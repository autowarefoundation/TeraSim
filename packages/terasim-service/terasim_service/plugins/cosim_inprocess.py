"""TeraSimCoSimInProcessPlugin: the co-simulation plugin (single-process link).

The CARLA-facing co-sim loop (client thread) and the TeraSim simulation loop
(sim thread) live in ONE process and exchange commands/state as Python
objects. This replaced the transports of the earlier co-sim stages (Redis
lists polled by a FastAPI service, then gRPC RPCs between two processes),
both of which have been removed:

  Redis "control" key polling / Tick RPC   -> tick_async() + threading.Event
  Redis "agent_commands" list / RPC field  -> AgentCommand objects (no JSON)
  Redis "state" keys / RPC state_json      -> TickResult.state (dict, no JSON)

One tick_async() call = deliver this step's agent commands, run exactly one
SUMO step, publish the post-step state. The client thread and the sim loop
rendezvous through two events (_tick_requested / _step_done).

Threading contract: a SINGLE co-sim client thread, calling
tick_async() -> handle.result() strictly in that order (the next tick_async
only after the previous handle resolved). The published state dict is a fresh
snapshot each step and is never mutated by the plugin afterwards.

The simulation-state construction (_build_simulation_state) and agent-command
application (_apply_agent_command) used to live in the Redis-era
TeraSimCoSimPlugin base class shared by all transports; with the other
transports gone they are part of this class.
"""

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Optional

import numpy as np
from terasim.overlay import traci
from terasim.simulator import Simulator
from terasim_nde_nade.adversity import ConstructionAdversity

from ..utils import AgentCommand, SimulationState, SUMOSignal
from ..utils.sumo_lane_geometry import (
    adapt_lookahead_distances_for_compiled_paths,
    build_external_state_lateral_action_lookahead,
    compile_lane_shapes,
    extract_next_link_lane_ids,
    reconstruct_position_from_lane_geometry,
    select_route_aware_lane_projection,
)
from .base import BasePlugin


def interpolate_by_distance(points, step):
    """
    Interpolate a tuple of tuples so that the distance between each point is equal to 'step'.

    Args:
        points (tuple of tuple): Original shape, e.g., ((x1, y1), (x2, y2), ...)
        step (float): Desired distance between points.

    Returns:
        list of list: Interpolated points as [[x, y], ...] with equal spacing.
    """
    points = np.array(points, dtype=np.float32)
    # Compute distances between consecutive points
    deltas = np.diff(points, axis=0)
    seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    cumulative = np.insert(np.cumsum(seg_lengths), 0, 0)
    total_length = cumulative[-1]
    if total_length == 0:
        return [points[0].tolist()]
    # Generate equally spaced distances
    num_points = int(np.floor(total_length / step)) + 1
    distances = np.linspace(0, total_length, num_points)
    # Interpolate x and y separately
    x_interp = np.interp(distances, cumulative, points[:, 0])
    y_interp = np.interp(distances, cumulative, points[:, 1])
    return [[float(x), float(y)] for x, y in zip(x_interp, y_interp)]


def generate_construction_zone_shape(lane_shape, lane_width, direction):
    """
    Generate a construction zone shape based on the lane shape and lane width.
    The first ten points of the lane_shape are offset laterally, with the offset
    gradually changing from direction * lane_width/2 to -direction * lane_width/2.
    The remaining points are offset by a constant -direction * lane_width/2.

    Args:
        lane_shape (list of list): The lane shape as a list of [x, y] points.
        lane_width (float): The width of the lane.
        direction (int): -1 for from left to right, 1 for from right to left.

    Returns:
        list of list: The offset lane shape.
    """
    n = min(10, len(lane_shape))
    construction_zone_shape = []
    for i, pt in enumerate(lane_shape):
        pt = np.array(pt)
        # Compute tangent direction
        if i < len(lane_shape) - 1:
            next_pt = np.array(lane_shape[i + 1])
            dir_vec = next_pt - pt
        else:
            prev_pt = np.array(lane_shape[i - 1])
            dir_vec = pt - prev_pt
        norm = np.linalg.norm(dir_vec)
        if norm == 0:
            dir_vec = np.array([1.0, 0.0])
        else:
            dir_vec = dir_vec / norm
        # Normal vector (perpendicular)
        normal = np.array([-dir_vec[1], dir_vec[0]]) * direction * -1

        # Compute offset
        if i < n:
            # Linear interpolation from +lane_width/2 to -lane_width/2
            alpha = i / (n - 1) if n > 1 else 0
            offset_val = (1 - alpha) * (lane_width / 2) + alpha * (-lane_width / 2)
        else:
            offset_val = - lane_width / 2

        offset_pt = pt + normal * offset_val
        construction_zone_shape.append(offset_pt.tolist())
    return construction_zone_shape


DEFAULT_COSIM_PLUGIN_CONFIG = {
    "name": "terasim_cosim_plugin",
    "priority": {
        "before_env": {
            "start": -90,
            "step": -90,
            "stop": -90,
        },
        "after_env": {
            "start": 90,
            "step": 90,
            "stop": 90,
        },
    },
}


@dataclass
class TickResult:
    """Snapshot of the co-sim state after a step (or at rest)."""

    status: str
    state: Optional[dict]  # SimulationState.model_dump(); None before the first build
    completed_sumo_time: float
    completed_tick_count: int


class TickHandle:
    """Future-like handle for one requested SUMO step.

    result() blocks until the sim thread finishes that step (or the
    simulation ends) and returns the post-step TickResult.
    """

    def __init__(self, plugin: "TeraSimCoSimInProcessPlugin", resolved: Optional[TickResult] = None):
        self._plugin = plugin
        self._resolved = resolved  # pre-resolved for pass-through (ended) calls

    def result(self, timeout: float = 300.0) -> TickResult:
        if self._resolved is not None:
            return self._resolved
        if not self._plugin._step_done.wait(timeout=timeout):
            raise TimeoutError(
                f"SUMO step did not complete within {timeout:.0f}s"
            )
        return self._plugin.get_result()


class TeraSimCoSimInProcessPlugin(BasePlugin):
    """Co-simulation plugin driven by a same-process co-sim client."""

    # Longest time function_before_env_step keeps waiting for a tick request
    # before auto-stopping.
    IDLE_TIMEOUT_S = 600.0

    # cadence (in steps) for pruning per-vehicle caches of departed ids
    CACHE_PRUNE_EVERY_STEPS = 1200

    LATERAL_SPEED_EPSILON = 0.05
    LATERAL_DIRECTION_MIN_ALIGNMENT = 0.5
    LATERAL_DIRECTION_WARNING_INTERVAL = 100

    def __init__(
        self,
        simulation_uuid: str,
        plugin_config: dict = DEFAULT_COSIM_PLUGIN_CONFIG,
        base_dir: str = "output",
        auto_run: bool = False,
    ):
        """Initialize the Co-Simulation plugin.

        Args:
            simulation_uuid (str): Unique identifier for the simulation instance.
            plugin_config (dict, optional): Configuration for the plugin. Defaults to DEFAULT_COSIM_PLUGIN_CONFIG.
            base_dir (str, optional): Base directory for the log file. Defaults to "output".
            auto_run (bool, optional): Must stay False: this link is strictly
                lock-stepped (one tick_async = one SUMO step).
        """
        super().__init__(simulation_uuid, plugin_config)
        if auto_run:
            # auto_run would advance SUMO without tick requests; this link is
            # strictly lock-stepped, so reject it early.
            raise ValueError("TeraSimCoSimInProcessPlugin requires auto_run=False")
        self.base_dir = base_dir

        # Setup logging
        self.logger = self._setup_logger(base_dir)

        # This plugin logs on the per-step hot path while holding the GIL, so
        # DEBUG-level chatter (e.g. the per-command dump) stays off unless
        # explicitly requested; INFO keeps the step-finished measurement line.
        if os.getenv("TERASIM_COSIM_LOG_DEBUG", "") in ("", "0", "false", "no"):
            self.logger.setLevel(logging.INFO)

        # Maintain controlled agents in each step, assuming each agent can be controlled by only one command
        self.controlled_agents_each_step = set()

        # Physical co-simulation is opt-in. Phase A assimilates CARLA pose and
        # motion immediately; the normal priority-10 SUMO step is the only
        # Phase B time advance.
        feedback_actor_value = os.getenv("CARLA_COSIM_ACKERMANN_FEEDBACK_ACTORS", "")
        self.physics_feedback_actor_ids = {
            actor_id.strip()
            for actor_id in feedback_actor_value.split(",")
            if actor_id.strip()
        }
        self.physics_feedback_mode = os.getenv(
            "CARLA_COSIM_ACKERMANN_FEEDBACK_MODE", "off"
        ).strip().lower()
        self.physics_assimilation_mode = os.getenv(
            "CARLA_COSIM_ACKERMANN_FEEDBACK_ASSIMILATION_MODE", "legacy"
        ).strip().lower()
        self.physics_external_state_enabled = (
            self.physics_feedback_mode == "apply"
            and self.physics_assimilation_mode == "external_state"
            and bool(self.physics_feedback_actor_ids)
        )
        self.physics_step_length = max(
            0.0, float(os.getenv("CARLA_COSIM_STEP_LENGTH", "0.05"))
        )
        self.physics_strict_lane_hint = self._parse_bool_env(
            "CARLA_COSIM_ACKERMANN_FEEDBACK_EXTERNAL_STATE_STRICT_LANE_HINT", True
        )
        self.physics_validate_external_state = self._parse_bool_env(
            "CARLA_COSIM_ACKERMANN_FEEDBACK_VALIDATE_EXTERNAL_STATE", True
        )
        self.physics_match_threshold = max(
            0.0,
            float(
                os.getenv(
                    "CARLA_COSIM_ACKERMANN_FEEDBACK_BACKGROUND_MOVE_TO_MAX_DISTANCE",
                    "8.0",
                )
            ),
        )
        self.physics_position_tolerance = max(
            0.0,
            float(
                os.getenv(
                    "CARLA_COSIM_ACKERMANN_FEEDBACK_EXTERNAL_STATE_POSITION_TOLERANCE",
                    "0.001",
                )
            ),
        )
        self.physics_feedback_observations = {}
        self.physics_lateral_direction_states = {}
        self.physics_lane_geometry_cache = {}
        self.physics_lookahead_path_cache = {}
        self.physics_lookahead_min_distance = max(
            0.1, float(os.getenv("CARLA_COSIM_ACKERMANN_LOOKAHEAD_MIN_DISTANCE", "7.0"))
        )
        self.physics_lookahead_max_distance = max(
            self.physics_lookahead_min_distance,
            float(os.getenv("CARLA_COSIM_ACKERMANN_LOOKAHEAD_MAX_DISTANCE", "15.0")),
        )
        if self.physics_external_state_enabled:
            self.logger.info(
                "Physical co-sim Phase A enabled: actors=%s strict_lane=%s",
                sorted(self.physics_feedback_actor_ids),
                self.physics_strict_lane_hint,
            )

        # Cache construction zone shapes
        self.construction_zone_shapes = None

        # Initialize last orientations cache
        self.last_orientations = {}  # {vehicle_id: (last_orientation, last_time)}

        self.state_filter_enabled = self._parse_bool_env(
            "TERASIM_COSIM_STATE_FILTER", False
        )
        self.state_filter_center_id = os.getenv(
            "TERASIM_COSIM_STATE_FILTER_CENTER_ID", "AV"
        )
        self.state_filter_radius = self._parse_optional_float(
            os.getenv("TERASIM_COSIM_STATE_FILTER_RADIUS", "")
        )
        self.state_filter_missing_center_logged = False
        self.state_filter_error_logged = False
        if self.state_filter_enabled:
            self.logger.info(
                "TeraSim co-sim state filter enabled: center=%s radius=%s",
                self.state_filter_center_id,
                self.state_filter_radius,
            )

        self.lane_relative_position_enabled = self._parse_bool_env(
            "TERASIM_COSIM_LANE_RELATIVE_POSITION",
            self.physics_external_state_enabled,
        )
        if self.lane_relative_position_enabled:
            self.logger.info(
                "TeraSim co-sim lane-relative reconstructed positions enabled "
                "for filtered state vehicles"
            )

        # lon/lat per agent costs one convertGeo (projection) per vehicle per
        # step, and the in-process consumer (CarlaCosim) converts coordinates
        # from x/y itself and never reads lon/lat, so skip it by default
        # (TERASIM_COSIM_STATE_LONLAT=1 re-enables it for external consumers
        # of the recorded state).
        self.state_lonlat_enabled = self._parse_bool_env(
            "TERASIM_COSIM_STATE_LONLAT", False
        )
        # Per-vehicle static attributes (length/width/height/type are constant
        # in SUMO) and the static half of the traffic-light details; both were
        # re-fetched/re-serialized every step.
        self._static_attr_cache = {}
        self._tls_static_cache = None
        self._cache_prune_countdown = self.CACHE_PRUNE_EVERY_STEPS

        # In-process rendezvous state (client thread <-> sim thread)
        self._lock = threading.Lock()
        self._status = "created"
        self._state = None  # dict (SimulationState.model_dump())
        self._completed_sumo_time = 0.0
        self._completed_tick_count = 0
        self._pending_commands = []  # list[AgentCommand]
        self._stop_requested = False
        self._ready = threading.Event()  # set once wait_for_tick is reached (or startup failed)
        self._tick_requested = threading.Event()
        self._step_done = threading.Event()
        self._client_serial = threading.Lock()  # serialize concurrent tick_async calls

    @staticmethod
    def _parse_bool_env(name, default=False):
        value = os.getenv(name)
        if value in (None, ""):
            return default
        return value.strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _parse_optional_float(value):
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _setup_logger(self, base_dir: str) -> logging.Logger:
        """Setup logger for the plugin.

        Args:
            base_dir (str): Base directory for the log file.

        Returns:
            logging.Logger: Logger instance for the plugin.
        """
        logger = logging.getLogger(f"{self.plugin_name}-{self.simulation_uuid}")
        logger.setLevel(logging.DEBUG)

        # Create a rotating file handler
        file_handler = RotatingFileHandler(
            f"{base_dir}/{self.plugin_name}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    # ------------------------------------------------------------------
    # client-side API (called from the co-sim client thread)
    # ------------------------------------------------------------------
    def wait_until_ready(self, timeout: float) -> bool:
        """Block until the simulation reaches wait_for_tick (SUMO loaded).

        Returns True when the plugin is ready for tick_async; False on
        timeout or when the simulation already ended during startup.
        """
        if not self._ready.wait(timeout=timeout):
            return False
        with self._lock:
            return self._status == "wait_for_tick"

    def tick_async(self, commands) -> TickHandle:
        """Request one SUMO step (non-blocking).

        commands: list of dicts {agent_id, agent_type, command_type, data}
        (same shape the earlier transports carried as JSON). Returns a
        TickHandle whose .result(timeout) yields the post-step TickResult.
        """
        with self._client_serial:
            with self._lock:
                if self._status in ("finished", "error") or self._stop_requested:
                    return TickHandle(self, resolved=self._result_locked())
                self._pending_commands = [
                    AgentCommand.model_validate(c) for c in commands
                ]
            self._step_done.clear()
            self._tick_requested.set()
            return TickHandle(self)

    def get_result(self) -> TickResult:
        """Fetch the latest state without advancing the simulation."""
        with self._lock:
            return self._result_locked()

    def request_stop(self):
        """Ask the simulation loop to stop (idempotent, thread-safe)."""
        self.logger.info("Stop requested by the co-sim client")
        self._stop_requested = True

    def abort(self, status: str = "error"):
        """Mark the simulation as ended on behalf of a dead sim thread.

        Called by the runner when sim.run() raises: releases a client blocked
        in wait_until_ready()/result() so the process can shut down.
        """
        self._finish(status)
        self._ready.set()

    # ------------------------------------------------------------------
    # lifecycle hooks
    # ------------------------------------------------------------------
    def inject(self, simulator: Simulator, ctx):
        """Inject the plugin into the simulation.

        Args:
            simulator (Simulator): The simulator object.
            ctx (dict): The context information.
        """
        self.ctx = ctx
        self.simulator = simulator

        simulator.start_pipeline.hook(f"{self.plugin_name}_before_env_start", self.function_before_env_start, priority=self.plugin_priority["before_env"]["start"])
        simulator.start_pipeline.hook(f"{self.plugin_name}_after_env_start", self.function_after_env_start, priority=self.plugin_priority["after_env"]["start"])
        simulator.step_pipeline.hook(f"{self.plugin_name}_before_env_step", self.function_before_env_step, priority=self.plugin_priority["before_env"]["step"])
        simulator.step_pipeline.hook(f"{self.plugin_name}_after_env_step", self.function_after_env_step, priority=self.plugin_priority["after_env"]["step"])
        simulator.stop_pipeline.hook(f"{self.plugin_name}_before_env_stop", self.function_before_env_stop, priority=self.plugin_priority["before_env"]["stop"])
        simulator.stop_pipeline.hook(f"{self.plugin_name}_after_env_stop", self.function_after_env_stop, priority=self.plugin_priority["after_env"]["stop"])

    def function_before_env_start(self, simulator: Simulator, ctx):
        self._set_status("initializing")
        self.logger.info(
            f"Simulation UUID: {self.simulation_uuid}, start initialization!"
        )
        return True

    def function_after_env_start(self, simulator: Simulator, ctx):
        try:
            # Build an initial state so the client can seed its render
            # pipeline (e.g. AV shape init) before the first tick.
            try:
                state = self._build_simulation_state(simulator)
                with self._lock:
                    self._state = state.model_dump()
            except Exception as e:
                self.logger.warning(f"Initial state build failed (non-fatal): {e}")
            self._set_status("wait_for_tick")
            self._ready.set()
            self.logger.info(
                f"Simulation UUID: {self.simulation_uuid}, finish initialization!"
            )
            return True
        except Exception as e:
            self.logger.exception(f"Unexpected error after start: {e}")
            self.abort("error")
            return False

    def function_before_env_step(self, simulator: Simulator, ctx):
        idle_start = time.time()
        while True:
            if self._stop_requested:
                self.logger.info("Stopping simulation")
                simulator.running = False  # stop the main loop
                return False
            if time.time() - idle_start > self.IDLE_TIMEOUT_S:
                self.logger.warning("No tick request for %.0fs, auto-stopping", self.IDLE_TIMEOUT_S)
                simulator.running = False
                return False
            if self._tick_requested.wait(timeout=0.1):
                self._tick_requested.clear()
                break

        # Apply the commands delivered with this tick request.
        with self._lock:
            commands = self._pending_commands
            self._pending_commands = []
        # Phase A observations are valid only for the current SUMO step.
        # A filtered actor may disappear from one exported state and reappear
        # later; never let its previous observation pass Phase B validation.
        self.physics_feedback_observations.clear()
        self.controlled_agents_each_step.clear()
        for command in commands:
            self._apply_agent_command(command)

        self._set_status("running")
        self.logger.debug("Simulation step started")
        return True

    def function_after_env_step(self, simulator: Simulator, ctx):
        try:
            state = self._build_simulation_state(simulator)
        except Exception as e:
            self.logger.exception(f"State build failed, stopping simulation: {e}")
            self._finish("error")
            return False
        completed_sumo_time = traci.simulation.getTime()
        with self._lock:
            self._state = state.model_dump()
            self._completed_sumo_time = completed_sumo_time
            self._completed_tick_count += 1
            self._status = "ticked"
            completed_tick_count = self._completed_tick_count
        self._step_done.set()
        # One line per step on purpose: with the RPC observation endpoint
        # gone, this log line (console handler prints asctime) is the external
        # interface for step-rate / clock-ratio / vehicle-count measurement.
        # vehicles= is the TOTAL SUMO vehicle count (the measurement x-axis; it
        # must not shrink when TERASIM_COSIM_STATE_FILTER trims the published
        # state); vehicles_state= is what actually went into the state.
        try:
            state_vehicle_count = state.agent_count.get("vehicle", -1)
        except Exception:
            state_vehicle_count = -1
        try:
            total_vehicle_count = traci.vehicle.getIDCount()
        except Exception:
            total_vehicle_count = state_vehicle_count
        self.logger.info(
            "Simulation step finished! completed_sumo_time=%s completed_tick_count=%s "
            "vehicles=%s vehicles_state=%s",
            completed_sumo_time,
            completed_tick_count,
            total_vehicle_count,
            state_vehicle_count,
        )
        return True

    def function_before_env_stop(self, simulator: Simulator, ctx):
        pass

    def function_after_env_stop(self, simulator: Simulator, ctx):
        self.physics_lateral_direction_states.clear()
        self._finish("finished")
        self.logger.info(f"Simulation {self.simulation_uuid} finished!")

    # ------------------------------------------------------------------
    # simulation-state construction (shared with no other transport since
    # the Redis and gRPC paths were removed; kept factored for reuse)
    # ------------------------------------------------------------------
    def get_vehicle_vru_ids(self):
        """Get all vehicle and VRU IDs in the simulation."""
        all_ids = set(traci.vehicle.getIDList() + traci.person.getIDList())
        # Separate by type in one pass: construction objects, VRUs, and regular vehicles
        construction_ids, vru_ids, vehicle_ids = [], [], []
        for agent_id in all_ids:
            if agent_id.startswith("CONSTRUCTION_"):
                construction_ids.append(agent_id)
            elif "VRU" in agent_id:
                vru_ids.append(agent_id)
            else:
                vehicle_ids.append(agent_id)
        return vehicle_ids, vru_ids, construction_ids

    def _filter_vehicle_ids_for_state(self, vehicle_ids):
        if (
            not self.state_filter_enabled
            or self.state_filter_radius is None
            or self.state_filter_radius <= 0
        ):
            return vehicle_ids, {}

        try:
            if self.state_filter_center_id not in vehicle_ids:
                if not self.state_filter_missing_center_logged:
                    self.logger.warning(
                        "State filter center vehicle %s is missing; writing all vehicles",
                        self.state_filter_center_id,
                    )
                    self.state_filter_missing_center_logged = True
                return vehicle_ids, {}

            center_position = traci.vehicle.getPosition3D(self.state_filter_center_id)
            position_cache = {self.state_filter_center_id: center_position}
            radius_sq = self.state_filter_radius * self.state_filter_radius
            filtered_vehicle_ids = []
            for vid in vehicle_ids:
                if vid in position_cache:
                    position = position_cache[vid]
                else:
                    position = traci.vehicle.getPosition3D(vid)
                    position_cache[vid] = position
                dx = position[0] - center_position[0]
                dy = position[1] - center_position[1]
                if vid == self.state_filter_center_id or dx * dx + dy * dy <= radius_sq:
                    filtered_vehicle_ids.append(vid)

            self.state_filter_missing_center_logged = False
            self.state_filter_error_logged = False
            return filtered_vehicle_ids, position_cache
        except Exception as e:
            if not self.state_filter_error_logged:
                self.logger.warning("State filter failed; writing all vehicles: %s", e)
                self.state_filter_error_logged = True
            return vehicle_ids, {}

    @staticmethod
    def _as_finite_float(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _is_physics_feedback_actor(self, actor_id):
        return bool(
            self.physics_external_state_enabled
            and actor_id != "AV"
            and (
                actor_id in self.physics_feedback_actor_ids
                or "*" in self.physics_feedback_actor_ids
            )
        )

    @staticmethod
    def _physics_lane_index(lane_id):
        try:
            return int(lane_id.rsplit("_", 1)[1])
        except (AttributeError, IndexError, ValueError):
            return -1

    def _physics_lane_geometry(self, lane_id):
        geometry = self.physics_lane_geometry_cache.get(lane_id)
        if geometry is not None:
            return geometry
        edge_id = traci.lane.getEdgeID(lane_id)
        geometry = {
            "lane_id": lane_id,
            "edge_id": edge_id,
            "lane_index": self._physics_lane_index(lane_id),
            "shape": tuple(tuple(point) for point in traci.lane.getShape(lane_id)),
            "length": traci.lane.getLength(lane_id),
        }
        self.physics_lane_geometry_cache[lane_id] = geometry
        return geometry

    def _apply_physics_external_state(self, command, x, y):
        """Phase A: assimilate a CARLA state without advancing SUMO time."""
        actor_id = command.agent_id
        data = command.data
        immediate_move = getattr(traci.vehicle, "moveToXYImmediate", None)
        if not callable(immediate_move):
            raise RuntimeError(
                "physical co-sim requires patched SUMO moveToXYImmediate"
            )

        position = (self._as_finite_float(x), self._as_finite_float(y))
        sumo_angle = self._as_finite_float(data.get("sumo_angle"))
        speed = self._as_finite_float(data.get("speed"))
        acceleration = self._as_finite_float(data.get("acceleration"))
        source_frame = data.get("source_carla_frame")
        position_z = self._as_finite_float(data.get("z"))
        if (
            None in position
            or sumo_angle is None
            or speed is None
            or speed < 0.0
            or acceleration is None
            or not isinstance(source_frame, int)
        ):
            raise RuntimeError(
                f"invalid physical feedback state actor={actor_id} frame={source_frame}"
            )

        lane_id = traci.vehicle.getLaneID(actor_id)
        if not lane_id:
            raise RuntimeError(
                f"physical feedback actor has no current SUMO lane actor={actor_id}"
            )
        geometry = self._physics_lane_geometry(lane_id)
        projection = select_route_aware_lane_projection(
            position,
            sumo_angle,
            [geometry],
            position_z=position_z,
            current_lane_id=lane_id,
            max_distance=self.physics_match_threshold,
            max_heading_error=180.0,
            prefer_current_lane=True,
        )
        if projection is None:
            # Do not kill the whole run when one actor's measured pose cannot be
            # projected onto its current lane (e.g. lanes whose SUMO centerline is
            # offset from the xodr geometry): skip assimilation for this actor this
            # tick and let the SUMO-side model drive the mirror instead. A global
            # raise here ended the entire run twice, deterministically, on the
            # Kashiwanoha map (physics-inprocess-findings.md §2).
            skips = getattr(self, "_physics_mapping_skips", None)
            if skips is None:
                skips = {}
                self._physics_mapping_skips = skips
            skips[actor_id] = skips.get(actor_id, 0) + 1
            count = skips[actor_id]
            if count <= 3 or count % 200 == 0:
                self.logger.warning(
                    "physical feedback current-lane mapping failed; skipping "
                    "assimilation this tick actor=%s lane=%s position=%s count=%d",
                    actor_id,
                    lane_id,
                    position,
                    count,
                )
            return None

        requested_route = tuple(traci.vehicle.getRoute(actor_id))
        phase_a_time = traci.simulation.getTime()
        immediate_move(
            actor_id,
            projection["edge_id"],
            projection["lane_index"],
            position[0],
            position[1],
            sumo_angle,
            1,
            self.physics_match_threshold,
            self.physics_strict_lane_hint,
        )
        traci.vehicle.setSpeed(actor_id, -1)
        traci.vehicle.setPreviousSpeed(actor_id, speed, acceleration)

        observed_position = traci.vehicle.getPosition(actor_id)
        observed_angle = traci.vehicle.getAngle(actor_id)
        observed_speed = traci.vehicle.getSpeed(actor_id)
        observed_acceleration = traci.vehicle.getAcceleration(actor_id)
        observed_lane_id = traci.vehicle.getLaneID(actor_id)
        observed_time = traci.simulation.getTime()
        observed_route = tuple(traci.vehicle.getRoute(actor_id))
        position_error = math.hypot(
            observed_position[0] - position[0],
            observed_position[1] - position[1],
        )
        angle_error = abs((observed_angle - sumo_angle + 180.0) % 360.0 - 180.0)
        valid = bool(
            abs(observed_time - phase_a_time) <= 1e-9
            and position_error <= self.physics_position_tolerance
            and angle_error <= 1e-6
            and abs(observed_speed - speed) <= 1e-6
            and abs(observed_acceleration - acceleration) <= 1e-6
            and observed_route == requested_route
            and (
                not self.physics_strict_lane_hint
                or observed_lane_id == projection["lane_id"]
            )
        )
        if self.physics_validate_external_state and not valid:
            raise RuntimeError(
                "physical feedback validation failed "
                f"actor={actor_id} lane={projection['lane_id']}->{observed_lane_id} "
                f"position_error={position_error:.6g} angle_error={angle_error:.6g}"
            )

        lane_length = None
        lane_position = None
        try:
            lane_position = traci.vehicle.getLanePosition(actor_id)
            lane_length = traci.lane.getLength(observed_lane_id)
        except Exception:
            pass
        self.physics_feedback_observations[actor_id] = {
            "position": tuple(observed_position[:2]),
            "sumo_angle": observed_angle,
            "speed": observed_speed,
            "acceleration": observed_acceleration,
            "requested_lane_id": projection["lane_id"],
            "lane_id": observed_lane_id,
            "lane_position": lane_position,
            "lane_length": lane_length,
            "phase_a_time": phase_a_time,
            "source_carla_frame": source_frame,
        }
        return True

    def _physics_lane_change_action(self, vehicle_id, current_lane_id):
        try:
            left_state = traci.vehicle.getLaneChangeState(vehicle_id, 1)[1]
            right_state = traci.vehicle.getLaneChangeState(vehicle_id, -1)[1]
        except Exception:
            return "none", ""
        wants_left = bool(left_state & traci.constants.LCA_LEFT)
        wants_right = bool(right_state & traci.constants.LCA_RIGHT)
        if wants_left == wants_right:
            return "none", ""
        return ("left" if wants_left else "right"), ""

    def _physics_compiled_lookahead_path(self, vehicle_id, current_lane_id):
        try:
            current_lane_id = traci.vehicle.getLaneID(vehicle_id)
        except Exception:
            return None
        try:
            next_links = tuple(traci.vehicle.getNextLinks(vehicle_id) or ())[:1]
        except Exception:
            return None

        next_lane_ids = extract_next_link_lane_ids(next_links)
        lane_ids = []
        for lane_id in [current_lane_id, *next_lane_ids]:
            if lane_id and lane_id not in lane_ids:
                lane_ids.append(lane_id)
        if not lane_ids:
            return None

        key = tuple(lane_ids)
        if key in self.physics_lookahead_path_cache:
            return self.physics_lookahead_path_cache[key]

        shapes = []
        valid_ids = []
        for lane_id in lane_ids:
            try:
                geometry = self._physics_lane_geometry(lane_id)
            except Exception:
                break
            if len(geometry["shape"]) < 2:
                break
            shapes.append(geometry["shape"])
            valid_ids.append(lane_id)

        compiled = compile_lane_shapes(shapes)
        if compiled is not None:
            self.physics_lookahead_path_cache[key] = compiled
            self.physics_lookahead_path_cache.setdefault(tuple(valid_ids), compiled)
        return compiled

    def _physics_previous_lateral_direction(
        self, vehicle_id, observation, lateral_speed
    ):
        """Return only a direction measured in the immediately preceding step."""
        states = getattr(self, "physics_lateral_direction_states", None)
        if states is None:
            self.physics_lateral_direction_states = {}
            states = self.physics_lateral_direction_states
        state = states.get(vehicle_id)
        if state is None or abs(lateral_speed) <= self.LATERAL_SPEED_EPSILON + 1e-9:
            return None
        previous_speed = self._as_finite_float(state.get("confirmed_lateral_speed"))
        confirmed_time = self._as_finite_float(state.get("confirmed_sumo_time"))
        phase_a_time = self._as_finite_float(observation.get("phase_a_time"))
        if (
            previous_speed is None
            or abs(previous_speed) <= self.LATERAL_SPEED_EPSILON + 1e-9
            or confirmed_time is None
            or phase_a_time is None
            or abs(confirmed_time - phase_a_time) > 1e-9
        ):
            return None
        return state.get("confirmed_world_direction")

    def _prune_departed_physics_lateral_directions(self, all_vehicle_ids):
        states = getattr(self, "physics_lateral_direction_states", {})
        if not states:
            return
        alive_vehicle_ids = set(all_vehicle_ids)
        for stale_id in [key for key in states if key not in alive_vehicle_ids]:
            del states[stale_id]

    def _update_physics_lateral_direction_state(
        self,
        vehicle_id,
        *,
        phase_b_time,
        lateral_speed,
        lane_change_intent,
        action,
    ):
        """Update diagnostic state without letting SUMO intent steer CARLA."""
        states = getattr(self, "physics_lateral_direction_states", None)
        if states is None:
            self.physics_lateral_direction_states = {}
            states = self.physics_lateral_direction_states
        state = states.get(vehicle_id, {})
        previous_unresolved = int(state.get("unresolved_count", 0))
        source = action.get("lateral_direction_source", "inactive")

        if not action.get("valid", False):
            return previous_unresolved, False

        if abs(lateral_speed) <= self.LATERAL_SPEED_EPSILON + 1e-9:
            if previous_unresolved:
                self.logger.info(
                    "Physical co-sim lateral direction recovered: actor=%s "
                    "unresolved_frames=%s source=inactive",
                    vehicle_id,
                    previous_unresolved,
                )
            states.pop(vehicle_id, None)
            return 0, False

        world_lateral_speed = self._as_finite_float(
            action.get("world_lateral_speed")
        )
        if source == "phase_b_delta" and world_lateral_speed not in (None, 0.0):
            state = self._record_confirmed_physics_lateral_direction(
                states,
                vehicle_id,
                phase_b_time,
                lateral_speed,
                world_lateral_speed,
                action,
                previous_unresolved,
            )
        elif source == "previous_confirmed":
            state["unresolved_count"] = 0
            states[vehicle_id] = state
        elif source == "route_only":
            return self._record_unresolved_physics_lateral_direction(
                states,
                state,
                vehicle_id,
                previous_unresolved,
                lateral_speed,
                lane_change_intent,
            )
        intent_conflict = self._physics_lateral_intent_conflict(
            lane_change_intent, world_lateral_speed
        )
        state["last_lane_change_intent"] = lane_change_intent
        state["last_intent_conflict"] = intent_conflict
        return 0, intent_conflict

    def _record_confirmed_physics_lateral_direction(
        self,
        states,
        vehicle_id,
        phase_b_time,
        lateral_speed,
        world_lateral_speed,
        action,
        previous_unresolved,
    ):
        normal_x = self._as_finite_float(action.get("world_left_normal_x"))
        normal_y = self._as_finite_float(action.get("world_left_normal_y"))
        state = states.get(vehicle_id, {})
        if normal_x is not None and normal_y is not None:
            direction_sign = math.copysign(1.0, world_lateral_speed)
            state = {
                "confirmed_world_direction": (
                    direction_sign * normal_x,
                    direction_sign * normal_y,
                ),
                "confirmed_sumo_time": phase_b_time,
                "confirmed_lateral_speed": lateral_speed,
                "unresolved_count": 0,
            }
            states[vehicle_id] = state
        if previous_unresolved:
            self.logger.info(
                "Physical co-sim lateral direction recovered: actor=%s "
                "unresolved_frames=%s source=phase_b_delta",
                vehicle_id,
                previous_unresolved,
            )
        return state

    def _record_unresolved_physics_lateral_direction(
        self,
        states,
        state,
        vehicle_id,
        previous_unresolved,
        lateral_speed,
        lane_change_intent,
    ):
        unresolved_count = previous_unresolved + 1
        state["unresolved_count"] = unresolved_count
        states[vehicle_id] = state
        if (
            unresolved_count == 1
            or unresolved_count % self.LATERAL_DIRECTION_WARNING_INTERVAL == 0
        ):
            self.logger.warning(
                "Physical co-sim lateral direction unresolved: actor=%s "
                "frames=%s lateral_speed=%.6g intent=%s; using route-only lookahead",
                vehicle_id,
                unresolved_count,
                lateral_speed,
                lane_change_intent,
            )
        return unresolved_count, False

    @staticmethod
    def _physics_lateral_intent_conflict(lane_change_intent, world_lateral_speed):
        if world_lateral_speed in (None, 0.0):
            return False
        intent_sign = {"left": 1.0, "right": -1.0}.get(lane_change_intent)
        return bool(
            intent_sign is not None
            and intent_sign != math.copysign(1.0, world_lateral_speed)
        )

    def _populate_physics_action_state(self, vehicle_id, vehicle_state):
        """Export the Phase-B SUMO action consumed by CARLA in the next frame."""
        if not self._is_physics_feedback_actor(vehicle_id):
            states = getattr(self, "physics_lateral_direction_states", None)
            if states is not None:
                states.pop(vehicle_id, None)
            return
        observation = self.physics_feedback_observations.get(vehicle_id)
        if observation is None:
            return

        phase_b_time = traci.simulation.getTime()
        phase_delta = phase_b_time - observation["phase_a_time"]
        if abs(phase_delta - self.physics_step_length) > 1e-9:
            raise RuntimeError(
                "physical co-sim Phase B time mismatch "
                f"actor={vehicle_id} delta={phase_delta} "
                f"expected={self.physics_step_length}"
            )

        vehicle_state["feedback_observed_x"] = observation["position"][0]
        vehicle_state["feedback_observed_y"] = observation["position"][1]
        vehicle_state["feedback_observed_sumo_angle"] = observation["sumo_angle"]
        vehicle_state["feedback_observed_speed"] = observation["speed"]
        vehicle_state["feedback_observed_acceleration"] = observation["acceleration"]
        vehicle_state["feedback_requested_lane_id"] = observation[
            "requested_lane_id"
        ]
        vehicle_state["feedback_observed_lane_id"] = observation["lane_id"]
        vehicle_state["feedback_phase_a_sumo_time"] = observation["phase_a_time"]
        vehicle_state["feedback_source_carla_frame"] = observation[
            "source_carla_frame"
        ]
        vehicle_state["feedback_longitudinal_error"] = None
        try:
            vehicle_state["sumo_desired_speed"] = traci.vehicle.getSpeedWithoutTraCI(
                vehicle_id
            )
        except Exception:
            vehicle_state["sumo_desired_speed"] = vehicle_state["speed"]
        try:
            vehicle_state["sumo_emergency_decel"] = traci.vehicle.getEmergencyDecel(
                vehicle_id
            )
        except Exception:
            vehicle_state["sumo_emergency_decel"] = None

        try:
            current_lane_id = traci.vehicle.getLaneID(vehicle_id)
        except Exception:
            current_lane_id = vehicle_state.get("lane_id", "")
        current_position_error = ""
        try:
            current_position = tuple(traci.vehicle.getPosition(vehicle_id)[:2])
        except Exception:
            current_position = (vehicle_state["x"], vehicle_state["y"])
            current_position_error = "phase_b_position_api_failed"
        try:
            current_lane_position = traci.vehicle.getLanePosition(vehicle_id)
        except Exception:
            current_lane_position = vehicle_state.get("lane_position")
        try:
            sumo_route = tuple(traci.vehicle.getRoute(vehicle_id))
        except Exception:
            sumo_route = ()
        try:
            sumo_slope = traci.vehicle.getSlope(vehicle_id)
        except Exception:
            sumo_slope = 0.0

        vehicle_state["lane_id"] = current_lane_id
        vehicle_state["lane_position"] = current_lane_position
        vehicle_state["sumo_route"] = sumo_route
        vehicle_state["sumo_slope"] = sumo_slope
        vehicle_state["external_state_maneuver_source_lane_id"] = ""
        vehicle_state["external_state_maneuver_target_lane_id"] = ""

        observed_lane_position = observation.get("lane_position")
        observed_lane_length = observation.get("lane_length")
        if observed_lane_position is not None and current_lane_position is not None:
            if current_lane_id == observation["lane_id"]:
                progress_delta = current_lane_position - observed_lane_position
            elif observed_lane_length is not None:
                progress_delta = (
                    observed_lane_length
                    - observed_lane_position
                    + current_lane_position
                )
            else:
                progress_delta = None
            if progress_delta is not None:
                vehicle_state["feedback_longitudinal_error"] = (
                    progress_delta
                    - float(observation["speed"]) * self.physics_step_length
                )

        intent, target_lane_id = self._physics_lane_change_action(
            vehicle_id, current_lane_id
        )
        vehicle_state["sumo_lane_change_intent"] = intent
        vehicle_state["sumo_lane_change_target_lane_id"] = target_lane_id
        try:
            lateral_speed = traci.vehicle.getLateralSpeed(vehicle_id)
        except Exception:
            lateral_speed = 0.0
        vehicle_state["lateral_speed"] = lateral_speed
        previous_lateral_direction = self._physics_previous_lateral_direction(
            vehicle_id, observation, lateral_speed
        )

        compiled_path = self._physics_compiled_lookahead_path(
            vehicle_id, current_lane_id
        )
        base_distance = min(
            self.physics_lookahead_max_distance,
            max(self.physics_lookahead_min_distance, vehicle_state["speed"]),
        )
        effective_distances, heading_changes = (
            adapt_lookahead_distances_for_compiled_paths(
                [compiled_path],
                [current_position],
                [base_distance],
            )
        )
        lookahead_distance = effective_distances[0]
        vehicle_state["lookahead_distance"] = lookahead_distance
        vehicle_state["lookahead_heading_change"] = heading_changes[0]
        vehicle_state["lookahead_origin_x"] = current_position[0]
        vehicle_state["lookahead_origin_y"] = current_position[1]

        if current_position_error:
            action = {
                "valid": False,
                "error": current_position_error,
                "mode": "route",
                "lookahead": None,
                "route_lookahead": None,
                "lateral_displacement": 0.0,
                "target_lateral_distance": None,
            }
        else:
            action = build_external_state_lateral_action_lookahead(
                compiled_path,
                current_position,
                lookahead_distance,
                lateral_speed=lateral_speed,
                desired_speed=vehicle_state["sumo_desired_speed"],
                min_forward_speed=0.2,
                lateral_speed_epsilon=0.05,
                phase_a_position=observation["position"],
                phase_step_length=self.physics_step_length,
                previous_world_lateral_direction=previous_lateral_direction,
                previous_direction_min_alignment=self.LATERAL_DIRECTION_MIN_ALIGNMENT,
                z=vehicle_state["z"],
            )

        unresolved_count, intent_conflict = self._update_physics_lateral_direction_state(
            vehicle_id,
            phase_b_time=phase_b_time,
            lateral_speed=lateral_speed,
            lane_change_intent=intent,
            action=action,
        )

        vehicle_state["lookahead_action_mode"] = action.get("mode", "route")
        vehicle_state["lookahead_action_valid"] = bool(action.get("valid", False))
        vehicle_state["lookahead_action_error"] = action.get("error", "")
        vehicle_state["lookahead_action_warning"] = action.get("warning", "")
        vehicle_state["lookahead_lateral_direction_source"] = action.get(
            "lateral_direction_source", "inactive"
        )
        vehicle_state["lookahead_lateral_direction_unresolved_count"] = (
            unresolved_count
        )
        vehicle_state["lookahead_lane_change_intent_conflict"] = intent_conflict
        vehicle_state["lookahead_lateral_horizon_displacement"] = float(
            action.get("lateral_displacement", 0.0)
        )
        vehicle_state["lookahead_target_lateral_distance"] = action.get(
            "target_lateral_distance"
        )
        vehicle_state["lookahead_route_tangent_x"] = action.get(
            "route_tangent_x", 0.0
        )
        vehicle_state["lookahead_route_tangent_y"] = action.get(
            "route_tangent_y", 0.0
        )
        vehicle_state["lookahead_world_left_normal_x"] = action.get(
            "world_left_normal_x"
        )
        vehicle_state["lookahead_world_left_normal_y"] = action.get(
            "world_left_normal_y"
        )
        vehicle_state["lookahead_phase_b_lateral_delta"] = action.get(
            "phase_b_lateral_delta"
        )
        vehicle_state["lookahead_expected_phase_b_lateral_distance"] = action.get(
            "expected_phase_b_lateral_distance"
        )
        vehicle_state["lookahead_world_lateral_speed"] = action.get(
            "world_lateral_speed", 0.0
        )
        vehicle_state["lookahead_lane_change_blend"] = 0.0
        route_lookahead = action.get("route_lookahead")
        if route_lookahead is not None:
            vehicle_state["lookahead_route_x"] = route_lookahead[0]
            vehicle_state["lookahead_route_y"] = route_lookahead[1]
            vehicle_state["lookahead_route_z"] = route_lookahead[2]
        lookahead = action.get("lookahead")
        vehicle_state["lookahead_position_valid"] = False
        if lookahead is not None:
            vehicle_state["lookahead_x"] = lookahead[0]
            vehicle_state["lookahead_y"] = lookahead[1]
            vehicle_state["lookahead_z"] = lookahead[2]
            vehicle_state["lookahead_position_valid"] = True

    def _populate_lane_relative_position(self, vehicle_id, vehicle_state):
        """Fill the lane-relative fields of a vehicle-state dict (opt-in path)."""
        if not self.lane_relative_position_enabled:
            return

        lane_id = traci.vehicle.getLaneID(vehicle_id)
        if not lane_id:
            return
        lane_position = traci.vehicle.getLanePosition(vehicle_id)
        lateral_offset = traci.vehicle.getLateralLanePosition(vehicle_id)
        lane_shape = traci.lane.getShape(lane_id)
        try:
            lane_length = traci.lane.getLength(lane_id)
        except Exception:
            lane_length = None
        reconstructed = reconstruct_position_from_lane_geometry(
            lane_shape,
            lane_position,
            lateral_offset,
            vehicle_state["z"],
            lane_length,
            (vehicle_state["x"], vehicle_state["y"]),
        )

        vehicle_state["lane_id"] = lane_id
        vehicle_state["lane_position"] = lane_position
        vehicle_state["lateral_offset"] = lateral_offset
        if reconstructed is None:
            return
        (
            vehicle_state["reconstructed_x"],
            vehicle_state["reconstructed_y"],
            vehicle_state["reconstructed_z"],
        ) = reconstructed
        vehicle_state["reconstructed_position_valid"] = True

    def _build_simulation_state(self, simulator):
        """Collect the current simulation state from SUMO into a SimulationState.

        Pure state construction (no network I/O); raises on TraCI errors.
        """
        simulation_state = SimulationState()
        simulation_time = traci.simulation.getTime()
        simulation_state.simulation_time = simulation_time

        # Get all interested agent IDs
        all_vehicle_ids, vru_ids, construction_ids = self.get_vehicle_vru_ids()
        vehicle_ids, vehicle_position_cache = self._filter_vehicle_ids_for_state(
            all_vehicle_ids
        )
        self._prune_departed_physics_lateral_directions(all_vehicle_ids)
        simulation_state.agent_count = {
            "vehicle": len(vehicle_ids),
            "vru": len(vru_ids),
            "construction": len(construction_ids),
        }

        # Occasionally drop departed ids from the per-vehicle caches (they are
        # keyed by SUMO id and would otherwise grow for the whole run).
        self._cache_prune_countdown -= 1
        if self._cache_prune_countdown <= 0:
            self._cache_prune_countdown = self.CACHE_PRUNE_EVERY_STEPS
            alive = set(all_vehicle_ids)
            alive.update(vru_ids)
            for cache in (
                self.last_orientations,
                self._static_attr_cache,
            ):
                for stale_id in [key for key in cache if key not in alive]:
                    del cache[stale_id]

        # Add vehicle states (plain dicts in the AgentStateSimplified shape;
        # scalar math via the math module — numpy scalar ufuncs are several
        # times slower and this loop runs per vehicle per step).
        lonlat_enabled = self.state_lonlat_enabled
        static_attrs = self._static_attr_cache
        last_orientations = self.last_orientations
        vehicles = {}
        for vid in vehicle_ids:
            position = vehicle_position_cache.get(vid)
            if position is None:
                position = traci.vehicle.getPosition3D(vid)
            x, y, z = position
            if lonlat_enabled:
                lon, lat = traci.simulation.convertGeo(x, y)
            else:
                lon = lat = 0.0
            sumo_angle = traci.vehicle.getAngle(vid)
            try:
                sumo_slope = traci.vehicle.getSlope(vid)
            except Exception:
                sumo_slope = 0.0
            orientation = math.radians((90.0 - sumo_angle) % 360.0)
            static = static_attrs.get(vid)
            if static is None:
                static = (
                    traci.vehicle.getLength(vid),
                    traci.vehicle.getWidth(vid),
                    traci.vehicle.getHeight(vid),
                    traci.vehicle.getTypeID(vid),
                )
                static_attrs[vid] = static
            last_orientation, last_time = last_orientations.get(vid, (orientation, simulation_time))
            dt = simulation_time - last_time
            if dt > 0:
                dtheta = orientation - last_orientation
                angular_velocity = math.atan2(math.sin(dtheta), math.cos(dtheta)) / dt
            else:
                angular_velocity = 0.0
            last_orientations[vid] = (orientation, simulation_time)
            vehicle_state = {
                "x": x,
                "y": y,
                "z": z,
                "lane_id": "",
                "lane_position": 0.0,
                "lateral_offset": 0.0,
                "reconstructed_x": 0.0,
                "reconstructed_y": 0.0,
                "reconstructed_z": 0.0,
                "reconstructed_position_valid": False,
                "lon": lon,
                "lat": lat,
                "sumo_angle": sumo_angle,
                "sumo_slope": sumo_slope,
                "length": static[0],
                "width": static[1],
                "height": static[2],
                "speed": traci.vehicle.getSpeed(vid),
                "orientation": orientation,
                "acceleration": traci.vehicle.getAcceleration(vid),
                "angular_velocity": angular_velocity,
                "type": static[3],
            }
            if self.lane_relative_position_enabled:
                self._populate_lane_relative_position(vid, vehicle_state)
            self._populate_physics_action_state(vid, vehicle_state)
            vehicles[vid] = vehicle_state

        simulation_state.agent_details["vehicle"] = vehicles

        # Add VRU states
        # Get current vehicle and person lists to determine actual object type
        current_vehicle_list = traci.vehicle.getIDList()
        current_person_list = traci.person.getIDList()

        vrus = {}
        for vru_id in vru_ids:
            # Determine if this VRU is actually a vehicle or person
            if vru_id in current_vehicle_list:
                # VRU is actually a vehicle (disguised as pedestrian)
                domain = traci.vehicle
            elif vru_id in current_person_list:
                # VRU is actually a person
                domain = traci.person
            else:
                # VRU ID not found in either list, log warning and skip
                self.logger.warning(f"VRU ID {vru_id} not found in vehicle or person lists, skipping")
                continue

            x, y, z = domain.getPosition3D(vru_id)
            if lonlat_enabled:
                lon, lat = traci.simulation.convertGeo(x, y)
            else:
                lon = lat = 0.0
            sumo_angle = domain.getAngle(vru_id)
            orientation = math.radians((90.0 - sumo_angle) % 360.0)
            angular_velocity = 0.0
            if domain is traci.vehicle:
                last_orientation, last_time = last_orientations.get(vru_id, (orientation, simulation_time))
                dt = simulation_time - last_time
                if dt > 0:
                    dtheta = orientation - last_orientation
                    angular_velocity = math.atan2(math.sin(dtheta), math.cos(dtheta)) / dt
                last_orientations[vru_id] = (orientation, simulation_time)
                acceleration = domain.getAcceleration(vru_id)
            else:
                acceleration = (
                    domain.getAcceleration(vru_id)
                    if hasattr(domain, "getAcceleration")
                    else 0.0
                )

            vrus[vru_id] = {
                "x": x,
                "y": y,
                "z": z,
                "lane_id": "",
                "lane_position": 0.0,
                "lateral_offset": 0.0,
                "reconstructed_x": 0.0,
                "reconstructed_y": 0.0,
                "reconstructed_z": 0.0,
                "reconstructed_position_valid": False,
                "lon": lon,
                "lat": lat,
                "sumo_angle": sumo_angle,
                "length": domain.getLength(vru_id),
                "width": domain.getWidth(vru_id),
                "height": domain.getHeight(vru_id),
                "speed": domain.getSpeed(vru_id),
                "orientation": orientation,
                "acceleration": acceleration,
                "angular_velocity": angular_velocity,
                "type": domain.getTypeID(vru_id),
            }

        simulation_state.agent_details["vru"] = vrus

        # Add construction objects
        construction_objects = {}
        for cid in construction_ids:
            x, y, z = traci.vehicle.getPosition3D(cid)
            if lonlat_enabled:
                lon, lat = traci.simulation.convertGeo(x, y)
            else:
                lon = lat = 0.0
            sumo_angle = traci.vehicle.getAngle(cid)
            construction_objects[cid] = {
                "x": x,
                "y": y,
                "z": z,
                "lane_id": "",
                "lane_position": 0.0,
                "lateral_offset": 0.0,
                "reconstructed_x": 0.0,
                "reconstructed_y": 0.0,
                "reconstructed_z": 0.0,
                "reconstructed_position_valid": False,
                "lon": lon,
                "lat": lat,
                "sumo_angle": sumo_angle,
                "length": traci.vehicle.getLength(cid),
                "width": traci.vehicle.getWidth(cid),
                "height": traci.vehicle.getHeight(cid),
                "speed": traci.vehicle.getSpeed(cid),
                "orientation": math.radians((90.0 - sumo_angle) % 360.0),
                "acceleration": traci.vehicle.getAcceleration(cid),
                "angular_velocity": 0.0,
                "type": traci.vehicle.getTypeID(cid),
            }

        simulation_state.construction_objects = construction_objects

        # Add traffic light states. The program/parameter block is static per
        # TLS, so it is resolved from the SUMO net and JSON-encoded exactly
        # once; only the current signal string changes per step.
        if self._tls_static_cache is None:
            self._tls_static_cache = {}
            for tl_id in traci.trafficlight.getIDList():
                tls_information = {
                    "programs": {}
                }
                tls = self.simulator.sumo_net.getTLS(tl_id)
                programs = tls.getPrograms()
                for program_id, program in programs.items():
                    # Get the program parameters
                    program_parameters = program.getParams()
                    tls_information["programs"][program_id] = {
                        "parameters": program_parameters
                    }
                self._tls_static_cache[tl_id] = json.dumps(tls_information)

        traffic_lights = {}
        for tl_id, information in self._tls_static_cache.items():
            sumo_signal = SUMOSignal()
            sumo_signal.x, sumo_signal.y = 0, 0
            sumo_signal.tls = traci.trafficlight.getRedYellowGreenState(tl_id)
            sumo_signal.information = information
            traffic_lights[tl_id] = sumo_signal

        simulation_state.traffic_light_details = traffic_lights

        # Add construction zone shapes
        if self.construction_zone_shapes is None and simulator.env.static_adversity is not None and simulator.env.static_adversity.adversities is not None:
            self.construction_zone_shapes = {}
            for adversity in simulator.env.static_adversity.adversities:
                if isinstance(adversity, ConstructionAdversity):
                    lane_shape = traci.lane.getShape(adversity._lane_id)
                    if lane_shape: # convert to list of lists
                        lane_shape = interpolate_by_distance(lane_shape, 2.0)
                        lane_index = int(adversity._lane_id.split("_")[-1])
                        edge_id = traci.lane.getEdgeID(adversity._lane_id)
                        if lane_index == 0:
                            # From right to left
                            direction = 1
                        elif lane_index == traci.edge.getLaneNumber(edge_id) - 1:
                            # From left to right
                            direction = -1
                        else:
                            # Middle lane, no construction zone
                            continue
                        construction_zone_shape = generate_construction_zone_shape(lane_shape, traci.lane.getWidth(adversity._lane_id), direction)
                        self.construction_zone_shapes[adversity._lane_id] = construction_zone_shape

        simulation_state.construction_zone_details = self.construction_zone_shapes
        return simulation_state

    # ------------------------------------------------------------------
    # agent command application
    # ------------------------------------------------------------------
    def _apply_agent_command(self, command):
        """Apply a parsed AgentCommand to the running simulation."""
        try:
            if command.agent_id != '':
                if command.agent_type not in ["vehicle", "vru"]:
                    self.logger.error(f"Invalid agent type: {command.agent_type}")
                    return False
                if command.agent_id in self.controlled_agents_each_step:
                    self.logger.debug(f"Agent {command.agent_id} is already controlled")
                    return True
                self.controlled_agents_each_step.add(command.agent_id)
                if command.command_type == "set_state":
                    # Check that exactly one of position or lonlat is present
                    has_position = "position" in command.data
                    has_lonlat = "lonlat" in command.data
                    if not (has_position ^ has_lonlat):  # XOR operation ensures exactly one is True
                        self.logger.error("Must specify exactly one of position or lonlat")
                        return False
                    if "position" in command.data:
                        x, y = command.data["position"]
                    elif "lonlat" in command.data:
                        lon, lat = command.data["lonlat"]
                        x, y = traci.simulation.convertGeo(lon, lat, fromGeo=True)
                    if (
                        command.agent_type == "vehicle"
                        and "source_carla_frame" in command.data
                        and self._is_physics_feedback_actor(command.agent_id)
                    ):
                        return self._apply_physics_external_state(command, x, y)
                    if command.agent_type == "vehicle":
                        # 3-cosim fix: keepRoute=0 (snap to closest lane in the network),
                        # not 2 (free / off-road). With keepRoute=2 an externally-driven
                        # vehicle (e.g. the Autoware ego mirrored as SUMO "AV" via control_av)
                        # lands slightly off the lane centerline -> getLaneID()=="" -> it drops
                        # out of the AV context subscription -> NADE stops controlling traffic
                        # around it -> background vehicles no longer yield and rear-end the ego.
                        # keepRoute=0 keeps the AV on a lane so SUMO traffic avoids it.
                        traci.vehicle.moveToXY(
                            command.agent_id, "", 0, x, y, command.data.get("sumo_angle", 0), 0
                        )

                        # 3-cosim fix (dense maps): right after moveToXY, append one
                        # successor edge so the externally-driven AV's route is never a single
                        # terminal edge. With keepRoute=0 a dense network can map the AV onto an
                        # off-route edge, collapsing its route to that one edge; the AV then reaches
                        # that edge's end, SUMO retires it as "arrived", NADE stops with
                        # finish_reason "AV_left", and the cosim crashes. The AV's pose is driven
                        # entirely by moveToXY (it mirrors the Autoware ego), so this 2-edge route is
                        # only a decoy to keep it alive -- NOT a fixed plan, which is correct because
                        # the Autoware ego chooses its path dynamically.
                        # (No getIDList membership pre-check: materializing the
                        # full id tuple every step is O(total vehicles), and the
                        # try/except below already tolerates a missing AV.)
                        if command.agent_id == "AV":
                            try:
                                cur = traci.vehicle.getRoadID("AV")
                                if cur and not cur.startswith(":"):  # skip junction-internal edges
                                    nxt = ""
                                    for lk in traci.lane.getLinks(traci.vehicle.getLaneID("AV")):
                                        if lk and lk[0]:
                                            e = traci.lane.getEdgeID(lk[0])
                                            if e and not e.startswith(":"):
                                                nxt = e
                                                break
                                    if nxt:
                                        traci.vehicle.setRoute("AV", [cur, nxt])
                            except Exception:
                                pass

                        if "speed" in command.data:
                            traci.vehicle.setPreviousSpeed(command.agent_id, command.data["speed"])
                    else:  # VRU type
                        # Check if VRU is actually a vehicle or person
                        current_vehicle_list = traci.vehicle.getIDList()
                        current_person_list = traci.person.getIDList()

                        if command.agent_id in current_vehicle_list:
                            # VRU is actually a vehicle (disguised as pedestrian)
                            traci.vehicle.moveToXY(
                                command.agent_id, "", 0, x, y, command.data.get("sumo_angle", 0), 2
                            )
                            if "speed" in command.data:
                                traci.vehicle.setPreviousSpeed(command.agent_id, command.data["speed"])
                        elif command.agent_id in current_person_list:
                            # VRU is actually a person
                            traci.person.moveToXY(
                                command.agent_id, "", x, y, command.data.get("sumo_angle", 0), 2
                            )
                            if "speed" in command.data:
                                traci.person.setSpeed(command.agent_id, command.data["speed"])
                        else:
                            self.logger.error(f"VRU ID {command.agent_id} not found in vehicle or person lists")
                            return False


                if self.logger.isEnabledFor(logging.DEBUG):
                    # Guarded: the model_dump_json() alone is measurable at one
                    # command per step, and this fires on the hot path.
                    self.logger.debug(f"Agent command executed: {command.model_dump_json()}")
                return True

        except Exception as e:
            self.logger.error(f"Error handling agent command: {e}")
            if (
                command.agent_type == "vehicle"
                and "source_carla_frame" in command.data
                and self._is_physics_feedback_actor(command.agent_id)
            ):
                raise
            return False

    # ------------------------------------------------------------------
    # internal state helpers
    # ------------------------------------------------------------------
    def _set_status(self, status: str):
        with self._lock:
            self._status = status

    def _finish(self, status: str):
        with self._lock:
            self._status = status
        # Release a client waiting on a step that will never come.
        self._step_done.set()

    def _result_locked(self) -> TickResult:
        return TickResult(
            status=self._status,
            state=self._state,
            completed_sumo_time=self._completed_sumo_time,
            completed_tick_count=self._completed_tick_count,
        )
