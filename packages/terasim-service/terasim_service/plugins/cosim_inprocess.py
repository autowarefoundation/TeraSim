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
    apply_external_state_lateral_maneuver_target,
    build_active_maneuver_lane_corridor,
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
            offset_val = -lane_width / 2

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

    def __init__(
        self,
        plugin: "TeraSimCoSimInProcessPlugin",
        resolved: Optional[TickResult] = None,
    ):
        self._plugin = plugin
        self._resolved = resolved  # pre-resolved for pass-through (ended) calls

    def result(self, timeout: float = 300.0) -> TickResult:
        if self._resolved is not None:
            return self._resolved
        if not self._plugin._step_done.wait(timeout=timeout):
            raise TimeoutError(f"SUMO step did not complete within {timeout:.0f}s")
        return self._plugin.get_result()


class TeraSimCoSimInProcessPlugin(BasePlugin):
    """Co-simulation plugin driven by a same-process co-sim client."""

    # Longest time function_before_env_step keeps waiting for a tick request
    # before auto-stopping.
    IDLE_TIMEOUT_S = 600.0

    # cadence (in steps) for pruning per-vehicle caches of departed ids
    CACHE_PRUNE_EVERY_STEPS = 1200

    LATERAL_SPEED_EPSILON = 0.001
    LATERAL_START_DELTA_D_EPSILON = 0.002
    LATERAL_START_CONFIRM_FRAMES = 3
    LATERAL_STOP_DELTA_D_EPSILON = 0.0005
    LATERAL_STOP_CONFIRM_FRAMES = 5
    LATERAL_ARMED_IDLE_FRAMES = 3
    LATERAL_CENTER_TOLERANCE = 0.01
    LATERAL_DIRECTION_MIN_ALIGNMENT = 0.5
    LATERAL_DIRECTION_WARNING_INTERVAL = 100
    CANONICAL_TARGET_LATERAL_JUMP_LIMIT = 0.10

    def __init__(
        self,
        simulation_uuid: str,
        plugin_config: dict = DEFAULT_COSIM_PLUGIN_CONFIG,
        base_dir: str = "output",
        auto_run: bool = False,
        control_av: bool = True,
        av_control_mode: str = "external",
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
        self.control_av = control_av
        self.av_control_mode = str(av_control_mode).strip().lower()
        if self.av_control_mode not in {"external", "sumo"}:
            raise ValueError(f"invalid AV control mode: {self.av_control_mode!r}")
        if self.av_control_mode == "sumo" and self.control_av:
            raise ValueError("SUMO-controlled AV cannot use external control_av commands")
        self.av_nde_lane_change_request = None
        self.av_nde_lane_change_request_count = 0
        self.av_nde_lane_change_decision = {}
        self._av_nde_controller_identity = None

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
            actor_id.strip() for actor_id in feedback_actor_value.split(",") if actor_id.strip()
        }
        self.physics_feedback_mode = (
            os.getenv("CARLA_COSIM_ACKERMANN_FEEDBACK_MODE", "off").strip().lower()
        )
        self.physics_background_invalid_policy = (
            os.getenv("CARLA_COSIM_ACKERMANN_BACKGROUND_INVALID_POLICY", "demote").strip().lower()
        )
        self.physics_assimilation_mode = (
            os.getenv("CARLA_COSIM_ACKERMANN_FEEDBACK_ASSIMILATION_MODE", "legacy").strip().lower()
        )
        self.physics_external_state_enabled = (
            self.physics_feedback_mode == "apply"
            and self.physics_assimilation_mode == "external_state"
            and bool(self.physics_feedback_actor_ids)
        )
        self.physics_lateral_maneuver_enabled = self._parse_bool_env(
            "CARLA_COSIM_ACKERMANN_LATERAL_MANEUVER_ENABLED", False
        )
        self.physics_lateral_maneuver_target_enabled = self._parse_bool_env(
            "CARLA_COSIM_ACKERMANN_LATERAL_MANEUVER_TARGET_ENABLED", False
        )
        if (
            self.physics_lateral_maneuver_target_enabled
            and not self.physics_lateral_maneuver_enabled
        ):
            raise ValueError(
                "CARLA_COSIM_ACKERMANN_LATERAL_MANEUVER_TARGET_ENABLED requires "
                "CARLA_COSIM_ACKERMANN_LATERAL_MANEUVER_ENABLED"
            )
        self.physics_internal_lane_transition_assist_enabled = self._parse_bool_env(
            "CARLA_COSIM_ACKERMANN_INTERNAL_LANE_TRANSITION_ASSIST_ENABLED", False
        )
        self.physics_step_length = max(0.0, float(os.getenv("CARLA_COSIM_STEP_LENGTH", "0.05")))
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
        self.physics_feedback_actor_membership = set()
        self.physics_feedback_generations = {}
        self.physics_lateral_direction_states = {}
        self.physics_rolling_path_states = {}
        self.physics_lane_geometry_cache = {}
        self.physics_lookahead_path_cache = {}
        self.physics_lookahead_min_distance = max(
            0.1, float(os.getenv("CARLA_COSIM_ACKERMANN_LOOKAHEAD_MIN_DISTANCE", "7.0"))
        )
        self.physics_lookahead_max_distance = max(
            self.physics_lookahead_min_distance,
            float(os.getenv("CARLA_COSIM_ACKERMANN_LOOKAHEAD_MAX_DISTANCE", "15.0")),
        )
        self.physics_steering_preview_min_distance = float(
            os.getenv("CARLA_COSIM_ACKERMANN_STEERING_PREVIEW_MIN_DISTANCE", "2.8")
        )
        self.physics_steering_preview_time_headway = float(
            os.getenv("CARLA_COSIM_ACKERMANN_STEERING_PREVIEW_TIME_HEADWAY", "0.5")
        )
        self.physics_steering_preview_max_distance = float(
            os.getenv("CARLA_COSIM_ACKERMANN_STEERING_PREVIEW_MAX_DISTANCE", "5.0")
        )
        preview_settings = (
            self.physics_steering_preview_min_distance,
            self.physics_steering_preview_time_headway,
            self.physics_steering_preview_max_distance,
        )
        if (
            not all(math.isfinite(value) for value in preview_settings)
            or self.physics_steering_preview_min_distance <= 0.0
            or self.physics_steering_preview_time_headway < 0.0
            or self.physics_steering_preview_max_distance
            < self.physics_steering_preview_min_distance
        ):
            raise ValueError("invalid Ackermann steering preview configuration")
        if self.physics_external_state_enabled:
            self.logger.info(
                "Physical co-sim Phase A enabled: actors=%s strict_lane=%s "
                "lateral_maneuver=%s lateral_maneuver_target=%s "
                "internal_lane_transition_assist=%s",
                sorted(self.physics_feedback_actor_ids),
                self.physics_strict_lane_hint,
                self.physics_lateral_maneuver_enabled,
                self.physics_lateral_maneuver_target_enabled,
                self.physics_internal_lane_transition_assist_enabled,
            )

        # Cache construction zone shapes
        self.construction_zone_shapes = None

        # Initialize last orientations cache
        self.last_orientations = {}  # {vehicle_id: (last_orientation, last_time)}

        self.state_filter_enabled = self._parse_bool_env("TERASIM_COSIM_STATE_FILTER", False)
        self.state_filter_center_id = os.getenv("TERASIM_COSIM_STATE_FILTER_CENTER_ID", "AV")
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
        self.state_lonlat_enabled = self._parse_bool_env("TERASIM_COSIM_STATE_LONLAT", False)
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
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
                self._pending_commands = [AgentCommand.model_validate(c) for c in commands]
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

        simulator.start_pipeline.hook(
            f"{self.plugin_name}_before_env_start",
            self.function_before_env_start,
            priority=self.plugin_priority["before_env"]["start"],
        )
        simulator.start_pipeline.hook(
            f"{self.plugin_name}_after_env_start",
            self.function_after_env_start,
            priority=self.plugin_priority["after_env"]["start"],
        )
        simulator.step_pipeline.hook(
            f"{self.plugin_name}_before_env_step",
            self.function_before_env_step,
            priority=self.plugin_priority["before_env"]["step"],
        )
        simulator.step_pipeline.hook(
            f"{self.plugin_name}_after_env_step",
            self.function_after_env_step,
            priority=self.plugin_priority["after_env"]["step"],
        )
        simulator.stop_pipeline.hook(
            f"{self.plugin_name}_before_env_stop",
            self.function_before_env_stop,
            priority=self.plugin_priority["before_env"]["stop"],
        )
        simulator.stop_pipeline.hook(
            f"{self.plugin_name}_after_env_stop",
            self.function_after_env_stop,
            priority=self.plugin_priority["after_env"]["stop"],
        )

    def function_before_env_start(self, simulator: Simulator, ctx):
        self._reset_av_lane_change_state("simulation_start")
        self.physics_feedback_actor_membership.clear()
        getattr(self, "physics_feedback_generations", {}).clear()
        self.physics_lateral_direction_states.clear()
        self.physics_rolling_path_states.clear()
        self.av_nde_lane_change_request = None
        self.av_nde_lane_change_request_count = 0
        self.av_nde_lane_change_decision = {}
        self._av_nde_controller_identity = None
        self._set_status("initializing")
        self.logger.info(f"Simulation UUID: {self.simulation_uuid}, start initialization!")
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
            self.logger.info(f"Simulation UUID: {self.simulation_uuid}, finish initialization!")
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
        previous_membership = set(getattr(self, "physics_feedback_actor_membership", set()))
        self.physics_feedback_observations.clear()
        self.controlled_agents_each_step.clear()
        for command in commands:
            self._apply_agent_command(command)

        self._set_status("running")

        current_membership = set(self.physics_feedback_observations)
        self.physics_feedback_actor_membership = current_membership
        for actor_id in previous_membership - current_membership:
            self.physics_lateral_direction_states.pop(actor_id, None)
            self.physics_rolling_path_states.pop(actor_id, None)
            getattr(self, "physics_feedback_generations", {}).pop(actor_id, None)
            self.logger.info(
                "Physical feedback actor left active membership: actor=%s; "
                "cleared maneuver and rolling-path state",
                actor_id,
            )
        self.logger.debug("Simulation step started")
        return True

    def function_after_env_step(self, simulator: Simulator, ctx):
        try:
            self._sync_av_lane_change_lifecycle(simulator)
            self._capture_av_nde_lane_change_request(simulator)
            self._capture_av_nde_lane_change_decision(simulator)
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
        self._reset_av_lane_change_state("simulation_stop")
        return True

    def function_after_env_stop(self, simulator: Simulator, ctx):
        self.physics_lateral_direction_states.clear()
        self.physics_rolling_path_states.clear()
        getattr(self, "physics_feedback_generations", {}).clear()
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
            and (actor_id != "AV" or not getattr(self, "control_av", True))
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

    def _physics_external_state_lane_projection(
        self,
        actor_id,
        lane_id,
        position,
        sumo_angle,
        position_z,
        speed,
        requested_route,
    ):
        """Map feedback to the current lane, with a stopped-internal-lane escape.

        Phase A normally keeps the current SUMO lane authoritative. At the end
        of an internal lane, however, a vehicle that stopped during an
        emergency brake can enter a feedback loop: every CARLA pose is mapped
        back to the last centimetres of the internal lane, so Phase B never
        gets enough motion to cross onto the route successor. Once an AV is
        already inside the junction it is safe to transfer its unchanged world
        pose to the immediate route successor at very low speed.
        """
        geometry = self._physics_lane_geometry(lane_id)
        current_projection = select_route_aware_lane_projection(
            position,
            sumo_angle,
            [geometry],
            position_z=position_z,
            current_lane_id=lane_id,
            max_distance=self.physics_match_threshold,
            max_heading_error=180.0,
            prefer_current_lane=True,
        )
        if current_projection is None:
            return None
        current_projection["internal_lane_transition_assist"] = False
        if not getattr(self, "physics_internal_lane_transition_assist_enabled", False):
            return current_projection

        transition_distance = max(
            0.0,
            float(
                getattr(
                    self,
                    "physics_internal_lane_transition_distance",
                    0.25,
                )
            ),
        )
        transition_max_speed = max(
            0.0,
            float(
                getattr(
                    self,
                    "physics_internal_lane_transition_max_speed",
                    0.2,
                )
            ),
        )
        remaining = max(
            0.0,
            float(geometry["length"]) - float(current_projection["lane_position"]),
        )
        if (
            not lane_id.startswith(":")
            or speed > transition_max_speed
            or remaining > transition_distance
        ):
            return current_projection

        try:
            immediate_links = tuple(traci.lane.getLinks(lane_id) or ())
        except Exception:
            return current_projection
        successor_lane_ids = extract_next_link_lane_ids(immediate_links)
        successor_geometries = []
        for successor_lane_id in successor_lane_ids:
            if not successor_lane_id or successor_lane_id == lane_id:
                continue
            try:
                successor_geometry = self._physics_lane_geometry(successor_lane_id)
            except Exception:
                continue
            if successor_geometry["edge_id"] not in requested_route:
                continue
            successor_geometries.append(successor_geometry)

        successor_projection = select_route_aware_lane_projection(
            position,
            sumo_angle,
            successor_geometries,
            position_z=position_z,
            lane_switch_hysteresis=0.0,
            max_distance=self.physics_match_threshold,
            max_heading_error=90.0,
        )
        if (
            successor_projection is None
            or successor_projection["lane_position"] > transition_distance
        ):
            return current_projection

        successor_projection["internal_lane_transition_assist"] = True
        successor_projection["transition_from_lane_id"] = lane_id
        successor_projection["transition_remaining_distance"] = remaining
        return successor_projection

    def _apply_physics_external_state(self, command, x, y):  # noqa: C901
        """Phase A: assimilate a CARLA state without advancing SUMO time."""
        actor_id = command.agent_id
        data = command.data
        immediate_move = getattr(traci.vehicle, "moveToXYImmediate", None)
        if not callable(immediate_move):
            raise RuntimeError("physical co-sim requires patched SUMO moveToXYImmediate")

        position = (self._as_finite_float(x), self._as_finite_float(y))
        sumo_angle = self._as_finite_float(data.get("sumo_angle"))
        speed = self._as_finite_float(data.get("speed"))
        acceleration = self._as_finite_float(data.get("acceleration"))
        source_frame = data.get("source_carla_frame")
        physics_seed_only = bool(data.get("physics_seed_only", False))
        initialization_generation = data.get("physics_initialization_generation", 0)
        generation_valid = (
            isinstance(initialization_generation, int)
            and not isinstance(initialization_generation, bool)
            and initialization_generation >= 0
        )
        position_z = self._as_finite_float(data.get("z"))
        rear_axle_data = data.get("rear_axle_position")
        rear_axle_position = (
            None
            if not isinstance(rear_axle_data, (list, tuple)) or len(rear_axle_data) < 2
            else (
                self._as_finite_float(rear_axle_data[0]),
                self._as_finite_float(rear_axle_data[1]),
            )
        )
        front_to_control_offset = self._as_finite_float(data.get("front_to_rear_axle_distance"))
        if (
            None in position
            or rear_axle_position is None
            or None in rear_axle_position
            or front_to_control_offset is None
            or front_to_control_offset <= 0.0
            or sumo_angle is None
            or speed is None
            or speed < 0.0
            or acceleration is None
            or not isinstance(source_frame, int)
            or not generation_valid
        ):
            raise RuntimeError(
                f"invalid physical feedback state actor={actor_id} frame={source_frame}"
            )

        generations = getattr(self, "physics_feedback_generations", None)
        if generations is None:
            generations = {}
            self.physics_feedback_generations = generations
        previous_generation = generations.get(actor_id)
        if previous_generation is not None and previous_generation != initialization_generation:
            self.physics_lateral_direction_states.pop(actor_id, None)
            self.physics_rolling_path_states.pop(actor_id, None)
            self.logger.info(
                "Physical feedback initialization generation changed: "
                "actor=%s generation=%s->%s; cleared maneuver and rolling-path state",
                actor_id,
                previous_generation,
                initialization_generation,
            )
        generations[actor_id] = initialization_generation

        lane_id = traci.vehicle.getLaneID(actor_id)
        if not lane_id:
            raise RuntimeError(f"physical feedback actor has no current SUMO lane actor={actor_id}")
        requested_route = tuple(traci.vehicle.getRoute(actor_id))
        projection = self._physics_external_state_lane_projection(
            actor_id,
            lane_id,
            position,
            sumo_angle,
            position_z,
            speed,
            requested_route,
        )
        if projection is None:
            raise RuntimeError(
                "physical feedback current-lane mapping failed "
                f"actor={actor_id} lane={lane_id} position={position}"
            )

        # CARLA is authoritative for the complete front-centre pose. Do not
        # preserve SUMO's previous lateral reference during a maneuver.
        assimilated_position = position

        phase_a_time = traci.simulation.getTime()
        if projection.get("internal_lane_transition_assist"):
            traci.vehicle.moveTo(actor_id, projection["lane_id"], 0.0)
        immediate_move(
            actor_id,
            projection["edge_id"],
            projection["lane_index"],
            assimilated_position[0],
            assimilated_position[1],
            sumo_angle,
            1,
            self.physics_match_threshold,
            self.physics_strict_lane_hint,
        )
        if projection.get("internal_lane_transition_assist"):
            self.logger.info(
                "Physical feedback advanced stopped actor across internal lane "
                "boundary: actor=%s lane=%s->%s remaining=%.6g speed=%.6g",
                actor_id,
                projection.get("transition_from_lane_id"),
                projection["lane_id"],
                projection.get("transition_remaining_distance", 0.0),
                speed,
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
            observed_position[0] - assimilated_position[0],
            observed_position[1] - assimilated_position[1],
        )
        angle_error = abs((observed_angle - sumo_angle + 180.0) % 360.0 - 180.0)
        valid = bool(
            abs(observed_time - phase_a_time) <= 1e-9
            and position_error <= self.physics_position_tolerance
            and angle_error <= 1e-6
            and abs(observed_speed - speed) <= 1e-6
            and abs(observed_acceleration - acceleration) <= 1e-6
            and observed_route == requested_route
            and (not self.physics_strict_lane_hint or observed_lane_id == projection["lane_id"])
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
            "position": tuple(position),
            "sumo_assimilated_position": tuple(observed_position[:2]),
            "rear_axle_position": tuple(rear_axle_position),
            "front_to_control_offset": front_to_control_offset,
            "z": position_z,
            "requested_sumo_angle": sumo_angle,
            "sumo_angle": observed_angle,
            "speed": observed_speed,
            "acceleration": observed_acceleration,
            "requested_lane_id": projection["lane_id"],
            "lane_id": observed_lane_id,
            "lane_position": lane_position,
            "lane_length": lane_length,
            "phase_a_time": phase_a_time,
            "source_carla_frame": source_frame,
            "physics_seed_only": physics_seed_only,
            "initialization_generation": initialization_generation,
            "internal_lane_transition_assist": bool(
                projection.get("internal_lane_transition_assist")
            ),
            "lateral_reference_preserved": False,
            "preserved_sumo_lateral_offset": None,
        }
        return True

    def _reset_av_lane_change_state(self, reason):
        vehicle = getattr(self, "_av_nde_vehicle", None)
        controller = getattr(vehicle, "controller", None)
        if controller is not None:
            controller.reset_control_state(reason)
        model = getattr(vehicle, "decision_model", None)
        if model is not None:
            model._reset()
        self.av_nde_lane_change_request = None
        self.av_nde_lane_change_request_count = 0
        self.av_nde_lane_change_decision = {}
        self._av_nde_controller_identity = None
        self._av_nde_vehicle = None
        self.physics_lateral_direction_states.pop("AV", None)
        self.physics_rolling_path_states.pop("AV", None)

    def _sync_av_lane_change_lifecycle(self, simulator):
        env = getattr(simulator, "env", None)
        vehicles = getattr(env, "vehicle_list", {})
        vehicle = vehicles.get("AV") if hasattr(vehicles, "get") else None
        mode = (self.av_control_mode, getattr(env, "av_control_mode", None))
        old_vehicle = getattr(self, "_av_nde_vehicle", None)
        old_mode = getattr(self, "_av_nde_authority", mode)
        if old_vehicle is not None and (vehicle is not old_vehicle or mode != old_mode):
            self._reset_av_lane_change_state("av_exit_or_authority_changed")
        self._av_nde_vehicle = vehicle
        self._av_nde_authority = mode

    def _capture_av_nde_lane_change_decision(self, simulator):
        if self.av_control_mode != "sumo":
            self.av_nde_lane_change_decision = {}
            return
        env = getattr(simulator, "env", None)
        vehicle_list = getattr(env, "vehicle_list", {})
        vehicle = vehicle_list.get("AV") if hasattr(vehicle_list, "get") else None
        decision_model = getattr(vehicle, "decision_model", None)
        decision = getattr(decision_model, "last_lane_change_decision", None)
        self.av_nde_lane_change_decision = dict(decision or {})
        if self.av_nde_lane_change_decision:
            record = dict(self.av_nde_lane_change_decision)
            record["request_count"] = self.av_nde_lane_change_request_count
            record["issued_request"] = self.av_nde_lane_change_request
            with open(os.path.join(self.base_dir, "av_lane_change_decisions.jsonl"), "a") as stream:
                stream.write(json.dumps(record) + "\n")
        fatal_reason = str(self.av_nde_lane_change_decision.get("fatal_reason") or "")
        if fatal_reason:
            raise RuntimeError(
                f"{fatal_reason}: lane="
                f"{self.av_nde_lane_change_decision.get('source_lane_id', '')} "
                f"target={self.av_nde_lane_change_decision.get('target_lane_id', '')} "
                f"remaining={self.av_nde_lane_change_decision.get('lane_remaining')} "
                f"required={self.av_nde_lane_change_decision.get('required_distance')}"
            )

    def _capture_av_nde_lane_change_request(self, simulator):
        if self.av_control_mode != "sumo":
            return
        env = getattr(simulator, "env", None)
        vehicle_list = getattr(env, "vehicle_list", {})
        vehicle = vehicle_list.get("AV") if hasattr(vehicle_list, "get") else None
        controller = getattr(vehicle, "controller", None)
        if controller is None:
            return
        controller_identity = id(controller)
        if self._av_nde_controller_identity != controller_identity:
            self._av_nde_controller_identity = controller_identity
            self.av_nde_lane_change_request = None
            self.av_nde_lane_change_request_count = 0
        request = getattr(controller, "last_lane_change_request", None)
        request_count = int(getattr(controller, "lane_change_request_count", 0) or 0)
        if request is None or request_count <= self.av_nde_lane_change_request_count:
            return
        self.av_nde_lane_change_request = dict(request)
        self.av_nde_lane_change_request_count = request_count
        self.logger.info(
            "Captured AV NDE lane-change request: count=%s direction=%s source=%s target=%s",
            request_count,
            request.get("direction", "none"),
            request.get("source_lane_id", ""),
            request.get("target_lane_id", ""),
        )

    def _fresh_av_nde_lane_change_request(self, vehicle_id, current_lane_id):
        if vehicle_id != "AV" or self.av_control_mode != "sumo":
            return None
        request = self.av_nde_lane_change_request
        if not request:
            return None
        state = self.physics_lateral_direction_states.get(vehicle_id, {})
        count = int(request.get("count", 0))
        if count and count <= int(state.get("request_generation_consumed", 0)):
            return None
        try:
            age = float(traci.simulation.getTime()) - float(request["simulation_time"])
        except (KeyError, TypeError, ValueError):
            return None
        max_age = max(1e-6, 2.0 * self.physics_step_length)
        if age < -1e-6 or age > max_age + 1e-6:
            return None
        source_lane_id = str(request.get("source_lane_id") or "")
        target_lane_id = str(request.get("target_lane_id") or "")
        if current_lane_id not in {source_lane_id, target_lane_id}:
            return None
        direction = str(request.get("direction") or "none")
        if direction not in {"left", "right"}:
            return None
        return request

    def _av_nde_controller_diagnostics(self):
        diagnostics = {
            "busy": False,
            "command_end_reason": "",
        }
        env = getattr(getattr(self, "simulator", None), "env", None)
        vehicle_list = getattr(env, "vehicle_list", {})
        vehicle = vehicle_list.get("AV") if hasattr(vehicle_list, "get") else None
        controller = getattr(vehicle, "controller", None)
        if controller is not None:
            diagnostics["busy"] = bool(getattr(controller, "is_busy", False))
            diagnostics["command_end_reason"] = str(
                getattr(controller, "last_command_end_reason", "") or ""
            )
        return diagnostics

    def _physics_lane_change_action(self, vehicle_id, current_lane_id):
        nde_request = self._fresh_av_nde_lane_change_request(vehicle_id, current_lane_id)
        try:
            left_state = traci.vehicle.getLaneChangeState(vehicle_id, 1)[1]
            right_state = traci.vehicle.getLaneChangeState(vehicle_id, -1)[1]
        except Exception:
            left_state = right_state = 0
        wants_left = bool(left_state & traci.constants.LCA_LEFT)
        wants_right = bool(right_state & traci.constants.LCA_RIGHT)
        if vehicle_id == "AV" and self.av_control_mode == "sumo":
            # Raw/post-TraCI intent is diagnostic only for an explicitly
            # controlled SUMO AV. Only the accepted one-shot NDE request may
            # authorize armed/active state.
            raw_direction = (
                "left"
                if wants_left and not wants_right
                else "right" if wants_right and not wants_left else "none"
            )
            if nde_request is not None:
                return (
                    str(nde_request["direction"]),
                    str(nde_request.get("target_lane_id") or ""),
                    "terasim_nde_request",
                    False,
                )
            return raw_direction, "", "none", False
        if wants_left != wants_right:
            direction = "left" if wants_left else "right"
            if nde_request is not None and nde_request.get("direction") == direction:
                return (
                    direction,
                    str(nde_request.get("target_lane_id") or ""),
                    "terasim_nde_request",
                    True,
                )
            return direction, "", "sumo_intent", True
        if nde_request is not None:
            return (
                str(nde_request["direction"]),
                str(nde_request.get("target_lane_id") or ""),
                "terasim_nde_request",
                False,
            )
        return "none", "", "none", False

    def _warn_unverified_physics_lookahead_path(self, key, lane_ids, compiled):
        truncated_index = compiled.get("shape_join_truncated_index")
        if truncated_index is None:
            return
        previous_lane_id = lane_ids[truncated_index - 1] if truncated_index > 0 else ""
        truncated_lane_id = lane_ids[truncated_index] if truncated_index < len(lane_ids) else ""
        self.logger.warning(
            "Physical co-sim lookahead path truncated at unverified lane "
            "boundary: lanes=%s boundary=%s->%s endpoint_distance=%.6g "
            "projection_error=%s",
            key,
            previous_lane_id,
            truncated_lane_id,
            compiled.get("shape_join_endpoint_distance") or 0.0,
            compiled.get("shape_join_projection_error"),
        )

    @staticmethod
    def _physics_direct_link_via_lane_id(source_lane_id, destination_lane_id):
        """Return the unique immediate via lane for one exact SUMO link.

        ``vehicle.getNextLinks`` reports the first internal lane and final
        destination, but a connection may contain another internal via lane.
        Follow ``lane.getLinks`` only when its destination exactly matches the
        already-authoritative destination; ambiguous links remain unexpanded
        and are rejected later by the geometry continuity invariant.
        """
        try:
            links = tuple(traci.lane.getLinks(source_lane_id) or ())
        except Exception:
            return None

        via_lane_ids = set()
        matched = False
        for link in links:
            if not link or link[0] != destination_lane_id:
                continue
            matched = True
            via_lane_id = ""
            if len(link) > 4 and isinstance(link[4], str):
                via_lane_id = link[4]
            elif len(link) > 1 and isinstance(link[1], str):
                via_lane_id = link[1]
            via_lane_ids.add(via_lane_id)
        if not matched or len(via_lane_ids) != 1:
            return None
        return next(iter(via_lane_ids))

    def _physics_expand_nested_via_lanes(self, source_lane_id, destination_lane_id):
        """Resolve nested internal lanes without inferring any geometry."""
        expanded = []
        current_lane_id = source_lane_id
        visited = {source_lane_id, destination_lane_id}
        for _ in range(8):
            via_lane_id = self._physics_direct_link_via_lane_id(
                current_lane_id,
                destination_lane_id,
            )
            if not via_lane_id:
                return tuple(expanded)
            if via_lane_id in visited:
                return ()
            expanded.append(via_lane_id)
            visited.add(via_lane_id)
            current_lane_id = via_lane_id
        return ()

    @staticmethod
    def _physics_link_destination_and_via(link):
        """Parse one lane/getNextLinks tuple without treating it as topology."""
        if not link or not isinstance(link[0], str) or not link[0]:
            return None
        destination_lane_id = link[0]
        via_lane_id = ""
        if len(link) > 4 and isinstance(link[4], str):
            via_lane_id = link[4]
        elif len(link) > 1 and isinstance(link[1], str):
            via_lane_id = link[1]
        return destination_lane_id, via_lane_id

    def _physics_direct_route_candidates(self, source_lane_id, successor_edge_id=None):
        """Return only successors proven by lane.getLinks(source_lane_id)."""
        try:
            links = tuple(traci.lane.getLinks(source_lane_id) or ())
        except Exception:
            return ()
        candidates = []
        for link in links:
            parsed = self._physics_link_destination_and_via(link)
            if parsed is None:
                continue
            destination_lane_id, reported_via_lane_id = parsed
            try:
                destination_edge_id = traci.lane.getEdgeID(destination_lane_id)
            except Exception:
                continue
            if successor_edge_id and destination_edge_id != successor_edge_id:
                continue
            via_lane_ids = self._physics_expand_nested_via_lanes(
                source_lane_id,
                destination_lane_id,
            )
            if reported_via_lane_id and (
                not via_lane_ids or via_lane_ids[0] != reported_via_lane_id
            ):
                continue
            candidates.append(
                {
                    "source_lane_id": source_lane_id,
                    "destination_lane_id": destination_lane_id,
                    "destination_edge_id": destination_edge_id,
                    "via_lane_ids": tuple(via_lane_ids),
                    "lane_ids": (
                        source_lane_id,
                        *tuple(via_lane_ids),
                        destination_lane_id,
                    ),
                }
            )
        return tuple(candidates)

    def _physics_planned_link_selector(self, vehicle_id):
        """Read SUMO's plan as a selector, never as proof of a connection."""
        try:
            links = tuple(traci.vehicle.getNextLinks(vehicle_id) or ())
        except Exception:
            return ()
        selectors = []
        for link in links:
            parsed = self._physics_link_destination_and_via(link)
            if parsed is not None and parsed not in selectors:
                selectors.append(parsed)
        return tuple(selectors)

    def _physics_route_successor_edge_id(self, vehicle_id, current_lane_id, selectors):
        try:
            route = tuple(traci.vehicle.getRoute(vehicle_id) or ())
            route_index = int(traci.vehicle.getRouteIndex(vehicle_id))
            current_edge_id = traci.lane.getEdgeID(current_lane_id)
        except Exception:
            route = ()
            route_index = -1
            current_edge_id = ""

        if route and 0 <= route_index < len(route):
            if not current_lane_id.startswith(":") and route[route_index] == current_edge_id:
                if route_index + 1 < len(route):
                    return route[route_index + 1]
            elif current_lane_id.startswith(":") and route_index + 1 < len(route):
                # During an internal lane SUMO retains the source route index.
                return route[route_index + 1]
            for index in range(max(0, route_index), len(route) - 1):
                if route[index] == current_edge_id:
                    return route[index + 1]

        for destination_lane_id, _via_lane_id in selectors:
            try:
                return traci.lane.getEdgeID(destination_lane_id)
            except Exception:
                continue
        return ""

    @staticmethod
    def _physics_select_direct_candidate(candidates, selectors):
        if len(candidates) == 1:
            return candidates[0], "unique_direct_connection"
        if not candidates:
            return None, "no_direct_connection"
        selected = []
        for candidate in candidates:
            for destination_lane_id, via_lane_id in selectors:
                if destination_lane_id != candidate["destination_lane_id"]:
                    continue
                candidate_via = candidate["via_lane_ids"][0] if candidate["via_lane_ids"] else ""
                if via_lane_id and candidate_via != via_lane_id:
                    continue
                selected.append(candidate)
                break
        if len(selected) == 1:
            return selected[0], "planned_selector_verified_direct_connection"
        return None, "ambiguous_direct_connections"

    def _physics_adjacent_intent_lane_id(self, current_lane_id, intent):
        if current_lane_id.startswith(":") or intent not in {"left", "right"}:
            return ""
        lane_index = self._physics_lane_index(current_lane_id)
        target_index = lane_index + (1 if intent == "left" else -1)
        if target_index < 0:
            return ""
        try:
            edge_id = traci.lane.getEdgeID(current_lane_id)
            target_lane_id = f"{edge_id}_{target_index}"
            if traci.lane.getEdgeID(target_lane_id) != edge_id:
                return ""
        except Exception:
            return ""
        return target_lane_id

    def _physics_extend_verified_route_sequence(self, vehicle_id, candidate, selectors):
        """Extend a direct candidate along route edges using verified links only."""
        lane_ids = list(candidate["lane_ids"])
        try:
            route = tuple(traci.vehicle.getRoute(vehicle_id) or ())
            route_index = int(traci.vehicle.getRouteIndex(vehicle_id))
        except Exception:
            return tuple(lane_ids), 0, ""
        if not route:
            return tuple(lane_ids), 0, ""

        destination_edge_id = candidate["destination_edge_id"]
        destination_route_index = None
        for index in range(max(0, route_index), len(route)):
            if route[index] == destination_edge_id:
                destination_route_index = index
                break
        if destination_route_index is None:
            return tuple(lane_ids), 0, "destination_not_in_route"

        current_lane_id = candidate["destination_lane_id"]
        extension_count = 0
        for next_edge_id in route[destination_route_index + 1 : destination_route_index + 4]:
            candidates = self._physics_direct_route_candidates(
                current_lane_id,
                next_edge_id,
            )
            next_candidate, reason = self._physics_select_direct_candidate(
                candidates,
                selectors,
            )
            if next_candidate is None:
                error = reason if reason.startswith("ambiguous") else ""
                return tuple(lane_ids), extension_count, error
            lane_ids.extend(next_candidate["lane_ids"][1:])
            current_lane_id = next_candidate["destination_lane_id"]
            extension_count += 1
        return tuple(lane_ids), extension_count, ""

    @staticmethod
    def _physics_on_final_route_edge(vehicle_id, current_lane_id):
        try:
            route = tuple(traci.vehicle.getRoute(vehicle_id))
            index = int(traci.vehicle.getRouteIndex(vehicle_id))
            return bool(
                route
                and index == len(route) - 1
                and traci.lane.getEdgeID(current_lane_id) == route[index]
            )
        except Exception:
            return False

    def _physics_resolve_route_lane_sequence(self, vehicle_id, current_lane_id):  # noqa: C901
        """Resolve a continuous immediate route sequence from SUMO topology.

        getNextLinks may describe a different lane while SUMO plans a
        route-required lane change. It therefore selects only among candidates
        independently proven with lane.getLinks.
        """
        selectors = self._physics_planned_link_selector(vehicle_id)
        if self._physics_on_final_route_edge(vehicle_id, current_lane_id):
            return {
                "actual_lane_id": current_lane_id,
                "reference_lane_id": current_lane_id,
                "lane_ids": (current_lane_id,),
                "successor_edge_id": "",
                "selection_reason": "final_route_edge",
                "error": "",
                "planned_selectors": selectors,
            }
        successor_edge_id = self._physics_route_successor_edge_id(
            vehicle_id,
            current_lane_id,
            selectors,
        )
        candidates = self._physics_direct_route_candidates(
            current_lane_id,
            successor_edge_id or None,
        )
        candidate, reason = self._physics_select_direct_candidate(candidates, selectors)
        reference_lane_id = current_lane_id
        if candidate is None and current_lane_id.startswith(":"):
            immediate_candidates = self._physics_direct_route_candidates(current_lane_id)
            immediate_candidate, immediate_reason = self._physics_select_direct_candidate(
                immediate_candidates,
                selectors,
            )
            if immediate_candidate is not None:
                candidate = immediate_candidate
                reason = f"internal_immediate:{immediate_reason}"
            elif immediate_reason == "ambiguous_direct_connections":
                reason = "ambiguous_internal_connections"

        if candidate is None and not current_lane_id.startswith(":"):
            maneuver_state = getattr(self, "physics_lateral_direction_states", {}).get(
                vehicle_id,
                {},
            )
            (
                intent,
                _target_lane_id,
                _authorization_source,
                _sumo_intent_evidence,
            ) = self._physics_lane_change_action(
                vehicle_id,
                current_lane_id,
            )
            # Keep the source-lane basis during ``armed``. Switching one full
            # lane at the first intent frame creates a false target jump. Once
            # the canonical 3-frame evidence confirms ``active``, the latched
            # direction remains authoritative if the intent bit disappears.
            reference_direction = intent
            if maneuver_state.get("phase") == "active" and reference_direction == "none":
                reference_direction = maneuver_state.get(
                    "authorized_direction",
                    "none",
                )
            adjacent_lane_id = ""
            if maneuver_state.get("phase") == "active":
                adjacent_lane_id = self._physics_adjacent_intent_lane_id(
                    current_lane_id,
                    reference_direction,
                )
            if adjacent_lane_id:
                adjacent_candidates = self._physics_direct_route_candidates(
                    adjacent_lane_id,
                    successor_edge_id or None,
                )
                adjacent_candidate, adjacent_reason = self._physics_select_direct_candidate(
                    adjacent_candidates,
                    selectors,
                )
                if adjacent_candidate is not None:
                    candidate = adjacent_candidate
                    reference_lane_id = adjacent_lane_id
                    reason = f"intent_target_lane:{adjacent_reason}"
                elif adjacent_reason == "ambiguous_direct_connections":
                    # No route leaves the actual lane yet. Keep its verified
                    # prefix while SUMO performs the required lane change;
                    # target/preview path-end checks still bound that prefix.
                    reason = (
                        "ambiguous_intent_target_connections"
                        if candidates
                        else "awaiting_required_lane_change:ambiguous_adjacent_connections"
                    )

        if candidate is None:
            error = reason if reason.startswith("ambiguous") else ""
            if reason == "no_direct_connection" and successor_edge_id:
                reason = "awaiting_required_lane_change"
            return {
                "actual_lane_id": current_lane_id,
                "reference_lane_id": current_lane_id,
                "lane_ids": (current_lane_id,),
                "successor_edge_id": successor_edge_id,
                "selection_reason": reason,
                "error": error,
                "planned_selectors": selectors,
            }

        lane_ids, extension_count, extension_error = self._physics_extend_verified_route_sequence(
            vehicle_id,
            candidate,
            selectors,
        )
        if extension_count:
            reason = f"{reason}:extended_{extension_count}"
        return {
            "actual_lane_id": current_lane_id,
            "reference_lane_id": reference_lane_id,
            "lane_ids": lane_ids,
            "successor_edge_id": successor_edge_id,
            "selection_reason": reason,
            "error": extension_error,
            "planned_selectors": selectors,
            "verified_extension_count": extension_count,
        }

    def _physics_rolling_lane_entries(  # noqa: C901
        self,
        vehicle_id,
        current_lane_id,
        future_lane_ids,
        history_distance,
    ):
        """Keep enough authoritative predecessor geometry for one vehicle."""
        states = getattr(self, "physics_rolling_path_states", None)
        if states is None:
            self.physics_rolling_path_states = {}
            states = self.physics_rolling_path_states
        previous = states.get(vehicle_id, {})
        entries = list(previous.get("entries") or ())
        next_occurrence = int(previous.get("next_occurrence", 0))
        future_lane_ids = tuple(future_lane_ids or ())
        if not future_lane_ids or future_lane_ids[0] != current_lane_id:
            future_lane_ids = (current_lane_id, *future_lane_ids)

        previous_current_index = int(previous.get("current_index", 0))
        current_index = None
        if entries and previous.get("current_lane_id") == current_lane_id:
            if 0 <= previous_current_index < len(entries):
                if entries[previous_current_index][1] == current_lane_id:
                    current_index = previous_current_index
        if current_index is None:
            for index in range(max(0, previous_current_index + 1), len(entries)):
                if entries[index][1] == current_lane_id:
                    current_index = index
                    break

        if current_index is None:
            entries = []
            for lane_id in future_lane_ids:
                entries.append((next_occurrence, lane_id))
                next_occurrence += 1
            current_index = 0
        else:
            refreshed = entries[: current_index + 1]
            old_future_index = current_index + 1
            for lane_id in future_lane_ids[1:]:
                if old_future_index < len(entries) and entries[old_future_index][1] == lane_id:
                    refreshed.append(entries[old_future_index])
                    old_future_index += 1
                else:
                    refreshed.append((next_occurrence, lane_id))
                    next_occurrence += 1
            entries = refreshed

        retained_history_distance = 0.0
        retained_start = current_index
        while retained_start > 0 and retained_history_distance < history_distance:
            retained_start -= 1
            lane_id = entries[retained_start][1]
            try:
                retained_history_distance += float(self._physics_lane_geometry(lane_id)["length"])
            except Exception:
                retained_start = 0
                retained_history_distance = math.inf
                break
        if retained_start:
            entries = entries[retained_start:]
            current_index -= retained_start

        state = {
            "entries": tuple(entries),
            "current_lane_id": current_lane_id,
            "current_index": current_index,
            "next_occurrence": next_occurrence,
            "history_distance": float(history_distance),
            "retained_history_distance": float(retained_history_distance),
        }
        states[vehicle_id] = state
        return tuple(entries), state

    def _physics_compiled_lookahead_path(
        self,
        vehicle_id,
        current_lane_id,
        history_distance=0.0,
    ):
        try:
            current_lane_id = traci.vehicle.getLaneID(vehicle_id)
        except Exception:
            return None
        resolution = self._physics_resolve_route_lane_sequence(vehicle_id, current_lane_id)
        reference_lane_id = resolution["reference_lane_id"]
        lane_ids = list(resolution["lane_ids"])
        entries, rolling_state = self._physics_rolling_lane_entries(
            vehicle_id,
            reference_lane_id,
            lane_ids,
            max(0.0, float(history_distance)),
        )
        lane_ids = [entry[1] for entry in entries]
        occurrence_key = tuple(f"{entry[0]}:{entry[1]}" for entry in entries)
        current_index = int(rolling_state["current_index"])

        cache_key = tuple(lane_ids)
        compiled = self.physics_lookahead_path_cache.get(cache_key)
        if compiled is None:
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

            compiled = compile_lane_shapes(shapes, lane_ids=valid_ids)
            if compiled is not None:
                self._warn_unverified_physics_lookahead_path(
                    cache_key,
                    valid_ids,
                    compiled,
                )
                self.physics_lookahead_path_cache[cache_key] = compiled
                self.physics_lookahead_path_cache.setdefault(tuple(valid_ids), compiled)
        if compiled is None:
            return None

        result_cache_key = (
            "__rolling__",
            occurrence_key,
            current_index,
            resolution["actual_lane_id"],
            resolution["reference_lane_id"],
            resolution["selection_reason"],
            float(rolling_state["history_distance"]),
            float(rolling_state["retained_history_distance"]),
        )
        cached_result = self.physics_lookahead_path_cache.get(result_cache_key)
        if cached_result is not None:
            return cached_result
        result = dict(compiled)
        valid_count = len(result.get("lane_ids", ()))
        result["canonical_path_key"] = occurrence_key[:valid_count]
        result["route_occurrence_ids"] = tuple(entry[0] for entry in entries[:valid_count])
        result["rolling_current_lane_index"] = current_index
        result["rolling_history_lane_ids"] = tuple(lane_ids[:current_index])
        result["rolling_history_distance"] = float(rolling_state["history_distance"])
        result["rolling_retained_history_distance"] = float(
            rolling_state["retained_history_distance"]
        )
        result["canonical_actual_lane_id"] = resolution["actual_lane_id"]
        result["canonical_reference_lane_id"] = resolution["reference_lane_id"]
        result["canonical_route_successor_edge_id"] = resolution["successor_edge_id"]
        result["canonical_route_selection_reason"] = resolution["selection_reason"]
        result["canonical_route_selection_error"] = resolution["error"]
        result["canonical_planned_link_selectors"] = resolution["planned_selectors"]
        self.physics_lookahead_path_cache[result_cache_key] = result
        return result

    def _physics_previous_lateral_direction(self, vehicle_id, observation=None, lateral_speed=None):
        """Return the direction latched for the complete active maneuver."""
        states = getattr(self, "physics_lateral_direction_states", {})
        state = states.get(vehicle_id, {})
        if state.get("phase") != "active":
            return None
        return state.get("confirmed_world_direction")

    def _physics_maneuver_corridor_ribbons(self, source_lane_id, target_lane_id, current_lane_id):
        source_lane_ids = (source_lane_id,)
        target_lane_ids = (target_lane_id,)
        if current_lane_id not in {source_lane_id, target_lane_id}:
            # A longitudinal connection does not complete a lateral maneuver.
            # Extend both ribbons only through one uniquely verified, adjacent
            # pair of connections containing the actual current lane.
            lane_delta = self._physics_lane_index(target_lane_id) - self._physics_lane_index(
                source_lane_id
            )
            continuations = []
            for source in self._physics_direct_route_candidates(source_lane_id):
                for target in self._physics_direct_route_candidates(
                    target_lane_id, source["destination_edge_id"]
                ):
                    source_ids, target_ids = source["lane_ids"], target["lane_ids"]
                    if len(source_ids) != len(target_ids) or current_lane_id not in (
                        *source_ids[1:],
                        *target_ids[1:],
                    ):
                        continue
                    if any(
                        traci.lane.getEdgeID(left) != traci.lane.getEdgeID(right)
                        or self._physics_lane_index(right) - self._physics_lane_index(left)
                        != lane_delta
                        for left, right in zip(source_ids, target_ids)
                    ):
                        continue
                    continuations.append((source_ids, target_ids))
            if len(continuations) != 1:
                return None
            source_lane_ids, target_lane_ids = continuations[0]

        ribbons = []
        for lane_ids in (source_lane_ids, target_lane_ids):
            widths = [float(traci.lane.getWidth(lane_id)) for lane_id in lane_ids]
            if any(not math.isfinite(width) or width <= 0.0 for width in widths):
                raise ValueError("invalid corridor lane width")
            shapes = [traci.lane.getShape(lane_id) for lane_id in lane_ids]
            compiled = compile_lane_shapes(shapes, lane_ids=lane_ids)
            if compiled is None or compiled["shape_join_accepted_count"] != len(lane_ids):
                raise ValueError("unverified corridor geometry join")
            ribbons.append((compiled["points"], min(widths)))
        return *ribbons, len(source_lane_ids) > 1

    def _physics_active_maneuver_lane_corridor(
        self,
        maneuver_state,
        action,
        *,
        current_lane_id,
    ):
        """Build a fail-closed source/target envelope for one active maneuver."""

        def invalid(reason):
            return {"valid": False, "reason": reason}

        if maneuver_state.get("phase") != "active":
            return None
        source_lane_id = str(maneuver_state.get("source_lane_id") or "")
        target_lane_id = str(maneuver_state.get("target_lane_id") or "")
        if not source_lane_id or not target_lane_id or source_lane_id == target_lane_id:
            return invalid("active_corridor_lane_ids_missing")
        source_index = self._physics_lane_index(source_lane_id)
        target_index = self._physics_lane_index(target_lane_id)
        if source_index < 0 or target_index < 0 or abs(target_index - source_index) != 1:
            return invalid("active_corridor_lanes_not_adjacent")
        try:
            source_edge_id = traci.lane.getEdgeID(source_lane_id)
            target_edge_id = traci.lane.getEdgeID(target_lane_id)
            if source_edge_id != target_edge_id:
                return invalid("active_corridor_lanes_not_same_edge")
            ribbons = self._physics_maneuver_corridor_ribbons(
                source_lane_id, target_lane_id, current_lane_id
            )
            if ribbons is None:
                return invalid("active_corridor_current_lane_mismatch")
            (
                (source_lane_shape, source_lane_width),
                (target_lane_shape, target_lane_width),
                continued,
            ) = ribbons
        except Exception:
            return invalid("active_corridor_lane_api_failed")
        corridor = build_active_maneuver_lane_corridor(
            action.get("canonical_phase_b"),
            source_lane_shape,
            source_lane_width,
            target_lane_shape,
            target_lane_width,
        )
        corridor["source_lane_id"] = source_lane_id
        corridor["target_lane_id"] = target_lane_id
        if corridor["valid"] and continued:
            corridor["reason"] = "source_target_lane_continuation_envelope"
        return corridor

    def _physics_maneuver_state(self, vehicle_id, current_lane_id):
        states = getattr(self, "physics_lateral_direction_states", None)
        if states is None:
            self.physics_lateral_direction_states = {}
            states = self.physics_lateral_direction_states
        defaults = {
            "phase": "lane_keep",
            "source_lane_id": current_lane_id,
            "current_lane_id": current_lane_id,
            "target_lane_id": "",
            "primary_lane_switched": False,
            "confirmed_world_direction": None,
            "confirmed_sumo_time": None,
            "confirmed_lateral_speed": None,
            "held_lateral_offset": None,
            "held_canonical_d": None,
            "held_world_anchor": None,
            "held_path_key": (),
            "last_current_lateral_offset": None,
            "canonical_path_key": (),
            "canonical_path_switched": False,
            "last_valid_front_target": None,
            "last_valid_control_target": None,
            "last_valid_steering_preview": None,
            "last_valid_path_tracker": None,
            "last_valid_origin": None,
            "last_valid_target_normal": None,
            "last_valid_preview_normal": None,
            "last_valid_route_tangent": None,
            "last_valid_maneuver_phase": None,
            "last_valid_target_d": None,
            "canonical_invariant_valid": True,
            "canonical_invariant_reason": "",
            "canonical_invariant_warning": "",
            "canonical_fallback_applied": False,
            "canonical_front_jump": None,
            "canonical_control_jump": None,
            "canonical_steering_preview_jump": None,
            "canonical_front_step": None,
            "canonical_control_step": None,
            "canonical_steering_preview_step": None,
            "canonical_fallback_reason": "",
            "canonical_fallback_age": 0,
            "zero_motion_frames": 0,
            "start_candidate_frames": 0,
            "start_candidate_sign": 0,
            "start_candidate_source_lane_id": "",
            "armed_idle_frames": 0,
            "armed_origin_phase": "lane_keep",
            "active_frames": 0,
            "unresolved_count": 0,
            "evidence_eligible": False,
            "evidence_exclusion_reason": "not_sampled",
            "canonical_motion_evidence": False,
            "canonical_delta_sign": 0,
            "speed_motion_evidence": False,
            "intent_evidence": False,
            "maneuver_authorized": False,
            "maneuver_authorization_source": "none",
            "authorized_direction": "none",
            "confirmed_direction_sign": 0,
            "last_confirmed_direction_sign": 0,
            "last_lane_change_intent": "none",
            "last_intent_conflict": False,
            "transition_reason": "initialized",
        }
        state = states.get(vehicle_id)
        if state is None:
            state = defaults
            states[vehicle_id] = state
        else:
            for key, value in defaults.items():
                state.setdefault(key, value)
        return state

    def _physics_force_lane_keep_state(self, vehicle_id, current_lane_id):
        """Reset maneuver control state while retaining diagnostic sampling."""
        state = self._physics_maneuver_state(vehicle_id, current_lane_id)
        state.update(
            {
                "phase": "lane_keep",
                "source_lane_id": current_lane_id,
                "current_lane_id": current_lane_id,
                "target_lane_id": "",
                "primary_lane_switched": False,
                "confirmed_world_direction": None,
                "confirmed_sumo_time": None,
                "confirmed_lateral_speed": None,
                "held_lateral_offset": None,
                "held_canonical_d": None,
                "held_world_anchor": None,
                "held_path_key": (),
                "zero_motion_frames": 0,
                "start_candidate_frames": 0,
                "start_candidate_sign": 0,
                "start_candidate_source_lane_id": "",
                "armed_idle_frames": 0,
                "armed_origin_phase": "lane_keep",
                "active_frames": 0,
                "unresolved_count": 0,
                "evidence_eligible": False,
                "evidence_exclusion_reason": "maneuver_disabled",
                "canonical_motion_evidence": False,
                "canonical_delta_sign": 0,
                "speed_motion_evidence": False,
                "intent_evidence": False,
                "maneuver_authorized": False,
                "maneuver_authorization_source": "none",
                "authorized_direction": "none",
                "confirmed_direction_sign": 0,
                "last_confirmed_direction_sign": 0,
                "last_lane_change_intent": "none",
                "last_intent_conflict": False,
                "transition_reason": "maneuver_disabled",
            }
        )
        return state

    def _prune_departed_physics_lateral_directions(self, all_vehicle_ids):
        alive_vehicle_ids = set(all_vehicle_ids)
        for states in (
            getattr(self, "physics_lateral_direction_states", {}),
            getattr(self, "physics_rolling_path_states", {}),
        ):
            for stale_id in [key for key in states if key not in alive_vehicle_ids]:
                del states[stale_id]

    @staticmethod
    def _physics_canonical_target_jump(
        previous_target,
        previous_origin,
        previous_normal,
        current_target,
        current_origin,
        current_normal,
    ):
        try:
            previous_target = np.asarray(previous_target[:2], dtype=np.float64)
            previous_origin = np.asarray(previous_origin[:2], dtype=np.float64)
            previous_normal = np.asarray(previous_normal[:2], dtype=np.float64)
            current_target = np.asarray(current_target[:2], dtype=np.float64)
            current_origin = np.asarray(current_origin[:2], dtype=np.float64)
            current_normal = np.asarray(current_normal[:2], dtype=np.float64)
        except (TypeError, ValueError, IndexError):
            return None
        arrays = (
            previous_target,
            previous_origin,
            previous_normal,
            current_target,
            current_origin,
            current_normal,
        )
        if not all(np.all(np.isfinite(value)) for value in arrays):
            return None
        mean_normal = previous_normal + current_normal
        norm = float(np.linalg.norm(mean_normal))
        if norm <= 1e-12:
            mean_normal = current_normal
            norm = float(np.linalg.norm(mean_normal))
        if norm <= 1e-12:
            return None
        mean_normal /= norm
        previous_vector = previous_target - previous_origin
        current_vector = current_target - current_origin
        return abs(float(np.dot(current_vector - previous_vector, mean_normal)))

    def _physics_validate_canonical_action(  # noqa: C901
        self,
        state,
        action,
        *,
        current_position,
        phase_a_rear_axle_position,
        canonical_path_key,
        canonical_path_switched,
        vehicle_id="",
    ):
        """Validate front/rear/preview targets or reuse one matching preview frame."""
        state["canonical_path_switched"] = bool(canonical_path_switched)
        state["canonical_invariant_valid"] = bool(action.get("valid", False))
        state["canonical_invariant_reason"] = action.get("error", "")
        state["canonical_invariant_warning"] = ""
        state["canonical_fallback_applied"] = False
        state["canonical_fallback_reason"] = ""
        state["canonical_front_jump"] = None
        state["canonical_control_jump"] = None
        state["canonical_steering_preview_jump"] = None
        state["canonical_front_step"] = None
        state["canonical_control_step"] = None
        state["canonical_steering_preview_step"] = None

        if action.get("valid", False) and not action.get("control_reference_valid", False):
            action["valid"] = False
            action["error"] = "missing_rear_axle_control_reference"
        if action.get("valid", False) and action.get("control_target_ahead") is False:
            action["valid"] = False
            action["error"] = "control_target_behind"
        if action.get("valid", False) and action.get("steering_preview_target_ahead") is False:
            action["valid"] = False
            action["error"] = "steering_preview_behind"
        if action.get("valid", False) and not action.get("path_tracker_valid", False):
            action["valid"] = False
            action["error"] = "missing_canonical_path_tracker_reference"

        front_target = action.get("lookahead")
        control_target = action.get("control_lookahead")
        steering_preview = action.get("steering_preview")
        target_normal = (
            action.get("target_world_left_normal_x"),
            action.get("target_world_left_normal_y"),
        )
        preview_normal = (
            action.get("steering_preview_normal_x"),
            action.get("steering_preview_normal_y"),
        )
        candidate_values = []
        for point in (front_target, control_target, steering_preview):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                candidate_values.append(None)
            else:
                candidate_values.extend(
                    (self._as_finite_float(point[0]), self._as_finite_float(point[1]))
                )
        candidate_values.extend(
            self._as_finite_float(value) for value in (*target_normal, *preview_normal)
        )
        path_tracker_fields = (
            "path_tracker_center_curvature",
            "path_tracker_front_curvature",
            "path_tracker_rear_curvature",
            "path_tracker_reference_tangent_x",
            "path_tracker_reference_tangent_y",
            "path_tracker_front_cross_track_error",
        )
        candidate_values.extend(
            self._as_finite_float(action.get(field)) for field in path_tracker_fields
        )
        if action.get("valid", False) and any(value is None for value in candidate_values):
            action["valid"] = False
            action["error"] = "non_finite_canonical_target"

        if action.get("valid", False):
            previous_origin = state.get("last_valid_origin")
            previous_normal = state.get("last_valid_target_normal")
            previous_preview_normal = state.get("last_valid_preview_normal")
            if (
                previous_origin is not None
                and previous_normal is not None
                and previous_preview_normal is not None
            ):
                state["canonical_front_step"] = self._physics_canonical_target_jump(
                    state.get("last_valid_front_target"),
                    previous_origin,
                    previous_normal,
                    front_target,
                    current_position,
                    target_normal,
                )
                state["canonical_control_step"] = self._physics_canonical_target_jump(
                    state.get("last_valid_control_target"),
                    previous_origin,
                    previous_normal,
                    control_target,
                    current_position,
                    target_normal,
                )
                state["canonical_steering_preview_step"] = self._physics_canonical_target_jump(
                    state.get("last_valid_steering_preview"),
                    previous_origin,
                    previous_preview_normal,
                    steering_preview,
                    current_position,
                    preview_normal,
                )

        if action.get("valid", False) and canonical_path_switched:
            previous_origin = state.get("last_valid_origin")
            previous_normal = state.get("last_valid_target_normal")
            previous_preview_normal = state.get("last_valid_preview_normal")
            if (
                previous_origin is not None
                and previous_normal is not None
                and previous_preview_normal is not None
            ):
                front_jump = self._physics_canonical_target_jump(
                    state.get("last_valid_front_target"),
                    previous_origin,
                    previous_normal,
                    front_target,
                    current_position,
                    target_normal,
                )
                control_jump = self._physics_canonical_target_jump(
                    state.get("last_valid_control_target"),
                    previous_origin,
                    previous_normal,
                    control_target,
                    current_position,
                    target_normal,
                )
                preview_jump = self._physics_canonical_target_jump(
                    state.get("last_valid_steering_preview"),
                    previous_origin,
                    previous_preview_normal,
                    steering_preview,
                    current_position,
                    preview_normal,
                )
                state["canonical_front_jump"] = front_jump
                state["canonical_control_jump"] = control_jump
                state["canonical_steering_preview_jump"] = preview_jump
                jumps = (front_jump, control_jump, preview_jump)
                if any(jump is None for jump in jumps):
                    action["valid"] = False
                    action["error"] = "canonical_target_jump_unavailable"
                elif max(jumps) > self.CANONICAL_TARGET_LATERAL_JUMP_LIMIT:
                    state["canonical_invariant_warning"] = "canonical_target_lateral_jump"
                    logger = getattr(self, "logger", None)
                    if logger is not None:
                        logger.warning(
                            "Canonical target lateral jump actor=%s "
                            "front=%.6f control=%.6f preview=%.6f limit=%.6f; "
                            "continuing control",
                            vehicle_id or "unknown",
                            front_jump,
                            control_jump,
                            preview_jump,
                            self.CANONICAL_TARGET_LATERAL_JUMP_LIMIT,
                        )

        if action.get("valid", False):
            state["canonical_path_key"] = tuple(canonical_path_key or ())
            state["last_valid_front_target"] = tuple(front_target)
            state["last_valid_control_target"] = tuple(control_target)
            state["last_valid_steering_preview"] = tuple(steering_preview)
            state["last_valid_path_tracker"] = {
                "path_tracker_valid": True,
                **{field: action.get(field) for field in path_tracker_fields},
            }
            state["last_valid_origin"] = tuple(current_position[:2])
            state["last_valid_target_normal"] = tuple(target_normal)
            state["last_valid_preview_normal"] = tuple(preview_normal)
            state["last_valid_route_tangent"] = (
                action.get("route_tangent_x"),
                action.get("route_tangent_y"),
            )
            state["last_valid_maneuver_phase"] = action.get("maneuver_phase")
            state["last_valid_target_d"] = action.get("target_lateral_distance")
            state["canonical_fallback_age"] = 0
            state["canonical_invariant_valid"] = True
            state["canonical_invariant_reason"] = ""
            return action

        reason = action.get("error") or "invalid_canonical_target"
        previous_front = state.get("last_valid_front_target")
        previous_control = state.get("last_valid_control_target")
        previous_preview = state.get("last_valid_steering_preview")
        previous_path_tracker = state.get("last_valid_path_tracker")
        previous_path_tracker_values = (
            []
            if not isinstance(previous_path_tracker, dict)
            else [
                self._as_finite_float(previous_path_tracker.get(field))
                for field in path_tracker_fields
            ]
        )
        tangent_values = (
            self._as_finite_float(action.get("route_tangent_x")),
            self._as_finite_float(action.get("route_tangent_y")),
        )
        if None in tangent_values:
            tangent_values = tuple(state.get("last_valid_route_tangent") or (None, None))
        target_d = self._as_finite_float(action.get("target_lateral_distance"))
        previous_target_d = self._as_finite_float(state.get("last_valid_target_d"))
        same_path = tuple(state.get("canonical_path_key") or ()) == tuple(canonical_path_key or ())
        same_phase = state.get("last_valid_maneuver_phase") == action.get("maneuver_phase")
        matching_d = bool(
            target_d is not None
            and previous_target_d is not None
            and abs(target_d - previous_target_d) <= 0.05
        )
        try:
            rear = np.asarray(phase_a_rear_axle_position[:2], dtype=np.float64)
            control = np.asarray(previous_control[:2], dtype=np.float64)
            preview = np.asarray(previous_preview[:2], dtype=np.float64)
            tangent = np.asarray(tangent_values, dtype=np.float64)
            reusable = bool(
                same_path
                and same_phase
                and matching_d
                and bool(previous_path_tracker.get("path_tracker_valid", False))
                and len(previous_path_tracker_values) == len(path_tracker_fields)
                and all(value is not None for value in previous_path_tracker_values)
                and int(state.get("canonical_fallback_age", 0)) < 1
                and np.all(np.isfinite(rear))
                and np.all(np.isfinite(control))
                and np.all(np.isfinite(preview))
                and np.all(np.isfinite(tangent))
                and float(np.dot(control - rear, tangent)) > 1e-9
                and float(np.dot(preview - rear, tangent)) > 1e-9
                and isinstance(previous_front, (list, tuple))
                and len(previous_front) >= 2
                and all(math.isfinite(float(value)) for value in previous_front[:2])
            )
        except (TypeError, ValueError, IndexError):
            reusable = False
        state["canonical_invariant_valid"] = False
        state["canonical_invariant_reason"] = reason
        if not reusable:
            return action

        action["valid"] = True
        action["error"] = ""
        action["warning"] = f"canonical_fallback:{reason}"
        action["mode"] = "canonical_fallback"
        action["lookahead"] = tuple(previous_front)
        action["control_lookahead"] = tuple(previous_control)
        action["steering_preview"] = tuple(previous_preview)
        action.update(previous_path_tracker)
        action["control_target_ahead"] = True
        action["steering_preview_target_ahead"] = True
        state["canonical_fallback_applied"] = True
        state["canonical_fallback_reason"] = reason
        state["canonical_fallback_age"] = 1
        return action

    @staticmethod
    def _physics_adjacent_lane_id(current_lane_id, direction):
        try:
            edge_id, lane_index = current_lane_id.rsplit("_", 1)
            lane_index = int(lane_index)
        except (AttributeError, TypeError, ValueError):
            return ""
        index_delta = {"left": 1, "right": -1}.get(direction)
        if index_delta is None or lane_index + index_delta < 0:
            return ""
        return f"{edge_id}_{lane_index + index_delta}"

    @staticmethod
    def _physics_active_maneuver_reached_route_successor(
        maneuver_state,
        current_lane_id,
        canonical_lane_ids,
    ):
        """Return whether an active maneuver has left its target via a verified path.

        The rolling canonical path is assembled only from ``lane.getLinks``-verified
        connections. Once SUMO has selected the target lane, reaching its direct
        route successor (with only internal lanes in between) is therefore stronger
        completion evidence than residual sub-centimetre ``speedLat`` noise.
        """
        if maneuver_state.get("phase") != "active" or not maneuver_state.get(
            "primary_lane_switched", False
        ):
            return False
        source_lane_id = str(maneuver_state.get("source_lane_id") or "")
        target_lane_id = str(maneuver_state.get("target_lane_id") or "")
        current_lane_id = str(current_lane_id or "")
        lane_ids = tuple(str(lane_id) for lane_id in (canonical_lane_ids or ()))
        if (
            not target_lane_id
            or not current_lane_id
            or current_lane_id in {source_lane_id, target_lane_id}
        ):
            return False

        current_indices = [
            index for index, lane_id in enumerate(lane_ids) if lane_id == current_lane_id
        ]
        target_indices = [
            index for index, lane_id in enumerate(lane_ids) if lane_id == target_lane_id
        ]
        for target_index in target_indices:
            for current_index in current_indices:
                if target_index >= current_index:
                    continue
                if all(
                    lane_id.startswith(":")
                    for lane_id in lane_ids[target_index + 1 : current_index]
                ):
                    return True
        return False

    def _physics_transition_maneuver_state(  # noqa: C901
        self,
        vehicle_id,
        *,
        phase_b_time,
        current_lane_id,
        current_lateral_offset,
        lateral_speed,
        lane_change_intent,
        action,
        current_position=None,
        canonical_path_key=(),
        canonical_lane_ids=(),
        authorization_source="none",
        authorization_direction="none",
        sumo_intent_evidence=None,
    ):
        """Advance maneuver evidence/state without selecting a control target."""
        state = self._physics_maneuver_state(vehicle_id, current_lane_id)
        states = self.physics_lateral_direction_states
        previous_phase = state["phase"]
        previous_lane_id = state.get("current_lane_id", current_lane_id)
        previous_intent = state.get("last_lane_change_intent", "none")
        current_lateral_distance = self._as_finite_float(action.get("current_lateral_distance"))
        raw_lateral_offset = self._as_finite_float(current_lateral_offset)
        canonical_delta_d = self._as_finite_float(action.get("canonical_delta_d"))
        lateral_speed_value = self._as_finite_float(lateral_speed)
        normal_x = self._as_finite_float(action.get("world_left_normal_x"))
        normal_y = self._as_finite_float(action.get("world_left_normal_y"))
        path_switched = bool(action.get("canonical_path_switched", False))
        lane_switched = bool(previous_lane_id and previous_lane_id != current_lane_id)

        state["last_raw_lateral_offset"] = raw_lateral_offset
        if current_lateral_distance is not None:
            state["last_current_lateral_offset"] = current_lateral_distance
        if action.get("held_reprojected"):
            reprojected_d = self._as_finite_float(action.get("held_canonical_d"))
            if reprojected_d is not None:
                state["held_canonical_d"] = reprojected_d
                state["held_lateral_offset"] = reprojected_d
                state["held_path_key"] = tuple(canonical_path_key or ())

        normal_valid = False
        if normal_x is not None and normal_y is not None:
            normal_norm = math.hypot(normal_x, normal_y)
            if normal_norm > 1e-12:
                normal_x /= normal_norm
                normal_y /= normal_norm
                normal_valid = True

        exclusion_reason = ""
        if not action.get("valid", False):
            exclusion_reason = "invalid_action:" + (action.get("error") or "unknown")
        elif canonical_delta_d is None:
            exclusion_reason = "canonical_delta_d_unavailable"
        elif lateral_speed_value is None:
            exclusion_reason = "non_finite_lateral_speed"
        elif not normal_valid:
            exclusion_reason = "canonical_normal_unavailable"
        elif path_switched and lane_switched:
            exclusion_reason = "canonical_path_and_lane_switched"
        elif path_switched:
            exclusion_reason = "canonical_path_switched"
        elif lane_switched:
            exclusion_reason = "primary_lane_switched"

        evidence_eligible = not exclusion_reason
        delta_sign = (
            0 if canonical_delta_d in (None, 0.0) else (1 if canonical_delta_d > 0.0 else -1)
        )
        canonical_motion = bool(
            evidence_eligible and abs(canonical_delta_d) >= self.LATERAL_START_DELTA_D_EPSILON
        )
        speed_motion = bool(
            lateral_speed_value is not None
            and abs(lateral_speed_value) > self.LATERAL_SPEED_EPSILON
        )
        intent_evidence = (
            lane_change_intent in {"left", "right"}
            if sumo_intent_evidence is None
            else bool(sumo_intent_evidence)
        )
        nde_request_evidence = authorization_source == "terasim_nde_request"
        explicit_av = vehicle_id == "AV" and getattr(self, "av_control_mode", "external") == "sumo"
        authorization_evidence = nde_request_evidence or (intent_evidence and not explicit_av)
        if nde_request_evidence:
            request = getattr(self, "av_nde_lane_change_request", None) or {}
            state["request_generation_consumed"] = int(request.get("count", 0))
        if intent_evidence and not explicit_av:
            authorization_direction = lane_change_intent
        stop_candidate = bool(
            evidence_eligible
            and not speed_motion
            and abs(canonical_delta_d) <= self.LATERAL_STOP_DELTA_D_EPSILON
        )
        route_successor_completion = (
            previous_phase == "active"
            and lane_switched
            and self._physics_active_maneuver_reached_route_successor(
                state,
                current_lane_id,
                canonical_lane_ids,
            )
        )

        state["evidence_eligible"] = evidence_eligible
        state["evidence_exclusion_reason"] = exclusion_reason
        state["canonical_motion_evidence"] = canonical_motion
        state["canonical_delta_sign"] = delta_sign
        state["speed_motion_evidence"] = speed_motion
        state["intent_evidence"] = intent_evidence
        if authorization_evidence:
            state["maneuver_authorized"] = True
            if nde_request_evidence:
                state["maneuver_authorization_source"] = "terasim_nde_request"
            elif state.get("maneuver_authorization_source") != "terasim_nde_request":
                state["maneuver_authorization_source"] = "sumo_intent"
            state["authorized_direction"] = authorization_direction
        elif previous_phase in {"armed", "active"}:
            # Preserve the provenance after a one-shot request or intent ends.
            state["maneuver_authorized"] = True
        else:
            state["maneuver_authorized"] = False
            state["maneuver_authorization_source"] = "none"

        if previous_intent != "none" and lane_change_intent == "none":
            self.logger.info(
                "Physical co-sim lane-change intent disappeared: actor=%s "
                "phase=%s source_lane=%s current_lane=%s",
                vehicle_id,
                previous_phase,
                state.get("source_lane_id", ""),
                current_lane_id,
            )

        candidate_authorized = previous_phase == "armed" or authorization_evidence
        if previous_phase != "active" and candidate_authorized:
            previous_candidate_sign = int(state.get("start_candidate_sign", 0))
            previous_candidate_frames = int(state.get("start_candidate_frames", 0))
            authorized_sign = 1 if state.get("authorized_direction") == "left" else -1
            if canonical_motion and (not explicit_av or delta_sign == authorized_sign):
                if previous_candidate_sign == delta_sign:
                    state["start_candidate_frames"] = previous_candidate_frames + 1
                else:
                    if previous_candidate_frames and previous_candidate_sign != delta_sign:
                        self.logger.info(
                            "Physical co-sim lateral candidate direction reset: "
                            "actor=%s old_sign=%s new_sign=%s frames=%s",
                            vehicle_id,
                            previous_candidate_sign,
                            delta_sign,
                            previous_candidate_frames,
                        )
                    state["start_candidate_sign"] = delta_sign
                    state["start_candidate_frames"] = 1
                    state["start_candidate_source_lane_id"] = current_lane_id
            else:
                state["start_candidate_frames"] = 0
                state["start_candidate_sign"] = 0
                state["start_candidate_source_lane_id"] = ""
        else:
            state["start_candidate_frames"] = 0
            state["start_candidate_sign"] = 0
            state["start_candidate_source_lane_id"] = ""

        start_confirmed = bool(
            previous_phase != "active"
            and int(state.get("start_candidate_frames", 0)) >= self.LATERAL_START_CONFIRM_FRAMES
        )

        if previous_phase == "active":
            state["active_frames"] = int(state.get("active_frames", 0)) + 1
            if lane_switched and not state.get("primary_lane_switched", False):
                expected_target = state.get("target_lane_id", "")
                source_lane = state.get("source_lane_id", "")
                observed_adjacent = current_lane_id in {
                    self._physics_adjacent_lane_id(source_lane, "left"),
                    self._physics_adjacent_lane_id(source_lane, "right"),
                }
                if current_lane_id == expected_target or (
                    not expected_target and observed_adjacent
                ):
                    state["target_lane_id"] = current_lane_id
                    state["primary_lane_switched"] = True
                    self.logger.info(
                        "Physical co-sim primary lane switched: actor=%s "
                        "source=%s target=%s phase=%s",
                        vehicle_id,
                        source_lane,
                        current_lane_id,
                        previous_phase,
                    )

            if route_successor_completion:
                state["zero_motion_frames"] = self.LATERAL_STOP_CONFIRM_FRAMES
                state["transition_reason"] = "active_target_lane_route_successor"
            elif stop_candidate:
                state["zero_motion_frames"] = int(state.get("zero_motion_frames", 0)) + 1
                state["transition_reason"] = "active_stop_candidate"
            else:
                state["zero_motion_frames"] = 0
                if not evidence_eligible:
                    state["transition_reason"] = "active_evidence_excluded:" + exclusion_reason
                elif canonical_motion or speed_motion:
                    state["transition_reason"] = "active_motion"
                else:
                    state["transition_reason"] = "active_between_thresholds"

            if int(state.get("zero_motion_frames", 0)) >= self.LATERAL_STOP_CONFIRM_FRAMES:
                final_offset = current_lateral_distance
                if final_offset is None:
                    final_offset = self._as_finite_float(state.get("last_current_lateral_offset"))
                if final_offset is None:
                    state["transition_reason"] = "completion_offset_unavailable"
                    state["zero_motion_frames"] = 0
                    self.logger.warning(
                        "Physical co-sim lateral completion offset unavailable: "
                        "actor=%s; keeping maneuver active",
                        vehicle_id,
                    )
                else:
                    state["held_canonical_d"] = final_offset
                    state["held_lateral_offset"] = final_offset
                    state["held_world_anchor"] = None
                    if current_position is not None:
                        try:
                            anchor = tuple(float(value) for value in current_position[:2])
                        except (TypeError, ValueError, IndexError):
                            anchor = ()
                        if len(anchor) == 2 and all(math.isfinite(value) for value in anchor):
                            state["held_world_anchor"] = anchor
                    state["held_path_key"] = tuple(canonical_path_key or ())
                    state["last_confirmed_direction_sign"] = int(
                        state.get("confirmed_direction_sign", 0)
                    )
                    state["confirmed_world_direction"] = None
                    state["confirmed_direction_sign"] = 0
                    state["confirmed_sumo_time"] = None
                    state["confirmed_lateral_speed"] = None
                    state["armed_idle_frames"] = 0
                    if abs(final_offset) < self.LATERAL_CENTER_TOLERANCE:
                        state["phase"] = "lane_keep"
                        state["source_lane_id"] = current_lane_id
                        state["target_lane_id"] = ""
                        state["primary_lane_switched"] = False
                        state["held_lateral_offset"] = None
                        state["held_canonical_d"] = None
                        state["held_world_anchor"] = None
                        state["held_path_key"] = ()
                        state["transition_reason"] = (
                            "completed_at_lane_center_after_target_successor"
                            if route_successor_completion
                            else "completed_at_lane_center"
                        )
                    else:
                        state["phase"] = "hold"
                        state["transition_reason"] = (
                            "completed_with_final_offset_after_target_successor"
                            if route_successor_completion
                            else "completed_with_final_offset"
                        )
        elif start_confirmed:
            direction_sign = int(state["start_candidate_sign"])
            state["phase"] = "active"
            state["source_lane_id"] = state.get("start_candidate_source_lane_id") or state.get(
                "source_lane_id"
            )
            authorized_direction = state.get("authorized_direction", "none")
            if not state.get("target_lane_id") and authorized_direction in {
                "left",
                "right",
            }:
                state["target_lane_id"] = self._physics_adjacent_lane_id(
                    state["source_lane_id"], authorized_direction
                )
            state["primary_lane_switched"] = False
            state["active_frames"] = 1
            state["zero_motion_frames"] = 0
            state["armed_idle_frames"] = 0
            state["confirmed_direction_sign"] = direction_sign
            state["last_confirmed_direction_sign"] = direction_sign
            state["confirmed_world_direction"] = (
                direction_sign * normal_x,
                direction_sign * normal_y,
            )
            state["confirmed_sumo_time"] = phase_b_time
            state["confirmed_lateral_speed"] = lateral_speed_value
            state["held_lateral_offset"] = None
            state["held_canonical_d"] = None
            state["held_world_anchor"] = None
            state["held_path_key"] = ()
            state["transition_reason"] = "canonical_motion_confirmed"
        else:
            auxiliary_evidence = authorization_evidence or speed_motion or canonical_motion
            if previous_phase == "armed":
                if auxiliary_evidence:
                    state["armed_idle_frames"] = 0
                    state["transition_reason"] = (
                        "canonical_motion_candidate"
                        if canonical_motion
                        else "armed_waiting_for_canonical_motion"
                    )
                elif evidence_eligible:
                    state["armed_idle_frames"] = int(state.get("armed_idle_frames", 0)) + 1
                    if state["armed_idle_frames"] >= self.LATERAL_ARMED_IDLE_FRAMES:
                        return_phase = state.get("armed_origin_phase", "lane_keep")
                        state["phase"] = return_phase if return_phase == "hold" else "lane_keep"
                        state["armed_idle_frames"] = 0
                        state["target_lane_id"] = (
                            state.get("target_lane_id", "") if state["phase"] == "hold" else ""
                        )
                        state["transition_reason"] = "armed_evidence_cancelled"
                    else:
                        state["transition_reason"] = "armed_idle_candidate"
                else:
                    state["armed_idle_frames"] = 0
                    state["transition_reason"] = "armed_evidence_excluded:" + exclusion_reason
            elif authorization_evidence:
                state["armed_origin_phase"] = "hold" if previous_phase == "hold" else "lane_keep"
                state["phase"] = "armed"
                state["source_lane_id"] = current_lane_id
                state["target_lane_id"] = self._physics_adjacent_lane_id(
                    current_lane_id, authorization_direction
                )
                state["primary_lane_switched"] = False
                state["active_frames"] = 0
                state["zero_motion_frames"] = 0
                state["armed_idle_frames"] = 0
                state["transition_reason"] = (
                    "request_authorized" if nde_request_evidence else "intent_authorized"
                )
            elif previous_phase == "hold":
                state["phase"] = "hold"
                state["transition_reason"] = (
                    "uncommanded_lateral_recovery"
                    if speed_motion or canonical_motion
                    else "holding_final_offset"
                )
            else:
                state["phase"] = "lane_keep"
                state["source_lane_id"] = current_lane_id
                state["target_lane_id"] = ""
                state["primary_lane_switched"] = False
                state["active_frames"] = 0
                state["zero_motion_frames"] = 0
                state["armed_idle_frames"] = 0
                state["transition_reason"] = (
                    "uncommanded_lateral_recovery"
                    if speed_motion or canonical_motion
                    else "lane_keep"
                )

        if state["phase"] == "hold" and action.get("valid", False):
            held_value = self._as_finite_float(state.get("held_canonical_d"))
            phase_b_frame = action.get("canonical_phase_b") or {}
            if held_value is not None:
                projected_x = self._as_finite_float(phase_b_frame.get("projected_x"))
                projected_y = self._as_finite_float(phase_b_frame.get("projected_y"))
                hold_normal_x = self._as_finite_float(phase_b_frame.get("normal_x"))
                hold_normal_y = self._as_finite_float(phase_b_frame.get("normal_y"))
                if None not in (projected_x, projected_y, hold_normal_x, hold_normal_y):
                    state["held_world_anchor"] = (
                        projected_x + hold_normal_x * held_value,
                        projected_y + hold_normal_y * held_value,
                    )
                    state["held_path_key"] = tuple(canonical_path_key or ())

        conflict_direction = (
            state.get("confirmed_direction_sign", 0)
            if state["phase"] == "active"
            else (delta_sign if canonical_motion else 0)
        )
        intent_conflict = self._physics_lateral_intent_conflict(
            lane_change_intent, conflict_direction
        )
        state["current_lane_id"] = current_lane_id
        state["last_lane_change_intent"] = lane_change_intent
        state["last_intent_conflict"] = intent_conflict
        states[vehicle_id] = state

        if state["phase"] != previous_phase:
            self.logger.info(
                "Physical co-sim lateral maneuver %s->%s: actor=%s reason=%s "
                "canonical_delta_d=%s candidate_frames=%s stop_frames=%s",
                previous_phase,
                state["phase"],
                vehicle_id,
                state.get("transition_reason", ""),
                canonical_delta_d,
                state.get("start_candidate_frames", 0),
                state.get("zero_motion_frames", 0),
            )
        return state, intent_conflict

    @staticmethod
    def _physics_lateral_intent_conflict(lane_change_intent, world_lateral_speed):
        if world_lateral_speed in (None, 0.0):
            return False
        intent_sign = {"left": 1.0, "right": -1.0}.get(lane_change_intent)
        return bool(
            intent_sign is not None and intent_sign != math.copysign(1.0, world_lateral_speed)
        )

    def _populate_physics_action_state(self, vehicle_id, vehicle_state):
        """Export the Phase-B SUMO action consumed by CARLA in the next frame."""
        if not self._is_physics_feedback_actor(vehicle_id):
            states = getattr(self, "physics_lateral_direction_states", None)
            if states is not None:
                states.pop(vehicle_id, None)
            rolling_states = getattr(self, "physics_rolling_path_states", None)
            if rolling_states is not None:
                rolling_states.pop(vehicle_id, None)
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
        vehicle_state["feedback_requested_sumo_angle"] = observation.get("requested_sumo_angle")
        vehicle_state["feedback_observed_speed"] = observation["speed"]
        vehicle_state["feedback_observed_acceleration"] = observation["acceleration"]
        vehicle_state["feedback_rear_axle_x"] = observation["rear_axle_position"][0]
        vehicle_state["feedback_rear_axle_y"] = observation["rear_axle_position"][1]
        vehicle_state["front_to_rear_axle_distance"] = observation["front_to_control_offset"]
        vehicle_state["feedback_requested_lane_id"] = observation["requested_lane_id"]
        vehicle_state["feedback_observed_lane_id"] = observation["lane_id"]
        vehicle_state["feedback_phase_a_sumo_time"] = observation["phase_a_time"]
        vehicle_state["feedback_source_carla_frame"] = observation["source_carla_frame"]
        vehicle_state["feedback_physics_seed_only"] = bool(
            observation.get("physics_seed_only", False)
        )
        vehicle_state["feedback_initialization_generation"] = observation.get(
            "initialization_generation", 0
        )
        vehicle_state["feedback_internal_lane_transition_assist_applied"] = bool(
            observation.get("internal_lane_transition_assist", False)
        )
        vehicle_state["feedback_position_skipped_for_lane_change"] = bool(
            observation.get("lateral_reference_preserved", False)
        )
        vehicle_state["feedback_lateral_reference_preserved"] = bool(
            observation.get("lateral_reference_preserved", False)
        )
        vehicle_state["feedback_preserved_sumo_lateral_offset"] = observation.get(
            "preserved_sumo_lateral_offset"
        )
        assimilated_position = observation.get("sumo_assimilated_position")
        vehicle_state["feedback_assimilated_x"] = (
            assimilated_position[0] if assimilated_position is not None else None
        )
        vehicle_state["feedback_assimilated_y"] = (
            assimilated_position[1] if assimilated_position is not None else None
        )
        maneuver_enabled = bool(getattr(self, "physics_lateral_maneuver_enabled", False))
        vehicle_state["external_state_lateral_maneuver_enabled"] = maneuver_enabled
        maneuver_target_enabled = bool(
            getattr(self, "physics_lateral_maneuver_target_enabled", False)
        )
        vehicle_state["external_state_lateral_maneuver_target_enabled"] = maneuver_target_enabled
        vehicle_state["external_state_internal_lane_transition_assist_enabled"] = bool(
            getattr(self, "physics_internal_lane_transition_assist_enabled", False)
        )
        vehicle_state["feedback_longitudinal_error"] = None
        try:
            vehicle_state["sumo_desired_speed"] = traci.vehicle.getSpeedWithoutTraCI(vehicle_id)
        except Exception:
            vehicle_state["sumo_desired_speed"] = vehicle_state["speed"]
        try:
            vehicle_state["sumo_emergency_decel"] = traci.vehicle.getEmergencyDecel(vehicle_id)
        except Exception:
            vehicle_state["sumo_emergency_decel"] = None

        vehicle_state["sumo_leader_id"] = ""
        vehicle_state["sumo_leader_gap"] = None
        vehicle_state["sumo_leader_speed"] = None
        vehicle_state["sumo_leader_acceleration"] = None
        vehicle_state["sumo_leader_lane_id"] = ""
        vehicle_state["sumo_leader_lane_position"] = None
        try:
            leader_info = traci.vehicle.getLeader(vehicle_id, dist=200.0)
        except Exception:
            leader_info = None
        if leader_info:
            leader_id, leader_gap = leader_info
            vehicle_state["sumo_leader_id"] = leader_id
            vehicle_state["sumo_leader_gap"] = leader_gap
            try:
                vehicle_state["sumo_leader_speed"] = traci.vehicle.getSpeed(leader_id)
                vehicle_state["sumo_leader_acceleration"] = traci.vehicle.getAcceleration(leader_id)
                vehicle_state["sumo_leader_lane_id"] = traci.vehicle.getLaneID(leader_id)
                vehicle_state["sumo_leader_lane_position"] = traci.vehicle.getLanePosition(
                    leader_id
                )
            except Exception:
                # The leader may leave between getLeader() and the detail calls.
                pass

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
            sumo_route_index = int(traci.vehicle.getRouteIndex(vehicle_id))
        except Exception:
            sumo_route_index = None
        try:
            sumo_route_distance = float(traci.vehicle.getDistance(vehicle_id))
        except Exception:
            sumo_route_distance = None
        try:
            sumo_slope = traci.vehicle.getSlope(vehicle_id)
        except Exception:
            sumo_slope = 0.0

        vehicle_state["lane_id"] = current_lane_id
        vehicle_state["lane_position"] = current_lane_position
        vehicle_state["sumo_route"] = sumo_route
        vehicle_state["sumo_route_index"] = sumo_route_index
        vehicle_state["sumo_route_distance"] = sumo_route_distance
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
                    observed_lane_length - observed_lane_position + current_lane_position
                )
            else:
                progress_delta = None
            if progress_delta is not None:
                vehicle_state["feedback_longitudinal_error"] = (
                    progress_delta - float(observation["speed"]) * self.physics_step_length
                )

        (
            intent,
            target_lane_id,
            authorization_source,
            sumo_intent_evidence,
        ) = self._physics_lane_change_action(vehicle_id, current_lane_id)
        authorization_direction = intent if authorization_source != "none" else "none"
        vehicle_state["sumo_lane_change_intent"] = intent
        vehicle_state["sumo_lane_change_target_lane_id"] = target_lane_id
        try:
            lateral_speed = traci.vehicle.getLateralSpeed(vehicle_id)
        except Exception:
            lateral_speed = 0.0
        vehicle_state["lateral_speed"] = lateral_speed
        try:
            current_lateral_offset = traci.vehicle.getLateralLanePosition(vehicle_id)
            vehicle_state["lateral_offset"] = current_lateral_offset
        except Exception:
            current_lateral_offset = None
        maneuver_state = self._physics_maneuver_state(vehicle_id, current_lane_id)
        if maneuver_enabled:
            previous_lateral_direction = self._physics_previous_lateral_direction(
                vehicle_id, observation, lateral_speed
            )
        else:
            maneuver_state = self._physics_force_lane_keep_state(vehicle_id, current_lane_id)
            previous_lateral_direction = None

        front_to_rear_distance = max(
            0.0,
            float(observation["front_to_control_offset"]),
        )
        curvature_half_window = max(1.0, 0.5 * front_to_rear_distance)
        rolling_history_distance = front_to_rear_distance + curvature_half_window + 0.5
        compiled_path = self._physics_compiled_lookahead_path(
            vehicle_id,
            current_lane_id,
            history_distance=rolling_history_distance,
        )
        canonical_path_key = (
            tuple(compiled_path.get("canonical_path_key", ())) if compiled_path is not None else ()
        )
        previous_canonical_path_key = tuple(maneuver_state.get("canonical_path_key") or ())
        canonical_path_switched = bool(
            previous_canonical_path_key
            and canonical_path_key
            and previous_canonical_path_key != canonical_path_key
        )
        try:
            current_lane_width = traci.lane.getWidth(current_lane_id)
        except Exception:
            current_lane_width = None
        base_distance = min(
            self.physics_lookahead_max_distance,
            max(self.physics_lookahead_min_distance, vehicle_state["speed"]),
        )
        effective_distances, heading_changes = adapt_lookahead_distances_for_compiled_paths(
            [compiled_path],
            [current_position],
            [base_distance],
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
                maneuver_enabled=maneuver_enabled,
                maneuver_target_enabled=False,
                lateral_speed=lateral_speed,
                desired_speed=vehicle_state["sumo_desired_speed"],
                min_forward_speed=0.2,
                lateral_speed_epsilon=self.LATERAL_SPEED_EPSILON,
                lateral_motion_epsilon=self.LATERAL_START_DELTA_D_EPSILON,
                phase_a_position=observation["position"],
                phase_a_evidence_position=observation.get(
                    "sumo_assimilated_position", observation["position"]
                ),
                phase_a_rear_axle_position=observation["rear_axle_position"],
                phase_a_speed=observation["speed"],
                front_to_control_offset=observation["front_to_control_offset"],
                steering_preview_min_distance=getattr(
                    self, "physics_steering_preview_min_distance", 2.8
                ),
                steering_preview_time_headway=getattr(
                    self, "physics_steering_preview_time_headway", 0.5
                ),
                steering_preview_max_distance=getattr(
                    self, "physics_steering_preview_max_distance", 5.0
                ),
                phase_step_length=self.physics_step_length,
                previous_world_lateral_direction=previous_lateral_direction,
                previous_direction_min_alignment=self.LATERAL_DIRECTION_MIN_ALIGNMENT,
                maneuver_phase=maneuver_state["phase"],
                armed_origin_phase=maneuver_state.get("armed_origin_phase", "lane_keep"),
                held_lateral_offset=maneuver_state.get("held_lateral_offset"),
                held_canonical_d=maneuver_state.get("held_canonical_d"),
                held_world_anchor=maneuver_state.get("held_world_anchor"),
                canonical_path_key=canonical_path_key,
                canonical_path_switched=canonical_path_switched,
                lane_width=current_lane_width,
                curvature_half_window=curvature_half_window,
                z=vehicle_state["z"],
            )
        route_selection_error = (
            compiled_path.get("canonical_route_selection_error", "")
            if compiled_path is not None
            else ""
        )
        if route_selection_error:
            action.update(valid=False, error=route_selection_error)

        if maneuver_enabled:
            maneuver_state, intent_conflict = self._physics_transition_maneuver_state(
                vehicle_id,
                phase_b_time=phase_b_time,
                current_lane_id=current_lane_id,
                current_lateral_offset=current_lateral_offset,
                lateral_speed=lateral_speed,
                lane_change_intent=intent,
                action=action,
                current_position=current_position,
                canonical_path_key=canonical_path_key,
                canonical_lane_ids=(
                    tuple(compiled_path.get("lane_ids", ())) if compiled_path is not None else ()
                ),
                authorization_source=authorization_source,
                authorization_direction=authorization_direction,
                sumo_intent_evidence=sumo_intent_evidence,
            )
        else:
            maneuver_state = self._physics_force_lane_keep_state(vehicle_id, current_lane_id)
            intent_conflict = False

        action = apply_external_state_lateral_maneuver_target(
            action,
            maneuver_enabled=maneuver_enabled,
            maneuver_target_enabled=maneuver_target_enabled,
            maneuver_phase=maneuver_state["phase"],
            armed_origin_phase=maneuver_state.get("armed_origin_phase", "lane_keep"),
            held_canonical_d=maneuver_state.get("held_canonical_d"),
            phase_a_rear_axle_position=observation["rear_axle_position"],
            lane_width=current_lane_width,
            active_maneuver_corridor=self._physics_active_maneuver_lane_corridor(
                maneuver_state,
                action,
                current_lane_id=current_lane_id,
            ),
        )
        action = self._physics_validate_canonical_action(
            maneuver_state,
            action,
            current_position=current_position,
            phase_a_rear_axle_position=observation["rear_axle_position"],
            canonical_path_key=canonical_path_key,
            canonical_path_switched=canonical_path_switched,
            vehicle_id=vehicle_id,
        )
        unresolved_count = int(maneuver_state.get("unresolved_count", 0))

        vehicle_state["external_state_maneuver_phase"] = maneuver_state["phase"]
        vehicle_state["external_state_maneuver_armed_origin_phase"] = maneuver_state.get(
            "armed_origin_phase", "lane_keep"
        )
        vehicle_state["external_state_maneuver_target_pre_state_phase"] = action.get(
            "maneuver_target_pre_state_phase", "lane_keep"
        )
        vehicle_state["external_state_maneuver_target_post_state_phase"] = action.get(
            "maneuver_target_post_state_phase", maneuver_state["phase"]
        )
        vehicle_state["external_state_maneuver_target_applied"] = bool(
            action.get("maneuver_target_applied", False)
        )
        vehicle_state["external_state_maneuver_target_source"] = action.get(
            "maneuver_target_source", "route_center"
        )
        vehicle_state["external_state_maneuver_target_selected_d"] = action.get(
            "maneuver_target_selected_d"
        )
        target_world_anchor = action.get("maneuver_target_world_anchor") or (None, None)
        vehicle_state["external_state_maneuver_target_world_anchor_x"] = target_world_anchor[0]
        vehicle_state["external_state_maneuver_target_world_anchor_y"] = target_world_anchor[1]
        vehicle_state["external_state_maneuver_target_d_outside_lane"] = action.get(
            "maneuver_target_d_outside_lane"
        )
        vehicle_state["external_state_maneuver_target_d_outside_primary_lane"] = action.get(
            "maneuver_target_d_outside_primary_lane"
        )
        vehicle_state["external_state_maneuver_target_safety_region"] = action.get(
            "maneuver_target_safety_region", "primary_lane"
        )
        vehicle_state["external_state_active_maneuver_corridor_valid"] = action.get(
            "active_maneuver_corridor_valid"
        )
        vehicle_state["external_state_active_maneuver_corridor_reason"] = action.get(
            "active_maneuver_corridor_reason", ""
        )
        vehicle_state["external_state_active_maneuver_corridor_d_min"] = action.get(
            "active_maneuver_corridor_d_min"
        )
        vehicle_state["external_state_active_maneuver_corridor_d_max"] = action.get(
            "active_maneuver_corridor_d_max"
        )
        vehicle_state["external_state_active_maneuver_corridor_source_center_d"] = action.get(
            "active_maneuver_corridor_source_center_d"
        )
        vehicle_state["external_state_active_maneuver_corridor_target_center_d"] = action.get(
            "active_maneuver_corridor_target_center_d"
        )
        vehicle_state["external_state_active_maneuver_corridor_center_separation"] = action.get(
            "active_maneuver_corridor_center_separation"
        )
        vehicle_state["external_state_maneuver_source_lane_id"] = maneuver_state.get(
            "source_lane_id", ""
        )
        vehicle_state["external_state_maneuver_current_lane_id"] = current_lane_id
        vehicle_state["external_state_maneuver_target_lane_id"] = maneuver_state.get(
            "target_lane_id", ""
        )
        vehicle_state["external_state_maneuver_primary_lane_switched"] = bool(
            maneuver_state.get("primary_lane_switched", False)
        )
        vehicle_state["external_state_maneuver_current_lateral_offset"] = action.get(
            "current_lateral_distance"
        )
        vehicle_state["external_state_maneuver_preview_lateral_offset"] = action.get(
            "preview_lateral_distance"
        )
        vehicle_state["external_state_maneuver_held_lateral_offset"] = maneuver_state.get(
            "held_canonical_d"
        )
        vehicle_state["external_state_maneuver_active_frame_count"] = int(
            maneuver_state.get("active_frames", 0)
        )
        vehicle_state["external_state_maneuver_zero_motion_frame_count"] = int(
            maneuver_state.get("zero_motion_frames", 0)
        )
        vehicle_state["external_state_maneuver_transition_reason"] = maneuver_state.get(
            "transition_reason", ""
        )
        vehicle_state["external_state_maneuver_evidence_eligible"] = bool(
            maneuver_state.get("evidence_eligible", False)
        )
        vehicle_state["external_state_maneuver_evidence_exclusion_reason"] = maneuver_state.get(
            "evidence_exclusion_reason", ""
        )
        vehicle_state["external_state_maneuver_canonical_motion_evidence"] = bool(
            maneuver_state.get("canonical_motion_evidence", False)
        )
        vehicle_state["external_state_maneuver_canonical_delta_sign"] = int(
            maneuver_state.get("canonical_delta_sign", 0)
        )
        vehicle_state["external_state_maneuver_speed_motion_evidence"] = bool(
            maneuver_state.get("speed_motion_evidence", False)
        )
        vehicle_state["external_state_maneuver_intent_evidence"] = bool(
            maneuver_state.get("intent_evidence", False)
        )
        vehicle_state["external_state_maneuver_authorized"] = bool(
            maneuver_state.get("maneuver_authorized", False)
        )
        vehicle_state["external_state_maneuver_authorization_source"] = maneuver_state.get(
            "maneuver_authorization_source", "none"
        )
        vehicle_state["external_state_maneuver_authorized_direction"] = maneuver_state.get(
            "authorized_direction", "none"
        )
        vehicle_state["external_state_maneuver_start_candidate_sign"] = int(
            maneuver_state.get("start_candidate_sign", 0)
        )
        vehicle_state["external_state_maneuver_start_candidate_frame_count"] = int(
            maneuver_state.get("start_candidate_frames", 0)
        )
        vehicle_state["external_state_maneuver_armed_idle_frame_count"] = int(
            maneuver_state.get("armed_idle_frames", 0)
        )
        vehicle_state["external_state_maneuver_confirmed_direction_sign"] = int(
            maneuver_state.get("confirmed_direction_sign", 0)
        )
        vehicle_state["external_state_maneuver_confirmed_sumo_time"] = maneuver_state.get(
            "confirmed_sumo_time"
        )
        confirmed_direction = maneuver_state.get("confirmed_world_direction") or (
            None,
            None,
        )
        vehicle_state["external_state_maneuver_confirmed_direction_x"] = confirmed_direction[0]
        vehicle_state["external_state_maneuver_confirmed_direction_y"] = confirmed_direction[1]
        vehicle_state["external_state_maneuver_switch_evidence_excluded"] = "switched" in (
            maneuver_state.get("evidence_exclusion_reason", "")
        )
        vehicle_state["sumo_lane_change_target_lane_id"] = maneuver_state.get("target_lane_id", "")
        vehicle_state["lookahead_action_mode"] = action.get("mode", "route")
        vehicle_state["lookahead_action_valid"] = bool(action.get("valid", False))
        vehicle_state["lookahead_action_error"] = action.get("error", "")
        vehicle_state["lookahead_action_warning"] = action.get("warning", "")
        vehicle_state["lookahead_lateral_direction_source"] = action.get(
            "lateral_direction_source", "inactive"
        )
        vehicle_state["lookahead_lateral_direction_unresolved_count"] = unresolved_count
        vehicle_state["lookahead_lane_change_intent_conflict"] = intent_conflict
        vehicle_state["lookahead_lateral_horizon_displacement"] = float(
            action.get("lateral_displacement", 0.0)
        )
        vehicle_state["lookahead_target_lateral_distance"] = action.get("target_lateral_distance")
        vehicle_state["lookahead_route_tangent_x"] = action.get("route_tangent_x", 0.0)
        vehicle_state["lookahead_route_tangent_y"] = action.get("route_tangent_y", 0.0)
        vehicle_state["lookahead_world_left_normal_x"] = action.get("world_left_normal_x")
        vehicle_state["lookahead_world_left_normal_y"] = action.get("world_left_normal_y")
        vehicle_state["lookahead_target_route_tangent_x"] = action.get("target_route_tangent_x")
        vehicle_state["lookahead_target_route_tangent_y"] = action.get("target_route_tangent_y")
        vehicle_state["lookahead_target_world_left_normal_x"] = action.get(
            "target_world_left_normal_x"
        )
        vehicle_state["lookahead_target_world_left_normal_y"] = action.get(
            "target_world_left_normal_y"
        )
        vehicle_state["lookahead_phase_b_lateral_delta"] = action.get("phase_b_lateral_delta")
        vehicle_state["canonical_actual_lane_id"] = (
            compiled_path.get("canonical_actual_lane_id", "") if compiled_path is not None else ""
        )
        vehicle_state["canonical_reference_lane_id"] = (
            compiled_path.get("canonical_reference_lane_id", "")
            if compiled_path is not None
            else ""
        )
        vehicle_state["canonical_route_successor_edge_id"] = (
            compiled_path.get("canonical_route_successor_edge_id", "")
            if compiled_path is not None
            else ""
        )
        vehicle_state["canonical_route_selection_reason"] = (
            compiled_path.get("canonical_route_selection_reason", "")
            if compiled_path is not None
            else ""
        )
        vehicle_state["canonical_route_selection_error"] = route_selection_error
        vehicle_state["canonical_planned_link_selectors"] = (
            compiled_path.get("canonical_planned_link_selectors", ())
            if compiled_path is not None
            else ()
        )
        vehicle_state["lookahead_expected_phase_b_lateral_distance"] = action.get(
            "expected_phase_b_lateral_distance"
        )
        vehicle_state["lookahead_world_lateral_speed"] = action.get("world_lateral_speed", 0.0)
        vehicle_state["canonical_route_lane_ids"] = (
            tuple(compiled_path.get("lane_ids", ())) if compiled_path is not None else ()
        )
        vehicle_state["canonical_route_path_key"] = canonical_path_key
        vehicle_state["canonical_route_occurrence_ids"] = (
            tuple(compiled_path.get("route_occurrence_ids", ()))
            if compiled_path is not None
            else ()
        )
        vehicle_state["canonical_rolling_history_lane_ids"] = (
            tuple(compiled_path.get("rolling_history_lane_ids", ()))
            if compiled_path is not None
            else ()
        )
        vehicle_state["canonical_rolling_history_distance"] = rolling_history_distance
        vehicle_state["canonical_rolling_retained_history_distance"] = (
            compiled_path.get("rolling_retained_history_distance")
            if compiled_path is not None
            else None
        )
        vehicle_state["canonical_route_path_switched"] = bool(
            maneuver_state.get("canonical_path_switched", False)
        )
        phase_a_projection = action.get("canonical_phase_a") or {}
        phase_a_evidence_projection = action.get("canonical_evidence_phase_a") or {}
        vehicle_state["canonical_evidence_phase_a_s"] = phase_a_evidence_projection.get("s")
        vehicle_state["canonical_evidence_phase_a_d"] = phase_a_evidence_projection.get("d")
        phase_b_projection = action.get("canonical_phase_b") or {}
        vehicle_state["canonical_phase_a_s"] = phase_a_projection.get("s")
        vehicle_state["canonical_phase_a_d"] = phase_a_projection.get("d")
        vehicle_state["canonical_phase_a_projection_distance"] = phase_a_projection.get(
            "projection_distance"
        )
        vehicle_state["canonical_phase_a_projection_along_residual"] = phase_a_projection.get(
            "projection_along_residual"
        )
        vehicle_state["canonical_phase_a_projection_endpoint_clamped"] = bool(
            phase_a_projection.get("projection_endpoint_clamped", False)
        )
        vehicle_state["canonical_phase_b_s"] = phase_b_projection.get("s")
        vehicle_state["canonical_phase_b_d"] = phase_b_projection.get("d")
        vehicle_state["canonical_phase_b_projection_distance"] = phase_b_projection.get(
            "projection_distance"
        )
        vehicle_state["canonical_phase_b_projection_along_residual"] = phase_b_projection.get(
            "projection_along_residual"
        )
        vehicle_state["canonical_phase_b_projection_endpoint_clamped"] = bool(
            phase_b_projection.get("projection_endpoint_clamped", False)
        )
        vehicle_state["canonical_delta_d"] = action.get("canonical_delta_d")
        for field in (
            "phase_a_to_b_delta_x",
            "phase_a_to_b_delta_y",
            "phase_a_to_b_velocity_x",
            "phase_a_to_b_velocity_y",
            "phase_a_to_b_requested_longitudinal_speed",
            "phase_a_to_b_requested_lateral_speed",
            "phase_a_to_b_motion_heading_radians",
            "phase_a_to_b_motion_heading_degrees",
        ):
            vehicle_state[field] = action.get(field)
        vehicle_state["phase_a_to_b_motion_heading_valid"] = bool(
            action.get("phase_a_to_b_motion_heading_valid", False)
        )
        vehicle_state["phase_a_to_b_motion_heading_invalid_reason"] = action.get(
            "phase_a_to_b_motion_heading_invalid_reason", ""
        )
        vehicle_state["phase_a_to_b_lateral_motion_requested"] = bool(
            action.get("phase_a_to_b_lateral_motion_requested", False)
        )
        vehicle_state["canonical_target_s"] = action.get("canonical_target_s")
        vehicle_state["canonical_target_d"] = action.get("target_lateral_distance")
        vehicle_state["canonical_curvature_half_window"] = action.get(
            "canonical_curvature_half_window"
        )
        vehicle_state["path_tracker_valid"] = bool(action.get("path_tracker_valid", False))
        vehicle_state["path_tracker_center_curvature"] = action.get("path_tracker_center_curvature")
        vehicle_state["path_tracker_front_curvature"] = action.get("path_tracker_front_curvature")
        vehicle_state["path_tracker_rear_curvature"] = action.get("path_tracker_rear_curvature")
        vehicle_state["path_tracker_reference_tangent_x"] = action.get(
            "path_tracker_reference_tangent_x"
        )
        vehicle_state["path_tracker_reference_tangent_y"] = action.get(
            "path_tracker_reference_tangent_y"
        )
        vehicle_state["path_tracker_front_cross_track_error"] = action.get(
            "path_tracker_front_cross_track_error"
        )
        vehicle_state["path_tracker_motion_heading_applied"] = bool(
            action.get("path_tracker_motion_heading_applied", False)
        )
        vehicle_state["path_tracker_reference_heading_source"] = action.get(
            "path_tracker_reference_heading_source", "route_tangent"
        )
        vehicle_state["path_tracker_motion_heading_reason"] = action.get(
            "path_tracker_motion_heading_reason", ""
        )
        vehicle_state["canonical_held_d"] = maneuver_state.get("held_canonical_d")
        vehicle_state["canonical_hold_reprojected"] = bool(action.get("held_reprojected", False))
        vehicle_state["canonical_invariant_valid"] = bool(
            maneuver_state.get("canonical_invariant_valid", False)
        )
        vehicle_state["canonical_invariant_reason"] = maneuver_state.get(
            "canonical_invariant_reason", ""
        )
        vehicle_state["canonical_invariant_warning"] = maneuver_state.get(
            "canonical_invariant_warning", ""
        )
        vehicle_state["canonical_fallback_applied"] = bool(
            maneuver_state.get("canonical_fallback_applied", False)
        )
        vehicle_state["canonical_fallback_reason"] = maneuver_state.get(
            "canonical_fallback_reason", ""
        )
        vehicle_state["canonical_fallback_age"] = int(
            maneuver_state.get("canonical_fallback_age", 0)
        )
        vehicle_state["canonical_front_target_lateral_jump"] = maneuver_state.get(
            "canonical_front_jump"
        )
        vehicle_state["canonical_control_target_lateral_jump"] = maneuver_state.get(
            "canonical_control_jump"
        )
        vehicle_state["canonical_steering_preview_lateral_jump"] = maneuver_state.get(
            "canonical_steering_preview_jump"
        )
        vehicle_state["canonical_front_target_lateral_change"] = maneuver_state.get(
            "canonical_front_step"
        )
        vehicle_state["canonical_control_target_lateral_change"] = maneuver_state.get(
            "canonical_control_step"
        )
        vehicle_state["canonical_steering_preview_lateral_change"] = maneuver_state.get(
            "canonical_steering_preview_step"
        )
        vehicle_state["steering_preview_requested_distance"] = action.get(
            "steering_preview_requested_distance"
        )
        vehicle_state["steering_preview_effective_distance"] = action.get(
            "steering_preview_effective_distance"
        )
        vehicle_state["steering_preview_front_s"] = action.get("steering_preview_front_s")
        vehicle_state["steering_preview_front_d"] = action.get("steering_preview_front_d")
        vehicle_state["steering_preview_tangent_x"] = action.get("steering_preview_tangent_x")
        vehicle_state["steering_preview_tangent_y"] = action.get("steering_preview_tangent_y")
        vehicle_state["steering_preview_normal_x"] = action.get("steering_preview_normal_x")
        vehicle_state["steering_preview_normal_y"] = action.get("steering_preview_normal_y")
        vehicle_state["front_rear_distance_error"] = action.get("front_rear_distance_error")
        vehicle_state["front_rear_distance_consistent"] = bool(
            action.get("front_rear_distance_consistent", False)
        )
        vehicle_state["canonical_current_d_outside_lane"] = action.get("current_d_outside_lane")
        vehicle_state["canonical_held_d_outside_lane"] = action.get("held_d_outside_lane")
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
        control_lookahead = action.get("control_lookahead")
        vehicle_state["control_lookahead_position_valid"] = False
        if control_lookahead is not None:
            vehicle_state["control_lookahead_x"] = control_lookahead[0]
            vehicle_state["control_lookahead_y"] = control_lookahead[1]
            vehicle_state["control_lookahead_z"] = control_lookahead[2]
            vehicle_state["control_lookahead_position_valid"] = True
        steering_preview = action.get("steering_preview")
        vehicle_state["steering_preview_position_valid"] = False
        if steering_preview is not None:
            vehicle_state["steering_preview_x"] = steering_preview[0]
            vehicle_state["steering_preview_y"] = steering_preview[1]
            vehicle_state["steering_preview_z"] = steering_preview[2]
            vehicle_state["steering_preview_position_valid"] = True

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

    def _populate_av_lane_change_diagnostics(self, vid, vehicle_state):
        if vid != "AV":
            return
        request = dict(self.av_nde_lane_change_request or {})
        controller_diagnostics = self._av_nde_controller_diagnostics()
        vehicle_state.update(
            {
                "av_control_mode": self.av_control_mode,
                "av_lane_change_decision": dict(self.av_nde_lane_change_decision),
                "av_lane_change_request": request,
                "effective_speed_mode": int(traci.vehicle.getSpeedMode(vid)),
                "nde_lane_change_decision_source": str(request.get("decision_source") or ""),
                "nde_lane_change_command": str(request.get("direction") or "none"),
                "nde_lane_change_request_count": int(
                    request.get("count", self.av_nde_lane_change_request_count) or 0
                ),
                "nde_lane_change_request_time": request.get("simulation_time"),
                "nde_lane_change_source_lane_id": str(request.get("source_lane_id") or ""),
                "nde_lane_change_target_lane_id": str(request.get("target_lane_id") or ""),
                "nde_lane_change_duration": request.get("duration"),
                "nde_lane_change_model_state": request.get("model_lane_change_state"),
                "nde_lane_change_post_state": request.get("post_lane_change_state"),
                "nde_controller_busy": controller_diagnostics["busy"],
                "nde_command_end_reason": controller_diagnostics["command_end_reason"],
            }
        )
        try:
            vehicle_state["effective_lane_change_mode"] = int(traci.vehicle.getLaneChangeMode(vid))
        except Exception:
            vehicle_state["effective_lane_change_mode"] = None

    def _build_simulation_state(self, simulator):
        """Collect the current simulation state from SUMO into a SimulationState.

        Pure state construction (no network I/O); raises on TraCI errors.
        """
        simulation_state = SimulationState()
        simulation_time = traci.simulation.getTime()
        simulation_state.simulation_time = simulation_time

        # Get all interested agent IDs
        all_vehicle_ids, vru_ids, construction_ids = self.get_vehicle_vru_ids()
        vehicle_ids, vehicle_position_cache = self._filter_vehicle_ids_for_state(all_vehicle_ids)
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
            # The patched immediate external-state move currently makes
            # libsumo return a 2-D tuple for the moved actor even through
            # getPosition3D(). Preserve the CARLA feedback height in that
            # case; regular SUMO actors continue to use their 3-D value.
            if len(position) >= 3:
                x, y, z = position[:3]
            else:
                x, y = position[:2]
                observation = getattr(self, "physics_feedback_observations", {}).get(vid, {})
                z = observation.get("z")
                z = 0.0 if z is None else z
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
            self._populate_av_lane_change_diagnostics(vid, vehicle_state)
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
                self.logger.warning(
                    f"VRU ID {vru_id} not found in vehicle or person lists, skipping"
                )
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
                last_orientation, last_time = last_orientations.get(
                    vru_id, (orientation, simulation_time)
                )
                dt = simulation_time - last_time
                if dt > 0:
                    dtheta = orientation - last_orientation
                    angular_velocity = math.atan2(math.sin(dtheta), math.cos(dtheta)) / dt
                last_orientations[vru_id] = (orientation, simulation_time)
                acceleration = domain.getAcceleration(vru_id)
            else:
                acceleration = (
                    domain.getAcceleration(vru_id) if hasattr(domain, "getAcceleration") else 0.0
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
                tls_information = {"programs": {}}
                tls = self.simulator.sumo_net.getTLS(tl_id)
                programs = tls.getPrograms()
                for program_id, program in programs.items():
                    # Get the program parameters
                    program_parameters = program.getParams()
                    tls_information["programs"][program_id] = {"parameters": program_parameters}
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
        if (
            self.construction_zone_shapes is None
            and simulator.env.static_adversity is not None
            and simulator.env.static_adversity.adversities is not None
        ):
            self.construction_zone_shapes = {}
            for adversity in simulator.env.static_adversity.adversities:
                if isinstance(adversity, ConstructionAdversity):
                    lane_shape = traci.lane.getShape(adversity._lane_id)
                    if lane_shape:  # convert to list of lists
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
                        construction_zone_shape = generate_construction_zone_shape(
                            lane_shape,
                            traci.lane.getWidth(adversity._lane_id),
                            direction,
                        )
                        self.construction_zone_shapes[adversity._lane_id] = construction_zone_shape

        simulation_state.construction_zone_details = self.construction_zone_shapes
        return simulation_state

    # ------------------------------------------------------------------
    # agent command application
    # ------------------------------------------------------------------
    def _remove_background_vehicle(self, actor_id, reason, metadata=None):
        """Remove one non-AV vehicle from SUMO and clear its feedback state."""
        if not actor_id or actor_id == "AV":
            self.logger.error(
                "Refusing automatic SUMO removal for protected actor=%r reason=%s",
                actor_id,
                reason,
            )
            return False
        try:
            traci.vehicle.remove(actor_id)
        except Exception as exc:
            self.logger.warning(
                "Background SUMO vehicle removal failed: actor=%s reason=%s error=%s",
                actor_id,
                reason,
                exc,
            )
            return False
        self.physics_feedback_observations.pop(actor_id, None)
        self.physics_feedback_actor_membership.discard(actor_id)
        self.physics_lateral_direction_states.pop(actor_id, None)
        self.physics_rolling_path_states.pop(actor_id, None)
        self.physics_feedback_generations.pop(actor_id, None)
        details = metadata or {}
        self.logger.warning(
            "Removed collided SUMO background vehicle: actor=%s reason=%s "
            "frame=%s lane=%s position=%s",
            actor_id,
            reason,
            details.get("collision_frame"),
            details.get("sumo_lane_id", ""),
            details.get("carla_location"),
        )
        return True

    def _apply_agent_command(self, command):
        """Apply a parsed AgentCommand to the running simulation."""
        try:
            if command.agent_id != "":
                if command.agent_type not in ["vehicle", "vru"]:
                    self.logger.error(f"Invalid agent type: {command.agent_type}")
                    return False
                if command.agent_id in self.controlled_agents_each_step:
                    self.logger.debug(f"Agent {command.agent_id} is already controlled")
                    return True
                self.controlled_agents_each_step.add(command.agent_id)
                if command.command_type == "remove":
                    if command.agent_type != "vehicle":
                        self.logger.error("Only SUMO vehicles support remove commands")
                        return False
                    return self._remove_background_vehicle(
                        command.agent_id,
                        command.data.get("reason", "collision"),
                        metadata=command.data,
                    )
                if command.command_type != "set_state":
                    self.logger.error(f"Invalid agent command type: {command.command_type}")
                    return False
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
                            command.agent_id,
                            "",
                            0,
                            x,
                            y,
                            command.data.get("sumo_angle", 0),
                            0,
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
                                command.agent_id,
                                "",
                                0,
                                x,
                                y,
                                command.data.get("sumo_angle", 0),
                                2,
                            )
                            if "speed" in command.data:
                                traci.vehicle.setPreviousSpeed(
                                    command.agent_id, command.data["speed"]
                                )
                        elif command.agent_id in current_person_list:
                            # VRU is actually a person
                            traci.person.moveToXY(
                                command.agent_id,
                                "",
                                x,
                                y,
                                command.data.get("sumo_angle", 0),
                                2,
                            )
                            if "speed" in command.data:
                                traci.person.setSpeed(command.agent_id, command.data["speed"])
                        else:
                            self.logger.error(
                                f"VRU ID {command.agent_id} not found in vehicle or person lists"
                            )
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
                if command.agent_id != "AV" and self.physics_background_invalid_policy == "demote":
                    self.physics_feedback_observations.pop(command.agent_id, None)
                    self.logger.warning(
                        "Background physical feedback rejected; continuing so CARLA "
                        "can demote actor=%s reason=%s",
                        command.agent_id,
                        e,
                    )
                    return False
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
