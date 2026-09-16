import json
import math
import os
import random
import re
import statistics
import threading
import time
import xml.etree.ElementTree as ET

import carla
import yaml

from .ackermann_control import (
    AckermannControllerTuning,
    AckermannEmergencyBrakeTuning,
    AckermannTuning,
    compute_ackermann_control_values,
    compute_direct_brake_value,
    horizontal_speed,
)
from .tools import (
    carla_to_sumo,
    create_bike_blueprint,
    create_bikeandmotor_blueprint,
    create_motor_blueprint,
    create_pedestrian_blueprint,
    create_police_car_blueprint,
    create_vehicle_blueprint,
    destroy_all_actors,
    draw_text,
    get_actor_id_from_attribute,
    log_spawn_actor_failure,
    spawn_actor,
    sumo_point_to_carla,
    sumo_to_carla,
)

AV_SUMO_ID = "AV"
SUMO_CARLA_TLS_LINK_PREFIX = "linkSignalID:"
VEHICLE_CONTROL_MODE_TELEPORT = "teleport"
VEHICLE_CONTROL_MODE_ACKERMANN_PHYSICS = "ackermann_physics"


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        print(f"Warning: invalid {name}={value!r}; using {default}.", flush=True)
        return default


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Warning: invalid {name}={value!r}; using {default}.", flush=True)
        return default


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


class CarlaCosim(object):
    def __init__(self, args):
        self.args = args

        self.client = carla.Client(args.carla_host, args.carla_port)
        self.client.set_timeout(getattr(args, 'carla_timeout', 10.0))

        self.world = self.client.get_world()
        if args.map_name:
            print(f"Loading map {args.map_name}")
            try:
                self.world = self.client.load_world(args.map_name)
            except:
                print(f"Map {args.map_name} not found. Loading default map.")
        else:
            print("No map name provided. Loading default map.")

        self.traffic_lights = self.world.get_actors().filter("traffic.traffic_light")
        for traffic_light in self.traffic_lights:
            traffic_light.set_state(carla.TrafficLightState.Off)
            traffic_light.freeze(True)

        self.control_av = args.control_av
        self.initialize_av = False
        self.av_shape = []
        self.step_length = args.step_length
        # Stage 3a clock-master mode (run_cosim --tick_mode master): this
        # process owns world.tick() on a fixed step_length wall-clock cadence.
        # Mutually exclusive with args.passive_tick (follow mode).
        self.tick_master = bool(getattr(args, "tick_master", False))
        self._next_tick_deadline = None  # monotonic deadline of the next tick
        # world.tick() duration stats (master mode): p50/p95/max + 5ms-bin
        # histogram printed every ~60s. Cheap (2 monotonic calls + one append
        # per cycle), so always on; this is the direct measurement of the
        # world-update time whose variance shifts frame-done arrival intervals.
        self._tick_times_ms = []
        self._tick_time_hist = [0] * 11  # 0-5,5-10,...,45-50,50+ ms
        # Timing must always be read together with the vehicle count (lesson
        # 2026-07-28): min/max of CARLA-synced vehicles within the same window.
        self._tick_veh_min = None
        self._tick_veh_max = None
        # Stage 3b async mode (run_cosim --tick_mode async): nobody ticks the
        # world; the CARLA server free-runs with a variable delta and this
        # process paces SUMO on a step_length wall-clock grid (_tick_async).
        self.tick_async = bool(getattr(args, "tick_async", False))
        # Catch-up cap: a late cycle runs at most this many SUMO steps in one
        requested_vehicle_control_mode = os.environ.get(
            "CARLA_COSIM_VEHICLE_CONTROL_MODE", VEHICLE_CONTROL_MODE_TELEPORT
        )
        self.vehicle_control_mode = str(requested_vehicle_control_mode).strip().lower()
        if self.vehicle_control_mode not in {
            VEHICLE_CONTROL_MODE_TELEPORT,
            VEHICLE_CONTROL_MODE_ACKERMANN_PHYSICS,
        }:
            raise ValueError(
                "CARLA_COSIM_VEHICLE_CONTROL_MODE must be teleport or ackermann_physics"
            )
        self.ackermann_physics_enabled = (
            self.vehicle_control_mode == VEHICLE_CONTROL_MODE_ACKERMANN_PHYSICS
        )
        feedback_mode = os.environ.get(
            "CARLA_COSIM_ACKERMANN_FEEDBACK_MODE", "off"
        ).strip().lower()
        if feedback_mode not in {"off", "shadow", "apply"}:
            raise ValueError(
                "CARLA_COSIM_ACKERMANN_FEEDBACK_MODE must be off, shadow, or apply"
            )
        feedback_actor_value = os.environ.get(
            "CARLA_COSIM_ACKERMANN_FEEDBACK_ACTORS", ""
        )
        self.ackermann_feedback_actor_ids = {
            actor_id.strip()
            for actor_id in feedback_actor_value.split(",")
            if actor_id.strip()
        }
        self.ackermann_feedback_apply_enabled = bool(
            self.ackermann_physics_enabled
            and feedback_mode == "apply"
            and self.ackermann_feedback_actor_ids
        )
        if self.ackermann_feedback_apply_enabled:
            if not self.tick_master:
                raise ValueError(
                    "physical co-sim requires --tick_mode master so Phase A/B "
                    "and CARLA frames remain serial"
                )
            if AV_SUMO_ID in self.ackermann_feedback_actor_ids:
                raise ValueError(
                    "Autoware owns the AV; physical feedback actors must be "
                    "background SUMO IDs or '*'"
                )
        self._ackermann_actor_state = {}
        self._physics_feedback_frames = {}
        self._physics_feedback_failures = {}
        self._pending_authoritative_action_error = None
        self._ackermann_feedback_state = {}
        self._ackermann_fail_closed_reasons = {}
        self.ackermann_feedback_ack_max_frame_lag = max(
            0, _env_int("CARLA_COSIM_ACKERMANN_FEEDBACK_ACK_MAX_FRAME_LAG", 2)
        )
        self.ackermann_feedback_ack_failure_limit = max(
            1, _env_int("CARLA_COSIM_ACKERMANN_FEEDBACK_ACK_FAILURE_LIMIT", 3)
        )
        ackermann_restart_enter_speed = max(
            0.0,
            _env_float("CARLA_COSIM_ACKERMANN_RESTART_ENTER_SPEED", 0.05),
        )
        ackermann_restart_release_speed = max(
            ackermann_restart_enter_speed,
            _env_float("CARLA_COSIM_ACKERMANN_RESTART_RELEASE_SPEED", 0.2),
        )
        self.ackermann_tuning = AckermannTuning(
            wheel_base=max(
                0.1, _env_float("CARLA_COSIM_ACKERMANN_WHEEL_BASE", 2.8)
            ),
            max_steer_rad=max(
                0.0, _env_float("CARLA_COSIM_ACKERMANN_MAX_STEER_RAD", 0.6)
            ),
            max_steer_rate_rad_s=max(
                0.0,
                _env_float(
                    "CARLA_COSIM_ACKERMANN_MAX_STEER_RATE_RAD_S", 0.6
                ),
            ),
            position_speed_gain=max(
                0.0,
                _env_float("CARLA_COSIM_ACKERMANN_POSITION_SPEED_GAIN", 1.0),
            ),
            kp_speed=_env_float("CARLA_COSIM_ACKERMANN_KP_SPEED", 0.8),
            kp_position=_env_float("CARLA_COSIM_ACKERMANN_KP_POSITION", 0.15),
            max_accel=max(
                0.0, _env_float("CARLA_COSIM_ACKERMANN_MAX_ACCEL", 3.0)
            ),
            max_decel=max(
                0.0, _env_float("CARLA_COSIM_ACKERMANN_MAX_DECEL", 6.0)
            ),
            restart_enter_speed=ackermann_restart_enter_speed,
            restart_release_speed=ackermann_restart_release_speed,
            restart_speed_epsilon=max(
                0.0,
                _env_float("CARLA_COSIM_ACKERMANN_RESTART_SPEED_EPSILON", 1e-3),
            ),
            restart_max_target_speed=max(
                ackermann_restart_release_speed,
                _env_float("CARLA_COSIM_ACKERMANN_RESTART_MAX_TARGET_SPEED", 0.3),
            ),
        )
        self.ackermann_controller_tuning = AckermannControllerTuning(
            speed_kp=max(
                0.0,
                _env_float(
                    "CARLA_COSIM_ACKERMANN_CONTROLLER_SPEED_KP", 1.0
                ),
            ),
            speed_ki=max(
                0.0,
                _env_float(
                    "CARLA_COSIM_ACKERMANN_CONTROLLER_SPEED_KI", 0.0
                ),
            ),
            speed_kd=max(
                0.0,
                _env_float(
                    "CARLA_COSIM_ACKERMANN_CONTROLLER_SPEED_KD", 0.0
                ),
            ),
            accel_kp=max(
                0.0,
                _env_float(
                    "CARLA_COSIM_ACKERMANN_CONTROLLER_ACCEL_KP", 0.05
                ),
            ),
            accel_ki=max(
                0.0,
                _env_float(
                    "CARLA_COSIM_ACKERMANN_CONTROLLER_ACCEL_KI", 0.0
                ),
            ),
            accel_kd=max(
                0.0,
                _env_float(
                    "CARLA_COSIM_ACKERMANN_CONTROLLER_ACCEL_KD", 0.0
                ),
            ),
        )
        emergency_engage_decel = max(
            0.0,
            _env_float(
                "CARLA_COSIM_ACKERMANN_EMERGENCY_BRAKE_ENGAGE_DECEL", 4.0
            ),
        )
        self.ackermann_emergency_brake_tuning = AckermannEmergencyBrakeTuning(
            enabled=_env_bool(
                "CARLA_COSIM_ACKERMANN_EMERGENCY_BRAKE_ENABLED", True
            ),
            engage_decel=emergency_engage_decel,
            release_decel=min(
                emergency_engage_decel,
                max(
                    0.0,
                    _env_float(
                        "CARLA_COSIM_ACKERMANN_EMERGENCY_BRAKE_RELEASE_DECEL",
                        1.0,
                    ),
                ),
            ),
            release_ticks=max(
                1,
                _env_int(
                    "CARLA_COSIM_ACKERMANN_EMERGENCY_BRAKE_RELEASE_TICKS", 3
                ),
            ),
            stop_speed=max(
                0.0,
                _env_float(
                    "CARLA_COSIM_ACKERMANN_EMERGENCY_BRAKE_STOP_SPEED", 0.2
                ),
            ),
            min_brake=min(
                1.0,
                max(
                    0.0,
                    _env_float(
                        "CARLA_COSIM_ACKERMANN_EMERGENCY_BRAKE_MIN_BRAKE", 0.5
                    ),
                ),
            ),
        )
        self.ackermann_warn_error_m = max(
            0.0, _env_float("CARLA_COSIM_ACKERMANN_WARN_ERROR_M", 3.0)
        )
        self.ackermann_snap_error_m = max(
            self.ackermann_warn_error_m,
            _env_float("CARLA_COSIM_ACKERMANN_SNAP_ERROR_M", 8.0),
        )
        self.ackermann_warning_interval = max(
            0.0, _env_float("CARLA_COSIM_ACKERMANN_WARNING_INTERVAL", 2.0)
        )
        self.ackermann_control_log_records = _env_bool(
            "CARLA_COSIM_ACKERMANN_CONTROL_LOG_RECORDS", False
        )
        control_log_actor_value = os.environ.get(
            "CARLA_COSIM_ACKERMANN_CONTROL_LOG_ACTORS", ""
        )
        self.ackermann_control_log_actor_ids = {
            actor_id.strip()
            for actor_id in control_log_actor_value.split(",")
            if actor_id.strip()
        }
        self.collision_sensor_enabled = _env_bool(
            "CARLA_COSIM_COLLISION_SENSOR_ENABLED", False
        )
        self.collision_log_path = os.environ.get(
            "CARLA_COSIM_COLLISION_LOG", "/app/outputs/carla_collision_events.jsonl"
        ).strip()
        self.collision_summary_path = os.environ.get(
            "CARLA_COSIM_COLLISION_SUMMARY", "/app/outputs/carla_collision_summary.json"
        ).strip()
        self.collision_episode_gap_frames = max(
            1, _env_int("CARLA_COSIM_COLLISION_EPISODE_GAP_FRAMES", 10)
        )
        self.initialization_diagnostics_enabled = _env_bool(
            "CARLA_COSIM_INITIALIZATION_DIAGNOSTICS_ENABLED", False
        )
        self.initialization_log_path = os.environ.get(
            "CARLA_COSIM_INITIALIZATION_LOG",
            "/app/outputs/carla_physics_initialization.jsonl",
        ).strip()
        self._diagnostic_lock = threading.Lock()
        self._collision_sensors = {}
        self._collision_seen_frame_pairs = set()
        self._collision_last_pair_frame = {}
        self._collision_raw_event_count = 0
        self._collision_unique_frame_count = 0
        self._collision_episode_count = 0
        self._collision_episode_counts_by_pair = {}
        self._initialization_failure_counts = {}
        if self.collision_sensor_enabled:
            self._reset_diagnostic_path(self.collision_log_path)
            self._reset_diagnostic_path(self.collision_summary_path)
        if self.initialization_diagnostics_enabled:
            self._reset_diagnostic_path(self.initialization_log_path)
        if self.ackermann_feedback_apply_enabled:
            print(
                "CARLA physical co-sim enabled for SUMO background actors "
                f"{sorted(self.ackermann_feedback_actor_ids)!r}; "
                "Autoware ego remains externally owned.",
                flush=True,
            )
        # go; debt beyond it is dropped (SUMO slow motion, never a long step
        # burst) and reported as dropped= in [async-timing].
        self._async_catchup_max_steps = 5
        # Cycles without a new CARLA frame before warning that the world looks
        # stuck in synchronous mode again (a relaunched bridge re-applies it;
        # the runbook says restart this process to clear the settings).
        self._async_stall_warn_cycles = 40  # ~2s at 50ms
        # [async-timing] window stats, printed every ~60s like [tick-timing].
        # Window-scoped (reset after each print):
        self._async_work_ms = []
        self._async_sumo_ms = []
        self._async_write_ms = []
        self._async_overrun_cycles = 0
        self._async_catchup_steps = 0
        self._async_dropped_periods = 0
        self._async_frames_advanced = 0
        self._async_delta_ms_samples = []
        self._async_delta_over_100ms = 0
        self._async_veh_min = None
        self._async_veh_max = None
        self._async_sumo_veh_min = None
        self._async_sumo_veh_max = None
        self._async_window_start = None
        # Cumulative (never reset; feeds the drift figures):
        self._async_prev_frame = None
        self._async_stalled_cycles = 0
        self._async_sumo_steps_total = 0
        self._async_start_wall = None
        self._async_start_carla_elapsed = None
        self._async_last_carla_elapsed = None
        self._spawn_failures = {}
        self._missing_angle_warnings = set()
        self._invalid_location_warnings = set()
        self.use_lane_relative_position = _env_bool(
            "CARLA_COSIM_USE_LANE_RELATIVE_POSITION",
            bool(getattr(args, "use_lane_relative_position", False))
            or self.ackermann_feedback_apply_enabled,
        )
        if self.use_lane_relative_position:
            print("CARLA co-sim lane-relative reconstructed positions enabled.", flush=True)
        self.spawn_failure_backoff_seconds = max(
            0.0,
            _env_float("CARLA_COSIM_SPAWN_FAILURE_BACKOFF_SECONDS", 5.0),
        )
        self.spawn_failure_backoff_max_seconds = max(
            self.spawn_failure_backoff_seconds,
            _env_float("CARLA_COSIM_SPAWN_FAILURE_BACKOFF_MAX_SECONDS", 30.0),
        )
        self.spawn_max_attempts = max(
            1,
            _env_int("CARLA_COSIM_SPAWN_MAX_ATTEMPTS", 3),
        )
        self.batch_transform_enabled = _env_bool("CARLA_COSIM_BATCH_TRANSFORM", False)
        if self.batch_transform_enabled:
            print("CARLA co-sim ApplyTransform batching enabled.", flush=True)
        self.batch_spawn_enabled = _env_bool("CARLA_COSIM_BATCH_SPAWN", False)
        if self.batch_spawn_enabled:
            print("CARLA co-sim SpawnActor batching enabled.", flush=True)
        self.spawn_z_clearance = max(
            0.0,
            _env_float("CARLA_COSIM_SPAWN_Z_CLEARANCE", self.SPAWN_Z_CLEARANCE),
        )
        print(
            f"CARLA co-sim spawn Z clearance: {self.spawn_z_clearance:.1f}m.",
            flush=True,
        )
        self.actor_filter_enabled = _env_bool("CARLA_COSIM_ACTOR_FILTER", False)
        self.actor_filter_center_id = (
            os.environ.get("CARLA_COSIM_ACTOR_FILTER_CENTER_ID") or AV_SUMO_ID
        )
        self.actor_filter_radius = max(
            0.0,
            _env_float("CARLA_COSIM_ACTOR_FILTER_RADIUS", 300.0),
        )
        self._actor_filter_missing_center_warned = False
        if self.actor_filter_enabled:
            print(
                "CARLA co-sim actor radius filter enabled: "
                f"center={self.actor_filter_center_id} "
                f"radius={self.actor_filter_radius:.1f}m.",
                flush=True,
            )

        # Per-tick CARLA round-trips replaced by caches. Every RPC on this
        # thread runs while holding the GIL, so beyond its own latency it
        # stalls the sim thread's Python phases (command apply / state build).
        self._bp_library = self.world.get_blueprint_library()
        self._last_world_frame = None  # from wait_for_tick()/tick(); avoids get_snapshot()
        self._av_actor = None  # cached ego handle; re-resolved on failure/reconcile
        self.ego_label_enabled = _env_bool("CARLA_COSIM_EGO_LABEL", True)
        # role_name -> actor indexes, maintained incrementally at spawn/destroy.
        # A full world scan runs once at start and then every N ticks as a
        # consistency net (0 = never reconcile).
        self._vehicle_actor_index = {}
        self._pedestrian_actor_index = {}
        self._actor_index_seeded = False
        self._actor_index_reconcile_every = max(
            0, int(_env_float("CARLA_COSIM_ACTOR_INDEX_RECONCILE_EVERY", 600))
        )
        self._ticks_since_reconcile = 0

        self.vehicle_blueprints = create_vehicle_blueprint(self.world)
        self.motor_blueprints = create_motor_blueprint(self.world)
        self.pedestrian_blueprints = create_pedestrian_blueprint(self.world)
        self.police_car_blueprints = create_police_car_blueprint(self.world)
        self.bike_blueprints = create_bike_blueprint(self.world)
        self.bikeandmotor_blueprints = create_bikeandmotor_blueprint(self.world)

        # Connect to TeraSim: single-process mode only (terasim_service.run_cosim).
        # The TeraSim simulation loop runs on another thread of THIS process;
        # commands and states are exchanged as Python objects (no JSON). The
        # runner already waited for wait_for_tick.
        self.inprocess_plugin = getattr(args, "inprocess_plugin", None)
        if self.inprocess_plugin is None:
            raise ValueError(
                "CarlaCosim requires args.inprocess_plugin "
                "(TeraSimCoSimInProcessPlugin); the former Redis/HTTP and gRPC "
                "transports were removed. Use terasim_service.run_cosim."
            )
        self._inproc_tick_handle = None
        # Seed the render pipeline with the initial (post-warmup) state so the
        # first tick behaves like later ones (AV shape init etc.).
        self._inproc_prev_state = self.inprocess_plugin.get_result().state

        # Auto-calibrate SUMO-CARLA coordinate transformation
        self.sumo_carla_offset = [0.0, 0.0]
        self._coord_transformer = None
        self._sumo_net_offset = [0.0, 0.0]
        self._xodr_origin_utm = [0.0, 0.0]
        net_file = self._get_net_file_from_config(args.terasim_config)
        if net_file:
            result = self._calibrate_sumo_carla_offset(net_file)
            if result is not None:
                # Offset-based mode
                self.sumo_carla_offset = result
                print(f"SUMO-CARLA coordinate offset: dx={self.sumo_carla_offset[0]:.2f}, dy={self.sumo_carla_offset[1]:.2f}")
            else:
                print("Using projection-based coordinate transformation")

    @staticmethod
    def _get_net_file_from_config(config_path):
        """Extract SUMO net file path from the TeraSim scenario YAML config."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Try input.sumo_net_file first, then environment.parameters.sumo_net_file_path
            net_file = config.get('input', {}).get('sumo_net_file')
            if not net_file:
                net_file = config.get('environment', {}).get('parameters', {}).get('sumo_net_file_path')
            return net_file
        except Exception as e:
            print(f"Warning: Could not read config for net file path: {e}")
            return None

    @staticmethod
    def _parse_xodr_origin(xodr_proj):
        """Extract lat_0/lon_0 and UTM zone from an xodr geoReference proj string.
        Returns (origin_lat, origin_lon, utm_zone) or (None, None, None) if not parseable.
        """
        lat_0 = lon_0 = utm_zone = None
        m = re.search(r'\+lat_0=([0-9.eE+-]+)', xodr_proj)
        if m:
            lat_0 = float(m.group(1))
        m = re.search(r'\+lon_0=([0-9.eE+-]+)', xodr_proj)
        if m:
            lon_0 = float(m.group(1))
        m = re.search(r'\+zone=(\d+)', xodr_proj)
        if m:
            utm_zone = int(m.group(1))
        return lat_0, lon_0, utm_zone

    def _calibrate_sumo_carla_offset(self, net_file):
        """Build a coordinate transformer between SUMO net.xml and CARLA (xodr) coordinate systems.

        OpenDRIVE files from Lanelet2 pipelines use coordinates that are local offsets from the
        geoReference origin (lat_0, lon_0) projected into standard UTM. pyproj ignores lat_0/lon_0
        for +proj=utm, so we handle this by:
        1. Detecting the SUMO coordinate system (EPSG:3857 for Lanelet2 conversions)
        2. Converting SUMO CRS -> standard UTM (matching xodr zone)
        3. Subtracting the geoReference origin (projected to the same UTM) to get xodr-local coords

        Returns [offset_x, offset_y] for simple offset mode, or sets self._coord_transformer
        for full projection-based conversion (returns None).
        """
        # Parse SUMO net.xml <location> for projection info
        sumo_proj = None
        sumo_net_offset = [0.0, 0.0]
        try:
            tree = ET.parse(net_file)
            root = tree.getroot()
            loc_elem = root.find('.//location')
            if loc_elem is not None:
                sumo_proj = loc_elem.get('projParameter', '!')
                offset_str = loc_elem.get('netOffset', '0.00,0.00')
                parts = offset_str.split(',')
                sumo_net_offset = [float(parts[0]), float(parts[1])]
                print(f"SUMO net.xml: projParameter='{sumo_proj}', netOffset={sumo_net_offset}")
        except Exception as e:
            print(f"Warning: Could not parse SUMO net file {net_file}: {e}")

        # Get xodr geoReference from CARLA map
        xodr_proj = None
        try:
            opendrive_str = self.world.get_map().to_opendrive()
            xodr_tree = ET.fromstring(opendrive_str)
            geo_elem = xodr_tree.find('.//geoReference')
            if geo_elem is not None and geo_elem.text:
                xodr_proj = geo_elem.text.strip()
                print(f"CARLA xodr geoReference: '{xodr_proj}'")
        except Exception as e:
            print(f"Warning: Could not get xodr geoReference from CARLA: {e}")

        # Attempt projection-based transformation with origin offset handling
        if xodr_proj:
            try:
                import pyproj

                # Parse xodr origin (lat_0, lon_0) and UTM zone
                origin_lat, origin_lon, utm_zone = self._parse_xodr_origin(xodr_proj)

                # Determine SUMO CRS
                sumo_crs = None
                if sumo_proj and sumo_proj != '!':
                    sumo_crs = pyproj.CRS(sumo_proj)
                elif sumo_proj == '!':
                    # Detect CRS empirically from coordinate ranges
                    tree = ET.parse(net_file)
                    root = tree.getroot()
                    conv_boundary = root.find('.//location').get('convBoundary', '')
                    cb_parts = conv_boundary.split(',')
                    sample_x = (float(cb_parts[0]) + float(cb_parts[2])) / 2
                    sample_y = (float(cb_parts[1]) + float(cb_parts[3])) / 2

                    wgs84 = pyproj.CRS('EPSG:4326')
                    for crs_code in ['EPSG:3857', 'EPSG:32654', 'EPSG:6677']:
                        try:
                            candidate = pyproj.CRS(crs_code)
                            to_wgs84 = pyproj.Transformer.from_crs(candidate, wgs84, always_xy=True)
                            lon, lat = to_wgs84.transform(sample_x, sample_y)
                            if 100.0 < lon < 180.0 and -60.0 < lat < 85.0:
                                sumo_crs = candidate
                                print(f"Detected SUMO CRS as {crs_code} (sample -> lon={lon:.4f}, lat={lat:.4f})")
                                break
                        except Exception:
                            continue

                if sumo_crs is None:
                    print("Warning: Could not determine SUMO CRS. Falling back to empirical calibration.")
                    return self._empirical_calibration(net_file)

                # Build transformer: SUMO CRS -> standard UTM (same zone as xodr)
                if utm_zone:
                    utm_crs = pyproj.CRS(f'EPSG:326{utm_zone:02d}')
                else:
                    # Default to UTM zone from xodr proj string
                    utm_crs = pyproj.CRS(xodr_proj)

                self._coord_transformer = pyproj.Transformer.from_crs(sumo_crs, utm_crs, always_xy=True)
                self._sumo_net_offset = sumo_net_offset

                # Compute xodr origin in standard UTM
                self._xodr_origin_utm = [0.0, 0.0]
                if origin_lat is not None and origin_lon is not None:
                    wgs84 = pyproj.CRS('EPSG:4326')
                    to_utm = pyproj.Transformer.from_crs(wgs84, utm_crs, always_xy=True)
                    ox, oy = to_utm.transform(origin_lon, origin_lat)
                    self._xodr_origin_utm = [ox, oy]
                    print(f"xodr origin ({origin_lat:.6f}, {origin_lon:.6f}) in UTM: ({ox:.2f}, {oy:.2f})")

                print(f"Using projection-based transform: SUMO -> UTM{utm_zone} - origin")
                return None  # Signal to use transformer instead of offset

            except Exception as e:
                print(f"Warning: Projection-based calibration failed: {e}")
                import traceback
                traceback.print_exc()

        # Fallback: empirical median-based calibration
        return self._empirical_calibration(net_file)

    def _empirical_calibration(self, net_file):
        """Fallback: compute offset by comparing matching road coordinates."""
        sumo_edges = {}
        try:
            tree = ET.parse(net_file)
            root = tree.getroot()
            for edge_elem in root.iter('edge'):
                edge_id = edge_elem.get('id', '')
                if edge_id.startswith(':'):
                    continue
                for lane_elem in edge_elem.iter('lane'):
                    shape_str = lane_elem.get('shape', '')
                    if shape_str:
                        points = [tuple(map(float, p.split(','))) for p in shape_str.split()]
                        mid = points[len(points) // 2]
                        sumo_edges[edge_id] = (mid[0], mid[1])
                        break
        except Exception as e:
            print(f"Warning: Could not parse net file: {e}")
            return [0.0, 0.0]

        carla_roads = {}
        try:
            for w in self.world.get_map().generate_waypoints(200.0):
                rid = str(w.road_id)
                if rid not in carla_roads:
                    carla_roads[rid] = (w.transform.location.x, w.transform.location.y)
        except Exception as e:
            print(f"Warning: Could not get CARLA waypoints: {e}")
            return [0.0, 0.0]

        dxs, dys = [], []
        for edge_id, (sx, sy) in sumo_edges.items():
            if edge_id in carla_roads:
                cx, cy = carla_roads[edge_id]
                dxs.append(cx - sx)
                dys.append(cy + sy)

        if len(dxs) < 10:
            print(f"Warning: Only {len(dxs)} matching roads. Offset may be inaccurate.")
            if not dxs:
                return [0.0, 0.0]

        offset_x = statistics.median(dxs)
        offset_y = statistics.median(dys)
        print(f"Empirical calibration from {len(dxs)} matching roads")
        return [offset_x, offset_y]

    def _is_ackermann_feedback_actor(self, actor_id):
        return bool(
            self.ackermann_feedback_apply_enabled
            and actor_id != AV_SUMO_ID
            and (
                actor_id in self.ackermann_feedback_actor_ids
                or "*" in self.ackermann_feedback_actor_ids
            )
        )

    def _uses_ackermann_physics(self, actor_id):
        return self._is_ackermann_feedback_actor(actor_id)

    def _waits_for_first_phase_a_feedback(self, actor_id):
        """Keep non-master actors coasting until their first feedback is queued."""
        return bool(
            not getattr(self, "tick_master", False)
            and self._is_ackermann_feedback_actor(actor_id)
            and actor_id not in self._physics_feedback_frames
        )

    def _carla_transform_to_sumo_feedback_state(
        self,
        transform,
        shape,
        front_bumper_local_x=None,
        rear_axle_local_x=None,
    ):
        location = transform.location
        rotation = transform.rotation
        values = [
            self._as_finite_float(location.x),
            self._as_finite_float(location.y),
            self._as_finite_float(location.z),
            self._as_finite_float(rotation.yaw),
        ]
        if any(value is None for value in values):
            return None

        _, _, carla_z, carla_yaw = values
        if front_bumper_local_x is None:
            front_bumper_local_x = shape[0] / 2.0
        if rear_axle_local_x is None:
            rear_axle_local_x = -0.5 * getattr(
                getattr(self, "ackermann_tuning", AckermannTuning()),
                "wheel_base",
                2.8,
            )

        front_point = self._transform_local_x_point(
            transform, front_bumper_local_x
        )
        rear_axle_point = self._transform_local_x_point(
            transform, rear_axle_local_x
        )
        sumo_location = self._carla_point_to_sumo_position(*front_point)
        rear_axle_location = self._carla_point_to_sumo_position(*rear_axle_point)
        sumo_rotation = [rotation.pitch, carla_yaw + 90.0, rotation.roll]

        sumo_values = [
            self._as_finite_float(sumo_location[0]),
            self._as_finite_float(sumo_location[1]),
            self._as_finite_float(sumo_location[2]),
            self._as_finite_float(sumo_rotation[1]),
            self._as_finite_float(rear_axle_location[0]),
            self._as_finite_float(rear_axle_location[1]),
        ]
        if any(value is None for value in sumo_values):
            return None
        return {
            "position": [sumo_values[0], sumo_values[1]],
            # moveToXY only accepts x/y, so carry the physical front elevation
            # separately for route-candidate validation in the SUMO plugin.
            "position_z": sumo_values[2],
            "sumo_angle": sumo_values[3] % 360.0,
            "rear_axle_position": [sumo_values[4], sumo_values[5]],
        }

    def _carla_actor_to_sumo_feedback(self, actor_id, actor, vehicle_info):
        shape = [
            float(vehicle_info.get("length", 5.0)),
            float(vehicle_info.get("width", 1.8)),
            float(vehicle_info.get("height", 1.5)),
        ]
        state = self._ackermann_actor_state.setdefault(actor_id, {})
        self._initialize_ackermann_actor_geometry(actor, actor_id, state)
        transform = actor.get_transform()
        sumo_state = self._carla_transform_to_sumo_feedback_state(
            transform,
            shape,
            front_bumper_local_x=state.get(
                "front_bumper_local_x_m", shape[0] / 2.0
            ),
            rear_axle_local_x=state.get(
                "rear_axle_local_x_m", -0.5 * self.ackermann_tuning.wheel_base
            ),
        )
        velocity = actor.get_velocity()
        acceleration = actor.get_acceleration()
        speed = self._as_finite_float(horizontal_speed(velocity))
        acceleration_x = self._as_finite_float(getattr(acceleration, "x", None))
        acceleration_y = self._as_finite_float(getattr(acceleration, "y", None))
        if acceleration_x is None or acceleration_y is None:
            longitudinal_acceleration = None
        else:
            yaw = math.radians(float(transform.rotation.yaw))
            longitudinal_acceleration = self._as_finite_float(
                acceleration_x * math.cos(yaw) + acceleration_y * math.sin(yaw)
            )
        if sumo_state is None or speed is None or longitudinal_acceleration is None:
            raise ValueError("non_finite_feedback_state")
        return {
            "position": sumo_state["position"],
            "z": sumo_state["position_z"],
            "speed": speed,
            "acceleration": longitudinal_acceleration,
            "sumo_angle": sumo_state["sumo_angle"],
            "rear_axle_position": sumo_state["rear_axle_position"],
        }

    def _build_physics_feedback_commands(self):
        if not self.ackermann_feedback_apply_enabled:
            return []
        state = self._inproc_prev_state
        if not isinstance(state, dict):
            return []
        vehicles = state.get("agent_details", {}).get("vehicle", {})
        frame = self._last_world_frame
        if not isinstance(frame, int):
            return []

        commands = []
        # Preserve the feature implementation's deterministic Phase A order.
        # moveToXYImmediate updates SUMO's lane-change neighborhood eagerly, so
        # applying identical feedback states in dictionary insertion order can
        # produce a different car-following decision for lane-changing actors.
        for actor_id in sorted(vehicles):
            vehicle_info = vehicles[actor_id]
            if not self._is_ackermann_feedback_actor(actor_id):
                continue
            actor = self._vehicle_actor_index.get(actor_id)
            if actor is None:
                continue
            actor_state = self._ackermann_actor_state.setdefault(actor_id, {})
            if actor_state.get("physics_initialization_pending"):
                continue
            previous_frame = self._physics_feedback_frames.get(actor_id)
            if previous_frame is not None and frame <= previous_frame:
                reason = "stale_or_duplicate_carla_frame"
                self._register_physics_feedback_failure(actor_id, reason)
                self._apply_physics_fail_closed_brake(reason)
                continue
            try:
                feedback = self._carla_actor_to_sumo_feedback(
                    actor_id, actor, vehicle_info
                )
            except Exception as exc:
                reason = f"feedback_collection:{type(exc).__name__}"
                print(
                    f"Warning: physical feedback collection failed for "
                    f"{actor_id}: {exc}",
                    flush=True,
                )
                self._register_physics_feedback_failure(actor_id, reason)
                self._apply_physics_fail_closed_brake(reason)
                continue
            feedback["source_carla_frame"] = frame
            commands.append(
                {
                    "agent_id": actor_id,
                    "agent_type": "vehicle",
                    "command_type": "set_state",
                    "data": feedback,
                }
            )
            self._physics_feedback_frames[actor_id] = frame
            self._ackermann_feedback_state[actor_id] = {
                "feedback_status": "queued",
                "source_carla_frame": frame,
            }
            self._clear_physics_feedback_failure(actor_id)
        return commands

    @staticmethod
    def _full_brake_control(steer=0.0):
        return carla.VehicleControl(
            throttle=0.0,
            steer=max(-1.0, min(1.0, float(steer))),
            brake=1.0,
            hand_brake=False,
            reverse=False,
        )

    def _set_ackermann_fail_closed_reason(self, actor_id, reason):
        reasons = getattr(self, "_ackermann_fail_closed_reasons", None)
        if reasons is None:
            reasons = {}
            self._ackermann_fail_closed_reasons = reasons
        reason_key = reason.partition(":")[0]
        previous_key = reasons.get(actor_id)
        if previous_key == reason_key:
            return
        reasons[actor_id] = reason_key
        print(
            f"Physical co-sim actor={actor_id} fail-closed: {reason}",
            flush=True,
        )

    def _clear_ackermann_fail_closed_reason(self, actor_id):
        reasons = getattr(self, "_ackermann_fail_closed_reasons", None)
        if not reasons:
            return
        previous = reasons.pop(actor_id, None)
        if previous is not None:
            print(
                f"Physical co-sim actor={actor_id} control recovered",
                flush=True,
            )

    def _register_physics_feedback_failure(self, actor_id, reason):
        actor_state = self._ackermann_actor_state.setdefault(actor_id, {})
        current_frame = getattr(self, "_last_world_frame", None)
        if (
            isinstance(current_frame, int)
            and actor_state.get("last_feedback_failure_frame") == current_frame
        ):
            return self._physics_feedback_failures.get(actor_id, 1)
        if isinstance(current_frame, int):
            actor_state["last_feedback_failure_frame"] = current_frame
        failures = self._physics_feedback_failures.get(actor_id, 0) + 1
        self._physics_feedback_failures[actor_id] = failures
        self._ackermann_feedback_state[actor_id] = {
            "feedback_status": "rejected",
            "feedback_reason": reason,
        }
        self._set_ackermann_fail_closed_reason(actor_id, reason)
        if failures >= self.ackermann_feedback_ack_failure_limit:
            self._pending_authoritative_action_error = {
                "actor_id": actor_id,
                "reason": reason,
            }
        return failures

    def _clear_physics_feedback_failure(self, actor_id):
        self._physics_feedback_failures.pop(actor_id, None)

    def _apply_physics_fail_closed_brake(self, reason):
        count = 0
        for actor_id, actor in list((self._vehicle_actor_index or {}).items()):
            if actor is None or not self._is_ackermann_feedback_actor(actor_id):
                continue
            try:
                actor.apply_control(self._full_brake_control())
                count += 1
            except Exception:
                pass
        if count:
            print(
                f"Physical co-sim fail-closed brake: actors={count} reason={reason}",
                flush=True,
            )

    def _sumo_point_to_carla_location(self, sumo_location, z_offset=0.0):
        return sumo_point_to_carla(
            sumo_location, self._get_carla_offset(sumo_location, z_offset)
        )

    def _carla_point_to_sumo_position(self, carla_x, carla_y, carla_z=0.0):
        """Convert a physical CARLA point to SUMO coordinates without shape correction."""
        if getattr(self, "_coord_transformer", None) is not None:
            sumo_x, sumo_y = self._transform_xodr_to_sumo(carla_x, -carla_y)
        else:
            sumo_x = carla_x - self.sumo_carla_offset[0]
            sumo_y = -carla_y + self.sumo_carla_offset[1]
        return [sumo_x, sumo_y, carla_z]

    @staticmethod
    def _transform_local_x_point(transform, local_x):
        yaw = math.radians(transform.rotation.yaw)
        return (
            transform.location.x + math.cos(yaw) * local_x,
            transform.location.y + math.sin(yaw) * local_x,
            transform.location.z,
        )

    @staticmethod
    def _phase_aligned_longitudinal_error(
        current_rear_axle,
        desired_rear_axle,
        current_velocity,
        desired_heading,
        step_length,
    ):
        """Compare both rear-axle positions at the SUMO target time t + dt."""
        predicted_x = current_rear_axle[0] + float(
            getattr(current_velocity, "x", 0.0)
        ) * max(0.0, step_length)
        predicted_y = current_rear_axle[1] + float(
            getattr(current_velocity, "y", 0.0)
        ) * max(0.0, step_length)
        return (
            (desired_rear_axle[0] - predicted_x) * math.cos(desired_heading)
            + (desired_rear_axle[1] - predicted_y) * math.sin(desired_heading)
        )

    @classmethod
    def _phase_aligned_front_progress_error(
        cls,
        current_transform,
        desired_transform,
        front_bumper_local_x,
        current_velocity,
        desired_heading,
        step_length,
    ):
        """Return rear-axle progress error without mixing yaw/lateral error into it.

        SUMO and CARLA physical front positions are symmetric. Translating both
        front positions by the same rigid front-to-rear-axle offset cancels in
        their signed path-progress difference, while rotating the desired offset
        with SUMO lane yaw would incorrectly turn heading error into longitudinal
        error on curves.
        """
        current_front = cls._transform_local_x_point(
            current_transform, front_bumper_local_x
        )
        desired_front = cls._transform_local_x_point(
            desired_transform, front_bumper_local_x
        )
        return cls._phase_aligned_longitudinal_error(
            current_front,
            desired_front,
            current_velocity,
            desired_heading,
            step_length,
        )

    def _sumo_front_to_carla_transform(
        self,
        sumo_location,
        sumo_rotation,
        shape,
        offset,
        front_bumper_local_x=None,
    ):
        """Map SUMO's front-center reference point to a CARLA actor origin."""
        if front_bumper_local_x is None:
            front_bumper_local_x = shape[0] / 2.0
        front_location = sumo_point_to_carla(sumo_location, offset)
        carla_yaw = sumo_rotation[1] - 90.0
        heading = math.radians(carla_yaw)
        actor_location = carla.Location(
            x=front_location.x - math.cos(heading) * front_bumper_local_x,
            y=front_location.y - math.sin(heading) * front_bumper_local_x,
            z=front_location.z,
        )
        return carla.Transform(
            actor_location,
            carla.Rotation(
                pitch=sumo_rotation[0],
                yaw=carla_yaw,
                roll=sumo_rotation[2],
            ),
        )

    def _resolve_sumo_lookahead_location(self, veh_id, veh_info, sumo_location, sumo_angle):
        authoritative_feedback = bool(
            getattr(self, "ackermann_feedback_apply_enabled", False)
        ) and self._is_ackermann_feedback_apply_actor(veh_id)
        if authoritative_feedback and veh_info.get("lookahead_action_valid") is False:
            reason = veh_info.get("lookahead_action_error") or "invalid_action_geometry"
            raise ValueError(f"invalid authoritative SUMO lookahead for actor {veh_id}: {reason}")

        if bool(veh_info.get("lookahead_position_valid", False)):
            lookahead = [
                self._as_finite_float(veh_info.get("lookahead_x")),
                self._as_finite_float(veh_info.get("lookahead_y")),
                self._as_finite_float(veh_info.get("lookahead_z")),
            ]
            if lookahead[2] is None:
                lookahead[2] = sumo_location[2]
            if lookahead[0] is not None and lookahead[1] is not None:
                return lookahead
            if bool(
                getattr(self, "ackermann_feedback_apply_enabled", False)
            ) and self._is_ackermann_feedback_apply_actor(veh_id):
                raise ValueError(f"invalid authoritative SUMO lookahead for actor {veh_id}")

        if authoritative_feedback and veh_info.get("lookahead_action_mode") in {
            "route",
            "route_only_unresolved_lateral",
            "sumo_lateral_velocity",
            "deferred",
        }:
            raise ValueError(f"missing authoritative SUMO lookahead for actor {veh_id}")

        desired_speed = self._resolve_ackermann_desired_speed(veh_id, veh_info)
        lookahead_distance = min(15.0, max(7.0, desired_speed))
        heading = math.radians(90.0 - sumo_angle)
        return [
            sumo_location[0] + math.cos(heading) * lookahead_distance,
            sumo_location[1] + math.sin(heading) * lookahead_distance,
            sumo_location[2],
        ]

    @staticmethod
    def _set_actor_simulate_physics(actor, enabled):
        try:
            actor.set_simulate_physics(enabled)
        except Exception:
            pass

    @staticmethod
    def _ackermann_actor_footprint(actor, transform):
        """Return an oriented 2-D physical footprint for a CARLA vehicle."""
        try:
            bounding_box = actor.bounding_box
            box_location = bounding_box.location
            extent = bounding_box.extent
            actor_yaw = math.radians(float(transform.rotation.yaw))
            box_yaw = math.radians(float(getattr(bounding_box.rotation, "yaw", 0.0)))
            center_x = (
                float(transform.location.x)
                + math.cos(actor_yaw) * float(box_location.x)
                - math.sin(actor_yaw) * float(box_location.y)
            )
            center_y = (
                float(transform.location.y)
                + math.sin(actor_yaw) * float(box_location.x)
                + math.cos(actor_yaw) * float(box_location.y)
            )
            yaw = actor_yaw + box_yaw
            forward = (math.cos(yaw), math.sin(yaw))
            left = (-math.sin(yaw), math.cos(yaw))
            center_z = float(transform.location.z) + float(box_location.z)
            return {
                "center": (center_x, center_y),
                "axes": (forward, left),
                "half_extents": (float(extent.x), float(extent.y)),
                "center_z": center_z,
                "half_z": float(extent.z),
            }
        except Exception:
            return None

    @staticmethod
    def _ackermann_footprints_overlap(first, second, clearance=0.05):
        if abs(first["center_z"] - second["center_z"]) > (
            first["half_z"] + second["half_z"] + clearance
        ):
            return False

        delta = (
            second["center"][0] - first["center"][0],
            second["center"][1] - first["center"][1],
        )
        for axis in first["axes"] + second["axes"]:
            center_distance = abs(delta[0] * axis[0] + delta[1] * axis[1])
            first_radius = sum(
                half_extent
                * abs(local_axis[0] * axis[0] + local_axis[1] * axis[1])
                for half_extent, local_axis in zip(
                    first["half_extents"], first["axes"]
                )
            )
            second_radius = sum(
                half_extent
                * abs(local_axis[0] * axis[0] + local_axis[1] * axis[1])
                for half_extent, local_axis in zip(
                    second["half_extents"], second["axes"]
                )
            )
            if center_distance > first_radius + second_radius + clearance:
                return False
        return True

    def _ackermann_spawn_transform_is_clear(self, actor, veh_id, target_transform):
        footprint = self._ackermann_actor_footprint(actor, target_transform)
        if footprint is None:
            return True, None

        for other_id, other_actor in (getattr(self, "_vehicle_actor_index", None) or {}).items():
            if other_id == veh_id or other_actor is actor:
                continue
            other_state = self._ackermann_actor_state.get(other_id, {})
            if other_state.get("physics_ground_transform_reserved"):
                other_transform = other_state.get("physics_initialization_transform")
            elif other_state.get("physics_initialization_pending"):
                # Elevated, non-physical actors do not reserve road space until
                # one of them has been admitted to the physical scene.
                continue
            else:
                try:
                    other_transform = other_actor.get_transform()
                except Exception:
                    continue
            if other_transform is None:
                continue
            other_footprint = self._ackermann_actor_footprint(
                other_actor, other_transform
            )
            if other_footprint is not None and self._ackermann_footprints_overlap(
                footprint, other_footprint
            ):
                return False, other_id
        return True, None

    @staticmethod
    def _transform_with_z_offset(transform, z_offset):
        return carla.Transform(
            carla.Location(
                x=transform.location.x,
                y=transform.location.y,
                z=transform.location.z + z_offset,
            ),
            carla.Rotation(
                pitch=transform.rotation.pitch,
                yaw=transform.rotation.yaw,
                roll=transform.rotation.roll,
            ),
        )

    def _prepare_ackermann_actor_physics(
        self,
        actor,
        veh_id,
        initial_speed,
        ground_transform,
        elevated_transform,
    ):
        state = self._ackermann_actor_state.setdefault(veh_id, {})
        self._set_actor_simulate_physics(actor, False)
        self._zero_ackermann_actor_motion(actor)
        state["physics_enabled"] = False
        state["physics_initialization_transform"] = ground_transform
        state["physics_initialization_speed"] = initial_speed
        state["physics_initialization_pending"] = True
        attempt = int(state.get("physics_reinitialization_count", 0)) + 1
        self._record_initialization_diagnostic(
            "prepared",
            actor,
            veh_id,
            expected_transform=ground_transform,
            expected_speed=initial_speed,
            attempt=attempt,
        )
        is_clear, blocking_actor_id = self._ackermann_spawn_transform_is_clear(
            actor, veh_id, ground_transform
        )
        if not is_clear:
            reason = f"overlap={blocking_actor_id}"
            if self._register_ackermann_initialization_failure(
                actor,
                veh_id,
                initial_speed,
                ground_transform,
                reason,
            ):
                return False
            state["physics_overlap_deferred"] = True
            state["physics_overlap_blocking_actor"] = blocking_actor_id
            state["physics_ground_transform_reserved"] = False
            actor.set_transform(elevated_transform)
            self._record_initialization_diagnostic(
                "overlap_deferred",
                actor,
                veh_id,
                expected_transform=ground_transform,
                expected_speed=initial_speed,
                attempt=attempt,
                extra={"blocking_vehicle_id": blocking_actor_id},
            )
            print(
                f"CARLA physics spawn deferred for {veh_id!r}: physical footprint "
                f"overlaps {blocking_actor_id!r}.",
                flush=True,
            )
            return False

        state["physics_ground_transform_reserved"] = True
        actor.set_transform(ground_transform)
        state["physics_ground_transform_applied_frame"] = self._current_carla_frame()
        state["physics_ground_transform_wait_pending"] = True
        # Never enable physics in the same CARLA frame as set_transform. CARLA
        # needs one completed world tick to update the body, wheels, and road
        # contacts at the requested ground transform.
        return False

    @staticmethod
    def _reset_diagnostic_path(path):
        if not path:
            return
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8"):
            pass

    def _append_diagnostic_jsonl(self, path, payload):
        if not path:
            return
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        lock = getattr(self, "_diagnostic_lock", None)
        if lock is None:
            with open(path, "a", encoding="utf-8") as output:
                output.write(line + "\n")
            return
        with lock:
            with open(path, "a", encoding="utf-8") as output:
                output.write(line + "\n")

    @staticmethod
    def _diagnostic_vector(vector):
        if vector is None:
            return None
        return {
            axis: float(getattr(vector, axis, 0.0))
            for axis in ("x", "y", "z")
        }

    @classmethod
    def _diagnostic_transform(cls, transform):
        if transform is None:
            return None
        return {
            "location": cls._diagnostic_vector(transform.location),
            "rotation": {
                axis: float(getattr(transform.rotation, axis, 0.0))
                for axis in ("pitch", "yaw", "roll")
            },
        }

    @classmethod
    def _diagnostic_actor_snapshot(cls, actor):
        snapshot = {
            "actor_id": int(getattr(actor, "id", -1)),
            "type_id": str(getattr(actor, "type_id", "")),
        }
        try:
            snapshot["role_name"] = actor.attributes.get("role_name", "")
        except Exception:
            snapshot["role_name"] = ""
        for name, getter in (
            ("transform", "get_transform"),
            ("velocity", "get_velocity"),
            ("acceleration", "get_acceleration"),
            ("angular_velocity", "get_angular_velocity"),
        ):
            try:
                value = getattr(actor, getter)()
            except Exception:
                value = None
            snapshot[name] = (
                cls._diagnostic_transform(value)
                if name == "transform"
                else cls._diagnostic_vector(value)
            )
        try:
            control = actor.get_control()
            snapshot["control"] = {
                name: getattr(control, name, None)
                for name in (
                    "throttle", "steer", "brake", "hand_brake", "reverse", "gear"
                )
            }
        except Exception:
            snapshot["control"] = None
        return snapshot

    def _record_initialization_diagnostic(
        self,
        event_type,
        actor,
        veh_id,
        expected_transform=None,
        expected_speed=None,
        reason=None,
        attempt=None,
        extra=None,
    ):
        if not getattr(self, "initialization_diagnostics_enabled", False):
            return
        frame = self._current_carla_frame()
        payload = {
            "event": event_type,
            "wall_time": time.time(),
            "carla_frame": frame,
            "vehicle_id": veh_id,
            "attempt": attempt,
            "reason": reason,
            "expected_speed": self._as_finite_float(expected_speed),
            "expected_transform": self._diagnostic_transform(expected_transform),
            "actual": self._diagnostic_actor_snapshot(actor),
        }
        if extra:
            payload["extra"] = extra
        if event_type == "failure":
            category = str(reason or "unknown").split("=", 1)[0].split(":", 1)[0]
            counts = getattr(self, "_initialization_failure_counts", None)
            if counts is not None:
                counts[category] = int(counts.get(category, 0)) + 1
        self._append_diagnostic_jsonl(self.initialization_log_path, payload)

    def _ensure_collision_sensor(self, actor, veh_id):
        if not getattr(self, "collision_sensor_enabled", False) or actor is None:
            return None
        sensors = getattr(self, "_collision_sensors", None)
        if sensors is None:
            return None
        existing = sensors.get(veh_id)
        try:
            if existing is not None and existing.is_alive:
                return existing
        except Exception:
            pass
        sensors.pop(veh_id, None)
        try:
            blueprint = self.world.get_blueprint_library().find("sensor.other.collision")
            sensor = self.world.spawn_actor(
                blueprint,
                carla.Transform(),
                attach_to=actor,
            )
            sensor.listen(
                lambda event, vehicle_id=veh_id, vehicle_actor=actor: self._on_collision_event(
                    vehicle_id, vehicle_actor, event
                )
            )
        except Exception as exc:
            print(
                f"Warning: failed to attach CARLA collision sensor to {veh_id!r}: {exc}",
                flush=True,
            )
            return None
        sensors[veh_id] = sensor
        return sensor

    def _on_collision_event(self, veh_id, actor, event):
        frame = int(getattr(event, "frame", self._current_carla_frame() or -1))
        other = getattr(event, "other_actor", None)
        actor_id = int(getattr(actor, "id", -1))
        other_id = int(getattr(other, "id", -1))
        pair_ids = tuple(sorted((actor_id, other_id)))
        pair_key = f"{pair_ids[0]}:{pair_ids[1]}"
        frame_pair = (frame, pair_ids)
        lock = getattr(self, "_diagnostic_lock", None) or threading.Lock()
        with lock:
            self._collision_raw_event_count += 1
            duplicate_frame_pair = frame_pair in self._collision_seen_frame_pairs
            if not duplicate_frame_pair:
                self._collision_seen_frame_pairs.add(frame_pair)
                self._collision_unique_frame_count += 1
            previous_frame = self._collision_last_pair_frame.get(pair_ids)
            is_new_episode = (
                not duplicate_frame_pair
                and (
                    previous_frame is None
                    or frame - previous_frame > self.collision_episode_gap_frames
                )
            )
            if not duplicate_frame_pair:
                self._collision_last_pair_frame[pair_ids] = frame
            if is_new_episode:
                self._collision_episode_count += 1
                self._collision_episode_counts_by_pair[pair_key] = (
                    int(self._collision_episode_counts_by_pair.get(pair_key, 0)) + 1
                )
            episode_count = self._collision_episode_count
        payload = {
            "event": "carla_collision",
            "wall_time": time.time(),
            "carla_frame": frame,
            "carla_timestamp": self._as_finite_float(getattr(event, "timestamp", None)),
            "sensor_vehicle_id": veh_id,
            "pair_actor_ids": list(pair_ids),
            "pair_key": pair_key,
            "duplicate_frame_pair": duplicate_frame_pair,
            "new_episode": is_new_episode,
            "episode_count": episode_count,
            "normal_impulse": self._diagnostic_vector(
                getattr(event, "normal_impulse", None)
            ),
            "vehicle": self._diagnostic_actor_snapshot(actor),
            "other_actor": self._diagnostic_actor_snapshot(other) if other is not None else None,
        }
        self._append_diagnostic_jsonl(self.collision_log_path, payload)
        if not duplicate_frame_pair:
            other_role = ""
            try:
                other_role = other.attributes.get("role_name", "")
            except Exception:
                pass
            print(
                "CARLACollision "
                f"frame={frame} vehicle={veh_id!r} other={other_role or other_id!r} "
                f"new_episode={is_new_episode} pair={pair_key}",
                flush=True,
            )

    def _remove_collision_sensor(self, veh_id):
        sensor = (getattr(self, "_collision_sensors", None) or {}).pop(veh_id, None)
        if sensor is None:
            return
        try:
            sensor.stop()
        except Exception:
            pass
        try:
            sensor.destroy()
        except Exception:
            pass

    def _prune_collision_sensors(self, active_vehicle_ids):
        active = set(active_vehicle_ids)
        for veh_id in list(getattr(self, "_collision_sensors", {}) or {}):
            if veh_id not in active:
                self._remove_collision_sensor(veh_id)

    def _write_collision_summary(self):
        if not getattr(self, "collision_sensor_enabled", False):
            return
        summary = {
            "raw_sensor_events": int(getattr(self, "_collision_raw_event_count", 0)),
            "unique_frame_pairs": int(getattr(self, "_collision_unique_frame_count", 0)),
            "contact_episodes": int(getattr(self, "_collision_episode_count", 0)),
            "episode_gap_frames": int(getattr(self, "collision_episode_gap_frames", 10)),
            "episodes_by_pair": dict(
                sorted((getattr(self, "_collision_episode_counts_by_pair", {}) or {}).items())
            ),
            "initialization_failures_by_reason": dict(
                sorted((getattr(self, "_initialization_failure_counts", {}) or {}).items())
            ),
        }
        path = getattr(self, "collision_summary_path", "")
        if path:
            with open(path, "w", encoding="utf-8") as output:
                json.dump(summary, output, indent=2, sort_keys=True)
                output.write("\n")
        print("CARLA collision summary " + json.dumps(summary, sort_keys=True), flush=True)

    def _shutdown_collision_sensors(self):
        for veh_id in list(getattr(self, "_collision_sensors", {}) or {}):
            self._remove_collision_sensor(veh_id)
        self._write_collision_summary()

    def _current_carla_frame(self):
        try:
            return int(self.world.get_snapshot().frame)
        except Exception:
            return None

    @staticmethod
    def _zero_ackermann_actor_motion(actor):
        zero_velocity = carla.Vector3D(0.0, 0.0, 0.0)
        try:
            actor.set_target_velocity(zero_velocity)
        except Exception:
            pass
        try:
            actor.set_target_angular_velocity(zero_velocity)
        except Exception:
            pass

    @staticmethod
    def _rotation_error_degrees(actual, expected, attribute):
        actual_value = float(getattr(actual, attribute, 0.0))
        expected_value = float(getattr(expected, attribute, 0.0))
        return abs((actual_value - expected_value + 180.0) % 360.0 - 180.0)

    def _ackermann_spawn_stability_failure(
        self,
        actor,
        expected_transform,
        expected_speed,
    ):
        try:
            actual_transform = actor.get_transform()
            velocity = actor.get_velocity()
        except Exception as exc:
            return f"state_error:{type(exc).__name__}"

        z_error = abs(
            float(actual_transform.location.z) - float(expected_transform.location.z)
        )
        horizontal_velocity = horizontal_speed(velocity)
        vertical_speed = abs(float(getattr(velocity, "z", 0.0)))
        speed_error = abs(horizontal_velocity - max(0.0, float(expected_speed or 0.0)))
        pitch_error = self._rotation_error_degrees(
            actual_transform.rotation, expected_transform.rotation, "pitch"
        )
        roll_error = self._rotation_error_degrees(
            actual_transform.rotation, expected_transform.rotation, "roll"
        )

        if z_error > self.ACKERMANN_SPAWN_MAX_Z_ERROR:
            return f"z_error={z_error:.3f}m"
        if speed_error > self.ACKERMANN_SPAWN_MAX_SPEED_ERROR:
            return f"speed_error={speed_error:.3f}m/s"
        if vertical_speed > self.ACKERMANN_SPAWN_MAX_VERTICAL_SPEED:
            return f"vertical_speed={vertical_speed:.3f}m/s"
        if (
            pitch_error > self.ACKERMANN_SPAWN_MAX_TILT_ERROR
            or roll_error > self.ACKERMANN_SPAWN_MAX_TILT_ERROR
        ):
            return f"tilt_error=pitch:{pitch_error:.3f},roll:{roll_error:.3f}deg"
        return None

    def _restart_ackermann_actor_physics_initialization(
        self,
        actor,
        veh_id,
        initial_speed,
        ground_transform,
        reason,
    ):
        state = self._ackermann_actor_state.setdefault(veh_id, {})
        if self._register_ackermann_initialization_failure(
            actor,
            veh_id,
            initial_speed,
            ground_transform,
            reason,
        ):
            return False
        retry_count = int(state.get("physics_reinitialization_count", 0))
        self._set_actor_simulate_physics(actor, False)
        self._zero_ackermann_actor_motion(actor)
        state["physics_enabled"] = False
        state["physics_initialization_pending"] = True
        state["physics_initialization_transform"] = ground_transform
        state["physics_initialization_speed"] = initial_speed
        state["physics_overlap_deferred"] = True
        state["physics_ground_transform_reserved"] = False
        state["physics_stabilization_pending"] = False
        state["physics_stable_ticks"] = 0
        state.pop("physics_enabled_frame", None)
        state.pop("physics_last_stability_frame", None)
        state.pop("physics_ground_transform_applied_frame", None)
        state.pop("physics_ground_transform_wait_pending", None)
        elevated_transform = self._transform_with_z_offset(
            ground_transform, self.spawn_z_clearance
        )
        try:
            actor.set_transform(elevated_transform)
        except Exception:
            pass
        print(
            f"Warning: restarting CARLA physics initialization for {veh_id!r}: "
            f"{reason} (attempt={retry_count}).",
            flush=True,
        )
        return False

    def _register_ackermann_initialization_failure(
        self,
        actor,
        veh_id,
        initial_speed,
        ground_transform,
        reason,
    ):
        """Count one failed initialization attempt and abandon at the limit."""
        state = self._ackermann_actor_state.setdefault(veh_id, {})
        failure_count = int(state.get("physics_reinitialization_count", 0)) + 1
        state["physics_reinitialization_count"] = failure_count
        self._record_initialization_diagnostic(
            "failure",
            actor,
            veh_id,
            expected_transform=ground_transform,
            expected_speed=initial_speed,
            reason=reason,
            attempt=failure_count,
        )
        max_attempts = max(1, int(getattr(self, "spawn_max_attempts", 3)))
        if failure_count < max_attempts:
            return False

        state["physics_initialization_abandoned"] = True
        state["physics_initialization_pending"] = True
        state["physics_enabled"] = False
        self._set_actor_simulate_physics(actor, False)
        self._zero_ackermann_actor_motion(actor)
        self._record_initialization_diagnostic(
            "abandoned",
            actor,
            veh_id,
            expected_transform=ground_transform,
            expected_speed=initial_speed,
            reason=reason,
            attempt=failure_count,
        )
        self._mark_spawn_abandoned("vehicle", veh_id, failure_count, reason)
        self._remove_collision_sensor(veh_id)

        actor_index = getattr(self, "_vehicle_actor_index", None)
        if actor_index is not None and actor_index.get(veh_id) is actor:
            actor_index.pop(veh_id, None)
        pending_entries = getattr(self, "_pending_actor_index_entries", None)
        if pending_entries is not None:
            pending_entries.pop(veh_id, None)

        try:
            self.client.apply_batch_sync(
                [carla.command.DestroyActor(actor.id)],
                False,
            )
        except Exception:
            try:
                actor.destroy()
            except Exception:
                pass
        print(
            f"Warning: abandoning CARLA spawn for {veh_id!r} after "
            f"{failure_count} failed initialization attempts: {reason}. "
            "SUMO simulation will continue without this CARLA actor.",
            flush=True,
        )
        return True

    def _monitor_ackermann_actor_physics_stability(
        self,
        actor,
        veh_id,
        initial_speed,
        initial_transform,
    ):
        state = self._ackermann_actor_state.setdefault(veh_id, {})
        current_frame = self._current_carla_frame()
        enabled_frame = state.get("physics_enabled_frame")
        last_frame = state.get("physics_last_stability_frame")
        if current_frame is not None:
            if enabled_frame is not None and current_frame <= enabled_frame:
                return False
            if last_frame is not None and current_frame <= last_frame:
                return False

        failure = self._ackermann_spawn_stability_failure(
            actor,
            initial_transform,
            initial_speed,
        )
        if failure is not None:
            return self._restart_ackermann_actor_physics_initialization(
                actor,
                veh_id,
                initial_speed,
                initial_transform,
                failure,
            )

        state["physics_last_stability_frame"] = current_frame
        stable_ticks = int(state.get("physics_stable_ticks", 0)) + 1
        state["physics_stable_ticks"] = stable_ticks
        if stable_ticks < self.ACKERMANN_SPAWN_STABILITY_TICKS:
            return False

        state["physics_stabilization_pending"] = False
        state["physics_initialization_pending"] = False
        state.pop("physics_ground_transform_reserved", None)
        state.pop("physics_initialization_transform", None)
        state.pop("physics_initialization_speed", None)
        self._record_initialization_diagnostic(
            "stable",
            actor,
            veh_id,
            expected_transform=initial_transform,
            expected_speed=initial_speed,
            attempt=int(state.get("physics_reinitialization_count", 0)) + 1,
            extra={"stable_ticks": stable_ticks},
        )
        print(
            f"CARLA physics initialization stable for {veh_id!r} after "
            f"{stable_ticks} tick(s).",
            flush=True,
        )
        return True

    def _initialize_ackermann_actor_velocity(self, actor, veh_id, speed, transform):
        initial_speed = self._as_finite_float(speed)
        if initial_speed is None:
            return False
        initial_speed = max(0.0, initial_speed)
        yaw = math.radians(transform.rotation.yaw)
        velocity = carla.Vector3D(
            initial_speed * math.cos(yaw),
            initial_speed * math.sin(yaw),
            0.0,
        )
        try:
            actor.set_target_velocity(velocity)
        except Exception as exc:
            print(
                f"Warning: failed to initialize CARLA velocity for {veh_id!r}: {exc}",
                flush=True,
            )
            return False

        print(
            f"CARLA initial velocity applied to {veh_id!r}: {initial_speed:.3f}m/s.",
            flush=True,
        )
        return True

    def _initialize_ackermann_actor_geometry(self, actor, veh_id, state):
        if state.get("geometry_from_physics"):
            return
        # Immediately after spawning, CARLA 0.9.16 may temporarily report the
        # actor transform as (0, 0) while wheel positions already contain
        # world-space centimetres. Retry on subsequent feedback frames.
        geometry_attempts = int(state.get("geometry_attempts", 0))
        if geometry_attempts >= 10:
            return
        state["geometry_attempts"] = geometry_attempts + 1
        state["geometry_attempted"] = True
        fallback_tuning = getattr(self, "ackermann_tuning", AckermannTuning())
        fallback_wheel_base = float(fallback_tuning.wheel_base)
        state["wheel_base_m"] = fallback_wheel_base
        state["rear_axle_local_x_m"] = -0.5 * fallback_wheel_base

        try:
            bounding_box = actor.bounding_box
            front_bumper_local_x = float(
                bounding_box.location.x + bounding_box.extent.x
            )
            if 0.1 <= front_bumper_local_x <= 10.0:
                state["front_bumper_local_x_m"] = front_bumper_local_x
                state["front_bumper_from_bounding_box"] = True
        except Exception:
            pass

        try:
            physics_control = actor.get_physics_control()
            wheels = list(getattr(physics_control, "wheels", ()) or ())
            wheel_positions = [
                (float(wheel.position.x), float(wheel.position.y))
                for wheel in wheels
                if getattr(wheel, "position", None) is not None
            ]
        except Exception:
            return
        if len(wheel_positions) < 2:
            return

        # CARLA 0.9.16 can expose wheel positions as world coordinates in
        # centimetres, while test doubles and other releases use actor-local
        # coordinates. Validate both interpretations and prefer the geometry
        # whose axle midpoint is closest to the actor origin.
        scale = (
            0.01
            if max(abs(value) for point in wheel_positions for value in point) > 20.0
            else 1.0
        )
        scaled_positions = [(x * scale, y * scale) for x, y in wheel_positions]

        def resolve_geometry(local_x_values):
            local_x_values = sorted(local_x_values)
            split = max(1, len(local_x_values) // 2)
            rear_axle_x = sum(local_x_values[:split]) / split
            front_values = local_x_values[split:]
            if not front_values:
                return None
            front_axle_x = sum(front_values) / len(front_values)
            wheel_base = front_axle_x - rear_axle_x
            if not (0.5 <= wheel_base <= 6.0 and -5.0 <= rear_axle_x <= 2.0):
                return None
            return rear_axle_x, front_axle_x, wheel_base

        candidates = []
        direct_geometry = resolve_geometry([x for x, _ in scaled_positions])
        if direct_geometry is not None:
            candidates.append(direct_geometry)

        try:
            transform = actor.get_transform()
            yaw = math.radians(float(transform.rotation.yaw))
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            world_local_x = [
                cos_yaw * (x - float(transform.location.x))
                + sin_yaw * (y - float(transform.location.y))
                for x, y in scaled_positions
            ]
            world_geometry = resolve_geometry(world_local_x)
            if world_geometry is not None:
                candidates.append(world_geometry)
        except Exception:
            pass

        if not candidates:
            return
        rear_axle_x, front_axle_x, wheel_base = min(
            candidates,
            key=lambda geometry: abs(0.5 * (geometry[0] + geometry[1])),
        )
        state["wheel_base_m"] = wheel_base
        state["rear_axle_local_x_m"] = rear_axle_x
        state["geometry_from_physics"] = True

    def _ensure_ackermann_actor_physics(
        self,
        actor,
        veh_id,
        initial_speed=None,
        initial_transform=None,
    ):
        state = self._ackermann_actor_state.setdefault(veh_id, {})
        if state.get("physics_stabilization_pending"):
            expected_transform = initial_transform or state.get(
                "physics_initialization_transform"
            )
            expected_speed = (
                initial_speed
                if initial_speed is not None
                else state.get("physics_initialization_speed", 0.0)
            )
            if expected_transform is None:
                return False
            return self._monitor_ackermann_actor_physics_stability(
                actor,
                veh_id,
                expected_speed,
                expected_transform,
            )

        if not state.get("physics_enabled"):
            if state.get("physics_overlap_deferred"):
                if initial_transform is None:
                    return False
                state["physics_initialization_transform"] = initial_transform
                state["physics_initialization_speed"] = initial_speed
                state["physics_initialization_pending"] = True
                is_clear, blocking_actor_id = self._ackermann_spawn_transform_is_clear(
                    actor, veh_id, initial_transform
                )
                if not is_clear:
                    reason = f"overlap={blocking_actor_id}"
                    if self._register_ackermann_initialization_failure(
                        actor,
                        veh_id,
                        initial_speed,
                        initial_transform,
                        reason,
                    ):
                        return False
                    state["physics_overlap_blocking_actor"] = blocking_actor_id
                    actor.set_transform(
                        self._transform_with_z_offset(
                            initial_transform, self.spawn_z_clearance
                        )
                    )
                    return False
                state["physics_overlap_deferred"] = False
                state.pop("physics_overlap_blocking_actor", None)
                state["physics_ground_transform_reserved"] = True
                actor.set_transform(initial_transform)
                state["physics_ground_transform_applied_frame"] = (
                    self._current_carla_frame()
                )
                state["physics_ground_transform_wait_pending"] = True
                return False

            pending_transform = state.get("physics_initialization_transform")
            if pending_transform is None and initial_transform is not None:
                pending_transform = initial_transform
                state["physics_initialization_transform"] = initial_transform
                state["physics_initialization_speed"] = initial_speed
                state["physics_initialization_pending"] = True

            if pending_transform is not None:
                if state.get("physics_ground_transform_wait_pending"):
                    applied_frame = state.get("physics_ground_transform_applied_frame")
                    current_frame = self._current_carla_frame()
                    if (
                        applied_frame is not None
                        and current_frame is not None
                        and current_frame <= applied_frame
                    ):
                        return False
                    if applied_frame is None or current_frame is None:
                        state["physics_ground_transform_wait_pending"] = False
                        return False
                    state["physics_ground_transform_wait_pending"] = False
                try:
                    actual_transform = actor.get_transform()
                    dx = actual_transform.location.x - pending_transform.location.x
                    dy = actual_transform.location.y - pending_transform.location.y
                    dz = actual_transform.location.z - pending_transform.location.z
                    yaw_error = self._rotation_error_degrees(
                        actual_transform.rotation, pending_transform.rotation, "yaw"
                    )
                    transform_ready = (
                        math.hypot(dx, dy) <= 1.0
                        and abs(dz) <= self.ACKERMANN_SPAWN_MAX_Z_ERROR
                        and yaw_error <= 5.0
                    )
                except Exception:
                    transform_ready = False
                    actual_transform = None
                if not transform_ready:
                    return self._restart_ackermann_actor_physics_initialization(
                        actor,
                        veh_id,
                        initial_speed,
                        pending_transform,
                        "ground_transform_not_ready",
                    )
                is_clear, blocking_actor_id = self._ackermann_spawn_transform_is_clear(
                    actor, veh_id, pending_transform
                )
                if not is_clear:
                    reason = f"overlap={blocking_actor_id}"
                    if self._register_ackermann_initialization_failure(
                        actor,
                        veh_id,
                        initial_speed,
                        pending_transform,
                        reason,
                    ):
                        return False
                    state["physics_overlap_deferred"] = True
                    state["physics_overlap_blocking_actor"] = blocking_actor_id
                    state["physics_ground_transform_reserved"] = False
                    actor.set_transform(
                        self._transform_with_z_offset(
                            pending_transform, self.spawn_z_clearance
                        )
                    )
                    return False
            else:
                actual_transform = initial_transform

            self._zero_ackermann_actor_motion(actor)
            self._set_actor_simulate_physics(actor, True)
            state["physics_enabled"] = True
            self._record_initialization_diagnostic(
                "physics_enabled",
                actor,
                veh_id,
                expected_transform=pending_transform or initial_transform,
                expected_speed=initial_speed,
                attempt=int(state.get("physics_reinitialization_count", 0)) + 1,
            )
            if initial_speed is not None and initial_transform is not None:
                state["initial_velocity_applied"] = self._initialize_ackermann_actor_velocity(
                    actor,
                    veh_id,
                    initial_speed,
                    actual_transform or initial_transform,
                )
                state["physics_initialization_speed"] = initial_speed
            if pending_transform is not None:
                state["physics_stabilization_pending"] = True
                state["physics_stable_ticks"] = 0
                state["physics_enabled_frame"] = self._current_carla_frame()
                state.pop("physics_last_stability_frame", None)
            else:
                state["physics_initialization_pending"] = False

        self._initialize_ackermann_actor_geometry(actor, veh_id, state)
        if not state.get("controller_settings_attempted"):
            state["controller_settings_attempted"] = True
            tuning = self.ackermann_controller_tuning
            try:
                settings = carla.AckermannControllerSettings(
                    tuning.speed_kp,
                    tuning.speed_ki,
                    tuning.speed_kd,
                    tuning.accel_kp,
                    tuning.accel_ki,
                    tuning.accel_kd,
                )
                actor.apply_ackermann_controller_settings(settings)
            except Exception as exc:
                state["controller_settings_applied"] = False
                print(
                    f"Warning: failed to apply CARLA Ackermann controller settings "
                    f"for {veh_id!r}: {exc}",
                    flush=True,
                )
            else:
                state["controller_settings_applied"] = True
                print(
                    f"CARLA Ackermann controller settings applied to {veh_id!r}: "
                    f"speed=({tuning.speed_kp:.4g},{tuning.speed_ki:.4g},{tuning.speed_kd:.4g}) "
                    f"accel=({tuning.accel_kp:.4g},{tuning.accel_ki:.4g},{tuning.accel_kd:.4g})",
                    flush=True,
                )

        return not state.get("physics_initialization_pending", False)

    def _ensure_actor_teleport_mode(self, actor, veh_id):
        state = self._ackermann_actor_state.setdefault(veh_id, {})
        if state.get("physics_enabled") is False:
            return
        self._set_actor_simulate_physics(actor, False)
        state.clear()
        state["physics_enabled"] = False

    def _warn_ackermann_position_error(self, veh_id, position_error):
        if self.ackermann_warn_error_m <= 0.0 or position_error < self.ackermann_warn_error_m:
            return
        state = self._ackermann_actor_state.setdefault(veh_id, {})
        now = time.monotonic()
        last_warning_time = state.get("last_warning_time", 0.0)
        if now - last_warning_time < self.ackermann_warning_interval:
            return
        state["last_warning_time"] = now
        print(
            f"Warning: Ackermann vehicle {veh_id!r} is {position_error:.2f}m "
            "from its SUMO desired position.",
            flush=True,
        )

    def _is_ackermann_feedback_apply_actor(self, veh_id):
        return self.ackermann_feedback_apply_enabled and (
            self._is_ackermann_feedback_actor(veh_id)
        )

    def _is_ackermann_feedback_healthy(self, veh_id, feedback, observed_frame):
        state = self._ackermann_actor_state.setdefault(veh_id, {})
        if feedback.get("feedback_status") != "queued":
            failures = self._register_physics_feedback_failure(
                veh_id, feedback.get("feedback_reason", "feedback_not_queued")
            )
            state["feedback_ack_failures"] = failures
            state["feedback_frame_lag"] = None
            state["feedback_frame_mismatch"] = True
            return False

        expected_frame = self._as_finite_float(feedback.get("source_carla_frame"))
        observed_frame = self._as_finite_float(observed_frame)
        frame_matches = (
            expected_frame is not None
            and observed_frame is not None
            and expected_frame == observed_frame
        )
        if frame_matches:
            self._clear_physics_feedback_failure(veh_id)
            state["feedback_ack_failures"] = 0
        else:
            frame_lag = (
                abs(expected_frame - observed_frame)
                if expected_frame is not None and observed_frame is not None
                else None
            )
            reason = (
                "feedback_frame_lag_exceeded"
                if frame_lag is not None
                and frame_lag > self.ackermann_feedback_ack_max_frame_lag
                else "feedback_frame_mismatch"
            )
            failures = self._register_physics_feedback_failure(
                veh_id, reason
            )
            state["feedback_ack_failures"] = failures
        state["feedback_frame_lag"] = (
            expected_frame - observed_frame
            if expected_frame is not None and observed_frame is not None
            else None
        )
        state["feedback_frame_mismatch"] = not frame_matches
        return frame_matches

    def _resolve_ackermann_desired_speed(self, veh_id, veh_info):
        speed_key = (
            "sumo_desired_speed"
            if self._is_ackermann_feedback_apply_actor(veh_id)
            else "speed"
        )
        desired_speed = self._as_finite_float(veh_info.get(speed_key))
        if desired_speed is None and speed_key != "speed":
            desired_speed = self._as_finite_float(veh_info.get("speed"))
        return max(0.0, desired_speed or 0.0)

    def _resolve_ackermann_max_decel(self, veh_info):
        emergency_decel = self._as_finite_float(veh_info.get("sumo_emergency_decel"))
        if emergency_decel is not None and emergency_decel > 0.0:
            return emergency_decel
        return self.ackermann_tuning.max_decel

    def _resolve_ackermann_longitudinal_target(
        self,
        veh_id,
        veh_info,
        current_speed,
        longitudinal_error=0.0,
    ):
        state = self._ackermann_actor_state.setdefault(veh_id, {})
        max_decel = self._resolve_ackermann_max_decel(veh_info)
        state["sumo_emergency_decel"] = max_decel
        requested_acceleration = self._as_finite_float(veh_info.get("acceleration"))
        state["sumo_requested_acceleration"] = requested_acceleration
        if not self._is_ackermann_feedback_apply_actor(veh_id):
            desired_speed = self._resolve_ackermann_desired_speed(veh_id, veh_info)
            state["applied_desired_acceleration"] = None
            state.pop("restart_target_speed", None)
            state["restart_active"] = False
            state["sumo_action_invalid"] = False
            return desired_speed, None

        sumo_next_speed = self._as_finite_float(veh_info.get("sumo_desired_speed"))
        observed_speed = self._as_finite_float(veh_info.get("feedback_observed_speed"))
        longitudinal_error = self._as_finite_float(longitudinal_error) or 0.0
        state["sumo_desired_speed"] = sumo_next_speed
        state["feedback_observed_speed"] = observed_speed
        state["longitudinal_position_error"] = longitudinal_error
        invalid_action = (
            sumo_next_speed is None
            or sumo_next_speed < 0.0
            or requested_acceleration is None
            or self.step_length <= 0.0
        )
        state["sumo_action_invalid"] = invalid_action
        if invalid_action:
            state.pop("restart_target_speed", None)
            state["restart_active"] = False
            state["applied_desired_acceleration"] = None
            state["longitudinal_velocity_error"] = None
            return 0.0, -max_decel

        desired_acceleration = min(
            self.ackermann_tuning.max_accel,
            max(-max_decel, requested_acceleration),
        )
        state["longitudinal_velocity_error"] = sumo_next_speed - current_speed
        state["applied_desired_acceleration"] = desired_acceleration
        speed_target = sumo_next_speed
        restart_target = self._as_finite_float(state.get("restart_target_speed"))
        restart_active = (
            bool(state.get("restart_active")) and restart_target is not None
        )
        restart_cancelled = (
            requested_acceleration <= 0.0
            or sumo_next_speed <= self.ackermann_tuning.restart_speed_epsilon
            or current_speed >= self.ackermann_tuning.restart_release_speed
        )
        if restart_cancelled:
            state.pop("restart_target_speed", None)
            state["restart_active"] = False
        else:
            restart_requested = (
                current_speed <= self.ackermann_tuning.restart_enter_speed
                and sumo_next_speed
                > current_speed + self.ackermann_tuning.restart_speed_epsilon
            )
            if restart_requested:
                if restart_target is None:
                    restart_target = sumo_next_speed
                else:
                    restart_target += requested_acceleration * self.step_length
                restart_target = min(
                    self.ackermann_tuning.restart_max_target_speed,
                    max(0.0, restart_target),
                )
                restart_active = True
                state["restart_target_speed"] = restart_target
                state["restart_active"] = True

            if restart_active and restart_target is not None:
                speed_target = max(speed_target, restart_target)

        return speed_target, desired_acceleration

    def _current_direct_brake_steer(self, veh_id, vehicle):
        try:
            applied_steer = self._as_finite_float(vehicle.get_control().steer)
        except Exception:
            applied_steer = None
        if applied_steer is not None:
            return min(1.0, max(-1.0, applied_steer))

        state = self._ackermann_actor_state.setdefault(veh_id, {})
        ackermann_steer = self._as_finite_float(state.get("steer")) or 0.0
        max_steer = self.ackermann_tuning.max_steer_rad
        if max_steer <= 0.0:
            return 0.0
        return min(1.0, max(-1.0, ackermann_steer / max_steer))

    def _make_direct_brake_control(self, veh_id, vehicle, brake=1.0, ackermann_steer=None):
        direct_steer = self._as_finite_float(ackermann_steer)
        if direct_steer is None:
            direct_steer = self._current_direct_brake_steer(veh_id, vehicle)
        else:
            max_steer = self.ackermann_tuning.max_steer_rad
            direct_steer = (
                0.0 if max_steer <= 0.0 else min(1.0, max(-1.0, direct_steer / max_steer))
            )
        return carla.VehicleControl(
            throttle=0.0,
            steer=direct_steer,
            brake=min(1.0, max(0.0, float(brake))),
            hand_brake=False,
            reverse=False,
        )

    def _mark_ackermann_authoritative_fail_closed(
        self, veh_id, veh_info, reason
    ):
        state = self._ackermann_actor_state.setdefault(veh_id, {})
        state["control_mode"] = "fail_closed_brake"
        state["sumo_action_invalid"] = True
        state["sumo_emergency_decel"] = self._resolve_ackermann_max_decel(
            veh_info
        )
        state["sumo_requested_acceleration"] = self._as_finite_float(
            veh_info.get("acceleration")
        )
        state["sumo_desired_speed"] = self._as_finite_float(
            veh_info.get("sumo_desired_speed")
        )
        state["feedback_observed_speed"] = self._as_finite_float(
            veh_info.get("feedback_observed_speed")
        )
        state["applied_desired_acceleration"] = None
        state["longitudinal_position_error"] = None
        state["longitudinal_velocity_error"] = None
        state.pop("restart_target_speed", None)
        state["restart_active"] = False
        state["emergency_brake_command"] = 1.0
        state["fail_closed_reason"] = reason
        if getattr(self, "_pending_authoritative_action_error", None) is None:
            self._pending_authoritative_action_error = {
                "actor_id": veh_id,
                "reason": reason,
            }

    def _raise_pending_authoritative_action_error(self):
        pending = getattr(self, "_pending_authoritative_action_error", None)
        if pending is None:
            return
        self._pending_authoritative_action_error = None
        actor_id = pending["actor_id"]
        reason = pending["reason"]
        self._apply_ackermann_fail_closed_brake(reason)
        raise RuntimeError(
            f"invalid authoritative SUMO action actor={actor_id} reason={reason}"
        )

    def _should_record_ackermann_control_trace(self, veh_id):
        if not getattr(self, "ackermann_control_log_records", False):
            return False
        control_log_actor_ids = getattr(
            self, "ackermann_control_log_actor_ids", set()
        )
        return (
            not control_log_actor_ids
            or "*" in control_log_actor_ids
            or veh_id in control_log_actor_ids
        )

    def _record_fail_closed_ackermann_control_trace(
        self,
        *,
        veh_id,
        veh_info,
        vehicle,
        control,
        reason,
        current_transform=None,
        current_velocity=None,
        position_error=None,
    ):
        if not self._should_record_ackermann_control_trace(veh_id):
            return
        try:
            if current_transform is None:
                current_transform = vehicle.get_transform()
            if current_velocity is None:
                current_velocity = vehicle.get_velocity()
            current_speed = horizontal_speed(current_velocity)
            self._record_ackermann_control_trace(
                veh_id=veh_id,
                veh_info=veh_info,
                vehicle=vehicle,
                current_transform=current_transform,
                current_speed=current_speed,
                target_speed=0.0,
                target_acceleration=-self._resolve_ackermann_max_decel(veh_info),
                position_error=position_error,
                feedback_unhealthy=False,
                target_behind=False,
                control_mode="fail_closed_brake",
                commanded_throttle=control.throttle,
                commanded_brake=control.brake,
                commanded_steer=control.steer,
                fail_closed_reason=reason,
            )
        except Exception as trace_error:
            # Diagnostics must never prevent the already-selected full brake.
            try:
                try:
                    carla_frame = int(self.world.get_snapshot().frame)
                except Exception:
                    carla_frame = None
                terasim_states = getattr(self, "_inproc_prev_state", {})
                simulation_time = (
                    terasim_states.get("simulation_time")
                    if isinstance(terasim_states, dict)
                    else None
                )
                state = self._ackermann_actor_state.setdefault(veh_id, {})
                trace = {
                    "actor_id": veh_id,
                    "carla_frame": carla_frame,
                    "simulation_time": simulation_time,
                    "action_source_carla_frame": veh_info.get(
                        "feedback_source_carla_frame"
                    ),
                    "phase_a_requested_lane_id": veh_info.get(
                        "feedback_requested_lane_id"
                    ),
                    "phase_a_observed_lane_id": veh_info.get(
                        "feedback_observed_lane_id"
                    ),
                    "phase_b_live_lane_id": veh_info.get("lane_id"),
                    "sumo_route": veh_info.get("sumo_route", ()),
                    "external_state_maneuver_current_lane_id": veh_info.get(
                        "lane_id"
                    ),
                    "external_state_maneuver_source_lane_id": veh_info.get(
                        "external_state_maneuver_source_lane_id", ""
                    ),
                    "external_state_maneuver_target_lane_id": veh_info.get(
                        "external_state_maneuver_target_lane_id", ""
                    ),
                    "lookahead_action_error": veh_info.get(
                        "lookahead_action_error", ""
                    ),
                    "fail_closed_reason": reason,
                    "sumo_desired_speed": self._as_finite_float(
                        veh_info.get("sumo_desired_speed")
                    ),
                    "sumo_requested_acceleration": state.get(
                        "sumo_requested_acceleration"
                    ),
                    "sumo_emergency_decel": state.get("sumo_emergency_decel"),
                    "restart_active": bool(state.get("restart_active")),
                    "restart_target_speed": state.get("restart_target_speed"),
                    "longitudinal_position_error": state.get(
                        "longitudinal_position_error"
                    ),
                    "longitudinal_velocity_error": state.get(
                        "longitudinal_velocity_error"
                    ),
                    "control_mode": "fail_closed_brake",
                    "commanded_throttle": self._as_finite_float(
                        getattr(control, "throttle", None)
                    ),
                    "commanded_brake": self._as_finite_float(
                        getattr(control, "brake", None)
                    ),
                    "commanded_steer": self._as_finite_float(
                        getattr(control, "steer", None)
                    ),
                    "trace_degraded": True,
                    "trace_error": (
                        f"{type(trace_error).__name__}: {trace_error}"
                    ),
                }
                print(
                    "AckermannControlTrace "
                    + json.dumps(trace, sort_keys=True, default=str),
                    flush=True,
                )
            except Exception as fallback_error:
                print(
                    "Warning: fail-closed AckermannControlTrace failed for "
                    f"{veh_id}: {fallback_error}",
                    flush=True,
                )

    def _update_ackermann_emergency_brake(
        self, veh_id, vehicle, current_speed, ackermann_steer=None
    ):
        state = self._ackermann_actor_state.setdefault(veh_id, {})
        tuning = getattr(
            self,
            "ackermann_emergency_brake_tuning",
            AckermannEmergencyBrakeTuning(),
        )
        requested_acceleration = self._as_finite_float(
            state.get("sumo_requested_acceleration")
        )
        authoritative = bool(
            getattr(self, "ackermann_feedback_apply_enabled", False)
        ) and self._is_ackermann_feedback_apply_actor(veh_id)
        active = bool(state.get("emergency_brake_active", False))
        release_ticks = int(state.get("emergency_brake_release_ticks", 0))

        if not tuning.enabled:
            active = False
            release_ticks = 0
        elif authoritative and (
            requested_acceleration is None or requested_acceleration >= 0.0
        ):
            active = False
            release_ticks = 0
        elif active:
            if current_speed <= tuning.stop_speed:
                active = False
                release_ticks = 0
            elif (
                requested_acceleration is not None
                and requested_acceleration >= -tuning.release_decel
            ):
                release_ticks += 1
                if release_ticks >= tuning.release_ticks:
                    active = False
                    release_ticks = 0
            else:
                release_ticks = 0
        elif (
            current_speed > tuning.stop_speed
            and requested_acceleration is not None
            and requested_acceleration <= -tuning.engage_decel
        ):
            active = True
            release_ticks = 0

        state["emergency_brake_active"] = active
        state["emergency_brake_release_ticks"] = release_ticks
        state["control_mode"] = "emergency_brake" if active else "ackermann"
        if not active:
            state["emergency_brake_command"] = 0.0
            return None

        max_decel = self._as_finite_float(state.get("sumo_emergency_decel"))
        if max_decel is None or max_decel <= 0.0:
            max_decel = self.ackermann_tuning.max_decel
        brake = compute_direct_brake_value(
            requested_acceleration or 0.0,
            max_decel,
            0.0 if authoritative else tuning.min_brake,
        )
        state["emergency_brake_command"] = brake
        if not authoritative:
            ackermann_steer = None
        return self._make_direct_brake_control(
            veh_id,
            vehicle,
            brake,
            ackermann_steer=ackermann_steer,
        )

    def _neutralize_ackermann_steer(self, veh_id):
        state = self._ackermann_actor_state.setdefault(veh_id, {})
        steer = self._as_finite_float(state.get("steer")) or 0.0
        max_delta = self.ackermann_tuning.max_steer_rate_rad_s * self.step_length
        if max_delta <= 0.0 or abs(steer) <= max_delta:
            return 0.0
        return steer - math.copysign(max_delta, steer)

    def _make_brake_ackermann_control(self, veh_id):
        state = self._ackermann_actor_state.setdefault(veh_id, {})
        if self._is_ackermann_feedback_apply_actor(veh_id):
            steer = self._neutralize_ackermann_steer(veh_id)
            state["steer"] = steer
        else:
            steer = self._as_finite_float(state.get("steer")) or 0.0
        max_decel = self._as_finite_float(state.get("sumo_emergency_decel"))
        if max_decel is None or max_decel <= 0.0:
            max_decel = self.ackermann_tuning.max_decel
        return carla.VehicleAckermannControl(
            steer=steer,
            speed=0.0,
            acceleration=max_decel,
            jerk=0.0,
        )

    def _apply_ackermann_fail_closed_brake(self, reason):
        actors = []
        for actor_id, actor in list((self._vehicle_actor_index or {}).items()):
            if actor is not None and self._is_ackermann_feedback_actor(actor_id):
                actors.append((actor_id, actor))
        for actor_id, actor in actors:
            try:
                actor.apply_control(
                    self._make_direct_brake_control(actor_id, actor, brake=1.0)
                )
            except Exception as exc:
                print(
                    f"Warning: fail-closed brake failed for {actor_id}: {exc}",
                    flush=True,
                )
        if actors:
            print(
                f"Ackermann fail-closed direct brake applied to "
                f"{len(actors)} actor(s): {reason}",
                flush=True,
            )
        return len(actors)

    def _build_ackermann_control(
        self,
        veh_id,
        veh_info,
        vehicle,
        sumo_location,
        sumo_angle,
        desired_transform,
    ):
        state = self._ackermann_actor_state.setdefault(veh_id, {})
        feedback_actor = self._is_ackermann_feedback_apply_actor(veh_id)
        state.pop("fail_closed_reason", None)
        if feedback_actor:
            state["last_action_source_carla_frame"] = veh_info.get(
                "feedback_source_carla_frame"
            )
            phase_b_sumo_angle = self._as_finite_float(sumo_angle)
            state["last_phase_b_target_sumo_angle"] = phase_b_sumo_angle
            state["last_phase_b_target_carla_yaw"] = (
                (phase_b_sumo_angle - 90.0) % 360.0
                if phase_b_sumo_angle is not None
                else None
            )
        action_pose = [
            self._as_finite_float(value)
            for value in (*sumo_location[:3], sumo_angle)
        ]
        if feedback_actor and any(value is None for value in action_pose):
            reason = "invalid_authoritative_action_pose"
            self._mark_ackermann_authoritative_fail_closed(veh_id, veh_info, reason)
            control = self._make_direct_brake_control(veh_id, vehicle, brake=1.0)
            self._record_fail_closed_ackermann_control_trace(
                veh_id=veh_id,
                veh_info=veh_info,
                vehicle=vehicle,
                control=control,
                reason=reason,
            )
            return control
        try:
            current_transform = vehicle.get_transform()
            current_velocity = vehicle.get_velocity()
        except Exception:
            state["control_mode"] = "ackermann"
            return self._make_brake_ackermann_control(veh_id)

        desired_location = desired_transform.location
        current_location = current_transform.location
        rear_axle_local_x = state.get(
            "rear_axle_local_x_m",
            -0.5 * self.ackermann_tuning.wheel_base,
        )
        current_rear_axle = self._transform_local_x_point(
            current_transform, rear_axle_local_x
        )
        desired_rear_axle = self._transform_local_x_point(
            desired_transform, rear_axle_local_x
        )
        position_error = math.hypot(
            current_rear_axle[0] - desired_rear_axle[0],
            current_rear_axle[1] - desired_rear_axle[1],
        )
        self._warn_ackermann_position_error(veh_id, position_error)

        if (
            self.ackermann_snap_error_m > 0.0
            and position_error >= self.ackermann_snap_error_m
            and not state.get("snap_used", False)
            and not self._is_ackermann_feedback_apply_actor(veh_id)
        ):
            try:
                vehicle.set_transform(desired_transform)
                current_transform = desired_transform
                current_location = desired_location
                state["steer"] = 0.0
                state["snap_used"] = True
            except Exception:
                pass

        try:
            lookahead_sumo_location = self._resolve_sumo_lookahead_location(
                veh_id, veh_info, sumo_location, sumo_angle
            )
            lookahead_location = self._sumo_point_to_carla_location(
                lookahead_sumo_location
            )
        except Exception as exc:
            if feedback_actor:
                reason = (
                    veh_info.get("lookahead_action_error")
                    or str(exc)
                    or "invalid_authoritative_lookahead"
                )
                self._mark_ackermann_authoritative_fail_closed(veh_id, veh_info, reason)
                control = self._make_direct_brake_control(
                    veh_id, vehicle, brake=1.0
                )
                self._record_fail_closed_ackermann_control_trace(
                    veh_id=veh_id,
                    veh_info=veh_info,
                    vehicle=vehicle,
                    control=control,
                    reason=reason,
                    current_transform=current_transform,
                    current_velocity=current_velocity,
                    position_error=position_error,
                )
                return control
            state["control_mode"] = "ackermann"
            return self._make_brake_ackermann_control(veh_id)

        current_speed = horizontal_speed(current_velocity)
        desired_heading = math.radians(desired_transform.rotation.yaw)
        longitudinal_error = self._as_finite_float(
            veh_info.get("feedback_longitudinal_error")
        )
        if longitudinal_error is None:
            longitudinal_error = self._phase_aligned_front_progress_error(
                current_transform,
                desired_transform,
                state.get(
                    "front_bumper_local_x_m",
                    0.5 * float(veh_info.get("length", 5.0)),
                ),
                current_velocity,
                desired_heading,
                self.step_length,
            )
        desired_speed, feedback_desired_acceleration = (
            self._resolve_ackermann_longitudinal_target(
                veh_id,
                veh_info,
                current_speed,
                longitudinal_error=longitudinal_error,
            )
        )
        values = compute_ackermann_control_values(
            current_x=current_location.x,
            current_y=current_location.y,
            yaw_degrees=current_transform.rotation.yaw,
            current_speed=current_speed,
            desired_x=desired_location.x,
            desired_y=desired_location.y,
            lookahead_x=lookahead_location.x,
            lookahead_y=lookahead_location.y,
            desired_speed=desired_speed,
            previous_steer=state.get("steer"),
            dt=self.step_length,
            tuning=self.ackermann_tuning,
            control_point_local_x=state.get(
                "rear_axle_local_x_m",
                -0.5 * self.ackermann_tuning.wheel_base,
            ),
            wheel_base=state.get("wheel_base_m", self.ackermann_tuning.wheel_base),
        )
        final_steer = values.steer
        final_speed = values.speed
        final_acceleration = (
            values.acceleration
            if feedback_desired_acceleration is None
            else feedback_desired_acceleration
        )
        feedback = self._ackermann_feedback_state.get(veh_id, {})
        feedback_unhealthy = (
            feedback_actor
            and not self._is_ackermann_feedback_healthy(
                veh_id, feedback, veh_info.get("feedback_source_carla_frame")
            )
        )
        target_behind = (
            feedback_actor and values.lookahead_local_x <= 0.0
        )
        invalid_action = feedback_actor and bool(state.get("sumo_action_invalid"))
        fail_closed = feedback_unhealthy or target_behind or invalid_action
        fail_closed_reason = ""
        if feedback_unhealthy:
            fail_closed_reason = "feedback_unhealthy"
        elif target_behind:
            fail_closed_reason = "target_behind"
        elif invalid_action:
            fail_closed_reason = "invalid_sumo_longitudinal_action"
        if fail_closed_reason:
            state["fail_closed_reason"] = fail_closed_reason
        if fail_closed:
            final_steer = self._neutralize_ackermann_steer(veh_id)
            final_speed = 0.0
            final_acceleration = -self._resolve_ackermann_max_decel(veh_info)

        if fail_closed:
            emergency_control = self._make_direct_brake_control(
                veh_id, vehicle, brake=1.0
            )
            state["control_mode"] = "fail_closed_brake"
            state["emergency_brake_command"] = 1.0
        else:
            emergency_control = self._update_ackermann_emergency_brake(
                veh_id,
                vehicle,
                current_speed,
                ackermann_steer=final_steer,
            )
        if emergency_control is None:
            control_mode = "ackermann"
            commanded_throttle = None
            commanded_brake = None
            commanded_steer = final_steer
        else:
            control_mode = state.get("control_mode", "emergency_brake")
            commanded_throttle = emergency_control.throttle
            commanded_brake = emergency_control.brake
            commanded_steer = emergency_control.steer

        self._record_ackermann_control_trace(
            veh_id=veh_id,
            veh_info=veh_info,
            vehicle=vehicle,
            current_transform=current_transform,
            current_speed=current_speed,
            target_speed=final_speed,
            target_acceleration=final_acceleration,
            position_error=position_error,
            feedback_unhealthy=bool(feedback_unhealthy),
            target_behind=bool(target_behind),
            control_values=values,
            control_mode=control_mode,
            commanded_throttle=commanded_throttle,
            commanded_brake=commanded_brake,
            commanded_steer=commanded_steer,
            fail_closed_reason=fail_closed_reason,
        )
        state["steer"] = final_steer
        state["last_position_error"] = position_error
        if emergency_control is not None:
            return emergency_control
        return carla.VehicleAckermannControl(
            steer=final_steer,
            speed=final_speed,
            # CARLA 0.9.16 takes Abs(acceleration) and uses it as the symmetric
            # acceleration/deceleration limit. Target speed determines direction.
            acceleration=abs(final_acceleration),
            jerk=values.jerk,
        )

    def _record_ackermann_control_trace(
        self,
        *,
        veh_id,
        veh_info,
        vehicle,
        current_transform,
        current_speed,
        target_speed,
        target_acceleration,
        position_error,
        feedback_unhealthy,
        target_behind,
        control_values=None,
        control_mode="ackermann",
        commanded_throttle=None,
        commanded_brake=None,
        commanded_steer=None,
        fail_closed_reason="",
    ):
        if not self._should_record_ackermann_control_trace(veh_id):
            return

        feedback = getattr(self, "_ackermann_feedback_state", {}).get(
            veh_id, {}
        )
        state = self._ackermann_actor_state.setdefault(veh_id, {})
        try:
            snapshot = self.world.get_snapshot()
            carla_frame = int(snapshot.frame)
        except Exception:
            carla_frame = None
        previous_speed = self._as_finite_float(state.get("trace_previous_speed"))
        previous_frame = state.get("trace_previous_frame")
        speed_delta_acceleration = None
        if (
            previous_speed is not None
            and carla_frame is not None
            and isinstance(previous_frame, int)
            and carla_frame > previous_frame
        ):
            dt = (carla_frame - previous_frame) * self.step_length
            if dt > 0.0:
                speed_delta_acceleration = (current_speed - previous_speed) / dt
        state["trace_previous_speed"] = current_speed
        state["trace_previous_frame"] = carla_frame

        longitudinal_acceleration = None
        try:
            acceleration = vehicle.get_acceleration()
            yaw = math.radians(current_transform.rotation.yaw)
            longitudinal_acceleration = (
                float(acceleration.x) * math.cos(yaw)
                + float(acceleration.y) * math.sin(yaw)
            )
        except Exception:
            pass

        applied_throttle = None
        applied_brake = None
        applied_steer = None
        try:
            applied_control = vehicle.get_control()
            applied_throttle = self._as_finite_float(
                getattr(applied_control, "throttle", None)
            )
            applied_brake = self._as_finite_float(getattr(applied_control, "brake", None))
            applied_steer = self._as_finite_float(getattr(applied_control, "steer", None))
        except Exception:
            pass

        trace = {
            "actor_id": veh_id,
            "carla_frame": carla_frame,
            "simulation_time": (self._inproc_prev_state or {}).get(
                "simulation_time"
            ),
            "action_source_carla_frame": veh_info.get(
                "feedback_source_carla_frame"
            ),
            "phase_a_sumo_time": self._as_finite_float(
                veh_info.get("feedback_phase_a_sumo_time")
            ),
            "phase_a_observed_x": self._as_finite_float(
                veh_info.get("feedback_observed_x")
            ),
            "phase_a_observed_y": self._as_finite_float(
                veh_info.get("feedback_observed_y")
            ),
            "phase_a_requested_sumo_angle": self._as_finite_float(
                feedback.get("feedback_sumo_angle")
            ),
            "phase_a_observed_sumo_angle": self._as_finite_float(
                veh_info.get("feedback_observed_sumo_angle")
            ),
            "phase_b_target_sumo_angle": self._as_finite_float(
                veh_info.get("sumo_angle")
            ),
            "phase_b_target_carla_yaw": (
                (float(veh_info["sumo_angle"]) - 90.0) % 360.0
                if self._as_finite_float(veh_info.get("sumo_angle")) is not None
                else None
            ),
            "sumo_x": self._as_finite_float(veh_info.get("x")),
            "sumo_y": self._as_finite_float(veh_info.get("y")),
            "sumo_lane_id": veh_info.get("lane_id"),
            "phase_b_live_lane_id": veh_info.get("lane_id"),
            "sumo_route": veh_info.get("sumo_route", ()),
            "sumo_lane_position": self._as_finite_float(
                veh_info.get("lane_position")
            ),
            "sumo_lateral_offset": self._as_finite_float(
                veh_info.get("lateral_offset")
            ),
            "sumo_lookahead_x": self._as_finite_float(veh_info.get("lookahead_x")),
            "sumo_lookahead_y": self._as_finite_float(veh_info.get("lookahead_y")),
            "sumo_route_lookahead_x": self._as_finite_float(veh_info.get("lookahead_route_x")),
            "sumo_route_lookahead_y": self._as_finite_float(veh_info.get("lookahead_route_y")),
            "lookahead_action_mode": veh_info.get("lookahead_action_mode"),
            "lookahead_action_valid": bool(veh_info.get("lookahead_action_valid", True)),
            "lookahead_action_error": veh_info.get("lookahead_action_error", ""),
            "lookahead_action_warning": veh_info.get(
                "lookahead_action_warning", ""
            ),
            "lookahead_lateral_direction_source": veh_info.get(
                "lookahead_lateral_direction_source", "inactive"
            ),
            "lookahead_lateral_direction_unresolved_count": int(
                veh_info.get("lookahead_lateral_direction_unresolved_count", 0)
                or 0
            ),
            "lookahead_lane_change_intent_conflict": bool(
                veh_info.get("lookahead_lane_change_intent_conflict", False)
            ),
            "external_state_maneuver_current_lane_id": veh_info.get("lane_id"),
            "external_state_maneuver_source_lane_id": veh_info.get(
                "external_state_maneuver_source_lane_id", ""
            ),
            "external_state_maneuver_target_lane_id": veh_info.get(
                "external_state_maneuver_target_lane_id", ""
            ),
            "fail_closed_reason": (
                fail_closed_reason or state.get("fail_closed_reason", "")
            ),
            "lookahead_lateral_horizon_displacement": self._as_finite_float(
                veh_info.get("lookahead_lateral_horizon_displacement")
            ),
            "lookahead_target_lateral_distance": self._as_finite_float(
                veh_info.get("lookahead_target_lateral_distance")
            ),
            "lookahead_route_tangent_x": self._as_finite_float(
                veh_info.get("lookahead_route_tangent_x")
            ),
            "lookahead_route_tangent_y": self._as_finite_float(
                veh_info.get("lookahead_route_tangent_y")
            ),
            "lookahead_world_left_normal_x": self._as_finite_float(
                veh_info.get("lookahead_world_left_normal_x")
            ),
            "lookahead_world_left_normal_y": self._as_finite_float(
                veh_info.get("lookahead_world_left_normal_y")
            ),
            "phase_b_lateral_delta": self._as_finite_float(
                veh_info.get("lookahead_phase_b_lateral_delta")
            ),
            "expected_phase_b_lateral_distance": self._as_finite_float(
                veh_info.get("lookahead_expected_phase_b_lateral_distance")
            ),
            "world_lateral_speed": self._as_finite_float(
                veh_info.get("lookahead_world_lateral_speed")
            ),
            "lookahead_origin_x": self._as_finite_float(
                veh_info.get("lookahead_origin_x")
            ),
            "lookahead_origin_y": self._as_finite_float(
                veh_info.get("lookahead_origin_y")
            ),
            "carla_x": self._as_finite_float(current_transform.location.x),
            "carla_y": self._as_finite_float(current_transform.location.y),
            "carla_yaw": self._as_finite_float(current_transform.rotation.yaw),
            "phase_a_observed_acceleration": self._as_finite_float(
                veh_info.get("feedback_observed_acceleration")
            ),
            "phase_a_requested_lane_id": veh_info.get(
                "feedback_requested_lane_id"
            ),
            "phase_a_observed_lane_id": veh_info.get("feedback_observed_lane_id"),
            "sumo_lane_change_intent": veh_info.get(
                "sumo_lane_change_intent", "none"
            ),
            "sumo_lane_change_target_lane_id": veh_info.get(
                "sumo_lane_change_target_lane_id", ""
            ),
            "sumo_desired_speed": self._as_finite_float(veh_info.get("sumo_desired_speed")),
            "sumo_reported_acceleration": self._as_finite_float(veh_info.get("acceleration")),
            "feedback_observed_speed": self._as_finite_float(veh_info.get("feedback_observed_speed")),
            "sumo_requested_acceleration": state.get("sumo_requested_acceleration"),
            "sumo_emergency_decel": state.get("sumo_emergency_decel"),
            "restart_active": bool(state.get("restart_active")),
            "restart_target_speed": state.get("restart_target_speed"),
            "longitudinal_position_error": state.get(
                "longitudinal_position_error"
            ),
            "longitudinal_velocity_error": state.get(
                "longitudinal_velocity_error"
            ),
            "ackermann_target_speed": target_speed,
            "ackermann_target_acceleration": target_acceleration,
            "control_mode": control_mode,
            "emergency_brake_active": bool(
                state.get("emergency_brake_active", False)
            ),
            "emergency_brake_release_ticks": int(
                state.get("emergency_brake_release_ticks", 0)
            ),
            "commanded_throttle": self._as_finite_float(commanded_throttle),
            "commanded_brake": self._as_finite_float(commanded_brake),
            "commanded_steer": self._as_finite_float(commanded_steer),
            "carla_speed": current_speed,
            "carla_speed_delta_acceleration": speed_delta_acceleration,
            "carla_longitudinal_acceleration": longitudinal_acceleration,
            "carla_applied_throttle": applied_throttle,
            "carla_applied_brake": applied_brake,
            "carla_applied_steer": applied_steer,
            "position_error": position_error,
            "lookahead_distance": self._as_finite_float(
                veh_info.get("lookahead_distance")
            ),
            "lookahead_heading_change": self._as_finite_float(
                veh_info.get("lookahead_heading_change")
            ),
            "lookahead_lane_change_blend": self._as_finite_float(
                veh_info.get("lookahead_lane_change_blend")
            ),
            "sumo_lateral_speed": self._as_finite_float(
                veh_info.get("lateral_speed")
            ),
            "feedback_position_skipped_for_lane_change": bool(
                veh_info.get("feedback_position_skipped_for_lane_change", False)
            ),
            "wheel_base_m": self._as_finite_float(state.get("wheel_base_m")),
            "rear_axle_local_x_m": self._as_finite_float(
                state.get("rear_axle_local_x_m")
            ),
            "front_bumper_local_x_m": self._as_finite_float(
                state.get("front_bumper_local_x_m")
            ),
            "front_bumper_from_bounding_box": bool(
                state.get("front_bumper_from_bounding_box", False)
            ),
            "geometry_from_physics": bool(state.get("geometry_from_physics", False)),
            "pure_pursuit_raw_steer": self._as_finite_float(
                getattr(control_values, "raw_steer", None)
            ),
            "pure_pursuit_clamped_steer": self._as_finite_float(
                getattr(control_values, "clamped_steer", None)
            ),
            "pure_pursuit_command_steer": self._as_finite_float(
                getattr(control_values, "steer", None)
            ),
            "lookahead_local_x": self._as_finite_float(
                getattr(control_values, "lookahead_local_x", None)
            ),
            "lookahead_local_y": self._as_finite_float(
                getattr(control_values, "lookahead_local_y", None)
            ),
            "control_point_x": self._as_finite_float(
                getattr(control_values, "control_point_x", None)
            ),
            "control_point_y": self._as_finite_float(
                getattr(control_values, "control_point_y", None)
            ),
            "feedback_unhealthy": feedback_unhealthy,
            "target_behind": target_behind,
        }
        print("AckermannControlTrace " + json.dumps(trace, sort_keys=True), flush=True)

    def tick(self):
        """One co-sim step. Dispatches on the tick mode.

        follow (default): _tick_follow() - the passive pipeline (the bridge
        owns world.tick(); this process follows via wait_for_tick).
        master: _tick_master() - feature-parity state apply, fixed-cadence
        world.tick(), feedback collection, and one pending SUMO step.
        async: _tick_async() - nobody ticks; wall-clock-paced SUMO steps
        against a free-running CARLA server.
        """
        if self.tick_master:
            return self._tick_master()
        if self.tick_async:
            return self._tick_async()
        return self._tick_follow()

    def _resolve_pending_master_tick(self):
        """Resolve the prior SUMO request before advancing another CARLA frame."""
        if self._inproc_tick_handle is None:
            return True
        try:
            result = self._inproc_tick_handle.result(timeout=300.0)
        except Exception as exc:
            reason = f"inprocess_tick_error:{type(exc).__name__}"
            print(f"TeraSim in-process tick failed: {exc}. Exiting...", flush=True)
            self._apply_physics_fail_closed_brake(reason)
            return False
        if result.status in ("finished", "error"):
            print(f"TeraSim ended (status={result.status}). Exiting...", flush=True)
            self._apply_physics_fail_closed_brake(result.status)
            return False
        if result.state is not None:
            self._inproc_prev_state = result.state
        return True

    def _tick_master(self):
        """One cycle as the physical co-simulation clock master.

        Preserve the validated feature pipeline: resolve the SUMO step requested
        on the previous cycle -> apply its state to CARLA -> advance exactly one
        CARLA frame -> collect that frame's AV/background feedback -> request one
        SUMO step. The newly requested step is resolved at the beginning of the
        next cycle, so at most one SUMO step is in flight.

        Deadlines are pinned to a step_length wall-clock grid and never wait
        for anyone (R15). An overrun cycle fires immediately and re-pins the
        grid to "now + step_length" (no debt is carried over; a chronically
        slow world update turns into slow motion, not a tick burst).
        """
        if not self._resolve_pending_master_tick():
            return False

        if self._inproc_prev_state is not None:
            self.sync_cosim_actor_to_carla(self._inproc_prev_state)
            if not getattr(self.args, "skip_tls", False):
                self.sync_cosim_tls_to_carla(self._inproc_prev_state)

        now = time.monotonic()
        if self._next_tick_deadline is not None and now < self._next_tick_deadline:
            time.sleep(self._next_tick_deadline - now)
            self._next_tick_deadline += self.step_length
        else:
            self._next_tick_deadline = time.monotonic() + self.step_length

        t0 = time.monotonic()
        self._last_world_frame = self.world.tick()
        tick_ms = (time.monotonic() - t0) * 1000.0
        self._tick_times_ms.append(tick_ms)
        self._tick_time_hist[min(int(tick_ms // 5), 10)] += 1
        nveh = len(self._vehicle_actor_index)
        self._tick_veh_min = nveh if self._tick_veh_min is None else min(self._tick_veh_min, nveh)
        self._tick_veh_max = nveh if self._tick_veh_max is None else max(self._tick_veh_max, nveh)
        if len(self._tick_times_ms) >= 1200:
            arr = sorted(self._tick_times_ms)
            n = len(arr)
            hist = "/".join(str(c) for c in self._tick_time_hist)
            print(
                f"[tick-timing] world.tick ms: p50={arr[n // 2]:.1f} "
                f"p95={arr[int(n * 0.95)]:.1f} max={arr[-1]:.1f} n={n} "
                f"hist(5ms bins,0->50+)={hist} "
                f"carla_veh={self._tick_veh_min}-{self._tick_veh_max}",
                flush=True,
            )
            self._tick_times_ms = []
            self._tick_time_hist = [0] * 11
            self._tick_veh_min = None
            self._tick_veh_max = None

        commands = []
        if self.control_av:
            av_command = self._build_av_command()
            if av_command is not None:
                commands.append(av_command)
        commands.extend(self._build_physics_feedback_commands())

        try:
            self._inproc_tick_handle = self.inprocess_plugin.tick_async(commands)
        except Exception as exc:
            reason = f"inprocess_tick_request_error:{type(exc).__name__}"
            print(f"TeraSim in-process tick request failed: {exc}. Exiting...", flush=True)
            self._apply_physics_fail_closed_brake(reason)
            return False
        return True

    def _tick_async(self):
        """One cycle in async mode (free-running CARLA, wall-paced SUMO).

        Nobody calls world.tick(): run_cosim cleared synchronous_mode, so the
        server advances by itself with a variable delta (sim time tracks real
        time by construction). This loop paces SUMO on the same step_length
        wall-clock grid as _tick_master; a cycle that arrives late additionally
        runs the missed periods as extra SUMO steps so SUMO time keeps tracking
        real time, capped at
        _async_catchup_max_steps (debt beyond the cap is dropped = slow motion,
        never a long burst). All CARLA reads in the cycle (get_snapshot(),
        get_transform()) are served from the client-side state cache fed by the
        server's frame stream, so no part of the cycle waits on the server.
        """
        now = time.monotonic()
        steps_due = 1
        if self._next_tick_deadline is None:
            self._next_tick_deadline = now + self.step_length
        elif now < self._next_tick_deadline:
            time.sleep(self._next_tick_deadline - now)
            self._next_tick_deadline += self.step_length
        else:
            # Late: also run the periods that already elapsed. Past the cap,
            # drop the remaining debt and re-pin the grid to "now".
            missed = 1 + int((now - self._next_tick_deadline) / self.step_length)
            self._async_overrun_cycles += 1
            if missed > self._async_catchup_max_steps:
                self._async_dropped_periods += missed - self._async_catchup_max_steps
                steps_due = self._async_catchup_max_steps
                self._next_tick_deadline = time.monotonic() + self.step_length
            else:
                steps_due = missed
                self._next_tick_deadline += missed * self.step_length
            self._async_catchup_steps += steps_due - 1

        t0 = time.monotonic()
        if self._async_window_start is None:
            self._async_window_start = t0
        snapshot = self.world.get_snapshot()
        frame = getattr(snapshot, "frame", None)
        self._last_world_frame = frame
        timestamp = getattr(snapshot, "timestamp", None)
        if timestamp is not None:
            if self._async_start_wall is None:
                self._async_start_wall = t0
                self._async_start_carla_elapsed = timestamp.elapsed_seconds
            self._async_last_carla_elapsed = timestamp.elapsed_seconds
            delta = getattr(timestamp, "delta_seconds", None)
            if delta:
                self._async_delta_ms_samples.append(delta * 1000.0)
                if delta > 0.1:
                    # Above 100ms per frame CARLA's physics substepping
                    # (<=10 substeps of <=10ms) can no longer cover the frame.
                    self._async_delta_over_100ms += 1
        if frame is not None:
            if self._async_prev_frame is not None:
                advanced = frame - self._async_prev_frame
                if advanced > 0:
                    self._async_frames_advanced += advanced
                    self._async_stalled_cycles = 0
                else:
                    self._async_stalled_cycles += 1
                    if self._async_stalled_cycles == self._async_stall_warn_cycles:
                        print(
                            "[async-timing] WARN: no new CARLA frame for "
                            f"{self._async_stall_warn_cycles} cycles - the "
                            "world looks stuck in synchronous mode (a "
                            "relaunched bridge re-applies it; restart this "
                            "process to clear the settings)",
                            flush=True,
                        )
            self._async_prev_frame = frame

        sumo_ms = 0.0
        for _ in range(steps_due):
            commands = []
            if self.control_av:
                av_command = self._build_av_command()
                if av_command is not None:
                    commands.append(av_command)
            s0 = time.monotonic()
            handle = self.inprocess_plugin.tick_async(commands)
            try:
                result = handle.result(timeout=300.0)
            except TimeoutError as e:
                print(f"TeraSim in-process tick failed: {e}. Exiting...")
                return False
            sumo_ms += (time.monotonic() - s0) * 1000.0
            self._async_sumo_steps_total += 1
            if result.status in ("finished", "error"):
                print(f"TeraSim ended (status={result.status}). Exiting...")
                return False
            if result.state is not None:
                self._inproc_prev_state = result.state

        w0 = time.monotonic()
        if self._inproc_prev_state is not None:
            self.sync_cosim_actor_to_carla(self._inproc_prev_state)
            if not getattr(self.args, "skip_tls", False):
                self.sync_cosim_tls_to_carla(self._inproc_prev_state)
        done = time.monotonic()

        self._async_sumo_ms.append(sumo_ms)
        self._async_write_ms.append((done - w0) * 1000.0)
        self._async_work_ms.append((done - t0) * 1000.0)
        nveh = len(self._vehicle_actor_index)
        self._async_veh_min = nveh if self._async_veh_min is None else min(self._async_veh_min, nveh)
        self._async_veh_max = nveh if self._async_veh_max is None else max(self._async_veh_max, nveh)
        state = self._inproc_prev_state
        if isinstance(state, dict):
            sveh = len(state.get("agent_details", {}).get("vehicle", {}))
            self._async_sumo_veh_min = (
                sveh if self._async_sumo_veh_min is None else min(self._async_sumo_veh_min, sveh)
            )
            self._async_sumo_veh_max = (
                sveh if self._async_sumo_veh_max is None else max(self._async_sumo_veh_max, sveh)
            )
        if len(self._async_work_ms) >= 1200:
            self._print_async_timing_window(done)
        return True

    def _print_async_timing_window(self, now):
        """Emit the ~60s [async-timing] line and reset the window counters.

        Timing comes with the vehicle counts (carla_veh = actors synced into
        CARLA, sumo_veh = vehicles in the SUMO state), the CARLA frame stats
        (fps + sampled per-frame delta; over100ms counts frames that break the
        physics substep guarantee), and the cumulative drift of SUMO time and
        CARLA time against the wall clock (a growing negative sumo drift means
        dropped catch-up debt = SUMO slow motion).
        """
        work = sorted(self._async_work_ms)
        sumo = sorted(self._async_sumo_ms)
        write = sorted(self._async_write_ms)
        n = len(work)
        window_s = max(1e-9, now - self._async_window_start)
        fps = self._async_frames_advanced / window_s
        if self._async_delta_ms_samples:
            deltas = sorted(self._async_delta_ms_samples)
            delta_txt = (
                f"frame_ms p50={deltas[len(deltas) // 2]:.1f} "
                f"max={deltas[-1]:.1f} over100ms={self._async_delta_over_100ms}"
            )
        else:
            delta_txt = "frame_ms n/a"
        wall_elapsed = now - self._async_start_wall
        sumo_drift = self._async_sumo_steps_total * self.step_length - wall_elapsed
        if self._async_last_carla_elapsed is not None:
            carla_drift = (
                self._async_last_carla_elapsed - self._async_start_carla_elapsed
            ) - wall_elapsed
            drift_txt = f"drift_s sumo={sumo_drift:+.2f} carla={carla_drift:+.2f}"
        else:
            drift_txt = f"drift_s sumo={sumo_drift:+.2f} carla=n/a"
        print(
            f"[async-timing] cycle work ms: p50={work[n // 2]:.1f} "
            f"p95={work[int(n * 0.95)]:.1f} max={work[-1]:.1f} "
            f"(sumo p50={sumo[n // 2]:.1f} write p50={write[n // 2]:.1f}) n={n} | "
            f"overrun={self._async_overrun_cycles} "
            f"catchup={self._async_catchup_steps} "
            f"dropped={self._async_dropped_periods} | "
            f"carla_fps={fps:.1f} {delta_txt} | {drift_txt} | "
            f"carla_veh={self._async_veh_min}-{self._async_veh_max} "
            f"sumo_veh={self._async_sumo_veh_min}-{self._async_sumo_veh_max}",
            flush=True,
        )
        self._async_work_ms = []
        self._async_sumo_ms = []
        self._async_write_ms = []
        self._async_overrun_cycles = 0
        self._async_catchup_steps = 0
        self._async_dropped_periods = 0
        self._async_frames_advanced = 0
        self._async_delta_ms_samples = []
        self._async_delta_over_100ms = 0
        self._async_veh_min = None
        self._async_veh_max = None
        self._async_sumo_veh_min = None
        self._async_sumo_veh_max = None
        self._async_window_start = None

    def _tick_follow(self):
        """One co-sim step over the in-process link (single process, no RPC).

        Pipeline parity with the former two-process transports: the state
        rendered into CARLA is the previous step's state, and the SUMO step
        requested this tick computes on the sim thread while this thread
        renders into CARLA and waits for the CARLA tick.
        """
        # Resolve the step requested on the previous tick.
        if self._inproc_tick_handle is not None:
            try:
                result = self._inproc_tick_handle.result(timeout=300.0)
            except TimeoutError as e:
                print(f"TeraSim in-process tick failed: {e}. Exiting...")
                return False
            if result.status in ("finished", "error"):
                print(f"TeraSim ended (status={result.status}). Exiting...")
                return False
            if result.state is not None:
                self._inproc_prev_state = result.state

        commands = []
        if self.control_av:
            av_command = self._build_av_command()
            if av_command is not None:
                commands.append(av_command)

        # Request the next SUMO step; it runs on the sim thread while this
        # thread renders below and waits for the CARLA tick.
        self._inproc_tick_handle = self.inprocess_plugin.tick_async(commands)

        # Render the previous step's state (one step of latency, as the former
        # transports also rendered the state before requesting its tick).
        if self._inproc_prev_state is not None:
            self.sync_cosim_actor_to_carla(self._inproc_prev_state)
            if not getattr(self.args, "skip_tls", False):
                self.sync_cosim_tls_to_carla(self._inproc_prev_state)

        # passive mode: the ego-side bridge (autoware_carla_interface) is the sole
        # owner of world.tick(). CarlaCosim does not tick the world; it waits for the
        # bridge's tick so the two clients stay synchronized on one CARLA server.
        if getattr(self.args, "passive_tick", False):
            snapshot = self.world.wait_for_tick()
            self._last_world_frame = getattr(snapshot, "frame", None)
        else:
            self._last_world_frame = self.world.tick()
        return True

    def _build_av_command(self):
        # 3-cosim: the ego that drives in CARLA is the Autoware ego (role "ego_vehicle"), not the
        # SUMO-spawned "AV". Read that actor's pose and build a set_state command for the SUMO AV
        # so background traffic avoids it. Returns None when the command cannot be built yet
        # (actor missing / av_shape not initialized). The caller (tick) attaches it to the
        # tick_async request.
        av_role = getattr(self.args, "av_carla_role", AV_SUMO_ID)

        if not self.av_shape:
            # av_shape is filled by initialize_av in sync_cosim_actor_to_carla, which runs later in
            # the same tick. Skip until then to avoid indexing an empty shape on the first tick.
            return None

        # Resolve the ego handle once and reuse it; the full actor walk
        # (get_actor_id_from_attribute) is a per-actor attribute conversion
        # over the whole world and only runs again when the cached handle
        # fails or after an index reconcile clears it.
        AV = self._av_actor
        transform = None
        if AV is not None:
            try:
                transform = AV.get_transform()
            except Exception:
                AV = None
        if AV is None:
            vehicle_status, carla_id = get_actor_id_from_attribute(self.world, av_role)
            if not vehicle_status:
                print(f"AV source actor (role={av_role}) not found in Carla simulation.")
                return None
            AV = self.world.get_actor(carla_id)
            if AV is None:
                print(f"AV actor {carla_id} not resolvable this loop; command skipped.", flush=True)
                return None
            self._av_actor = AV
            transform = AV.get_transform()
        if self.ego_label_enabled:
            draw_text(self.world, transform.location + carla.Location(z=2.5), AV_SUMO_ID)
        # draw_point(
        #     self.world,
        #     size=0.05,
        #     color=(255, 0, 0),
        #     location=transform.location + carla.Location(z=2.5),
        #     life_time=0,
        # )

        velocity = AV.get_velocity()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

        # Reverse transform: CARLA -> SUMO
        if self._coord_transformer is not None:
            # Direct reverse: CARLA location -> xodr coords -> UTM -> SUMO CRS
            # CARLA: x = xodr_x, y = -xodr_y
            xodr_x = transform.location.x
            xodr_y = -transform.location.y
            sumo_x, sumo_y = self._transform_xodr_to_sumo(xodr_x, xodr_y)
            # Apply vehicle shape correction (SUMO position is front bumper).
            # Heading in the SUMO/ENU frame (y north) is -yaw: CARLA yaw lives in a
            # y-south frame, so the rotation sense is mirrored (same convention as
            # carla_to_sumo in tools.py). Using +yaw flips the correction's north
            # component: up to a full car length backward on N/S roads and half a
            # car sideways on diagonals -- enough to map the AV onto the wrong lane.
            yaw = math.radians(-transform.rotation.yaw)
            sumo_x += math.cos(yaw) * self.av_shape[0] / 2.0
            sumo_y += math.sin(yaw) * self.av_shape[0] / 2.0
            av_sumo_location = [sumo_x, sumo_y, transform.location.z]
            av_sumo_rotation = [transform.rotation.pitch, transform.rotation.yaw + 90, transform.rotation.roll]
        else:
            av_offset = [self.sumo_carla_offset[0], self.sumo_carla_offset[1], 0.0]
            av_sumo_location, av_sumo_rotation = carla_to_sumo(
                transform.location,
                transform.rotation,
                self.av_shape,
                av_offset
            )

        av_command = {
            "agent_id": AV_SUMO_ID,
            "agent_type": "vehicle",
            "command_type": "set_state",
            "data": {
                "position": [av_sumo_location[0], av_sumo_location[1]],
                "speed": speed,
                "sumo_angle": av_sumo_rotation[1],
            }
        }

        return av_command

    def sync_cosim_tls_to_carla(self, terasim_states):
        if not terasim_states:
            print("terasim_states not available.")
            return

        if "traffic_light_details" not in terasim_states:
            print("No traffic light details available.")
            return

        terasim_tls_data = terasim_states["traffic_light_details"]

        for node_id, node_info in terasim_tls_data.items():
            sumo_tls = node_info["tls"]
            sumo_information = json.loads(node_info["information"])
            parameters = None
            for program_id, program in sumo_information["programs"].items():
                try:
                    parameters = program["parameters"]
                    break
                except KeyError:
                    print(f"KeyError: Node ({node_id}) Program ({program}) does not have 'parameters' key.")
                    continue
            if parameters is None:
                print(f"Traffic Lights within Node ({node_id}) is not synchronized with Carla.")
                continue

            for i in range(len(sumo_tls)):
                param_key = f"{SUMO_CARLA_TLS_LINK_PREFIX}{i}"
                carla_landmark_ids = parameters.get(param_key, "")
                if carla_landmark_ids == "":
                    continue
                carla_landmark_ids = carla_landmark_ids.split(" ")
                for landmark_id in carla_landmark_ids:
                    light_id = int(landmark_id)
                    light_actor = self.world.get_actor(light_id)
                    if not light_actor:
                        print(f"Traffic light with ID {light_id} not found in CARLA.")
                        continue

                    # Defensive guard: CARLA may return a non-TrafficLight Actor
                    # when SUMO's TLS program parameters are not mapped to a
                    # real CARLA landmark_id (e.g. netconvert --tls.guess nets
                    # like Town01). Calling set_state on such an actor raises
                    # AttributeError and aborts the whole cosim tick.
                    if not isinstance(light_actor, carla.TrafficLight):
                        continue

                    light_state = sumo_tls[i]
                    if light_state == "G" or light_state == "g":
                        light_actor.set_state(carla.TrafficLightState.Green)
                    elif light_state == "Y" or light_state == "y":
                        light_actor.set_state(carla.TrafficLightState.Yellow)
                    elif light_state == "R" or light_state == "r":
                        light_actor.set_state(carla.TrafficLightState.Red)

    @staticmethod
    def _actor_xy(actor_info):
        try:
            return float(actor_info["x"]), float(actor_info["y"])
        except (KeyError, TypeError, ValueError):
            return None

    def _filter_actor_details_by_radius(self, vehicles, vrus):
        if not self.actor_filter_enabled:
            return vehicles, vrus

        center_info = vehicles.get(self.actor_filter_center_id)
        center_xy = self._actor_xy(center_info) if center_info is not None else None
        if center_xy is None:
            if not self._actor_filter_missing_center_warned:
                print(
                    "Warning: CARLA co-sim actor radius filter center "
                    f"{self.actor_filter_center_id!r} is not available; "
                    "using unfiltered actors.",
                    flush=True,
                )
                self._actor_filter_missing_center_warned = True
            return vehicles, vrus

        center_x, center_y = center_xy
        radius_squared = self.actor_filter_radius * self.actor_filter_radius

        filtered_vehicles = {}
        for veh_id, veh_info in vehicles.items():
            if veh_id in {self.actor_filter_center_id, AV_SUMO_ID}:
                filtered_vehicles[veh_id] = veh_info
                continue
            vehicle_xy = self._actor_xy(veh_info)
            if vehicle_xy is None:
                filtered_vehicles[veh_id] = veh_info
                continue
            dx = vehicle_xy[0] - center_x
            dy = vehicle_xy[1] - center_y
            if dx * dx + dy * dy <= radius_squared:
                filtered_vehicles[veh_id] = veh_info

        return filtered_vehicles, vrus

    def sync_cosim_actor_to_carla(self, terasim_states):
        """Update all actors in cosim to CARLA from the given state dict."""
        if not terasim_states:
            print("terasim_states not available.")
            return

        if "agent_details" not in terasim_states:
            print("No agent details available.")
            return

        if "vehicle" not in terasim_states["agent_details"]:
            print("No vehicle details available.")
            return

        if "vru" not in terasim_states["agent_details"]:
            print("No VRU details available.")
            return

        cosim_id_record = set()
        # Frame number comes from the last wait_for_tick()/tick() result; the
        # extra get_snapshot() round-trip only happens before the first tick.
        current_frame = self._last_world_frame
        if current_frame is None:
            current_frame = self.world.get_snapshot().frame
        # Incrementally-maintained indexes; full world scan only at seed time
        # and every N ticks as a consistency net.
        if not self._actor_index_seeded or (
            self._actor_index_reconcile_every > 0
            and self._ticks_since_reconcile >= self._actor_index_reconcile_every
        ):
            self._build_actor_role_indexes()
            self._av_actor = None  # re-resolve the ego handle on the same cadence
        self._ticks_since_reconcile += 1
        vehicle_actor_index = self._vehicle_actor_index
        pedestrian_actor_index = self._pedestrian_actor_index
        vehicles = terasim_states["agent_details"]["vehicle"]
        vrus = terasim_states["agent_details"]["vru"]
        vehicles, vrus = self._filter_actor_details_by_radius(vehicles, vrus)
        transform_batch = []
        ackermann_batch = []
        spawn_requests = []

        for veh_id, veh_info in vehicles.items():
            if self.control_av and veh_id == AV_SUMO_ID:
                if not self.initialize_av:
                    self.initialize_av = True
                    self.av_shape = [
                        veh_info["length"],
                        veh_info["width"],
                        veh_info["height"],
                    ]
                    print("AV is initialized based on SUMO state.")
                # In control_av / 3-cosim mode, the CARLA-side ego actor already
                # represents the AV. Do not spawn a second SUMO AV into CARLA.
                continue

            self._process_vehicle(
                veh_id,
                veh_info,
                cosim_id_record,
                carla_actor=vehicle_actor_index.get(veh_id),
                actor_index=vehicle_actor_index,
                current_frame=current_frame,
                transform_batch=transform_batch,
                ackermann_batch=ackermann_batch,
                spawn_requests=spawn_requests,
            )

        for vru_id, vru_info in vrus.items():
            vru_actor_index = (
                vehicle_actor_index
                if self._vru_uses_vehicle_blueprint(vru_info)
                else pedestrian_actor_index
            )
            self._process_vru(
                vru_id,
                vru_info,
                cosim_id_record,
                carla_actor=vru_actor_index.get(vru_id),
                actor_index=vru_actor_index,
                current_frame=current_frame,
                transform_batch=transform_batch,
                spawn_requests=spawn_requests,
            )

        self._flush_actor_spawn_batch(spawn_requests, transform_batch)
        self._flush_actor_transform_batch(transform_batch)
        self._flush_actor_ackermann_batch(ackermann_batch)
        self._prune_collision_sensors(vehicles.keys())
        self._raise_pending_authoritative_action_error()
        self._cleanup_stale_actors(cosim_id_record)
        self._prune_spawn_failures(vehicles.keys(), vrus.keys())

        # self.sync_cosim_tls_to_carla()

    def _transform_sumo_to_xodr(self, sumo_x, sumo_y):
        """Transform SUMO coordinates to xodr/CARLA coordinate system.

        Steps: SUMO internal coords -> raw CRS coords -> standard UTM -> subtract xodr origin.
        Returns (xodr_x, xodr_y) where xodr_y still needs to be negated for CARLA.
        """
        if self._coord_transformer is not None:
            # SUMO internal coords = raw coords + netOffset
            raw_x = sumo_x - self._sumo_net_offset[0]
            raw_y = sumo_y - self._sumo_net_offset[1]
            # Transform to standard UTM
            utm_x, utm_y = self._coord_transformer.transform(raw_x, raw_y)
            # Subtract xodr origin to get xodr-local coordinates
            xodr_x = utm_x - self._xodr_origin_utm[0]
            xodr_y = utm_y - self._xodr_origin_utm[1]
            return xodr_x, xodr_y
        return None, None

    def _transform_xodr_to_sumo(self, xodr_x, xodr_y):
        """Reverse transform: xodr/CARLA coordinates -> SUMO coordinates.

        Steps: add xodr origin -> standard UTM -> SUMO CRS -> add netOffset.
        """
        if self._coord_transformer is not None:
            # xodr-local -> standard UTM
            utm_x = xodr_x + self._xodr_origin_utm[0]
            utm_y = xodr_y + self._xodr_origin_utm[1]
            # Reverse transform: UTM -> SUMO CRS
            # _coord_transformer goes SUMO CRS -> UTM, we need the inverse
            raw_x, raw_y = self._coord_transformer.transform(utm_x, utm_y, direction='INVERSE')
            # Add netOffset
            sumo_x = raw_x + self._sumo_net_offset[0]
            sumo_y = raw_y + self._sumo_net_offset[1]
            return sumo_x, sumo_y
        return None, None

    def _get_carla_offset(self, sumo_location, z_offset):
        """Get the offset for sumo_to_carla, incorporating coordinate transformation.
        If using projection-based transform, converts SUMO coords to xodr coords and
        computes the effective offset. Otherwise, returns the calibrated static offset.
        """
        if self._coord_transformer is not None:
            xodr_x, xodr_y = self._transform_sumo_to_xodr(sumo_location[0], sumo_location[1])
            # sumo_to_carla computes: carla_x = sumo_x - cos*shape/2 + offset_x
            #                          carla_y = -(sumo_y - sin*shape/2) + offset_y
            # We want: carla_x ≈ xodr_x, carla_y ≈ -xodr_y
            # So: offset_x = xodr_x - sumo_x (approximately, ignoring shape term)
            #     offset_y = -xodr_y - (-sumo_y) = sumo_y - xodr_y
            return [xodr_x - sumo_location[0], sumo_location[1] - xodr_y, z_offset]
        return [self.sumo_carla_offset[0], self.sumo_carla_offset[1], z_offset]

    # Elevated spawn height to avoid collision with OpenDRIVE-generated road geometry
    # (guardrails, curbs, barriers). After spawn, correct transform is set immediately.
    SPAWN_Z_CLEARANCE = 5.0
    ACKERMANN_SPAWN_STABILITY_TICKS = 3
    ACKERMANN_SPAWN_MAX_Z_ERROR = 0.5
    ACKERMANN_SPAWN_MAX_SPEED_ERROR = 3.0
    ACKERMANN_SPAWN_MAX_VERTICAL_SPEED = 1.0
    ACKERMANN_SPAWN_MAX_TILT_ERROR = 10.0
    SPAWN_RETRY_FRAMES = 10
    SPAWN_RETRY_DISTANCE = 5.0

    @staticmethod
    def _spawn_failure_key(actor_type, actor_id):
        return actor_type, actor_id

    def _fresh_blueprint(self, blueprint):
        # The library handle is cached at init; fetching it per spawn was one
        # blocking RPC per newly-entering vehicle.
        try:
            return self._bp_library.find(blueprint.id)
        except Exception:
            return blueprint

    def _select_vehicle_blueprint(self, veh_id, veh_info):
        if "BIKE" in veh_info["type"]:
            blueprint = random.choice(self.bike_blueprints)
        elif "MOTOR" in veh_info["type"]:
            blueprint = random.choice(self.motor_blueprints)
        elif "POLICE" in veh_info["type"]:
            blueprint = random.choice(self.police_car_blueprints)
        else:
            blueprint = random.choice(self.vehicle_blueprints)
        blueprint = self._fresh_blueprint(blueprint)
        blueprint.set_attribute("role_name", veh_id)
        if veh_id == AV_SUMO_ID:
            blueprint.set_attribute("color", "255, 0, 0")
        else:
            blueprint.set_attribute("color", "0, 102, 204")
        return blueprint

    def _select_vru_blueprint(self, vru_id, vru_info):
        if "BIKE" in vru_info["type"]:
            blueprint = random.choice(self.bike_blueprints)
        elif "MOTOR" in vru_info["type"]:
            blueprint = random.choice(self.motor_blueprints)
        else:
            blueprint = random.choice(self.pedestrian_blueprints)
        blueprint = self._fresh_blueprint(blueprint)
        blueprint.set_attribute("role_name", vru_id)
        return blueprint

    @staticmethod
    def _empty_spawn_batch_stats():
        return {
            "spawn_calls": 0,
            "spawn_total": 0.0,
            "spawn_max": 0.0,
            "spawn_success": 0,
            "spawn_failed": 0,
            "apply_batch_sync_time": 0.0,
            "apply_batch_sync_max": 0.0,
            "apply_batch_sync_success_time": 0.0,
            "apply_batch_sync_failed_time": 0.0,
        }

    def _queue_actor_spawn(self, spawn_requests, request):
        if self.batch_spawn_enabled and spawn_requests is not None:
            spawn_requests.append(request)
            return True
        return False

    def _flush_actor_spawn_batch(self, spawn_requests, transform_batch=None):
        stats = self._empty_spawn_batch_stats()
        if not spawn_requests:
            return stats

        commands = [
            carla.command.SpawnActor(request["blueprint"], request["spawn_transform"]).then(
                carla.command.SetSimulatePhysics(carla.command.FutureActor, False)
            )
            for request in spawn_requests
        ]

        total_start = time.perf_counter()
        apply_start = time.perf_counter()
        # due_tick_cue=False: with True, in synchronous mode this call blocks until the
        # next world tick. Spawns happen every time a vehicle enters the sync radius, so
        # the blocking variant locks the whole co-sim loop to 2 CARLA ticks per step
        # (measured: client wait p50 ~70ms vs the 51ms tick period). The responses
        # (actor ids) are still returned without the tick; an actor that is not yet
        # resolvable falls into the existing spawn-failure retry path.
        responses = self.client.apply_batch_sync(commands, False)
        apply_elapsed = time.perf_counter() - apply_start

        stats["spawn_calls"] = len(spawn_requests)
        stats["apply_batch_sync_time"] = apply_elapsed
        stats["apply_batch_sync_max"] = apply_elapsed

        response_count = len(responses)
        success_count = sum(1 for response in responses if not response.error)
        failed_count = response_count - success_count
        stats["spawn_success"] = success_count
        stats["spawn_failed"] = failed_count
        if response_count:
            stats["apply_batch_sync_success_time"] = apply_elapsed * success_count / response_count
            stats["apply_batch_sync_failed_time"] = apply_elapsed * failed_count / response_count

        for request, response in zip(spawn_requests, responses):
            actor_type = request["actor_type"]
            actor_id = request["actor_id"]
            if response.error:
                log_spawn_actor_failure(
                    self.world,
                    request["blueprint"],
                    request["spawn_transform"],
                    actor_id,
                    response.error,
                )
                self._record_spawn_failure(
                    actor_type, actor_id, request["sumo_location"], request["current_frame"]
                )
                continue

            actor = self.world.get_actor(response.actor_id)
            if actor is None:
                self._record_spawn_failure(
                    actor_type, actor_id, request["sumo_location"], request["current_frame"]
                )
                continue

            self._clear_spawn_failure(actor_type, actor_id)
            actor_index = request.get("actor_index")
            if actor_index is not None:
                actor_index[actor_id] = actor
            if actor_type == "vehicle":
                self._ensure_collision_sensor(actor, actor_id)
            if request.get("enable_physics_after_spawn", False):
                self._ackermann_actor_state.pop(actor_id, None)
                actor_state = self._ackermann_actor_state.setdefault(actor_id, {})
                self._initialize_ackermann_actor_geometry(
                    actor, actor_id, actor_state
                )
                post_spawn_transform = self._sumo_front_to_carla_transform(
                    request["sumo_location"],
                    request["sumo_rotation"],
                    request["shape"],
                    request["post_spawn_offset"],
                    actor_state.get("front_bumper_local_x_m"),
                )
                self._prepare_ackermann_actor_physics(
                    actor,
                    actor_id,
                    request["actor_info"].get("speed"),
                    post_spawn_transform,
                    request["spawn_transform"],
                )
            else:
                self._queue_actor_transform(
                    actor, request["post_spawn_transform"], transform_batch, cosim_id=actor_id
                )
                if actor_type == "vru":
                    self._apply_vru_walker_control(
                        request["actor_info"], request["sumo_angle"], actor
                    )

        stats["spawn_total"] = time.perf_counter() - total_start
        stats["spawn_max"] = stats["spawn_total"]
        return stats

    def _apply_vru_walker_control(self, vru_info, sumo_angle, pedestrian):
        if self._vru_uses_vehicle_blueprint(vru_info):
            return
        radians = math.radians(90 - sumo_angle)
        orientation = math.atan2(math.sin(radians), math.cos(radians))
        direction_x, direction_y = math.cos(orientation), math.sin(orientation)
        walker_control = carla.WalkerControl(
            direction=carla.Vector3D(direction_x, direction_y, 0),
            speed=vru_info["speed"],
        )
        try:
            pedestrian.apply_control(walker_control)
        except Exception:
            pass

    def _queue_actor_transform(self, actor, transform, transform_batch=None, cosim_id=None):
        """Queue or apply a CARLA actor transform according to batching settings.

        cosim_id ties the queued command back to the persistent actor index so
        a failed apply (actor gone) drops the stale handle instead of relying
        on a per-tick world rescan.
        """
        if self.batch_transform_enabled and transform_batch is not None:
            transform_batch.append(
                (carla.command.ApplyTransform(actor.id, transform), cosim_id)
            )
            return
        actor.set_transform(transform)

    def _flush_actor_transform_batch(self, transform_batch):
        """Apply queued actor transforms in one CARLA batch call."""
        if not transform_batch:
            return []
        responses = self.client.apply_batch_sync(
            [command for command, _ in transform_batch], False
        )
        for (_, cosim_id), response in zip(transform_batch, responses):
            if response.error and cosim_id is not None:
                # Stale handle (e.g. actor destroyed outside this loop): forget
                # it so the next state entry re-spawns instead of erroring.
                self._drop_actor_from_indexes(cosim_id)
        return responses

    def _queue_actor_ackermann_control(
        self, actor, control, ackermann_batch=None, direct_vehicle_control=False
    ):
        command_name = (
            "ApplyVehicleControl"
            if direct_vehicle_control
            else "ApplyVehicleAckermannControl"
        )
        command_type = getattr(carla.command, command_name, None)
        if ackermann_batch is not None and command_type is not None:
            ackermann_batch.append(command_type(actor.id, control))
            return
        if direct_vehicle_control:
            actor.apply_control(control)
        else:
            actor.apply_ackermann_control(control)

    def _flush_actor_ackermann_batch(self, ackermann_batch):
        if not ackermann_batch:
            return []
        return self.client.apply_batch_sync(ackermann_batch, False)

    def _build_actor_role_indexes(self):
        """Full world scan seeding the persistent role_name -> actor indexes.

        Runs at start (also picks up leftovers from a previous run so cleanup
        can retire them) and every N ticks as a consistency net; between scans
        the indexes are maintained incrementally at spawn/destroy/transform
        error, so the per-tick get_actors() walk with per-actor attribute
        conversion is gone from the hot path.
        """
        vehicle_actor_index = {}
        pedestrian_actor_index = {}
        actors = self.world.get_actors()
        world_actor_count = 0
        for actor in actors:
            world_actor_count += 1
            role_name = actor.attributes.get("role_name")
            if not role_name:
                continue
            if actor.type_id.startswith("vehicle."):
                vehicle_actor_index.setdefault(role_name, actor)
            elif actor.type_id.startswith("walker.pedestrian."):
                pedestrian_actor_index.setdefault(role_name, actor)
        self._last_actor_index_world_actor_count = world_actor_count
        self._last_actor_index_vehicle_actor_count = len(vehicle_actor_index)
        self._last_actor_index_pedestrian_actor_count = len(pedestrian_actor_index)
        self._vehicle_actor_index = vehicle_actor_index
        self._pedestrian_actor_index = pedestrian_actor_index
        self._actor_index_seeded = True
        self._ticks_since_reconcile = 0
        return vehicle_actor_index, pedestrian_actor_index

    def _drop_actor_from_indexes(self, cosim_id):
        """Forget a cached actor handle (destroyed or failed); the next state
        entry for that id goes through the spawn path again."""
        self._vehicle_actor_index.pop(cosim_id, None)
        self._pedestrian_actor_index.pop(cosim_id, None)

    def _cleanup_stale_actors(self, cosim_id_record):
        """Destroy CARLA actors whose SUMO counterpart left the state.

        Index-based (no world scan) and batched: one DestroyActor batch per
        tick instead of one blocking destroy() RPC per stale actor.
        """
        protected = getattr(self.args, "protected_roles", None) or ["AV"]
        destroy_commands = []
        for index in (self._vehicle_actor_index, self._pedestrian_actor_index):
            stale_ids = [
                cosim_id
                for cosim_id in index
                if cosim_id not in cosim_id_record and cosim_id not in protected
            ]
            for cosim_id in stale_ids:
                actor = index.pop(cosim_id)
                destroy_commands.append(carla.command.DestroyActor(actor.id))
        active_vehicle_ids = set(self._vehicle_actor_index)
        for cache in (
            self._ackermann_actor_state,
            self._physics_feedback_frames,
            self._physics_feedback_failures,
            self._ackermann_feedback_state,
            self._ackermann_fail_closed_reasons,
        ):
            for stale_id in [
                actor_id
                for actor_id in cache
                if actor_id not in active_vehicle_ids
            ]:
                cache.pop(stale_id, None)
        if destroy_commands:
            self.client.apply_batch_sync(destroy_commands, False)

    @staticmethod
    def _vru_uses_vehicle_blueprint(vru_info):
        return "BIKE" in vru_info["type"] or "MOTOR" in vru_info["type"]

    def _should_retry_spawn(self, actor_type, actor_id, sumo_location, current_frame):
        failure = self._spawn_failures.get(self._spawn_failure_key(actor_type, actor_id))
        if failure is None:
            return True
        if failure.get("abandoned"):
            return False
        if actor_id == AV_SUMO_ID:
            return True

        dx = sumo_location[0] - failure["x"]
        dy = sumo_location[1] - failure["y"]
        if dx * dx + dy * dy >= self.SPAWN_RETRY_DISTANCE * self.SPAWN_RETRY_DISTANCE:
            return True

        next_retry_time = failure.get("next_retry_time")
        if next_retry_time is not None:
            return time.monotonic() >= next_retry_time
        return current_frame >= failure.get("next_retry_frame", current_frame)

    def _record_spawn_failure(self, actor_type, actor_id, sumo_location, current_frame):
        key = self._spawn_failure_key(actor_type, actor_id)
        previous = self._spawn_failures.get(key, {})
        failures = previous.get("failures", 0) + 1
        exponent = min(failures - 1, 30)
        delay = min(
            self.spawn_failure_backoff_max_seconds,
            self.spawn_failure_backoff_seconds * (2 ** exponent),
        )
        abandoned = failures >= max(1, int(getattr(self, "spawn_max_attempts", 3)))
        self._spawn_failures[key] = {
            "failures": failures,
            "abandoned": abandoned,
            "next_retry_frame": current_frame + self.SPAWN_RETRY_FRAMES,
            "next_retry_time": time.monotonic() + delay,
            "x": sumo_location[0],
            "y": sumo_location[1],
        }
        if abandoned:
            print(
                f"Warning: abandoning CARLA {actor_type} spawn for {actor_id!r} "
                f"after {failures} failed attempts. SUMO simulation will continue.",
                flush=True,
            )

    def _mark_spawn_abandoned(self, actor_type, actor_id, failures, reason):
        self._spawn_failures[self._spawn_failure_key(actor_type, actor_id)] = {
            "failures": int(failures),
            "abandoned": True,
            "reason": str(reason),
        }

    def _is_spawn_abandoned(self, actor_type, actor_id):
        failure = self._spawn_failures.get(self._spawn_failure_key(actor_type, actor_id))
        return bool(failure and failure.get("abandoned"))

    def _clear_spawn_failure(self, actor_type, actor_id):
        self._spawn_failures.pop(self._spawn_failure_key(actor_type, actor_id), None)

    @staticmethod
    def _as_finite_float(value):
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isfinite(number):
            return number
        return None

    def _warn_invalid_sumo_location_once(self, actor_type, actor_id, raw_location):
        key = (actor_type, actor_id)
        if key in self._invalid_location_warnings:
            return
        self._invalid_location_warnings.add(key)
        print(
            f"Warning: {actor_type} {actor_id!r} has invalid SUMO location "
            f"{raw_location}; skipping this tick.",
            flush=True,
        )

    def _resolve_sumo_location(self, actor_type, actor_id, actor_info, prefer_lane_relative=False):
        use_reconstructed = (
            prefer_lane_relative
            and bool(actor_info.get("reconstructed_position_valid", False))
        )
        if use_reconstructed:
            raw_location = {
                "x": actor_info.get("reconstructed_x"),
                "y": actor_info.get("reconstructed_y"),
                "z": actor_info.get("reconstructed_z"),
            }
        else:
            raw_location = {
                "x": actor_info.get("x"),
                "y": actor_info.get("y"),
                "z": actor_info.get("z"),
            }
        sumo_location = [
            self._as_finite_float(raw_location["x"]),
            self._as_finite_float(raw_location["y"]),
            self._as_finite_float(raw_location["z"]),
        ]
        if use_reconstructed and any(value is None for value in sumo_location):
            raw_location = {
                "x": actor_info.get("x"),
                "y": actor_info.get("y"),
                "z": actor_info.get("z"),
            }
            sumo_location = [
                self._as_finite_float(raw_location["x"]),
                self._as_finite_float(raw_location["y"]),
                self._as_finite_float(raw_location["z"]),
            ]
        if any(value is None for value in sumo_location):
            self._warn_invalid_sumo_location_once(actor_type, actor_id, raw_location)
            return None
        return sumo_location

    def _warn_missing_sumo_angle_once(self, actor_type, actor_id, action):
        key = (actor_type, actor_id, action)
        if key in self._missing_angle_warnings:
            return
        self._missing_angle_warnings.add(key)
        print(
            f"Warning: {actor_type} {actor_id!r} has missing sumo_angle; {action}.",
            flush=True,
        )

    def _resolve_sumo_angle(self, actor_type, actor_id, actor_info, carla_actor=None):
        angle = self._as_finite_float(actor_info.get("sumo_angle"))
        if angle is not None:
            return angle

        orientation = self._as_finite_float(actor_info.get("orientation"))
        if orientation is not None:
            self._warn_missing_sumo_angle_once(actor_type, actor_id, "using orientation fallback")
            return (90.0 - math.degrees(orientation)) % 360.0

        if carla_actor is not None:
            self._warn_missing_sumo_angle_once(actor_type, actor_id, "using previous CARLA yaw fallback")
            return (carla_actor.get_transform().rotation.yaw + 90.0) % 360.0

        self._warn_missing_sumo_angle_once(actor_type, actor_id, "skipping this tick")
        return None

    def _prune_spawn_failures(self, vehicle_ids, vru_ids):
        active_keys = {
            self._spawn_failure_key("vehicle", vehicle_id) for vehicle_id in vehicle_ids
        }
        active_keys.update(
            self._spawn_failure_key("vru", vru_id) for vru_id in vru_ids
        )
        for key in list(self._spawn_failures):
            if key not in active_keys:
                self._spawn_failures.pop(key, None)

    def _process_vehicle(
        self,
        veh_id,
        veh_info,
        cosim_id_record,
        carla_actor=None,
        actor_index=None,
        current_frame=None,
        transform_batch=None,
        ackermann_batch=None,
        spawn_requests=None,
    ):
        """Process a vehicle actor."""
        if self._is_spawn_abandoned("vehicle", veh_id):
            return
        cosim_id_record.add(veh_id)

        sumo_location = self._resolve_sumo_location(
            "vehicle",
            veh_id,
            veh_info,
            prefer_lane_relative=self.use_lane_relative_position,
        )
        if sumo_location is None:
            return
        sumo_angle = self._resolve_sumo_angle("vehicle", veh_id, veh_info, carla_actor)
        if sumo_angle is None:
            return
        sumo_slope = self._as_finite_float(veh_info.get("sumo_slope")) or 0.0
        sumo_rotation = [sumo_slope, sumo_angle, 0.0]
        shape = [veh_info["length"], veh_info["width"], veh_info["height"]]
        uses_ackermann_physics = self._uses_ackermann_physics(veh_id)

        vehicle = carla_actor
        if vehicle is not None:
            self._ensure_collision_sensor(vehicle, veh_id)
        if vehicle is None:
            if current_frame is None:
                current_frame = self.world.get_snapshot().frame
            if not self._should_retry_spawn("vehicle", veh_id, sumo_location, current_frame):
                return
            blueprint = self._select_vehicle_blueprint(veh_id, veh_info)
            # Spawn elevated to avoid collision with road geometry, then set correct transform
            sumo_offset = self._get_carla_offset(sumo_location, self.spawn_z_clearance)
            spawn_transform = (
                self._sumo_front_to_carla_transform(
                    sumo_location, sumo_rotation, shape, sumo_offset
                )
                if uses_ackermann_physics
                else sumo_to_carla(sumo_location, sumo_rotation, shape, sumo_offset)
            )
            sumo_offset_correct = self._get_carla_offset(sumo_location, 0.0)
            carla_trasform = (
                self._sumo_front_to_carla_transform(
                    sumo_location, sumo_rotation, shape, sumo_offset_correct
                )
                if uses_ackermann_physics
                else sumo_to_carla(
                    sumo_location, sumo_rotation, shape, sumo_offset_correct
                )
            )
            if self._queue_actor_spawn(
                spawn_requests,
                {
                    "actor_type": "vehicle",
                    "actor_id": veh_id,
                    "actor_info": veh_info,
                    "blueprint": blueprint,
                    "spawn_transform": spawn_transform,
                    "post_spawn_transform": carla_trasform,
                    "sumo_location": sumo_location,
                    "sumo_rotation": sumo_rotation,
                    "shape": shape,
                    "post_spawn_offset": sumo_offset_correct,
                    "current_frame": current_frame,
                    "actor_index": actor_index,
                    "sumo_angle": sumo_angle,
                    "enable_physics_after_spawn": uses_ackermann_physics,
                },
            ):
                return
            carla_id = spawn_actor(
                self.client,
                blueprint,
                spawn_transform,
                world=self.world,
                actor_role=veh_id,
            )
            if carla_id > 0:
                self._clear_spawn_failure("vehicle", veh_id)
                vehicle = self.world.get_actor(carla_id)
                if vehicle is None:
                    self._record_spawn_failure("vehicle", veh_id, sumo_location, current_frame)
                    return
                if actor_index is not None:
                    actor_index[veh_id] = vehicle
                self._ensure_collision_sensor(vehicle, veh_id)
                if uses_ackermann_physics:
                    self._ackermann_actor_state.pop(veh_id, None)
                    actor_state = self._ackermann_actor_state.setdefault(veh_id, {})
                    self._initialize_ackermann_actor_geometry(
                        vehicle, veh_id, actor_state
                    )
                    carla_trasform = self._sumo_front_to_carla_transform(
                        sumo_location,
                        sumo_rotation,
                        shape,
                        sumo_offset_correct,
                        actor_state.get("front_bumper_local_x_m"),
                    )
                    self._prepare_ackermann_actor_physics(
                        vehicle,
                        veh_id,
                        veh_info.get("speed"),
                        carla_trasform,
                        spawn_transform,
                    )
                else:
                    # Immediately set the correct road-level transform.
                    vehicle.set_transform(carla_trasform)
            else:
                self._record_spawn_failure("vehicle", veh_id, sumo_location, current_frame)
        else:
            sumo_offset = self._get_carla_offset(sumo_location, 0.0)
            carla_trasform = (
                self._sumo_front_to_carla_transform(
                    sumo_location, sumo_rotation, shape, sumo_offset
                )
                if uses_ackermann_physics
                else sumo_to_carla(sumo_location, sumo_rotation, shape, sumo_offset)
            )
            if uses_ackermann_physics:
                physics_ready = self._ensure_ackermann_actor_physics(
                    vehicle,
                    veh_id,
                    veh_info.get("speed"),
                    carla_trasform,
                )
                if not physics_ready:
                    return
                # Non-master pipelines can reach actor processing before their
                # first feedback is queued, so preserve their one-frame gate.
                # Master applies state before its CARLA tick and then collects
                # feedback, matching the feature pipeline without this delay.
                if self._waits_for_first_phase_a_feedback(veh_id):
                    return
                actor_state = self._ackermann_actor_state.setdefault(veh_id, {})
                carla_trasform = self._sumo_front_to_carla_transform(
                    sumo_location,
                    sumo_rotation,
                    shape,
                    sumo_offset,
                    actor_state.get("front_bumper_local_x_m"),
                )
                control = self._build_ackermann_control(
                    veh_id,
                    veh_info,
                    vehicle,
                    sumo_location,
                    sumo_angle,
                    carla_trasform,
                )
                self._queue_actor_ackermann_control(
                    vehicle,
                    control,
                    ackermann_batch,
                    direct_vehicle_control=(
                        actor_state.get("control_mode")
                        in {
                            "emergency_brake",
                            "fail_closed_brake",
                        }
                    ),
                )
            else:
                self._ensure_actor_teleport_mode(vehicle, veh_id)
                self._queue_actor_transform(vehicle, carla_trasform, transform_batch)

    def _process_vru(
        self,
        vru_id,
        vru_info,
        cosim_id_record,
        carla_actor=None,
        actor_index=None,
        current_frame=None,
        transform_batch=None,
        spawn_requests=None,
    ):
        """Process a pedestrian actor."""
        cosim_id_record.add(vru_id)

        sumo_location = self._resolve_sumo_location("vru", vru_id, vru_info)
        if sumo_location is None:
            return
        sumo_angle = self._resolve_sumo_angle("vru", vru_id, vru_info, carla_actor)
        if sumo_angle is None:
            return
        sumo_rotation = [0.0, sumo_angle, 0.0]
        shape = [vru_info["length"], vru_info["width"], vru_info["height"]]

        pedestrian = carla_actor
        carla_id = pedestrian.id if pedestrian is not None else -1
        if pedestrian is None:
            if current_frame is None:
                current_frame = self.world.get_snapshot().frame
            if not self._should_retry_spawn("vru", vru_id, sumo_location, current_frame):
                return
            blueprint = self._select_vru_blueprint(vru_id, vru_info)
            # Spawn elevated to avoid collision with road geometry
            sumo_offset = self._get_carla_offset(sumo_location, self.spawn_z_clearance)
            spawn_transform = sumo_to_carla(sumo_location, sumo_rotation, shape, sumo_offset)
            z_off = 0.0 if "BIKE" in vru_info["type"] else shape[2] / 2.0
            sumo_offset_correct = self._get_carla_offset(sumo_location, z_off)
            carla_trasform = sumo_to_carla(sumo_location, sumo_rotation, shape, sumo_offset_correct)
            if self._queue_actor_spawn(
                spawn_requests,
                {
                    "actor_type": "vru",
                    "actor_id": vru_id,
                    "actor_info": vru_info,
                    "blueprint": blueprint,
                    "spawn_transform": spawn_transform,
                    "post_spawn_transform": carla_trasform,
                    "sumo_location": sumo_location,
                    "current_frame": current_frame,
                    "actor_index": actor_index,
                    "sumo_angle": sumo_angle,
                },
            ):
                return
            carla_id = spawn_actor(
                self.client,
                blueprint,
                spawn_transform,
                world=self.world,
                actor_role=vru_id,
            )
            if carla_id > 0:
                self._clear_spawn_failure("vru", vru_id)
                pedestrian = self.world.get_actor(carla_id)
                if pedestrian is None:
                    self._record_spawn_failure("vru", vru_id, sumo_location, current_frame)
                    return
                if actor_index is not None:
                    actor_index[vru_id] = pedestrian
                pedestrian.set_transform(carla_trasform)
            else:
                self._record_spawn_failure("vru", vru_id, sumo_location, current_frame)
        else:
            z_off = 0.0 if "BIKE" in vru_info["type"] else shape[2] / 2.0
            sumo_offset = self._get_carla_offset(sumo_location, z_off)
            carla_trasform = sumo_to_carla(sumo_location, sumo_rotation, shape, sumo_offset)
            self._queue_actor_transform(pedestrian, carla_trasform, transform_batch, cosim_id=vru_id)

        if carla_id > 0:
            self._apply_vru_walker_control(vru_info, sumo_angle, pedestrian)

    def close(self):
        """
        Cleans synchronization and resets the simulation settings.
        """
        self._shutdown_collision_sensors()

        # In all three tick modes (follow, master AND async) the ego side is
        # still alive when this process exits, so the world settings and the
        # ego are preserved. In follow mode the bridge owns synchronous_mode;
        # in master mode nobody ticks after us and the operating rule is the
        # same as a dead bridge today: restart CARLA before the next run; in
        # async mode the world just keeps free-running.
        preserve_world = (
            getattr(self.args, "passive_tick", False)
            or self.tick_master
            or self.tick_async
        )
        if not preserve_world:
            # Configuring carla simulation in async mode.
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        # Destroy actors. In 3-cosim, keep ego (protected_roles) and clear only the
        # SUMO-spawned background vehicles/pedestrians; otherwise destroy everything.
        if preserve_world:
            protected = getattr(self.args, "protected_roles", None) or ["AV"]
            for actor in self.world.get_actors().filter("vehicle.*"):
                if actor.attributes.get("role_name") not in protected:
                    actor.destroy()
            for actor in self.world.get_actors().filter("walker.*"):
                actor.destroy()
        else:
            destroy_all_actors(self.world)

        # stop TeraSim
        self.inprocess_plugin.request_stop()
