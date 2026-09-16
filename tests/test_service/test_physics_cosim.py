import json
import tempfile
import unittest
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import patch

from terasim_service.plugins import cosim_inprocess
from terasim_service.plugins.cosim_inprocess import TeraSimCoSimInProcessPlugin
from terasim_service.utils import AgentCommand
from terasim_service.utils import base as service_base
from terasim_service.utils.carla import cosim as carla_cosim
from terasim_service.utils.carla.ackermann_control import (
    AckermannTuning,
    compute_ackermann_control_values,
    compute_direct_brake_value,
)
from terasim_service.utils.carla.cosim import CarlaCosim
from terasim_service.utils.sumo_lane_geometry import (
    build_external_state_lateral_action_lookahead,
    compile_lane_shapes,
)


class _FakeVehicle:
    def __init__(self, calls):
        self.calls = calls
        self.position = (10.0, 1.0)
        self.angle = 90.0
        self.speed = 4.0
        self.acceleration = 0.5

    def getLaneID(self, actor_id):
        return "edge_0_0"

    def getRoute(self, actor_id):
        return ("edge_0", "edge_1")

    def moveToXYImmediate(
        self,
        actor_id,
        edge_id,
        lane_index,
        x,
        y,
        angle,
        keep_route,
        threshold,
        strict_lane,
    ):
        self.calls.append(("move", actor_id, edge_id, lane_index, strict_lane))
        self.position = (x, y)
        self.angle = angle

    def setSpeed(self, actor_id, speed):
        self.calls.append(("setSpeed", speed))

    def setPreviousSpeed(self, actor_id, speed, acceleration):
        self.calls.append(("setPreviousSpeed", speed, acceleration))
        self.speed = speed
        self.acceleration = acceleration

    def getPosition(self, actor_id):
        return self.position

    def getAngle(self, actor_id):
        return self.angle

    def getSpeed(self, actor_id):
        return self.speed

    def getAcceleration(self, actor_id):
        return self.acceleration

    def getLanePosition(self, actor_id):
        return self.position[0]


class _FakeLane:
    @staticmethod
    def getEdgeID(lane_id):
        return "edge_0"

    @staticmethod
    def getShape(lane_id):
        return ((0.0, 1.0), (100.0, 1.0))

    @staticmethod
    def getLength(lane_id):
        return 100.0


class PhysicalCosimTest(unittest.TestCase):
    def test_initial_inprocess_state_exports_sumo_road_slope(self):
        fake_vehicle = SimpleNamespace(
            getPosition3D=lambda _actor_id: (10.0, 20.0, 1.0),
            getAngle=lambda _actor_id: 90.0,
            getSlope=lambda _actor_id: 3.25,
            getLength=lambda _actor_id: 4.8,
            getWidth=lambda _actor_id: 1.8,
            getHeight=lambda _actor_id: 1.5,
            getTypeID=lambda _actor_id: "vehicle_passenger",
            getSpeed=lambda _actor_id: 5.0,
            getAcceleration=lambda _actor_id: 0.25,
            getIDList=lambda: ("vehicle0",),
        )
        fake_traci = SimpleNamespace(
            vehicle=fake_vehicle,
            person=SimpleNamespace(getIDList=lambda: ()),
            simulation=SimpleNamespace(getTime=lambda: 500.05),
        )

        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.get_vehicle_vru_ids = lambda: (("vehicle0",), (), ())
        plugin._filter_vehicle_ids_for_state = lambda vehicle_ids: (vehicle_ids, {})
        plugin._cache_prune_countdown = 1
        plugin.last_orientations = {}
        plugin._static_attr_cache = {}
        plugin.state_lonlat_enabled = False
        plugin.lane_relative_position_enabled = False
        plugin._populate_physics_action_state = lambda _actor_id, _state: None
        plugin._tls_static_cache = {}
        plugin.construction_zone_shapes = {}

        with patch.object(cosim_inprocess, "traci", fake_traci):
            state = plugin._build_simulation_state(SimpleNamespace())

        self.assertEqual(
            state.agent_details["vehicle"]["vehicle0"]["sumo_slope"],
            3.25,
        )

    def test_inprocess_lane_reconstruction_matches_feature_inputs(self):
        fake_traci = SimpleNamespace(
            vehicle=SimpleNamespace(
                getLaneID=lambda _actor_id: "edge_0_0",
                getLanePosition=lambda _actor_id: 50.0,
                getLateralLanePosition=lambda _actor_id: 1.0,
            ),
            lane=SimpleNamespace(
                getShape=lambda _lane_id: ((0.0, 0.0), (80.0, 0.0)),
                getLength=lambda _lane_id: 100.0,
            ),
        )
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.lane_relative_position_enabled = True
        state = {"x": 40.0, "y": -1.0, "z": 2.0}

        with patch.object(cosim_inprocess, "traci", fake_traci):
            plugin._populate_lane_relative_position("vehicle0", state)

        self.assertEqual(state["lane_id"], "edge_0_0")
        self.assertAlmostEqual(state["reconstructed_x"], 40.0)
        self.assertAlmostEqual(state["reconstructed_y"], -1.0)
        self.assertAlmostEqual(state["reconstructed_z"], 2.0)
        self.assertTrue(state["reconstructed_position_valid"])

    def test_ackermann_steer_is_rate_limited_toward_lookahead(self):
        values = compute_ackermann_control_values(
            current_x=0.0,
            current_y=0.0,
            yaw_degrees=0.0,
            current_speed=5.0,
            desired_x=0.25,
            desired_y=0.0,
            lookahead_x=7.0,
            lookahead_y=2.0,
            desired_speed=5.5,
            previous_steer=0.0,
            dt=0.05,
            tuning=AckermannTuning(max_steer_rate_rad_s=0.6),
        )
        self.assertGreater(values.steer, 0.0)
        self.assertLessEqual(values.steer, 0.03 + 1e-12)
        self.assertGreater(values.lookahead_local_x, 0.0)

    def test_direct_brake_mapping_has_minimum_and_saturates(self):
        self.assertAlmostEqual(
            compute_direct_brake_value(-1.0, 8.0, 0.5), 0.5
        )
        self.assertAlmostEqual(
            compute_direct_brake_value(-20.0, 8.0, 0.5), 1.0
        )

    def test_lateral_action_lookahead_uses_sumo_lateral_velocity(self):
        path = compile_lane_shapes([((0.0, 0.0), (20.0, 0.0))])
        result = build_external_state_lateral_action_lookahead(
            path,
            (0.0, 0.0),
            7.0,
            lateral_speed=1.0,
            desired_speed=5.0,
            phase_a_position=(0.0, -0.05),
            phase_step_length=0.05,
            z=0.0,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["mode"], "sumo_lateral_velocity")
        self.assertAlmostEqual(result["lookahead"][0], 7.0)
        self.assertAlmostEqual(result["lookahead"][1], 1.4)
        self.assertAlmostEqual(result["lateral_displacement"], 1.4)
        self.assertAlmostEqual(result["phase_b_lateral_delta"], 0.05)
        self.assertAlmostEqual(result["world_lateral_speed"], 1.0)

    def test_lateral_action_uses_world_motion_not_lateral_speed_sign(self):
        path = compile_lane_shapes([((0.0, 0.0), (20.0, 0.0))])
        result = build_external_state_lateral_action_lookahead(
            path,
            (0.0, 0.0),
            7.0,
            lateral_speed=-2.0,
            desired_speed=5.0,
            phase_a_position=(0.0, -0.1),
            phase_step_length=0.05,
        )
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["lateral_displacement"], 2.8)
        self.assertAlmostEqual(result["world_lateral_speed"], 2.0)

    def test_unresolved_lateral_delta_returns_valid_route_only_lookahead(self):
        path = compile_lane_shapes([((0.0, 0.0), (20.0, 0.0))])
        result = build_external_state_lateral_action_lookahead(
            path,
            (1.0, 0.0),
            7.0,
            lateral_speed=0.95,
            desired_speed=5.0,
            phase_a_position=(0.75, 0.0),
            phase_step_length=0.05,
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["error"], "")
        self.assertEqual(result["warning"], "unresolved_phase_b_lateral_direction")
        self.assertEqual(result["mode"], "route_only_unresolved_lateral")
        self.assertEqual(result["lateral_direction_source"], "route_only")
        self.assertEqual(result["lookahead"], result["route_lookahead"])
        self.assertAlmostEqual(result["lateral_displacement"], 0.0)
        self.assertAlmostEqual(result["world_lateral_speed"], 0.0)

    def test_unresolved_lateral_delta_reuses_aligned_previous_direction(self):
        path = compile_lane_shapes([((0.0, 0.0), (20.0, 0.0))])
        result = build_external_state_lateral_action_lookahead(
            path,
            (1.0, 0.0),
            7.0,
            lateral_speed=-1.0,
            desired_speed=5.0,
            phase_a_position=(0.75, 0.0),
            phase_step_length=0.05,
            previous_world_lateral_direction=(0.0, 1.0),
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["warning"], "")
        self.assertEqual(result["lateral_direction_source"], "previous_confirmed")
        self.assertAlmostEqual(result["world_lateral_speed"], 1.0)
        self.assertAlmostEqual(result["lookahead"][1], 1.4)

    def test_unresolved_lateral_delta_rejects_unaligned_previous_direction(self):
        path = compile_lane_shapes([((0.0, 0.0), (20.0, 0.0))])
        result = build_external_state_lateral_action_lookahead(
            path,
            (1.0, 0.0),
            7.0,
            lateral_speed=1.0,
            desired_speed=5.0,
            phase_a_position=(0.75, 0.0),
            phase_step_length=0.05,
            previous_world_lateral_direction=(1.0, 0.0),
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["lateral_direction_source"], "route_only")
        self.assertEqual(result["warning"], "unresolved_phase_b_lateral_direction")

    def test_invalid_route_or_missing_phase_a_remains_fail_closed(self):
        path = compile_lane_shapes([((0.0, 0.0), (20.0, 0.0))])
        missing_route = build_external_state_lateral_action_lookahead(
            None,
            (1.0, 0.0),
            7.0,
            lateral_speed=1.0,
            desired_speed=5.0,
            phase_a_position=(1.0, 0.0),
        )
        missing_phase_a = build_external_state_lateral_action_lookahead(
            path,
            (1.0, 0.0),
            7.0,
            lateral_speed=1.0,
            desired_speed=5.0,
            phase_a_position=None,
        )
        non_finite_position = build_external_state_lateral_action_lookahead(
            path,
            (float("nan"), 0.0),
            7.0,
            lateral_speed=1.0,
            desired_speed=5.0,
            phase_a_position=(1.0, 0.0),
        )

        self.assertFalse(missing_route["valid"])
        self.assertEqual(missing_route["error"], "missing_route_geometry")
        self.assertFalse(missing_phase_a["valid"])
        self.assertEqual(
            missing_phase_a["error"], "missing_phase_a_world_position"
        )
        self.assertFalse(non_finite_position["valid"])
        self.assertEqual(non_finite_position["error"], "non_finite_route_input")

    def test_phase_a_immediate_move_does_not_advance_sumo_time(self):
        calls = []
        fake_vehicle = _FakeVehicle(calls)
        fake_traci = SimpleNamespace(
            vehicle=fake_vehicle,
            lane=_FakeLane(),
            simulation=SimpleNamespace(getTime=lambda: 12.5),
        )

        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.physics_lane_geometry_cache = {}
        plugin.physics_match_threshold = 8.0
        plugin.physics_strict_lane_hint = True
        plugin.physics_validate_external_state = True
        plugin.physics_position_tolerance = 0.001
        plugin.physics_feedback_observations = {}

        command = AgentCommand(
            agent_id="vehicle0",
            agent_type="vehicle",
            command_type="set_state",
            data={
                "position": [10.0, 1.0],
                "z": 0.0,
                "sumo_angle": 90.0,
                "speed": 4.0,
                "acceleration": 0.5,
                "source_carla_frame": 123,
            },
        )
        with patch.object(cosim_inprocess, "traci", fake_traci):
            self.assertTrue(
                plugin._apply_physics_external_state(command, 10.0, 1.0)
            )
        self.assertEqual(
            [call[0] for call in calls],
            ["move", "setSpeed", "setPreviousSpeed"],
        )
        observation = plugin.physics_feedback_observations["vehicle0"]
        self.assertAlmostEqual(observation["phase_a_time"], 12.5)
        self.assertEqual(observation["source_carla_frame"], 123)
        self.assertEqual(observation["requested_lane_id"], "edge_0_0")

    def test_phase_b_adapter_exports_live_route_and_world_lateral_motion(self):
        fake_vehicle = SimpleNamespace(
            getSpeedWithoutTraCI=lambda _actor_id: 5.0,
            getEmergencyDecel=lambda _actor_id: 8.0,
            getLaneID=lambda _actor_id: "edge_current_0",
            getPosition=lambda _actor_id: (10.0, 0.0),
            getLanePosition=lambda _actor_id: 10.0,
            getRoute=lambda _actor_id: ("edge_current", "edge_next"),
            getSlope=lambda _actor_id: 0.0,
            getLaneChangeState=lambda _actor_id, _direction: (0, 0),
            getLateralSpeed=lambda _actor_id: -1.0,
            getNextLinks=lambda _actor_id: (),
        )
        fake_lane = SimpleNamespace(
            getEdgeID=lambda _lane_id: "edge_current",
            getShape=lambda _lane_id: ((0.0, 0.0), (100.0, 0.0)),
            getLength=lambda _lane_id: 100.0,
        )
        fake_traci = SimpleNamespace(
            vehicle=fake_vehicle,
            lane=fake_lane,
            simulation=SimpleNamespace(getTime=lambda: 12.55),
            constants=SimpleNamespace(LCA_LEFT=1, LCA_RIGHT=2),
        )

        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.physics_external_state_enabled = True
        plugin.physics_feedback_actor_ids = {"*"}
        plugin.physics_step_length = 0.05
        plugin.physics_lookahead_min_distance = 7.0
        plugin.physics_lookahead_max_distance = 15.0
        plugin.physics_lane_geometry_cache = {}
        plugin.physics_lookahead_path_cache = {}
        plugin.physics_feedback_observations = {
            "vehicle0": {
                "position": (9.75, -0.05),
                "sumo_angle": 90.0,
                "speed": 5.0,
                "acceleration": 0.0,
                "requested_lane_id": "edge_requested_0",
                "lane_id": "edge_current_0",
                "lane_position": 9.75,
                "lane_length": 100.0,
                "phase_a_time": 12.5,
                "source_carla_frame": 77,
            }
        }
        vehicle_state = {
            "x": 999.0,
            "y": 999.0,
            "z": 0.0,
            "speed": 5.0,
            "lane_id": "stale_0",
            "lane_position": 0.0,
        }

        with patch.object(cosim_inprocess, "traci", fake_traci):
            plugin._populate_physics_action_state("vehicle0", vehicle_state)

        self.assertEqual(
            vehicle_state["feedback_requested_lane_id"], "edge_requested_0"
        )
        self.assertEqual(
            vehicle_state["feedback_observed_lane_id"], "edge_current_0"
        )
        self.assertEqual(vehicle_state["lane_id"], "edge_current_0")
        self.assertEqual(vehicle_state["sumo_route"], ("edge_current", "edge_next"))
        self.assertAlmostEqual(vehicle_state["feedback_longitudinal_error"], 0.0)
        self.assertAlmostEqual(vehicle_state["lookahead_origin_x"], 10.0)
        self.assertAlmostEqual(vehicle_state["lookahead_origin_y"], 0.0)
        self.assertAlmostEqual(
            vehicle_state["lookahead_phase_b_lateral_delta"], 0.05
        )
        self.assertAlmostEqual(vehicle_state["lookahead_world_lateral_speed"], 1.0)
        self.assertTrue(vehicle_state["lookahead_position_valid"])

    def test_phase_b_lane_transition_reuses_only_immediately_confirmed_direction(self):
        phase_b_time = {"value": 12.55}
        lateral_speed = {"value": 0.95}
        fake_vehicle = SimpleNamespace(
            getSpeedWithoutTraCI=lambda _actor_id: 5.0,
            getEmergencyDecel=lambda _actor_id: 8.0,
            getLaneID=lambda _actor_id: "edge_125_0",
            getPosition=lambda _actor_id: (10.0, 0.0),
            getLanePosition=lambda _actor_id: 10.0,
            getRoute=lambda _actor_id: ("edge_125", "edge_next"),
            getSlope=lambda _actor_id: 0.0,
            getLaneChangeState=lambda _actor_id, direction: (
                (0, 0) if direction == 1 else (0, 2)
            ),
            getLateralSpeed=lambda _actor_id: lateral_speed["value"],
            getNextLinks=lambda _actor_id: (),
        )
        fake_lane = SimpleNamespace(
            getEdgeID=lambda _lane_id: "edge_125",
            getShape=lambda _lane_id: ((0.0, 0.0), (100.0, 0.0)),
            getLength=lambda _lane_id: 100.0,
        )
        fake_traci = SimpleNamespace(
            vehicle=fake_vehicle,
            lane=fake_lane,
            simulation=SimpleNamespace(getTime=lambda: phase_b_time["value"]),
            constants=SimpleNamespace(LCA_LEFT=1, LCA_RIGHT=2),
        )
        warnings = []
        recoveries = []

        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.logger = SimpleNamespace(
            warning=lambda *args: warnings.append(args),
            info=lambda *args: recoveries.append(args),
        )
        plugin.physics_external_state_enabled = True
        plugin.physics_feedback_actor_ids = {"*"}
        plugin.physics_step_length = 0.05
        plugin.physics_lookahead_min_distance = 7.0
        plugin.physics_lookahead_max_distance = 15.0
        plugin.physics_lane_geometry_cache = {}
        plugin.physics_lookahead_path_cache = {}
        plugin.physics_lateral_direction_states = {
            "vehicle1593": {
                "confirmed_world_direction": (0.0, 1.0),
                "confirmed_sumo_time": 12.5,
                "confirmed_lateral_speed": 0.95,
                "unresolved_count": 0,
            }
        }
        plugin.physics_feedback_observations = {
            "vehicle1593": {
                "position": (9.75, 0.0),
                "sumo_angle": 90.0,
                "speed": 5.0,
                "acceleration": 0.0,
                "requested_lane_id": "edge_125_1",
                "lane_id": "edge_125_1",
                "lane_position": 9.75,
                "lane_length": 100.0,
                "phase_a_time": 12.5,
                "source_carla_frame": 77,
            }
        }

        def vehicle_state():
            return {
                "x": 10.0,
                "y": 0.0,
                "z": 0.0,
                "speed": 5.0,
                "lane_id": "edge_125_0",
                "lane_position": 10.0,
            }

        with patch.object(cosim_inprocess, "traci", fake_traci):
            first = vehicle_state()
            plugin._populate_physics_action_state("vehicle1593", first)
            self.assertTrue(first["lookahead_action_valid"])
            self.assertEqual(
                first["lookahead_lateral_direction_source"], "previous_confirmed"
            )
            self.assertAlmostEqual(first["lookahead_world_lateral_speed"], 0.95)
            self.assertTrue(first["lookahead_lane_change_intent_conflict"])
            self.assertEqual(
                plugin.physics_lateral_direction_states["vehicle1593"][
                    "confirmed_sumo_time"
                ],
                12.5,
            )

            phase_b_time["value"] = 12.6
            plugin.physics_feedback_observations["vehicle1593"][
                "phase_a_time"
            ] = 12.55
            second = vehicle_state()
            plugin._populate_physics_action_state("vehicle1593", second)
            self.assertTrue(second["lookahead_action_valid"])
            self.assertEqual(second["lookahead_action_error"], "")
            self.assertEqual(
                second["lookahead_action_warning"],
                "unresolved_phase_b_lateral_direction",
            )
            self.assertEqual(second["lookahead_lateral_direction_source"], "route_only")
            self.assertEqual(
                second["lookahead_lateral_direction_unresolved_count"], 1
            )
            self.assertEqual(second["lookahead_x"], second["lookahead_route_x"])
            self.assertEqual(second["lookahead_y"], second["lookahead_route_y"])

            phase_b_time["value"] = 12.65
            plugin.physics_feedback_observations["vehicle1593"][
                "phase_a_time"
            ] = 12.6
            third = vehicle_state()
            plugin._populate_physics_action_state("vehicle1593", third)
            self.assertTrue(third["lookahead_action_valid"])
            self.assertEqual(
                third["lookahead_lateral_direction_unresolved_count"], 2
            )
            self.assertEqual(len(warnings), 1)

            lateral_speed["value"] = 0.0
            phase_b_time["value"] = 12.7
            plugin.physics_feedback_observations["vehicle1593"][
                "phase_a_time"
            ] = 12.65
            inactive = vehicle_state()
            plugin._populate_physics_action_state("vehicle1593", inactive)
            self.assertEqual(inactive["lookahead_lateral_direction_source"], "inactive")
            self.assertNotIn("vehicle1593", plugin.physics_lateral_direction_states)
            self.assertEqual(len(recoveries), 1)

    def test_departed_actor_lateral_direction_history_is_pruned_immediately(self):
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.physics_lateral_direction_states = {
            "active": {"confirmed_sumo_time": 12.5},
            "departed": {"confirmed_sumo_time": 12.5},
        }

        plugin._prune_departed_physics_lateral_directions(["active"])

        self.assertEqual(
            plugin.physics_lateral_direction_states,
            {"active": {"confirmed_sumo_time": 12.5}},
        )

    def test_unresolved_lateral_warning_is_rate_limited(self):
        warnings = []
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.logger = SimpleNamespace(
            warning=lambda *args: warnings.append(args),
        )
        plugin.physics_lateral_direction_states = {}
        state = {}

        for previous_count in range(100):
            unresolved_count, conflict = (
                plugin._record_unresolved_physics_lateral_direction(
                    plugin.physics_lateral_direction_states,
                    state,
                    "vehicle1593",
                    previous_count,
                    0.95,
                    "right",
                )
            )
            self.assertEqual(unresolved_count, previous_count + 1)
            self.assertFalse(conflict)

        self.assertEqual(len(warnings), 2)
        self.assertEqual(warnings[0][2], 1)
        self.assertEqual(warnings[1][2], 100)

    def test_route_only_lateral_warning_is_not_authoritative_action_error(self):
        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_feedback_apply_enabled = True
        cosim.ackermann_feedback_actor_ids = {"*"}
        location = cosim._resolve_sumo_lookahead_location(
            "vehicle1593",
            {
                "lookahead_action_valid": True,
                "lookahead_action_warning": "unresolved_phase_b_lateral_direction",
                "lookahead_action_mode": "route_only_unresolved_lateral",
                "lookahead_position_valid": True,
                "lookahead_x": 17.0,
                "lookahead_y": 0.0,
                "lookahead_z": 0.0,
            },
            [10.0, 0.0, 0.0],
            90.0,
        )

        self.assertEqual(location, [17.0, 0.0, 0.0])

        with self.assertRaisesRegex(ValueError, "missing authoritative SUMO lookahead"):
            cosim._resolve_sumo_lookahead_location(
                "vehicle1593",
                {
                    "lookahead_action_valid": True,
                    "lookahead_action_warning": (
                        "unresolved_phase_b_lateral_direction"
                    ),
                    "lookahead_action_mode": "route_only_unresolved_lateral",
                    "lookahead_position_valid": False,
                },
                [10.0, 0.0, 0.0],
                90.0,
            )

    def test_create_simulator_passes_optional_step_length(self):
        config = {
            "input": {
                "sumo_net_file": "net.xml",
                "sumo_config_file": "sumo.sumocfg",
            },
            "simulator": {
                "parameters": {
                    "num_tries": 10,
                    "gui_flag": False,
                    "realtime_flag": False,
                    "sumo_output_file_types": [],
                    "sumo_seed": 42,
                    "step_length": 0.05,
                }
            },
        }
        with patch.object(service_base, "Simulator") as simulator:
            service_base.create_simulator(config, "output")
        self.assertEqual(simulator.call_args.kwargs["step_length"], 0.05)

    def test_phase_a_observations_are_cleared_at_each_tick_start(self):
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin._stop_requested = False
        plugin._tick_requested = Event()
        plugin._tick_requested.set()
        plugin._pending_commands = []
        plugin._lock = Lock()
        plugin.controlled_agents_each_step = set()
        plugin.physics_feedback_observations = {
            "filtered_vehicle": {"phase_a_time": 1.0}
        }
        plugin._set_status = lambda status: None
        plugin.logger = SimpleNamespace(debug=lambda message: None)

        self.assertTrue(plugin.function_before_env_step(None, None))
        self.assertEqual(plugin.physics_feedback_observations, {})

    def test_master_tick_matches_feature_pipeline_and_defers_new_result(self):
        events = []
        frames = iter((101, 102))

        class FakeHandle:
            def __init__(self, response):
                self.response = response
                self.result_calls = 0

            def result(self, timeout):
                self.result_calls += 1
                events.append(("previous_result", timeout))
                return self.response

        requested_handles = []

        def request_tick(commands):
            events.append(("request_sumo", commands))
            handle = FakeHandle(
                SimpleNamespace(status="ticked", state={"state": "post_step"})
            )
            requested_handles.append(handle)
            return handle

        cosim = object.__new__(CarlaCosim)
        cosim._inproc_tick_handle = None
        cosim._inproc_prev_state = {"state": "initial"}
        cosim._next_tick_deadline = None
        cosim.step_length = 0.05
        cosim.args = SimpleNamespace(skip_tls=True)
        cosim.control_av = True
        cosim.inprocess_plugin = SimpleNamespace(tick_async=request_tick)
        cosim.world = SimpleNamespace(
            tick=lambda: events.append("carla_tick") or next(frames)
        )
        cosim.sync_cosim_actor_to_carla = lambda state: events.append(
            ("apply_state", state)
        )
        cosim._build_av_command = lambda: events.append(
            ("build_av", cosim._last_world_frame)
        ) or {"agent_id": "AV", "source_carla_frame": cosim._last_world_frame}
        cosim._build_physics_feedback_commands = lambda: events.append(
            ("build_feedback", cosim._last_world_frame)
        ) or [
            {
                "agent_id": "vehicle0",
                "source_carla_frame": cosim._last_world_frame,
            }
        ]
        cosim._apply_physics_fail_closed_brake = lambda reason: events.append(
            ("brake", reason)
        )
        cosim._tick_times_ms = []
        cosim._tick_time_hist = [0] * 11
        cosim._tick_veh_min = None
        cosim._tick_veh_max = None
        cosim._vehicle_actor_index = {}

        self.assertTrue(cosim._tick_master())
        first_handle = requested_handles[0]
        self.assertEqual(first_handle.result_calls, 0)
        self.assertEqual(
            events,
            [
                ("apply_state", {"state": "initial"}),
                "carla_tick",
                ("build_av", 101),
                ("build_feedback", 101),
                (
                    "request_sumo",
                    [
                        {"agent_id": "AV", "source_carla_frame": 101},
                        {"agent_id": "vehicle0", "source_carla_frame": 101},
                    ],
                ),
            ],
        )

        events.clear()
        self.assertTrue(cosim._tick_master())
        self.assertEqual(first_handle.result_calls, 1)
        self.assertEqual(requested_handles[1].result_calls, 0)
        self.assertEqual(
            events,
            [
                ("previous_result", 300.0),
                ("apply_state", {"state": "post_step"}),
                "carla_tick",
                ("build_av", 102),
                ("build_feedback", 102),
                (
                    "request_sumo",
                    [
                        {"agent_id": "AV", "source_carla_frame": 102},
                        {"agent_id": "vehicle0", "source_carla_frame": 102},
                    ],
                ),
            ],
        )

    def test_master_does_not_delay_first_ackermann_control_for_phase_a_feedback(self):
        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_feedback_apply_enabled = True
        cosim.ackermann_feedback_actor_ids = {"*"}
        cosim._physics_feedback_frames = {}

        cosim.tick_master = True
        self.assertFalse(cosim._waits_for_first_phase_a_feedback("vehicle0"))

        cosim.tick_master = False
        self.assertTrue(cosim._waits_for_first_phase_a_feedback("vehicle0"))

        cosim._physics_feedback_frames["vehicle0"] = 101
        self.assertFalse(cosim._waits_for_first_phase_a_feedback("vehicle0"))

    def test_physics_feedback_commands_use_feature_actor_id_order(self):
        collected_actor_ids = []
        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_feedback_apply_enabled = True
        cosim._inproc_prev_state = {
            "agent_details": {
                "vehicle": {
                    "vehicle9": {"speed": 9.0},
                    "AV": {"speed": 0.0},
                    "vehicle10": {"speed": 10.0},
                    "vehicle2": {"speed": 2.0},
                }
            }
        }
        cosim._last_world_frame = 123
        cosim._vehicle_actor_index = {
            actor_id: SimpleNamespace(id=actor_id)
            for actor_id in ("vehicle9", "vehicle10", "vehicle2")
        }
        cosim._ackermann_actor_state = {}
        cosim._physics_feedback_frames = {}
        cosim._ackermann_feedback_state = {}
        cosim._is_ackermann_feedback_actor = lambda actor_id: actor_id != "AV"

        def collect(actor_id, _actor, vehicle_info):
            collected_actor_ids.append(actor_id)
            return {"speed": vehicle_info["speed"]}

        cosim._carla_actor_to_sumo_feedback = collect
        cosim._clear_physics_feedback_failure = lambda _actor_id: None

        commands = cosim._build_physics_feedback_commands()

        self.assertEqual(
            collected_actor_ids,
            ["vehicle10", "vehicle2", "vehicle9"],
        )
        self.assertEqual(
            [command["agent_id"] for command in commands],
            ["vehicle10", "vehicle2", "vehicle9"],
        )
        self.assertTrue(
            all(command["data"]["source_carla_frame"] == 123 for command in commands)
        )

    def test_master_previous_step_failure_brakes_before_advancing_carla(self):
        events = []

        class FailedHandle:
            @staticmethod
            def result(timeout):
                events.append(("previous_result", timeout))
                raise TimeoutError("SUMO did not finish")

        cosim = object.__new__(CarlaCosim)
        cosim._inproc_tick_handle = FailedHandle()
        cosim._inproc_prev_state = {"state": "stale"}
        cosim._apply_physics_fail_closed_brake = lambda reason: events.append(
            ("brake", reason)
        )
        cosim.sync_cosim_actor_to_carla = lambda _state: events.append("apply_state")
        cosim.world = SimpleNamespace(tick=lambda: events.append("carla_tick"))
        cosim.inprocess_plugin = SimpleNamespace(
            tick_async=lambda _commands: events.append("request_sumo")
        )

        self.assertFalse(cosim._tick_master())
        self.assertEqual(
            events,
            [
                ("previous_result", 300.0),
                ("brake", "inprocess_tick_error:TimeoutError"),
            ],
        )

    def test_master_ended_step_brakes_before_advancing_carla(self):
        events = []
        handle = SimpleNamespace(
            result=lambda timeout: events.append(("previous_result", timeout))
            or SimpleNamespace(status="error", state=None)
        )
        cosim = object.__new__(CarlaCosim)
        cosim._inproc_tick_handle = handle
        cosim._inproc_prev_state = {"state": "stale"}
        cosim._apply_physics_fail_closed_brake = lambda reason: events.append(
            ("brake", reason)
        )
        cosim.sync_cosim_actor_to_carla = lambda _state: events.append("apply_state")
        cosim.world = SimpleNamespace(tick=lambda: events.append("carla_tick"))

        self.assertFalse(cosim._tick_master())
        self.assertEqual(
            events,
            [("previous_result", 300.0), ("brake", "error")],
        )

    def test_master_request_failure_brakes_after_current_carla_frame(self):
        events = []
        cosim = object.__new__(CarlaCosim)
        cosim._inproc_tick_handle = None
        cosim._inproc_prev_state = {"state": "initial"}
        cosim._next_tick_deadline = None
        cosim.step_length = 0.05
        cosim.args = SimpleNamespace(skip_tls=True)
        cosim.control_av = False
        cosim.world = SimpleNamespace(
            tick=lambda: events.append("carla_tick") or 201
        )
        cosim.sync_cosim_actor_to_carla = lambda state: events.append(
            ("apply_state", state)
        )
        cosim._build_physics_feedback_commands = lambda: events.append(
            ("build_feedback", cosim._last_world_frame)
        ) or []

        def request_tick(_commands):
            events.append("request_sumo")
            raise RuntimeError("request rejected")

        cosim.inprocess_plugin = SimpleNamespace(tick_async=request_tick)
        cosim._apply_physics_fail_closed_brake = lambda reason: events.append(
            ("brake", reason)
        )
        cosim._tick_times_ms = []
        cosim._tick_time_hist = [0] * 11
        cosim._tick_veh_min = None
        cosim._tick_veh_max = None
        cosim._vehicle_actor_index = {}

        self.assertFalse(cosim._tick_master())
        self.assertEqual(
            events,
            [
                ("apply_state", {"state": "initial"}),
                "carla_tick",
                ("build_feedback", 201),
                "request_sumo",
                ("brake", "inprocess_tick_request_error:RuntimeError"),
            ],
        )

    def test_tick_dispatch_keeps_follow_and_async_paths_separate(self):
        cosim = object.__new__(CarlaCosim)
        calls = []
        cosim._tick_master = lambda: calls.append("master") or "master"
        cosim._tick_async = lambda: calls.append("async") or "async"
        cosim._tick_follow = lambda: calls.append("follow") or "follow"

        cosim.tick_master = False
        cosim.tick_async = False
        self.assertEqual(cosim.tick(), "follow")
        cosim.tick_async = True
        self.assertEqual(cosim.tick(), "async")
        cosim.tick_master = True
        self.assertEqual(cosim.tick(), "master")
        self.assertEqual(calls, ["follow", "async", "master"])

    def test_physics_initialization_waits_one_completed_carla_frame(self):
        calls = []

        class FakeActor:
            id = 10

            def __init__(self, transform):
                self.transform = transform
                self.velocity = carla_cosim.carla.Vector3D(0.0, 0.0, 0.0)

            def set_simulate_physics(self, enabled):
                calls.append(("physics", enabled))

            def set_transform(self, transform):
                calls.append(("transform", transform))
                self.transform = transform

            def set_target_velocity(self, velocity):
                calls.append(("velocity", velocity.x, velocity.y, velocity.z))
                self.velocity = velocity

            def set_target_angular_velocity(self, velocity):
                calls.append(("angular_velocity", velocity.x, velocity.y, velocity.z))

            def get_transform(self):
                return self.transform

            def get_velocity(self):
                return self.velocity

            def get_physics_control(self):
                return SimpleNamespace(wheels=[])

            def apply_ackermann_controller_settings(self, settings):
                calls.append(("configure", settings))

        transform = carla_cosim.carla.Transform(
            carla_cosim.carla.Location(x=1.0, y=2.0, z=0.5),
            carla_cosim.carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        elevated = CarlaCosim._transform_with_z_offset(transform, 5.0)
        actor = FakeActor(elevated)
        cosim = object.__new__(CarlaCosim)
        cosim._ackermann_actor_state = {}
        cosim._vehicle_actor_index = {"vehicle0": actor}
        cosim.spawn_max_attempts = 3
        cosim.spawn_z_clearance = 5.0
        cosim.initialization_diagnostics_enabled = False
        cosim.ackermann_tuning = AckermannTuning()
        cosim.ackermann_controller_tuning = SimpleNamespace(
            speed_kp=1.0,
            speed_ki=0.0,
            speed_kd=0.0,
            accel_kp=0.05,
            accel_ki=0.0,
            accel_kd=0.0,
        )
        frame = {"value": 100}
        cosim.world = SimpleNamespace(
            get_snapshot=lambda: SimpleNamespace(frame=frame["value"])
        )

        self.assertFalse(
            cosim._prepare_ackermann_actor_physics(
                actor, "vehicle0", 4.0, transform, elevated
            )
        )
        self.assertFalse(
            cosim._ensure_ackermann_actor_physics(
                actor, "vehicle0", 4.0, transform
            )
        )
        self.assertNotIn(("physics", True), calls)

        frame["value"] = 101
        self.assertFalse(
            cosim._ensure_ackermann_actor_physics(
                actor, "vehicle0", 4.0, transform
            )
        )
        self.assertIn(("physics", True), calls)

        frame["value"] = 102
        self.assertFalse(
            cosim._ensure_ackermann_actor_physics(
                actor, "vehicle0", 4.0, transform
            )
        )
        frame["value"] = 103
        self.assertFalse(
            cosim._ensure_ackermann_actor_physics(
                actor, "vehicle0", 4.0, transform
            )
        )
        frame["value"] = 104
        self.assertTrue(
            cosim._ensure_ackermann_actor_physics(
                actor, "vehicle0", 4.0, transform
            )
        )
        state = cosim._ackermann_actor_state["vehicle0"]
        self.assertFalse(state["physics_initialization_pending"])
        self.assertEqual(state["physics_stable_ticks"], 3)

    def test_feedback_frame_failure_brakes_then_propagates_on_third_failure(self):
        cosim = object.__new__(CarlaCosim)
        cosim._ackermann_actor_state = {}
        cosim._physics_feedback_failures = {}
        cosim._ackermann_feedback_state = {}
        cosim._ackermann_fail_closed_reasons = {}
        cosim._pending_authoritative_action_error = None
        cosim.ackermann_feedback_ack_max_frame_lag = 2
        cosim.ackermann_feedback_ack_failure_limit = 3

        feedback = {"feedback_status": "queued", "source_carla_frame": 100}
        self.assertFalse(cosim._is_ackermann_feedback_healthy("vehicle0", feedback, 99))
        self.assertFalse(cosim._is_ackermann_feedback_healthy("vehicle0", feedback, 99))
        self.assertIsNone(cosim._pending_authoritative_action_error)
        self.assertFalse(cosim._is_ackermann_feedback_healthy("vehicle0", feedback, 99))
        self.assertEqual(
            cosim._pending_authoritative_action_error["actor_id"], "vehicle0"
        )

        calls = []
        cosim._apply_ackermann_fail_closed_brake = calls.append
        with self.assertRaisesRegex(RuntimeError, "feedback_frame_mismatch"):
            cosim._raise_pending_authoritative_action_error()
        self.assertEqual(calls, ["feedback_frame_mismatch"])

    def test_newly_initialized_actor_waits_for_first_phase_a_feedback(self):
        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_feedback_apply_enabled = True
        cosim.ackermann_feedback_actor_ids = {"*"}
        cosim._physics_feedback_frames = {}

        self.assertTrue(cosim._waits_for_first_phase_a_feedback("vehicle0"))
        self.assertFalse(cosim._waits_for_first_phase_a_feedback("AV"))

        cosim._physics_feedback_frames["vehicle0"] = 101
        self.assertFalse(cosim._waits_for_first_phase_a_feedback("vehicle0"))

    def test_restart_target_is_bounded_and_releases_at_feature_threshold(self):
        cosim = object.__new__(CarlaCosim)
        cosim._ackermann_actor_state = {}
        cosim.ackermann_tuning = AckermannTuning()
        cosim.step_length = 0.05
        cosim._is_ackermann_feedback_apply_actor = lambda _actor_id: True
        vehicle_info = {
            "sumo_desired_speed": 0.1,
            "feedback_observed_speed": 0.0,
            "acceleration": 2.0,
            "sumo_emergency_decel": 8.0,
        }

        target, acceleration = cosim._resolve_ackermann_longitudinal_target(
            "vehicle0", vehicle_info, 0.0
        )
        self.assertAlmostEqual(target, 0.1)
        self.assertAlmostEqual(acceleration, 2.0)
        self.assertTrue(cosim._ackermann_actor_state["vehicle0"]["restart_active"])

        for _ in range(10):
            target, _ = cosim._resolve_ackermann_longitudinal_target(
                "vehicle0", vehicle_info, 0.0
            )
        self.assertAlmostEqual(target, 0.3)

        cosim._resolve_ackermann_longitudinal_target(
            "vehicle0", vehicle_info, 0.2
        )
        self.assertFalse(cosim._ackermann_actor_state["vehicle0"]["restart_active"])

    def test_emergency_brake_retains_steer_and_releases_after_hysteresis(self):
        cosim = object.__new__(CarlaCosim)
        cosim._ackermann_actor_state = {
            "vehicle0": {
                "sumo_requested_acceleration": -5.0,
                "sumo_emergency_decel": 8.0,
            }
        }
        cosim.ackermann_tuning = AckermannTuning()
        cosim.ackermann_emergency_brake_tuning = (
            carla_cosim.AckermannEmergencyBrakeTuning()
        )
        cosim.ackermann_feedback_apply_enabled = True
        cosim._is_ackermann_feedback_apply_actor = lambda _actor_id: True

        control = cosim._update_ackermann_emergency_brake(
            "vehicle0", SimpleNamespace(), 5.0, ackermann_steer=0.3
        )
        self.assertIsNotNone(control)
        self.assertAlmostEqual(control.brake, 0.625)
        self.assertAlmostEqual(control.steer, 0.5)

        state = cosim._ackermann_actor_state["vehicle0"]
        state["sumo_requested_acceleration"] = -0.5
        for _ in range(2):
            self.assertIsNotNone(
                cosim._update_ackermann_emergency_brake(
                    "vehicle0", SimpleNamespace(), 5.0, ackermann_steer=0.3
                )
            )
        self.assertIsNone(
            cosim._update_ackermann_emergency_brake(
                "vehicle0", SimpleNamespace(), 5.0, ackermann_steer=0.3
            )
        )

    def test_healthy_phase_aligned_action_builds_ackermann_command(self):
        transform = carla_cosim.carla.Transform(
            carla_cosim.carla.Location(x=0.0, y=0.0, z=0.0),
            carla_cosim.carla.Rotation(yaw=0.0),
        )
        actor = SimpleNamespace(
            get_transform=lambda: transform,
            get_velocity=lambda: carla_cosim.carla.Vector3D(0.0, 0.0, 0.0),
            get_acceleration=lambda: carla_cosim.carla.Vector3D(0.0, 0.0, 0.0),
            get_control=lambda: carla_cosim.carla.VehicleControl(),
        )
        cosim = object.__new__(CarlaCosim)
        cosim._ackermann_actor_state = {
            "vehicle0": {
                "wheel_base_m": 2.8,
                "rear_axle_local_x_m": -1.4,
                "front_bumper_local_x_m": 2.5,
            }
        }
        cosim._ackermann_feedback_state = {
            "vehicle0": {"feedback_status": "queued", "source_carla_frame": 100}
        }
        cosim._physics_feedback_failures = {}
        cosim._ackermann_fail_closed_reasons = {}
        cosim._pending_authoritative_action_error = None
        cosim.ackermann_feedback_ack_max_frame_lag = 2
        cosim.ackermann_feedback_ack_failure_limit = 3
        cosim.ackermann_feedback_apply_enabled = True
        cosim.ackermann_tuning = AckermannTuning()
        cosim.ackermann_emergency_brake_tuning = (
            carla_cosim.AckermannEmergencyBrakeTuning()
        )
        cosim.ackermann_control_log_records = False
        cosim.ackermann_warn_error_m = 3.0
        cosim.ackermann_snap_error_m = 8.0
        cosim.ackermann_warning_interval = 2.0
        cosim.step_length = 0.05
        cosim._is_ackermann_feedback_apply_actor = lambda _actor_id: True
        cosim._sumo_point_to_carla_location = lambda location: (
            carla_cosim.carla.Location(x=location[0], y=location[1], z=location[2])
        )
        vehicle_info = {
            "length": 5.0,
            "sumo_desired_speed": 5.0,
            "feedback_observed_speed": 0.0,
            "feedback_source_carla_frame": 100,
            "feedback_longitudinal_error": 0.0,
            "acceleration": 1.0,
            "sumo_emergency_decel": 8.0,
            "lookahead_action_valid": True,
            "lookahead_position_valid": True,
            "lookahead_x": 10.0,
            "lookahead_y": 0.0,
            "lookahead_z": 0.0,
        }

        control = cosim._build_ackermann_control(
            "vehicle0", vehicle_info, actor, [0.0, 0.0, 0.0], 90.0, transform
        )
        self.assertAlmostEqual(control.speed, 5.0)
        self.assertAlmostEqual(control.acceleration, 1.0)
        self.assertEqual(
            cosim._ackermann_actor_state["vehicle0"]["control_mode"], "ackermann"
        )

    def test_initialization_diagnostic_and_footprint_overlap(self):
        overlapping = {
            "center": (0.0, 0.0),
            "axes": ((1.0, 0.0), (0.0, 1.0)),
            "half_extents": (2.0, 1.0),
            "center_z": 0.5,
            "half_z": 0.5,
        }
        separated = dict(overlapping, center=(10.0, 0.0))
        self.assertTrue(
            CarlaCosim._ackermann_footprints_overlap(overlapping, overlapping)
        )
        self.assertFalse(
            CarlaCosim._ackermann_footprints_overlap(overlapping, separated)
        )

        cosim = object.__new__(CarlaCosim)
        cosim.initialization_diagnostics_enabled = True
        cosim._diagnostic_lock = None
        cosim._initialization_failure_counts = {}
        cosim._current_carla_frame = lambda: 123
        with tempfile.TemporaryDirectory() as directory:
            cosim.initialization_log_path = f"{directory}/initialization.jsonl"
            actor = SimpleNamespace(id=10, type_id="vehicle.test", attributes={})
            cosim._record_initialization_diagnostic(
                "failure", actor, "vehicle0", reason="overlap=vehicle1", attempt=1
            )
            with open(cosim.initialization_log_path, encoding="utf-8") as output:
                record = json.loads(output.read())
        self.assertEqual(record["carla_frame"], 123)
        self.assertEqual(record["vehicle_id"], "vehicle0")
        self.assertEqual(cosim._initialization_failure_counts["overlap"], 1)

    def test_physics_selection_never_takes_autoware_ego(self):
        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_feedback_apply_enabled = True
        cosim.ackermann_feedback_actor_ids = {"*"}
        self.assertTrue(cosim._is_ackermann_feedback_actor("vehicle0"))
        self.assertFalse(cosim._is_ackermann_feedback_actor("AV"))

        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.physics_external_state_enabled = True
        plugin.physics_feedback_actor_ids = {"*"}
        self.assertTrue(plugin._is_physics_feedback_actor("vehicle0"))
        self.assertFalse(plugin._is_physics_feedback_actor("AV"))


if __name__ == "__main__":
    unittest.main()
