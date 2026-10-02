import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

from addict import Dict
from terasim_nde_nade.envs.nade_with_av import NADEWithAV
from terasim_nde_nade.utils import CommandType, NDECommand
from terasim_nde_nade.vehicle.nde_controller import NDEController
from terasim_nde_nade.vehicle.nde_vehicle_factory import NDEVehicleFactory
from terasim_service.plugins.cosim_inprocess import TeraSimCoSimInProcessPlugin
from terasim_service.run_cosim import (
    resolve_av_control_settings,
    validate_sumo_av_physics_configuration,
)
from terasim_service.utils.base import resolve_environment_av_control_mode


class _FakeSimulation:
    time = 10.0

    @classmethod
    def getTime(cls):
        return cls.time

    @staticmethod
    def getDeltaT():
        return 0.05


class _FakeVehicleDomain:
    def __init__(self):
        self.road_id = "edge_32"
        self.lane_id = "edge_32_1"
        self.lane_index = 1
        self.route = ("edge_31", "edge_32", "edge_35")
        self.route_index = 1
        self.left_state = 2 | 8
        self.right_state = 4
        self.calls = []
        self.lane_change_mode = 0

    def getRoadID(self, _veh_id):
        return self.road_id

    def getLaneID(self, _veh_id):
        return self.lane_id

    def getLaneIndex(self, _veh_id):
        return self.lane_index

    def getRoute(self, _veh_id):
        return self.route

    def getRouteIndex(self, _veh_id):
        return self.route_index

    def getLaneChangeState(self, _veh_id, direction):
        state = self.left_state if direction > 0 else self.right_state
        return state, state

    def setSpeedMode(self, veh_id, mode):
        self.calls.append(("speed_mode", veh_id, mode))

    def setLaneChangeMode(self, veh_id, mode):
        self.calls.append(("lane_mode", veh_id, mode))
        self.lane_change_mode = mode

    def getLaneChangeMode(self, _veh_id):
        return self.lane_change_mode

    def changeLaneRelative(self, veh_id, offset, duration):
        self.calls.append(("change_relative", veh_id, offset, duration))

    def setColor(self, veh_id, color):
        self.calls.append(("color", veh_id, color))

    def subscribeContext(self, veh_id, command, radius, variables):
        self.calls.append(("subscribe_context", veh_id, command, radius, variables))

    def setSpeed(self, veh_id, speed):
        self.calls.append(("speed", veh_id, speed))


class _FakeLaneDomain:
    def __init__(self):
        self.links = {
            "edge_32_1": (),
            "edge_32_2": (("edge_35_0",),),
            "edge_32_0": (),
        }

    def getLinks(self, lane_id):
        return self.links.get(lane_id, ())

    @staticmethod
    def getEdgeID(lane_id):
        return lane_id.rsplit("_", 1)[0]


class AVControlModeTest(unittest.TestCase):
    def setUp(self):
        _FakeSimulation.time = 10.0
        self.vehicle = _FakeVehicleDomain()
        self.lane = _FakeLaneDomain()
        self.fake_traci = SimpleNamespace(
            vehicle=self.vehicle,
            lane=self.lane,
            edge=SimpleNamespace(getLaneNumber=lambda _edge: 3),
            simulation=_FakeSimulation,
            constants=SimpleNamespace(
                LCA_LEFT=2,
                LCA_RIGHT=4,
                LCA_STRATEGIC=8,
                LCA_BLOCKED=1 << 20,
                CMD_GET_VEHICLE_VARIABLE=0xA4,
                VAR_DISTANCE=0x84,
                VAR_POSITION=0x42,
            ),
        )

    def test_control_mode_resolution_defaults_external_and_rejects_conflicts(self):
        self.assertEqual(resolve_environment_av_control_mode({}), "external")
        self.assertEqual(resolve_environment_av_control_mode({"av_debug_control": True}), "sumo")
        self.assertEqual(resolve_av_control_settings(None, None, {}), ("external", True))
        self.assertEqual(resolve_av_control_settings("sumo", None, {}), ("sumo", False))
        self.assertEqual(resolve_av_control_settings(None, False, {}), ("external", False))
        with self.assertRaises(ValueError):
            resolve_environment_av_control_mode(
                {"av_control_mode": "sumo", "av_debug_control": False}
            )
        with self.assertRaises(ValueError):
            resolve_av_control_settings("sumo", True, {})

    def test_sumo_mode_requires_complete_carla_physics_feedback_configuration(self):
        complete = {
            "CARLA_COSIM_VEHICLE_CONTROL_MODE": "ackermann_physics",
            "CARLA_COSIM_ACKERMANN_FEEDBACK_MODE": "apply",
            "CARLA_COSIM_ACKERMANN_FEEDBACK_ASSIMILATION_MODE": "external_state",
            "CARLA_COSIM_ACKERMANN_FEEDBACK_ACTORS": "AV",
            "CARLA_COSIM_ACKERMANN_LATERAL_MANEUVER_ENABLED": "1",
            "CARLA_COSIM_ACKERMANN_LATERAL_MANEUVER_TARGET_ENABLED": "1",
        }
        args = SimpleNamespace(tick_mode="master")
        with patch.dict("os.environ", complete, clear=True):
            validate_sumo_av_physics_configuration(args)
        incomplete = dict(complete, CARLA_COSIM_VEHICLE_CONTROL_MODE="transform")
        with patch.dict("os.environ", incomplete, clear=True):
            with self.assertRaisesRegex(ValueError, "ackermann_physics"):
                validate_sumo_av_physics_configuration(args)

    def test_factory_uses_safe_sumo_adapter_and_zero_av_lane_mode(self):
        factory_module = "terasim_nde_nade.vehicle.nde_vehicle_factory"
        fake_vehicle = object()
        with (
            patch(f"{factory_module}.NDEEgoSensor", return_value=object()),
            patch(
                f"{factory_module}.AVSumoLaneChangeDecisionModel", return_value=object()
            ) as model,
            patch(f"{factory_module}.NDEController", return_value=object()) as controller,
            patch(f"{factory_module}.Vehicle", return_value=fake_vehicle),
        ):
            result = NDEVehicleFactory(Dict(av_control_mode="sumo", AV_cfg=Dict())).create_vehicle(
                "AV", object()
            )
        self.assertIs(result, fake_vehicle)
        model.assert_called_once_with(
            cfg=Dict(av_control_mode="sumo", AV_cfg=Dict()),
            MOBIL_lc_flag=True,
            stochastic_acc_flag=False,
            reroute=False,
            dynamically_change_vtype=False,
        )
        controller.assert_called_once_with(
            ANY,
            preserve_sumo_speed_control=True,
            release_lane_change_mode=0,
        )

    def test_factory_keeps_external_av_adapter_as_default(self):
        factory_module = "terasim_nde_nade.vehicle.nde_vehicle_factory"
        with (
            patch(f"{factory_module}.NDEEgoSensor", return_value=object()),
            patch(f"{factory_module}.NDEDecisionModel", return_value=object()) as model,
            patch(f"{factory_module}.NDEController", return_value=object()) as controller,
            patch(f"{factory_module}.Vehicle", return_value=object()),
        ):
            NDEVehicleFactory(Dict(AV_cfg=Dict())).create_vehicle("AV", object())
        model.assert_called_once()
        controller.assert_called_once_with(
            ANY,
            preserve_sumo_speed_control=False,
            release_lane_change_mode=0,
        )

    def test_sumo_spawn_disables_autonomous_lane_changes(self):
        env = object.__new__(NADEWithAV)
        env.av_cfg = Dict(type="NDE_URBAN")
        env.cache_radius = 100.0
        env.configured_av_control_mode = "sumo"
        env.av_control_mode = "pending"
        env.add_vehicle = Mock()
        with patch(
            "terasim_nde_nade.envs.nade_with_av.traci",
            self.fake_traci,
        ):
            env.add_av_with_sumo_control("edge_394", "edge_394_0", 0.810798662808417, 0.0)
        self.assertIn(("speed_mode", "AV", 31), self.vehicle.calls)
        self.assertIn(("lane_mode", "AV", 0), self.vehicle.calls)
        self.assertIn(("speed", "AV", -1), self.vehicle.calls)
        self.assertEqual(env.av_control_mode, "sumo")

    def test_external_warmup_does_not_enable_sumo_lane_changes(self):
        env = object.__new__(NADEWithAV)
        env.av_cfg = Dict(type="NDE_URBAN")
        env.cache_radius = 100.0
        env.configured_av_control_mode = "external"
        env.av_control_mode = "pending"
        env.add_vehicle = Mock()
        with patch(
            "terasim_nde_nade.envs.nade_with_av.traci",
            self.fake_traci,
        ):
            env.add_av_with_sumo_control("edge_394", "edge_394_0", 0.810798662808417, 0.0)
        self.assertIn(("speed_mode", "AV", 31), self.vehicle.calls)
        self.assertIn(("lane_mode", "AV", 0), self.vehicle.calls)
        self.assertIn(("speed", "AV", -1), self.vehicle.calls)

    def test_explicit_controller_request_keeps_lane_change_mode_zero(self):
        with patch(
            "terasim_nde_nade.vehicle.nde_controller.traci",
            self.fake_traci,
        ):
            controller = NDEController(
                object(),
                preserve_sumo_speed_control=True,
                release_lane_change_mode=0,
            )
            command = NDECommand(
                command_type=CommandType.LEFT,
                duration=8.0,
                info={
                    "decision_source": "sumo_model_strategic",
                    "source_lane_id": "edge_32_1",
                    "target_lane_id": "edge_32_2",
                },
            )
            controller.execute_control_command("AV", command, {})
            controller.execute_control_command("AV", command, {})
            relative_calls = [call for call in self.vehicle.calls if call[0] == "change_relative"]
            self.assertEqual(relative_calls, [("change_relative", "AV", 1, 8.0)])
            self.assertEqual(controller.lane_change_request_count, 1)
            self.assertTrue(controller.is_busy)
            self.assertIn(("speed_mode", "AV", 31), self.vehicle.calls)
            self.assertNotIn(("lane_mode", "AV", 1621), self.vehicle.calls)

            _FakeSimulation.time = 18.1
            controller._update_controller_status("AV")
            self.assertFalse(controller.is_busy)
            self.assertEqual(
                self.vehicle.calls[-2:],
                [
                    ("speed_mode", "AV", 31),
                    ("lane_mode", "AV", 0),
                ],
            )

    def test_background_controller_restores_lane_change_mode_1621(self):
        with patch(
            "terasim_nde_nade.vehicle.nde_controller.traci",
            self.fake_traci,
        ):
            controller = NDEController(object())
            command = NDECommand(command_type=CommandType.RIGHT, duration=1.0)
            _FakeSimulation.time = 20.0
            controller.execute_control_command("vehicle1", command, {})
            _FakeSimulation.time = 21.1
            controller._update_controller_status("vehicle1")
        self.assertEqual(
            self.vehicle.calls[-2:],
            [
                ("speed_mode", "vehicle1", 31),
                ("lane_mode", "vehicle1", 1621),
            ],
        )

    def test_plugin_accepts_fresh_nde_request_without_sumo_intent(self):
        with tempfile.TemporaryDirectory() as output_dir:
            plugin = TeraSimCoSimInProcessPlugin(
                "nde-request",
                base_dir=output_dir,
                control_av=False,
                av_control_mode="sumo",
            )
            plugin.physics_step_length = 0.05
            plugin.av_nde_lane_change_request = {
                "simulation_time": 10.0,
                "direction": "left",
                "source_lane_id": "edge_32_1",
                "target_lane_id": "edge_32_2",
            }
            self.vehicle.left_state = 0
            self.vehicle.right_state = 0
            _FakeSimulation.time = 10.05
            with patch(
                "terasim_service.plugins.cosim_inprocess.traci",
                self.fake_traci,
            ):
                direction, target, source, sumo_intent = plugin._physics_lane_change_action(
                    "AV", "edge_32_1"
                )
        self.assertEqual(
            (direction, target, source),
            (
                "left",
                "edge_32_2",
                "terasim_nde_request",
            ),
        )
        self.assertFalse(sumo_intent)

    def test_nde_authorization_source_remains_latched_after_sumo_intent_appears(self):
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.logger = SimpleNamespace(info=lambda *args: None, warning=lambda *args: None)
        plugin.physics_lateral_direction_states = {}
        action = {
            "valid": True,
            "current_lateral_distance": 0.0,
            "canonical_delta_d": 0.0,
            "world_left_normal_x": 0.0,
            "world_left_normal_y": 1.0,
            "canonical_path_switched": False,
        }
        state, _ = plugin._physics_transition_maneuver_state(
            "AV",
            phase_b_time=1.0,
            current_lane_id="edge_32_1",
            current_lateral_offset=0.0,
            lateral_speed=0.2,
            lane_change_intent="left",
            action=action,
            authorization_source="terasim_nde_request",
            authorization_direction="left",
            sumo_intent_evidence=False,
        )
        self.assertEqual(state["phase"], "armed")
        self.assertEqual(state["maneuver_authorization_source"], "terasim_nde_request")

        state, _ = plugin._physics_transition_maneuver_state(
            "AV",
            phase_b_time=1.05,
            current_lane_id="edge_32_1",
            current_lateral_offset=0.0,
            lateral_speed=0.2,
            lane_change_intent="left",
            action=action,
            authorization_source="sumo_intent",
            authorization_direction="left",
            sumo_intent_evidence=True,
        )
        self.assertEqual(state["phase"], "armed")
        self.assertEqual(state["maneuver_authorization_source"], "terasim_nde_request")


if __name__ == "__main__":
    unittest.main()
