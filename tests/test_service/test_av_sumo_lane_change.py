"""Safety and lifecycle regressions for explicit SUMO-controlled AV requests."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from terasim.overlay import traci
from terasim_nde_nade.utils import CommandType
from terasim_nde_nade.vehicle.av_sumo_lane_change_decision_model import (
    AVSumoLaneChangeDecisionModel,
)
from terasim_nde_nade.vehicle.nde_controller import NDEController
from terasim_service.plugins.cosim_inprocess import TeraSimCoSimInProcessPlugin


class ExplicitAVDecisionTest(unittest.TestCase):
    def setUp(self):
        self.states = {1: (traci.constants.LCA_LEFT | traci.constants.LCA_STRATEGIC, 0), -1: (0, 0)}
        self.links = {
            "edge_32_0": (),
            "edge_32_1": (),
            "edge_32_2": (("edge_35_0", True, True, False, ":node_1_0", "", "", 10),),
        }
        self.api = SimpleNamespace(
            constants=traci.constants,
            simulation=SimpleNamespace(getTime=Mock(return_value=10.0), getDeltaT=lambda: 0.05),
            vehicle=SimpleNamespace(
                getLaneID=Mock(return_value="edge_32_1"),
                getRoadID=Mock(return_value="edge_32"),
                getLaneIndex=Mock(return_value=1),
                getRoute=lambda _: ("edge_31", "edge_32", "edge_35"),
                getRouteIndex=lambda _: 1,
                getVehicleClass=lambda _: "passenger",
                getLaneChangeState=lambda _, direction: self.states[direction],
                couldChangeLane=Mock(return_value=True),
                getNeighbors=Mock(return_value=()),
                getLanePosition=Mock(return_value=20.0),
                getSpeed=Mock(return_value=9.51),
                getLength=lambda _: 5.0,
                getTypeID=lambda _: "type",
                setSpeedMode=Mock(),
                setLaneChangeMode=Mock(),
                getLaneChangeMode=lambda _: 0,
                changeLaneRelative=Mock(),
            ),
            lane=SimpleNamespace(
                getLinks=lambda lane: self.links[lane],
                getEdgeID=lambda lane: lane.rsplit("_", 1)[0],
                getAllowed=Mock(return_value=()),
                getDisallowed=Mock(return_value=()),
                getLength=Mock(return_value=172.858),
                getWidth=lambda _: 3.2,
                getShape=lambda lane: (
                    (0.0, int(lane.rsplit("_", 1)[1]) * 3.2),
                    (172.858, int(lane.rsplit("_", 1)[1]) * 3.2),
                ),
            ),
            edge=SimpleNamespace(getLaneNumber=lambda _: 3),
            vehicletype=SimpleNamespace(getMaxSpeedLat=Mock(return_value=1.0)),
        )
        for module in ("av_sumo_lane_change_decision_model", "nde_controller"):
            patcher = patch("terasim_nde_nade.vehicle." + module + ".traci", self.api)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.model = AVSumoLaneChangeDecisionModel()
        self.controller = NDEController(
            object(), preserve_sumo_speed_control=True, release_lane_change_mode=0
        )
        self.model._agent = SimpleNamespace(controller=self.controller)

    def decide(self):
        command, info = self.model.derive_control_command_from_observation(
            {"ego": {"veh_id": "AV"}}
        )
        return command, info["lane_change_decision"]

    def test_strategic_model_state_works_with_zero_post_state(self):
        command, decision = self.decide()
        self.assertEqual(command.command_type, CommandType.LEFT)
        self.assertTrue(decision["route_required"])
        self.assertEqual(decision["post_lane_change_state"], 0)
        self.assertAlmostEqual(decision["required_distance"], 9.51 * 4.2 + 5.0)
        self.api.vehicle.couldChangeLane.assert_called_once_with("AV", 1, state=self.states[1][0])

    def test_right_strategic_and_ordinary_reasons(self):
        self.links["edge_32_0"] = self.links.pop("edge_32_2")
        self.links["edge_32_2"] = ()
        self.states = {
            1: (0, 0),
            -1: (traci.constants.LCA_RIGHT | traci.constants.LCA_STRATEGIC, 0),
        }
        self.assertEqual(self.decide()[0].command_type, CommandType.RIGHT)
        self.links["edge_32_1"] = self.links["edge_32_0"]
        for reason in ("SPEEDGAIN", "KEEPRIGHT", "COOPERATIVE"):
            with self.subTest(reason=reason):
                self.model._reset()
                self.states[-1] = (
                    traci.constants.LCA_RIGHT | getattr(traci.constants, "LCA_" + reason),
                    0,
                )
                command, decision = self.decide()
                self.assertEqual(command.command_type, CommandType.RIGHT)
                self.assertFalse(decision["route_required"])

    def test_all_blocked_masks_and_could_change_are_required(self):
        initial = self.states[1][0]
        for name in ("BLOCKED", "INSUFFICIENT_SPACE", "INSUFFICIENT_SPEED"):
            with self.subTest(name=name):
                self.states[1] = (initial | getattr(traci.constants, "LCA_" + name), 0)
                self.assertEqual(self.decide()[0].command_type, CommandType.DEFAULT)
        self.states[1] = (initial, 0)
        self.api.vehicle.couldChangeLane.return_value = False
        self.assertEqual(self.decide()[0].command_type, CommandType.DEFAULT)

    def test_conflict_rejected_even_if_opposite_request_blocked(self):
        self.states[-1] = (
            traci.constants.LCA_RIGHT | traci.constants.LCA_SPEEDGAIN | traci.constants.LCA_BLOCKED,
            0,
        )
        command, decision = self.decide()
        self.assertEqual(command.command_type, CommandType.DEFAULT)
        self.assertEqual(decision["skip_reason"], "left_right_conflict")

    def test_sublane_only_and_post_only_are_diagnostic(self):
        for model, post in (
            (traci.constants.LCA_LEFT | traci.constants.LCA_SUBLANE, 0),
            (0, self.states[1][0]),
        ):
            self.states[1] = (model, post)
            self.assertEqual(self.decide()[0].command_type, CommandType.DEFAULT)

    def test_ambiguous_connector_and_vehicle_restriction_rejected(self):
        link = self.links["edge_32_2"][0]
        self.links["edge_32_2"] += (link[:4] + (":another_connector",) + link[5:],)
        self.assertEqual(self.decide()[0].command_type, CommandType.DEFAULT)
        self.links["edge_32_2"] = (link,)
        self.api.lane.getDisallowed.return_value = ("passenger",)
        self.assertEqual(self.decide()[0].command_type, CommandType.DEFAULT)

    def test_custom_lane_geometry_jump_is_rejected_without_changing_map(self):
        self.api.lane.getShape = lambda lane: (
            (0.0, int(lane[-1]) * 4.1),
            (172.858, int(lane[-1]) * 4.1),
        )
        command, decision = self.decide()
        self.assertEqual(command.command_type, CommandType.DEFAULT)
        self.assertEqual(
            decision["direction_evaluations"][0]["evaluation_reason"],
            "lane_switch_geometry_discontinuous",
        )

    def test_final_route_edge_does_not_select_connections_beyond_route(self):
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin._physics_planned_link_selector = lambda _: ()
        self.api.vehicle.getRoute = lambda _: ("edge_32",)
        self.api.vehicle.getRouteIndex = lambda _: 0
        with patch("terasim_service.plugins.cosim_inprocess.traci", self.api):
            result = plugin._physics_resolve_route_lane_sequence("AV", "edge_32_1")
        self.assertEqual(result["error"], "")
        self.assertEqual(result["lane_ids"], ("edge_32_1",))

    def test_partial_sublane_permission_does_not_allow_overlapping_target(self):
        self.api.vehicle.getNeighbors.return_value = (("blocker", -5.0),)
        command, decision = self.decide()
        self.assertEqual(command.command_type, CommandType.DEFAULT)
        self.assertEqual(
            decision["direction_evaluations"][0]["evaluation_reason"], "target_neighbor_overlap"
        )

    def test_missing_adjacent_lane_rejected(self):
        self.api.vehicle.getLaneIndex.return_value = 2
        self.assertEqual(self.decide()[0].command_type, CommandType.DEFAULT)

    def test_edge398_late_discretionary_request_rejected(self):
        self.api.vehicle.getRoadID.return_value = "edge_398"
        self.api.vehicle.getLaneID.return_value = "edge_398_0"
        self.api.vehicle.getLaneIndex.return_value = 0
        self.api.vehicle.getRoute = lambda _: ("edge_398", "edge_412")
        self.api.vehicle.getRouteIndex = lambda _: 0
        self.api.vehicle.getLanePosition.return_value = 172.858 - 14.13
        self.links = {"edge_398_0": (("edge_412_0",),), "edge_398_1": (("edge_412_0",),)}
        self.states[1] = (traci.constants.LCA_LEFT | traci.constants.LCA_SPEEDGAIN, 0)
        command, decision = self.decide()
        self.assertEqual(command.command_type, CommandType.DEFAULT)
        self.assertEqual(decision["source_lane_id"], "edge_398_0")
        self.assertAlmostEqual(decision["lane_remaining"], 14.13)
        self.assertGreater(decision["required_distance"], decision["lane_remaining"])
        self.assertEqual(decision["skip_reason"], "insufficient_remaining_distance")
        self.assertEqual(decision["fatal_reason"], "")

    def test_late_route_required_request_fails_closed_even_without_intent(self):
        self.api.vehicle.getLanePosition.return_value = 172.858 - 14.13
        for state in (self.states[1][0], 0):
            self.states[1] = (state, 0)
            command, decision = self.decide()
            self.assertEqual(command.command_type, CommandType.DEFAULT)
            self.assertEqual(decision["fatal_reason"], "route_required_lane_change_unavailable")

    def test_invalid_max_lateral_speed_fails_closed(self):
        self.api.vehicletype.getMaxSpeedLat.return_value = 0
        command, decision = self.decide()
        self.assertEqual(command.command_type, CommandType.DEFAULT)
        self.assertEqual(decision["fatal_reason"], "route_required_lane_change_unavailable")

    def test_request_once_across_busy_blocked_expiry_and_stale_command(self):
        command, _ = self.decide()
        self.controller.execute_control_command("AV", command, {})
        self.states[1] = (self.states[1][0] | traci.constants.LCA_BLOCKED, 0)
        self.api.vehicle.getLanePosition.return_value = 170.0
        for _ in range(3):
            next_command, decision = self.decide()
            self.controller.execute_control_command("AV", next_command, {})
            self.assertFalse(decision["fatal_reason"])
        self.controller._update_controller_status("AV", current_time=19.0)
        self.states[1] = (traci.constants.LCA_LEFT | traci.constants.LCA_STRATEGIC, 0)
        self.assertEqual(self.decide()[1]["skip_reason"], "duplicate_request_latched")
        self.controller.execute_control_command("AV", command, {})
        self.api.vehicle.changeLaneRelative.assert_called_once_with("AV", 1, 8.0)
        self.assertEqual(self.api.vehicle.setLaneChangeMode.call_args.args, ("AV", 0))
        self.assertEqual(self.api.vehicle.setSpeedMode.call_args.args, ("AV", 31))

    def test_request_disappearance_allows_new_generation(self):
        self.decide()
        self.states[1] = (0, 0)
        self.decide()
        self.states[1] = (traci.constants.LCA_LEFT | traci.constants.LCA_STRATEGIC, 0)
        self.assertEqual(self.decide()[1]["request_generation"], 2)

    def test_exit_authority_change_and_restart_clear_all_state(self):
        for event in ("exit", "authority", "restart"):
            with self.subTest(event=event), tempfile.TemporaryDirectory() as output:
                plugin = TeraSimCoSimInProcessPlugin(
                    "test", base_dir=output, control_av=False, av_control_mode="sumo"
                )
                vehicle = self.model._agent
                vehicle.decision_model = self.model
                env = SimpleNamespace(vehicle_list={"AV": vehicle}, av_control_mode="sumo")
                sim = SimpleNamespace(env=env)
                plugin._sync_av_lane_change_lifecycle(sim)
                command, _ = self.decide()
                self.controller.execute_control_command("AV", command, {})
                plugin._capture_av_nde_lane_change_request(sim)
                plugin.physics_lateral_direction_states["AV"] = {"phase": "active"}
                if event == "exit":
                    env.vehicle_list.clear()
                    plugin._sync_av_lane_change_lifecycle(sim)
                elif event == "authority":
                    env.av_control_mode = "external"
                    plugin._sync_av_lane_change_lifecycle(sim)
                else:
                    plugin.function_before_env_start(sim, {})
                self.assertFalse(self.controller.is_busy)
                self.assertIsNone(self.model._latched_request_key)
                self.assertIsNone(plugin.av_nde_lane_change_request)
                self.assertNotIn("AV", plugin.physics_lateral_direction_states)

    def test_plugin_persists_explicit_failure_reason(self):
        self.api.vehicle.getLanePosition.return_value = 170.0
        self.decide()
        with tempfile.TemporaryDirectory() as output:
            plugin = TeraSimCoSimInProcessPlugin(
                "test", base_dir=output, control_av=False, av_control_mode="sumo"
            )
            sim = SimpleNamespace(
                env=SimpleNamespace(vehicle_list={"AV": SimpleNamespace(decision_model=self.model)})
            )
            with self.assertRaisesRegex(RuntimeError, "route_required_lane_change_unavailable"):
                plugin._capture_av_nde_lane_change_decision(sim)

    def test_transient_intent_and_mm_recovery_cannot_authorize(self):
        with tempfile.TemporaryDirectory() as output:
            plugin = TeraSimCoSimInProcessPlugin(
                "test", base_dir=output, control_av=False, av_control_mode="sumo"
            )
            for frame in range(10):
                state, _ = plugin._physics_transition_maneuver_state(
                    "AV",
                    phase_b_time=500.0 + frame * 0.05,
                    current_lane_id="edge_394_0",
                    current_lateral_offset=0.003,
                    lateral_speed=0.02,
                    lane_change_intent="left" if frame % 2 else "none",
                    action={
                        "valid": True,
                        "canonical_delta_d": 0.001,
                        "current_lateral_distance": 0.003,
                        "world_left_normal_x": 0.0,
                        "world_left_normal_y": 1.0,
                        "canonical_path_switched": False,
                    },
                    authorization_source="sumo_intent",
                    authorization_direction="left",
                    sumo_intent_evidence=True,
                )
                self.assertEqual(state["phase"], "lane_keep")
                self.assertFalse(state["maneuver_authorized"])


if __name__ == "__main__":
    unittest.main()
