import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path
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
    adapt_lookahead_distances_for_compiled_paths,
    apply_external_state_lateral_maneuver_target,
    build_active_maneuver_lane_corridor,
    build_external_state_lateral_action_lookahead,
    compile_lane_shapes,
    find_lookahead_positions_from_compiled_paths,
    project_position_to_compiled_route,
    sample_compiled_route_curvature,
    sample_compiled_route_frame,
)


class _FakeVehicle:
    def __init__(self, calls):
        self.calls = calls
        self.position = (10.0, 1.0)
        self.angle = 90.0
        self.speed = 4.0
        self.acceleration = 0.5
        self.lateral_position = 0.0
        self.lane_id = "edge_0_0"
        self.route = ("edge_0", "edge_1")

    def getLaneID(self, actor_id):
        return self.lane_id

    def getRoute(self, actor_id):
        return self.route

    def moveTo(self, actor_id, lane_id, position):
        self.calls.append(("moveTo", actor_id, lane_id, position))
        self.lane_id = lane_id

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
        self.lane_id = f"{edge_id}_{lane_index}"

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

    def getLateralLanePosition(self, actor_id):
        return self.lateral_position


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
    @staticmethod
    def _maneuver_evidence_action(
        *,
        canonical_delta_d=0.0,
        current_d=0.0,
        valid=True,
        path_switched=False,
    ):
        return {
            "valid": valid,
            "error": "" if valid else "fixture_invalid",
            "canonical_delta_d": canonical_delta_d,
            "current_lateral_distance": current_d,
            "world_left_normal_x": 0.0,
            "world_left_normal_y": 1.0,
            "canonical_path_switched": path_switched,
            "canonical_phase_b": None,
            "lateral_motion_detected": abs(canonical_delta_d) >= 0.002,
            "lateral_direction_source": "canonical_delta_d",
            "phase_b_lateral_delta": canonical_delta_d,
            "world_lateral_speed": 0.0,
        }

    def test_lateral_experimental_features_default_off_and_allow_opt_in(self):
        flag_names = {
            "CARLA_COSIM_ACKERMANN_LATERAL_MANEUVER_ENABLED": "",
            "CARLA_COSIM_ACKERMANN_LATERAL_MANEUVER_TARGET_ENABLED": "",
            "CARLA_COSIM_ACKERMANN_INTERNAL_LANE_TRANSITION_ASSIST_ENABLED": "",
        }
        with (
            tempfile.TemporaryDirectory() as output_dir,
            patch.dict("os.environ", flag_names, clear=False),
        ):
            plugin = TeraSimCoSimInProcessPlugin("feature-flags-default", base_dir=output_dir)
            self.assertFalse(plugin.physics_lateral_maneuver_enabled)
            self.assertFalse(plugin.physics_lateral_maneuver_target_enabled)
            self.assertFalse(plugin.physics_internal_lane_transition_assist_enabled)
            self.assertEqual(plugin.physics_steering_preview_min_distance, 2.8)
            self.assertEqual(plugin.physics_steering_preview_time_headway, 0.5)
            self.assertEqual(plugin.physics_steering_preview_max_distance, 5.0)

        enabled_flags = {key: "1" for key in flag_names}
        enabled_flags.update(
            {
                "CARLA_COSIM_ACKERMANN_STEERING_PREVIEW_MIN_DISTANCE": "3.0",
                "CARLA_COSIM_ACKERMANN_STEERING_PREVIEW_TIME_HEADWAY": "0.6",
                "CARLA_COSIM_ACKERMANN_STEERING_PREVIEW_MAX_DISTANCE": "6.0",
            }
        )
        with (
            tempfile.TemporaryDirectory() as output_dir,
            patch.dict("os.environ", enabled_flags, clear=False),
        ):
            plugin = TeraSimCoSimInProcessPlugin("feature-flags-enabled", base_dir=output_dir)
            self.assertTrue(plugin.physics_lateral_maneuver_enabled)
            self.assertTrue(plugin.physics_lateral_maneuver_target_enabled)
            self.assertTrue(plugin.physics_internal_lane_transition_assist_enabled)
            self.assertEqual(plugin.physics_steering_preview_min_distance, 3.0)
            self.assertEqual(plugin.physics_steering_preview_time_headway, 0.6)
            self.assertEqual(plugin.physics_steering_preview_max_distance, 6.0)

        with (
            tempfile.TemporaryDirectory() as output_dir,
            patch.dict(
                "os.environ",
                {
                    "CARLA_COSIM_ACKERMANN_LATERAL_MANEUVER_ENABLED": "0",
                    "CARLA_COSIM_ACKERMANN_LATERAL_MANEUVER_TARGET_ENABLED": "1",
                },
                clear=False,
            ),
            self.assertRaisesRegex(ValueError, "TARGET_ENABLED requires"),
        ):
            TeraSimCoSimInProcessPlugin("invalid-maneuver-target", base_dir=output_dir)

        with (
            tempfile.TemporaryDirectory() as output_dir,
            patch.dict(
                "os.environ",
                {
                    "CARLA_COSIM_ACKERMANN_STEERING_PREVIEW_MIN_DISTANCE": "5.0",
                    "CARLA_COSIM_ACKERMANN_STEERING_PREVIEW_MAX_DISTANCE": "2.8",
                },
                clear=False,
            ),
            self.assertRaisesRegex(ValueError, "invalid Ackermann steering preview configuration"),
        ):
            TeraSimCoSimInProcessPlugin("invalid-preview", base_dir=output_dir)

    def test_phase_aligned_canonical_target_error_uses_sumo_world_normal(self):
        current = SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        )
        desired = SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=1.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        )
        velocity = SimpleNamespace(x=0.0, y=0.0)
        error = CarlaCosim._phase_aligned_front_canonical_target_error(
            current, desired, 0.0, velocity, 0.0, -1.0, 0.05
        )
        self.assertAlmostEqual(error, 1.0)
        self.assertIsNone(
            CarlaCosim._phase_aligned_front_canonical_target_error(
                current, desired, 0.0, velocity, 0.0, 0.0, 0.05
            )
        )

    def test_canonical_target_error_threshold_warns_once_per_episode_and_continues(self):
        cosim = object.__new__(CarlaCosim)
        cosim._ackermann_actor_state = {}
        cosim.canonical_target_error_warn_m = 0.30
        info = {"lane_id": "edge_0_0", "external_state_maneuver_phase": "active"}

        with patch("builtins.print") as warning:
            self.assertTrue(cosim._update_canonical_target_error_warning("AV", 0.316, info))
            self.assertTrue(cosim._update_canonical_target_error_warning("AV", 0.314, info))
            self.assertFalse(cosim._update_canonical_target_error_warning("AV", 0.299, info))
            self.assertTrue(cosim._update_canonical_target_error_warning("AV", -0.301, info))

        self.assertEqual(warning.call_count, 2)
        self.assertIn("phase=active", warning.call_args_list[0].args[0])
        state = cosim._ackermann_actor_state["AV"]
        self.assertTrue(state["canonical_target_error_warning_active"])
        self.assertEqual(state["canonical_target_error_warning_episodes"], 2)

    def test_initial_inprocess_state_accepts_2d_position_and_exports_slope(self):
        fake_vehicle = SimpleNamespace(
            getPosition3D=lambda _actor_id: (10.0, 20.0),
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
        plugin.physics_feedback_observations = {"vehicle0": {"z": 7.25}}
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
        self.assertEqual(state.agent_details["vehicle"]["vehicle0"]["z"], 7.25)

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

    def test_ackermann_lateral_gain_increases_pure_pursuit_steer(self):
        common = {
            "current_x": 0.0,
            "current_y": 0.0,
            "yaw_degrees": 0.0,
            "current_speed": 5.0,
            "desired_x": 1.0,
            "desired_y": 0.0,
            "lookahead_x": 5.0,
            "lookahead_y": 1.0,
            "desired_speed": 5.0,
        }
        baseline = compute_ackermann_control_values(
            **common, tuning=AckermannTuning(lateral_gain=1.0)
        )
        boosted = compute_ackermann_control_values(
            **common, tuning=AckermannTuning(lateral_gain=1.5)
        )

        self.assertGreater(boosted.raw_steer, baseline.raw_steer)

    def test_canonical_front_path_tracker_combines_feedforward_heading_and_cross_track(self):
        tuning = AckermannTuning(wheel_base=2.8, lateral_gain=1.5)
        values = compute_ackermann_control_values(
            current_x=0.0,
            current_y=0.0,
            yaw_degrees=0.0,
            current_speed=5.0,
            desired_x=1.0,
            desired_y=0.0,
            lookahead_x=3.0,
            lookahead_y=2.0,
            desired_speed=5.0,
            tuning=tuning,
            path_curvature=0.1,
            path_heading_degrees=10.0,
            path_cross_track_error=0.2,
        )

        expected_feedforward = math.atan(2.8 * 0.1)
        expected_heading = math.radians(10.0)
        expected_cross_track = -math.atan2(1.5 * 0.2, 5.0)
        self.assertEqual(values.lateral_control_mode, "canonical_front_path")
        self.assertAlmostEqual(values.path_feedforward_steer, expected_feedforward)
        self.assertAlmostEqual(values.path_heading_feedback_steer, expected_heading)
        self.assertAlmostEqual(values.path_cross_track_feedback_steer, expected_cross_track)
        self.assertAlmostEqual(
            values.raw_steer,
            expected_feedforward + expected_heading + expected_cross_track,
        )

    def test_canonical_front_path_cross_track_is_smoothly_blended_below_half_mps(self):
        common = {
            "current_x": 0.0,
            "current_y": 0.0,
            "yaw_degrees": 0.0,
            "desired_x": 1.0,
            "desired_y": 0.0,
            "lookahead_x": 3.0,
            "lookahead_y": 0.0,
            "desired_speed": 1.0,
            "path_curvature": 0.0,
            "path_heading_degrees": 0.0,
            "path_cross_track_error": 0.2,
            "tuning": AckermannTuning(lateral_gain=1.5),
        }
        stopped = compute_ackermann_control_values(current_speed=0.0, **common)
        quarter = compute_ackermann_control_values(current_speed=0.25, **common)
        half = compute_ackermann_control_values(current_speed=0.5, **common)
        above = compute_ackermann_control_values(current_speed=1.0, **common)

        self.assertEqual(stopped.path_cross_track_low_speed_blend, 0.0)
        self.assertEqual(stopped.path_cross_track_feedback_steer, 0.0)
        self.assertAlmostEqual(quarter.path_cross_track_low_speed_blend, 0.5)
        self.assertAlmostEqual(
            quarter.path_cross_track_feedback_steer,
            0.5 * quarter.path_cross_track_feedback_unblended_steer,
        )
        self.assertEqual(half.path_cross_track_low_speed_blend, 1.0)
        self.assertEqual(above.path_cross_track_low_speed_blend, 1.0)
        self.assertAlmostEqual(
            half.path_cross_track_feedback_steer,
            half.path_cross_track_feedback_unblended_steer,
        )
        self.assertAlmostEqual(
            above.path_cross_track_feedback_steer,
            -math.atan2(1.5 * 0.2, 1.0),
        )

    def test_canonical_front_path_tracker_rejects_partial_inputs(self):
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            compute_ackermann_control_values(
                current_x=0.0,
                current_y=0.0,
                yaw_degrees=0.0,
                current_speed=3.0,
                desired_x=1.0,
                desired_y=0.0,
                lookahead_x=3.0,
                lookahead_y=0.0,
                desired_speed=3.0,
                path_curvature=0.1,
            )

    def test_canonical_route_curvature_is_signed(self):
        left_path = compile_lane_shapes([((0.0, 0.0), (5.0, 0.0), (9.0, 2.0), (12.0, 6.0))])
        right_path = compile_lane_shapes([((0.0, 0.0), (5.0, 0.0), (9.0, -2.0), (12.0, -6.0))])
        straight_path = compile_lane_shapes([((0.0, 0.0), (20.0, 0.0))])

        self.assertGreater(sample_compiled_route_curvature(left_path, 5.0), 0.0)
        self.assertLess(sample_compiled_route_curvature(right_path, 5.0), 0.0)
        self.assertAlmostEqual(sample_compiled_route_curvature(straight_path, 5.0), 0.0)

    def test_authoritative_path_tracker_reflects_sumo_frame_into_carla(self):
        cosim = object.__new__(CarlaCosim)
        heading = 0.2
        result = cosim._resolve_authoritative_path_tracker(
            {
                "path_tracker_valid": True,
                "path_tracker_rear_curvature": 0.1,
                "path_tracker_reference_tangent_x": math.cos(heading),
                "path_tracker_reference_tangent_y": math.sin(heading),
                "path_tracker_front_cross_track_error": 0.3,
            }
        )

        self.assertAlmostEqual(result["path_curvature"], -0.1)
        self.assertAlmostEqual(result["path_heading_degrees"], -math.degrees(heading))
        self.assertAlmostEqual(result["path_cross_track_error"], -0.3)

    def test_direct_brake_mapping_has_minimum_and_saturates(self):
        self.assertAlmostEqual(compute_direct_brake_value(-1.0, 8.0, 0.5), 0.5)
        self.assertAlmostEqual(compute_direct_brake_value(-20.0, 8.0, 0.5), 1.0)

    def test_lane_shape_stitch_preserves_continuous_geometry(self):
        path = compile_lane_shapes(
            [
                ((0.0, 0.0), (5.0, 0.0)),
                ((5.0, 0.0), (10.0, 0.0)),
            ]
        )

        self.assertIsNotNone(path)
        self.assertEqual(path["shape_join_accepted_count"], 2)
        self.assertEqual(path["shape_join_trimmed_count"], 0)
        self.assertIsNone(path["shape_join_truncated_index"])
        self.assertTrue((path["deltas"][:, 0] > 0.0).all())
        self.assertTrue((path["deltas"][:, 1] == 0.0).all())
        self.assertAlmostEqual(path["last_end"][0], 10.0)

    def test_lane_shape_stitch_snaps_small_endpoint_gap(self):
        path = compile_lane_shapes(
            [
                ((0.0, 0.0), (5.0, 0.0)),
                ((5.02, 0.0), (10.0, 0.0)),
            ]
        )

        self.assertIsNotNone(path)
        self.assertEqual(path["shape_join_accepted_count"], 2)
        self.assertIsNone(path["shape_join_truncated_index"])
        self.assertTrue((path["deltas"][:, 0] > 0.0).all())
        self.assertAlmostEqual(path["starts"][-1][0], 5.0)
        self.assertAlmostEqual(path["last_end"][0], 10.0)

    def test_lane_shape_stitch_trims_overlapping_successor_prefix(self):
        path = compile_lane_shapes(
            [
                ((0.0, 0.0), (10.0, 0.0)),
                ((10.0, 0.0), (11.0, 0.0)),
                ((10.0, 0.0), (20.0, 0.0)),
            ]
        )
        target = find_lookahead_positions_from_compiled_paths(
            [path],
            [(9.0, 0.0)],
            [3.0],
            [0.0],
        )[0]

        self.assertEqual(path["shape_join_accepted_count"], 3)
        self.assertEqual(path["shape_join_trimmed_count"], 1)
        self.assertAlmostEqual(path["shape_join_trimmed_distance"], 1.0)
        self.assertIsNone(path["shape_join_truncated_index"])
        self.assertTrue((path["deltas"][:, 0] > 0.0).all())
        self.assertAlmostEqual(target[0], 12.0)
        self.assertAlmostEqual(target[1], 0.0)

    def test_vehicle3429_shape_overlap_does_not_create_false_curve_cap(self):
        current_shape = (
            (989.820, 171.613),
            (991.504, 172.735),
            (993.195, 173.849),
            (994.890, 174.955),
            (996.611, 176.056),
        )
        internal_shape = (
            (996.611, 176.056),
            (996.820, 176.193),
        )
        successor_shape = (
            (996.595, 176.046),
            (999.429, 177.903),
            (1002.263, 179.758),
        )
        path = compile_lane_shapes([current_shape, internal_shape, successor_shape])
        origin = (989.88649752698, 171.65730535942)
        base_distance = 8.481613937655627
        distances, heading_changes = adapt_lookahead_distances_for_compiled_paths(
            [path],
            [origin],
            [base_distance],
        )
        target = find_lookahead_positions_from_compiled_paths(
            [path],
            [origin],
            distances,
            [0.0],
        )[0]

        self.assertEqual(path["shape_join_trimmed_count"], 1)
        self.assertAlmostEqual(path["shape_join_trimmed_distance"], 0.268764, places=5)
        self.assertLess(heading_changes[0], 0.05)
        self.assertAlmostEqual(distances[0], base_distance)
        self.assertGreater(target[0], origin[0])
        self.assertGreater(target[1], origin[1])

        cosim = object.__new__(CarlaCosim)
        cosim._ackermann_actor_state = {}
        cosim.ackermann_tuning = AckermannTuning()
        cosim.ackermann_max_lateral_accel = 3.0
        cosim.step_length = 0.05
        cosim._is_ackermann_feedback_apply_actor = lambda _actor_id: True
        vehicle_info = {
            "sumo_desired_speed": base_distance,
            "feedback_observed_speed": 8.383862316260045,
            "acceleration": 1.9550324279116538,
            "sumo_emergency_decel": 9.0,
            "lookahead_heading_change": heading_changes[0],
            "lookahead_distance": distances[0],
        }

        speed_target, acceleration = cosim._resolve_ackermann_longitudinal_target(
            "vehicle3429",
            vehicle_info,
            8.383862316260045,
        )

        self.assertAlmostEqual(speed_target, base_distance)
        self.assertAlmostEqual(acceleration, 1.9550324279116538)
        self.assertFalse(cosim._ackermann_actor_state["vehicle3429"]["curve_speed_cap_active"])

    def test_lane_shape_stitch_preserves_real_curve(self):
        heading_change = 0.77057
        path = compile_lane_shapes(
            [
                ((0.0, 0.0), (1.0, 0.0)),
                (
                    (1.0, 0.0),
                    (
                        1.0 + 10.0 * math.cos(heading_change),
                        10.0 * math.sin(heading_change),
                    ),
                ),
            ]
        )
        _, heading_changes = adapt_lookahead_distances_for_compiled_paths(
            [path],
            [(0.0, 0.0)],
            [3.7035],
        )

        self.assertAlmostEqual(heading_changes[0], heading_change)

    def test_lane_shape_stitch_truncates_unverified_boundary(self):
        path = compile_lane_shapes(
            [
                ((0.0, 0.0), (10.0, 0.0)),
                ((20.0, 0.0), (30.0, 0.0)),
            ]
        )

        self.assertIsNotNone(path)
        self.assertEqual(path["shape_join_accepted_count"], 1)
        self.assertEqual(path["shape_join_truncated_index"], 1)
        self.assertAlmostEqual(path["shape_join_endpoint_distance"], 10.0)
        self.assertAlmostEqual(path["shape_join_projection_error"], 10.0)
        self.assertAlmostEqual(path["last_end"][0], 10.0)
        self.assertTrue((path["deltas"][:, 0] > 0.0).all())

    def test_lane_shape_stitch_handles_invalid_and_fully_overlapped_shapes(self):
        invalid = compile_lane_shapes(
            [
                ((0.0, 0.0), (10.0, 0.0)),
                ((float("nan"), 0.0), (20.0, 0.0)),
            ]
        )
        zero_length = compile_lane_shapes([((0.0, 0.0), (0.0, 0.0))])
        fully_overlapped = compile_lane_shapes(
            [
                ((0.0, 0.0), (10.0, 0.0)),
                ((0.0, 0.0), (10.0, 0.0)),
            ]
        )

        self.assertEqual(invalid["shape_join_truncated_index"], 1)
        self.assertAlmostEqual(invalid["last_end"][0], 10.0)
        self.assertIsNone(zero_length)
        self.assertEqual(fully_overlapped["shape_join_accepted_count"], 2)
        self.assertEqual(fully_overlapped["shape_join_trimmed_count"], 1)
        self.assertAlmostEqual(fully_overlapped["last_end"][0], 10.0)

    def test_unverified_lane_boundary_warns_once_per_cached_path(self):
        fake_traci = SimpleNamespace(
            vehicle=SimpleNamespace(
                getLaneID=lambda _actor_id: "edge_0_0",
                getRoute=lambda _actor_id: ("edge_0", "edge_1"),
                getRouteIndex=lambda _actor_id: 0,
                getNextLinks=lambda _actor_id: (
                    (
                        "edge_1_0",
                        ":node_0_0_0",
                        True,
                        True,
                        False,
                        "M",
                        "s",
                        0.25,
                    ),
                ),
            ),
            lane=SimpleNamespace(
                getEdgeID=lambda lane_id: {
                    "edge_0_0": "edge_0",
                    "edge_1_0": "edge_1",
                    ":node_0_0_0": ":node_0",
                }[lane_id],
                getLinks=lambda lane_id: {
                    "edge_0_0": (
                        (
                            "edge_1_0",
                            True,
                            True,
                            False,
                            ":node_0_0_0",
                            "M",
                            "s",
                            0.25,
                        ),
                    ),
                    ":node_0_0_0": (("edge_1_0", True, True, False, "", "M", "s", 0.25),),
                }.get(lane_id, ()),
            ),
        )
        shapes = {
            "edge_0_0": ((0.0, 0.0), (10.0, 0.0)),
            ":node_0_0_0": ((10.0, 0.0), (11.0, 0.0)),
            "edge_1_0": ((20.0, 0.0), (30.0, 0.0)),
        }
        warnings = []
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.physics_lookahead_path_cache = {}
        plugin._physics_lane_geometry = lambda lane_id: {"shape": shapes[lane_id]}
        plugin.logger = SimpleNamespace(warning=lambda *args: warnings.append(args))

        with patch.object(cosim_inprocess, "traci", fake_traci):
            first = plugin._physics_compiled_lookahead_path("vehicle0", "edge_0_0")
            second = plugin._physics_compiled_lookahead_path("vehicle0", "edge_0_0")

        self.assertIs(first, second)
        self.assertEqual(first["shape_join_truncated_index"], 2)
        self.assertEqual(len(warnings), 1)
        self.assertIn("lookahead path truncated", warnings[0][0])

    def test_rolling_path_keeps_body_length_of_predecessor_and_occurrences(self):
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.physics_rolling_path_states = {}
        lengths = {"edge_a_0": 4.0, ":via_0": 0.25, "edge_b_0": 12.0}
        plugin._physics_lane_geometry = lambda lane_id: {"length": lengths[lane_id]}

        first, _ = plugin._physics_rolling_lane_entries(
            "AV",
            "edge_a_0",
            ("edge_a_0", ":via_0", "edge_b_0"),
            6.244,
        )
        second, _ = plugin._physics_rolling_lane_entries(
            "AV",
            ":via_0",
            (":via_0", "edge_b_0"),
            6.244,
        )
        third, third_state = plugin._physics_rolling_lane_entries(
            "AV",
            "edge_b_0",
            ("edge_b_0", "edge_a_0"),
            6.244,
        )

        self.assertEqual(
            tuple(lane_id for _, lane_id in first),
            ("edge_a_0", ":via_0", "edge_b_0"),
        )
        self.assertEqual(
            tuple(lane_id for _, lane_id in second),
            ("edge_a_0", ":via_0", "edge_b_0"),
        )
        self.assertEqual(
            tuple(lane_id for _, lane_id in third),
            ("edge_a_0", ":via_0", "edge_b_0", "edge_a_0"),
        )
        self.assertEqual(first[0][0], 0)
        self.assertNotEqual(first[0][0], third[-1][0])
        self.assertGreaterEqual(third_state["retained_history_distance"], 4.25)

    def test_internal_lane_lookahead_inserts_immediate_successor(self):
        fake_traci = SimpleNamespace(
            vehicle=SimpleNamespace(
                getLaneID=lambda _actor_id: ":node_119_3_0",
                getRoute=lambda _actor_id: ("edge_source", "edge_1235", "edge_1352"),
                getRouteIndex=lambda _actor_id: 0,
                getNextLinks=lambda _actor_id: (
                    (
                        "edge_1352_0",
                        ":node_121_0_0",
                        True,
                        True,
                        False,
                        "M",
                        "s",
                        0.25,
                    ),
                ),
            ),
            lane=SimpleNamespace(
                getEdgeID=lambda lane_id: {
                    ":node_119_3_0": ":node_119",
                    "edge_1235_0": "edge_1235",
                    ":node_121_0_0": ":node_121",
                    "edge_1352_0": "edge_1352",
                }[lane_id],
                getLinks=lambda lane_id: {
                    ":node_119_3_0": (("edge_1235_0", True, True, False, "", "M", "s", 112.981),),
                    "edge_1235_0": (
                        (
                            "edge_1352_0",
                            True,
                            True,
                            False,
                            ":node_121_0_0",
                            "M",
                            "s",
                            0.25,
                        ),
                    ),
                    ":node_121_0_0": (("edge_1352_0", True, True, False, "", "M", "s", 0.25),),
                }.get(lane_id, ()),
            ),
        )
        shapes = {
            ":node_119_3_0": ((0.0, 0.0), (1.0, 0.0)),
            "edge_1235_0": ((1.0, 0.0), (10.0, 0.0)),
            ":node_121_0_0": ((10.0, 0.0), (11.0, 0.0)),
            "edge_1352_0": ((11.0, 0.0), (20.0, 0.0)),
        }
        warnings = []
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.physics_lookahead_path_cache = {}
        plugin._physics_lane_geometry = lambda lane_id: {"shape": shapes[lane_id]}
        plugin.logger = SimpleNamespace(warning=lambda *args: warnings.append(args))

        with patch.object(cosim_inprocess, "traci", fake_traci):
            path = plugin._physics_compiled_lookahead_path("vehicle3563", ":node_119_3_0")

        self.assertEqual(path["shape_join_accepted_count"], 4)
        self.assertIsNone(path["shape_join_truncated_index"])
        self.assertAlmostEqual(path["last_end"][0], 20.0)
        self.assertAlmostEqual(path["last_end"][1], 0.0)
        self.assertEqual(warnings, [])

    def test_internal_lane_lookahead_expands_nested_via_chain(self):
        links = {
            ":node_1042_0_0": (("edge_412_0", True, True, False, "", "M", "s", 65.0),),
            "edge_412_0": (
                (
                    "edge_400_1",
                    True,
                    True,
                    False,
                    ":ia_300008_16_1",
                    "o",
                    "r",
                    52.0,
                ),
            ),
            ":ia_300008_16_1": (
                (
                    "edge_400_1",
                    True,
                    True,
                    False,
                    ":ia_300008_25_1",
                    "m",
                    "r",
                    52.0,
                ),
            ),
            ":ia_300008_25_1": (("edge_400_1", True, True, False, "", "M", "r", 29.0),),
        }
        fake_traci = SimpleNamespace(
            vehicle=SimpleNamespace(
                getLaneID=lambda _actor_id: ":node_1042_0_0",
                getRoute=lambda _actor_id: ("edge_source", "edge_412", "edge_400"),
                getRouteIndex=lambda _actor_id: 0,
                getNextLinks=lambda _actor_id: (
                    (
                        "edge_400_1",
                        True,
                        True,
                        False,
                        ":ia_300008_16_1",
                        "o",
                        "r",
                        52.0,
                    ),
                ),
            ),
            lane=SimpleNamespace(
                getEdgeID=lambda lane_id: {
                    ":node_1042_0_0": ":node_1042",
                    "edge_412_0": "edge_412",
                    ":ia_300008_16_1": ":ia_300008",
                    ":ia_300008_25_1": ":ia_300008",
                    "edge_400_1": "edge_400",
                }[lane_id],
                getLinks=lambda lane_id: links.get(lane_id, ()),
            ),
        )
        shapes = {
            ":node_1042_0_0": ((0.0, 0.0), (0.25, 0.0)),
            "edge_412_0": ((0.25, 0.0), (10.0, 0.0)),
            ":ia_300008_16_1": ((10.0, 0.0), (30.0, 0.0)),
            ":ia_300008_25_1": ((30.0, 0.0), (31.0, 0.0)),
            "edge_400_1": ((31.0, 0.0), (40.0, 0.0)),
        }
        warnings = []
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.physics_lookahead_path_cache = {}
        plugin._physics_lane_geometry = lambda lane_id: {"shape": shapes[lane_id]}
        plugin.logger = SimpleNamespace(warning=lambda *args: warnings.append(args))

        with patch.object(cosim_inprocess, "traci", fake_traci):
            path = plugin._physics_compiled_lookahead_path("AV", ":node_1042_0_0")

        self.assertEqual(
            path["lane_ids"],
            (
                ":node_1042_0_0",
                "edge_412_0",
                ":ia_300008_16_1",
                ":ia_300008_25_1",
                "edge_400_1",
            ),
        )
        self.assertEqual(path["shape_join_accepted_count"], 5)
        self.assertIsNone(path["shape_join_truncated_index"])
        self.assertAlmostEqual(path["last_end"][0], 40.0)
        self.assertEqual(warnings, [])

    def test_route_required_lane_change_uses_verified_adjacent_lane_topology(self):
        links = {
            "edge_1127_2": (("edge_1130_0", True, True, False, ":ia_300007_13_0", "M", "s", 3.0),),
            ":ia_300007_13_0": (("edge_1130_0", True, True, False, "", "M", "s", 3.0),),
            "edge_1127_1": (("edge_417_0", True, True, False, ":ia_300007_11_0", "M", "l", 3.0),),
            ":ia_300007_11_0": (("edge_417_0", True, True, False, "", "M", "l", 3.0),),
        }
        edges = {
            "edge_1127_2": "edge_1127",
            "edge_1127_1": "edge_1127",
            "edge_1130_0": "edge_1130",
            "edge_417_0": "edge_417",
            ":ia_300007_13_0": ":ia_300007",
            ":ia_300007_11_0": ":ia_300007",
        }
        lane_change_state = {"right": True}
        fake_traci = SimpleNamespace(
            constants=SimpleNamespace(LCA_LEFT=1, LCA_RIGHT=2),
            vehicle=SimpleNamespace(
                getLaneID=lambda _actor_id: "edge_1127_2",
                getRoute=lambda _actor_id: ("edge_1125", "edge_1127", "edge_417"),
                getRouteIndex=lambda _actor_id: 1,
                getNextLinks=lambda _actor_id: (
                    (
                        "edge_417_0",
                        True,
                        True,
                        False,
                        ":ia_300007_11_0",
                        "M",
                        "l",
                        3.0,
                    ),
                ),
                getLaneChangeState=lambda _actor_id, direction: (
                    0,
                    2 if direction == -1 and lane_change_state["right"] else 0,
                ),
            ),
            lane=SimpleNamespace(
                getEdgeID=lambda lane_id: edges[lane_id],
                getLinks=lambda lane_id: links.get(lane_id, ()),
            ),
        )
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.physics_lateral_direction_states = {
            "vehicle1904": {
                "phase": "active",
                "authorized_direction": "right",
            }
        }

        with patch.object(cosim_inprocess, "traci", fake_traci):
            resolution = plugin._physics_resolve_route_lane_sequence(
                "vehicle1904",
                "edge_1127_2",
            )
            lane_change_state["right"] = False
            plugin.physics_lateral_direction_states["vehicle1904"] = {
                "phase": "armed",
                "authorized_direction": "right",
            }
            waiting = plugin._physics_resolve_route_lane_sequence(
                "vehicle1904",
                "edge_1127_2",
            )
            plugin.physics_lateral_direction_states["vehicle1904"]["phase"] = "lane_keep"

        self.assertEqual(resolution["reference_lane_id"], "edge_1127_1")
        self.assertEqual(
            resolution["lane_ids"],
            ("edge_1127_1", ":ia_300007_11_0", "edge_417_0"),
        )
        self.assertNotIn(":ia_300007_13_0", resolution["lane_ids"])
        self.assertTrue(resolution["selection_reason"].startswith("intent_target_lane:"))
        self.assertEqual(resolution["error"], "")
        self.assertEqual(waiting["lane_ids"], ("edge_1127_2",))
        self.assertEqual(waiting["selection_reason"], "awaiting_required_lane_change")

    def test_lane_keep_target_ignores_sumo_lateral_velocity(self):
        path = compile_lane_shapes([((0.0, 0.0), (20.0, 0.0))])
        result = build_external_state_lateral_action_lookahead(
            path,
            (0.0, 0.0),
            7.0,
            maneuver_enabled=True,
            maneuver_target_enabled=True,
            lateral_speed=1.0,
            desired_speed=5.0,
            phase_a_position=(0.0, -0.05),
            phase_step_length=0.05,
            z=0.0,
        )

        self.assertTrue(result["valid"])
        self.assertTrue(result["lateral_motion_detected"])
        self.assertEqual(result["mode"], "route")
        self.assertEqual(result["maneuver_phase"], "lane_keep")
        self.assertEqual(result["maneuver_target_source"], "route_center")
        self.assertFalse(result["maneuver_target_applied"])
        self.assertEqual(result["lookahead"], result["route_lookahead"])
        self.assertAlmostEqual(result["target_lateral_distance"], 0.0)
        self.assertAlmostEqual(result["lateral_displacement"], 0.0)
        self.assertAlmostEqual(result["phase_b_lateral_delta"], 0.05)
        self.assertAlmostEqual(result["expected_phase_b_lateral_distance"], 0.05)
        self.assertAlmostEqual(result["world_lateral_speed"], 0.0)

    def test_route_only_lookahead_restores_off_center_origin(self):
        path = compile_lane_shapes([((0.0, 0.0), (20.0, 0.0))])
        result = build_external_state_lateral_action_lookahead(
            path,
            (1.0, 2.0),
            7.0,
            lateral_speed=0.0,
            desired_speed=5.0,
            phase_a_position=(1.0, 2.0),
            phase_step_length=0.05,
            z=0.0,
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["mode"], "route")
        self.assertEqual(result["route_lookahead"], (8.0, 0.0, 0.0))
        self.assertEqual(result["lookahead"], result["route_lookahead"])

    def test_disabled_maneuver_forces_lane_keep_with_raw_motion_evidence(self):
        path = compile_lane_shapes([((0.0, 0.0), (20.0, 0.0))])
        for phase, held_offset in (("active", None), ("hold", 1.25)):
            with self.subTest(phase=phase):
                result = build_external_state_lateral_action_lookahead(
                    path,
                    (1.0, 0.5),
                    7.0,
                    maneuver_enabled=False,
                    lateral_speed=2.0,
                    desired_speed=5.0,
                    phase_a_position=(1.0, 0.4),
                    maneuver_phase=phase,
                    held_lateral_offset=held_offset,
                )

                self.assertTrue(result["valid"])
                self.assertTrue(result["lateral_motion_detected"])
                self.assertFalse(result["maneuver_enabled"])
                self.assertEqual(result["maneuver_phase"], "lane_keep")
                self.assertEqual(result["mode"], "route")
                self.assertEqual(result["target_lateral_distance"], 0.0)
                self.assertEqual(result["lookahead"], result["route_lookahead"])

        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.physics_lateral_direction_states = {
            "vehicle0": {
                "phase": "hold",
                "held_lateral_offset": 1.25,
                "active_frames": 20,
                "zero_motion_frames": 2,
            }
        }
        state = plugin._physics_force_lane_keep_state("vehicle0", "edge_0_0")
        self.assertEqual(state["phase"], "lane_keep")
        self.assertEqual(state["source_lane_id"], "edge_0_0")
        self.assertEqual(state["target_lane_id"], "")
        self.assertIsNone(state["held_lateral_offset"])
        self.assertEqual(state["active_frames"], 0)
        self.assertEqual(state["transition_reason"], "maneuver_disabled")

    def test_active_target_uses_phase_b_canonical_d(self):
        path = compile_lane_shapes([((0.0, 0.0), (20.0, 0.0))])
        result = build_external_state_lateral_action_lookahead(
            path,
            (1.0, 0.5),
            7.0,
            maneuver_enabled=True,
            maneuver_target_enabled=True,
            phase_a_position=(1.0, 0.45),
            maneuver_phase="active",
            z=0.0,
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["route_lookahead"], (8.0, 0.0, 0.0))
        self.assertEqual(result["mode"], "active_current")
        self.assertTrue(result["maneuver_target_applied"])
        self.assertEqual(result["maneuver_target_source"], "phase_b_canonical_d")
        self.assertAlmostEqual(result["target_lateral_distance"], 0.5)
        self.assertAlmostEqual(result["lookahead"][0], 8.0)
        self.assertAlmostEqual(result["lookahead"][1], 0.5)
        self.assertAlmostEqual(result["control_lookahead"][1], 0.5)
        self.assertAlmostEqual(result["steering_preview"][1], 0.5)
        self.assertEqual(result["maneuver_target_world_anchor"], (1.0, 0.5))
        self.assertAlmostEqual(result["lateral_displacement"], 0.0)
        self.assertAlmostEqual(result["world_lateral_speed"], 0.0)

    def test_active_target_ignores_raw_lateral_inputs(self):
        path = compile_lane_shapes([((0.0, 0.0), (20.0, 0.0))])
        common = {
            "maneuver_enabled": True,
            "maneuver_target_enabled": True,
            "phase_a_position": (1.0, 0.45),
            "maneuver_phase": "active",
        }
        left_raw = build_external_state_lateral_action_lookahead(
            path,
            (1.0, 0.5),
            7.0,
            lateral_speed=2.0,
            desired_speed=0.2,
            previous_world_lateral_direction=(0.0, 1.0),
            **common,
        )
        right_raw = build_external_state_lateral_action_lookahead(
            path,
            (1.0, 0.5),
            7.0,
            lateral_speed=-2.0,
            desired_speed=20.0,
            previous_world_lateral_direction=(0.0, -1.0),
            **common,
        )

        self.assertTrue(left_raw["valid"])
        self.assertTrue(right_raw["valid"])
        for field in (
            "lookahead",
            "control_lookahead",
            "steering_preview",
            "target_lateral_distance",
            "maneuver_target_source",
        ):
            self.assertEqual(left_raw[field], right_raw[field])
        self.assertAlmostEqual(left_raw["world_lateral_speed"], 0.0)
        self.assertAlmostEqual(right_raw["world_lateral_speed"], 0.0)

    def test_state_phase_alone_selects_maneuver_target(self):
        path = compile_lane_shapes([((0.0, 0.0), (20.0, 0.0))])
        for phase, armed_origin, expected_mode, expected_d in (
            ("lane_keep", "lane_keep", "route", 0.0),
            ("armed", "lane_keep", "route", 0.0),
            ("active", "lane_keep", "active_current", 0.4),
            ("hold", "hold", "hold", 0.25),
            ("armed", "hold", "armed_hold", 0.25),
        ):
            with self.subTest(phase=phase, armed_origin=armed_origin):
                result = build_external_state_lateral_action_lookahead(
                    path,
                    (1.0, 0.4),
                    0.5,
                    maneuver_enabled=True,
                    maneuver_target_enabled=True,
                    lateral_speed=-9.0,
                    desired_speed=0.01,
                    phase_a_position=(1.0, 0.35),
                    maneuver_phase=phase,
                    armed_origin_phase=armed_origin,
                    held_canonical_d=0.25,
                )

                self.assertTrue(result["valid"])
                self.assertEqual(result["mode"], expected_mode)
                self.assertAlmostEqual(result["target_lateral_distance"], expected_d)
                self.assertEqual(
                    result["maneuver_target_applied"],
                    phase in {"active", "hold"} or armed_origin == "hold",
                )

    def test_active_low_speed_uses_current_frenet_offset_without_preview(self):
        path = compile_lane_shapes([((0.0, 0.0), (20.0, 0.0))])
        result = build_external_state_lateral_action_lookahead(
            path,
            (1.0, 0.4),
            0.5,
            maneuver_enabled=True,
            maneuver_target_enabled=True,
            lateral_speed=1.0,
            desired_speed=0.2,
            phase_a_position=(1.0, 0.35),
            maneuver_phase="active",
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["mode"], "active_current")
        self.assertAlmostEqual(result["current_lateral_distance"], 0.4)
        self.assertAlmostEqual(result["target_lateral_distance"], 0.4)
        self.assertAlmostEqual(result["lateral_displacement"], 0.0)

    def test_active_curve_offset_uses_lookahead_point_normal(self):
        path = compile_lane_shapes([((0.0, 0.0), (5.0, 0.0), (5.0, 5.0))])
        result = build_external_state_lateral_action_lookahead(
            path,
            (4.0, 0.5),
            2.0,
            maneuver_enabled=True,
            maneuver_target_enabled=True,
            lateral_speed=0.0,
            desired_speed=5.0,
            phase_a_position=(4.0, 0.5),
            maneuver_phase="active",
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["mode"], "active_current")
        self.assertEqual(result["maneuver_target_source"], "phase_b_canonical_d")
        target_d = result["canonical_phase_b"]["d"]
        self.assertAlmostEqual(result["target_lateral_distance"], target_d)
        target_frame = sample_compiled_route_frame(path, result["canonical_target_s"])
        self.assertAlmostEqual(
            result["lookahead"][0],
            target_frame["x"] + target_d * target_frame["normal_x"],
        )
        self.assertAlmostEqual(
            result["lookahead"][1],
            target_frame["y"] + target_d * target_frame["normal_y"],
        )
        self.assertAlmostEqual(result["target_world_left_normal_x"], target_frame["normal_x"])
        self.assertAlmostEqual(result["target_world_left_normal_y"], target_frame["normal_y"])

    def test_canonical_route_projection_has_signed_d_and_continuous_normal(self):
        path = compile_lane_shapes([((0.0, 0.0), (5.0, 0.0), (5.0, 5.0))], lane_ids=("edge_0_0",))
        left = project_position_to_compiled_route(path, (4.0, 0.5))
        right = project_position_to_compiled_route(path, (4.0, -0.5))

        self.assertEqual(path["lane_ids"], ("edge_0_0",))
        self.assertAlmostEqual(left["s"], 4.0)
        self.assertAlmostEqual(left["d"], 0.4021523246407367)
        self.assertAlmostEqual(right["d"], -0.4021523246407367)
        before = sample_compiled_route_frame(path, 5.0 - 1e-6)
        after = sample_compiled_route_frame(path, 5.0 + 1e-6)
        normal_alignment = (
            before["normal_x"] * after["normal_x"] + before["normal_y"] * after["normal_y"]
        )
        self.assertGreater(normal_alignment, 1.0 - 1e-10)

    def test_endpoint_projection_separates_lateral_d_from_along_residual(self):
        path = compile_lane_shapes([((0.0, 0.0), (10.0, 0.0))])
        point = (-0.4688, 0.0177)
        projection = project_position_to_compiled_route(path, point)

        self.assertAlmostEqual(projection["s"], 0.0)
        self.assertAlmostEqual(projection["d"], 0.0177)
        self.assertAlmostEqual(projection["projection_along_residual"], -0.4688)
        self.assertAlmostEqual(projection["projection_distance"], math.hypot(0.4688, 0.0177))
        self.assertTrue(projection["projection_endpoint_clamped"])

    def test_canonical_route_rejects_disconnected_reverse_and_non_finite_geometry(self):
        disconnected = compile_lane_shapes([((0.0, 0.0), (1.0, 0.0)), ((2.0, 0.0), (3.0, 0.0))])
        self.assertIsNotNone(disconnected)
        result = build_external_state_lateral_action_lookahead(
            disconnected,
            (0.5, 0.0),
            0.5,
            phase_a_position=(0.5, 0.0),
            phase_a_rear_axle_position=(-3.4, 0.0),
            front_to_control_offset=3.9,
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["error"], "disconnected_route_geometry")
        self.assertIsNone(compile_lane_shapes([((0.0, 0.0), (1.0, 0.0), (0.01, 0.001))]))
        self.assertIsNone(compile_lane_shapes([((0.0, 0.0), (math.nan, 0.0))]))

    def test_front_target_and_rear_axle_control_target_share_canonical_frame(self):
        path = compile_lane_shapes([((0.0, 0.0), (5.0, 0.0), (5.0, 5.0))])
        result = build_external_state_lateral_action_lookahead(
            path,
            (4.0, 0.0),
            2.0,
            phase_a_position=(3.95, 0.0),
            phase_a_rear_axle_position=(0.05, 0.0),
            front_to_control_offset=3.9,
        )

        self.assertTrue(result["valid"])
        self.assertIsNotNone(result["canonical_phase_a"])
        self.assertIsNotNone(result["canonical_phase_b"])
        self.assertAlmostEqual(
            result["canonical_delta_d"],
            result["canonical_phase_b"]["d"] - result["canonical_phase_a"]["d"],
        )
        front = result["lookahead"]
        control = result["control_lookahead"]
        preview = result["steering_preview"]
        self.assertIsNotNone(preview)
        self.assertAlmostEqual(result["steering_preview_effective_distance"], 2.8)
        self.assertAlmostEqual(math.dist(front[:2], control[:2]), 3.9)
        target_tangent = (
            result["target_route_tangent_x"],
            result["target_route_tangent_y"],
        )
        self.assertAlmostEqual(front[0] - control[0], 3.9 * target_tangent[0])
        self.assertAlmostEqual(front[1] - control[1], 3.9 * target_tangent[1])
        self.assertTrue(result["control_reference_valid"])
        self.assertTrue(result["control_target_ahead"])

    def test_speed_adaptive_steering_preview_distance_and_three_targets(self):
        path = compile_lane_shapes(
            [((0.0, 0.0), (40.0, 0.0), (50.0, 10.0))],
            lane_ids=("edge_0_0",),
        )
        expected = (
            (0.0, 0.0, 2.8),
            (5.0, 2.5, 2.8),
            (8.0, 4.0, 4.0),
            (10.0, 5.0, 5.0),
            (20.0, 10.0, 5.0),
        )
        for speed, requested, effective in expected:
            with self.subTest(speed=speed):
                result = build_external_state_lateral_action_lookahead(
                    path,
                    (20.0, 0.0),
                    0.5,
                    phase_a_position=(19.95, 0.0),
                    phase_a_rear_axle_position=(16.05, 0.0),
                    phase_a_speed=speed,
                    front_to_control_offset=3.9,
                )
                self.assertTrue(result["valid"])
                self.assertEqual(result["target_lateral_distance"], 0.0)
                self.assertEqual(result["steering_preview_front_d"], 0.0)
                self.assertAlmostEqual(result["steering_preview_requested_distance"], requested)
                self.assertAlmostEqual(result["steering_preview_effective_distance"], effective)
                self.assertAlmostEqual(result["canonical_target_s"], 20.5)
                self.assertAlmostEqual(result["steering_preview_front_s"], 20.0 + effective)
                self.assertAlmostEqual(
                    math.dist(result["lookahead"][:2], result["control_lookahead"][:2]),
                    3.9,
                )
                self.assertTrue(result["front_rear_distance_consistent"])
                self.assertLessEqual(result["front_rear_distance_error"], 1e-12)
                preview_front = sample_compiled_route_frame(
                    path, result["steering_preview_front_s"]
                )
                self.assertAlmostEqual(
                    result["steering_preview"][0],
                    preview_front["x"] - 3.9 * preview_front["tangent_x"],
                )
                self.assertAlmostEqual(
                    result["steering_preview"][1],
                    preview_front["y"] - 3.9 * preview_front["tangent_y"],
                )
                self.assertTrue(result["control_target_ahead"])
                self.assertTrue(result["steering_preview_target_ahead"])

    def test_steering_preview_rejects_invalid_configuration_and_path_end_clamp(self):
        path = compile_lane_shapes([((0.0, 0.0), (10.0, 0.0))])
        invalid_settings = build_external_state_lateral_action_lookahead(
            path,
            (2.0, 0.0),
            0.5,
            phase_a_position=(1.95, 0.0),
            phase_a_rear_axle_position=(-1.95, 0.0),
            front_to_control_offset=3.9,
            steering_preview_min_distance=5.0,
            steering_preview_max_distance=2.8,
        )
        self.assertFalse(invalid_settings["valid"])
        self.assertEqual(invalid_settings["error"], "invalid_steering_preview_configuration")

        path_end = build_external_state_lateral_action_lookahead(
            path,
            (8.0, 0.0),
            0.5,
            phase_a_position=(7.95, 0.0),
            phase_a_rear_axle_position=(4.05, 0.0),
            front_to_control_offset=3.9,
        )
        self.assertFalse(path_end["valid"])
        self.assertEqual(path_end["error"], "steering_preview_path_end")
        self.assertIsNone(path_end["steering_preview"])

    def test_hold_latches_canonical_d_not_raw_poslat_and_reprojects_world_anchor(self):
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.logger = SimpleNamespace(warning=lambda *args: None, info=lambda *args: None)
        plugin.physics_lateral_direction_states = {
            "vehicle0": {
                "phase": "active",
                "source_lane_id": "edge_0_0",
                "current_lane_id": "edge_0_0",
                "target_lane_id": "edge_1_0",
            }
        }
        stopped = self._maneuver_evidence_action(current_d=0.6)
        for frame in range(1, 7):
            state, _ = plugin._physics_transition_maneuver_state(
                "vehicle0",
                phase_b_time=frame * 0.05,
                current_lane_id="edge_1_0",
                current_lateral_offset=-1.575,
                lateral_speed=0.0,
                lane_change_intent="none",
                action=stopped,
                current_position=(10.0, 0.6),
                canonical_path_key=("edge_0_0", "edge_1_0"),
            )
            if frame <= 5:
                self.assertEqual(state["phase"], "active")

        self.assertEqual(state["phase"], "hold")
        self.assertEqual(state["zero_motion_frames"], 5)
        self.assertAlmostEqual(state["last_raw_lateral_offset"], -1.575)
        self.assertAlmostEqual(state["held_canonical_d"], 0.6)
        self.assertAlmostEqual(state["held_lateral_offset"], 0.6)
        self.assertEqual(state["held_world_anchor"], (10.0, 0.6))

        new_path = compile_lane_shapes([((0.0, 1.0), (30.0, 1.0))], lane_ids=("edge_1_0",))
        reprojected = build_external_state_lateral_action_lookahead(
            new_path,
            (10.0, 0.6),
            0.5,
            maneuver_enabled=True,
            maneuver_target_enabled=True,
            phase_a_position=(10.0, 0.6),
            phase_a_rear_axle_position=(6.1, 0.6),
            front_to_control_offset=3.9,
            maneuver_phase="hold",
            held_canonical_d=state["held_canonical_d"],
            held_world_anchor=state["held_world_anchor"],
            canonical_path_key=("edge_1_0",),
            canonical_path_switched=True,
            lane_width=3.15,
        )
        self.assertTrue(reprojected["valid"])
        self.assertTrue(reprojected["held_reprojected"])
        self.assertAlmostEqual(reprojected["held_canonical_d"], -0.4)
        self.assertAlmostEqual(reprojected["target_lateral_distance"], -0.4)
        self.assertAlmostEqual(reprojected["lookahead"][1], 0.6)

    def test_canonical_path_switch_jump_and_previous_target_fallback(self):
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.physics_lateral_direction_states = {}
        state = plugin._physics_maneuver_state("vehicle0", "edge_a_0")
        first_path = compile_lane_shapes([((0.0, 0.0), (30.0, 0.0))], lane_ids=("edge_a_0",))
        first = build_external_state_lateral_action_lookahead(
            first_path,
            (10.0, 0.0),
            7.0,
            phase_a_position=(10.0, 0.0),
            phase_a_rear_axle_position=(6.1, 0.0),
            front_to_control_offset=3.9,
            canonical_path_key=("edge_a_0",),
        )
        first = plugin._physics_validate_canonical_action(
            state,
            first,
            current_position=(10.0, 0.0),
            phase_a_rear_axle_position=(6.1, 0.0),
            canonical_path_key=("edge_a_0",),
            canonical_path_switched=False,
        )
        self.assertTrue(first["valid"])

        second_path = compile_lane_shapes([((10.0, 0.0), (40.0, 0.0))], lane_ids=("edge_b_0",))
        second = build_external_state_lateral_action_lookahead(
            second_path,
            (10.05, 0.0),
            7.0,
            phase_a_position=(10.0, 0.0),
            phase_a_rear_axle_position=(6.15, 0.0),
            front_to_control_offset=3.9,
            canonical_path_key=("edge_b_0",),
            canonical_path_switched=True,
        )
        second = plugin._physics_validate_canonical_action(
            state,
            second,
            current_position=(10.05, 0.0),
            phase_a_rear_axle_position=(6.15, 0.0),
            canonical_path_key=("edge_b_0",),
            canonical_path_switched=True,
        )
        self.assertTrue(second["valid"])
        self.assertAlmostEqual(state["canonical_front_jump"], 0.0)
        self.assertAlmostEqual(state["canonical_control_jump"], 0.0)
        self.assertAlmostEqual(state["canonical_steering_preview_jump"], 0.0)

        missing = dict(
            second,
            valid=False,
            error="transient_preview_failure",
            lookahead=None,
            control_lookahead=None,
            steering_preview=None,
        )
        fallback = plugin._physics_validate_canonical_action(
            state,
            missing,
            current_position=(10.1, 0.0),
            phase_a_rear_axle_position=(6.2, 0.0),
            canonical_path_key=("edge_b_0",),
            canonical_path_switched=False,
        )
        self.assertTrue(fallback["valid"])
        self.assertEqual(fallback["mode"], "canonical_fallback")
        self.assertTrue(state["canonical_fallback_applied"])
        self.assertEqual(state["canonical_invariant_reason"], "transient_preview_failure")
        self.assertEqual(state["canonical_fallback_reason"], "transient_preview_failure")
        self.assertEqual(state["canonical_fallback_age"], 1)

        missing_again = dict(
            second,
            valid=False,
            error="second_transient_preview_failure",
            lookahead=None,
            control_lookahead=None,
            steering_preview=None,
        )
        rejected = plugin._physics_validate_canonical_action(
            state,
            missing_again,
            current_position=(10.15, 0.0),
            phase_a_rear_axle_position=(6.25, 0.0),
            canonical_path_key=("edge_b_0",),
            canonical_path_switched=False,
        )
        self.assertFalse(rejected["valid"])
        self.assertFalse(state["canonical_fallback_applied"])

    def test_hold_path_switch_jump_over_limit_warns_and_continues(self):
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.physics_lateral_direction_states = {}
        state = plugin._physics_maneuver_state("vehicle0", "edge_a_0")
        state["phase"] = "hold"
        first_path = compile_lane_shapes([((0.0, 0.0), (30.0, 0.0))], lane_ids=("edge_a_0",))
        first = build_external_state_lateral_action_lookahead(
            first_path,
            (10.0, 0.2),
            0.5,
            maneuver_enabled=True,
            maneuver_target_enabled=True,
            phase_a_position=(10.0, 0.2),
            phase_a_rear_axle_position=(6.1, 0.2),
            front_to_control_offset=3.9,
            maneuver_phase="hold",
            held_canonical_d=0.2,
            canonical_path_key=("edge_a_0",),
        )
        first = plugin._physics_validate_canonical_action(
            state,
            first,
            current_position=(10.0, 0.2),
            phase_a_rear_axle_position=(6.1, 0.2),
            canonical_path_key=("edge_a_0",),
            canonical_path_switched=False,
        )
        self.assertTrue(first["valid"])

        shifted_path = compile_lane_shapes([((0.0, 0.2), (30.0, 0.2))], lane_ids=("edge_b_0",))
        shifted = build_external_state_lateral_action_lookahead(
            shifted_path,
            (10.05, 0.2),
            0.5,
            maneuver_enabled=True,
            maneuver_target_enabled=True,
            phase_a_position=(10.0, 0.2),
            phase_a_rear_axle_position=(6.15, 0.2),
            front_to_control_offset=3.9,
            maneuver_phase="hold",
            held_canonical_d=0.2,
            canonical_path_key=("edge_b_0",),
            canonical_path_switched=True,
        )
        shifted = plugin._physics_validate_canonical_action(
            state,
            shifted,
            current_position=(10.05, 0.2),
            phase_a_rear_axle_position=(6.15, 0.2),
            canonical_path_key=("edge_b_0",),
            canonical_path_switched=True,
        )

        self.assertTrue(shifted["valid"])
        self.assertEqual(shifted["error"], "")
        self.assertGreater(state["canonical_front_jump"], 0.10)
        self.assertEqual(state["canonical_invariant_warning"], "canonical_target_lateral_jump")
        self.assertTrue(state["canonical_invariant_valid"])
        self.assertFalse(state["canonical_fallback_applied"])

    def test_invalid_held_canonical_d_fails_closed(self):
        path = compile_lane_shapes([((0.0, 0.0), (30.0, 0.0))])
        result = build_external_state_lateral_action_lookahead(
            path,
            (10.0, 0.0),
            0.5,
            maneuver_enabled=True,
            maneuver_target_enabled=True,
            phase_a_position=(10.0, 0.0),
            phase_a_rear_axle_position=(6.1, 0.0),
            front_to_control_offset=3.9,
            maneuver_phase="hold",
            held_canonical_d=2.0,
            lane_width=3.0,
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["error"], "maneuver_target_d_outside_lane")

    def test_missing_non_finite_and_behind_maneuver_targets_fail_closed(self):
        path = compile_lane_shapes([((0.0, 0.0), (30.0, 0.0))])
        common = {
            "maneuver_enabled": True,
            "maneuver_target_enabled": True,
            "phase_a_position": (10.0, 0.0),
            "front_to_control_offset": 3.9,
            "lane_width": 3.15,
        }
        missing_hold = build_external_state_lateral_action_lookahead(
            path,
            (10.0, 0.0),
            0.5,
            phase_a_rear_axle_position=(6.1, 0.0),
            maneuver_phase="hold",
            **common,
        )
        non_finite_hold = build_external_state_lateral_action_lookahead(
            path,
            (10.0, 0.0),
            0.5,
            phase_a_rear_axle_position=(6.1, 0.0),
            maneuver_phase="hold",
            held_canonical_d=float("nan"),
            **common,
        )
        behind = build_external_state_lateral_action_lookahead(
            path,
            (10.0, 0.2),
            0.5,
            phase_a_rear_axle_position=(20.0, 0.2),
            maneuver_phase="active",
            **common,
        )

        self.assertFalse(missing_hold["valid"])
        self.assertEqual(missing_hold["error"], "missing_held_canonical_d")
        self.assertFalse(non_finite_hold["valid"])
        self.assertEqual(non_finite_hold["error"], "non_finite_route_input")
        self.assertFalse(behind["valid"])
        self.assertEqual(behind["error"], "control_target_behind")

    def test_active_target_outside_lane_fails_closed(self):
        path = compile_lane_shapes([((0.0, 0.0), (30.0, 0.0))])
        result = build_external_state_lateral_action_lookahead(
            path,
            (10.0, 2.0),
            0.5,
            maneuver_enabled=True,
            maneuver_target_enabled=True,
            phase_a_position=(10.0, 1.9),
            phase_a_rear_axle_position=(6.1, 1.9),
            front_to_control_offset=3.9,
            maneuver_phase="active",
            lane_width=3.0,
        )

        self.assertFalse(result["valid"])
        self.assertEqual(result["error"], "maneuver_target_d_outside_lane")
        self.assertTrue(result["maneuver_target_d_outside_lane"])

    def test_active_target_uses_source_target_lane_corridor(self):
        target_path = compile_lane_shapes([((0.0, 3.15), (30.0, 3.15))])
        reference_frame = project_position_to_compiled_route(target_path, (10.0, 3.15))
        corridor = build_active_maneuver_lane_corridor(
            reference_frame,
            ((0.0, 0.0), (30.0, 0.0)),
            3.0,
            ((0.0, 3.15), (30.0, 3.15)),
            3.0,
        )
        self.assertTrue(corridor["valid"])
        self.assertAlmostEqual(corridor["d_min"], -4.66)
        self.assertAlmostEqual(corridor["d_max"], 1.51)

        between_lanes = build_external_state_lateral_action_lookahead(
            target_path,
            (10.0, 1.4),
            0.5,
            maneuver_enabled=True,
            maneuver_target_enabled=True,
            phase_a_position=(9.6, 1.35),
            phase_a_rear_axle_position=(5.7, 1.35),
            front_to_control_offset=3.9,
            maneuver_phase="active",
            lane_width=3.0,
            active_maneuver_corridor=corridor,
        )
        self.assertTrue(between_lanes["valid"])
        self.assertTrue(between_lanes["maneuver_target_d_outside_primary_lane"])
        self.assertFalse(between_lanes["maneuver_target_d_outside_lane"])
        self.assertFalse(between_lanes["current_d_outside_lane"])
        self.assertEqual(
            between_lanes["maneuver_target_safety_region"],
            "source_target_lane_corridor",
        )

        outside_both_lanes = build_external_state_lateral_action_lookahead(
            target_path,
            (10.0, -2.0),
            0.5,
            maneuver_enabled=True,
            maneuver_target_enabled=True,
            phase_a_position=(9.6, -2.05),
            phase_a_rear_axle_position=(5.7, -2.05),
            front_to_control_offset=3.9,
            maneuver_phase="active",
            lane_width=3.0,
            active_maneuver_corridor=corridor,
        )
        self.assertFalse(outside_both_lanes["valid"])
        self.assertEqual(outside_both_lanes["error"], "maneuver_target_d_outside_lane")
        self.assertTrue(outside_both_lanes["maneuver_target_d_outside_lane"])

    def test_active_corridor_invalid_geometry_fails_closed(self):
        target_path = compile_lane_shapes([((0.0, 3.15), (30.0, 3.15))])
        reference_frame = project_position_to_compiled_route(target_path, (10.0, 3.15))
        invalid_corridor = build_active_maneuver_lane_corridor(
            reference_frame,
            ((0.0, 0.0), (30.0, 0.0)),
            3.0,
            ((0.0, 3.15), (30.0, 3.15)),
            float("nan"),
        )
        self.assertFalse(invalid_corridor["valid"])

        result = build_external_state_lateral_action_lookahead(
            target_path,
            (10.0, 1.4),
            0.5,
            maneuver_enabled=True,
            maneuver_target_enabled=True,
            phase_a_position=(9.6, 1.35),
            phase_a_rear_axle_position=(5.7, 1.35),
            front_to_control_offset=3.9,
            maneuver_phase="active",
            lane_width=3.0,
            active_maneuver_corridor=invalid_corridor,
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["error"], "active_maneuver_corridor_invalid")

    def test_runtime_active_corridor_requires_adjacent_same_edge_lanes(self):
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        state = {
            "phase": "active",
            "source_lane_id": "edge_0_0",
            "target_lane_id": "edge_0_1",
        }
        target_path = compile_lane_shapes([((0.0, 3.15), (30.0, 3.15))])
        action = {"canonical_phase_b": project_position_to_compiled_route(target_path, (10.0, 1.4))}
        lane_shapes = {
            "edge_0_0": ((0.0, 0.0), (30.0, 0.0)),
            "edge_0_1": ((0.0, 3.15), (30.0, 3.15)),
        }
        fake_lane = SimpleNamespace(
            getEdgeID=lambda lane_id: lane_id.rsplit("_", 1)[0],
            getShape=lambda lane_id: lane_shapes[lane_id],
            getWidth=lambda lane_id: 3.0,
        )
        with patch.object(cosim_inprocess.traci, "lane", fake_lane):
            corridor = plugin._physics_active_maneuver_lane_corridor(
                state,
                action,
                current_lane_id="edge_0_1",
            )
        self.assertTrue(corridor["valid"])

        state["target_lane_id"] = "edge_0_2"
        corridor = plugin._physics_active_maneuver_lane_corridor(
            state,
            action,
            current_lane_id="edge_0_0",
        )
        self.assertFalse(corridor["valid"])
        self.assertEqual(corridor["reason"], "active_corridor_lanes_not_adjacent")

    def test_active_maneuver_completes_on_verified_target_route_successor(self):
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.logger = SimpleNamespace(warning=lambda *args: None, info=lambda *args: None)
        plugin.physics_lateral_direction_states = {
            "vehicle0": {
                "phase": "active",
                "source_lane_id": "edge_31_0",
                "current_lane_id": "edge_31_1",
                "target_lane_id": "edge_31_1",
                "primary_lane_switched": True,
            }
        }
        moving = self._maneuver_evidence_action(
            canonical_delta_d=-0.0015,
            current_d=0.0,
        )

        state, _ = plugin._physics_transition_maneuver_state(
            "vehicle0",
            phase_b_time=23.7,
            current_lane_id="edge_32_2",
            current_lateral_offset=0.0,
            lateral_speed=-0.03,
            lane_change_intent="none",
            action=moving,
            current_position=(10.0, 0.0),
            canonical_path_key=(
                "3:edge_31_1",
                "4::node_987_0_2",
                "5:edge_32_2",
            ),
            canonical_lane_ids=(
                "edge_31_1",
                ":node_987_0_2",
                "edge_32_2",
            ),
        )

        self.assertEqual(state["phase"], "lane_keep")
        self.assertEqual(state["source_lane_id"], "edge_32_2")
        self.assertEqual(state["target_lane_id"], "")
        self.assertEqual(
            state["transition_reason"],
            "completed_at_lane_center_after_target_successor",
        )

        state = {
            "phase": "active",
            "source_lane_id": "edge_31_0",
            "target_lane_id": "edge_31_1",
            "primary_lane_switched": True,
        }
        self.assertFalse(
            plugin._physics_active_maneuver_reached_route_successor(
                state,
                "unverified_0",
                ("edge_31_1", "other_edge_0", "unverified_0"),
            )
        )

    def test_armed_persists_for_blocked_intent_then_confirms_real_motion(self):
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.logger = SimpleNamespace(warning=lambda *args: None, info=lambda *args: None)
        plugin.physics_lateral_direction_states = {}
        still = self._maneuver_evidence_action()

        for frame in range(316):
            state, _ = plugin._physics_transition_maneuver_state(
                "vehicle0",
                phase_b_time=1.0 + frame * 0.05,
                current_lane_id="edge_0_0",
                current_lateral_offset=0.0,
                lateral_speed=0.0,
                lane_change_intent="left",
                action=still,
            )
            self.assertEqual(state["phase"], "armed")
            self.assertEqual(state["start_candidate_frames"], 0)

        moving = self._maneuver_evidence_action(canonical_delta_d=0.0025, current_d=0.1)
        for frame in range(1, 4):
            state, _ = plugin._physics_transition_maneuver_state(
                "vehicle0",
                phase_b_time=16.8 + frame * 0.05,
                current_lane_id="edge_0_0",
                current_lateral_offset=0.1,
                lateral_speed=0.05,
                lane_change_intent="left",
                action=moving,
            )
            self.assertEqual(state["phase"], "active" if frame == 3 else "armed")

    def test_armed_cancel_same_lane_motion_and_completion_offset_fallback(self):
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.logger = SimpleNamespace(warning=lambda *args: None, info=lambda *args: None)
        plugin.physics_lateral_direction_states = {}
        still = self._maneuver_evidence_action()

        state, _ = plugin._physics_transition_maneuver_state(
            "vehicle0",
            phase_b_time=1.0,
            current_lane_id="edge_0_0",
            current_lateral_offset=0.0,
            lateral_speed=0.0,
            lane_change_intent="left",
            action=still,
        )
        self.assertEqual(state["phase"], "armed")

        for frame in range(1, 4):
            state, _ = plugin._physics_transition_maneuver_state(
                "vehicle0",
                phase_b_time=1.0 + frame * 0.05,
                current_lane_id="edge_0_0",
                current_lateral_offset=0.0,
                lateral_speed=0.0,
                lane_change_intent="none",
                action=still,
            )
            self.assertEqual(state["phase"], "armed" if frame < 3 else "lane_keep")

        moving = self._maneuver_evidence_action(canonical_delta_d=0.0025, current_d=0.3)
        for frame in range(1, 4):
            state, _ = plugin._physics_transition_maneuver_state(
                "vehicle0",
                phase_b_time=1.15 + frame * 0.05,
                current_lane_id="edge_0_0",
                current_lateral_offset=0.3,
                lateral_speed=0.05,
                lane_change_intent="none",
                action=moving,
            )
            self.assertEqual(state["phase"], "lane_keep")
            self.assertEqual(state["start_candidate_frames"], 0)
            self.assertEqual(state["transition_reason"], "uncommanded_lateral_recovery")

        for frame in range(1, 4):
            state, _ = plugin._physics_transition_maneuver_state(
                "vehicle0",
                phase_b_time=1.30 + frame * 0.05,
                current_lane_id="edge_0_0",
                current_lateral_offset=0.3,
                lateral_speed=0.05,
                lane_change_intent="left",
                action=moving,
            )
            self.assertEqual(state["phase"], "armed" if frame < 3 else "active")

        self.assertEqual(state["source_lane_id"], "edge_0_0")
        self.assertEqual(state["target_lane_id"], "edge_0_1")
        self.assertEqual(state["confirmed_direction_sign"], 1)

        settled = self._maneuver_evidence_action(current_d=0.3)
        for frame in range(1, 6):
            state, _ = plugin._physics_transition_maneuver_state(
                "vehicle0",
                phase_b_time=1.45 + frame * 0.05,
                current_lane_id="edge_0_0",
                current_lateral_offset=None,
                lateral_speed=0.0,
                lane_change_intent="none",
                action=settled,
            )
            self.assertEqual(state["phase"], "active" if frame < 5 else "hold")
        self.assertAlmostEqual(state["held_lateral_offset"], 0.3)

        plugin.physics_lateral_direction_states["vehicle_missing"] = {"phase": "active"}
        unavailable = self._maneuver_evidence_action(current_d=None)
        for frame in range(1, 6):
            state, _ = plugin._physics_transition_maneuver_state(
                "vehicle_missing",
                phase_b_time=2.0 + frame * 0.05,
                current_lane_id="edge_0_0",
                current_lateral_offset=None,
                lateral_speed=0.0,
                lane_change_intent="none",
                action=unavailable,
            )
        self.assertEqual(state["phase"], "active")
        self.assertEqual(state["transition_reason"], "completion_offset_unavailable")

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
            maneuver_enabled=True,
            maneuver_target_enabled=True,
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
        self.assertEqual(missing_phase_a["error"], "missing_phase_a_world_position")
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
                "rear_axle_position": [6.1, 1.0],
                "front_to_rear_axle_distance": 3.9,
                "source_carla_frame": 123,
            },
        )
        with patch.object(cosim_inprocess, "traci", fake_traci):
            self.assertTrue(plugin._apply_physics_external_state(command, 10.0, 1.0))
        self.assertEqual(
            [call[0] for call in calls],
            ["move", "setSpeed", "setPreviousSpeed"],
        )
        observation = plugin.physics_feedback_observations["vehicle0"]
        self.assertAlmostEqual(observation["phase_a_time"], 12.5)
        self.assertEqual(observation["source_carla_frame"], 123)
        self.assertEqual(observation["requested_lane_id"], "edge_0_0")

    def test_phase_a_assimilates_complete_carla_pose(self):
        calls = []
        fake_vehicle = _FakeVehicle(calls)
        fake_vehicle.lateral_position = 0.6
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
                "position": [10.0, 1.2],
                "z": 0.0,
                "sumo_angle": 90.0,
                "speed": 4.0,
                "acceleration": 0.5,
                "rear_axle_position": [6.1, 1.2],
                "front_to_rear_axle_distance": 3.9,
                "source_carla_frame": 123,
            },
        )
        with patch.object(cosim_inprocess, "traci", fake_traci):
            self.assertTrue(plugin._apply_physics_external_state(command, 10.0, 1.2))

        observation = plugin.physics_feedback_observations["vehicle0"]
        self.assertEqual(observation["position"], (10.0, 1.2))
        self.assertAlmostEqual(observation["sumo_assimilated_position"][0], 10.0)
        self.assertAlmostEqual(observation["sumo_assimilated_position"][1], 1.2)
        self.assertFalse(observation["lateral_reference_preserved"])
        self.assertIsNone(observation["preserved_sumo_lateral_offset"])

    def test_phase_a_advances_stopped_internal_lane_endpoint_to_successor(self):
        calls = []
        fake_vehicle = _FakeVehicle(calls)
        fake_vehicle.position = (9.99, 0.0)
        fake_vehicle.speed = 0.0
        fake_vehicle.acceleration = 0.0
        fake_vehicle.lane_id = ":junction_0_0"
        lane_shapes = {
            ":junction_0_0": ((0.0, 0.0), (10.0, 0.0)),
            "edge_1_0": ((10.0, 0.0), (20.0, 0.0)),
        }
        fake_lane = SimpleNamespace(
            getEdgeID=lambda lane_id: ":junction_0" if lane_id.startswith(":") else "edge_1",
            getShape=lambda lane_id: lane_shapes[lane_id],
            getLength=lambda lane_id: 10.0,
            getLinks=lambda _lane_id: (("edge_1_0", True, True, False, "", "M", "s", 10.0),),
        )
        fake_traci = SimpleNamespace(
            vehicle=fake_vehicle,
            lane=fake_lane,
            simulation=SimpleNamespace(getTime=lambda: 12.5),
        )
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.physics_lane_geometry_cache = {}
        plugin.physics_match_threshold = 8.0
        plugin.physics_strict_lane_hint = True
        plugin.physics_validate_external_state = True
        plugin.physics_position_tolerance = 0.001
        plugin.physics_feedback_observations = {}
        plugin.physics_internal_lane_transition_assist_enabled = True
        plugin.logger = SimpleNamespace(info=lambda *_args: None)
        command = AgentCommand(
            agent_id="vehicle0",
            agent_type="vehicle",
            command_type="set_state",
            data={
                "position": [9.99, 0.0],
                "z": 0.0,
                "sumo_angle": 90.0,
                "speed": 0.0,
                "acceleration": 0.0,
                "rear_axle_position": [6.09, 0.0],
                "front_to_rear_axle_distance": 3.9,
                "source_carla_frame": 124,
            },
        )

        with patch.object(cosim_inprocess, "traci", fake_traci):
            self.assertTrue(plugin._apply_physics_external_state(command, 9.99, 0.0))

        self.assertEqual(calls[0], ("moveTo", "vehicle0", "edge_1_0", 0.0))
        self.assertEqual(calls[1], ("move", "vehicle0", "edge_1", 0, True))
        observation = plugin.physics_feedback_observations["vehicle0"]
        self.assertEqual(observation["requested_lane_id"], "edge_1_0")
        self.assertEqual(observation["lane_id"], "edge_1_0")
        self.assertTrue(observation["internal_lane_transition_assist"])

    def test_phase_a_keeps_internal_lane_when_transition_assist_is_disabled(self):
        calls = []
        fake_vehicle = _FakeVehicle(calls)
        fake_vehicle.position = (9.99, 0.0)
        fake_vehicle.speed = 0.0
        fake_vehicle.acceleration = 0.0
        fake_vehicle.lane_id = ":junction_0_0"
        lane_shapes = {
            ":junction_0_0": ((0.0, 0.0), (10.0, 0.0)),
            "edge_1_0": ((10.0, 0.0), (20.0, 0.0)),
        }
        fake_lane = SimpleNamespace(
            getEdgeID=lambda lane_id: (":junction_0" if lane_id.startswith(":") else "edge_1"),
            getShape=lambda lane_id: lane_shapes[lane_id],
            getLength=lambda _lane_id: 10.0,
            getLinks=lambda _lane_id: (("edge_1_0", True, True, False, "", "M", "s", 10.0),),
        )
        fake_traci = SimpleNamespace(
            vehicle=fake_vehicle,
            lane=fake_lane,
            simulation=SimpleNamespace(getTime=lambda: 12.5),
        )
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.physics_lane_geometry_cache = {}
        plugin.physics_match_threshold = 8.0
        plugin.physics_strict_lane_hint = True
        plugin.physics_validate_external_state = True
        plugin.physics_position_tolerance = 0.001
        plugin.physics_feedback_observations = {}
        plugin.physics_internal_lane_transition_assist_enabled = False
        command = AgentCommand(
            agent_id="vehicle0",
            agent_type="vehicle",
            command_type="set_state",
            data={
                "position": [9.99, 0.0],
                "z": 0.0,
                "sumo_angle": 90.0,
                "speed": 0.0,
                "acceleration": 0.0,
                "rear_axle_position": [6.09, 0.0],
                "front_to_rear_axle_distance": 3.9,
                "source_carla_frame": 124,
            },
        )

        with patch.object(cosim_inprocess, "traci", fake_traci):
            self.assertTrue(plugin._apply_physics_external_state(command, 9.99, 0.0))

        self.assertEqual(calls[0], ("move", "vehicle0", ":junction_0", 0, True))
        observation = plugin.physics_feedback_observations["vehicle0"]
        self.assertEqual(observation["requested_lane_id"], ":junction_0_0")
        self.assertFalse(observation["internal_lane_transition_assist"])

    def test_phase_a_does_not_advance_regular_lane_endpoint(self):
        calls = []
        fake_vehicle = _FakeVehicle(calls)
        fake_vehicle.position = (99.99, 1.0)
        fake_vehicle.speed = 0.0
        fake_vehicle.acceleration = 0.0
        fake_lane = SimpleNamespace(
            getEdgeID=lambda _lane_id: "edge_0",
            getShape=lambda _lane_id: ((0.0, 1.0), (100.0, 1.0)),
            getLength=lambda _lane_id: 100.0,
            getLinks=lambda _lane_id: (("edge_1_0", True, True, False, "", "M", "s", 10.0),),
        )
        fake_traci = SimpleNamespace(
            vehicle=fake_vehicle,
            lane=fake_lane,
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
                "position": [99.99, 1.0],
                "z": 0.0,
                "sumo_angle": 90.0,
                "speed": 0.0,
                "acceleration": 0.0,
                "rear_axle_position": [96.09, 1.0],
                "front_to_rear_axle_distance": 3.9,
                "source_carla_frame": 125,
            },
        )

        with patch.object(cosim_inprocess, "traci", fake_traci):
            self.assertTrue(plugin._apply_physics_external_state(command, 99.99, 1.0))

        self.assertEqual(calls[0], ("move", "vehicle0", "edge_0", 0, True))
        self.assertFalse(
            plugin.physics_feedback_observations["vehicle0"]["internal_lane_transition_assist"]
        )

    def test_phase_b_adapter_exports_live_route_and_world_lateral_motion(self):
        fake_vehicle = SimpleNamespace(
            getSpeedWithoutTraCI=lambda _actor_id: 5.0,
            getEmergencyDecel=lambda _actor_id: 8.0,
            getLeader=lambda _actor_id, dist: ("leader0", 12.5),
            getSpeed=lambda _actor_id: 4.5,
            getAcceleration=lambda _actor_id: -0.25,
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
        plugin.logger = SimpleNamespace(warning=lambda *args: None, info=lambda *args: None)
        plugin.physics_external_state_enabled = True
        plugin.physics_lateral_maneuver_enabled = True
        plugin.physics_lateral_maneuver_target_enabled = False
        plugin.physics_internal_lane_transition_assist_enabled = False
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
                "rear_axle_position": (5.85, -0.05),
                "front_to_control_offset": 3.9,
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

        self.assertEqual(vehicle_state["feedback_requested_lane_id"], "edge_requested_0")
        self.assertEqual(vehicle_state["feedback_observed_lane_id"], "edge_current_0")
        self.assertEqual(vehicle_state["lane_id"], "edge_current_0")
        self.assertEqual(vehicle_state["sumo_route"], ("edge_current", "edge_next"))
        self.assertEqual(vehicle_state["sumo_leader_id"], "leader0")
        self.assertAlmostEqual(vehicle_state["sumo_leader_gap"], 12.5)
        self.assertAlmostEqual(vehicle_state["sumo_leader_speed"], 4.5)
        self.assertAlmostEqual(vehicle_state["sumo_leader_acceleration"], -0.25)
        self.assertEqual(vehicle_state["sumo_leader_lane_id"], "edge_current_0")
        self.assertAlmostEqual(vehicle_state["sumo_leader_lane_position"], 10.0)
        self.assertAlmostEqual(vehicle_state["lookahead_origin_y"], 0.0)
        self.assertAlmostEqual(vehicle_state["lookahead_phase_b_lateral_delta"], 0.05)
        self.assertAlmostEqual(vehicle_state["canonical_delta_d"], 0.05)
        self.assertEqual(vehicle_state["lateral_speed"], -1.0)
        self.assertEqual(vehicle_state["external_state_maneuver_phase"], "lane_keep")
        self.assertTrue(vehicle_state["external_state_maneuver_canonical_motion_evidence"])
        self.assertEqual(vehicle_state["external_state_maneuver_start_candidate_frame_count"], 0)
        self.assertEqual(
            vehicle_state["external_state_maneuver_transition_reason"],
            "uncommanded_lateral_recovery",
        )
        self.assertFalse(vehicle_state["external_state_lateral_maneuver_target_enabled"])
        self.assertEqual(vehicle_state["lookahead_action_mode"], "route")
        self.assertAlmostEqual(vehicle_state["lookahead_target_lateral_distance"], 0.0)
        self.assertAlmostEqual(vehicle_state["lookahead_world_lateral_speed"], 0.0)
        self.assertTrue(vehicle_state["lookahead_position_valid"])

    def test_measured_lane_change_timeline_stays_active_and_holds_final_offset(self):
        lane_centers = {"edge_394_0": 0.0, "edge_394_1": 3.15}

        for intent, direction_sign, source_lane, target_lane in (
            ("left", 1.0, "edge_394_0", "edge_394_1"),
            ("right", -1.0, "edge_394_1", "edge_394_0"),
        ):
            with self.subTest(intent=intent):
                events = []
                plugin = object.__new__(TeraSimCoSimInProcessPlugin)
                plugin.logger = SimpleNamespace(
                    warning=lambda *args: events.append(("warning", args)),
                    info=lambda *args: events.append(("info", args)),
                )
                plugin.physics_lateral_direction_states = {}
                source_center = lane_centers[source_lane]
                target_center = lane_centers[target_lane]
                source_path = compile_lane_shapes([((0.0, source_center), (100.0, source_center))])
                target_path = compile_lane_shapes([((0.0, target_center), (100.0, target_center))])

                def action(path, world_y, phase_a_y, speed):
                    state = plugin._physics_maneuver_state("vehicle1593", source_lane)
                    return build_external_state_lateral_action_lookahead(
                        path,
                        (10.0, world_y),
                        0.5,
                        maneuver_enabled=True,
                        maneuver_target_enabled=False,
                        lateral_speed=speed,
                        desired_speed=5.0,
                        phase_a_position=(10.0, phase_a_y),
                        previous_world_lateral_direction=(
                            plugin._physics_previous_lateral_direction("vehicle1593")
                        ),
                        maneuver_phase=state["phase"],
                        held_lateral_offset=state.get("held_lateral_offset"),
                        held_canonical_d=state.get("held_canonical_d"),
                    )

                def target(base_action, state):
                    return apply_external_state_lateral_maneuver_target(
                        base_action,
                        maneuver_enabled=True,
                        maneuver_target_enabled=True,
                        maneuver_phase=state["phase"],
                        armed_origin_phase=state.get("armed_origin_phase", "lane_keep"),
                        held_canonical_d=state.get("held_canonical_d"),
                    )

                armed_action = action(source_path, source_center, source_center, 0.0)
                state, _ = plugin._physics_transition_maneuver_state(
                    "vehicle1593",
                    phase_b_time=1.0,
                    current_lane_id=source_lane,
                    current_lateral_offset=0.0,
                    lateral_speed=0.0,
                    lane_change_intent=intent,
                    action=armed_action,
                )
                armed_action = target(armed_action, state)
                self.assertEqual(state["phase"], "armed")
                self.assertEqual(state["target_lane_id"], target_lane)
                self.assertEqual(armed_action["mode"], "route")

                for frame in range(1, 4):
                    phase_a_y = source_center + direction_sign * 0.0025 * (frame - 1)
                    world_y = source_center + direction_sign * 0.0025 * frame
                    start_action = action(source_path, world_y, phase_a_y, direction_sign * 0.05)
                    state, _ = plugin._physics_transition_maneuver_state(
                        "vehicle1593",
                        phase_b_time=1.0 + frame * 0.05,
                        current_lane_id=source_lane,
                        current_lateral_offset=world_y - source_center,
                        lateral_speed=direction_sign * 0.05,
                        lane_change_intent=intent,
                        action=start_action,
                    )
                    start_action = target(start_action, state)
                    self.assertEqual(
                        start_action["mode"], "route" if frame < 3 else "active_current"
                    )
                    self.assertEqual(state["phase"], "armed" if frame < 3 else "active")

                self.assertTrue(start_action["lateral_motion_detected"])
                self.assertEqual(start_action["maneuver_target_source"], "phase_b_canonical_d")
                self.assertEqual(state["confirmed_direction_sign"], int(direction_sign))

                before_switch_y = source_center + direction_sign * 1.525
                before_switch_action = action(
                    source_path,
                    before_switch_y,
                    before_switch_y - direction_sign * 0.05,
                    direction_sign * 1.0,
                )
                state, _ = plugin._physics_transition_maneuver_state(
                    "vehicle1593",
                    phase_b_time=3.05,
                    current_lane_id=source_lane,
                    current_lateral_offset=direction_sign * 1.525,
                    lateral_speed=direction_sign * 1.0,
                    lane_change_intent=intent,
                    action=before_switch_action,
                )
                before_switch_action = target(before_switch_action, state)
                self.assertEqual(state["phase"], "active")

                switch_y = target_center - direction_sign * 1.575
                switch_action = action(
                    target_path,
                    switch_y,
                    switch_y - direction_sign * 0.05,
                    direction_sign * 1.0,
                )
                state, _ = plugin._physics_transition_maneuver_state(
                    "vehicle1593",
                    phase_b_time=3.10,
                    current_lane_id=target_lane,
                    current_lateral_offset=-direction_sign * 1.575,
                    lateral_speed=direction_sign * 1.0,
                    lane_change_intent=intent,
                    action=switch_action,
                )
                switch_action = target(switch_action, state)
                world_target_jump = abs(
                    switch_action["lookahead"][1] - before_switch_action["lookahead"][1]
                )
                self.assertLess(world_target_jump, 0.10)
                self.assertTrue(state["primary_lane_switched"])
                self.assertFalse(state["evidence_eligible"])
                self.assertIn("lane_switched", state["evidence_exclusion_reason"])
                self.assertEqual(state["phase"], "active")
                self.assertEqual(state["confirmed_direction_sign"], int(direction_sign))

                after_switch_y = target_center - direction_sign * 1.525
                intent_gone_action = action(
                    target_path,
                    after_switch_y,
                    after_switch_y - direction_sign * 0.05,
                    direction_sign * 1.0,
                )
                state, _ = plugin._physics_transition_maneuver_state(
                    "vehicle1593",
                    phase_b_time=3.15,
                    current_lane_id=target_lane,
                    current_lateral_offset=-direction_sign * 1.525,
                    lateral_speed=direction_sign * 1.0,
                    lane_change_intent="none",
                    action=intent_gone_action,
                )
                intent_gone_action = target(intent_gone_action, state)
                self.assertEqual(state["phase"], "active")
                self.assertEqual(
                    intent_gone_action["maneuver_target_source"], "phase_b_canonical_d"
                )

                final_offset = -direction_sign * 0.675
                final_y = target_center + final_offset
                final_motion_action = action(
                    target_path,
                    final_y,
                    final_y - direction_sign * 0.0025,
                    direction_sign * 0.05,
                )
                state, _ = plugin._physics_transition_maneuver_state(
                    "vehicle1593",
                    phase_b_time=4.50,
                    current_lane_id=target_lane,
                    current_lateral_offset=final_offset,
                    lateral_speed=direction_sign * 0.05,
                    lane_change_intent="none",
                    action=final_motion_action,
                )
                self.assertEqual(state["phase"], "active")
                final_motion_action = target(final_motion_action, state)

                for frame in range(1, 6):
                    zero_action = action(target_path, final_y, final_y, 0.0)
                    state, _ = plugin._physics_transition_maneuver_state(
                        "vehicle1593",
                        phase_b_time=4.50 + frame * 0.05,
                        current_lane_id=target_lane,
                        current_lateral_offset=final_offset,
                        lateral_speed=0.0,
                        lane_change_intent="none",
                        action=zero_action,
                    )
                    self.assertEqual(state["phase"], "active" if frame < 5 else "hold")
                    zero_action = target(zero_action, state)

                self.assertAlmostEqual(state["held_canonical_d"], final_offset)
                hold_action = target(action(target_path, final_y, final_y, 0.0), state)
                self.assertEqual(hold_action["maneuver_phase"], "hold")
                self.assertAlmostEqual(hold_action["target_lateral_distance"], final_offset)
                self.assertAlmostEqual(hold_action["lookahead"][1], final_y)
                self.assertTrue(
                    any(item[0] == "info" and "intent disappeared" in item[1][0] for item in events)
                )
                self.assertTrue(
                    any(
                        item[0] == "info"
                        and "lateral maneuver %s->%s" in item[1][0]
                        and item[1][1:3] == ("active", "hold")
                        for item in events
                    )
                )

    def test_centered_completion_returns_to_lane_keep_and_hold_can_restart(self):
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.logger = SimpleNamespace(warning=lambda *args: None, info=lambda *args: None)
        plugin.physics_lateral_direction_states = {
            "vehicle0": {
                "phase": "active",
                "source_lane_id": "edge_0_0",
                "current_lane_id": "edge_0_0",
                "target_lane_id": "edge_0_1",
                "confirmed_world_direction": (0.0, 1.0),
                "confirmed_direction_sign": 1,
            }
        }
        zero_action = self._maneuver_evidence_action()
        for frame in range(1, 7):
            state, _ = plugin._physics_transition_maneuver_state(
                "vehicle0",
                phase_b_time=1.0 + frame * 0.05,
                current_lane_id="edge_0_1",
                current_lateral_offset=0.0,
                lateral_speed=0.0,
                lane_change_intent="none",
                action=zero_action,
            )
            self.assertEqual(state["phase"], "active" if frame < 6 else "lane_keep")

        state.update(
            phase="hold",
            held_lateral_offset=0.675,
            held_canonical_d=0.675,
            held_world_anchor=(10.0, 0.675),
            current_lane_id="edge_0_1",
        )
        restart_action = self._maneuver_evidence_action(canonical_delta_d=-0.0025, current_d=0.6725)
        for frame in range(1, 4):
            state, _ = plugin._physics_transition_maneuver_state(
                "vehicle0",
                phase_b_time=1.30 + frame * 0.05,
                current_lane_id="edge_0_1",
                current_lateral_offset=0.6725,
                lateral_speed=-0.05,
                lane_change_intent="right",
                action=restart_action,
            )
            self.assertEqual(state["phase"], "armed" if frame < 3 else "active")
            if frame < 3:
                self.assertAlmostEqual(state["held_canonical_d"], 0.675)

        self.assertEqual(state["transition_reason"], "canonical_motion_confirmed")
        self.assertEqual(state["confirmed_direction_sign"], -1)
        self.assertIsNone(state["held_canonical_d"])

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

    def test_auxiliary_evidence_spike_and_direction_flips_never_confirm_active(self):
        events = []
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.logger = SimpleNamespace(
            warning=lambda *args: events.append(("warning", args)),
            info=lambda *args: events.append(("info", args)),
        )
        plugin.physics_lateral_direction_states = {}
        still = self._maneuver_evidence_action()

        for frame in range(1, 101):
            state, conflict = plugin._physics_transition_maneuver_state(
                "auxiliary",
                phase_b_time=frame * 0.05,
                current_lane_id="edge_125_0",
                current_lateral_offset=0.0,
                lateral_speed=0.95,
                lane_change_intent="right",
                action=still,
            )
            self.assertEqual(state["phase"], "armed")
            self.assertEqual(state["start_candidate_frames"], 0)
            self.assertFalse(conflict)

        spike = self._maneuver_evidence_action(canonical_delta_d=0.0025)
        state, _ = plugin._physics_transition_maneuver_state(
            "spike",
            phase_b_time=1.0,
            current_lane_id="edge_125_0",
            current_lateral_offset=0.0,
            lateral_speed=0.0,
            lane_change_intent="none",
            action=spike,
        )
        self.assertEqual(state["phase"], "lane_keep")
        self.assertEqual(state["transition_reason"], "uncommanded_lateral_recovery")
        for frame in range(1, 4):
            state, _ = plugin._physics_transition_maneuver_state(
                "spike",
                phase_b_time=1.0 + frame * 0.05,
                current_lane_id="edge_125_0",
                current_lateral_offset=0.0,
                lateral_speed=0.0,
                lane_change_intent="none",
                action=still,
            )
        self.assertEqual(state["phase"], "lane_keep")

        for frame, delta_d in enumerate((0.0025, -0.0025) * 5, start=1):
            state, _ = plugin._physics_transition_maneuver_state(
                "alternating",
                phase_b_time=frame * 0.05,
                current_lane_id="edge_125_0",
                current_lateral_offset=0.0,
                lateral_speed=0.05,
                lane_change_intent="none",
                action=self._maneuver_evidence_action(canonical_delta_d=delta_d),
            )
            self.assertEqual(state["phase"], "lane_keep")
            self.assertEqual(state["start_candidate_frames"], 0)

        self.assertEqual([item for item in events if item[0] == "warning"], [])

    def test_maneuver_state_does_not_change_route_target_until_target_flag_enabled(self):
        path = compile_lane_shapes([((0.0, 0.0), (30.0, 0.0))])
        for phase in ("armed", "active", "hold"):
            with self.subTest(phase=phase):
                result = build_external_state_lateral_action_lookahead(
                    path,
                    (10.0, 0.4),
                    0.5,
                    maneuver_enabled=True,
                    maneuver_target_enabled=False,
                    lateral_speed=1.0,
                    desired_speed=5.0,
                    phase_a_position=(10.0, 0.35),
                    phase_a_rear_axle_position=(6.1, 0.35),
                    front_to_control_offset=3.9,
                    maneuver_phase=phase,
                    held_canonical_d=0.4,
                )
                self.assertTrue(result["valid"])
                self.assertEqual(result["mode"], "route")
                self.assertAlmostEqual(result["target_lateral_distance"], 0.0)
                self.assertAlmostEqual(result["lookahead"][1], 0.0)
                self.assertAlmostEqual(result["canonical_delta_d"], 0.05)
                self.assertTrue(result["lateral_motion_detected"])
                self.assertFalse(result["maneuver_target_enabled"])

    def test_full_pose_feedback_uses_carla_phase_a_for_maneuver_evidence(self):
        path = compile_lane_shapes([((0.0, 0.0), (30.0, 0.0))])
        result = build_external_state_lateral_action_lookahead(
            path,
            (10.0, 0.675),
            0.5,
            maneuver_enabled=True,
            maneuver_target_enabled=True,
            lateral_speed=0.0,
            phase_a_position=(10.0, 0.2),
            phase_a_evidence_position=(10.0, 0.2),
            phase_a_rear_axle_position=(6.1, 0.2),
            phase_a_speed=5.0,
            front_to_control_offset=3.9,
            maneuver_phase="active",
        )

        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["canonical_phase_a"]["d"], 0.2)
        self.assertAlmostEqual(result["canonical_evidence_phase_a"]["d"], 0.2)
        self.assertAlmostEqual(result["canonical_delta_d"], 0.475)
        self.assertTrue(result["lateral_motion_detected"])
        self.assertAlmostEqual(result["path_tracker_front_cross_track_error"], -0.475)

    def test_active_moving_heading_uses_same_step_phase_a_to_b_motion(self):
        path = compile_lane_shapes([((0.0, 0.0), (30.0, 0.0))])
        for lateral_speed in (-1.0, 1.0):
            with self.subTest(lateral_speed=lateral_speed):
                phase_b = (10.4, 0.05 * lateral_speed)
                result = build_external_state_lateral_action_lookahead(
                    path,
                    phase_b,
                    0.5,
                    maneuver_enabled=True,
                    maneuver_target_enabled=True,
                    lateral_speed=lateral_speed,
                    phase_a_position=(10.0, 0.0),
                    phase_a_evidence_position=(10.0, 0.0),
                    phase_a_rear_axle_position=(6.1, 0.0),
                    phase_a_speed=8.0,
                    front_to_control_offset=3.9,
                    phase_step_length=0.05,
                    maneuver_phase="active",
                )

                self.assertTrue(result["valid"])
                self.assertAlmostEqual(result["phase_a_to_b_requested_longitudinal_speed"], 8.0)
                self.assertAlmostEqual(
                    result["phase_a_to_b_requested_lateral_speed"], lateral_speed
                )
                expected_heading = math.atan2(lateral_speed, 8.0)
                self.assertAlmostEqual(
                    result["phase_a_to_b_motion_heading_radians"], expected_heading
                )
                self.assertTrue(result["path_tracker_motion_heading_applied"])
                self.assertEqual(
                    result["path_tracker_reference_heading_source"],
                    "phase_a_to_b_motion",
                )
                self.assertAlmostEqual(
                    math.atan2(
                        result["path_tracker_reference_tangent_y"],
                        result["path_tracker_reference_tangent_x"],
                    ),
                    expected_heading,
                )

    def test_active_moving_heading_rejects_backward_motion_and_blends_low_speed(self):
        path = compile_lane_shapes([((0.0, 0.0), (30.0, 0.0))])
        common = {
            "maneuver_enabled": True,
            "maneuver_target_enabled": True,
            "lateral_speed": 1.0,
            "phase_a_evidence_position": (10.0, 0.0),
            "phase_a_rear_axle_position": (6.1, 0.0),
            "phase_a_speed": 0.0,
            "front_to_control_offset": 3.9,
            "phase_step_length": 0.05,
            "maneuver_phase": "active",
        }
        low_speed = build_external_state_lateral_action_lookahead(
            path,
            (10.01, 0.05),
            0.5,
            phase_a_position=(10.0, 0.0),
            **common,
        )
        backward = build_external_state_lateral_action_lookahead(
            path,
            (9.95, 0.05),
            0.5,
            phase_a_position=(10.0, 0.0),
            **common,
        )

        self.assertTrue(low_speed["valid"])
        self.assertFalse(low_speed["path_tracker_motion_heading_applied"])
        self.assertEqual(
            low_speed["path_tracker_motion_heading_reason"],
            "low_forward_speed_route_tangent",
        )
        self.assertFalse(backward["valid"])
        self.assertEqual(backward["error"], "backward_phase_a_to_b_motion")

    def test_path_or_lane_switch_resets_candidate_without_ending_active(self):
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.logger = SimpleNamespace(warning=lambda *args: None, info=lambda *args: None)
        plugin.physics_lateral_direction_states = {}
        moving = self._maneuver_evidence_action(canonical_delta_d=0.0025)

        for frame in range(1, 3):
            state, _ = plugin._physics_transition_maneuver_state(
                "vehicle0",
                phase_b_time=frame * 0.05,
                current_lane_id="edge_0_0",
                current_lateral_offset=0.0,
                lateral_speed=0.05,
                lane_change_intent="left",
                action=moving,
            )
        self.assertEqual(state["phase"], "armed")
        self.assertEqual(state["start_candidate_frames"], 2)

        excluded = self._maneuver_evidence_action(canonical_delta_d=0.0025, path_switched=True)
        state, _ = plugin._physics_transition_maneuver_state(
            "vehicle0",
            phase_b_time=0.15,
            current_lane_id="edge_0_0",
            current_lateral_offset=0.0,
            lateral_speed=0.05,
            lane_change_intent="none",
            action=excluded,
        )
        self.assertEqual(state["phase"], "armed")
        self.assertEqual(state["start_candidate_frames"], 0)
        self.assertFalse(state["evidence_eligible"])
        self.assertEqual(state["evidence_exclusion_reason"], "canonical_path_switched")

        for frame in range(1, 4):
            state, _ = plugin._physics_transition_maneuver_state(
                "vehicle0",
                phase_b_time=0.15 + frame * 0.05,
                current_lane_id="edge_0_0",
                current_lateral_offset=0.0,
                lateral_speed=0.05,
                lane_change_intent="none",
                action=moving,
            )
        self.assertEqual(state["phase"], "active")

        state, _ = plugin._physics_transition_maneuver_state(
            "vehicle0",
            phase_b_time=0.35,
            current_lane_id="edge_0_1",
            current_lateral_offset=-1.575,
            lateral_speed=1.0,
            lane_change_intent="none",
            action=moving,
        )
        self.assertEqual(state["phase"], "active")
        self.assertFalse(state["evidence_eligible"])
        self.assertEqual(state["evidence_exclusion_reason"], "primary_lane_switched")
        self.assertTrue(state["primary_lane_switched"])
        self.assertEqual(state["target_lane_id"], "edge_0_1")

    def test_diagnostic_warning_is_not_authoritative_action_error(self):
        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_feedback_apply_enabled = True
        cosim.ackermann_feedback_actor_ids = {"*"}
        location = cosim._resolve_sumo_lookahead_location(
            "vehicle1593",
            {
                "lookahead_action_valid": True,
                "lookahead_action_warning": "diagnostic_only",
                "lookahead_action_mode": "active_current",
                "lookahead_position_valid": True,
                "lookahead_x": 20.9,
                "lookahead_y": 0.0,
                "lookahead_z": 0.0,
                "control_lookahead_position_valid": True,
                "control_lookahead_x": 17.0,
                "control_lookahead_y": 0.0,
                "control_lookahead_z": 0.0,
                "steering_preview_position_valid": True,
                "steering_preview_x": 19.0,
                "steering_preview_y": 0.0,
                "steering_preview_z": 0.0,
            },
            [10.0, 0.0, 0.0],
            90.0,
        )

        self.assertEqual(location, [19.0, 0.0, 0.0])

        with self.assertRaisesRegex(ValueError, "missing authoritative SUMO steering preview"):
            cosim._resolve_sumo_lookahead_location(
                "vehicle1593",
                {
                    "lookahead_action_valid": True,
                    "lookahead_action_warning": ("diagnostic_only"),
                    "lookahead_action_mode": "active_current",
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
        plugin.physics_feedback_observations = {"filtered_vehicle": {"phase_a_time": 1.0}}
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
            handle = FakeHandle(SimpleNamespace(status="ticked", state={"state": "post_step"}))
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
        cosim.world = SimpleNamespace(tick=lambda: events.append("carla_tick") or next(frames))
        cosim.sync_cosim_actor_to_carla = lambda state: events.append(("apply_state", state))
        cosim._build_av_command = lambda: events.append(("build_av", cosim._last_world_frame)) or {
            "agent_id": "AV",
            "source_carla_frame": cosim._last_world_frame,
        }
        cosim._build_physics_feedback_commands = lambda: events.append(
            ("build_feedback", cosim._last_world_frame)
        ) or [
            {
                "agent_id": "vehicle0",
                "source_carla_frame": cosim._last_world_frame,
            }
        ]
        cosim._apply_physics_fail_closed_brake = lambda reason: events.append(("brake", reason))
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

    def test_master_waits_for_first_ackermann_phase_a_feedback(self):
        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_feedback_apply_enabled = True
        cosim.ackermann_physics_enabled = True
        cosim.ackermann_feedback_actor_ids = {"*"}
        cosim._physics_feedback_frames = {}

        cosim.tick_master = True
        self.assertTrue(cosim._waits_for_first_phase_a_feedback("vehicle0"))

        cosim.tick_master = False
        self.assertTrue(cosim._waits_for_first_phase_a_feedback("vehicle0"))

        cosim._physics_feedback_frames["vehicle0"] = 101
        self.assertFalse(cosim._waits_for_first_phase_a_feedback("vehicle0"))

    def test_physics_feedback_exports_rear_axle_and_front_distance(self):
        transform = carla_cosim.carla.Transform(
            carla_cosim.carla.Location(x=0.0, y=0.0, z=0.0),
            carla_cosim.carla.Rotation(yaw=0.0),
        )
        actor = SimpleNamespace(
            get_transform=lambda: transform,
            get_velocity=lambda: carla_cosim.carla.Vector3D(4.0, 0.0, 0.0),
            get_acceleration=lambda: carla_cosim.carla.Vector3D(0.5, 0.0, 0.0),
        )
        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_tuning = AckermannTuning()
        cosim.sumo_carla_offset = (0.0, 0.0)
        cosim._coord_transformer = None
        cosim._ackermann_actor_state = {
            "vehicle0": {
                "geometry_from_physics": True,
                "front_bumper_local_x_m": 2.5,
                "rear_axle_local_x_m": -1.4,
            }
        }

        feedback = cosim._carla_actor_to_sumo_feedback(
            "vehicle0", actor, {"length": 5.0, "width": 1.8, "height": 1.5}
        )

        self.assertEqual(feedback["position"], [2.5, 0.0])
        self.assertEqual(feedback["rear_axle_position"], [-1.4, 0.0])
        self.assertAlmostEqual(feedback["front_to_rear_axle_distance"], 3.9)

    def test_physics_feedback_commands_use_feature_actor_id_order(self):
        collected_actor_ids = []
        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_feedback_apply_enabled = True
        cosim.ackermann_physics_enabled = True
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
        self.assertTrue(all(command["data"]["source_carla_frame"] == 123 for command in commands))

    def test_physics_feedback_seed_waits_for_enabled_stabilizing_physics(self):
        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_feedback_apply_enabled = True
        cosim.ackermann_physics_enabled = True
        cosim.physics_radius_enabled = True
        cosim._physics_active_vehicle_ids = {"vehicle0"}
        cosim._inproc_prev_state = {"agent_details": {"vehicle": {"vehicle0": {"speed": 3.0}}}}
        cosim._last_world_frame = 100
        cosim._vehicle_actor_index = {"vehicle0": SimpleNamespace(id=7)}
        cosim._ackermann_actor_state = {
            "vehicle0": {
                "physics_initialization_pending": True,
                "physics_enabled": False,
                "physics_stabilization_pending": False,
                "physics_feedback_seed_pending": True,
                "physics_initialization_generation": 1,
            }
        }
        cosim._physics_feedback_frames = {}
        cosim._ackermann_feedback_state = {}
        cosim._is_ackermann_feedback_actor = lambda _actor_id: True
        cosim._carla_actor_to_sumo_feedback = lambda *_args: self.fail(
            "feedback must not be collected before physics is enabled"
        )

        self.assertEqual(cosim._build_physics_feedback_commands(), [])

        cosim._ackermann_actor_state["vehicle0"].update(
            {
                "physics_enabled": True,
                "physics_stabilization_pending": True,
                "physics_ground_transform_wait_pending": True,
            }
        )
        self.assertEqual(cosim._build_physics_feedback_commands(), [])

    def test_physics_feedback_seed_sends_full_pose_without_control(self):
        transitions = []
        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_feedback_apply_enabled = True
        cosim.ackermann_physics_enabled = True
        cosim.physics_radius_enabled = True
        cosim._physics_active_vehicle_ids = {"vehicle0"}
        cosim._inproc_prev_state = {
            "simulation_time": 503.35,
            "agent_details": {"vehicle": {"vehicle0": {"speed": 3.0}}},
        }
        cosim._last_world_frame = 100
        cosim._vehicle_actor_index = {"vehicle0": SimpleNamespace(id=7)}
        cosim._ackermann_actor_state = {
            "vehicle0": {
                "physics_initialization_pending": True,
                "physics_enabled": True,
                "physics_stabilization_pending": True,
                "physics_feedback_seed_pending": True,
                "physics_initialization_generation": 2,
            }
        }
        cosim._physics_feedback_frames = {}
        cosim._ackermann_feedback_state = {}
        cosim._is_ackermann_feedback_actor = lambda _actor_id: True
        cosim._carla_actor_to_sumo_feedback = lambda *_args: {
            "position": [10.0, 1.0],
            "sumo_angle": 90.0,
        }
        cosim._clear_physics_feedback_failure = lambda _actor_id: None
        cosim._record_actor_mode_transition = lambda actor_id, mode, **details: (
            transitions.append((actor_id, mode, details))
        )

        commands = cosim._build_physics_feedback_commands()

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]["command_type"], "set_state")
        self.assertTrue(commands[0]["data"]["physics_seed_only"])
        self.assertEqual(commands[0]["data"]["physics_initialization_generation"], 2)
        self.assertEqual(commands[0]["data"]["source_carla_frame"], 100)
        state = cosim._ackermann_actor_state["vehicle0"]
        self.assertEqual(state["physics_feedback_seed_sent_frames"], 1)
        self.assertEqual(state["physics_feedback_seed_first_sent_frame"], 100)
        self.assertEqual(state["physics_feedback_seed_start_frame"], 100)
        self.assertEqual(transitions[0][1], "physics_seeding")

    def test_physics_feedback_seed_resolves_active_before_first_control(self):
        for actor_id in ("vehicle776", "vehicle2387"):
            with self.subTest(actor_id=actor_id):
                cosim = object.__new__(CarlaCosim)
                cosim.ACKERMANN_SPAWN_STABILITY_TICKS = 3
                cosim.inprocess_plugin = SimpleNamespace(LATERAL_ARMED_IDLE_FRAMES=3)
                cosim._physics_feedback_frames = {}
                cosim._inproc_prev_state = {"simulation_time": 503.45}
                cosim._last_world_frame = 103
                state = {
                    "physics_initialization_generation": 4,
                    "physics_feedback_seed_pending": True,
                    "physics_feedback_seed_started_with_maneuver": False,
                    "physics_feedback_seed_acknowledged_frames": 0,
                }

                phases = ("armed", "armed", "active")
                for index, phase in enumerate(phases, start=101):
                    vehicle_info = {
                        "feedback_physics_seed_only": True,
                        "feedback_initialization_generation": 4,
                        "feedback_source_carla_frame": index,
                        "sumo_lane_change_intent": ("left" if index == 101 else "none"),
                        "lane_id": "edge_394_0",
                        "canonical_phase_b_d": 0.1 * (index - 100),
                        "external_state_maneuver_phase": phase,
                        "external_state_maneuver_intent_evidence": index == 101,
                        "external_state_maneuver_authorized": index == 101,
                        "external_state_maneuver_canonical_motion_evidence": True,
                        "external_state_maneuver_speed_motion_evidence": True,
                        "lookahead_action_valid": True,
                        "external_state_maneuver_target_selected_d": 0.3,
                    }
                    cosim._physics_feedback_frames[actor_id] = index
                    cosim._acknowledge_physics_feedback_seed(actor_id, vehicle_info, state)
                    ready, failure = cosim._physics_feedback_seed_control_gate(
                        actor_id, vehicle_info, state
                    )

                self.assertTrue(ready)
                self.assertEqual(failure, "")
                self.assertTrue(state["physics_feedback_seed_complete"])
                self.assertEqual(state["physics_feedback_seed_acknowledged_frames"], 3)
                self.assertEqual(state["physics_control_start_phase"], "active")
                self.assertAlmostEqual(state["physics_control_start_target_d"], 0.3)
                self.assertEqual(state["physics_feedback_seed_first_intent"], "left")

    def test_physics_feedback_seed_warns_and_starts_for_unauthorized_lateral_motion(self):
        cosim = object.__new__(CarlaCosim)
        cosim.ACKERMANN_SPAWN_STABILITY_TICKS = 3
        cosim.inprocess_plugin = SimpleNamespace(LATERAL_ARMED_IDLE_FRAMES=3)
        cosim._physics_feedback_frames = {"vehicle0": 103}
        cosim._inproc_prev_state = {"simulation_time": 12.5}
        cosim._last_world_frame = 103
        state = {
            "physics_initialization_generation": 1,
            "physics_feedback_seed_pending": True,
            "physics_feedback_seed_started_with_maneuver": False,
            "physics_feedback_seed_acknowledged_frames": 3,
        }
        vehicle_info = {
            "feedback_source_carla_frame": 103,
            "lookahead_action_valid": True,
            "external_state_maneuver_phase": "lane_keep",
            "external_state_maneuver_authorized": False,
            "external_state_maneuver_canonical_motion_evidence": True,
            "external_state_maneuver_speed_motion_evidence": False,
        }

        ready, failure = cosim._physics_feedback_seed_control_gate("vehicle0", vehicle_info, state)

        self.assertTrue(ready)
        self.assertEqual(failure, "")
        self.assertTrue(state["physics_feedback_seed_complete"])
        self.assertEqual(
            state["physics_feedback_seed_handoff_warning"],
            "physics_seed_uncommanded_lateral_motion",
        )
        self.assertEqual(state["physics_control_start_phase"], "lane_keep")

    def test_physics_feedback_seed_starts_control_while_intent_remains_armed(self):
        cosim = object.__new__(CarlaCosim)
        cosim.ACKERMANN_SPAWN_STABILITY_TICKS = 3
        cosim.inprocess_plugin = SimpleNamespace(LATERAL_ARMED_IDLE_FRAMES=3)
        cosim._physics_feedback_frames = {"vehicle0": 103}
        cosim._inproc_prev_state = {"simulation_time": 12.5}
        cosim._last_world_frame = 103
        state = {
            "physics_initialization_generation": 1,
            "physics_feedback_seed_pending": True,
            "physics_feedback_seed_started_with_maneuver": True,
            "physics_feedback_seed_acknowledged_frames": 3,
        }
        vehicle_info = {
            "feedback_source_carla_frame": 103,
            "lookahead_action_valid": True,
            "external_state_maneuver_phase": "armed",
            "external_state_maneuver_authorized": True,
            "external_state_maneuver_canonical_motion_evidence": False,
            "external_state_maneuver_speed_motion_evidence": False,
        }

        ready, failure = cosim._physics_feedback_seed_control_gate("vehicle0", vehicle_info, state)

        self.assertTrue(ready)
        self.assertEqual(failure, "")
        self.assertTrue(state["physics_feedback_seed_complete"])
        self.assertTrue(state["physics_feedback_seed_completed_while_armed"])
        self.assertEqual(state["physics_control_start_phase"], "armed")

    def test_feedback_generation_change_clears_maneuver_and_rolling_path(self):
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
        plugin.physics_feedback_generations = {"vehicle0": 1}
        plugin.physics_lateral_direction_states = {"vehicle0": {"phase": "active"}}
        plugin.physics_rolling_path_states = {"vehicle0": {"entries": ("stale",)}}
        plugin.logger = SimpleNamespace(info=lambda *_args: None)
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
                "rear_axle_position": [6.1, 1.0],
                "front_to_rear_axle_distance": 3.9,
                "source_carla_frame": 123,
                "physics_seed_only": True,
                "physics_initialization_generation": 2,
            },
        )

        with patch.object(cosim_inprocess, "traci", fake_traci):
            self.assertTrue(plugin._apply_physics_external_state(command, 10.0, 1.0))

        self.assertEqual(plugin.physics_feedback_generations["vehicle0"], 2)
        self.assertNotIn("vehicle0", plugin.physics_lateral_direction_states)
        self.assertNotIn("vehicle0", plugin.physics_rolling_path_states)
        observation = plugin.physics_feedback_observations["vehicle0"]
        self.assertTrue(observation["physics_seed_only"])
        self.assertEqual(observation["initialization_generation"], 2)

    def test_initialization_static_terrain_contact_is_warning_only(self):
        cosim = object.__new__(CarlaCosim)
        cosim._ackermann_actor_state = {
            "vehicle0": {
                "physics_initialization_pending": True,
                "physics_feedback_seed_pending": True,
                "physics_initialization_generation": 3,
            }
        }
        cosim._diagnostic_lock = Lock()
        cosim._collision_raw_event_count = 0
        cosim._collision_seen_frame_pairs = set()
        cosim._collision_last_pair_frame = {}
        cosim._collision_unique_frame_count = 0
        cosim._collision_episode_count = 0
        cosim._collision_episode_counts_by_pair = {}
        cosim._initialization_contact_warning_seen_frame_pairs = set()
        cosim._initialization_contact_warning_last_pair_frame = {}
        cosim._initialization_contact_warning_event_count = 0
        cosim._initialization_contact_warning_episode_count = 0
        cosim._initialization_contact_warning_counts_by_vehicle = {}
        cosim.collision_episode_gap_frames = 10
        cosim.collision_log_path = "collision.jsonl"
        payloads = []
        cosim._append_diagnostic_jsonl = lambda _path, payload: payloads.append(payload)
        cosim._diagnostic_actor_snapshot = lambda value: {"actor_id": value.id}
        cosim._diagnostic_vector = lambda _value: {"x": 0.0, "y": 0.0, "z": 0.0}
        actor = SimpleNamespace(id=7)
        terrain = SimpleNamespace(id=0, type_id="static.terrain", attributes={})

        with patch("builtins.print"):
            cosim._on_collision_event(
                "vehicle0",
                actor,
                SimpleNamespace(
                    frame=100,
                    timestamp=5.0,
                    other_actor=terrain,
                    normal_impulse=SimpleNamespace(),
                ),
            )

        self.assertEqual(cosim._collision_episode_count, 0)
        self.assertEqual(cosim._collision_unique_frame_count, 0)
        self.assertEqual(cosim._initialization_contact_warning_event_count, 1)
        self.assertEqual(cosim._initialization_contact_warning_episode_count, 1)
        self.assertEqual(cosim._initialization_contact_warning_counts_by_vehicle, {"vehicle0": 1})
        self.assertEqual(payloads[0]["event"], "carla_initialization_contact_warning")
        self.assertTrue(payloads[0]["warning_only"])
        self.assertEqual(payloads[0]["initialization_generation"], 3)

        other_vehicle = SimpleNamespace(id=8, type_id="vehicle.test", attributes={})
        with patch("builtins.print"):
            cosim._on_collision_event(
                "vehicle0",
                actor,
                SimpleNamespace(
                    frame=101,
                    timestamp=5.05,
                    other_actor=other_vehicle,
                    normal_impulse=SimpleNamespace(),
                ),
            )

        self.assertEqual(cosim._collision_episode_count, 1)
        self.assertEqual(cosim._collision_unique_frame_count, 1)
        self.assertEqual(payloads[1]["event"], "carla_collision")
        self.assertFalse(payloads[1]["warning_only"])

        cosim._ackermann_actor_state["vehicle0"]["physics_initialization_pending"] = False
        cosim._ackermann_actor_state["vehicle0"]["physics_feedback_seed_pending"] = False
        with patch("builtins.print"):
            cosim._on_collision_event(
                "vehicle0",
                actor,
                SimpleNamespace(
                    frame=120,
                    timestamp=6.0,
                    other_actor=terrain,
                    normal_impulse=SimpleNamespace(),
                ),
            )

        self.assertEqual(cosim._collision_episode_count, 2)
        self.assertEqual(cosim._collision_unique_frame_count, 2)
        self.assertEqual(payloads[2]["event"], "carla_collision")
        self.assertFalse(payloads[2]["warning_only"])

    def test_remove_command_deletes_only_background_sumo_vehicle_and_clears_state(self):
        removed = []
        fake_traci = SimpleNamespace(
            vehicle=SimpleNamespace(remove=lambda actor_id: removed.append(actor_id))
        )
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.logger = SimpleNamespace(
            error=lambda *_args: None,
            warning=lambda *_args: None,
        )
        plugin.controlled_agents_each_step = set()
        plugin.physics_feedback_observations = {"vehicle0": {}}
        plugin.physics_feedback_actor_membership = {"vehicle0"}
        plugin.physics_lateral_direction_states = {"vehicle0": {}}
        plugin.physics_rolling_path_states = {"vehicle0": {}}
        plugin.physics_feedback_generations = {"vehicle0": 2}
        command = AgentCommand(
            agent_id="vehicle0",
            agent_type="vehicle",
            command_type="remove",
            data={
                "reason": "vehicle_collision",
                "collision_frame": 100,
                "sumo_lane_id": "edge_1_0",
                "carla_location": {"x": 1.0, "y": 2.0, "z": 0.0},
            },
        )

        with patch.object(cosim_inprocess, "traci", fake_traci):
            self.assertTrue(plugin._apply_agent_command(command))
            self.assertFalse(
                plugin._apply_agent_command(command.model_copy(update={"agent_id": "AV"}))
            )

        self.assertEqual(removed, ["vehicle0"])
        self.assertNotIn("vehicle0", plugin.physics_feedback_observations)
        self.assertNotIn("vehicle0", plugin.physics_feedback_actor_membership)
        self.assertNotIn("vehicle0", plugin.physics_lateral_direction_states)
        self.assertNotIn("vehicle0", plugin.physics_rolling_path_states)
        self.assertNotIn("vehicle0", plugin.physics_feedback_generations)

    def test_background_feedback_error_warns_for_demotion_instead_of_aborting_tick(self):
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        warnings = []
        plugin.logger = SimpleNamespace(
            error=lambda *_args: None,
            warning=lambda *args: warnings.append(args),
            isEnabledFor=lambda _level: False,
        )
        plugin.controlled_agents_each_step = set()
        plugin.physics_feedback_observations = {"vehicle0": {"stale": True}}
        plugin.physics_background_invalid_policy = "demote"
        plugin._is_physics_feedback_actor = lambda actor_id: actor_id in {"vehicle0", "AV"}
        plugin._apply_physics_external_state = lambda *_args: (_ for _ in ()).throw(
            RuntimeError("current-lane mapping failed")
        )
        background_command = AgentCommand(
            agent_id="vehicle0",
            agent_type="vehicle",
            command_type="set_state",
            data={"position": [1.0, 2.0], "source_carla_frame": 100},
        )

        self.assertFalse(plugin._apply_agent_command(background_command))
        self.assertNotIn("vehicle0", plugin.physics_feedback_observations)
        self.assertTrue(warnings)

        plugin.controlled_agents_each_step.clear()
        av_command = background_command.model_copy(update={"agent_id": "AV"})
        with self.assertRaisesRegex(RuntimeError, "current-lane mapping failed"):
            plugin._apply_agent_command(av_command)

    def test_vehicle_collision_remove_policy_is_warning_not_formal_collision(self):
        cosim = object.__new__(CarlaCosim)
        cosim.background_collision_policy = "remove"
        cosim.args = SimpleNamespace(protected_roles=["AV"])
        cosim._ackermann_actor_state = {}
        cosim._diagnostic_lock = Lock()
        cosim._collision_raw_event_count = 0
        cosim._collision_seen_frame_pairs = set()
        cosim._collision_last_pair_frame = {}
        cosim._collision_unique_frame_count = 0
        cosim._collision_episode_count = 0
        cosim._collision_episode_counts_by_pair = {}
        cosim._initialization_contact_warning_seen_frame_pairs = set()
        cosim._initialization_contact_warning_last_pair_frame = {}
        cosim._initialization_contact_warning_event_count = 0
        cosim._initialization_contact_warning_episode_count = 0
        cosim._initialization_contact_warning_counts_by_vehicle = {}
        cosim._collision_recovery_warning_seen_frame_pairs = set()
        cosim._collision_recovery_warning_last_pair_frame = {}
        cosim._collision_recovery_warning_event_count = 0
        cosim._collision_recovery_warning_episode_count = 0
        cosim._collision_recovery_warning_counts_by_vehicle = {}
        cosim._pending_collision_vehicle_removals = {}
        cosim._collision_vehicle_removal_tombstones = {}
        cosim.collision_episode_gap_frames = 10
        cosim.collision_log_path = "collision.jsonl"
        first = SimpleNamespace(id=7, type_id="vehicle.test", attributes={"role_name": "vehicle0"})
        second = SimpleNamespace(id=8, type_id="vehicle.test", attributes={"role_name": "vehicle1"})
        cosim._vehicle_actor_index = {"vehicle0": first, "vehicle1": second}
        cosim._inproc_prev_state = {
            "simulation_time": 510.0,
            "agent_details": {
                "vehicle": {
                    "vehicle0": {"lane_id": "edge_1_0", "x": 10.0, "y": 20.0},
                    "vehicle1": {"lane_id": "edge_1_1", "x": 11.0, "y": 20.0},
                }
            },
        }
        payloads = []
        cosim._append_diagnostic_jsonl = lambda _path, payload: payloads.append(payload)
        cosim._diagnostic_actor_snapshot = lambda actor: {
            "actor_id": actor.id,
            "transform": {"location": {"x": float(actor.id), "y": 2.0, "z": 0.0}},
        }
        cosim._diagnostic_vector = lambda _value: {"x": 0.0, "y": 0.0, "z": 0.0}

        with patch("builtins.print"):
            cosim._on_collision_event(
                "vehicle0",
                first,
                SimpleNamespace(
                    frame=100,
                    timestamp=5.0,
                    other_actor=second,
                    normal_impulse=SimpleNamespace(),
                ),
            )

        self.assertEqual(cosim._collision_episode_count, 0)
        self.assertEqual(cosim._collision_recovery_warning_episode_count, 1)
        self.assertEqual(
            set(cosim._pending_collision_vehicle_removals),
            {"vehicle0", "vehicle1"},
        )
        self.assertEqual(payloads[0]["event"], "carla_collision_recovery_warning")
        self.assertTrue(payloads[0]["warning_only"])

        terrain = SimpleNamespace(id=0, type_id="static.terrain", attributes={})
        self.assertEqual(
            set(cosim._collision_deletable_background_actors("vehicle0", first, terrain)),
            {"vehicle0"},
        )
        cosim._pending_collision_vehicle_removals.clear()
        cosim._ackermann_actor_state = {"vehicle0": {"physics_initialization_pending": True}}
        with patch("builtins.print"):
            cosim._on_collision_event(
                "vehicle0",
                first,
                SimpleNamespace(
                    frame=101,
                    timestamp=5.05,
                    other_actor=terrain,
                    normal_impulse=SimpleNamespace(),
                ),
            )
        self.assertEqual(cosim._pending_collision_vehicle_removals, {})
        self.assertEqual(cosim._initialization_contact_warning_episode_count, 1)

        cosim._ackermann_actor_state["vehicle0"] = {}
        with patch("builtins.print"):
            cosim._on_collision_event(
                "vehicle0",
                first,
                SimpleNamespace(
                    frame=102,
                    timestamp=5.10,
                    other_actor=terrain,
                    normal_impulse=SimpleNamespace(),
                ),
            )
        self.assertEqual(set(cosim._pending_collision_vehicle_removals), {"vehicle0"})
        self.assertEqual(cosim._collision_episode_count, 0)
        self.assertEqual(cosim._collision_recovery_warning_episode_count, 2)

    def test_vehicle_collision_schedules_background_but_never_av_removal(self):
        cosim = object.__new__(CarlaCosim)
        cosim.background_collision_policy = "remove"
        cosim.args = SimpleNamespace(protected_roles=["AV"])
        cosim._diagnostic_lock = Lock()
        cosim._pending_collision_vehicle_removals = {}
        cosim._collision_vehicle_removal_tombstones = {}
        cosim._inproc_prev_state = {
            "simulation_time": 510.0,
            "agent_details": {
                "vehicle": {
                    "vehicle0": {"lane_id": "edge_1_0", "x": 10.0, "y": 20.0},
                    "AV": {"lane_id": "edge_1_1", "x": 11.0, "y": 20.0},
                }
            },
        }
        background = SimpleNamespace(
            id=7, type_id="vehicle.test", attributes={"role_name": "vehicle0"}
        )
        av = SimpleNamespace(id=8, type_id="vehicle.test", attributes={"role_name": "AV"})
        cosim._vehicle_actor_index = {"vehicle0": background, "AV": av}
        cosim._diagnostic_actor_snapshot = lambda actor: {
            "transform": {"location": {"x": float(actor.id), "y": 2.0, "z": 0.0}}
        }
        payloads = []
        cosim.collision_log_path = "collision.jsonl"
        cosim._append_diagnostic_jsonl = lambda _path, payload: payloads.append(payload)

        with patch("builtins.print"):
            cosim._schedule_collision_vehicle_removals(
                "vehicle0",
                background,
                av,
                {
                    "warning_only": False,
                    "new_episode": True,
                    "pair_key": "7:8",
                    "carla_frame": 100,
                },
            )

        self.assertEqual(set(cosim._pending_collision_vehicle_removals), {"vehicle0"})
        record = cosim._pending_collision_vehicle_removals["vehicle0"]
        self.assertEqual(record["sumo_lane_id"], "edge_1_0")
        self.assertEqual(record["carla_location"], {"x": 7.0, "y": 2.0, "z": 0.0})
        self.assertEqual(payloads[0]["event"], "collision_vehicle_removal_scheduled")

    def test_collision_removal_destroys_carla_actor_and_waits_for_sumo_absence(self):
        destroyed = []
        actor = SimpleNamespace(id=7, destroy=lambda: destroyed.append(7))
        cosim = object.__new__(CarlaCosim)
        cosim.background_collision_policy = "remove"
        cosim._diagnostic_lock = Lock()
        cosim._pending_collision_vehicle_removals = {
            "vehicle0": {
                "actor_id": "vehicle0",
                "collision_frame": 100,
                "sumo_lane_id": "edge_1_0",
                "carla_location": {"x": 1.0, "y": 2.0, "z": 0.0},
                "reason": "vehicle_collision",
                "remove_attempts": 0,
            }
        }
        cosim._collision_vehicle_removal_tombstones = {}
        cosim._collision_vehicle_removal_history = []
        cosim._vehicle_actor_index = {"vehicle0": actor}
        cosim._pedestrian_actor_index = {}
        cosim._collision_sensors = {}
        cosim._ackermann_actor_state = {"vehicle0": {}}
        cosim._physics_feedback_frames = {"vehicle0": 1}
        cosim._physics_feedback_failures = {}
        cosim._ackermann_feedback_state = {}
        cosim._ackermann_fail_closed_reasons = {}
        cosim._pending_background_demotions = {}
        cosim._physics_quarantined_vehicle_ids = {}
        cosim._physics_active_vehicle_ids = {"vehicle0"}
        cosim._actor_filter_active_vehicle_ids = {"vehicle0"}
        cosim.world = SimpleNamespace(get_snapshot=lambda: SimpleNamespace(frame=101))
        payloads = []
        cosim.collision_log_path = "collision.jsonl"
        cosim._append_diagnostic_jsonl = lambda _path, payload: payloads.append(payload)

        commands = cosim._build_collision_removal_commands()

        self.assertEqual(destroyed, [7])
        self.assertNotIn("vehicle0", cosim._vehicle_actor_index)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]["command_type"], "remove")
        self.assertEqual(commands[0]["agent_id"], "vehicle0")
        self.assertEqual(
            cosim._filter_collision_removal_tombstones({"vehicle0": {"x": 1.0}}),
            {},
        )
        self.assertIn("vehicle0", cosim._collision_vehicle_removal_tombstones)
        with patch("builtins.print"):
            self.assertEqual(cosim._filter_collision_removal_tombstones({}), {})
        self.assertNotIn("vehicle0", cosim._collision_vehicle_removal_tombstones)
        self.assertEqual(
            cosim._collision_vehicle_removal_history[0]["event"],
            "collision_vehicle_removed_from_sumo",
        )
        self.assertEqual(
            [payload["event"] for payload in payloads],
            ["collision_vehicle_removed_from_carla", "collision_vehicle_removed_from_sumo"],
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
        cosim._apply_physics_fail_closed_brake = lambda reason: events.append(("brake", reason))
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
        cosim._apply_physics_fail_closed_brake = lambda reason: events.append(("brake", reason))
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
        cosim.world = SimpleNamespace(tick=lambda: events.append("carla_tick") or 201)
        cosim.sync_cosim_actor_to_carla = lambda state: events.append(("apply_state", state))
        cosim._build_physics_feedback_commands = (
            lambda: events.append(("build_feedback", cosim._last_world_frame)) or []
        )

        def request_tick(_commands):
            events.append("request_sumo")
            raise RuntimeError("request rejected")

        cosim.inprocess_plugin = SimpleNamespace(tick_async=request_tick)
        cosim._apply_physics_fail_closed_brake = lambda reason: events.append(("brake", reason))
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
        cosim.world = SimpleNamespace(get_snapshot=lambda: SimpleNamespace(frame=frame["value"]))

        self.assertFalse(
            cosim._prepare_ackermann_actor_physics(actor, "vehicle0", 4.0, transform, elevated)
        )
        self.assertFalse(cosim._ensure_ackermann_actor_physics(actor, "vehicle0", 4.0, transform))
        self.assertNotIn(("physics", True), calls)

        frame["value"] = 101
        self.assertFalse(cosim._ensure_ackermann_actor_physics(actor, "vehicle0", 4.0, transform))
        self.assertIn(("physics", True), calls)

        frame["value"] = 102
        self.assertFalse(cosim._ensure_ackermann_actor_physics(actor, "vehicle0", 4.0, transform))
        frame["value"] = 103
        self.assertFalse(cosim._ensure_ackermann_actor_physics(actor, "vehicle0", 4.0, transform))
        frame["value"] = 104
        self.assertTrue(cosim._ensure_ackermann_actor_physics(actor, "vehicle0", 4.0, transform))
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
        self.assertEqual(cosim._pending_authoritative_action_error["actor_id"], "vehicle0")

        calls = []
        cosim._apply_ackermann_fail_closed_brake = calls.append
        with self.assertRaisesRegex(RuntimeError, "feedback_frame_mismatch"):
            cosim._raise_pending_authoritative_action_error()
        self.assertEqual(calls, ["feedback_frame_mismatch"])

    def test_newly_initialized_actor_waits_for_first_phase_a_feedback(self):
        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_feedback_apply_enabled = True
        cosim.ackermann_physics_enabled = True
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
            target, _ = cosim._resolve_ackermann_longitudinal_target("vehicle0", vehicle_info, 0.0)
        self.assertAlmostEqual(target, 0.3)

        cosim._resolve_ackermann_longitudinal_target("vehicle0", vehicle_info, 0.2)
        self.assertFalse(cosim._ackermann_actor_state["vehicle0"]["restart_active"])

    def test_curve_speed_cap_is_disabled_by_default(self):
        cosim = object.__new__(CarlaCosim)
        cosim._ackermann_actor_state = {}
        cosim.ackermann_tuning = AckermannTuning()
        cosim.step_length = 0.05
        cosim._is_ackermann_feedback_apply_actor = lambda _actor_id: True
        vehicle_info = {
            "sumo_desired_speed": 13.08,
            "feedback_observed_speed": 13.0,
            "acceleration": 2.0,
            "sumo_emergency_decel": 9.0,
            "lookahead_heading_change": 0.77057,
            "lookahead_distance": 3.7035,
        }

        target, acceleration = cosim._resolve_ackermann_longitudinal_target(
            "vehicle0", vehicle_info, 13.0
        )

        self.assertAlmostEqual(target, 13.08)
        self.assertAlmostEqual(acceleration, 2.0)
        state = cosim._ackermann_actor_state["vehicle0"]
        self.assertFalse(state["curve_speed_cap_active"])
        self.assertIsNone(state["curve_speed_cap_curvature"])
        self.assertAlmostEqual(state["curve_speed_cap_pre_target_speed"], 13.08)

    def test_curve_speed_cap_ignores_straight_and_invalid_lookahead(self):
        cosim = object.__new__(CarlaCosim)
        cosim._ackermann_actor_state = {}
        cosim.ackermann_tuning = AckermannTuning()
        cosim.ackermann_max_lateral_accel = 3.0
        cosim.step_length = 0.05
        cosim._is_ackermann_feedback_apply_actor = lambda _actor_id: True
        vehicle_info = {
            "sumo_desired_speed": 13.08,
            "feedback_observed_speed": 13.0,
            "acceleration": 2.0,
            "sumo_emergency_decel": 9.0,
            "lookahead_heading_change": 0.0,
            "lookahead_distance": 3.7035,
        }

        vehicle_info["lookahead_heading_change"] = 0.77057
        cosim._resolve_ackermann_longitudinal_target("vehicle0", vehicle_info, 13.0)
        self.assertTrue(cosim._ackermann_actor_state["vehicle0"]["curve_speed_cap_active"])

        for heading_change, distance in (
            (0.0, 3.7035),
            (0.77057, 0.0),
            (float("nan"), 3.7035),
            (0.77057, float("inf")),
        ):
            vehicle_info["lookahead_heading_change"] = heading_change
            vehicle_info["lookahead_distance"] = distance
            target, acceleration = cosim._resolve_ackermann_longitudinal_target(
                "vehicle0", vehicle_info, 13.0
            )
            self.assertAlmostEqual(target, 13.08)
            self.assertAlmostEqual(acceleration, 2.0)
            state = cosim._ackermann_actor_state["vehicle0"]
            self.assertFalse(state["curve_speed_cap_active"])
            self.assertIsNone(state["curve_speed_cap_curvature"])
            self.assertIsNone(state["curve_speed_cap_speed_limit"])
            self.assertIsNone(state["curve_speed_cap_required_acceleration"])

    def test_curve_speed_cap_matches_vehicle2311_trace_and_bounds_deceleration(self):
        cosim = object.__new__(CarlaCosim)
        cosim._ackermann_actor_state = {}
        cosim.ackermann_tuning = AckermannTuning()
        cosim.ackermann_max_lateral_accel = 3.0
        cosim.step_length = 0.05
        cosim._is_ackermann_feedback_apply_actor = lambda _actor_id: True
        vehicle_info = {
            "sumo_desired_speed": 13.08,
            "feedback_observed_speed": 13.0,
            "acceleration": 2.0,
            "sumo_emergency_decel": 9.0,
            "lookahead_heading_change": 0.77057,
            "lookahead_distance": 3.7035,
        }

        target, acceleration = cosim._resolve_ackermann_longitudinal_target(
            "vehicle2311", vehicle_info, 13.0
        )

        curvature = 2.0 * math.sin(0.77057 / 2.0) / 3.7035
        expected_limit = math.sqrt(3.0 / curvature)
        self.assertAlmostEqual(target, 3.844678, places=5)
        self.assertAlmostEqual(target, expected_limit)
        self.assertAlmostEqual(acceleration, -9.0)
        self.assertAlmostEqual(vehicle_info["sumo_desired_speed"], 13.08)
        state = cosim._ackermann_actor_state["vehicle2311"]
        self.assertAlmostEqual(state["curve_speed_cap_pre_target_speed"], 13.08)
        self.assertAlmostEqual(state["curve_speed_cap_curvature"], curvature)
        self.assertAlmostEqual(state["curve_speed_cap_speed_limit"], expected_limit)
        self.assertTrue(state["curve_speed_cap_active"])
        self.assertAlmostEqual(state["curve_speed_cap_required_acceleration"], -20.820636, places=5)
        self.assertAlmostEqual(state["applied_desired_acceleration"], -9.0)

    def test_curve_speed_cap_does_not_decelerate_below_limit(self):
        cosim = object.__new__(CarlaCosim)
        cosim._ackermann_actor_state = {}
        cosim.ackermann_tuning = AckermannTuning()
        cosim.ackermann_max_lateral_accel = 3.0
        cosim.step_length = 0.05
        cosim._is_ackermann_feedback_apply_actor = lambda _actor_id: True
        vehicle_info = {
            "sumo_desired_speed": 13.08,
            "feedback_observed_speed": 2.0,
            "acceleration": 2.0,
            "sumo_emergency_decel": 9.0,
            "lookahead_heading_change": 0.77057,
            "lookahead_distance": 3.7035,
        }

        target, acceleration = cosim._resolve_ackermann_longitudinal_target(
            "vehicle0", vehicle_info, 2.0
        )

        self.assertAlmostEqual(target, 3.844678, places=5)
        self.assertAlmostEqual(acceleration, 2.0)
        state = cosim._ackermann_actor_state["vehicle0"]
        self.assertTrue(state["curve_speed_cap_active"])
        self.assertIsNone(state["curve_speed_cap_required_acceleration"])

    def test_curve_speed_cap_is_applied_after_restart_target(self):
        cosim = object.__new__(CarlaCosim)
        cosim._ackermann_actor_state = {}
        cosim.ackermann_tuning = AckermannTuning()
        cosim.ackermann_max_lateral_accel = 0.001
        cosim.step_length = 0.05
        cosim._is_ackermann_feedback_apply_actor = lambda _actor_id: True
        vehicle_info = {
            "sumo_desired_speed": 0.1,
            "feedback_observed_speed": 0.0,
            "acceleration": 2.0,
            "sumo_emergency_decel": 9.0,
            "lookahead_heading_change": math.pi,
            "lookahead_distance": 0.1,
        }

        target, acceleration = cosim._resolve_ackermann_longitudinal_target(
            "vehicle0", vehicle_info, 0.0
        )

        self.assertLess(target, 0.1)
        self.assertAlmostEqual(target, math.sqrt(0.001 / 20.0))
        self.assertAlmostEqual(acceleration, 2.0)
        self.assertTrue(cosim._ackermann_actor_state["vehicle0"]["restart_active"])
        self.assertTrue(cosim._ackermann_actor_state["vehicle0"]["curve_speed_cap_active"])

    def test_ackermann_trace_includes_curve_speed_cap_state(self):
        cosim = object.__new__(CarlaCosim)
        cosim._should_record_ackermann_control_trace = lambda _actor_id: True
        cosim._ackermann_feedback_state = {}
        cosim._ackermann_actor_state = {
            "vehicle0": {
                "curve_speed_cap_max_lateral_acceleration": 3.0,
                "curve_speed_cap_pre_target_speed": 13.08,
                "curve_speed_cap_curvature": 0.2,
                "curve_speed_cap_speed_limit": 3.84,
                "curve_speed_cap_active": True,
                "curve_speed_cap_required_acceleration": -20.0,
                "exact_control_target_local_x": 0.85,
                "exact_control_target_local_y": -0.1,
                "phase_aligned_canonical_target_error": 0.125,
            }
        }
        cosim.world = SimpleNamespace(get_snapshot=lambda: SimpleNamespace(frame=17))
        cosim.step_length = 0.05
        cosim._inproc_prev_state = {"simulation_time": 12.5}
        vehicle = SimpleNamespace(
            get_acceleration=lambda: SimpleNamespace(x=0.0, y=0.0),
            get_control=lambda: SimpleNamespace(throttle=0.0, brake=0.0, steer=0.0),
        )
        transform = SimpleNamespace(
            location=SimpleNamespace(x=1.0, y=2.0),
            rotation=SimpleNamespace(yaw=0.0),
        )

        with patch("builtins.print") as print_mock:
            cosim._record_ackermann_control_trace(
                veh_id="vehicle0",
                veh_info={
                    "sumo_desired_speed": 13.08,
                    "sumo_leader_id": "leader0",
                    "sumo_leader_gap": 12.5,
                    "sumo_leader_speed": 4.5,
                    "sumo_leader_acceleration": -0.25,
                    "sumo_leader_lane_id": "edge_current_0",
                    "sumo_leader_lane_position": 10.0,
                    "external_state_lateral_maneuver_enabled": False,
                    "external_state_lateral_maneuver_target_enabled": False,
                    "external_state_maneuver_armed_origin_phase": "hold",
                    "external_state_maneuver_target_pre_state_phase": "active",
                    "external_state_maneuver_target_post_state_phase": "hold",
                    "external_state_maneuver_target_applied": True,
                    "external_state_maneuver_target_source": "held_canonical_d",
                    "external_state_maneuver_target_selected_d": -0.675,
                    "external_state_maneuver_target_world_anchor_x": 10.0,
                    "external_state_maneuver_target_world_anchor_y": 20.0,
                    "external_state_maneuver_target_d_outside_lane": False,
                    "external_state_internal_lane_transition_assist_enabled": False,
                    "feedback_internal_lane_transition_assist_applied": False,
                    "lookahead_target_world_left_normal_x": 0.25,
                    "lookahead_target_world_left_normal_y": 0.75,
                    "control_lookahead_position_valid": True,
                    "control_lookahead_x": 11.0,
                    "control_lookahead_y": 12.0,
                    "steering_preview_position_valid": True,
                    "steering_preview_x": 14.0,
                    "steering_preview_y": 15.0,
                    "steering_preview_requested_distance": 6.5,
                    "steering_preview_effective_distance": 5.0,
                    "steering_preview_front_s": 6.05,
                    "steering_preview_front_d": -0.675,
                    "steering_preview_tangent_x": 0.6,
                    "steering_preview_tangent_y": 0.8,
                    "steering_preview_normal_x": -0.8,
                    "steering_preview_normal_y": 0.6,
                    "front_rear_distance_error": 1e-12,
                    "front_rear_distance_consistent": True,
                    "canonical_route_lane_ids": ("edge_394_0", "edge_394_1"),
                    "canonical_route_path_switched": True,
                    "canonical_phase_a_s": 1.0,
                    "canonical_phase_a_d": -0.6,
                    "canonical_phase_b_s": 1.05,
                    "canonical_phase_b_d": -0.675,
                    "canonical_delta_d": -0.075,
                    "canonical_target_s": 1.55,
                    "canonical_target_d": -0.675,
                    "canonical_held_d": -0.675,
                    "canonical_hold_reprojected": True,
                    "canonical_invariant_valid": True,
                    "canonical_invariant_reason": "",
                    "canonical_invariant_warning": "canonical_target_lateral_jump",
                    "canonical_fallback_applied": False,
                    "canonical_fallback_reason": "",
                    "canonical_fallback_age": 0,
                    "canonical_front_target_lateral_jump": 0.01,
                    "canonical_control_target_lateral_jump": 0.02,
                    "canonical_steering_preview_lateral_jump": 0.03,
                    "canonical_front_target_lateral_change": 0.04,
                    "canonical_control_target_lateral_change": 0.05,
                    "canonical_steering_preview_lateral_change": 0.06,
                    "canonical_current_d_outside_lane": False,
                    "canonical_held_d_outside_lane": False,
                    "feedback_rear_axle_x": 5.1,
                    "feedback_rear_axle_y": 6.2,
                    "front_to_rear_axle_distance": 3.9,
                    "external_state_maneuver_phase": "hold",
                    "external_state_maneuver_source_lane_id": "edge_394_0",
                    "external_state_maneuver_target_lane_id": "edge_394_1",
                    "external_state_maneuver_primary_lane_switched": True,
                    "external_state_maneuver_current_lateral_offset": -0.675,
                    "external_state_maneuver_preview_lateral_offset": -0.675,
                    "external_state_maneuver_held_lateral_offset": -0.675,
                    "external_state_maneuver_active_frame_count": 70,
                    "external_state_maneuver_zero_motion_frame_count": 5,
                    "external_state_maneuver_transition_reason": ("completed_with_final_offset"),
                    "external_state_maneuver_evidence_eligible": True,
                    "external_state_maneuver_evidence_exclusion_reason": "",
                    "external_state_maneuver_canonical_motion_evidence": False,
                    "external_state_maneuver_canonical_delta_sign": -1,
                    "external_state_maneuver_speed_motion_evidence": False,
                    "external_state_maneuver_intent_evidence": False,
                    "external_state_maneuver_start_candidate_sign": 0,
                    "external_state_maneuver_start_candidate_frame_count": 0,
                    "external_state_maneuver_armed_idle_frame_count": 0,
                    "external_state_maneuver_confirmed_direction_sign": 0,
                    "external_state_maneuver_confirmed_sumo_time": None,
                    "external_state_maneuver_confirmed_direction_x": None,
                    "external_state_maneuver_confirmed_direction_y": None,
                    "external_state_maneuver_switch_evidence_excluded": False,
                },
                vehicle=vehicle,
                current_transform=transform,
                current_speed=13.0,
                target_speed=3.84,
                target_acceleration=-9.0,
                position_error=0.0,
                feedback_unhealthy=False,
                target_behind=False,
            )

        line = print_mock.call_args.args[0]
        trace = json.loads(line.removeprefix("AckermannControlTrace "))
        self.assertEqual(trace["sumo_desired_speed"], 13.08)
        self.assertEqual(trace["sumo_leader_id"], "leader0")
        self.assertAlmostEqual(trace["sumo_leader_gap"], 12.5)
        self.assertAlmostEqual(trace["sumo_leader_speed"], 4.5)
        self.assertAlmostEqual(trace["sumo_leader_acceleration"], -0.25)
        self.assertEqual(trace["sumo_leader_lane_id"], "edge_current_0")
        self.assertAlmostEqual(trace["sumo_leader_lane_position"], 10.0)
        self.assertEqual(trace["curve_speed_cap_max_lateral_acceleration"], 3.0)
        self.assertEqual(trace["curve_speed_cap_pre_target_speed"], 13.08)
        self.assertEqual(trace["curve_speed_cap_curvature"], 0.2)
        self.assertEqual(trace["curve_speed_cap_speed_limit"], 3.84)
        self.assertTrue(trace["curve_speed_cap_active"])
        self.assertEqual(trace["curve_speed_cap_required_acceleration"], -20.0)
        self.assertFalse(trace["external_state_lateral_maneuver_enabled"])
        self.assertFalse(trace["external_state_lateral_maneuver_target_enabled"])
        self.assertFalse(trace["external_state_internal_lane_transition_assist_enabled"])
        self.assertEqual(trace["external_state_maneuver_armed_origin_phase"], "hold")
        self.assertEqual(trace["external_state_maneuver_target_pre_state_phase"], "active")
        self.assertEqual(trace["external_state_maneuver_target_post_state_phase"], "hold")
        self.assertTrue(trace["external_state_maneuver_target_applied"])
        self.assertEqual(trace["external_state_maneuver_target_source"], "held_canonical_d")
        self.assertEqual(trace["external_state_maneuver_target_selected_d"], -0.675)
        self.assertEqual(trace["external_state_maneuver_target_world_anchor_x"], 10.0)
        self.assertEqual(trace["external_state_maneuver_target_world_anchor_y"], 20.0)
        self.assertFalse(trace["external_state_maneuver_target_d_outside_lane"])
        self.assertFalse(trace["feedback_internal_lane_transition_assist_applied"])
        self.assertEqual(trace["lookahead_target_world_left_normal_x"], 0.25)
        self.assertEqual(trace["lookahead_target_world_left_normal_y"], 0.75)
        self.assertEqual(trace["sumo_control_lookahead_x"], 11.0)
        self.assertEqual(trace["sumo_control_lookahead_y"], 12.0)
        self.assertTrue(trace["control_lookahead_position_valid"])
        self.assertEqual(trace["sumo_steering_preview_x"], 14.0)
        self.assertEqual(trace["sumo_steering_preview_y"], 15.0)
        self.assertTrue(trace["steering_preview_position_valid"])
        self.assertEqual(trace["steering_preview_requested_distance"], 6.5)
        self.assertEqual(trace["steering_preview_effective_distance"], 5.0)
        self.assertEqual(trace["steering_preview_front_s"], 6.05)
        self.assertEqual(trace["steering_preview_front_d"], -0.675)
        self.assertEqual(trace["steering_preview_tangent_x"], 0.6)
        self.assertEqual(trace["steering_preview_tangent_y"], 0.8)
        self.assertEqual(trace["steering_preview_normal_x"], -0.8)
        self.assertEqual(trace["steering_preview_normal_y"], 0.6)
        self.assertEqual(trace["front_rear_distance_error"], 1e-12)
        self.assertTrue(trace["front_rear_distance_consistent"])
        self.assertEqual(trace["canonical_route_lane_ids"], ["edge_394_0", "edge_394_1"])
        self.assertTrue(trace["canonical_route_path_switched"])
        self.assertEqual(trace["canonical_phase_a_s"], 1.0)
        self.assertEqual(trace["canonical_phase_a_d"], -0.6)
        self.assertEqual(trace["canonical_phase_b_s"], 1.05)
        self.assertEqual(trace["canonical_phase_b_d"], -0.675)
        self.assertEqual(trace["canonical_delta_d"], -0.075)
        self.assertEqual(trace["canonical_target_s"], 1.55)
        self.assertEqual(trace["canonical_target_d"], -0.675)
        self.assertEqual(trace["canonical_held_d"], -0.675)
        self.assertTrue(trace["canonical_hold_reprojected"])
        self.assertTrue(trace["canonical_invariant_valid"])
        self.assertEqual(trace["canonical_invariant_warning"], "canonical_target_lateral_jump")
        self.assertFalse(trace["canonical_fallback_applied"])
        self.assertEqual(trace["canonical_fallback_reason"], "")
        self.assertEqual(trace["canonical_fallback_age"], 0)
        self.assertEqual(trace["canonical_front_target_lateral_jump"], 0.01)
        self.assertEqual(trace["canonical_control_target_lateral_jump"], 0.02)
        self.assertEqual(trace["canonical_steering_preview_lateral_jump"], 0.03)
        self.assertEqual(trace["exact_control_target_local_x"], 0.85)
        self.assertEqual(trace["exact_control_target_local_y"], -0.1)
        self.assertEqual(trace["canonical_front_target_lateral_change"], 0.04)
        self.assertEqual(trace["canonical_control_target_lateral_change"], 0.05)
        self.assertEqual(trace["canonical_steering_preview_lateral_change"], 0.06)
        self.assertFalse(trace["canonical_current_d_outside_lane"])
        self.assertFalse(trace["canonical_held_d_outside_lane"])
        self.assertEqual(trace["feedback_rear_axle_x"], 5.1)
        self.assertEqual(trace["feedback_rear_axle_y"], 6.2)
        self.assertEqual(trace["front_to_rear_axle_distance"], 3.9)
        self.assertEqual(trace["external_state_maneuver_phase"], "hold")
        self.assertEqual(trace["external_state_maneuver_source_lane_id"], "edge_394_0")
        self.assertEqual(trace["external_state_maneuver_target_lane_id"], "edge_394_1")
        self.assertTrue(trace["external_state_maneuver_primary_lane_switched"])
        self.assertAlmostEqual(trace["external_state_maneuver_held_lateral_offset"], -0.675)
        self.assertEqual(trace["external_state_maneuver_active_frame_count"], 70)
        self.assertEqual(trace["external_state_maneuver_zero_motion_frame_count"], 5)
        self.assertEqual(
            trace["external_state_maneuver_transition_reason"],
            "completed_with_final_offset",
        )
        self.assertTrue(trace["external_state_maneuver_evidence_eligible"])
        self.assertEqual(trace["external_state_maneuver_evidence_exclusion_reason"], "")
        self.assertFalse(trace["external_state_maneuver_canonical_motion_evidence"])
        self.assertEqual(trace["external_state_maneuver_canonical_delta_sign"], -1)
        self.assertFalse(trace["external_state_maneuver_speed_motion_evidence"])
        self.assertFalse(trace["external_state_maneuver_intent_evidence"])
        self.assertEqual(trace["external_state_maneuver_start_candidate_sign"], 0)
        self.assertEqual(trace["external_state_maneuver_start_candidate_frame_count"], 0)
        self.assertEqual(trace["external_state_maneuver_armed_idle_frame_count"], 0)
        self.assertEqual(trace["external_state_maneuver_confirmed_direction_sign"], 0)
        self.assertIsNone(trace["external_state_maneuver_confirmed_sumo_time"])
        self.assertIsNone(trace["external_state_maneuver_confirmed_direction_x"])
        self.assertIsNone(trace["external_state_maneuver_confirmed_direction_y"])
        self.assertFalse(trace["external_state_maneuver_switch_evidence_excluded"])
        self.assertEqual(trace["phase_aligned_canonical_target_error"], 0.125)

    def test_emergency_brake_retains_steer_and_releases_after_hysteresis(self):
        cosim = object.__new__(CarlaCosim)
        cosim._ackermann_actor_state = {
            "vehicle0": {
                "sumo_requested_acceleration": -5.0,
                "sumo_emergency_decel": 8.0,
            }
        }
        cosim.ackermann_tuning = AckermannTuning()
        cosim.ackermann_emergency_brake_tuning = carla_cosim.AckermannEmergencyBrakeTuning()
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
        cosim.ackermann_emergency_brake_tuning = carla_cosim.AckermannEmergencyBrakeTuning()
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
            "lookahead_x": 13.9,
            "lookahead_y": 0.0,
            "lookahead_z": 0.0,
            "control_lookahead_position_valid": True,
            "control_lookahead_x": 10.0,
            "control_lookahead_y": 0.0,
            "control_lookahead_z": 0.0,
            "steering_preview_position_valid": True,
            "steering_preview_x": 12.0,
            "steering_preview_y": 0.0,
            "steering_preview_z": 0.0,
            "path_tracker_valid": True,
            "path_tracker_rear_curvature": 0.0,
            "path_tracker_reference_tangent_x": 1.0,
            "path_tracker_reference_tangent_y": 0.0,
            "path_tracker_front_cross_track_error": 0.0,
        }

        control = cosim._build_ackermann_control(
            "vehicle0", vehicle_info, actor, [0.0, 0.0, 0.0], 90.0, transform
        )
        self.assertAlmostEqual(control.speed, 5.0)
        self.assertAlmostEqual(control.acceleration, 1.0)
        self.assertEqual(cosim._ackermann_actor_state["vehicle0"]["control_mode"], "ackermann")

        vehicle_info["control_lookahead_x"] = -2.0
        fail_closed = cosim._build_ackermann_control(
            "vehicle0", vehicle_info, actor, [0.0, 0.0, 0.0], 90.0, transform
        )
        self.assertEqual(fail_closed.brake, 1.0)
        self.assertEqual(
            cosim._ackermann_actor_state["vehicle0"]["control_mode"],
            "fail_closed_brake",
        )
        self.assertIn(
            "rear reference target behind",
            cosim._ackermann_actor_state["vehicle0"]["fail_closed_reason"],
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
        self.assertTrue(CarlaCosim._ackermann_footprints_overlap(overlapping, overlapping))
        self.assertFalse(CarlaCosim._ackermann_footprints_overlap(overlapping, separated))

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

    def test_ackermann_trace_file_is_dedicated_jsonl(self):
        cosim = object.__new__(CarlaCosim)
        cosim._diagnostic_lock = Lock()
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "ackermann_trace.jsonl"
            cosim.ackermann_control_trace_path = str(trace_path)
            cosim._emit_ackermann_control_trace({"actor_id": "AV", "frame": 1})
            cosim._emit_ackermann_control_trace({"actor_id": "AV", "frame": 2})
            records = [json.loads(line) for line in trace_path.read_text().splitlines()]

        self.assertEqual([record["frame"] for record in records], [1, 2])

    def test_actor_radius_filter_uses_exit_hysteresis(self):
        cosim = object.__new__(CarlaCosim)
        cosim.actor_filter_enabled = True
        cosim.actor_filter_center_id = "AV"
        cosim.actor_filter_radius = 300.0
        cosim.actor_filter_hysteresis = 20.0
        cosim._actor_filter_active_vehicle_ids = set()
        cosim._actor_filter_missing_center_warned = False

        def vehicles_at(distance):
            return {
                "AV": {"x": 0.0, "y": 0.0},
                "vehicle0": {"x": distance, "y": 0.0},
            }

        filtered, _ = cosim._filter_actor_details_by_radius(vehicles_at(299.0), {})
        self.assertEqual(set(filtered), {"AV", "vehicle0"})
        filtered, _ = cosim._filter_actor_details_by_radius(vehicles_at(310.0), {})
        self.assertEqual(set(filtered), {"AV", "vehicle0"})
        filtered, _ = cosim._filter_actor_details_by_radius(vehicles_at(321.0), {})
        self.assertEqual(set(filtered), {"AV"})
        filtered, _ = cosim._filter_actor_details_by_radius(vehicles_at(310.0), {})
        self.assertEqual(set(filtered), {"AV"})
        filtered, _ = cosim._filter_actor_details_by_radius(vehicles_at(299.0), {})
        self.assertEqual(set(filtered), {"AV", "vehicle0"})

    def test_physics_radius_uses_exit_hysteresis_and_keeps_av(self):
        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_physics_enabled = True
        cosim.ackermann_feedback_apply_enabled = True
        cosim.ackermann_feedback_actor_ids = {"*"}
        cosim.control_av = False
        cosim.physics_radius_enabled = True
        cosim.physics_radius_center_id = "AV"
        cosim.physics_radius = 100.0
        cosim.physics_radius_hysteresis = 10.0
        cosim._physics_active_vehicle_ids = set()

        def select_at(distance):
            vehicles = {
                "AV": {"x": 0.0, "y": 0.0},
                "vehicle0": {"x": distance, "y": 0.0},
            }
            cosim._update_physics_active_vehicle_ids(vehicles)
            return set(cosim._physics_active_vehicle_ids)

        self.assertEqual(select_at(99.0), {"AV", "vehicle0"})
        self.assertEqual(select_at(105.0), {"AV", "vehicle0"})
        self.assertEqual(select_at(111.0), {"AV"})
        self.assertEqual(select_at(105.0), {"AV"})
        self.assertEqual(select_at(99.0), {"AV", "vehicle0"})
        self.assertTrue(cosim._uses_ackermann_physics("AV"))
        self.assertTrue(cosim._uses_ackermann_physics("vehicle0"))

        cosim._update_physics_active_vehicle_ids({"vehicle0": {"x": 0.0, "y": 0.0}})
        self.assertEqual(cosim._physics_active_vehicle_ids, {"AV"})
        self.assertFalse(cosim._uses_ackermann_physics("vehicle0"))

    def test_feedback_membership_exit_clears_vehicle_maneuver_and_path_state(self):
        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin._stop_requested = False
        plugin._tick_requested = Event()
        plugin._tick_requested.set()
        plugin._lock = Lock()
        plugin._pending_commands = []
        plugin.physics_feedback_observations = {}
        plugin.physics_feedback_actor_membership = {"vehicle0"}
        plugin.physics_feedback_generations = {"vehicle0": 7}
        plugin.physics_lateral_direction_states = {"vehicle0": {"phase": "active"}}
        plugin.physics_rolling_path_states = {"vehicle0": {"entries": ()}}
        plugin.controlled_agents_each_step = set()
        plugin._set_status = lambda _status: None
        plugin.logger = SimpleNamespace(info=lambda *args: None, debug=lambda *args: None)
        simulator = SimpleNamespace(running=True)

        self.assertTrue(plugin.function_before_env_step(simulator, None))
        self.assertEqual(plugin.physics_feedback_actor_membership, set())
        self.assertNotIn("vehicle0", plugin.physics_feedback_generations)
        self.assertNotIn("vehicle0", plugin.physics_lateral_direction_states)
        self.assertNotIn("vehicle0", plugin.physics_rolling_path_states)

    def test_background_invalid_policy_demotes_until_hysteresis_exit(self):
        applied = []
        actor = SimpleNamespace(apply_control=lambda control: applied.append(control))
        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_background_invalid_policy = "demote"
        cosim._pending_authoritative_action_error = None
        cosim._pending_background_demotions = {}
        cosim._physics_active_vehicle_ids = {"AV", "vehicle0"}
        cosim._physics_quarantined_vehicle_ids = {}
        cosim._background_demotion_history = []
        cosim._physics_actor_distances = {"vehicle0": 95.0}
        cosim._vehicle_actor_index = {"vehicle0": actor}
        cosim._physics_feedback_frames = {"vehicle0": 12}
        cosim._physics_feedback_failures = {"vehicle0": 3}
        cosim._ackermann_feedback_state = {"vehicle0": {"feedback_status": "queued"}}
        cosim._ackermann_actor_state = {}
        cosim._actor_sync_modes = {}
        cosim._actor_filter_active_vehicle_ids = {"AV", "vehicle0"}
        cosim._last_world_frame = 12
        cosim._inproc_prev_state = {"simulation_time": 500.0}
        cosim.actor_mode_trace_path = ""
        cosim._full_brake_control = lambda: "full_brake"

        cosim._queue_authoritative_failure("vehicle0", "invalid_route")
        self.assertIsNone(cosim._pending_authoritative_action_error)
        cosim._handle_pending_background_demotions()

        self.assertEqual(applied, ["full_brake"])
        self.assertNotIn("vehicle0", cosim._physics_active_vehicle_ids)
        self.assertIn("vehicle0", cosim._physics_quarantined_vehicle_ids)
        self.assertEqual(len(cosim._background_demotion_history), 1)
        self.assertEqual(cosim._actor_sync_modes["vehicle0"], "quarantined_transform")

        cosim.ackermann_physics_enabled = True
        cosim.ackermann_feedback_apply_enabled = True
        cosim.ackermann_feedback_actor_ids = {"*"}
        cosim.control_av = False
        cosim.physics_radius_enabled = True
        cosim.physics_radius_center_id = "AV"
        cosim.physics_radius = 100.0
        cosim.physics_radius_hysteresis = 10.0

        def update(distance):
            cosim._update_physics_active_vehicle_ids(
                {"AV": {"x": 0.0, "y": 0.0}, "vehicle0": {"x": distance, "y": 0.0}}
            )

        update(105.0)
        self.assertNotIn("vehicle0", cosim._physics_active_vehicle_ids)
        self.assertIn("vehicle0", cosim._physics_quarantined_vehicle_ids)
        update(111.0)
        self.assertNotIn("vehicle0", cosim._physics_active_vehicle_ids)
        self.assertNotIn("vehicle0", cosim._physics_quarantined_vehicle_ids)
        update(99.0)
        self.assertIn("vehicle0", cosim._physics_active_vehicle_ids)

    def test_background_demotion_policy_never_demotes_av(self):
        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_background_invalid_policy = "demote"
        cosim._pending_authoritative_action_error = None
        cosim._pending_background_demotions = {}

        cosim._queue_authoritative_failure("AV", "invalid_route")

        self.assertEqual(cosim._pending_background_demotions, {})
        self.assertEqual(cosim._pending_authoritative_action_error["actor_id"], "AV")

    def test_physics_feedback_commands_exclude_transform_only_actors(self):
        collected_actor_ids = []
        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_physics_enabled = True
        cosim.ackermann_feedback_apply_enabled = True
        cosim.physics_radius_enabled = True
        cosim._physics_active_vehicle_ids = {"AV", "vehicle2"}
        cosim._inproc_prev_state = {
            "agent_details": {
                "vehicle": {
                    "vehicle9": {"speed": 9.0},
                    "vehicle2": {"speed": 2.0},
                }
            }
        }
        cosim._last_world_frame = 123
        cosim._vehicle_actor_index = {
            actor_id: SimpleNamespace(id=actor_id) for actor_id in ("vehicle9", "vehicle2")
        }
        cosim._ackermann_actor_state = {}
        cosim._physics_feedback_frames = {}
        cosim._ackermann_feedback_state = {}
        cosim._is_ackermann_feedback_actor = lambda _actor_id: True
        cosim._carla_actor_to_sumo_feedback = lambda actor_id, _actor, vehicle_info: (
            collected_actor_ids.append(actor_id) or {"speed": vehicle_info["speed"]}
        )
        cosim._clear_physics_feedback_failure = lambda _actor_id: None

        commands = cosim._build_physics_feedback_commands()

        self.assertEqual(collected_actor_ids, ["vehicle2"])
        self.assertEqual([command["agent_id"] for command in commands], ["vehicle2"])

    def test_fail_closed_brake_excludes_transform_only_actors(self):
        applied = []

        def actor(actor_id):
            return SimpleNamespace(apply_control=lambda _control: applied.append(actor_id))

        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_physics_enabled = True
        cosim.physics_radius_enabled = True
        cosim._physics_active_vehicle_ids = {"AV", "vehicle0"}
        cosim._vehicle_actor_index = {
            "vehicle0": actor("vehicle0"),
            "vehicle1": actor("vehicle1"),
        }
        cosim._is_ackermann_feedback_actor = lambda _actor_id: True
        cosim._full_brake_control = lambda: object()

        cosim._apply_physics_fail_closed_brake("test")

        self.assertEqual(applied, ["vehicle0"])

    def test_teleport_transition_disables_physics_and_clears_feedback_state(self):
        calls = []
        cosim = object.__new__(CarlaCosim)
        cosim._ackermann_actor_state = {
            "vehicle0": {
                "physics_enabled": True,
                "physics_initialization_pending": True,
            }
        }
        cosim._physics_feedback_frames = {"vehicle0": 12}
        cosim._physics_feedback_failures = {"vehicle0": 1}
        cosim._ackermann_feedback_state = {"vehicle0": {"feedback_status": "queued"}}
        cosim._ackermann_fail_closed_reasons = {"vehicle0": "test"}
        cosim._set_actor_simulate_physics = lambda actor, enabled: calls.append((actor, enabled))
        actor = SimpleNamespace(id=7)

        cosim._ensure_actor_teleport_mode(actor, "vehicle0")

        self.assertEqual(calls, [(actor, False)])
        self.assertEqual(cosim._ackermann_actor_state["vehicle0"], {"physics_enabled": False})
        self.assertNotIn("vehicle0", cosim._physics_feedback_frames)
        self.assertNotIn("vehicle0", cosim._physics_feedback_failures)
        self.assertNotIn("vehicle0", cosim._ackermann_feedback_state)
        self.assertNotIn("vehicle0", cosim._ackermann_fail_closed_reasons)

    def test_physics_selection_never_takes_autoware_ego(self):
        cosim = object.__new__(CarlaCosim)
        cosim.ackermann_feedback_apply_enabled = True
        cosim.ackermann_feedback_actor_ids = {"*"}
        cosim.control_av = True
        self.assertTrue(cosim._is_ackermann_feedback_actor("vehicle0"))
        self.assertFalse(cosim._is_ackermann_feedback_actor("AV"))

        cosim.control_av = False
        self.assertTrue(cosim._is_ackermann_feedback_actor("AV"))

        plugin = object.__new__(TeraSimCoSimInProcessPlugin)
        plugin.physics_external_state_enabled = True
        plugin.physics_feedback_actor_ids = {"*"}
        plugin.control_av = True
        self.assertTrue(plugin._is_physics_feedback_actor("vehicle0"))
        self.assertFalse(plugin._is_physics_feedback_actor("AV"))

        plugin.control_av = False
        self.assertTrue(plugin._is_physics_feedback_actor("AV"))


if __name__ == "__main__":
    unittest.main()
