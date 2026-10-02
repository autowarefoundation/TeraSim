"""Validate SUMO model intent before requesting one explicit AV lane change."""

from __future__ import annotations

import math

from loguru import logger
from terasim.overlay import traci

from ..utils import CommandType, NDECommand
from .nde_decision_model import NDEDecisionModel


class AVSumoLaneChangeDecisionModel(NDEDecisionModel):
    """Keep car-following in SUMO and lateral authority in NDEController."""

    DURATION = 8.0
    MAX_LATERAL_DISCONTINUITY = 0.30
    REASONS = ("STRATEGIC", "SPEEDGAIN", "KEEPRIGHT", "COOPERATIVE", "SUBLANE")

    def __init__(self, cfg=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset()

    def _reset(self):
        self.last_lane_change_decision = {}
        self._latched_request_key = None
        self._request_generation = 0

    @staticmethod
    def _constant(name):
        return int(getattr(traci.constants, "LCA_" + name))

    def _result(self, decision, command=None):
        if command is None:
            command = NDECommand(command_type=CommandType.DEFAULT, prob=1)
        decision["request_generation"] = self._request_generation
        self.last_lane_change_decision = dict(decision)
        return command, {
            "ndd_command_distribution": {"normal": command},
            "command_cache": command,
            "lane_change_decision": dict(decision),
        }

    @staticmethod
    def _route_targets(lane_id, next_edge_id):
        # Retain distinct internal connectors as well as final lanes: two
        # connectors reaching one destination are still ambiguous geometry.
        return tuple(
            sorted(
                {
                    (str(link[0]), str(link[4]) if len(link) > 4 else "")
                    for link in traci.lane.getLinks(lane_id)
                    if link[0] and traci.lane.getEdgeID(link[0]) == next_edge_id
                }
            )
        )

    def _evaluate_direction(self, veh_id, source, direction):
        model_state, post_state = map(int, traci.vehicle.getLaneChangeState(veh_id, direction))
        reason_bits = [name.lower() for name in self.REASONS if model_state & self._constant(name)]
        blocked_mask = (
            self._constant("BLOCKED")
            | self._constant("INSUFFICIENT_SPACE")
            | self._constant("INSUFFICIENT_SPEED")
        )
        target_index = source["lane_index"] + direction
        target = ""
        if 0 <= target_index < traci.edge.getLaneNumber(source["edge_id"]):
            target = f"{source['edge_id']}_{target_index}"
        targets = (
            self._route_targets(target, source["next_route_edge_id"])
            if target and source["next_route_edge_id"]
            else ()
        )
        required = bool(source["route_required"])
        has_direction = bool(model_state & self._constant("LEFT" if direction > 0 else "RIGHT"))
        could_change = (
            bool(traci.vehicle.couldChangeLane(veh_id, direction, state=model_state))
            if has_direction
            else False
        )
        allowed = traci.lane.getAllowed(target) if target else ()
        disallowed = traci.lane.getDisallowed(target) if target else ()
        vehicle_class = traci.vehicle.getVehicleClass(veh_id)
        neighbor_modes = (0, 2) if direction > 0 else (1, 3)
        neighbors = (
            [
                (str(neighbor), float(gap))
                for mode in neighbor_modes
                for neighbor, gap in traci.vehicle.getNeighbors(veh_id, mode)
            ]
            if has_direction and target
            else []
        )
        # SL2015 can report an unblocked partial sublane move even when the
        # vehicle beside us prevents a complete primary-lane change.
        target_overlap = any(not math.isfinite(gap) or gap < 0 for _, gap in neighbors)
        checks = (
            (has_direction, "no_direction_intent"),
            (
                any(reason != "sublane" for reason in reason_bits),
                "sublane_only_or_no_supported_reason",
            ),
            (not model_state & blocked_mask, "blocked"),
            (bool(target), "adjacent_lane_missing"),
            (could_change, "could_change_lane_false"),
            (not target_overlap, "target_neighbor_overlap"),
            (
                (not allowed or vehicle_class in allowed) and vehicle_class not in disallowed,
                "target_lane_disallows_vehicle",
            ),
            (len(targets) <= 1, "target_connection_ambiguous"),
            (not required or "strategic" in reason_bits, "route_required_without_strategic_bit"),
            (not source["next_route_edge_id"] or len(targets) == 1, "target_route_disconnected"),
        )
        reason = next((reason for passed, reason in checks if not passed), "eligible")
        result = {
            "direction": "left" if direction > 0 else "right",
            "index_offset": direction,
            "model_lane_change_state": model_state,
            "post_lane_change_state": post_state,
            "reason_bits": reason_bits,
            "blocked": bool(model_state & blocked_mask),
            "blocked_bits": model_state & blocked_mask,
            "has_direction": has_direction,
            "could_change_lane": could_change,
            "target_neighbors": neighbors,
            "target_neighbor_overlap": target_overlap,
            "target_lane_id": target,
            "route_targets": targets,
            "route_required": required,
            "eligible": reason == "eligible",
            "evaluation_reason": reason,
        }
        if target:
            result.update(self._distance_envelope(veh_id, source["source_lane_id"], target))
            if result["eligible"] and not result["geometry_continuous"]:
                result["eligible"] = False
                result["evaluation_reason"] = "lane_switch_geometry_discontinuous"
        return result

    @staticmethod
    def _center_separation(source_lane_id, target_lane_id, lane_position, lane_length):
        source = tuple(traci.lane.getShape(source_lane_id))
        target = tuple(traci.lane.getShape(target_lane_id))
        segments = [(a, b, math.dist(a, b)) for a, b in zip(source, source[1:])]
        distance = sum(length for _, _, length in segments) * min(
            1.0, max(0.0, lane_position / lane_length)
        )
        point = None
        for a, b, length in segments:
            if length > 0 and distance <= length + 1e-8:
                point = tuple(x + (y - x) * distance / length for x, y in zip(a, b))
                break
            distance -= length
        if point is None:
            raise ValueError("invalid source lane centerline")
        separations = []
        for a, b in zip(target, target[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            squared_length = dx * dx + dy * dy
            if squared_length > 0:
                fraction = min(
                    1.0,
                    max(0.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / squared_length),
                )
                separations.append(
                    math.hypot(point[0] - a[0] - fraction * dx, point[1] - a[1] - fraction * dy)
                )
        separation = min(separations, default=math.nan)
        if not math.isfinite(separation) or separation <= 0:
            raise ValueError("invalid adjacent lane center separation")
        return separation

    def _distance_envelope(self, veh_id, source_lane_id, target_lane_id):
        lane_length = float(traci.lane.getLength(source_lane_id))
        lane_position = float(traci.vehicle.getLanePosition(veh_id))
        speed = float(traci.vehicle.getSpeed(veh_id))
        length = float(traci.vehicle.getLength(veh_id))
        max_speed_lat = float(traci.vehicletype.getMaxSpeedLat(traci.vehicle.getTypeID(veh_id)))
        if (
            not all(
                math.isfinite(value)
                for value in (lane_length, lane_position, speed, length, max_speed_lat)
            )
            or min(lane_length, length, max_speed_lat) <= 0
            or speed < 0
        ):
            raise ValueError("invalid lane-change distance inputs")
        separation = self._center_separation(
            source_lane_id, target_lane_id, lane_position, lane_length
        )
        maneuver_time = separation / max_speed_lat + 1.0
        required = speed * maneuver_time + max(5.0, length)
        # SUMO transfers posLat by half the two lane widths at a primary-lane
        # switch. Custom centerlines may be farther apart, producing a world
        # pose jump even with identity/full-pose feedback. Check the maneuver
        # corridor without changing the map or smoothing SUMO's Phase B pose.
        nominal_separation = 0.5 * (
            traci.lane.getWidth(source_lane_id) + traci.lane.getWidth(target_lane_id)
        )
        end_position = min(lane_length, lane_position + speed * maneuver_time)
        sample_count = max(1, math.ceil(max(0.0, end_position - lane_position)))
        discontinuity = max(
            abs(
                self._center_separation(
                    source_lane_id,
                    target_lane_id,
                    lane_position + (end_position - lane_position) * index / sample_count,
                    lane_length,
                )
                - nominal_separation
            )
            for index in range(sample_count + 1)
        )
        remaining = max(0.0, lane_length - lane_position)
        return {
            "current_speed": speed,
            "lane_remaining": remaining,
            "lane_center_separation": separation,
            "max_speed_lat": max_speed_lat,
            "maneuver_time": maneuver_time,
            "required_distance": required,
            "distance_sufficient": remaining >= required,
            "predicted_lane_switch_discontinuity": discontinuity,
            "geometry_continuous": discontinuity <= self.MAX_LATERAL_DISCONTINUITY,
        }

    def _select(self, veh_id, decision):
        evaluations = [
            self._evaluate_direction(veh_id, decision, direction) for direction in (1, -1)
        ]
        decision["direction_evaluations"] = evaluations
        intents = [item for item in evaluations if item["has_direction"]]
        # A blocked request is still a request. Only disappearance or a new
        # source/target releases the latch, never changing eligibility alone.
        keys = {
            (decision["route_index"], decision["source_lane_id"], item["target_lane_id"])
            for item in intents
        }
        if self._latched_request_key not in keys:
            self._latched_request_key = None
        controller = getattr(self._agent, "controller", None)
        if controller is not None and controller.is_busy:
            decision["skip_reason"] = "controller_busy"
            return self._result(decision)
        if self._latched_request_key is not None:
            decision["skip_reason"] = "duplicate_request_latched"
            return self._result(decision)
        # Reject simultaneous raw directional requests even if only one is safe.
        conflict = len(intents) > 1 or any(
            item["model_lane_change_state"] & self._constant("LEFT")
            and item["model_lane_change_state"] & self._constant("RIGHT")
            for item in evaluations
        )
        candidates = [item for item in evaluations if item["eligible"]]
        if conflict or len(candidates) != 1:
            decision["skip_reason"] = "left_right_conflict" if conflict else "no_safe_model_request"
            # Route-required failure is checked even if the model is blocked,
            # silent or the connector is ambiguous: never wait at the endpoint.
            envelopes = [item for item in evaluations if item.get("target_lane_id")]
            if decision["route_required"] and (
                not envelopes or all(not item["distance_sufficient"] for item in envelopes)
            ):
                decision["fatal_reason"] = "route_required_lane_change_unavailable"
            return self._result(decision)
        candidate = candidates[0]
        decision.update(candidate)
        if not candidate["distance_sufficient"]:
            decision["skip_reason"] = "insufficient_remaining_distance"
            if decision["route_required"]:
                decision["fatal_reason"] = "route_required_lane_change_unavailable"
            else:
                logger.warning(
                    "Skipping late AV SUMO lane-change request: lane={} target={} "
                    "remaining={:.3f} required={:.3f}",
                    decision["source_lane_id"],
                    candidate["target_lane_id"],
                    candidate["lane_remaining"],
                    candidate["required_distance"],
                )
            return self._result(decision)
        self._latched_request_key = (
            decision["route_index"],
            decision["source_lane_id"],
            candidate["target_lane_id"],
        )
        self._request_generation += 1
        decision.update(
            command=candidate["direction"],
            issued=True,
            skip_reason="",
            request_generation=self._request_generation,
        )
        command = NDECommand(
            command_type=CommandType.LEFT if candidate["index_offset"] > 0 else CommandType.RIGHT,
            duration=self.DURATION,
            prob=1,
            info=dict(decision),
        )
        return self._result(decision, command)

    def derive_control_command_from_observation(self, obs_dict):
        veh_id = str(obs_dict["ego"]["veh_id"])
        decision = {
            "simulation_time": float(traci.simulation.getTime()),
            "vehicle_id": veh_id,
            "decision_source": "sumo_model",
            "command": "none",
            "issued": False,
            "fatal_reason": "",
        }
        env = getattr(getattr(self._agent, "simulator", None), "env", None)
        if env is not None and getattr(env, "av_control_mode", "sumo") != "sumo":
            self._reset()
            decision["skip_reason"] = "sumo_authority_inactive"
            return self._result(decision)
        try:
            source = str(traci.vehicle.getLaneID(veh_id))
            edge = str(traci.vehicle.getRoadID(veh_id))
            route = tuple(traci.vehicle.getRoute(veh_id))
            index = int(traci.vehicle.getRouteIndex(veh_id))
            decision.update(
                source_lane_id=source,
                edge_id=edge,
                lane_index=int(traci.vehicle.getLaneIndex(veh_id)),
                route_index=index,
            )
            if edge.startswith(":") or not 0 <= index < len(route):
                self._latched_request_key = None
                decision["skip_reason"] = "internal_or_invalid_route_position"
                return self._result(decision)
            next_edge = route[index + 1] if index + 1 < len(route) else ""
            decision["next_route_edge_id"] = next_edge
            decision["route_required"] = bool(
                next_edge and not self._route_targets(source, next_edge)
            )
            return self._select(veh_id, decision)
        except Exception as exc:
            decision["skip_reason"] = "decision_input_invalid"
            decision["decision_error"] = f"{type(exc).__name__}: {exc}"
            if decision.get("route_required"):
                decision["fatal_reason"] = "route_required_lane_change_unavailable"
            logger.warning("AV SUMO lane-change decision skipped: {}", decision["decision_error"])
            return self._result(decision)
