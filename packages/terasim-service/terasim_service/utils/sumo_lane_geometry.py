import math

import numpy as np


def extract_next_link_lane_ids(next_links):
    """Return internal and destination lane IDs from TraCI ``getNextLinks`` data."""
    lane_ids = []
    for link in next_links or []:
        # SUMO 1.23: (lane, priority, opened, foe, via, state, direction, length).
        # Some older wrappers expose via at index 1, so accept both layouts.
        for index in (4, 1, 0):
            if index >= len(link):
                continue
            lane_id = link[index]
            if isinstance(lane_id, str) and lane_id and lane_id not in lane_ids:
                lane_ids.append(lane_id)
    return lane_ids


def _as_2d_points(shape):
    try:
        points = [(float(point[0]), float(point[1])) for point in shape]
    except (TypeError, ValueError, IndexError):
        return []
    return points


def _as_3d_points(shape):
    try:
        points = [
            (float(point[0]), float(point[1]), float(point[2]))
            for point in shape
        ]
    except (TypeError, ValueError, IndexError):
        return []
    return points


def project_elevation_to_lane_shape(lane_shape_3d, position):
    """Interpolate lane elevation at the closest x/y point on a 3D shape."""
    points = _as_3d_points(lane_shape_3d)
    if len(points) < 2:
        return None

    try:
        pos_x = float(position[0])
        pos_y = float(position[1])
    except (TypeError, ValueError, IndexError):
        return None

    best = None
    for start, end in zip(points, points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        segment_length_sq = dx * dx + dy * dy
        if segment_length_sq <= 0.0:
            continue
        t = ((pos_x - start[0]) * dx + (pos_y - start[1]) * dy) / segment_length_sq
        t = min(1.0, max(0.0, t))
        proj_x = start[0] + dx * t
        proj_y = start[1] + dy * t
        distance_sq = (pos_x - proj_x) ** 2 + (pos_y - proj_y) ** 2
        if best is None or distance_sq < best["distance_sq"]:
            best = {
                "distance_sq": distance_sq,
                "projected_z": start[2] + (end[2] - start[2]) * t,
            }
    return None if best is None else best["projected_z"]


def flatten_lane_shapes(lane_shapes):
    """Flatten lane shape polylines while dropping repeated junction points."""
    points = []
    for lane_shape in lane_shapes or []:
        for point in _as_2d_points(lane_shape):
            if points and point == points[-1]:
                continue
            points.append(point)
    return points


def _project_distance_on_polyline(points, position):
    if len(points) < 2:
        return None

    try:
        pos_x = float(position[0])
        pos_y = float(position[1])
    except (TypeError, ValueError, IndexError):
        return None

    travelled = 0.0
    best_distance_sq = None
    best_projected_distance = 0.0

    for start, end in zip(points, points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        segment_length_sq = dx * dx + dy * dy
        if segment_length_sq <= 0.0:
            continue
        segment_length = math.sqrt(segment_length_sq)
        t = ((pos_x - start[0]) * dx + (pos_y - start[1]) * dy) / segment_length_sq
        t = min(1.0, max(0.0, t))
        proj_x = start[0] + dx * t
        proj_y = start[1] + dy * t
        distance_sq = (pos_x - proj_x) ** 2 + (pos_y - proj_y) ** 2
        if best_distance_sq is None or distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_projected_distance = travelled + segment_length * t
        travelled += segment_length

    return best_projected_distance


def project_position_to_lane_shape(lane_shape, position, lane_length=None):
    """Project an x/y position onto a lane shape.

    The returned lane position uses SUMO's lane length rather than the raw
    polyline length when ``lane_length`` is provided. This matters because
    ``vehicle.moveTo`` expects the distance from the lane start to the vehicle
    front bumper in SUMO lane coordinates.
    """
    points = _as_2d_points(lane_shape)
    if len(points) < 2:
        return None

    try:
        pos_x = float(position[0])
        pos_y = float(position[1])
    except (TypeError, ValueError, IndexError):
        return None

    travelled = 0.0
    total_length = 0.0
    best = None
    for start, end in zip(points, points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        segment_length = math.hypot(dx, dy)
        if segment_length <= 0.0:
            continue
        segment_length_sq = segment_length * segment_length
        t = ((pos_x - start[0]) * dx + (pos_y - start[1]) * dy) / segment_length_sq
        t = min(1.0, max(0.0, t))
        proj_x = start[0] + dx * t
        proj_y = start[1] + dy * t
        distance = math.hypot(pos_x - proj_x, pos_y - proj_y)
        projected_distance = travelled + segment_length * t
        sumo_angle = (90.0 - math.degrees(math.atan2(dy, dx))) % 360.0
        if best is None or distance < best["distance"]:
            best = {
                "shape_position": projected_distance,
                "distance": distance,
                "sumo_angle": sumo_angle,
                "projected_x": proj_x,
                "projected_y": proj_y,
            }
        travelled += segment_length
        total_length += segment_length

    if best is None or total_length <= 0.0:
        return None

    if lane_length is None:
        sumo_lane_length = total_length
    else:
        try:
            sumo_lane_length = max(0.0, float(lane_length))
        except (TypeError, ValueError):
            return None
    best["lane_position"] = min(
        sumo_lane_length,
        max(0.0, best["shape_position"] / total_length * sumo_lane_length),
    )
    best["shape_length"] = total_length
    return best


def _angle_difference_degrees(first, second):
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def select_route_aware_lane_projection(
    position,
    sumo_angle,
    lane_candidates,
    position_z=None,
    current_lane_id="",
    lane_switch_hysteresis=0.35,
    heading_weight=0.02,
    max_distance=None,
    max_elevation_error=None,
    max_heading_error=90.0,
    prefer_current_lane=False,
):
    """Select the best projection among caller-supplied route-aware lanes.

    This function deliberately does not search the whole SUMO network. The
    caller is responsible for supplying only the current edge, adjacent lanes,
    route successors and applicable internal lanes.
    """
    try:
        sumo_angle = float(sumo_angle)
        lane_switch_hysteresis = max(0.0, float(lane_switch_hysteresis))
        heading_weight = max(0.0, float(heading_weight))
        max_heading_error = max(0.0, float(max_heading_error))
        if max_distance is not None:
            max_distance = max(0.0, float(max_distance))
        if position_z is not None:
            position_z = float(position_z)
        if max_elevation_error is not None:
            max_elevation_error = max(0.0, float(max_elevation_error))
    except (TypeError, ValueError):
        return None

    best = None
    current_lane_projection = None
    for candidate in lane_candidates or []:
        lane_id = candidate.get("lane_id")
        if not lane_id:
            continue
        projection = project_position_to_lane_shape(
            candidate.get("shape"),
            position,
            candidate.get("length"),
        )
        if projection is None:
            continue
        heading_error = _angle_difference_degrees(
            sumo_angle,
            projection["sumo_angle"],
        )
        if heading_error > max_heading_error:
            continue
        if max_distance is not None and projection["distance"] > max_distance:
            continue
        projected_z = project_elevation_to_lane_shape(
            candidate.get("shape3d"),
            position,
        )
        elevation_error = None
        if projected_z is not None and position_z is not None:
            elevation_error = abs(position_z - projected_z)
            if (
                max_elevation_error is not None
                and elevation_error > max_elevation_error
            ):
                continue

        continuity_penalty = 0.0 if lane_id == current_lane_id else lane_switch_hysteresis
        score = projection["distance"] + heading_weight * heading_error + continuity_penalty
        result = {
            **projection,
            "lane_id": lane_id,
            "edge_id": candidate.get("edge_id", ""),
            "lane_index": candidate.get("lane_index", -1),
            "route_offset": candidate.get("route_offset", 0.0),
            "heading_error": heading_error,
            "projected_z": projected_z,
            "elevation_error": elevation_error,
            "score": score,
        }
        if lane_id == current_lane_id:
            current_lane_projection = result
        if best is None or result["score"] < best["score"]:
            best = result
    if prefer_current_lane and current_lane_projection is not None:
        return current_lane_projection
    return best


def point_at_distance_on_polyline(points, target_distance, z=0.0):
    """Return a 3D point at the requested travelled distance along a polyline."""
    if len(points) < 2:
        return None

    try:
        target_distance = max(0.0, float(target_distance))
        z = float(z)
    except (TypeError, ValueError):
        return None

    travelled = 0.0
    last_valid_end = None
    for start, end in zip(points, points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        segment_length = math.hypot(dx, dy)
        if segment_length <= 0.0:
            continue
        last_valid_end = end
        if travelled + segment_length >= target_distance:
            ratio = (target_distance - travelled) / segment_length
            return (
                start[0] + dx * ratio,
                start[1] + dy * ratio,
                z,
            )
        travelled += segment_length

    if last_valid_end is None:
        return None
    return last_valid_end[0], last_valid_end[1], z


def find_lookahead_position_from_lane_shapes(
    lane_shapes, current_position, lookahead_distance, z=0.0
):
    """Find a lookahead point on lane-shape polylines near a moving SUMO vehicle."""
    points = flatten_lane_shapes(lane_shapes)
    projected_distance = _project_distance_on_polyline(points, current_position)
    if projected_distance is None:
        return None
    return point_at_distance_on_polyline(points, projected_distance + lookahead_distance, z)


def compile_lane_shapes(lane_shapes):
    """Compile lane polylines into NumPy arrays reusable across simulation ticks."""
    points = np.asarray(flatten_lane_shapes(lane_shapes), dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        return None

    starts = points[:-1]
    deltas = points[1:] - starts
    length_sq = np.einsum("ij,ij->i", deltas, deltas)
    valid = length_sq > 0.0
    if not np.any(valid):
        return None

    starts = starts[valid]
    deltas = deltas[valid]
    length_sq = length_sq[valid]
    lengths = np.sqrt(length_sq)
    cumulative_starts = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(lengths[:-1]))
    )
    cumulative_ends = cumulative_starts + lengths
    return {
        "starts": starts,
        "deltas": deltas,
        "length_sq": length_sq,
        "lengths": lengths,
        "cumulative_starts": cumulative_starts,
        "cumulative_ends": cumulative_ends,
        "last_end": starts[-1] + deltas[-1],
    }


def adapt_lookahead_distances_for_compiled_paths(
    compiled_paths,
    current_positions,
    base_distances,
    min_curve_distance=3.5,
    curve_start_radians=math.radians(5.0),
    curve_full_scale_radians=math.radians(45.0),
):
    """Shorten lookahead when the cached route bends inside the base horizon."""
    count = len(compiled_paths)
    if len(current_positions) != count or len(base_distances) != count:
        raise ValueError("compiled path and vehicle input lengths must match")

    try:
        min_curve_distance = max(0.1, float(min_curve_distance))
        curve_start_radians = max(0.0, float(curve_start_radians))
        curve_full_scale_radians = max(
            curve_start_radians + 1e-6, float(curve_full_scale_radians)
        )
    except (TypeError, ValueError):
        raise ValueError("invalid adaptive lookahead settings") from None

    effective = [max(0.0, float(distance)) for distance in base_distances]
    heading_changes = [0.0] * count
    groups = {}
    for index, compiled_path in enumerate(compiled_paths):
        if compiled_path is not None:
            groups.setdefault(id(compiled_path), (compiled_path, []))[1].append(index)

    for compiled_path, indices in groups.values():
        try:
            positions = np.asarray(
                [current_positions[index] for index in indices], dtype=np.float64
            )
            distances = np.maximum(
                0.0, np.asarray([base_distances[index] for index in indices], dtype=np.float64)
            )
        except (TypeError, ValueError):
            continue
        if positions.ndim != 2 or positions.shape[1] != 2:
            continue

        starts = compiled_path["starts"]
        deltas = compiled_path["deltas"]
        length_sq = compiled_path["length_sq"]
        lengths = compiled_path["lengths"]
        cumulative_starts = compiled_path["cumulative_starts"]
        cumulative_ends = compiled_path["cumulative_ends"]

        offsets = positions[:, None, :] - starts[None, :, :]
        ratios = np.einsum("bsi,si->bs", offsets, deltas) / length_sq[None, :]
        ratios = np.clip(ratios, 0.0, 1.0)
        projections = starts[None, :, :] + ratios[..., None] * deltas[None, :, :]
        distance_sq = np.sum((positions[:, None, :] - projections) ** 2, axis=2)
        nearest_indices = np.argmin(distance_sq, axis=1)
        rows = np.arange(len(indices))
        projected_distances = (
            cumulative_starts[nearest_indices]
            + lengths[nearest_indices] * ratios[rows, nearest_indices]
        )
        base_targets = projected_distances + distances
        target_indices = np.searchsorted(cumulative_ends, base_targets, side="left")
        target_indices = np.minimum(target_indices, len(lengths) - 1)

        unit_directions = deltas / lengths[:, None]
        current_directions = unit_directions[nearest_indices]
        dots = np.einsum("bi,si->bs", current_directions, unit_directions)
        angle_differences = np.arccos(np.clip(dots, -1.0, 1.0))
        segment_indices = np.arange(len(lengths))[None, :]
        within_horizon = (
            (segment_indices >= nearest_indices[:, None])
            & (segment_indices <= target_indices[:, None])
        )
        max_heading_changes = np.max(
            np.where(within_horizon, angle_differences, 0.0), axis=1
        )
        curve_factors = np.clip(
            (max_heading_changes - curve_start_radians)
            / (curve_full_scale_radians - curve_start_radians),
            0.0,
            1.0,
        )
        minimums = np.minimum(distances, min_curve_distance)
        adapted = distances - curve_factors * (distances - minimums)

        for row, result_index in enumerate(indices):
            effective[result_index] = float(adapted[row])
            heading_changes[result_index] = float(max_heading_changes[row])

    return effective, heading_changes


def find_lookahead_positions_from_compiled_paths(
    compiled_paths,
    current_positions,
    lookahead_distances,
    z_values,
):
    """Calculate route lookahead points for multiple vehicles in vectorized groups.

    Vehicles sharing the same compiled route path are projected together. The
    result matches the scalar lane-shape lookup, while avoiding repeated
    polyline flattening and per-segment Python loops.
    """
    count = len(compiled_paths)
    if not (
        len(current_positions) == count
        and len(lookahead_distances) == count
        and len(z_values) == count
    ):
        raise ValueError("compiled path and vehicle input lengths must match")

    results = [None] * count
    groups = {}
    for index, compiled_path in enumerate(compiled_paths):
        if compiled_path is not None:
            groups.setdefault(id(compiled_path), (compiled_path, []))[1].append(index)

    for compiled_path, indices in groups.values():
        try:
            positions = np.asarray(
                [current_positions[index] for index in indices], dtype=np.float64
            )
            distances = np.maximum(
                0.0,
                np.asarray(
                    [lookahead_distances[index] for index in indices], dtype=np.float64
                ),
            )
            z_array = np.asarray([z_values[index] for index in indices], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if positions.ndim != 2 or positions.shape[1] != 2:
            continue

        starts = compiled_path["starts"]
        deltas = compiled_path["deltas"]
        length_sq = compiled_path["length_sq"]
        lengths = compiled_path["lengths"]
        cumulative_starts = compiled_path["cumulative_starts"]
        cumulative_ends = compiled_path["cumulative_ends"]

        offsets = positions[:, None, :] - starts[None, :, :]
        ratios = np.einsum("bsi,si->bs", offsets, deltas) / length_sq[None, :]
        ratios = np.clip(ratios, 0.0, 1.0)
        projections = starts[None, :, :] + ratios[..., None] * deltas[None, :, :]
        distance_sq = np.sum((positions[:, None, :] - projections) ** 2, axis=2)
        nearest_indices = np.argmin(distance_sq, axis=1)
        rows = np.arange(len(indices))
        projected_distances = (
            cumulative_starts[nearest_indices]
            + lengths[nearest_indices] * ratios[rows, nearest_indices]
        )
        target_distances = projected_distances + distances
        target_indices = np.searchsorted(cumulative_ends, target_distances, side="left")
        beyond_path = target_indices >= len(lengths)
        target_indices = np.minimum(target_indices, len(lengths) - 1)
        target_ratios = (
            target_distances - cumulative_starts[target_indices]
        ) / lengths[target_indices]
        target_ratios = np.clip(target_ratios, 0.0, 1.0)
        target_points = (
            starts[target_indices] + target_ratios[:, None] * deltas[target_indices]
        )
        target_points[beyond_path] = compiled_path["last_end"]

        for row, result_index in enumerate(indices):
            results[result_index] = (
                float(target_points[row, 0]),
                float(target_points[row, 1]),
                float(z_array[row]),
            )

    return results


def build_external_state_lateral_action_lookahead(
    compiled_path,
    current_position,
    lookahead_distance,
    *,
    lateral_speed=0.0,
    desired_speed=None,
    min_forward_speed=0.2,
    lateral_speed_epsilon=0.05,
    phase_a_position=None,
    phase_step_length=0.05,
    previous_world_lateral_direction=None,
    previous_direction_min_alignment=0.5,
    z=0.0,
):
    """Build a rolling lookahead from live route geometry and SUMO lateral speed.

    The live Phase B front position remains the origin. SUMO route geometry supplies
    longitudinal displacement. The raw Phase A-to-B world displacement determines
    which side of the live route SUMO moved toward; only the magnitude of speedLat
    is used. If one Phase-B step has no measurable lateral displacement, a direction
    confirmed from the immediately preceding step may be projected onto the live
    route normal. Otherwise the action remains valid and uses route-only lookahead.
    SUMO lane-change intent is deliberately not an input to steering direction.
    """

    def invalid(reason):
        return {
            "valid": False,
            "error": reason,
            "mode": "route",
            "lookahead": None,
            "route_lookahead": None,
            "lateral_displacement": 0.0,
            "target_lateral_distance": None,
            "route_tangent_x": None,
            "route_tangent_y": None,
            "world_left_normal_x": None,
            "world_left_normal_y": None,
            "phase_b_lateral_delta": None,
            "expected_phase_b_lateral_distance": None,
            "world_lateral_speed": 0.0,
            "warning": "",
            "lateral_direction_source": "inactive",
        }

    try:
        origin = np.asarray(
            [float(current_position[0]), float(current_position[1])],
            dtype=np.float64,
        )
        distance = max(0.0, float(lookahead_distance))
        phase_step_length = float(phase_step_length)
        z = float(z)
        phase_a = (
            None
            if phase_a_position is None
            else np.asarray(
                [float(phase_a_position[0]), float(phase_a_position[1])],
                dtype=np.float64,
            )
        )
    except (TypeError, ValueError, IndexError):
        return invalid("invalid_route_input")
    if (
        not np.all(np.isfinite(origin))
        or not math.isfinite(distance)
        or not math.isfinite(phase_step_length)
        or phase_step_length <= 0.0
        or not math.isfinite(z)
        or (phase_a is not None and not np.all(np.isfinite(phase_a)))
    ):
        return invalid("non_finite_route_input")
    if compiled_path is None:
        return invalid("missing_route_geometry")

    try:
        starts = np.asarray(compiled_path["starts"], dtype=np.float64)
        deltas = np.asarray(compiled_path["deltas"], dtype=np.float64)
        length_sq = np.asarray(compiled_path["length_sq"], dtype=np.float64)
        lengths = np.asarray(compiled_path["lengths"], dtype=np.float64)
        cumulative_starts = np.asarray(compiled_path["cumulative_starts"], dtype=np.float64)
        cumulative_ends = np.asarray(compiled_path["cumulative_ends"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return invalid("invalid_route_geometry")
    if (
        starts.ndim != 2
        or starts.shape[1] != 2
        or len(starts) == 0
        or deltas.shape != starts.shape
        or len(length_sq) != len(starts)
        or len(lengths) != len(starts)
        or not all(
            np.all(np.isfinite(values))
            for values in (starts, deltas, length_sq, lengths, cumulative_starts, cumulative_ends)
        )
        or np.any(length_sq <= 0.0)
        or np.any(lengths <= 0.0)
    ):
        return invalid("invalid_route_geometry")

    offsets = origin[None, :] - starts
    ratios = np.einsum("si,si->s", offsets, deltas) / length_sq
    ratios = np.clip(ratios, 0.0, 1.0)
    projections = starts + ratios[:, None] * deltas
    nearest_index = int(np.argmin(np.sum((origin[None, :] - projections) ** 2, axis=1)))
    projected_point = projections[nearest_index]
    projected_distance = (
        cumulative_starts[nearest_index] + lengths[nearest_index] * ratios[nearest_index]
    )
    target_distance = projected_distance + distance
    target_index = int(np.searchsorted(cumulative_ends, target_distance, side="left"))
    if target_index >= len(lengths):
        target_point = starts[-1] + deltas[-1]
    else:
        target_ratio = (target_distance - cumulative_starts[target_index]) / lengths[target_index]
        target_ratio = min(1.0, max(0.0, float(target_ratio)))
        target_point = starts[target_index] + target_ratio * deltas[target_index]

    tangent = deltas[nearest_index] / lengths[nearest_index]
    left_normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    phase_b_lateral_delta = (
        None
        if phase_a is None
        else float(np.dot(origin - phase_a, left_normal))
    )
    route_displacement = target_point - projected_point
    route_lookahead_xy = origin + route_displacement
    route_lookahead = (
        float(route_lookahead_xy[0]),
        float(route_lookahead_xy[1]),
        z,
    )
    result = {
        "valid": True,
        "error": "",
        "warning": "",
        "mode": "route",
        "lookahead": route_lookahead,
        "route_lookahead": route_lookahead,
        "lateral_displacement": 0.0,
        "target_lateral_distance": None,
        "route_tangent_x": float(tangent[0]),
        "route_tangent_y": float(tangent[1]),
        "world_left_normal_x": float(left_normal[0]),
        "world_left_normal_y": float(left_normal[1]),
        "phase_b_lateral_delta": phase_b_lateral_delta,
        "expected_phase_b_lateral_distance": None,
        "world_lateral_speed": 0.0,
        "lateral_direction_source": "inactive",
    }

    try:
        lateral_speed = float(lateral_speed)
        lateral_speed_epsilon = max(0.0, float(lateral_speed_epsilon))
    except (TypeError, ValueError):
        return invalid("invalid_lateral_action")
    if not all(math.isfinite(value) for value in (lateral_speed, lateral_speed_epsilon)):
        return invalid("non_finite_lateral_action")
    result["expected_phase_b_lateral_distance"] = abs(lateral_speed) * phase_step_length
    if abs(lateral_speed) <= lateral_speed_epsilon + 1e-9:
        return result
    if phase_b_lateral_delta is None:
        result["valid"] = False
        result["error"] = "missing_phase_a_world_position"
        result["lookahead"] = None
        return result
    direction_sign = None
    if abs(phase_b_lateral_delta) > 1e-9:
        direction_sign = math.copysign(1.0, phase_b_lateral_delta)
        result["lateral_direction_source"] = "phase_b_delta"
    elif previous_world_lateral_direction is not None:
        try:
            previous_direction = np.asarray(
                [
                    float(previous_world_lateral_direction[0]),
                    float(previous_world_lateral_direction[1]),
                ],
                dtype=np.float64,
            )
            previous_direction_min_alignment = min(
                1.0, max(0.0, float(previous_direction_min_alignment))
            )
        except (TypeError, ValueError, IndexError):
            previous_direction = None
        if previous_direction is not None and np.all(np.isfinite(previous_direction)):
            previous_norm = float(np.linalg.norm(previous_direction))
            if previous_norm > 1e-12:
                previous_direction = previous_direction / previous_norm
                alignment = float(np.dot(previous_direction, left_normal))
                if abs(alignment) >= previous_direction_min_alignment:
                    direction_sign = math.copysign(1.0, alignment)
                    result["lateral_direction_source"] = "previous_confirmed"

    if direction_sign is None:
        result["mode"] = "route_only_unresolved_lateral"
        result["warning"] = "unresolved_phase_b_lateral_direction"
        result["lateral_direction_source"] = "route_only"
        return result

    world_lateral_speed = direction_sign * abs(lateral_speed)
    result["world_lateral_speed"] = world_lateral_speed

    try:
        desired_speed = float(desired_speed)
        min_forward_speed = max(0.0, float(min_forward_speed))
    except (TypeError, ValueError):
        return invalid("invalid_lateral_action")
    if not all(math.isfinite(value) for value in (desired_speed, min_forward_speed)):
        return invalid("non_finite_lateral_action")

    lateral_displacement = 0.0
    mode = "deferred"
    if desired_speed > min_forward_speed:
        lateral_displacement = world_lateral_speed * distance / desired_speed
        mode = "sumo_lateral_velocity"

    route_lookahead_xy = route_lookahead_xy + left_normal * lateral_displacement
    result["mode"] = mode
    result["lateral_displacement"] = float(lateral_displacement)
    result["lookahead"] = (
        float(route_lookahead_xy[0]),
        float(route_lookahead_xy[1]),
        z,
    )
    return result


def project_position_by_sumo_angle(position, sumo_angle, distance, z=0.0):
    """Project a SUMO position forward using SUMO's heading convention."""
    try:
        x = float(position[0])
        y = float(position[1])
        sumo_angle = float(sumo_angle)
        distance = float(distance)
        z = float(z)
    except (TypeError, ValueError, IndexError):
        return None
    heading = math.radians(90.0 - sumo_angle)
    return (
        x + math.cos(heading) * distance,
        y + math.sin(heading) * distance,
        z,
    )


def _smoothstep(value, start, full):
    try:
        value = abs(float(value))
        start = max(0.0, float(start))
        full = max(start + 1e-6, float(full))
    except (TypeError, ValueError):
        return 0.0
    ratio = min(1.0, max(0.0, (value - start) / (full - start)))
    return ratio * ratio * (3.0 - 2.0 * ratio)


def blend_lane_change_lookahead(
    route_lookahead,
    current_position,
    sumo_angle,
    distance,
    lateral_speed=0.0,
    lateral_offset=0.0,
    z=0.0,
    speed_start=0.05,
    speed_full=0.35,
    offset_start=0.15,
    offset_full=0.75,
):
    """Blend a route target with the instantaneous SUMO lane-change heading.

    The blend is zero for ordinary lane following, and approaches one while
    SUMO reports meaningful lateral motion or lane-relative displacement.
    This keeps the target continuous when SUMO switches the current lane ID.
    """
    heading_lookahead = project_position_by_sumo_angle(
        current_position, sumo_angle, distance, z
    )
    if route_lookahead is None:
        return heading_lookahead, 0.0
    if heading_lookahead is None:
        return route_lookahead, 0.0

    blend = max(
        _smoothstep(lateral_speed, speed_start, speed_full),
        _smoothstep(lateral_offset, offset_start, offset_full),
    )
    if blend <= 0.0:
        return route_lookahead, 0.0
    inverse = 1.0 - blend
    return (
        inverse * float(route_lookahead[0]) + blend * heading_lookahead[0],
        inverse * float(route_lookahead[1]) + blend * heading_lookahead[1],
        inverse * float(route_lookahead[2]) + blend * heading_lookahead[2],
    ), blend


def reconstruct_position_from_lane_geometry(
    lane_shape,
    lane_position: float,
    lateral_offset: float,
    z: float = 0.0,
    lane_length: float | None = None,
    reference_position=None,
):
    """Reconstruct x/y/z, using raw x/y to disambiguate lateral sign.

    SUMO's internal lateral sign changes with the network drive side, while a
    lane shape alone carries no drive-side metadata. ``reference_position``
    selects the geometrically matching side without changing lane progress.
    """
    if not lane_shape or len(lane_shape) < 2:
        return None

    try:
        points = [(float(point[0]), float(point[1])) for point in lane_shape]
        target_distance = max(0.0, float(lane_position))
        lateral_offset = float(lateral_offset)
        z = float(z)
        lane_length = None if lane_length is None else float(lane_length)
        reference_xy = (
            None
            if reference_position is None
            else (float(reference_position[0]), float(reference_position[1]))
        )
    except (TypeError, ValueError, IndexError):
        return None

    def with_lateral_offset(center_x, center_y, normal_x, normal_y):
        primary = (
            center_x + normal_x * lateral_offset,
            center_y + normal_y * lateral_offset,
            z,
        )
        if reference_xy is None or lateral_offset == 0.0:
            return primary
        mirrored = (
            center_x - normal_x * lateral_offset,
            center_y - normal_y * lateral_offset,
            z,
        )
        primary_error = math.hypot(
            primary[0] - reference_xy[0], primary[1] - reference_xy[1]
        )
        mirrored_error = math.hypot(
            mirrored[0] - reference_xy[0], mirrored[1] - reference_xy[1]
        )
        return mirrored if mirrored_error < primary_error else primary

    segments = []
    shape_length = 0.0
    for start, end in zip(points, points[1:]):
        segment_length = math.hypot(end[0] - start[0], end[1] - start[1])
        segments.append((start, end, segment_length))
        shape_length += segment_length
    if lane_length is not None and lane_length > 0.0 and shape_length > 0.0:
        # SUMO lanePosition uses the lane's declared length. Repaired or
        # simplified lane shapes may have a different polyline length, so use
        # normalized lane progress rather than treating both metres as equal.
        progress = min(1.0, target_distance / lane_length)
        target_distance = progress * shape_length

    travelled = 0.0
    last_segment = None
    for start, end, segment_length in segments:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if segment_length <= 0.0:
            continue
        last_segment = (start, dx, dy, segment_length)
        if travelled + segment_length >= target_distance:
            ratio = (target_distance - travelled) / segment_length
            center_x = start[0] + dx * ratio
            center_y = start[1] + dy * ratio
            normal_x = -dy / segment_length
            normal_y = dx / segment_length
            return with_lateral_offset(center_x, center_y, normal_x, normal_y)
        travelled += segment_length

    if last_segment is None:
        return None

    start, dx, dy, segment_length = last_segment
    end_x = start[0] + dx
    end_y = start[1] + dy
    normal_x = -dy / segment_length
    normal_y = dx / segment_length
    return with_lateral_offset(end_x, end_y, normal_x, normal_y)
