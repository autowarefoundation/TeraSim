import math

import numpy as np

LANE_SHAPE_JOIN_TOLERANCE = 0.05
CANONICAL_AMBIGUOUS_TURN_DOT = math.cos(math.radians(170.0))
CANONICAL_CURVATURE_HALF_WINDOW = 0.5
CANONICAL_MIN_BODY_CURVATURE_HALF_WINDOW = 1.0


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
        points = [(float(point[0]), float(point[1]), float(point[2])) for point in shape]
    except (TypeError, ValueError, IndexError):
        return []
    return points


def _normalized_lane_shape(shape):
    points = []
    for point in _as_2d_points(shape):
        if not all(math.isfinite(value) for value in point):
            return []
        if points and point == points[-1]:
            continue
        points.append(point)
    return points if len(points) >= 2 else []


def _project_point_to_polyline(points, position):
    travelled = 0.0
    best = None
    for segment_index, (start, end) in enumerate(zip(points, points[1:])):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_sq = dx * dx + dy * dy
        if length_sq <= 0.0:
            continue
        length = math.sqrt(length_sq)
        ratio = ((position[0] - start[0]) * dx + (position[1] - start[1]) * dy) / length_sq
        ratio = min(1.0, max(0.0, ratio))
        projected = (start[0] + ratio * dx, start[1] + ratio * dy)
        distance_sq = (position[0] - projected[0]) ** 2 + (position[1] - projected[1]) ** 2
        progress = travelled + ratio * length
        candidate = {
            "distance_sq": distance_sq,
            "progress": progress,
            "segment_index": segment_index,
            "segment_ratio": ratio,
            "direction": (dx / length, dy / length),
        }
        if best is None or (distance_sq, progress) < (
            best["distance_sq"],
            best["progress"],
        ):
            best = candidate
        travelled += length
    return best


def _last_polyline_direction(points):
    for start, end in reversed(tuple(zip(points, points[1:]))):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length > 0.0:
            return dx / length, dy / length
    return None


def _stitch_lane_shapes(lane_shapes, join_tolerance=LANE_SHAPE_JOIN_TOLERANCE):
    """Build one forward polyline without inventing lane-boundary segments."""
    try:
        join_tolerance = max(0.0, float(join_tolerance))
    except (TypeError, ValueError):
        join_tolerance = LANE_SHAPE_JOIN_TOLERANCE

    shapes = list(lane_shapes or [])
    points = []
    diagnostics = {
        "shape_join_input_count": len(shapes),
        "shape_join_accepted_count": 0,
        "shape_join_trimmed_count": 0,
        "shape_join_trimmed_distance": 0.0,
        "shape_join_truncated_index": None,
        "shape_join_endpoint_distance": None,
        "shape_join_projection_error": None,
    }

    for shape_index, lane_shape in enumerate(shapes):
        shape_points = _normalized_lane_shape(lane_shape)
        if not shape_points:
            diagnostics["shape_join_truncated_index"] = shape_index
            break

        if not points:
            points.extend(shape_points)
            diagnostics["shape_join_accepted_count"] += 1
            continue

        endpoint = points[-1]
        endpoint_distance = math.dist(endpoint, shape_points[0])
        diagnostics["shape_join_endpoint_distance"] = endpoint_distance
        if endpoint_distance <= join_tolerance:
            points.extend(shape_points[1:])
            diagnostics["shape_join_accepted_count"] += 1
            continue

        projection = _project_point_to_polyline(shape_points, endpoint)
        previous_direction = _last_polyline_direction(points)
        projection_error = None if projection is None else math.sqrt(projection["distance_sq"])
        diagnostics["shape_join_projection_error"] = projection_error
        aligned = bool(
            projection is not None
            and previous_direction is not None
            and (
                previous_direction[0] * projection["direction"][0]
                + previous_direction[1] * projection["direction"][1]
            )
            > 0.0
        )
        if not (
            projection is not None
            and projection_error <= join_tolerance
            and projection["progress"] > 0.0
            and aligned
        ):
            diagnostics["shape_join_truncated_index"] = shape_index
            break

        suffix_index = projection["segment_index"] + 1
        if projection["segment_ratio"] >= 1.0 - 1e-12:
            suffix_index += 1
        points.extend(shape_points[suffix_index:])
        diagnostics["shape_join_accepted_count"] += 1
        diagnostics["shape_join_trimmed_count"] += 1
        diagnostics["shape_join_trimmed_distance"] += projection["progress"]

    return points, diagnostics


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
    """Build a canonical forward polyline from ordered lane shapes."""
    points, _ = _stitch_lane_shapes(lane_shapes)
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
            if max_elevation_error is not None and elevation_error > max_elevation_error:
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


def compile_lane_shapes(lane_shapes, lane_ids=None):
    """Compile an ordered SUMO route into a canonical forward polyline.

    Segment geometry remains piecewise linear, while vertex tangents blend the
    incoming and outgoing segment directions.  Interpolating those tangents at
    projection and target points keeps the route-left normal continuous across
    ordinary polyline vertices without inventing connector positions.
    """
    points, diagnostics = _stitch_lane_shapes(lane_shapes)
    points = np.asarray(points, dtype=np.float64)
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
    unit_directions = deltas / lengths[:, None]
    vertex_tangents = np.empty((len(unit_directions) + 1, 2), dtype=np.float64)
    vertex_tangents[0] = unit_directions[0]
    vertex_tangents[-1] = unit_directions[-1]
    for vertex_index in range(1, len(unit_directions)):
        incoming = unit_directions[vertex_index - 1]
        outgoing = unit_directions[vertex_index]
        direction_dot = float(np.dot(incoming, outgoing))
        if not math.isfinite(direction_dot) or direction_dot <= CANONICAL_AMBIGUOUS_TURN_DOT:
            return None
        blended = incoming + outgoing
        norm = float(np.linalg.norm(blended))
        if not math.isfinite(norm) or norm <= 1e-9:
            return None
        vertex_tangents[vertex_index] = blended / norm

    cumulative_starts = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(lengths[:-1])))
    cumulative_ends = cumulative_starts + lengths
    return {
        "points": points,
        "starts": starts,
        "deltas": deltas,
        "length_sq": length_sq,
        "lengths": lengths,
        "unit_directions": unit_directions,
        "vertex_tangents": vertex_tangents,
        "cumulative_starts": cumulative_starts,
        "cumulative_ends": cumulative_ends,
        "total_length": float(cumulative_ends[-1]),
        "last_end": starts[-1] + deltas[-1],
        "lane_ids": tuple(lane_ids or ()),
        **diagnostics,
    }


def _compiled_route_tangent(compiled_path, segment_index, segment_ratio):
    try:
        vertex_tangents = np.asarray(compiled_path["vertex_tangents"], dtype=np.float64)
        segment_index = int(segment_index)
        ratio = min(1.0, max(0.0, float(segment_ratio)))
        tangent = (1.0 - ratio) * vertex_tangents[segment_index] + ratio * vertex_tangents[
            segment_index + 1
        ]
        norm = float(np.linalg.norm(tangent))
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return tangent / norm


def project_position_to_compiled_route(compiled_path, position, *, progress_hint=None):
    """Project a world point to canonical route-Frenet coordinates.

    ``d`` is positive along the world-left normal of the ordered SUMO route.
    The optional progress hint is used only to break equal-distance projection
    ties at overlapping geometry; Euclidean distance remains authoritative.
    """
    if compiled_path is None:
        return None
    try:
        point = np.asarray([float(position[0]), float(position[1])], dtype=np.float64)
        starts = np.asarray(compiled_path["starts"], dtype=np.float64)
        deltas = np.asarray(compiled_path["deltas"], dtype=np.float64)
        length_sq = np.asarray(compiled_path["length_sq"], dtype=np.float64)
        lengths = np.asarray(compiled_path["lengths"], dtype=np.float64)
        cumulative_starts = np.asarray(compiled_path["cumulative_starts"], dtype=np.float64)
        hint = None if progress_hint is None else float(progress_hint)
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if (
        not np.all(np.isfinite(point))
        or starts.ndim != 2
        or starts.shape[1] != 2
        or len(starts) == 0
        or deltas.shape != starts.shape
        or len(length_sq) != len(starts)
        or len(lengths) != len(starts)
        or len(cumulative_starts) != len(starts)
        or np.any(length_sq <= 0.0)
        or np.any(lengths <= 0.0)
        or not all(
            np.all(np.isfinite(values))
            for values in (starts, deltas, length_sq, lengths, cumulative_starts)
        )
        or (hint is not None and not math.isfinite(hint))
    ):
        return None

    offsets = point[None, :] - starts
    unclamped_ratios = np.einsum("si,si->s", offsets, deltas) / length_sq
    ratios = unclamped_ratios
    ratios = np.clip(ratios, 0.0, 1.0)
    projections = starts + ratios[:, None] * deltas
    distance_sq = np.sum((point[None, :] - projections) ** 2, axis=1)
    progress = cumulative_starts + lengths * ratios
    if hint is None:
        index = int(np.argmin(distance_sq))
    else:
        # Stable tie handling matters at lane endpoints and overlapping
        # internal-lane geometry.  Do not let the hint override a genuinely
        # closer segment.
        minimum = float(np.min(distance_sq))
        tied = np.flatnonzero(distance_sq <= minimum + 1e-12)
        index = int(
            min(
                tied,
                key=lambda candidate: (
                    abs(float(progress[candidate]) - hint),
                    -float(progress[candidate]),
                    int(candidate),
                ),
            )
        )

    tangent = _compiled_route_tangent(compiled_path, index, ratios[index])
    if tangent is None:
        return None
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    projected = projections[index]
    projected_offset = point - projected
    lateral_dot = float(np.dot(projected_offset, normal))
    along_residual = float(np.dot(projected_offset, tangent))
    distance = math.sqrt(float(distance_sq[index]))
    d = 0.0 if abs(lateral_dot) <= 1e-12 else lateral_dot
    endpoint_clamped = bool(unclamped_ratios[index] < 0.0 or unclamped_ratios[index] > 1.0)
    values = (
        *projected,
        *tangent,
        *normal,
        progress[index],
        d,
        distance,
        along_residual,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return None
    return {
        "s": float(progress[index]),
        "d": d,
        "projected_x": float(projected[0]),
        "projected_y": float(projected[1]),
        "tangent_x": float(tangent[0]),
        "tangent_y": float(tangent[1]),
        "normal_x": float(normal[0]),
        "normal_y": float(normal[1]),
        "segment_index": index,
        "segment_ratio": float(ratios[index]),
        "distance": distance,
        "projection_distance": distance,
        "projection_along_residual": along_residual,
        "projection_endpoint_clamped": endpoint_clamped,
    }


def build_active_maneuver_lane_corridor(  # noqa: C901
    reference_frame,
    source_lane_shape,
    source_lane_width,
    target_lane_shape,
    target_lane_width,
    *,
    lane_width_tolerance=0.01,
):
    """Build the lateral envelope joining source and target lane outer edges.

    The returned bounds use the same canonical world-left normal as
    ``reference_frame``. The envelope spans both adjacent lane ribbons so a
    SUMO lane-change trajectory may pass between their centerlines.
    """

    def invalid(reason):
        return {
            "valid": False,
            "reason": reason,
            "d_min": None,
            "d_max": None,
            "source_center_d": None,
            "target_center_d": None,
            "source_lane_width": None,
            "target_lane_width": None,
            "center_separation": None,
        }

    try:
        reference_point = np.asarray(
            [reference_frame["projected_x"], reference_frame["projected_y"]],
            dtype=np.float64,
        )
        reference_tangent = np.asarray(
            [reference_frame["tangent_x"], reference_frame["tangent_y"]],
            dtype=np.float64,
        )
        reference_normal = np.asarray(
            [reference_frame["normal_x"], reference_frame["normal_y"]],
            dtype=np.float64,
        )
        source_lane_width = float(source_lane_width)
        target_lane_width = float(target_lane_width)
        lane_width_tolerance = max(0.0, float(lane_width_tolerance))
    except (KeyError, TypeError, ValueError, IndexError):
        return invalid("invalid_active_corridor_input")
    if (
        not all(
            np.all(np.isfinite(value))
            for value in (reference_point, reference_tangent, reference_normal)
        )
        or not all(
            math.isfinite(value)
            for value in (source_lane_width, target_lane_width, lane_width_tolerance)
        )
        or source_lane_width <= 0.0
        or target_lane_width <= 0.0
    ):
        return invalid("non_finite_active_corridor_input")

    tangent_norm = float(np.linalg.norm(reference_tangent))
    normal_norm = float(np.linalg.norm(reference_normal))
    if tangent_norm <= 1e-12 or normal_norm <= 1e-12:
        return invalid("invalid_active_corridor_reference_frame")
    reference_tangent /= tangent_norm
    reference_normal /= normal_norm

    lane_projections = []
    for shape in (source_lane_shape, target_lane_shape):
        lane_path = compile_lane_shapes([shape])
        if lane_path is None:
            return invalid("invalid_active_corridor_lane_shape")
        projection = project_position_to_compiled_route(lane_path, reference_point)
        if projection is None or projection.get("projection_endpoint_clamped"):
            return invalid("active_corridor_lane_projection_invalid")
        lane_tangent = np.asarray(
            [projection["tangent_x"], projection["tangent_y"]], dtype=np.float64
        )
        alignment = float(np.dot(reference_tangent, lane_tangent))
        if not math.isfinite(alignment) or alignment < math.cos(math.radians(45.0)):
            return invalid("active_corridor_lane_heading_mismatch")
        lane_center = np.asarray(
            [projection["projected_x"], projection["projected_y"]], dtype=np.float64
        )
        lane_projections.append(float(np.dot(lane_center - reference_point, reference_normal)))

    source_center_d, target_center_d = lane_projections
    center_separation = abs(target_center_d - source_center_d)
    if center_separation <= 1e-6:
        return invalid("active_corridor_lane_centers_coincident")
    # SUMO topology proves adjacency at the caller. This geometry guard only
    # rejects a clearly disconnected/corrupt pair.
    if center_separation > 2.0 * max(source_lane_width, target_lane_width):
        return invalid("active_corridor_lane_centers_disconnected")

    d_min = (
        min(
            source_center_d - 0.5 * source_lane_width,
            target_center_d - 0.5 * target_lane_width,
        )
        - lane_width_tolerance
    )
    d_max = (
        max(
            source_center_d + 0.5 * source_lane_width,
            target_center_d + 0.5 * target_lane_width,
        )
        + lane_width_tolerance
    )
    if not all(math.isfinite(value) for value in (d_min, d_max)) or d_min >= d_max:
        return invalid("invalid_active_corridor_bounds")
    return {
        "valid": True,
        "reason": "source_target_lane_envelope",
        "d_min": d_min,
        "d_max": d_max,
        "source_center_d": source_center_d,
        "target_center_d": target_center_d,
        "source_lane_width": source_lane_width,
        "target_lane_width": target_lane_width,
        "center_separation": center_separation,
    }


def sample_compiled_route_frame(compiled_path, route_s, z=0.0):
    """Sample point, continuous tangent and left normal at canonical progress."""
    if compiled_path is None:
        return None
    try:
        starts = np.asarray(compiled_path["starts"], dtype=np.float64)
        deltas = np.asarray(compiled_path["deltas"], dtype=np.float64)
        lengths = np.asarray(compiled_path["lengths"], dtype=np.float64)
        cumulative_starts = np.asarray(compiled_path["cumulative_starts"], dtype=np.float64)
        cumulative_ends = np.asarray(compiled_path["cumulative_ends"], dtype=np.float64)
        route_s = float(route_s)
        z = float(z)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        len(starts) == 0
        or not math.isfinite(route_s)
        or not math.isfinite(z)
        or not all(
            np.all(np.isfinite(values))
            for values in (starts, deltas, lengths, cumulative_starts, cumulative_ends)
        )
    ):
        return None
    clamped_s = min(float(cumulative_ends[-1]), max(0.0, route_s))
    index = int(np.searchsorted(cumulative_ends, clamped_s, side="left"))
    index = min(index, len(lengths) - 1)
    ratio = (clamped_s - cumulative_starts[index]) / lengths[index]
    ratio = min(1.0, max(0.0, float(ratio)))
    point = starts[index] + ratio * deltas[index]
    tangent = _compiled_route_tangent(compiled_path, index, ratio)
    if tangent is None:
        return None
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    return {
        "s": clamped_s,
        "x": float(point[0]),
        "y": float(point[1]),
        "z": z,
        "tangent_x": float(tangent[0]),
        "tangent_y": float(tangent[1]),
        "normal_x": float(normal[0]),
        "normal_y": float(normal[1]),
        "segment_index": index,
        "segment_ratio": ratio,
        "clamped": abs(clamped_s - route_s) > 1e-12,
    }


def sample_compiled_route_curvature(
    compiled_path,
    route_s,
    half_window=CANONICAL_CURVATURE_HALF_WINDOW,
):
    """Return signed curvature from continuous canonical-route tangents."""
    if compiled_path is None:
        return None
    try:
        route_s = float(route_s)
        half_window = float(half_window)
        total_length = float(compiled_path["total_length"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not all(math.isfinite(value) for value in (route_s, half_window, total_length))
        or half_window <= 0.0
        or total_length <= 0.0
        or route_s < 0.0
        or route_s > total_length
    ):
        return None

    before_s = max(0.0, route_s - half_window)
    after_s = min(total_length, route_s + half_window)
    span = after_s - before_s
    if span <= 1e-6:
        return None
    before = sample_compiled_route_frame(compiled_path, before_s)
    after = sample_compiled_route_frame(compiled_path, after_s)
    if before is None or after is None:
        return None
    before_tangent = (before["tangent_x"], before["tangent_y"])
    after_tangent = (after["tangent_x"], after["tangent_y"])
    cross = before_tangent[0] * after_tangent[1] - before_tangent[1] * after_tangent[0]
    dot = before_tangent[0] * after_tangent[0] + before_tangent[1] * after_tangent[1]
    heading_delta = math.atan2(cross, dot)
    curvature = heading_delta / span
    if not math.isfinite(curvature):
        return None
    return curvature


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
        curve_full_scale_radians = max(curve_start_radians + 1e-6, float(curve_full_scale_radians))
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
        within_horizon = (segment_indices >= nearest_indices[:, None]) & (
            segment_indices <= target_indices[:, None]
        )
        max_heading_changes = np.max(np.where(within_horizon, angle_differences, 0.0), axis=1)
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
                np.asarray([lookahead_distances[index] for index in indices], dtype=np.float64),
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
        target_ratios = (target_distances - cumulative_starts[target_indices]) / lengths[
            target_indices
        ]
        target_ratios = np.clip(target_ratios, 0.0, 1.0)
        target_points = starts[target_indices] + target_ratios[:, None] * deltas[target_indices]
        target_points[beyond_path] = compiled_path["last_end"]

        for row, result_index in enumerate(indices):
            results[result_index] = (
                float(target_points[row, 0]),
                float(target_points[row, 1]),
                float(z_array[row]),
            )

    return results


# Keep selection plus front/rear/tracker invariants in one atomic operation.
def apply_external_state_lateral_maneuver_target(  # noqa: C901
    action,
    *,
    maneuver_enabled=False,
    maneuver_target_enabled=False,
    maneuver_phase="lane_keep",
    armed_origin_phase="lane_keep",
    held_canonical_d=None,
    phase_a_rear_axle_position=None,
    lane_width=None,
    lane_width_tolerance=0.01,
    active_maneuver_corridor=None,
):
    """Apply a post-transition canonical maneuver target to a base action.

    Evidence and state transitions deliberately happen before this function.
    The selected target depends only on the confirmed maneuver phase and
    canonical Phase-B/held offsets; raw SUMO ``posLat`` and ``speedLat`` are
    never used here.
    """
    result = dict(action or {})
    pre_state_phase = str(result.get("maneuver_phase") or "lane_keep")
    maneuver_phase = str(maneuver_phase or "lane_keep")
    if maneuver_phase not in {"lane_keep", "armed", "active", "hold"}:
        maneuver_phase = "lane_keep"
    armed_origin_phase = "hold" if armed_origin_phase == "hold" else "lane_keep"
    maneuver_enabled = bool(maneuver_enabled)
    maneuver_target_enabled = bool(maneuver_target_enabled)
    result["maneuver_enabled"] = maneuver_enabled
    result["maneuver_target_enabled"] = maneuver_target_enabled
    result["maneuver_target_pre_state_phase"] = pre_state_phase
    result["maneuver_target_post_state_phase"] = maneuver_phase if maneuver_enabled else "lane_keep"
    result["maneuver_phase"] = maneuver_phase if maneuver_enabled else "lane_keep"
    result["maneuver_target_armed_origin_phase"] = armed_origin_phase
    result["maneuver_target_applied"] = False
    result["maneuver_target_source"] = "route_center"
    result["maneuver_target_selected_d"] = 0.0
    result["maneuver_target_world_anchor"] = None
    result["maneuver_target_d_outside_lane"] = False
    result["maneuver_target_d_outside_primary_lane"] = False
    result["maneuver_target_safety_region"] = "primary_lane"
    result["active_maneuver_corridor_valid"] = None
    result["active_maneuver_corridor_reason"] = "not_active"
    result["active_maneuver_corridor_d_min"] = None
    result["active_maneuver_corridor_d_max"] = None
    result["active_maneuver_corridor_source_center_d"] = None
    result["active_maneuver_corridor_target_center_d"] = None
    result["active_maneuver_corridor_center_separation"] = None
    result["lateral_displacement"] = 0.0
    result["world_lateral_speed"] = 0.0
    result["warning"] = ""

    if not result.get("valid", False):
        return result
    if maneuver_enabled and maneuver_target_enabled and result.get("canonical_phase_a") is None:
        result["valid"] = False
        result["error"] = "missing_phase_a_world_position"
        return result

    selected_d = 0.0
    mode = "route"
    source = "route_center"
    target_applied = False
    if maneuver_enabled and maneuver_target_enabled:
        if maneuver_phase == "active":
            phase_b_frame = result.get("canonical_phase_b") or {}
            selected_d = phase_b_frame.get("d")
            mode = "active_current"
            source = "phase_b_canonical_d"
            target_applied = True
        elif maneuver_phase == "hold" or (
            maneuver_phase == "armed" and armed_origin_phase == "hold"
        ):
            selected_d = (
                result.get("held_canonical_d") if held_canonical_d is None else held_canonical_d
            )
            mode = "hold" if maneuver_phase == "hold" else "armed_hold"
            source = "held_canonical_d"
            target_applied = True

    try:
        selected_d = float(selected_d)
        lane_width = None if lane_width is None else float(lane_width)
        lane_width_tolerance = max(0.0, float(lane_width_tolerance))
    except (TypeError, ValueError):
        result["valid"] = False
        result["error"] = (
            "missing_active_canonical_d"
            if maneuver_phase == "active"
            else "missing_held_canonical_d"
        )
        return result
    if not math.isfinite(selected_d) or (
        lane_width is not None and (not math.isfinite(lane_width) or lane_width <= 0.0)
    ):
        result["valid"] = False
        result["error"] = "non_finite_maneuver_target"
        return result

    target_outside_primary_lane = bool(
        lane_width is not None and abs(selected_d) > 0.5 * lane_width + lane_width_tolerance
    )
    target_outside_lane = target_outside_primary_lane
    if target_applied and maneuver_phase == "active" and active_maneuver_corridor is not None:
        result["active_maneuver_corridor_valid"] = bool(
            active_maneuver_corridor.get("valid", False)
        )
        result["active_maneuver_corridor_reason"] = str(
            active_maneuver_corridor.get("reason") or "active_corridor_invalid"
        )
        for field in (
            "d_min",
            "d_max",
            "source_center_d",
            "target_center_d",
            "center_separation",
        ):
            result[f"active_maneuver_corridor_{field}"] = active_maneuver_corridor.get(field)
        if not result["active_maneuver_corridor_valid"]:
            result["valid"] = False
            result["error"] = "active_maneuver_corridor_invalid"
            return result
        try:
            corridor_d_min = float(active_maneuver_corridor["d_min"])
            corridor_d_max = float(active_maneuver_corridor["d_max"])
        except (KeyError, TypeError, ValueError):
            result["valid"] = False
            result["error"] = "active_maneuver_corridor_invalid"
            return result
        if (
            not math.isfinite(corridor_d_min)
            or not math.isfinite(corridor_d_max)
            or corridor_d_min >= corridor_d_max
        ):
            result["valid"] = False
            result["error"] = "active_maneuver_corridor_invalid"
            return result
        target_outside_lane = not corridor_d_min <= selected_d <= corridor_d_max
        result["maneuver_target_safety_region"] = "source_target_lane_corridor"
        phase_b_d = (result.get("canonical_phase_b") or {}).get("d")
        if phase_b_d is not None:
            result["current_d_outside_lane"] = not (
                corridor_d_min <= float(phase_b_d) <= corridor_d_max
            )
    result["maneuver_target_d_outside_lane"] = target_outside_lane
    result["maneuver_target_d_outside_primary_lane"] = target_outside_primary_lane
    result["maneuver_target_selected_d"] = selected_d
    result["maneuver_target_source"] = source
    result["maneuver_target_applied"] = target_applied
    if target_applied and target_outside_lane:
        result["valid"] = False
        result["error"] = "maneuver_target_d_outside_lane"
        return result

    try:
        target_center = np.asarray(result["route_lookahead"][:2], dtype=np.float64)
        target_tangent = np.asarray(
            [result["target_route_tangent_x"], result["target_route_tangent_y"]],
            dtype=np.float64,
        )
        target_normal = np.asarray(
            [result["target_world_left_normal_x"], result["target_world_left_normal_y"]],
            dtype=np.float64,
        )
        preview_center = np.asarray(
            [
                result["steering_preview_front_center_x"],
                result["steering_preview_front_center_y"],
            ],
            dtype=np.float64,
        )
        preview_tangent = np.asarray(
            [result["steering_preview_tangent_x"], result["steering_preview_tangent_y"]],
            dtype=np.float64,
        )
        preview_normal = np.asarray(
            [result["steering_preview_normal_x"], result["steering_preview_normal_y"]],
            dtype=np.float64,
        )
        phase_b_frame = result["canonical_phase_b"]
        route_tangent = np.asarray(
            [result["route_tangent_x"], result["route_tangent_y"]], dtype=np.float64
        )
        phase_a_frame = result.get("canonical_phase_a")
        center_curvature = float(result["path_tracker_center_curvature"])
        control_offset = float(result["front_to_control_offset"])
        z = float(result["route_lookahead"][2])
        phase_a_rear = (
            None
            if phase_a_rear_axle_position is None
            else np.asarray(phase_a_rear_axle_position[:2], dtype=np.float64)
        )
    except (KeyError, TypeError, ValueError, IndexError):
        result["valid"] = False
        result["error"] = "invalid_maneuver_target_geometry"
        return result
    arrays = (
        target_center,
        target_tangent,
        target_normal,
        preview_center,
        preview_tangent,
        preview_normal,
        route_tangent,
    )
    if (
        not all(np.all(np.isfinite(value)) for value in arrays)
        or (phase_a_rear is not None and not np.all(np.isfinite(phase_a_rear)))
        or not all(math.isfinite(value) for value in (center_curvature, control_offset, z))
    ):
        result["valid"] = False
        result["error"] = "non_finite_maneuver_target_geometry"
        return result

    offset_curvature_denominator = 1.0 - selected_d * center_curvature
    if offset_curvature_denominator <= 1e-6:
        result["valid"] = False
        result["error"] = "invalid_offset_path_curvature"
        return result
    front_curvature = center_curvature / offset_curvature_denominator
    curvature_ratio = control_offset * front_curvature
    if abs(curvature_ratio) >= 1.0 - 1e-9:
        result["valid"] = False
        result["error"] = "unreachable_front_path_curvature"
        return result
    rear_curvature = front_curvature / math.sqrt(
        max(1e-12, 1.0 - curvature_ratio * curvature_ratio)
    )
    route_heading = math.atan2(route_tangent[1], route_tangent[0])
    front_reference_heading = route_heading
    motion_heading_applied = False
    reference_heading_source = "route_tangent"
    motion_heading_reason = "maneuver_not_active"
    if maneuver_enabled and maneuver_target_enabled and maneuver_phase == "active":
        requested_longitudinal_speed = result.get("phase_a_to_b_requested_longitudinal_speed")
        motion_heading = result.get("phase_a_to_b_motion_heading_radians")
        lateral_motion_requested = bool(result.get("phase_a_to_b_lateral_motion_requested", False))
        if requested_longitudinal_speed is not None and requested_longitudinal_speed < -1e-9:
            result["valid"] = False
            result["error"] = "backward_phase_a_to_b_motion"
            return result
        if lateral_motion_requested:
            if result.get("phase_a_to_b_motion_heading_valid", False):
                try:
                    front_reference_heading = float(motion_heading)
                except (TypeError, ValueError):
                    result["valid"] = False
                    result["error"] = "non_finite_phase_a_to_b_motion"
                    return result
                if not math.isfinite(front_reference_heading):
                    result["valid"] = False
                    result["error"] = "non_finite_phase_a_to_b_motion"
                    return result
                motion_heading_applied = True
                reference_heading_source = "phase_a_to_b_motion"
                motion_heading_reason = "active_lateral_motion"
            else:
                motion_heading_reason = result.get(
                    "phase_a_to_b_motion_heading_invalid_reason", "invalid_motion_heading"
                )
                if motion_heading_reason != "low_forward_speed_route_tangent":
                    result["valid"] = False
                    result["error"] = motion_heading_reason
                    return result
        else:
            motion_heading_reason = "lateral_request_stopped_route_tangent"
    elif maneuver_phase == "armed":
        motion_heading_reason = "armed_route_tangent"
    elif maneuver_phase == "hold":
        motion_heading_reason = "hold_route_tangent"
    elif not (maneuver_enabled and maneuver_target_enabled):
        motion_heading_reason = "maneuver_target_disabled"

    reference_heading = front_reference_heading - math.atan(control_offset * rear_curvature)
    result["path_tracker_center_curvature"] = center_curvature
    result["path_tracker_front_curvature"] = float(front_curvature)
    result["path_tracker_rear_curvature"] = float(rear_curvature)
    result["path_tracker_reference_tangent_x"] = math.cos(reference_heading)
    result["path_tracker_reference_tangent_y"] = math.sin(reference_heading)
    result["path_tracker_motion_heading_applied"] = motion_heading_applied
    result["path_tracker_reference_heading_source"] = reference_heading_source
    result["path_tracker_motion_heading_reason"] = motion_heading_reason
    result["path_tracker_front_cross_track_error"] = (
        None if phase_a_frame is None else float(phase_a_frame["d"] - selected_d)
    )
    result["path_tracker_valid"] = bool(
        phase_a_frame is not None
        and result.get("control_reference_valid", False)
        and control_offset > 0.0
    )

    front_xy = target_center + target_normal * selected_d
    control_xy = front_xy - target_tangent * control_offset
    preview_front_xy = preview_center + preview_normal * selected_d
    steering_preview_xy = preview_front_xy - preview_tangent * control_offset
    if not all(
        math.isfinite(float(value)) for value in (*front_xy, *control_xy, *steering_preview_xy)
    ):
        result["valid"] = False
        result["error"] = "non_finite_control_target"
        return result
    distance_error = abs(float(np.linalg.norm(front_xy - control_xy)) - control_offset)
    if not math.isfinite(distance_error) or distance_error > 1e-6:
        result["valid"] = False
        result["error"] = "front_rear_distance_mismatch"
        return result

    result["front_rear_distance_error"] = distance_error
    result["front_rear_distance_consistent"] = True
    result["mode"] = mode
    result["target_lateral_distance"] = selected_d
    result["preview_lateral_distance"] = selected_d
    result["lookahead"] = (float(front_xy[0]), float(front_xy[1]), z)
    result["control_lookahead"] = (float(control_xy[0]), float(control_xy[1]), z)
    result["steering_preview"] = (
        float(steering_preview_xy[0]),
        float(steering_preview_xy[1]),
        z,
    )
    result["steering_preview_front_d"] = selected_d
    result["lateral_direction_source"] = source
    result["maneuver_target_world_anchor"] = (
        float(phase_b_frame["projected_x"] + phase_b_frame["normal_x"] * selected_d),
        float(phase_b_frame["projected_y"] + phase_b_frame["normal_y"] * selected_d),
    )
    if phase_a_rear is not None:
        result["control_target_ahead"] = bool(
            float(np.dot(control_xy - phase_a_rear, route_tangent)) > 1e-9
        )
        result["steering_preview_target_ahead"] = bool(
            float(np.dot(steering_preview_xy - phase_a_rear, route_tangent)) > 1e-9
        )
        if not result["control_target_ahead"]:
            result["valid"] = False
            result["error"] = "control_target_behind"
        elif not result["steering_preview_target_ahead"]:
            result["valid"] = False
            result["error"] = "steering_preview_behind"
    return result


def build_external_state_lateral_action_lookahead(
    compiled_path,
    current_position,
    lookahead_distance,
    *,
    maneuver_enabled=False,
    maneuver_target_enabled=False,
    lateral_speed=0.0,
    desired_speed=None,
    min_forward_speed=0.2,
    lateral_speed_epsilon=0.001,
    lateral_motion_epsilon=0.0001,
    phase_a_position=None,
    phase_a_evidence_position=None,
    phase_a_rear_axle_position=None,
    phase_a_speed=0.0,
    front_to_control_offset=None,
    steering_preview_min_distance=2.8,
    steering_preview_time_headway=0.5,
    steering_preview_max_distance=5.0,
    phase_step_length=0.05,
    previous_world_lateral_direction=None,
    previous_direction_min_alignment=0.5,
    maneuver_phase="lane_keep",
    armed_origin_phase="lane_keep",
    held_lateral_offset=None,
    held_canonical_d=None,
    held_world_anchor=None,
    canonical_path_key=None,
    canonical_path_switched=False,
    lane_width=None,
    lane_width_tolerance=0.01,
    active_maneuver_corridor=None,
    curvature_half_window=None,
    z=0.0,
):
    """Build canonical front and rear-axle control targets from Phase-B SUMO state.

    The existing ``lookahead`` value remains the desired SUMO front-centre
    target. ``control_lookahead`` is its exact rear-axle reference, while
    ``steering_preview`` is retained for continuity and ahead diagnostics.
    """

    def invalid(reason):
        return {
            "valid": False,
            "error": reason,
            "mode": "route",
            "lookahead": None,
            "control_lookahead": None,
            "steering_preview": None,
            "route_lookahead": None,
            "lateral_displacement": 0.0,
            "target_lateral_distance": None,
            "route_tangent_x": None,
            "route_tangent_y": None,
            "world_left_normal_x": None,
            "world_left_normal_y": None,
            "target_route_tangent_x": None,
            "target_route_tangent_y": None,
            "target_world_left_normal_x": None,
            "target_world_left_normal_y": None,
            "phase_b_lateral_delta": None,
            "canonical_delta_d": None,
            "phase_a_to_b_delta_x": None,
            "phase_a_to_b_delta_y": None,
            "phase_a_to_b_velocity_x": None,
            "phase_a_to_b_velocity_y": None,
            "phase_a_to_b_requested_longitudinal_speed": None,
            "phase_a_to_b_requested_lateral_speed": None,
            "phase_a_to_b_motion_heading_radians": None,
            "phase_a_to_b_motion_heading_degrees": None,
            "phase_a_to_b_motion_heading_valid": False,
            "phase_a_to_b_motion_heading_invalid_reason": reason,
            "phase_a_to_b_lateral_motion_requested": False,
            "expected_phase_b_lateral_distance": None,
            "world_lateral_speed": 0.0,
            "current_lateral_distance": None,
            "preview_lateral_distance": None,
            "held_lateral_offset": None,
            "held_canonical_d": None,
            "held_world_anchor": None,
            "held_reprojected": False,
            "maneuver_enabled": bool(maneuver_enabled),
            "maneuver_target_enabled": bool(maneuver_target_enabled),
            "maneuver_phase": maneuver_phase if bool(maneuver_enabled) else "lane_keep",
            "maneuver_target_pre_state_phase": maneuver_phase,
            "maneuver_target_post_state_phase": maneuver_phase,
            "maneuver_target_armed_origin_phase": armed_origin_phase,
            "maneuver_target_applied": False,
            "maneuver_target_source": "route_center",
            "maneuver_target_selected_d": None,
            "maneuver_target_world_anchor": None,
            "maneuver_target_d_outside_lane": None,
            "maneuver_target_d_outside_primary_lane": None,
            "maneuver_target_safety_region": "primary_lane",
            "active_maneuver_corridor_valid": None,
            "active_maneuver_corridor_reason": reason,
            "active_maneuver_corridor_d_min": None,
            "active_maneuver_corridor_d_max": None,
            "active_maneuver_corridor_source_center_d": None,
            "active_maneuver_corridor_target_center_d": None,
            "active_maneuver_corridor_center_separation": None,
            "lateral_motion_detected": False,
            "warning": "",
            "lateral_direction_source": "inactive",
            "canonical_path_key": tuple(canonical_path_key or ()),
            "canonical_path_switched": bool(canonical_path_switched),
            "canonical_phase_a": None,
            "canonical_evidence_phase_a": None,
            "canonical_phase_b": None,
            "canonical_target_s": None,
            "steering_preview_requested_distance": None,
            "steering_preview_effective_distance": None,
            "steering_preview_front_s": None,
            "steering_preview_front_d": None,
            "steering_preview_front_center_x": None,
            "steering_preview_front_center_y": None,
            "steering_preview_tangent_x": None,
            "steering_preview_tangent_y": None,
            "steering_preview_normal_x": None,
            "steering_preview_normal_y": None,
            "front_to_control_offset": None,
            "front_rear_distance_error": None,
            "front_rear_distance_consistent": False,
            "control_reference_valid": False,
            "control_target_ahead": None,
            "steering_preview_target_ahead": None,
            "current_d_outside_lane": None,
            "held_d_outside_lane": None,
            "path_tracker_valid": False,
            "path_tracker_center_curvature": None,
            "path_tracker_front_curvature": None,
            "path_tracker_rear_curvature": None,
            "path_tracker_reference_tangent_x": None,
            "path_tracker_reference_tangent_y": None,
            "path_tracker_front_cross_track_error": None,
            "path_tracker_motion_heading_applied": False,
            "path_tracker_reference_heading_source": "route_tangent",
            "path_tracker_motion_heading_reason": reason,
            "canonical_curvature_half_window": None,
        }

    try:
        origin = np.asarray(
            [float(current_position[0]), float(current_position[1])], dtype=np.float64
        )
        distance = max(0.0, float(lookahead_distance))
        phase_step_length = float(phase_step_length)
        min_forward_speed = max(0.0, float(min_forward_speed))
        phase_a_speed = float(phase_a_speed)
        steering_preview_min_distance = float(steering_preview_min_distance)
        steering_preview_time_headway = float(steering_preview_time_headway)
        steering_preview_max_distance = float(steering_preview_max_distance)
        maneuver_enabled = bool(maneuver_enabled)
        maneuver_phase = str(maneuver_phase or "lane_keep")
        if maneuver_phase not in {"lane_keep", "armed", "active", "hold"}:
            maneuver_phase = "lane_keep"
        legacy_held = None if held_lateral_offset is None else float(held_lateral_offset)
        held_canonical_d = legacy_held if held_canonical_d is None else float(held_canonical_d)
        z = float(z)
        phase_a = (
            None
            if phase_a_position is None
            else np.asarray(
                [float(phase_a_position[0]), float(phase_a_position[1])],
                dtype=np.float64,
            )
        )
        phase_a_evidence = (
            phase_a
            if phase_a_evidence_position is None
            else np.asarray(
                [
                    float(phase_a_evidence_position[0]),
                    float(phase_a_evidence_position[1]),
                ],
                dtype=np.float64,
            )
        )
        phase_a_rear = (
            None
            if phase_a_rear_axle_position is None
            else np.asarray(
                [
                    float(phase_a_rear_axle_position[0]),
                    float(phase_a_rear_axle_position[1]),
                ],
                dtype=np.float64,
            )
        )
        anchor = (
            None
            if held_world_anchor is None
            else np.asarray(
                [float(held_world_anchor[0]), float(held_world_anchor[1])],
                dtype=np.float64,
            )
        )
        control_offset_supplied = front_to_control_offset is not None
        control_offset = 0.0 if front_to_control_offset is None else float(front_to_control_offset)
        lane_width = None if lane_width is None else float(lane_width)
        lane_width_tolerance = max(0.0, float(lane_width_tolerance))
        curvature_half_window = (
            max(
                CANONICAL_MIN_BODY_CURVATURE_HALF_WINDOW,
                0.5 * control_offset,
            )
            if curvature_half_window is None
            else float(curvature_half_window)
        )
    except (TypeError, ValueError, IndexError):
        return invalid("invalid_route_input")
    finite_arrays = [origin]
    finite_arrays.extend(
        value for value in (phase_a, phase_a_evidence, phase_a_rear, anchor) if value is not None
    )
    finite_scalars = [
        distance,
        phase_step_length,
        min_forward_speed,
        phase_a_speed,
        control_offset,
        lane_width_tolerance,
        steering_preview_min_distance,
        steering_preview_time_headway,
        steering_preview_max_distance,
        curvature_half_window,
        z,
    ]
    finite_scalars.extend(value for value in (held_canonical_d, lane_width) if value is not None)
    if (
        not all(np.all(np.isfinite(value)) for value in finite_arrays)
        or not all(math.isfinite(value) for value in finite_scalars)
        or phase_step_length <= 0.0
        or control_offset < 0.0
        or curvature_half_window <= 0.0
        or (lane_width is not None and lane_width <= 0.0)
    ):
        return invalid("non_finite_route_input")
    if (
        steering_preview_min_distance <= 0.0
        or steering_preview_time_headway < 0.0
        or steering_preview_max_distance < steering_preview_min_distance
    ):
        return invalid("invalid_steering_preview_configuration")
    if compiled_path is None:
        return invalid("missing_route_geometry")
    if compiled_path.get("shape_join_truncated_index") is not None:
        return invalid("disconnected_route_geometry")

    phase_b_frame = project_position_to_compiled_route(compiled_path, origin)
    if phase_b_frame is None:
        return invalid("invalid_phase_b_projection")
    center_curvature = sample_compiled_route_curvature(
        compiled_path,
        phase_b_frame["s"],
        half_window=curvature_half_window,
    )
    if center_curvature is None:
        return invalid("invalid_canonical_curvature")

    phase_a_frame = (
        None
        if phase_a is None
        else project_position_to_compiled_route(
            compiled_path, phase_a, progress_hint=phase_b_frame["s"]
        )
    )
    if phase_a is not None and phase_a_frame is None:
        return invalid("invalid_phase_a_projection")
    phase_a_evidence_frame = (
        None
        if phase_a_evidence is None
        else project_position_to_compiled_route(
            compiled_path, phase_a_evidence, progress_hint=phase_b_frame["s"]
        )
    )
    if phase_a_evidence is not None and phase_a_evidence_frame is None:
        return invalid("invalid_phase_a_evidence_projection")
    target_frame = sample_compiled_route_frame(compiled_path, phase_b_frame["s"] + distance, z=z)
    if target_frame is None or target_frame.get("clamped"):
        return invalid("invalid_target_projection")
    preview_requested_distance = steering_preview_time_headway * max(0.0, phase_a_speed)
    preview_effective_distance = min(
        steering_preview_max_distance,
        max(steering_preview_min_distance, preview_requested_distance),
    )
    preview_front_frame = sample_compiled_route_frame(
        compiled_path, phase_b_frame["s"] + preview_effective_distance, z=z
    )
    if preview_front_frame is None or preview_front_frame.get("clamped"):
        return invalid("steering_preview_path_end")

    tangent = np.asarray([phase_b_frame["tangent_x"], phase_b_frame["tangent_y"]], dtype=np.float64)
    left_normal = np.asarray(
        [phase_b_frame["normal_x"], phase_b_frame["normal_y"]], dtype=np.float64
    )
    target_point = np.asarray([target_frame["x"], target_frame["y"]], dtype=np.float64)
    target_tangent = np.asarray(
        [target_frame["tangent_x"], target_frame["tangent_y"]], dtype=np.float64
    )
    target_left_normal = np.asarray(
        [target_frame["normal_x"], target_frame["normal_y"]], dtype=np.float64
    )
    preview_front_point = np.asarray(
        [preview_front_frame["x"], preview_front_frame["y"]], dtype=np.float64
    )
    preview_tangent = np.asarray(
        [preview_front_frame["tangent_x"], preview_front_frame["tangent_y"]],
        dtype=np.float64,
    )
    preview_left_normal = np.asarray(
        [preview_front_frame["normal_x"], preview_front_frame["normal_y"]],
        dtype=np.float64,
    )
    phase_b_lateral_delta = (
        None if phase_a is None else float(np.dot(origin - phase_a, left_normal))
    )
    canonical_delta_d = (
        None
        if phase_a_evidence_frame is None
        else float(phase_b_frame["d"] - phase_a_evidence_frame["d"])
    )
    phase_a_to_b_delta = None if phase_a is None else origin - phase_a
    phase_a_to_b_velocity = (
        None if phase_a_to_b_delta is None else phase_a_to_b_delta / phase_step_length
    )
    requested_longitudinal_speed = (
        None if phase_a_to_b_velocity is None else float(np.dot(phase_a_to_b_velocity, tangent))
    )
    requested_lateral_speed = (
        None if phase_a_to_b_velocity is None else float(np.dot(phase_a_to_b_velocity, left_normal))
    )
    motion_heading = (
        None
        if phase_a_to_b_velocity is None
        else math.atan2(phase_a_to_b_velocity[1], phase_a_to_b_velocity[0])
    )
    motion_heading_valid = bool(
        requested_longitudinal_speed is not None
        and requested_longitudinal_speed > min_forward_speed
        and motion_heading is not None
        and math.isfinite(motion_heading)
    )
    if requested_longitudinal_speed is None:
        motion_heading_reason = "phase_a_position_unavailable"
    elif requested_longitudinal_speed < -1e-9:
        motion_heading_reason = "backward_phase_a_to_b_motion"
    elif requested_longitudinal_speed <= min_forward_speed:
        motion_heading_reason = "low_forward_speed_route_tangent"
    elif not motion_heading_valid:
        motion_heading_reason = "non_finite_phase_a_to_b_motion"
    else:
        motion_heading_reason = "valid"
    current_lateral_distance = float(phase_b_frame["d"])
    held_reprojected = False
    if bool(canonical_path_switched) and anchor is not None and held_canonical_d is not None:
        held_projection = project_position_to_compiled_route(
            compiled_path, anchor, progress_hint=phase_b_frame["s"]
        )
        if held_projection is None:
            return invalid("held_anchor_reprojection_failed")
        held_canonical_d = float(held_projection["d"])
        held_reprojected = True

    current_d_outside_lane = None
    held_d_outside_lane = None
    if lane_width is not None:
        maximum_d = 0.5 * lane_width + lane_width_tolerance
        current_d_outside_lane = abs(current_lateral_distance) > maximum_d
        held_d_outside_lane = bool(
            held_canonical_d is not None and abs(held_canonical_d) > maximum_d
        )

    route_lookahead = (float(target_point[0]), float(target_point[1]), z)
    result = {
        "valid": True,
        "error": "",
        "warning": "",
        "mode": "route",
        "lookahead": route_lookahead,
        "control_lookahead": None,
        "steering_preview": None,
        "route_lookahead": route_lookahead,
        "lateral_displacement": 0.0,
        "target_lateral_distance": 0.0,
        "route_tangent_x": float(tangent[0]),
        "route_tangent_y": float(tangent[1]),
        "world_left_normal_x": float(left_normal[0]),
        "world_left_normal_y": float(left_normal[1]),
        "target_route_tangent_x": float(target_tangent[0]),
        "target_route_tangent_y": float(target_tangent[1]),
        "target_world_left_normal_x": float(target_left_normal[0]),
        "target_world_left_normal_y": float(target_left_normal[1]),
        "phase_b_lateral_delta": phase_b_lateral_delta,
        "canonical_delta_d": canonical_delta_d,
        "phase_a_to_b_delta_x": (
            None if phase_a_to_b_delta is None else float(phase_a_to_b_delta[0])
        ),
        "phase_a_to_b_delta_y": (
            None if phase_a_to_b_delta is None else float(phase_a_to_b_delta[1])
        ),
        "phase_a_to_b_velocity_x": (
            None if phase_a_to_b_velocity is None else float(phase_a_to_b_velocity[0])
        ),
        "phase_a_to_b_velocity_y": (
            None if phase_a_to_b_velocity is None else float(phase_a_to_b_velocity[1])
        ),
        "phase_a_to_b_requested_longitudinal_speed": requested_longitudinal_speed,
        "phase_a_to_b_requested_lateral_speed": requested_lateral_speed,
        "phase_a_to_b_motion_heading_radians": motion_heading,
        "phase_a_to_b_motion_heading_degrees": (
            None if motion_heading is None else math.degrees(motion_heading)
        ),
        "phase_a_to_b_motion_heading_valid": motion_heading_valid,
        "phase_a_to_b_motion_heading_invalid_reason": motion_heading_reason,
        "phase_a_to_b_lateral_motion_requested": False,
        "expected_phase_b_lateral_distance": None,
        "world_lateral_speed": 0.0,
        "current_lateral_distance": current_lateral_distance,
        "preview_lateral_distance": current_lateral_distance,
        "held_lateral_offset": held_canonical_d,
        "held_canonical_d": held_canonical_d,
        "held_world_anchor": None if anchor is None else (float(anchor[0]), float(anchor[1])),
        "held_reprojected": held_reprojected,
        "maneuver_enabled": maneuver_enabled,
        "maneuver_target_enabled": maneuver_target_enabled,
        "maneuver_phase": maneuver_phase if maneuver_enabled else "lane_keep",
        "maneuver_target_pre_state_phase": maneuver_phase,
        "maneuver_target_post_state_phase": maneuver_phase,
        "maneuver_target_armed_origin_phase": armed_origin_phase,
        "maneuver_target_applied": False,
        "maneuver_target_source": "route_center",
        "maneuver_target_selected_d": 0.0,
        "maneuver_target_world_anchor": None,
        "maneuver_target_d_outside_lane": False,
        "lateral_motion_detected": False,
        "lateral_direction_source": "inactive",
        "canonical_path_key": tuple(canonical_path_key or compiled_path.get("lane_ids", ())),
        "canonical_path_switched": bool(canonical_path_switched),
        "canonical_phase_a": phase_a_frame,
        "canonical_evidence_phase_a": phase_a_evidence_frame,
        "canonical_phase_b": phase_b_frame,
        "canonical_target_s": float(target_frame["s"]),
        "steering_preview_requested_distance": float(preview_requested_distance),
        "steering_preview_effective_distance": float(preview_effective_distance),
        "steering_preview_front_s": float(preview_front_frame["s"]),
        "steering_preview_front_d": 0.0,
        "steering_preview_front_center_x": float(preview_front_point[0]),
        "steering_preview_front_center_y": float(preview_front_point[1]),
        "steering_preview_tangent_x": float(preview_tangent[0]),
        "steering_preview_tangent_y": float(preview_tangent[1]),
        "steering_preview_normal_x": float(preview_left_normal[0]),
        "steering_preview_normal_y": float(preview_left_normal[1]),
        "front_to_control_offset": float(control_offset),
        "front_rear_distance_error": None,
        "front_rear_distance_consistent": False,
        "control_reference_valid": bool(control_offset_supplied and phase_a_rear is not None),
        "control_target_ahead": None,
        "steering_preview_target_ahead": None,
        "current_d_outside_lane": current_d_outside_lane,
        "held_d_outside_lane": held_d_outside_lane,
        "path_tracker_valid": False,
        "path_tracker_center_curvature": float(center_curvature),
        "path_tracker_front_curvature": None,
        "path_tracker_rear_curvature": None,
        "path_tracker_reference_tangent_x": None,
        "path_tracker_reference_tangent_y": None,
        "path_tracker_front_cross_track_error": None,
        "path_tracker_motion_heading_applied": False,
        "path_tracker_reference_heading_source": "route_tangent",
        "path_tracker_motion_heading_reason": "maneuver_not_active",
        "canonical_curvature_half_window": float(curvature_half_window),
    }

    result = apply_external_state_lateral_maneuver_target(
        result,
        maneuver_enabled=False,
        maneuver_target_enabled=False,
        maneuver_phase="lane_keep",
        armed_origin_phase="lane_keep",
        held_canonical_d=None,
        phase_a_rear_axle_position=phase_a_rear,
        lane_width=lane_width,
        lane_width_tolerance=lane_width_tolerance,
        active_maneuver_corridor=active_maneuver_corridor,
    )
    if not result.get("valid", False):
        return result

    try:
        lateral_speed = float(lateral_speed)
        lateral_speed_epsilon = max(0.0, float(lateral_speed_epsilon))
        lateral_motion_epsilon = max(0.0, float(lateral_motion_epsilon))
    except (TypeError, ValueError):
        return invalid("invalid_lateral_action")
    if not all(
        math.isfinite(value)
        for value in (lateral_speed, lateral_speed_epsilon, lateral_motion_epsilon)
    ):
        return invalid("non_finite_lateral_action")
    result["expected_phase_b_lateral_distance"] = abs(lateral_speed) * phase_step_length
    speed_motion = abs(lateral_speed) > lateral_speed_epsilon
    displacement_motion = bool(
        canonical_delta_d is not None and abs(canonical_delta_d) > lateral_motion_epsilon
    )
    lateral_motion_detected = speed_motion or displacement_motion
    result["lateral_motion_detected"] = lateral_motion_detected
    result["phase_a_to_b_lateral_motion_requested"] = bool(
        requested_lateral_speed is not None and abs(requested_lateral_speed) > lateral_speed_epsilon
    )

    return apply_external_state_lateral_maneuver_target(
        result,
        maneuver_enabled=maneuver_enabled,
        maneuver_target_enabled=maneuver_target_enabled,
        maneuver_phase=maneuver_phase,
        armed_origin_phase=armed_origin_phase,
        held_canonical_d=held_canonical_d,
        phase_a_rear_axle_position=phase_a_rear,
        lane_width=lane_width,
        lane_width_tolerance=lane_width_tolerance,
        active_maneuver_corridor=active_maneuver_corridor,
    )


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
    heading_lookahead = project_position_by_sumo_angle(current_position, sumo_angle, distance, z)
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
        primary_error = math.hypot(primary[0] - reference_xy[0], primary[1] - reference_xy[1])
        mirrored_error = math.hypot(mirrored[0] - reference_xy[0], mirrored[1] - reference_xy[1])
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
