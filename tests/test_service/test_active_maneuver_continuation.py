"""A longitudinal connection must preserve the active lateral lane envelope."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from terasim_service.plugins import cosim_inprocess
from terasim_service.plugins.cosim_inprocess import TeraSimCoSimInProcessPlugin
from terasim_service.utils.sumo_lane_geometry import (
    compile_lane_shapes,
    project_position_to_compiled_route,
)


def corridor_fixture():
    shapes = {}
    links = {}
    widths = {}
    for index in range(2):
        y = 3.2 * index
        shapes[f"in_{index}"] = ((0.0, y), (10.0, y))
        shapes[f":join_0_{index}"] = ((10.0, y), (10.25, y))
        shapes[f"out_{index}"] = ((10.25, y), (30.0, y))
        links[f"in_{index}"] = ((f"out_{index}", f":join_0_{index}"),)
        links[f":join_0_{index}"] = ((f"out_{index}", ""),)
    widths.update({lane: 3.2 for lane in shapes})
    lane = SimpleNamespace(
        getLinks=lambda lane_id: links.get(lane_id, ()),
        getEdgeID=lambda lane_id: lane_id.rsplit("_", 1)[0],
        getShape=shapes.__getitem__,
        getWidth=widths.__getitem__,
    )
    return lane, shapes, links, widths


def evaluate(lane, current, position=(10.1, 0.6), reverse=False):
    plugin = object.__new__(TeraSimCoSimInProcessPlugin)
    state = {
        "phase": "active",
        "source_lane_id": "in_1" if reverse else "in_0",
        "target_lane_id": "in_0" if reverse else "in_1",
        "primary_lane_switched": False,
    }
    before = deepcopy(state)
    path = compile_lane_shapes([((0.0, 0.0), (30.0, 0.0))])
    action = {"canonical_phase_b": project_position_to_compiled_route(path, position)}
    with patch.object(cosim_inprocess.traci, "lane", lane):
        result = plugin._physics_active_maneuver_lane_corridor(
            state, action, current_lane_id=current
        )
    assert state == before  # Connection traversal is not lane-change completion.
    return result


@pytest.mark.parametrize("current", [":join_0_0", ":join_0_1", "out_0", "out_1"])
@pytest.mark.parametrize("reverse", [False, True])
def test_verified_connection_preserves_both_lane_ribbons(current, reverse):
    lane, _, _, _ = corridor_fixture()
    result = evaluate(lane, current, reverse=reverse)
    assert result["valid"]
    assert result["reason"] == "source_target_lane_continuation_envelope"
    assert result["d_min"] == pytest.approx(-1.61)
    assert result["d_max"] == pytest.approx(4.81)


def test_short_internal_lane_uses_verified_predecessor_geometry():
    lane, _, _, _ = corridor_fixture()
    # The canonical front projection may still be just before the short via lane.
    assert evaluate(lane, ":join_0_0", (9.98, 0.6))["valid"]


@pytest.mark.parametrize("current", ["unrelated_0", "later_0"])
def test_unverified_current_lane_is_rejected(current):
    lane, _, _, _ = corridor_fixture()
    result = evaluate(lane, current)
    assert not result["valid"]
    assert result["reason"] == "active_corridor_current_lane_mismatch"


def test_missing_counterpart_connection_is_rejected():
    lane, _, links, _ = corridor_fixture()
    links.pop("in_1")
    assert not evaluate(lane, ":join_0_0")["valid"]


def test_ambiguous_connection_is_rejected():
    lane, _, links, _ = corridor_fixture()
    links["in_1"] *= 2
    assert not evaluate(lane, ":join_0_0")["valid"]


def test_merging_connections_are_not_treated_as_adjacent_ribbons():
    lane, _, links, _ = corridor_fixture()
    links["in_1"] = (("out_0", ":join_0_1"),)
    links[":join_0_1"] = (("out_0", ""),)
    assert not evaluate(lane, ":join_0_0")["valid"]


@pytest.mark.parametrize("width", [float("nan"), 0.0, -1.0])
def test_invalid_continuation_width_is_rejected(width):
    lane, _, _, widths = corridor_fixture()
    widths["out_1"] = width
    assert not evaluate(lane, ":join_0_0")["valid"]


def test_continuation_does_not_widen_a_narrower_lane():
    lane, _, _, widths = corridor_fixture()
    widths["out_1"] = 2.0
    result = evaluate(lane, ":join_0_0")
    assert result["valid"]
    assert result["d_max"] == pytest.approx(4.21)


def test_disconnected_geometry_is_still_rejected():
    lane, shapes, _, _ = corridor_fixture()
    for name in ["in_1", ":join_0_1", "out_1"]:
        shapes[name] = tuple((x, 30.0) for x, _ in shapes[name])
    assert not evaluate(lane, ":join_0_0")["valid"]


@pytest.mark.parametrize(
    "shape",
    [((11.0, 3.2), (12.0, 3.2)), (), ((10.0, 3.2), (float("nan"), 3.2))],
)
def test_invalid_longitudinal_geometry_join_is_rejected(shape):
    lane, shapes, _, _ = corridor_fixture()
    shapes[":join_0_1"] = shape
    assert not evaluate(lane, ":join_0_0")["valid"]
