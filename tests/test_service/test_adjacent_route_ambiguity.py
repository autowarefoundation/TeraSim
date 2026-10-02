"""A future lane's choices must not invalidate usable current-lane geometry."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from terasim_service.plugins import cosim_inprocess
from terasim_service.plugins.cosim_inprocess import TeraSimCoSimInProcessPlugin
from terasim_service.utils.sumo_lane_geometry import (
    build_external_state_lateral_action_lookahead,
    compile_lane_shapes,
)


def resolve(*, actual_choices=(), adjacent_choices=(0, 1, 2), selectors=(0, 1, 2)):
    links = {
        "in_0": tuple((f"out_{i}", f":join_0_{i}") for i in actual_choices),
        "in_1": tuple((f"out_{i}", f":join_0_{i}") for i in adjacent_choices),
        **{f":join_0_{i}": ((f"out_{i}", ""),) for i in range(3)},
    }
    fake = SimpleNamespace(
        lane=SimpleNamespace(
            getEdgeID=lambda lane: lane.rsplit("_", 1)[0],
            getLinks=lambda lane: links.get(lane, ()),
        ),
        vehicle=SimpleNamespace(
            getRoute=lambda _: ("in", "out"),
            getRouteIndex=lambda _: 0,
            getNextLinks=lambda _: tuple((f"out_{i}", f":join_0_{i}") for i in selectors),
        ),
    )
    plugin = object.__new__(TeraSimCoSimInProcessPlugin)
    plugin.physics_lateral_direction_states = {"v": {"phase": "active"}}
    plugin._physics_lane_change_action = lambda *_: ("left", "in_1", "sumo_intent", True)
    with patch.object(cosim_inprocess, "traci", fake):
        return plugin._physics_resolve_route_lane_sequence("v", "in_0")


def test_ambiguous_adjacent_choices_keep_actual_lane_prefix():
    result = resolve()
    assert result["error"] == ""
    assert result["lane_ids"] == ("in_0",)
    assert result["reference_lane_id"] == "in_0"
    assert result["selection_reason"] == (
        "awaiting_required_lane_change:ambiguous_adjacent_connections"
    )


@pytest.mark.parametrize(
    ("x", "error"),
    [(10.0, ""), (28.0, "steering_preview_path_end"), (29.9, "invalid_target_projection")],
)
def test_deferred_choice_is_bounded_by_existing_target_and_preview_checks(x, error):
    result = resolve()
    assert result["error"] == ""
    # Only the actual lane is compiled. No unselected connection is appended.
    shapes = {"in_0": ((0.0, 0.0), (30.0, 0.0))}
    path = compile_lane_shapes([shapes[lane] for lane in result["lane_ids"]])
    action = build_external_state_lateral_action_lookahead(path, (x, 0.0), 0.5)
    assert action["valid"] == (not error)
    assert action["error"] == error


def test_ambiguity_on_actual_lane_is_still_rejected():
    result = resolve(actual_choices=(0, 1))
    assert result["error"] == "ambiguous_intent_target_connections"
    assert result["lane_ids"] == ("in_0",)


@pytest.mark.parametrize("selected", [0, 1, 2])
def test_unique_verified_adjacent_selector_keeps_existing_behavior(selected):
    result = resolve(selectors=(selected,))
    assert result["error"] == ""
    assert result["reference_lane_id"] == "in_1"
    assert result["lane_ids"] == ("in_1", f":join_0_{selected}", f"out_{selected}")
