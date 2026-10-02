"""Focused gates for the dedicated SUMO external-state build."""

from __future__ import annotations

import importlib
import math
import os
import shutil
import subprocess
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

STEP_LENGTH = 0.05
POSITION_TOLERANCE = 1e-6


def _find_binary(name: str) -> str:
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / name
        if candidate.is_file():
            return str(candidate)
    candidate = shutil.which(name)
    if candidate:
        return candidate
    pytest.skip(f"{name} binary is not available")


def _write_network(tmp_path: Path) -> tuple[Path, Path]:
    nodes = tmp_path / "external-state.nod.xml"
    edges = tmp_path / "external-state.edg.xml"
    network = tmp_path / "external-state.net.xml"
    routes = tmp_path / "external-state.rou.xml"
    nodes.write_text(
        textwrap.dedent(
            """\
            <nodes>
                <node id="start" x="0" y="0" type="dead_end"/>
                <node id="end" x="500" y="0" type="dead_end"/>
            </nodes>
            """
        ),
        encoding="utf-8",
    )
    edges.write_text(
        textwrap.dedent(
            """\
            <edges>
                <edge id="road" from="start" to="end" numLanes="2" speed="30"/>
            </edges>
            """
        ),
        encoding="utf-8",
    )
    routes.write_text(
        textwrap.dedent(
            """\
            <routes>
                <vType id="car" accel="2.6" decel="4.5" emergencyDecel="9"
                       sigma="0" length="5" maxSpeed="30"/>
                <route id="route" edges="road"/>
                <vehicle id="ego" type="car" route="route" depart="0"
                         departLane="0" departPos="20" departSpeed="10"/>
            </routes>
            """
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            _find_binary("netconvert"),
            "--node-files",
            str(nodes),
            "--edge-files",
            str(edges),
            "--output-file",
            str(network),
            "--no-warnings",
            "true",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return network, routes


def _start_backend(backend_name: str, network: Path, routes: Path, *extra_args: str):
    backend = pytest.importorskip(backend_name)
    if not hasattr(backend.vehicle, "moveToXYImmediate"):
        pytest.skip("requires the dedicated SUMO moveToXYImmediate build")
    command = [
        _find_binary("sumo"),
        "--net-file",
        str(network),
        "--route-files",
        str(routes),
        "--step-length",
        str(STEP_LENGTH),
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
    ]
    command.extend(extra_args)
    if backend_name == "traci":
        backend.start(command, numRetries=5)
    else:
        backend.start(command)
    return backend


@pytest.mark.integration
@pytest.mark.requires_sumo
@pytest.mark.parametrize("backend_name", ["traci", "libsumo"])
def test_external_state_is_immediate_and_phase_b_steps_once(
    tmp_path: Path, backend_name: str
) -> None:
    """TraCI and libsumo must expose the same Phase-A/Phase-B contract."""
    network, routes = _write_network(tmp_path)
    backend = _start_backend(backend_name, network, routes)
    try:
        backend.simulationStep()
        assert tuple(backend.vehicle.getIDList()) == ("ego",)
        target = (60.0, backend.vehicle.getPosition("ego")[1])
        angle = backend.vehicle.getAngle("ego")
        phase_a_time = backend.simulation.getTime()
        backend.vehicle.moveToXYImmediate(
            "ego", "road", 0, target[0], target[1], angle, 1, 10.0, True
        )
        backend.vehicle.setSpeed("ego", -1)
        backend.vehicle.setPreviousSpeed("ego", 10.0, 0.25)

        assert backend.simulation.getTime() == phase_a_time
        assert math.dist(backend.vehicle.getPosition("ego"), target) < POSITION_TOLERANCE
        assert backend.vehicle.getSpeed("ego") == pytest.approx(10.0, abs=1e-6)
        assert backend.vehicle.getAcceleration("ego") == pytest.approx(0.25, abs=1e-6)

        phase_a_lane_position = backend.vehicle.getLanePosition("ego")
        backend.simulationStep()
        assert backend.simulation.getTime() == pytest.approx(phase_a_time + STEP_LENGTH)
        assert backend.vehicle.getLaneID("ego") == "road_0"
        assert backend.vehicle.getLanePosition("ego") > phase_a_lane_position
    finally:
        backend.close()


@pytest.mark.integration
@pytest.mark.requires_sumo
def test_traci_and_libsumo_publish_the_immediate_api() -> None:
    for backend_name in ("traci", "libsumo"):
        module = importlib.import_module(backend_name)
        if not hasattr(module.vehicle, "moveToXYImmediate"):
            pytest.skip("requires the dedicated SUMO moveToXYImmediate build")
        assert module.constants.MOVE_TO_XY_IMMEDIATE == 0xF8


@pytest.mark.integration
@pytest.mark.requires_sumo
@pytest.mark.parametrize("backend_name", ["traci", "libsumo"])
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("switch_phase", ["phase_a_rebase", "phase_b_motion"])
@pytest.mark.parametrize("extra_separation", [0.0, 0.25])
def test_external_lane_switch_preserves_world_progress_with_offset_lane_origins(
    tmp_path: Path,
    backend_name: str,
    direction: int,
    switch_phase: str,
    extra_separation: float,
) -> None:
    """A primary-lane coordinate rebase must not add a four-metre world jump."""
    network, routes = _write_network(tmp_path)
    source_index = 1 if direction < 0 else 0
    target_index = source_index + direction
    tree = ET.parse(network)
    target_lane = tree.find(f"./edge[@id='road']/lane[@index='{target_index}']")
    assert target_lane is not None
    # Same width and length, but an adjacent custom shape with a different
    # longitudinal origin, as seen on a real network. Cover both jump signs.
    origin_offset = direction * 4.0
    points = [tuple(map(float, p.split(","))) for p in target_lane.get("shape").split()]
    target_lane.set(
        "shape",
        " ".join(f"{x + origin_offset},{y + direction * extra_separation}" for x, y in points),
    )
    tree.write(network)
    route_tree = ET.parse(routes)
    route_tree.find("vehicle").set("departLane", str(source_index))
    route_tree.write(routes)
    backend = _start_backend(backend_name, network, routes, "--lateral-resolution", "0.4")
    try:
        backend.simulationStep()
        backend.vehicle.setLaneChangeMode("ego", 0)
        backend.vehicle.setSpeedMode("ego", 31)
        backend.vehicle.setSpeed("ego", 10.0)
        backend.vehicle.changeLaneRelative("ego", direction, 8.0)
        x, source_y = backend.vehicle.getPosition("ego")
        switched = False
        # Cover an envelope already crossed by Phase A and one crossed by
        # Phase B motion. Both must preserve longitudinal world progress.
        final_lateral = 1.70 + extra_separation if switch_phase == "phase_a_rebase" else 1.59999
        for lateral in (0.1, 0.3, 0.6, 1.0, 1.45, final_lateral):
            x += 0.5
            y = source_y + direction * lateral
            backend.vehicle.moveToXYImmediate(
                "ego", "road", source_index, x, y, 90.0, 1, 10.0, True
            )
            backend.vehicle.setPreviousSpeed("ego", 10.0, 0.0)
            assert backend.vehicle.getLaneIndex("ego") == source_index
            assert backend.vehicle.getPosition("ego") == pytest.approx((x, y), abs=1e-6)
            distance_before = backend.vehicle.getDistance("ego")
            backend.simulationStep()
            phase_b_x, phase_b_y = backend.vehicle.getPosition("ego")
            assert phase_b_x - x == pytest.approx(0.5, abs=1e-6)
            assert abs(phase_b_y - y) <= 0.1
            assert backend.vehicle.getDistance("ego") - distance_before == pytest.approx(
                0.5, abs=1e-6
            )
            assert backend.vehicle.getLaneChangeMode("ego") == 0
            assert backend.vehicle.getSpeedMode("ego") == 31
            if backend.vehicle.getLaneIndex("ego") == target_index:
                switched = True
                break
        assert switched, "fixture must exercise a native primary-lane switch"
    finally:
        backend.close()
