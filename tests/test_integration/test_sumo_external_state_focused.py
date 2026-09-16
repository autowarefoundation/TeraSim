"""Focused gates for the dedicated SUMO external-state build."""

from __future__ import annotations

import importlib
import math
import os
import shutil
import subprocess
import textwrap
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


def _start_backend(backend_name: str, network: Path, routes: Path):
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
        assert backend.simulation.getTime() == pytest.approx(
            phase_a_time + STEP_LENGTH
        )
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
