import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytest.importorskip("sumolib")


def load_generator_module():
    script_path = Path(__file__).parents[2] / "scripts" / "generate_sumo_artifacts_from_net.py"
    spec = importlib.util.spec_from_file_location("generate_sumo_artifacts_from_net", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeEdge:
    def __init__(self, edge_id, special=False):
        self.edge_id = edge_id
        self.special = special

    def getID(self):
        return self.edge_id

    def isSpecial(self):
        return self.special


class FakeNet:
    def __init__(self, edge_ids):
        self.edges = {edge_id: FakeEdge(edge_id) for edge_id in edge_ids}

    def getEdge(self, edge_id):
        return self.edges[edge_id]


def test_filter_regular_route_edges_removes_internal_special_and_consecutive_duplicates():
    module = load_generator_module()

    route_edges = [
        FakeEdge("edge_755"),
        FakeEdge(":node_1195_2"),
        FakeEdge("edge_760"),
        FakeEdge("edge_760"),
        FakeEdge("edge_special", special=True),
        FakeEdge("edge_2381"),
        FakeEdge(":node_673_1"),
        FakeEdge("edge_2381"),
        FakeEdge("edge_2456"),
    ]

    regular_edges = module.filter_regular_route_edges(route_edges)

    assert [edge.getID() for edge in regular_edges] == [
        "edge_755",
        "edge_760",
        "edge_2381",
        "edge_2456",
    ]


def test_get_saved_av_route_edges_uses_saved_ids_and_filters_internal_duplicates():
    module = load_generator_module()

    route_edges = module.get_saved_av_route_edges(
        FakeNet(["edge_459", "edge_426", "edge_3"]),
        ["edge_459", ":node_1_0", "edge_426", "edge_426", "edge_3"],
    )

    assert [edge.getID() for edge in route_edges] == ["edge_459", "edge_426", "edge_3"]


def test_load_av_route_edge_ids_from_sumo_route_xml(tmp_path):
    module = load_generator_module()
    route_path = tmp_path / "teleport-mirai-loop.rou.xml"
    route_path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<routes>
  <route id="other" edges="edge_1 edge_2" />
  <route id="teleport-mirai-loop" edges="edge_459 edge_357 edge_426" />
</routes>
""",
        encoding="utf-8",
    )

    route_edges, route_id = module.load_av_route_edge_ids(route_path)

    assert route_id == "teleport-mirai-loop"
    assert route_edges == ["edge_459", "edge_357", "edge_426"]


def test_load_av_route_edge_ids_from_text_file(tmp_path):
    module = load_generator_module()
    route_path = tmp_path / "av_route.txt"
    route_path.write_text("edge_459, edge_357\nedge_426\n", encoding="utf-8")

    route_edges, route_id = module.load_av_route_edge_ids(route_path)

    assert route_id is None
    assert route_edges == ["edge_459", "edge_357", "edge_426"]


def test_ensure_vtypes_and_av_route_writes_only_regular_route_edges(tmp_path):
    module = load_generator_module()
    routes_path = tmp_path / "vehicles.rou.xml"
    routes_path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<routes>
  <vType id="vehicle_passenger" vClass="passenger" />
</routes>
""",
        encoding="utf-8",
    )

    module.ensure_vtypes_and_av_route(
        routes_path,
        ["edge_755", ":node_1195_2", "edge_760", "edge_760"],
    )

    route = ET.parse(routes_path).getroot().find("route[@id='av_route']")
    assert route is not None
    assert route.get("edges") == "edge_755 edge_760"

    root = ET.parse(routes_path).getroot()
    for vtype_id in module.STOPLINE_GAP_VTYPE_IDS:
        vtype = root.find(f"vType[@id='{vtype_id}']")
        assert vtype is not None
        assert vtype.get("jmStoplineGap") == module.SUMO_STOPLINE_GAP
