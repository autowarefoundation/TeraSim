#!/usr/bin/env python3
"""Generate minimal SUMO artifacts for a TeraSim map directory.

This script is intentionally lightweight so it can run inside the existing
`terasim-service:cosim` image without installing the full `terasim-envgen`
dependency stack.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import sumolib


SUMO_STOPLINE_GAP = "4.0"
STOPLINE_GAP_VTYPE_IDS = ("NDE_URBAN", "vehicle_passenger")

NDE_VTYPES = [
    {
        "id": "NDE_URBAN",
        "length": "5.00",
        "width": "1.85",
        "minGap": "3.28",
        "maxSpeed": "10.00",
        "carFollowModel": "IDM",
        "accel": "1.84",
        "decel": "1.29",
        "tau": "1.17",
        "emergencyDecel": "7.06",
        "lcSpeedGain": "0",
        "lcCooperative": "0",
        "lcKeepRight": "1",
        "speedFactor": "normc(1,0.1,0.8,1.2)",
        "jmStoplineGap": SUMO_STOPLINE_GAP,
    },
    {
        "id": "NDE_HIGHWAY",
        "length": "5.00",
        "width": "1.85",
        "minGap": "5.92",
        "maxSpeed": "28.31",
        "carFollowModel": "IDM",
        "accel": "5.95",
        "decel": "5.96",
        "tau": "1.72",
        "emergencyDecel": "7.06",
        "lcSpeedGain": "0",
        "lcCooperative": "0",
        "lcKeepRight": "1",
        "speedFactor": "normc(1,0.1,0.8,1.2)",
    },
]


def ensure_metadata(metadata_path: Path) -> dict:
    if metadata_path.exists():
        data = json.loads(metadata_path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{metadata_path} must contain a JSON object")
    else:
        data = {}
    metadata_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return data


def get_random_trips_script() -> str:
    candidates = [
        os.environ.get("SUMO_RANDOM_TRIPS"),
        "/usr/local/lib/python3.10/site-packages/sumo/tools/randomTrips.py",
        "/usr/share/sumo/tools/randomTrips.py",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("Could not locate randomTrips.py")


def is_regular_route_edge(edge: sumolib.net.edge.Edge) -> bool:
    edge_id = edge.getID()
    return not edge_id.startswith(":") and not edge.isSpecial()


def filter_regular_route_edges(
    route_edges: list[sumolib.net.edge.Edge],
) -> list[sumolib.net.edge.Edge]:
    regular_route_edges = []
    previous_edge_id = None
    for edge in route_edges:
        if not is_regular_route_edge(edge):
            continue
        edge_id = edge.getID()
        if edge_id == previous_edge_id:
            continue
        regular_route_edges.append(edge)
        previous_edge_id = edge_id
    return regular_route_edges


def filter_regular_route_edge_ids(route_edge_ids: list[str]) -> list[str]:
    regular_route_edge_ids = []
    previous_edge_id = None
    for edge_id in route_edge_ids:
        if edge_id.startswith(":"):
            continue
        if edge_id == previous_edge_id:
            continue
        regular_route_edge_ids.append(edge_id)
        previous_edge_id = edge_id
    return regular_route_edge_ids


def parse_route_edge_tokens(route_text: str) -> list[str]:
    return [edge_id.strip() for edge_id in route_text.replace(",", " ").split() if edge_id.strip()]


def load_av_route_edge_ids(route_path: Path, route_id: str | None = None) -> tuple[list[str], str | None]:
    route_text = route_path.read_text(encoding="utf-8")
    try:
        root = ET.fromstring(route_text)
    except ET.ParseError:
        return parse_route_edge_tokens(route_text), None

    route_elements = [elem for elem in root.findall(".//route") if elem.get("edges")]
    if not route_elements:
        raise ValueError(f"No <route edges=...> entry found in {route_path}")

    selected_route = None
    if route_id:
        for route in route_elements:
            if route.get("id") == route_id:
                selected_route = route
                break
        if selected_route is None:
            raise ValueError(f"Route id {route_id!r} not found in {route_path}")
    else:
        preferred_ids = ["av_route", route_path.stem]
        stem_without_outer_suffix = route_path.with_suffix("").stem
        if stem_without_outer_suffix not in preferred_ids:
            preferred_ids.append(stem_without_outer_suffix)
        for preferred_id in preferred_ids:
            for route in route_elements:
                if route.get("id") == preferred_id:
                    selected_route = route
                    break
            if selected_route is not None:
                break
        if selected_route is None:
            if len(route_elements) != 1:
                route_ids = [route.get("id") for route in route_elements]
                raise ValueError(
                    f"Multiple route entries found in {route_path}; specify --av-route-id. "
                    f"Available route ids: {route_ids}"
                )
            selected_route = route_elements[0]

    return parse_route_edge_tokens(selected_route.get("edges", "")), selected_route.get("id")


def get_saved_av_route_edges(
    sumo_net: sumolib.net.Net,
    route_edge_ids: list[str],
) -> list[sumolib.net.edge.Edge]:
    regular_route_edge_ids = filter_regular_route_edge_ids(route_edge_ids)
    route_edges = []
    missing_edge_ids = []
    for edge_id in regular_route_edge_ids:
        try:
            route_edges.append(sumo_net.getEdge(edge_id))
        except Exception:
            missing_edge_ids.append(edge_id)
    if missing_edge_ids:
        raise ValueError(f"Saved AV route edges not found in SUMO net: {missing_edge_ids}")
    return route_edges


def generate_av_fallback_route(net_path: Path, seed: int | None = None) -> list[sumolib.net.edge.Edge]:
    if seed is not None:
        random.seed(seed)

    sumo_net = sumolib.net.readNet(str(net_path), withInternal=True)
    regular_edges = [edge for edge in sumo_net.getEdges() if is_regular_route_edge(edge)]
    if not regular_edges:
        raise RuntimeError(f"No regular edges found in {net_path}")

    peripheral_edges = []
    for edge in regular_edges:
        if len(edge.getIncoming()) <= 1 or len(edge.getOutgoing()) <= 1:
            peripheral_edges.append(edge)

    candidates = peripheral_edges if len(peripheral_edges) >= 2 else regular_edges
    if len(candidates) > 128:
        candidates = random.sample(candidates, 128)

    best_route_edges = None
    best_score = -1.0

    for i, src in enumerate(candidates):
        for dst in candidates[i + 1:]:
            try:
                route_edges, _ = sumo_net.getShortestPath(src, dst)
                if not route_edges:
                    route_edges, _ = sumo_net.getShortestPath(dst, src)
                if not route_edges:
                    continue
                score = sum(edge.getLength() for edge in route_edges)
                if score > best_score:
                    best_score = score
                    best_route_edges = route_edges
            except Exception:
                continue

    if best_route_edges:
        regular_route_edges = filter_regular_route_edges(list(best_route_edges))
        if regular_route_edges:
            return regular_route_edges
    return [max(regular_edges, key=lambda edge: edge.getLength())]


def save_av_route_metadata(
    net_path: Path,
    metadata_path: Path,
    *,
    seed: int | None = None,
    force_new_route: bool = False,
    av_route_file: Path | None = None,
    av_route_id: str | None = None,
) -> list[str]:
    metadata = ensure_metadata(metadata_path)
    sumo_net = sumolib.net.readNet(str(net_path), withInternal=True)

    if force_new_route:
        metadata.pop("av_route", None)
        metadata.pop("av_route_sumo", None)
        metadata.pop("av_route_edge_ids", None)

    if av_route_file is not None:
        av_route_edge_ids, selected_route_id = load_av_route_edge_ids(av_route_file, av_route_id)
        av_route_objects = get_saved_av_route_edges(sumo_net, av_route_edge_ids)
        metadata["av_route_name"] = selected_route_id or av_route_file.stem
        metadata["av_route_source"] = "route_file"
        metadata["av_route_file"] = str(av_route_file)
    elif metadata.get("av_route_edge_ids"):
        try:
            av_route_objects = get_saved_av_route_edges(sumo_net, metadata["av_route_edge_ids"])
        except Exception as exc:
            print(
                f"Warning: saved av_route_edge_ids could not be used ({exc}); "
                "falling back to av_route coordinates.",
                file=sys.stderr,
            )
            av_route_objects = []
    else:
        av_route_objects = []

    if not av_route_objects:
        if metadata.get("av_route"):
            try:
                av_route_xy = [
                    sumo_net.convertLonLat2XY(point[1], point[0])
                    for point in metadata["av_route"]
                ]
                av_route_edges = sumolib.route.mapTrace(
                    av_route_xy,
                    sumo_net,
                    delta=10,
                    fillGaps=100,
                    verbose=False,
                )
            except Exception:
                av_route_edges = []
            av_route_objects = av_route_edges

        if not av_route_objects:
            av_route_objects = generate_av_fallback_route(net_path, seed=seed)

    av_route_objects = filter_regular_route_edges(list(av_route_objects))
    if not av_route_objects:
        av_route_objects = generate_av_fallback_route(net_path, seed=seed)

    av_route_with_internal = sumolib.route.addInternal(sumo_net, av_route_objects)

    latlon_points = []
    try:
        for edge in av_route_with_internal:
            for x, y in edge.getShape():
                lon, lat = sumo_net.convertXY2LonLat(x, y)
                latlon_points.append([lat, lon])
    except Exception:
        latlon_points = []

    av_route_edge_ids = filter_regular_route_edge_ids([edge.getID() for edge in av_route_objects])
    metadata["av_route_sumo"] = latlon_points
    metadata["av_route_edge_ids"] = av_route_edge_ids
    if "av_route" not in metadata and latlon_points:
        metadata["av_route"] = latlon_points
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    return av_route_edge_ids


def create_minimal_routes(routes_path: Path) -> None:
    root = ET.Element("routes")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(routes_path, encoding="utf-8", xml_declaration=True)


def generate_vehicle_routes(
    net_path: Path,
    trips_path: Path,
    routes_path: Path,
    *,
    end_time: int,
    period: float,
    seed: int,
) -> None:
    random_trips = get_random_trips_script()
    base_cmd = [
        sys.executable,
        random_trips,
        "-n",
        str(net_path),
        "-o",
        str(trips_path),
        "-r",
        str(routes_path),
        "-e",
        str(end_time),
        "-p",
        str(period),
        "--random",
        "--seed",
        str(seed),
        "--lanes",
        "--length",
        "--vehicle-class",
        "passenger",
        "--vclass",
        "passenger",
        "--prefix",
        "vehicle",
        "--validate",
        "--route-file",
        str(routes_path),
    ]

    commands = [
        base_cmd + ["--fringe-factor", "100", "--allow-fringe"],
        base_cmd + ["--fringe-factor", "1.0", "--allow-fringe"],
    ]

    last_error = None
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc

    create_minimal_routes(routes_path)
    if last_error:
        print("Warning: randomTrips.py failed; created minimal routes file instead.", file=sys.stderr)
        if last_error.stderr:
            print(last_error.stderr, file=sys.stderr)


def ensure_vtypes_and_av_route(routes_path: Path, av_route_edges: list[str]) -> None:
    av_route_edges = filter_regular_route_edge_ids(av_route_edges)
    if not av_route_edges:
        raise ValueError("AV route must contain at least one regular SUMO edge")

    if routes_path.exists():
        tree = ET.parse(routes_path)
        root = tree.getroot()
    else:
        root = ET.Element("routes")
        tree = ET.ElementTree(root)

    existing_vtypes = {elem.get("id") for elem in root.findall("vType")}
    insert_index = 0
    for vtype in reversed(NDE_VTYPES):
        if vtype["id"] in existing_vtypes:
            continue
        elem = ET.Element("vType", vtype)
        root.insert(insert_index, elem)
    for vtype_id in STOPLINE_GAP_VTYPE_IDS:
        vtype = root.find(f"vType[@id='{vtype_id}']")
        if vtype is not None:
            vtype.set("jmStoplineGap", SUMO_STOPLINE_GAP)
    existing_route = root.find("route[@id='av_route']")
    if existing_route is not None:
        root.remove(existing_route)
    root.append(ET.Element("route", {"id": "av_route", "edges": " ".join(av_route_edges)}))

    ET.indent(tree, space="  ")
    tree.write(routes_path, encoding="utf-8", xml_declaration=True)


def create_sumo_config(net_path: Path, routes_path: Path, cfg_path: Path, *, end_time: int) -> None:
    root = ET.Element("configuration")
    input_section = ET.SubElement(root, "input")
    ET.SubElement(input_section, "net-file", {"value": net_path.name})
    ET.SubElement(input_section, "route-files", {"value": routes_path.name})
    ET.SubElement(input_section, "step-length", {"value": "0.1"})

    time_section = ET.SubElement(root, "time")
    ET.SubElement(time_section, "begin", {"value": "0"})
    ET.SubElement(time_section, "end", {"value": str(end_time)})

    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "lateral-resolution", {"value": "0.5"})
    ET.SubElement(processing, "time-to-teleport", {"value": "-1"})
    ET.SubElement(processing, "collision.action", {"value": "warn"})
    ET.SubElement(processing, "collision.check-junctions", {"value": "true"})
    ET.SubElement(processing, "collision.mingap-factor", {"value": "0"})

    report = ET.SubElement(root, "report")
    ET.SubElement(report, "verbose", {"value": "true"})
    ET.SubElement(report, "no-step-log", {"value": "true"})

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(cfg_path, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SUMO artifacts from a SUMO network.")
    parser.add_argument("--net", required=True, help="Path to the SUMO network file.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated files.")
    parser.add_argument("--metadata", default=None, help="Path to metadata.json (default: <output-dir>/metadata.json).")
    parser.add_argument("--end-time", type=int, default=3600, help="Simulation end time in seconds.")
    parser.add_argument("--period", type=float, default=2.0, help="Vehicle generation period for randomTrips.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument(
        "--av-route-seed",
        type=int,
        default=None,
        help="Random seed used only for AV route generation. Defaults to --seed.",
    )
    parser.add_argument(
        "--force-new-av-route",
        action="store_true",
        help="Discard any saved AV route metadata and generate a fresh AV route.",
    )
    parser.add_argument(
        "--av-route-file",
        default=None,
        help=(
            "Optional AV route file. Supports SUMO route XML with <route edges=...> "
            "or a plain text edge list. When omitted, existing metadata/fallback behavior is used."
        ),
    )
    parser.add_argument(
        "--av-route-id",
        default=None,
        help="Route id to read from --av-route-file when the file contains multiple routes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    net_path = Path(args.net).resolve()
    output_dir = Path(args.output_dir).resolve()
    metadata_path = Path(args.metadata).resolve() if args.metadata else output_dir / "metadata.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    trips_path = output_dir / "trips.trips.xml"
    routes_path = output_dir / "vehicles.rou.xml"
    cfg_path = output_dir / "simulation.sumocfg"
    av_route_seed = args.seed if args.av_route_seed is None else args.av_route_seed

    av_route_edges = save_av_route_metadata(
        net_path,
        metadata_path,
        seed=av_route_seed,
        force_new_route=args.force_new_av_route,
        av_route_file=Path(args.av_route_file).resolve() if args.av_route_file else None,
        av_route_id=args.av_route_id,
    )
    generate_vehicle_routes(
        net_path,
        trips_path,
        routes_path,
        end_time=args.end_time,
        period=args.period,
        seed=args.seed,
    )
    ensure_vtypes_and_av_route(routes_path, av_route_edges)
    create_sumo_config(net_path, routes_path, cfg_path, end_time=args.end_time)

    print(f"Generated metadata: {metadata_path}")
    print(f"Generated trips: {trips_path}")
    print(f"Generated routes: {routes_path}")
    print(f"Generated SUMO config: {cfg_path}")
    print(f"AV route seed: {av_route_seed}")
    print(f"Forced new AV route: {args.force_new_av_route}")
    print(f"Generated AV route edges: {' '.join(av_route_edges)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
