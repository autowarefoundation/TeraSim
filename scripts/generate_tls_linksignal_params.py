#!/usr/bin/env python3
"""Generate SUMO tlLogic linkSignalID params from map-conversion signal mappings."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUMO_CARLA_TLS_LINK_PREFIX = "linkSignalID:"
OD_PREFIX = "od:"
OPENDRIVE_SIGNAL_ID_OFFSET = 2_000_000
DEFAULT_KNOWN_UNMAPPED_RECORDS = 11


@dataclass
class NetContext:
    tree: ET.ElementTree
    root: ET.Element
    tls_logic_by_id: dict[str, ET.Element]
    tls_ids: set[str]
    phase_state_lengths: dict[str, list[int]]
    link_indexes_by_tls_and_node: dict[str, dict[str, set[int]]]
    all_link_indexes_by_tls: dict[str, set[int]]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_opendrive_signal_id(signal_id: Any) -> str:
    """Normalize the 2000000+N signal id convention to OpenDRIVE id N."""
    value = str(signal_id).strip()
    if value.startswith(OD_PREFIX):
        value = value[len(OD_PREFIX) :]

    try:
        numeric_value = int(value)
    except ValueError:
        return value

    if numeric_value >= OPENDRIVE_SIGNAL_ID_OFFSET:
        numeric_value -= OPENDRIVE_SIGNAL_ID_OFFSET
    return str(numeric_value)


def opendrive_token(signal_id: Any) -> str:
    return f"{OD_PREFIX}{normalize_opendrive_signal_id(signal_id)}"


def sort_signal_tokens(tokens: set[str]) -> list[str]:
    def sort_key(token: str) -> tuple[int, int | str]:
        value = token[len(OD_PREFIX) :] if token.startswith(OD_PREFIX) else token
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return sorted(tokens, key=sort_key)


def extract_via_node_id(via: str | None) -> str | None:
    if not via:
        return None

    body = via[1:] if via.startswith(":") else via
    parts = body.rsplit("_", 2)
    if len(parts) == 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return parts[0]

    parts = body.rsplit("_", 1)
    if len(parts) == 2 and parts[-1].isdigit():
        return parts[0]

    return body


def parse_sumo_net(net_path: Path) -> NetContext:  # noqa: C901
    tree = ET.parse(net_path)
    root = tree.getroot()

    tls_logic_by_id: dict[str, ET.Element] = {}
    phase_state_lengths: dict[str, list[int]] = {}
    for tl_logic in root.findall("tlLogic"):
        tls_id = tl_logic.get("id")
        if not tls_id:
            continue
        tls_logic_by_id[tls_id] = tl_logic
        phase_state_lengths.setdefault(tls_id, []).extend(
            [len(phase.get("state", "")) for phase in tl_logic.findall("phase")]
        )

    edge_to_node: dict[str, str] = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        to_node = edge.get("to")
        if edge_id and to_node:
            edge_to_node[edge_id] = to_node

    link_indexes_by_tls_and_node: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    all_link_indexes_by_tls: dict[str, set[int]] = defaultdict(set)

    for connection in root.findall("connection"):
        tls_id = connection.get("tl")
        link_index_text = connection.get("linkIndex")
        if not tls_id or link_index_text is None:
            continue

        try:
            link_index = int(link_index_text)
        except ValueError:
            continue

        all_link_indexes_by_tls[tls_id].add(link_index)

        candidate_nodes = set()
        via_node = extract_via_node_id(connection.get("via"))
        if via_node:
            candidate_nodes.add(via_node)

        # The SUMO incoming edge endpoint is the junction/node reached by the "from" edge.
        from_edge_id = connection.get("from")
        if from_edge_id and from_edge_id in edge_to_node:
            candidate_nodes.add(edge_to_node[from_edge_id])

        for node_id in candidate_nodes:
            link_indexes_by_tls_and_node[tls_id][node_id].add(link_index)

    return NetContext(
        tree=tree,
        root=root,
        tls_logic_by_id=tls_logic_by_id,
        tls_ids=set(tls_logic_by_id),
        phase_state_lengths=phase_state_lengths,
        link_indexes_by_tls_and_node=link_indexes_by_tls_and_node,
        all_link_indexes_by_tls=all_link_indexes_by_tls,
    )


def load_lanelet_signal_map(opendrive_lanelet_mapping_path: Path) -> dict[str, list[Any]]:
    mapping = load_json(opendrive_lanelet_mapping_path)
    signal_mapping = mapping.get("traffic_light_signal_mapping", {})
    lanelet_to_signal = signal_mapping.get("lanelet2_tl_id_to_signal_ids", {})
    return {str(lanelet_id): signal_ids for lanelet_id, signal_ids in lanelet_to_signal.items()}


def choose_target_tls(record: dict[str, Any], target_tls_ids: set[str]) -> tuple[str | None, str]:
    for tls_id in record.get("actual_sumo_tls_ids", []) or []:
        if tls_id in target_tls_ids:
            return tls_id, "actual_sumo_tls_ids"

    planned_tls_id = record.get("planned_sumo_tls_id")
    if planned_tls_id in target_tls_ids:
        return planned_tls_id, "planned_sumo_tls_id"

    return None, "not_in_target_net"


def record_signal_tokens(
    record: dict[str, Any],
    lanelet_to_signal_ids: dict[str, list[Any]],
) -> tuple[set[str], list[str]]:
    tokens: set[str] = set()
    matched_lanelet_ids: list[str] = []

    # The generated mappings join through regulatory element ids.
    # Keep traffic light way ids as fallback for compatible future mappings.
    for field_name in ("lanelet_regulatory_element_ids", "lanelet_traffic_light_way_ids"):
        for lanelet_id in record.get(field_name, []) or []:
            lanelet_key = str(lanelet_id)
            signal_ids = lanelet_to_signal_ids.get(lanelet_key)
            if not signal_ids:
                continue
            matched_lanelet_ids.append(lanelet_key)
            for signal_id in signal_ids:
                tokens.add(opendrive_token(signal_id))

    return tokens, matched_lanelet_ids


def is_valid_link_index(net_context: NetContext, tls_id: str, link_index: int) -> bool:
    phase_lengths = net_context.phase_state_lengths.get(tls_id, [])
    return (
        bool(phase_lengths)
        and 0 <= link_index < min(phase_lengths)
        and link_index in net_context.all_link_indexes_by_tls.get(tls_id, set())
    )


def remove_existing_linksignal_params(tl_logic: ET.Element) -> None:
    for param in list(tl_logic.findall("param")):
        key = param.get("key", "")
        if key.startswith(SUMO_CARLA_TLS_LINK_PREFIX):
            tl_logic.remove(param)


def copy_map_metadata(source_net: Path, output_net: Path) -> None:
    source_metadata = source_net.parent / "metadata.json"
    if not source_metadata.exists():
        return

    target_metadata = output_net.parent / "metadata.json"
    if source_metadata.resolve() == target_metadata.resolve():
        return

    target_metadata.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_metadata, target_metadata)


def write_generated_sumocfg(  # noqa: C901
    source_sumocfg: Path,
    output_sumocfg: Path,
    output_net: Path,
) -> None:
    tree = ET.parse(source_sumocfg)
    root = tree.getroot()
    input_element = root.find("input")
    if input_element is None:
        input_element = ET.SubElement(root, "input")

    net_file = input_element.find("net-file")
    if net_file is None:
        net_file = ET.SubElement(input_element, "net-file")
    net_file.set("value", str(output_net.resolve()))

    source_dir = source_sumocfg.parent
    for child in list(input_element):
        if child.tag == "net-file":
            continue
        if not (child.tag.endswith("-file") or child.tag.endswith("-files")):
            continue
        value = child.get("value")
        if not value:
            continue

        parts = []
        changed = False
        for raw_part in value.split(","):
            part = raw_part.strip()
            if not part:
                continue
            path = Path(part)
            if not path.is_absolute():
                path = (source_dir / path).resolve()
                changed = True
            parts.append(str(path))
        if changed:
            child.set("value", ",".join(parts))

    output_sumocfg.parent.mkdir(parents=True, exist_ok=True)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    tree.write(output_sumocfg, encoding="utf-8", xml_declaration=True)


def generate_tls_linksignal_params(  # noqa: C901
    sumo_net: Path,
    signal_id_mapping: Path,
    opendrive_lanelet_mapping: Path,
    sumocfg: Path,
    output_net: Path,
    output_sumocfg: Path,
    report_path: Path,
    min_coverage: float,
    known_unmapped_records: int = DEFAULT_KNOWN_UNMAPPED_RECORDS,
) -> dict[str, Any]:
    sources = {
        path.resolve() for path in (sumo_net, signal_id_mapping, opendrive_lanelet_mapping, sumocfg)
    }
    outputs = [path.resolve() for path in (output_net, output_sumocfg, report_path)]
    if sources.intersection(outputs) or len(set(outputs)) != len(outputs):
        raise ValueError("Output paths must be distinct from all inputs and each other")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be between 0 and 1")
    net_context = parse_sumo_net(sumo_net)
    signal_mapping_data = load_json(signal_id_mapping)
    lanelet_to_signal_ids = load_lanelet_signal_map(opendrive_lanelet_mapping)
    records = signal_mapping_data.get("lanelet_to_sumo", [])

    params_by_tls: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    target_candidate_link_indexes_by_tls: dict[str, set[int]] = defaultdict(set)
    skipped_records: list[dict[str, Any]] = []
    skipped_by_reason: Counter[str] = Counter()
    target_tls_source_counts: Counter[str] = Counter()
    mapped_record_count = 0
    invalid_link_index_count = 0

    schema_version = signal_mapping_data.get("schema_version", 1)
    source_record_status_counts = Counter(record.get("resolution_status") for record in records)
    if schema_version >= 3:
        records = [
            record
            for record in signal_mapping_data["sumo_link_to_lanelet_signal"]
            if record.get("sync_eligible")
        ]
        for record_index, record in enumerate(records):
            tls_id = record.get("actual_sumo_tls_id")
            link_index = record.get("linkIndex")
            target_tls_source_counts["explicit_sumo_link"] += 1
            if tls_id in net_context.tls_ids and isinstance(link_index, int):
                target_candidate_link_indexes_by_tls[tls_id].add(link_index)
            if not isinstance(link_index, int) or not is_valid_link_index(
                net_context, tls_id, link_index
            ):
                invalid_link_index_count += 1
                skipped_by_reason["invalid_link_index"] += 1
                skipped_records.append(
                    {
                        "record_index": record_index,
                        "reason": "invalid_link_index",
                        "target_tls_id": tls_id,
                        "link_index": link_index,
                    }
                )
                continue
            tokens, _ = record_signal_tokens(record, lanelet_to_signal_ids)
            if not tokens:
                skipped_by_reason["signal_id_not_found"] += 1
                skipped_records.append(
                    {
                        "record_index": record_index,
                        "reason": "signal_id_not_found",
                        "target_tls_id": tls_id,
                        "link_index": link_index,
                    }
                )
                continue
            params_by_tls[tls_id][link_index].update(tokens)
            mapped_record_count += 1
    else:
        for record_index, record in enumerate(records):
            if record.get("resolution_status") != "mapped":
                skipped_by_reason["unmapped_resolution_status"] += 1
                skipped_records.append(
                    {
                        "record_index": record_index,
                        "reason": "unmapped_resolution_status",
                        "planned_sumo_tls_id": record.get("planned_sumo_tls_id"),
                        "planned_sumo_node_ids": record.get("planned_sumo_node_ids", []),
                        "source_reason": record.get("reason"),
                    }
                )
                continue

            target_tls_id, tls_source = choose_target_tls(record, net_context.tls_ids)
            if target_tls_id is None:
                skipped_by_reason["target_tls_not_found"] += 1
                skipped_records.append(
                    {
                        "record_index": record_index,
                        "reason": "target_tls_not_found",
                        "planned_sumo_tls_id": record.get("planned_sumo_tls_id"),
                        "actual_sumo_tls_ids": record.get("actual_sumo_tls_ids", []),
                    }
                )
                continue
            target_tls_source_counts[tls_source] += 1

            signal_tokens, matched_lanelet_ids = record_signal_tokens(record, lanelet_to_signal_ids)
            if not signal_tokens:
                skipped_by_reason["signal_id_not_found"] += 1
                skipped_records.append(
                    {
                        "record_index": record_index,
                        "reason": "signal_id_not_found",
                        "target_tls_id": target_tls_id,
                        "lanelet_regulatory_element_ids": record.get(
                            "lanelet_regulatory_element_ids", []
                        ),
                        "lanelet_traffic_light_way_ids": record.get(
                            "lanelet_traffic_light_way_ids", []
                        ),
                    }
                )
                continue

            planned_nodes = [str(node_id) for node_id in record.get("planned_sumo_node_ids", [])]
            link_indexes: set[int] = set()
            for node_id in planned_nodes:
                link_indexes.update(
                    net_context.link_indexes_by_tls_and_node.get(target_tls_id, {}).get(
                        node_id, set()
                    )
                )

            if not link_indexes:
                skipped_by_reason["link_index_not_found"] += 1
                skipped_records.append(
                    {
                        "record_index": record_index,
                        "reason": "link_index_not_found",
                        "target_tls_id": target_tls_id,
                        "planned_sumo_node_ids": planned_nodes,
                        "matched_lanelet_ids": matched_lanelet_ids,
                        "signal_tokens": sort_signal_tokens(signal_tokens),
                    }
                )
                continue

            target_candidate_link_indexes_by_tls[target_tls_id].update(
                net_context.all_link_indexes_by_tls.get(target_tls_id, set())
            )

            wrote_any_link = False
            for link_index in sorted(link_indexes):
                if not is_valid_link_index(net_context, target_tls_id, link_index):
                    invalid_link_index_count += 1
                    skipped_by_reason["invalid_link_index"] += 1
                    skipped_records.append(
                        {
                            "record_index": record_index,
                            "reason": "invalid_link_index",
                            "target_tls_id": target_tls_id,
                            "link_index": link_index,
                            "phase_state_lengths": net_context.phase_state_lengths.get(
                                target_tls_id, []
                            ),
                        }
                    )
                    continue
                params_by_tls[target_tls_id][link_index].update(signal_tokens)
                wrote_any_link = True

            if wrote_any_link:
                mapped_record_count += 1

    for tl_logic in net_context.root.findall("tlLogic"):
        # Replacing metadata must not retain stale tokens from an earlier generation.
        remove_existing_linksignal_params(tl_logic)
        for link_index, tokens in sorted(params_by_tls.get(tl_logic.get("id"), {}).items()):
            ET.SubElement(
                tl_logic,
                "param",
                {
                    "key": f"{SUMO_CARLA_TLS_LINK_PREFIX}{link_index}",
                    "value": " ".join(sort_signal_tokens(tokens)),
                },
            )

    written_link_indexes_by_tls = {
        tls_id: set(link_params) for tls_id, link_params in params_by_tls.items()
    }
    candidate_link_indexes = set()
    written_link_indexes = set()
    for tls_id, link_indexes in target_candidate_link_indexes_by_tls.items():
        for link_index in link_indexes:
            candidate_link_indexes.add((tls_id, link_index))
    for tls_id, link_indexes in written_link_indexes_by_tls.items():
        for link_index in link_indexes:
            written_link_indexes.add((tls_id, link_index))

    # Schema 3 includes 899 eligible connection records (882 distinct TLS/link keys
    # for the pinned map). Keep the planned conservative 899-record denominator.
    coverage_denominator = len(records) if schema_version >= 3 else len(candidate_link_indexes)
    coverage = (
        len(written_link_indexes & candidate_link_indexes) / coverage_denominator
        if coverage_denominator
        else 0.0
    )
    record_coverage = mapped_record_count / len(records) if records else 1.0

    report: dict[str, Any] = {
        "coverage": coverage,
        "coverage_denominator": coverage_denominator,
        "coverage_mode": (
            "written_links_per_sync_eligible_record"
            if schema_version >= 3
            else "target_tls_link_index"
        ),
        "schema_version": schema_version,
        "source_record_status_counts": dict(source_record_status_counts),
        "min_coverage": min_coverage,
        "record_coverage": record_coverage,
        "records_total": len(records),
        "records_mapped_to_linksignal": mapped_record_count,
        "records_skipped": len(records) - mapped_record_count,
        "known_unmapped_records": known_unmapped_records,
        "known_unmapped_records_matched": (
            skipped_by_reason["unmapped_resolution_status"] == known_unmapped_records
        ),
        "skipped_by_reason": dict(sorted(skipped_by_reason.items())),
        "target_tls_source_counts": dict(sorted(target_tls_source_counts.items())),
        "tls_with_linksignal_params": len(params_by_tls),
        "params_written": sum(len(link_params) for link_params in params_by_tls.values()),
        "candidate_link_indexes": len(candidate_link_indexes),
        "link_indexes_written": len(written_link_indexes & candidate_link_indexes),
        "invalid_link_index_count": invalid_link_index_count,
        "source_files": {
            "sumo_net": str(sumo_net),
            "signal_id_mapping": str(signal_id_mapping),
            "opendrive_lanelet_mapping": str(opendrive_lanelet_mapping),
            "sumocfg": str(sumocfg),
        },
        "output_files": {
            "sumo_net": str(output_net),
            "sumocfg": str(output_sumocfg),
            "report": str(report_path),
        },
        "skipped_records": skipped_records,
    }

    token_bindings = defaultdict(list)
    for tls_id, link_params in sorted(params_by_tls.items()):
        for index, tokens in sorted(link_params.items()):
            for token in sort_signal_tokens(tokens):
                token_bindings[token].append({"tls_id": tls_id, "link_index": index})
    report["unique_signal_tokens"] = len(token_bindings)
    report["signal_bindings"] = dict(sorted(token_bindings.items()))
    report["ambiguous_signal_bindings"] = {
        token: bindings
        for token, bindings in sorted(token_bindings.items())
        if len({binding["tls_id"] for binding in bindings}) > 1
        or any(
            len(
                {
                    phase.get("state")[binding["link_index"]].lower()
                    for binding in bindings
                    if binding["tls_id"] == tls_id
                }
            )
            > 1
            for tls_id in {binding["tls_id"] for binding in bindings}
            for phase in net_context.tls_logic_by_id[tls_id].findall("phase")
        )
    }
    report["source_sha256"] = {
        key: hashlib.sha256(Path(value).read_bytes()).hexdigest()
        for key, value in report["source_files"].items()
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if not report["params_written"] or invalid_link_index_count:
        raise RuntimeError(
            f"Invalid TLS mapping: params={report['params_written']}, "
            f"invalid link indexes={invalid_link_index_count}. See {report_path}"
        )
    if coverage < min_coverage:
        raise RuntimeError(
            f"TLS linkSignalID coverage {coverage:.3f} is below required {min_coverage:.3f}. "
            f"See report: {report_path}"
        )

    output_net.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(net_context.tree, space="  ")
    net_context.tree.write(output_net, encoding="utf-8", xml_declaration=True)
    write_generated_sumocfg(sumocfg, output_sumocfg, output_net)
    copy_map_metadata(sumo_net, output_net)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inject linkSignalID params into a SUMO net from map-conversion mappings."
    )
    parser.add_argument("--sumo-net", required=True, type=Path)
    parser.add_argument("--signal-id-mapping", required=True, type=Path)
    parser.add_argument("--opendrive-lanelet-mapping", required=True, type=Path)
    parser.add_argument("--sumocfg", required=True, type=Path)
    parser.add_argument("--output-net", required=True, type=Path)
    parser.add_argument("--output-sumocfg", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument(
        "--known-unmapped-records",
        type=int,
        default=DEFAULT_KNOWN_UNMAPPED_RECORDS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = generate_tls_linksignal_params(
            sumo_net=args.sumo_net,
            signal_id_mapping=args.signal_id_mapping,
            opendrive_lanelet_mapping=args.opendrive_lanelet_mapping,
            sumocfg=args.sumocfg,
            output_net=args.output_net,
            output_sumocfg=args.output_sumocfg,
            report_path=args.report,
            min_coverage=args.min_coverage,
            known_unmapped_records=args.known_unmapped_records,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "Generated TLS linkSignalID params: "
        f"coverage={report['coverage']:.3f}, "
        f"record_coverage={report['record_coverage']:.3f}, "
        f"params={report['params_written']}, "
        f"net={report['output_files']['sumo_net']}, "
        f"sumocfg={report['output_files']['sumocfg']}, "
        f"report={report['output_files']['report']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
