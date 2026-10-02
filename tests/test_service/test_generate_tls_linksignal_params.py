import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


def load_generator_module():
    script_path = Path(__file__).parents[2] / "scripts" / "generate_tls_linksignal_params.py"
    spec = importlib.util.spec_from_file_location("generate_tls_linksignal_params", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_synthetic_inputs(tmp_path, tls_id="tls_a"):
    net_path = tmp_path / "input.net.xml"
    net_path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<net>
  <edge id="incoming_a" from="upstream_a" to="node_a">
    <lane id="incoming_a_0" index="0" speed="13.89" length="10.0" />
  </edge>
  <edge id="incoming_b" from="upstream_b" to="node_b">
    <lane id="incoming_b_0" index="0" speed="13.89" length="10.0" />
  </edge>
  <edge id="out_a" from="node_a" to="downstream_a">
    <lane id="out_a_0" index="0" speed="13.89" length="10.0" />
  </edge>
  <edge id="out_b" from="node_b" to="downstream_b">
    <lane id="out_b_0" index="0" speed="13.89" length="10.0" />
  </edge>
  <tlLogic id="{tls_id}" type="static" programID="0" offset="0">
    <phase duration="10" state="Gr" />
    <param key="linkSignalID:0" value="legacy" />
  </tlLogic>
  <connection from="incoming_a" to="out_a" via=":node_a_0_0" tl="{tls_id}" linkIndex="0" />
  <connection from="incoming_b" to="out_b" via=":node_b_0_0" tl="{tls_id}" linkIndex="1" />
</net>
""",
        encoding="utf-8",
    )

    route_path = tmp_path / "vehicles.rou.xml"
    route_path.write_text("<routes />\n", encoding="utf-8")

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text('{"map": "synthetic"}\n', encoding="utf-8")

    sumocfg_path = tmp_path / "simulation.sumocfg"
    sumocfg_path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <input>
    <net-file value="input.net.xml" />
    <route-files value="vehicles.rou.xml" />
    <step-length value="0.1" />
  </input>
</configuration>
""",
        encoding="utf-8",
    )

    signal_mapping_path = tmp_path / "signal_id_mapping.json"
    signal_mapping_path.write_text(
        json.dumps(
            {
                "lanelet_to_sumo": [
                    {
                        "actual_sumo_tls_ids": [],
                        "lanelet_regulatory_element_ids": ["reg_a"],
                        "lanelet_traffic_light_way_ids": [],
                        "planned_sumo_node_ids": ["node_a"],
                        "planned_sumo_tls_id": tls_id,
                        "resolution_status": "mapped",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    opendrive_mapping_path = tmp_path / "tl_mapping_test.mapping.json"
    opendrive_mapping_path.write_text(
        json.dumps(
            {
                "traffic_light_signal_mapping": {
                    "lanelet2_tl_id_to_signal_ids": {
                        "reg_a": [2000466, 467],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    return net_path, signal_mapping_path, opendrive_mapping_path, sumocfg_path, route_path


def test_generate_linksignal_params_writes_od_tokens_and_sumocfg(tmp_path):
    module = load_generator_module()
    net_path, signal_mapping_path, od_mapping_path, sumocfg_path, route_path = (
        write_synthetic_inputs(tmp_path)
    )
    output_net = tmp_path / "out" / "tls_synced.net.xml"
    output_sumocfg = tmp_path / "out" / "simulation_tls_synced.sumocfg"
    report_path = tmp_path / "out" / "report.json"

    report = module.generate_tls_linksignal_params(
        sumo_net=net_path,
        signal_id_mapping=signal_mapping_path,
        opendrive_lanelet_mapping=od_mapping_path,
        sumocfg=sumocfg_path,
        output_net=output_net,
        output_sumocfg=output_sumocfg,
        report_path=report_path,
        min_coverage=0.0,
    )

    tl_logic = ET.parse(output_net).getroot().find("tlLogic")
    params = {param.get("key"): param.get("value") for param in tl_logic.findall("param")}
    assert params == {"linkSignalID:0": "od:466 od:467"}
    assert report["records_mapped_to_linksignal"] == 1

    generated_sumocfg = ET.parse(output_sumocfg).getroot()
    assert generated_sumocfg.find("input/net-file").get("value") == str(output_net)
    assert generated_sumocfg.find("input/route-files").get("value") == str(route_path.resolve())
    assert generated_sumocfg.find("input/step-length").get("value") == "0.1"
    assert (output_net.parent / "metadata.json").read_text(
        encoding="utf-8"
    ) == '{"map": "synthetic"}\n'


def test_generate_linksignal_params_prefers_actual_tls_id_when_present(tmp_path):
    module = load_generator_module()
    net_path, signal_mapping_path, od_mapping_path, sumocfg_path, _ = write_synthetic_inputs(
        tmp_path, tls_id="actual_tls"
    )
    mapping = json.loads(signal_mapping_path.read_text(encoding="utf-8"))
    mapping["lanelet_to_sumo"][0]["actual_sumo_tls_ids"] = ["actual_tls"]
    mapping["lanelet_to_sumo"][0]["planned_sumo_tls_id"] = "planned_tls"
    signal_mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    output_net = tmp_path / "out" / "tls_synced.net.xml"
    report = module.generate_tls_linksignal_params(
        sumo_net=net_path,
        signal_id_mapping=signal_mapping_path,
        opendrive_lanelet_mapping=od_mapping_path,
        sumocfg=sumocfg_path,
        output_net=output_net,
        output_sumocfg=tmp_path / "out" / "simulation_tls_synced.sumocfg",
        report_path=tmp_path / "out" / "report.json",
        min_coverage=0.0,
    )

    assert report["target_tls_source_counts"] == {"actual_sumo_tls_ids": 1}
    tl_logic = ET.parse(output_net).getroot().find("tlLogic[@id='actual_tls']")
    assert tl_logic.find("param[@key='linkSignalID:0']").get("value") == "od:466 od:467"


def test_generate_linksignal_params_rejects_low_coverage(tmp_path):
    module = load_generator_module()
    net_path, signal_mapping_path, od_mapping_path, sumocfg_path, _ = write_synthetic_inputs(
        tmp_path
    )
    output_dir = tmp_path / "out"
    report_path = output_dir / "report.json"

    with pytest.raises(RuntimeError, match="below required"):
        module.generate_tls_linksignal_params(
            sumo_net=net_path,
            signal_id_mapping=signal_mapping_path,
            opendrive_lanelet_mapping=od_mapping_path,
            sumocfg=sumocfg_path,
            output_net=output_dir / "tls_synced.net.xml",
            output_sumocfg=output_dir / "simulation_tls_synced.sumocfg",
            report_path=report_path,
            min_coverage=0.75,
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["coverage"] == 0.5


def run_generator(tmp_path, *, mapping=None, min_coverage=0.0):
    module = load_generator_module()
    net, ids, od, cfg, _ = write_synthetic_inputs(tmp_path)
    if mapping is not None:
        ids.write_text(json.dumps(mapping))
    args = dict(
        sumo_net=net,
        signal_id_mapping=ids,
        opendrive_lanelet_mapping=od,
        sumocfg=cfg,
        output_net=tmp_path / "out/net.xml",
        output_sumocfg=tmp_path / "out/config.sumocfg",
        report_path=tmp_path / "out/report.json",
        min_coverage=min_coverage,
    )
    return module, args


def v3_row(index=0, eligible=True, reg="reg_a"):
    return {
        "actual_sumo_tls_id": "tls_a",
        "linkIndex": index,
        "sync_eligible": eligible,
        "lanelet_regulatory_element_ids": [reg],
    }


def test_schema_three_uses_explicit_links_and_keeps_missing_signals_in_denominator(tmp_path):
    mapping = {
        "schema_version": 3,
        "lanelet_to_sumo": [{"resolution_status": "excluded_non_vehicle"}],
        "sumo_link_to_lanelet_signal": [v3_row(), v3_row(1, reg="missing"), v3_row(1, False)],
    }
    module, args = run_generator(tmp_path, mapping=mapping)
    report = module.generate_tls_linksignal_params(**args)
    assert report["coverage"] == 0.5
    assert report["coverage_denominator"] == 2
    assert report["params_written"] == 1
    assert report["source_record_status_counts"] == {"excluded_non_vehicle": 1}
    assert report["skipped_by_reason"] == {"signal_id_not_found": 1}


@pytest.mark.parametrize("index", [-1, 2, "bad"])
def test_invalid_explicit_index_does_not_write_generated_net(tmp_path, index):
    module, args = run_generator(
        tmp_path, mapping={"schema_version": 3, "sumo_link_to_lanelet_signal": [v3_row(index)]}
    )
    with pytest.raises(RuntimeError, match="Invalid TLS mapping"):
        module.generate_tls_linksignal_params(**args)
    assert not args["output_net"].exists()
    assert json.loads(args["report_path"].read_text())["invalid_link_index_count"] == 1


def test_empty_generation_is_failure_even_with_zero_required_coverage(tmp_path):
    module, args = run_generator(tmp_path, mapping={"lanelet_to_sumo": []})
    with pytest.raises(RuntimeError, match="params=0"):
        module.generate_tls_linksignal_params(**args)
    assert not args["output_net"].exists()


def test_input_overwrite_is_rejected(tmp_path):
    module, args = run_generator(tmp_path)
    before = args["sumo_net"].read_bytes()
    args["output_net"] = args["sumo_net"]
    with pytest.raises(ValueError, match="distinct"):
        module.generate_tls_linksignal_params(**args)
    assert args["sumo_net"].read_bytes() == before


def test_only_linksignal_metadata_changes_and_all_programs_are_updated(tmp_path):
    module, args = run_generator(tmp_path)
    root = ET.parse(args["sumo_net"]).getroot()
    other = ET.SubElement(root, "tlLogic", id="tls_a", programID="1", type="static", offset="5")
    ET.SubElement(other, "phase", duration="10", state="yr")
    ET.ElementTree(root).write(args["sumo_net"])
    before = args["sumo_net"].read_bytes()
    module.generate_tls_linksignal_params(**args)
    generated = ET.parse(args["output_net"]).getroot()
    assert all(
        t.find("param[@key='linkSignalID:0']") is not None for t in generated.findall("tlLogic")
    )
    for tree in (root, generated):
        for element in tree.iter():
            element.text = element.tail = None
        for logic in tree.findall("tlLogic"):
            module.remove_existing_linksignal_params(logic)
    assert ET.tostring(root) == ET.tostring(generated)
    assert args["sumo_net"].read_bytes() == before


def test_ambiguous_bindings_are_reported_without_choosing_a_winner(tmp_path):
    module, args = run_generator(
        tmp_path,
        mapping={"schema_version": 3, "sumo_link_to_lanelet_signal": [v3_row(0), v3_row(1)]},
    )
    report = module.generate_tls_linksignal_params(**args)
    assert set(report["ambiguous_signal_bindings"]) == {"od:466", "od:467"}
    assert report["params_written"] == 2
