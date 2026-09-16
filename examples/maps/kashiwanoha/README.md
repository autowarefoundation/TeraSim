# Kashiwanoha campus map (SUMO net for co-simulation)

SUMO network of the Kashiwanoha campus area, for Autoware x CARLA x TeraSim
co-simulation. Converted directly from the public Kashiwanoha lanelet2 map:

- Source: https://github.com/tier4/scenario_simulator_v2
  `map/kashiwanoha_map/map/private_road_and_walkway_ele_fix/lanelet2_map.osm`
  (commit 6a805e47), License: **Apache-2.0** (TIER IV, Inc.)
- Converter: https://github.com/scien-inc/lanelet2_to_sumo (`ll2sumo`, commit 59d111a)
  with SUMO/netconvert 1.26.0. This file is a derivative work of the Apache-2.0
  licensed map above.
- Coordinate frame: `<location>` holds raw UTM zone 54N coordinates with
  `netOffset="0,0"` (the converter runs netconvert with
  `--offset.disable-normalization`), so the CARLA co-simulation layer can
  transform SUMO -> UTM -> xodr-local against the matching OpenDRIVE
  `geoReference`.

Files:

| file | purpose |
|---|---|
| kashiwanoha.net.xml | SUMO network (58 edges, 36 junctions, 1 traffic light) |
| kashiwanoha.trips.xml | random background traffic (seed 42, period 12s; vType inlined) |
| kashiwanoha_empty.trips.xml | vType only, no background traffic |
| kashiwanoha_dt005.sumocfg / _notraffic.sumocfg | step 0.05s configs |
| vtypes.add.xml | NDE_URBAN vehicle type (referenced by trips generation) |

## Regenerating

```bash
# 1) net (from the lanelet2 map, in the lanelet2_to_sumo checkout)
docker build -t ll2sumo:latest .
docker run --rm -v "$PWD/map:/data/input:ro" -v "$PWD/out:/data/out" ll2sumo:latest \
  --input /data/input/lanelet2_map.osm --out-dir /data/out/kashiwanoha \
  --lane-change-mode unrestricted

# 2) background traffic (safe weights come from the converter)
python3 "$SUMO_HOME/tools/randomTrips.py" -n network.net.xml -a vtypes.add.xml \
  -o kashiwanoha.trips.xml --weights-prefix randomtrips.safe \
  --trip-attributes 'type="NDE_URBAN"' --validate -p 12 --seed 42
# SUMO 1.26 duarouter no longer copies the vType into the trip output, so the
# <vType> element from vtypes.add.xml is inlined into kashiwanoha.trips.xml by hand.
```

The converter also emits `signal_id_mapping.json` (lanelet2 traffic-light regulatory
element <-> SUMO tlLogic link mapping) and `randomtrips.safe.*` weight files. They are
not committed here (`*.json` is gitignored in this repo); regenerate them with the
conversion command above when the co-simulation layer needs them.

Validated against the lanelet2 source: all 81 road lanelets exported, 0 reversed
lanes, all 103 lanelet2 successor pairs reproduced, lane centre lines within
0.52 m (median 0.001 m) of the lanelet2 centre lines.

Known limitations: lane widths are not exported, so SUMO uses its 3.2 m default
(the map's real lanes are 2.86-3.75 m, median 3.08 m); crosswalk and walkway
lanelets are not exported (4 + 1 ignored); the single traffic light gets an
inferred Japanese-style static program, not a real signal plan.
