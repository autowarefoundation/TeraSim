# CARLA OpenDRIVE and SUMO Maps Collection

This directory contains OpenDRIVE map files for the CARLA simulator and corresponding SUMO network files, intended for testing and developing an OpenDRIVE-to-SUMO converter.

## 📁 Directory structure

```
examples/xodr_sumo_maps/
├── xodr/                   # OpenDRIVE map files (converter input)
│   ├── Town01.xodr         # Basic city map (486KB)
│   ├── Town02.xodr         # Small town map (602KB)
│   ├── Town03.xodr         # Complex city map (2.2MB)
│   ├── Town04.xodr         # Highway map (2.0MB)
│   ├── Town05.xodr         # Urban map (2.8MB)
│   └── Town06.xodr         # Long highway map (2.0MB)
├── sumo/                   # SUMO networks converted from the maps above
│   ├── Town01.net.xml
│   ├── Town04.net.xml
│   └── Town05.net.xml
└── README.md               # This file
```

## 🗺️ Map details

### Town01 – Basic city map
- Type: urban roads
- Size: 410.68 x 344.26 m
- Features: basic test scenario with intersections and straight roads
- Use: initial algorithm testing, connectivity verification

### Town02 – Small town map
- Type: compact town
- Size: 205.59 x 204.48 m
- Features: compact layout, suitable for quick tests
- Use: algorithm validation, performance tests

### Town03 – Complex city map
- Type: complex urban environment
- Features: multi-lane roads, complex intersections, multiple road types
- Use: complex scenario testing, intersection handling validation

### Town04 – Highway map
- Type: highway
- Features: ramps, lane-count changes, high-speed scenarios
- Use: ramp merging tests, variable lane-width handling

### Town06 – Long highway map
- Type: long-distance highway
- Features: long highway stretches with multiple exits/entrances
- Use: long-range path planning, high-speed scenario testing

## 🚗 Inspecting the converted networks

The `sumo/` directory holds networks already converted from the maps above, so
they can be opened without running the converter first:

```bash
sumo-gui -n examples/xodr_sumo_maps/sumo/Town01.net.xml
```

## 🛠️ Converter testing

### Using the Python converter
```bash
# Test basic map
python scripts/xodr_to_sumo_converter.py -i examples/xodr_sumo_maps/xodr/Town01.xodr -o output_town01.net.xml

# Test highway map (includes ramps)
python scripts/xodr_to_sumo_converter.py -i examples/xodr_sumo_maps/xodr/Town04.xodr -o output_town04.net.xml

# Test complex city map
python scripts/xodr_to_sumo_converter.py -i examples/xodr_sumo_maps/xodr/Town03.xodr -o output_town03.net.xml
```

### Comparison with the official netconvert
```bash
# Use the official netconvert tool
netconvert --opendrive-files examples/xodr_sumo_maps/xodr/Town01.xodr -o official_town01.net.xml

# Compare results
sumo-gui -n output_town01.net.xml     # Python converter result
sumo-gui -n official_town01.net.xml   # Official converter result
```

## 📊 Testing recommendations

### Suggested test order
1. Town01 — verify basic functionality
2. Town02 — test compact scenarios
3. Town04 — verify ramps and highway behavior
4. Town03 — test complex intersections
5. Town06 — test long-distance scenarios

### Key verification points
- ✅ Geometry handling: straight lines, arcs, spirals
- ✅ Lane types: driving, entry, exit, onRamp, offRamp
- ✅ Connectivity: road connections, ramp merges/splits
- ✅ Intersections: handling of complex junctions
- ✅ Coordinate systems: UTM projection and geographic references

## 📚 References

- CARLA OpenDRIVE docs: https://carla.readthedocs.io/en/latest/adv_opendrive/
- CARLA-SUMO co-simulation: https://carla.readthedocs.io/en/latest/adv_sumo/
- OpenDRIVE standard: https://www.asam.net/standards/detail/opendrive/
- SUMO network format: https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html

## 🐛 Known issues

1. Town03 complexity: contains many complex geometries and may require optimized handling
2. Town04 ramps: ramp connection logic needs careful verification
3. Coordinate systems: ensure geographic reference is properly set

## 📈 Performance metrics

After conversion, evaluate using metrics such as:
- Number of edges vs original road count
- Number of nodes vs number of intersections
- Number of connections vs expected connections
- Total lane count consistency
- Geometric fidelity (arc smoothness)