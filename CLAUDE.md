# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TeraSim is an open-source traffic simulation platform designed for naturalistic and adversarial testing of autonomous vehicles (AVs), built upon SUMO (Simulation of Urban MObility).

This is a **reduced fork of mcity/TeraSim**, reworked for three-way co-simulation (Autoware × CARLA × TeraSim): the core simulation engine, the NDE-NADE algorithms, a single-process CARLA co-simulation link (`terasim-service`), and FCD visualization tools. The upstream extras (TeraSim-World/Cosmos, environment generation, dataset tooling, the Redis/FastAPI service) were removed from this fork.

## Common Development Commands

### Environment Setup
```bash
# Native (conda)
./setup_environment.sh

# Docker (co-simulation image)
docker build -f Dockerfile.cosim -t terasim-service:inprocess .
```

### Testing
```bash
# Run all tests with coverage (config in the root pyproject.toml)
pytest

# Run a specific test file
pytest tests/test_service/test_physics_cosim.py
```

### Code Quality
```bash
black packages/
isort packages/
ruff check packages/
```

### Running Simulations

The runners are generic; everything map-specific lives in the scenario YAML.
Bundled ready-to-run scenarios: `cosim_town01.yaml` and `cosim_kashiwanoha_dt005.yaml`
(co-sim + standalone) and `Mcity_safety_assessment.yaml` (standalone NADE only —
CARLA has no Mcity map).

```bash
# CARLA co-simulation, single process (a CARLA server must be running)
python -m terasim_service.run_cosim --config examples/scenarios/cosim_town01.yaml

# Standalone NADE run (no CARLA), GUI controlled by the scenario yaml
python scripts/run_experiments_debug.py --config examples/scenarios/Mcity_safety_assessment.yaml

# FCD trajectory visualization
python scripts/visualize_fcd.py configs/visulation/example.yaml
```

## Architecture Overview

### Monorepo Structure
```
TeraSim/
├── packages/               # Python packages (uv workspace members)
│   ├── terasim/           # Core simulation engine
│   ├── terasim-nde-nade/  # NDE-NADE algorithms for naturalistic/adversarial environments
│   ├── terasim-service/   # Single-process CARLA co-simulation link
│   └── terasim-vis/       # Visualization tools (FCD plots/videos)
├── examples/              # Scenario YAMLs, SUMO maps, co-sim launcher script
├── configs/visulation/    # visualize_fcd.py config example
├── scripts/               # Runners and tooling
├── docs/                  # OpenDRIVE / SUMO plain-XML format notes
└── tests/                 # Test suites (core, NDE-NADE)
```

### Core Components

**Simulator (`packages/terasim/terasim/simulator.py`)**: Central orchestrator managing SUMO integration, synchronization, and simulation lifecycle. Key methods:
- `__init__()`: Configure SUMO, GUI, output paths, traffic scale
- `bind_env()`: Attach environment to simulator
- `start()`: Initialize SUMO and agents
- `run()`: Execute simulation steps
- `close()`: Clean up resources

**Environment System (`packages/terasim/terasim/envs/`)**: Testing environment abstractions:
- `BaseEnv`: Abstract base with lifecycle hooks (`on_start`, `on_step`, `on_stop`)
- `EnvTemplate`: Standard testing scenario implementation
- NADE environments (`terasim-nde-nade`) control adversarial background traffic

**Agent Architecture**: Modular sensor-decision-controller design:
- **Agent** (`agent/agent.py`): Base class for all entities
- **Vehicle** (`vehicle/vehicle.py`): Vehicle-specific agent
- **Sensors** (`vehicle/sensors/`): EgoSensor, LocalSensor for perception
- **Decision Models** (`vehicle/decision_models/`): IDMModel, SUMOModel for behavior
- **Controllers** (`vehicle/controllers/`): HighEfficiencyController, SUMOMoveController for actuation
- **Factories** (`vehicle/factories/`): VehicleFactory for creating configured vehicles

**Pipeline System (`terasim/pipeline.py`)**: Ordered execution framework with priority-based scheduling for simulation steps.

### Co-simulation Link

**TeraSim Service (`packages/terasim-service/`)**: single-process CARLA co-simulation link:
- The TeraSim simulation loop (sim thread) and the CARLA-facing client (main thread) run in ONE
  process (`terasim_service.run_cosim`), rendezvousing through `TeraSimCoSimInProcessPlugin`
  (two threading.Events; states/commands passed as plain Python objects, no serialization)
- `CarlaCosim` (`utils/carla/cosim.py`) mirrors SUMO background traffic into CARLA and feeds an
  externally driven ego (role_name `ego_vehicle`) back into SUMO as the "AV"
- Three tick modes (`--tick_mode`): `follow` (default) — the ego-side bridge owns `world.tick()`
  and this process only follows via `world.wait_for_tick()`; `master` — this
  process is the clock master, ticking CARLA on a fixed `step_length` cadence and running
  one SUMO step serially inside each cycle (the bridge must run with `tick_follower:=true`);
  `async` — nobody ticks: this process clears `synchronous_mode` so CARLA
  free-runs with a variable delta, and paces SUMO on a `step_length` wall-clock timer
  with catch-up for late cycles (the bridge must run with `tick_follower:=true` and must
  be started BEFORE this process — the last settings write wins)
- The former transports (Redis lists + FastAPI service, then gRPC) were removed

### Key Design Patterns

1. **Factory Pattern**: Vehicle and agent creation with customizable components
2. **Observer Pattern**: Sensors observe environment, decision models react
3. **Strategy Pattern**: Interchangeable decision models and controllers
4. **Pipeline Pattern**: Ordered, prioritized execution of simulation steps

## Development Notes

### SUMO Integration
- SUMO 1.23.1 required (installed automatically)
- TraCI for real-time control, libsumo for performance
- Supports GUI (`sumo-gui`) and headless (`sumo`) modes
- Network files: `.net.xml` (topology), `.rou.xml` (routes), `.sumocfg` (config)

### Agent Lifecycle
1. **Creation**: Factory creates agent with configured components
2. **Registration**: Agent registers with SUMO via simulator
3. **Execution Loop**: `sense()` → `decide()` → `control()` each step
4. **Cleanup**: Proper removal from SUMO and simulator

### Testing Strategy
- Unit tests per package in `tests/test_*/`
- Markers: `@pytest.mark.slow`, `@pytest.mark.requires_sumo`

### Package Dependencies
- **Core**: eclipse-sumo==1.23.1, numpy==1.26.4, scipy, attrs, bidict
- **Service**: pydantic, pyyaml, loguru, utm (plus the CARLA client wheel at runtime)
- **NDE-NADE**: Cython extensions for performance
- **Visualization**: matplotlib, plotly, folium

## Common Development Tasks

### Adding New Vehicle Behavior
1. Create decision model in `packages/terasim/terasim/vehicle/decision_models/`
2. Inherit from `AgentDecisionModel`, implement `decide()` method
3. Create custom `VehicleFactory` with new model
4. Test with scenario in `examples/scenarios/`

### Creating Custom Environment
1. Extend `BaseEnv` in `packages/terasim/terasim/envs/`
2. Implement `on_start()`, `on_step()`, `on_stop()` methods
3. Define termination conditions and metrics
4. Bind to simulator with `sim.bind_env(env)`

### Adding Adversarial Behavior
1. Create adversity in `packages/terasim-nde-nade/terasim_nde_nade/adversity/`
2. Define trigger conditions and behavior modifications
3. Register in the scenario yaml (`adversity_cfg`, see `examples/scenarios/cosim_town01.yaml`)
4. Test with NADE environment

### Running with CARLA Co-simulation
1. Start CARLA server
2. Run `python -m terasim_service.run_cosim --config <scenario yaml> --carla_port <port>`
   (single process: TeraSimCoSimInProcessPlugin + CarlaCosim client)
