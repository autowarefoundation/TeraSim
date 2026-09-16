# TeraSim Test Suite

Tests for the packages in this repository. None of them need CARLA to be
running; three integration tests need a SUMO build carrying the
`setExternalState` patch (see `apps/sumo_external_state/`) and skip with a
reason when that build is absent.

## Layout

```
tests/
├── conftest.py                 # shared fixtures, marker registration, env setup
├── test_service/               # in-process CARLA co-simulation link
│   └── test_physics_cosim.py   # physical (Ackermann) co-simulation, 33 tests
├── test_integration/           # SUMO-in-the-loop checks
│   └── test_sumo_external_state_focused.py
└── test_envgen/                # SUMO artifact generation
    └── test_generate_sumo_artifacts_from_net.py
```

## Running

```bash
# everything -- this is what CI runs
uv run --with carla==0.9.16 pytest tests

# one file, or one test by name
pytest tests/test_service/test_physics_cosim.py
pytest tests/test_service/test_physics_cosim.py -k ackermann

# skip the tests that need a SUMO installation
pytest tests -m "not requires_sumo"

# without the coverage report that pyproject.toml enables by default
pytest tests --no-cov
```

`carla` is imported by the co-simulation module under test but is not a package
dependency, so install it into the test environment -- CI does this with
`uv run --with carla==0.9.16`.

## Markers

| marker | meaning |
|---|---|
| `integration` | drives a real SUMO process |
| `requires_sumo` | needs a SUMO installation |
| `requires_gui` | needs a display (skipped automatically when `CI` is set) |
| `slow` | long-running |

## Fixtures

`conftest.py` sets `TERASIM_TEST_MODE`, `SUMO_HOME` and `TERASIM_LOG_LEVEL` for
every test (autouse), registers the markers above, and skips GUI tests in CI.
It also offers path and configuration fixtures (`project_root`, `temp_dir`,
`base_config` and friends) that the current tests do not use.
