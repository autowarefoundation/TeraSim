# Repository Guidelines

This is a reduced fork of mcity/TeraSim, reworked for three-way
co-simulation (Autoware × CARLA × TeraSim). See `CLAUDE.md` for the
full guide (layout, commands, architecture); it applies to all coding agents.

- `packages/`: `terasim` (core), `terasim-nde-nade` (adversarial algorithms),
  `terasim-service` (single-process CARLA co-simulation link), `terasim-vis` (FCD visualization).
- `tests/`: pytest suites (`pytest` from the repo root; config in `pyproject.toml`).
- Formatting: `black packages/` + `isort packages/`; lint with `ruff check packages/`.
- Co-simulation entry point: `python -m terasim_service.run_cosim --config <scenario yaml> --carla_port <port>`.
