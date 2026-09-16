<div align="center">
<p align="center">

<img src="docs/figure/logo.png" height="100px">

</p>
</div>

<p align="center">
<strong>Naturalistic and adversarial traffic simulation for discovering unknown unsafe events</strong>
</p>

---

## Overview

TeraSim is an open-source platform for automated autonomous-vehicle simulation.
Its objective is to **efficiently uncover real-world unknown unsafe events** by
automatically creating diverse and statistically realistic traffic environments:

- **Naturalistic driving environment (NDE)** — background traffic derived from
  large-scale naturalistic driving data, with statistical realism.
- **Adversarial scenario synthesis (NADE)** — rare, high-risk interactions such
  as aggressive cut-ins and unexpected crossings, injected into that traffic to
  reach the failures that matter for safety validation.
- Built on [SUMO](https://www.eclipse.org/sumo/), and able to drive third-party
  simulators such as [CARLA](https://carla.org/) and
  [Autoware](https://github.com/autowarefoundation/autoware).

## About this fork

This is a reduced fork of [mcity/TeraSim](https://github.com/mcity/TeraSim),
reworked for **three-way co-simulation: Autoware × CARLA × TeraSim**. Autoware
drives the ego vehicle through
[`autoware_carla_interface`](https://github.com/autowarefoundation/autoware_universe/tree/main/simulator/autoware_carla_interface),
and TeraSim supplies the background traffic around it.

- **Single-process CARLA link** (`terasim-service`): the TeraSim loop and the
  CARLA client run as two threads of one process, exchanging states and commands
  as plain Python objects. TeraSim traffic is mirrored into a running CARLA
  server; the externally driven ego is fed back into SUMO so the background
  traffic reacts to it. TeraSim can follow the ego side's clock or own it.
- **Optional physics-based background vehicles**: selected CARLA vehicles are
  driven by Ackermann control, and their measured pose is written back into
  SUMO.

Detailed documentation of the co-simulation setup will follow.

## Running

Everything map-specific (SUMO net, routes, adversity setup, run time) lives in a
scenario YAML under `examples/scenarios/`, which you pass to the runner. Three
maps ship with the repository: **Town01**, **Kashiwanoha** and **Mcity**.

```bash
# Install (Python 3.10-3.12, gcc/g++ for the Cython extensions)
conda create -n terasim python=3.10 -y && conda activate terasim
./setup_environment.sh

# Standalone NADE run, on SUMO alone
python scripts/run_experiments_debug.py --config examples/scenarios/Mcity_safety_assessment.yaml

# Co-simulation against a running CARLA server
python -m terasim_service.run_cosim --config examples/scenarios/cosim_town01_dt005.yaml
```

For the full three-way setup, build the image and bring up the compose file,
which runs `examples/scripts/run_3cosim_inprocess.sh` against a CARLA server
that an Autoware bridge is already attached to:

```bash
docker build -f Dockerfile.cosim -t terasim-service:inprocess .
SCENARIO=/app/examples/scenarios/cosim_town01_dt005.yaml \
    docker compose -f docker-compose.cosim-inprocess.yml up
```

Physics-based background vehicles additionally need the patched SUMO build from
`Dockerfile.sumo-external-state`, which layers onto the image above.

## **📄 License**

- **TeraSim Core and other packages**: Apache 2.0 License
- **Visualization Tools**: MIT License

This project includes modified code from [SumoNetVis](https://github.com/patmalcolm91/SumoNetVis) licensed under the MIT License.
