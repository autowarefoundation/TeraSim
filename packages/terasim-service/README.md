# terasim-service

The co-simulation link between TeraSim and CARLA.

TeraSim's SUMO background traffic is mirrored into a running CARLA server, and
an externally driven ego vehicle — Autoware through `autoware_carla_interface`,
for example — is fed back into SUMO as the "AV", so the background traffic
reacts to it.

The TeraSim simulation loop and the CARLA client run as two threads of a single
process and exchange states and commands as plain Python objects, so a run is a
single command against a CARLA server.

## Layout

| module | what it does |
|---|---|
| `run_cosim.py` | entry point: starts the TeraSim thread and the CARLA loop |
| `plugins/cosim_inprocess.py` | the rendezvous between those two threads |
| `utils/carla/cosim.py` | the CARLA side: traffic mirroring, ego feedback, tick modes, physical control |
| `utils/carla/ackermann_control.py` | the control law for physics-driven vehicles |
| `utils/sumo_lane_geometry.py` | maps measured CARLA poses back onto SUMO lanes |
| `utils/messages/` | the state and command structures that cross the link |

## Running

```bash
python -m terasim_service.run_cosim --config examples/scenarios/cosim_town01_dt005.yaml
```

A CARLA server must already be running with the matching map. Everything
map-specific lives in the scenario YAML; `--help` lists the remaining options.

### Tick modes (`--tick_mode`)

| mode | who owns the clock |
|---|---|
| `follow` (default) | the ego-side bridge ticks CARLA; this process follows it |
| `master` | this process ticks CARLA on a fixed cadence, and the bridge must run with `tick_follower:=true` |
| `async` | nobody ticks: CARLA free-runs and SUMO is paced against the wall clock |

### Physics-based background vehicles (optional)

By default background vehicles are teleported to the positions SUMO computes.
They can instead be driven by Ackermann control in CARLA, with their measured
pose written back into SUMO:

```bash
export CARLA_COSIM_VEHICLE_CONTROL_MODE=ackermann_physics
export CARLA_COSIM_ACKERMANN_FEEDBACK_MODE=apply
export CARLA_COSIM_ACKERMANN_FEEDBACK_ACTORS='*'          # or a comma-separated list
export CARLA_COSIM_ACKERMANN_FEEDBACK_ASSIMILATION_MODE=external_state
```

All four are required: the feedback path stays off if the actor list is empty.
`external_state` additionally needs a SUMO build carrying the patch in
[`apps/sumo_external_state/`](../../apps/sumo_external_state/). The remaining
knobs are documented in
[`docs/carla_ackermann_feedback.md`](../../docs/carla_ackermann_feedback.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
