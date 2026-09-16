# Physical co-simulation SUMO external-state build

This directory contains the pinned SUMO 1.23.1 patch used by the optional
Autoware x CARLA x TeraSim physical co-simulation mode.

The patch adds the low-level API used by the service's `setExternalState`
operation:

```python
traci.vehicle.moveToXYImmediate(
    vehicle_id,
    edge_id,
    lane_index,
    x,
    y,
    angle,
    keepRoute=1,
    matchThreshold=8.0,
    strictLaneHint=True,
)
```

Unlike normal `moveToXY`, this call updates pose immediately without advancing
SUMO time. It also completes the remote-control operation immediately, rebases
the lane-change model onto the realized CARLA lateral state, releases stale
TraCI speed overrides, and lets the in-process service apply CARLA measured
speed and acceleration with `setPreviousSpeed`.

The master cycle preserves the validated feature pipeline:

1. Resolve the SUMO step requested by the previous cycle and apply its Phase B
   state and Ackermann control to CARLA. The first cycle uses the initial state
   already published by the in-process plugin.
2. Advance CARLA frame N exactly once under the co-sim clock master.
3. Build the Autoware ego observation, when present, and the selected background
   vehicle feedback from completed CARLA frame N. Background feedback commands
   are emitted in actor-ID order, matching the feature implementation and
   keeping eager lane-change neighborhood updates deterministic.
4. Request one SUMO step. Phase A immediately assimilates the background
   feedback without changing SUMO time, TeraSim/NADE plans, and the existing
   priority-10 `simulationStep()` advances SUMO exactly 0.05 seconds before
   Phase B exports the desired state.
5. Keep that request pending and resolve it at the beginning of the next master
   cycle. The newly requested step is never awaited in the same cycle, and at
   most one SUMO step is in flight.

The Autoware ego is never selected for Ackermann ownership. It keeps the normal
CARLA-to-SUMO AV observation path.

Build from the repository root:

```bash
docker build \
  -t terasim-service:physics-cosim-20260817 \
  -f Dockerfile.sumo-external-state .
```

Enable the mode only with the dedicated image and master tick:

```bash
export CARLA_COSIM_VEHICLE_CONTROL_MODE=ackermann_physics
export CARLA_COSIM_ACKERMANN_FEEDBACK_MODE=apply
export CARLA_COSIM_ACKERMANN_FEEDBACK_ACTORS='*'
export CARLA_COSIM_ACKERMANN_FEEDBACK_ASSIMILATION_MODE=external_state
export CARLA_COSIM_ACKERMANN_FEEDBACK_EXTERNAL_STATE_STRICT_LANE_HINT=1
export CARLA_COSIM_STEP_LENGTH=0.05
export USE_LIBSUMO=0
python -m terasim_service.run_cosim --tick_mode master ...
```

The feature-tip control defaults are retained: feedback ack lag is 2 frames,
the consecutive ack failure limit is 3, restart enter/release speeds are
0.05/0.2 m/s, and the bounded restart target is at most 0.3 m/s. These remain
overridable with the existing `CARLA_COSIM_ACKERMANN_*` environment variables.

Diagnostics are opt-in and do not alter control behavior:

```bash
export CARLA_COSIM_ACKERMANN_CONTROL_LOG_RECORDS=1
export CARLA_COSIM_ACKERMANN_CONTROL_LOG_ACTORS='*'
export CARLA_COSIM_INITIALIZATION_DIAGNOSTICS_ENABLED=1
export CARLA_COSIM_INITIALIZATION_LOG=/app/outputs/carla_physics_initialization.jsonl
export CARLA_COSIM_COLLISION_SENSOR_ENABLED=1
export CARLA_COSIM_COLLISION_LOG=/app/outputs/carla_collision_events.jsonl
export CARLA_COSIM_COLLISION_SUMMARY=/app/outputs/carla_collision_summary.json
export CARLA_COSIM_SPAWN_MAX_ATTEMPTS=3
```

The dedicated physical launcher sets `USE_LIBSUMO=0`, matching the 3-way
workflow. This runs the patched SUMO engine as a TraCI child process instead
of loading libsumo into TeraSim's simulation thread. Both generated Python
APIs expose `moveToXYImmediate`; focused integration tests cover TraCI and
libsumo, and the physical 3-way run is validated through TraCI.

Acceptance runs on a dense urban map whose CARLA world and SUMO net describe
the same road geometry. The two-way physical run is exercised for 12,000
synchronized steps with TraCI and again with libsumo before adding the Autoware
ego for the 6,000-step three-way run. A map whose SUMO net only approximates
the CARLA geometry is not an acceptance map: the feedback path needs the two
descriptions to agree on lane centrelines.

An active SUMO lateral speed does not by itself prove a usable steering
direction: Phase A and Phase B can report the same world position for one
step. The service first uses the current measured lateral displacement, then a
world direction measured in the immediately preceding SUMO step. If neither
is usable, it keeps the action valid and drives toward the route-only
lookahead with warning `unresolved_phase_b_lateral_direction`. SUMO
lane-change intent is recorded only as a diagnostic conflict and never selects
the steering direction. This warning does not escalate to fail-closed, even if
it repeats.

`examples/scripts/check_physics_motion.py` requires a non-ego CARLA vehicle to
show both displacement and non-zero speed. The personal one-command launcher
uses it after the normal 3-way health check, so actor existence alone is not
reported as physical co-sim success.

All flags default to the legacy teleport behavior. Missing immediate-move API,
current-lane projection failure, frame mismatch, non-finite or missing route
geometry, SUMO assimilation failure, or excessive position error fails closed
instead of rematching another lane. An unresolved lateral direction alone is
the warning-level route-only case described above, not an invalid lookahead. In
strict-lane mode, SUMO never rematches an adjacent, predecessor, or unrelated
internal lane.
