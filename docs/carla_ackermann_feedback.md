# CARLA Ackermann feedback co-simulation

This mode keeps selected CARLA vehicles under physics-based Ackermann control and writes their
observed pose and speed back to TeraSim before the next SUMO step.

## Enable the loop

Use synchronous CARLA mode and set:

```bash
export CARLA_COSIM_VEHICLE_CONTROL_MODE=ackermann_physics
export CARLA_COSIM_ACKERMANN_FEEDBACK_MODE=apply
export CARLA_COSIM_ACKERMANN_FEEDBACK_ACTORS=AV
```

The position mode is `moveTo`, which prioritizes longitudinal time consistency:

```bash
export CARLA_COSIM_ACKERMANN_FEEDBACK_POSITION_MODE=moveTo
```

`*` selects all background vehicles, but feedback should first be validated with `AV` only.
`shadow` may be used instead of `apply` to observe and log feedback without changing SUMO state.

The normal HTTP/Redis path queues one ordered command batch before requesting the next TeraSim
tick. For the direct gRPC path, pass `--direct_addr HOST:PORT`; the same ordered `set_state`
commands are carried in the next `Tick` request.

## Position synchronization

Before each SUMO simulation step, `moveTo` projects the CARLA front-bumper position onto the
vehicle current SUMO lane, calls `vehicle.moveTo`, and then calls `setPreviousSpeed`. Unlike
`moveToXY`, `moveTo` takes effect immediately, so SUMO computes the next speed from a current CARLA
observation.

CARLA lateral offset does not select a different SUMO lane. SUMO owns lane-change decisions and the
feedback updates longitudinal progress on the lane that SUMO selected in its preceding step. This
avoids an instantaneous centerline-to-centerline jump caused by choosing an adjacent lane from
CARLA x/y.

If the current-lane projection exceeds the configured distance or heading limits, feedback fails
closed: TeraSim refuses the SUMO step, returns an error to the CARLA client, and the client applies
a zero-speed emergency-deceleration Ackermann command before stopping.

## Longitudinal diagnostics

Set the following to emit one JSON record per controlled frame:

```bash
export CARLA_COSIM_ACKERMANN_CONTROL_LOG_RECORDS=1
```

Each `AckermannControlTrace` contains SUMO desired speed and acceleration, the accepted CARLA
feedback speed, unclamped requested acceleration, SUMO vehicle-type `emergencyDecel`, the applied
Ackermann speed/acceleration targets, and measured CARLA speed and longitudinal acceleration.
CARLA braking uses the vehicle-specific SUMO `emergencyDecel` when it is available; the configured
Ackermann maximum deceleration is the fallback.

When CARLA remains in its standstill range but SUMO requests a positive next speed and positive
acceleration, the controller accumulates a small restart speed target from SUMO's requested
acceleration. The same logic applies to every selected feedback actor, including background
vehicles selected by `*`. The assist is cancelled immediately when SUMO requests zero/negative
acceleration or a zero next speed, and normal control resumes after CARLA reaches the release speed.

The restart thresholds can be tuned with:

```bash
export CARLA_COSIM_ACKERMANN_RESTART_ENTER_SPEED=0.05
export CARLA_COSIM_ACKERMANN_RESTART_RELEASE_SPEED=0.2
export CARLA_COSIM_ACKERMANN_RESTART_SPEED_EPSILON=0.001
export CARLA_COSIM_ACKERMANN_RESTART_MAX_TARGET_SPEED=0.3
```

`AckermannControlTrace` records `restart_active` and `restart_target_speed` while the assist is
active.

## Constraints

- CARLA must be synchronous.
- The CARLA client must own `world.tick()`; `passive_tick` is unsupported.
- `control_av` cannot be combined with feedback ownership of the AV.
- Feedback mode requires `ackermann_physics`.
- `moveTo` does not apply CARLA lateral offset or yaw inside SUMO.
- Validate AV-only feedback before expanding to background vehicles.

Each feedback command contains the CARLA frame number. TeraSim exposes the accepted observed
speed and frame in the next simulation state, allowing the controller to detect missing or stale
feedback and brake safely.
