#!/bin/bash
# =============================================================================
# Three-way co-simulation (Autoware x CARLA x TeraSim), single-process variant.
#
# TeraSim injects its background traffic into a CARLA server that is already
# running and already owns the ego vehicle, so this container never spawns or
# controls the ego. Called as the container entrypoint from
# docker-compose.cosim-inprocess.yml.
#
# The TeraSim loop and the CARLA client live in one process
# (terasim_service.run_cosim): states and commands are passed as Python objects
# instead of crossing a transport boundary.
#
# Prerequisite: a CARLA server with the matching map, plus whatever drives the
# ego vehicle (for example Autoware through a CARLA bridge), is already up.
# =============================================================================
set -u
cd /app

CARLA_HOST="${CARLA_HOST:-127.0.0.1}"
CARLA_PORT="${CARLA_PORT:-2000}"
SCENARIO="${SCENARIO:-/app/examples/scenarios/cosim_town01_dt005.yaml}"
# Tick mode: follow = the ego-side bridge calls world.tick() and TeraSim follows;
#            master = TeraSim owns the clock and ticks CARLA on a fixed wall-clock
#                     period (the bridge must then run in follower mode).
TICK_MODE="${COSIM_TICK_MODE:-follow}"
# CARLA client RPC timeout in seconds. A freshly started server can block inside
# world.tick() while it warms up its rendering pipeline; raise this (e.g. 3600)
# if the first tick exceeds the default.
CARLA_TIMEOUT="${CARLA_TIMEOUT:-600}"

echo "=========================================="
echo " TeraSim 3-cosim (in-process)"
echo "  CARLA :${CARLA_PORT}"
echo "  scenario: ${SCENARIO}"
echo "  tick_mode: ${TICK_MODE}"
echo "=========================================="

# -- Step 1: remove every non-ego vehicle from CARLA (makes the run idempotent) --
#   Background vehicles left behind by a previous run outlive this container,
#   because the CARLA server does. They make the next injection fail with
#   "collision at spawn position". The ego (ego_vehicle/hero) is preserved.
echo "[1/2] cleaning up non-ego vehicles on CARLA :${CARLA_PORT}"
python - <<PY
import carla, time
c = carla.Client("${CARLA_HOST}", ${CARLA_PORT}); c.set_timeout(60.0)
w = c.get_world()
vs = []
for i in range(12):
    vs = list(w.get_actors().filter("*vehicle*"))
    if len(vs) > 0:
        break
    time.sleep(0.5)
keep = ("ego_vehicle", "hero")
victims = [v for v in vs if v.attributes.get("role_name") not in keep]
print("      CARLA total=%d ego=%d destroy=%d" % (len(vs), len(vs) - len(victims), len(victims)))
if victims:
    c.apply_batch([carla.command.DestroyActor(v.id) for v in victims])
    time.sleep(2)
    print("      after cleanup=%d" % len(list(w.get_actors().filter("*vehicle*"))))
PY

# -- Step 2: the single-process runner (TeraSim + CARLA client) --
#   exec replaces bash so that SIGTERM from `docker stop` reaches python directly;
#   the runner shuts TeraSim down and clears its vehicles from CARLA on the way out.
echo "[2/2] run terasim_service.run_cosim (single process)"
echo "------------------------------------------"
exec python -m terasim_service.run_cosim \
  --config "${SCENARIO}" \
  --carla_host "${CARLA_HOST}" \
  --carla_port "${CARLA_PORT}" \
  --carla_timeout "${CARLA_TIMEOUT}" \
  --tick_mode "${TICK_MODE}"
