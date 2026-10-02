"""Traffic-light reflection tests; no live CARLA connection is needed."""

import json
from types import SimpleNamespace

import pytest
from terasim_service.utils.carla import traffic_light_sync as tls


class FakeLight:
    def __init__(self, actor_id, od_id, xyz=(0, 0, 0), stale_readback=False):
        self.id = actor_id
        self.od_id = od_id
        self.xyz = xyz
        self.stale_readback = stale_readback
        self.state = "off"
        self.states = []

    def get_opendrive_id(self):
        return self.od_id

    def get_transform(self):
        return SimpleNamespace(
            location=SimpleNamespace(x=self.xyz[0], y=self.xyz[1], z=self.xyz[2])
        )

    def get_state(self):
        return "red" if self.stale_readback else self.state

    def set_state(self, state):
        self.states.append(state)
        self.state = state


def landmark(od_id, xyz, kind="1000001"):
    return SimpleNamespace(id=od_id, type=kind, transform=FakeLight(0, "", xyz).get_transform())


class FakeWorld:
    def __init__(self, actors, landmarks=()):
        self.actors = {a.id: a for a in actors}
        self.landmarks = landmarks
        self.lookups = []

    def get_map(self):
        return SimpleNamespace(get_all_landmarks=lambda: self.landmarks)

    def get_actor(self, actor_id):
        self.lookups.append(actor_id)
        return self.actors.get(actor_id)


class FreezableLight(FakeLight):
    frozen = False

    def freeze(self, value):
        self.frozen = value


class SynchronousWorld(FakeWorld):
    """Lists no actors until it has ticked, as a synchronous world does for a new client."""

    ticked = False

    def get_actors(self):
        lights = list(self.actors.values()) if self.ticked else []
        return SimpleNamespace(filter=lambda pattern: lights)


@pytest.fixture(autouse=True)
def fake_carla(monkeypatch):
    monkeypatch.setattr(
        tls,
        "carla",
        SimpleNamespace(
            TrafficLight=FakeLight,
            TrafficLightState=SimpleNamespace(Green="green", Yellow="yellow", Red="red", Off="off"),
        ),
    )


def detail(signal, parameters):
    return {
        "tls": signal,
        "information": json.dumps({"programs": {"0": {"parameters": parameters}}}),
    }


def test_actor_id_precedes_opendrive_and_lookup_is_cached():
    actor = FakeLight(84, "999")
    od_actor = FakeLight(900, "2000466")
    world = FakeWorld([actor, od_actor])
    sync = tls.TrafficLightSynchronizer(world, [actor, od_actor])
    assert sync.resolve("84") is actor
    assert sync.resolve("od:466") is od_actor
    assert sync.resolve("od:2000466") is od_actor
    assert sync.resolve("466") is od_actor
    sync.resolve("84")
    assert world.lookups == [84, 466]


def test_position_matching_is_one_to_one_typed_and_distance_limited():
    known = FakeLight(100, "2000000")
    low = FakeLight(10, "", (10, 0, 0))
    high = FakeLight(20, "", (10, 0, 0))
    nearby = FakeLight(30, None, (20, 0, 0))
    far = FakeLight(40, "", (30, 0, 0))
    actors = [known, high, far, low, nearby]
    marks = [
        landmark("2000000", (0, 0, 0)),
        landmark("2000425", (10, 0, 0)),
        landmark("2000423", (10, 0, 0)),
        landmark("2000423", (10, 0, 0)),
        landmark("2000385", (20.02, 0, 0)),
        landmark("500", (30.06, 0, 0)),
        landmark("999", (30, 0, 0), "205"),
    ]
    sync = tls.TrafficLightSynchronizer(FakeWorld(actors, marks), actors)
    assert sync.resolve("od:423") is low
    assert sync.resolve("od:425") is high
    assert sync.resolve("od:385") is nearby
    assert sync.resolve("od:999") is None
    assert sync.resolve("od:500") is None


def test_position_matching_can_be_disabled(monkeypatch):
    monkeypatch.setenv("CARLA_COSIM_TLS_POSITION_MATCH_MAX_DISTANCE", "0")
    actors = [FakeLight(1, "10"), FakeLight(2, "", (5, 0, 0))]
    sync = tls.TrafficLightSynchronizer(
        FakeWorld(actors, [landmark("10", (0, 0, 0)), landmark("20", (5, 0, 0))]), actors
    )
    assert sync.resolve("od:10") is actors[0]
    assert sync.resolve("od:20") is None


@pytest.mark.parametrize("signal,expected", [("Gg", "green"), ("Yy", "yellow"), ("Rr", "red")])
def test_sync_reflects_each_supported_case(signal, expected):
    actors = [FakeLight(1, "11"), FakeLight(2, "22")]
    sync = tls.TrafficLightSynchronizer(FakeWorld(actors), actors)
    sync.sync({"node": detail(signal, {"linkSignalID:0": "1", "linkSignalID:1": "od:22"})})
    assert [a.state for a in actors] == [expected, expected]


def test_legacy_last_writer_order_is_preserved():
    actor = FakeLight(1, "11")
    sync = tls.TrafficLightSynchronizer(FakeWorld([actor]), [actor])
    sync.sync(
        {
            "first": detail("Gr", {"linkSignalID:1": "od:11", "linkSignalID:0": "od:11"}),
            "second": detail("y", {"linkSignalID:0": "od:11"}),
        }
    )
    assert actor.states == ["green", "red", "yellow"]


def test_non_light_missing_and_invalid_tokens_warn_once(capsys):
    world = FakeWorld([])
    world.actors[9] = SimpleNamespace(id=9)
    sync = tls.TrafficLightSynchronizer(world, [])
    node = detail("g", {"linkSignalID:0": "9 999 od:404 invalid"})
    sync.sync({"node": node})
    first = capsys.readouterr().out
    sync.sync({"node": node})
    assert "Unresolved" in first
    assert capsys.readouterr().out == ""


def test_stale_readback_actor_still_receives_state_commands(capsys):
    actors = [FakeLight(1, "10"), FakeLight(2, "", (5, 0, 0), stale_readback=True)]
    world = FakeWorld(actors, [landmark("10", (0, 0, 0)), landmark("20", (5, 0, 0))])
    sync = tls.TrafficLightSynchronizer(world, actors)
    assert sync.resolve("od:20") is actors[1]
    capsys.readouterr()  # Discard the one-time position inference message.
    sync.sync({"n": detail("g", {"linkSignalID:0": "od:20"})})
    assert actors[1].state == "green"  # Server-side color is updated.
    assert actors[1].get_state() == "red"  # The client snapshot is still incorrect.
    assert actors[1].states == ["green"]
    assert capsys.readouterr().out == ""


def test_service_wrapper_accepts_inprocess_snapshot_and_off_is_noop():
    from terasim_service.utils.carla.cosim import CarlaCosim

    actor = FakeLight(1, "10")
    cosim = CarlaCosim.__new__(CarlaCosim)
    cosim.traffic_light_sync = tls.TrafficLightSynchronizer(FakeWorld([actor]), [actor])
    snapshot = {"traffic_light_details": {"n": detail("g", {"linkSignalID:0": "od:10"})}}
    cosim.sync_cosim_tls_to_carla(snapshot)
    assert actor.state == "green"
    cosim.traffic_light_sync = None
    snapshot["traffic_light_details"]["n"]["tls"] = "r"
    cosim.sync_cosim_tls_to_carla(snapshot)
    assert actor.state == "green"


@pytest.mark.parametrize("mode", ["master", "follow", "async"])
@pytest.mark.parametrize("enabled", [False, True])
def test_tick_modes_reflect_snapshot_without_changing_tick_ownership(mode, enabled):
    from terasim_service.utils.carla.cosim import CarlaCosim

    events = []
    actor = FakeLight(1, "10")
    prior = {"traffic_light_details": {"n": detail("g", {"linkSignalID:0": "od:10"})}}
    following = {"traffic_light_details": {"n": detail("r", {"linkSignalID:0": "od:10"})}}
    cosim = CarlaCosim.__new__(CarlaCosim)
    cosim.traffic_light_sync = tls.TrafficLightSynchronizer(FakeWorld([actor]), [actor])
    cosim.args = SimpleNamespace(skip_tls=not enabled, passive_tick=True)
    cosim.control_av = False
    cosim.step_length = 0.05
    cosim._inproc_tick_handle = None
    cosim._inproc_prev_state = prior
    cosim._next_tick_deadline = None
    cosim._vehicle_actor_index = {}
    cosim.sync_cosim_actor_to_carla = lambda state: None
    cosim._build_collision_removal_commands = lambda: []
    cosim._build_physics_feedback_commands = lambda: []
    cosim.inprocess_plugin = SimpleNamespace(
        tick_async=lambda commands: SimpleNamespace(
            result=lambda timeout: SimpleNamespace(status="ticked", state=following)
        )
    )
    cosim.world = SimpleNamespace(
        tick=lambda: events.append("tick") or 1,
        wait_for_tick=lambda: events.append("wait") or SimpleNamespace(frame=1),
        get_snapshot=lambda: SimpleNamespace(frame=1, timestamp=None),
    )
    cosim._tick_times_ms = []
    cosim._tick_time_hist = [0] * 11
    cosim._tick_veh_min = cosim._tick_veh_max = None
    for name in (
        "window_start",
        "prev_frame",
        "veh_min",
        "veh_max",
        "sumo_veh_min",
        "sumo_veh_max",
    ):
        setattr(cosim, "_async_" + name, None)
    cosim._async_sumo_steps_total = 0
    for name in ("sumo_ms", "write_ms", "work_ms"):
        setattr(cosim, "_async_" + name, [])
    assert getattr(cosim, "_tick_" + mode)()
    assert events == ({"master": ["tick"], "follow": ["wait"], "async": []}[mode])
    assert actor.state == ("off" if not enabled else "red" if mode == "async" else "green")


def test_offset_token_also_resolves_unprefixed_actor_opendrive_id():
    actor = FakeLight(1, "466")
    sync = tls.TrafficLightSynchronizer(FakeWorld([actor]), [actor])
    assert sync.resolve("od:2000466") is actor


def test_traffic_lights_missing_at_startup_are_picked_up_after_the_world_ticks(monkeypatch, capsys):
    from terasim_service.utils.carla import cosim as cosim_module
    from terasim_service.utils.carla.cosim import CarlaCosim

    monkeypatch.setattr(cosim_module, "carla", tls.carla)

    actor = FreezableLight(1, "10")
    world = SynchronousWorld([actor])
    cosim = CarlaCosim.__new__(CarlaCosim)
    cosim.world = world
    cosim.args = SimpleNamespace(skip_tls=False)
    cosim._initialize_traffic_lights()
    assert "No CARLA traffic lights visible yet" in capsys.readouterr().out
    snapshot = {"traffic_light_details": {"n": detail("g", {"linkSignalID:0": "od:10"})}}

    cosim.sync_cosim_tls_to_carla(snapshot)
    assert actor.states == []

    world.ticked = True
    cosim.sync_cosim_tls_to_carla(snapshot)
    assert actor.frozen
    assert actor.states == ["off", "green"]

    snapshot["traffic_light_details"]["n"]["tls"] = "r"
    cosim.sync_cosim_tls_to_carla(snapshot)
    assert actor.states == ["off", "green", "red"]
    assert capsys.readouterr().out.count("after startup") == 1
