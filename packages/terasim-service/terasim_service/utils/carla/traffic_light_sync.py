"""SUMO-to-CARLA traffic-light synchronization.

Static OpenDRIVE lookup is built once; legacy actor-ID parameters stay supported.
The caller owns CARLA ticking and the existing initial Off/freeze operation.
"""

import json
import math
import os

import carla

SUMO_CARLA_TLS_LINK_PREFIX = "linkSignalID:"
OPENDRIVE_TLS_LINK_PREFIX = "od:"
OPENDRIVE_SIGNAL_ID_OFFSET = 2_000_000
DEFAULT_TLS_POSITION_MATCH_MAX_DISTANCE = 0.05


class TrafficLightSynchronizer:
    def __init__(self, world, traffic_lights):
        self.world = world
        self._warnings = set()
        self._resolved_tokens = {}
        self._bindings = {}
        try:
            distance = float(os.getenv("CARLA_COSIM_TLS_POSITION_MATCH_MAX_DISTANCE", "0.05"))
            if not math.isfinite(distance):
                raise ValueError("distance must be finite")
        except ValueError:
            distance = DEFAULT_TLS_POSITION_MATCH_MAX_DISTANCE
            self._warn_once("distance", "Invalid position-match distance; using 0.05 m")
        try:
            landmarks = world.get_map().get_all_landmarks()
        except Exception as exc:
            landmarks = []
            self._warn_once("landmarks", f"Could not read OpenDRIVE landmarks: {exc}")
        self.opendrive_traffic_lights = self._build_opendrive_traffic_light_map(
            traffic_lights,
            landmarks,
            max(0.0, distance),
        )

    def _warn_once(self, key, message):
        if key not in self._warnings:
            self._warnings.add(key)
            print(f"[TLS] Warning: {message}", flush=True)

    @staticmethod
    def _opendrive_traffic_light_id_aliases(opendrive_id):
        opendrive_id = str(opendrive_id).strip()
        if not opendrive_id:
            return []

        aliases = [opendrive_id]
        if opendrive_id.isdigit():
            numeric_id = int(opendrive_id)
            if numeric_id >= OPENDRIVE_SIGNAL_ID_OFFSET:
                aliases.append(str(numeric_id - OPENDRIVE_SIGNAL_ID_OFFSET))
        return aliases

    @staticmethod
    def _location_xyz(item):
        get_transform = getattr(item, "get_transform", None)
        transform = get_transform() if callable(get_transform) else getattr(item, "transform", None)
        location = getattr(transform, "location", None)
        if location is None:
            return None
        try:
            return float(location.x), float(location.y), float(location.z)
        except (AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def _sortable_identifier(value):
        value = str(value)
        if value.isdigit():
            return 0, int(value)
        return 1, value

    @classmethod
    def _infer_opendrive_traffic_light_ids_by_position(  # noqa: C901
        cls,
        traffic_lights,
        landmarks,
        mapped_traffic_lights,
        max_distance,
    ):
        """Match empty-ID actors to otherwise-unclaimed traffic-light landmarks.

        CARLA can create a traffic-light actor for an OpenDRIVE signal while returning an
        empty string from get_opendrive_id(). Landmarks still retain the signal ID and
        transform, so match those actors one-to-one by position. Actor ID is the deterministic
        tie-breaker when multiple signals occupy exactly the same position.
        """
        if not landmarks or max_distance <= 0.0:
            return []

        unmapped_actors = []
        for traffic_light in traffic_lights:
            get_opendrive_id = getattr(traffic_light, "get_opendrive_id", None)
            try:
                opendrive_id = get_opendrive_id() if callable(get_opendrive_id) else None
            except Exception:
                opendrive_id = None
            if opendrive_id is None or not str(opendrive_id).strip():
                if cls._location_xyz(traffic_light) is not None:
                    unmapped_actors.append(traffic_light)
        if not unmapped_actors:
            return []

        # Learn the traffic-light landmark type from IDs CARLA resolved successfully. This
        # prevents static signs and other OpenDRIVE landmarks from becoming candidates.
        traffic_light_landmark_types = set()
        for landmark in landmarks:
            landmark_id = str(getattr(landmark, "id", "")).strip()
            if not landmark_id:
                continue
            aliases = cls._opendrive_traffic_light_id_aliases(landmark_id)
            if any(alias in mapped_traffic_lights for alias in aliases):
                traffic_light_landmark_types.add(str(getattr(landmark, "type", "")))
        if not traffic_light_landmark_types:
            return []

        candidate_locations = {}
        for landmark in landmarks:
            landmark_id = str(getattr(landmark, "id", "")).strip()
            if not landmark_id:
                continue
            if str(getattr(landmark, "type", "")) not in traffic_light_landmark_types:
                continue
            aliases = cls._opendrive_traffic_light_id_aliases(landmark_id)
            if any(alias in mapped_traffic_lights for alias in aliases):
                continue
            location = cls._location_xyz(landmark)
            if location is not None:
                candidate_locations.setdefault(landmark_id, set()).add(location)

        candidate_pairs = []
        for traffic_light in unmapped_actors:
            actor_location = cls._location_xyz(traffic_light)
            actor_id = getattr(traffic_light, "id", "")
            for landmark_id, locations in candidate_locations.items():
                distance = min(
                    math.dist(actor_location, landmark_location) for landmark_location in locations
                )
                if distance <= max_distance:
                    candidate_pairs.append(
                        (
                            distance,
                            cls._sortable_identifier(actor_id),
                            cls._sortable_identifier(landmark_id),
                            traffic_light,
                            landmark_id,
                        )
                    )

        inferred = []
        matched_actor_ids = set()
        matched_landmark_ids = set()
        for distance, _, _, traffic_light, landmark_id in sorted(candidate_pairs):
            actor_id = getattr(traffic_light, "id", "")
            if actor_id in matched_actor_ids or landmark_id in matched_landmark_ids:
                continue
            inferred.append((traffic_light, landmark_id, distance))
            matched_actor_ids.add(actor_id)
            matched_landmark_ids.add(landmark_id)
        return inferred

    @classmethod
    def _build_opendrive_traffic_light_map(
        cls,
        traffic_lights,
        landmarks=None,
        position_match_max_distance=DEFAULT_TLS_POSITION_MATCH_MAX_DISTANCE,
    ):
        opendrive_traffic_lights = {}
        for traffic_light in traffic_lights:
            get_opendrive_id = getattr(traffic_light, "get_opendrive_id", None)
            if not callable(get_opendrive_id):
                continue

            try:
                opendrive_id = get_opendrive_id()
            except Exception as e:
                print(
                    f"Warning: Could not read OpenDRIVE id for CARLA traffic light "
                    f"{getattr(traffic_light, 'id', '<unknown>')}: {e}",
                    flush=True,
                )
                continue

            if opendrive_id is None or str(opendrive_id) == "":
                continue
            for opendrive_id_alias in cls._opendrive_traffic_light_id_aliases(opendrive_id):
                opendrive_traffic_lights[opendrive_id_alias] = traffic_light

        inferred = cls._infer_opendrive_traffic_light_ids_by_position(
            traffic_lights,
            landmarks or [],
            opendrive_traffic_lights,
            position_match_max_distance,
        )
        for traffic_light, opendrive_id, distance in inferred:
            for opendrive_id_alias in cls._opendrive_traffic_light_id_aliases(opendrive_id):
                opendrive_traffic_lights[opendrive_id_alias] = traffic_light
            actor_id = getattr(traffic_light, "id", "<unknown>")
            print(
                f"Inferred OpenDRIVE signal id {opendrive_id} for CARLA traffic light "
                f"actor {actor_id} from position (distance={distance:.4f}m).",
                flush=True,
            )

        if inferred:
            print(
                f"Inferred {len(inferred)} CARLA traffic-light OpenDRIVE ids by actor.id "
                f"and position (max_distance={position_match_max_distance:.3f}m).",
                flush=True,
            )
        return opendrive_traffic_lights

    def _resolve_traffic_light_actor(self, link_signal_id):
        link_signal_id = str(link_signal_id).strip()
        if not link_signal_id:
            return None, "empty linkSignalID"

        if link_signal_id.startswith(OPENDRIVE_TLS_LINK_PREFIX):
            opendrive_id = link_signal_id[len(OPENDRIVE_TLS_LINK_PREFIX) :].strip()
            light_actor = next(
                (
                    self.opendrive_traffic_lights[alias]
                    for alias in self._opendrive_traffic_light_id_aliases(opendrive_id)
                    if alias in self.opendrive_traffic_lights
                ),
                None,
            )
            return light_actor, f"OpenDRIVE signal id {opendrive_id}"

        try:
            actor_id = int(link_signal_id)
        except ValueError:
            return None, f"invalid linkSignalID {link_signal_id!r}"

        light_actor = self.world.get_actor(actor_id)
        if light_actor:
            return light_actor, f"CARLA actor id {actor_id}"

        light_actor = self.opendrive_traffic_lights.get(str(actor_id))
        return light_actor, f"OpenDRIVE signal id {actor_id}"

    def resolve(self, token):
        """Resolve and cache a token, including unsuccessful lookups."""
        if token not in self._resolved_tokens:
            actor, description = self._resolve_traffic_light_actor(token)
            if actor is None or not isinstance(actor, carla.TrafficLight):
                actor = None
                self._warn_once(("resolve", token), f"Unresolved traffic light: {description}")
            self._resolved_tokens[token] = actor
        return self._resolved_tokens[token]

    def _node_bindings(self, node_id, information):
        cached = self._bindings.get(node_id)
        if cached is not None and cached[0] == information:
            return cached[1]
        try:
            programs = json.loads(information)["programs"]
            parameters = next(
                program["parameters"] for program in programs.values() if "parameters" in program
            )
            bindings = []
            for key, value in parameters.items():
                if not key.startswith(SUMO_CARLA_TLS_LINK_PREFIX):
                    continue
                index = int(key[len(SUMO_CARLA_TLS_LINK_PREFIX) :])
                if index < 0:
                    continue
                for token in str(value).split():
                    actor = self.resolve(token)
                    if actor is not None:
                        bindings.append((index, token, actor))
            # Match the original linkIndex loop, including last-writer behavior.
            bindings.sort(key=lambda item: item[0])
        except (ValueError, KeyError, StopIteration, TypeError) as exc:
            bindings = []
            self._warn_once(("information", node_id), f"Invalid TLS {node_id} information: {exc}")
        self._bindings[node_id] = (information, bindings)
        return bindings

    def sync(self, traffic_light_details):
        """Apply the supplied SUMO snapshot without requesting a CARLA tick."""
        states = {
            "g": carla.TrafficLightState.Green,
            "y": carla.TrafficLightState.Yellow,
            "r": carla.TrafficLightState.Red,
        }
        for node_id, node in traffic_light_details.items():
            signal = node.get("tls", "")
            for index, token, actor in self._node_bindings(node_id, node.get("information", "")):
                if index >= len(signal):
                    self._warn_once(
                        ("index", node_id, index), f"Out-of-range TLS {node_id}:{index}"
                    )
                    continue
                state = states.get(signal[index].lower())
                if state is None:
                    continue
                try:
                    actor.set_state(state)
                except RuntimeError as exc:
                    self._warn_once(
                        ("set_state", actor.id), f"Cannot update actor {actor.id}: {exc}"
                    )
