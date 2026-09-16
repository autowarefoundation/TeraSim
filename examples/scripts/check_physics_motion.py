#!/usr/bin/env python3
"""Verify that a CARLA background vehicle moves under physics co-sim."""

import argparse
import math
import sys

import carla


EGO_ROLES = {"AV", "ego_vehicle", "hero"}


def horizontal_speed(velocity):
    return math.hypot(velocity.x, velocity.y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--frames", type=int, default=2400)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--min-displacement", type=float, default=1.0)
    parser.add_argument("--min-speed", type=float, default=0.5)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    world.wait_for_tick(args.timeout)

    tracks = {}
    winner = None
    for frame_index in range(1, args.frames + 1):
        world.wait_for_tick(args.timeout)
        for actor in world.get_actors().filter("vehicle.*"):
            role = actor.attributes.get("role_name", "")
            if role in EGO_ROLES:
                continue
            location = actor.get_location()
            speed = horizontal_speed(actor.get_velocity())
            track = tracks.setdefault(
                actor.id,
                {
                    "role": role,
                    "first_x": location.x,
                    "first_y": location.y,
                    "max_displacement": 0.0,
                    "max_speed": 0.0,
                },
            )
            displacement = math.hypot(
                location.x - track["first_x"],
                location.y - track["first_y"],
            )
            track["max_displacement"] = max(
                track["max_displacement"], displacement
            )
            track["max_speed"] = max(track["max_speed"], speed)
            if (
                track["max_displacement"] >= args.min_displacement
                and track["max_speed"] >= args.min_speed
            ):
                winner = track
                break
        if winner is not None:
            print(
                "PHYSICS_MOTION_OK "
                f"actor={winner['role']} frame_count={frame_index} "
                f"displacement={winner['max_displacement']:.3f}m "
                f"max_speed={winner['max_speed']:.3f}m/s"
            )
            return 0

    best = max(
        tracks.values(),
        key=lambda track: (track["max_displacement"], track["max_speed"]),
        default=None,
    )
    if best is None:
        detail = "no background vehicle observed"
    else:
        detail = (
            f"best_actor={best['role']} "
            f"displacement={best['max_displacement']:.3f}m "
            f"max_speed={best['max_speed']:.3f}m/s"
        )
    print(f"PHYSICS_MOTION_FAIL frames={args.frames} {detail}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
