"""Stream live Vive Tracker poses over ZMQ to a LAN receiver (e.g. Ubuntu).

Windows side (sender). Reuses ``sim_teleop.tracker.read_tracker_poses`` so the
wire format matches what ``record_tracker`` saves to disk: a 4x4 homogeneous
pose in the SteamVR ``TrackingUniverseStanding`` frame.

Run from the repository root:

    & ".venv\\Scripts\\python.exe" -m sim_teleop.stream_pose --port 1234

On the other end (Ubuntu), subscribe with::

    python -m sim_teleop.receive_pose --host <windows_lan_ip> --port 1234

Wire message (JSON, sent every frame)::

    {
      "timestamp": 1780000000.123,
      "trackers": [
        {"serial": "3B-A33M...", "tracker_pose": [[...4x4...]]},
        ...
      ]
    }
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import openvr
import zmq

from .tracker import (
    DEFAULT_TRACKER_MAPPING_PATH,
    TRACKER_SERIAL_PREFIX,
    load_tracker_mapping,
    read_tracker_poses,
    tracker_pose_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream live Vive Tracker poses over ZMQ (LAN).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1234,
        help="TCP port to bind the ZMQ publisher (default: 1234).",
    )
    parser.add_argument(
        "--host",
        default="*",
        help="Interface to bind. '*' binds all interfaces for LAN access.",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=120.0,
        help="Publish frequency in Hz (default: 120).",
    )
    parser.add_argument(
        "--serial-prefix",
        default=TRACKER_SERIAL_PREFIX,
        help="Only stream trackers whose serial starts with this prefix.",
    )
    parser.add_argument(
        "--tracker-mapping",
        type=Path,
        default=DEFAULT_TRACKER_MAPPING_PATH,
        help=(
            "JSON role->serial mapping used to label streamed trackers "
            f"(default: {DEFAULT_TRACKER_MAPPING_PATH})."
        ),
    )
    args = parser.parse_args()
    if args.frequency <= 0.0:
        raise ValueError("--frequency must be positive.")

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    # Drop old frames instead of buffering a backlog: real-time pose stream.
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://{args.host}:{args.port}")
    print(f"[STREAM] ZMQ PUB bound on tcp://{args.host}:{args.port}", flush=True)

    openvr.init(openvr.VRApplication_Other)
    vr_system = openvr.VRSystem()
    print("[STREAM] Waiting for VR system to detect trackers...", flush=True)
    time.sleep(2.0)
    tracker_mapping = load_tracker_mapping(args.tracker_mapping)
    if tracker_mapping:
        mapping_text = ", ".join(
            f"{role}={serial}" for role, serial in tracker_mapping.items()
        )
        print(f"[STREAM] Mapping: {mapping_text}", flush=True)

    dt = 1.0 / args.frequency
    last_status = 0.0

    try:
        while True:
            loop_start = time.time()
            poses = read_tracker_poses(vr_system, args.serial_prefix)
            trackers = tracker_pose_records(poses, tracker_mapping)
            if trackers:
                msg = {
                    "timestamp": loop_start,
                    "trackers": trackers,
                }
                try:
                    socket.send_json(msg, flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass

            now = time.time()
            if now - last_status >= 1.0:
                serials = (
                    ", ".join(
                        f"{tracker.get('role', '?')}:{tracker['serial']}"
                        if "role" in tracker
                        else tracker["serial"]
                        for tracker in trackers
                    )
                    if trackers
                    else "none"
                )
                print(
                    f"[STREAM] trackers={len(trackers)} @ "
                    f"{args.frequency:.0f}Hz  [{serials}]",
                    flush=True,
                )
                last_status = now

            sleep_s = max(dt - (time.time() - loop_start), 0.0)
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("[STREAM] Stopping...", flush=True)
    finally:
        openvr.shutdown()
        socket.close(linger=0)
        context.term()
        print("[STREAM] Done.", flush=True)


if __name__ == "__main__":
    main()
