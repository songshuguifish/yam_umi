"""Vive Tracker sensor process."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from sim_teleop.tracker import (
    DEFAULT_TRACKER_MAPPING_PATH,
    TRACKER_SERIAL_PREFIX,
    load_tracker_mapping,
    read_tracker_poses,
)

from .ring_buffer import SharedMemoryRingBuffer
from .timebase import Timebase, midpoint


def tracker_sample_example() -> dict[str, object]:
    return {
        "timestamp": np.float64(0.0),
        "read_start_timestamp": np.float64(0.0),
        "read_end_timestamp": np.float64(0.0),
        "left_eef_pose": np.eye(4, dtype=np.float64),
        "right_eef_pose": np.eye(4, dtype=np.float64),
        "left_eef_valid": np.int8(0),
        "right_eef_valid": np.int8(0),
        "num_trackers": np.int16(0),
    }


class TrackerProcess(mp.Process):
    """Poll OpenVR and publish left/right tracker poses to a ring buffer."""

    def __init__(
        self,
        ring_buffer: SharedMemoryRingBuffer,
        timebase: Timebase,
        *,
        tracker_mapping: Path | None = DEFAULT_TRACKER_MAPPING_PATH,
        serial_prefix: str | None = TRACKER_SERIAL_PREFIX,
        frequency: float = 120.0,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        if frequency <= 0:
            raise ValueError("frequency must be positive")
        self.ring_buffer = ring_buffer
        self.timebase = timebase
        self.tracker_mapping = tracker_mapping
        self.serial_prefix = serial_prefix
        self.frequency = float(frequency)
        self.verbose = verbose
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.error: Exception | None = None

    @property
    def is_ready(self) -> bool:
        return self.ready_event.is_set()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        import openvr

        mapping = load_tracker_mapping(self.tracker_mapping)
        left_serial = mapping.get("left_eef")
        right_serial = mapping.get("right_eef")
        dt = 1.0 / self.frequency
        next_tick = time.perf_counter()

        try:
            openvr.init(openvr.VRApplication_Other)
            vr_system = openvr.VRSystem()
            time.sleep(1.0)
            print(
                "[TRACKER] OpenVR ready, waiting for a valid pose "
                f"(left={left_serial} right={right_serial})",
                flush=True,
            )

            consecutive_errors = 0
            while not self.stop_event.is_set():
                try:
                    read_start = self.timebase.now()
                    poses = read_tracker_poses(vr_system, self.serial_prefix)
                    read_end = self.timebase.now()
                except Exception as exc:  # noqa: BLE001 - transient OpenVR error
                    consecutive_errors += 1
                    if consecutive_errors == 1 or consecutive_errors % 100 == 0:
                        print(
                            f"[TRACKER] pose read error (#{consecutive_errors}): {exc}",
                            flush=True,
                        )
                    next_tick += dt
                    sleep_s = next_tick - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    else:
                        next_tick = time.perf_counter()
                    continue
                consecutive_errors = 0

                by_serial = {serial: pose for serial, pose in poses}
                left_pose = by_serial.get(left_serial) if left_serial else None
                right_pose = by_serial.get(right_serial) if right_serial else None
                sample = {
                    "timestamp": midpoint(read_start, read_end),
                    "read_start_timestamp": read_start,
                    "read_end_timestamp": read_end,
                    "left_eef_pose": (
                        left_pose if left_pose is not None else np.eye(4)
                    ),
                    "right_eef_pose": (
                        right_pose if right_pose is not None else np.eye(4)
                    ),
                    "left_eef_valid": int(left_pose is not None),
                    "right_eef_valid": int(right_pose is not None),
                    "num_trackers": len(poses),
                }
                self.ring_buffer.put(sample)

                if not self.ready_event.is_set() and poses:
                    print(
                        f"[TRACKER] tracking {len(poses)} tracker(s) "
                        f"(L={sample['left_eef_valid']} "
                        f"R={sample['right_eef_valid']})",
                        flush=True,
                    )
                    self.ready_event.set()

                if self.verbose:
                    print(
                        "[TRACKER] "
                        f"n={len(poses)} "
                        f"L={sample['left_eef_valid']} "
                        f"R={sample['right_eef_valid']}",
                        flush=True,
                    )

                next_tick += dt
                sleep_s = next_tick - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.perf_counter()
        except Exception as exc:  # noqa: BLE001 - fatal (init/shutdown) error
            self.error = exc
            print(f"[TRACKER] FATAL: {exc}", flush=True)
        finally:
            try:
                openvr.shutdown()
            except Exception:
                pass
            print("[TRACKER] Stopped.", flush=True)


def _format_pose(name: str, pose: np.ndarray, valid: int) -> str:
    if not valid:
        return f"{name}=INVALID"
    pos = pose[:3, 3]
    return f"{name}=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test Vive tracker process.")
    parser.add_argument("--frequency", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--buffer-size", type=int, default=256)
    parser.add_argument(
        "--tracker-mapping",
        type=Path,
        default=DEFAULT_TRACKER_MAPPING_PATH,
    )
    parser.add_argument(
        "--serial-prefix",
        default=TRACKER_SERIAL_PREFIX,
        help="Use empty string to disable prefix filtering.",
    )
    args = parser.parse_args()

    serial_prefix = args.serial_prefix if args.serial_prefix else None
    timebase = Timebase.create()
    ring = SharedMemoryRingBuffer(tracker_sample_example(), capacity=args.buffer_size)
    proc = TrackerProcess(
        ring,
        timebase,
        tracker_mapping=args.tracker_mapping,
        serial_prefix=serial_prefix,
        frequency=args.frequency,
    )
    proc.start()
    if not proc.ready_event.wait(timeout=8.0):
        proc.stop()
        proc.join(timeout=2.0)
        raise SystemExit("ERROR: tracker process did not become ready.")

    start = time.time()
    try:
        while time.time() - start < args.duration:
            try:
                latest = ring.get_latest()
            except IndexError:
                time.sleep(0.05)
                continue
            left = _format_pose(
                "left",
                latest["left_eef_pose"],
                int(latest["left_eef_valid"]),
            )
            right = _format_pose(
                "right",
                latest["right_eef_pose"],
                int(latest["right_eef_valid"]),
            )
            print(
                "t={:.6f} n={} {} {} count={}".format(
                    float(latest["timestamp"]),
                    int(latest["num_trackers"]),
                    left,
                    right,
                    ring.count,
                ),
                flush=True,
            )
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        proc.stop()
        proc.join(timeout=3.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)


if __name__ == "__main__":
    main()
