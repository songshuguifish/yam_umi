"""Record a single Vive Tracker pose stream to HuMI-style episode JSON.

With --gripper, a BRT encoder is read in parallel and each frame also stores
the raw encoder value and the calibrated normalised gripper position
(0 = closed, 1 = open). The same O/C keys used in the live teleop viewer
re-calibrate the encoder on the fly.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import openvr

from .gripper import (
    CALIBRATION_FILE,
    EncoderCalibration,
    create_instrument,
    find_serial_port,
    read_raw,
)
from .tracker import (
    DEFAULT_TRACKER_MAPPING_PATH,
    TRACKER_SERIAL_PREFIX,
    load_tracker_mapping,
    read_tracker_poses,
    tracker_pose_records,
)


def _get_key() -> str | None:
    try:
        import msvcrt  # type: ignore
    except ImportError:
        return None
    if not msvcrt.kbhit():
        return None
    key = msvcrt.getwch()
    if key == "\x1b":
        return "esc"
    return key.lower()


def _new_session(output_dir: Path, metadata: dict) -> Path:
    session_dir = output_dir / datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return session_dir


def _save_episode(session_dir: Path, start_ts: float, frames: list[dict]) -> Path | None:
    if not frames:
        return None
    ts = datetime.fromtimestamp(start_ts).strftime("%Y.%m.%d_%H.%M.%S.%f")
    out_path = session_dir / f"tracker_recording_{ts}.json"
    payload = {
        "schema": "vive_tracker_pose_episode_v1",
        "metadata": "metadata.json",
        "episode": frames,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def _primary_tracker(trackers: list[dict], mapping: dict[str, str]) -> dict | None:
    """Choose the legacy single-tracker pose, preferring left_eef."""
    if not trackers:
        return None
    for role in ("left_eef", "right_eef"):
        serial = mapping.get(role)
        if serial is None:
            continue
        for tracker in trackers:
            if tracker["serial"] == serial:
                return tracker
    return trackers[0]


def _connect_gripper(
    port: str | None,
    baudrate: int,
    slave: int,
) -> tuple[object | None, str]:
    """Open the BRT encoder if one is reachable.

    Returns ``(instrument, port_or_reason)``. When the instrument is None the
    second element is a human-readable reason; otherwise it is the resolved
    serial port name. Failures degrade gracefully so tracker recording still
    works without a gripper attached.
    """
    resolved = port or find_serial_port()
    if resolved is None:
        return None, "no serial port found"
    try:
        inst = create_instrument(resolved, slave_addr=slave, baudrate=baudrate)
    except Exception as exc:  # noqa: BLE001 — keep recording without the gripper
        return None, f"cannot open {resolved}: {exc}"
    if read_raw(inst) is None:
        inst.serial.close()
        return None, "encoder did not respond (check wiring/baudrate/slave)"
    return inst, resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one Vive Tracker pose stream.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data/tracker_poses"),
        help="Root directory for tracker pose sessions.",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        type=float,
        default=120.0,
        help="Recording frequency in Hz.",
    )
    parser.add_argument(
        "--serial-prefix",
        default=TRACKER_SERIAL_PREFIX,
        help="Only record trackers whose serial starts with this prefix.",
    )
    parser.add_argument(
        "--tracker-mapping",
        type=Path,
        default=DEFAULT_TRACKER_MAPPING_PATH,
        help=(
            "JSON role->serial mapping used to label trackers "
            f"(default: {DEFAULT_TRACKER_MAPPING_PATH})."
        ),
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start recording immediately instead of waiting for S.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop and save after this many seconds once recording starts.",
    )
    # ── Gripper (BRT encoder) ────────────────────────────────────────────────
    parser.add_argument(
        "-g",
        "--gripper",
        action="store_true",
        help="Also record the BRT gripper encoder (adds gripper_raw/gripper_norm).",
    )
    parser.add_argument(
        "--gripper-port",
        default=None,
        help="Serial port for the BRT encoder (e.g. COM6). Auto-detect if omitted.",
    )
    parser.add_argument("--gripper-baudrate", type=int, default=9600)
    parser.add_argument(
        "--gripper-slave", type=int, default=1, help="Modbus slave address."
    )
    parser.add_argument(
        "--gripper-calibration",
        type=Path,
        default=None,
        help=(
            "Encoder calibration JSON (raw_open/raw_closed). "
            f"Defaults to {CALIBRATION_FILE}."
        ),
    )
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0.0:
        raise ValueError("--duration must be positive.")

    openvr.init(openvr.VRApplication_Other)
    vr_system = openvr.VRSystem()
    time.sleep(2.0)
    tracker_mapping = load_tracker_mapping(args.tracker_mapping)

    # ── Gripper ──────────────────────────────────────────────────────────────
    gripper_inst: object | None = None
    gripper_cal: EncoderCalibration | None = None
    gripper_port: str | None = None
    cal_path = args.gripper_calibration or CALIBRATION_FILE
    if args.gripper:
        gripper_inst, info = _connect_gripper(
            args.gripper_port, args.gripper_baudrate, args.gripper_slave
        )
        gripper_cal = EncoderCalibration.load(cal_path)
        if gripper_inst is not None:
            gripper_port = info
            print(f"[GRIPPER] encoder on {gripper_port}  {gripper_cal}", flush=True)
        else:
            print(f"[GRIPPER] disabled: {info}", flush=True)

    metadata = {
        "schema": "vive_tracker_pose_session_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frequency": args.frequency,
        "duration": args.duration,
        "serial_prefix": args.serial_prefix,
        "tracker_mapping_config": str(args.tracker_mapping),
        "tracker_mapping": tracker_mapping,
        "tracking_universe": "TrackingUniverseStanding",
        "controls": {
            "s": "start recording",
            "t": "stop and save",
            "q_or_esc": "save active recording and quit",
            "o": "record current encoder value as gripper OPEN (with --gripper)",
            "c": "record current encoder value as gripper CLOSED (with --gripper)",
        },
        "gripper": {
            "enabled": args.gripper,
            "port": gripper_port,
            "baudrate": args.gripper_baudrate if args.gripper else None,
            "slave": args.gripper_slave if args.gripper else None,
            "calibration_path": str(cal_path) if args.gripper else None,
            "calibration": (
                {
                    "raw_open": gripper_cal.raw_open,
                    "raw_closed": gripper_cal.raw_closed,
                }
                if args.gripper and gripper_cal is not None
                else None
            ),
        },
    }
    session_dir = _new_session(args.output_dir, metadata)

    frames: list[dict] = []
    recording = False
    start_ts: float | None = None
    last_status = 0.0
    last_raw: int | None = None
    dt = 1.0 / args.frequency

    if args.auto_start:
        recording = True
        start_ts = time.time()

    print(f"[TRACKER] Session: {session_dir}", flush=True)
    print("[TRACKER] S=start  T=stop/save  Q/Esc=save+quit", flush=True)
    if args.gripper:
        print("[GRIPPER] O=open calib  C=closed calib", flush=True)
    if tracker_mapping:
        mapping_text = ", ".join(
            f"{role}={serial}" for role, serial in tracker_mapping.items()
        )
        print(f"[TRACKER] Mapping: {mapping_text}", flush=True)

    try:
        while True:
            loop_start = time.time()
            tracker_poses = read_tracker_poses(vr_system, args.serial_prefix)
            trackers = tracker_pose_records(tracker_poses, tracker_mapping)

            gripper_raw: int | None = None
            gripper_norm: float | None = None
            if gripper_inst is not None:
                raw = read_raw(gripper_inst)
                if raw is not None:
                    last_raw = raw
                    gripper_raw = raw
                    if gripper_cal is not None and gripper_cal.is_ready:
                        gripper_norm = gripper_cal.normalise(raw)

            if trackers:
                if recording:
                    primary = _primary_tracker(trackers, tracker_mapping)
                    frame = {
                        "timestamp": loop_start,
                        "trackers": trackers,
                        "poses_by_role": {
                            tracker["role"]: tracker["tracker_pose"]
                            for tracker in trackers
                            if "role" in tracker
                        },
                    }
                    if primary is not None:
                        # Legacy fields keep existing single-tracker replay usable.
                        frame["serial"] = primary["serial"]
                        frame["tracker_pose"] = primary["tracker_pose"]
                    if args.gripper:
                        frame["gripper_raw"] = gripper_raw
                        frame["gripper_norm"] = gripper_norm
                    frames.append(frame)

            key = _get_key()
            if key == "s":
                if not recording:
                    frames.clear()
                    start_ts = time.time()
                    recording = True
                    print("[TRACKER] START", flush=True)
                else:
                    print("[TRACKER] Already recording.", flush=True)
            elif key == "t":
                if recording and start_ts is not None:
                    out_path = _save_episode(session_dir, start_ts, frames)
                    print(f"[TRACKER] SAVED {out_path}", flush=True)
                    frames.clear()
                    start_ts = None
                    recording = False
                else:
                    print("[TRACKER] Not recording.", flush=True)
            elif key == "o":
                if args.gripper and last_raw is not None and gripper_cal is not None:
                    gripper_cal.raw_open = last_raw
                    gripper_cal.save(cal_path)
                    print(f"[GRIPPER] OPEN raw={last_raw}  {gripper_cal}", flush=True)
            elif key == "c":
                if args.gripper and last_raw is not None and gripper_cal is not None:
                    gripper_cal.raw_closed = last_raw
                    gripper_cal.save(cal_path)
                    print(f"[GRIPPER] CLOSED raw={last_raw}  {gripper_cal}", flush=True)
            elif key in ("q", "esc"):
                if recording and start_ts is not None:
                    out_path = _save_episode(session_dir, start_ts, frames)
                    print(f"[TRACKER] SAVED {out_path}", flush=True)
                break

            now = time.time()
            if now - last_status >= 1.0:
                status = "REC" if recording else "IDLE"
                found = len(trackers)
                count = len(frames)
                labels = ", ".join(
                    f"{tracker.get('role', '?')}:{tracker['serial']}"
                    if "role" in tracker
                    else tracker["serial"]
                    for tracker in trackers
                )
                line = f"[{status}] trackers={found} frames={count} [{labels}]"
                if args.gripper:
                    if gripper_norm is not None:
                        line += f"  grip[raw={gripper_raw} norm={gripper_norm:.2f}]"
                    elif gripper_raw is not None:
                        line += f"  grip[raw={gripper_raw} (uncalibrated)]"
                    else:
                        line += "  grip[no signal]"
                print(line, flush=True)
                last_status = now

            if (
                args.duration is not None
                and recording
                and start_ts is not None
                and loop_start - start_ts >= args.duration
            ):
                out_path = _save_episode(session_dir, start_ts, frames)
                print(f"[TRACKER] SAVED {out_path}", flush=True)
                break

            sleep_s = max(dt - (time.time() - loop_start), 0.0)
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        if recording and start_ts is not None:
            out_path = _save_episode(session_dir, start_ts, frames)
            print(f"[TRACKER] SAVED {out_path}", flush=True)
    finally:
        if gripper_inst is not None:
            gripper_inst.serial.close()  # type: ignore[attr-defined]
            print("[GRIPPER] Port closed.", flush=True)
        openvr.shutdown()
        print("[TRACKER] Done.", flush=True)


if __name__ == "__main__":
    main()
