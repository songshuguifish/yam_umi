"""Collect a short raw session from encoder, Vive trackers, and RealSense RGB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .encoder_process import EncoderProcess, encoder_sample_example
from .ring_buffer import SharedMemoryRingBuffer
from .timebase import Timebase
from .tracker_process import TrackerProcess, tracker_sample_example


DEFAULT_REALSENSE_PYTHON = Path(sys.executable)

# The camera recorder does a ~1s warmup after pipeline.start before recording.
# Launching it first lets that warmup overlap sensor startup, and recording for
# `duration + CAMERA_WARMUP_S` keeps the camera window covering the main loop.
CAMERA_WARMUP_S = 3.0


def _session_dir(root: Path) -> Path:
    session = root / time.strftime("session_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=False)
    return session


def _save_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def _start_realsense_recorder(
    python_exe: Path,
    output_dir: Path,
    *,
    duration: float,
    width: int,
    height: int,
    fps: int,
    max_cameras: int | None,
    timebase: Timebase,
    ready_file: Path | None = None,
) -> subprocess.Popen:
    cmd = [
        str(python_exe),
        "-m",
        "sim_teleop.data_collection.realsense_rgb_record",
        "--output-dir",
        str(output_dir),
        "--duration",
        str(duration),
        "--width",
        str(width),
        "--height",
        str(height),
        "--fps",
        str(fps),
        "--wall0",
        str(timebase.wall0),
        "--perf0",
        str(timebase.perf0),
    ]
    if max_cameras is not None:
        cmd.extend(["--max-cameras", str(max_cameras)])
    if ready_file is not None:
        cmd.extend(["--ready-file", str(ready_file)])
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_for_ready_file(
    ready_file: Path,
    process: subprocess.Popen,
    *,
    timeout_s: float,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if ready_file.exists():
            return
        if process.poll() is not None:
            raise RuntimeError(f"RealSense recorder exited before ready: {process.returncode}")
        time.sleep(0.05)
    raise RuntimeError(f"RealSense recorder did not become ready: {ready_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("data/raw_smoke"))
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--encoder-frequency", type=float, default=30.0)
    parser.add_argument("--tracker-frequency", type=float, default=120.0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--max-cameras", type=int, default=None)
    parser.add_argument(
        "--encoder-port",
        action="append",
        default=None,
        help=(
            "Encoder serial port. Repeat for multiple encoders, e.g. "
            "--encoder-port COM10 --encoder-port COM8. If omitted, auto-detect one."
        ),
    )
    parser.add_argument("--encoder-baudrate", type=int, default=9600)
    parser.add_argument("--encoder-slave", type=int, default=1)
    parser.add_argument("--realsense-python", type=Path, default=DEFAULT_REALSENSE_PYTHON)
    args = parser.parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be positive")

    session_dir = _session_dir(args.output_dir)
    lowdim_dir = session_dir / "lowdim"
    camera_dir = session_dir / "cameras"
    timebase = Timebase.create()

    encoder_ports = args.encoder_port or [None]
    encoder_capacity = int(args.duration * args.encoder_frequency * 2) + 64
    tracker_capacity = int(args.duration * args.tracker_frequency * 2) + 64
    encoder_rings = [
        SharedMemoryRingBuffer(encoder_sample_example(), encoder_capacity)
        for _ in encoder_ports
    ]
    tracker_ring = SharedMemoryRingBuffer(tracker_sample_example(), tracker_capacity)

    encoders = [
        EncoderProcess(
            ring,
            timebase,
            port=port,
            baudrate=args.encoder_baudrate,
            slave_addr=args.encoder_slave,
            frequency=args.encoder_frequency,
        )
        for ring, port in zip(encoder_rings, encoder_ports)
    ]
    tracker = TrackerProcess(
        tracker_ring,
        timebase,
        frequency=args.tracker_frequency,
    )

    rs_proc = None
    metadata = {
        "schema": "raw_smoke_session_v1",
        "session_dir": str(session_dir),
        "duration": args.duration,
        "timebase": {"wall0": timebase.wall0, "perf0": timebase.perf0},
        "encoder_ports": encoder_ports,
        "encoder_frequency": args.encoder_frequency,
        "encoder_baudrate": args.encoder_baudrate,
        "encoder_slave": args.encoder_slave,
        "tracker_frequency": args.tracker_frequency,
        "camera": {
            "width": args.camera_width,
            "height": args.camera_height,
            "fps": args.camera_fps,
            "max_cameras": args.max_cameras,
        },
    }

    try:
        print(f"[COLLECT] Session: {session_dir}", flush=True)

        # Launch cameras first and wait until the recorder has entered its
        # frame loop before starting lowdim sensors. The extra duration keeps
        # cameras running through encoder/tracker startup plus the main loop.
        if args.realsense_python.exists():
            ready_file = camera_dir / "realsense_ready.json"
            rs_proc = _start_realsense_recorder(
                args.realsense_python,
                camera_dir,
                duration=args.duration + CAMERA_WARMUP_S,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
                max_cameras=args.max_cameras,
                timebase=timebase,
                ready_file=ready_file,
            )
            _wait_for_ready_file(
                ready_file,
                rs_proc,
                timeout_s=max(args.duration + 15.0, 20.0),
            )
        else:
            print(f"[COLLECT] RealSense python not found: {args.realsense_python}")

        for idx, encoder in enumerate(encoders):
            encoder.start()
            if not encoder.ready_event.wait(timeout=5.0):
                raise RuntimeError(
                    f"encoder{idx} process did not become ready "
                    f"(port={encoder_ports[idx] or 'auto'}, "
                    f"baudrate={args.encoder_baudrate}, slave={args.encoder_slave})"
                )
        tracker.start()
        if not tracker.ready_event.wait(timeout=8.0):
            raise RuntimeError("tracker process did not become ready")

        start = time.time()
        while time.time() - start < args.duration:
            try:
                enc_latest = [ring.get_latest() for ring in encoder_rings]
                trk = tracker_ring.get_latest()
                enc_text = " ".join(
                    f"enc{i}_raw={int(enc['raw'])} enc{i}_valid={int(enc['valid'])}"
                    for i, enc in enumerate(enc_latest)
                )
                print(
                    "[COLLECT] "
                    f"{enc_text} "
                    f"trk_n={int(trk['num_trackers'])} "
                    f"L={int(trk['left_eef_valid'])} R={int(trk['right_eef_valid'])}",
                    flush=True,
                )
            except IndexError:
                pass
            time.sleep(0.5)

        for encoder in encoders:
            encoder.stop()
        tracker.stop()
        for encoder in encoders:
            encoder.join(timeout=3.0)
        tracker.join(timeout=3.0)

        if rs_proc is not None:
            rs_stdout, _ = rs_proc.communicate(timeout=max(args.duration + 10.0, 15.0))
            metadata["realsense_returncode"] = rs_proc.returncode
            metadata["realsense_stdout"] = rs_stdout
            print(rs_stdout, end="", flush=True)

        enc_counts = [min(ring.count, encoder_capacity) for ring in encoder_rings]
        trk_k = min(tracker_ring.count, tracker_capacity)
        encoder_data_list = [
            ring.get_last_k(k)
            for ring, k in zip(encoder_rings, enc_counts)
        ]
        tracker_data = tracker_ring.get_last_k(trk_k)
        for idx, encoder_data in enumerate(encoder_data_list):
            suffix = "" if len(encoder_data_list) == 1 else str(idx)
            _save_npz(lowdim_dir / f"encoder{suffix}.npz", encoder_data)
        _save_npz(lowdim_dir / "tracker.npz", tracker_data)
        metadata["encoder_samples"] = [int(k) for k in enc_counts]
        metadata["tracker_samples"] = int(trk_k)
        metadata["encoder_actual_hz"] = []
        metadata["encoder_mean_read_ms"] = []
        for enc_k, encoder_data in zip(enc_counts, encoder_data_list):
            if enc_k > 1:
                enc_duration = float(
                    encoder_data["timestamp"][-1] - encoder_data["timestamp"][0]
                )
                metadata["encoder_actual_hz"].append(float((enc_k - 1) / enc_duration))
                metadata["encoder_mean_read_ms"].append(
                    float(
                        np.mean(
                            encoder_data["read_end_timestamp"]
                            - encoder_data["read_start_timestamp"]
                        )
                        * 1000.0
                    )
                )
            else:
                metadata["encoder_actual_hz"].append(0.0)
                metadata["encoder_mean_read_ms"].append(0.0)
        if trk_k > 1:
            trk_duration = float(tracker_data["timestamp"][-1] - tracker_data["timestamp"][0])
            metadata["tracker_actual_hz"] = float((trk_k - 1) / trk_duration)
        print(
            f"[COLLECT] Saved encoder={enc_counts} tracker={trk_k} samples",
            flush=True,
        )
    finally:
        for encoder in encoders:
            encoder.stop()
        tracker.stop()
        for encoder in encoders:
            if encoder.is_alive():
                encoder.terminate()
        if tracker.is_alive():
            tracker.terminate()
        if rs_proc is not None and rs_proc.poll() is None:
            rs_proc.terminate()
        (session_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        print(f"[COLLECT] Metadata: {session_dir / 'metadata.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
