"""Hot-start sensors once, then record episodes with r/s/q hotkeys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from gripper.encoder import EncoderCalibration

from .calibrate_encoders import DEFAULT_CONFIG_PATH as DEFAULT_ENCODER_CONFIG_PATH
from .calibrate_encoders import resolve_ports
from .encoder_process import EncoderProcess, encoder_sample_example
from .ring_buffer import SharedMemoryRingBuffer
from .timebase import Timebase
from .tracker_process import TrackerProcess, tracker_sample_example


DEFAULT_REALSENSE_PYTHON = Path(sys.executable)


def _session_dir(root: Path) -> Path:
    session = root / time.strftime("session_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=False)
    return session


def _save_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def _slice_by_window(
    ring: SharedMemoryRingBuffer,
    count_start: int,
    t_start: float,
    t_stop: float,
) -> tuple[dict[str, np.ndarray], bool]:
    produced = ring.count - count_start
    if produced <= 0:
        empty = {
            key: np.empty((0,) + spec.shape, dtype=spec.dtype)
            for key, spec in ring.specs.items()
        }
        return empty, False
    truncated = produced > ring.capacity
    batch = ring.get_last_k(min(produced, ring.capacity))
    ts = batch["timestamp"]
    mask = (ts >= t_start) & (ts <= t_stop)
    return {key: value[mask] for key, value in batch.items()}, truncated


def _read_key(prompt: str, valid: set[str]) -> str:
    print(prompt, flush=True)
    try:
        import msvcrt

        while True:
            key = msvcrt.getwch().lower()
            if key in valid:
                print(key, flush=True)
                return key
    except ImportError:
        while True:
            value = input("> ").strip().lower()
            if value in valid:
                return value


def _role_label(role: str) -> str:
    return role.removesuffix("_encoder")


def _load_encoder_config(path: Path) -> tuple[dict, list[tuple[str, str]]]:
    config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    resolved = resolve_ports(path) if path.exists() else []
    return config, resolved


class _CameraServer:
    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc

    def _send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def start_episode(self, cameras_dir: Path, *, timeout_s: float) -> bool:
        ready = cameras_dir / "realsense_ready.json"
        done = cameras_dir / "realsense_done.json"
        for marker in (ready, done):
            if marker.exists():
                marker.unlink()
        self._send(f"START {cameras_dir}")
        return self._wait(ready, timeout_s=timeout_s)

    def stop_episode(self, cameras_dir: Path, *, timeout_s: float) -> bool:
        self._send("STOP")
        return self._wait(cameras_dir / "realsense_done.json", timeout_s=timeout_s)

    def quit(self) -> None:
        try:
            self._send("QUIT")
        except Exception:
            pass

    def _wait(self, marker: Path, *, timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if marker.exists():
                return True
            if self.proc.poll() is not None:
                return False
            time.sleep(0.02)
        return False


def _start_camera_server(
    python_exe: Path,
    *,
    width: int,
    height: int,
    fps: int,
    max_cameras: int | None,
    timebase: Timebase,
    ready_file: Path,
    log_file: Path,
) -> _CameraServer:
    cmd = [
        str(python_exe),
        "-m",
        "sim_teleop.data_collection.realsense_rgb_record",
        "--serve",
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
        "--ready-file",
        str(ready_file),
    ]
    if max_cameras is not None:
        cmd.extend(["--max-cameras", str(max_cameras)])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=open(log_file, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    return _CameraServer(proc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("data/sessions"))
    parser.add_argument("--encoder-frequency", type=float, default=30.0)
    parser.add_argument("--tracker-frequency", type=float, default=120.0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--max-cameras", type=int, default=3)
    parser.add_argument("--max-episode-s", type=float, default=180.0)
    parser.add_argument("--encoder-mapping", type=Path, default=DEFAULT_ENCODER_CONFIG_PATH)
    parser.add_argument("--realsense-python", type=Path, default=DEFAULT_REALSENSE_PYTHON)
    parser.add_argument("--no-camera", action="store_true")
    args = parser.parse_args()

    session_dir = _session_dir(args.output_dir)
    timebase = Timebase.create()
    print(f"[SESSION] {session_dir}", flush=True)

    encoder_config, resolved_ports = _load_encoder_config(args.encoder_mapping)
    role_entries = encoder_config.get("roles", {})
    cap_enc = int(args.max_episode_s * args.encoder_frequency) + 256
    cap_trk = int(args.max_episode_s * args.tracker_frequency) + 256

    encoder_rings: dict[str, SharedMemoryRingBuffer] = {}
    encoders: dict[str, EncoderProcess] = {}
    for role, port in resolved_ports:
        entry = role_entries.get(role) or {}
        cal_data = entry.get("calibration") or {}
        stroke_mm = cal_data.get("stroke_mm")
        if stroke_mm is None and entry.get("gripper_length_m") is not None:
            stroke_mm = float(entry["gripper_length_m"]) * 1000.0
        if stroke_mm is None and entry.get("gripper_length") is not None:
            stroke_mm = float(entry["gripper_length"]) * 1000.0
        calibration = EncoderCalibration(
            raw_closed=cal_data.get("raw_closed"),
            raw_open=cal_data.get("raw_open"),
            stroke_mm=stroke_mm,
        )
        ring = SharedMemoryRingBuffer(encoder_sample_example(), cap_enc)
        encoder_rings[role] = ring
        encoders[role] = EncoderProcess(
            ring,
            timebase,
            port=port,
            usb_serial=entry.get("usb_serial"),
            baudrate=int(entry.get("baudrate", encoder_config.get("baudrate", 9600))),
            slave_addr=int(entry.get("slave", encoder_config.get("slave", 1))),
            frequency=args.encoder_frequency,
            calibration=calibration,
            side=_role_label(role),
        )
        print(f"[SESSION] encoder {role}: port={port} {calibration}", flush=True)

    tracker_ring = SharedMemoryRingBuffer(tracker_sample_example(), cap_trk)
    tracker = TrackerProcess(tracker_ring, timebase, frequency=args.tracker_frequency)
    camera: _CameraServer | None = None
    episode_index = 0
    metadata = {
        "schema": "hotkey_session_v1",
        "session_dir": str(session_dir),
        "timebase": {"wall0": timebase.wall0, "perf0": timebase.perf0},
        "encoder_mapping": encoder_config,
        "encoder_resolved_ports": {role: port for role, port in resolved_ports},
        "encoder_frequency": args.encoder_frequency,
        "tracker_frequency": args.tracker_frequency,
        "camera": None
        if args.no_camera
        else {
            "width": args.camera_width,
            "height": args.camera_height,
            "fps": args.camera_fps,
            "max_cameras": args.max_cameras,
        },
        "episodes": [],
    }

    try:
        if not args.no_camera:
            ready_file = session_dir / "cameras_rig_ready.json"
            camera = _start_camera_server(
                args.realsense_python,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
                max_cameras=args.max_cameras,
                timebase=timebase,
                ready_file=ready_file,
                log_file=session_dir / "realsense_serve.log",
            )
            print("[SESSION] warming up cameras...", flush=True)
            if not camera._wait(ready_file, timeout_s=30.0):
                raise RuntimeError(
                    "RealSense serve did not become ready "
                    f"(see {session_dir / 'realsense_serve.log'})"
                )
            print("[SESSION] cameras ready", flush=True)

        for role, proc in encoders.items():
            proc.start()
            if not proc.ready_event.wait(timeout=5.0):
                raise RuntimeError(f"encoder {role} did not become ready")
        tracker.start()
        if not tracker.ready_event.wait(timeout=8.0):
            raise RuntimeError("tracker process did not become ready")
        print("[SESSION] all sensors hot. r=start, s=stop, q=quit", flush=True)

        while True:
            cmd = _read_key(
                f"\n[SESSION] episode {episode_index}: press r to START, q to quit",
                {"r", "q"},
            )
            if cmd == "q":
                break

            ep_dir = session_dir / f"episode_{episode_index:03d}"
            cameras_dir = ep_dir / "cameras"
            t_start = timebase.now()
            enc_count_start = {role: ring.count for role, ring in encoder_rings.items()}
            trk_count_start = tracker_ring.count

            if camera is not None and not camera.start_episode(
                cameras_dir, timeout_s=15.0
            ):
                print("[SESSION] WARNING: camera did not confirm START.", flush=True)

            stop_cmd = _read_key(
                "[SESSION] recording... press s to STOP/save, q to STOP/save and quit",
                {"s", "q"},
            )
            t_stop = timebase.now()

            cam_frames = None
            if camera is not None:
                if camera.stop_episode(cameras_dir, timeout_s=20.0):
                    try:
                        done = json.loads(
                            (cameras_dir / "realsense_done.json").read_text()
                        )
                        cam_frames = done.get("frames")
                    except Exception:
                        pass
                else:
                    print("[SESSION] WARNING: camera did not confirm STOP.", flush=True)

            lowdim_dir = ep_dir / "lowdim"
            ep_meta = {
                "episode": episode_index,
                "t_start": t_start,
                "t_stop": t_stop,
                "duration": t_stop - t_start,
                "camera_frames": cam_frames,
                "encoder_samples": {},
            }
            for role, ring in encoder_rings.items():
                data, truncated = _slice_by_window(
                    ring, enc_count_start[role], t_start, t_stop
                )
                label = _role_label(role)
                _save_npz(lowdim_dir / f"encoder_{label}.npz", data)
                ep_meta["encoder_samples"][role] = int(len(data["timestamp"]))
                if truncated:
                    print(f"[SESSION] WARNING: encoder {role} ring truncated.")
            trk_data, trk_trunc = _slice_by_window(
                tracker_ring, trk_count_start, t_start, t_stop
            )
            _save_npz(lowdim_dir / "tracker.npz", trk_data)
            ep_meta["tracker_samples"] = int(len(trk_data["timestamp"]))
            if trk_trunc:
                print("[SESSION] WARNING: tracker ring truncated.")

            ep_dir.mkdir(parents=True, exist_ok=True)
            (ep_dir / "metadata.json").write_text(
                json.dumps(ep_meta, indent=2), encoding="utf-8"
            )
            metadata["episodes"].append(ep_meta)
            print(
                f"[SESSION] saved episode {episode_index}: "
                f"dur={ep_meta['duration']:.1f}s "
                f"enc={ep_meta['encoder_samples']} "
                f"trk={ep_meta['tracker_samples']} "
                f"cam_frames={cam_frames} -> {ep_dir}",
                flush=True,
            )
            episode_index += 1
            if stop_cmd == "q":
                break
    except KeyboardInterrupt:
        print("\n[SESSION] interrupted.", flush=True)
    finally:
        if camera is not None:
            camera.quit()
        for proc in encoders.values():
            proc.stop()
        tracker.stop()
        for proc in encoders.values():
            proc.join(timeout=3.0)
            if proc.is_alive():
                proc.terminate()
        tracker.join(timeout=3.0)
        if tracker.is_alive():
            tracker.terminate()
        if camera is not None:
            try:
                camera.proc.wait(timeout=10.0)
            except Exception:
                camera.proc.terminate()
        metadata["episode_count"] = episode_index
        (session_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(
            f"[SESSION] done. {episode_index} episode(s). "
            f"Metadata: {session_dir / 'metadata.json'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
