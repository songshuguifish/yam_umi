"""Hot-start sensors once, then record episodes with c/q hotkeys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

from gripper.encoder import EncoderCalibration

from .calibrate_encoders import DEFAULT_CONFIG_PATH as DEFAULT_ENCODER_CONFIG_PATH
from .calibrate_encoders import resolve_ports
from .encoder_process import EncoderProcess, encoder_sample_example
from .ring_buffer import SharedMemoryRingBuffer
from .timebase import Timebase
from .tracker_process import TrackerProcess, tracker_sample_example

try:
    from .mochi_status import MochiStatus
    from .mochi_status import push as mochi_push
except Exception:
    class MochiStatus:
        IDLE = "idle"
        WORKING = "working"
        WAITING = "waiting"
        DONE = "done"
        ERROR = "error"

    def mochi_push(state: str) -> None:
        return None


DEFAULT_REALSENSE_PYTHON = Path(sys.executable)
SESSION_METADATA_README = """# Session metadata guide

This session directory contains one hot-start recording run driven by
`collect_session`.

## Layout

- `metadata.json`: session-level summary, sensor config, and episode list.
- `episode_*/metadata.json`: one episode summary per recording.
- `episode_*/cameras/README_metadata.md`: camera-side file guide.
- `episode_*/lowdim/*.npz`: tracker and encoder samples sliced by episode time.

## Important session fields

- `timebase.wall0` / `timebase.perf0`: shared host-time anchor used by every
  process in this session.
- `encoder_mapping`: resolved encoder role/port/calibration config.
- `encoder_resolved_ports`: role to COM-port mapping actually used at runtime.
- `encoder_raw_only`: true means encoder `normalized` and `metric` are NaN.
- `episodes`: list of saved episodes. Each entry contains `t_start`, `t_stop`,
  `duration`, camera frame count, and lowdim sample counts.

## How to read an episode

1. Open `episode_NNN/metadata.json` for the episode window.
2. Open `episode_NNN/cameras/metadata.json` for camera timing details.
3. Use `episode_NNN/cameras/cam*/color_timestamps.npy` as the primary video
   timeline.
4. Use `episode_NNN/lowdim/tracker.npz` and `encoder_*.npz` to align lowdim
   data onto the camera timestamps.

The `color_timestamps.npy` arrays and the lowdim `timestamp` arrays share the
same host timebase, so they can be aligned directly.
"""


def _session_dir(root: Path) -> Path:
    session = root / time.strftime("session_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=False)
    return session


def _save_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def _write_metadata_readme(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


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


def _flush_pending_keys() -> None:
    try:
        import msvcrt
    except ImportError:
        return
    while msvcrt.kbhit():
        msvcrt.getwch()


def _read_key(prompt: str, valid: set[str]) -> str:
    _flush_pending_keys()
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


def _play_prompt_sound(kind: str, *, enabled: bool = True) -> None:
    if not enabled:
        return

    patterns = {
        "start": [(1200, 120), (1500, 120)],
        "stop": [(520, 260)],
        "error": [(420, 120), (420, 120), (420, 120)],
    }
    pattern = patterns[kind]

    def worker() -> None:
        try:
            import winsound

            for freq, duration_ms in pattern:
                winsound.Beep(freq, duration_ms)
                time.sleep(0.04)
        except Exception:
            print("\a", end="", flush=True)

    threading.Thread(target=worker, daemon=True).start()


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
    parser.set_defaults(encoder_raw_only=True)
    parser.add_argument(
        "--encoder-raw-only",
        dest="encoder_raw_only",
        action="store_true",
        help="Record encoder raw values only; normalized/metric are NaN (default).",
    )
    parser.add_argument(
        "--encoder-normalize",
        dest="encoder_raw_only",
        action="store_false",
        help="Compute encoder normalized/metric values from calibration.",
    )
    parser.add_argument("--realsense-python", type=Path, default=DEFAULT_REALSENSE_PYTHON)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument(
        "--no-cue-sound",
        action="store_true",
        help="Disable start/stop system prompt sounds.",
    )
    parser.add_argument(
        "--no-mochi",
        action="store_true",
        help="Disable Clawd Mochi status bridge (recording-state reflection).",
    )
    args = parser.parse_args()
    cue_sound = not args.no_cue_sound
    mochi_enabled = not args.no_mochi

    def mochi(state: str) -> None:
        if mochi_enabled:
            mochi_push(state)

    session_dir = _session_dir(args.output_dir)
    _write_metadata_readme(session_dir / "README_metadata.md", SESSION_METADATA_README)
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
        calibration = (
            EncoderCalibration()
            if args.encoder_raw_only
            else EncoderCalibration(
                raw_closed=cal_data.get("raw_closed"),
                raw_open=cal_data.get("raw_open"),
                stroke_mm=stroke_mm,
            )
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
            raw_only=args.encoder_raw_only,
            side=_role_label(role),
        )
        cal_text = "RAW_ONLY" if args.encoder_raw_only else str(calibration)
        print(f"[SESSION] encoder {role}: port={port} {cal_text}", flush=True)

    tracker_ring = SharedMemoryRingBuffer(tracker_sample_example(), cap_trk)
    tracker = TrackerProcess(tracker_ring, timebase, frequency=args.tracker_frequency)
    camera: _CameraServer | None = None
    episode_index = 0
    metadata = {
        "schema": "hotkey_session_v1",
        "session_dir": str(session_dir),
        "metadata_readme": "README_metadata.md",
        "timebase": {"wall0": timebase.wall0, "perf0": timebase.perf0},
        "encoder_mapping": encoder_config,
        "encoder_resolved_ports": {role: port for role, port in resolved_ports},
        "encoder_raw_only": args.encoder_raw_only,
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
        "field_descriptions": {
            "session_dir": "Absolute session root path.",
            "timebase": "Shared host-time anchor used by cameras, tracker, and encoders.",
            "encoder_mapping": "Loaded encoder role/port/calibration config.",
            "encoder_resolved_ports": "Runtime-resolved role to serial-port mapping.",
            "encoder_raw_only": "True when only raw encoder values were recorded.",
            "encoder_frequency": "Encoder polling rate in Hz.",
            "tracker_frequency": "Tracker polling rate in Hz.",
            "camera": "Camera configuration used for this session, or null.",
            "episodes": "Episode summaries appended as each recording is saved.",
        },
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
        print("[SESSION] all sensors hot. c=开始录制, q=退出/结束录制", flush=True)
        mochi(MochiStatus.IDLE)

        while True:
            mochi(MochiStatus.WAITING)
            cmd = _read_key(
                f"\n[SESSION] episode {episode_index}: 按 c 开始录制, 按 q 退出",
                {"c", "q"},
            )
            if cmd == "q":
                break

            ep_dir = session_dir / f"episode_{episode_index:03d}"
            cameras_dir = ep_dir / "cameras"
            t_start = timebase.now()
            _play_prompt_sound("start", enabled=cue_sound)
            print("[SESSION] 开始录制", flush=True)
            mochi(MochiStatus.WORKING)
            enc_count_start = {role: ring.count for role, ring in encoder_rings.items()}
            trk_count_start = tracker_ring.count

            if camera is not None and not camera.start_episode(
                cameras_dir, timeout_s=15.0
            ):
                print("[SESSION] WARNING: camera did not confirm START.", flush=True)

            _read_key(
                "[SESSION] recording... 按 q 结束录制并保存",
                {"q"},
            )
            t_stop = timebase.now()
            _play_prompt_sound("stop", enabled=cue_sound)
            print("[SESSION] 结束录制，正在保存...", flush=True)

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
                "metadata_readme": "cameras/README_metadata.md",
                "t_start": t_start,
                "t_stop": t_stop,
                "duration": t_stop - t_start,
                "camera_frames": cam_frames,
                "encoder_samples": {},
                "field_descriptions": {
                    "episode": "Zero-based episode index.",
                    "t_start": "Shared host-time timestamp when recording started.",
                    "t_stop": "Shared host-time timestamp when recording stopped.",
                    "duration": "Episode duration in seconds.",
                    "camera_frames": "Number of frames written by the camera subprocess.",
                    "encoder_samples": "Per-encoder sample counts saved for this episode.",
                    "tracker_samples": "Tracker sample count saved for this episode.",
                },
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
            mochi(MochiStatus.DONE)
            episode_index += 1
    except KeyboardInterrupt:
        print("\n[SESSION] interrupted.", flush=True)
    finally:
        _exc = sys.exc_info()[1]
        mochi(
            MochiStatus.ERROR
            if _exc is not None and not isinstance(_exc, KeyboardInterrupt)
            else MochiStatus.IDLE
        )
        if camera is not None:
            camera.quit()
        for proc in encoders.values():
            if proc.pid is not None:
                proc.stop()
        if tracker.pid is not None:
            tracker.stop()
        for proc in encoders.values():
            if proc.pid is None:
                continue
            proc.join(timeout=3.0)
            if proc.is_alive():
                proc.terminate()
        if tracker.pid is not None:
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
