"""RGB-only RealSense diagnostic for the data-collection stack."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np


def _import_realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "pyrealsense2 is not installed in this Python environment. "
            "Try: third_party\\HuMI-main\\.realsense-env\\Scripts\\python.exe "
            "-m sim_teleop.data_collection.realsense_rgb_check"
        ) from exc
    return rs


def _import_cv2():
    try:
        import cv2
    except ImportError:
        return None
    return cv2


def _devices(rs, *, attempts: int = 3, backoff_s: float = 2.0) -> list:
    """Enumerate connected RealSense devices, skipping platform cameras.

    ``query_devices()`` can transiently raise ``RuntimeError`` when a device
    is mid-recovery / FW-update (e.g. "Failed to create FW update device").
    Retry with backoff before giving up — the glitch usually clears on the
    next enumeration (a fresh ``rs.context()`` each attempt).
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            devices = []
            for dev in rs.context().query_devices():
                name = dev.get_info(rs.camera_info.name)
                if name.lower() == "platform camera":
                    continue
                devices.append(dev)
            return devices
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(
                f"[RS] device enumeration failed (attempt {attempt}/{attempts}): {exc}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(backoff_s)
    raise RuntimeError(
        f"RealSense device enumeration failed after {attempts} attempts: {last_exc}"
    )


def _device_info(rs, dev) -> dict[str, str]:
    out = {}
    for field in ("name", "product_line", "serial_number", "firmware_version"):
        try:
            out[field] = dev.get_info(getattr(rs.camera_info, field))
        except Exception:
            pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--max-cameras", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/realsense_rgb_smoke"),
    )
    args = parser.parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")

    rs = _import_realsense()
    cv2 = _import_cv2()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    devices = _devices(rs)
    if args.max_cameras is not None:
        devices = devices[: args.max_cameras]
    print(f"[RS] pyrealsense2={getattr(rs, '__version__', 'ok')}", flush=True)
    print(f"[RS] devices={len(devices)}", flush=True)
    if not devices:
        raise SystemExit("ERROR: no RealSense devices found.")

    pipelines = []
    serials = []
    try:
        for idx, dev in enumerate(devices):
            info = _device_info(rs, dev)
            serial = info.get("serial_number", f"camera_{idx}")
            serials.append(serial)
            print(f"[RS] cam{idx} {info}", flush=True)

            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(
                rs.stream.color,
                args.width,
                args.height,
                rs.format.bgr8,
                args.fps,
            )
            pipeline = rs.pipeline()
            pipeline.start(cfg)
            pipelines.append(pipeline)

        time.sleep(1.0)
        last_frames = {}
        for frame_idx in range(args.frames):
            for cam_idx, pipeline in enumerate(pipelines):
                recv_start = time.time()
                frameset = pipeline.wait_for_frames(timeout_ms=5000)
                recv_end = time.time()
                color_frame = frameset.get_color_frame()
                if not color_frame:
                    print(f"[RS] cam{cam_idx} frame={frame_idx} missing color")
                    continue
                image = np.asanyarray(color_frame.get_data())
                device_ts = color_frame.get_timestamp() / 1000.0
                last_frames[cam_idx] = image
                print(
                    "[RS] cam{} frame={} shape={} dtype={} "
                    "host_mid={:.6f} recv_dt_ms={:.2f} device_ts={:.6f}".format(
                        cam_idx,
                        frame_idx,
                        tuple(image.shape),
                        image.dtype,
                        recv_start + 0.5 * (recv_end - recv_start),
                        (recv_end - recv_start) * 1000.0,
                        device_ts,
                    ),
                    flush=True,
                )

        if cv2 is not None:
            for cam_idx, image in last_frames.items():
                out_path = args.output_dir / f"cam{cam_idx}_{serials[cam_idx]}.png"
                cv2.imwrite(str(out_path), image)
                print(f"[RS] saved {out_path}", flush=True)
        else:
            print("[RS] cv2 not installed; skipping PNG save.", flush=True)
    finally:
        for pipeline in pipelines:
            pipeline.stop()
        print("[RS] stopped", flush=True)


if __name__ == "__main__":
    main()
