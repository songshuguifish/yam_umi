"""Calibrate RealSense RGB camera roles and write a serial-role config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .realsense_rgb_record import (
    DEFAULT_CAMERA_CONFIG_PATH,
    METADATA_FIELDS,
    _device_info,
    _devices,
    _frame_metadata,
    _import_cv2,
    _import_realsense,
    _timestamp_domain,
)


DEFAULT_ROLES = ("left_cam", "right_cam", "egocam")


def _capture_samples(
    rs,
    cv2,
    devices: list,
    output_dir: Path,
    *,
    width: int,
    height: int,
    fps: int,
    warmup_frames: int,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for observed_index, dev in enumerate(devices):
        info = _device_info(rs, dev)
        serial = info.get("serial_number", f"camera_{observed_index}")
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        pipeline = rs.pipeline()
        try:
            pipeline.start(cfg)
            color_frame = None
            for _ in range(max(warmup_frames, 1)):
                frameset = pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frameset.get_color_frame()
            if color_frame is None:
                raise RuntimeError(f"No color frame from RealSense {serial}")
            image = np.asanyarray(color_frame.get_data())
            sample_path = output_dir / f"cam{observed_index}_{serial}.png"
            cv2.imwrite(str(sample_path), image)
            metadata_support = {
                "frame_counter": not np.isnan(
                    _frame_metadata(rs, color_frame, "frame_counter")
                ),
                **{
                    metadata_name: not np.isnan(
                        _frame_metadata(rs, color_frame, metadata_name)
                    )
                    for metadata_name in METADATA_FIELDS.values()
                },
            }
            # Factory color intrinsics from the active stream profile.
            intr = (
                pipeline.get_active_profile()
                .get_stream(rs.stream.color)
                .as_video_stream_profile()
                .get_intrinsics()
            )
            intrinsics = {
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "ppx": float(intr.ppx),
                "ppy": float(intr.ppy),
                "coeffs": [float(c) for c in intr.coeffs],
                "width": int(intr.width),
                "height": int(intr.height),
            }
            rows.append(
                {
                    "observed_camera_index": observed_index,
                    "serial_number": serial,
                    "model": info.get("name", ""),
                    "firmware_version": info.get("firmware_version", ""),
                    "product_line": info.get("product_line", ""),
                    "timestamp_domain": _timestamp_domain(color_frame),
                    "metadata": metadata_support,
                    "intrinsics": intrinsics,
                    "sample_path": str(sample_path),
                }
            )
        finally:
            pipeline.stop()
    return rows


def _print_rows(rows: list[dict]) -> None:
    print("[RS-CAL] Captured samples:")
    for row in rows:
        supported = ", ".join(
            name for name, ok in row["metadata"].items() if ok
        ) or "none"
        print(
            "  cam{idx}: serial={serial} fw={fw} domain={domain} "
            "fx={fx:.1f} fy={fy:.1f} metadata=[{metadata}] sample={sample}".format(
                idx=row["observed_camera_index"],
                serial=row["serial_number"],
                fw=row["firmware_version"],
                domain=row["timestamp_domain"],
                fx=row["intrinsics"]["fx"],
                fy=row["intrinsics"]["fy"],
                metadata=supported,
                sample=row["sample_path"],
            )
        )


def _select_role(role: str, rows: list[dict]) -> dict:
    by_index = {str(row["observed_camera_index"]): row for row in rows}
    by_serial = {row["serial_number"]: row for row in rows}
    while True:
        value = input(f"[RS-CAL] {role} serial or cam index: ").strip()
        row = by_index.get(value) or by_serial.get(value)
        if row is not None:
            return row
        print("[RS-CAL] Unknown camera. Enter one of:")
        for candidate in rows:
            print(
                f"  {candidate['observed_camera_index']}  "
                f"{candidate['serial_number']}  {candidate['sample_path']}"
            )


def _build_config(rows_by_role: dict[str, dict], *, width: int, height: int, fps: int) -> dict:
    return {
        "schema": "realsense_camera_roles_v2",
        "recording_host": "same_host",
        "capture": {
            "streams": ["rgb", "metadata"],
            "width": width,
            "height": height,
            "fps": fps,
        },
        "roles": {
            role: {
                "serial_number": row["serial_number"],
                "observed_camera_index": row["observed_camera_index"],
                "model": row["model"],
                "product_line": row["product_line"],
                "firmware_version": row["firmware_version"],
                "timestamp_domain": row["timestamp_domain"],
                "metadata": row["metadata"],
                "intrinsics": row["intrinsics"],
                "sample_path": row["sample_path"],
            }
            for role, row in rows_by_role.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-cameras", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=Path("data/realsense_camera_calibration"),
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=DEFAULT_CAMERA_CONFIG_PATH,
    )
    parser.add_argument("--roles", nargs="+", default=list(DEFAULT_ROLES))
    args = parser.parse_args()

    rs = _import_realsense()
    cv2 = _import_cv2()
    devices = _devices(rs)
    if args.max_cameras is not None:
        devices = devices[: args.max_cameras]
    if not devices:
        raise SystemExit("ERROR: no RealSense devices found.")

    run_dir = args.sample_dir / time.strftime("calib_%Y%m%d_%H%M%S")
    rows = _capture_samples(
        rs,
        cv2,
        devices,
        run_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
    )
    _print_rows(rows)

    rows_by_role = {}
    used_serials = set()
    for role in args.roles:
        row = _select_role(role, rows)
        if row["serial_number"] in used_serials:
            raise SystemExit(f"ERROR: duplicate camera selected for role {role}")
        used_serials.add(row["serial_number"])
        rows_by_role[role] = row

    config = _build_config(rows_by_role, width=args.width, height=args.height, fps=args.fps)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    print(f"[RS-CAL] Wrote {args.output_config}")


if __name__ == "__main__":
    main()
