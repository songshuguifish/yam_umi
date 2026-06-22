"""Analyze timestamp sync across recorded RealSense RGB camera streams.

Three diagnostics on a recorded session (standalone cameras dir or a
``collect_smoke`` session directory):

  1. Per-camera self-check — host fps/jitter, timestamp domain, dropped frames
     (via ``frame_counter``), and ``global_time`` vs ``host_mid`` epoch offset.
  2. Cross-camera nearest-frame skew for a chosen clock (``--kind``).
  3. Camera <-> tracker / encoder alignment from ``lowdim/*.npz`` when present.

All wall-clock sources (host midpoint, ``global_time``, ``time_of_arrival``,
``backend_timestamp``) share one epoch on a single host, so absolute deltas
between them are meaningful. ``frame_timestamp`` / ``sensor_timestamp`` are
per-device microsecond uptime counters and are only meaningful per-stream.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# kind -> (filename, scale to seconds, note)
KIND_FILES: dict[str, tuple[str, float, str]] = {
    "color": ("color_timestamps.npy", 1.0, "host midpoint, seconds"),
    "device": ("device_timestamps.npy", 1.0, "global_time, seconds"),
    "time-of-arrival": ("metadata_time_of_arrival.npy", 1e-3, "ms -> seconds, wall-clock"),
    "backend-timestamp": ("metadata_backend_timestamp.npy", 1e-3, "ms -> seconds, wall-clock"),
    "frame-timestamp": ("metadata_frame_timestamp.npy", 1e-6, "us -> seconds, per-device uptime"),
    "sensor-timestamp": ("metadata_sensor_timestamp.npy", 1e-6, "us -> seconds, per-device uptime"),
}

# Absolute timestamps of these are NOT comparable across cameras (per-device
# free-running counters); only per-stream dt is meaningful.
PER_DEVICE_KINDS = {"frame-timestamp", "sensor-timestamp"}


def _resolve_cameras_dir(path: Path) -> Path:
    if (path / "cameras").is_dir():
        return path / "cameras"
    return path


def _session_dir(cameras_dir: Path) -> Path:
    return cameras_dir.parent if cameras_dir.name == "cameras" else cameras_dir


def _load_kind(cameras_dir: Path, kind: str) -> dict[str, np.ndarray]:
    fname, scale, _ = KIND_FILES[kind]
    result: dict[str, np.ndarray] = {}
    for cam_dir in sorted(cameras_dir.glob("cam*")):
        ts_path = cam_dir / fname
        if not ts_path.exists():
            continue
        ts = np.load(ts_path).astype(float)
        if len(ts) > 0:
            result[cam_dir.name] = ts * scale
    return result


def _load_str(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [str(x) for x in np.load(path, allow_pickle=True)]


def _nearest_delta_ms(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(other, ref)
    idx0 = np.clip(idx - 1, 0, len(other) - 1)
    idx1 = np.clip(idx, 0, len(other) - 1)
    d0 = other[idx0] - ref
    d1 = other[idx1] - ref
    use1 = np.abs(d1) < np.abs(d0)
    delta = np.where(use1, d1, d0)
    return delta * 1000.0


def _stats_ms(delta_ms: np.ndarray) -> dict[str, float]:
    abs_delta = np.abs(delta_ms)
    return {
        "signed_mean_ms": float(np.mean(delta_ms)),
        "signed_std_ms": float(np.std(delta_ms)),
        "abs_mean_ms": float(np.mean(abs_delta)),
        "abs_p95_ms": float(np.percentile(abs_delta, 95)),
        "abs_max_ms": float(np.max(abs_delta)),
    }


def _fps(ts: np.ndarray) -> float:
    if len(ts) <= 1:
        return 0.0
    return float((len(ts) - 1) / (ts[-1] - ts[0]))


def _fmt_stats(stats: dict[str, float]) -> str:
    return (
        f"signed_mean={stats['signed_mean_ms']:+.2f}ms "
        f"signed_std={stats['signed_std_ms']:.2f}ms "
        f"abs_mean={stats['abs_mean_ms']:.2f}ms "
        f"abs_p95={stats['abs_p95_ms']:.2f}ms "
        f"abs_max={stats['abs_max_ms']:.2f}ms"
    )


def _per_stream(cameras_dir: Path) -> None:
    print("[SELF] Per-stream health (host_mid clock):")
    any_found = False
    for cam_dir in sorted(cameras_dir.glob("cam*")):
        color_p = cam_dir / "color_timestamps.npy"
        if not color_p.exists():
            continue
        any_found = True
        color = np.load(color_p).astype(float)
        dt_ms = np.diff(color) * 1000.0 if len(color) > 1 else np.array([])
        jitter = (
            f"dt_mean={dt_ms.mean():.2f}ms dt_std={dt_ms.std():.2f}ms"
            if len(dt_ms)
            else "dt=n/a"
        )
        domain = _load_str(cam_dir / "timestamp_domain.npy")
        domain_str = domain[0] if domain else "n/a"
        drops = _dropped_frames(cam_dir)
        print(
            f"  {cam_dir.name}: frames={len(color)} fps={_fps(color):.2f} "
            f"domain={domain_str} drops={drops} {jitter}"
        )
    if not any_found:
        print("  (no color_timestamps.npy found)")


def _dropped_frames(cam_dir: Path) -> int | str:
    fc_p = cam_dir / "frame_counter.npy"
    if not fc_p.exists():
        return "n/a"
    fc = np.load(fc_p).astype(float)
    if len(fc) < 2 or np.isnan(fc).any():
        return "n/a"
    return int(fc[-1] - fc[0]) - (len(fc) - 1)


def _epoch_check(cameras_dir: Path) -> None:
    print("[SELF] global_time vs host_mid (epoch consistency / pull latency):")
    found = False
    for cam_dir in sorted(cameras_dir.glob("cam*")):
        color_p = cam_dir / "color_timestamps.npy"
        dev_p = cam_dir / "device_timestamps.npy"
        if not color_p.exists() or not dev_p.exists():
            continue
        found = True
        color = np.load(color_p).astype(float)
        dev = np.load(dev_p).astype(float)
        n = min(len(color), len(dev))
        offset = (color[:n] - dev[:n]) * 1000.0
        print(
            f"  {cam_dir.name}: host_mid-device mean={np.mean(offset):+.1f}ms "
            f"std={np.std(offset):.2f}ms (n={n})"
        )
    if not found:
        print("  (need color_timestamps.npy + device_timestamps.npy)")


def _cross_camera(timestamps: dict[str, np.ndarray], kind: str) -> None:
    note = KIND_FILES[kind][2]
    print(f"[CROSS] clock={kind} ({note})")
    if not timestamps:
        print(f"  No '{kind}' timestamp files found.")
        return
    names = list(timestamps)
    print(f"  cameras={names}")
    if kind in PER_DEVICE_KINDS:
        print(
            f"  NOTE: '{kind}' is a per-device uptime counter; absolute "
            "cross-camera deltas are NOT meaningful. Skipping (see per-stream dt above)."
        )
        return
    if len(names) < 2:
        print("  Need >=2 cameras for cross-camera skew.")
        return
    ref_name = names[0]
    ref = timestamps[ref_name]
    print(f"  Reference: {ref_name}")
    for name in names[1:]:
        delta_ms = _nearest_delta_ms(ref, timestamps[name])
        print(f"  {name} vs {ref_name}: {_fmt_stats(_stats_ms(delta_ms))}")


def _multimodal(cameras_dir: Path) -> None:
    lowdim = _session_dir(cameras_dir) / "lowdim"
    color = {
        d.name: np.load(d / "color_timestamps.npy").astype(float)
        for d in sorted(cameras_dir.glob("cam*"))
        if (d / "color_timestamps.npy").exists()
    }
    if not lowdim.is_dir() or not color:
        if not lowdim.is_dir():
            print("[MIXED] No lowdim/ directory; skipping camera<->tracker/encoder.")
        return

    ref_name = next(iter(color))
    cam_ts = color[ref_name]
    print(f"[MIXED] Camera<->lowdim alignment (ref={ref_name}, clock=host_mid, shared Timebase)")

    def _align(other: np.ndarray, label: str) -> None:
        lo = max(cam_ts.min(), other.min())
        hi = min(cam_ts.max(), other.max())
        cam_o = cam_ts[(cam_ts >= lo) & (cam_ts <= hi)]
        o_o = other[(other >= lo) & (other <= hi)]
        print(
            f"  {ref_name} vs {label}: overlap={hi - lo:.2f}s "
            f"cam_n={len(cam_o)} {label}_n={len(o_o)}"
        )
        if len(cam_o) and len(o_o) > 1:
            print(f"    {_fmt_stats(_stats_ms(_nearest_delta_ms(cam_o, o_o)))}")

    trk_p = lowdim / "tracker.npz"
    if trk_p.exists():
        trk = np.load(trk_p)
        ts = trk["timestamp"]
        lv = float(trk["left_eef_valid"].mean())
        rv = float(trk["right_eef_valid"].mean())
        print(
            f"  tracker: n={len(ts)} valid_frac left={lv:.2f} right={rv:.2f}"
        )
        _align(ts, "tracker")
    for enc_p in sorted(lowdim.glob("encoder*.npz")):
        enc = np.load(enc_p)
        ts = enc["timestamp"]
        read_ms = (enc["read_end_timestamp"] - enc["read_start_timestamp"]) * 1000.0
        print(f"  {enc_p.stem}: n={len(ts)} read_mean={np.mean(read_ms):.1f}ms")
        _align(ts, enc_p.stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Session directory, its cameras/ directory, or a standalone cameras dir.",
    )
    parser.add_argument(
        "--kind",
        choices=list(KIND_FILES),
        default="device",
        help=(
            "Clock for cross-camera skew (default: device/global_time). "
            "color=device host_mid; device/time-of-arrival/backend-timestamp are "
            "wall-clock (cross-camera OK); frame/sensor-timestamp are per-device "
            "us counters (cross-camera skipped)."
        ),
    )
    args = parser.parse_args()

    cameras_dir = _resolve_cameras_dir(args.input)
    print(f"[SYNC] cameras_dir={cameras_dir}")
    print(f"[SYNC] kind={args.kind}\n")

    _per_stream(cameras_dir)
    print()
    _epoch_check(cameras_dir)
    print()
    _cross_camera(_load_kind(cameras_dir, args.kind), args.kind)
    print()
    _multimodal(cameras_dir)


if __name__ == "__main__":
    main()
