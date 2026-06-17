"""Interactive Rerun visualization of recorded Vive Tracker pose episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_ROOT = Path("data/tracker_poses")
_AXIS_COLORS = ([220, 40, 40], [40, 200, 60], [60, 120, 255])  # X Y Z


def latest_episode(root: Path) -> Path:
    files = sorted(root.rglob("tracker_recording_*.json"))
    if not files:
        raise SystemExit(f"No tracker_recording_*.json under {root}")
    return files[-1]


def _label_for_tracker(tracker: dict) -> str:
    return str(tracker.get("role") or tracker["serial"])


def _entity_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in label)


def _frame_pose(frame: dict, tracker_role: str | None, frame_idx: int) -> list:
    if tracker_role is None:
        return frame["tracker_pose"]
    poses_by_role = frame.get("poses_by_role", {})
    if tracker_role in poses_by_role:
        return poses_by_role[tracker_role]
    for tracker in frame.get("trackers", []):
        if tracker.get("role") == tracker_role or tracker["serial"] == tracker_role:
            return tracker["tracker_pose"]
    raise KeyError(f"Frame {frame_idx} has no tracker role/serial {tracker_role!r}")


def load_episode(
    path: Path,
    *,
    tracker_role: str | None,
    all_trackers: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return timestamps and one or more labeled pose arrays."""
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data["episode"]
    if not frames:
        raise SystemExit(f"Empty episode in {path}")
    ts = np.array([f["timestamp"] for f in frames], dtype=float)

    if all_trackers:
        labels = [_label_for_tracker(t) for t in frames[0].get("trackers", [])]
        if not labels:
            raise SystemExit(f"No multi-tracker data in {path}")
        pose_series = {label: [] for label in labels}
        for i, frame in enumerate(frames):
            frame_trackers = {
                _label_for_tracker(tracker): tracker["tracker_pose"]
                for tracker in frame.get("trackers", [])
            }
            for label in labels:
                if label not in frame_trackers:
                    raise KeyError(f"Frame {i} has no tracker {label!r}")
                pose_series[label].append(frame_trackers[label])
        return ts, {
            label: np.array(poses, dtype=float)
            for label, poses in pose_series.items()
        }

    label = tracker_role or "tracker"
    poses = np.array(
        [_frame_pose(frame, tracker_role, i) for i, frame in enumerate(frames)],
        dtype=float,
    )
    return ts, {label: poses}


def poses_to_tum_rows(ts: np.ndarray, poses: np.ndarray) -> str:
    """Convert 4x4 poses to TUM rows: timestamp tx ty tz qx qy qz qw."""
    xyz = poses[:, :3, 3]
    quat_xyzw = Rotation.from_matrix(poses[:, :3, :3]).as_quat(scalar_first=False)
    lines = [
        f"{ts[i]:.6f} {xyz[i, 0]:.6f} {xyz[i, 1]:.6f} {xyz[i, 2]:.6f} "
        f"{quat_xyzw[i, 0]:.6f} {quat_xyzw[i, 1]:.6f} "
        f"{quat_xyzw[i, 2]:.6f} {quat_xyzw[i, 3]:.6f}"
        for i in range(len(ts))
    ]
    return "\n".join(lines) + "\n"


def _log_triad(
    rr,
    path: str,
    pos: np.ndarray,
    quat_xyzw: np.ndarray,
    length: float,
    radius: float,
) -> None:
    """Log an RGB coordinate frame as 3 short segments."""
    rot = Rotation.from_quat(quat_xyzw).as_matrix()
    world_axes = (rot @ (np.eye(3) * length).T).T
    tips = pos + world_axes
    strips = [[pos.tolist(), tips[i].tolist()] for i in range(3)]
    rr.log(path, rr.LineStrips3D(strips, radii=radius, colors=_AXIS_COLORS))


def run_rerun(
    ts: np.ndarray,
    pose_series: dict[str, np.ndarray],
    axis_length: float,
) -> None:
    import rerun as rr
    from rerun import blueprint as rrb

    all_xyz = np.concatenate([poses[:, :3, 3] for poses in pose_series.values()])
    span = float((all_xyz.max(axis=0) - all_xyz.min(axis=0)).max())
    triad_r = max(span * 0.01, 0.003)
    line_r = max(span * 0.014, 0.006)

    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(origin="/", contents="$origin/**", name="Tracker"),
    )
    rr.init("yam_umi/tracker_pose", spawn=True, default_blueprint=blueprint)

    traj_colors = ([80, 200, 255], [255, 120, 80], [220, 220, 80], [200, 120, 255])
    prepared = {}
    for idx, (label, poses) in enumerate(pose_series.items()):
        xyz = poses[:, :3, 3]
        quat_xyzw = Rotation.from_matrix(poses[:, :3, :3]).as_quat(scalar_first=False)
        step = max(1, len(xyz) // 80)
        path = xyz[::step]
        segs = [[path[j].tolist(), path[j + 1].tolist()] for j in range(len(path) - 1)]
        color = traj_colors[idx % len(traj_colors)]
        prepared[label] = {
            "entity": _entity_label(label),
            "xyz": xyz,
            "quat_xyzw": quat_xyzw,
            "segs": segs,
            "seg_colors": [color] * len(segs),
        }

    for i in range(len(ts)):
        rr.set_time("time", timestamp=float(ts[i]))
        rr.set_time("frame", sequence=i)
        for label, item in prepared.items():
            entity = item["entity"]
            xyz = item["xyz"]
            quat_xyzw = item["quat_xyzw"]
            rr.log(
                f"{entity}/trajectory/full",
                rr.LineStrips3D(item["segs"], radii=line_r, colors=item["seg_colors"]),
            )
            _log_triad(
                rr,
                f"{entity}/frames/start",
                xyz[0],
                quat_xyzw[0],
                axis_length * 2.0,
                triad_r * 1.4,
            )
            _log_triad(
                rr,
                f"{entity}/frames/end",
                xyz[-1],
                quat_xyzw[-1],
                axis_length * 2.0,
                triad_r * 1.4,
            )
            _log_triad(
                rr,
                f"{entity}/tracker/current",
                xyz[i],
                quat_xyzw[i],
                axis_length * 1.5,
                triad_r,
            )
            rr.log(f"{entity}/label", rr.TextDocument(label))


def _write_tum_files(
    episode_path: Path,
    tum_out: Path | None,
    ts: np.ndarray,
    pose_series: dict[str, np.ndarray],
) -> list[Path]:
    written = []
    if len(pose_series) == 1:
        _, poses = next(iter(pose_series.items()))
        tum_path = tum_out or episode_path.with_suffix(".tum")
        tum_path.write_text(poses_to_tum_rows(ts, poses), encoding="utf-8")
        return [tum_path]

    if tum_out is not None and tum_out.suffix:
        out_dir = tum_out.parent
        stem = tum_out.stem
    else:
        out_dir = tum_out or episode_path.parent
        stem = episode_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, poses in pose_series.items():
        tum_path = out_dir / f"{stem}_{_entity_label(label)}.tum"
        tum_path.write_text(poses_to_tum_rows(ts, poses), encoding="utf-8")
        written.append(tum_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Episode JSON. If omitted, auto-pick the latest under --root.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Search root for auto-picking the latest episode.",
    )
    parser.add_argument(
        "--tum-out",
        type=Path,
        default=None,
        help="Write TUM trajectory file here. For multiple trackers, use as dir/prefix.",
    )
    parser.add_argument(
        "--tracker-role",
        default=None,
        help="Tracker role or serial to visualize, e.g. left_eef or right_eef.",
    )
    parser.add_argument(
        "--all-trackers",
        action="store_true",
        help="Visualize all trackers in the recording at once.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.06,
        help="Triad axis length in meters.",
    )
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="Only export TUM files; skip the Rerun viewer.",
    )
    args = parser.parse_args()
    if args.all_trackers and args.tracker_role is not None:
        raise SystemExit("Use either --all-trackers or --tracker-role, not both.")

    episode_path = args.input or latest_episode(args.root)
    ts, pose_series = load_episode(
        episode_path,
        tracker_role=args.tracker_role,
        all_trackers=args.all_trackers,
    )
    print(
        f"[VIZ] Loaded {len(ts)} frames ({ts[-1] - ts[0]:.2f}s) from\n"
        f"      {episode_path}"
    )
    print(f"[VIZ] Trackers: {', '.join(pose_series)}")

    tum_paths = _write_tum_files(episode_path, args.tum_out, ts, pose_series)
    for tum_path in tum_paths:
        print(f"[VIZ] Wrote TUM trajectory -> {tum_path}")

    if args.no_rerun:
        print("[VIZ] --no-rerun set; skipping viewer.")
        return

    run_rerun(ts, pose_series, args.axis_length)
    print("[VIZ] Rerun viewer launched; scrub the bottom timeline to replay.")
    print("[VIZ] Use --all-trackers for both hands, or --tracker-role left_eef/right_eef.")


if __name__ == "__main__":
    main()
