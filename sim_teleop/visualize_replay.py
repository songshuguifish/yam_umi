"""Replay a recorded collect_smoke session in Rerun.

Shows, on one scrubable timeline:
  - the 3 RealSense RGB streams (left_cam / right_cam / egocam) as image panels,
  - the Vive tracker poses as 3D coordinate triads in the SteamVR world frame,
  - the gripper raw value as a scalar curve.

All streams are logged on a shared "time" timeline (wall-clock seconds from the
shared Timebase), so cameras (30 Hz), tracker (120 Hz), and encoder (30 Hz) stay
aligned as you scrub.

Cameras with a mount (parent_tracker_role + intrinsics) in the mounts config are
nested under their tracker — a Pinhole frustum + the RGB image follow the arm in
the 3D view. Cameras without a mount (e.g. egocam) are shown as standalone 2D
image panels. Mounts are best-guess starting values; verify placement in the
viewer and iterate. See sim_teleop/configs/camera_mounts.json.

Run from the repository root:

    & ".venv\\Scripts\\python.exe" -m sim_teleop.visualize_replay
    & ".venv\\Scripts\\python.exe" -m sim_teleop.visualize_replay data/raw_smoke/session_...
    & ".venv\\Scripts\\python.exe" -m sim_teleop.visualize_replay --save replay.rrd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("data/raw_smoke")
DEFAULT_MOUNTS_PATH = Path(__file__).resolve().parent / "configs" / "camera_mounts.json"
DEFAULT_CAMERA_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "realsense_cameras.json"
)


def _latest_session(root: Path) -> Path:
    sessions = sorted(root.glob("session_*"))
    if not sessions:
        raise SystemExit(f"No session_* under {root}")
    return sessions[-1]


def _load_cameras(cameras_dir: Path) -> list[dict]:
    meta = json.loads((cameras_dir / "metadata.json").read_text(encoding="utf-8"))
    cams = []
    for entry in meta.get("cameras", []):
        idx = entry["camera_index"]
        role = entry.get("role") or f"cam{idx}"
        cam_dir = cameras_dir / f"cam{idx}"
        ts_path = cam_dir / "color_timestamps.npy"
        video_path = cam_dir / "color.mp4"
        if not ts_path.exists() or not video_path.exists():
            continue
        ts = np.load(ts_path).astype(float)
        cams.append(
            {
                "role": role,
                "idx": idx,
                "ts": ts,
                "video": video_path,
            }
        )
    return cams


def _entity(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in label)


def _load_mounts(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("cameras", {})


def _load_intrinsics(path: Path) -> dict[str, dict]:
    """role -> intrinsics dict, from realsense_cameras.json (camera property)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        role: cfg["intrinsics"]
        for role, cfg in data.get("roles", {}).items()
        if "intrinsics" in cfg
    }


def _camera_base_path(role: str, mount: dict | None) -> str:
    """Entity path for a camera: nested under its tracker if mounted, else standalone."""
    ent = _entity(role)
    parent = mount.get("parent_tracker_role") if mount else None
    if parent:
        return f"world/{_entity(parent)}/{ent}"
    return f"cams/{ent}"


def _log_trackers(rr, tracker: dict, axis_length: float) -> None:
    from scipy.spatial.transform import Rotation

    ts = tracker["timestamp"]
    role_keys = [
        ("left_eef", "left_eef_pose", "left_eef_valid"),
        ("right_eef", "right_eef_pose", "right_eef_valid"),
    ]
    for role, pose_key, valid_key in role_keys:
        if pose_key not in tracker.files:
            continue
        poses = tracker[pose_key]  # (N, 4, 4)
        valid = tracker[valid_key]
        ent = _entity(role)
        logged = 0
        for i in range(len(ts)):
            if not int(valid[i]):
                continue
            t = poses[i][:3, 3].tolist()
            q = Rotation.from_matrix(poses[i][:3, :3]).as_quat(scalar_first=False).tolist()
            rr.set_time("time", timestamp=float(ts[i]))
            rr.log(
                f"world/{ent}",
                rr.Transform3D(translation=t, quaternion=rr.Quaternion(xyzw=q)),
                rr.TransformAxes3D(axis_length),
            )
            logged += 1
        print(f"[REPLAY] {role}: logged {logged}/{len(ts)} poses", flush=True)


def _log_cameras(
    rr,
    cv2,
    cams: list[dict],
    mounts: dict[str, dict],
    intrinsics_by_role: dict[str, dict],
    jpeg_quality: int,
    image_plane_distance: float,
) -> None:
    for cam in cams:
        role = cam["role"]
        mount = mounts.get(role)
        parent = mount.get("parent_tracker_role") if mount else None
        intr = intrinsics_by_role.get(role)
        base = _camera_base_path(role, mount)
        image_path = f"{base}/image" if parent else base

        cap = cv2.VideoCapture(str(cam["video"]))
        if not cap.isOpened():
            print(f"[REPLAY] {role}: could not open {cam['video']}", flush=True)
            continue
        ts = cam["ts"]
        static_logged = False
        i = 0
        while i < len(ts):
            ok, frame = cap.read()
            if not ok:
                break
            rr.set_time("time", timestamp=float(ts[i]))
            # Log the static mount transform + Pinhole once (at the first frame),
            # inside the time stream so the viewer renders the frustum.
            if not static_logged:
                if parent:
                    wxyz = mount["rotation_wxyz"]
                    quat_xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
                    rr.log(
                        base,
                        rr.Transform3D(
                            translation=mount["translation_m"],
                            quaternion=rr.Quaternion(xyzw=quat_xyzw),
                        ),
                    )
                if parent and intr:
                    k = [
                        [intr["fx"], 0.0, intr["ppx"]],
                        [0.0, intr["fy"], intr["ppy"]],
                        [0.0, 0.0, 1.0],
                    ]
                    rr.log(
                        base,
                        rr.Pinhole(
                            image_from_camera=k,
                            width=intr["width"],
                            height=intr["height"],
                            camera_xyz=rr.ViewCoordinates.RDF,
                            image_plane_distance=image_plane_distance,
                        ),
                    )
                static_logged = True
            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            )
            if ok:
                rr.log(
                    image_path,
                    rr.EncodedImage(contents=bytes(buf), media_type="image/jpeg"),
                )
            i += 1
        cap.release()
        place = f"under world/{_entity(parent)}" if parent else "standalone"
        print(f"[REPLAY] {role}: logged {i} frames ({place})", flush=True)


def _log_gripper(rr, encoder: dict) -> None:
    ts = encoder["timestamp"]
    raw = encoder["raw"]
    valid = encoder["valid"] if "valid" in encoder.files else np.ones_like(raw)
    for i in range(len(ts)):
        rr.set_time("time", timestamp=float(ts[i]))
        rr.log(
            "gripper/raw",
            rr.Scalars(float(raw[i]) if int(valid[i]) else float("nan")),
        )


def run(
    session: Path,
    *,
    mounts_path: Path,
    camera_config_path: Path,
    axis_length: float,
    jpeg_quality: int,
    image_plane_distance: float,
    save: Path | None,
) -> None:
    import rerun as rr
    import cv2
    from rerun import blueprint as rrb

    cameras_dir = session / "cameras"
    lowdim = session / "lowdim"
    cams = _load_cameras(cameras_dir)
    mounts = _load_mounts(mounts_path)
    intrinsics = _load_intrinsics(camera_config_path)
    tracker = (
        np.load(lowdim / "tracker.npz") if (lowdim / "tracker.npz").exists() else None
    )
    enc_files = sorted(lowdim.glob("encoder*.npz"))
    encoder = np.load(enc_files[0]) if enc_files else None

    if not cams and tracker is None:
        raise SystemExit(f"No camera or tracker data found in {session}")

    views = []
    if tracker is not None:
        views.append(
            rrb.Spatial3DView(origin="/", contents="world/**", name="Trackers + cams 3D")
        )
    for cam in cams:
        base = _camera_base_path(cam["role"], mounts.get(cam["role"]))
        views.append(
            rrb.Spatial2DView(origin=base, contents=f"{base}/**", name=cam["role"])
        )
    if encoder is not None:
        views.append(
            rrb.TimeSeriesView(origin="gripper", contents="gripper/**", name="Gripper")
        )
    blueprint = rrb.Blueprint(*views)

    rr.init("yam_umi/replay", spawn=True, default_blueprint=blueprint)

    # SteamVR TrackingUniverseStanding is right-handed with +Y up.
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    print(f"[REPLAY] session: {session}", flush=True)
    print(f"[REPLAY] mounts: {mounts_path} ({len(mounts)} cameras)", flush=True)
    print(
        f"[REPLAY] intrinsics: {camera_config_path} ({len(intrinsics)} cameras)",
        flush=True,
    )
    if tracker is not None:
        _log_trackers(rr, tracker, axis_length)
    _log_cameras(rr, cv2, cams, mounts, intrinsics, jpeg_quality, image_plane_distance)
    if encoder is not None:
        _log_gripper(rr, encoder)
        print(f"[REPLAY] gripper: logged {len(encoder['timestamp'])} samples", flush=True)

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(save))
        print(f"[REPLAY] saved -> {save}", flush=True)

    print("[REPLAY] viewer launched; scrub the bottom timeline to replay.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Session directory. If omitted, auto-pick the latest under --root.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Search root for auto-picking the latest session.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.08,
        help="Tracker triad axis length in meters.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for logged camera frames (1-100).",
    )
    parser.add_argument(
        "--image-plane-distance",
        type=float,
        default=0.3,
        help="Distance (m) of the Pinhole image plane in the 3D view — "
        "visualization-only; larger = bigger frustum/image. "
        "Image-plane width ~= dist * width / fx (~0.49m at 0.3).",
    )
    parser.add_argument(
        "--mounts-config",
        type=Path,
        default=DEFAULT_MOUNTS_PATH,
        help="camera_mounts.json with tracker mount transforms.",
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=DEFAULT_CAMERA_CONFIG_PATH,
        help="realsense_cameras.json with camera intrinsics.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Also save the recording to an .rrd file at this path.",
    )
    args = parser.parse_args()

    session = args.input or _latest_session(args.root)
    run(
        session,
        mounts_path=args.mounts_config,
        camera_config_path=args.camera_config,
        axis_length=args.axis_length,
        jpeg_quality=args.jpeg_quality,
        image_plane_distance=args.image_plane_distance,
        save=args.save,
    )


if __name__ == "__main__":
    main()
