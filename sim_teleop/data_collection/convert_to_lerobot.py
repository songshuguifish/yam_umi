"""Convert hot-start raw sessions into a LeRobot v2.1-style dataset.

The converter treats camera host timestamps as the master timeline. Tracker
poses and gripper encoder samples are interpolated onto the selected camera
frames, then packed as:

    [left xyz, left rotation, left gripper, right xyz, right rotation, right gripper]

Tracker poses are recorded as T_world_tracker. The config stores a fixed
T_output_tracker mount transform. First compute:

    T_world_output = T_world_tracker @ inv(T_output_tracker)

For UMI-style datasets, the default config then exports each arm relative to
the first valid frame of that episode:

    T_relative_output(t) = inv(T_world_output(0)) @ T_world_output(t)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation, Slerp


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "lerobot_conversion.json"
DEFAULT_CAMERA_CONFIG = Path(__file__).parents[1] / "configs" / "realsense_cameras.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _episode_dirs(session: Path) -> list[Path]:
    episodes = sorted(
        p for p in session.glob("episode_*") if (p / "lowdim").exists()
    )
    if episodes:
        return episodes
    if (session / "lowdim").exists():
        return [session]
    raise FileNotFoundError(f"No episode lowdim data found under {session}")


def _camera_serial_roles(camera_config: Path) -> dict[str, str]:
    if not camera_config.exists():
        return {}
    data = _load_json(camera_config)
    roles = data.get("roles", {})
    return {
        str(entry["serial_number"]): role
        for role, entry in roles.items()
        if isinstance(entry, dict) and entry.get("serial_number")
    }


def _load_camera_dirs(
    episode: Path,
    *,
    camera_config: Path,
) -> dict[str, Path]:
    cameras_root = episode / "cameras"
    meta_path = cameras_root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing camera metadata: {meta_path}")

    serial_roles = _camera_serial_roles(camera_config)
    meta = _load_json(meta_path)
    out: dict[str, Path] = {}
    for cam in meta.get("cameras", []):
        idx = cam.get("camera_index")
        serial = str(cam.get("serial_number", ""))
        role = serial_roles.get(serial) or cam.get("role")
        if role and idx is not None:
            out[str(role)] = cameras_root / f"cam{idx}"
    return out


def _load_timestamps(cam_dir: Path) -> np.ndarray:
    ts = np.load(cam_dir / "color_timestamps.npy").astype(np.float64)
    if ts.ndim != 1 or len(ts) == 0:
        raise ValueError(f"Bad timestamps in {cam_dir}")
    return ts


def _nearest_indices(source_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(source_ts, target_ts, side="left")
    idx = np.clip(idx, 0, len(source_ts) - 1)
    prev_idx = np.clip(idx - 1, 0, len(source_ts) - 1)
    use_prev = np.abs(source_ts[prev_idx] - target_ts) < np.abs(source_ts[idx] - target_ts)
    return np.where(use_prev, prev_idx, idx)


def _interp_pose(
    ts: np.ndarray,
    poses: np.ndarray,
    valid: np.ndarray,
    target_ts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = valid.astype(bool) & np.isfinite(ts)
    if not np.any(mask):
        return np.repeat(np.eye(4)[None, :, :], len(target_ts), axis=0), np.zeros(len(target_ts), dtype=bool)

    src_ts = ts[mask].astype(np.float64)
    src_poses = poses[mask].astype(np.float64)
    order = np.argsort(src_ts)
    src_ts = src_ts[order]
    src_poses = src_poses[order]

    unique_ts, unique_idx = np.unique(src_ts, return_index=True)
    src_ts = unique_ts
    src_poses = src_poses[unique_idx]
    if len(src_ts) == 1:
        inside = np.ones(len(target_ts), dtype=bool)
        return np.repeat(src_poses[:1], len(target_ts), axis=0), inside

    clipped = np.clip(target_ts, src_ts[0], src_ts[-1])
    xyz = np.column_stack(
        [np.interp(clipped, src_ts, src_poses[:, axis, 3]) for axis in range(3)]
    )
    rots = Rotation.from_matrix(src_poses[:, :3, :3])
    slerp = Slerp(src_ts, rots)
    out = np.repeat(np.eye(4)[None, :, :], len(target_ts), axis=0)
    out[:, :3, :3] = slerp(clipped).as_matrix()
    out[:, :3, 3] = xyz
    inside = (target_ts >= src_ts[0]) & (target_ts <= src_ts[-1])
    return out, inside


def _world_output_pose(
    world_tracker: np.ndarray,
    t_output_tracker: np.ndarray,
) -> np.ndarray:
    return world_tracker @ np.linalg.inv(t_output_tracker)


def _matrix_to_pose_vector(transform: np.ndarray, rotation: str) -> np.ndarray:
    rot = transform[:3, :3]
    if rotation == "euler_xyz":
        rot_vec = Rotation.from_matrix(rot).as_euler("xyz", degrees=False)
    elif rotation == "rotvec":
        rot_vec = Rotation.from_matrix(rot).as_rotvec()
    elif rotation == "quat_wxyz":
        quat_xyzw = Rotation.from_matrix(rot).as_quat()
        rot_vec = quat_xyzw[[3, 0, 1, 2]]
    elif rotation == "rot6d":
        # Zhou et al. 6D representation: concatenate the first two columns.
        rot_vec = rot[:, :2].reshape(6, order="F")
    else:
        raise ValueError(
            f"Unsupported rotation mode {rotation!r}; expected one of "
            "'euler_xyz', 'rotvec', 'quat_wxyz', 'rot6d'."
        )
    return np.concatenate([transform[:3, 3], rot_vec]).astype(np.float32)


def _load_encoder(lowdim: Path, arm_cfg: dict[str, Any]) -> dict[str, np.ndarray] | None:
    names = [arm_cfg.get("encoder_file", "")]
    names.extend(arm_cfg.get("encoder_fallback_files", []))
    for name in names:
        if not name:
            continue
        path = lowdim / name
        if path.exists():
            data = np.load(path)
            return {key: data[key] for key in data.files}
    return None


def _interp_encoder(
    data: dict[str, np.ndarray] | None,
    target_ts: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    if data is None or "timestamp" not in data:
        return np.zeros(len(target_ts), dtype=np.float32)
    key = "metric" if mode == "metric" and "metric" in data else "normalized"
    if key not in data:
        return np.zeros(len(target_ts), dtype=np.float32)
    ts = data["timestamp"].astype(np.float64)
    values = data[key].astype(np.float64)
    valid = data.get("valid", np.ones(len(ts), dtype=np.int8)).astype(bool)
    mask = valid & np.isfinite(ts) & np.isfinite(values)
    if not np.any(mask):
        return np.zeros(len(target_ts), dtype=np.float32)
    ts = ts[mask]
    values = values[mask]
    order = np.argsort(ts)
    ts = ts[order]
    values = values[order]
    unique_ts, unique_idx = np.unique(ts, return_index=True)
    values = values[unique_idx]
    return np.interp(target_ts, unique_ts, values).astype(np.float32)


def _build_state(
    episode: Path,
    target_ts: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    lowdim = episode / "lowdim"
    tracker_path = lowdim / "tracker.npz"
    if not tracker_path.exists():
        raise FileNotFoundError(f"Missing tracker data: {tracker_path}")
    tracker = np.load(tracker_path)

    gripper_mode = config.get("gripper", {}).get("mode", "normalized")
    pose_mode = config.get("pose_mode", "relative_to_first")
    rotation = config.get("rotation", "rot6d")
    state_parts: list[np.ndarray] = []
    diagnostics: dict[str, Any] = {"arms": {}}
    for side in ("left", "right"):
        arm_cfg = config["arms"][side]
        pose_key = arm_cfg.get("tracker_pose_key", f"{side}_eef_pose")
        valid_key = arm_cfg.get("tracker_valid_key", f"{side}_eef_valid")
        poses, inside = _interp_pose(
            tracker["timestamp"],
            tracker[pose_key],
            tracker[valid_key],
            target_ts,
        )
        mount = arm_cfg.get("t_output_tracker", arm_cfg.get("t_ee_tracker"))
        t_output_tracker = np.asarray(mount, dtype=np.float64)
        output_poses = np.asarray(
            [_world_output_pose(pose, t_output_tracker) for pose in poses],
            dtype=np.float64,
        )
        base_index = 0
        base_pose = output_poses[0]
        if pose_mode == "relative_to_first":
            valid_indices = np.flatnonzero(inside)
            if len(valid_indices):
                base_index = int(valid_indices[0])
                base_pose = output_poses[base_index]
            base_inv = np.linalg.inv(base_pose)
            output_poses = np.asarray([base_inv @ pose for pose in output_poses])
        elif pose_mode != "absolute":
            raise ValueError(
                f"Unsupported pose_mode {pose_mode!r}; expected "
                "'relative_to_first' or 'absolute'."
            )
        ee = np.asarray(
            [_matrix_to_pose_vector(pose, rotation) for pose in output_poses],
            dtype=np.float32,
        )
        gripper = _interp_encoder(_load_encoder(lowdim, arm_cfg), target_ts, mode=gripper_mode)
        state_parts.append(np.column_stack([ee, gripper]).astype(np.float32))
        diagnostics["arms"][side] = {
            "tracker_valid_fraction_on_timeline": float(np.mean(inside)),
            "gripper_mode": gripper_mode,
            "output_frame": config.get("output_frame", "output"),
            "pose_mode": pose_mode,
            "rotation": rotation,
            "base_frame_index": base_index,
            "base_world_output_pose": base_pose.tolist(),
        }
    return np.column_stack(state_parts).astype(np.float32), diagnostics


def _fixed_size_list_array(values: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


def _write_episode_parquet(
    path: Path,
    *,
    state: np.ndarray,
    episode_index: int,
    global_start_index: int,
    task_index: int,
    fps: int,
) -> None:
    n = len(state)
    actions = np.vstack([state[1:], state[-1:]]).astype(np.float32)
    table = pa.table(
        {
            "observation.state": _fixed_size_list_array(state),
            "actions": _fixed_size_list_array(actions),
            "timestamp": pa.array(np.arange(n, dtype=np.float32) / float(fps)),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64)),
            "index": pa.array(np.arange(global_start_index, global_start_index + n, dtype=np.int64)),
            "task_index": pa.array(np.full(n, task_index, dtype=np.int64)),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _video_info(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    try:
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()


def _write_aligned_video(
    src_path: Path,
    src_ts: np.ndarray,
    target_ts: np.ndarray,
    out_path: Path,
    *,
    fps: int,
    resize: tuple[int, int] | None,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if resize is None and len(src_ts) == len(target_ts) and np.allclose(src_ts, target_ts, atol=1e-3):
        shutil.copy2(src_path, out_path)
        info = _video_info(out_path)
        info["codec"] = "copied"
        return info

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_path}")
    src_info = _video_info(src_path)
    width = resize[0] if resize else src_info["width"]
    height = resize[1] if resize else src_info["height"]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {out_path}")

    indices = _nearest_indices(src_ts, target_ts)
    last_idx = -1
    frame = None
    try:
        for idx in indices:
            idx = int(idx)
            if idx != last_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Could not read frame {idx} from {src_path}")
                last_idx = idx
            out_frame = frame
            if resize is not None:
                out_frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            writer.write(out_frame)
    finally:
        writer.release()
        cap.release()

    return {
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "frames": int(len(target_ts)),
        "codec": "mp4v",
    }


def _stats(values: np.ndarray) -> dict[str, Any]:
    if values.ndim == 1:
        v = values[:, None]
    else:
        v = values
    return {
        "min": np.nanmin(v, axis=0).astype(float).tolist(),
        "max": np.nanmax(v, axis=0).astype(float).tolist(),
        "mean": np.nanmean(v, axis=0).astype(float).tolist(),
        "std": np.nanstd(v, axis=0).astype(float).tolist(),
        "count": [int(len(values))],
    }


def _feature_info(
    *,
    video_infos: dict[str, dict[str, Any]],
    state_dim: int,
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": ["observation.state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": ["actions"],
        },
    }
    for key, info in video_infos.items():
        features[key] = {
            "dtype": "video",
            "shape": [int(info["height"]), int(info["width"]), 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": int(info["height"]),
                "video.width": int(info["width"]),
                "video.codec": info.get("codec", "mp4v"),
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": int(round(info.get("fps", 30))),
                "video.channels": 3,
                "has_audio": False,
            },
        }
    features.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return features


def convert(
    session: Path,
    output: Path,
    *,
    config_path: Path,
    camera_config: Path,
    task: str,
    resize: tuple[int, int] | None,
) -> None:
    config = _load_json(config_path)
    fps = int(config.get("fps", 30))
    video_keys = config["video_keys"]
    episodes = _episode_dirs(session)

    output.mkdir(parents=True, exist_ok=True)
    episode_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    global_index = 0
    total_frames = 0
    video_infos: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {
        "source_session": str(session),
        "config": str(config_path),
        "camera_config": str(camera_config),
        "episodes": [],
    }

    for episode_index, ep in enumerate(episodes):
        cam_dirs = _load_camera_dirs(ep, camera_config=camera_config)
        ref_role = "egocam" if "egocam" in cam_dirs else sorted(cam_dirs)[0]
        target_ts = _load_timestamps(cam_dirs[ref_role])
        n = len(target_ts)
        state, state_diag = _build_state(ep, target_ts, config)

        chunk = episode_index // int(config.get("chunks_size", 1000))
        parquet_path = output / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        _write_episode_parquet(
            parquet_path,
            state=state,
            episode_index=episode_index,
            global_start_index=global_index,
            task_index=0,
            fps=fps,
        )

        ep_video_info: dict[str, dict[str, Any]] = {}
        for role, video_key in video_keys.items():
            if role not in cam_dirs:
                continue
            cam_dir = cam_dirs[role]
            src_ts = _load_timestamps(cam_dir)
            out_video = output / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"
            info = _write_aligned_video(
                cam_dir / "color.mp4",
                src_ts,
                target_ts,
                out_video,
                fps=fps,
                resize=resize,
            )
            ep_video_info[video_key] = info
            video_infos.setdefault(video_key, info)

        episode_rows.append({"episode_index": episode_index, "tasks": [task], "length": n})
        actions = np.vstack([state[1:], state[-1:]]).astype(np.float32)
        timestamps = np.arange(n, dtype=np.float32) / float(fps)
        stats = {
            "observation.state": _stats(state),
            "actions": _stats(actions),
            "timestamp": _stats(timestamps),
            "frame_index": _stats(np.arange(n, dtype=np.int64)),
            "episode_index": _stats(np.full(n, episode_index, dtype=np.int64)),
            "index": _stats(np.arange(global_index, global_index + n, dtype=np.int64)),
            "task_index": _stats(np.zeros(n, dtype=np.int64)),
        }
        for key, info in ep_video_info.items():
            stats[key] = {
                "min": [[[0.0]], [[0.0]], [[0.0]]],
                "max": [[[1.0]], [[1.0]], [[1.0]]],
                "mean": [[[math.nan]], [[math.nan]], [[math.nan]]],
                "std": [[[math.nan]], [[math.nan]], [[math.nan]]],
                "count": [int(info["frames"])],
            }
        stats_rows.append({"episode_index": episode_index, "stats": stats})
        diagnostics["episodes"].append(
            {
                "episode_index": episode_index,
                "source": str(ep),
                "reference_camera_role": ref_role,
                "frames": n,
                "camera_roles": {role: str(path) for role, path in cam_dirs.items()},
                **state_diag,
            }
        )
        global_index += n
        total_frames += n

    _write_jsonl(output / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task}])
    _write_jsonl(output / "meta" / "episodes.jsonl", episode_rows)
    _write_jsonl(output / "meta" / "episodes_stats.jsonl", stats_rows)
    _write_json(
        output / "meta" / "info.json",
        {
            "codebase_version": "v2.1",
            "robot_type": config.get("robot_type", "yam_umi_bimanual"),
            "total_episodes": len(episodes),
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": len(video_infos),
            "total_chunks": max(1, math.ceil(len(episodes) / int(config.get("chunks_size", 1000)))),
            "chunks_size": int(config.get("chunks_size", 1000)),
            "fps": fps,
            "splits": {"train": f"0:{len(episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": _feature_info(video_infos=video_infos, state_dim=state.shape[1]),
        },
    )
    _write_json(output / "meta" / "conversion_diagnostics.json", diagnostics)
    print(f"[LEROBOT] wrote {len(episodes)} episode(s), {total_frames} frames -> {output}", flush=True)


def _resize_arg(value: str) -> tuple[int, int] | None:
    if value.lower() in {"none", "source", "keep"}:
        return None
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Raw session dir or hotkey session root.")
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--task", default="default")
    parser.add_argument(
        "--resize",
        type=_resize_arg,
        default=None,
        help="Video output size as WIDTHxHEIGHT, e.g. 320x240. Default keeps source size.",
    )
    args = parser.parse_args()
    convert(
        args.session,
        args.output_dir,
        config_path=args.config,
        camera_config=args.camera_config,
        task=args.task,
        resize=args.resize,
    )


if __name__ == "__main__":
    main()
