"""Replay a recorded Vive Tracker trajectory on the YAM MuJoCo model."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .model_assets import EE_SITE, TRACKER_ALIGNED_EE_SITE, TRACKER_SITE
from .robot import RobotBundle, ik_mink
from .transform import ee_delta


N_ARM = 6
DEFAULT_ROOT = Path("data/tracker_poses")
EE_BOUNDS_MIN = np.array([-0.4, -0.4, 0.05])
EE_BOUNDS_MAX = np.array([0.4, 0.4, 0.5])
EE_WORKSPACE_CENTER = np.array([0.0, 0.0, 0.275])
MAX_EE_STEP_M = 0.03
INITIAL_Q_PRESETS = {
    "zero": np.zeros(N_ARM),
    "ready": np.array([0.0, 1.0, 1.0, 0.0, 0.5, 0.0]),
}


@dataclass
class ReplayResult:
    timestamps: np.ndarray
    qpos: np.ndarray
    targets: np.ndarray
    realized: np.ndarray
    ik_ok: np.ndarray
    pos_err: np.ndarray
    ori_err_deg: np.ndarray


def _resolve_recording(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.glob("tracker_recording_*.json"))
    if not candidates:
        candidates = sorted(path.glob("session_*/tracker_recording_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No tracker_recording_*.json found under {path}")
    return candidates[-1]


def _frame_pose(frame: dict, tracker_role: str | None, frame_idx: int) -> list:
    if tracker_role is None:
        return frame["tracker_pose"]
    poses_by_role = frame.get("poses_by_role", {})
    if tracker_role in poses_by_role:
        return poses_by_role[tracker_role]
    for tracker in frame.get("trackers", []):
        if tracker.get("role") == tracker_role:
            return tracker["tracker_pose"]
    raise KeyError(f"Frame {frame_idx} has no tracker role {tracker_role!r}")


def _load_recording(
    path: Path,
    tracker_role: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("episode", [])
    if not frames:
        raise ValueError(f"No frames in {path}")
    timestamps = np.asarray([frame["timestamp"] for frame in frames], dtype=float)
    poses = np.asarray(
        [_frame_pose(frame, tracker_role, i) for i, frame in enumerate(frames)],
        dtype=float,
    )
    return timestamps, poses


def _joint6_axis(choice: str) -> np.ndarray | None:
    if choice == "config":
        return None
    if choice == "positive":
        return np.array([0.0, 0.0, 1.0])
    if choice == "negative":
        return np.array([0.0, 0.0, -1.0])
    raise ValueError(f"Unsupported joint6 axis choice: {choice}")


def _parse_initial_q(text: str | None, preset: str) -> np.ndarray:
    if text is None:
        return INITIAL_Q_PRESETS[preset].copy()
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != N_ARM:
        raise ValueError(f"--initial-q expects {N_ARM} comma-separated values.")
    return np.asarray(values, dtype=float)


def _rotation_error_deg(target: np.ndarray, realized: np.ndarray) -> float:
    delta = target[:3, :3].T @ realized[:3, :3]
    cos_angle = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _rotation_residual(target: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()


def _joint_bounds(xml_path: str) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(xml_path)
    lower = np.full(N_ARM, -np.inf)
    upper = np.full(N_ARM, np.inf)
    for j in range(N_ARM):
        if model.jnt_limited[j]:
            lower[j], upper[j] = model.jnt_range[j]
    return lower, upper


def _limit_step(pos: np.ndarray, prev: np.ndarray) -> np.ndarray:
    delta = pos - prev
    norm = float(np.linalg.norm(delta))
    if norm > MAX_EE_STEP_M and norm > 0.0:
        return prev + delta * (MAX_EE_STEP_M / norm)
    return pos


def _compute_targets(
    bundle: RobotBundle,
    timestamps: np.ndarray,
    tracker_poses: np.ndarray,
    initial_q: np.ndarray,
    *,
    stride: int,
    max_frames: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = timestamps[::stride]
    tracker_poses = tracker_poses[::stride]
    if max_frames is not None:
        timestamps = timestamps[:max_frames]
        tracker_poses = tracker_poses[:max_frames]

    control_init = bundle.kin.fk(initial_q, bundle.control_site)
    tracker_init_inv = np.linalg.inv(tracker_poses[0])

    targets = []
    for tracker_pose in tracker_poses:
        target_delta = ee_delta(
            tracker_init_inv,
            tracker_pose,
            bundle.t_ee_track,
            bundle.t_ee_track_inv,
        )
        targets.append(control_init @ target_delta)
    return timestamps - timestamps[0], np.asarray(targets)


def _smooth_targets(targets: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return targets.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = targets.copy()
    for i in range(len(targets)):
        lo = max(0, i - half)
        hi = min(len(targets), i + half + 1)
        out[i, :3, 3] = targets[lo:hi, :3, 3].mean(axis=0)
        out[i, :3, :3] = Rotation.from_matrix(targets[lo:hi, :3, :3]).mean().as_matrix()
    return out


def _apply_teleop_safety(targets: np.ndarray) -> np.ndarray:
    out = targets.copy()
    prev = out[0, :3, 3].copy()
    for i in range(len(out)):
        pos = np.clip(out[i, :3, 3], EE_BOUNDS_MIN, EE_BOUNDS_MAX)
        pos = _limit_step(pos, prev)
        out[i, :3, 3] = pos
        prev = pos.copy()
    return out


def _anchor_targets(targets: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "home":
        return targets.copy(), np.zeros(3)
    if mode != "center":
        raise ValueError(f"Unsupported anchor mode: {mode}")
    out = targets.copy()
    positions = out[:, :3, 3]
    bbox_center = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    offset = EE_WORKSPACE_CENTER - bbox_center
    out[:, :3, 3] += offset
    return out, offset


def _precompute(
    bundle: RobotBundle,
    timestamps: np.ndarray,
    targets: np.ndarray,
    initial_q: np.ndarray,
    *,
    position_cost: float,
    orientation_cost: float,
    posture_cost: float,
    max_iters: int,
) -> ReplayResult:
    q_prev = initial_q.copy()
    qpos: list[np.ndarray] = []
    realized: list[np.ndarray] = []
    ik_ok: list[bool] = []
    pos_err: list[float] = []
    ori_err_deg: list[float] = []

    for i, target in enumerate(targets):
        q_sol, ok = ik_mink(
            bundle,
            q_prev,
            target,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            posture_cost=posture_cost,
            max_iters=max_iters,
        )
        q_prev = q_sol[:N_ARM].copy()
        actual = bundle.kin.fk(q_prev, bundle.control_site)

        qpos.append(q_prev)
        realized.append(actual)
        ik_ok.append(bool(ok))
        pos_err.append(float(np.linalg.norm(target[:3, 3] - actual[:3, 3])))
        ori_err_deg.append(_rotation_error_deg(target, actual))

        if i % 100 == 0 or i == len(targets) - 1:
            print(
                f"[REPLAY] IK {i + 1}/{len(targets)} "
                f"ok={ok} pos_err={pos_err[-1]:.4f} ori={ori_err_deg[-1]:.1f}deg",
                flush=True,
            )

    return ReplayResult(
        timestamps=timestamps - timestamps[0],
        qpos=np.asarray(qpos),
        targets=np.asarray(targets),
        realized=np.asarray(realized),
        ik_ok=np.asarray(ik_ok, dtype=bool),
        pos_err=np.asarray(pos_err, dtype=float),
        ori_err_deg=np.asarray(ori_err_deg, dtype=float),
    )


def _window_residual(
    q_flat: np.ndarray,
    *,
    bundle: RobotBundle,
    targets: np.ndarray,
    q_before: np.ndarray,
    q_before2: np.ndarray | None,
    q_home: np.ndarray,
    position_weight: float,
    orientation_weight: float,
    velocity_weight: float,
    acceleration_weight: float,
    posture_weight: float,
) -> np.ndarray:
    q_seq = q_flat.reshape(-1, N_ARM)
    residuals: list[np.ndarray] = []

    for q, target in zip(q_seq, targets):
        actual = bundle.kin.fk(q, bundle.control_site)
        residuals.append(np.sqrt(position_weight) * (actual[:3, 3] - target[:3, 3]))
        residuals.append(np.sqrt(orientation_weight) * _rotation_residual(target, actual))
        if posture_weight > 0.0:
            residuals.append(np.sqrt(posture_weight) * (q - q_home))

    if velocity_weight > 0.0:
        prev = q_before
        for q in q_seq:
            residuals.append(np.sqrt(velocity_weight) * (q - prev))
            prev = q

    if acceleration_weight > 0.0:
        prev2 = q_before if q_before2 is None else q_before2
        prev1 = q_before
        for q in q_seq:
            residuals.append(np.sqrt(acceleration_weight) * (q - 2.0 * prev1 + prev2))
            prev2 = prev1
            prev1 = q

    return np.concatenate(residuals)


def _precompute_windowed(
    bundle: RobotBundle,
    timestamps: np.ndarray,
    targets: np.ndarray,
    initial_q: np.ndarray,
    *,
    action_frames: int,
    exec_frames: int,
    position_weight: float,
    orientation_weight: float,
    velocity_weight: float,
    acceleration_weight: float,
    posture_weight: float,
    max_nfev: int,
) -> ReplayResult:
    if action_frames <= 0:
        raise ValueError("--action-frames must be positive.")
    if exec_frames <= 0:
        raise ValueError("--exec-frames must be positive.")

    q_home = initial_q.copy()
    q_prev = q_home.copy()
    q_prev2: np.ndarray | None = None
    lower, upper = _joint_bounds(bundle.xml_path)

    out_q: list[np.ndarray] = []
    out_targets: list[np.ndarray] = []
    out_realized: list[np.ndarray] = []
    out_ok: list[bool] = []
    out_pos_err: list[float] = []
    out_ori_err: list[float] = []
    out_ts: list[float] = []

    t = 0
    while t < len(targets):
        window_targets = targets[t : min(t + action_frames, len(targets))]
        horizon = len(window_targets)
        guess = np.repeat(q_prev[None, :], horizon, axis=0)
        bounds = (
            np.tile(lower, horizon),
            np.tile(upper, horizon),
        )
        result = least_squares(
            _window_residual,
            guess.ravel(),
            bounds=bounds,
            max_nfev=max_nfev,
            verbose=0,
            kwargs={
                "bundle": bundle,
                "targets": window_targets,
                "q_before": q_prev,
                "q_before2": q_prev2,
                "q_home": q_home,
                "position_weight": position_weight,
                "orientation_weight": orientation_weight,
                "velocity_weight": velocity_weight,
                "acceleration_weight": acceleration_weight,
                "posture_weight": posture_weight,
            },
        )
        q_seq = result.x.reshape(horizon, N_ARM)
        execute = min(exec_frames, horizon)

        for i in range(execute):
            q = q_seq[i].copy()
            target = targets[t + i]
            actual = bundle.kin.fk(q, bundle.control_site)
            pos_err = float(np.linalg.norm(target[:3, 3] - actual[:3, 3]))
            ori_err = _rotation_error_deg(target, actual)

            out_ts.append(float(timestamps[t + i]))
            out_q.append(q)
            out_targets.append(target)
            out_realized.append(actual)
            out_ok.append(bool(pos_err < 0.01 and ori_err < 5.0))
            out_pos_err.append(pos_err)
            out_ori_err.append(ori_err)

            q_prev2 = q_prev.copy()
            q_prev = q.copy()

        if t % max(exec_frames * 10, 1) == 0 or t + execute >= len(targets):
            print(
                f"[REPLAY] window {min(t + execute, len(targets))}/{len(targets)} "
                f"cost={result.cost:.4g} pos={out_pos_err[-1]:.4f} "
                f"ori={out_ori_err[-1]:.1f}deg",
                flush=True,
            )
        t += execute

    return ReplayResult(
        timestamps=np.asarray(out_ts) - out_ts[0],
        qpos=np.asarray(out_q),
        targets=np.asarray(out_targets),
        realized=np.asarray(out_realized),
        ik_ok=np.asarray(out_ok, dtype=bool),
        pos_err=np.asarray(out_pos_err, dtype=float),
        ori_err_deg=np.asarray(out_ori_err, dtype=float),
    )


def _add_sphere(
    scn: mujoco.MjvScene,
    pos: np.ndarray,
    *,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).ravel(),
        np.array(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def _add_frame(
    scn: mujoco.MjvScene,
    pose: np.ndarray,
    *,
    length: float,
    width: float,
) -> None:
    colors = (
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 1.0),
        (0.0, 0.2, 1.0, 1.0),
    )
    pos = pose[:3, 3]
    rot = pose[:3, :3]
    for axis in range(3):
        if scn.ngeom >= scn.maxgeom:
            return
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            np.array(colors[axis], dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            width,
            pos,
            pos + length * rot[:, axis],
        )
        scn.ngeom += 1


def _site_pose(data: mujoco.MjData, site_id: int) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    pose[:3, 3] = data.site_xpos[site_id]
    return pose


def _draw_overlay(
    viewer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    *,
    control_site: str,
) -> None:
    scn = viewer.user_scn
    scn.ngeom = 0

    _add_sphere(scn, target[:3, 3], radius=0.014, rgba=(1.0, 0.85, 0.0, 1.0))
    _add_frame(scn, target, length=0.09, width=0.004)

    for site_name, color in (
        (control_site, (0.0, 1.0, 1.0, 1.0)),
        (TRACKER_SITE, (1.0, 0.0, 1.0, 1.0)),
    ):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            continue
        pose = _site_pose(data, site_id)
        _add_sphere(scn, pose[:3, 3], radius=0.011, rgba=color)
        _add_frame(scn, pose, length=0.06, width=0.003)


def _set_model_qpos(model: mujoco.MjModel, data: mujoco.MjData, arm_q: np.ndarray) -> None:
    data.qpos[:N_ARM] = arm_q[:N_ARM]
    for j in range(N_ARM, model.nq):
        lo, hi = model.jnt_range[j]
        data.qpos[j] = lo + 0.5 * (hi - lo)


def _play_mujoco(
    xml_path: str,
    result: ReplayResult,
    *,
    control_site: str,
    speed: float,
    loop: bool,
    hide_mesh: bool,
) -> None:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    duration = float(result.timestamps[-1]) if len(result.timestamps) > 1 else 0.0
    idx = 0
    paused = False
    should_quit = False
    start_wall = time.time()
    pause_t = 0.0

    def on_key(key: int) -> None:
        nonlocal idx, paused, should_quit, start_wall, pause_t
        if key in (ord("Q"), ord("q")):
            should_quit = True
        elif key == ord(" "):
            paused = not paused
            if paused:
                pause_t = float(result.timestamps[idx])
            else:
                start_wall = time.time() - pause_t / speed
        elif key in (ord("R"), ord("r")):
            idx = 0
            start_wall = time.time()
            paused = False
        elif key in (ord("N"), ord("n")):
            idx = min(idx + 1, len(result.qpos) - 1)
            paused = True
            pause_t = float(result.timestamps[idx])
        elif key in (ord("B"), ord("b")):
            idx = max(idx - 1, 0)
            paused = True
            pause_t = float(result.timestamps[idx])

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        if hide_mesh:
            viewer.opt.geomgroup[:] = 0
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        print("[REPLAY] Keys: Space=pause  N/B=step  R=restart  Q=quit")
        print("[REPLAY] yellow=target, cyan=control_site, magenta=tracker_site")

        while viewer.is_running() and not should_quit:
            if not paused and duration > 0.0:
                replay_t = (time.time() - start_wall) * speed
                if loop:
                    replay_t = replay_t % duration
                elif replay_t >= duration:
                    replay_t = duration
                    paused = True
                    pause_t = replay_t
                idx = int(np.searchsorted(result.timestamps, replay_t, side="right") - 1)
                idx = int(np.clip(idx, 0, len(result.qpos) - 1))

            _set_model_qpos(model, data, result.qpos[idx])
            mujoco.mj_forward(model, data)
            _draw_overlay(
                viewer,
                model,
                data,
                result.targets[idx],
                control_site=control_site,
            )
            viewer.sync()
            time.sleep(1.0 / 60.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help="Recording JSON, session directory, or tracker recording root.",
    )
    parser.add_argument(
        "--control-site",
        choices=[EE_SITE, TRACKER_ALIGNED_EE_SITE],
        default=TRACKER_ALIGNED_EE_SITE,
        help="Replay target site. Defaults to the validated tracker-aligned ee_site.",
    )
    parser.add_argument(
        "--tracker-role",
        default=None,
        help=(
            "Tracker role to replay from a multi-tracker recording, e.g. "
            "left_eef or right_eef. Defaults to the legacy top-level tracker_pose."
        ),
    )
    parser.add_argument(
        "--joint6-axis",
        choices=["config", "positive", "negative"],
        default="positive",
        help="MuJoCo joint6 axis override. Defaults to the validated positive axis.",
    )
    parser.add_argument(
        "--initial-pose",
        choices=sorted(INITIAL_Q_PRESETS),
        default="ready",
        help="Named initial arm configuration used as the replay anchor.",
    )
    parser.add_argument(
        "--initial-q",
        default=None,
        help="Comma-separated six-joint initial q override, e.g. 0,1,1,0,0.5,0.",
    )
    parser.add_argument("--position-cost", type=float, default=1.0)
    parser.add_argument("--orientation-cost", type=float, default=1.0)
    parser.add_argument("--posture-cost", type=float, default=0.001)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--ik-mode", choices=["causal", "window"], default="causal")
    parser.add_argument("--action-frames", type=int, default=12)
    parser.add_argument("--exec-frames", type=int, default=4)
    parser.add_argument("--velocity-cost", type=float, default=0.05)
    parser.add_argument("--acceleration-cost", type=float, default=0.2)
    parser.add_argument("--window-max-nfev", type=int, default=25)
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--anchor",
        choices=["center", "home"],
        default="home",
        help=(
            "Where to place the relative tracker trajectory in robot workspace. "
            "home keeps the robot home pose; center is only for visualization debug."
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Centered offline smoothing window in frames. Use 1 to disable.",
    )
    parser.add_argument(
        "--no-safety",
        action="store_true",
        help="Disable teleop position bounds and max-step limiting.",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--hide-mesh", action="store_true")
    parser.add_argument(
        "--precompute-only",
        action="store_true",
        help="Solve IK and print stats without opening the MuJoCo viewer.",
    )
    args = parser.parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    if args.speed <= 0:
        raise ValueError("--speed must be positive.")
    if args.smooth_window <= 0:
        raise ValueError("--smooth-window must be positive.")

    recording = _resolve_recording(args.input)
    timestamps, tracker_poses = _load_recording(recording, args.tracker_role)
    print(
        f"[REPLAY] Loaded {len(timestamps)} frames "
        f"({timestamps[-1] - timestamps[0]:.3f}s) from {recording}",
        flush=True,
    )
    if args.tracker_role is not None:
        print(f"[REPLAY] tracker_role={args.tracker_role}", flush=True)

    bundle = RobotBundle(
        control_site=args.control_site,
        joint6_axis=_joint6_axis(args.joint6_axis),
    )
    initial_q = _parse_initial_q(args.initial_q, args.initial_pose)
    initial_pose = bundle.kin.fk(initial_q, args.control_site)
    print(
        f"[REPLAY] model={bundle.xml_path} control_site={args.control_site} "
        f"joint6_axis={args.joint6_axis}",
        flush=True,
    )
    print(
        f"[REPLAY] initial_q={np.round(initial_q, 4)} "
        f"{args.control_site}_pos={np.round(initial_pose[:3, 3], 4)}",
        flush=True,
    )

    replay_timestamps, targets = _compute_targets(
        bundle,
        timestamps,
        tracker_poses,
        initial_q,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    raw_pos = targets[:, :3, 3]
    print(
        "[REPLAY] raw target pos range "
        f"min={np.round(raw_pos.min(axis=0), 3)} "
        f"max={np.round(raw_pos.max(axis=0), 3)}",
        flush=True,
    )
    targets, anchor_offset = _anchor_targets(targets, args.anchor)
    if args.anchor != "home":
        anchored_pos = targets[:, :3, 3]
        print(
            f"[REPLAY] anchor={args.anchor} offset={np.round(anchor_offset, 3)} "
            f"range min={np.round(anchored_pos.min(axis=0), 3)} "
            f"max={np.round(anchored_pos.max(axis=0), 3)}",
            flush=True,
        )
    targets = _smooth_targets(targets, args.smooth_window)
    if not args.no_safety:
        targets = _apply_teleop_safety(targets)
        safe_pos = targets[:, :3, 3]
        print(
            "[REPLAY] safety target pos range "
            f"min={np.round(safe_pos.min(axis=0), 3)} "
            f"max={np.round(safe_pos.max(axis=0), 3)}",
            flush=True,
        )

    if args.ik_mode == "window":
        result = _precompute_windowed(
            bundle,
            replay_timestamps,
            targets,
            initial_q,
            action_frames=args.action_frames,
            exec_frames=args.exec_frames,
            position_weight=args.position_cost,
            orientation_weight=args.orientation_cost,
            velocity_weight=args.velocity_cost,
            acceleration_weight=args.acceleration_cost,
            posture_weight=args.posture_cost,
            max_nfev=args.window_max_nfev,
        )
    else:
        result = _precompute(
            bundle,
            replay_timestamps,
            targets,
            initial_q,
            position_cost=args.position_cost,
            orientation_cost=args.orientation_cost,
            posture_cost=args.posture_cost,
            max_iters=args.max_iters,
        )
    print(
        "[REPLAY] IK ok "
        f"{int(result.ik_ok.sum())}/{len(result.ik_ok)} "
        f"mean_pos={result.pos_err.mean():.4f}m "
        f"max_pos={result.pos_err.max():.4f}m "
        f"mean_ori={result.ori_err_deg.mean():.1f}deg "
        f"max_ori={result.ori_err_deg.max():.1f}deg",
        flush=True,
    )

    if args.precompute_only:
        return

    _play_mujoco(
        bundle.xml_path,
        result,
        control_site=args.control_site,
        speed=args.speed,
        loop=args.loop,
        hide_mesh=args.hide_mesh,
    )


if __name__ == "__main__":
    main()
