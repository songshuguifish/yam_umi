"""OpenVR Vive Tracker reading."""
import json
from pathlib import Path

import numpy as np
import openvr

TRACKER_SERIAL_PREFIX = "3B-A33M"
DEFAULT_TRACKER_MAPPING_PATH = Path(__file__).with_name("configs") / "tracker_mapping.json"


def read_tracker_poses(
    vr_system: openvr.IVRSystem,
    serial_prefix: str | None = TRACKER_SERIAL_PREFIX,
) -> list[tuple[str, np.ndarray]]:
    """Return valid Vive Tracker poses as (serial, 4x4 matrix) pairs."""
    poses = vr_system.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount,
    )
    out: list[tuple[str, np.ndarray]] = []
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if not poses[i].bPoseIsValid:
            continue
        if vr_system.getTrackedDeviceClass(i) != openvr.TrackedDeviceClass_GenericTracker:
            continue
        serial = vr_system.getStringTrackedDeviceProperty(
            i, openvr.Prop_SerialNumber_String
        )
        if serial_prefix is not None and not serial.startswith(serial_prefix):
            continue
        m34 = poses[i].mDeviceToAbsoluteTracking
        mat = np.eye(4)
        for r in range(3):
            for c in range(4):
                mat[r, c] = m34[r][c]
        out.append((serial, mat))
    return out


def read_pose(vr_system: openvr.IVRSystem) -> "np.ndarray | None":
    """Return the first valid Vive Tracker pose as a 4x4 matrix, or None."""
    tracker_poses = read_tracker_poses(vr_system)
    return tracker_poses[0][1] if tracker_poses else None


def load_tracker_mapping(path: Path | None = DEFAULT_TRACKER_MAPPING_PATH) -> dict[str, str]:
    """Load role -> tracker serial mapping, ignoring empty roles."""
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Tracker mapping must be a JSON object: {path}")
    return {str(role): str(serial) for role, serial in data.items() if serial}


def order_tracker_poses(
    tracker_poses: list[tuple[str, np.ndarray]],
    mapping: dict[str, str],
) -> list[tuple[str, np.ndarray]]:
    """Return mapped trackers first, then any extra trackers."""
    by_serial = {serial: pose for serial, pose in tracker_poses}
    ordered: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for serial in mapping.values():
        pose = by_serial.get(serial)
        if pose is not None:
            ordered.append((serial, pose))
            seen.add(serial)
    for serial, pose in tracker_poses:
        if serial not in seen:
            ordered.append((serial, pose))
    return ordered


def tracker_pose_records(
    tracker_poses: list[tuple[str, np.ndarray]],
    mapping: dict[str, str],
) -> list[dict]:
    """Convert tracker poses to JSON-ready records with optional role labels."""
    role_by_serial = {serial: role for role, serial in mapping.items()}
    return [
        {
            "serial": serial,
            **({"role": role_by_serial[serial]} if serial in role_by_serial else {}),
            "tracker_pose": pose.tolist(),
        }
        for serial, pose in order_tracker_poses(tracker_poses, mapping)
    ]
