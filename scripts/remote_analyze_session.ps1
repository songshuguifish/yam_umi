param(
  [string]$Repo = "$env:USERPROFILE\Desktop\yam_umi",
  [string]$Session
)

$ErrorActionPreference = "Stop"
Set-Location $Repo

if (-not $Session) {
  throw "Provide -Session, e.g. data/raw_smoke_remote_check/session_YYYYmmdd_HHMMSS"
}

Write-Host "[REMOTE-ANALYZE] repo=$Repo"
Write-Host "[REMOTE-ANALYZE] session=$Session"

$summary = @'
from pathlib import Path
import json
import numpy as np
import sys

session = Path(sys.argv[1])
meta = json.loads((session / "metadata.json").read_text())
cam_meta = json.loads((session / "cameras" / "metadata.json").read_text())
encoder_paths = sorted((session / "lowdim").glob("encoder*.npz"))
if not encoder_paths:
    encoder_paths = [session / "lowdim" / "encoder.npz"]
encoders = [np.load(path) for path in encoder_paths]
trk = np.load(session / "lowdim" / "tracker.npz")
enc_ts = encoders[0]["timestamp"]
trk_ts = trk["timestamp"]

print("[SUMMARY] encoder_samples", meta.get("encoder_samples"))
print("[SUMMARY] encoder_actual_hz", meta.get("encoder_actual_hz"))
for path, enc in zip(encoder_paths, encoders):
    print(
        "[SUMMARY] encoder_file",
        path.name,
        "valid_frac",
        float(np.mean(enc["valid"])),
        "raw_minmax",
        int(enc["raw"].min()),
        int(enc["raw"].max()),
    )
print("[SUMMARY] tracker_samples", meta.get("tracker_samples"))
print("[SUMMARY] tracker_actual_hz", meta.get("tracker_actual_hz"))
print("[SUMMARY] tracker_valid_frac_LR", float(np.mean(trk["left_eef_valid"])), float(np.mean(trk["right_eef_valid"])))

for cam in cam_meta["cameras"]:
    cam_dir = session / "cameras" / f"cam{cam['camera_index']}"
    color = np.load(cam_dir / "color_timestamps.npy")
    domain = np.load(cam_dir / "timestamp_domain.npy")
    fc = np.load(cam_dir / "frame_counter.npy")
    toa = np.load(cam_dir / "metadata_time_of_arrival.npy")
    overlap_start = max(float(color[0]), float(enc_ts[0]), float(trk_ts[0]))
    overlap_end = min(float(color[-1]), float(enc_ts[-1]), float(trk_ts[-1]))
    drops = int(np.sum(np.diff(fc) != 1)) if not np.isnan(fc).any() else -1
    print(
        "[SUMMARY] camera",
        cam["camera_index"],
        cam.get("role"),
        cam["serial_number"],
        "frames",
        len(color),
        "domain",
        sorted(set(map(str, domain))),
        "drops",
        drops,
        "toa_nan",
        int(np.isnan(toa).sum()),
        "overlap_all_s",
        max(0.0, overlap_end - overlap_start),
    )
'@

$summary | & ".\.venv\Scripts\python.exe" - $Session
& ".\.venv\Scripts\python.exe" -m sim_teleop.data_collection.analyze_realsense_sync $Session --kind color
& ".\.venv\Scripts\python.exe" -m sim_teleop.data_collection.analyze_realsense_sync $Session --kind device
