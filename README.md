# yam_umi

Bimanual manipulation **data collection** on Windows. Record synchronized
RealSense RGB video, BRT gripper encoders, and Vive Trackers into raw
time-aligned episodes, then convert them to a LeRobot v2.1 dataset for
imitation learning. The same sensor stack also drives live YAM arm + gripper
teleoperation inside MuJoCo, plus a standalone gripper-encoder calibration
tool.

Typical flow: `collect_session` (hotkey recorder) → raw episode tree →
`convert_to_lerobot` → trainable dataset.

## Repository layout

```
sim_teleop/data_collection/   Data collection core: hotkey recorder, per-sensor processes, LeRobot conversion
sim_teleop/                   Live teleoperation: Vive Tracker → IK (J-PARSE / mink) → YAM arm + gripper in MuJoCo
gripper/                      Standalone BRT Modbus-RTU encoder reader + interactive calibration (no mujoco/openvr needed)
scripts/                      Collection/calibration launchers + remote hardware-check helpers (PowerShell)
```

`third_party/` (vendored HuMI / i2rt / pyroki source), `data/` (recorded
sessions), and the Python `.venv/` are **git-ignored** — they are local-only
dependencies and data, not part of this repo.

## Environment

The known-good Windows environment is the local venv at the repo root:

```powershell
.venv\Scripts\python.exe
```

Key dependencies (must be installed in that venv): `openvr`, `mujoco`, `numpy`,
`mink`, `pyroki`, `jax`, `jaxlie`, `yourdfpy`, `minimalmodbus`, `pyserial`.

`sim_teleop` also depends on vendored source trees under `third_party/`
(HuMI / i2rt / `yam_ik_controller`); `sim_teleop.robot` resolves those paths
for the current layout.

## Quick start

### Record a raw session (cameras + encoders + trackers)

`collect_session` hot-starts every sensor once (RealSense cameras, BRT gripper
encoders, Vive Trackers), then records many episodes interactively. Each
episode writes H.264 video to `episode_NNN/cameras/` and slices the
encoder/tracker streams by the episode time window into `episode_NNN/lowdim/`.

```powershell
& ".\.venv\Scripts\python.exe" -m sim_teleop.data_collection.collect_session -o data/sessions
```

Encoder raw counts are recorded by default; calibrate open/closed endpoints
endpoints later — `normalized`/`metric` are saved as NaN):

```powershell
& ".\.venv\Scripts\python.exe" -m sim_teleop.data_collection.collect_session `
  -o data\pokeumi_202606241148
```

Hotkeys (typed in the console running the collector; start/stop are also
beep-cued):

```text
c   start a new episode
q   stop + save the episode (during recording), or quit (at the episode menu)
```

Recorded layout (a `README_metadata.md` field guide is auto-written next to
each `metadata.json`):

```text
data/sessions/session_YYYYmmdd_HHMMSS/
  metadata.json                 session: schema, timebase, sensor config, episodes[]
  episode_NNN/
    metadata.json               episode: t_start/t_stop/duration/counts
    cameras/                    cam*/color.mp4 + per-frame timestamp .npy + metadata.json
    lowdim/                     encoder_left.npz  encoder_right.npz  tracker.npz
```

All streams share one host timebase, so `lowdim/*.npz` aligns directly onto
`cameras/cam*/color_timestamps.npy`. For the full process architecture
(one `mp.Process` per sensor device, always-on ring buffer + windowed slice),
CLI options, `.npz` column schemas, and camera metadata details, see
[sim_teleop/README.md](sim_teleop/README.md#raw-sensor-data-collection-cameras--encoders--trackers).

### Convert a raw session to LeRobot v2.1

After recording with the hotkey collector, convert a session directory into the
same top-level layout as `third_party/mapo_tofu_secondhalf_0527_1`:

```powershell
& ".\.venv\Scripts\python.exe" -m sim_teleop.data_collection.convert_to_lerobot `
  data\sessions\session_YYYYMMDD_HHMMSS `
  -o data\lerobot_yam_umi `
  --task mapo_tofu `
  --resize 320x240
```

The exported state/action vector is 20D:
`left xyz+rot6d+gripper, right xyz+rot6d+gripper`. By default, the pose frame is
`link6` and rotation uses the 6D representation formed by concatenating the
first two rotation-matrix columns. Camera frames are the master timeline;
tracker poses and encoder values are interpolated onto those frame timestamps. Edit
`sim_teleop/configs/lerobot_conversion.json` after measuring a better fixed
tracker mount transform.

Gripper values come from `encoder_left.npz` / `encoder_right.npz`
`normalized`, where `0 = fully_closed` and `1 = fully_open`. The open/closed
raw endpoints are saved in `sim_teleop/configs/encoder_mapping.json`:

```json
"calibration": {
  "raw_open": 703,
  "raw_closed": 883,
  "span": -180,
  "stroke_mm": 85.0,
  "convention": "0=fully_closed, 1=fully_open"
}
```

Run the calibration by USB serial instead of COM port, because COM numbers can
change across boots. For the current gripper convention, first set the encoder
hardware zero at fully closed, then capture fully open:

```powershell
scripts\calibrate_encoders.cmd closed-zero --role left_encoder --usb-serial 5B90108980
scripts\calibrate_encoders.cmd closed-zero --role right_encoder --usb-serial 5B90108259
```

The pose conversion uses a fixed tracker mount transform. The recorded Vive
pose is `T_world_tracker`; first the converter computes the absolute link6
pose:

```text
T_world_link6 = T_world_tracker @ inv(T_link6_tracker)
```

For UMI-style data, the exported pose is relative to the first valid frame in
each episode and each arm:

```text
T_relative_link6(t) = inv(T_world_link6(0)) @ T_world_link6(t)
```

This removes the arbitrary SteamVR/OpenVR world origin from the learning state.
Set `pose_mode` to `absolute` in `sim_teleop/configs/lerobot_conversion.json`
only if absolute world-frame poses are intentionally needed.

The default `T_link6_tracker` is not estimated from the recording; it comes
from the hand-defined tracker axis mounting plus the measured mount offset.
The current config assumes the tracker is 4.394 mm "back" along link6 `-Z`:

```text
T_link6_tracker =
[[ 0,  0,  1,  0.0000],
 [ 1,  0,  0,  0.0000],
 [ 0,  1,  0, -0.004394],
 [ 0,  0,  0,  1.0000]]
```

### Calibrate the gripper encoder (standalone, COM device only)

Plug in the BRT encoder over USB-serial and run from the repo root:

```powershell
& ".\.venv\Scripts\python.exe" -m gripper.calibrate            # interactive
& ".\.venv\Scripts\python.exe" -m gripper.calibrate --show     # just stream raw values
& ".\.venv\Scripts\python.exe" -m gripper.calibrate --open 703 --closed 883   # set directly
& ".\.venv\Scripts\python.exe" -m gripper.calibrate --zero     # hardware zero-set (persistent)
```

See [gripper/calibrate.py](gripper/calibrate.py) for all options.
Calibration is saved to `encoder_calibration.json` (git-ignored).

If raw readings jump or wrap (e.g. `11 → 1000` mid-stroke), the encoder's zero
point is sitting inside the gripper's travel. Move to **fully closed** and run
`--zero` once to push the zero out of the working range, **then** re-run
calibration — zeroing is persistent and shifts every raw value, so it
invalidates the saved `encoder_calibration.json`.

### Live teleoperation (needs SteamVR + tracker + encoder)

```powershell
& ".\.venv\Scripts\python.exe" -m sim_teleop --ik-method jparse
& ".\.venv\Scripts\python.exe" -m sim_teleop --port COM6       # explicit encoder port
```

Viewer keys: `R` reset tracker reference and enter CONTROL mode, `O`/`C` record
gripper open/closed, `Q` quit. See [sim_teleop/README.md](sim_teleop/README.md)
for the full guide.

## Notes

- Encoder convention: normalised `0 = fully closed`, `1 = fully open`. The map
  is correct regardless of which endpoint has the larger raw value.
- The BRT encoder also supports a persistent hardware zero-set (register
  0x0008, exposed as `gripper.calibrate --zero`). It is **not** needed for
  normal use — the software endpoint calibration removes any dependence on the
  hardware zero — but it fixes raw readings that jump/wrap across the boundary
  when the zero lands inside the gripper's travel. Zeroing invalidates the
  saved calibration, so re-run it afterwards.
- The `gripper` package is the single source of truth for encoder logic;
  `sim_teleop/gripper.py` is a thin re-export so the teleop pipeline is
  unaffected.
