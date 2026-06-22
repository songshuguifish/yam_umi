# sim_teleop

Clean YAM teleoperation package extracted from the older HuMI
`humi_data_collection/packages/htc_interface` scripts.

## Current Scope

`sim_teleop` currently supports live teleoperation and YAM episode recording:

1. Read one or more Vive Trackers from OpenVR.
2. Convert tracker motion into end-effector motion.
3. Solve YAM arm IK with J-PARSE or mink.
4. Drive the YAM arm and LINEAR_4310 gripper in a MuJoCo viewer.
5. Optionally read a BRT encoder for gripper open/close.
6. Save tracker-first YAM teleop episodes with `--record-dir`.
7. Record raw Vive Tracker trajectories without MuJoCo via
   `python -m sim_teleop.record_tracker`.
8. Stream raw Vive Tracker poses over ZMQ to a LAN receiver (e.g. Ubuntu)
   via `python -m sim_teleop.stream_pose` — see Live Pose Streaming below.

It does not yet implement MuJoCo replay.

## Model Assets

The vendored i2rt model is modular: YAM arm XML, LINEAR_4310 gripper XML, and
gripper mount config are stored separately and combined at runtime. For data
collection and replay, `sim_teleop` materializes the specific model we use:

```text
sim_teleop/models/
  yam_linear_4310_tracker.xml
  yam_linear_4310_tracker.frames.urdf
  yam_linear_4310_tracker.meta.json
```

`yam_linear_4310_tracker.xml` is the authoritative MuJoCo replay model. It
contains the arm, LINEAR_4310 gripper, `grasp_site`, `ee_site`, and
`tracker_site`.

`yam_linear_4310_tracker.frames.urdf` is a lightweight kinematic reference
URDF. It starts from the YAM arm URDF and adds fixed `grasp_site`, `ee_site`,
and `tracker_site` frames. `ee_site` is colocated with `grasp_site` but uses
the tracker orientation, so `ee_site -> tracker_site` is pure translation.
MuJoCo remains the source for replay and gripper geometry.

The metadata records `T_grasp_tracker`, `T_link6_grasp`, source asset paths,
and the relative tracker replay rule:

```text
T_tracker_delta = inv(T_world_tracker_0) @ T_world_tracker_t
T_grasp_delta = T_grasp_tracker @ T_tracker_delta @ inv(T_grasp_tracker)
T_world_grasp_target = T_world_grasp_0 @ T_grasp_delta
```

Regenerate the assets after changing tracker mount geometry:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.export_model
```

## Main Entry

Run from the repository root:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop --ik-method jparse --resolution 1024
```

Useful variants:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop --ik-method mink --resolution 1024
& ".venv\Scripts\python.exe" -m sim_teleop --control-site ee_site --ik-method jparse --resolution 1024
& ".venv\Scripts\python.exe" -m sim_teleop --control-site ee_site --ik-method mink --joint6-axis positive --resolution 1024
& ".venv\Scripts\python.exe" -m sim_teleop --port COM5 --resolution 1024
& ".venv\Scripts\python.exe" -m sim_teleop --record-dir data/yam_teleop
```

`--control-site ee_site` controls the tracker-aligned `ee_site` instead of
`grasp_site`. This is useful for isolating whether the tracker/grasp relative
rotation is causing a teleoperation issue.

`--joint6-axis positive` is an experimental A/B test that overrides MuJoCo
`joint6` from the i2rt config value `0 0 -1` to `0 0 1`.

Check the exported MuJoCo model before starting OpenVR:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop --check-model-only
```

By default, teleop loads:

```text
sim_teleop/models/yam_linear_4310_tracker.xml
```

You can override it explicitly:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop --model-xml sim_teleop/models/yam_linear_4310_tracker.xml
```

Minimal MuJoCo model visualization, with mesh hidden and only `grasp_site`,
`ee_site`, and `tracker_site` highlighted:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.visualize_model
```

For a terminal-only check:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.visualize_model --check-only
& ".venv\Scripts\python.exe" -m sim_teleop.check_frames
```

Tracker-pose-only recording:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.record_tracker -o data/tracker_poses
```

Tracker link / mount validation:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.validate_tracker_link --urdf-only
& ".venv\Scripts\python.exe" -m sim_teleop.validate_tracker_link
```

Stable model asset export:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.export_model
```

Viewer keys:

```text
R  reset tracker reference and enter CONTROL mode
S  start recording an episode, if --record-dir is set
T  stop and save the active episode
O  record current encoder value as gripper open
C  record current encoder value as gripper closed
Q  save active recording, quit the loop, and close OpenVR
```

## Environment

The active Windows environment should live at the repository root:

```text
.venv
```

It was copied from the older HTC interface venv and verified with:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop --help
```

Key dependencies (versions from the verified `.venv`):

```text
# Teleoperation / hardware
openvr==2.12.1401            Vive Tracker / OpenVR pose polling
pyserial==3.5                COM port for the BRT gripper encoder
minimalmodbus==2.1.1         Modbus for the gripper controller
python-can==4.5.0            CAN bus support

# Simulation / IK
mujoco==3.9.0                YAM arm + LINEAR_4310 viewer / replay
mink==1.1.1                  whole-body IK solver
pyroki                       (local install) IK solver
jax==0.10.1, jaxlib==0.10.1  J-PARSE IK backend
jaxlie==1.5.0                SE(3) / SO(3) Lie algebra
yourdfpy==0.0.60             URDF kinematic reference

# Network / streaming
pyzmq==27.1.0                LAN pose streaming (PUB/SUB)

# Math / data
numpy==2.4.6
scipy==1.17.1
pandas==3.0.3
pyarrow==24.0.0

# Visualization (episode review)
evo==1.36.5                  trajectory metrics / plotting (TUM)
rerun-sdk==0.33.0            interactive 3D pose viewer
matplotlib==3.11.0
trimesh==4.12.2

# Config / CLI
tyro==1.0.13                 CLI argument parsing
pydantic==2.13.4
rich==14.3.4
loguru==0.7.3
```

Dump the full frozen environment anytime with:

```powershell
& ".venv\Scripts\python.exe" -m pip freeze
```

The package also needs source dependencies from the vendored HuMI tree:

```text
third_party/HuMI-main/third_party/i2rt-main
third_party/HuMI-main/humi_data_collection/packages/htc_interface/yam_ik_controller
```

`sim_teleop.robot` resolves those paths automatically for the current repo
layout and for the old `packages/htc_interface` layout.

## Live Pose Streaming (Windows → Ubuntu)

Stream live Vive Tracker poses from the Windows machine (SteamVR) to another
machine on the LAN (e.g. Ubuntu) over ZeroMQ. The receiver needs no SteamVR
install and can drive IK, logging, or teleop with the live pose.

```text
Windows (SteamVR + tracker)                Ubuntu (LAN)
┌────────────────────────────┐           ┌─────────────────────────┐
│ stream_pose.py             │   ZMQ     │ receive_pose.py         │
│  openvr → 4x4 pose         │ ────────► │  zmq.SUB connect        │
│  zmq.PUB bind tcp://*:1234 │   LAN     │   tcp://<WIN_IP>:1234   │
└────────────────────────────┘           └─────────────────────────┘
```

Each frame is JSON, matching `record_tracker` on-disk multi-tracker schema:

```json
{
  "timestamp": 1780000000.123,
  "trackers": [
    {"serial": "3B-A33M02233", "role": "left_eef", "tracker_pose": [[...4x4...]]},
    {"serial": "3B-A33M01660", "role": "right_eef", "tracker_pose": [[...4x4...]]}
  ]
}
```

`tracker_pose` is a 4x4 homogeneous matrix in the SteamVR
`TrackingUniverseStanding` frame (raw, no robot-frame transform). The sender
reuses `sim_teleop.tracker.read_tracker_poses`, so streamed poses are directly
comparable to recorded ones.

### Windows side (sender)

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.stream_pose --port 1234
```

Options:

```text
--port PORT          ZMQ publisher port (default 1234)
--host HOST          bind interface; '*' = all / LAN (default '*')
--frequency HZ       publish rate (default 120)
--serial-prefix PFX  only stream trackers whose serial starts with PFX
                     (default TRACKER_SERIAL_PREFIX from tracker.py)
```

Find the Windows LAN IP with `ipconfig` (e.g. `192.168.1.10`).

### Ubuntu side (receiver)

Install the single dependency (no openvr/SteamVR needed):

```bash
pip install pyzmq
```

Run from the repo root. The receiver has no `sim_teleop` imports, so it also
works as a standalone file:

```bash
python -m sim_teleop.receive_pose --host 192.168.1.10 --port 1234
# or directly without the package:
python sim_teleop/receive_pose.py --host 192.168.1.10 --port 1234
```

Options:

```text
--host WIN_IP   IP of the Windows sender (default 127.0.0.1)
--port PORT     port the sender binds (default 1234)
--print-rate N  print one frame every N seconds; 0 = every frame (default 0.5)
```

Use the pose in your own code by polling the newest frame:

```python
import zmq, numpy as np
sock = zmq.Context().socket(zmq.SUB)
sock.setsockopt(zmq.CONFLATE, 1); sock.setsockopt(zmq.RCVHWM, 1)
sock.setsockopt_string(zmq.SUBSCRIBE, "")
sock.connect("tcp://192.168.1.10:1234")
msg = sock.recv_json()
pose = np.array(msg["trackers"][0]["tracker_pose"])  # 4x4
pos, rot = pose[:3, 3], pose[:3, :3]
```

### Streaming Checklist

1. Both machines on the same subnet; `ping <WIN_IP>` works from Ubuntu.
2. Allow the publisher port through the Windows firewall. On the first-run
   popup check "Private network", or open TCP 1234 manually. This is the most
   common reason Ubuntu sees no data.
3. SteamVR detects the tracker before starting `stream_pose`.
4. PUB/SUB drops the first few frames after a subscriber connects ("slow
   joiner"). The receiver keeps only the newest frame via `CONFLATE`, so this
   is fine for real-time use.

## Relation To Old HuMI Code

Old Windows-side tracker recording:

```powershell
& ".venv\Scripts\python.exe" -m htc_scripts.record_pose --rpc.serve --rec.output-dir data/my-tracker-recordings
```

Old full-body IK and data processing:

```bash
uv run online-ik <task_name> --rpc-address tcp://<windows_ip>:4242
uv run offline-ik <task_name> -i data/my-raw-recordings
uv run replay -i data/my-raw-recordings_ik_recomputed
uv run run-pipeline data/my-recording-session --gopro_timezone +08:00
```

That pipeline targets HuMI/G1 full-body data. For YAM MuJoCo teleop, reuse the
episode idea, but keep a separate YAM-specific schema.

## YAM Recording Schema

Tracker-only file layout:

```text
data/tracker_poses/session_YYYYmmdd_HHMMSS/
  metadata.json
  tracker_recording_YYYY.mm.dd_HH.MM.SS.ffffff.json
```

Tracker-only episode frame:

```json
{
  "timestamp": 1780000000.123,
  "serial": "3B-A33M02233",
  "tracker_pose": [[...], [...], [...], [...]],
  "trackers": [
    {"serial": "3B-A33M02233", "role": "left_eef", "tracker_pose": [[...]]},
    {"serial": "3B-A33M01660", "role": "right_eef", "tracker_pose": [[...]]}
  ],
  "poses_by_role": {
    "left_eef": [[...], [...], [...], [...]],
    "right_eef": [[...], [...], [...], [...]]
  }
}
```

`serial` and top-level `tracker_pose` are kept for compatibility with the
single-tracker replay path and prefer `left_eef` when a mapping is configured.
All poses in one frame are returned by the same OpenVR
`TrackingUniverseStanding` query, so the tracker matrices share one SteamVR
standing coordinate system.

Teleop file layout:

```text
data/yam_teleop/session_YYYYmmdd_HHMMSS/
  metadata.json
  recording_YYYY.mm.dd_HH.MM.SS.ffffff.json
```

Recommended episode frame fields:

```json
{
  "timestamp": 1780000000.123,
  "tracker_pose": [[...], [...], [...], [...]],
  "target_ee_pose": [[...], [...], [...], [...]],
  "realized_ee_pose": [[...], [...], [...], [...]],
  "arm_q": [0, 0, 0, 0, 0, 0],
  "ik_ok": true,
  "ik_error_m": 0.001,
  "mode": "CONTROL"
}
```

Minimum fields for tracker-trajectory replay:

```text
timestamp
tracker_pose
```

Useful fields for debugging and future training:

```text
tracker_pose
target_ee_pose
realized_ee_pose
ik_ok
ik_error_m
```

## Suggested Next Entries

Replay:

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.replay_tracker data/tracker_poses
& ".venv\Scripts\python.exe" -m sim_teleop.replay_tracker data/tracker_poses --hide-mesh --loop
& ".venv\Scripts\python.exe" -m sim_teleop.replay_tracker data/tracker_poses --precompute-only
& ".venv\Scripts\python.exe" -m sim_teleop.replay_tracker data/tracker_poses --ik-mode window --initial-pose ready --stride 2 --action-frames 12 --exec-frames 4 --hide-mesh
```

Replay defaults to the validated setup:

```text
control_site = ee_site
joint6_axis  = positive
IK           = mink
initial_pose = ready
```

The MuJoCo replay viewer uses yellow for the target control pose, cyan for the
realized control site, and magenta for `tracker_site`.

Note: the vendored YAM URDF currently does not contain a tracker link. The
current validation path injects a `tracker_site` into the MuJoCo XML and checks
that the FK-derived `T_EE_TRACK` matches the configured mount transform.

## Episode Visualization (Rerun + evo)

Review recorded tracker episodes in an interactive 3D Rerun viewer (each tracker
shown as a moving triad plus its smoothed trajectory) and/or export TUM
trajectories for evo metrics.

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.visualize_tracker                          # auto-pick the latest episode under --root
& ".venv\Scripts\python.exe" -m sim_teleop.visualize_tracker <episode.json>           # specify a file
& ".venv\Scripts\python.exe" -m sim_teleop.visualize_tracker <f> --axis-length 0.08   # triad axis length in meters
& ".venv\Scripts\python.exe" -m sim_teleop.visualize_tracker --all-trackers           # show every tracker at once
& ".venv\Scripts\python.exe" -m sim_teleop.visualize_tracker --tracker-role right_eef # pick one by role
& ".venv\Scripts\python.exe" -m sim_teleop.visualize_tracker --no-rerun --tum-out data/traj.tum  # TUM export only
```

Options:

```text
input                 Episode JSON; omit to auto-pick the latest under --root
--root ROOT           Search root for auto-picking the latest episode
--tracker-role ROLE   Tracker role or serial to visualize (left_eef, right_eef, ...)
--all-trackers        Visualize all trackers in the recording at once
--axis-length M       Triad axis length in meters
--tum-out PATH        Write a TUM trajectory file; with multiple trackers use a dir/prefix
--no-rerun            Only export TUM files; skip the Rerun viewer
```

## Raw Sensor Data Collection (cameras + encoders + trackers)

`sim_teleop.data_collection` records synchronized raw sensor streams —
RealSense RGB video (PyAV H.264), BRT gripper encoders, and Vive Trackers —
for later conversion to a LeRobot dataset. Two collectors share the same
sensor processes:

- **`collect_smoke`** — one fixed-duration episode; a pipeline smoke test.
- **`collect_session`** — hot-start sensors once, then record many episodes
  interactively with `r` / `s` / `q` hotkeys. Use this for real collection.

### Running collect_session

Requires `.venv` with `pyrealsense2` + `av` (PyAV) in addition to the
teleop stack, plus an encoder mapping config (default
`sim_teleop/configs/encoder_mapping.json`, produced by `calibrate_encoders`):

```powershell
& ".venv\Scripts\python.exe" -m sim_teleop.data_collection.collect_session -o data/sessions
```

The collector hot-starts every sensor first, then waits for hotkeys:

```text
r   start a new episode (cameras START via the recorder's serve mode)
s   stop and save the current episode
q   stop, save, and quit
```

Each episode writes cameras to `episode_NNN/cameras/cam*/color.mp4`
(H.264, `yuv420p`, CRF 18) with per-frame timestamp `.npy` arrays, and
slices the encoder/tracker ring buffers by the episode time window into
`episode_NNN/lowdim/*.npz`.

Options:

```text
-o, --output-dir DIR            session root (default data/sessions)
--camera-width/--height/--fps   RealSense stream config (640/480/30)
--max-cameras N                 cap number of cameras (default 3)
--max-episode-s FLOAT           ring-buffer capacity guard in seconds (default 180)
--encoder-mapping PATH          encoder role→port+calibration config
--no-camera                     run without RealSense (encoders + trackers only)
--encoder-frequency FLOAT       encoder sampling rate (default 30)
--tracker-frequency FLOAT       tracker sampling rate (default 120)
```

### Output layout

```text
data/sessions/session_YYYYmmdd_HHMMSS/
  metadata.json                          session-level (schema=hotkey_session_v1, episodes[])
  cameras_rig_ready.json                 camera serve warmed up
  realsense_serve.log                    camera subprocess stdout
  episode_000/
    metadata.json                        episode-level (t_start/t_stop/duration/counts)
    cameras/
      realsense_ready.json / realsense_done.json
      cam0/color.mp4 + color_timestamps.npy + device_timestamps.npy + ...
      cam1/...
    lowdim/
      encoder_left.npz  encoder_right.npz  tracker.npz
  episode_001/ ...
```

### Downstream

Raw `.mp4` + `.npz` episodes feed a LeRobot dataset conversion step
(videos re-encoded to AV1 on import). Recording uses H.264 specifically
because PyAV/decord read it reliably and the conversion re-encodes, so
the recording codec and the final dataset codec are decoupled.

## Pre-collection Checklist

1. SteamVR detects the tracker.
2. `python -m sim_teleop --help` works in the selected venv.
3. `TRACKER_SERIAL_PREFIX` in `tracker.py` matches the mounted tracker.
4. Pressing `R` in the MuJoCo viewer enters CONTROL mode.
5. Console error stays small during slow 6-DOF tracker motion.
6. If using the gripper encoder, press `O` and `C` once and confirm
   `encoder_calibration.json` is saved.
7. Run one short recording and immediately replay it before collecting a
   full session.
