# yam_umi

YAM arm + gripper teleoperation on Windows: read a Vive Tracker and a BRT
encoder, solve arm IK, and drive the YAM arm and LINEAR_4310 gripper inside a
MuJoCo simulation. Also includes a standalone gripper calibration tool that
needs nothing but a USB-serial encoder.

## Repository layout

```
sim_teleop/   Live teleoperation: Vive Tracker → IK (J-PARSE / mink) → YAM arm + gripper in MuJoCo
gripper/      Standalone BRT Modbus-RTU encoder reader + interactive calibration (no mujoco/openvr needed)
scripts/      One-off helpers: render the YAM URDF / MuJoCo XML to PNGs
```

`third_party/` (vendored HuMI / i2rt / pyroki source), `data/` (tracker
recordings), and the Python `.venv/` are **git-ignored** — they are local-only
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
