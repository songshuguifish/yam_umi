"""Re-export of the standalone gripper package for backwards compatibility.

The implementation now lives in the top-level `gripper` package
(see gripper/encoder.py). This shim keeps `sim_teleop.__main__` working
unchanged.
"""
from gripper.encoder import (  # noqa: F401
    CALIBRATION_FILE,
    EncoderCalibration,
    create_instrument,
    find_serial_port,
    read_raw,
    reset_zero,
    set_midpoint,
)
