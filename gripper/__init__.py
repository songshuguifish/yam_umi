"""BRT encoder → gripper normalisation (standalone package).

Self-contained: only needs numpy + minimalmodbus + pyserial. Use this for
encoder testing and calibration without pulling in the full sim_teleop stack
(mujoco / openvr / etc.).
"""
from .encoder import (
    CALIBRATION_FILE,
    EncoderCalibration,
    create_instrument,
    find_serial_port,
    read_raw,
    reset_zero,
    set_midpoint,
)

__all__ = [
    "CALIBRATION_FILE",
    "EncoderCalibration",
    "create_instrument",
    "find_serial_port",
    "read_raw",
    "reset_zero",
    "set_midpoint",
]
