"""BRT encoder → gripper normalisation (standalone package).

Self-contained: only needs numpy + minimalmodbus + pyserial. Use this for
encoder testing and calibration without pulling in the full sim_teleop stack
(mujoco / openvr / etc.).
"""
from .encoder import (
    CALIBRATION_FILE,
    EncoderCalibration,
    create_instrument,
    find_port_by_usb_serial,
    find_serial_port,
    read_raw,
    reset_zero,
    resolve_serial_port,
    set_midpoint,
    usb_serial_from_port_info,
)

__all__ = [
    "CALIBRATION_FILE",
    "EncoderCalibration",
    "create_instrument",
    "find_port_by_usb_serial",
    "find_serial_port",
    "read_raw",
    "reset_zero",
    "resolve_serial_port",
    "set_midpoint",
    "usb_serial_from_port_info",
]
