"""BRT encoder → gripper normalisation.

Reads a BRT Modbus-RTU encoder on a USB-serial port (CH340) and maps the
raw register value to a normalised [0, 1] gripper position.
Calibration (open/closed raw values) is persisted to encoder_calibration.json.

BRT register map (manual p.12) — only the ones we touch:
    0x0000   single-turn value (≤16 bit); the gripper position we read.
    0x0008   reset zero point — write 1, current position becomes raw 0.
    0x000E   set midpoint    — write 1, current position becomes the midpoint.
(Other registers — baudrate 0x0005, mode 0x0006, virtual multi-turn
0x0000~0x0001, angular velocity 0x0003, single-turn2 0x0025~0x0026 — exist on
the device but are unused here.)

Hardware zero-set (reset_zero) is persistent and stored in the encoder, not in
software. Use it when the current zero point falls inside the gripper's travel,
which makes raw readings jump or wrap across the 0/fulscale boundary (e.g.
11 → 1000). Zeroing at an endpoint (fully closed) moves the whole stroke into a
single non-wrapping region. NOTE: zeroing shifts every raw value, so any saved
encoder_calibration.json is invalidated and must be re-recorded afterwards.
"""
import json
from pathlib import Path

import numpy as np

try:
    import minimalmodbus
    import minimalmodbus as _mm
except ImportError:
    minimalmodbus = None  # type: ignore[assignment]
    _mm = None

try:
    import serial.tools.list_ports as _list_ports
except ImportError:
    _list_ports = None  # type: ignore[assignment]

CALIBRATION_FILE = Path(__file__).parent.parent / "encoder_calibration.json"

# BRT Modbus registers (manual p.12).
REG_SINGLE_TURN = 0x0000   # single-turn value (≤16 bit) — gripper position
REG_RESET_ZERO = 0x0008    # write 1 → current position = raw 0 (persistent)
REG_SET_MIDPOINT = 0x000E  # write 1 → current position = midpoint (persistent)


def find_serial_port() -> "str | None":
    """Auto-detect a USB-serial port (CH340, CP210x, FTDI, etc.)."""
    if _list_ports is None:
        return None
    ports = _list_ports.comports()
    if not ports:
        return None
    for p in ports:
        desc = (p.description or "").lower()
        mfg = (p.manufacturer or "").lower()
        hwid = (p.hwid or "").lower()
        if any(kw in desc or kw in mfg or kw in hwid
               for kw in ["usb", "ch340", "ch9344", "cp210", "ftdi",
                           "pl2303", "usb-serial"]):
            return p.device
    return ports[0].device


def create_instrument(
    port: str,
    slave_addr: int = 1,
    baudrate: int = 9600,
) -> "minimalmodbus.Instrument":
    """Create a minimalmodbus Instrument for the BRT encoder."""
    if _mm is None:
        raise ImportError("minimalmodbus is not installed")
    inst = _mm.Instrument(port, slave_addr)
    inst.serial.baudrate = baudrate
    inst.serial.bytesize = 8
    inst.serial.parity = _mm.serial.PARITY_NONE
    inst.serial.stopbits = 1
    inst.serial.timeout = 1.0
    inst.mode = _mm.MODE_RTU
    return inst


def read_raw(inst: "minimalmodbus.Instrument") -> "int | None":
    """Read the single-turn register 0x0000. Returns None on failure."""
    try:
        return inst.read_register(REG_SINGLE_TURN, functioncode=3)
    except Exception:
        return None


def reset_zero(inst: "minimalmodbus.Instrument") -> None:
    """Hardware zero-set: the current position becomes raw 0.

    Persistent — stored in the encoder itself, not in software. Use it when the
    current zero point sits inside the gripper's travel, which makes raw
    readings jump/wrap across the 0/fulscale boundary. Zero at an endpoint
    (fully closed) so the whole stroke stays in a non-wrapping region.

    NOTE: this shifts every raw value, invalidating any saved calibration
    (raw_open/raw_closed); re-run calibration afterwards.
    """
    inst.write_register(REG_RESET_ZERO, 1, functioncode=6)


def set_midpoint(inst: "minimalmodbus.Instrument") -> None:
    """Hardware midpoint-set: the current position becomes the midpoint value.

    Persistent — stored in the encoder. Same invalidates-calibration caveat as
    :func:`reset_zero`.
    """
    inst.write_register(REG_SET_MIDPOINT, 1, functioncode=6)


class EncoderCalibration:
    """Linear map from raw encoder value to normalised [0, 1] gripper position.

    Convention: 0 = fully closed, 1 = fully open.
    The map is correct regardless of which of raw_open / raw_closed is larger
    (the span carries the sign), so the physical open/closed ordering does not
    need to match the numeric ordering.
    """

    def __init__(
        self,
        raw_closed: "int | None" = None,
        raw_open: "int | None" = None,
    ) -> None:
        self.raw_closed = raw_closed
        self.raw_open = raw_open

    @property
    def is_ready(self) -> bool:
        return (
            self.raw_closed is not None
            and self.raw_open is not None
            and self.raw_open != self.raw_closed
        )

    def normalise(self, raw: int) -> float:
        """Map raw value to [0, 1]. Returns 0.0 if not calibrated."""
        if not self.is_ready:
            return 0.0
        span = self.raw_open - self.raw_closed  # type: ignore[operator]
        return float(np.clip((raw - self.raw_closed) / span, 0.0, 1.0))

    def save(self, path: Path = CALIBRATION_FILE) -> None:
        path.write_text(json.dumps(
            {"raw_closed": self.raw_closed, "raw_open": self.raw_open}, indent=2
        ))
        print(f"Calibration saved → {path}")

    @classmethod
    def load(cls, path: Path = CALIBRATION_FILE) -> "EncoderCalibration":
        if not path.exists():
            return cls()
        try:
            d = json.loads(path.read_text())
            return cls(raw_closed=d.get("raw_closed"), raw_open=d.get("raw_open"))
        except Exception:
            return cls()

    def __repr__(self) -> str:
        if self.is_ready:
            return (
                f"EncoderCalibration(closed={self.raw_closed}, "
                f"open={self.raw_open}, span={self.raw_open - self.raw_closed})"  # type: ignore[operator]
            )
        return "EncoderCalibration(NOT READY — record open & closed with O/C)"
