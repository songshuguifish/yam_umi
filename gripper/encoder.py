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
from dataclasses import dataclass
from pathlib import Path
import re

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

# Per-side (left/right) gripper binding: COM port + open/closed raw + physical
# stroke. Lives next to tracker_mapping.json and is git-ignored (machine- and
# hardware-specific: ports and raw values change across machines / re-zeroing).
GRIPPER_MAPPING_FILE = (
    Path(__file__).parent.parent
    / "sim_teleop"
    / "configs"
    / "gripper_mapping.json"
)
SIDES = ("left", "right")
USB_SERIAL_RE = re.compile(r"(?:SER|SERIAL)=([^ ]+)", re.IGNORECASE)

# BRT Modbus registers (manual p.12).
REG_SINGLE_TURN = 0x0000   # single-turn value (≤16 bit) — gripper position
REG_RESET_ZERO = 0x0008    # write 1 → current position = raw 0 (persistent)
REG_SET_MIDPOINT = 0x000E  # write 1 → current position = midpoint (persistent)


def usb_serial_from_port_info(port_info) -> str:
    """Return the stable USB serial from a pyserial ListPortInfo object."""
    serial_number = getattr(port_info, "serial_number", None)
    if serial_number:
        return str(serial_number)
    match = USB_SERIAL_RE.search(getattr(port_info, "hwid", "") or "")
    return match.group(1) if match else ""


def find_port_by_usb_serial(usb_serial: str) -> "str | None":
    """Resolve a stable USB serial number to the current Windows COM port."""
    if _list_ports is None:
        return None
    for port_info in _list_ports.comports():
        if usb_serial_from_port_info(port_info) == usb_serial:
            return port_info.device
    return None


def resolve_serial_port(
    *,
    port: "str | None" = None,
    usb_serial: "str | None" = None,
    baudrate: int = 9600,
    slave_addr: int = 1,
    probe: bool = True,
) -> "str | None":
    """Resolve explicit COM, USB serial, or auto-detected encoder port."""
    if port:
        return port
    if usb_serial:
        return find_port_by_usb_serial(usb_serial)
    return find_serial_port(
        baudrate=baudrate,
        slave_addr=slave_addr,
        probe=probe,
    )


def find_serial_port(
    *,
    baudrate: int = 9600,
    slave_addr: int = 1,
    probe: bool = True,
) -> "str | None":
    """Auto-detect the BRT encoder's USB-serial port.

    Candidates are USB-serial adapters (CH340, CP210x, FTDI, PL2303, ...).
    When ``probe`` is true (default), each candidate is opened and read once;
    the first that returns a valid register value wins. This avoids silently
    grabbing the wrong device when several USB-serial adapters are plugged in.
    Returns None if nothing matches / responds (callers should then require an
    explicit ``--port``).
    """
    if _list_ports is None:
        return None
    ports = _list_ports.comports()
    if not ports:
        return None

    keywords = ("ch340", "ch9344", "cp210", "ftdi", "pl2303", "usb-serial", "usb")
    matched: list[str] = []
    for p in ports:
        haystack = " ".join(
            (p.description or "", p.manufacturer or "", p.hwid or "")
        ).lower()
        if any(kw in haystack for kw in keywords):
            matched.append(p.device)
    # Keyword-matched ports first; fall back to every port if none matched.
    ordered = matched if matched else [p.device for p in ports]

    if not probe:
        return ordered[0]

    for port in ordered:
        try:
            inst = create_instrument(port, slave_addr=slave_addr, baudrate=baudrate)
        except Exception:
            continue
        inst.serial.timeout = 0.3  # BRT replies in ~28ms; keep probes fast
        try:
            if read_raw(inst) is not None:
                return port
        except Exception:
            pass
        finally:
            try:
                inst.serial.close()
            except Exception:
                pass
    return None


def probe_ports(
    *,
    baudrate: int = 9600,
    slave_addr: int = 1,
) -> "list[tuple[str, int | None]]":
    """Probe every serial port and return ``(port, raw)`` for each.

    ``raw`` is the BRT single-turn register value if the port responds, else
    None. Used to discover how many encoders are attached and — by wiggling one
    gripper at a time and watching which port's ``raw`` changes — to decide
    which COM port is the left vs the right gripper.
    """
    if _list_ports is None:
        return []
    results: list[tuple[str, int | None]] = []
    for p in _list_ports.comports():
        try:
            inst = create_instrument(p.device, slave_addr=slave_addr, baudrate=baudrate)
        except Exception:
            results.append((p.device, None))
            continue
        inst.serial.timeout = 0.3  # BRT replies in ~28ms; keep probes fast
        try:
            results.append((p.device, read_raw(inst)))
        except Exception:
            results.append((p.device, None))
        finally:
            try:
                inst.serial.close()
            except Exception:
                pass
    return results


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
        stroke_mm: "float | None" = None,
    ) -> None:
        self.raw_closed = raw_closed
        self.raw_open = raw_open
        # Physical gripper stroke (fully-closed → fully-open jaw opening), in mm.
        # Manually measured. Lets normalise() be scaled to a metric opening.
        self.stroke_mm = stroke_mm

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

    def metric_m(self, raw: int) -> float:
        """Map raw value to a metric jaw opening in metres.

        Returns NaN if the stroke length is unknown (normalise() is unitless).
        """
        if self.stroke_mm is None:
            return float("nan")
        return self.normalise(raw) * (self.stroke_mm / 1000.0)

    def save(self, path: Path = CALIBRATION_FILE) -> None:
        payload = {"raw_closed": self.raw_closed, "raw_open": self.raw_open}
        if self.stroke_mm is not None:
            payload["stroke_mm"] = self.stroke_mm
        path.write_text(json.dumps(payload, indent=2))
        print(f"Calibration saved → {path}")

    @classmethod
    def load(cls, path: Path = CALIBRATION_FILE) -> "EncoderCalibration":
        if not path.exists():
            return cls()
        try:
            d = json.loads(path.read_text())
            return cls(
                raw_closed=d.get("raw_closed"),
                raw_open=d.get("raw_open"),
                stroke_mm=d.get("stroke_mm"),
            )
        except Exception:
            return cls()

    def __repr__(self) -> str:
        if self.is_ready:
            stroke = (
                f", stroke={self.stroke_mm}mm" if self.stroke_mm is not None else ""
            )
            return (
                f"EncoderCalibration(closed={self.raw_closed}, "
                f"open={self.raw_open}, span={self.raw_open - self.raw_closed}{stroke})"  # type: ignore[operator]
            )
        return "EncoderCalibration(NOT READY — record open & closed with O/C)"


@dataclass
class GripperSide:
    """One side's gripper binding: COM port + per-side encoder calibration."""

    side: str
    port: "str | None"
    baudrate: int
    slave_addr: int
    calibration: EncoderCalibration
    usb_serial: "str | None" = None

    def to_dict(self) -> dict:
        return {
            "port": self.port,
            "usb_serial": self.usb_serial,
            "baudrate": self.baudrate,
            "slave_addr": self.slave_addr,
            "raw_open": self.calibration.raw_open,
            "raw_closed": self.calibration.raw_closed,
            "stroke_mm": self.calibration.stroke_mm,
        }

    @classmethod
    def from_dict(cls, side: str, d: dict) -> "GripperSide":
        return cls(
            side=side,
            port=d.get("port"),
            baudrate=int(d.get("baudrate", 9600)),
            slave_addr=int(d.get("slave_addr", 1)),
            calibration=EncoderCalibration(
                raw_closed=d.get("raw_closed"),
                raw_open=d.get("raw_open"),
                stroke_mm=d.get("stroke_mm"),
            ),
            usb_serial=d.get("usb_serial"),
        )


def _empty_side(side: str) -> GripperSide:
    return GripperSide(
        side=side,
        port=None,
        baudrate=9600,
        slave_addr=1,
        calibration=EncoderCalibration(),
        usb_serial=None,
    )


def load_gripper_mapping(
    path: Path = GRIPPER_MAPPING_FILE,
) -> "dict[str, GripperSide]":
    """Load the per-side gripper mapping. Missing sides come back uncalibrated."""
    sides = {side: _empty_side(side) for side in SIDES}
    if not path.exists():
        return sides
    try:
        data = json.loads(path.read_text())
        for side, entry in data.get("sides", {}).items():
            if side in sides and isinstance(entry, dict):
                sides[side] = GripperSide.from_dict(side, entry)
    except Exception:
        pass
    return sides


def save_gripper_side(
    side: GripperSide,
    path: Path = GRIPPER_MAPPING_FILE,
) -> None:
    """Merge one side into the gripper mapping file (creating it if needed)."""
    if side.side not in SIDES:
        raise ValueError(f"unknown side {side.side!r}; expected one of {SIDES}")
    mapping = load_gripper_mapping(path)
    mapping[side.side] = side
    payload = {
        "schema": "gripper_mapping_v1",
        "sides": {name: mapping[name].to_dict() for name in SIDES},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Gripper mapping saved → {path} (side={side.side})")
