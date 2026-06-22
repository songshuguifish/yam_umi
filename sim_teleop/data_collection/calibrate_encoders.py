"""Bind and calibrate left/right gripper encoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

import serial.tools.list_ports

from gripper.encoder import (
    create_instrument,
    find_port_by_usb_serial,
    read_raw,
    reset_zero,
    usb_serial_from_port_info,
)


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "encoder_mapping.json"
)
DEFAULT_ROLES = ("left_encoder", "right_encoder")


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": "encoder_roles_v1",
            "recording_host": "same_host",
            "baudrate": 9600,
            "slave": 1,
            "roles": {role: None for role in DEFAULT_ROLES},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _save_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[ENC-CAL] wrote {path}")


def _ports() -> list[dict[str, str]]:
    rows = []
    for port in serial.tools.list_ports.comports():
        rows.append(
            {
                "device": port.device,
                "description": port.description or "",
                "manufacturer": port.manufacturer or "",
                "hwid": port.hwid or "",
            }
        )
    return rows


def _usb_serial(row: dict[str, str]) -> str:
    class PortInfo:
        serial_number = ""
        hwid = row.get("hwid", "")

    return usb_serial_from_port_info(PortInfo)


def _port_by_usb_serial(usb_serial: str) -> str:
    port = find_port_by_usb_serial(usb_serial)
    if port is None:
        raise SystemExit(f"could not find encoder usb_serial={usb_serial!r}")
    return port


def _print_ports() -> None:
    rows = _ports()
    if not rows:
        print("[ENC-CAL] no serial ports found")
        return
    for row in rows:
        print(
            "[ENC-CAL] {device}: {description} {manufacturer} usb_serial={usb_serial} {hwid}".format(
                usb_serial=_usb_serial(row),
                **row,
            )
        )


def _sample_stable(inst, n: int = 10, dt: float = 0.05) -> int | None:
    vals = []
    for _ in range(n):
        value = read_raw(inst)
        if value is not None:
            vals.append(value)
        time.sleep(dt)
    if not vals:
        return None
    return int(statistics.median(vals))


def _read_port(port: str, *, baudrate: int, slave: int) -> int:
    inst = create_instrument(port, slave_addr=slave, baudrate=baudrate)
    try:
        value = _sample_stable(inst)
        if value is None:
            raise RuntimeError(f"encoder did not respond on {port}")
        return value
    finally:
        inst.serial.close()


def _role_entry(
    *,
    port: str,
    role: str,
    raw: int,
    gripper_length: float | None,
    calibration: dict[str, int] | None,
) -> dict[str, Any]:
    port_meta = next((row for row in _ports() if row["device"] == port), {})
    out: dict[str, Any] = {
        "port": port,
        "device": port_meta.get("description") or port_meta.get("device") or "",
        "manufacturer": port_meta.get("manufacturer", ""),
        "usb_serial": _usb_serial(port_meta),
        "hwid": port_meta.get("hwid", ""),
        "verified_raw": raw,
        "notes": f"Bound as {role}.",
    }
    if gripper_length is not None:
        out["gripper_length_m"] = gripper_length
    if calibration is not None:
        cal_out: dict[str, Any] = {
            **calibration,
            "convention": "0=fully_closed, 1=fully_open",
        }
        if gripper_length is not None:
            cal_out["stroke_mm"] = gripper_length * 1000.0
        out["calibration"] = {
            **cal_out,
        }
    return out


def _prompt(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    if not value and default is not None:
        return default
    return value


def _capture_endpoint(port: str, *, baudrate: int, slave: int, name: str) -> int:
    print(f"[ENC-CAL] Move gripper to FULLY {name.upper()}, then press Enter.")
    input()
    raw = _read_port(port, baudrate=baudrate, slave=slave)
    print(f"[ENC-CAL] {name} raw={raw}")
    return raw


def _reset_zero_at_open(port: str, *, baudrate: int, slave: int) -> tuple[int, int]:
    print("[ENC-CAL] Move gripper to FULLY OPEN, then press Enter.")
    print("[ENC-CAL] The encoder hardware zero will be reset at this position.")
    input()
    inst = create_instrument(port, slave_addr=slave, baudrate=baudrate)
    try:
        before = _sample_stable(inst)
        if before is None:
            raise RuntimeError(f"encoder did not respond on {port}")
        print(f"[ENC-CAL] open raw before zero={before}")
        reset_zero(inst)
        time.sleep(0.2)
        after = _sample_stable(inst)
        if after is None:
            raise RuntimeError(f"encoder did not respond after zero on {port}")
        print(f"[ENC-CAL] open raw after zero={after}")
        return before, after
    finally:
        inst.serial.close()


def bind_role(
    *,
    config_path: Path,
    role: str,
    port: str,
    baudrate: int,
    slave: int,
    gripper_length: float | None,
    do_calibrate: bool,
    raw_open: int | None = None,
    raw_closed: int | None = None,
) -> None:
    if role not in DEFAULT_ROLES:
        raise SystemExit(f"role must be one of: {', '.join(DEFAULT_ROLES)}")

    raw = _read_port(port, baudrate=baudrate, slave=slave)
    print(f"[ENC-CAL] {role} {port} raw={raw}")

    calibration = None
    if raw_open is not None or raw_closed is not None:
        if raw_open is None or raw_closed is None:
            raise SystemExit("--raw-open and --raw-closed must be provided together")
        if raw_open == raw_closed:
            raise SystemExit("open and closed raw values are identical")
        calibration = {
            "raw_open": raw_open,
            "raw_closed": raw_closed,
            "span": raw_open - raw_closed,
        }
    elif do_calibrate:
        raw_open = _capture_endpoint(port, baudrate=baudrate, slave=slave, name="open")
        raw_closed = _capture_endpoint(
            port,
            baudrate=baudrate,
            slave=slave,
            name="closed",
        )
        if raw_open == raw_closed:
            raise SystemExit("open and closed raw values are identical")
        calibration = {
            "raw_open": raw_open,
            "raw_closed": raw_closed,
            "span": raw_open - raw_closed,
        }

    config = _load_config(config_path)
    config["baudrate"] = baudrate
    config["slave"] = slave
    config.setdefault("roles", {})
    config["roles"][role] = _role_entry(
        port=port,
        role=role,
        raw=raw,
        gripper_length=gripper_length,
        calibration=calibration,
    )
    for missing_role in DEFAULT_ROLES:
        config["roles"].setdefault(missing_role, None)
    _save_config(config_path, config)


def open_zero_calibrate(
    *,
    config_path: Path,
    role: str,
    port: str,
    baudrate: int,
    slave: int,
    gripper_length: float | None,
) -> None:
    if role not in DEFAULT_ROLES:
        raise SystemExit(f"role must be one of: {', '.join(DEFAULT_ROLES)}")

    zero_before, raw_open = _reset_zero_at_open(
        port,
        baudrate=baudrate,
        slave=slave,
    )
    raw_closed = _capture_endpoint(
        port,
        baudrate=baudrate,
        slave=slave,
        name="closed",
    )
    if raw_open == raw_closed:
        raise SystemExit("open and closed raw values are identical")

    calibration = {
        "raw_open": raw_open,
        "raw_closed": raw_closed,
        "span": raw_open - raw_closed,
        "zero_reference": "fully_open",
        "raw_open_before_zero": zero_before,
    }
    config = _load_config(config_path)
    config["baudrate"] = baudrate
    config["slave"] = slave
    config.setdefault("roles", {})
    config["roles"][role] = _role_entry(
        port=port,
        role=role,
        raw=raw_open,
        gripper_length=gripper_length,
        calibration=calibration,
    )
    for missing_role in DEFAULT_ROLES:
        config["roles"].setdefault(missing_role, None)
    _save_config(config_path, config)
    print(
        "[ENC-CAL] saved open-zero calibration: "
        f"role={role} raw_open={raw_open} raw_closed={raw_closed} "
        f"span={raw_open - raw_closed}"
    )


def resolve_ports(config_path: Path) -> list[tuple[str, str]]:
    config = _load_config(config_path)
    current_ports = _ports()
    resolved = []
    for role in DEFAULT_ROLES:
        entry = config.get("roles", {}).get(role)
        if not entry:
            continue
        usb_serial = entry.get("usb_serial", "")
        saved_hwid = entry.get("hwid", "")
        saved_port = entry.get("port", "")
        match = None
        if usb_serial:
            match = next(
                (row for row in current_ports if _usb_serial(row) == usb_serial),
                None,
            )
        if match is None and saved_hwid:
            match = next(
                (row for row in current_ports if row.get("hwid") == saved_hwid),
                None,
            )
        if match is None and saved_port:
            match = next(
                (row for row in current_ports if row.get("device") == saved_port),
                None,
            )
        if match is None:
            raise SystemExit(f"could not resolve {role} to a current COM port")
        resolved.append((role, match["device"]))
    return resolved


def wizard(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    print("[ENC-CAL] current config:")
    print(json.dumps(config, indent=2))
    print()
    _print_ports()
    print()

    role = _prompt("Role", "left_encoder")
    port = _prompt("Port, e.g. COM4")
    length_text = _prompt("Gripper length in meters, empty to skip", "")
    gripper_length = float(length_text) if length_text else None
    do_calibrate = _prompt("Capture open/closed calibration now? y/N", "N").lower() == "y"
    bind_role(
        config_path=args.config,
        role=role,
        port=port,
        baudrate=args.baudrate,
        slave=args.slave,
        gripper_length=gripper_length,
        do_calibrate=do_calibrate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--baudrate", type=int, default=9600)
    parser.add_argument("--slave", type=int, default=1)
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List serial ports.")

    read_p = sub.add_parser("read", help="Read one encoder port once.")
    read_p.add_argument("--port", default=None)
    read_p.add_argument("--usb-serial", default=None)

    bind_p = sub.add_parser("bind", help="Bind a port to left/right encoder.")
    bind_p.add_argument("--role", choices=DEFAULT_ROLES, required=True)
    bind_p.add_argument("--port", default=None)
    bind_p.add_argument("--usb-serial", default=None)
    bind_p.add_argument("--gripper-length", type=float, default=None)
    bind_p.add_argument("--calibrate", action="store_true")
    bind_p.add_argument("--raw-open", type=int, default=None)
    bind_p.add_argument("--raw-closed", type=int, default=None)

    oz_p = sub.add_parser(
        "open-zero",
        help="Reset hardware zero at fully open, then capture fully closed.",
    )
    oz_p.add_argument("--role", choices=DEFAULT_ROLES, required=True)
    oz_p.add_argument("--port", default=None)
    oz_p.add_argument("--usb-serial", default=None)
    oz_p.add_argument("--gripper-length", type=float, default=None)

    resolve_p = sub.add_parser("resolve", help="Resolve saved roles to current COM ports.")
    resolve_p.add_argument("--plain", action="store_true", help="Print ports only.")

    sub.add_parser("wizard", help="Interactive bind/calibration wizard.")
    args = parser.parse_args()

    if args.cmd is None or args.cmd == "wizard":
        wizard(args)
    elif args.cmd == "list":
        _print_ports()
    elif args.cmd == "read":
        if args.port is None and args.usb_serial is None:
            raise SystemExit("read requires --port or --usb-serial")
        port = args.port or _port_by_usb_serial(args.usb_serial)
        print(_read_port(port, baudrate=args.baudrate, slave=args.slave))
    elif args.cmd == "bind":
        if args.port is None and args.usb_serial is None:
            raise SystemExit("bind requires --port or --usb-serial")
        port = args.port or _port_by_usb_serial(args.usb_serial)
        bind_role(
            config_path=args.config,
            role=args.role,
            port=port,
            baudrate=args.baudrate,
            slave=args.slave,
            gripper_length=args.gripper_length,
            do_calibrate=args.calibrate,
            raw_open=args.raw_open,
            raw_closed=args.raw_closed,
        )
    elif args.cmd == "open-zero":
        if args.port is None and args.usb_serial is None:
            raise SystemExit("open-zero requires --port or --usb-serial")
        port = args.port or _port_by_usb_serial(args.usb_serial)
        open_zero_calibrate(
            config_path=args.config,
            role=args.role,
            port=port,
            baudrate=args.baudrate,
            slave=args.slave,
            gripper_length=args.gripper_length,
        )
    elif args.cmd == "resolve":
        resolved = resolve_ports(args.config)
        if args.plain:
            for _, port in resolved:
                print(port)
        else:
            for role, port in resolved:
                print(f"{role}={port}")
    else:
        raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
