"""Interactive BRT encoder → gripper calibration.

Run from the repository root:

    & ".venv\\Scripts\\python.exe" -m gripper.calibrate
    & ".venv\\Scripts\\python.exe" -m gripper.calibrate --port COM6
    & ".venv\\Scripts\\python.exe" -m gripper.calibrate --open 703 --closed 883   # non-interactive
    & ".venv\\Scripts\\python.exe" -m gripper.calibrate --zero    # hardware zero-set (persistent)
    & ".venv\\Scripts\\python.exe" -m gripper.calibrate --list     # probe ports, ID left/right
    & ".venv\\Scripts\\python.exe" -m gripper.calibrate --side left --port COM3 --stroke-mm 85

Per-side mode (--side left|right): saves the open/closed raw values, the bound
COM port, and the physical stroke (--stroke-mm) into
sim_teleop/configs/gripper_mapping.json instead of the single global
encoder_calibration.json. Use it when two independent encoders (one per
gripper) each need their own calibration. Identify which COM port is which side
with --list (wiggle one gripper; the port whose raw changes is that side).

Workflow (interactive):
    1. Port is auto-detected (override with --port).
    2. You are prompted to move the gripper fully OPEN, then press Enter.
    3. You are prompted to move the gripper fully CLOSED, then press Enter.
    4. Stable samples are captured for each endpoint and saved to
       encoder_calibration.json (repo root).

Direction note: the normalisation map is correct regardless of whether the
OPEN value is larger or smaller than the CLOSED value, so physical ordering
does not matter.

Hardware zero-set (--zero): writes the BRT reset-zero register so the current
position becomes raw 0, stored persistently in the encoder. Use it when raw
readings jump/wrap across the boundary (e.g. 11 → 1000) because the zero point
sits inside the gripper's travel. Move to FULLY CLOSED first, then --zero, then
re-run calibration — zeroing shifts every raw value and invalidates the saved
encoder_calibration.json.
"""
from __future__ import annotations

import argparse
import statistics
import time

from .encoder import (
    CALIBRATION_FILE,
    SIDES,
    EncoderCalibration,
    GripperSide,
    create_instrument,
    load_gripper_mapping,
    probe_ports,
    read_raw,
    reset_zero,
    resolve_serial_port,
    save_gripper_side,
    set_midpoint,
    usb_serial_from_port_info,
)


def _resolve_stroke_mm(side: str, explicit, existing):
    """Pick the stroke (mm): explicit flag > prompt > existing mapping value."""
    if explicit is not None:
        return explicit
    val = input(
        f"Enter {side} gripper stroke (fully-closed → fully-open) in mm "
        f"[blank to keep {existing}]: "
    ).strip()
    if not val:
        return existing
    try:
        return float(val)
    except ValueError:
        print("  (not a number — keeping existing)")
        return existing


def _save_calibration(args, raw_open, raw_closed, port) -> EncoderCalibration:
    """Save to the per-side gripper_mapping.json (--side) or the global file."""
    if args.side:
        existing = load_gripper_mapping()[args.side].calibration.stroke_mm
        stroke = _resolve_stroke_mm(args.side, args.stroke_mm, existing)
        cal = EncoderCalibration(
            raw_closed=raw_closed, raw_open=raw_open, stroke_mm=stroke
        )
        save_gripper_side(
            GripperSide(
                side=args.side,
                port=port,
                usb_serial=args.usb_serial,
                baudrate=args.baudrate,
                slave_addr=args.slave,
                calibration=cal,
            )
        )
        return cal
    cal = EncoderCalibration(raw_open=raw_open, raw_closed=raw_closed)
    cal.save()
    return cal


def _sample_stable(inst, n: int = 10, dt: float = 0.05) -> int | None:
    """Take n samples and return their median. Returns None if all fail."""
    vals = []
    for _ in range(n):
        v = read_raw(inst)
        if v is not None:
            vals.append(v)
        time.sleep(dt)
    if not vals:
        return None
    return int(statistics.median(vals))


def _monitor(inst, seconds: float | None = None) -> None:
    """Print raw values until interrupted."""
    t0 = time.time()
    try:
        while seconds is None or time.time() - t0 < seconds:
            raw = read_raw(inst)
            if raw is None:
                text = "raw=None"
            else:
                text = f"raw={int(raw):4d}"
            print(f"\r{text}    ", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    print()


def _capture(inst, prompt: str) -> int | None:
    """Show live values, wait for Enter, return a stable sample."""
    print(f"\n{prompt}")
    print("(live value shown below — move the gripper now, press Enter to capture)")
    print(">>> ", end="", flush=True)
    # Stream values on the same line until Enter is pressed.
    import threading

    captured: dict[str, int | None] = {"v": None}
    done = threading.Event()

    def reader():
        while not done.is_set():
            captured["v"] = _sample_stable(inst, n=5, dt=0.03)
            print(f"\r>>> raw≈{captured['v']}    ", end="", flush=True)
            time.sleep(0.1)

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    try:
        input()
    finally:
        done.set()
        th.join(timeout=1.0)
    print()
    return captured["v"]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Interactive BRT encoder → gripper calibration"
    )
    p.add_argument("-p", "--port", default=None,
                   help="Serial port (e.g. COM6). Auto-detect if omitted.")
    p.add_argument("--usb-serial", default=None,
                   help="Stable USB serial number. Preferred over COM ports "
                        "when the encoder's COM number can change.")
    p.add_argument("--baudrate", type=int, default=9600)
    p.add_argument("--slave", type=int, default=1, help="Modbus slave address.")
    p.add_argument("--side", choices=SIDES, default=None,
                   help="Calibrate one gripper side; saves to the per-side "
                        "gripper_mapping.json (binds this --port to the side) "
                        "instead of the single global encoder_calibration.json.")
    p.add_argument("--stroke-mm", type=float, default=None,
                   help="Physical jaw stroke (fully-closed → fully-open) in mm "
                        "for --side. Prompted if omitted and not already set.")
    p.add_argument("--list", action="store_true",
                   help="Probe every serial port for a BRT encoder and exit. "
                        "Wiggle one gripper to see which port's raw changes.")
    p.add_argument("--open", type=int, default=None,
                   help="Set raw_open directly (non-interactive).")
    p.add_argument("--closed", type=int, default=None,
                   help="Set raw_closed directly (non-interactive).")
    p.add_argument("--show", action="store_true",
                   help="Only stream live raw values (Ctrl-C to quit).")
    p.add_argument("--reset", action="store_true",
                   help="Delete the saved calibration and exit.")
    p.add_argument("--zero", action="store_true",
                   help="Hardware zero-set: current position becomes raw 0 "
                        "(persistent, stored in the encoder). Move to the "
                        "desired zero position first; invalidates calibration.")
    p.add_argument("--midpoint", action="store_true",
                   help="Hardware midpoint-set (persistent, stored in encoder).")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Skip the confirmation prompt for --zero/--midpoint.")
    args = p.parse_args()

    # ── List / probe ports ──────────────────────────────────────────────────
    if args.list:
        print("Probing serial ports for BRT encoders "
              f"(baudrate={args.baudrate}, slave={args.slave})...")
        results = probe_ports(baudrate=args.baudrate, slave_addr=args.slave)
        if not results:
            print("  (no serial ports found)")
        try:
            import serial.tools.list_ports as list_ports
            serial_by_port = {
                item.device: usb_serial_from_port_info(item)
                for item in list_ports.comports()
            }
        except Exception:
            serial_by_port = {}
        for port, raw in results:
            tag = f"raw={raw}" if raw is not None else "no response"
            usb = serial_by_port.get(port, "")
            usb_text = f" usb_serial={usb}" if usb else ""
            print(f"  {port}:{usb_text} {tag}")
        mapping = load_gripper_mapping()
        print("\nCurrent gripper_mapping.json bindings:")
        for side in SIDES:
            gs = mapping[side]
            print(f"  {side}: port={gs.port} {gs.calibration}")
        print("\nTip: wiggle ONE gripper and re-run --list; the port whose raw "
              "changes is that side. Prefer serial binding, e.g. "
              "-m gripper.calibrate --side left --usb-serial <SER> "
              "--stroke-mm <mm>")
        return

    # ── Reset ──────────────────────────────────────────────────────────────
    if args.reset:
        if CALIBRATION_FILE.exists():
            CALIBRATION_FILE.unlink()
            print(f"Deleted {CALIBRATION_FILE}")
        else:
            print("No calibration file to delete.")
        return

    # ── Port ───────────────────────────────────────────────────────────────
    port = resolve_serial_port(
        port=args.port,
        usb_serial=args.usb_serial,
        baudrate=args.baudrate,
        slave_addr=args.slave,
    )
    if port is None:
        raise SystemExit("ERROR: no serial port found (pass --port COMx or --usb-serial SERIAL)")
    print(f"Using port: {port}")

    inst = create_instrument(port, slave_addr=args.slave, baudrate=args.baudrate)
    try:
        probe = read_raw(inst)
        if probe is None:
            raise SystemExit(
                "ERROR: encoder did not respond. Check wiring/baudrate/slave addr."
            )
        print(f"Encoder OK. current raw = {probe}")

        # ── Hardware zero-set / midpoint ──────────────────────────────────
        if args.zero or args.midpoint:
            if args.zero:
                action = "reset zero (current position → raw 0)"
                print("\nNOTE: move to the position you want to become raw 0.")
            else:
                action = "set midpoint"
            print(f"\nWARNING: about to {action}.")
            print("  This is PERSISTENT (stored in the encoder hardware).")
            print("  The saved encoder_calibration.json will be INVALIDATED;")
            print("  re-run calibration (O/C or --open/--closed) afterwards.")
            if not args.yes:
                if input("Proceed? [y/N] ").strip().lower() != "y":
                    print("Cancelled.")
                    return
            if args.zero:
                reset_zero(inst)
                print("Zero reset: current position is now raw 0.")
            if args.midpoint:
                set_midpoint(inst)
                print("Midpoint set.")
            # The first read right after a write can time out while the encoder
            # applies the change; retry a few times before reporting.
            after = None
            for _ in range(5):
                after = read_raw(inst)
                if after is not None:
                    break
                time.sleep(0.1)
            print(f"Read-back: raw = {after}")
            if args.show:
                print("Streaming raw values (Ctrl-C to quit)...")
                _monitor(inst)
            return

        # ── Show-only mode ─────────────────────────────────────────────────
        if args.show:
            print("Streaming raw values (Ctrl-C to quit)...")
            _monitor(inst)
            return

        # ── Non-interactive set ────────────────────────────────────────────
        if args.open is not None and args.closed is not None:
            cal = _save_calibration(args, args.open, args.closed, port)
            print(cal)
            return

        # ── Interactive capture ────────────────────────────────────────────
        if args.side:
            print(f"\nCurrent {args.side} calibration:",
                  load_gripper_mapping()[args.side].calibration)
        else:
            print("\nCurrent calibration:", EncoderCalibration.load())

        raw_open = args.open
        if raw_open is None:
            raw_open = _capture(inst, "STEP 1: move gripper to FULLY OPEN")
        if raw_open is None:
            raise SystemExit("ERROR: failed to read OPEN value.")

        raw_closed = args.closed
        if raw_closed is None:
            raw_closed = _capture(inst, "STEP 2: move gripper to FULLY CLOSED")
        if raw_closed is None:
            raise SystemExit("ERROR: failed to read CLOSED value.")

        if raw_open == raw_closed:
            raise SystemExit(
                f"ERROR: open ({raw_open}) == closed ({raw_closed}); "
                "calibration needs two distinct values."
            )
        cal = _save_calibration(args, raw_open, raw_closed, port)
        print("\nSaved calibration:", cal)

        # ── Verify ────────────────────────────────────────────────────────
        print("\nVerification (move the gripper and watch the normalised value):")
        print("0.0 = closed, 1.0 = open. Ctrl-C to quit.")
        _monitor(inst)
        # after monitor ends, show a final reading
        v = read_raw(inst)
        if v is not None:
            print(f"final: raw={v}  normalised={cal.normalise(v):.3f}")
    finally:
        inst.serial.close()
        print("Port closed.")


if __name__ == "__main__":
    main()
