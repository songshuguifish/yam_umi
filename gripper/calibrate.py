"""Interactive BRT encoder → gripper calibration.

Run from the repository root:

    & ".venv\\Scripts\\python.exe" -m gripper.calibrate
    & ".venv\\Scripts\\python.exe" -m gripper.calibrate --port COM6
    & ".venv\\Scripts\\python.exe" -m gripper.calibrate --open 703 --closed 883   # non-interactive
    & ".venv\\Scripts\\python.exe" -m gripper.calibrate --zero    # hardware zero-set (persistent)

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
    EncoderCalibration,
    create_instrument,
    find_serial_port,
    read_raw,
    reset_zero,
    set_midpoint,
)


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
    """Print raw values until KeyboardInterrupt (or for `seconds`)."""
    t0 = time.time()
    try:
        while seconds is None or time.time() - t0 < seconds:
            print(f"\rraw = {read_raw(inst)}", end="", flush=True)
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
    p.add_argument("--baudrate", type=int, default=9600)
    p.add_argument("--slave", type=int, default=1, help="Modbus slave address.")
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
                        "(persistent, stored in the encoder). Move to FULLY "
                        "CLOSED first; invalidates the saved calibration.")
    p.add_argument("--midpoint", action="store_true",
                   help="Hardware midpoint-set (persistent, stored in encoder).")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Skip the confirmation prompt for --zero/--midpoint.")
    args = p.parse_args()

    # ── Reset ──────────────────────────────────────────────────────────────
    if args.reset:
        if CALIBRATION_FILE.exists():
            CALIBRATION_FILE.unlink()
            print(f"Deleted {CALIBRATION_FILE}")
        else:
            print("No calibration file to delete.")
        return

    # ── Port ───────────────────────────────────────────────────────────────
    port = args.port or find_serial_port()
    if port is None:
        raise SystemExit("ERROR: no serial port found (pass --port COMx)")
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
                print("\nNOTE: for a gripper, move to FULLY CLOSED first so the")
                print("      whole stroke stays in a non-wrapping region.")
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
            return

        # ── Show-only mode ─────────────────────────────────────────────────
        if args.show:
            print("Streaming raw values (Ctrl-C to quit)...")
            _monitor(inst)
            return

        # ── Non-interactive set ────────────────────────────────────────────
        if args.open is not None and args.closed is not None:
            cal = EncoderCalibration(raw_open=args.open, raw_closed=args.closed)
            cal.save()
            print(cal)
            return

        # ── Interactive capture ────────────────────────────────────────────
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

        cal = EncoderCalibration(raw_open=raw_open, raw_closed=raw_closed)
        if not cal.is_ready:
            raise SystemExit(
                f"ERROR: open ({raw_open}) == closed ({raw_closed}); "
                "calibration needs two distinct values."
            )
        cal.save()
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
