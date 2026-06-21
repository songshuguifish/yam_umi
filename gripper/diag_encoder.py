"""Quick BRT encoder diagnostic.

Reads mode + single-turn + angular-velocity + turns in a loop so you can see
WHICH registers respond while you move the gripper. Use this before reaching
for --zero: if raw is stuck at 0 and never changes, the problem is not the
zero-point location but either the encoder mode (0x0006 != 0 → query reads of
0x0000 may not return live values) or the physical magnet/coupling.

Run from the repo root:
    & ".venv\\Scripts\\python.exe" -m gripper.diag_encoder
    & ".venv\\Scripts\\python.exe" -m gripper.diag_encoder -p COM8
"""
from __future__ import annotations

import argparse
import time

from .encoder import create_instrument, find_serial_port


def _rd(inst, reg: int):
    try:
        return inst.read_register(reg, functioncode=3)
    except Exception:
        return "ERR"


def main() -> None:
    p = argparse.ArgumentParser(description="BRT encoder diagnostic")
    p.add_argument("-p", "--port", default=None, help="Serial port, e.g. COM8.")
    p.add_argument("--baudrate", type=int, default=9600)
    p.add_argument("--slave", type=int, default=1)
    p.add_argument("-n", "--count", type=int, default=20)
    p.add_argument("--dt", type=float, default=0.3)
    args = p.parse_args()

    port = args.port or find_serial_port()
    if port is None:
        raise SystemExit("ERROR: no serial port found (pass -p COMx)")
    print(f"Using port: {port}")

    inst = create_instrument(port, slave_addr=args.slave, baudrate=args.baudrate)
    try:
        mode = _rd(inst, 0x0006)
        print(f"mode (0x0006) = {mode}   (0=query, 1=auto single-turn, "
              f"4=auto multi-turn, 5=auto angular-velocity)")
        print("Now move the gripper slowly — watch which columns change:\n")
        print(f"{'#':>3}  {'single 0x0000':>14}  {'ang.vel 0x0003':>15}  "
              f"{'turns 0x0002':>13}")
        for i in range(args.count):
            s = _rd(inst, 0x0000)
            v = _rd(inst, 0x0003)
            t = _rd(inst, 0x0002)
            print(f"{i + 1:>3}  {str(s):>14}  {str(v):>15}  {str(t):>13}",
                  flush=True)
            time.sleep(args.dt)
    finally:
        inst.serial.close()
        print("\nPort closed.")


if __name__ == "__main__":
    main()
