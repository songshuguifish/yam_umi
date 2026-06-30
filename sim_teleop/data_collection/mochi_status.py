"""Optional Clawd Mochi status client — direct Bluetooth Low Energy.

Pushes coarse recording lifecycle states straight to a Clawd Mochi desk
display (ESP32-C3) over Bluetooth Low Energy — no separate bridge daemon
needed. Only five states are understood (see MochiStatus).

A background daemon thread owns one warm BLE connection: it scans for the
mochi, connects, and forwards queued states over the GATT characteristic.
``push()`` only validates and enqueues, so it never blocks the recorder.

Gracefully degrades to a no-op when ``bleak`` is not installed, no Bluetooth
adapter is present, or the mochi is off — data collection is never affected.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import Optional

try:
    from bleak import BleakClient, BleakScanner

    _BLEAK = True
except Exception:  # pragma: no cover - bleak is optional
    _BLEAK = False


# Must match the firmware (clawd_mochi.ino). The mochi advertises its SERVICE
# UUID but NOT a local name (the name set via NimBLEDevice::init is only readable
# after connecting), so we discover it by service UUID. CHAR_UUID is the
# characteristic that accepts a state string, reachable only after connecting.
MOCHI_NAME = "Clawd-Mochi"
MOCHI_SERVICE = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"  # addServiceUUID() in .ino
CHAR_UUID = "1c95d5e3-df7e-4c5a-8b45-9e0c0a4e0e10"
SCAN_TIMEOUT = 10.0
CONNECT_TIMEOUT = 15.0
POLL_INTERVAL = 0.05


class MochiStatus:
    """The five states understood by the mochi firmware."""

    IDLE = "idle"          # resting / warming sensors / between runs
    WORKING = "working"    # an episode is being recorded
    WAITING = "waiting"    # needs operator input (press c / calibrate)
    DONE = "done"          # episode saved successfully
    ERROR = "error"        # sensor / camera / save failure


_VALID = frozenset(
    {
        MochiStatus.IDLE,
        MochiStatus.WORKING,
        MochiStatus.WAITING,
        MochiStatus.DONE,
        MochiStatus.ERROR,
    }
)

_queue: deque[str] = deque()
_lock = threading.Lock()
_started = False
_start_lock = threading.Lock()


def push(state: str) -> None:
    """Queue a state to send to the mochi over BLE. Non-blocking; never raises."""
    if state not in _VALID:
        raise ValueError(f"unknown mochi state: {state!r}")
    if not _BLEAK:
        return
    _ensure_worker()
    with _lock:
        _queue.append(state)


def _ensure_worker() -> None:
    global _started
    with _start_lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_worker, name="mochi-ble", daemon=True).start()


def _worker() -> None:
    try:
        asyncio.run(_ble_loop())
    except Exception:
        pass


async def _find_mochi():
    # The firmware advertises its service UUID but no local name, so discover by
    # service UUID (fast + reliable); fall back to name matching for other builds.
    try:
        devs = await BleakScanner.discover(
            timeout=SCAN_TIMEOUT, service_uuids=[MOCHI_SERVICE]
        )
        if devs:
            return devs[0]
        for d in await BleakScanner.discover(timeout=SCAN_TIMEOUT):
            if d.name and MOCHI_NAME in d.name:
                return d
    except Exception:
        return None
    return None


def _drain() -> Optional[str]:
    """Pop all queued states; return the most recent (latest wins)."""
    state: Optional[str] = None
    with _lock:
        while _queue:
            state = _queue.popleft()
    return state


async def _ble_loop() -> None:
    """Keep one warm BLE connection open and forward queued states forever."""
    while True:
        dev = await _find_mochi()
        if dev is None:
            await asyncio.sleep(5)
            continue
        try:
            async with BleakClient(dev.address, timeout=CONNECT_TIMEOUT) as client:
                # re-send idle on (re)connect so the mochi is never stuck
                await client.write_gatt_char(CHAR_UUID, MochiStatus.IDLE.encode())
                while client.is_connected:
                    state = _drain()
                    if state is not None:
                        await client.write_gatt_char(CHAR_UUID, state.encode())
                    await asyncio.sleep(POLL_INTERVAL)
        except Exception:
            pass
        await asyncio.sleep(3)
