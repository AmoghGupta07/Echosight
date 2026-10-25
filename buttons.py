"""
Push-to-talk button sources.

The assistant only needs three things from a button: clear(), wait_press(), was_pressed().
`Button` provides them; *sources* feed presses into it:

  MediaKeySource  the play/pause button on Bluetooth earphones (Linux, via evdev)
  EnterKeySource  the Enter key, only when running in a terminal (handy over SSH / on a PC)

How the earphone button reaches us: when earphones connect, BlueZ (AVRCP) creates a Linux input
device (named like "Your Earbuds (AVRCP)") that emits media key events. We read those events
directly, so it works headless, at boot, with no desktop and no media player running.

Different earphones send different codes for the same button (play/pause toggle, or separate
play and pause), so all of them count as a press. Run `python buttons.py` to see what yours sends.

Environment:
  SG_BUTTON_DEVICE   only listen to input devices whose name contains this text (case-insensitive),
                     e.g. SG_BUTTON_DEVICE=avrcp or the earphones' name. Default: any media-key device.
"""
import os
import queue
import select
import sys
import threading
import time
from typing import Callable, Optional

# Linux input event codes (linux/input-event-codes.h)
KEY_PLAYPAUSE, KEY_PLAYCD, KEY_PAUSECD, KEY_PLAY = 164, 200, 201, 207
MEDIA_CODES = frozenset({KEY_PLAYPAUSE, KEY_PLAYCD, KEY_PAUSECD, KEY_PLAY})
KEY_A = 30                          # a real keyboard has this; earphones never do

DEBOUNCE_S = 0.35                   # ignore a second press this soon after the first
RESCAN_S = 3.0                      # how often to look for newly connected earphones
DEVICE_MATCH = os.environ.get("SG_BUTTON_DEVICE", "").strip().lower()


class Button:
    """Thread-safe press queue. Same interface the old EnterKeyButton had."""

    def __init__(self):
        self._q: queue.Queue[float] = queue.Queue()
        self._sources: list = []

    def press(self):
        self._q.put(time.time())

    def clear(self):
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def wait_press(self):
        return self._q.get()

    def was_pressed(self) -> bool:
        try:
            self._q.get_nowait()
            return True
        except queue.Empty:
            return False

    def close(self):
        for s in self._sources:
            s.stop()


class EnterKeySource:
    """Enter key = button. Only useful with a terminal attached."""

    def __init__(self, button: Button):
        self.button = button

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="enter-key").start()
        return self

    def _run(self):
        while True:
            try:
                input()
            except EOFError:
                return
            self.button.press()

    def stop(self):
        pass


class MediaKeySource:
    """Bluetooth earphone play/pause (and play, pause) -> button presses. Handles hot-plugging."""

    def __init__(self, button: Button, on_change: Optional[Callable[[bool], None]] = None,
                 device_match: str = DEVICE_MATCH, debounce_s: float = DEBOUNCE_S,
                 log_raw: bool = False, verbose: bool = True):
        self.button = button
        self.on_change = on_change
        self.device_match = (device_match or "").lower()
        self.debounce_s = debounce_s
        self.log_raw = log_raw
        self.verbose = verbose
        self.connected = False
        self._devs: dict = {}                 # path -> evdev.InputDevice
        self._last_press = 0.0
        self._warned_perm = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="media-button")

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _log(self, msg: str):
        if self.verbose:
            print(f"[button] {msg}")

    # ---- the part that decides what counts as a press (unit-tested without hardware)
    def handle_key(self, code: int, value: int, now: Optional[float] = None) -> bool:
        """value: 1 = key down, 0 = up, 2 = auto-repeat. Only a fresh key-down counts."""
        if self.log_raw:
            print(f"[button] raw key code={code} value={value}")
        if value != 1 or code not in MEDIA_CODES:
            return False
        now = time.time() if now is None else now
        if now - self._last_press < self.debounce_s:
            return False
        self._last_press = now
        self.button.press()
        if self.log_raw:
            print("[button] -> PRESS")
        return True

    # ---- device handling
    def _scan(self):
        from evdev import InputDevice, list_devices
        for path in list_devices():
            if path in self._devs:
                continue
            try:
                dev = InputDevice(path)
            except PermissionError:
                if not self._warned_perm:
                    self._warned_perm = True
                    self._log("permission denied on /dev/input/*; add your user to the 'input' "
                              "group (sudo usermod -aG input $USER) and log in again")
                continue
            except OSError:
                continue
            keys = set(dev.capabilities().get(1, []))        # 1 = EV_KEY
            ok = bool(keys & MEDIA_CODES) and KEY_A not in keys
            if ok and self.device_match and self.device_match not in (dev.name or "").lower():
                ok = False
            if not ok:
                dev.close()
                continue
            self._devs[path] = dev
            self._log(f"listening to {dev.name!r} ({path})")

    def _drop(self, path: str):
        dev = self._devs.pop(path, None)
        if dev is not None:
            self._log(f"lost {dev.name!r}")
            try:
                dev.close()
            except Exception:
                pass

    def _update_connected(self):
        now_connected = bool(self._devs)
        if now_connected != self.connected:
            self.connected = now_connected
            if self.on_change:
                try:
                    self.on_change(now_connected)
                except Exception as e:
                    print(f"[button] on_change failed: {e}")

    def _run(self):
        next_scan = 0.0
        while not self._stop.is_set():
            if time.time() >= next_scan:
                try:
                    self._scan()
                except Exception as e:
                    self._log(f"scan failed: {type(e).__name__}: {e}")
                next_scan = time.time() + RESCAN_S
                self._update_connected()
            if not self._devs:
                self._stop.wait(0.5)
                continue
            try:
                ready, _, _ = select.select(list(self._devs.values()), [], [], 0.5)
            except (OSError, ValueError):                 # a device vanished mid-select
                for path, dev in list(self._devs.items()):
                    try:
                        dev.fileno()
                        os.fstat(dev.fileno())
                    except (OSError, ValueError):
                        self._drop(path)
                self._update_connected()
                continue
            for dev in ready:
                try:
                    for ev in dev.read():
                        if ev.type == 1:                  # EV_KEY
                            self.handle_key(ev.code, ev.value)
                except BlockingIOError:
                    pass                                  # nothing to read after all
                except OSError:                           # earphones disconnected
                    self._drop(dev.path)
            self._update_connected()


def make_button(on_media_change: Optional[Callable[[bool], None]] = None,
                use_enter: bool = True, use_media: bool = True) -> Button:
    """Build the button the assistant uses, with every source that is available here."""
    button = Button()
    if use_enter and sys.stdin is not None and sys.stdin.isatty():
        src = EnterKeySource(button).start()
        button._sources.append(src)
    if use_media:
        try:
            import evdev  # noqa: F401
        except ImportError:
            if sys.platform.startswith("linux"):
                print("[button] python 'evdev' is not installed, so the earphone button is "
                      "disabled (sudo apt install python3-evdev, or pip install evdev)")
        else:
            src = MediaKeySource(button, on_change=on_media_change).start()
            button._sources.append(src)
    return button


# ---------------- standalone test: python buttons.py ----------------
if __name__ == "__main__":
    try:
        from evdev import InputDevice, list_devices
    except ImportError:
        sys.exit("evdev is not installed: sudo apt install python3-evdev   (Linux only)")
    print("Input devices visible to this user:")
    for p in list_devices():
        try:
            d = InputDevice(p)
            media = bool(set(d.capabilities().get(1, [])) & MEDIA_CODES)
            print(f"  {p}  {d.name!r}{'   <- has media keys' if media else ''}")
        except PermissionError:
            print(f"  {p}  (permission denied: add your user to the 'input' group)")
    print("\nPress the play/pause button on your earphones (Ctrl+C to quit).")
    print("Raw key codes are printed so you can see exactly what your earphones send.\n")
    b = Button()
    MediaKeySource(b, log_raw=True).start()
    try:
        while True:
            b.wait_press()
            print("*** BUTTON PRESS ***")
    except KeyboardInterrupt:
        print("\nBye.")
