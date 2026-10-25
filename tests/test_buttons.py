"""Earphone-button logic with a fake evdev. Run: python tests/test_buttons.py"""
import os, sys, time, types, errno
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import buttons
from buttons import Button, MediaKeySource

ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")

# 1) press rules
b = Button(); s = MediaKeySource(b, verbose=False)
check("play/pause key-down counts", s.handle_key(164, 1, now=100.0) and b.was_pressed())
check("key-up ignored", not s.handle_key(164, 0, now=101.0) and not b.was_pressed())
check("auto-repeat ignored", not s.handle_key(164, 2, now=102.0))
check("other keys ignored (next track, volume)", not s.handle_key(163, 1, now=103.0) and not s.handle_key(115, 1, now=104.0))
s.handle_key(164, 1, now=200.0); b.clear()
check("double event within 0.35 s is one press", not s.handle_key(201, 1, now=200.2) and not b.was_pressed())
check("press after debounce works", s.handle_key(201, 1, now=200.5) and b.was_pressed())
for code in (200, 201, 207):
    b.clear(); check(f"code {code} accepted", s.handle_key(code, 1, now=300.0 + code) and b.was_pressed())
b.press(); b.press(); b.clear()
check("clear() empties queue", not b.was_pressed())

# 2) full device loop with fake evdev: hotplug, keyboard ignored, events, disconnect
class Ev:
    def __init__(self, c, v): self.type, self.code, self.value = 1, c, v
class FakeDev:
    def __init__(self, path, name, keys):
        self.path, self.name, self._keys = path, name, keys
        self.r, self.w = os.pipe(); self.queue = []; self.dead = False
    def capabilities(self): return {1: self._keys}
    def fileno(self): return self.r
    def close(self):
        for fd in (self.r, self.w):
            try: os.close(fd)
            except OSError: pass
    def read(self):
        os.read(self.r, 1)
        if self.dead: raise OSError(errno.ENODEV, "gone")
        while self.queue: yield self.queue.pop(0)
    def send(self, *events): self.queue += events; os.write(self.w, b"x")
    def kill(self): self.dead = True; os.write(self.w, b"x")

present = {}
fake = types.ModuleType("evdev")
fake.list_devices = lambda: list(present)
fake.InputDevice = lambda p: present[p]
sys.modules["evdev"] = fake
buttons.RESCAN_S = 0.2

earbuds = FakeDev("/dev/input/event7", "Galaxy Buds (AVRCP)", [164, 163, 165])
kbd = FakeDev("/dev/input/event0", "USB Keyboard", [30, 31, 164])
mouse = FakeDev("/dev/input/event1", "Mouse", [272])
present.update({earbuds.path: earbuds, kbd.path: kbd, mouse.path: mouse})
changes = []
b = Button(); s = MediaKeySource(b, on_change=changes.append, verbose=False).start()
time.sleep(0.6)
check("earbuds found, keyboard + mouse ignored", list(s._devs) == [earbuds.path] and changes == [True], str(changes))
earbuds.send(Ev(164, 1), Ev(164, 0)); time.sleep(0.2)
check("press delivered from device", b.was_pressed() and not b.was_pressed())
del present[earbuds.path]; earbuds.kill(); time.sleep(0.4)   # real node vanishes on disconnect
check("disconnect detected", changes == [True, False] and not s._devs, str(changes))
earbuds2 = FakeDev("/dev/input/event8", "Galaxy Buds (AVRCP)", [164]); present[earbuds2.path] = earbuds2
time.sleep(0.6)
check("reconnect picked up (new event node)", changes == [True, False, True], str(changes))
earbuds2.send(Ev(200, 1)); time.sleep(0.2)
check("press after reconnect", b.was_pressed())
s.stop()

# 3) name filter
present.clear(); present[earbuds2.path] = earbuds2
s2 = MediaKeySource(Button(), device_match="sony", verbose=False); s2._scan()
check("SG_BUTTON_DEVICE filter excludes non-matching", not s2._devs)

print("\nALL PASS" if ok else "\nFAILURES"); sys.exit(0 if ok else 1)
