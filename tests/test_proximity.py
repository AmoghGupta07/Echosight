"""Proximity alert logic + HC-SR04 timing maths, no hardware. Run: python tests/test_proximity.py"""
import os, sys, threading, types, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import proximity
from proximity import ProximityMonitor, FAR, PERIOD_S

ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")

class FakeAudio:
    def __init__(self): self.beeps, self.hazards, self.says = [], [], []
    def beep(self, f, ms=0, volume=0): self.beeps.append((f, ms, volume))
    def hazard(self, text, key=None, cooldown=4.0): self.hazards.append(text)
    def say(self, text, priority=3, **kw): self.says.append(text)

def run(mon, readings, t0=0.0):
    t = t0
    for r in readings:
        mon.step(r, t); t += PERIOD_S
    return t

def make():
    a = FakeAudio(); return a, ProximityMonitor(a, verbose=False)

# 1) nothing near: silent
a, m = make(); run(m, [200.0] * 30 + [FAR] * 10)
check("far readings are silent", not a.beeps and not a.hazards and not m.alerting)

# 2) approach: one spoken warning, beeps start only below 30 cm
a, m = make(); run(m, [100, 60, 40, 31, 29.9])
check("one reading under 30 cm is not enough (stray echo)", not m.alerting and not a.beeps)
run(m, [28, 27], t0=0.35)
check("two readings under 30 cm raise the alert", m.alerting and len(a.hazards) == 1 and len(a.beeps) >= 1,
      f"hazards={a.hazards} beeps={len(a.beeps)}")
check("spoken text", a.hazards == ["Obstacle ahead"])

# 3) single stray far reading mid-alert does not drop it
a, m = make(); t = run(m, [20, 20, 20]); n = len(a.hazards)
t = run(m, [FAR, 20, FAR, 19], t0=t)
check("dropouts don't end the alert", m.alerting and len(a.hazards) == n)

# 4) hysteresis: 32 cm keeps alert, >35 cm for >0.4 s ends it
a, m = make(); t = run(m, [20, 20, 20])
t = run(m, [32] * 20, t0=t)
check("32 cm keeps alert (hysteresis)", m.alerting)
t = run(m, [60] * 3, t0=t)
check("alert held briefly after object leaves", m.alerting)
t = run(m, [60] * 10, t0=t)
check("alert ends once clear", not m.alerting)
b = len(a.beeps); run(m, [60] * 20, t0=t)
check("no beeps when clear", len(a.beeps) == b)

# 5) beeps get faster and higher the closer you are
def beeps_in_3s(d):
    a, m = make(); run(m, [d] * int(3 / PERIOD_S)); return a.beeps
far, near = beeps_in_3s(28), beeps_in_3s(9)
check("closer = more beeps", len(near) > 2 * len(far), f"{len(far)} @28cm vs {len(near)} @9cm")
check("closer = higher pitch", near[0][0] > far[0][0], f"{far[0][0]:.0f} -> {near[0][0]:.0f} Hz")
check("first beep is immediate", beeps_in_3s(25) != [])

# 6) second episode warns again
a, m = make(); t = run(m, [20] * 4 + [100] * 12 + [20] * 4)
check("new episode speaks again", len(a.hazards) == 2)

# 7) sensor fault: announce once, repeat after 2 min, recovery message
a, m = make(); t = run(m, [None] * 19)
check("19 failures: not yet a fault", not a.hazards)
t = run(m, [None] * 5, t0=t)
check("fault announced once", a.hazards == ["Obstacle sensor is not responding."] and m.faulty)
t = run(m, [None] * 100, t0=t)
check("not repeated within 2 min", len(a.hazards) == 1)
t = run(m, [None], t0=t + 125)
check("repeated after 2 min", len(a.hazards) == 2)
run(m, [100.0], t0=t + 1)
check("recovery announced", a.says == ["Obstacle sensor is working again."] and not m.faulty)

# 8) fault mid-alert stops beeping (no stuck beeps on a dead sensor)
a, m = make(); t = run(m, [15] * 5); t = run(m, [None] * 25, t0=t); b = len(a.beeps)
run(m, [None] * 20, t0=t)
check("no beeps while faulty", not m.alerting and len(a.beeps) == b)

# 9) HC-SR04 maths with fake gpiozero: 30 cm echo, far, none
class Pin:
    when_changed = None
    pin_factory = None
class FakeOut:
    def __init__(self, *a, **k): self.hook = None
    def on(self): pass
    def off(self):
        if self.hook: self.hook()
    def close(self): pass
class FakeIn:
    def __init__(self, *a, **k):
        self.pin = Pin(); self.pin_factory = types.SimpleNamespace(ticks_diff=lambda a, b: a - b)
    def close(self): pass
fake = types.ModuleType("gpiozero"); fake.DigitalInputDevice = FakeIn; fake.DigitalOutputDevice = FakeOut
sys.modules["gpiozero"] = fake
s = proximity.HCSR04()
def respond(echo_s):
    def go():
        if echo_s is None: return
        threading.Timer(0.002, lambda: (s._echo.pin.when_changed(10.0, 1),
                                        s._echo.pin.when_changed(10.0 + echo_s, 0))).start()
    s._trig.hook = go
cm_to_s = lambda cm: cm * 2 / 100 / 343.0
respond(cm_to_s(30)); r = s.measure()
check("30 cm echo -> ~30 cm", r is not None and abs(r - 30) < 0.5, f"got {r}")
respond(cm_to_s(10)); r = s.measure()
check("10 cm echo -> ~10 cm", r is not None and abs(r - 10) < 0.5, f"got {r}")
respond(0.038); r = s.measure()
check("38 ms 'nothing in range' pulse -> FAR", r == FAR, f"got {r}")
respond(None); t0 = time.time(); r = s.measure()
check("no echo -> None after timeout", r is None and time.time() - t0 < 0.2, f"got {r}")

print("\nALL PASS" if ok else "\nFAILURES"); sys.exit(0 if ok else 1)
