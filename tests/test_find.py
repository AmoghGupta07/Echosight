"""Finder + tracker with a simulated camera and Gemini. Run: python tests/test_find.py"""
import sys, time, threading, types
from collections import deque
import numpy as np, cv2

try:
    import pydantic  # use the real one when installed
except ImportError:
    stub = types.ModuleType("pydantic")
    class BaseModel:
        def __init__(self, **kw):
            for k, v in kw.items(): setattr(self, k, v)
    stub.BaseModel = BaseModel
    sys.modules["pydantic"] = stub
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import find_object as fo
from find_object import ObjectFinder, Detection, BoxTracker, parse_find_request
from audio_out import AudioManager

W, H, FPS = 480, 270, 10
print("tracker available:", BoxTracker.available(), "->", fo._make_cv_tracker()[1])

# ---------------- simulated world ----------------
rng = np.random.default_rng(1)
BG = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 2.5)
BG = cv2.addWeighted(BG, 0.6, rng.integers(60, 120, (H, W, 3), dtype=np.uint8), 0.4, 0)
PATCH = np.clip(np.array([30, 30, 220]) + rng.integers(-25, 25, (48, 48, 3)), 0, 255).astype(np.uint8)

def truth(t, hidden_until=0.0):
    """Object centre x drifts right->centre while it grows (user turning + walking closer)."""
    if t < hidden_until: return None
    tt = t - hidden_until
    cx = max(0.5, 0.85 - 0.07 * tt)
    wf = min(0.6, 0.06 + 0.04 * tt)
    hf = wf * 1.2
    cy = 0.55
    return (cy - hf / 2, cx - wf / 2, cy + hf / 2, cx + wf / 2)   # ymin,xmin,ymax,xmax

def render(box):
    img = BG.copy()
    if box:
        y0, x0, y1, x1 = [int(round(v)) for v in (box[0]*H, box[1]*W, box[2]*H, box[3]*W)]
        y0, x0, y1, x1 = max(0,y0), max(0,x0), min(H,y1), min(W,x1)
        if x1 - x0 > 2 and y1 - y0 > 2:
            img[y0:y1, x0:x1] = cv2.resize(PATCH, (x1 - x0, y1 - y0))
    return img

class FakeCam:
    def __init__(self, hidden_until=0.0):
        self.t0 = time.time(); self.hidden = hidden_until
        self.buf = deque(maxlen=FPS * 12); self.run = True
        threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        nxt = time.time()
        while self.run:
            t = time.time()
            self.buf.append((t, render(truth(t - self.t0, self.hidden))))
            nxt += 1 / FPS; time.sleep(max(0, nxt - time.time()))
    def is_low_light(self): return False
    def best_recent(self, w=0.6):
        while not self.buf: time.sleep(0.01)
        t, f = self.buf[-1]; return t, f, f
    def frames_since(self, ts): return [(t, f) for t, f in list(self.buf) if t > ts]
    def truth_at(self, t): return truth(t - self.t0, self.hidden)

class FakeGemini:
    def __init__(self, cam, latency=2.0, fail=False):
        self.cam, self.latency, self.fail, self.calls = cam, latency, fail, 0
    def ask_structured(self, prompt, jpegs, schema):
        self.calls += 1
        img = cv2.imdecode(np.frombuffer(jpegs[0], np.uint8), cv2.IMREAD_COLOR)
        time.sleep(self.latency)
        if self.fail: raise RuntimeError("503 simulated outage")
        b, g, r = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
        ys, xs = np.where((r > 150) & (g < 110) & (b < 110))
        if len(xs) < 40: return Detection(found=False)
        h, w = img.shape[:2]
        box = [int(ys.min()/h*1000), int(xs.min()/w*1000), int(ys.max()/h*1000), int(xs.max()/w*1000)]
        wf = (xs.max() - xs.min()) / w
        return Detection(found=True, box_2d=box, distance_m=round(0.35 / max(wf, 0.02), 1), confidence="high")

class FakeButton:
    def __init__(self, at=None): self.at, self.t0, self.fired = at, time.time(), False
    def clear(self): self.t0 = time.time()
    def was_pressed(self):
        if self.at is not None and not self.fired and time.time() - self.t0 > self.at:
            self.fired = True; return True
        return False

class FakeStream:
    def __init__(self, rate, cb): self.cb, self.run = cb, False; self.rate = rate
    def start(self):
        self.run = True; threading.Thread(target=self._l, daemon=True).start()
    def _l(self):
        nxt = time.time()
        while self.run:
            self.cb(np.zeros((512, 1), np.float32), 512, None, None)
            nxt += 512 / 8000; time.sleep(max(0, nxt - time.time()))
    def stop(self): self.run = False
    def close(self): pass

def make_audio(t0, log):
    a = AudioManager(rate=8000, synth_fn=lambda txt: np.full(int(0.28 * len(txt.split()) * 8000), 0.1, np.float32),
                     stream_factory=FakeStream, verbose=False).start()
    real_say = a.say
    def say(text, *args, **kw):
        log.append((round(time.time() - t0, 1), text)); return real_say(text, *args, **kw)
    a.say = say
    return a

def run_scenario(name, hidden_until=0.0, latency=2.0, fail=False, cancel_at=None):
    print(f"\n===== {name} =====")
    cam = FakeCam(hidden_until); time.sleep(0.5)
    t0 = time.time(); log = []
    audio = make_audio(t0, log)
    gem = FakeGemini(cam, latency, fail)
    res = ObjectFinder(cam, gem, audio, FakeButton(cancel_at)).run("red cup")
    audio.wait_idle(5); audio.close(); cam.run = False
    for t, txt in log: print(f"  [{t:5.1f}s] {txt}")
    print(f"  -> result={res}, gemini calls={gem.calls}, total {time.time()-t0:.1f}s")
    return res, log, gem.calls

ok = True
def check(label, cond):
    global ok; ok &= bool(cond); print(("PASS  " if cond else "FAIL  ") + label)

# --- 1) tracker catch-up accuracy (the core claim) ---
print("\n===== tracker catch-up accuracy =====")
cam = FakeCam(); time.sleep(3.5)
frames = list(cam.buf)
(t_a, f_a) = frames[5]                      # the frame Gemini "looked at"
box_a = cam.truth_at(t_a)
tr = BoxTracker(); assert tr.init(f_a, box_a)
last = None
for t, f in fo._subsample(frames[6:], fo.CATCHUP_MAX_FRAMES):
    if not tr.update(f): print("  tracker lost"); break
    last = t
true_now = cam.truth_at(last)
c = lambda b: ((b[0]+b[2])/2, (b[1]+b[3])/2)
(ty, tx), (ry, rx) = c(tr.box), c(true_now)
print(f"  replayed 8 sampled frames out of {len(frames)-6} ({last - t_a:.1f}s of motion)")
print(f"  centre tracked ({tx:.3f},{ty:.3f}) vs true ({rx:.3f},{ry:.3f}) -> horizontal error {abs(tx-rx)*fo.HFOV_DEG:.1f} degrees")
print(f"  (size: tracked width {tr.box[3]-tr.box[1]:.2f} vs true {true_now[3]-true_now[1]:.2f}: size lags while approaching - expected)")
check("catch-up lands on the object's CURRENT direction (< 4 degrees)", abs(tx - rx) * fo.HFOV_DEG < 4)
cam.run = False

# --- 2) full scenarios ---
res, log, calls = run_scenario("A: object visible, 2s Gemini latency, user approaches")
txt = " | ".join(m for _, m in log)
check("finds it", res == "found")
check("announces direction then updates between Gemini calls", sum(1 for _, m in log if "right" in m or "ahead" in m or "left" in m) >= 2)
check("final instruction is to reach", "Reach" in log[-1][1])

res, log, calls = run_scenario("B: object hidden for 6s (scanning hints), then appears", hidden_until=6.0)
check("gives scanning hints while searching", sum(1 for _, m in log if "don't see" in m or "turn" in m.lower() or "Not yet" in m or "nothing" in m.lower()) >= 2)
check("finds it once visible", res == "found")

res, log, calls = run_scenario("C: user presses the button after 3s", hidden_until=60.0, cancel_at=3.0)
check("cancels cleanly", res == "cancelled" and log[-1][1] == "Okay, stopped.")

res, log, calls = run_scenario("D: Gemini outage", fail=True, latency=0.3)
check("stops after repeated Gemini failures", res == "error" and calls == 3)

print("\nQUERY PARSING")
for q, want in [("Find my keys", "keys"), ("where is the exit", None), ("Where are my glasses?", "glasses")]:
    check(f"parse {q!r}", parse_find_request(q) == want)

print("\nALL PASS" if ok else "\nSOME FAILED")
