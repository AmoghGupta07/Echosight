"""Audio manager logic with a simulated speaker. Run: python tests/test_audio_logic.py"""
import sys, time, threading
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from audio_out import AudioManager, HAZARD, NAV, REPLY, INFO

RATE, BLOCK = 8000, 512

class FakeStream:
    """Calls the audio callback in real time and records what a speaker would play."""
    def __init__(self, rate, cb, log):
        self.cb, self.log, self.run = cb, log, False
        self.t0 = None
    def start(self):
        self.run = True; self.t0 = time.time()
        threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        nxt = time.time()
        while self.run:
            out = np.zeros((BLOCK, 1), dtype=np.float32)
            self.cb(out, BLOCK, None, None)
            lvl = round(float(np.max(np.abs(out))), 2)
            self.log.append((round(time.time() - self.t0, 2), lvl))
            nxt += BLOCK / RATE
            time.sleep(max(0, nxt - time.time()))
    def stop(self): self.run = False
    def close(self): pass

def synth(text):
    name, dur = text.split(":")
    level = (ord(name[0]) - 64) * 0.05          # A=.05, B=.10, H=.40 ...
    return np.full(int(float(dur) * RATE), level, dtype=np.float32)

def make():
    log = []
    m = AudioManager(rate=RATE, synth_fn=synth, stream_factory=lambda r, cb: FakeStream(r, cb, log), verbose=False)
    return m.start(), log

def sequence(log):
    """Collapse the recording into [(start_time, level, duration)] ignoring silence."""
    segs, cur = [], None
    for t, lvl in log:
        if lvl > 0:
            if cur and cur[1] == lvl: cur[2] = t - cur[0] + BLOCK / RATE
            else:
                cur = [t, lvl, BLOCK / RATE]; segs.append(cur)
        else:
            cur = None
    return [(round(a, 1), l, round(d, 1)) for a, l, d in segs]

def names(seq):  # level -> letter
    return "".join(chr(int(round(l / 0.05)) + 64) for _, l, _ in seq)

ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")

# 1) hazard interrupts a reply, reply RESUMES afterwards
m, log = make()
m.say("A:2.0", REPLY); time.sleep(0.8)
m.say("H:0.5", HAZARD); m.wait_idle(8); time.sleep(0.2)
seq = sequence(log); m.close()
check("hazard interrupts reply, reply resumes", names(seq) == "AHA", f"{seq}")
if len(seq) == 3:
    played_a = seq[0][2] + seq[2][2]
    check("resumed reply replays a little (rewind) but not everything", 2.0 < played_a < 2.8, f"total A audio {played_a:.1f}s")
    check("hazard started promptly (<0.5s after request)", seq[1][0] - 0.8 < 0.5, f"started at {seq[1][0]}s, requested 0.8s")

# 2) newer NAV replaces older waiting NAV (only newest matters)
m, log = make()
m.say("B:0.8", NAV, group="nav", replace=True)
time.sleep(0.2)
m.say("C:0.4", NAV, group="nav", replace=True)
m.say("D:0.4", NAV, group="nav", replace=True)
m.say("E:0.4", NAV, group="nav", replace=True)
m.wait_idle(8); time.sleep(0.2)
seq = sequence(log); m.close()
check("NAV replace keeps playing + newest only", names(seq) == "BE", names(seq))

# 3) NAV interrupts a reply, but reply resumes after
m, log = make()
m.say("A:1.5", REPLY); time.sleep(0.5)
m.say("C:0.4", NAV); m.wait_idle(8); time.sleep(0.2)
seq = sequence(log); m.close()
check("NAV interrupts reply then reply resumes", names(seq) == "ACA", names(seq))

# 4) stop() silences replies but keeps hazards
m, log = make()
m.say("A:2.0", REPLY); m.say("B:2.0", REPLY); time.sleep(0.4)
m.say("H:0.4", HAZARD, group="hazard"); time.sleep(0.05)
m.stop(); m.wait_idle(8); time.sleep(0.2)
seq = sequence(log); m.close()
check("stop() cancels replies, hazard still speaks", names(seq).endswith("H") and "B" not in names(seq) and names(seq).count("A") <= 1, names(seq))

# 5) stop() with no hazard = silence
m, log = make()
m.say("A:2.0", REPLY); time.sleep(0.4); m.stop(); t_stop = time.time() - log[0][0] if False else None
time.sleep(0.5); n_before = len(log)
seq_after = [x for x in log[-6:] if x[1] > 0]
m.close()
check("stop() silences current speech quickly", len(seq_after) == 0, f"{seq_after}")

# 6) hazard de-duplication (cooldown)
m, log = make()
r = [m.hazard("H:0.3", key="car", cooldown=2.0), m.hazard("H:0.3", key="car", cooldown=2.0), m.hazard("G:0.3", key="step", cooldown=2.0)]
m.wait_idle(8); m.close()
check("same hazard suppressed within cooldown, different one allowed", r == [True, False, True], str(r))

# 7) stale INFO dropped when higher-priority speech hogs the speaker; same-priority order kept
m, log = make()
m.say("A:1.0", REPLY)
m.say("I:0.3", INFO)           # waits behind the reply
m.say("B:0.3", REPLY)
m.wait_idle(8); time.sleep(0.2)
seq = sequence(log); m.close()
check("INFO waits for replies (order A B I)", names(seq) == "ABI", names(seq))

# 8) beep mixes over speech (level exceeds speech alone)
m, log = make()
m.say("A:1.0", REPLY); time.sleep(0.3); m.beep(1000, 100, 0.3); time.sleep(0.3)
peak = max(l for _, l in log); m.close()
check("beep plays on top of speech", peak > 0.05 + 0.2, f"peak level {peak}")

# 9) wait_idle abort
m, log = make()
m.say("A:3.0", REPLY); t = time.time()
r = m.wait_idle(timeout=10, abort=lambda: time.time() - t > 0.3)
m.stop(); m.close()
check("wait_idle abort works", r is False and time.time() - t < 1.5)

print("\nALL PASS" if ok else "\nSOME FAILED")
