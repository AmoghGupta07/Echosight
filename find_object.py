"""
Guided "find my object": Gemini finds it, a local tracker keeps the guidance real-time.

  1. "Find my keys"  ->  parse_find_request() extracts "keys".
  2. Gemini looks at ONE frame and returns JSON {found, box_2d, distance_m, confidence}.
     That answer takes seconds, so it describes the past. The tracker is started on THAT
     frame and replays every frame captured since, ending up where the object is now.
  3. From then on OpenCV follows the object ~8x per second on the device. Guidance
     ("a little to your left", "turn right") comes from the tracked box, so it keeps up with
     head movement. Gemini re-checks every few seconds to correct drift or a wrong object.
  4. Distance is updated locally from how the box grows as you approach.
  5. Not visible -> scanning hints. Within reach -> "reach forward". Button cancels.

If no OpenCV tracker is available it degrades to Gemini-only guidance.

Test on its own:   python find_object.py "water bottle"
"""
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel

from audio_out import NAV
from camera import to_jpeg

# ---------------- Config ----------------
HFOV_DEG = 52.0          # horizontal field of view (Logitech C270: ~60 deg diagonal -> ~52 wide)
MAX_SEARCH_S = 90        # give up after this long
MAX_ERRORS = 3           # consecutive Gemini failures before stopping
VERIFY_S = 3.0           # while tracking, re-check with Gemini this often (also re-anchors drift)
TICK_S = 0.1
MIN_SPEAK_GAP_S = 1.6    # don't chatter: at most one steering message per this long
REMIND_S = 6.0           # "keep going" if nothing has changed for this long
MAX_TRACK_FRAMES = 2     # per tick: only the newest frames, so slow CPUs never fall behind
CATCHUP_MAX_FRAMES = 8   # replay at most this many frames when catching up after a Gemini answer
STEP_M = 0.7             # metres per walking step
ARRIVED_M = 0.6          # closer than this = within reach
ARRIVED_AREA = 0.20      # box covers this fraction of the frame = within reach
ARRIVE_CONFIRM = 2       # consecutive ticks needed, so one glitchy box can't end the search

Box = tuple              # (ymin, xmin, ymax, xmax), each 0..1


# ---------------- Gemini schema + prompt ----------------
class Detection(BaseModel):
    found: bool
    box_2d: Optional[list[int]] = None      # [ymin, xmin, ymax, xmax], 0-1000
    distance_m: Optional[float] = None
    confidence: Optional[str] = None        # "high" | "medium" | "low"


DETECT_PROMPT = """This is a frame from the camera on a blind person's glasses.
Decide whether the following is clearly visible: {target}

Rules:
- found=true only if you can actually see it. If unsure, found=false. Never guess.
- box_2d is its bounding box as [ymin, xmin, ymax, xmax], each 0-1000 (fraction of image height/width times 1000).
- distance_m is your best estimate of its distance from the camera in metres.
- confidence is "high", "medium" or "low".
If it is not visible, return found=false and leave the other fields out."""


# ---------------- Understanding the request ----------------
_FIND = re.compile(
    r"\b(?:(?:help me )?(?:find|look for|locate|search for)|where(?:'s| is| are))\s+(?P<t>.+)", re.I)


def parse_find_request(query: str) -> Optional[str]:
    """'Can you find my water bottle please?' -> 'water bottle'. None if not a find request."""
    m = _FIND.search(query)
    if not m:
        return None
    t = re.split(r"\b(?:please|for me|right now|around here|in here)\b", m.group("t"), flags=re.I)[0]
    t = t.strip(" .,?!")
    # "where is ..." is only a find request when it's about the user's own things
    if m.group(0).lower().startswith("where") and not re.match(r"my\b", t, re.I):
        return None
    t = re.sub(r"^(?:my|the|a|an)\s+", "", t, flags=re.I)
    if not t or len(t.split()) > 5 or re.match(r"(?:out|a way|the way)\b", t, re.I):
        return None
    return t


# ---------------- Geometry -> words ----------------
def valid_box(det: Detection) -> Optional[Box]:
    if not det.found or not det.box_2d or len(det.box_2d) != 4:
        return None
    ymin, xmin, ymax, xmax = [min(max(v, 0), 1000) / 1000 for v in det.box_2d]
    if xmax - xmin < 0.01 or ymax - ymin < 0.01:
        return None
    return (ymin, xmin, ymax, xmax)


def box_area(b: Box) -> float:
    return max(0.0, b[3] - b[1]) * max(0.0, b[2] - b[0])


def estimate_distance(box: Box, ref_area: Optional[float], ref_dist: Optional[float]) -> Optional[float]:
    """Gemini's distance at detection time, rescaled as the box grows or shrinks (d ~ 1/sqrt(area))."""
    if ref_dist is None or not ref_area:
        return ref_dist
    a = box_area(box)
    if a <= 0:
        return ref_dist
    return max(0.2, ref_dist * math.sqrt(ref_area / a))


def distance_phrase(d: Optional[float]) -> str:
    if d is None:
        return ""
    if d < ARRIVED_M:
        return "within arm's reach"
    if d <= 7:
        n = max(1, round(d / STEP_M))
        return f"about {n} step{'s' if n != 1 else ''} away"
    return f"about {round(d)} metres away"


@dataclass
class Guidance:
    category: str            # changes only when the spoken direction would change
    direction: str           # "a little to your left"
    distance_text: str
    dist_bucket: Optional[int]
    arrived: bool
    side: str
    bearing: float           # degrees, negative = left


def guidance_from_box(box: Box, distance_m: Optional[float]) -> Guidance:
    ymin, xmin, ymax, xmax = box
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    bearing = (cx - 0.5) * HFOV_DEG
    side = "left" if bearing < 0 else "right"
    a = abs(bearing)
    if a < 6:
        cat, text = "ahead", "straight ahead"
    elif a < 15:
        cat, text = f"slight_{side}", f"a little to your {side}"
    else:
        cat, text = side, f"turn {side}"
    if cy < 0.2:
        cat, text = cat + "_up", text + ", look up"
    elif cy > 0.85:
        cat, text = cat + "_down", text + ", look down"
    arrived = (distance_m is not None and distance_m <= ARRIVED_M) or box_area(box) >= ARRIVED_AREA
    bucket = None if distance_m is None else max(1, round(distance_m / STEP_M)) // 2
    return Guidance(cat, text, distance_phrase(distance_m), bucket, arrived, side, bearing)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


SCAN_HINTS = [
    "I don't see it yet. Slowly turn your head to the right.",
    "Not yet. Keep turning slowly.",
    "Still nothing. Try looking lower, at tables and the floor.",
    "Nothing yet. Now slowly turn the other way.",
    "Still can't see it. Keep sweeping slowly.",
]


def _subsample(frames, k):
    """At most k frames, evenly spread, always including the newest."""
    n = len(frames)
    if n <= k:
        return frames
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [frames[i] for i in idx]


# ---------------- Local tracker ----------------
def _make_cv_tracker():
    import cv2
    for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"):
        fn = getattr(cv2, name, None) or getattr(getattr(cv2, "legacy", None), name, None)
        if fn:
            try:
                return fn(), name
            except Exception:
                continue
    return None, None


class BoxTracker:
    """Follows one box through a sequence of small frames. Boxes are normalised 0..1."""

    def __init__(self):
        self._t = None
        self.name = None
        self.box: Optional[Box] = None

    @staticmethod
    def available() -> bool:
        return _make_cv_tracker()[0] is not None

    def init(self, frame, box: Box) -> bool:
        h, w = frame.shape[:2]
        ymin, xmin, ymax, xmax = box
        rect = (int(xmin * w), int(ymin * h),
                max(8, int((xmax - xmin) * w)), max(8, int((ymax - ymin) * h)))
        self._t, self.name = _make_cv_tracker()
        self.box = box
        if self._t is None:
            return True                       # no tracker: just remember the box
        try:
            ret = self._t.init(frame, rect)
            return True if ret is None else bool(ret)
        except Exception as e:
            print(f"[find] tracker init failed: {e}")
            return False

    def update(self, frame) -> bool:
        if self._t is None:
            return self.box is not None
        ok, rect = self._t.update(frame)
        if not ok:
            return False
        x, y, w, h = rect
        H, W = frame.shape[:2]
        if w < 4 or h < 4:
            return False
        self.box = (max(0.0, y / H), max(0.0, x / W), min(1.0, (y + h) / H), min(1.0, (x + w) / W))
        return True


# ---------------- The loop ----------------
class _Request:
    def __init__(self, ts, small):
        self.ts, self.small = ts, small
        self.started = time.time()
        self.done = threading.Event()
        self.result: Optional[Detection] = None
        self.error: Optional[Exception] = None


class ObjectFinder:
    def __init__(self, cam, gemini, audio, button):
        self.cam, self.gemini, self.audio, self.button = cam, gemini, audio, button

    # --- helpers ---
    def _say(self, text, replace=True):
        self.audio.say(text, NAV, group="nav", replace=replace)

    def _launch(self, target: str) -> _Request:
        ts, frame, small = self.cam.best_recent(0.6)
        jpeg = to_jpeg(frame, 896, 75)
        req = _Request(ts, small)
        prompt = DETECT_PROMPT.format(target=target)

        def work():
            try:
                req.result = self.gemini.ask_structured(prompt, [jpeg], Detection)
            except Exception as e:
                req.error = e
            finally:
                req.done.set()

        threading.Thread(target=work, daemon=True, name="find-gemini").start()
        return req

    def _cancel(self) -> str:
        self.audio.stop()
        self._say("Okay, stopped.", replace=False)
        return "cancelled"

    # --- main loop ---
    def run(self, target: str) -> str:
        """Returns 'found', 'cancelled', 'timeout' or 'error'."""
        cam, button = self.cam, self.button
        intro = f"Okay, looking for {target}. I'll use the camera. Press the button any time to stop."
        if cam.is_low_light():
            intro += " It's quite dark, so I may miss things."
        self._say(intro, replace=False)
        button.clear()

        tracker = BoxTracker()
        verify_s = VERIFY_S if BoxTracker.available() else 0.0
        if verify_s == 0.0:
            print("[find] no OpenCV tracker available: Gemini-only guidance")

        t0 = time.time()
        req: Optional[_Request] = None
        tracking = seen = low_conf = lost_said = False
        last_ts = last_verify = last_say = 0.0
        ref_area = ref_dist = None
        last_cat = last_bucket = None
        errors = misses = neg = arrive_n = 0

        while time.time() - t0 < MAX_SEARCH_S:
            if button.was_pressed():
                return self._cancel()
            now = time.time()

            # 1) keep one Gemini request in flight (back-to-back while searching)
            if req is None and (not tracking or now - last_verify >= verify_s):
                req = self._launch(target)
                last_verify = now

            # 2) advance the tracker through the newest frames
            if tracking:
                for t, f in cam.frames_since(last_ts)[-MAX_TRACK_FRAMES:]:
                    last_ts = t
                    if not tracker.update(f):
                        tracking = False
                        print("[find] tracker lost the object")
                        if seen and not lost_said:
                            self._say("I lost it. Turn back a little.")
                            lost_said = True
                        break

            # 3) a Gemini answer has arrived
            fallback_box = None
            gem_arrived = False
            if req is not None and req.done.is_set():
                r, req = req, None
                if r.error is not None:
                    errors += 1
                    print(f"[find] Gemini error {errors}/{MAX_ERRORS}: {type(r.error).__name__}: {r.error}")
                    if errors >= MAX_ERRORS:
                        self.audio.stop()
                        self._say("I'm having trouble reaching the assistant, so I'm stopping the search.",
                                  replace=False)
                        return "error"
                else:
                    errors = 0
                    det = r.result
                    box = valid_box(det)
                    print(f"[find] Gemini ({time.time() - r.started:.1f}s): found={det.found} "
                          f"box={det.box_2d} dist={det.distance_m} conf={det.confidence}")
                    if box:
                        misses = neg = 0
                        lost_said = False
                        ok = r.small is not None and tracker.init(r.small, box)
                        if ok:   # catch up: replay every frame captured while Gemini was thinking
                            last_ts = r.ts
                            for t, f in _subsample(cam.frames_since(r.ts), CATCHUP_MAX_FRAMES):
                                if not tracker.update(f):
                                    ok = False
                                    break
                                last_ts = t
                        tracking = ok
                        ref_area, ref_dist = box_area(box), det.distance_m
                        gem_arrived = guidance_from_box(box, det.distance_m).arrived
                        if not ok:
                            fallback_box = box          # act on the (older) Gemini box once
                        if not seen and det.confidence == "low":
                            low_conf = True
                    else:
                        neg += 1
                        if tracking and neg < 2:
                            pass                          # one miss: trust the tracker for now
                        else:
                            tracking = False
                            misses += 1
                            if seen and misses == 1:
                                if not lost_said:
                                    self._say("I lost it. Turn back a little.")
                                    lost_said = True
                            else:
                                self._say(SCAN_HINTS[(misses - 1) % len(SCAN_HINTS)])

            # 4) steer from the tracked box
            box_now = tracker.box if tracking else fallback_box
            if box_now is not None:
                dist = estimate_distance(box_now, ref_area, ref_dist)
                g = guidance_from_box(box_now, dist)
                arrive_n = arrive_n + 1 if g.arrived else 0
                if arrive_n >= ARRIVE_CONFIRM or gem_arrived or (g.arrived and fallback_box is not None):
                    if abs(g.bearing) < 8:
                        self._say("It's right in front of you. Reach forward.")
                    else:
                        self._say(f"It's right here, a little to your {g.side}. Reach {g.side}.")
                    return "found"

                gap = now - last_say
                dir_changed = g.category != last_cat
                dist_changed = g.dist_bucket != last_bucket
                if (last_bucket is not None and g.dist_bucket is not None
                        and 0 < g.dist_bucket - last_bucket < 2):
                    dist_changed = False   # estimate jitter: never announce "farther" unless it's clearly real
                if not seen or ((dir_changed or dist_changed) and gap >= MIN_SPEAK_GAP_S):
                    parts = []
                    if not seen or dir_changed:
                        parts.append(g.direction)
                    if g.distance_text and (not seen or dist_changed):
                        parts.append(g.distance_text)
                    if parts:
                        lead = ""
                        if not seen:
                            lead = ("I think I see it, but I'm not sure. " if low_conf else "I see it. ")
                        self._say(lead + _cap(", ".join(parts)) + ".")
                    last_say, last_cat, last_bucket, seen = now, g.category, g.dist_bucket, True
                elif gap >= REMIND_S:
                    self._say("Keep going.")
                    last_say = now

            time.sleep(TICK_S)

        self._say(f"I couldn't find {target}. Try moving to a different spot and ask me again.")
        return "timeout"


# ---------------- Stand-alone test ----------------
def benchmark_tracker():
    """How fast is the tracker on THIS machine? Run it on the Pi:  python find_object.py --bench"""
    import numpy as np
    rng = np.random.default_rng(0)
    frames = [np.clip(rng.integers(0, 255, (270, 480, 3)), 0, 255).astype(np.uint8) for _ in range(20)]
    tr = BoxTracker()
    tr.init(frames[0], (0.4, 0.4, 0.6, 0.6))
    t0 = time.time()
    for f in frames[1:]:
        tr.update(f)
    ms = (time.time() - t0) / 19 * 1000
    print(f"tracker: {_make_cv_tracker()[1]}, {ms:.0f} ms/frame -> max ~{1000 / max(ms, 1):.0f} updates/s")
    print("Catch-up after a Gemini answer costs about", f"{ms * CATCHUP_MAX_FRAMES / 1000:.1f}s.")


if __name__ == "__main__":
    import sys
    if "--bench" in sys.argv:
        benchmark_tracker()
        sys.exit()

    from audio_out import AudioManager
    from camera import CameraBuffer
    from gemini_client import GeminiAssistant
    from voice_loop import EnterKeyButton

    target = " ".join(sys.argv[1:]) or "water bottle"
    audio = AudioManager().start()
    cam = CameraBuffer().start()
    time.sleep(2)
    result = ObjectFinder(cam, GeminiAssistant(), audio, EnterKeyButton()).run(target)
    audio.wait_idle(timeout=10)
    print("Result:", result)
    cam.stop()
    audio.close()
