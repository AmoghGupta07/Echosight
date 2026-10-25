"""
Step 2: Camera thread + rolling frame buffer.

Runs continuously in the background and keeps the last few seconds of frames.
When the user asks something, call get_query_jpegs() to grab the best recent
frames (sharpest in each time slice, so head-motion blur is avoided), already
resized and JPEG-encoded for Gemini.

Works with a Pi Camera (picamera2) or a USB webcam (OpenCV), auto-detected.
Run this file directly to test:  python camera.py
"""
import sys
import threading
import time
from collections import deque

import cv2

# ---------------- Config ----------------
WIDTH, HEIGHT = 1280, 720
FPS = 15
BUFFER_SECONDS = 3
TRACK_SECONDS = 10           # small frames kept longer, so the tracker can catch up after a slow Gemini call
TRACK_WIDTH = 480            # width of the small frames used for tracking and quality metrics
LOW_LIGHT_THRESHOLD = 40     # mean gray level (0-255) below this = "dark"
USB_CAMERA_INDEX = None      # None = scan indices 0..19 for the first camera that gives frames
# V4L2 on Linux/Pi, DirectShow on Windows, default elsewhere (macOS)
_CV_BACKEND = (cv2.CAP_V4L2 if sys.platform.startswith("linux")
               else cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)


class CameraBuffer:
    def __init__(self, width=WIDTH, height=HEIGHT, fps=FPS, seconds=BUFFER_SECONDS):
        self.width, self.height, self.fps = width, height, fps
        self._buf = deque(maxlen=fps * seconds)   # items: (timestamp, frame, sharpness, brightness)
        self._track = deque(maxlen=fps * TRACK_SECONDS)   # items: (timestamp, small BGR frame)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._read = None        # function returning one BGR frame
        self._grab = None        # optional: blocks until the camera has a new frame (USB path)
        self._close = None
        self.backend = None

    # ---------- lifecycle ----------
    def start(self):
        self._open_camera()
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera")
        self._thread.start()
        # wait for first frame so callers never see an empty buffer
        t0 = time.time()
        while not self._buf and time.time() - t0 < 5:
            time.sleep(0.05)
        if not self._buf:
            raise RuntimeError("Camera started but produced no frames")
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._close:
            self._close()

    def _open_camera(self):
        try:
            from picamera2 import Picamera2

            cam = Picamera2()
            cfg = cam.create_video_configuration(
                main={
                    "size": (self.width, self.height),
                    "format": "RGB888",
                },
                controls={"FrameRate": self.fps},
            )
            cam.configure(cfg)
            cam.start()

            self._read = lambda: cam.capture_array("main")
            self._close = lambda: (cam.stop(), cam.close())
            self.backend = "picamera2"

        except Exception as e:
            print(
                f"[camera] picamera2 unavailable ({e}); "
                "trying USB webcam"
            )

            cap = self._open_usb()

            def read():
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("USB camera read failed")
                return frame

            self._grab = None
            self._read = read
            self._close = cap.release
            self.backend = "opencv"

    def _open_usb(self):
        """Open a USB webcam using a working OpenCV backend."""

        candidates = (
            [USB_CAMERA_INDEX]
            if USB_CAMERA_INDEX is not None
            else range(20)
        )

        for i in candidates:
            cap = cv2.VideoCapture(i, _CV_BACKEND)

            if not cap.isOpened():
                cap.release()
                continue

            # Use the same resolution that worked in test_camera.py.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Do not force MJPEG; let the camera choose its format.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Warm up the camera and find a valid frame.
            ok = False
            frame = None

            for _ in range(10):
                ok, frame = cap.read()

                if ok and frame is not None:
                    break

                time.sleep(0.1)

            if ok and frame is not None:
                actual_h, actual_w = frame.shape[:2]

                print(
                    f"[camera] using camera index {i} "
                    f"({actual_w}x{actual_h})"
                )

                return cap

            cap.release()

        raise RuntimeError(
            "No working USB camera found. "
            "Check Windows camera permissions and close other "
            "applications using the webcam."
        )

    # ---------- capture loop ----------
    def _run(self):
        interval = 1.0 / self.fps
        failures = 0
        last_t = 0.0
        while not self._stop.is_set():
            try:
                if self._grab:
                    # USB: pull every frame the camera produces so the queue never goes
                    # stale, but only decode (the expensive part) at the target FPS.
                    self._grab()
                    if time.time() - last_t < interval:
                        continue
                t0 = time.time()
                last_t = t0
                frame = self._read()
                failures = 0
            except Exception as e:
                failures += 1
                print(f"[camera] read error ({failures}): {e}")
                if failures > 20:
                    print("[camera] too many failures, stopping camera thread")
                    return
                time.sleep(0.2)
                continue

            # Small copy: used for cheap quality metrics AND kept for the object tracker
            small = cv2.resize(frame, (TRACK_WIDTH, int(TRACK_WIDTH * frame.shape[0] / frame.shape[1])))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            bright = float(gray.mean())

            with self._lock:
                self._buf.append((t0, frame, sharp, bright))
                self._track.append((t0, small))

            if not self._grab:
                time.sleep(max(0.0, interval - (time.time() - t0)))

    # ---------- public API ----------
    def latest(self):
        """Most recent BGR frame (numpy array)."""
        with self._lock:
            return self._buf[-1][1]

    def brightness(self, window_s=1.0):
        """Average brightness (0-255) over the last window_s seconds."""
        cutoff = time.time() - window_s
        with self._lock:
            vals = [b for t, _, _, b in self._buf if t >= cutoff]
        return sum(vals) / len(vals) if vals else 0.0

    def is_low_light(self):
        return self.brightness() < LOW_LIGHT_THRESHOLD

    def get_frames(self, n=3, window_s=2.0):
        """
        Return up to n BGR frames, oldest first. The window is split into n time
        slices and the sharpest frame in each slice is chosen.
        """
        cutoff = time.time() - window_s
        with self._lock:
            items = [it for it in self._buf if it[0] >= cutoff] or list(self._buf)[-1:]
        if n <= 1 or len(items) <= 1:
            return [max(items, key=lambda it: it[2])[1]]

        n = min(n, len(items))
        size = len(items) / n
        picked = []
        for i in range(n):
            chunk = items[int(i * size): int((i + 1) * size)] or [items[-1]]
            picked.append(max(chunk, key=lambda it: it[2])[1])
        return picked

    def best_recent(self, window_s=0.6):
        """
        Sharpest frame of the last window_s seconds, as (timestamp, full_frame, small_frame).
        The timestamp lets the tracker catch up from exactly this frame later.
        """
        cutoff = time.time() - window_s
        with self._lock:
            items = [it for it in self._buf if it[0] >= cutoff] or list(self._buf)[-1:]
            ts, frame, _, _ = max(items, key=lambda it: it[2])
            small = next((s for t, s in self._track if t == ts), None)
        return ts, frame, small

    def frames_since(self, ts):
        """Small frames captured after ts, oldest first: [(timestamp, small_frame), ...]."""
        with self._lock:
            return [(t, s) for t, s in self._track if t > ts]

    def get_query_jpegs(self, n=3, window_s=2.0, max_side=1024, quality=70):
        """Best recent frames as JPEG bytes, ready to send to Gemini."""
        return [to_jpeg(f, max_side, quality) for f in self.get_frames(n, window_s)]

    def measured_fps(self, window_s=2.0):
        cutoff = time.time() - window_s
        with self._lock:
            ts = [t for t, *_ in self._buf if t >= cutoff]
        return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else 0.0


def to_jpeg(frame, max_side=1024, quality=70) -> bytes:
    """Downscale so the longest side <= max_side, then JPEG-encode."""
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


# ---------------- Self-test ----------------
if __name__ == "__main__":
    cam = CameraBuffer().start()
    print(f"Backend: {cam.backend}")
    time.sleep(3)  # let the buffer fill
    print(f"Measured FPS:   {cam.measured_fps():.1f}")
    print(f"Brightness:     {cam.brightness():.0f}/255  (low light: {cam.is_low_light()})")

    t = time.time()
    jpegs = cam.get_query_jpegs(n=3)
    print(f"Grabbed {len(jpegs)} frames in {(time.time() - t) * 1000:.0f} ms, "
          f"sizes: {[len(j) // 1024 for j in jpegs]} KB")
    for i, j in enumerate(jpegs):
        with open(f"test_frame_{i}.jpg", "wb") as f:
            f.write(j)
    print("Saved test_frame_0..2.jpg  (open them to check focus, angle, and exposure)")
    cam.stop()
