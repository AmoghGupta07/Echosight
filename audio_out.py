"""
Step 6: Audio output manager.

One place that owns the speaker. Everything that wants to talk or beep goes through here.

  Priorities (lower number = more important)
    HAZARD  1  "Car approaching, left"   interrupts anything, de-duplicated, expires fast
    NAV     2  "A little to your left"   interrupts replies; newer message replaces older one
    REPLY   3  Gemini answers            if interrupted by a hazard, RESUMES afterwards
    INFO    4  status chatter            only when idle, expires if it waits too long

  Design
    - A synthesis thread turns text into audio ahead of time (Piper, loaded once).
    - A player thread picks the most important ready utterance.
    - A tiny mixer feeds ONE persistent output stream, so beeps can play on top of speech
      (needed for ultrasonic proximity beeps in the next step).
    - say() never blocks. stop() silences replies instantly (hazards are kept).

Used by assistant.py and find_object.py. Run audio_test.py to hear it work.
"""
import itertools
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

# ---------------- Config ----------------
HAZARD, NAV, REPLY, INFO = 1, 2, 3, 4
MAX_AGE = {HAZARD: 3.0, NAV: 4.0, REPLY: None, INFO: 8.0}   # seconds a message may wait

PIPER_VOICE = "voices/en_US-lessac-medium.onnx"
SAMPLE_RATE = 22050          # adopted from the voice file automatically if it differs
SPEAKER_DEVICE = None        # None = system default; or index/name from sd.query_devices()
RESUME_REWIND_S = 0.4        # when a reply resumes after a hazard, back up this far


@dataclass(eq=False)         # identity comparison: audio arrays can't be compared with ==
class Utterance:
    seq: int
    priority: int
    text: str
    group: Optional[str]
    created: float
    audio: Optional[np.ndarray] = None
    synthesizing: bool = False
    cancelled: bool = False
    interrupted: bool = False
    pos: int = 0


class AudioManager:
    def __init__(self, voice_path: str = PIPER_VOICE, rate: int = SAMPLE_RATE,
                 device=SPEAKER_DEVICE, synth_fn: Optional[Callable[[str], np.ndarray]] = None,
                 stream_factory=None, verbose: bool = True):
        self.voice_path, self.rate, self.device = voice_path, rate, device
        self._synth_fn = synth_fn
        self._stream_factory = stream_factory
        self.verbose = verbose

        self._cv = threading.Condition()          # guards the queue and playback state
        self._items: list[Utterance] = []
        self._now: Optional[Utterance] = None
        self._seq = itertools.count()
        self._key_times: dict = {}

        self._mix = threading.Lock()              # guards what the audio callback reads
        self._speech: Optional[np.ndarray] = None
        self._pos = 0
        self._speech_done = threading.Event()
        self._tones: list[list] = []              # [buffer, position]

        self._voice = None
        self._stream = None
        self._running = False

    # ---------------- lifecycle ----------------
    def start(self) -> "AudioManager":
        if self._synth_fn is None:
            self._load_piper()
        self._open_stream()
        self._running = True
        threading.Thread(target=self._synth_loop, daemon=True, name="tts-synth").start()
        threading.Thread(target=self._play_loop, daemon=True, name="tts-play").start()
        return self

    def close(self):
        self._running = False
        with self._cv:
            self._cv.notify_all()
        self._speech_done.set()
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

    def _load_piper(self):
        """Load the voice once (fast). Falls back to the slower CLI if the API differs."""
        try:
            from piper import PiperVoice
            self._voice = PiperVoice.load(self.voice_path)
            sr = getattr(getattr(self._voice, "config", None), "sample_rate", None)
            if sr and sr != self.rate:
                print(f"[audio] voice runs at {sr} Hz; using that")
                self.rate = int(sr)
            self._log(f"Piper loaded in-process ({self.rate} Hz)")
        except Exception as e:
            self._voice = None
            print(f"[audio] Piper Python API unavailable ({e}); using slower command-line mode")

    def _open_stream(self):
        if self._stream_factory:
            self._stream = self._stream_factory(self.rate, self._callback)
        else:
            import sounddevice as sd
            self._stream = sd.OutputStream(
                samplerate=self.rate, channels=1, dtype="float32",
                device=self.device, callback=self._callback)
        self._stream.start()

    def _log(self, msg):
        if self.verbose:
            print(f"[audio] {msg}")

    # ---------------- public API ----------------
    def say(self, text: str, priority: int = REPLY, group: Optional[str] = None,
            replace: bool = False, key=None, cooldown: float = 0.0) -> bool:
        """
        Queue speech; returns immediately.
          group+replace : drop still-waiting messages of the same group (only the newest matters)
          key+cooldown  : skip if the same key was spoken less than `cooldown` seconds ago
        A message more important than the one playing interrupts it.
        """
        text = (text or "").strip()
        if not text:
            return False
        now = time.time()
        with self._cv:
            if key is not None and cooldown > 0:
                if now - self._key_times.get(key, 0.0) < cooldown:
                    return False
                self._key_times[key] = now
            if replace and group is not None:
                self._items = [u for u in self._items if u.group != group]
            self._items.append(Utterance(next(self._seq), priority, text, group, now))
            if self._now is not None and priority < self._now.priority:
                self._interrupt_locked()
            self._cv.notify_all()
        return True

    def hazard(self, text: str, key=None, cooldown: float = 4.0) -> bool:
        """Convenience for safety alerts: top priority, never repeated within `cooldown`."""
        return self.say(text, HAZARD, group="hazard", replace=True,
                        key=key if key is not None else text, cooldown=cooldown)

    def beep(self, freq: float = 880, ms: float = 110, volume: float = 0.25):
        """Immediate tone, mixed on top of whatever is playing (not queued)."""
        n = int(self.rate * ms / 1000)
        buf = (volume * np.sin(2 * np.pi * freq * np.arange(n) / self.rate)).astype(np.float32)
        fade = min(n // 4, int(0.01 * self.rate))           # 10 ms edges: no clicks
        if fade > 0:
            buf[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            buf[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        with self._mix:
            self._tones.append([buf, 0])

    def stop(self, keep_hazards: bool = True):
        """Silence everything now (the button press). Hazard alerts survive by default."""
        with self._cv:
            keep = [u for u in self._items if keep_hazards and u.priority == HAZARD]
            for u in self._items:
                if u not in keep:
                    u.cancelled = True
            self._items = keep
            now = self._now
            if now is not None and not (keep_hazards and now.priority == HAZARD):
                now.cancelled = True
                self._interrupt_locked()
            self._cv.notify_all()

    def wait_idle(self, timeout: Optional[float] = None,
                  abort: Optional[Callable[[], bool]] = None) -> bool:
        """Block until all queued speech has finished. False on timeout or abort()."""
        deadline = None if timeout is None else time.time() + timeout
        with self._cv:
            while self._items or self._now is not None:
                if abort and abort():
                    return False
                step = 0.05
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return False
                    step = min(step, remaining)
                self._cv.wait(timeout=step)
        return True

    @property
    def busy(self) -> bool:
        with self._cv:
            return bool(self._items) or self._now is not None

    # ---------------- internals ----------------
    def _interrupt_locked(self):
        """Stop the utterance currently playing. Caller holds self._cv."""
        item = self._now
        with self._mix:
            if self._speech is None or item is None:
                return
            item.pos = self._pos
            self._speech = None
            self._speech_done.set()
        item.interrupted = True

    def _drop_stale_locked(self):
        now = time.time()
        kept = []
        for u in self._items:
            age = MAX_AGE.get(u.priority)
            if age is not None and now - u.created > age:
                self._log(f"dropped stale: {u.text!r}")
            else:
                kept.append(u)
        self._items = kept

    def _synth_loop(self):
        while self._running:
            with self._cv:
                while self._running:
                    waiting = [u for u in self._items if u.audio is None and not u.synthesizing]
                    if waiting:
                        item = min(waiting, key=lambda u: (u.priority, u.seq))
                        item.synthesizing = True
                        break
                    self._cv.wait(timeout=0.2)
                else:
                    return
            try:
                audio = self._synthesize(item.text)
            except Exception as e:
                print(f"[audio] synthesis failed for {item.text!r}: {e}")
                audio = np.zeros(0, dtype=np.float32)
            with self._cv:
                item.audio = audio
                item.synthesizing = False
                self._cv.notify_all()

    def _play_loop(self):
        while self._running:
            with self._cv:
                while self._running:
                    self._drop_stale_locked()
                    top = min(self._items, key=lambda u: (u.priority, u.seq), default=None)
                    if top is not None and top.audio is not None:
                        break
                    self._cv.wait(timeout=0.1)
                else:
                    return
                self._items.remove(top)
                self._now = top
                if len(top.audio) == 0:
                    self._now = None
                    self._cv.notify_all()
                    continue
                with self._mix:                       # start playback while still holding the lock
                    self._speech, self._pos = top.audio, 0
                    self._speech_done.clear()

            self._speech_done.wait()                  # finished, or interrupted

            with self._cv:
                item, self._now = self._now, None
                if (item is not None and item.interrupted and not item.cancelled
                        and item.priority == REPLY and item.audio is not None):
                    back = int(RESUME_REWIND_S * self.rate)       # resume where it was cut off
                    item.audio = item.audio[max(0, item.pos - back):]
                    item.interrupted, item.pos, item.created = False, 0, time.time()
                    self._items.append(item)
                self._cv.notify_all()

    def _callback(self, outdata, frames, time_info, status):
        """Runs on the audio thread: keep it short and never block."""
        out = np.zeros(frames, dtype=np.float32)
        with self._mix:
            if self._speech is not None:
                chunk = self._speech[self._pos:self._pos + frames]
                out[:len(chunk)] += chunk
                self._pos += len(chunk)
                if self._pos >= len(self._speech):
                    self._speech = None
                    self._speech_done.set()
            if self._tones:
                alive = []
                for tone in self._tones:
                    buf, pos = tone
                    chunk = buf[pos:pos + frames]
                    out[:len(chunk)] += chunk
                    tone[1] = pos + len(chunk)
                    if tone[1] < len(buf):
                        alive.append(tone)
                self._tones = alive
        np.clip(out, -1.0, 1.0, out=out)
        outdata[:, 0] = out

    # ---------------- text to speech ----------------
    def _synthesize(self, text: str) -> np.ndarray:
        if self._synth_fn is not None:
            return self._synth_fn(text)
        if self._voice is not None:
            try:
                raw = b"".join(c.audio_int16_bytes for c in self._voice.synthesize(text))
                if raw:
                    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception as e:
                print(f"[audio] Piper API call failed ({e}); switching to command-line mode")
                self._voice = None
        result = subprocess.run(
            [sys.executable, "-m", "piper", "--model", self.voice_path, "--output-raw"],
            input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0 or not result.stdout:
            raise RuntimeError(result.stderr.decode(errors="ignore")[-200:])
        return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
