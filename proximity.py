"""
Ultrasonic collision warning (HC-SR04) for the glasses.

Runs in its own thread from the moment the program starts. It never touches the camera, Gemini
or the internet, so it keeps working when those are slow or down.

  every ~70 ms: ping -> distance in cm
  object closer than WARN_CM (30 cm):
      - immediately: parking-sensor beeps, mixed on top of whatever is playing
        (faster and higher-pitched the closer the object gets)
      - spoken once per episode, as a top-priority hazard: "Obstacle ahead"
  object gone (beyond WARN_CM + 5 cm for a moment): beeps stop
  sensor fault (no echo at all: loose wire, dead module): spoken warning, repeated every 2 minutes,
      because silence would otherwise be mistaken for "all clear"

Wiring (HC-SR04 is a 5 V part; the Pi's GPIO is 3.3 V only, so ECHO needs a voltage divider):
  VCC  -> Pin 2  (5 V)
  GND  -> Pin 6  (GND)
  TRIG -> Pin 16 (GPIO 23)
  ECHO -> 1 kOhm -> Pin 18 (GPIO 24) -> 2 kOhm -> GND        (see README for the diagram)

Run:
  python proximity.py --check    live distance readout (wiring test, no audio)
  python proximity.py --demo     simulated approaching object, plays the real beeps (no sensor needed)

Environment (all optional):
  SG_SENSOR=0     disable the sensor entirely (e.g. on a PC)
  SG_WARN_CM=30   warning distance in cm
  SG_TRIG_GPIO / SG_ECHO_GPIO   BCM GPIO numbers (default 23 / 24)
"""
import os
import statistics
import sys
import threading
import time
from collections import deque
from typing import Optional

from audio_out import HAZARD, INFO

# ---------------- Config ----------------
TRIG_GPIO = int(os.environ.get("SG_TRIG_GPIO", 23))     # BCM numbering (Pin 16)
ECHO_GPIO = int(os.environ.get("SG_ECHO_GPIO", 24))     # BCM numbering (Pin 18)

WARN_CM = float(os.environ.get("SG_WARN_CM", 30))       # alert starts below this
CLEAR_MARGIN_CM = 5.0                                   # ...and ends only above WARN_CM + this (no flapping)
CLOSE_CM = 8.0                                          # at/below this the beeps are fastest
MAX_CM = 400.0                                          # HC-SR04 limit; farther = "nothing there"

PERIOD_S = 0.07                                         # HC-SR04 needs >= 60 ms between pings
RELEASE_S = 0.40                                        # keep warning this long after the last near reading

BEEP_SLOW_S, BEEP_FAST_S = 0.50, 0.10                   # gap between beeps at WARN_CM / at CLOSE_CM
BEEP_HZ_FAR, BEEP_HZ_NEAR = 1500, 2400                  # pitch at WARN_CM / at CLOSE_CM (UI beeps are 660/990)
BEEP_MS = 60
BEEP_VOLUME = 0.5

SPEECH = "Obstacle ahead"
SPEECH_COOLDOWN_S = 8.0                                 # don't repeat the words while standing at a wall

FAULT_LIMIT = 20                                        # consecutive failed pings (~1.4 s) = sensor fault
FAULT_REPEAT_S = 120.0

FAR = float("inf")                                      # "valid ping, nothing within range"


# ---------------- Sensor driver ----------------
class HCSR04:
    """
    HC-SR04 on gpiozero (uses lgpio on the Pi 5, RPi.GPIO does not work there).
    Echo timing comes from kernel edge timestamps, not from Python polling, so CPU load from
    Piper/OpenCV does not distort the distance.
    """
    SPEED_OF_SOUND = 343.0          # m/s at ~20 C (error ~0.17 % per degree: irrelevant at 30 cm)

    def __init__(self, trig: int = TRIG_GPIO, echo: int = ECHO_GPIO, timeout_s: float = 0.06):
        from gpiozero import DigitalInputDevice, DigitalOutputDevice
        self.timeout_s = timeout_s
        self._trig = DigitalOutputDevice(trig, initial_value=False)
        self._echo = DigitalInputDevice(echo, pull_up=False, bounce_time=None)   # pulled DOWN: a loose
        self._rise = self._fall = None                                            # echo wire = no echo
        self._done = threading.Event()
        self._echo.pin.when_changed = self._on_edge

    def _on_edge(self, ticks, state):
        if state:
            self._rise = ticks
        else:
            self._fall = ticks
            self._done.set()

    def measure(self) -> Optional[float]:
        """cm, FAR (valid ping, nothing in range) or None (no usable echo: wiring/sensor fault)."""
        self._rise = self._fall = None
        self._done.clear()
        self._trig.on()
        time.sleep(0.00002)                      # >= 10 us trigger pulse
        self._trig.off()
        if not self._done.wait(self.timeout_s):
            return None                          # the echo pin never went high and low
        rise, fall = self._rise, self._fall
        if rise is None or fall is None:
            return None                          # caught only one edge
        seconds = self._echo.pin_factory.ticks_diff(fall, rise)
        cm = seconds * self.SPEED_OF_SOUND * 100.0 / 2.0
        return FAR if cm > MAX_CM else max(cm, 0.0)

    def close(self):
        for dev in (self._trig, self._echo):
            try:
                dev.close()
            except Exception:
                pass


# ---------------- Alert logic ----------------
class ProximityMonitor:
    """
    Turns raw readings into beeps and speech. step() holds all the logic and takes the time as a
    parameter, so it is tested without hardware (tests/test_proximity.py).
    """

    def __init__(self, audio, sensor=None, warn_cm: float = WARN_CM, close_cm: float = CLOSE_CM,
                 clear_margin_cm: float = CLEAR_MARGIN_CM, period_s: float = PERIOD_S,
                 release_s: float = RELEASE_S, fault_limit: int = FAULT_LIMIT,
                 fault_repeat_s: float = FAULT_REPEAT_S, verbose: bool = True):
        self.audio, self.sensor = audio, sensor
        self.warn_cm = warn_cm
        self.clear_cm = warn_cm + clear_margin_cm
        self.close_cm = min(close_cm, warn_cm - 1)
        self.period_s, self.release_s = period_s, release_s
        self.fault_limit, self.fault_repeat_s = fault_limit, fault_repeat_s
        self.verbose = verbose

        self._recent: deque = deque(maxlen=3)
        self.alerting = False
        self.faulty = False
        self.distance: Optional[float] = None      # latest nearest-object estimate in cm (None = unknown)
        self._last_d = warn_cm
        self._last_near_t = 0.0
        self._next_beep = 0.0
        self._faults = 0
        self._fault_said = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- the logic
    def step(self, reading: Optional[float], now: Optional[float] = None) -> None:
        now = time.time() if now is None else now

        if reading is None:                                    # ping failed
            self._faults += 1
            if self._faults >= self.fault_limit:
                if not self.faulty or now - self._fault_said >= self.fault_repeat_s:
                    self.faulty, self._fault_said = True, now
                    self.alerting = False
                    self._recent.clear()
                    self.distance = None
                    if self.verbose:
                        print("[proximity] SENSOR FAULT: no echo")
                    self.audio.hazard("Obstacle sensor is not responding.", key="sensor-fault",
                                      cooldown=self.fault_repeat_s)
            return

        self._faults = 0
        if self.faulty:                                        # a valid ping after a fault
            self.faulty = False
            self._recent.clear()
            if self.verbose:
                print("[proximity] sensor recovered")
            self.audio.say("Obstacle sensor is working again.", INFO)

        self._recent.append(reading)
        near = [r for r in self._recent if r < self.warn_cm]
        holding = [r for r in self._recent if r < self.clear_cm]
        self.distance = reading if reading != FAR else None

        if not self.alerting:
            if len(near) >= 2:                                 # 2 of the last 3: ignores one stray echo
                self.alerting = True
                self._last_near_t = now
                self._next_beep = now                          # first beep right away
                self._last_d = statistics.median(near)
                if self.verbose:
                    print(f"[proximity] OBSTACLE at ~{self._last_d:.0f} cm")
                self.audio.hazard(SPEECH, key="proximity", cooldown=SPEECH_COOLDOWN_S)
            else:
                return
        else:
            if holding:                                        # still something close: keep warning
                self._last_near_t = now
                self._last_d = statistics.median(holding)
            elif now - self._last_near_t > self.release_s:     # it's gone
                self.alerting = False
                if self.verbose:
                    print("[proximity] clear")
                return

        if now >= self._next_beep:
            self.audio.beep(self.beep_freq(self._last_d), BEEP_MS, BEEP_VOLUME)
            self._next_beep = now + self.beep_interval(self._last_d)

    def _closeness(self, d: float) -> float:
        """0.0 at/below close_cm ... 1.0 at/above warn_cm."""
        span = max(self.warn_cm - self.close_cm, 1.0)
        return min(max((d - self.close_cm) / span, 0.0), 1.0)

    def beep_interval(self, d: float) -> float:
        return BEEP_FAST_S + self._closeness(d) * (BEEP_SLOW_S - BEEP_FAST_S)

    def beep_freq(self, d: float) -> float:
        return BEEP_HZ_NEAR + self._closeness(d) * (BEEP_HZ_FAR - BEEP_HZ_NEAR)

    # ---- the thread
    def start(self) -> "ProximityMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True, name="proximity")
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        close = getattr(self.sensor, "close", None)
        if close:
            close()

    def _run(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                reading = self.sensor.measure()
            except Exception as e:
                print(f"[proximity] read error: {type(e).__name__}: {e}")
                reading = None
            try:
                self.step(reading, t0)
            except Exception as e:                             # never let the safety thread die
                print(f"[proximity] logic error: {type(e).__name__}: {e}")
            rest = self.period_s - (time.time() - t0)
            if rest > 0:
                self._stop.wait(rest)


def _on_raspberry_pi() -> bool:
    try:
        with open("/proc/device-tree/model", "rb") as f:
            return b"Raspberry Pi" in f.read()
    except OSError:
        return False


def start_proximity(audio) -> Optional[ProximityMonitor]:
    """Start the collision warning. Returns None (and says so) if there is no sensor to run."""
    if os.environ.get("SG_SENSOR", "1") == "0":
        print("[proximity] disabled (SG_SENSOR=0)")
        return None
    if not _on_raspberry_pi():
        print("[proximity] not a Raspberry Pi: obstacle sensor disabled")
        return None
    try:
        sensor = HCSR04()
    except Exception as e:
        print(f"[proximity] cannot open GPIO: {type(e).__name__}: {e}\n"
              "            (Pi 5 needs: sudo apt install python3-lgpio python3-gpiozero, "
              "and your user in the 'gpio' group)")
        audio.hazard("Obstacle sensor unavailable.", key="sensor-unavailable")
        return None
    print(f"[proximity] HC-SR04 on GPIO{TRIG_GPIO} (trig) / GPIO{ECHO_GPIO} (echo), "
          f"warning below {WARN_CM:.0f} cm")
    return ProximityMonitor(audio, sensor).start()


# ---------------- standalone tools ----------------
def _check():
    sensor = HCSR04()
    print(f"HC-SR04 live readout (trig GPIO{TRIG_GPIO}, echo GPIO{ECHO_GPIO}). Ctrl+C to stop.")
    print("Wave your hand in front of it: the number should follow. Alert zone is "
          f"below {WARN_CM:.0f} cm.\n")
    try:
        while True:
            r = sensor.measure()
            if r is None:
                print("no echo: check wiring (5 V, GND, trigger, echo divider)      ", end="\r")
            elif r == FAR:
                print("nothing in range (> 4 m)                                     ", end="\r")
            else:
                flag = "  <-- WARN" if r < WARN_CM else ""
                print(f"{r:6.1f} cm{flag}                                         ", end="\r")
            time.sleep(PERIOD_S)
    except KeyboardInterrupt:
        print("\nBye.")
    finally:
        sensor.close()


class _ScriptedSensor:
    """Walks toward an obstacle, stops, then walks away. For --demo."""
    def __init__(self):
        self.i = 0

    def measure(self):
        t = self.i * PERIOD_S
        self.i += 1
        if t < 1.5:
            return 200.0
        if t < 5.5:                                   # approach 200 -> 6 cm
            return 200.0 - (t - 1.5) / 4.0 * 194.0
        if t < 7.5:
            return 6.0
        if t < 10.0:                                  # back away
            return 6.0 + (t - 7.5) / 2.5 * 120.0
        return FAR


def _demo():
    from audio_out import AudioManager
    print("Starting audio...")
    audio = AudioManager().start()
    print("Simulated walk toward a wall, then away. You should hear beeps start at 30 cm, speed up "
          "and rise in pitch, then stop.\n")
    mon = ProximityMonitor(audio, _ScriptedSensor()).start()
    try:
        time.sleep(12)
    except KeyboardInterrupt:
        pass
    mon.stop()
    audio.wait_idle(timeout=5)
    audio.close()


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        _check()
