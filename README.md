# Echosight

A camera and a voice assistant that describe surroundings, read text, and guide the user to a
named object. The assistant runs on a PC for development and moves to a Raspberry Pi 5 for the
glasses.

> **Safety:** this is an aid, not a replacement for a cane or guide dog. It can be wrong. It must
> never be the only thing a person relies on to cross a road or avoid a drop. Test it with blind or
> low-vision users and an orientation-and-mobility instructor before real-world use.

## What it does

```
 button ──► beep ──► record ──► SpeechRecognition (STT) ──► route the request
                                                           │
   "find my keys"        ───► guided search: Gemini finds it once, a local tracker steers
   "what time is it"     ───► answered locally, instantly (no internet)
   non-visual question   ───► Gemini, text only
   anything else         ───► best camera frames + Gemini ("what's in front of me", "read this")

 Gemini streams its answer ──► sentence by sentence ──► AudioManager (Piper TTS) ──► earphones
```

`AudioManager` is a priority queue: hazard alerts interrupt everything, navigation guidance
interrupts replies, replies resume after a hazard, and the button silences everything except
hazards. Beeps are mixed on top of speech.

## Status

| Part | State |
|---|---|
| Voice loop (push-to-talk, STT via SpeechRecognition, offline TTS) | Built, tested on the PC |
| Camera ring buffer (sharpest-frame picking, low-light check) | Built, tested on the PC with a Logitech C270 |
| Gemini client (streaming, minimal thinking, model fallback, structured JSON) | Built, tested on the PC |
| Reading mode, scene description, follow-up questions, local time/date | Built, tested on the PC |
| Find-my-object with a local tracker | Built; logic verified in simulation; tried by the author on the PC |
| Priority audio manager | Built; logic verified in simulation; run `audio_test.py` to hear it |
| **Raspberry Pi setup** | **Not yet tested.** Instructions below are the intended procedure |
| Bluetooth earphone play/pause as the push-to-talk button | Built (`buttons.py`); logic tested with a fake input device, **not yet with real earphones** |
| Ultrasonic collision warning, HC-SR04, 30 cm (`proximity.py`) | Built; alert logic and distance maths tested in simulation, **not yet with the real sensor** |
| Autostart on boot (systemd user service) | Built (`install_service.sh`); not yet run on a Pi |
| Low-light fallback using the ultrasonic sensor | Not built |
| On-device hazard detection (offline) | Not built |
| Offline fallbacks, battery announcements | Not built (sensor-fault announcements are built) |

## Files

| File | Purpose |
|---|---|
| `assistant.py` | **Main program.** Runs everything |
| `audio_out.py` | Speaker manager: priorities, interruption, resume, beep mixing |
| `buttons.py` | Push-to-talk sources: Bluetooth earphone play/pause (evdev) + Enter key. `python buttons.py` shows what your earphones send |
| `proximity.py` | HC-SR04 driver + collision warning (beeps + "Obstacle ahead"). `--check` live readout, `--demo` hear the beeps |
| `smartglasses.service`, `install_service.sh` | Start-on-boot service and its installer |
| `voice_loop.py` | Mic recording, voice-activity detection, SpeechRecognition STT, button class. Also runs standalone as an echo test |
| `camera.py` | Camera thread and frame buffers (Pi Camera via `picamera2`, or USB webcam via OpenCV) |
| `gemini_client.py` | Gemini API: system prompt, streaming, model fallback, structured output |
| `find_object.py` | Guided object search with a local tracker |
| `audio_test.py` | Listening test for the audio priorities |
| `mic_check.py` | Live microphone level meter, to tune `MIN_RMS` / `REL_LEVEL` |
| `tests/` | Simulation tests that need no hardware or internet |
| `setup.sh`, `requirements.txt` | Installation |
| `CHAT_SUMMARY.md` | Condensed history of the design decisions and measured results |

---

## 1. Setup on a PC (Linux, macOS, Windows)

You need Python 3.10 or newer, a microphone, speakers or headphones, and a webcam.

```bash
cd smartglasses
python -m venv .venv
source .venv/bin/activate                 # fish: source .venv/bin/activate.fish
                                          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m piper.download_voices en_US-lessac-medium --data-dir voices
```

- On Linux you also need PortAudio: `sudo apt install libportaudio2` (Debian/Ubuntu/Pi OS).
- Run every script **from the project folder**: the voice file is found at `voices/...`.

### Gemini API key

Get a key at https://aistudio.google.com/apikey and keep it out of the code, chat, and git.

```bash
# safest: put the key in a file with an editor, then load it (not typed into the terminal)
chmod 600 ~/.gemini_key
export GEMINI_API_KEY=$(cat ~/.gemini_key)       # bash / zsh
set -x GEMINI_API_KEY (cat ~/.gemini_key)        # fish
# Windows PowerShell:  $env:GEMINI_API_KEY="your-key"
```

If a key is ever exposed, revoke it in AI Studio and create a new one.

Model names change often. List what your key can use, and optionally set a preference chain
(first = preferred; the rest are automatic fallbacks if one is overloaded):

```bash
python gemini_client.py --models
export GEMINI_MODEL="gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.5-flash"
```

### Run order (each step isolates one part)

```bash
python camera.py 2>/dev/null          # 1. camera works? prints FPS, saves test_frame_0..2.jpg
python gemini_client.py 2>/dev/null   # 2. camera -> Gemini -> text (prints time to first sentence)
python voice_loop.py 2>/dev/null      # 3. mic -> SpeechRecognition -> Piper echo ("You said: ...")
python audio_test.py                  # 4. hear hazard/interrupt/resume/beep behaviour
python assistant.py 2>/dev/null       # 5. the whole thing
```

Drop `2>/dev/null` (PowerShell: `2>$null`) when something fails, so you can see the error. That
redirect only hides harmless OpenCV "Corrupt JPEG data" warnings.

**Using the assistant:** press **Enter** (the stand-in for the button), wait for the low beep, speak,
press **Enter** again. Pressing Enter while it is talking silences it and starts listening.

Things to say: "What's in front of me?", "Read this" (hold text at arm's length), "What colour is
it?" (follow-up), "Find my keys", "Where is my phone?", "What time is it?", "Stop".

---

## 2. Setup on the Raspberry Pi 5

> Untested as a whole. The code is cross-platform, but expect to tune devices and speed.

**Hardware:** Pi 5 (8 GB recommended), active cooler, official 27 W supply while developing
(a 5 V/5 A power bank for the final build), a decent A2 microSD card (32 GB+), USB webcam or Pi
Camera, a microphone, earphones or bone-conduction headphones.

1. **Flash Raspberry Pi OS (64-bit)** with Raspberry Pi Imager. Use Desktop while developing, Lite
   for the final build. Before writing the card, open the customisation settings and set the
   hostname, user and password, Wi-Fi, locale, and **enable SSH**.
2. **First boot:**
   ```bash
   sudo apt update && sudo apt full-upgrade -y && sudo reboot
   ```
3. **Copy the project over** (`scp -r smartglasses user@glasses.local:~`) or unzip it, then:
   ```bash
   cd ~/smartglasses
   bash setup.sh                 # USB webcam
   bash setup.sh --picamera      # Pi Camera Module (adds picamera2)
   ```
4. **Check the devices:**
   ```bash
   arecord -l && aplay -l                                       # mic and speaker
   python -c "import sounddevice as sd; print(sd.query_devices())"
   lsusb && v4l2-ctl --list-devices                             # USB webcam
   rpicam-hello --list-cameras                                  # Pi Camera only
   ```
   A USB webcam is **not** seen by `rpicam-hello`; that is normal. The code scans for it.
5. **Select audio devices** if the defaults are wrong: set `MIC_DEVICE` in `voice_loop.py` and
   `SPEAKER_DEVICE` in `audio_out.py` (and in `voice_loop.py` if you use its standalone test) to an
   index or name from the list above. Bluetooth earphones need to be the system default output
   (PipeWire) to work with the default device. If `speech_test.py` recognizes speech but the
   assistant does not, set `MIC_DEVICE` to the same input index used by `speech_test.py`
   (`device_index=2` means `MIC_DEVICE = 2`). The system default may be a different device,
   such as a virtual audio cable.
6. **Set the API key** as above, then follow the run order in section 1.
7. **Measure and tune speed:**
   ```bash
   python find_object.py --bench        # tracker speed on this hardware
   ```
   For fully offline speech-to-text, `pip install pocketsphinx` and set `STT_ENGINE = "sphinx"` in
   `voice_loop.py` (less accurate than the default Google web recognizer).
8. **Earphone button, ultrasonic sensor, autostart:** sections 2a, 2b and 2c below. Over SSH, the
   Enter key also works as the button.

If a `pip` package has no wheel for your Python or ARM version, install what you can from `apt`
(e.g. `python3-opencv`, `python3-numpy`) and use `python -m venv --system-site-packages .venv`.

### 2a. Earphone play/pause as the button

Pair the earphones once, and mark them **trusted** so they reconnect by themselves after a reboot:

```bash
bluetoothctl
  scan on                      # put the earphones in pairing mode; note their MAC address
  pair AA:BB:CC:DD:EE:FF
  trust AA:BB:CC:DD:EE:FF      # without this they will not auto-connect at boot
  connect AA:BB:CC:DD:EE:FF
  exit
```

Then check that the Pi sees the button:

```bash
python buttons.py              # press play/pause on the earphones
```

You should see `raw key code=164 value=1` and `*** BUTTON PRESS ***`. Earphones differ: some send
play/pause (164), some separate play (200) / pause (201); all count. If nothing appears:
the earphones are not connected, or your user is not in the `input` group
(`sudo usermod -aG input $USER`, then log out and in), or they send a code we don't list (tell me
what `python buttons.py` prints and it is a one-line change). `SG_BUTTON_DEVICE=<part of name>`
restricts which input device is used. The assistant says "Earphone button connected" when it
finds them and "disconnected" if they drop.

Use: press once, wait for the low beep, speak, press again. Pressing while it is talking silences
it and starts listening. Note: the earphone *microphone* only works in Bluetooth headset mode
(HFP), which has lower audio quality than music mode; if speech recognition sounds poor, test with
a different `MIC_DEVICE`.

### 2b. Ultrasonic collision warning (HC-SR04)

**Safety of the Pi:** the HC-SR04 runs on 5 V and its ECHO pin outputs 5 V. The Pi's GPIO is 3.3 V
and is **not 5 V tolerant**: connect ECHO directly and you can damage the pin or the Pi. Use a
voltage divider (below), or buy a 3.3 V-compatible module (HC-SR04P or "3-5.5 V" versions) which
needs no divider. Power the Pi **off** while wiring.

| HC-SR04 pin | Connects to | Pi header pin |
|---|---|---|
| VCC | 5 V | **Pin 2** |
| TRIG | GPIO 23 | **Pin 16** |
| ECHO | through the divider, to GPIO 24 | **Pin 18** |
| GND | Ground | **Pin 6** |

Voltage divider on ECHO: `ECHO -- 1 kOhm -- (junction) -- 2 kOhm -- GND`, and the **junction goes
to Pin 18**. That gives 5 V x 2/3 = 3.3 V. Two 1 kOhm resistors in series make the 2 kOhm. (330 Ohm
and 470 Ohm also work.) The divider's GND and the sensor's GND must be the same ground (Pin 6, or any
other GND pin). Different pins? Set `SG_TRIG_GPIO` / `SG_ECHO_GPIO` (BCM numbers) in the env file.

Test the wiring, then hear what the alert sounds like:

```bash
python proximity.py --check     # live cm readout; wave a hand in front, 'WARN' appears under 30 cm
python proximity.py --demo      # simulated walk toward a wall: beeps from 30 cm, faster and higher as you close in
```

What the user hears: closer than 30 cm -> parking-sensor beeps (every 0.5 s at 30 cm, every 0.1 s at
8 cm, rising in pitch) plus a spoken "Obstacle ahead" once per episode, which interrupts anything
else being said and is never silenced by the button. The warning clears when the object is more than
35 cm away. If the sensor stops answering (loose wire), it says "Obstacle sensor is not responding"
and repeats every 2 minutes, because silence would otherwise sound like "all clear".

**Limits you must know about.** At a normal walking speed (about 1.2 m/s) 30 cm is only about a
quarter of a second of warning, so treat it as a last-moment alarm; raising `SG_WARN_CM` to around
100 gives usable time to react. The beam is a narrow cone (about 15 degrees) straight ahead: it
misses things to the side, thin poles, and overhanging objects above it, and soft or steeply angled
surfaces (clothing, some walls) reflect poorly. Stairs going down and kerbs are not detected. It
complements a cane or guide dog; it does not replace them.

### 2c. Start automatically when the Pi boots

```bash
bash install_service.sh        # once; then edit ~/.config/smartglasses.env and add GEMINI_API_KEY=...
sudo reboot
```

This installs a systemd **user** service with "linger" enabled (so it starts at boot with nobody
logged in). It is a user service on purpose: Bluetooth earphone audio belongs to the user's PipeWire
session, which a system-wide service cannot reach. It restarts itself 3 s after any crash.

```bash
systemctl --user status smartglasses      # running?
journalctl --user -u smartglasses -f      # live log
systemctl --user restart smartglasses     # after changing code or the env file
systemctl --user stop smartglasses        # before running  python assistant.py  by hand
bash install_service.sh --uninstall
```

Test the real boot sequence once with the earphones *off* at power-on and switched on afterwards:
the sensor should work immediately, and "Earphone button connected" should be spoken once they pair.

---

## 3. Configuration reference

| Setting | File | Meaning |
|---|---|---|
| `PUSH_TO_TALK` | `assistant.py` | `True`: press to start, press again to finish (reliable in noise). `False`: auto-stop when you go quiet |
| `STT_ENGINE` | `voice_loop.py` | `google` (default; free web API, needs internet, audio is uploaded) or `sphinx` (offline PocketSphinx, less accurate) |
| `STT_LANGUAGE` | `voice_loop.py` | Recognition language, e.g. `en-US`, `en-GB`, `en-IN` |
| `MIC_DEVICE` / `SPEAKER_DEVICE` | `voice_loop.py`, `audio_out.py` | Device index or name; `None` = system default. Match `MIC_DEVICE` to the input used by `speech_test.py` |
| `MIN_RMS`, `REL_LEVEL`, `VAD_AGGRESSIVENESS` | `voice_loop.py` | Voice detection in auto-stop mode. Tune with `mic_check.py` |
| `WIDTH, HEIGHT`, `FPS`, `LOW_LIGHT_THRESHOLD` | `camera.py` | Capture size, rate, "too dark" cut-off |
| `GEMINI_MODEL` | environment | Comma-separated model chain |
| `SYSTEM_PROMPT` | `gemini_client.py` | How Gemini is told to speak (short, hazards first, never "safe to cross") |
| `SG_WARN_CM`, `SG_TRIG_GPIO`, `SG_ECHO_GPIO`, `SG_SENSOR=0` | environment / `~/.config/smartglasses.env` | Obstacle warning distance (default 30), sensor pins (default 23/24), or turn the sensor off (e.g. on a PC) |
| `SG_BUTTON_DEVICE` | environment | Only listen to the input device whose name contains this text |
| `BEEP_*`, `SPEECH`, `RELEASE_S`, `FAULT_*` | `proximity.py` | Beep pitch/speed/volume, spoken text, how long a warning lingers, sensor-fault timing |
| `HFOV_DEG` | `find_object.py` | Camera horizontal field of view. Sets how "left/right" is computed (52 for a C270) |
| `VERIFY_S`, `ARRIVED_M`, `ARRIVED_AREA` | `find_object.py` | Gemini re-check interval and "within reach" thresholds |
| `MAX_AGE`, `RESUME_REWIND_S` | `audio_out.py` | How long a queued message stays valid; how far a resumed reply backs up |

## 4. Tests that need no hardware

```bash
python tests/test_audio_logic.py     # ~25 s: priorities, interrupt, resume, replace, dedupe, mixing
python tests/test_find.py            # ~75 s: tracker catch-up + finder scenarios with a fake camera/Gemini
python tests/test_proximity.py       # alert logic, hysteresis, fault handling, HC-SR04 timing maths
python tests/test_buttons.py         # earphone button: debounce, reconnect, ignores keyboards
```

Use these to catch regressions after changes, and on the Pi to check the tracker runs there.

## 5. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `rpicam-hello`: "no cameras available" with a USB webcam | Expected. It only sees ribbon-cable cameras. Use `lsusb` / `v4l2-ctl --list-devices` |
| `picamera2 unavailable ... trying USB webcam` | Normal on a PC or with a USB camera |
| Lots of `Corrupt JPEG data` lines | Harmless webcam/USB noise (common with the C270). Hide with `2>/dev/null` |
| Camera FPS far below 10 | The C270 lowers its frame rate in dim light. Try `v4l2-ctl -d /dev/videoN --set-ctrl=exposure_dynamic_framerate=0`, a blue USB 3 port, or 640x480 |
| Gemini `503 ... high demand` | That model is overloaded. The fallback chain handles it; set `GEMINI_MODEL` to a list |
| Gemini answers take 10-25 s | Silent "thinking". The client already minimises it; run `pip install -U google-genai` and check `[gemini]` messages |
| Model name "not found" | Run `python gemini_client.py --models` and use a name from the list |
| `[stt] google request failed` / always "Sorry, I didn't catch that" | No internet, or the web recognizer is rate-limited. Check connectivity, or switch to `STT_ENGINE = "sphinx"` |
| Wrong words in noisy places | Use push-to-talk (`PUSH_TO_TALK = True`) and keep the mic close to the mouth |
| Recording never stops in a loud room | Auto-stop can't work without quiet. Use `PUSH_TO_TALK = True` |
| Piper error / voice not found | Run scripts from the project folder; re-run the `piper.download_voices` command |
| No sound, or wrong device | Set `SPEAKER_DEVICE` (`audio_out.py`) to an index from `sd.query_devices()` |
| "no OpenCV tracker available" | Your OpenCV build lacks CSRT. Guidance still works, from Gemini only (less smooth) |
| Earphone button does nothing | `python buttons.py` to see events; check `trust` + connected in `bluetoothctl`, `input` group (section 2a) |
| Sensor: `no echo: check wiring` | Check 5 V on Pin 2, GND, TRIG on Pin 16, ECHO divider junction on Pin 18; module needs 5 V, not 3.3 V |
| Sensor: readings jump or are always ~0 | Loose jumper wires, or no common ground between the divider and the Pi |
| Sensor: `cannot open GPIO` | `sudo apt install python3-lgpio python3-gpiozero`; user in `gpio` group; venv made with `--system-site-packages` (re-run `setup.sh`) |
| Service runs but no sound | `loginctl enable-linger $USER` done? Earphones paired and *trusted*? Check `journalctl --user -u smartglasses` |
| Pi throttling or random glitches | Use the official PSU and an active cooler |

## 6. Roadmap

1. Move to the Pi: device selection, speed tuning; verify earphone button and autostart on real hardware.
2. Ultrasonic: test on the real sensor and tune `WARN_CM`; low-light fallback using `cam.is_low_light()`.
3. Offline hazard detection (a small model for people, vehicles, stairs), and whether the AI HAT+
   is needed.
4. Reliability: no-internet behaviour with offline OCR, low-battery announcements, watchdog.
5. Polish: wake word, IMU to avoid blurry frames, mounting on a glasses frame, user testing.

