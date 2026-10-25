"""
Full assistant loop.

  button -> beep -> record -> SpeechRecognition STT -> route:
        "find my keys"        -> guided search with local tracking (find_object.py)
        "what time is it"     -> answered locally, instantly
        non-visual questions  -> Gemini, text only
        everything else       -> camera frames + Gemini
  Gemini streams sentences -> AudioManager speaks them (non-blocking, interruptible)

Button = play/pause on the Bluetooth earphones (Enter key also works when run in a terminal).
Button behaviour:
  - press while idle        -> start listening
  - press while it's talking -> silence it and start listening right away
  - recording stops by itself when you stop talking (a second press also stops it)
Hazard alerts survive the button: audio.stop() keeps them.

Collision warning (proximity.py): an HC-SR04 ultrasonic sensor runs in its own thread from the
first second, independent of camera/Gemini/internet. Closer than 30 cm -> fast beeps + "Obstacle ahead".

Run:  python assistant.py 2>/dev/null
"""
import re
import signal
import sys
import time
from datetime import datetime

from audio_out import INFO, REPLY, AudioManager
from buttons import make_button
from camera import CameraBuffer
from find_object import ObjectFinder, parse_find_request
from gemini_client import GeminiAssistant
from proximity import start_proximity
from voice_loop import record_until_silence, transcribe

# ---------------- Routing ----------------
# Questions answered locally, instantly, with no cloud call.
TIME_Q = re.compile(r"\bwhat(?:'s| is)?\s+(?:the\s+)?(?:current\s+)?(?:time|date|day)\b|\bwhat time\b", re.I)

# Clearly not about the surroundings -> skip the camera (saves time, data and privacy).
# Everything else defaults to USING the camera, because for a blind user
# "what is this?" / "is there anything here?" is far more likely than not.
NON_VISUAL = re.compile(
    r"\b(weather|joke|tell me a story|calculate|plus|minus|multiplied|divided|"
    r"capital of|who (?:is|was|invented)|define|meaning of|translate|how do you say)\b", re.I)

# Reading requests get one sharp, higher-resolution frame.
READ_Q = re.compile(
    r"\b(read|what does (?:it|this|that) say|label|sign|menu|letter|document|page|text|"
    r"expiry|expiration|ingredients|price)\b", re.I)

STOP_Q = re.compile(r"^\s*(stop|cancel|never ?mind|quiet)\W*$", re.I)


def main():
    # systemd stops us with SIGTERM: exit cleanly so the GPIO pins are released
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    print("Starting audio...")
    audio = AudioManager().start()
    # Safety first: the collision warning starts before the slow parts (camera, Gemini) and does
    # not depend on them. If they fail below, the sensor has already been running.
    proximity = start_proximity(audio)

    print("Starting camera...")
    cam = CameraBuffer().start()
    gemini = GeminiAssistant()

    def on_earphone_button(connected: bool):
        audio.say("Earphone button connected." if connected else "Earphone button disconnected.", INFO)

    button = make_button(on_media_change=on_earphone_button)
    finder = ObjectFinder(cam, gemini, audio, button)
    print(f"Camera: {cam.backend} | Gemini models: {', '.join(gemini.models)}")
    audio.say("Glasses ready.", REPLY)

    start_now = False            # True when a button press already interrupted a reply
    while True:
        if not start_now:
            button.clear()
            print("\nPress the earphone play/pause button (or Enter) to talk...")
            button.wait_press()
        start_now = False

        audio.stop()                          # silence any reply (hazard alerts are kept)
        audio.beep(660, 100)                  # "I'm listening"
        time.sleep(0.15)                      # let the beep finish before the mic opens
        pcm = record_until_silence(button)    # stops when you stop talking; a press also stops it
        if not pcm:
            audio.say("I didn't hear anything.", REPLY)
            continue

        t_start = time.time()
        query = transcribe(pcm)
        print(f"[STT {time.time() - t_start:.1f}s] {query!r}")
        if not query:
            audio.say("Sorry, I didn't catch that.", REPLY)
            continue
        if STOP_Q.match(query):
            continue

        # ---- 0) guided search: "find my keys"
        target = parse_find_request(query)
        if target:
            print(f"[find] target: {target!r}")
            finder.run(target)
            continue

        # ---- 1) local answers
        if TIME_Q.search(query):
            audio.say(datetime.now().strftime("It's %I:%M %p, %A, %d %B."), REPLY)
            continue

        # ---- 2) decide whether to use the camera
        jpegs, note = None, ""
        if not NON_VISUAL.search(query):
            if READ_Q.search(query):
                jpegs = cam.get_query_jpegs(n=1, window_s=1.5, max_side=1600, quality=85)
                note = "The user wants something read. Read the exact text you can see."
            else:
                jpegs = cam.get_query_jpegs(n=3, window_s=2.0)
            if cam.is_low_light():
                note += " The camera reports low light, so details may be unreliable."
            audio.beep(990, 70)               # audible cue: camera frames are being uploaded
            print(f"[camera] sending {len(jpegs)} frame(s), {sum(map(len, jpegs)) // 1024} KB")

        # ---- 3) Gemini, spoken sentence by sentence; a button press cuts in
        t_g = time.time()
        first = True
        interrupted = False
        try:
            for sentence in gemini.ask_stream(query, jpegs, note.strip()):
                if button.was_pressed():
                    audio.stop()
                    interrupted = True        # the press means "I want to talk now"
                    break
                if first:
                    print(f"[Gemini first sentence {time.time() - t_g:.1f}s | "
                          f"total since question {time.time() - t_start:.1f}s]")
                    first = False
                print("  >", sentence)
                audio.say(sentence, REPLY)    # returns immediately; plays in the background
        except Exception as e:
            print(f"[Gemini error] {type(e).__name__}: {e}")
            audio.say("Sorry, I couldn't reach the assistant. Please try again.", REPLY)
        start_now = interrupted


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        print("\nBye.")
