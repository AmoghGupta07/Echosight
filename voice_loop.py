"""
Step 1: Voice loop for smart glasses (Raspberry Pi 5)

  [Enter / button] -> record until you stop talking (VAD)
                   -> SpeechRecognition (Google web API, or offline PocketSphinx)
                   -> handle(text)            <-- Gemini plugs in here in step 3
                   -> Piper (offline TTS) -> speaker

For now handle() just echoes what you said, so you can verify the whole audio chain.
"""
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import webrtcvad
import speech_recognition as sr

# ---------------- Config ----------------
SAMPLE_RATE = 16000          # SpeechRecognition + webrtcvad both work with 16 kHz mono
FRAME_MS = 30                # webrtcvad accepts 10/20/30 ms frames
VAD_AGGRESSIVENESS = 2       # 0 (lenient) .. 3 (strict). Raise if noisy surroundings.
SILENCE_END_MS = 900         # stop recording after this much silence
START_TIMEOUT_S = 6          # give up if the user never speaks
MAX_RECORD_S = 30            # safety net only; silence normally ends the recording first
PREROLL_FRAMES = 10          # keep ~300 ms before speech starts so first word isn't clipped
MIN_RMS = 200                # frames quieter than this are never speech (int16 scale, 0..32767)
REL_LEVEL = 0.25             # after the user starts, frames below 25% of their level count as
                             # background (distant talkers are quieter than the wearer)
MIN_SPEECH_MS = 250          # shorter than this = a click or cough, not a question
DEBUG_AUDIO = True           # prints levels so you can tune MIN_RMS / REL_LEVEL

MIC_DEVICE = 2                # Realtek microphone used by speech_test.py; change for other systems
STT_ENGINE = "google"        # "google" = free web API, accurate, needs internet (audio is uploaded)
                             # "sphinx" = fully offline PocketSphinx, less accurate (pip install pocketsphinx)
STT_LANGUAGE = "en-US"       # BCP-47 code, e.g. "en-GB", "en-IN", "ta-IN" (sphinx: English only)
PIPER_VOICE = "voices/en_US-lessac-medium.onnx"
PIPER_RATE = 22050           # must match the voice's sample rate (medium voices = 22050)
SPEAKER_DEVICE = None        # None = system default; or an index/name from sd.query_devices()


# ---------------- Speech-to-text ----------------
recognizer = sr.Recognizer()


def _rms(frame: bytes) -> float:
    x = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def record_until_silence(button=None) -> bytes | None:
    """
    Record one spoken query; return raw int16 PCM, or None if nothing usable was said.

    Speech = WebRTC VAD says voice AND the frame is loud enough. Once the user starts,
    their own loudness becomes the reference: quieter voices (people behind them) are
    treated as silence, so the recording ends when the wearer stops talking.

    If a button is given, pressing it again ends the recording immediately (manual stop).
    """
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_samples = SAMPLE_RATE * FRAME_MS // 1000
    q: queue.Queue[bytes] = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(bytes(indata))

    preroll: list[bytes] = []
    frames: list[bytes] = []
    noise: list[float] = []
    started = False
    silent_ms = voiced_ms = 0
    ref = 0.0                      # decaying peak of the user's own loudness
    t0 = time.time()

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, blocksize=frame_samples, dtype="int16",
        channels=1, device=MIC_DEVICE, callback=callback,
    ):
        while True:
            frame = q.get()
            rms = _rms(frame)
            speech = rms >= MIN_RMS and vad.is_speech(frame, SAMPLE_RATE)

            # optional manual stop: a button press ends the recording right away
            if button is not None and button.was_pressed():
                if not started:
                    return None
                break

            if not started:
                preroll = (preroll + [frame])[-PREROLL_FRAMES:]
                if speech:
                    started, ref, voiced_ms = True, rms, FRAME_MS
                    frames.extend(preroll)
                else:
                    noise.append(rms)
                    if time.time() - t0 > START_TIMEOUT_S:
                        return None
                continue

            frames.append(frame)
            ref = max(ref * 0.995, rms) if speech else ref * 0.995
            if speech and rms >= REL_LEVEL * ref:
                voiced_ms += FRAME_MS
                silent_ms = 0
            else:
                silent_ms += FRAME_MS
            if silent_ms >= SILENCE_END_MS or time.time() - t0 > MAX_RECORD_S:
                break

    # drop most of the trailing silence/background, keep ~150 ms
    trim = max(0, silent_ms // FRAME_MS - 5)
    if trim:
        frames = frames[:-trim]

    if DEBUG_AUDIO:
        floor = float(np.median(noise)) if noise else 0.0
        print(f"[audio] {len(frames) * FRAME_MS / 1000:.1f}s kept, "
              f"{voiced_ms / 1000:.1f}s voiced, your level ~{ref:.0f}, room noise ~{floor:.0f}")
    if voiced_ms < MIN_SPEECH_MS:
        return None
    return b"".join(frames)


def transcribe(pcm: bytes) -> str:
    """Raw 16 kHz mono int16 PCM -> text ('' if nothing was understood or the service failed)."""
    audio = sr.AudioData(pcm, SAMPLE_RATE, 2)        # 2 bytes per sample = int16
    try:
        if STT_ENGINE == "sphinx":
            return recognizer.recognize_sphinx(audio).strip()
        return recognizer.recognize_google(audio, language=STT_LANGUAGE).strip()
    except sr.UnknownValueError:
        return ""                                    # audio was not intelligible speech
    except sr.RequestError as e:                     # no internet, quota, missing pocketsphinx...
        print(f"[stt] {STT_ENGINE} request failed: {e}")
        return ""


# ---------------- Push-to-talk ----------------
class EnterKeyButton:
    """
    Stand-in for the hardware button: pressing Enter = pressing the button.
    On the Pi this gets replaced by a GPIO button with the same three methods.
    """
    def __init__(self):
        self._q: queue.Queue[float] = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                input()
            except EOFError:
                return
            self._q.put(time.time())

    def clear(self):
        while not self._q.empty():
            self._q.get_nowait()

    def wait_press(self):
        return self._q.get()

    def was_pressed(self) -> bool:
        try:
            self._q.get_nowait()
            return True
        except queue.Empty:
            return False


def record_until_button(button, max_s: float = 15.0) -> bytes | None:
    """Record until the button is pressed again. Reliable in constant noise."""
    frame_samples = SAMPLE_RATE * FRAME_MS // 1000
    q: queue.Queue[bytes] = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(bytes(indata))

    frames: list[bytes] = []
    t0 = time.time()
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, blocksize=frame_samples, dtype="int16",
        channels=1, device=MIC_DEVICE, callback=callback,
    ):
        while time.time() - t0 < max_s:
            try:
                frames.append(q.get(timeout=0.1))
            except queue.Empty:
                pass
            if button.was_pressed():
                break

    if DEBUG_AUDIO and frames:
        level = float(np.percentile([_rms(f) for f in frames], 90))
        print(f"[audio] {len(frames) * FRAME_MS / 1000:.1f}s recorded, level ~{level:.0f}")
    if len(frames) * FRAME_MS < 300:
        return None
    return b"".join(frames)


# ---------------- Text-to-speech ----------------
def play_pcm(pcm: bytes, rate: int, block: bool = True) -> None:
    """Play raw mono int16 audio. Works on Windows, macOS, Linux and the Pi."""
    audio = np.frombuffer(pcm, dtype=np.int16)
    sd.play(audio, samplerate=rate, device=SPEAKER_DEVICE)
    if block:
        sd.wait()


def speak(text: str, block: bool = True) -> None:
    """Speak text with Piper. Interrupts anything already playing."""
    stop_speaking()
    result = subprocess.run(
        [sys.executable, "-m", "piper", "--model", PIPER_VOICE, "--output-raw"],
        input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0 or not result.stdout:
        print(f"[tts] Piper failed: {result.stderr.decode(errors='ignore')[-300:]}")
        print(f"[tts] (text was: {text})")
        return
    play_pcm(result.stdout, PIPER_RATE, block)


def stop_speaking() -> None:
    sd.stop()


# ---------------- Brain (placeholder) ----------------
def handle(query: str) -> str:
    """Step 3 replaces this with: decide if vision is needed -> frames -> Gemini."""
    return f"You said: {query}"


# ---------------- Main loop ----------------
def main():
    speak("Voice system ready.")
    while True:
        # Later: replace with wake word and/or GPIO push-to-talk button.
        input("\nPress Enter, then speak... ")
        stop_speaking()
        print("Listening...")
        pcm = record_until_silence()
        if not pcm:
            speak("I didn't hear anything.")
            continue

        t = time.time()
        text = transcribe(pcm)
        print(f"[STT {time.time() - t:.1f}s] {text!r}")
        if not text:
            speak("Sorry, I didn't catch that.")
            continue

        speak(handle(text))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_speaking()
        print("\nBye.")
