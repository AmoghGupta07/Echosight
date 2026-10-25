"""
Mic check: shows live loudness (RMS) and what the voice detector thinks.

  python mic_check.py            # 12 seconds
  python mic_check.py 20         # 20 seconds

Suggested routine (about 4 seconds each):
  1) stay completely silent
  2) say a normal sentence a few times, like you would to the glasses
  3) stay silent while a background voice or noise plays
Then note the typical RMS number for each. Those numbers set MIN_RMS and REL_LEVEL
in voice_loop.py.
"""
import sys
import time

import numpy as np
import sounddevice as sd
import webrtcvad

SR, FRAME_MS = 16000, 30
MIC_DEVICE = 2                       # same value as in voice_loop.py
VAD_AGGRESSIVENESS = 2

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 12
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
n = SR * FRAME_MS // 1000

print("  time    RMS  vad    level")
t0 = time.time()
acc = []
with sd.RawInputStream(samplerate=SR, blocksize=n, dtype="int16",
                       channels=1, device=MIC_DEVICE) as stream:
    while time.time() - t0 < secs:
        data, _ = stream.read(n)
        frame = bytes(data)
        x = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        acc.append((float(np.sqrt(np.mean(x * x))), vad.is_speech(frame, SR)))
        if len(acc) >= 4:                                   # one line per ~120 ms
            rms = max(a[0] for a in acc)
            voice = sum(a[1] for a in acc) >= 2
            bar = "#" * min(60, int(rms / 200))
            print(f"{time.time() - t0:5.1f}s {rms:6.0f}  {'VOICE' if voice else '  -  '} {bar}")
            acc = []
