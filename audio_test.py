"""
Listen-and-check demo of the audio manager. Run:  python audio_test.py
Each part tells you what you should HEAR.
"""
import time

from audio_out import HAZARD, INFO, NAV, REPLY, AudioManager

audio = AudioManager().start()
t0 = time.time()


def step(title, expect):
    print(f"\n[{time.time() - t0:5.1f}s] {title}\n        you should hear: {expect}")


time.sleep(0.5)

step("1. A hazard interrupts a long reply, then the reply carries on",
     "a long reply, cut off by 'Careful, step down ahead', then the reply resumes")
audio.say("This is a long answer from the assistant. It has several sentences in it.", REPLY)
audio.say("Here is the second sentence, and then a third one follows right after it.", REPLY)
time.sleep(2.2)
audio.hazard("Careful. Step down ahead.", key="step")
audio.wait_idle(30)

step("2. Hazards are not repeated in a flurry",
     "'Car approaching' ONCE, even though it is triggered three times")
for _ in range(3):
    audio.hazard("Car approaching on your left.", key="car", cooldown=5.0)
    time.sleep(0.3)
audio.wait_idle(15)

step("3. Newer navigation guidance replaces older, waiting guidance",
     "'A little left' and then ONLY the newest one, 'Straight ahead' - the middle ones are skipped")
for text in ["A little to your left.", "Turn left.", "A little to your right.", "Straight ahead."]:
    audio.say(text, NAV, group="nav", replace=True)
    time.sleep(0.15)
audio.wait_idle(15)

step("4. Beeps play ON TOP of speech (this is how proximity beeps will work)",
     "a sentence with a few beeps sounding over it")
audio.say("I am speaking a fairly long sentence while the beeps sound over the top of it.", REPLY)
time.sleep(0.6)
for _ in range(4):
    audio.beep(1000, 90)
    time.sleep(0.45)
audio.wait_idle(15)

step("5. Low-priority chatter waits until replies are finished",
     "'First reply', 'Second reply', and only then 'Status message'")
audio.say("Status message.", INFO)
audio.say("First reply.", REPLY)
audio.say("Second reply.", REPLY)
audio.wait_idle(15)

step("6. stop() (the button) silences a reply immediately",
     "the start of a long sentence, then silence")
audio.say("This sentence is long enough that you will certainly be able to cut it off in the middle, "
          "and nothing after this point should be heard at all.", REPLY)
time.sleep(1.6)
audio.stop()
time.sleep(1.5)

step("7. ...but stop() does not silence a hazard alert",
     "a reply starts, you press stop, but 'Obstacle ahead, stop' is still spoken")
audio.say("A reply that is about to be cancelled by the button press.", REPLY)
audio.hazard("Obstacle ahead. Stop.", key="obstacle")
time.sleep(0.2)
audio.stop()
audio.wait_idle(15)

print("\nDone.")
audio.close()
