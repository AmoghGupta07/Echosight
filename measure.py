"""
EchoSight measurement helper.  Collects the numbers the project report needs and stores them in
results.json.  Put this file in the project folder (next to assistant.py), activate the venv, then:

    python measure.py camera                # FPS, brightness, JPEG size (15 s)
    python measure.py ultrasonic            # accuracy at known distances + warn / no-warn confusion matrix
    python measure.py gemini  --runs 10     # time to first sentence + total time with real camera frames
    python measure.py stt     --runs 10     # speech-to-text delay and word error rate on 10 fixed phrases
    python measure.py tts                   # Piper time-to-audio for short and long sentences
    python measure.py system  --seconds 120 # CPU / RAM / temperature / throttling while the assistant runs
    python measure.py manual                # things read off a meter or timed by hand (power, battery, trials)
    python measure.py report                # print every stored result as tables you can paste to Claude

Each command ADDS to results.json (re-running a command replaces only its own entry).
Send results.json (and, if you like, the printed report) to get the write-up tables filled in.
Everything here only reads sensors/APIs; it never changes the assistant's own settings.
"""
import argparse
import json
import os
import statistics as st
import sys
import time

RESULTS = "results.json"


# ---------------------------------------------------------------- storage
def load():
    try:
        with open(RESULTS) as f:
            return json.load(f)
    except Exception:
        return {}


def save(key, value):
    data = load()
    value["_when"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data[key] = value
    with open(RESULTS, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[saved to {RESULTS} under '{key}']")


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": round(st.mean(xs), 3), "median": round(st.median(xs), 3),
            "std": round(st.pstdev(xs), 3), "min": round(min(xs), 3), "max": round(max(xs), 3)}


# ---------------------------------------------------------------- camera
def cmd_camera(a):
    from camera import CameraBuffer, to_jpeg
    out = {}
    for w, h in ((1280, 720), (640, 480)):
        print(f"Camera {w}x{h} for {a.seconds}s ...")
        cam = CameraBuffer(width=w, height=h).start()
        fps, bright = [], []
        t_end = time.time() + a.seconds
        while time.time() < t_end:
            time.sleep(2)
            fps.append(cam.measured_fps(2.0))
            bright.append(cam.brightness(2.0))
        t = time.time()
        jpg = to_jpeg(cam.latest(), 1024, 70)
        enc_ms = (time.time() - t) * 1000
        sizes = [len(j) // 1024 for j in cam.get_query_jpegs(n=3)]
        out[f"{w}x{h}"] = {"backend": cam.backend, "fps": stats(fps), "brightness": stats(bright),
                           "jpeg_kb_each_of_3": sizes, "jpeg_encode_ms": round(enc_ms, 1)}
        print(f"   FPS {stats(fps)}")
        cam.stop()
        time.sleep(1)
    out["note"] = "Count 'Corrupt JPEG' / 'read error' lines separately by running without 2>/dev/null."
    save("camera", out)


# ---------------------------------------------------------------- ultrasonic
def cmd_ultrasonic(a):
    from proximity import FAR, HCSR04, WARN_CM
    warn = a.warn if a.warn else WARN_CM
    print(f"Warning threshold used for the confusion matrix: {warn:g} cm "
          f"(set --warn 15 if SG_WARN_CM is not exported).")
    print("Use a ruler. Hold a flat object (book, box) square-on in front of the sensor.\n")
    sensor = HCSR04()
    rows, tp = [], dict(TP=0, FP=0, TN=0, FN=0)
    dists = [float(x) for x in a.distances.split(",")]
    for d in dists:
        input(f"Place the object at {d:g} cm, then press Enter ... ")
        vals, bad = [], 0
        for _ in range(a.samples):
            r = sensor.measure()
            if r is None or r == FAR:
                bad += 1
                vals.append(None)
            else:
                vals.append(r)
            time.sleep(0.07)
        good = [v for v in vals if v is not None]
        # replicate the assistant's rule: alert when 2 of the last 3 readings are < threshold
        window, alerts, evals = [], 0, 0
        for v in vals:
            window = (window + [v if v is not None else 1e9])[-3:]
            if len(window) == 3:
                evals += 1
                alerts += sum(1 for x in window if x < warn) >= 2
        truth_near = d < warn
        tp["TP" if truth_near else "FP"] += alerts if truth_near else alerts
        tp["FN" if truth_near else "TN"] += (evals - alerts)
        row = {"true_cm": d, "mean_cm": round(st.mean(good), 2) if good else None,
               "std_cm": round(st.pstdev(good), 2) if good else None,
               "error_cm": round(st.mean(good) - d, 2) if good else None,
               "no_echo_or_far": bad, "alert_rate": round(alerts / evals, 3) if evals else None}
        rows.append(row)
        print("  ", row)
    TP, FP, TN, FN = tp["TP"], tp["FP"], tp["TN"], tp["FN"]
    prec = TP / (TP + FP) if TP + FP else None
    rec = TP / (TP + FN) if TP + FN else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    acc = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN else None
    r4 = lambda x: None if x is None else round(x, 4)
    save("ultrasonic", {"warn_cm": warn, "samples_per_distance": a.samples, "rows": rows,
                        "confusion": tp, "accuracy": r4(acc), "precision": r4(prec),
                        "recall": r4(rec), "f1": r4(f1)})
    sensor.close()


# ---------------------------------------------------------------- gemini
QUERIES = ["What's in front of me?", "Describe what is in front of me in detail.",
           "Read this.", "Is there anything dangerous ahead?", "What colour is the object in front of me?"]


def cmd_gemini(a):
    from camera import CameraBuffer
    from gemini_client import GeminiAssistant
    cam = CameraBuffer().start()
    time.sleep(2)
    g = GeminiAssistant()
    first, total, kb, models, errs = [], [], [], [], 0
    for i in range(a.runs):
        q = QUERIES[i % len(QUERIES)]
        g.reset()
        jpegs = cam.get_query_jpegs(n=3)
        t0 = time.time()
        t_first = None
        try:
            for _ in g.ask_stream(q, jpegs):
                if t_first is None:
                    t_first = time.time() - t0
            first.append(t_first)
            total.append(time.time() - t0)
            kb.append(sum(map(len, jpegs)) // 1024)
            models.append(g.model)
            print(f"  run {i + 1}: first {t_first:.2f}s, total {total[-1]:.2f}s, {kb[-1]} KB, {g.model}")
        except Exception as e:
            errs += 1
            print(f"  run {i + 1}: ERROR {type(e).__name__}: {e}")
        time.sleep(1)
    cam.stop()
    save("gemini", {"runs": a.runs, "errors": errs, "first_sentence_s": stats(first),
                    "total_s": stats(total), "upload_kb": stats(kb), "models_used": sorted(set(models)),
                    "queries": QUERIES})


# ---------------------------------------------------------------- stt
PHRASES = ["what is in front of me", "read this label", "find my keys", "what time is it",
           "is there anything dangerous ahead", "where is my phone", "what colour is this",
           "describe the room", "how many people are here", "what does the sign say"]


def wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1][-1] / max(len(r), 1)


def cmd_stt(a):
    from voice_loop import SAMPLE_RATE, record_until_silence, transcribe
    delays, wers, rec_s, rows, empty = [], [], [], [], 0
    print("Speak each phrase as shown (use the same microphone/earphones as normal use).")
    for i in range(a.runs):
        ref = PHRASES[i % len(PHRASES)]
        input(f"\n[{i + 1}/{a.runs}] Press Enter, then say:  \"{ref}\"  ")
        pcm = record_until_silence()
        if not pcm:
            empty += 1
            rows.append({"ref": ref, "hyp": None})
            print("  nothing recorded")
            continue
        rec_s.append(len(pcm) / 2 / SAMPLE_RATE)
        t0 = time.time()
        hyp = transcribe(pcm)
        dt = time.time() - t0
        delays.append(dt)
        w = wer(ref, hyp)
        wers.append(w)
        rows.append({"ref": ref, "hyp": hyp, "wer": round(w, 3), "stt_s": round(dt, 2)})
        print(f"  heard: {hyp!r}   WER {w:.2f}   STT {dt:.2f}s")
    save("stt", {"engine": "SpeechRecognition (Google web)", "runs": a.runs, "nothing_recorded": empty,
                 "stt_delay_s": stats(delays), "wer": stats(wers), "recording_s": stats(rec_s), "rows": rows})


# ---------------------------------------------------------------- tts
def cmd_tts(a):
    from audio_out import AudioManager
    audio = AudioManager()
    audio._load_piper()
    texts = {"short (3 words)": "Obstacle ahead.",
             "medium (12 words)": "I can see a table with a laptop on it, about two steps ahead of you.",
             "long (30 words)": "In front of you there is a wooden desk with a laptop, a water bottle and some papers. "
                                "A chair is pulled out on your left, so walk slowly to avoid it."}
    out = {}
    audio._synthesize("warm up.")
    for k, t in texts.items():
        times, secs = [], 0
        for _ in range(5):
            t0 = time.time()
            pcm = audio._synthesize(t)
            times.append(time.time() - t0)
            secs = len(pcm) / audio.rate
        out[k] = {"synth_s": stats(times), "audio_s": round(secs, 2),
                  "real_time_factor": round(st.mean(times) / secs, 3)}
        print(f"  {k}: synth {st.mean(times):.2f}s for {secs:.1f}s of audio")
    save("tts", out)


# ---------------------------------------------------------------- system
def read(path):
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return ""


def cpu_times():
    p = read("/proc/stat").splitlines()[0].split()[1:]
    v = list(map(int, p))
    return sum(v), v[3] + v[4]


def mem_used_mb():
    m = {l.split(":")[0]: int(l.split()[1]) for l in read("/proc/meminfo").splitlines() if ":" in l}
    return (m["MemTotal"] - m["MemAvailable"]) / 1024, m["MemTotal"] / 1024


def temp_c():
    s = read("/sys/class/thermal/thermal_zone0/temp").strip()
    return int(s) / 1000 if s else None


def throttled():
    import subprocess
    try:
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        return None


def proc_rss_mb(pid):
    for l in read(f"/proc/{pid}/status").splitlines():
        if l.startswith("VmRSS"):
            return int(l.split()[1]) / 1024
    return None


def cmd_system(a):
    print(f"Sampling for {a.seconds}s. Use the assistant normally in another terminal meanwhile "
          f"(ask a few questions, walk towards an obstacle).")
    cpu, mem, tmp, rss = [], [], [], []
    t0, i0 = cpu_times()
    end = time.time() + a.seconds
    while time.time() < end:
        time.sleep(2)
        t1, i1 = cpu_times()
        cpu.append(100 * (1 - (i1 - i0) / max(t1 - t0, 1)))
        t0, i0 = t1, i1
        mem.append(mem_used_mb()[0])
        tmp.append(temp_c())
        if a.pid:
            rss.append(proc_rss_mb(a.pid))
    th = throttled()
    out = {"seconds": a.seconds, "cpu_percent_all_cores": stats(cpu), "ram_used_mb": stats(mem),
           "ram_total_mb": round(mem_used_mb()[1]), "cpu_temp_c": stats(tmp),
           "get_throttled": th,
           "get_throttled_help": "0x0 = never throttled/undervolted. Bit 0 (0x1) = undervoltage now, "
                                 "bit 16 (0x10000) = undervoltage has occurred since boot."}
    if a.pid:
        out["assistant_rss_mb"] = stats(rss)
    print(json.dumps(out, indent=2))
    save("system", out)


# ---------------------------------------------------------------- manual
def ask(prompt, cast=float):
    s = input(prompt + " (blank = skip): ").strip()
    if not s:
        return None
    try:
        return cast(s)
    except ValueError:
        return s


def cmd_manual(a):
    print("Enter what you measured by hand. Leave blank to skip. See MEASUREMENTS.md for how.\n")
    d = {
        "pi_idle_watts": ask("Pi idle power from a USB power meter (W)"),
        "pi_assistant_idle_watts": ask("Pi power with assistant running, waiting for button (W)"),
        "pi_peak_watts": ask("Pi peak power while Gemini/STT is active (W)"),
        "pi_voltage_under_load_v": ask("USB voltage at the Pi under load (V)"),
        "battery_runtime_hours": ask("Battery run time, full charge to Pi shut-off (hours)"),
        "undervoltage_warnings_seen": ask("Undervoltage / power warnings seen (yes/no/how many)", str),
        "boot_to_ready_s": ask("Seconds from power-on to spoken 'Glasses ready'"),
        "bt_connect_s": ask("Seconds for the Buds to reconnect after power-on"),
        "hazard_trials_total": ask("Obstacle walk-up trials (total)", int),
        "hazard_trials_warned_in_time": ask("...of which beeps/warning came before contact", int),
        "find_trials_total": ask("Find-my-object trials (total)", int),
        "find_trials_reached": ask("...of which you were guided to within reach", int),
        "find_mean_time_s": ask("Mean time to reach the object (s)"),
        "scene_trials_total": ask("Scene-description questions asked (total)", int),
        "scene_trials_correct": ask("...of which the answer was correct and useful", int),
        "read_trials_total": ask("Reading (OCR) trials (total)", int),
        "read_trials_correct": ask("...of which the text was read fully correctly", int),
        "low_light_trials_total": ask("Low-light trials (total)", int),
        "low_light_ultrasonic_warned": ask("...of which ultrasonic warned correctly", int),
        "low_light_camera_answer_ok": ask("...of which the camera description was still usable", int),
        "notes": ask("Anything else worth noting", str),
    }
    save("manual", d)


# ---------------------------------------------------------------- report
def cmd_report(a):
    d = load()
    if not d:
        print("No results yet.")
        return
    print(json.dumps(d, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("camera"); c.add_argument("--seconds", type=int, default=15); c.set_defaults(f=cmd_camera)
    u = sub.add_parser("ultrasonic")
    u.add_argument("--distances", default="5,10,12,15,18,20,30,50,100")
    u.add_argument("--samples", type=int, default=40)
    u.add_argument("--warn", type=float, default=15.0)
    u.set_defaults(f=cmd_ultrasonic)
    g = sub.add_parser("gemini"); g.add_argument("--runs", type=int, default=10); g.set_defaults(f=cmd_gemini)
    s = sub.add_parser("stt"); s.add_argument("--runs", type=int, default=10); s.set_defaults(f=cmd_stt)
    sub.add_parser("tts").set_defaults(f=cmd_tts)
    y = sub.add_parser("system"); y.add_argument("--seconds", type=int, default=120)
    y.add_argument("--pid", type=int, default=0, help="PID of assistant.py to log its own memory")
    y.set_defaults(f=cmd_system)
    sub.add_parser("manual").set_defaults(f=cmd_manual)
    sub.add_parser("report").set_defaults(f=cmd_report)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
