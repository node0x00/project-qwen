"""Run the tag approach over and over from different starting spots, unattended, and log
how close each run ends to the turn spot. Needs slam_view.py running and clear floor in
front of the dock (~1 m out, ~50 cm each side).

Each trial: from wherever the last one ended, back up a random 10-25 cm, turn a random
15-40 deg either way, drive forward a little (so it starts off the centre line and at an
angle), then press Find dock and wait.
    venv/bin/python approach_trials.py [n_trials]
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import json
import random
import sys
import time
import urllib.request

KEY = next(l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("QWEN_STREAM_KEY="))
BASE = "http://localhost:8080"
PULSE = 0.15


def cmd(q):
    urllib.request.urlopen(urllib.request.Request(f"{BASE}/cmd?key={KEY}&{q}", method="POST"), timeout=5)


def status():
    return json.load(urllib.request.urlopen(f"{BASE}/status?key={KEY}", timeout=5))


def pulses(l, r, n):
    for _ in range(n):
        cmd(f"test={l},{r},{PULSE}")
        time.sleep(PULSE + 0.35)


def wait_idle(timeout=420):
    t0 = time.time()
    time.sleep(2)
    while time.time() - t0 < timeout:
        st = status()
        if st["mode"] == "idle":
            return st["activity"], time.time() - t0
        time.sleep(1)
    cmd("stop=1")
    return "timed out", time.time() - t0


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    results = []
    for i in range(1, n + 1):
        if i > 1:
            back = random.randint(7, 16)                 # ~10-25 cm
            turn = random.randint(2, 4) * random.choice([-1, 1])
            fwd = random.randint(2, 6)
            print(f"trial {i}: reposition - back {back}, turn {turn:+d}, forward {fwd} pulses", flush=True)
            pulses(-1, -1, back)
            pulses(-1, 1, turn) if turn > 0 else pulses(1, -1, -turn)
            pulses(1, 1, fwd)
            time.sleep(1)
        cmd("find=1")
        result, secs = wait_idle()
        print(f"trial {i}: {secs:.0f}s  {result}", flush=True)
        results.append(result)
        if "READY" not in result and "lost" not in result:
            if "IR" in result or "no tag" in result or "timed out" in result:
                print("stopping trials - needs a look", flush=True)
                break
    print("\nSUMMARY")
    for i, r in enumerate(results, 1):
        print(f"  {i}: {r}")


if __name__ == "__main__":
    main()
