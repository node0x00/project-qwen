"""Measure how fast the robot drives and turns at a given speed, using the dock
AprilTag as the ruler (no odometry needed). Needs slam_view.py running and the
tag in view, ideally 70-120 cm ahead and roughly centred.

    venv/bin/python calib_motion.py [speed]

  forward: distance to the tag before/after a timed forward move  -> cm/s
  reverse: same, backing up (short: no rear sensors)              -> cm/s
  turn:    the tag's angle before/after a timed spin left, then
           right (which also brings it back)                       -> deg/s
Results are appended to motion_calib.json.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import json
import os
import statistics
import sys
import time
import urllib.request

from robot_control import read_tag

KEY = next(l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("QWEN_STREAM_KEY="))
BASE = "http://localhost:8080"
OUT = "motion_calib.json"


def cmd(q: str):
    urllib.request.urlopen(urllib.request.Request(f"{BASE}/cmd?key={KEY}&{q}", method="POST"), timeout=5)


def look(seconds: float = 1.5):
    """Median tag reading over `seconds` while standing still: (dist_cm, x_cm, bearing, yaw)."""
    samples, end = [], time.time() + seconds
    while time.time() < end:
        t = read_tag(0.3)
        if t:
            samples.append(t)
        time.sleep(0.1)
    if len(samples) < 3:
        return None
    return tuple(statistics.median(s[i] for s in samples) for i in range(4))


def move(left: int, right: int, seconds: float):
    cmd(f"test={left},{right},{seconds}")
    time.sleep(seconds + 1.0)          # move + settle (camera blur, robot rocking)


def main():
    speed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cmd("stop=1")
    time.sleep(0.5)
    start = look()
    if start is None:
        raise SystemExit("tag not in view - point the robot at it first")
    print(f"start: tag {start[0]:.1f} cm, x {start[1]:+.1f} cm, bearing {start[2]:+.1f}, yaw {start[3]:+.1f}")
    if start[0] < 60:
        raise SystemExit("too close for the forward test - start at least 60 cm from the tag")
    res = {"speed": speed, "time": time.strftime("%Y-%m-%d %H:%M")}

    # forward
    fwd_s = 3.0
    move(speed, speed, fwd_s)
    a = look()
    if a:
        res["forward_cm_s"] = round((start[0] - a[0]) / fwd_s, 2)
        print(f"forward {fwd_s}s: {start[0]:.1f} -> {a[0]:.1f} cm  = {res['forward_cm_s']} cm/s")
    # reverse (short)
    rev_s = 1.2
    b0 = a or look()
    move(-speed, -speed, rev_s)
    b = look()
    if b0 and b:
        res["reverse_cm_s"] = round((b[0] - b0[0]) / rev_s, 2)
        print(f"reverse {rev_s}s: {b0[0]:.1f} -> {b[0]:.1f} cm  = {res['reverse_cm_s']} cm/s")
    # turn left then right; the tag's yaw changes by exactly the robot's rotation
    turn_s = 1.0
    c0 = b or look()
    move(-speed, speed, turn_s)
    c = look()
    if c0 and c:
        res["turn_left_deg_s"] = round(abs(c[3] - c0[3]) / turn_s, 2)
        print(f"spin left {turn_s}s: yaw {c0[3]:+.1f} -> {c[3]:+.1f}, bearing {c0[2]:+.1f} -> {c[2]:+.1f}"
              f"  = {res['turn_left_deg_s']} deg/s")
    else:
        print("spin left: tag lost (turned too far) - try a shorter turn")
    d0 = c or look()
    move(speed, -speed, turn_s)
    d = look()
    if d0 and d:
        res["turn_right_deg_s"] = round(abs(d[3] - d0[3]) / turn_s, 2)
        print(f"spin right {turn_s}s: yaw {d0[3]:+.1f} -> {d[3]:+.1f}, bearing {d0[2]:+.1f} -> {d[2]:+.1f}"
              f"  = {res['turn_right_deg_s']} deg/s")
    cmd("stop=1")

    hist = json.load(open(OUT)) if os.path.exists(OUT) else []
    hist.append(res)
    json.dump(hist, open(OUT, "w"), indent=1)
    print("saved", res)


if __name__ == "__main__":
    main()
