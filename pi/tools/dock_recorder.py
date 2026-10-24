"""Record manual docking runs for calibration. Logs the tag pose, wheel counts and charging
~5x a second. For each dock it reports where you were lined up just before turning around
(the last steady pose while the tags were in view), and counts the dock as a success when
charging holds 3 s (charging flicker within 20 s doesn't start a new run).
    venv/bin/python dock_recorder.py        (Ctrl-C to stop; log in dock_runs.csv)"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import csv
import json
import time
import urllib.request

from robot_control import read_tag

KEY = next(l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("QWEN_STREAM_KEY="))


def status():
    return json.load(urllib.request.urlopen(f"http://localhost:8080/status?key={KEY}", timeout=3))


def charging_now():
    try:
        return "charging" in open("/dev/shm/slam_view.jpg.info").read()
    except OSError:
        return False


out = open("dock_runs.csv", "a", newline="")
w = csv.writer(out)
w.writerow(["time", "mode", "odom_l", "odom_r", "charging", "tag_dist", "tag_bearing", "lateral", "out", "heading", "event"])
last_odom, still_since = None, None
steady = None              # last steady pose with both tags in view: (time, lateral, out, heading)
charge_since, last_charge_end, run = None, 0.0, 0
print("recording - drive with WASD; pause 1-2 s facing the tags before you turn around", flush=True)
try:
    while True:
        time.sleep(0.2)
        try:
            st = status()
        except Exception:
            continue
        now = time.time()
        t = read_tag(0.4)
        odom = tuple(st.get("odom", [0, 0]))
        event = ""
        if odom != last_odom:                       # wheels moved: not still
            still_since, last_odom = None, odom
        else:
            still_since = still_since or now
        both = t is not None and len(t) >= 15 and abs(t[7] - t[11]) < 0.5
        if still_since and now - still_since >= 1.0 and both and (steady is None or steady[0] < still_since):
            steady = (now, t[4], t[5], t[6])
            event = f"steady: {t[5]:.0f} cm out, lateral {t[4]:+.1f} cm, heading {t[6]:+.1f}"
            print(time.strftime("%H:%M:%S"), event, flush=True)
        chg = charging_now()
        if chg:
            charge_since = charge_since or now
            if charge_since > 0 and now - charge_since >= 3.0:
                if now - last_charge_end > 20:      # a new dock, not a flicker of the last one
                    run += 1
                    event = "DOCKED (charging 3 s)"
                    if steady:
                        event += f" - last steady pose before the turn: {steady[2]:.0f} cm out, lateral {steady[1]:+.1f} cm, heading {steady[3]:+.1f}"
                    print(time.strftime("%H:%M:%S"), f"RUN {run}: {event}", flush=True)
                charge_since = -1
        else:
            if charge_since is not None:
                last_charge_end = now
            charge_since = None
        pose = [round(x, 1) for x in (t[0], t[2], t[4], t[5], t[6])] if t and len(t) >= 7 else ["", "", "", "", ""]
        w.writerow([round(now, 2), st.get("mode"), odom[0], odom[1], int(chg), *pose, event])
        out.flush()
except KeyboardInterrupt:
    pass
