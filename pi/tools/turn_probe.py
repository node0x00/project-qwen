"""Turn in place by small odometry amounts and compare with the tag BEARING change
(the direct measure of rotation). Uses the control page's turn= command."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import json, statistics, time, urllib.request
from robot_control import read_tag
KEY = next(l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("QWEN_STREAM_KEY="))
def cmd(q): urllib.request.urlopen(urllib.request.Request(f"http://localhost:8080/cmd?key={KEY}&{q}", method="POST"), timeout=5)
def status(): return json.load(urllib.request.urlopen(f"http://localhost:8080/status?key={KEY}", timeout=5))
def bearing():
    v = []
    for _ in range(15):
        t = read_tag(0.3)
        if t: v.append(t[2])
        time.sleep(0.06)
    return statistics.median(v) if len(v) > 3 else None
for deg in (8, -8, 8, -8):
    time.sleep(1.0)
    b0 = bearing()
    cmd(f"turn={deg}")
    time.sleep(1.0)
    while status()["mode"] != "idle":
        time.sleep(0.2)
    time.sleep(1.0)
    b1 = bearing()
    act = status()["activity"]
    print(f"turn {deg:+d}: {act} | tag bearing {b0} -> {b1} (robot turned {None if None in (b0, b1) else round(b1 - b0, 1)} deg)", flush=True)
