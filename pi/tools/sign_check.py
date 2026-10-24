"""Check every motion's effect on the two-tag pose (lateral, out, heading) and on the tags'
picture position, so the code's left/right/heading conventions can be verified."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import statistics
import time

from calib_motion import cmd
from robot_control import read_tag


def pose():
    got = []
    for _ in range(20):
        t = read_tag(0.3)
        if t and len(t) >= 15 and abs(t[7] - t[11]) < 0.5:     # both tags (board) only
            got.append(t)
        time.sleep(0.06)
    if len(got) < 4:
        return None
    m = lambda i: round(statistics.median(g[i] for g in got), 1)
    return {"bearing": m(2), "lat": m(4), "out": m(5), "head": m(6)}


def move(l, r, n):
    for _ in range(n):
        cmd(f"test={l},{r},0.15")
        time.sleep(0.5)
    time.sleep(1.0)


for name, l, r, n in [("D-direction (1,-1) x2", 1, -1, 2), ("A-direction (-1,1) x2", -1, 1, 2),
                      ("forward x3", 1, 1, 3), ("reverse x3", -1, -1, 3)]:
    a = pose()
    move(l, r, n)
    b = pose()
    if a and b:
        print(f"{name:24s} bearing {a['bearing']:+6.1f} -> {b['bearing']:+6.1f} | lat {a['lat']:+6.1f} -> {b['lat']:+6.1f}"
              f" | out {a['out']:5.1f} -> {b['out']:5.1f} | heading {a['head']:+6.1f} -> {b['head']:+6.1f}", flush=True)
    else:
        print(f"{name}: lost the board ({a}, {b})", flush=True)
