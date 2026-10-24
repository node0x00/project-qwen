"""Camera-measured turn vs the tag's bearing change (the truth for a turn in place)."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import statistics
import time

import numpy as np

from calib_motion import cmd
from dock_view import features, grab, _bf, RATIO, FX
from dock_view import match as dock_match
from robot_control import read_tag


def bearing():
    v = []
    for _ in range(12):
        t = read_tag(0.3)
        if t:
            v.append(t[2])
        time.sleep(0.06)
    return statistics.median(v) if v else None


def raw_shift(a, b):
    ka, da = features(a)
    kb, db = features(b)
    good = [m for m, n in (p for p in _bf.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < RATIO * n.distance]
    dx = np.array([ka[m.queryIdx].pt[0] - kb[m.trainIdx].pt[0] for m in good])
    return len(good), np.percentile(dx, [10, 50, 90]).round(0).tolist()


for l, r, name in ((1, -1, "right"), (-1, 1, "left"), (1, -1, "right"), (-1, 1, "left")):
    time.sleep(0.8)
    b0, i0 = bearing(), grab()
    for _ in range(2):
        cmd(f"test={l},{r},0.15")
        time.sleep(0.45)
    time.sleep(1.2)
    b1, i1 = bearing(), grab()
    n, h = dock_match(features(i1), features(i0))
    print(f"2 {name:5s} pulses: tag bearing {b0} -> {b1} (robot turned {None if None in (b0, b1) else round(b1 - b0, 1)} deg);"
          f" camera says {None if h is None else round(-h, 1)} deg; raw dx p10/50/90 {raw_shift(i1, i0)}", flush=True)
