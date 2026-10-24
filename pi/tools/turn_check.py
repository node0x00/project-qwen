"""Compare the camera-measured turn (scene shift) with the tag's heading change."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import time

from dock_view import features, grab
from dock_view import match as dock_match
from calib_motion import cmd
from robot_control import read_tag


def tag_head():
    vals = []
    for _ in range(15):
        t = read_tag(0.3)
        if t and len(t) >= 7:
            vals.append(t[6])
        time.sleep(0.07)
    vals.sort()
    return vals[len(vals) // 2] if vals else None


for pulses in (2, 3):
    time.sleep(1.0)
    h0, f0 = tag_head(), features(grab())
    for _ in range(pulses):
        cmd("test=1,-1,0.15")  # right
        time.sleep(0.45)
    time.sleep(1.0)
    h1, f1 = tag_head(), features(grab())
    n, heading = dock_match(f1, f0)
    meas = None if heading is None else heading
    print(f"{pulses} right pulses: tag heading {h0} -> {h1} (change {None if None in (h0, h1) else round(h1 - h0, 1)}); "
          f"camera-measured {meas if meas is None else round(meas, 1)} deg ({n} matches)", flush=True)
    for _ in range(pulses):                 # turn back
        cmd("test=-1,1,0.15")
        time.sleep(0.45)
