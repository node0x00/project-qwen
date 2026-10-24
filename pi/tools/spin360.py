"""Measure the continuous turn rate: spin left from facing the tag until the tag comes
back into view and passes its starting bearing (= one full turn). Ends facing the tag.
    venv/bin/python spin360.py [speed]"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import sys
import time

from calib_motion import cmd, look
from robot_control import read_tag

s = int(sys.argv[1]) if len(sys.argv) > 1 else 1
p0 = look(1.5)
if p0 is None:
    raise SystemExit("tag not in view")
b0 = p0[2]
print(f"start bearing {b0:+.1f}, dist {p0[0]:.1f}", flush=True)
cmd(f"test={-s},{s},20")
t0 = time.time()
lost_at = seen_again_at = None
while time.time() - t0 < 20:
    t = read_tag(0.25)
    el = time.time() - t0
    if t is None and lost_at is None and el > 0.3:
        lost_at = el
    if t is not None and lost_at is not None:
        if seen_again_at is None:
            seen_again_at = el
            print(f"tag back at {el:.2f}s, bearing {t[2]:+.1f}", flush=True)
        if t[2] >= b0:                       # bearing grows while spinning left
            cmd("stop=1")
            print(f"full turn: crossed start bearing at {el:.2f}s -> {360 / el:.1f} deg/s "
                  f"(lost {lost_at:.2f}s)", flush=True)
            break
    time.sleep(0.03)
else:
    cmd("stop=1")
    print(f"no full turn in 20 s (lost {lost_at}, back {seen_again_at})", flush=True)
time.sleep(1.2)
p1 = look(1.5)
print("end:", [round(v, 1) for v in p1] if p1 else None, flush=True)
