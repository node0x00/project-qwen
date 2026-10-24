"""Docking attempt with short pulses ("inching") from a hand-placed start:
facing the dock tag, centred, ~20 cm in front of the dock.

  1. reverse pulse size: 2 back pulses, measured with the tag, then 2 forward to return
  2. turn 180 left in pulses; the pulses made while the tag is still visible are
     measured and their average is used to count the rest
  3. reverse pulses until charge current appears or the distance budget is used

Needs slam_view.py running.   venv/bin/python dock_pulse.py [reverse_cm]
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import statistics
import sys
import time

from battery_monitor import CHARGE_CURRENT_SIGN, CHARGING_THRESHOLD_A, INA226
from calib_motion import cmd, look

PULSE_S = 0.15
SETTLE_S = 0.8
MARGIN_CM = 6
# The tag's bearing changes less than the robot actually turns (the camera sits ahead of
# the turning centre). First attempt: 41 pulses counted as 180 deg really turned ~220 deg.
ROTATION_PER_BEARING = 220 / 180


def pulse(left, right):
    cmd(f"test={left},{right},{PULSE_S}")
    time.sleep(PULSE_S + SETTLE_S)


def main():
    reverse_cm = float(sys.argv[1]) if len(sys.argv) > 1 else 20
    ina = INA226()
    charging = lambda: ina.current() * CHARGE_CURRENT_SIGN > CHARGING_THRESHOLD_A
    cmd("stop=1")
    time.sleep(0.5)
    p0 = look(2.0)
    if p0 is None:
        raise SystemExit("tag not in view")
    print(f"start: tag {p0[0]:.1f} cm  x {p0[1]:+.1f}  bearing {p0[2]:+.1f}  yaw {p0[3]:+.1f}", flush=True)

    # 1. reverse pulse size
    pulse(-1, -1); pulse(-1, -1)
    p1 = look(1.5)
    pulse(1, 1); pulse(1, 1)
    p2 = look(1.5)
    if not (p1 and p2):
        raise SystemExit("lost the tag during the pulse-size check")
    rev_cm = (p1[0] - p0[0]) / 2
    fwd_cm = (p1[0] - p2[0]) / 2
    print(f"per pulse: reverse {rev_cm:.2f} cm, forward {fwd_cm:.2f} cm; now {p2[0]:.1f} cm, "
          f"bearing {p2[2]:+.1f}", flush=True)
    if rev_cm < 0.3:
        raise SystemExit("reverse pulses barely move the robot")

    # 2. turn 180 left in pulses
    steps, prev, total, n = [], p2, 0.0, 0
    while True:
        pulse(-1, 1)
        n += 1
        cur = look(1.0)
        if cur and prev:
            d = (cur[2] - prev[2]) * ROTATION_PER_BEARING   # bearing grows while turning left
            if d > 0:
                steps.append(d)
        prev = cur
        per = statistics.mean(steps) if steps else 5.0
        total = sum(steps) + per * (n - len(steps))
        print(f"turn pulse {n}: {'tag ' + format(cur[2], '+.1f') if cur else 'tag out of view'}"
              f"  ~{total:.0f} deg (avg {per:.1f}/pulse from {len(steps)})", flush=True)
        if total >= 180 - per / 2:
            break
        if n > 80:
            raise SystemExit("too many pulses - stopping")

    # 3. reverse onto the pads
    budget = int((reverse_cm + MARGIN_CM) / rev_cm) + 1
    print(f"reversing: up to {budget} pulses (~{rev_cm:.1f} cm each)", flush=True)
    for k in range(1, budget + 1):
        cmd(f"test=-1,-1,{PULSE_S}")
        end = time.time() + PULSE_S + SETTLE_S
        while time.time() < end:
            if charging():
                cmd("stop=1")
                time.sleep(1.5)
                print(f"CHARGING after {k} reverse pulses (~{k * rev_cm:.0f} cm): "
                      f"{ina.current() * CHARGE_CURRENT_SIGN:+.3f} A", flush=True)
                return
            time.sleep(0.05)
    cmd("stop=1")
    print(f"no charge current after {budget} reverse pulses (~{budget * rev_cm:.0f} cm)", flush=True)


if __name__ == "__main__":
    main()
