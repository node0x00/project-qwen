"""First docking attempt from a hand-placed start: robot facing the dock tag,
centred, ~20 cm in front of the dock.

  1. measure reverse/forward speed at `speed` with the tag as ruler
     (back up, then drive back to the start)
  2. measure the turn rate: spin left 1 s, then right 2 s (two durations ->
     rate and start-up lag), then left 1 s back to the start
  3. turn 180 deg (timed), then reverse in short steps until charge current
     appears (INA226) or REVERSE_CM + margin is used up

Needs slam_view.py running.   venv/bin/python dock_attempt.py [reverse_cm] [speed]
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import statistics
import sys
import time

from battery_monitor import CHARGE_CURRENT_SIGN, CHARGING_THRESHOLD_A, INA226
from calib_motion import cmd, look

STEP_S = 1.2            # reverse in steps (the ESP32 allows 1.5 s of continuous reverse)
MARGIN_CM = 5


def timed(left, right, seconds, settle=1.0):
    cmd(f"test={left},{right},{seconds}")
    time.sleep(seconds + settle)


def main():
    reverse_cm = float(sys.argv[1]) if len(sys.argv) > 1 else 20
    s = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ina = INA226()
    charging = lambda: ina.current() * CHARGE_CURRENT_SIGN > CHARGING_THRESHOLD_A
    cmd("stop=1")
    time.sleep(0.5)
    if charging():
        raise SystemExit("already charging")
    p0 = look(2.0)
    if p0 is None:
        raise SystemExit("tag not in view")
    print(f"start: tag {p0[0]:.1f} cm  x {p0[1]:+.1f}  bearing {p0[2]:+.1f}  yaw {p0[3]:+.1f}", flush=True)

    # 1. linear speed
    timed(-s, -s, STEP_S)
    p1 = look()
    timed(s, s, STEP_S)
    p2 = look()
    if not (p1 and p2):
        raise SystemExit("lost the tag during the speed check")
    rev = (p1[0] - p0[0]) / STEP_S
    fwd = (p1[0] - p2[0]) / STEP_S
    print(f"reverse {rev:.2f} cm/s, forward {fwd:.2f} cm/s (per {STEP_S}s step)", flush=True)
    if rev <= 0.3:
        raise SystemExit("the robot barely moved backward - speed too low?")

    # 2. turn rate from two durations
    timed(-s, s, 1.0)
    q1 = look()
    timed(s, -s, 2.0)
    q2 = look()
    timed(-s, s, 1.0)
    q3 = look(2.0)
    if not (q1 and q2):
        raise SystemExit("lost the tag during the turn check - turns are faster than expected")
    a1 = abs(q1[3] - p2[3])          # 1 s spin
    a2 = abs(q2[3] - q1[3])          # 2 s spin
    rate = a2 - a1                   # deg per extra second
    lag = 1.0 - a1 / rate if rate > 0 else 0.0
    print(f"turn: 1s -> {a1:.1f} deg, 2s -> {a2:.1f} deg  => {rate:.1f} deg/s, start-up lag {lag:.2f}s", flush=True)
    if q3:
        print(f"back at start: bearing {q3[2]:+.1f}  yaw {q3[3]:+.1f}  dist {q3[0]:.1f}", flush=True)
    if not (3 < rate < 120):
        raise SystemExit("turn rate looks wrong - not turning 180")

    # 3. turn 180 and reverse onto the pads
    t180 = 180.0 / rate + max(0.0, lag)
    print(f"turning 180: spin left {t180:.1f}s", flush=True)
    timed(-s, s, t180, settle=1.5)
    budget = reverse_cm + MARGIN_CM
    moved = 0.0
    print(f"reversing up to {budget:.0f} cm in {STEP_S}s steps (~{rev * STEP_S:.1f} cm each)", flush=True)
    while moved < budget:
        cmd(f"test={-s},{-s},{STEP_S}")
        end = time.time() + STEP_S
        while time.time() < end:
            if charging():
                cmd("stop=1")
                print(f"CHARGING after ~{moved + rev * (STEP_S - (end - time.time())):.0f} cm", flush=True)
                time.sleep(2)
                print(f"charge current {ina.current() * CHARGE_CURRENT_SIGN:+.3f} A", flush=True)
                return
            time.sleep(0.05)
        moved += rev * STEP_S
        time.sleep(0.4)
        if charging():
            print(f"CHARGING after ~{moved:.0f} cm", flush=True)
            return
    cmd("stop=1")
    print(f"no charge current after ~{moved:.0f} cm - stopped", flush=True)


if __name__ == "__main__":
    main()
