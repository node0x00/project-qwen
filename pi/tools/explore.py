"""Robot-vacuum style exploring until the dock AprilTag is seen.

Drive forward slowly. When the front IR sees something (the ESP32 also refuses
forward motion on its own): stop, back up a little, turn a little, and try
again - repeat until clear. Stops as soon as the tag is in view.
Needs slam_view.py running (mono_live writes /dev/shm/slam_tag.txt).

    venv/bin/python explore.py [--speed 8] [--minutes 4]
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import argparse
import random
import time

from esp32_link import ESP32Link
from find_tag import caption, read_tag

BACKUP_S = 0.5      # a little - there are no rear sensors
TURN_S = 0.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=4)
    a = ap.parse_args()
    s = a.speed

    link = ESP32Link()
    link.start()
    time.sleep(1.0)
    link.lift(0)
    deadline = time.time() + a.minutes * 60
    turn_dir = random.choice([-1, 1])
    found = None
    try:
        while time.time() < deadline and not found:
            if not link.connected:
                caption("explore: ESP32 link lost - stopping")
                break
            if link.obstacle_ahead():
                # back up a little and turn, keep turning the same way until clear
                link.stop()
                caption(f"explore: obstacle ({link.state.last_blocked or 'IR'}) - back up + turn "
                        f"{'left' if turn_dir < 0 else 'right'}")
                time.sleep(0.3)
                link.drive_for(-s, -s, BACKUP_S)
                time.sleep(0.3)
                link.drive_for(turn_dir * s, -turn_dir * s, TURN_S)
                time.sleep(0.6)            # let the IR reading update and the camera look
                found = read_tag()
                continue
            caption(f"explore: forward @ {s}")
            link.drive(s, s)
            t_move = time.time()
            while time.time() < deadline:
                time.sleep(0.1)
                found = read_tag()
                if found or link.obstacle_ahead() or not link.connected:
                    break
                if time.time() - t_move > 30:   # long straight run: pick a new direction
                    turn_dir = random.choice([-1, 1])
                    link.stop()
                    time.sleep(0.3)
                    link.drive_for(turn_dir * s, -turn_dir * s, TURN_S * 2)
                    time.sleep(0.5)
                    break
            link.stop()
            if not found and not link.obstacle_ahead():
                turn_dir = random.choice([-1, 1])   # new random turn side for the next obstacle
    except BaseException as e:
        caption(f"explore: aborted ({e!r})")
        raise
    finally:
        link.stop()
        link.close()
    if found:
        d, x, b, y = found
        caption(f"explore: TAG FOUND  {d:.0f} cm  x {x:+.0f} cm  bearing {b:+.0f} deg  yaw {y:+.0f} deg")
    else:
        caption("explore: time up, tag not found")


if __name__ == "__main__":
    main()
