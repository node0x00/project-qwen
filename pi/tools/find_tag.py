"""Slowly spin in place in short steps until the dock AprilTag is in view.
Needs slam_view.py running (mono_live writes /dev/shm/slam_tag.txt).

    venv/bin/python find_tag.py [--speed 8] [--dir left|right] [--step 0.5] [--max-steps 60]
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import argparse
import time

from esp32_link import ESP32Link

TAG_FILE = "/dev/shm/slam_tag.txt"
CAPTION = "/dev/shm/slam_view.jpg.txt"


def caption(text: str) -> None:
    with open(CAPTION, "w") as f:
        f.write(text)
    print(time.strftime("%H:%M:%S"), text, flush=True)


def read_tag(max_age: float = 0.5):
    """(dist_cm, x_cm, bearing_deg, yaw_deg) if the tag was seen in the last max_age s."""
    try:
        parts = open(TAG_FILE).read().split()
    except OSError:
        return None
    if len(parts) < 6 or time.time() - float(parts[0]) > max_age:
        return None
    return tuple(float(p) for p in parts[2:6])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=int, default=8)
    ap.add_argument("--dir", choices=["left", "right"], default="left")
    ap.add_argument("--step", type=float, default=0.5, help="seconds of spin per step")
    ap.add_argument("--settle", type=float, default=0.8, help="pause after each step to look")
    ap.add_argument("--max-steps", type=int, default=60)
    a = ap.parse_args()
    l, r = (-a.speed, a.speed) if a.dir == "left" else (a.speed, -a.speed)

    link = ESP32Link()
    link.start()
    time.sleep(1.0)
    link.lift(0)
    found = None
    try:
        for i in range(1, a.max_steps + 1):
            found = read_tag()
            if found:
                break
            caption(f"search: spin {a.dir} step {i}/{a.max_steps} @ {a.speed}")
            link.drive_for(l, r, a.step)
            time.sleep(a.settle)
            if not link.connected:
                caption("search: ESP32 link lost - stopping")
                break
        if found is None:
            found = read_tag()
    except BaseException as e:
        caption(f"search: aborted ({e!r})")
        raise
    finally:
        link.stop()
        link.close()
    if found:
        d, x, b, y = found
        caption(f"search: TAG FOUND  {d:.0f} cm  x {x:+.0f} cm  bearing {b:+.0f} deg  yaw {y:+.0f} deg")
    else:
        caption("search: tag not found")


if __name__ == "__main__":
    main()
