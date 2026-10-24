"""Slow scripted moves for the ORB-SLAM ground test. Each step: 'f<sec>', 'b<sec>',
'l<sec>' (spin left), 'r<sec>' (spin right), 'p<sec>' (pause). Speed via --speed.
Captions go to the SLAM live view.

    venv/bin/python drive_seq.py --speed 20 f2.5 p2 b1.2
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import argparse
import time

from esp32_link import ESP32Link

CAPTION = "/dev/shm/slam_view.jpg.txt"


def caption(text: str) -> None:
    with open(CAPTION, "w") as f:
        f.write(text)
    print(time.strftime("%H:%M:%S"), text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=int, default=20)
    ap.add_argument("steps", nargs="+")
    a = ap.parse_args()
    s = a.speed
    moves = {"f": (s, s, "forward"), "b": (-s, -s, "reverse"), "l": (-s, s, "spin left"),
             "r": (s, -s, "spin right"), "p": (0, 0, "pause")}
    link = ESP32Link()
    link.start()
    time.sleep(1.0)
    link.lift(0)
    try:
        for step in a.steps:
            left, right, name = moves[step[0]]
            sec = float(step[1:])
            caption(f"drive: {name} {sec:.1f}s @ {s}" if left or right else f"drive: pause {sec:.1f}s")
            if left or right:
                blocked = link.drive_for(left, right, sec)
                if blocked:
                    caption(f"drive: BLOCKED by {blocked} - stopping sequence")
                    break
            else:
                time.sleep(sec)
            if not link.connected:
                caption("drive: ESP32 link lost - stopping")
                break
    except BaseException as e:            # make an early end visible in the log
        caption(f"drive: aborted ({e!r})")
        raise
    finally:
        link.stop()
        caption("drive: stopped")
        link.close()


if __name__ == "__main__":
    main()
