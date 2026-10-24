"""Bench check of the slowest drive speeds (wheels off the ground)."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import sys
import time

from esp32_link import ESP32Link

link = ESP32Link()
link.start()
time.sleep(1.0)
link.lift(0)
for speed in [int(s) for s in sys.argv[1:]] or [15, 20]:
    print(f"forward {speed} for 2 s ...", flush=True)
    blocked = link.drive_for(speed, speed, 2.0)
    print(f"  done, blocked by: {blocked}", flush=True)
    time.sleep(3.0)
link.stop()
link.close()
