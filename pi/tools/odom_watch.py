"""Print wheel odometry counts whenever they change, for N seconds (turn a wheel by hand)."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import sys
import time

from esp32_link import ESP32Link

link = ESP32Link()
link.start()
time.sleep(1.0)
link.reset_odom()
time.sleep(0.3)
print("watching odometry - turn the wheel by hand", flush=True)
last = link.odom()
t0 = time.time()
while time.time() - t0 < float(sys.argv[1] if len(sys.argv) > 1 else 30):
    time.sleep(0.1)
    now = link.odom()
    if now != last:
        print(f"{time.time() - t0:5.1f}s  left {now[0]:+6d}   right {now[1]:+6d}", flush=True)
        last = now
print("final", link.odom(), flush=True)
link.close()
