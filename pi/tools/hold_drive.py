"""Hold one motor command until killed (or max_s), then STOP. Bench use, wheels off the ground.
    venv/bin/python hold_drive.py <left> <right> [max_s]"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import signal
import sys
import time

from esp32_link import ESP32Link

left, right = int(sys.argv[1]), int(sys.argv[2])
max_s = float(sys.argv[3]) if len(sys.argv) > 3 else 600
stop = False


def on_signal(*_):
    global stop
    stop = True


signal.signal(signal.SIGTERM, on_signal)
signal.signal(signal.SIGINT, on_signal)
link = ESP32Link()
link.start()
time.sleep(1.0)
t0 = time.time()
link.drive(left, right)
print(f"holding M {left} {right}", flush=True)
try:
    while not stop and time.time() - t0 < max_s:
        time.sleep(0.1)
finally:
    link.stop()
    time.sleep(0.3)
    link.close()
    print(f"stopped after {time.time() - t0:.0f} s", flush=True)
