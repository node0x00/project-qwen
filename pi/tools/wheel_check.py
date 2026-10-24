"""One wheel, forward then backward (bench check of the direction signal).
    venv/bin/python wheel_check.py right|left [speed]"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import sys
import time

from esp32_link import ESP32Link

wheel = sys.argv[1]
s = int(sys.argv[2]) if len(sys.argv) > 2 else 1
link = ESP32Link()
link.start()
time.sleep(1.0)
try:
    for label, cmd, secs in (("FORWARD", s, 3.0), ("pause", 0, 2.0), ("BACKWARD", -s, 1.2)):
        print(time.strftime("%H:%M:%S"), f"{wheel} wheel {label}", flush=True)
        if cmd:
            link.drive_for(cmd, 0, secs) if wheel == "left" else link.drive_for(0, cmd, secs)
        else:
            time.sleep(secs)
finally:
    link.stop()
    link.close()
print("done", flush=True)
