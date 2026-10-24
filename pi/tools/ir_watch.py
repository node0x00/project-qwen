"""Print IR sensor changes for N seconds (bench check before driving)."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import sys
import time

from esp32_link import ESP32Link

link = ESP32Link()
link.start()
time.sleep(1.0)
print(f"link {link.connected}  lift {link.state.lift_pos}  IR {link.state.ir}", flush=True)
last = dict(link.state.ir)
t0 = time.time()
seen = {k: False for k in last}
while time.time() - t0 < float(sys.argv[1] if len(sys.argv) > 1 else 5):
    time.sleep(0.05)
    ir = dict(link.state.ir)
    if ir != last:
        changed = [f"{k}={'OBSTACLE' if v else 'clear'}" for k, v in ir.items() if last.get(k) != v]
        print(f"{time.time() - t0:5.1f}s  " + "  ".join(changed), flush=True)
        for k, v in ir.items():
            seen[k] |= v
        last = ir
print("triggered at least once:", {k: v for k, v in seen.items() if k != "head"}, flush=True)
link.close()
