"""Record a mapping drive for offline ORB-SLAM: every clean camera frame (~10 fps, from
mono_live's /dev/shm/cam.jpg) plus wheel counts and the dock-tag pose ~10x a second.
Drive with WASD on the control page while it runs (slow arcs - W+A / W+D - beat spins on the spot).
    venv/bin/python map_recorder.py [name]      (Ctrl-C / SIGTERM to stop)
Output: maps/<name>/frames/<unix_ns>.jpg and maps/<name>/odom.csv"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import csv
import json
import os
import shutil
import signal
import sys
import time
import urllib.request

KEY = next(l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("QWEN_STREAM_KEY="))
CAM = "/dev/shm/cam.jpg"
TAG = "/dev/shm/slam_tag.txt"

name = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d_%H%M")
out = os.path.join("maps", name)
os.makedirs(os.path.join(out, "frames"), exist_ok=True)
stop = False
signal.signal(signal.SIGTERM, lambda *_: globals().update(stop=True))


def status():
    return json.load(urllib.request.urlopen(f"http://localhost:8080/status?key={KEY}", timeout=2))


f = open(os.path.join(out, "odom.csv"), "a", newline="")
w = csv.writer(f)
w.writerow(["time", "odom_l", "odom_r", "tag_time", "tag_dist", "tag_bearing", "lateral", "out", "heading"])
last_mtime, last_log, frames = 0, 0.0, 0
print(f"recording to {out} - drive with WASD, slow arcs", flush=True)
try:
    while not stop:
        try:
            m = os.stat(CAM).st_mtime_ns
        except OSError:
            m = last_mtime
        if m != last_mtime:
            last_mtime = m
            try:
                shutil.copyfile(CAM, os.path.join(out, "frames", f"{m}.jpg"))   # rename-written, so whole
                frames += 1
            except OSError:
                pass
        now = time.time()
        if now - last_log >= 0.1:
            last_log = now
            try:
                odo = status().get("odom") or [None, None]
            except Exception:
                odo = [None, None]
            tag = [""] * 6
            try:
                p = open(TAG).read().split()
                if len(p) >= 9 and p[1] != "none":
                    tag = [p[0], p[2], p[4], p[6], p[7], p[8]]   # time dist bearing lateral out heading
            except OSError:
                pass
            w.writerow([f"{now:.3f}", odo[0], odo[1], *tag])
            if frames and frames % 100 == 0:
                f.flush()
        time.sleep(0.01)
except KeyboardInterrupt:
    pass
f.close()
print(f"stopped: {frames} frames in {out}", flush=True)
