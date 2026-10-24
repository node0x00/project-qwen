"""Save the "pre-push" picture (robot placed ~2 cm out from the pads, facing away) and
check how well image scale tells distance against the taught strip."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import json
import time

import cv2
import numpy as np

from battery_monitor import CHARGE_CURRENT_SIGN, INA226
from dock_view import RATIO, _bf, features, grab

ina = INA226()
print(f"charge current now {ina.current() * CHARGE_CURRENT_SIGN:+.3f} A (should be ~0: off the pads)")
frames = []
for _ in range(5):
    frames.append(grab())
    time.sleep(0.25)
pre = frames[-1]
cv2.imwrite("dock_ref/prepush.jpg", pre, [cv2.IMWRITE_JPEG_QUALITY, 92])
meta = json.load(open("dock_ref/refs.json"))
meta["prepush"] = {"file": "prepush.jpg", "cm": 2.0, "taught": time.strftime("%Y-%m-%d %H:%M")}
json.dump(meta, open("dock_ref/refs.json", "w"), indent=1)


def scale(a, b):
    ka, da = features(a)
    kb, db = features(b)
    good = [m for m, n in (p for p in _bf.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < RATIO * n.distance]
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    return float(np.hypot(M[0, 0], M[1, 0])), int(inl.sum())


print("noise, same spot:", [round(scale(frames[i], pre)[0], 4) for i in range(4)])
for r in meta["refs"][:12]:
    img = cv2.imread("dock_ref/" + r["file"], cv2.IMREAD_GRAYSCALE)
    sc, n = scale(img, pre)
    print(f"strip {r['cm']:5.1f} cm vs prepush: scale {sc:.4f} ({n} inliers)")
