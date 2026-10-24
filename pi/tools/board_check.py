"""Compare single-tag (both IPPE answers) and two-tag board poses on the current frame."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import math

import cv2
import numpy as np

from dock_view import grab

K = np.array([[645.0, 0, 308.087], [0, 645.0, 239.378], [0, 0, 1]])
D = np.array([-0.021926, -0.174402, 0, 0])
h = 79.0 / 2
obj = np.float32([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])


def pose(R, t):
    p = -R.T @ t
    f = R.T @ np.array([0, 0, 1.0])
    return round(float(p[0]) / 10, 1), round(float(p[2]) / 10, 1), round(math.degrees(math.atan2(f[0], -f[2])), 1)


img = grab()
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
cs, ids, _ = det.detectMarkers(img)
ids = ids.ravel().tolist()
c0, c1 = cs[ids.index(0)].reshape(4, 2), cs[ids.index(1)].reshape(4, 2)
print("ID0 centre x", c0.mean(0)[0].round(), " ID1 centre x", c1.mean(0)[0].round())
n, rv, tv, err = cv2.solvePnPGeneric(obj, c0, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
for k in range(n):
    R, _ = cv2.Rodrigues(rv[k])
    print(f"single tag answer {k}: (lat, out, heading) {pose(R, tv[k].ravel())}  err {float(err[k]):.3f}")
for spacing in (160, -160):
    obj8 = np.vstack([obj, obj + np.float32([spacing, 0, 0])])
    ok, r, t = cv2.solvePnP(obj8, np.vstack([c0, c1]), K, D, flags=cv2.SOLVEPNP_SQPNP)
    R, _ = cv2.Rodrigues(r)
    proj, _ = cv2.projectPoints(obj8, r, t, K, D)
    e = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - np.vstack([c0, c1])) ** 2, 1))))
    print(f"board, ID1 at {spacing:+d} mm: {pose(R, t.ravel())}  reprojection {e:.2f} px")
