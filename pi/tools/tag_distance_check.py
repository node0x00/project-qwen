"""Print the dock tag's distance as seen by the camera, using each calibration file.
Compare against a tape-measure distance to see which calibration is accurate.

    venv/bin/python tag_distance_check.py [seconds] [tag_mm]
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import sys
import time

import cv2
import numpy as np

CALIB_FILES = ["camera_calib.yaml"]
TAG_ID = 0


def load(path: str):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K, d = fs.getNode("camera_matrix").mat(), fs.getNode("dist_coeffs").mat()
    fs.release()
    return K, d


def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 20
    tag_mm = float(sys.argv[2]) if len(sys.argv) > 2 else 80.0
    calibs = {p: load(p) for p in CALIB_FILES}
    s = tag_mm / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    dists = {p: [] for p in calibs}
    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is None or TAG_ID not in ids.ravel():
            continue
        c = corners[list(ids.ravel()).index(TAG_ID)].reshape(4, 2)
        for p, (K, d) in calibs.items():
            ok, _, tvec = cv2.solvePnP(obj, c, K, d, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                dists[p].append(float(np.ravel(tvec)[2]))   # straight-ahead distance from the lens, mm
    cap.release()
    for p, v in dists.items():
        if v:
            print(f"{p:26s} {np.median(v) / 10:6.1f} cm  ({len(v)} frames)")
        else:
            print(f"{p:26s} tag not seen")


if __name__ == "__main__":
    main()
