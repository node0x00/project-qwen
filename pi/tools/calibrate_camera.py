"""Camera calibration from the printed 6x9 checkerboard (11 mm squares).

Hold the board in front of the camera and move it around; a short beep means a
view was captured, a double beep means calibration is done. Saves the result to
camera_calib.yaml (used by AprilTag pose and ORB-SLAM).

    venv/bin/python calibrate_camera.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import time

import cv2
import numpy as np

import audio_out

PATTERN = (6, 9)              # inner corners
SQUARE_MM = 11.0
WIDTH, HEIGHT = 640, 480
TARGET_VIEWS = 25
TIMEOUT_S = 6 * 60
MIN_GAP_S = 0.8
MIN_BOARD_PX = 120          # board must fill enough of the frame (hold it ~20-40 cm away)
OUT_FILE = "camera_calib.yaml"
VIEWS_FILE = "calib_views.npz"


def beep(times: int = 1) -> None:
    sr = 22050
    t = np.arange(int(sr * 0.12)) / sr
    tone = 0.3 * np.sin(2 * np.pi * 1000 * t)
    gap = np.zeros(int(sr * 0.08))
    audio_out.play(np.concatenate([np.concatenate([tone, gap])] * times).astype(np.float32), sr)


def pose_key(corners: np.ndarray) -> np.ndarray:
    """Board centre (px), apparent size and tilt - used to skip near-duplicate views."""
    pts = corners.reshape(-1, 2)
    centre = pts.mean(axis=0)
    size = np.sqrt(cv2.contourArea(cv2.convexHull(pts)))
    top = pts[PATTERN[0] - 1] - pts[0]
    side = pts[-PATTERN[0]] - pts[0]
    return np.array([centre[0], centre[1], size, np.linalg.norm(top) / np.linalg.norm(side) * 100,
                     np.degrees(np.arctan2(top[1], top[0]))])


def is_new(key: np.ndarray, keys: list[np.ndarray]) -> bool:
    for k in keys:
        if (np.hypot(*(key[:2] - k[:2])) < 60 and abs(key[2] - k[2]) < 0.12 * k[2]
                and abs(key[3] - k[3]) < 8 and abs(key[4] - k[4]) < 10):
            return False
    return True


def main() -> None:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("camera read failed")
    h, w = frame.shape[:2]
    print(f"camera {w}x{h}; waiting for the board...", flush=True)

    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM
    obj_pts, img_pts, keys = [], [], []
    t0 = last = time.time()
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    while len(img_pts) < TARGET_VIEWS and time.time() - t0 < TIMEOUT_S:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
        if not found or time.time() - last < MIN_GAP_S:
            continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)
        key = pose_key(corners)
        if key[2] < MIN_BOARD_PX or not is_new(key, keys):
            continue
        keys.append(key)
        obj_pts.append(objp)
        img_pts.append(corners)
        last = time.time()
        print(f"view {len(img_pts):2d}/{TARGET_VIEWS}  centre ({key[0]:.0f},{key[1]:.0f})  "
              f"size {key[2]:.0f}px  roll {key[4]:+.0f} deg  shape {key[3]:.0f}", flush=True)
        beep()
    cap.release()

    np.savez(VIEWS_FILE, img_pts=np.array(img_pts), w=w, h=h)   # so a failed fit can be re-run
    if len(img_pts) < 10:
        raise SystemExit(f"only {len(img_pts)} views - not enough to calibrate")
    fit_and_save(obj_pts, img_pts, w, h)
    beep(2)


def fit_and_save(obj_pts: list, img_pts: list, w: int, h: int) -> None:
    # Webcam pixels are square and tangential distortion is negligible - fixing those
    # keeps the fit from chasing noise. Then drop views that fit badly and refit.
    flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    K0 = np.array([[600.0, 0, w / 2], [0, 600.0, h / 2], [0, 0, 1]])
    for round_ in range(3):
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (w, h), K0.copy(), None,
                                                         flags=flags)
        errs = [float(np.sqrt(np.mean(np.sum(
                    (ip.reshape(-1, 2) - cv2.projectPoints(op, r, t, K, dist)[0].reshape(-1, 2)) ** 2, axis=1))))
                for op, ip, r, t in zip(obj_pts, img_pts, rvecs, tvecs)]
        keep = [i for i, e in enumerate(errs) if e <= max(2 * np.median(errs), 0.5)]
        if round_ == 2 or len(keep) == len(errs) or len(keep) < 10:
            break
        print(f"dropping {len(errs) - len(keep)} poorly fitting views", flush=True)
        obj_pts = [obj_pts[i] for i in keep]
        img_pts = [img_pts[i] for i in keep]
    fs = cv2.FileStorage(OUT_FILE, cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", w)
    fs.write("image_height", h)
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", dist)
    fs.write("rms_px", rms)
    fs.release()
    hfov = np.degrees(2 * np.arctan(w / (2 * K[0, 0])))
    print(f"\nDONE  {len(img_pts)} views  RMS {rms:.3f} px  "
          f"fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}  hfov {hfov:.1f} deg")
    print("dist", np.round(dist.ravel(), 4).tolist(), f"\nsaved {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
