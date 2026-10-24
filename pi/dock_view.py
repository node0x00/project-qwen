"""
Project Qwen - "docked view" matching for the blind part of docking.

When the robot backs onto the pads its camera faces away from the dock, into
the room. We teach it once what that view looks like: a strip of reference
pictures taken from the docked spot and then every forward pulse out to ~20 cm.
While reversing onto the dock later, the live picture is matched against the
strip with ORB features:

  - heading error: the room is far away, so a turned robot sees the matched
    features shifted sideways - median shift / focal length = angle
  - progress: the reference with the most good matches is roughly where we are

  - "pre-push" spot: a picture taken where a final push lands on the pads
    (~2 cm out). The scene's apparent size vs that picture tells distance to
    within a few mm near the dock (moving 1 cm changes it ~0.4%, noise ~0.01%).

References live in dock_ref/ (images + refs.json).
"""

import json
import os
import time

import cv2
import numpy as np

CAM_FRAME = "/dev/shm/cam.jpg"          # clean frames from mono_live (~10 Hz)
REF_DIR = "dock_ref"
FX = 645.0                              # camera_calib.yaml
N_FEATURES = 1000
RATIO = 0.75                            # Lowe ratio test
MIN_MATCHES = 25                        # fewer than this = "can't tell"

_orb = cv2.ORB_create(nfeatures=N_FEATURES)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def grab(max_age: float = 0.5):
    """Latest clean camera frame (grayscale), or None if stale/missing."""
    try:
        if time.time() - os.path.getmtime(CAM_FRAME) > max_age:
            return None
        img = cv2.imread(CAM_FRAME, cv2.IMREAD_GRAYSCALE)
    except OSError:
        return None
    return img


def features(gray):
    return _orb.detectAndCompute(gray, None)


def match(live, ref):
    """(n_good, heading_deg) between two (keypoints, descriptors) sets.
    heading_deg > 0: the robot points to the RIGHT of where it pointed for ref."""
    kl, dl = live
    kr, dr = ref
    if dl is None or dr is None or len(kl) < 10 or len(kr) < 10:
        return 0, None
    good = [m for m, n in (p for p in _bf.knnMatch(dl, dr, k=2) if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < MIN_MATCHES:
        return len(good), None
    dx = np.array([kl[m.queryIdx].pt[0] - kr[m.trainIdx].pt[0] for m in good])
    dy = np.array([kl[m.queryIdx].pt[1] - kr[m.trainIdx].pt[1] for m in good])
    # keep the consistent majority (drops wrong matches and very near objects)
    med_x, med_y = np.median(dx), np.median(dy)
    keep = (np.abs(dx - med_x) < 25) & (np.abs(dy - med_y) < 25)
    if keep.sum() < MIN_MATCHES:
        return int(keep.sum()), None
    shift = float(np.median(dx[keep]))
    # robot turned right -> the scene slides left in the picture (negative shift)
    return int(keep.sum()), float(np.degrees(np.arctan(-shift / FX)))


class DockView:
    def __init__(self, ref_dir: str = REF_DIR):
        self.ref_dir = ref_dir
        self.refs = []                  # [(cm_from_dock, (kp, des))]
        self.prepush = None             # (kp, des) of the pre-push picture
        self.load()

    @property
    def taught(self) -> bool:
        return len(self.refs) > 0

    def load(self):
        meta = os.path.join(self.ref_dir, "refs.json")
        self.refs = []
        if not os.path.exists(meta):
            return
        data = json.load(open(meta))
        for r in data["refs"]:
            img = cv2.imread(os.path.join(self.ref_dir, r["file"]), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.refs.append((r["cm"], features(img)))
        self.prepush = None
        if "prepush" in data:
            img = cv2.imread(os.path.join(self.ref_dir, data["prepush"]["file"]), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.prepush = features(img)

    def save(self, shots):
        """shots: [(cm_from_dock, gray image)] - replaces the old references."""
        os.makedirs(self.ref_dir, exist_ok=True)
        for f in os.listdir(self.ref_dir):
            os.remove(os.path.join(self.ref_dir, f))
        meta = []
        for i, (cm, img) in enumerate(shots):
            name = f"ref_{i:02d}.jpg"
            cv2.imwrite(os.path.join(self.ref_dir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            meta.append({"file": name, "cm": round(cm, 1)})
        json.dump({"taught": time.strftime("%Y-%m-%d %H:%M"), "refs": meta},  # (drops any old pre-push)
                  open(os.path.join(self.ref_dir, "refs.json"), "w"), indent=1)
        self.load()

    def save_prepush(self, gray):
        """Picture from the spot a final push lands from (~2 cm out, lined up)."""
        os.makedirs(self.ref_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.ref_dir, "prepush.jpg"), gray, [cv2.IMWRITE_JPEG_QUALITY, 92])
        meta_path = os.path.join(self.ref_dir, "refs.json")
        data = json.load(open(meta_path)) if os.path.exists(meta_path) else {"refs": []}
        data["prepush"] = {"file": "prepush.jpg", "cm": 2.0, "taught": time.strftime("%Y-%m-%d %H:%M")}
        json.dump(data, open(meta_path, "w"), indent=1)
        self.load()

    def locate(self, gray):
        """(cm_from_dock, heading_deg, n_good), or None.
        Progress comes from the best-matching reference; the heading is always measured
        against the DOCKED picture (the robot can veer a little while teaching, so the
        later pictures aren't all pointing exactly the docked way)."""
        live = features(gray)
        best = None
        docked_heading = None
        for i, (cm, ref) in enumerate(self.refs):
            n, heading = match(live, ref)
            if i == 0:
                docked_heading = heading
            if heading is not None and (best is None or n > best[2]):
                best = (cm, heading, n)
        if best is None:
            return None
        return best[0], docked_heading if docked_heading is not None else best[1], best[2]


def scale_to(live, ref):
    """How big the scene looks in `live` relative to `ref` (similarity fit, RANSAC):
    < 1 = live is farther out from the dock than ref. (scale, inliers) or None."""
    kl, dl = live
    kr, dr = ref
    if dl is None or dr is None:
        return None
    good = [m for m, n in (p for p in _bf.knnMatch(dl, dr, k=2) if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < MIN_MATCHES:
        return None
    pl = np.float32([kl[m.queryIdx].pt for m in good])
    pr = np.float32([kr[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(pl, pr, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or inl is None or inl.sum() < MIN_MATCHES:
        return None
    # M maps live -> ref. Farther out from the dock = closer to the room = the scene looks
    # BIGGER in live, so M shrinks it: scale < 1 means live is farther out than ref.
    return float(np.hypot(M[0, 0], M[1, 0])), int(inl.sum())
