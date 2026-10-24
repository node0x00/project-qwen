"""Turn an offline ORB-SLAM run (map_offline) into a 2D floor map.
    venv/bin/python build_map.py maps/<name>
ORB-SLAM starts a new map every time it loses track, so the wheel odometry is the backbone:
the whole drive is dead-reckoned from the wheel counts, and every SLAM map (with enough
keyframes) is placed onto it - rotation from the camera's facing vs the odometry heading,
scale from how far the wheels went, position from the path. The dock goes where the tag
readings say it was. Writes map.png (white free, black obstacles, grey unknown, blue path,
red dock) and map.npz (grid + origin + cell size) into the same folder."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools
import csv
import sys

import cv2
import numpy as np

MM_PER_COUNT = 8.5
WHEEL_SPACING_MM = 365.0
CELL_M = 0.02
ROBOT_RADIUS_M = 0.18           # swept free space around the driven path
OBSTACLE_BAND_M = (0.03, 0.60)  # height above the floor that counts as an obstacle for the robot
RAY_MAX_M = 2.5
HALF_FOV_DEG = 35               # C920e horizontal half field of view
MIN_KF = 8

d = sys.argv[1].rstrip("/")


def quat_rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot2(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


# --- wheel odometry: dead-reckoned path (x forward at the start, CCW angles)
rows = [r for r in csv.DictReader(open(f"{d}/odom.csv")) if r["odom_l"] not in ("", "None")]
ot = np.array([float(r["time"]) for r in rows])
cl = np.array([float(r["odom_l"]) for r in rows]) * MM_PER_COUNT / 1000
cr = np.array([float(r["odom_r"]) for r in rows]) * MM_PER_COUNT / 1000
dl, dr = np.diff(cl, prepend=cl[0]), np.diff(cr, prepend=cr[0])
oth = np.cumsum((dr - dl) / (WHEEL_SPACING_MM / 1000))
ds = (dl + dr) / 2
mid = oth - (dr - dl) / (WHEEL_SPACING_MM / 1000) / 2
ox, oy = np.cumsum(ds * np.cos(mid)), np.cumsum(ds * np.sin(mid))
print(f"odometry: {np.abs(ds).sum():.1f} m driven, {np.degrees(np.abs(np.diff(oth)).sum()):.0f} deg turned")


def odo_at(t):
    return (np.stack([np.interp(t, ot, ox), np.interp(t, ot, oy)], -1),
            np.interp(t, ot, np.unwrap(oth)))


# --- SLAM maps placed on the odometry
kf_all = np.loadtxt(f"{d}/keyframes.txt", ndmin=2)
pts_all = np.loadtxt(f"{d}/points.txt", ndmin=2)
placed = []          # (KF 2D positions, KF forward unit vectors, points 2D, point heights)
scales = []
for mid_ in np.unique(kf_all[:, 1]):
    kf = kf_all[kf_all[:, 1] == mid_]
    kf = kf[np.argsort(kf[:, 0])]
    if len(kf) < MIN_KF:
        continue
    t, C = kf[:, 0], kf[:, 2:5]
    R = np.array([quat_rot(q) for q in kf[:, 5:9]])
    up = -R[:, :, 1].mean(axis=0)
    up /= np.linalg.norm(up)
    f0 = R[0][:, 2] - up * (R[0][:, 2] @ up)
    e1 = f0 / np.linalg.norm(f0)
    e2 = np.cross(up, e1)
    S = np.stack([C @ e1, C @ e2], -1)
    yaw = np.arctan2(R[:, :, 2] @ e2, R[:, :, 2] @ e1)
    Q, th = odo_at(t)
    a = np.angle(np.exp(1j * wrap(th - yaw)).mean())
    slam_len = np.linalg.norm(np.diff(S, axis=0), axis=1).sum()
    odo_len = np.linalg.norm(np.diff(Q, axis=0), axis=1).sum()
    s = odo_len / slam_len if odo_len > 0.3 and slam_len > 1e-6 else None
    placed.append(dict(id=int(mid_), t=t, S=S, Q=Q, a=a, s=s, up=up, e1=e1, e2=e2, C=C, R=R,
                       X=pts_all[pts_all[:, 0] == mid_][:, 1:4]))
    if s:
        scales.append(s)
s_med = float(np.median(scales)) if scales else 1.0
P_kf, F_kf, Xs, Hs = [], [], [], []
for m in placed:
    s = m["s"] or s_med
    Rot = rot2(m["a"])
    T = (m["Q"] - s * m["S"] @ Rot.T).mean(0)
    P_kf.append(s * m["S"] @ Rot.T + T)
    Fw = np.stack([m["R"][:, :, 2] @ m["e1"], m["R"][:, :, 2] @ m["e2"]], -1) @ Rot.T
    F_kf.append(Fw / np.linalg.norm(Fw, axis=1, keepdims=True))
    X2 = np.stack([m["X"] @ m["e1"], m["X"] @ m["e2"]], -1)
    Xs.append(s * X2 @ Rot.T + T)
    Hs.append((m["X"] @ m["up"] - np.median(m["C"] @ m["up"])) * s)
    fit = np.linalg.norm(P_kf[-1] - m["Q"], axis=1).mean()
    print(f"  map {m['id']:2d}: {len(m['t']):3d} keyframes, {m['t'][-1] - m['t'][0]:4.0f} s, "
          f"scale {s:.2f}{'' if m['s'] else ' (median)'}, fits the wheel path to {fit * 100:.0f} cm")
print(f"{len(placed)} SLAM maps placed ({len(np.unique(kf_all[:, 1])) - len(placed)} tiny ones dropped)")
P_kf, F_kf = np.vstack(P_kf), np.vstack(F_kf)
X2, H = np.vstack(Xs), np.concatenate(Hs)
kf_owner = np.concatenate([np.full(len(p["t"]), i) for i, p in enumerate(placed)])
pt_owner = np.concatenate([np.full(len(p["X"]), i) for i, p in enumerate(placed)])

below = H[(H < -0.05) & (H > -0.6)]
if len(below) > 50:
    hist, edges = np.histogram(below, bins=np.arange(-0.6, -0.05, 0.01))
    floor = edges[np.argmax(hist)] + 0.005
else:
    floor = -0.20
print(f"floor {-floor * 100:.0f} cm below the camera")
Hf = H - floor
obst = (Hf > OBSTACLE_BAND_M[0]) & (Hf < OBSTACLE_BAND_M[1])
tall = Hf >= OBSTACLE_BAND_M[1]

# --- dock from the tag readings (fresh ones only): dock pose = robot pose (odometry) o (robot in dock frame)^-1
dock = None
tag = [r for r in rows if r["tag_dist"] and abs(float(r["tag_time"]) - float(r["time"])) < 0.5]
if len(tag) >= 3:
    tt = np.array([float(r["time"]) for r in tag])
    lat = np.array([float(r["lateral"]) for r in tag]) / 100
    out = np.array([float(r["out"]) for r in tag]) / 100
    hd = np.radians([float(r["heading"]) for r in tag])
    Qt, tht = odo_at(tt)
    # dock frame: x along the wall (camera right when facing it), y out from the wall;
    # facing the wall is -y, and + heading turns toward +x
    phi = -np.pi / 2 + hd
    th_dock = tht - phi
    origin = Qt - np.stack([np.cos(th_dock) * lat - np.sin(th_dock) * out,
                            np.sin(th_dock) * lat + np.cos(th_dock) * out], -1)
    a = np.angle(np.exp(1j * th_dock).mean())
    dock = (np.median(origin, axis=0), a)
    spread = np.linalg.norm(origin - dock[0], axis=1)
    print(f"dock from {len(tag)} tag readings, spread {np.median(spread) * 100:.0f} cm (odometry drift)")

# --- grid
path = np.stack([ox, oy], -1)
allxy = np.vstack([path, X2[obst]])
lo = np.percentile(allxy, 1, axis=0) - 0.5
hi = np.percentile(allxy, 99, axis=0) + 0.5
W, Hh = (np.ceil((hi - lo) / CELL_M)).astype(int)
grid = np.full((Hh, W), 128, np.uint8)


def cell(xy):
    c = np.floor((xy - lo) / CELL_M).astype(int)
    return np.stack([c[..., 0], Hh - 1 - c[..., 1]], axis=-1)       # x right, y up in the picture


rng = np.random.default_rng(0)
for k, (Pk, Fk) in enumerate(zip(P_kf, F_kf)):            # rays: free up to each seen obstacle
    own = obst & (pt_owner == kf_owner[k])
    v = X2[own] - Pk
    dist = np.linalg.norm(v, axis=1)
    cosang = (v @ Fk) / np.maximum(dist, 1e-9)
    hits = cell(X2[own][(dist < RAY_MAX_M) & (cosang > np.cos(np.radians(HALF_FOV_DEG)))])
    if len(hits) > 300:
        hits = hits[rng.choice(len(hits), 300, replace=False)]
    p0 = tuple(int(i) for i in cell(Pk))
    for c in hits:
        cv2.line(grid, p0, (int(c[0]), int(c[1])), 255, 1)
pc = cell(path[::5])
cv2.polylines(grid, [pc.astype(np.int32)], False, 255, 2 * int(ROBOT_RADIUS_M / CELL_M))
ob_c = cell(X2[obst])
for c in ob_c:
    if 0 <= c[0] < W and 0 <= c[1] < Hh:
        grid[c[1], c[0]] = 0
np.savez(f"{d}/map.npz", grid=grid, origin=lo, cell=CELL_M)

# --- picture
img = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
for m in np.arange(np.ceil(lo[0] * 2) / 2, hi[0], 0.5):        # 50 cm grid
    x = int((m - lo[0]) / CELL_M * 2)
    cv2.line(img, (x, 0), (x, img.shape[0]), (215, 215, 215), 1)
for m in np.arange(np.ceil(lo[1] * 2) / 2, hi[1], 0.5):
    y = int((Hh - (m - lo[1]) / CELL_M) * 2)
    cv2.line(img, (0, y), (img.shape[1], y), (215, 215, 215), 1)
for c in cell(X2[tall]):
    cv2.circle(img, (int(c[0] * 2), int(c[1] * 2)), 1, (150, 150, 150), -1)
for c in ob_c:
    cv2.circle(img, (int(c[0] * 2), int(c[1] * 2)), 1, (0, 0, 0), -1)
cv2.polylines(img, [(pc * 2).astype(np.int32)], False, (220, 120, 30), 2)
cv2.circle(img, tuple(int(i) for i in pc[0] * 2), 7, (60, 170, 60), -1)      # start: green
cv2.circle(img, tuple(int(i) for i in pc[-1] * 2), 7, (220, 120, 30), -1)    # end: blue
if dock:
    o, a = dock
    wall = [o + rot2(a) @ np.array([x, 0]) for x in (-0.3, 0.3)]           # the wall with the tags
    w0, w1 = (cell(np.array(w)) * 2 for w in wall)
    cv2.line(img, tuple(map(int, w0)), tuple(map(int, w1)), (40, 40, 220), 4)
    dc = cell(o) * 2
    cv2.circle(img, (int(dc[0]), int(dc[1])), 8, (40, 40, 220), -1)
cv2.putText(img, f"grid 50 cm   {len(placed)} SLAM pieces on the wheel path   green start, blue end, red dock",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
cv2.imwrite(f"{d}/map.png", img)
print(f"map.png: {W * CELL_M:.1f} x {Hh * CELL_M:.1f} m")
