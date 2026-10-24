"""
Project Qwen - lost-tag search: find the dock when a full turn on the spot shows no tags.

Docking starts with a 360 deg stop-and-look scan for the two AprilTags above the dock. If
that finds nothing (the robot was carried to another room, or the tags are behind
furniture), RobotController calls LostTagSearch.run():

  1. The tag tracker restarts with monocular ORB-SLAM on (dock_camera.set_slam). Tag
     detection keeps running in the same program, so a tag seen at any moment ends the search.
  2. The robot explores in legs of about LEG_MM, driving slowly, turning at <= ~9 deg/s
     (faster turns move features too far between frames and ORB-SLAM loses track) and
     stopping for the front IR sensors.
  3. Where it has been is kept in a coarse occupancy-style grid of visited and blocked
     cells. The next heading is the one whose cell LEG_MM ahead has been visited least, so
     the robot spreads out across the room instead of circling the same spot.
     Position comes from wheel odometry for distance; while ORB-SLAM is tracking, its
     camera heading replaces the wheel heading, which drifts with every slip on the floor.
  4. After every leg it repeats the 360 deg tag scan. Tags found -> normal docking resumes.
  5. After TIME_BUDGET_S, or if it gets boxed in, it gives up; the caller reports
     "no tags found after searching the room", and the power manager tells the owner.

    search = LostTagSearch(robot, dock_camera)
    robot.lost_tag_search = search
"""

import math
import threading
import time

from robot_control import MM_PER_COUNT, WHEEL_SPACING_MM, read_tag

POSE_FILE = "/dev/shm/slam_pose.txt"
TRACKING_OK = 2                # ORB_SLAM3::Tracking::OK
TIME_BUDGET_S = 240.0
LEG_MM = 900.0                 # drive this far between tag scans
LEG_STEP_MM = 150.0            # ...in short steps, checking for tags and obstacles between them
DRIVE_MM_S = 60.0
TURN_MM_S = 30.0               # wheel speed while turning: ~9 deg/s, slow enough for ORB-SLAM
CELL_MM = 500.0                # visited/blocked grid resolution
HEADINGS = 8                   # candidate directions, 45 deg apart
INIT_TIMEOUT_S = 20.0          # ORB-SLAM needs some forward motion to initialise
BACK_OFF_MM = 120.0            # after an obstacle


def _quat_yaw(qx, qy, qz, qw):
    """Heading on the floor of a camera-to-world rotation: the camera's forward axis (+z)
    projected onto the map's x/z plane (the camera sits level, so that's the floor)."""
    fx = 2 * (qx * qz + qy * qw)
    fz = 1 - 2 * (qx * qx + qy * qy)
    return math.atan2(fx, fz)


def read_slam_pose(max_age: float = 0.5):
    """(state, map_id, yaw) from the tag tracker, or None if stale/missing."""
    try:
        p = open(POSE_FILE).read().split()
    except OSError:
        return None
    if len(p) < 10 or time.time() - float(p[0]) > max_age:
        return None
    state, map_id = int(p[1]), int(p[2])
    return state, map_id, _quat_yaw(*map(float, p[6:10]))


class _Odometry:
    """Dead reckoning from the wheel counts (x forward at the start, CCW angles). While ORB-SLAM
    tracks, its heading (offset-corrected per SLAM map) replaces the wheel heading."""

    def __init__(self, link):
        self.link = link
        self.x = self.y = self.th = 0.0
        self._last = None
        self._slam_offset = {}         # map id -> (odometry heading - SLAM yaw) when first seen
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()

    def pose(self):
        with self._lock:
            return self.x, self.y, self.th

    def _run(self):
        while not self._stop.wait(0.05):
            l, r = self.link.odom()
            if self._last is None:
                self._last = (l, r)
                continue
            dl = (l - self._last[0]) * MM_PER_COUNT
            dr = (r - self._last[1]) * MM_PER_COUNT
            self._last = (l, r)
            dth = (dr - dl) / WHEEL_SPACING_MM
            ds = (dl + dr) / 2
            with self._lock:
                mid = self.th + dth / 2
                self.x += ds * math.cos(mid)
                self.y += ds * math.sin(mid)
                self.th += dth
                slam = read_slam_pose()
                if slam and slam[0] == TRACKING_OK:
                    # SLAM yaw grows clockwise seen from above (camera x right, z forward)
                    yaw = -slam[2]
                    off = self._slam_offset.setdefault(slam[1], self.th - yaw)
                    self.th = yaw + off


class LostTagSearch:
    def __init__(self, robot, dock_camera, time_budget_s: float = TIME_BUDGET_S):
        self.robot = robot
        self.dock_camera = dock_camera
        self.time_budget_s = time_budget_s
        self.visits = {}               # cell -> times visited
        self.blocked = set()           # cells where the front IR stopped us

    # ------------------------------------------------------------------ grid

    @staticmethod
    def _cell(x, y):
        return (round(x / CELL_MM), round(y / CELL_MM))

    def _mark(self, x, y):
        c = self._cell(x, y)
        self.visits[c] = self.visits.get(c, 0) + 1

    def _pick_heading(self, x, y, th):
        """Least-visited, unblocked direction; ties go to the smallest turn."""
        best = None
        for k in range(HEADINGS):
            a = th + k * 2 * math.pi / HEADINGS
            c = self._cell(x + LEG_MM * math.cos(a), y + LEG_MM * math.sin(a))
            if c in self.blocked:
                continue
            turn = abs(math.atan2(math.sin(a - th), math.cos(a - th)))
            score = (self.visits.get(c, 0), turn)
            if best is None or score < best[0]:
                best = (score, a)
        return None if best is None else best[1]

    # ------------------------------------------------------------------ motion

    def _tag_seen(self):
        return read_tag(0.5) is not None

    def _turn_to(self, odo, target):
        _, _, th = odo.pose()
        d = math.degrees(math.atan2(math.sin(target - th), math.cos(target - th)))
        if abs(d) < 8:
            return True
        return self.robot._odo_turn(abs(d), d > 0, speed=TURN_MM_S) is not None

    def _leg(self, odo):
        """Drive up to LEG_MM forward in short steps. Returns 'tag', 'blocked', 'done' or None (cancelled)."""
        driven = 0.0
        while driven < LEG_MM:
            if self._tag_seen():
                return "tag"
            if self.robot.link.obstacle_ahead():
                return "blocked"
            got = self.robot._odo_move(LEG_STEP_MM, LEG_STEP_MM, DRIVE_MM_S,
                                       stop_on=self.robot.link.obstacle_ahead, timeout=10)
            if got is None:
                return None
            x, y, _ = odo.pose()
            self._mark(x, y)
            driven += got
            if got < LEG_STEP_MM * 0.5:
                return "blocked"
        return "done"

    def _wait_for_slam(self, odo):
        """ORB-SLAM can't initialise from rotation alone: creep forward until it tracks."""
        end = time.time() + INIT_TIMEOUT_S
        while time.time() < end:
            p = read_slam_pose()
            if p and p[0] == TRACKING_OK:
                return True
            if self.robot.link.obstacle_ahead():
                return False
            if self.robot._odo_move(60, 60, 30, stop_on=self.robot.link.obstacle_ahead, timeout=5) is None:
                return False
        return False

    # ------------------------------------------------------------------ main

    def run(self) -> bool:
        """True if the tags came into view (the robot is left facing them)."""
        r = self.robot
        r._caption("search: no tags in a full turn - mapping the room to look for the dock")
        self.dock_camera.set_slam(True)
        odo = _Odometry(r.link)
        odo.start()
        start = time.time()
        try:
            if not self._wait_for_slam(odo):
                r._caption("search: ORB-SLAM hasn't started tracking - searching on wheel odometry")
            x, y, th = odo.pose()
            self._mark(x, y)
            legs = 0
            while time.time() - start < self.time_budget_s:
                if r._auto_cancel.is_set():
                    return False
                x, y, th = odo.pose()
                heading = self._pick_heading(x, y, th)
                if heading is None:
                    r._caption("search: boxed in - giving up")
                    return False
                if not self._turn_to(odo, heading):
                    return False
                if self._tag_seen():
                    return self._found(start, legs)
                result = self._leg(odo)
                if result is None:
                    return False
                if result == "tag":
                    return self._found(start, legs)
                x, y, th = odo.pose()
                if result == "blocked":
                    self.blocked.add(self._cell(x + LEG_MM / 2 * math.cos(th), y + LEG_MM / 2 * math.sin(th)))
                    r._odo_move(-BACK_OFF_MM, -BACK_OFF_MM, DRIVE_MM_S, timeout=6)
                legs += 1
                r._caption(f"search: leg {legs} ({result}), {len(self.visits)} cells covered - scanning for tags")
                if r._scan_for_tag() is not None:
                    return self._found(start, legs)
            r._caption(f"search: no tags after {self.time_budget_s / 60:.0f} min - giving up")
            return False
        finally:
            odo.stop()
            self.dock_camera.set_slam(False)

    def _found(self, start, legs):
        self.robot._caption(f"search: tags found after {time.time() - start:.0f} s, {legs} legs - docking")
        return True
