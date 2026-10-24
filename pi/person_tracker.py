"""
Project Qwen - keep a person in view during security mode by turning on the spot.

The robot sits in a corner where it can see most of the room, so it doesn't follow people
around: it only turns left/right, slowly, to keep the face (or the person, if they turn
away) centred in the camera. Security mode feeds it the target's horizontal position every
frame; turning happens in short pulses on its own thread, so the face loop never waits.

Before the first turn of an ARMED-ACTIVE spell it drives straight forward a little, so the
body can't scrape the wall while turning: 20 cm when it starts on the charger (off the pads),
10 cm otherwise. If it left the charger, `on_left_dock` is called when the spell ends, so
assistant.py can send it back to dock.

Speeds stay far below the old "speed 1" (~130 mm/s): 40 mm/s forward, 25 mm/s per wheel
when turning (~8 deg/s).
"""

import threading
import time

from robot_control import MM_PER_COUNT, WHEEL_SPACING_MM

FORWARD_OFF_DOCK_MM = 200      # charging: clear the wall and the dock contacts first
FORWARD_FREE_MM = 100          # not charging: a little room to turn
FORWARD_MM_S = 40
FORWARD_TIMEOUT_S = 10.0
TURN_MM_S = 25                 # per wheel, opposite directions: ~8 deg/s on the spot
CAMERA_HFOV_DEG = 70.0         # C920 horizontal field of view
DEADBAND_DEG = 6.0             # close enough to centre: don't move
MAX_PULSE_S = 0.6              # longest single turn before re-checking the picture
SETTLE_S = 0.25                # pause after a pulse so the next frame isn't blurred
TARGET_STALE_S = 0.8           # no fresh target this long: hold still

TURN_DEG_PER_S = 2 * TURN_MM_S / WHEEL_SPACING_MM * 180 / 3.141592653589793


class PersonTracker:
    def __init__(self, link, robot=None, is_charging=None, on_left_dock=None):
        """link: ESP32Link. robot: RobotController (its mode is set to "security" while the
        tracker owns the wheels, which keeps docking and idle animations out of the way).
        is_charging: () -> bool. on_left_dock: called when a spell ends after leaving the dock."""
        self.link = link
        self.robot = robot
        self.is_charging = is_charging or (lambda: False)
        self.on_left_dock = on_left_dock
        self._lock = threading.Lock()
        self._target = None            # (angle_deg, time)
        self._engaged = False
        self._left_dock = False
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------------ called by security mode

    def update(self, center_x: float, frame_width: int):
        """A person/face centred at center_x (pixels) in the latest frame."""
        angle = (center_x / frame_width - 0.5) * CAMERA_HFOV_DEG     # + = to the right
        with self._lock:
            self._target = (angle, time.time())
            first = not self._engaged
            self._engaged = True
        if first:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def release(self):
        """The ARMED-ACTIVE spell is over (back to motion-only, or disarmed)."""
        with self._lock:
            was = self._engaged
            self._engaged = False
            self._target = None
        if not was:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self.link.stop()
        if self.robot is not None and self.robot.mode == "security":
            self.robot.mode = "idle"
        if self._left_dock:
            self._left_dock = False
            if self.on_left_dock:
                self.on_left_dock()

    # ------------------------------------------------------------------ motion

    def _forward(self, mm: float) -> bool:
        l0, r0 = self.link.odom()
        self.link.drive_speed(FORWARD_MM_S, FORWARD_MM_S)
        end = time.time() + FORWARD_TIMEOUT_S
        try:
            while time.time() < end and not self._stop.is_set():
                if self.link.halted:
                    print("[track] emergency stop during the forward move - not tracking this time")
                    return False
                if self.link.obstacle_ahead():
                    print("[track] front IR - stopped the forward move early")
                    break
                l, r = self.link.odom()
                if ((l - l0) + (r - r0)) / 2 * MM_PER_COUNT >= mm:
                    break
                time.sleep(0.03)
        finally:
            self.link.stop_smooth(0.4)
        return not self._stop.is_set()

    def _run(self):
        if self.robot is not None:
            if self.robot.mode not in ("idle", "security"):
                print(f"[track] robot busy ({self.robot.mode}) - not tracking")
                return
            self.robot.mode = "security"
        charging = bool(self.is_charging())
        mm = FORWARD_OFF_DOCK_MM if charging else FORWARD_FREE_MM
        print(f"[track] person seen - {mm / 10:.0f} cm forward{' off the dock' if charging else ''}, then tracking")
        if self.link.halted or not self._forward(mm):
            return
        self._left_dock = charging
        while not self._stop.is_set():
            with self._lock:
                t = self._target
            if (self.link.halted or t is None or time.time() - t[1] > TARGET_STALE_S
                    or abs(t[0]) < DEADBAND_DEG):
                self._stop.wait(0.1)
                continue
            angle = t[0]
            secs = min(MAX_PULSE_S, abs(angle) / TURN_DEG_PER_S)
            if angle > 0:      # target right of centre -> turn right
                self.link.drive_speed(TURN_MM_S, -TURN_MM_S)
            else:
                self.link.drive_speed(-TURN_MM_S, TURN_MM_S)
            self._stop.wait(secs)
            self.link.stop()
            with self._lock:
                self._target = None        # wait for a frame taken after this turn
            self._stop.wait(SETTLE_S)
        self.link.stop()
