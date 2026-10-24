"""
Project Qwen - one owner for the drive motors: manual remote control from the
web page (WASD) and autonomous behaviours (explore), with instant take-over.

Modes:  idle    - motors stopped
        manual  - driven from the web page; a dead-man timer stops the motors
                  if the page stops sending (key released, tab closed, Wi-Fi drop)
        auto    - a behaviour thread drives; any manual key or STOP cancels it

The ESP32 keeps its own safety on top of this: front-IR obstacle reflex,
reverse speed/time caps, and a 500 ms watchdog.
"""

import math
import random
import statistics
import threading
import time

from battery_monitor import CHARGE_CURRENT_SIGN, CHARGING_THRESHOLD_A, INA226
from dock_view import DockView, features, grab, scale_to
from dock_view import match as dock_match
from esp32_link import ESP32Link

TAG_FILE = "/dev/shm/slam_tag.txt"
CAPTION_FILE = "/dev/shm/slam_view.jpg.txt"
MANUAL_DEADMAN_S = 0.35
MAX_SPEED = 30              # page slider cap (ESP32 caps again at 150)
MANUAL_TURN_MM_S = 30       # A / D: ~9 deg/s - fast spins break ORB-SLAM tracking (130 lost it often)
MANUAL_DRIVE_MM_S = 130     # W / S: same as the old plain-power speed 1 (~130 mm/s; docking teach used 20)
MIN_SPEED = 1               # 1 is the user's preferred slow speed (2026-10-13)
# Gentle speed changes (the ESP32's own ramp is near-instant at these low speeds):
ACCEL_PER_S = 20            # speed units per second when speeding up (0 -> 10 in ~0.5 s)
DECEL_PER_S = 30            # when slowing down / releasing a key (10 -> 0 in ~0.3 s)
RAMP_HZ = 25

# explore
BACKUP_S = 0.5              # no rear sensors: back up only a little...
MAX_BACKUPS_IN_ROW = 2      # ...and never more than twice before trying turns only
MAX_OBSTACLES_IN_ROW = 6    # then give up and report "stuck"
TURN_S = 0.7
STRAIGHT_RUN_S = 30

# centre on the dock tag (turn in place until it's in the middle of the picture)
CENTRE_TOL_DEG = 1.5        # "dead centre" (the band drawn on the live view)
CENTRE_SETTLE_S = 0.6       # let the robot stop rocking/coasting before looking again
CENTRE_MIN_PULSE_S = 0.05   # starting guess; grows if a pulse doesn't move the robot

# approach the dock tag: forward pulses, re-centring whenever it drifts off
APPROACH_TARGET_CM = 37     # turn spot (camera to tags), from the same docks: 32, 38, 38, 41, 36 cm
APPROACH_TOL_DEG = 2.5      # re-centre before the next forward pulse if further off than this
# Get onto the tag's centre line on the way in (pointing at the tag isn't enough - from the
# side it arrives at an angle). Steer toward the line, like merging into a lane.
LINE_TOL_CM = 1.5           # "on the line" (errors grow ~x2.5 on the way back, so be tight)
# The 180 deg turn-around shifts the robot sideways (wheels not exactly matched): it landed
# ~7 cm to its own left at the dock, twice in a row (2026-10-15). Line up that far to the
# other side before turning so the turn carries it onto the pads. (+ = tag's +x side.)
LINE_TARGET_CM = 9.0        # dock line, from 5 recorded manual docks (2026-10-17): +8.6, +8.6, +10.0, +9.4, +8.5 cm
HEADING_TOL_DEG = 3.0
KEEP_IN_VIEW_DEG = 20
FACE_TOL_DEG = 3.0          # "facing the wall" (median heading in the tag's frame; turn steps are ~2-5 deg)
SHUFFLE_ROUNDS = 3
ZIGZAG_ROUNDS = 10           # parallel-parking style sideways fixes at the turn spot
ZIGZAG_DEG = 15             # angle used for each leg (small enough to keep the tag in view)
ZIGZAG_EARLY_CM = 15        # still >4 cm off the line this close to the turn spot: zigzag now
STEP2_MAX_OFF_CM = 3.0      # step 2 won't finish further than this off the dock line...
STEP2_RELINE_OUT_CM = 70    # ...it backs out to here and lines up again instead
LINEUP_MAX_OUT_CM = 95      # farther out than this: drive toward the tags before sidestepping...
LINEUP_TARGET_OUT_CM = 80   # ...to about here
SIDESTEP_MAX_CM = 30.0      # cap per sidestep: a bad tag reading (2026-10-17: "96 cm off" 40 cm from the dock)
                            # can not send it off 1 m blind - the camera re-checks after every leg
SIDESTEP_MIN_CM = 2.5       # any real offset: 90 deg sidestep on odometry (the zigzag lost track of the side)
ZIGZAG_VIEW_DEG = 22        # the camera sees +-26 deg; keep the tag inside +-22 while zigzagging
MERGE_LOOKAHEAD_CM = 30     # merging onto the centre line: aim at a point this far ahead on it
MERGE_MAX_DEG = 18          # never aim more than this off straight-at-the-wall
BACKUP_MAX_PULSES = 30      # backing up for merging room: at most ~45 cm
SIDE_TEST_BEARING = 12      # side check needs the tag at least this far off to one side...
SIDE_TEST_MAX_BEARING = 17  # ...but not so far that driving forward pushes it out of view
SIDE_TEST_PULSES = 15       # at most this many forward pulses for the side check (stops at ~6 cm moved)
TURN_CREEP_CM = -0.8        # turning in place also moves the robot this far per pulse (- = backward);
                            # re-measured during each docking turn

# find the dock: turn in place in steps, stopping to look (the tag can't be read mid-spin)
SCAN_STEP_PULSES = 4        # ~22 deg per step (camera sees ~53 deg, so looks overlap)
SCAN_PULSE_GAP_S = 0.15
SCAN_TOTAL_DEG = 380        # a bit past a full turn (pulse counting is approximate)
SCAN_LOOK_S = 0.6           # how long to look at each stop
SCAN_MAX_STOPS = 40         # safety limit even if the measured turn stays small

# teach / back onto the dock (camera faces away from the dock, into the room)
PULSE_S = 0.15              # one "inch" at speed 1: ~1.2 cm forward, ~1.5 cm back, ~5.4 deg turn
FWD_CM_PER_PULSE = 1.2
# Wheel odometry (hall sensors, 2026-10-16): 60 counts per wheel turn on both wheels; the floor
# test against the tag camera says ~8.5 mm of travel per count. Spacing is approximate.
MM_PER_COUNT = 8.5
WHEEL_SPACING_MM = 365.0    # effective (turns in place slip a little): 180 on 330 turned ~200 (2026-10-16)
ODO_TURN_SPEED = 20         # mm/s per wheel when turning in place on odometry (slow: less slip)
ODO_DRIVE_SPEED = 30        # mm/s for straight odometry moves while lining up
ODO_BACK_SPEED = 30         # mm/s when backing onto the dock (user: 60 was too fast)
ODO_BACK_STEP_MM = 50       # back up this far, then re-check the heading against the docked view
BACK_PULSE_SPEED = 25       # back-on in short gentle pulses: 25 mm/s for BACK_PULSE_S, then ease to a stop
BACK_PULSE_S = 0.9          # long enough for the speed control to get a gentle grip
BACK_PULSES_PER_CHECK = 4   # pulses between heading checks (~4-5 cm)
BACK_TWIST_ABORT_DEG = 6.0  # a back-up step that twists more than this hit something: stop
BACK_STALL_FRACTION = 0.35
# touching the dock without charge: wiggle (out ~1 cm, turn a hair, back in), alternating sides
WIGGLE_ENABLED = False      # off: too big and too far out for a few-mm miss (user, 2026-10-17)
WIGGLE_OUT_MM = 10
WIGGLE_SEQUENCE = [(2, True), (2, False), (4, True), (4, False), (6, True), (6, False)]  # a step that covers less than this share of its distance = touching the dock
REV_CM_PER_PULSE = 1.5
TURN_DEG_PER_PULSE = 5.4
TEACH_PULSES = 16           # (old pulse teach)
TEACH_SPEED_MM_S = 15       # teach: creep out this slowly on the speed control...
TEACH_STEP_MM = 10          # ...taking a picture every 1 cm (wheel odometry)...
TEACH_DIST_MM = 250         # ...out to 25 cm
TURN_BLIND_DEG = 140        # turning around: count pulses this far without looking...
TURN_SEARCH_PULSES = 8      # after the odometry 180: up to 8 small search turns (10, 10, 20, 20, ... deg)
TURN_STEP_PULSES = 4        # pulses between looks while turning around (~22 deg)
TURN_MAX_STOPS = 60
ODO_TURN_DEG = 180
TURN_FOUND_DEG = 25         # recognised and within this of the docked heading = hand over
TURN_CREEP_PULSES = 3       # measure the creep-while-turning over the first pulses (tag still visible)
BACK_TOL_DEG = 3.0          # wider: fewer left-right-left corrections while backing on          # heading must be this close to the docked view before reversing
BACK_MAX_REVERSE = 45       # reverse pulses before giving up
BACK_RUN_S = 0.35           # continuous reverse per run (~4-5 cm at speed 1), heading re-checked between
BACK_MAX_RUNS = 14
REVERSE_START_L = 2         # starting reverse speeds (left wheel lags in reverse)
REVERSE_START_R = 1
BACK_STALL_SCALE = 0.003    # scene size changed less than this over a reverse run = it didn't move
BACK_TWIST_DEG = 2.0        # a reverse run that turns the robot more than this = one wheel lagging
BACK_MAX_WHEEL = 4          # never push a lagging wheel above this speed
BACK_MIN_TURN_S = 0.08      # 0.05 s turn pulses didn't move the wheels (2026-10-14 log)
# The pads bounce the robot back off after a pulse hits them, so "docked" means charge
# current held steadily; if it drops, nudge back with a short pulse and check again.
SEAT_STABLE_S = 3.0         # docked = charging held for 3 s (user's rule, 2026-10-16)
SEAT_SETTLE_S = 0.5         # after contact, a moment for the bounce before judging
SEAT_MIN_FRACTION = 0.95    # charging in (almost) every sample over those 3 s
SEAT_NUDGE_S = 0.08
SEAT_MAX_NUDGES = 10
# Final push (user's idea): from close in, one continuous reverse instead of pulses, kept on
# a moment after contact to press onto the pads, then released.
PUSH_FROM_CM = 3            # without a pre-push picture: push when the strip says this close...
PUSH_CONFIRM = 2            # ...this many readings in a row (the estimate is noisy)
PREPUSH_SCALE_OK = 0.997    # with a pre-push picture: push once the scene is within 0.3% of it (~<1 cm)
PUSH_MAX_S = 0.5            # ~7 cm at speed 1
PUSH_HOLD_S = 3.0           # keep pushing this long after charge current appears (user's manual test)
REVERSE_REARM_S = 1.2       # the ESP32 cuts reverse after 1.5 s; a zero blip restarts its timer
PUSH_TRIES = 3
PUSH_ALIGN_TOL_DEG = 1.0    # tighter heading before the push (a push can't correct itself)
PUSH_ALIGN_TRIES = 8


def read_tag(max_age: float = 0.5):
    """Dock tag seen in the last max_age s: (dist_cm, x_cm, bearing_deg, yaw_deg, lateral_cm,
    normal_cm, heading_deg) - the last three are the robot's place on the floor relative to
    the tag (sideways from its centre line, out from the wall, pointing angle)."""
    try:
        parts = open(TAG_FILE).read().split()
    except OSError:
        return None
    if len(parts) < 6 or time.time() - float(parts[0]) > max_age:
        return None
    return tuple(float(p) for p in parts[2:])


class RobotController:
    def __init__(self, link=None, camera_hook=None):
        """link: a shared, already-started ESP32Link (assistant.py), or None to own one.
        camera_hook: object with acquire()/release() run around every auto task (DockCamera)."""
        self._own_link = link is None
        self.link = link or ESP32Link()
        self.camera_hook = camera_hook
        self.battery_fn = None              # set by the power manager: () -> (volts, percent, charging)
        self.task = ""                      # name of the last auto task started (for Telegram resume)
        self.say = None                     # text-to-speech (assistant.py), for the page's video lines
        self.page_hooks = {}                # control-page actions owned by assistant.py (arm, alarm, volume)
        self.extra_status = None            # () -> dict merged into status() (security, alarm, volume)
        self.lost_tag_search = None         # LostTagSearch: runs when a full turn finds no tags
        self.on_dock_start = None           # optional callables around a full docking run (room lights)
        self.on_dock_end = None
        self.mode = "idle"
        self.activity = "stopped"
        self.speed = 1
        self._lock = threading.Lock()
        self._motion = (0, 0)               # target
        self._cur = [0.0, 0.0]              # what the ramp is currently sending
        self._sent = (0, 0)
        self._last_manual = 0.0
        self._spin = None                   # manual slow spin in progress (speed mode), or None
        self._auto_cancel = threading.Event()
        self._auto_thread = None
        self.dock_view = DockView()
        self.turn_creep = TURN_CREEP_CM
        self.tag_side = None        # which mirror-image pose solution is real: +1 / -1 / None (unknown)
        self.reverse_balance = {"l": REVERSE_START_L, "r": REVERSE_START_R}
        self._ina = None

    def start(self):
        if self._own_link:
            self.link.start()
            time.sleep(1.0)
        self.link.lift(0)                       # docking work: lift stays down
        threading.Thread(target=self._deadman_loop, daemon=True).start()
        threading.Thread(target=self._ramp_loop, daemon=True).start()

    def close(self):
        self.stop("shutting down")
        if self._own_link:
            self.link.close()

    # ------------------------------------------------------------------ status

    def lift_lock_reason(self) -> str:
        if self.link.halted:
            return "emergency stop from Telegram"
        if self.link.lift_allowed is not None and not self.link.lift_allowed(130):
            return "charging (servo current disturbs the contacts)"
        return ""

    def status(self) -> dict:
        st = self.link.state
        tag = read_tag()
        out = {
            "mode": self.mode, "activity": self.activity, "speed": self.speed,
            "motion": list(self._sent), "esp32": self.link.connected, "odom": list(self.link.odom()),
            "lift": st.lift_pos, "lift_locked": self.lift_lock_reason(),
            "ir": {k: v for k, v in st.ir.items() if k != "head"},
            "blocked": st.last_blocked if time.time() - st.last_blocked_at < 2 else None,
            "tag": None if tag is None else dict(zip(("dist_cm", "x_cm", "bearing", "yaw"), tag)),
        }
        if self.extra_status:
            try:
                out.update(self.extra_status())
            except Exception:
                pass
        return out

    def _caption(self, text: str):
        self.activity = text
        try:
            with open(CAPTION_FILE, "w") as f:
                f.write(f"[{self.mode}] {text}")
        except OSError:
            pass
        print(time.strftime("%H:%M:%S"), f"[{self.mode}] {text}", flush=True)

    # ------------------------------------------------------------------ commands

    def _drive(self, left: int, right: int):
        """Sets the target; _ramp_loop eases the motors toward it."""
        if (left, right) != self._motion:
            print(time.strftime("%H:%M:%S"), f"[{self.mode}] target {left:+d} {right:+d}", flush=True)
        self._motion = (left, right)

    def _ramp_loop(self):
        while True:
            time.sleep(1.0 / RAMP_HZ)
            with self._lock:
                for i, target in enumerate(self._motion):
                    cur = self._cur[i]
                    speeding_up = abs(target) > abs(cur) and (cur == 0 or (target > 0) == (cur > 0))
                    step = (ACCEL_PER_S if speeding_up else DECEL_PER_S) / RAMP_HZ
                    self._cur[i] = min(target, cur + step) if target > cur else max(target, cur - step)
                out = (round(self._cur[0]), round(self._cur[1]))
                if out != self._sent:
                    self._sent = out
                    if out == (0, 0):
                        self.link.drive(0, 0)       # ease to a halt (STOP would cut instantly)
                        self.link.stop()
                    else:
                        self.link.drive(*out)

    def stop(self, why: str = "STOP"):
        """Hard stop - no easing."""
        self._cancel_auto()
        with self._lock:
            self.mode = "idle"
            self._motion = (0, 0)
            self._cur = [0.0, 0.0]
            self._sent = (0, 0)
            self.link.stop()
        self._caption(why)

    def set_speed(self, speed: int):
        self.speed = max(MIN_SPEED, min(MAX_SPEED, int(speed)))

    def manual(self, keys: str):
        """keys: any of w/a/s/d currently held ('' = none). Called repeatedly while held."""
        if self.link.halted and keys:
            return
        self._cancel_auto()
        s = self.speed
        w, a, b, d = ("w" in keys), ("a" in keys), ("s" in keys), ("d" in keys)
        # Everything manual runs under the ESP32 speed control now (mm/s), so it can go far
        # slower than speed 1 in plain power mode (~130 mm/s at the wheels).
        v, t = MANUAL_DRIVE_MM_S, MANUAL_TURN_MM_S
        if w and not b:
            want = (v // 3, v) if a and not d else (v, v // 3) if d and not a else (v, v)
        elif b and not w:
            want = (-v // 3, -v) if a and not d else (-v, -v // 3) if d and not a else (-v, -v)
        elif a and not d:
            want = (-t, t)
        elif d and not a:
            want = (t, -t)
        else:
            want = None
        with self._lock:
            if self.mode != "manual":
                self.mode = "manual"
                self._caption("you have control")
            self._last_manual = time.time()
            if self._motion != (0, 0) or self._sent != (0, 0):      # leave plain power mode
                self._motion = (0, 0)
                self._sent = (0, 0)
                self.link.stop()
            if want != self._spin:
                self._spin = want
                if want is None:
                    self.link.stop()
                else:
                    self.link.drive_speed(*want)

    def test_motors(self, left: int, right: int, seconds: float):
        """Bench check: run the raw left/right command for `seconds` (with easing), then stop."""
        self._cancel_auto()
        with self._lock:
            self.mode = "test"
            self._drive(left, right)
        self._caption(f"motor test: left {left:+d}  right {right:+d} for {seconds:.1f}s")

        def finish():
            time.sleep(seconds)
            with self._lock:
                if self.mode == "test":
                    self._drive(0, 0)
                    self.mode = "idle"
        threading.Thread(target=finish, daemon=True).start()

    def sweep(self, wheel: str, direction: int, top: int = 25, step_s: float = 1.0):
        """Bench check (wheels off the ground): one wheel's command climbs 1 unit per step_s,
        shown in the caption, so you can say at which number it starts turning."""
        self._cancel_auto()
        self._auto_cancel.clear()
        with self._lock:
            self.mode = "test"

        def run():
            name = f"{wheel} wheel {'forward' if direction > 0 else 'backward'}"
            for v in range(1, top + 1):
                if self._auto_cancel.is_set() or self.mode != "test":
                    return
                cmd = direction * v
                with self._lock:
                    self._motion = (cmd, 0) if wheel == "left" else (0, cmd)
                    self._cur = [float(self._motion[0]), float(self._motion[1])]
                    if direction < 0:
                        # the ESP32 cuts reverse after 1.5 s; a zero command restarts that timer
                        # (its ramp makes the blip a few ms long)
                        self.link.drive(0, 0)
                        self.link.drive(*self._motion)
                        self._sent = self._motion
                self._caption(f"SWEEP {name}: speed {v}")
                time.sleep(step_s)
            with self._lock:
                self._drive(0, 0)
                self.mode = "idle"
            self._caption(f"SWEEP {name}: done")
        self._auto_thread = threading.Thread(target=run, daemon=True)
        self._auto_thread.start()

    def _deadman_loop(self):
        while True:
            time.sleep(0.05)
            with self._lock:
                stale = time.time() - self._last_manual > MANUAL_DEADMAN_S
                if self.mode == "manual" and self._motion != (0, 0) and stale:
                    self._drive(0, 0)
                if self._spin is not None and (stale or self.mode != "manual"):
                    self._spin = None
                    self.link.stop()

    # ------------------------------------------------------------------ autonomy

    def start_explore(self):
        self._start_auto(self._explore)

    def start_centre(self):
        self._start_auto(self._centre_on_tag)

    def start_find_dock(self):
        self._start_auto(self._find_dock)

    def _look_for_tag(self, seconds: float):
        end = time.time() + seconds
        while time.time() < end:
            t = read_tag(0.25)
            if t:
                return t
            if self._auto_cancel.wait(0.05):
                return None
        return None

    def _find_dock(self):
        """Find-dock button: scan, then approach (stops at the turn spot)."""
        if self._find_and_approach():
            p = self._pose()
            self._finish("READY at the turn spot" + (f": {p[5]:.0f} cm out, {p[4]:+.1f} cm off the line, "
                                                     f"heading {p[6]:+.1f} deg" if p else ""))

    def _find_and_approach(self, tries: int = 3) -> bool:
        """Scan for the tag and approach it; if the tag gets lost on the way, scan again."""
        for attempt in range(1, tries + 1):
            tag = self._scan_for_tag()
            if self._auto_cancel.is_set():
                return False
            if tag is None:
                self._finish("find: no tag in a full turn - ORB-SLAM fallback not built yet")
                return False
            if self._approach(APPROACH_TARGET_CM, finish=False):
                return True
            if self._auto_cancel.is_set():
                return False
            if "lost the tag" not in self.activity:
                self._finish(self.activity)
                return False
            self._caption(f"find: lost the tag on the way - scanning again ({attempt}/{tries})")
        self._finish("find: kept losing the tag - stopped")
        return False

    def _measured_turn(self, before, after):
        """Degrees the robot actually turned between two still frames (sideways shift of the
        matched scene; + = turned left), or None if they don't overlap enough to tell."""
        if before is None or after is None:
            return None
        n, heading = dock_match(features(after), features(before))
        return None if heading is None else -heading

    def _scan_for_tag(self):
        """Turn in place in ~22 deg steps, looking for the dock tag at each stop. The turn is
        MEASURED from the camera (scene shift between stops), not just counted - at low
        battery the pulses partly stall - and a step that barely turned gets more power.
        Returns the tag reading, or None after a full turn (or if cancelled)."""
        turned = 0.0
        stop = 0
        boost = 0                                  # extra speed for stalling turns
        prev_img = grab()
        while True:
            if not self._auto_pause(0.5):          # settle before looking
                return None
            tag = self._look_for_tag(SCAN_LOOK_S)
            if self._auto_cancel.is_set():
                return None
            if tag:
                self._caption(f"find: TAG FOUND after ~{turned:.0f} deg - {tag[0]:.0f} cm, "
                              f"{tag[2]:+.0f} deg -> approaching")
                return tag if self._auto_pause(0.5) else None
            if turned >= SCAN_TOTAL_DEG:
                return None
            stop += 1
            s = self.speed + boost
            self._caption(f"find: no tag yet - turning (stop {stop}, {turned:.0f} of {SCAN_TOTAL_DEG} deg"
                          f"{f', speed {s}' if boost else ''})")
            for _ in range(SCAN_STEP_PULSES):
                if not self._auto_move(-s, s, PULSE_S):
                    return None
                if not self._auto_pause(SCAN_PULSE_GAP_S):
                    return None
            img = self._still_frame()
            step = self._measured_turn(prev_img, img)
            prev_img = img
            expected = SCAN_STEP_PULSES * TURN_DEG_PER_PULSE
            if step is None:
                step = expected                    # too far to compare: trust the count
            elif abs(step) < 0.4 * expected and boost < 2:
                boost += 1                         # barely turned: low battery, push harder
            turned += abs(step)
            if stop > SCAN_MAX_STOPS:
                return None

    def start_approach(self, target_cm: float = APPROACH_TARGET_CM):
        self._start_auto(self._approach, target_cm)

    def _pose_both(self, n: int = 8, timeout: float = 2.5):
        """Median of n fresh readings: (bearing, dist, plus_side, minus_side) where each side is
        (lateral, out, heading) for one of the two mirror-image solutions. None if no tag."""
        got, end = [], time.time() + timeout
        while len(got) < n and time.time() < end:
            t = read_tag(0.2)
            if t is not None and len(t) >= 15 and (not got or t != got[-1]):
                got.append(t)
            if self._auto_cancel.wait(0.05):
                return None
        if len(got) < max(3, n // 2):
            return None
        med = [statistics.median(g[i] for g in got) for i in range(15)]
        return med[2], med[0], (med[7], med[8], med[9]), (med[11], med[12], med[13])

    def _pose(self, n: int = 8, timeout: float = 2.5):
        """Median pose (dist, x, bearing, yaw, lateral, out, heading) using the side we've
        worked out is real (see _which_side); falls back to the camera's own pick."""
        both = self._pose_both(n, timeout)
        if both is None:
            return None
        b, dist, plus, minus = both
        if self.tag_side is None:
            t = read_tag(0.5)
            if t is None:
                return None
            lat, out, head = t[4], t[5], t[6]
        else:
            lat, out, head = plus if self.tag_side > 0 else minus
        return dist, 0.0, b, 0.0, lat, out, head

    def _relock_side(self, expected_lat: float):
        """After a known sideways move: keep whichever solution lands nearest to where we
        expect to be (the real one moves the way we drove)."""
        both = self._pose_both()
        if both is None:
            return
        _, _, plus, minus = both
        self.tag_side = 1 if abs(plus[0] - expected_lat) <= abs(minus[0] - expected_lat) else -1
        print(time.strftime("%H:%M:%S"), f"[side] expected {expected_lat:+.1f}, + side {plus[0]:+.1f}, "
              f"- side {minus[0]:+.1f} -> {self.tag_side:+d}", flush=True)

    def _which_side(self) -> bool:
        """The tag alone can't tell which side of its centre line we're on (two mirror-image
        answers fit). Drive a few cm forward at an angle to the tag: the real answer's
        position moves the way the robot points; the mirror one doesn't."""
        s = self.speed
        both = self._pose_both()
        if both is None:
            return False
        if abs(both[2][0] - both[3][0]) < 1.0 and abs(both[2][2] - both[3][2]) < 1.0:
            self.tag_side = 1                     # both tags in view: one clear answer, no check needed
            self._caption("side check: both tags in view - position is unambiguous")
            return True
        b = both[0]
        if abs(b) > SIDE_TEST_MAX_BEARING:       # tag near the edge: driving forward would lose it
            left = b < 0                          # tag on the left: turn left to bring it in
            pulses = max(1, round((abs(b) - SIDE_TEST_BEARING - 2) / TURN_DEG_PER_PULSE))
            self._caption(f"side check: tag at {b:+.0f} deg - turning {'left' if left else 'right'} to bring it in")
            if not self._turn_pulses(pulses, left):
                return False
            if not self._auto_pause(CENTRE_SETTLE_S):
                return False
            both = self._pose_both()
            if both is None:
                return False
            b = both[0]
        if abs(b) < SIDE_TEST_BEARING:          # pointing near the tag doesn't tell: angle off first
            # swing the tag across to ~15 deg on the other side (turning right moves it left)
            left = b < 0
            pulses = max(2, round((abs(b) + 15) / TURN_DEG_PER_PULSE))
            self._caption(f"side check: angling {'left' if left else 'right'} so the tag sits off to one side")
            if not self._turn_pulses(pulses, left):
                return False
            if not self._auto_pause(CENTRE_SETTLE_S):
                return False
            both = self._pose_both()
            if both is None:
                return False
        _, _, a0, m0 = both
        self._caption(f"side check: driving forward at {both[0]:+.0f} deg to the tag until it has moved ~6 cm")
        a1 = m1 = None
        for k in range(SIDE_TEST_PULSES):
            if self.link.obstacle_ahead():
                break
            if not (self._auto_move(s, s, PULSE_S) and self._auto_pause(SCAN_PULSE_GAP_S)):
                return False
            if k % 3 == 2 or k == SIDE_TEST_PULSES - 1:     # check progress every few pulses
                if not self._auto_pause(CENTRE_SETTLE_S):
                    return False
                both = self._pose_both(n=5)
                if both is None:
                    return False
                _, _, a1, m1 = both
                if math.hypot(a1[0] - a0[0], a1[1] - a0[1]) >= 6.0:
                    break
        if a1 is None:
            return False

        def mismatch(p0, p1):
            dl, do = p1[0] - p0[0], p1[1] - p0[1]
            if math.hypot(dl, do) < 3.0:
                return None
            moved = math.degrees(math.atan2(dl, -do))        # direction of travel (0 = at the wall)
            pointing = (p0[2] + p1[2]) / 2
            return abs((moved - pointing + 180) % 360 - 180)

        ma, mm = mismatch(a0, a1), mismatch(m0, m1)
        print(time.strftime("%H:%M:%S"), f"[side] + side {a0} -> {a1} mismatch {ma}; "
              f"- side {m0} -> {m1} mismatch {mm}", flush=True)
        if ma is None or mm is None or abs(ma - mm) < 8:
            self.tag_side = None
            self._caption("side check: inconclusive - using the camera's best guess")
            return True
        self.tag_side = 1 if ma < mm else -1
        lat = (a1 if self.tag_side > 0 else m1)[0]
        self._caption(f"side check: robot is {abs(lat):.0f} cm to the tag's {'right' if lat > 0 else 'left'} "
                      f"(travel matched {min(ma, mm):.0f} vs {max(ma, mm):.0f} deg)")
        return True

    def _turn_pulses(self, n: int, left: bool) -> bool:
        """n turn pulses, cancelling the creep (turning isn't perfectly in place) with
        forward/back pulses whenever it adds up to one pulse's worth."""
        s = self.speed
        l, r = (-s, s) if left else (s, -s)
        owed = 0.0
        for _ in range(n):
            if not (self._auto_move(l, r, PULSE_S) and self._auto_pause(SCAN_PULSE_GAP_S)):
                return False
            owed -= self.turn_creep
            if abs(owed) >= FWD_CM_PER_PULSE:
                d = 1 if owed > 0 else -1
                if not (self._auto_move(d * s, d * s, PULSE_S) and self._auto_pause(SCAN_PULSE_GAP_S)):
                    return False
                owed -= d * (FWD_CM_PER_PULSE if d > 0 else REV_CM_PER_PULSE)
        return True

    def _face_wall(self, finish: bool = False, max_pulses: int = 25) -> bool:
        if self._face_wall_once(finish=False, max_pulses=max_pulses):
            return True
        if self._auto_cancel.is_set():
            return False
        # lost the tags (an inexact turn left it looking away): look round for them, then try again
        self._caption("face wall: lost the tags - looking for them")
        if self._scan_for_tag() is not None and self._face_wall_once(finish=finish, max_pulses=max_pulses):
            return True
        if finish and not self._auto_cancel.is_set():
            self._finish("face wall: couldn't line up")
        return False

    def _face_wall_once(self, finish: bool = False, max_pulses: int = 25) -> bool:
        """Turn in place until pointing straight at the wall (median heading in the tag's
        frame within FACE_TOL_DEG). Heading + means pointing to the tag's right, so turning
        left lowers it."""
        s = self.speed
        dur = BACK_MIN_TURN_S
        prev = None
        for _ in range(max_pulses):
            if not self._auto_pause(CENTRE_SETTLE_S):
                return False
            p = self._pose()
            if p is None:
                if finish:
                    self._finish("face wall: lost the tag")
                return False
            head = p[6]
            if abs(head) <= FACE_TOL_DEG:
                msg = (f"facing the wall: {p[5]:.0f} cm out, {p[4]:+.1f} cm off the centre line, "
                       f"heading {head:+.1f} deg")
                if finish:
                    self._finish(msg)
                else:
                    self._caption(msg)
                return True
            if prev is not None and abs(head - prev) < 0.5:
                dur = min(0.15, dur + 0.02)
            prev = head
            self._caption(f"face wall: heading {head:+.1f} deg -> turn {'left' if head > 0 else 'right'}")
            if self._odo_turn(min(abs(head), 15.0) * 0.6, head > 0) is None:   # slow centre-pivot spin
                return False
        if finish:
            self._finish("face wall: couldn't line up")
        return False

    def _approach(self, target_cm: float, max_steps: int = 250, finish: bool = True) -> bool:
        """Get onto the tag's centre line on the way in, then face the wall at target_cm.

        1. side check: which of the two mirror-image pose answers is real
        2. merge: aim a little toward the centre line (small angles, so the tag stays in
           view and every step is checked), creep forward, repeat - like changing lanes.
           The real answer is followed from step to step (it can only move a little).
        3. face the wall at the turn spot."""
        s = self.speed
        self.tag_side = None
        if not self._which_side():
            return self._approach_fail("lost the tag during the side check", finish)
        prev_lat = None
        both = self._pose_both()
        if both is not None and self.tag_side is not None:
            prev_lat = (both[2] if self.tag_side > 0 else both[3])[0]
        for n in range(max_steps):
            if not self._auto_pause(CENTRE_SETTLE_S):
                return False
            both = self._pose_both(n=5, timeout=2.0)
            if both is None:
                return self._approach_fail("lost the tag", finish)
            b, dist, plus, minus = both
            if prev_lat is None:
                lat, out, head = plus if (self.tag_side or 1) > 0 else minus
            else:                                   # follow the real answer: nearest to last time
                lat, out, head = plus if abs(plus[0] - prev_lat) <= abs(minus[0] - prev_lat) else minus
            prev_lat = lat
            if out <= target_cm + 0.5:
                break
            if out <= target_cm + ZIGZAG_EARLY_CM and abs(lat - LINE_TARGET_CM) > 2 * LINE_TOL_CM:
                break                               # still well off the line: zigzag from here (more room)
            if self.link.obstacle_ahead():
                return self._approach_fail(f"front IR sees something {out:.0f} cm from the wall", finish)
            want = -math.degrees(math.atan2(lat - LINE_TARGET_CM, max(MERGE_LOOKAHEAD_CM, out - target_cm)))
            want = max(-MERGE_MAX_DEG, min(MERGE_MAX_DEG, want))
            err = head - want
            if abs(b) > KEEP_IN_VIEW_DEG:
                left = b < 0                         # tag slipping off the left edge: turn left
                why = f"tag {b:+.0f} deg - keeping it in view"
            elif abs(err) > HEADING_TOL_DEG:
                left = err > 0                       # + heading = pointing right of the wall line
                why = f"heading {head:+.1f}, want {want:+.1f}"
            else:
                left_cm = out - target_cm
                dur = 0.3 if left_cm > 15 else 0.15      # 0.08 s forward pulses barely move it
                self._caption(f"merge: {out:.0f} cm out, {lat:+.1f} cm off the line, heading {head:+.1f} "
                              f"-> forward {dur:.2f}s")
                if not self._auto_move(s, s, dur):
                    return False
                continue
            dur = 0.15 if abs(err) > 8 else 0.08
            self._caption(f"merge: {out:.0f} cm out, {lat:+.1f} cm off the line -> turn "
                          f"{'left' if left else 'right'} ({why})")
            l, r = (-s, s) if left else (s, -s)
            if not self._auto_move(l, r, dur):
                return False
        else:
            return self._approach_fail("gave up after too many steps", finish)
        if prev_lat is not None:
            self.tag_side = 1 if prev_lat >= 0 else -1
        self._caption(f"approach: reached {target_cm:.0f} cm, {prev_lat:+.1f} cm off the line")
        if not self._zigzag(target_cm, prev_lat):
            if self._auto_cancel.is_set():
                return False
            p = self._pose()
            if p is None or abs(p[4] - LINE_TARGET_CM) > 2 * LINE_TOL_CM:
                return self._approach_fail("couldn't get onto the centre line"
                                           + (f" ({p[4]:+.0f} cm off)" if p else ""), finish)
        return self._face_wall(finish=finish)

    def _turn_to_heading(self, want: float, lat_hint: float, tries: int = 12):
        """Turn in place until the heading (tag frame) is within HEADING_TOL_DEG of want.
        Returns (lat, out, head) or None if the tag is lost."""
        s = self.speed
        for _ in range(tries):
            both = self._pose_both(n=5)
            if both is None:
                return None
            near = lambda p: abs(p[0] - lat_hint)
            lat, out, head = both[2] if near(both[2]) <= near(both[3]) else both[3]
            lat_hint = lat
            if abs(head - want) <= HEADING_TOL_DEG:
                return lat, out, head
            left = head > want
            if self._odo_turn(min(abs(head - want), 15.0) * 0.6, left) is None or not self._auto_pause(CENTRE_SETTLE_S):
                return None
        both = self._pose_both(n=5)
        if both is None:
            return None
        near = lambda p: abs(p[0] - lat_hint)
        return both[2] if near(both[2]) <= near(both[3]) else both[3]

    def _turn_measured(self, deg: float, left: bool) -> bool:
        """Turn about `deg` degrees in place, measuring the real rotation from the camera
        between short bursts (stalled low-battery pulses get more power)."""
        turned, boost = 0.0, 0
        prev = self._still_frame()
        for _ in range(30):
            if turned >= deg - 3:
                return True
            s = self.speed + boost
            n = 2 if deg - turned > 12 else 1
            l, r = (-s, s) if left else (s, -s)
            for _ in range(n):
                if not (self._auto_move(l, r, PULSE_S) and self._auto_pause(SCAN_PULSE_GAP_S)):
                    return False
            img = self._still_frame()
            if img is None:
                return False
            step = self._measured_turn(prev, img)
            prev = img
            expected = n * TURN_DEG_PER_PULSE
            if step is None:
                step = expected
            elif abs(step) < 0.4 * expected and boost < 2:
                boost += 1
            turned += abs(step)
        return turned >= deg - 6

    def _sidestep(self, err: float, head: float = 0.0) -> bool:
        """Sideways offset: turn to point straight along the wall toward the line (from the current
        heading, so it doesn't need to face the wall first), drive across, turn back to face the wall
        - on wheel odometry, then the camera re-checks. head: current heading in the tags' frame."""
        left = err > 0                                  # + = right of the line: go left (toward -x)
        dist = min(abs(err), SIDESTEP_MAX_CM) * 10.0    # cm -> mm, capped (re-measured after)
        first = 90.0 + head if left else 90.0 - head    # left turns lower the heading
        self._caption(f"sidestep: {err:+.0f} cm off the line -> turn {'left' if left else 'right'} {first:.0f}, "
                      f"drive {dist:.0f} mm, turn back (odometry)")
        if self._odo_turn(max(0.0, first), left) is None:
            return False
        got = self._odo_move(dist, dist, ODO_DRIVE_SPEED, stop_on=self.link.obstacle_ahead, timeout=20)
        if got is None or self.link.obstacle_ahead():   # cancelled, or the front IR stopped it
            return False
        if self._odo_turn(90, not left) is None:
            return False
        return self._auto_pause(CENTRE_SETTLE_S)

    def _zigzag(self, target_cm: float, lat: float) -> bool:
        """Close a sideways offset at the turn spot without losing sight of the tag, like
        parallel parking: angle away from the line and reverse (carries it toward the line),
        then angle toward the line and drive forward back to target_cm. Re-measured each leg."""
        s = self.speed
        for rnd in range(1, ZIGZAG_ROUNDS + 1):
            if abs(lat - LINE_TARGET_CM) <= LINE_TOL_CM:
                return True
            both = self._pose_both(n=5)
            if both is None:
                return False
            near = lambda p: abs(p[0] - lat)
            lat, out, head = both[2] if near(both[2]) <= near(both[3]) else both[3]
            if abs(lat - LINE_TARGET_CM) > SIDESTEP_MIN_CM:
                if not self._sidestep(lat - LINE_TARGET_CM):
                    return False
                both = self._pose_both(n=5)
                if both is None:
                    tag = self._scan_for_tag()            # turned back slightly off: find it again
                    if tag is None:
                        return False
                    both = self._pose_both(n=5)
                    if both is None:
                        return False
                exp = LINE_TARGET_CM
                lat, out, head = both[2] if abs(both[2][0] - exp) <= abs(both[3][0] - exp) else both[3]
                continue
            ray = math.degrees(math.atan2(abs(lat), out))           # how far off to the side the tag is
            room = ZIGZAG_VIEW_DEG - ray                           # angle we can use and still see it
            if room < 4:
                # back out for room, angled so the tag stays just inside the picture (facing the
                # wall squarely would push it out of view this far off to the side)
                ray_dir = -math.degrees(math.atan2(lat, out))
                want = ray_dir - math.copysign(ZIGZAG_VIEW_DEG - 2, ray_dir)
                self._caption(f"zigzag {rnd}: {lat:+.1f} cm off at {out:.0f} cm - backing out for room "
                              f"(heading {want:+.0f})")
                if self._turn_to_heading(want, lat) is None:
                    return False
                if self._odo_move(-100, -100, 60, timeout=6) is None:
                    return False
                continue
            deg = min(ZIGZAG_DEG, room)
            ang = math.copysign(deg, lat - LINE_TARGET_CM)                 # + lat: point right, reverse moves left
            # each leg (back at +ang, forward at -ang) moves sin(ang) x leg sideways
            leg_mm = min(250.0, abs(lat - LINE_TARGET_CM) * 10.0 / (2 * math.sin(math.radians(deg))))
            self._caption(f"zigzag {rnd}: {lat:+.1f} cm off the line -> angle {ang:+.0f}, reverse {leg_mm:.0f} mm")
            got = self._turn_to_heading(ang, lat)
            if got is None:
                return False
            if self._odo_move(-leg_mm, -leg_mm, 60, timeout=8) is None:
                return False
            if not self._auto_pause(CENTRE_SETTLE_S):
                return False
            got = self._turn_to_heading(-ang, got[0])
            if got is None:
                return False
            lat, out, head = got
            # forward leg: the same distance back in at -ang (this is what cancels the sideways
            # offset), then face the wall and drive straight in to the turn spot
            self._caption(f"zigzag {rnd}: now {lat:+.1f} cm off -> forward {leg_mm:.0f} mm at {-ang:+.0f} deg")
            if not self.link.obstacle_ahead():
                if self._odo_move(leg_mm, leg_mm, 60, timeout=8) is None:
                    return False
            got = self._turn_to_heading(0.0, lat)
            if got is None:
                return False
            lat, out, head = got
            fwd_mm = max(0.0, (out - target_cm) * 10.0)
            self._caption(f"zigzag {rnd}: {lat:+.1f} cm off, {out:.0f} cm out -> straight in {fwd_mm:.0f} mm")
            if fwd_mm > 5 and not self.link.obstacle_ahead():
                if self._odo_move(fwd_mm, fwd_mm, ODO_DRIVE_SPEED, timeout=20) is None:
                    return False
            if not self._auto_pause(CENTRE_SETTLE_S):
                return False
            both = self._pose_both(n=6)
            if both is None:
                return False
            near = lambda p: abs(p[0] - lat)
            lat, out, head = both[2] if near(both[2]) <= near(both[3]) else both[3]
        return abs(lat - LINE_TARGET_CM) <= LINE_TOL_CM

    def _approach_fail(self, why: str, finish: bool) -> bool:
        if self._auto_cancel.is_set():
            return False
        if finish:
            self._finish(f"approach: {why}")
        else:
            self._caption(f"approach: {why}")
        return False

    # ------------------------------------------------------------------ docking: blind half

    def charging(self) -> bool:
        try:
            if self._ina is None:
                self._ina = INA226()
            return self._ina.current() * CHARGE_CURRENT_SIGN > CHARGING_THRESHOLD_A
        except OSError:
            return False

    def battery(self):
        """(volts, percent, charging) from the INA226, or None."""
        if self.battery_fn is not None:
            return self.battery_fn()
        try:
            if self._ina is None:
                self._ina = INA226()
            from battery_monitor import voltage_to_percent
            v = self._ina.bus_voltage()
            return v, voltage_to_percent(v), self.charging()
        except OSError:
            return None

    def _start_auto(self, target, *args):
        if self.link.halted:
            self._caption('stopped from Telegram - send "resume" first')
            return
        self.task = getattr(target, "__name__", "")
        self._cancel_auto()
        with self._lock:
            self.mode = "auto"
            self._drive(0, 0)
        self._auto_cancel.clear()

        def run():
            hook = self.camera_hook
            if hook is not None:
                hook.acquire()
            try:
                target(*args)
            finally:
                if hook is not None:
                    hook.release()
        self._auto_thread = threading.Thread(target=run, daemon=True)
        self._auto_thread.start()

    def _finish(self, text: str):
        if self._auto_cancel.is_set():
            return
        with self._lock:
            self._drive(0, 0)
            self.mode = "idle"
        self._caption(text)

    def _still_frame(self):
        """A camera frame taken after the robot has stopped moving."""
        if not self._auto_pause(0.8):
            return None
        for _ in range(10):
            img = grab()
            if img is not None:
                return img
            if not self._auto_pause(0.1):
                return None
        return None

    def start_move(self, mm: float):
        """Drive straight `mm` (+ forward) on odometry at 80 mm/s; the front IR still stops it."""
        def run():
            got = self._odo_move(mm, mm, ODO_DRIVE_SPEED, timeout=25)
            if got is not None:
                self._finish(f"move: drove {got:.0f} of {abs(mm):.0f} mm")
        self._start_auto(run)

    def start_turn(self, deg: float):
        """Turn in place on odometry (+ = left)."""
        def run():
            got = self._odo_turn(abs(deg), deg > 0)
            if got is not None:
                l, r = self.link.odom()
                self._finish(f"turn: odometry {got:.1f} deg (counts L {l:+d} R {r:+d})")
        self._start_auto(run)

    def start_spin_calibration(self):
        self._start_auto(self._spin_calibration)

    def _spin_calibration(self):
        """With the tags in view and centred: small odometry turns (+-20 deg) measured by the
        tag heading. Real/odometry turn ratio gives the effective wheel spacing."""
        global WHEEL_SPACING_MM
        tag = self._scan_for_tag()
        if tag is None:
            self._finish("spin cal: can't find the tags")
            return
        if not self._centre_on_tag(finish=False):
            self._finish("spin cal: couldn't centre on the tags")
            return
        ratios = []
        for k, left in enumerate((True, False, True, False, False, True)):
            p0 = self._pose(n=10)
            if p0 is None:
                break
            turned = self._odo_turn(10, left)
            if turned is None:
                return
            if not self._auto_pause(0.8):
                return
            p1 = self._pose(n=10)
            if p1 is None:
                self._caption(f"spin cal: lost the tags on turn {k + 1}")
                break
            real = abs(p1[6] - p0[6])          # heading change in the tags' frame = real turn
            if turned > 3:
                ratios.append(real / turned)
            self._caption(f"spin cal {k + 1}: odometry {turned:.1f} deg, tags say {real:.1f} deg")
            print(time.strftime("%H:%M:%S"), f"[spincal] {'L' if left else 'R'} odo {turned:.1f} real {real:.1f}", flush=True)
        if len(ratios) < 3:
            self._finish("spin cal: not enough good turns")
            return
        ratio = statistics.median(ratios)
        if not 0.6 < ratio < 1.6:
            self._finish(f"spin cal: real/odometry {ratio:.3f} is implausible (wheels slipping or off the "
                         f"ground?) - keeping {WHEEL_SPACING_MM:.0f} mm")
            return
        new_spacing = WHEEL_SPACING_MM / ratio
        self._finish(f"spin cal: real/odometry {ratio:.3f} over {len(ratios)} turns -> effective spacing "
                     f"{new_spacing:.0f} mm (was {WHEEL_SPACING_MM:.0f}); now using it")
        WHEEL_SPACING_MM = new_spacing

    # ------------------------------------------------------------------ docking, step by step

    def start_step_lineup(self):
        self._start_auto(self._step_lineup)

    def _step_lineup(self):
        """Step 1: find the tags and get onto the dock's line, facing the wall - from wherever it
        is (best 60-90 cm out, where there's room to sidestep). No driving in. Sidesteps start from
        the current heading, so it never has to look away from the tags while still off to the side."""
        if self._scan_for_tag() is None:
            if self._auto_cancel.is_set():
                return
            if self.lost_tag_search is None or not self.lost_tag_search.run():
                if not self._auto_cancel.is_set():
                    self._finish("step 1: no tags in a full turn" if self.lost_tag_search is None
                                 else "step 1: no tags found after searching the room")
                return
        self.tag_side = None
        if not self._which_side():
            self._finish("step 1: lost the tags during the side check")
            return
        p = self._line_up("step 1")
        if p is None:
            return
        self._finish(f"STEP 1 DONE: lined up - {p[5]:.0f} cm out, {p[4] - LINE_TARGET_CM:+.1f} cm from the "
                     f"dock line, heading {p[6]:+.1f} deg")

    def _close_in(self, step: str) -> bool:
        """Far out (beyond LINEUP_MAX_OUT_CM): head straight for the tags first, in short legs,
        so the sidesteps happen in the clear space near the dock (2026-10-17: sidestepping at
        160 cm out ran into the beanbag)."""
        for _ in range(8):
            p = self._pose()
            if p is None:
                if self._scan_for_tag() is None:
                    if not self._auto_cancel.is_set():
                        self._finish(f"{step}: lost the tags")
                    return False
                continue
            dist, bearing, out = p[0], p[2], p[5]
            if out <= LINEUP_MAX_OUT_CM:
                return True
            if abs(bearing) > 4:                       # point at the tags (small odo turns overshoot)
                if self._odo_turn(min(abs(bearing), 30) * 0.6, bearing < 0) is None:
                    return False
                continue
            leg = min(30.0, dist - LINEUP_TARGET_OUT_CM) * 10.0
            self._caption(f"{step}: {out:.0f} cm out, {p[4] - LINE_TARGET_CM:+.0f} cm off the line -> "
                          f"closer to the tags first ({leg:.0f} mm)")
            got = self._odo_move(leg, leg, ODO_DRIVE_SPEED, stop_on=self.link.obstacle_ahead, timeout=15)
            if got is None:
                return False
            if self.link.obstacle_ahead():
                self._finish(f"{step}: front IR sees something on the way to the tags - stopped")
                return False
        return True

    def _line_up(self, step: str, rounds: int = 6):
        """Sidestep onto the dock line and face the wall; succeed only if it's still on the line
        after facing the wall (2026-10-17: it called -12 cm "lined up" and docked on ID 0).
        Returns the final pose, or None after calling _finish with the reason."""
        if not self._close_in(step):
            return None
        err = None
        for rnd in range(1, rounds + 1):
            p = self._pose()
            if p is None:
                if self._scan_for_tag() is None:
                    if not self._auto_cancel.is_set():
                        self._finish(f"{step}: lost the tags")
                    return None
                continue
            err = p[4] - LINE_TARGET_CM
            if abs(err) <= LINE_TOL_CM:
                if not self._face_wall():
                    if not self._auto_cancel.is_set():
                        self._finish(f"{step}: couldn't face the wall")
                    return None
                p = self._pose()
                if p is not None and abs(p[4] - LINE_TARGET_CM) <= LINE_TOL_CM:
                    return p
                continue                          # facing the wall moved it off: go again
            if not self._sidestep(err, p[6]):
                if not self._auto_cancel.is_set():
                    self._finish(f"{step}: sidestep stopped" + (" - something in front"
                                                                 if self.link.obstacle_ahead() else ""))
                return None
            if self._pose() is None and self._scan_for_tag() is None:
                self._finish(f"{step}: lost the tags after a sidestep")
                return None
        self._finish(f"{step}: couldn't get onto the dock line in {rounds} tries"
                     + (f" (last {err:+.1f} cm off)" if err is not None else ""))
        return None

    def start_step_forward(self):
        self._start_auto(self._step_forward)

    def _step_forward(self, target_cm: float = APPROACH_TARGET_CM, retry: int = 0):
        """Step 2: drive straight in to the turn spot in 10 cm odometry legs, re-facing the wall
        between legs (straight driving is the robot's most accurate move)."""
        for _ in range(15):
            p = self._pose()
            if p is None and self._scan_for_tag() is not None and self._face_wall():
                p = self._pose()                    # tags were out of view: found them again
            if p is None:
                if not self._auto_cancel.is_set():
                    self._finish("step 2: lost the tags")
                return
            left_mm = (p[5] - target_cm) * 10.0
            if left_mm <= 5:
                break
            if abs(p[6]) > HEADING_TOL_DEG and not self._face_wall():
                self._finish("step 2: couldn't face the wall")
                return
            if self.link.obstacle_ahead():
                self._finish(f"step 2: front IR sees something at {p[5]:.0f} cm out - stopped")
                return
            self._caption(f"step 2: {p[5]:.0f} cm out, {p[4] - LINE_TARGET_CM:+.1f} cm off the dock line -> "
                          f"forward {min(100.0, left_mm):.0f} mm")
            if self._odo_move(min(100.0, left_mm), min(100.0, left_mm), ODO_DRIVE_SPEED, timeout=10) is None:
                return
        # (no sidestep here: close to the wall only one tag fits in the picture and the heading
        #  becomes ambiguous - all sideways correction happens farther out)
        if not self._face_wall():
            self._finish("step 2: couldn't face the wall")
            return
        p = self._pose()
        if p is not None and abs(p[4] - LINE_TARGET_CM) > STEP2_MAX_OFF_CM:
            if retry >= 2:
                self._finish(f"step 2: still {p[4] - LINE_TARGET_CM:+.1f} cm off the dock line after 2 re-line-ups")
                return
            back_mm = (STEP2_RELINE_OUT_CM - p[5]) * 10.0
            self._caption(f"step 2: {p[4] - LINE_TARGET_CM:+.1f} cm off the dock line -> back out "
                          f"{back_mm:.0f} mm and line up again")
            if back_mm > 0 and self._odo_move(-back_mm, -back_mm, ODO_DRIVE_SPEED, timeout=25) is None:
                return
            if self._line_up("step 2") is None:
                return
            return self._step_forward(target_cm, retry + 1)
        self._finish("STEP 2 DONE: at the turn spot" + (f" - {p[5]:.0f} cm out, {p[4] - LINE_TARGET_CM:+.1f} cm from "
                                                         f"the dock line, heading {p[6]:+.1f} deg" if p else ""))

    def start_step_turn(self):
        self._start_auto(self._step_turn)

    def _step_turn(self):
        """Step 3: turn ~180 (slow centre-pivot spins, measured by the camera) until the docked
        view is recognised."""
        if not self.dock_view.taught:
            self._finish("step 3: teach the docked view first")
            return
        if self._turn_to_docked_view_pulses():
            img = self._still_frame()
            loc = self.dock_view.locate(img) if img is not None else None
            self._finish("STEP 3 DONE: facing away from the dock" +
                         (f" - heading {loc[1]:+.1f} deg vs the docked view" if loc else ""))
        elif not self._auto_cancel.is_set():
            self._finish("step 3: couldn't recognise the docked view")

    def start_dock(self):
        self._start_auto(self._dock_full)

    def start_dock_sequence(self):
        self._start_auto(self._dock_sequence)

    def _dock_sequence(self):
        """Steps 1-4 back to back (low battery / the Dock button). Each step finishes with
        its own result line; the sequence stops at the first one that isn't a success."""
        if self.on_dock_start:
            self.on_dock_start()
        try:
            self._dock_steps()
        finally:
            if self.on_dock_end:
                self.on_dock_end(self.activity)

    def _dock_steps(self):
        for fn, ok in ((self._step_lineup, "STEP 1 DONE"), (self._step_forward, "STEP 2 DONE"),
                       (self._step_turn, "STEP 3 DONE"), (self._dock_back, "DOCKED")):
            if self._auto_cancel.is_set():
                return
            with self._lock:
                self.mode = "auto"
            fn()
            if not self.activity.startswith(ok):
                return

    # ------------------------------------------------------------------ odometry moves

    def _odo(self):
        l, r = self.link.odom()
        return l * MM_PER_COUNT, r * MM_PER_COUNT

    def _odo_move(self, left_mm: float, right_mm: float, speed: float, stop_on=None, timeout: float = 20.0):
        """Drive each wheel the given distance (+ forward) under the ESP32 speed control, ease to
        a stop, and return the distance actually covered (mean of |left|, |right|), or None if
        cancelled. stop_on: optional callable checked as it goes; True ends the move early."""
        span = max(abs(left_mm), abs(right_mm))
        if span < 1:
            return 0.0
        vl, vr = speed * left_mm / span, speed * right_mm / span
        l0, r0 = self._odo()
        brake = speed * speed / (2 * 200.0)          # distance the 200 mm/s^2 ramp-down takes
        with self._lock:
            self._motion = (0, 0)
            self._sent = (0, 0)
        self.link.drive_speed(int(vl), int(vr))
        t0 = time.time()
        done = 0.0
        while time.time() - t0 < timeout:
            if self._auto_cancel.wait(0.03):
                self.link.stop()
                return None
            l, r = self._odo()
            done = (abs(l - l0) + abs(r - r0)) / 2
            if done >= span - brake:
                break
            if stop_on is not None and stop_on():
                break
            st = self.link.state
            if st.last_blocked and time.time() - st.last_blocked_at < 0.5:
                break
        self.link.stop_smooth(0.6)
        l, r = self._odo()
        return (abs(l - l0) + abs(r - r0)) / 2

    def _odo_turn(self, deg: float, left: bool, speed: float = ODO_TURN_SPEED):
        """Turn in place by `deg` degrees using the wheel counts (each wheel rolls an arc of
        spacing*pi*deg/360 in opposite directions). Returns degrees turned, or None if cancelled."""
        arc = WHEEL_SPACING_MM * math.pi * deg / 360.0
        got = self._odo_move(-arc if left else arc, arc if left else -arc, speed)
        return None if got is None else got * 360.0 / (WHEEL_SPACING_MM * math.pi)

    def _turn_to_docked_view(self) -> bool:
        """From facing the dock tag: turn ~180 deg on the wheel odometry (smooth, measured), then
        nudge in small odometry turns until the docked view is recognised."""
        self._caption("turn: 180 deg on wheel odometry")
        got = self._odo_turn(ODO_TURN_DEG, left=True)
        if got is None:
            return False
        print(time.strftime("%H:%M:%S"), f"[turn] odometry turn done: {got:.0f} deg", flush=True)
        for k in range(TURN_SEARCH_PULSES):
            img = self._still_frame()
            if img is None:
                return False
            loc = self.dock_view.locate(img)
            if loc is not None and abs(loc[1]) <= TURN_FOUND_DEG:
                self._caption(f"turn: docked view recognised after ~{got:.0f} deg, heading {loc[1]:+.1f} deg")
                return True
            # not recognised yet: turn a little more each way, alternating, widening
            step = 10.0 * (k // 2 + 1)
            left = k % 2 == 0
            self._caption(f"turn: looking for the docked view - {'left' if left else 'right'} {step:.0f} deg")
            if self._odo_turn(step, left) is None:
                return False
        return False

    def _turn_to_docked_view_pulses(self) -> bool:
        """From facing the dock tag: turn ~180 deg left. The rotation is MEASURED from the
        camera (scene shift between stops) so stalled low-battery pulses don't fool it, a
        stalling turn gets more power, and the backward creep is cancelled with small
        forward pulses. From ~140 deg on it looks for the docked view after every stop."""
        turned, boost, owed = 0.0, 0, 0.0
        prev = self._still_frame()
        t0 = self._fresh_tag()
        stops = 0
        while stops < TURN_MAX_STOPS:
            stops += 1
            s = self.speed + boost
            n = TURN_STEP_PULSES if turned < TURN_BLIND_DEG else 1
            self._caption(f"turn: {turned:.0f} deg so far -> turn ~{n * TURN_DEG_PER_PULSE:.0f} deg")
            # Speed-controlled spin (both wheels turning, kept in step) so the robot pivots on its
            # centre. Short pulses let one wheel stall and it swung round the other wheel - that
            # carried it ~35 cm sideways over a half turn (2026-10-16). The angle is still
            # measured by the camera below.
            if self._odo_turn(n * TURN_DEG_PER_PULSE, True, speed=ODO_TURN_SPEED) is None:
                return False
                owed -= self.turn_creep
                if abs(owed) >= FWD_CM_PER_PULSE:
                    d = 1 if owed > 0 else -1
                    if not (self._auto_move(d * self.speed, d * self.speed, PULSE_S)
                            and self._auto_pause(SCAN_PULSE_GAP_S)):
                        return False
                    owed -= d * (FWD_CM_PER_PULSE if d > 0 else REV_CM_PER_PULSE)
            img = self._still_frame()
            if img is None:
                return False
            step = self._measured_turn(prev, img)
            prev = img
            expected = n * TURN_DEG_PER_PULSE
            if step is None:
                step = expected
            elif abs(step) < 0.4 * expected and boost < 2:
                boost += 1
            turned += abs(step)
            if t0 is not None and stops == 1:      # first stop: tag still visible -> measure the creep
                t1 = self._fresh_tag(1.0)
                if t1 is not None and len(t0) >= 7 and len(t1) >= 7:
                    self.turn_creep = max(-3.0, min(3.0, -(t1[5] - t0[5]) / n))
                    print(time.strftime("%H:%M:%S"), f"[turn] creep {self.turn_creep:+.2f} cm/pulse", flush=True)
            if turned >= TURN_BLIND_DEG:
                loc = self.dock_view.locate(img)
                if loc is not None and abs(loc[1]) <= TURN_FOUND_DEG:
                    self._caption(f"turn: docked view recognised after {turned:.0f} deg, heading {loc[1]:+.1f} deg")
                    return True
            if turned > 300:
                return False
        return False

    def _dock_full(self):
        """One button: find the tag (360 scan) -> approach to the turn spot -> turn ~180
        -> back onto the dock (docked-view heading + pre-push spot) -> final push."""
        if self.charging():
            self._finish("dock: already on the dock and charging")
            return
        if not self.dock_view.taught:
            self._finish("dock: teach the docked view first")
            return
        self._caption("dock: 1-2/4 finding and approaching the tag")
        if not self._find_and_approach():
            return
        self._caption("dock: 3/4 turning around")
        # turning in place on this floor isn't reliable from wheel counts (the robot swings round
        # its caster and slips), so the turn-around keeps the camera-measured version that worked;
        # the odometry is used for the straight moves.
        if not self._turn_to_docked_view_pulses():
            if not self._auto_cancel.is_set():
                self._finish("dock: couldn't recognise the docked view after the turn - stopped")
            return
        self._caption("dock: 4/4 backing onto the dock")
        self._dock_back()                     # finishes with DOCKED or a reason

    def start_teach(self, force: bool = False):
        self._start_auto(self._teach, force)

    def _teach(self, force: bool = False):
        """Robot sitting correctly on the dock (charging): photograph the docked view, then creep
        straight out very slowly on the speed control, taking a picture every ~1 cm measured by
        the wheel odometry."""
        if not force and not self.charging():
            self._finish("teach: not charging - put the robot on the dock first")
            return
        img = self._still_frame()
        if img is None:
            self._finish("teach: no camera frames")
            return
        shots = [(0.0, img)]
        self._caption("teach: docked view saved, creeping out slowly")
        l0, r0 = self._odo()
        next_mm = TEACH_STEP_MM
        with self._lock:
            self._motion = (0, 0)
            self._sent = (0, 0)
        self.link.drive_speed(TEACH_SPEED_MM_S, TEACH_SPEED_MM_S)
        t0 = time.time()
        while time.time() - t0 < 40:
            if self._auto_cancel.wait(0.05):
                self.link.stop()
                return
            if self.link.obstacle_ahead():
                self._caption("teach: front IR sees something - stopping early")
                break
            l, r = self._odo()
            done = ((l - l0) + (r - r0)) / 2
            if done >= next_mm:
                frame = grab(0.3)
                if frame is not None:
                    shots.append((done / 10.0, frame))
                    self._caption(f"teach: picture {len(shots) - 1} at {done / 10.0:.1f} cm")
                next_mm += TEACH_STEP_MM
            if done >= TEACH_DIST_MM:
                break
        self.link.stop_smooth(0.6)
        self.dock_view.save(shots)
        self._finish(f"teach: saved {len(shots)} pictures (0-{shots[-1][0]:.0f} cm, every ~{TEACH_STEP_MM / 10:.0f} cm by "
                     f"odometry). Try 'Back onto dock' from here.")

    def start_dock_back(self):
        self._start_auto(self._dock_back)

    def _dock_back(self):
        """Facing away from the dock, roughly lined up: keep the heading matched to the docked
        view and reverse in short continuous runs until charge current appears; on contact keep
        pushing 3 s, release, and check it holds. (Distance from picture size proved unreliable
        in lamp light at night, so the charge current is the only stop signal - with a budget.)"""
        if not self.dock_view.taught:
            self._finish("back: no docked view taught yet")
            return
        s = self.speed
        min_turn = BACK_MIN_TURN_S
        last_turn_heading = None
        runs = 0
        # per-wheel reverse speed: start from what worked last time (the left wheel needed more
        # in reverse on 2026-10-15), then nudge a lagging wheel up as runs show it twisting
        rev = dict(self.reverse_balance)
        stalled_rounds = 0
        before_run = None                 # heading just before the last reverse run
        img_before = None                 # picture just before the last reverse run
        while runs < BACK_MAX_RUNS:
            if self.charging():
                if self._held_charging(SEAT_STABLE_S):
                    self._finish(f"DOCKED - charging steadily (after {runs} reverse runs)")
                    return
            img = self._still_frame()
            if img is None:
                return
            loc = self.dock_view.locate(img)
            if loc is None:
                self._finish("back: can't match the docked view - stopped")
                return
            _, heading, n = loc
            if img_before is not None:
                # did the last run actually move it back? (scene shrinks as it backs toward the dock)
                r = scale_to(features(img), features(img_before))
                img_before = None
                if r is not None and abs(r[0] - 1.0) < BACK_STALL_SCALE and max(rev.values()) < BACK_MAX_WHEEL:
                    rev["l"] += 1
                    rev["r"] += 1
                    self._caption(f"back: last run didn't move it (low battery?) -> reverse speeds L{rev['l']} R{rev['r']}")
            if before_run is not None:
                twist = heading - before_run      # how much the last reverse run turned the robot
                before_run = None
                if twist < -BACK_TWIST_DEG and rev["r"] < BACK_MAX_WHEEL:
                    rev["r"] += 1                 # swung left while reversing: right wheel lagging
                elif twist > BACK_TWIST_DEG and rev["l"] < BACK_MAX_WHEEL:
                    rev["l"] += 1                 # swung right: left wheel lagging
                if abs(twist) > BACK_TWIST_DEG:
                    self._caption(f"back: last run twisted {twist:+.1f} deg -> reverse speeds L{rev['l']} R{rev['r']}")
            if abs(heading) > BACK_TOL_DEG:
                if last_turn_heading is not None and abs(heading - last_turn_heading) < 0.3:
                    min_turn = min(0.15, min_turn + 0.02)
                dur = 0.15 if abs(heading) > 8 else max(min_turn, 0.1 if abs(heading) > 4 else min_turn)
                side = "left" if heading > 0 else "right"
                self._caption(f"back: heading {heading:+.1f} deg ({n} matches) -> turn {side} {dur:.2f}s")
                last_turn_heading = heading
                # slow speed-controlled spin (pivots on the centre; pulses swung round one wheel and
                # shifted the robot sideways). The odometry over-reads turns, so ask for less.
                if self._odo_turn(min(abs(heading), 15.0) * 0.6, heading > 0) is None:
                    return
                continue
            last_turn_heading = None
            runs += 1
            self._caption(f"back: heading {heading:+.1f} deg -> reverse run {runs}/{BACK_MAX_RUNS} "
                          f"(stops on charge current, then holds {PUSH_HOLD_S:.0f} s)")
            before_run = heading
            img_before = img
            # back up in short, very gentle pulses (speed-controlled, eased stop so the wheels don't
            # slip), stopping at charge current; a pulse that barely moves = touching the dock
            moved = 0.0
            for _ in range(BACK_PULSES_PER_CHECK):
                l0, r0 = self._odo()
                with self._lock:
                    self._motion = (0, 0)
                    self._sent = (0, 0)
                self.link.drive_speed(-BACK_PULSE_SPEED, -BACK_PULSE_SPEED)
                if self._auto_cancel.wait(BACK_PULSE_S):
                    self.link.stop()
                    return
                self.link.stop_smooth(0.4)
                l1, r1 = self._odo()
                step = (abs(l1 - l0) + abs(r1 - r0)) / 2
                moved += step
                if self.charging() or step < 2.0:
                    break
            if self._auto_cancel.is_set():
                return
            after = self._still_frame()
            loc2 = self.dock_view.locate(after) if after is not None else None
            twist = None if loc2 is None else loc2[1] - heading
            if twist is not None and abs(twist) > BACK_TWIST_ABORT_DEG:
                self._finish(f"back: twisted {twist:+.1f} deg in one step - hit something, stopped (check the contacts)")
                return
            stalled_rounds = stalled_rounds + 1 if moved < 4.0 else 0
            if not self.charging() and stalled_rounds >= 2:
                if WIGGLE_ENABLED and self._wiggle_onto_pads():
                    return
                if not self._auto_cancel.is_set():
                    self._finish(f"back: touching the dock after {runs} steps but no charge current, even after "
                                 f"wiggling (charger off, or contacts too far off?)")
                return
            contact = self.charging()
            if contact:
                contact = self._push(0.5, rev["l"], rev["r"]) or True   # the 3 s hold that keeps it on the pads
            if self._auto_cancel.is_set():
                return
            if contact or self.charging():
                self._caption("back: contact - checking it holds")
                if self._held_charging(SEAT_STABLE_S):
                    self.reverse_balance = dict(rev)
                    self._finish(f"DOCKED - charging steadily (after {runs} reverse runs, L{rev['l']} R{rev['r']})")
                    return
        self.reverse_balance = {k: max(1, v - 1) for k, v in rev.items()}   # keep the lean, drop the overshoot
        self._finish(f"back: no steady charge after {runs} reverse runs (~{runs * BACK_RUN_S * 13:.0f} cm) - stopped")

    def _wiggle_onto_pads(self) -> bool:
        """Touching the dock but not charging - the contacts are a few mm off the pads. Ease out a
        little, turn a hair one way, ease back in until it touches, check; then the other way, a bit
        more each round. True once charging holds 3 s (or if cancelled)."""
        for k, (deg, left) in enumerate(WIGGLE_SEQUENCE, 1):
            self._caption(f"wiggle {k}/{len(WIGGLE_SEQUENCE)}: out a little, turn {deg:.0f} deg "
                          f"{'left' if left else 'right'}, back in")
            if self._odo_move(WIGGLE_OUT_MM, WIGGLE_OUT_MM, 20, timeout=4) is None:
                return True
            if self._odo_turn(deg, left, speed=15) is None:
                return True
            if self._odo_move(-(WIGGLE_OUT_MM + 15), -(WIGGLE_OUT_MM + 15), 15, stop_on=self.charging,
                              timeout=6) is None:
                return True
            if self.charging() and self._held_charging(SEAT_STABLE_S):
                self._finish(f"DOCKED - charging steadily (found the pads on wiggle {k})")
                return True
            if self._odo_turn(deg, not left, speed=15) is None:     # undo the turn before the next try
                return True
        return False

    def _push(self, max_s: float, left: int = None, right: int = None) -> bool:
        """One continuous reverse: stops PUSH_HOLD_S after charge current appears, or at
        max_s. True if contact was made. left/right: per-wheel reverse speeds (default speed)."""
        s = self.speed
        ls = left if left is not None else s
        rs = right if right is not None else s
        contact_at = None
        with self._lock:
            if self._auto_cancel.is_set():
                return False
            self._drive(-ls, -rs)
        t0 = rearm = time.time()
        while time.time() - t0 < max_s or contact_at is not None:
            if contact_at is None and self.charging():
                contact_at = time.time()
                self._caption(f"push: contact - holding the push {PUSH_HOLD_S:.0f} s")
            if contact_at is not None and time.time() - contact_at >= PUSH_HOLD_S:
                break
            if time.time() - rearm >= REVERSE_REARM_S:
                with self._lock:            # keep reversing past the ESP32's 1.5 s reverse limit
                    self.link.drive(0, 0)
                    self.link.drive(-ls, -rs)
                rearm = time.time()
            if self._auto_cancel.wait(0.01):
                break
        with self._lock:
            self._drive(0, 0)
        return contact_at is not None

    def save_prepush(self):
        """Robot placed where a final push lands on the pads (~2 cm out): remember the view."""
        if self.mode != "idle":
            self._caption("pre-push: stop the robot first")
            return
        img = grab()
        if img is None:
            self._caption("pre-push: no camera frame")
            return
        self.dock_view.save_prepush(img)
        self._caption("pre-push spot saved - Back onto dock will push from here")

    def start_final_push(self):
        self._start_auto(self._final_push_only)

    def _final_push_only(self):
        """Robot placed a few cm from the pads, facing away: line up and push."""
        if not self.dock_view.taught:
            self._finish("push: no docked view taught yet")
            return
        if not self._final_push(0, None):
            self._finish(f"push: no steady charge after {PUSH_TRIES} pushes - stopped")

    def _heading_now(self):
        img = self._still_frame()
        if img is None:
            return None
        loc = self.dock_view.locate(img)
        return None if loc is None else loc[1]

    def _align_tight(self):
        """Turn in the smallest moving pulses until the heading is within PUSH_ALIGN_TOL_DEG.
        Returns the final heading (or None if the view can't be matched)."""
        s = self.speed
        dur = BACK_MIN_TURN_S
        heading = self._heading_now()
        for _ in range(PUSH_ALIGN_TRIES):
            if heading is None or abs(heading) <= PUSH_ALIGN_TOL_DEG:
                break
            self._caption(f"push: lining up first, heading {heading:+.1f} deg -> turn {dur:.2f}s")
            l, r = (-s, s) if heading > 0 else (s, -s)
            if not self._auto_move(l, r, dur):
                return None
            new = self._heading_now()
            if new is not None and abs(new - heading) < 0.3:
                dur = min(0.15, dur + 0.02)      # didn't move: slightly longer pulse
            heading = new
        return heading

    def _final_push(self, reverses: int, heading) -> bool:
        """Close in: line up tightly, continuous push onto the pads, check it holds, retry a
        couple of times if it bounced off. True when docked (or cancelled)."""
        for attempt in range(1, PUSH_TRIES + 1):
            heading = self._align_tight()
            if self._auto_cancel.is_set():
                return True
            if heading is None:
                self._finish("push: can't match the docked view - stopped")
                return True
            self._caption(f"push: final reverse onto the pads (try {attempt}, heading {heading:+.1f} deg)")
            contact = self._push(PUSH_MAX_S if attempt == 1 else 0.3)
            if self._auto_cancel.is_set():
                return True
            after = None if contact else self._heading_now()
            if after is not None:
                self._caption(f"push: missed - heading went {heading:+.1f} -> {after:+.1f} deg during the push")
                print(time.strftime("%H:%M:%S"), f"[push] veer {after - heading:+.1f} deg", flush=True)
            if contact or self.charging():
                self._caption(f"push: contact - checking it holds (try {attempt})")
                if self._held_charging(SEAT_STABLE_S):
                    self._finish(f"DOCKED - charging steadily (after {reverses} pulses + final push, try {attempt})")
                    return True
        return False

    def _held_charging(self, seconds: float) -> bool:
        """True if, after the bounce has settled, charge current is on for at least
        SEAT_MIN_FRACTION of `seconds` (sampled every 0.1 s). The robot often bounces off
        the pads and then rolls back on by itself, so judging right at contact fails it."""
        if self._auto_cancel.wait(SEAT_SETTLE_S):
            return False
        on = total = 0
        end = time.time() + seconds
        while time.time() < end:
            total += 1
            on += self.charging()
            if self._auto_cancel.wait(0.1):
                return False
        return total > 0 and on / total >= SEAT_MIN_FRACTION

    def _seat(self, reverses: int) -> bool:
        """Contact made: settle onto the pads. Short reverse nudges whenever the bounce
        breaks contact, until charging holds for SEAT_STABLE_S. True when docked."""
        s = self.speed
        for nudge in range(SEAT_MAX_NUDGES + 1):
            self._caption(f"seat: contact - checking it holds ({nudge} nudges so far)")
            if self._held_charging(SEAT_STABLE_S):
                self._finish(f"DOCKED - charging steadily (after {reverses} reverse pulses, {nudge} nudges)")
                return True
            if self._auto_cancel.is_set():
                return True
            if nudge == SEAT_MAX_NUDGES:
                break
            # contact lost (bounced off): one short nudge back, stop as soon as current returns
            with self._lock:
                self._drive(-s, -s)
            end = time.time() + SEAT_NUDGE_S
            while time.time() < end and not self.charging():
                if self._auto_cancel.wait(0.01):
                    return True
            with self._lock:
                self._drive(0, 0)
        return False

    def _fresh_tag(self, wait_s: float = 1.5):
        end = time.time() + wait_s
        while time.time() < end:
            t = read_tag(0.25)
            if t:
                return t
            if self._auto_cancel.wait(0.05):
                return None
        return None

    def _centre_on_tag(self, max_pulses: int = 40, finish: bool = True) -> bool:
        """Turn in place in short pulses until the tag sits in the middle of the picture
        (bearing within CENTRE_TOL_DEG). Pulses shrink as it gets close; the smallest
        pulse grows if it turns out too short to move the wheels."""
        s = self.speed
        min_pulse = CENTRE_MIN_PULSE_S
        result = "gave up after too many pulses"
        centred = False
        prev = None
        for n in range(max_pulses):
            if not self._auto_pause(CENTRE_SETTLE_S):
                return False
            tag = self._fresh_tag()
            if tag is None:
                result = "lost the tag"
                break
            b = tag[2]
            if prev is not None and abs(b - prev) < 0.3 and abs(prev) <= 3:
                min_pulse = min(0.15, min_pulse + 0.02)        # last tiny pulse didn't move it
            prev = b
            if abs(b) <= CENTRE_TOL_DEG:
                result = f"CENTRED: bearing {b:+.1f} deg, {tag[0]:.0f} cm, tag yaw {tag[3]:+.1f} deg"
                centred = True
                break
            dur = 0.15 if abs(b) > 8 else 0.08 if abs(b) > 3 else min_pulse
            side = "right" if b > 0 else "left"                 # + bearing = tag is right of centre
            self._caption(f"centre: tag {b:+.1f} deg -> turn {side} {dur:.2f}s")
            l, r = (s, -s) if b > 0 else (-s, s)
            if not self._auto_move(l, r, dur):
                return False
        if self._auto_cancel.is_set():
            return False
        if finish:
            self._finish(f"centre: {result}")
        else:
            self._caption(f"centre: {result}")
        return centred

    def _cancel_auto(self):
        self._auto_cancel.set()
        t = self._auto_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2)
        self._auto_thread = None

    def _auto_move(self, left: int, right: int, seconds: float) -> bool:
        """Drive for `seconds` unless cancelled; False if cancelled."""
        with self._lock:
            if self._auto_cancel.is_set():
                return False
            self._drive(left, right)
        end = time.time() + seconds
        while time.time() < end:
            if self._auto_cancel.wait(0.05):
                return False
        with self._lock:
            if not self._auto_cancel.is_set():
                self._drive(0, 0)
        return not self._auto_cancel.is_set()

    def _auto_pause(self, seconds: float) -> bool:
        return not self._auto_cancel.wait(seconds)

    def _explore(self, minutes: float = 4.0):
        """Robot-vacuum style: forward until the front IR sees something, then back up a
        little (at most twice in a row) and turn until clear. Stops when the dock tag is seen."""
        s = self.speed
        deadline = time.time() + minutes * 60
        turn_dir = random.choice([-1, 1])
        obstacles_in_row = backups_in_row = 0
        result = "time up, tag not found"
        while time.time() < deadline and not self._auto_cancel.is_set():
            tag = read_tag()
            if tag:
                result = f"TAG FOUND {tag[0]:.0f} cm, bearing {tag[2]:+.0f} deg"
                break
            if not self.link.connected:
                result = "ESP32 link lost"
                break
            if self.link.obstacle_ahead():
                obstacles_in_row += 1
                if obstacles_in_row > MAX_OBSTACLES_IN_ROW:
                    result = "stuck - obstacle won't clear, please help"
                    break
                side = "left" if turn_dir < 0 else "right"
                if backups_in_row < MAX_BACKUPS_IN_ROW:
                    backups_in_row += 1
                    self._caption(f"explore: obstacle - back up a little + turn {side}")
                    if not (self._auto_pause(0.3) and self._auto_move(-s, -s, BACKUP_S)):
                        break
                else:
                    self._caption(f"explore: obstacle - turn {side} (no more backing up)")
                if not (self._auto_pause(0.3) and self._auto_move(turn_dir * s, -turn_dir * s, TURN_S)
                        and self._auto_pause(0.6)):
                    break
                continue
            obstacles_in_row = backups_in_row = 0
            self._caption(f"explore: forward @ {s}")
            with self._lock:
                if self._auto_cancel.is_set():
                    break
                self._drive(s, s)
            t_move = time.time()
            while not self._auto_cancel.wait(0.1):
                if read_tag() or self.link.obstacle_ahead() or not self.link.connected:
                    break
                if time.time() > deadline:
                    break
                if time.time() - t_move > STRAIGHT_RUN_S:
                    turn_dir = random.choice([-1, 1])
                    self._auto_move(turn_dir * s, -turn_dir * s, TURN_S * 2)
                    break
            with self._lock:
                if not self._auto_cancel.is_set():
                    self._drive(0, 0)
            if not self.link.obstacle_ahead():
                turn_dir = random.choice([-1, 1])
        if self._auto_cancel.is_set():
            return                                  # taken over / stopped - they set the caption
        with self._lock:
            self._drive(0, 0)
            self.mode = "idle"
        self._caption(f"explore: {result}")
