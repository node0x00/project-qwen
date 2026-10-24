"""
Project Qwen - battery, charging and return-to-dock (inside assistant.py; replaces charge_manager.py).

  - Battery LEDs, solid at every level, charging or not (owner's spec 2026-10-17):
        red at 25% or lower, yellow 26-50%, green above 50%.
    The ESP32 firmware pulses the LEDs while "charging", switches to red at 20% and blinks
    below 5%, so the Pi sends what makes it show the spec instead: charging always 0, 21-25%
    sent as 20 (red), and never below 6 (no blink).
  - Says "Charging." when charging starts; at ~90% says "Battery full." and drives forward a
    little off the pads (the charger lifts the voltage, so 90% is a charging-voltage threshold).
  - At 25% or lower, not charging, robot idle: "Low battery, returning to dock." and runs the
    docking steps 1-4. If the 360 scan finds no tags it says so and alerts the owner on Telegram
    (no wandering). Other failures retry up to DOCK_ATTEMPTS times, then alert.
"""

import threading
import time

from battery_monitor import CHARGE_CURRENT_SIGN, CHARGING_THRESHOLD_A, INA226, voltage_to_percent

SAMPLE_S = 1.0
CHANGE_SAMPLES = 3            # charging must look the same this many samples in a row to count
BATT_SEND_S = 10.0            # the ESP32 shows "no data" if BATT is older than 60 s
FULL_CHARGING_V = 28.75       # ~90%: charging voltage held this long (28.55 V at rest)
FULL_HOLD_S = 60
FULL_PERCENT = 90
MOVE_OFF_MM = 130             # off the pads when full (same as the old speed 1 for 1 s)
LOW_PERCENT = 25
LOW_HOLD_S = 30               # must stay low this long with the motors still (they sag the voltage)
MANUAL_QUIET_S = 60           # don't take over within this long of someone driving it
DOCK_ATTEMPTS = 3
RETRY_BACK_OFF_MM = 400       # after touching the dock without charge: drive out before trying again
INFO_FILE = "/dev/shm/slam_view.jpg.info"
# Charging guard: while the robot drives itself, charging current that holds this long ends the
# move at once - it is docked, however it got there (e.g. lifted onto the pads by hand).
GUARD_SAMPLE_S = 0.15
GUARD_HOLD_S = 0.6


def led_percent(pct: float) -> int:
    """What to send so the firmware's thresholds (red <=20, critical blink <=5) show the
    owner's (red <=25, solid)."""
    p = int(round(pct))
    if p <= LOW_PERCENT:
        p = min(p, 20)
    return max(6, min(100, p))


class PowerManager:
    def __init__(self, link, robot, speak, notify=None):
        """link: ESP32Link. robot: RobotController. speak(text). notify(text) -> Telegram."""
        self.link = link
        self.robot = robot
        self.speak = speak
        self.notify = notify or (lambda text: None)
        self.ina = INA226()
        self.volts = None
        self.percent = None
        self.charging = None
        self.full = False
        self._stop = threading.Event()
        self._thread = None
        self._low_since = None
        self._returning = False       # a low-battery docking run is in progress / done
        self._gave_up = False         # stop trying until charged (or someone docks it)

    def reading(self):
        if self.volts is None:
            return None
        return self.volts, self.percent, bool(self.charging)

    def start(self):
        self.robot.battery_fn = self.reading
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        threading.Thread(target=self._charge_guard, daemon=True).start()

    def _charge_guard(self):
        # Only arms once the auto task has been off the charger, so a task that starts ON
        # the charger (driving off the pads when full) is not stopped by its own charge current.
        since, armed = None, False
        while not self._stop.wait(GUARD_SAMPLE_S):
            if self.robot.mode != "auto":
                since, armed = None, False
                continue
            try:
                charging = self.ina.current() * CHARGE_CURRENT_SIGN > CHARGING_THRESHOLD_A
            except OSError:
                continue
            if not charging:
                since, armed = None, True
                continue
            if not armed:
                continue
            since = since or time.time()
            if time.time() - since >= GUARD_HOLD_S and self.robot.mode == "auto":
                print("[power] charging while driving -> motors off, docked", flush=True)
                self.robot.stop("DOCKED - charging")
                since = None

    def stop(self):
        self._stop.set()

    def demo_full(self):
        """Control-page demo: exactly what happens at ~90 %, whatever the real charge."""
        print("[power] DEMO: battery full -> driving off the pads", flush=True)
        self.full = True
        self.link.send_battery(FULL_PERCENT, False)     # solid green
        self.speak("Battery full.")
        if self.robot.mode == "idle":
            self.robot.start_move(MOVE_OFF_MM)

    def describe(self) -> str:
        if self.volts is None:
            return "battery unknown"
        return f"{self.percent:.0f}%{' charging' if self.charging else ''} ({self.volts:.1f} V)"

    # ------------------------------------------------------------------ loop

    def _run(self):
        pending, pending_n = None, 0
        above_since = None
        last_send = last_log = last_info = 0.0
        while not self._stop.wait(SAMPLE_S):
            try:
                v = self.ina.bus_voltage()
                amps = self.ina.current() * CHARGE_CURRENT_SIGN
            except OSError as e:
                print(f"[power] INA226 read failed: {e}", flush=True)
                continue
            now_charging = amps > CHARGING_THRESHOLD_A
            self.volts, self.percent = v, voltage_to_percent(v)

            if self.charging is None:
                self.charging = now_charging
                print(f"[power] start: {v:.2f} V {amps:+.3f} A {'charging' if now_charging else 'not charging'}",
                      flush=True)
            elif now_charging != self.charging:
                pending_n = pending_n + 1 if pending == now_charging else 1
                pending = now_charging
                if pending_n >= CHANGE_SAMPLES:
                    self.charging, pending_n = now_charging, 0
                    self.full, above_since = False, None
                    print(f"[power] {'charger connected' if now_charging else 'charger disconnected'} "
                          f"({v:.2f} V, {amps:+.3f} A)", flush=True)
                    if now_charging:
                        self._returning = self._gave_up = False
                        self.speak("Charging.")
            else:
                pending_n = 0

            # full at ~90%: announce and drive off the pads
            if self.charging and not self.full:
                if v >= FULL_CHARGING_V:
                    above_since = above_since or time.time()
                    if time.time() - above_since >= FULL_HOLD_S:
                        self.full = True
                        print(f"[power] FULL: {v:.2f} V at {amps:.3f} A - driving off the pads", flush=True)
                        self.speak("Battery full.")
                        if self.robot.mode == "idle":
                            self.robot.start_move(MOVE_OFF_MM)
                else:
                    above_since = None

            self._check_low()

            now = time.time()
            if now - last_send >= BATT_SEND_S:
                self.link.send_battery(FULL_PERCENT if self.full else led_percent(self.percent), False)
                last_send = now
            if now - last_info >= 5:
                self._write_info()
                last_info = now
            if now - last_log >= 300:
                print(f"[power] {v:.2f} V {amps:+.3f} A {self.describe()}", flush=True)
                last_log = now

    def _write_info(self):
        """Battery and Pi temperature for the bottom-right of the docking view."""
        parts = [f"batt {self.percent:.0f}%{' charging' if self.charging else ''} ({self.volts:.1f} V)"]
        try:
            parts.append(f"Pi {int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000:.0f} C")
        except OSError:
            pass
        try:
            with open(INFO_FILE, "w") as f:
                f.write("  ".join(parts))
        except OSError:
            pass

    # ------------------------------------------------------------------ return to dock

    def _check_low(self):
        r = self.robot
        if r.link.halted:
            return                                  # emergency stop: no driving off on our own
        quiet = r.mode == "idle" and time.time() - r._last_manual > MANUAL_QUIET_S
        if self.charging or self.percent > LOW_PERCENT or not quiet:
            self._low_since = None
            if self.percent > LOW_PERCENT + 5:
                self._gave_up = False
            return
        if self._returning or self._gave_up:
            return
        self._low_since = self._low_since or time.time()
        if time.time() - self._low_since < LOW_HOLD_S:
            return
        self._returning = True
        threading.Thread(target=self._return_to_dock, daemon=True).start()

    def _wait_robot(self):
        time.sleep(2)
        while self.robot.mode == "auto" and not self._stop.is_set():
            time.sleep(0.5)
        return self.robot.activity

    def _return_to_dock(self):
        pct = self.percent
        print(f"[power] low battery ({pct:.0f}%) - returning to dock", flush=True)
        self.speak("Low battery, returning to dock.")
        self.notify(f"Low battery ({pct:.0f}%) - returning to the dock.")
        result = ""
        try:
            for attempt in range(1, DOCK_ATTEMPTS + 1):
                if self.robot.mode == "manual":
                    self._gave_up = True               # someone took over: leave it to them
                    return
                self.robot.start_dock_sequence()
                result = self._wait_robot()
                print(f"[power] dock attempt {attempt}: {result}", flush=True)
                if result.startswith("DOCKED") or self.charging:
                    self.notify(f"Docked and charging ({self.describe()}).")
                    return
                if "no tags" in result:             # full turn, then the ORB-SLAM search
                    self.speak("I can't find the dock.")
                    self.notify(f"Low battery ({self.describe()}) and I can't find the dock - "
                                f"no tags in a full turn. Please help me back.")
                    self._gave_up = True
                    return
                if self.robot.mode == "manual" or "STOP" in result:
                    self._gave_up = True               # someone stepped in: don't restart on our own
                    return
                if "touching the dock" in result and attempt < DOCK_ATTEMPTS:
                    self.robot.start_move(RETRY_BACK_OFF_MM)
                    self._wait_robot()
            self.speak("I couldn't dock.")
            self.notify(f"Low battery ({self.describe()}) - docking failed {DOCK_ATTEMPTS} times "
                        f"(last: {result}). Please help me back.")
            self._gave_up = True
        finally:
            self._returning = False
