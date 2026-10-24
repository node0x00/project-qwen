"""
Project Qwen - Pi <-> ESP32 serial link.

The ESP32 owns all real-time hardware (motors, IR obstacle reflex, scissor
lift, LEDs) and is wired directly to the Pi's UART: Pi GPIO14/15
(/dev/serial0 -> ttyAMA0, Bluetooth disabled) <-> ESP32 Serial2. Plain-text,
newline-terminated, 115200 baud - see esp32-firmware/README.md for the full
protocol.

This module:
  - runs a background reader thread that parses ESP32 messages (IR, LIFT_POS,
    POT, BLOCKED, PONG) into shared state,
  - re-sends the active motor command every MOTOR_KEEPALIVE_S while a
    movement is active (the ESP32 stops the motors after 500 ms of silence -
    that watchdog is the safety net if this process dies),
  - PINGs periodically and warns once by voice if the ESP32 stops answering.

Safety reflexes (obstacle stop, reverse limits) live on the ESP32, never
here - this side only reports what the ESP32 decided.

If the serial port can't be opened (e.g. on the dev Mac) it runs
disconnected: commands are dropped with a log line instead of crashing.

Standalone test (interactive):
    python esp32_link.py
"""

import threading
import time
from dataclasses import dataclass, field

SERIAL_PORT = "/dev/serial0"
SERIAL_BAUD = 115200

MOTOR_KEEPALIVE_S = 0.1     # resend M this often while moving (ESP32 watchdog is 0.5 s)
PING_INTERVAL_S = 2.0
PONG_TIMEOUT_S = 10.0       # no PONG for this long -> link considered down (generous: model
                            # loading at startup can stall this process for several seconds)
STARTUP_PONG_WAIT_S = 1.0   # start() waits this long for the first PONG
UNREACHABLE_WARNING = "I can't reach my motor controller."

STATUS_STATES = ("idle", "listening", "thinking", "recording", "armed")
IR_NAMES = ("bl", "bm", "br", "head")


@dataclass
class LinkState:
    ir: dict = field(default_factory=lambda: {n: False for n in IR_NAMES})
    ir_updated_at: float = 0.0
    lift_pos: int | None = None     # last LIFT_POS reported (deg)
    pot_deg: int | None = None      # last POT reported (deg)
    odom_left: int = 0              # wheel hall-sensor counts (ODOM), +forward once calibrated
    odom_right: int = 0
    odom_at: float = 0.0
    speed_left: int = 0           # measured wheel speeds (mm/s) while in speed mode
    speed_right: int = 0
    out_left: float = 0.0         # drive the speed controller is applying (0-255 units)
    out_right: float = 0.0
    speed_at: float = 0.0
    last_blocked: str | None = None
    last_blocked_at: float = 0.0
    last_pong_at: float = 0.0


class ESP32Link:
    def __init__(self, port: str = SERIAL_PORT, baud: int = SERIAL_BAUD, speak=None):
        """speak: optional callable(text) used for the one-time 'can't reach'
        voice warning (pass tts.speak)."""
        self.port = port
        self.baud = baud
        self.speak = speak
        self.state = LinkState()
        self.connected = False      # True while PONGs are arriving
        self.lift_allowed = None    # optional callable(deg) -> bool; False blocks that lift move (charging)
        self.halted = False         # software emergency stop (Telegram "stop"): no wheel or lift motion

        self._serial = None
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

        self._motion: tuple[int, int] | None = None   # active (left, right) or None
        self._motion_lock = threading.Lock()
        self._warned_unreachable = False

        self._blocked_event = threading.Event()
        self._lift_event = threading.Event()
        self._blocked_listeners: list = []

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> bool:
        """Opens the port and starts the background threads. Returns False
        (and stays usable in disconnected mode) if the port can't be opened."""
        try:
            import serial
            self._serial = serial.Serial(self.port, self.baud, timeout=0.2)
            self._serial.reset_input_buffer()
        except Exception as e:
            print(f"[esp32] couldn't open {self.port} ({e}) - running disconnected")
            self._serial = None
            return False

        self.state.last_pong_at = time.time()   # grace period before the first PONG
        self._stop_event.clear()
        for target in (self._reader_loop, self._keepalive_loop, self._ping_loop):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)
        # Opening the port can glitch the TX line, which would prefix the first
        # command with a garbage byte (the ESP32 then rejects it). A bare newline
        # first ends that partial line, then wait briefly for the first PONG so
        # callers right after start() already see connected=True.
        self._send("")
        self._send("PING")
        deadline = time.time() + STARTUP_PONG_WAIT_S
        while not self.connected and time.time() < deadline:
            time.sleep(0.02)
        print(f"[esp32] link open on {self.port} @ {self.baud}"
              + ("" if self.connected else " - no PONG yet"))
        return True

    def close(self):
        if self._serial is not None:
            self.stop()
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads.clear()
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    # ------------------------------------------------------------------ commands

    def drive(self, left: int, right: int):
        """Starts (or changes) a movement. Kept alive automatically until
        stop() or another drive() - the ESP32 still applies its own speed caps,
        ramping, and obstacle/reverse limits."""
        left, right = int(max(-255, min(255, left))), int(max(-255, min(255, right)))
        if self.halted and (left or right):
            return
        with self._motion_lock:
            self._motion = (left, right) if (left or right) else None
        self._blocked_event.clear()
        self._send(f"M {left} {right}")

    def drive_speed(self, left_mm_s: int, right_mm_s: int):
        """Speed mode: the ESP32 holds these real wheel speeds (mm/s, + forward) using
        the wheel odometry, whatever the battery voltage. Same safety rules as drive()."""
        left, right = int(max(-600, min(600, left_mm_s))), int(max(-600, min(600, right_mm_s)))
        if self.halted and (left or right):
            return
        with self._motion_lock:
            self._motion = (left, right, "V") if (left or right) else None
        self._blocked_event.clear()
        self._send(f"V {left} {right}")

    def stop_smooth(self, seconds: float = 1.0):
        """Ease to a halt (speed mode ramps down at its acceleration limit), then STOP."""
        with self._motion_lock:
            self._motion = (0, 0, "V")
        self._send("V 0 0")
        time.sleep(seconds)
        self.stop()

    def stop(self):
        with self._motion_lock:
            self._motion = None
        self._send("STOP")

    def drive_for(self, left: int, right: int, seconds: float) -> str | None:
        """Blocking helper: drive for `seconds`, then stop. Returns the
        sensor name if the ESP32 reported BLOCKED during the move, else None."""
        if self.halted:
            return "halted"
        self.drive(left, right)
        blocked = self._blocked_event.wait(timeout=seconds)
        self.stop()
        return self.state.last_blocked if blocked else None

    def halt(self):
        """Emergency stop: motors off now, and every wheel/lift command is ignored until resume()."""
        self.halted = True
        self.stop()

    def resume(self):
        self.halted = False

    def lift(self, deg: int, ms: int | None = None):
        if self.halted:
            print(f"[esp32] lift to {deg} skipped - emergency stop")
            self._lift_event.set()
            return False
        if self.lift_allowed is not None and not self.lift_allowed(deg):
            print(f"[esp32] lift to {deg} skipped - not allowed right now (charging)")
            self._lift_event.set()             # nobody waits for a move that won't happen
            return False
        self._lift_event.clear()
        self._send(f"LIFT {int(deg)}" + (f" {int(ms)}" if ms is not None else ""))

    def wait_lift(self, timeout: float) -> bool:
        """Waits for the LIFT_POS that ends the last lift() move."""
        return self._lift_event.wait(timeout=timeout)

    def send_battery(self, percent: int, charging: bool):
        self._send(f"BATT {int(round(percent))} {1 if charging else 0}")

    def set_status(self, state: str):
        if state not in STATUS_STATES:
            raise ValueError(f"unknown status {state!r}, expected one of {STATUS_STATES}")
        self._send(f"STATUS {state}")

    def reset_odom(self):
        """Zeroes both wheel counts on the ESP32."""
        self._send("ODOM_RESET")

    def odom(self) -> tuple[int, int]:
        with self._state_lock:
            return self.state.odom_left, self.state.odom_right

    def on_blocked(self, callback):
        """Registers callback(sensor_name), called from the reader thread on
        every BLOCKED message. Keep it quick (no blocking TTS here)."""
        self._blocked_listeners.append(callback)

    def obstacle_ahead(self) -> bool:
        with self._state_lock:
            return any(self.state.ir.values())

    # ------------------------------------------------------------------ internals

    def _send(self, line: str):
        if self._serial is None:
            print(f"[esp32] (disconnected) dropped: {line}")
            return
        try:
            with self._write_lock:
                self._serial.write((line + "\n").encode("ascii"))
        except Exception as e:
            print(f"[esp32] write failed: {e}")

    def _reader_loop(self):
        while not self._stop_event.is_set():
            try:
                raw = self._serial.readline()
            except Exception as e:
                print(f"[esp32] read failed: {e}")
                time.sleep(0.5)
                continue
            if raw:
                self._handle(raw.decode("ascii", errors="replace").strip())

    def _handle(self, line: str):
        parts = line.split()
        if not parts:
            return
        cmd, args = parts[0], parts[1:]
        now = time.time()
        try:
            if cmd == "IR" and len(args) == 4:
                with self._state_lock:
                    self.state.ir = {n: a == "1" for n, a in zip(IR_NAMES, args)}
                    self.state.ir_updated_at = now
            elif cmd == "ODOM" and len(args) == 2:
                with self._state_lock:
                    self.state.odom_left, self.state.odom_right = int(args[0]), int(args[1])
                    self.state.odom_at = now
            elif cmd == "SPD" and len(args) == 4:
                with self._state_lock:
                    self.state.speed_left, self.state.speed_right = int(args[0]), int(args[1])
                    self.state.out_left, self.state.out_right = float(args[2]), float(args[3])
                    self.state.speed_at = now
            elif cmd == "LIFT_POS" and len(args) == 1:
                self.state.lift_pos = int(args[0])
                self._lift_event.set()
            elif cmd == "POT" and len(args) == 1:
                self.state.pot_deg = int(args[0])
            elif cmd == "BLOCKED" and len(args) == 1:
                self.state.last_blocked = args[0]
                self.state.last_blocked_at = now
                self._blocked_event.set()
                print(f"[esp32] BLOCKED {args[0]}")
                for cb in list(self._blocked_listeners):
                    try:
                        cb(args[0])
                    except Exception as e:
                        print(f"[esp32] blocked listener failed: {e}")
            elif cmd == "PONG":
                self.state.last_pong_at = now
                if not self.connected:
                    print("[esp32] connected (PONG received)")
                self.connected = True
                self._warned_unreachable = False
        except ValueError:
            print(f"[esp32] malformed line: {line!r}")

    def _keepalive_loop(self):
        while not self._stop_event.wait(MOTOR_KEEPALIVE_S):
            with self._motion_lock:
                motion = self._motion
            if motion is not None:
                self._send(f"{motion[2] if len(motion) > 2 else 'M'} {motion[0]} {motion[1]}")

    def _ping_loop(self):
        while not self._stop_event.wait(PING_INTERVAL_S):
            self._send("PING")
            if time.time() - self.state.last_pong_at > PONG_TIMEOUT_S:
                if self.connected or not self._warned_unreachable:
                    print(f"[esp32] no PONG for {PONG_TIMEOUT_S:.0f}s - motor controller unreachable")
                self.connected = False
                if not self._warned_unreachable:
                    self._warned_unreachable = True
                    if self.speak is not None:
                        threading.Thread(target=self.speak, args=(UNREACHABLE_WARNING,),
                                         daemon=True).start()


# ---------------------------------------------------------------------- standalone test

if __name__ == "__main__":
    HELP = """Commands:
  ping                     check the link
  ir                       show IR state
  drive <l> <r> <secs>     drive for N seconds (robot on blocks!)
  stop                     stop motors
  lift <deg> [ms]          move the lift and wait for LIFT_POS
  batt <pct> <0|1>         battery LEDs
  status <state>           white LED: idle|listening|thinking|recording|armed
  watch <secs>             print state changes for N seconds
  q                        quit"""
    link = ESP32Link(speak=lambda t: print(f"Qwen (would say): {t}"))
    link.start()
    print(HELP)
    try:
        while True:
            try:
                parts = input("> ").split()
            except EOFError:
                break
            if not parts:
                continue
            c = parts[0].lower()
            try:
                if c == "q":
                    break
                elif c == "ping":
                    before = link.state.last_pong_at
                    link._send("PING")
                    time.sleep(0.3)
                    print("PONG" if link.state.last_pong_at > before else "no reply")
                elif c == "ir":
                    print(link.state.ir)
                elif c == "drive":
                    blocked = link.drive_for(int(parts[1]), int(parts[2]), float(parts[3]))
                    print(f"BLOCKED by {blocked}" if blocked else "done")
                elif c == "stop":
                    link.stop()
                elif c == "lift":
                    link.lift(int(parts[1]), int(parts[2]) if len(parts) > 2 else None)
                    print(f"LIFT_POS {link.state.lift_pos}" if link.wait_lift(25) else "no LIFT_POS")
                elif c == "batt":
                    link.send_battery(int(parts[1]), parts[2] == "1")
                elif c == "status":
                    link.set_status(parts[1])
                elif c == "watch":
                    end, last = time.time() + float(parts[1]), None
                    while time.time() < end:
                        snap = (dict(link.state.ir), link.state.lift_pos, link.state.pot_deg,
                                link.state.last_blocked, link.connected)
                        if snap != last:
                            print(f"ir={snap[0]} lift={snap[1]} pot={snap[2]} "
                                  f"blocked={snap[3]} connected={snap[4]}")
                            last = snap
                        time.sleep(0.05)
                else:
                    print(HELP)
            except (IndexError, ValueError) as e:
                print(f"bad arguments ({e})\n{HELP}")
    finally:
        link.close()
