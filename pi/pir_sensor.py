"""
Project Qwen - PIR motion sensor interface for security mode.

The PIR sensor is wired directly to the Pi's own GPIO (not the ESP32, which
only handles motors and is unrelated to this). Real hardware uses gpiozero's
MotionSensor, the standard/simplest approach for this on a Pi.

Since development/testing happens on a Mac with no GPIO hardware, a mock
implementation is also provided so the security state machine can be fully
exercised without real hardware: press 'm' in the terminal to simulate
motion (a raw keypress via termios/tty cbreak mode - no Enter needed, and no
other key does anything).

get_pir_sensor() tries the real sensor first and falls back to the mock on
any failure (missing gpiozero, or no GPIO backend available), so the same
calling code works unchanged on both the Pi and this Mac.
"""

import sys
import termios
import time
import tty

PIR_GPIO_PIN = 24  # BCM numbering - change to match your wiring


class PIRSensor:
    """Real hardware PIR via gpiozero.MotionSensor."""

    def __init__(self, pin: int = PIR_GPIO_PIN):
        from gpiozero import MotionSensor
        self._sensor = MotionSensor(pin)

    def wait_for_motion(self, timeout: float | None = None) -> bool:
        """Blocks until motion (True) or until `timeout` seconds pass (False)."""
        return bool(self._sensor.wait_for_motion(timeout))

    def motion_now(self) -> bool:
        return bool(self._sensor.motion_detected)


_shared = None   # one sensor per pin: gpiozero refuses a second MotionSensor on GPIO24


class MockPIRSensor:
    """Dev/test stand-in - blocks on a raw 'm' keypress (POSIX only, fine
    for both this Mac and the target Pi)."""

    def __init__(self, trigger_key: str = "m"):
        self.trigger_key = trigger_key
        self._prompted = False

    def motion_now(self) -> bool:
        return False

    def wait_for_motion(self, timeout: float | None = None) -> bool:
        """Blocks until 'm' is pressed (True) or `timeout` seconds pass (False)."""
        import select
        if not sys.stdin.isatty():           # running as a service: no keyboard to simulate with
            time.sleep(timeout or 0)
            return False
        if not self._prompted:
            print(f"[mock PIR] press '{self.trigger_key}' to simulate motion...")
            self._prompted = True
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        deadline = None if timeout is None else time.time() + timeout
        try:
            tty.setcbreak(fd)
            while True:
                wait = None if deadline is None else max(0.0, deadline - time.time())
                ready, _, _ = select.select([sys.stdin], [], [], wait)
                if not ready:
                    return False
                if sys.stdin.read(1).lower() == self.trigger_key:
                    print("[mock PIR] motion simulated")
                    self._prompted = False
                    return True
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def get_pir_sensor(pin: int = PIR_GPIO_PIN):
    """The PIR sensor, shared by everyone who asks (security mode and idle mode)."""
    global _shared
    if _shared is not None:
        return _shared
    try:
        _shared = PIRSensor(pin)
        print(f"[pir] Using real gpiozero MotionSensor on GPIO{pin} (BCM)")
        return _shared
    except Exception as e:
        print(f"[pir] Real PIR unavailable ({e!r}) - falling back to mock "
              f"keypress PIR (press 'm' in this terminal to simulate motion)")
        return MockPIRSensor()


if __name__ == "__main__":
    print("Testing PIR sensor (Ctrl+C to quit).")
    sensor = get_pir_sensor()
    try:
        while True:
            sensor.wait_for_motion()
            print(">>> MOTION DETECTED")
    except KeyboardInterrupt:
        print("\nStopped.")
