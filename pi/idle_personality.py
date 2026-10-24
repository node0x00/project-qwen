"""
Project Qwen - idle personality.

While the robot is in normal mode and nothing is happening, every so often
(random interval, IDLE_MIN_S..IDLE_MAX_S) it slowly raises the scissor lift
partway and lowers it back to where it was - a small "stretch" so it feels
alive.

Never runs while recording, during security mode, or while a conversation is
active (recent wake word / command), and never mid-move. The conditions are
passed in as a callable so this module stays independent of the rest.

Standalone test (does one animation right away):
    python idle_personality.py
"""

import random
import threading
import time

IDLE_MIN_S = 5 * 60
IDLE_MAX_S = 15 * 60

RAISE_MIN_DEG = 25           # how far up the "stretch" goes (random in this range)
RAISE_MAX_DEG = 50
RAISE_MS = 3500              # slow on purpose - the organic motion is the point
HOLD_MIN_S, HOLD_MAX_S = 1.0, 2.5
LOWER_MS = 4000

LIFT_MAX_DEG = 130
LIFT_MOVE_TIMEOUT_S = 10.0


class IdlePersonality:
    def __init__(self, link, can_animate):
        """can_animate: callable() -> bool, True only when nothing else is
        going on (normal mode, not recording, no active conversation)."""
        self.link = link
        self.can_animate = can_animate
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=LIFT_MOVE_TIMEOUT_S * 2)

    def _run(self):
        while not self._stop_event.wait(random.uniform(IDLE_MIN_S, IDLE_MAX_S)):
            if self.can_animate() and self.link.connected:
                self.animate_once()

    def animate_once(self):
        start = self.link.state.lift_pos
        start = 0 if start is None else start     # ESP32 boots at 0
        top = min(LIFT_MAX_DEG, start + random.randint(RAISE_MIN_DEG, RAISE_MAX_DEG))
        if top <= start:
            return                                # already at the top - nothing to show
        print(f"[idle] stretch {start} -> {top} -> {start}")
        self.link.lift(top, RAISE_MS)
        if not self.link.wait_lift(LIFT_MOVE_TIMEOUT_S):
            return
        # Hold briefly, but come back down early if something starts happening.
        hold_until = time.time() + random.uniform(HOLD_MIN_S, HOLD_MAX_S)
        while time.time() < hold_until and self.can_animate() and not self._stop_event.is_set():
            time.sleep(0.1)
        self.link.lift(start, LOWER_MS)
        self.link.wait_lift(LIFT_MOVE_TIMEOUT_S)


if __name__ == "__main__":
    from esp32_link import ESP32Link

    link = ESP32Link()
    link.start()
    time.sleep(1.0)
    try:
        IdlePersonality(link, can_animate=lambda: True).animate_once()
        print(f"done, lift at {link.state.lift_pos}")
    finally:
        link.close()
