"""
Project Qwen - white status LED (on the ESP32), driven from robot state.

The ESP32 shows the pattern; this module decides which one, from one place,
with a fixed priority:
  armed      - security mode is on (ARMED_IDLE or ARMED_ACTIVE)
  listening  - wake word heard, recording the command
  thinking   - transcribing / answering
  recording  - camera is recording video
  idle       - nothing going on

So talking to the robot while it records shows listening/thinking, then goes
back to recording. The state is re-sent every RESEND_S even when unchanged,
so the LED recovers if the ESP32 reboots.

Standalone test (cycles through every state):
    python status_led.py
"""

import threading

POLL_S = 0.2
RESEND_S = 5.0


class StatusLED:
    def __init__(self, link, is_recording, is_armed):
        """is_recording / is_armed: callables returning bool."""
        self.link = link
        self.is_recording = is_recording
        self.is_armed = is_armed
        self._voice_phase = "idle"     # "idle" | "listening" | "thinking"
        self._stop_event = threading.Event()
        self._thread = None
        self._last_sent = None

    def set_voice_phase(self, phase: str):
        if phase not in ("idle", "listening", "thinking"):
            raise ValueError(f"unknown voice phase {phase!r}")
        self._voice_phase = phase
        self._update()                 # don't wait for the next poll - feels laggy otherwise

    def current_state(self) -> str:
        if self.is_armed():
            return "armed"
        if self._voice_phase != "idle":
            return self._voice_phase
        if self.is_recording():
            return "recording"
        return "idle"

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.link.set_status("idle")

    def _update(self, force: bool = False):
        state = self.current_state()
        if force or state != self._last_sent:
            self.link.set_status(state)
            self._last_sent = state

    def _run(self):
        ticks_per_resend = int(RESEND_S / POLL_S)
        tick = 0
        while not self._stop_event.wait(POLL_S):
            tick += 1
            self._update(force=tick % ticks_per_resend == 0)


if __name__ == "__main__":
    import time
    from esp32_link import ESP32Link

    link = ESP32Link()
    link.start()
    for state in ("listening", "thinking", "recording", "armed", "idle"):
        print(f"STATUS {state} - watch the white LED for 3 s")
        link.set_status(state)
        time.sleep(3)
    link.close()
