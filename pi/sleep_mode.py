"""
Project Qwen - idle mode: no motion for IDLE_AFTER_S -> everything off except the PIR.

Asleep: camera, gesture watching and the voice pipeline (mic + wake word) are stopped, and the
idle animations pause, so "Hey Qwen" doesn't work until something moves. The LLM stays loaded
(instant answers on waking), and the battery/charging manager and the ESP32 link keep running,
so low-battery return-to-dock still works while asleep.

Any of these counts as activity and keeps it awake: PIR motion, talking to it, the robot driving
or docking, a recording, a security mode other than OFF (security manages the camera/voice
itself), or someone watching the live view.
"""

import threading
import time

IDLE_AFTER_S = 120
POLL_S = 0.5


class SleepMode:
    def __init__(self, pir, sleep_fn, wake_fn, busy_fns):
        """pir: sensor with motion_now(). sleep_fn / wake_fn: turn things off / back on.
        busy_fns: callables; any returning True counts as activity."""
        self.pir = pir
        self.sleep_fn = sleep_fn
        self.wake_fn = wake_fn
        self.busy_fns = busy_fns
        self.asleep = False
        self.last_activity = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()

    def poke(self):
        """Something happened (wake from code, e.g. a web command)."""
        self.last_activity = time.time()
        if self.asleep:
            self._wake("activity")

    def _busy(self) -> bool:
        for fn in self.busy_fns:
            try:
                if fn():
                    return True
            except Exception:
                pass
        return False

    def _run(self):
        while not self._stop.wait(POLL_S):
            try:
                motion = self.pir.motion_now()
            except Exception:
                motion = False
            if motion or self._busy():
                self.last_activity = time.time()
                if self.asleep:
                    self._wake("motion" if motion else "activity")
            elif not self.asleep and time.time() - self.last_activity >= IDLE_AFTER_S:
                self._sleep()

    def _sleep(self):
        with self._lock:
            if self.asleep:
                return
            self.asleep = True
        print(f"[idle] no motion for {IDLE_AFTER_S} s - camera, gestures and voice off "
              f"(PIR still watching)", flush=True)
        try:
            self.sleep_fn()
        except Exception as e:
            print(f"[idle] going to sleep failed: {e}", flush=True)

    def _wake(self, why: str):
        with self._lock:
            if not self.asleep:
                return
            self.asleep = False
        print(f"[idle] {why} - waking up", flush=True)
        try:
            self.wake_fn()
        except Exception as e:
            print(f"[idle] waking up failed: {e}", flush=True)
