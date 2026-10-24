"""
Project Qwen - client for the smart-home ESP-NOW hub (lights + door + keypad).

The hub (an ESP32-S3 running a web server; a separate project) relays commands over ESP-NOW to:
  node 1 / node 2  light-switch servos
  node 3           door-lock servo
  node 4           keypad + LCD + buzzer (3-minute countdown after a remote unlock)

Commands (fire-and-forget, sent from a background worker so callers never
wait on Wi-Fi):
  lights_on() / lights_off()     both lights
  door_unlock() / door_lock()    door only (no lights, no countdown)
  cancel_countdown()             owner verified by face -> keypad stops its countdown

Events: a poller reads GET /events?since=N twice a second and calls
on_event(type) for each new one:
  "exit"               keypad D pressed (exit button)
  "remote_unlock"      web page UNLOCK (door unlocked, lights on, keypad countdown started)
  "password_ok"        correct code entered during the countdown
  "password_wrong"
  "countdown_expired"  3 minutes passed without the code
Events older than STALE_EVENT_S (e.g. queued while the Pi was off) are ignored.

Standalone test (prints events as they happen; doesn't move anything):
    python smart_home.py
"""

import json
import os
import queue
import threading
import time
import urllib.request

HUB_URL = os.environ.get("QWEN_HUB_URL", "http://192.168.50.23")   # smart-home hub (set QWEN_HUB_URL in .env)
POLL_INTERVAL_S = 0.5
REQUEST_TIMEOUT_S = 4.0
COMMAND_RETRIES = 2
STALE_EVENT_S = 30.0


class SmartHomeHub:
    def __init__(self, base_url: str = HUB_URL, on_event=None):
        self.base_url = base_url.rstrip("/")
        self.on_event = on_event or (lambda event: None)
        self.reachable = None            # None = not polled yet
        self._commands: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._boot_id = None
        self._last_seq = None
        self.lights_known = None         # True/False after we (or a remote unlock) set them; None = unknown

    # ------------------------------------------------------------------ lifecycle

    def start(self):
        self._stop_event.clear()
        for target in (self._poll_loop, self._command_loop):
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop_event.set()
        self._commands.put(None)
        for t in self._threads:
            t.join(timeout=REQUEST_TIMEOUT_S + 1)
        self._threads.clear()

    # ------------------------------------------------------------------ commands

    def lights_on(self):
        self.lights_known = True
        self._send("/lights/on")

    def lights_off(self):
        self.lights_known = False
        self._send("/lights/off")

    def door_unlock(self):
        self._send("/door/unlock")

    def door_lock(self):
        self._send("/door/lock")

    def cancel_countdown(self):
        self._send("/countdown/cancel")

    def _send(self, path: str):
        print(f"[hub] -> {path}")
        self._commands.put(path)

    def _get(self, path: str) -> bytes:
        with urllib.request.urlopen(self.base_url + path, timeout=REQUEST_TIMEOUT_S) as r:
            return r.read()

    def _command_loop(self):
        while not self._stop_event.is_set():
            path = self._commands.get()
            if path is None:
                return
            for attempt in range(1 + COMMAND_RETRIES):
                try:
                    self._get(path)
                    break
                except Exception as e:
                    if attempt == COMMAND_RETRIES:
                        print(f"[hub] {path} failed: {e}")
                    else:
                        time.sleep(0.5)

    # ------------------------------------------------------------------ events

    def _poll_loop(self):
        while not self._stop_event.wait(POLL_INTERVAL_S):
            since = self._last_seq if self._last_seq is not None else 0
            try:
                data = json.loads(self._get(f"/events?since={since}"))
            except Exception as e:
                if self.reachable is not False:
                    print(f"[hub] unreachable at {self.base_url} ({e})")
                self.reachable = False
                continue
            if self.reachable is not True:
                print(f"[hub] connected at {self.base_url}")
            self.reachable = True

            boot, latest = data.get("boot"), data.get("latest", 0)
            if self._boot_id is None:
                # First contact: skip history, only react to what happens from now on.
                self._boot_id, self._last_seq = boot, latest
                continue
            if boot != self._boot_id:
                # Hub rebooted - its sequence numbers restarted. Re-read from the start
                # next round (stale events are filtered by age).
                print("[hub] hub rebooted, resyncing events")
                self._boot_id, self._last_seq = boot, 0
                continue

            for event in data.get("events", []):
                self._last_seq = max(self._last_seq, event["seq"])
                if event.get("age_ms", 0) / 1000 > STALE_EVENT_S:
                    print(f"[hub] ignoring stale event {event['type']}")
                    continue
                print(f"[hub] event: {event['type']}")
                try:
                    if event["type"] == "remote_unlock":
                        self.lights_known = True     # the hub turns the lights on itself
                    self.on_event(event["type"])
                except Exception as e:
                    print(f"[hub] event handler failed for {event['type']}: {e}")
            self._last_seq = max(self._last_seq, latest)


if __name__ == "__main__":
    hub = SmartHomeHub(on_event=lambda e: print(f">>> EVENT {e}"))
    hub.start()
    print(f"Polling {HUB_URL}/events - press D on the keypad, or UNLOCK on the web page. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        hub.stop()
