"""
Project Qwen - charging assistant (runs on its own while docked).

  - Says "Charging." when charging starts (and once at startup if it already is).
  - At ~90% (charging voltage held at FULL_CHARGING_V for FULL_HOLD_S): says
    "Battery full.", shows solid green on the battery LEDs, and drives forward a
    little to come off the charging pads (MOVE_OFF_ENABLED).
  - Keeps the battery LEDs updated (BATT) the rest of the time.

The charger lifts the measured voltage while it's connected, so "90%" is a
charging-voltage threshold: 28.75 V while charging ~ 28.55 V at rest (4.08 V/cell).
The 1N4007 on the pads caps charging near 28.9 V anyway.

    venv/bin/python charge_manager.py
Superseded by power_manager.py inside assistant.py (qwen.service) - don't run it alongside that or slam_view.py (shared ESP32 link).
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))  # run from pi/tools

import time

from battery_monitor import CHARGE_CURRENT_SIGN, CHARGING_THRESHOLD_A, INA226, voltage_to_percent
from esp32_link import ESP32Link

FULL_CHARGING_V = 28.75
FULL_HOLD_S = 60            # must stay at/above the threshold this long (no false "full")
FULL_PERCENT = 90
CHANGE_SAMPLES = 3          # charging must look the same this many samples in a row to count
SAMPLE_S = 1.0
BATT_SEND_S = 10.0          # the ESP32 shows "no data" if BATT is older than 60 s

# Driving off the dock (right wheel direction fixed 2026-10-13: DIR on GPIO5).
MOVE_OFF_ENABLED = True
MOVE_OFF_SPEED = 1          # ~13 cm in 1 s at speed 1
MOVE_OFF_S = 1.0

PIPER_MODEL_PATH = "models/en_US-lessac-medium.onnx"


class Voice:
    """Piper TTS, loaded on first use (keeps idle memory/CPU low)."""

    def __init__(self):
        self._voice = None

    def say(self, text: str):
        import numpy as np
        import audio_out
        if self._voice is None:
            from piper.voice import PiperVoice
            self._voice = PiperVoice.load(PIPER_MODEL_PATH)
        chunks, rate = [], None
        for chunk in self._voice.synthesize(text):
            rate = chunk.sample_rate
            chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        if chunks:
            audio_out.play(np.concatenate(chunks), rate)


def log(msg: str):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    sensor = INA226()
    link = ESP32Link()
    link.start()
    voice = Voice()

    charging = None          # debounced state
    pending, pending_n = None, 0
    above_since = None
    full = False
    last_send = 0.0
    last_log = 0.0

    while True:
        v = sensor.bus_voltage()
        amps = sensor.current() * CHARGE_CURRENT_SIGN
        now_charging = amps > CHARGING_THRESHOLD_A

        # debounce charging on/off
        if charging is None:
            charging = now_charging
            log(f"start: {v:.2f} V, {amps:+.3f} A, {'charging' if charging else 'not charging'}")
            if charging:
                voice.say("Charging.")
        elif now_charging != charging:
            pending_n = pending_n + 1 if pending == now_charging else 1
            pending = now_charging
            if pending_n >= CHANGE_SAMPLES:
                charging, pending_n = now_charging, 0
                log(f"{'charger connected' if charging else 'charger disconnected'} ({v:.2f} V, {amps:+.3f} A)")
                full = False             # off the pads (or re-docked): back to normal LEDs
                above_since = None
                if charging:
                    voice.say("Charging.")
        else:
            pending_n = 0

        # full at ~90%
        if charging and not full:
            if v >= FULL_CHARGING_V:
                above_since = above_since or time.time()
                if time.time() - above_since >= FULL_HOLD_S:
                    full = True
                    log(f"FULL: {v:.2f} V while charging at {amps:.3f} A")
                    link.send_battery(FULL_PERCENT, False)        # solid green
                    last_send = time.time()
                    voice.say("Battery full.")
                    if MOVE_OFF_ENABLED:
                        log("moving off the dock")
                        link.drive_for(MOVE_OFF_SPEED, MOVE_OFF_SPEED, MOVE_OFF_S)
                    else:
                        log("move-off disabled (right wheel direction fault) - staying on the pads")
            else:
                above_since = None

        # battery LEDs
        if time.time() - last_send >= BATT_SEND_S:
            if full:
                link.send_battery(FULL_PERCENT, False)
            else:
                link.send_battery(voltage_to_percent(v), charging)
            last_send = time.time()

        if time.time() - last_log >= 300:
            log(f"{v:.2f} V  {amps:+.3f} A  {'charging' if charging else 'not charging'}"
                f"{'  FULL' if full else ''}")
            last_log = time.time()
        time.sleep(SAMPLE_S)


if __name__ == "__main__":
    main()
