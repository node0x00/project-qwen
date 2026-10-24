"""
Project Qwen - battery monitor (INA226 over I2C).

Hardware: INA226 on I2C bus 1, address 0x40, wired on the CHARGING branch
only (IN+ from battery+, IN- to the charging pad +). So:
  - bus voltage  -> battery voltage (-> % via a 7S Li-ion lookup curve),
  - current      -> charging current only (motor current never flows through
                    the shunt). Because of the shunt orientation the sign is
                    probably NEGATIVE while charging - CHARGE_CURRENT_SIGN
                    flips it. Confirm with the standalone test below.

Battery: 7S Li-ion hoverboard pack. Full 29.4 V (4.20 V/cell), 0% of usable
range at 22.4 V (3.20 V/cell). Li-ion voltage vs. charge is non-linear, so %
comes from a per-cell lookup curve, not a straight line. Voltage sags under
motor load, so readings are smoothed with a slow moving average.

Every BATT_SEND_INTERVAL_S the state is sent to the ESP32 (battery LEDs).
Speaks "My battery is low" once when it drops to LOW_BATTERY_PCT.

If the INA226 isn't reachable (e.g. on the dev Mac), available stays False
and nothing crashes.

Standalone test (prints readings every second - plug/unplug the charger to
check CHARGE_CURRENT_SIGN):
    python battery_monitor.py
"""

import threading
import time

I2C_BUS = 1
INA226_ADDR = 0x40
SHUNT_OHMS = 0.1            # R100 on most INA226 breakouts - check yours

# Positive charging current after this sign is applied. Flip to +1 if the
# standalone test shows positive current while the charger is connected.
CHARGE_CURRENT_SIGN = -1
CHARGING_THRESHOLD_A = 0.05  # above this (after sign) counts as charging

SAMPLE_INTERVAL_S = 1.0
SMOOTHING_ALPHA = 0.05      # EMA weight per sample; ~20 s time constant at 1 Hz
BATT_SEND_INTERVAL_S = 10.0

LOW_BATTERY_PCT = 20
LOW_BATTERY_REARM_PCT = 25  # hysteresis: warn again only after recovering above this
DOCKING_PCT = 25            # needs_docking() threshold, for future auto-docking
LOW_BATTERY_MESSAGE = "My battery is low."

CELLS = 7
# Per-cell resting voltage -> % of usable range (3.20 V = 0%, 4.20 V = 100%).
# Typical Li-ion NMC discharge curve; refine from your own pack if you log a
# full discharge.
CELL_CURVE = [
    (3.20, 0), (3.30, 5), (3.40, 11), (3.50, 20), (3.60, 33), (3.70, 47),
    (3.80, 60), (3.90, 72), (4.00, 83), (4.10, 92), (4.20, 100),
]

# INA226 registers
_REG_CONFIG = 0x00
_REG_SHUNT = 0x01           # signed, 2.5 uV/LSB
_REG_BUS = 0x02             # 1.25 mV/LSB
_REG_MANUFACTURER = 0xFE    # 0x5449 ("TI")
_CONFIG = 0x4527            # 16-sample averaging, 1.1 ms conversions, continuous shunt + bus


def voltage_to_percent(pack_voltage: float) -> float:
    cell = pack_voltage / CELLS
    if cell <= CELL_CURVE[0][0]:
        return 0.0
    if cell >= CELL_CURVE[-1][0]:
        return 100.0
    for (v0, p0), (v1, p1) in zip(CELL_CURVE, CELL_CURVE[1:]):
        if v0 <= cell <= v1:
            return p0 + (p1 - p0) * (cell - v0) / (v1 - v0)
    return 0.0


class INA226:
    def __init__(self, bus: int = I2C_BUS, addr: int = INA226_ADDR):
        from smbus2 import SMBus
        self._bus = SMBus(bus)
        self._addr = addr
        manufacturer = self._read(_REG_MANUFACTURER)
        if manufacturer != 0x5449:
            raise RuntimeError(f"device at 0x{addr:02x} isn't an INA226 (manufacturer id 0x{manufacturer:04x})")
        self._write(_REG_CONFIG, _CONFIG)

    def _read(self, reg: int) -> int:
        word = self._bus.read_word_data(self._addr, reg)
        return ((word & 0xFF) << 8) | (word >> 8)    # INA226 is big-endian

    def _write(self, reg: int, value: int):
        self._bus.write_word_data(self._addr, reg, ((value & 0xFF) << 8) | (value >> 8))

    def bus_voltage(self) -> float:
        return self._read(_REG_BUS) * 1.25e-3

    def shunt_voltage(self) -> float:
        raw = self._read(_REG_SHUNT)
        if raw & 0x8000:
            raw -= 0x10000
        return raw * 2.5e-6

    def current(self) -> float:
        return self.shunt_voltage() / SHUNT_OHMS

    def close(self):
        self._bus.close()


class BatteryMonitor:
    def __init__(self, esp32_link=None, speak=None):
        """esp32_link: optional ESP32Link for the BATT LED updates.
        speak: optional callable(text) for the low-battery warning."""
        self.link = esp32_link
        self.speak = speak
        self.available = False
        self.voltage: float | None = None      # smoothed pack voltage
        self.percent: float | None = None
        self.charging = False
        self.charge_current: float = 0.0       # A, positive while charging

        self._sensor = None
        self._thread = None
        self._stop_event = threading.Event()
        self._low_warned = False
        self._last_send = 0.0

    def start(self) -> bool:
        try:
            self._sensor = INA226()
        except Exception as e:
            print(f"[battery] INA226 unavailable ({e}) - battery monitoring off")
            return False
        self.available = True
        self._sample()          # seed the average so the first reading is real
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[battery] {self.describe()}")
        return True

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._sensor is not None:
            self._sensor.close()

    def needs_docking(self) -> bool:
        return (self.available and not self.charging and self.percent is not None
                and self.percent <= DOCKING_PCT)

    def describe(self) -> str:
        """Spoken answer for 'what's your battery'."""
        if not self.available or self.percent is None:
            return "I can't read my battery right now."
        text = f"My battery is at {round(self.percent)} percent"
        return text + (", and I'm charging." if self.charging else ".")

    def _sample(self):
        v = self._sensor.bus_voltage()
        i = self._sensor.current() * CHARGE_CURRENT_SIGN
        self.voltage = v if self.voltage is None else self.voltage + SMOOTHING_ALPHA * (v - self.voltage)
        self.charge_current = i
        self.charging = i > CHARGING_THRESHOLD_A
        self.percent = voltage_to_percent(self.voltage)

    def _run(self):
        while not self._stop_event.wait(SAMPLE_INTERVAL_S):
            try:
                self._sample()
            except OSError as e:
                print(f"[battery] read failed: {e}")
                continue

            now = time.time()
            if self.link is not None and now - self._last_send >= BATT_SEND_INTERVAL_S:
                self.link.send_battery(self.percent, self.charging)
                self._last_send = now

            if self.charging or self.percent > LOW_BATTERY_REARM_PCT:
                self._low_warned = False
            elif self.percent <= LOW_BATTERY_PCT and not self._low_warned:
                self._low_warned = True
                print(f"[battery] low: {self.percent:.0f}%")
                if self.speak is not None:
                    threading.Thread(target=self.speak, args=(LOW_BATTERY_MESSAGE,), daemon=True).start()


if __name__ == "__main__":
    try:
        sensor = INA226()
    except Exception as e:
        raise SystemExit(f"INA226 not reachable: {e}")
    print("raw readings every second (Ctrl+C to stop). Plug/unplug the charger and watch "
          "'current' - if it goes POSITIVE while charging, set CHARGE_CURRENT_SIGN = +1.\n")
    try:
        while True:
            v, sv = sensor.bus_voltage(), sensor.shunt_voltage()
            i = sv / SHUNT_OHMS
            charging = i * CHARGE_CURRENT_SIGN > CHARGING_THRESHOLD_A
            print(f"bus {v:6.3f} V ({v / CELLS:.3f} V/cell, {voltage_to_percent(v):5.1f}%)   "
                  f"shunt {sv * 1000:+7.3f} mV   current {i:+6.3f} A   "
                  f"{'CHARGING' if charging else 'not charging'}", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        sensor.close()
