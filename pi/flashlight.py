"""
Project Qwen - flashlight on the Pi's GPIO.

The light is on GPIO23 (BCM, physical pin 16). It is wired directly between GPIO23
and GND (no MOSFET), so it must draw no more than ~16 mA - a GPIO pin can't
safely source more. A brighter light needs a MOSFET/transistor driver.
Real hardware uses gpiozero's OutputDevice.

Like pir_sensor.py, get_flashlight() falls back to a print-only mock when no
GPIO backend is available (e.g. the dev Mac), so calling code is unchanged.

Standalone test (toggles the light a few times):
    python flashlight.py
"""

FLASHLIGHT_GPIO_PIN = 23  # BCM numbering (physical pin 16; moved from GPIO26 for the perfboard build)


class Flashlight:
    """Real hardware flashlight via gpiozero.OutputDevice."""

    def __init__(self, pin: int = FLASHLIGHT_GPIO_PIN):
        from gpiozero import OutputDevice
        self._device = OutputDevice(pin, active_high=True, initial_value=False)

    @property
    def is_on(self) -> bool:
        return bool(self._device.value)

    def on(self):
        self._device.on()

    def off(self):
        self._device.off()

    def toggle(self):
        self._device.toggle()


class MockFlashlight:
    """Dev/test stand-in - just tracks and prints the state."""

    def __init__(self):
        self.is_on = False

    def on(self):
        self.is_on = True
        print("[mock flashlight] ON")

    def off(self):
        self.is_on = False
        print("[mock flashlight] OFF")

    def toggle(self):
        self.off() if self.is_on else self.on()


def get_flashlight(pin: int = FLASHLIGHT_GPIO_PIN):
    try:
        light = Flashlight(pin)
        print(f"[flashlight] Using real GPIO{pin} (BCM)")
        return light
    except Exception as e:
        print(f"[flashlight] Real GPIO unavailable ({e!r}) - using print-only mock")
        return MockFlashlight()


if __name__ == "__main__":
    import time

    light = get_flashlight()
    try:
        for _ in range(3):
            light.on()
            print("on")
            time.sleep(1)
            light.off()
            print("off")
            time.sleep(1)
    finally:
        light.off()
