# Project Qwen — ESP32 firmware

Real-time half of the robot: drive motors, IR obstacle reflex, scissor lift, manual height
knob, battery/status LEDs. The Pi sends high-level commands over a direct UART wire; every
safety reflex (watchdog, obstacle stop, reverse limits) lives here and never waits on the Pi.

Board: ESP32-WROOM-32 dev board (`esp32dev`). Arduino framework via PlatformIO,
`ESP32Servo` library (AVR `Servo.h` does not work on ESP32).

## Layout

| File | What it does |
|---|---|
| `include/config.h` | **All pins and tuning constants** — the only file you should need to edit on the bench |
| `include/modules.h` | Module interfaces |
| `src/main.cpp` | `setup()` / non-blocking `loop()` — calls each module's `update(now)` |
| `src/pilink.cpp` | Line protocol parser (Serial2 = Pi, Serial/USB = hand testing) |
| `src/ir.cpp` | Debounced IR sensors, reports `IR ...` |
| `src/motors.cpp` | Ramping, watchdog, obstacle reflex, reverse limits |
| `src/lift.cpp` | Eased scissor-lift moves + potentiometer arbitration |
| `src/leds.cpp` | Battery LEDs + white status LED |

## Build / flash / monitor

In VS Code (PlatformIO extension): open this folder, then
use the PlatformIO toolbar — ✓ build, → upload, 🔌 serial monitor.

From a terminal:
```bash
pio run                 # build
pio run -t upload       # flash over USB
pio device monitor      # USB serial monitor at 115200 (local echo on)
```
(`pio` lives at `~/.platformio/penv/bin/pio` if it's not on your PATH.)

## Wiring notes

**Pi link (UART, not USB)**

| ESP32 | Pi |
|---|---|
| GPIO22 (RX) | GPIO14 / TXD (pin 8) |
| GPIO21 (TX) | GPIO15 / RXD (pin 10) |
| GND | GND — **required**, the link doesn't work without a shared ground |

Both sides are 3.3 V logic, so no level shifter. The Pi side is already configured: serial
console removed, `dtoverlay=disable-bt`, `/dev/serial0 -> ttyAMA0`. Leave the ESP32's USB
port free for flashing and these debug logs.

**IR sensors**
- Active LOW by default (`IR_ACTIVE_LOW`). Inputs are pulled toward the *obstacle* level, so
  a disconnected sensor reads as "blocked" — the robot fails safe instead of driving blind.
- **GPIO35 (bottom-middle) has no internal pull resistors** — add an external 10 kΩ
  pull-down to GND for the same fail-safe behaviour (pull-up instead if you set
  `IR_ACTIVE_LOW = false`).
- If your IR modules are powered from 5 V, check their output doesn't exceed 3.3 V (most
  LM393 boards output VCC level — power them from 3.3 V or add a divider).
- GPIO15 is a boot strapping pin; a module holding it LOW at power-on only silences the
  ESP32's boot log. Harmless.

**Motors**
- PWM at 20 kHz (`MOTOR_PWM_FREQ`). If your controller expects ~1 kHz, lower it.
- Check the controller accepts 3.3 V logic on its PWM/DIR inputs.
- If a wheel spins backwards for "forward", flip `MOTOR_L_INVERT` / `MOTOR_R_INVERT`
  (mirrored hub motors usually need one of them).

**Lift servos**
- Power the MG996Rs from their own 5–6 V supply capable of several amps (each can draw
  >1 A when stalling), **not** the ESP32's 5 V pin. Tie that supply's GND to ESP32 GND.
- At power-on both servos jump once to `LIFT_BOOT_DEG` (0 = lowest). There is no position
  feedback, so this one move can't be eased.
- **Before the first full-range test**, verify `LIFT_TOP_MIRRORED` with the scissor
  mechanism disconnected (or with small moves like `LIFT 5` / `LIFT 10`) — if one servo is
  mirrored and the flag is wrong, the two servos will fight and twist the lift.

**Potentiometer**: wiper to GPIO34, ends to 3.3 V and GND (not 5 V). If the knob doesn't
reach 0/130 at its ends, set `POT_RAW_MIN` / `POT_RAW_MAX` from the raw readings.

**LEDs**: each through a series resistor (~220–330 Ω). GPIO14 (red) briefly outputs a PWM
signal during boot, so a short red flicker at power-on is normal.

## Protocol

115200 baud, one message per line (`\n`, `\r` ignored), max 64 chars.

Pi → ESP32

| Command | Meaning |
|---|---|
| `M <left> <right>` | Motor speeds −255..255. Capped to ±`MOTOR_MAX_SPEED` (150). **Must be resent at least every 500 ms** or the watchdog stops the motors. |
| `STOP` | Stop both motors immediately (no ramp) |
| `LIFT <deg> [ms]` | Move lift to 0–130 deg, eased; optional duration in ms (default scales with distance, ~25 ms/deg) |
| `BATT <percent> <0\|1>` | Battery % and charging flag for the LEDs |
| `STATUS <idle\|listening\|thinking\|recording\|armed>` | White LED pattern |
| `PING` | Replies `PONG` |

ESP32 → Pi

| Message | When |
|---|---|
| `IR <bl> <bm> <br> <head>` | On any change, and at least every 500 ms (1 = obstacle) |
| `LIFT_POS <deg>` | Lift finished moving |
| `POT <deg>` | Owner turned the knob (it now controls the lift) |
| `BLOCKED <bl\|bm\|br\|head>` | Forward motion refused/stopped by that sensor |
| `BLOCKED reverse` | Continuous reverse exceeded `MOTOR_MAX_REVERSE_MS` (1.5 s) |
| `PONG` | Reply to `PING` |

`BLOCKED` is rate-limited to one per 500 ms so a Pi resending `M` doesn't flood the link.

### Motion rules (motors.cpp)
Direction is judged by net motion `left + right`:
- **> 0 forward** — refused (motors stopped, `BLOCKED <sensor>`) while any IR sees an
  obstacle; also stopped instantly if an obstacle appears mid-move.
- **< 0 reverse** — wheel speeds capped to `MOTOR_MAX_REVERSE_SPEED` (70); after 1.5 s of
  continuous reverse the motors stop and further reverse is refused until any non-reverse
  command (e.g. `M 0 0` or a turn).
- **= 0 spin in place** (e.g. `M 100 -100`) — always allowed, even with an obstacle ahead.
- Speed changes ramp over `MOTOR_RAMP_MS` (250 ms for 0 → 150). `STOP`, the watchdog, the
  obstacle reflex and the reverse limit all stop *instantly*, not ramped.

### Lift / knob arbitration
The knob position at boot is the reference. Turning it more than 3 deg (`POT_DEADBAND_DEG`)
hands it control: `POT <deg>` is sent and the lift tracks the knob at 60 deg/s. The next
`LIFT` command from the Pi takes control back (the knob then has to be turned again).

## Testing by hand (serial monitor)

Commands typed in the USB serial monitor are handled exactly like the Pi's, with replies
printed back to USB. Lines starting with `[tx]` are what was sent to the Pi; `[rx usb]`
echoes what you typed; `ERR ...` means a malformed command.

Test each module with the robot **on blocks (wheels off the ground)**:

1. **Link** — `PING` → `PONG`.
2. **IR** — wave a hand in front of each sensor; watch `[tx] IR 0 0 0 0` change. Unplug a
   sensor: it should read `1` (fail-safe).
3. **Motors**
   - `M 100 100` → both wheels ramp up, then stop ~0.5 s later with
     `[motors] watchdog ...` — **that's correct**: one typed command only lasts 500 ms.
     To see sustained motion, paste the same line repeatedly or test from the Pi.
   - `M 255 255` → capped to 150.
   - Block a sensor, then `M 100 100` → `[tx] BLOCKED bl` (or whichever), wheels stay still.
   - With a sensor blocked, `M 100 -100` → spins (allowed).
   - `M -200 -200` → slow reverse (capped to 70).
   - `STOP` → instant stop.
4. **Lift** (verify `LIFT_TOP_MIRRORED` first!) — `LIFT 10`, `LIFT 0`, then `LIFT 130`,
   `LIFT 0 6000` (slow). Each ends with `[tx] LIFT_POS <deg>`. `LIFT 200` clamps to 130.
5. **Knob** — turn the potentiometer: `[lift] knob took over` + `[tx] POT <deg>`, lift
   follows. Then `LIFT 40` → Pi has control again.
6. **Battery LEDs** — at boot: yellow blink (no data). `BATT 80 0` green, `BATT 40 0`
   yellow, `BATT 15 0` red, `BATT 3 0` blinking red, `BATT 40 1` pulsing yellow. Wait 60 s
   without a `BATT` → back to yellow blink.
7. **Status LED** — `STATUS listening` solid, `STATUS thinking` slow pulse,
   `STATUS recording` blink, `STATUS armed` short blip every 3 s, `STATUS idle` off.

### Testing the real link from the Pi
Once wired, from the Pi:
```bash
sudo apt install -y python3-serial   # or pip install pyserial in the venv
python3 -m serial.tools.miniterm /dev/serial0 115200
```
Type `PING` → `PONG`; you should also see `IR ...` lines every 500 ms.
