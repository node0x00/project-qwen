// Project Qwen - ESP32 firmware: pins and tuning constants.
// Everything you might need to adjust on the real robot lives here.
#pragma once
#include <Arduino.h>

// ---------------------------------------------------------------------------
// Pi link (UART2, direct wire to Pi GPIO14/15 - NOT the USB port)
// ---------------------------------------------------------------------------
constexpr int PIN_LINK_RX = 22;   // <- Pi GPIO14 (TXD)
constexpr int PIN_LINK_TX = 21;   // -> Pi GPIO15 (RXD)
constexpr uint32_t LINK_BAUD = 115200;
constexpr uint32_t DEBUG_BAUD = 115200;   // USB Serial: human-readable logs only
constexpr size_t LINK_LINE_MAX = 64;           // longer lines are dropped as garbage

// ---------------------------------------------------------------------------
// IR obstacle sensors
// ---------------------------------------------------------------------------
constexpr int PIN_IR_BOTTOM_LEFT   = 15;
constexpr int PIN_IR_BOTTOM_MIDDLE = 35;  // input-only pin, no internal pull: add external 10k pull-down
constexpr int PIN_IR_BOTTOM_RIGHT  = 4;
constexpr int PIN_IR_HEAD          = 16;   // NOTE: 16 is left-wheel odometry now; needs a new pin if the head IR is fitted
// The head IR isn't fitted yet (servo weight limit). An unconnected sensor
// reads as "obstacle" (fail-safe), which would block all forward driving, so
// it's ignored until this is set back to true. Its code stays in place.
constexpr bool IR_HEAD_INSTALLED = false;

// Most IR obstacle modules pull their output LOW when they see something.
// Flip to false if yours are active HIGH.
constexpr bool IR_ACTIVE_LOW = true;
constexpr uint32_t IR_DEBOUNCE_MS = 30;
constexpr uint32_t IR_REPORT_INTERVAL_MS = 500;   // send IR at least this often

// ---------------------------------------------------------------------------
// Drive motors (PWM + DIR per hub-motor controller)
// ---------------------------------------------------------------------------
constexpr int PIN_MOTOR_L_PWM = 17;
constexpr int PIN_MOTOR_L_DIR = 13;
constexpr int PIN_MOTOR_R_PWM = 18;
constexpr int PIN_MOTOR_R_DIR = 5;    // soldered to GPIO5 on the perfboard (was 23 in the original pin map). Strapping pin: fine as an output after boot

constexpr uint32_t MOTOR_PWM_FREQ = 20000;   // 20 kHz: above hearing; lower if your controller wants ~1 kHz
constexpr int MOTOR_PWM_BITS = 10;           // hardware resolution; speeds stay in 0-255 units (x4 inside)
constexpr bool MOTOR_L_INVERT = false;       // flip if a wheel spins backwards for "forward"
constexpr bool MOTOR_R_INVERT = true;        // (hub motors are mirrored, one of these is often true)

constexpr int MOTOR_MAX_SPEED = 150;         // hard cap on |speed| regardless of what the Pi asks
constexpr uint32_t MOTOR_RAMP_MS = 250;      // time to ramp from 0 to MOTOR_MAX_SPEED
constexpr uint32_t MOTOR_WATCHDOG_MS = 500;  // stop if no M command for this long

// No rear sensors: reversing is slow and short.
constexpr int MOTOR_MAX_REVERSE_SPEED = 70;
constexpr uint32_t MOTOR_MAX_REVERSE_MS = 1500;   // max continuous reverse before forced stop

constexpr uint32_t BLOCKED_REPORT_INTERVAL_MS = 500;   // don't spam BLOCKED on every keepalive

// ---------------------------------------------------------------------------
// Wheel odometry: two of each hub motor's three hall sensors (Ha, Hb), tapped
// from the ZS-X11H hall connector through 10k/20k dividers (5 V -> 3.3 V).
// Ha/Hb are 120 deg apart, so (Ha,Hb) steps 10 -> 11 -> 01 -> 00 when turning
// one way: 4 counts per electrical cycle x 15 pole pairs = 60 counts/turn,
// with direction from the step order.
// ---------------------------------------------------------------------------
constexpr int PIN_ODOM_L_A = 16;   // board has no GPIO36/39 broken out (2026-10-15)
constexpr int PIN_ODOM_L_B = 23;
constexpr int PIN_ODOM_R_A = 32;
constexpr int PIN_ODOM_R_B = 19;
constexpr bool ODOM_L_INVERT = true;    // flip so "forward" counts up on each side
constexpr bool ODOM_R_INVERT = false;   // (mirrored hub motors: one usually needs flipping)
constexpr uint32_t ODOM_REPORT_MS = 50; // ODOM <left> <right> to the Pi at 20 Hz
constexpr float ODOM_MM_PER_COUNT = 8.14f;   // 155.4 mm wheel, 60 counts per turn (both wheels, 2026-10-16)
constexpr uint32_t SPEED_TIMEOUT_US = 300000; // no hall edge for this long = wheel stopped

// ---------------------------------------------------------------------------
// Speed control: "V <left mm/s> <right mm/s>" holds real wheel speeds using the
// odometry, so slow means slow at any battery voltage and both wheels match.
// Output is in the same 0-255 speed units as M (fractions allowed: 10-bit PWM).
// ---------------------------------------------------------------------------
constexpr uint32_t SPEED_LOOP_MS = 20;
constexpr float SPEED_FF = 0.025f;       // feed-forward: speed units per mm/s (measured ~40 mm/s per unit, wheels lifted)
constexpr float SPEED_KP = 0.008f;       // speed units per mm/s of error
constexpr float SPEED_KI = 0.060f;       // speed units per (mm/s x s) of accumulated error
constexpr float SPEED_I_MAX = 15.0f;     // integrator clamp (speed units)
constexpr float SPEED_MIN_OUT = 0.25f;   // smallest drive while a wheel should turn
constexpr float SPEED_MAX_OUT = 40.0f;   // cap in speed mode (well below MOTOR_MAX_SPEED)
constexpr int SPEED_MAX_MM_S = 600;      // refuse faster targets
constexpr float SPEED_ACCEL_MM_S2 = 200.0f;  // gentle starts/stops: the rear caster lifts and slams on hard ones
constexpr float SPEED_SYNC_GAIN = 12.0f;     // mm/s of correction per count the wheels drift apart (straight/spin)
constexpr float SPEED_REVERSE_MAX_MM = 450.0f;  // speed mode: reverse limit by DISTANCE (odometry), not time

// ---------------------------------------------------------------------------
// Scissor lift (2x MG996R, always the same angle)
// ---------------------------------------------------------------------------
constexpr int PIN_LIFT_SERVO_BOTTOM = 25;
constexpr int PIN_LIFT_SERVO_TOP    = 33;

constexpr int LIFT_MIN_DEG = 0;      // lowest
constexpr int LIFT_MAX_DEG = 130;    // highest
constexpr int LIFT_BOOT_DEG = 0;     // position commanded at power-on (servos jump here once)

// Set true if the top servo is mounted mirrored (it then gets 180 - angle).
// Verify on the real lift with small moves before trusting full range!
constexpr bool LIFT_TOP_MIRRORED = false;

constexpr int SERVO_MIN_US = 500;    // MG996R pulse range for 0..180 deg
constexpr int SERVO_MAX_US = 2500;

// When LIFT has no [ms], the move duration scales with distance - slow and organic.
constexpr uint32_t LIFT_MS_PER_DEG = 25;     // 130 deg -> ~3.3 s
constexpr uint32_t LIFT_MIN_MOVE_MS = 400;
constexpr uint32_t LIFT_MAX_MOVE_MS = 20000;  // clamp for [ms] from the Pi
constexpr uint32_t LIFT_UPDATE_MS = 20;       // servo refresh tick

// ---------------------------------------------------------------------------
// Manual height potentiometer
// ---------------------------------------------------------------------------
constexpr int PIN_POT = 34;               // ADC1 (works with WiFi on, unlike ADC2)
constexpr int POT_RAW_MIN = 0;            // calibrate: raw reading at the knob's "lowest" end
constexpr int POT_RAW_MAX = 4095;         // calibrate: raw reading at the knob's "highest" end
constexpr int POT_SAMPLES = 16;           // averaged per reading
constexpr uint32_t POT_READ_MS = 20;
constexpr int POT_DEADBAND_DEG = 3;       // knob must move more than this to take over
constexpr float POT_FOLLOW_DEG_PER_S = 60.0f;  // how fast the lift tracks the knob

// ---------------------------------------------------------------------------
// LEDs
// ---------------------------------------------------------------------------
constexpr int PIN_LED_GREEN  = 26;
constexpr int PIN_LED_YELLOW = 27;
constexpr int PIN_LED_RED    = 14;
// White status LED. Moved from GPIO19 to GPIO12. GPIO12 is a boot strapping pin:
// it must NOT be held HIGH at power-on (the ESP32 then selects 1.8 V flash and
// fails to boot), so wire the LED GPIO12 -> resistor -> LED -> GND, never a pull-up.
constexpr int PIN_LED_STATUS = 12;   // white

constexpr uint32_t LED_PWM_FREQ = 5000;
constexpr int LED_PWM_BITS = 8;

constexpr int BATT_GREEN_ABOVE = 50;     // > 50% green
constexpr int BATT_RED_AT_OR_BELOW = 20; // <= 20% red (21-50 yellow)
constexpr int BATT_CRITICAL_AT_OR_BELOW = 5;  // <= 5% blinking red
constexpr uint32_t BATT_STALE_MS = 60000;     // no BATT for this long -> "no data" pattern
