// Project Qwen - ESP32 firmware: module interfaces.
// Each module is non-blocking: begin() once in setup(), update(now) every loop().
#pragma once
#include <Arduino.h>

// Pi link: line protocol on Serial2 (Pi) and Serial (USB, for hand-typed testing).
namespace pilink {
enum class Source { Pi, Usb };
void begin();
void update(uint32_t now);
// Unsolicited ESP32 -> Pi message (no trailing newline). `echo` mirrors it to the USB log.
void send(const char* msg, bool echo = true);
}

// IR obstacle sensors (debounced).
namespace ir {
enum Sensor { BOTTOM_LEFT, BOTTOM_MIDDLE, BOTTOM_RIGHT, HEAD, COUNT };
void begin();
void update(uint32_t now);
bool obstacle(int sensor);
int firstObstacle();          // index of first sensor reporting an obstacle, or -1
const char* name(int sensor); // protocol name used in BLOCKED: bl / bm / br / head
}

// Drive motors: ramping, watchdog, obstacle reflex, reverse limits.
namespace motors {
void begin();
void command(int left, int right, uint32_t now);  // from "M <l> <r>"
void velocity(int leftMmS, int rightMmS, uint32_t now);  // from "V <l> <r>": held with odometry
void stop();                                      // immediate, no ramp
void update(uint32_t now);
}

// Scissor lift servos + manual potentiometer arbitration.
namespace lift {
void begin();
void moveTo(int deg, int32_t ms, uint32_t now);   // ms < 0 -> duration from distance
void update(uint32_t now);
}

// Wheel odometry from the hub motors' hall sensors (interrupt-driven).
namespace odom {
void begin();
void update(uint32_t now);      // sends ODOM <left> <right> every ODOM_REPORT_MS
void reset();                   // zero both counts
float speed(int wheel);         // measured wheel speed, mm/s (+ forward); 0 = left, 1 = right
void counts(int32_t& left, int32_t& right);   // current counts (+ forward), same as ODOM reports
}

// Battery LEDs + white status LED.
namespace leds {
void begin();
void setBattery(int percent, bool charging, uint32_t now);
bool setStatus(const char* name);   // false if the name is unknown
void update(uint32_t now);
}
