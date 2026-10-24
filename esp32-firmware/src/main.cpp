// Project Qwen - ESP32 firmware entry point.
//
// The ESP32 owns everything real-time: drive motors, IR obstacle reflex, scissor lift,
// manual height knob, and LEDs. The Pi sends high-level commands over a direct UART
// link (Serial2); see README.md for the protocol. No delay() anywhere in the loop -
// the IR safety checks must never be blocked.
#include <Arduino.h>
#include "config.h"
#include "modules.h"

void setup() {
  Serial.begin(DEBUG_BAUD);
  // Motors first, so the drive outputs are forced off as early as possible.
  motors::begin();
  ir::begin();
  pilink::begin();
  lift::begin();
  leds::begin();
  odom::begin();
  Serial.println("[boot] Project Qwen ESP32 firmware ready. Type commands here to test (e.g. PING).");
}

void loop() {
  const uint32_t now = millis();
  ir::update(now);       // fresh obstacle state before anything can move
  pilink::update(now);   // parse + dispatch incoming commands
  motors::update(now);
  lift::update(now);
  leds::update(now);
  odom::update(now);
}
