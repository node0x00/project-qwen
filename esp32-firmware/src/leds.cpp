// Battery LEDs (green / yellow / red) and the white status LED.
#include <ESP32PWM.h>
#include "config.h"
#include "modules.h"

namespace leds {
namespace {

enum class Status { Idle, Listening, Thinking, Recording, Armed };
const char* STATUS_NAMES[] = {"idle", "listening", "thinking", "recording", "armed"};

ESP32PWM pwmGreen, pwmYellow, pwmRed, pwmStatus;
Status status = Status::Idle;
bool haveBattery = false;
int batteryPct = 0;
bool charging = false;
uint32_t lastBattMs = 0;

// 0..1 brightness with gamma so pulses look even to the eye.
void set(ESP32PWM& pwm, float level) {
  level = level < 0 ? 0 : (level > 1 ? 1 : level);
  pwm.write((uint32_t)(level * level * 255.0f + 0.5f));
}

float pulse(uint32_t now, uint32_t periodMs, float minLevel) {
  float phase = (float)(now % periodMs) / periodMs;
  float s = 0.5f - 0.5f * cosf(2.0f * PI * phase);
  return minLevel + (1.0f - minLevel) * s;
}

bool blink(uint32_t now, uint32_t onMs, uint32_t periodMs) { return now % periodMs < onMs; }

void updateBattery(uint32_t now) {
  float g = 0, y = 0, r = 0;
  if (!haveBattery || now - lastBattMs > BATT_STALE_MS) {
    y = blink(now, 500, 1000) ? 1 : 0;   // no data: yellow blink
  } else {
    float level;
    if (charging) level = pulse(now, 3000, 0.08f);
    else if (batteryPct <= BATT_CRITICAL_AT_OR_BELOW) level = blink(now, 250, 500) ? 1 : 0;
    else level = 1;

    if (batteryPct > BATT_GREEN_ABOVE) g = level;
    else if (batteryPct > BATT_RED_AT_OR_BELOW) y = level;
    else r = level;
  }
  set(pwmGreen, g);
  set(pwmYellow, y);
  set(pwmRed, r);
}

void updateStatus(uint32_t now) {
  float w = 0;
  switch (status) {
    case Status::Idle:      w = 0; break;
    case Status::Listening: w = 1; break;
    case Status::Thinking:  w = pulse(now, 2000, 0.05f); break;
    case Status::Recording: w = blink(now, 400, 800) ? 1 : 0; break;
    case Status::Armed:     w = blink(now, 80, 3000) ? 1 : 0; break;
  }
  set(pwmStatus, w);
}

}  // namespace

void begin() {
  pwmGreen.attachPin(PIN_LED_GREEN, LED_PWM_FREQ, LED_PWM_BITS);
  pwmYellow.attachPin(PIN_LED_YELLOW, LED_PWM_FREQ, LED_PWM_BITS);
  pwmRed.attachPin(PIN_LED_RED, LED_PWM_FREQ, LED_PWM_BITS);
  pwmStatus.attachPin(PIN_LED_STATUS, LED_PWM_FREQ, LED_PWM_BITS);
}

void setBattery(int percent, bool isCharging, uint32_t now) {
  batteryPct = percent < 0 ? 0 : (percent > 100 ? 100 : percent);
  charging = isCharging;
  haveBattery = true;
  lastBattMs = now;
}

bool setStatus(const char* name) {
  for (int i = 0; i < 5; i++) {
    if (!strcasecmp(name, STATUS_NAMES[i])) {
      status = (Status)i;
      return true;
    }
  }
  return false;
}

void update(uint32_t now) {
  updateBattery(now);
  updateStatus(now);
}

}  // namespace leds
