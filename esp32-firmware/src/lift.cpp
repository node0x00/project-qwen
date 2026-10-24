// Scissor lift (2x MG996R) + manual height potentiometer.
//
// Both servos always get the same angle (top mirrored if LIFT_TOP_MIRRORED), clamped to
// 0-130. Pi LIFT commands move with an ease-in/out curve. The knob takes control only
// when it is physically turned beyond the deadband; the next LIFT command from the Pi
// takes control back.
#include <ESP32Servo.h>
#include "config.h"
#include "modules.h"

namespace lift {
namespace {

enum class Mode { Idle, Easing, FollowKnob };

Servo servoBottom, servoTop;
Mode mode = Mode::Idle;
float pos = LIFT_BOOT_DEG;

float easeFrom = 0, easeTo = 0;
uint32_t easeStart = 0, easeDuration = 1;

float knobTarget = 0;
bool knobMoving = false;
int potRef = 0;            // knob angle when last acknowledged; moves beyond deadband take over
int potDeg = 0;            // latest averaged knob angle
uint32_t lastPotRead = 0;
uint32_t lastTick = 0;

float clampDeg(float d) { return d < LIFT_MIN_DEG ? LIFT_MIN_DEG : (d > LIFT_MAX_DEG ? LIFT_MAX_DEG : d); }

int servoMicros(float deg) { return SERVO_MIN_US + (int)((SERVO_MAX_US - SERVO_MIN_US) * deg / 180.0f + 0.5f); }

void writeServos(float deg) {
  servoBottom.writeMicroseconds(servoMicros(deg));
  servoTop.writeMicroseconds(servoMicros(LIFT_TOP_MIRRORED ? 180.0f - deg : deg));
}

int readPotDeg() {
  long sum = 0;
  for (int i = 0; i < POT_SAMPLES; i++) sum += analogRead(PIN_POT);
  long raw = sum / POT_SAMPLES;
  long deg = (raw - POT_RAW_MIN) * (LIFT_MAX_DEG - LIFT_MIN_DEG) / (POT_RAW_MAX - POT_RAW_MIN) + LIFT_MIN_DEG;
  return (int)clampDeg((float)deg);
}

void sendPos(const char* tag, int deg) {
  char msg[24];
  snprintf(msg, sizeof(msg), "%s %d", tag, deg);
  pilink::send(msg);
}

void updateKnob(uint32_t now) {
  if (now - lastPotRead < POT_READ_MS) return;
  lastPotRead = now;
  potDeg = readPotDeg();
  if (abs(potDeg - potRef) <= POT_DEADBAND_DEG) return;

  potRef = potDeg;
  if (mode != Mode::FollowKnob) Serial.println("[lift] knob took over");
  mode = Mode::FollowKnob;
  knobTarget = potDeg;
  knobMoving = true;
  sendPos("POT", potDeg);
}

}  // namespace

void begin() {
  analogReadResolution(12);
  servoBottom.setPeriodHertz(50);
  servoTop.setPeriodHertz(50);
  servoBottom.attach(PIN_LIFT_SERVO_BOTTOM, SERVO_MIN_US, SERVO_MAX_US);
  servoTop.attach(PIN_LIFT_SERVO_TOP, SERVO_MIN_US, SERVO_MAX_US);
  pos = clampDeg(LIFT_BOOT_DEG);
  writeServos(pos);
  // Wherever the knob sits at boot is the reference - it must be turned to take over.
  potDeg = potRef = readPotDeg();
  lastTick = lastPotRead = millis();
}

void moveTo(int deg, int32_t ms, uint32_t now) {
  const float target = clampDeg((float)deg);
  const float distance = fabsf(target - pos);
  uint32_t duration;
  if (ms >= 0) duration = (uint32_t)ms > LIFT_MAX_MOVE_MS ? LIFT_MAX_MOVE_MS : (uint32_t)ms;
  else duration = max<uint32_t>(LIFT_MIN_MOVE_MS, (uint32_t)(distance * LIFT_MS_PER_DEG));

  potRef = potDeg;   // Pi takes control back; knob must move again to take over
  knobMoving = false;
  easeFrom = pos;
  easeTo = target;
  easeStart = now;
  easeDuration = duration > 0 ? duration : 1;
  mode = Mode::Easing;
  Serial.printf("[lift] %.0f -> %.0f deg over %lu ms\n", easeFrom, easeTo, (unsigned long)easeDuration);
}

void update(uint32_t now) {
  updateKnob(now);

  const uint32_t dt = now - lastTick;
  if (dt < LIFT_UPDATE_MS) return;
  lastTick = now;

  if (mode == Mode::Easing) {
    float t = (float)(now - easeStart) / easeDuration;
    if (t >= 1.0f) {
      pos = easeTo;
      mode = Mode::Idle;
      writeServos(pos);
      sendPos("LIFT_POS", (int)lroundf(pos));
      return;
    }
    const float eased = 0.5f - 0.5f * cosf(PI * t);   // ease in/out
    pos = easeFrom + (easeTo - easeFrom) * eased;
    writeServos(pos);
  } else if (mode == Mode::FollowKnob && knobMoving) {
    const float step = POT_FOLLOW_DEG_PER_S * dt / 1000.0f;
    if (fabsf(knobTarget - pos) <= step) {
      pos = knobTarget;
      knobMoving = false;
      writeServos(pos);
      sendPos("LIFT_POS", (int)lroundf(pos));
      return;
    }
    pos += knobTarget > pos ? step : -step;
    writeServos(pos);
  }
}

}  // namespace lift
