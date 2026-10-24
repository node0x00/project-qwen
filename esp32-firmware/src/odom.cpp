// Wheel odometry: counts hall-sensor steps on each hub motor.
//
// Each wheel has two hall lines (A, B) 120 electrical degrees apart. Their
// combined 2-bit state s = (A << 1) | B steps 2 -> 3 -> 1 -> 0 -> 2 one way and
// the reverse the other way, so every valid step is one count with a known
// direction. Same-state or two-step jumps (noise, or a GPIO36/39 glitch from
// the ADC) count as 0, so a brief spike cancels out instead of drifting.
#include "config.h"
#include "modules.h"

namespace odom {
namespace {

// delta[old][new]: +1 forward step, -1 backward step, 0 no/invalid step.
const int8_t DELTA[4][4] = {
    /* old 0 */ {0, -1, +1, 0},
    /* old 1 */ {+1, 0, 0, -1},
    /* old 2 */ {-1, 0, 0, +1},
    /* old 3 */ {0, +1, -1, 0},
};

volatile int32_t countL = 0, countR = 0;
volatile uint8_t stateL = 0, stateR = 0;
// speed: time of the last two valid hall steps per wheel (the hall steps are alternately
// 120 and 60 electrical degrees apart, so a two-step span is the steady measure)
volatile uint32_t lastUsL = 0, spanUsL = 0, prevStepUsL = 0;
volatile uint32_t lastUsR = 0, spanUsR = 0, prevStepUsR = 0;
volatile int8_t dirL = 0, dirR = 0;
portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
uint32_t lastReport = 0;

inline uint8_t readState(int pinA, int pinB) {
  return (uint8_t)((digitalRead(pinA) << 1) | digitalRead(pinB));
}

void IRAM_ATTR onLeft() {
  uint8_t s = readState(PIN_ODOM_L_A, PIN_ODOM_L_B);
  uint32_t t = micros();
  portENTER_CRITICAL_ISR(&mux);
  int8_t d = DELTA[stateL][s];
  if (d) {
    if (d != dirL) { spanUsL = 0; prevStepUsL = 0; }      // direction change: restart timing
    else if (lastUsL) { spanUsL = prevStepUsL ? (t - lastUsL) + prevStepUsL : 0; prevStepUsL = t - lastUsL; }
    lastUsL = t;
    dirL = d;
  }
  countL += d;
  stateL = s;
  portEXIT_CRITICAL_ISR(&mux);
}

void IRAM_ATTR onRight() {
  uint8_t s = readState(PIN_ODOM_R_A, PIN_ODOM_R_B);
  uint32_t t = micros();
  portENTER_CRITICAL_ISR(&mux);
  int8_t d = DELTA[stateR][s];
  if (d) {
    if (d != dirR) { spanUsR = 0; prevStepUsR = 0; }
    else if (lastUsR) { spanUsR = prevStepUsR ? (t - lastUsR) + prevStepUsR : 0; prevStepUsR = t - lastUsR; }
    lastUsR = t;
    dirR = d;
  }
  countR += d;
  stateR = s;
  portEXIT_CRITICAL_ISR(&mux);
}

float speedFrom(uint32_t last, uint32_t span, int8_t dir, bool invert) {
  const uint32_t age = micros() - last;
  if (!last || !span || !dir) return 0.0f;
  // no new step for a while: the wheel has slowed/stopped - use the age as an upper bound
  const uint32_t effective = age > span / 2 ? (age > SPEED_TIMEOUT_US ? 0 : max(span, 2 * age)) : span;
  if (!effective) return 0.0f;
  float v = 2.0f * ODOM_MM_PER_COUNT * 1e6f / (float)effective;
  if (dir < 0) v = -v;
  return invert ? -v : v;
}

}  // namespace

void begin() {
  // The dividers drive these lines; 36/39 have no internal pulls anyway.
  for (int p : {PIN_ODOM_L_A, PIN_ODOM_L_B, PIN_ODOM_R_A, PIN_ODOM_R_B}) pinMode(p, INPUT);
  stateL = readState(PIN_ODOM_L_A, PIN_ODOM_L_B);
  stateR = readState(PIN_ODOM_R_A, PIN_ODOM_R_B);
  attachInterrupt(digitalPinToInterrupt(PIN_ODOM_L_A), onLeft, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ODOM_L_B), onLeft, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ODOM_R_A), onRight, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ODOM_R_B), onRight, CHANGE);
}

void counts(int32_t& left, int32_t& right) {
  portENTER_CRITICAL(&mux);
  left = countL;
  right = countR;
  portEXIT_CRITICAL(&mux);
  if (ODOM_L_INVERT) left = -left;
  if (ODOM_R_INVERT) right = -right;
}

float speed(int wheel) {
  portENTER_CRITICAL(&mux);
  uint32_t last = wheel == 0 ? lastUsL : lastUsR;
  uint32_t span = wheel == 0 ? spanUsL : spanUsR;
  int8_t dir = wheel == 0 ? dirL : dirR;
  portEXIT_CRITICAL(&mux);
  return speedFrom(last, span, dir, wheel == 0 ? ODOM_L_INVERT : ODOM_R_INVERT);
}

void reset() {
  portENTER_CRITICAL(&mux);
  countL = countR = 0;
  portEXIT_CRITICAL(&mux);
}

void update(uint32_t now) {
  if (now - lastReport < ODOM_REPORT_MS) return;
  lastReport = now;
  portENTER_CRITICAL(&mux);
  int32_t l = countL, r = countR;
  portEXIT_CRITICAL(&mux);
  if (ODOM_L_INVERT) l = -l;
  if (ODOM_R_INVERT) r = -r;
  char msg[40];
  snprintf(msg, sizeof(msg), "ODOM %ld %ld", (long)l, (long)r);
  pilink::send(msg, false);    // 20 Hz - don't flood the USB log
}

}  // namespace odom
