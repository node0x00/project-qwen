// Drive motors: PWM + DIR per hub-motor controller.
//
// Safety rules (all enforced here, never dependent on the Pi):
//   - Watchdog: no M command for MOTOR_WATCHDOG_MS -> immediate stop.
//   - Obstacle reflex: any IR obstacle -> forward motion refused/stopped, BLOCKED sent.
//     Spinning in place and reversing stay allowed so the robot can get itself out.
//   - No rear sensors: reverse is capped to a low speed and a max continuous duration.
//
// "Forward", "reverse" and "spin" are judged by the net motion left + right:
//   > 0 forward, < 0 reverse, == 0 spin in place (e.g. M 100 -100).
#include <ESP32PWM.h>
#include "config.h"
#include "modules.h"

namespace motors {
namespace {

ESP32PWM pwmL, pwmR;
float targetL = 0, targetR = 0;
// speed mode (V command)
bool velMode = false;
float velL = 0, velR = 0;        // requested targets, mm/s
float rampL = 0, rampR = 0;      // targets after the acceleration limit
int32_t baseCountL = 0, baseCountR = 0;   // counts when the current V move started (for wheel sync)
int32_t revStartCountL = 0, revStartCountR = 0;
bool revActive = false;
float integL = 0, integR = 0;
uint32_t lastLoopMs = 0;
uint32_t lastSpdReportMs = 0;
float curL = 0, curR = 0;
uint32_t lastCmdMs = 0;
uint32_t lastUpdateMs = 0;
uint32_t reverseStartMs = 0;   // 0 = not currently reversing
bool reverseLocked = false;    // hit the reverse time limit; cleared by any non-reverse command
uint32_t lastBlockedMs = 0;
bool watchdogTripped = true;   // nothing to stop at boot

int clampSpeed(int v, int lim) { return v > lim ? lim : (v < -lim ? -lim : v); }
float clampf(float v, float lim) { return v > lim ? lim : (v < -lim ? -lim : v); }

void write(ESP32PWM& pwm, int dirPin, float speed, bool invert) {
  bool forward = speed >= 0;
  digitalWrite(dirPin, (forward != invert) ? HIGH : LOW);
  const uint32_t full = (1u << MOTOR_PWM_BITS) - 1;
  uint32_t duty = (uint32_t)(fabsf(speed) * full / 255.0f + 0.5f);   // speeds are in 0-255 units
  pwm.write(duty > full ? full : duty);
}

void output() {
  write(pwmL, PIN_MOTOR_L_DIR, curL, MOTOR_L_INVERT);
  write(pwmR, PIN_MOTOR_R_DIR, curR, MOTOR_R_INVERT);
}

void haltNow() {
  targetL = targetR = 0;
  curL = curR = 0;
  velMode = false;
  velL = velR = 0;
  rampL = rampR = 0;
  revActive = false;
  integL = integR = 0;
  reverseStartMs = 0;
  output();
}

void reportBlocked(const char* what, uint32_t now) {
  if (lastBlockedMs != 0 && now - lastBlockedMs < BLOCKED_REPORT_INTERVAL_MS) return;
  lastBlockedMs = now;
  char msg[24];
  snprintf(msg, sizeof(msg), "BLOCKED %s", what);
  pilink::send(msg);
}

float approach(float cur, float target, float step) {
  if (cur < target) return fminf(cur + step, (float)target);
  if (cur > target) return fmaxf(cur - step, (float)target);
  return cur;
}

}  // namespace

void begin() {
  pinMode(PIN_MOTOR_L_DIR, OUTPUT);
  pinMode(PIN_MOTOR_R_DIR, OUTPUT);
  pwmL.attachPin(PIN_MOTOR_L_PWM, MOTOR_PWM_FREQ, MOTOR_PWM_BITS);
  pwmR.attachPin(PIN_MOTOR_R_PWM, MOTOR_PWM_FREQ, MOTOR_PWM_BITS);
  haltNow();
  lastUpdateMs = millis();
}

// Shared safety gate for M and V: watchdog refresh, obstacle reflex, reverse lock.
// net > 0 forward, < 0 reverse. Returns false if the move was refused (motors halted).
bool allowMove(int net, uint32_t now) {
  lastCmdMs = now;
  watchdogTripped = false;
  if (net > 0) {
    int blocker = ir::firstObstacle();
    if (blocker >= 0) {
      haltNow();
      reportBlocked(ir::name(blocker), now);
      return false;
    }
  }
  if (net < 0) {
    if (reverseLocked) {
      haltNow();
      reportBlocked("reverse", now);
      return false;
    }
  } else {
    reverseLocked = false;
    reverseStartMs = 0;
  }
  return true;
}

void velocity(int leftMmS, int rightMmS, uint32_t now) {
  leftMmS = clampSpeed(leftMmS, SPEED_MAX_MM_S);
  rightMmS = clampSpeed(rightMmS, SPEED_MAX_MM_S);
  if (!allowMove(leftMmS + rightMmS, now)) return;
  // a direction change or a fresh start resets that wheel's integrator
  if (!velMode || (leftMmS > 0) != (velL > 0) || leftMmS == 0) integL = 0;
  if (!velMode || (rightMmS > 0) != (velR > 0) || rightMmS == 0) integR = 0;
  if (!velMode || leftMmS != (int)velL || rightMmS != (int)velR)
    odom::counts(baseCountL, baseCountR);          // new move: wheel sync starts from here
  if (!velMode) { rampL = 0; rampR = 0; lastLoopMs = now; }
  velMode = true;
  velL = (float)leftMmS;
  velR = (float)rightMmS;
}

// One wheel's speed controller: feed-forward + PI on measured speed. Output in 0-255 units,
// same sign as the target, never below SPEED_MIN_OUT while the wheel should turn.
float speedOut(float target, float measured, float& integ, float dt, float cap) {
  if (target == 0.0f) { integ = 0; return 0.0f; }
  const float err = target - measured;
  integ = clampf(integ + err * dt * SPEED_KI, SPEED_I_MAX);
  float out = target * SPEED_FF + err * SPEED_KP + integ;
  const float mag = fminf(fmaxf(fabsf(out), SPEED_MIN_OUT), cap);
  if ((out > 0) != (target > 0)) return target > 0 ? SPEED_MIN_OUT : -SPEED_MIN_OUT;  // never push backwards
  return target > 0 ? mag : -mag;
}

void command(int left, int right, uint32_t now) {
  velMode = false;
  lastCmdMs = now;
  watchdogTripped = false;

  left = clampSpeed(left, MOTOR_MAX_SPEED);
  right = clampSpeed(right, MOTOR_MAX_SPEED);
  const int net = left + right;

  if (net > 0) {
    int blocker = ir::firstObstacle();
    if (blocker >= 0) {
      haltNow();
      reportBlocked(ir::name(blocker), now);
      return;
    }
  }

  if (net < 0) {
    if (reverseLocked) {
      haltNow();
      reportBlocked("reverse", now);
      return;
    }
    left = clampSpeed(left, MOTOR_MAX_REVERSE_SPEED);
    right = clampSpeed(right, MOTOR_MAX_REVERSE_SPEED);
  } else {
    reverseLocked = false;
    reverseStartMs = 0;
  }

  targetL = left;
  targetR = right;
}

void stop() {
  haltNow();
  watchdogTripped = true;
  reverseLocked = false;   // a new reverse after STOP is a new command with its own time limit
  Serial.println("[motors] STOP");
}

void update(uint32_t now) {
  const uint32_t dt = now - lastUpdateMs;
  lastUpdateMs = now;

  // Watchdog: the Pi must keep sending M while it wants to move.
  if (!watchdogTripped && now - lastCmdMs > MOTOR_WATCHDOG_MS) {
    watchdogTripped = true;
    if (targetL || targetR || curL != 0 || curR != 0) {
      haltNow();
      Serial.println("[motors] watchdog: no M command for 500 ms, stopped");
    }
  }

  // Obstacle reflex while already moving forward (sensor changed mid-move).
  if (targetL + targetR > 0 || curL + curR > 0) {
    int blocker = ir::firstObstacle();
    if (blocker >= 0) {
      haltNow();
      reportBlocked(ir::name(blocker), now);
      Serial.printf("[motors] obstacle reflex: %s, stopped\n", ir::name(blocker));
    }
  }

  // Reverse limit. Speed mode: by distance (odometry), so slow reversing isn't cut into
  // jerky bursts. M mode: by time, as before.
  if (velMode) {
    if (velL + velR < 0) {
      int32_t cl, cr;
      odom::counts(cl, cr);
      if (!revActive) { revActive = true; revStartCountL = cl; revStartCountR = cr; }
      const float backMm = -((cl - revStartCountL) + (cr - revStartCountR)) * 0.5f * ODOM_MM_PER_COUNT;
      if (backMm > SPEED_REVERSE_MAX_MM) {
        haltNow();
        reverseLocked = true;
        reportBlocked("reverse", now);
        Serial.println("[motors] reverse distance limit reached, stopped");
      }
    } else {
      revActive = false;
    }
  } else if (targetL + targetR < 0) {
    if (reverseStartMs == 0) {
      reverseStartMs = now ? now : 1;
    } else if (now - reverseStartMs > MOTOR_MAX_REVERSE_MS) {
      haltNow();
      reverseLocked = true;
      reportBlocked("reverse", now);
      Serial.println("[motors] reverse time limit reached, stopped");
    }
  }

  // Speed mode: the controllers set the outputs directly (they are already smooth).
  if (velMode) {
    if (now - lastLoopMs >= SPEED_LOOP_MS) {
      const float loopDt = (now - lastLoopMs) / 1000.0f;
      lastLoopMs = now;
      const float cap = (velL + velR < 0) ? fminf(SPEED_MAX_OUT, (float)MOTOR_MAX_REVERSE_SPEED) : SPEED_MAX_OUT;
      const float sl = odom::speed(0), sr = odom::speed(1);
      // gentle acceleration / deceleration toward the requested speeds
      const float stepV = SPEED_ACCEL_MM_S2 * fminf(loopDt, 0.1f);
      rampL = approach(rampL, velL, stepV);
      rampR = approach(rampR, velR, stepV);
      // wheel sync: when both wheels should cover the same distance (straight or spin),
      // slow the one that's ahead in proportion to how many counts it leads by
      float tL = rampL, tR = rampR;
      if (fabsf(velL) == fabsf(velR) && velL != 0) {
        int32_t cl, cr;
        odom::counts(cl, cr);
        const float progL = (cl - baseCountL) * (velL > 0 ? 1.0f : -1.0f);
        const float progR = (cr - baseCountR) * (velR > 0 ? 1.0f : -1.0f);
        const float corr = clampf((progL - progR) * SPEED_SYNC_GAIN, fabsf(rampL) * 0.5f);
        tL = rampL - (velL > 0 ? corr : -corr);
        tR = rampR + (velR > 0 ? corr : -corr);
      }
      curL = speedOut(tL, sl, integL, fminf(loopDt, 0.1f), cap);
      curR = speedOut(tR, sr, integR, fminf(loopDt, 0.1f), cap);
      targetL = curL;
      targetR = curR;
      if (now - lastSpdReportMs >= 100) {
        lastSpdReportMs = now;
        char msg[56];
        snprintf(msg, sizeof(msg), "SPD %d %d %.2f %.2f", (int)sl, (int)sr, curL, curR);
        pilink::send(msg, false);
      }
    }
    output();
    return;
  }

  // Smooth acceleration toward the targets.
  if (dt > 0) {
    const float step = (float)MOTOR_MAX_SPEED * dt / MOTOR_RAMP_MS;
    curL = approach(curL, targetL, step);
    curR = approach(curR, targetR, step);
  }
  output();
}

}  // namespace motors
