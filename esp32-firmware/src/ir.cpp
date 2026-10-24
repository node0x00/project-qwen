// IR obstacle sensors: debounced, reported to the Pi on change and every 500 ms.
#include "config.h"
#include "modules.h"

namespace ir {
namespace {

const int PINS[COUNT] = {PIN_IR_BOTTOM_LEFT, PIN_IR_BOTTOM_MIDDLE, PIN_IR_BOTTOM_RIGHT, PIN_IR_HEAD};
const char* NAMES[COUNT] = {"bl", "bm", "br", "head"};

bool stable[COUNT];      // debounced: true = obstacle
bool raw[COUNT];
uint32_t rawSince[COUNT];
uint32_t lastReport = 0;
bool dirty = true;       // report immediately at boot

bool readObstacle(int i) {
  if (i == HEAD && !IR_HEAD_INSTALLED) return false;   // not fitted: always "clear"
  int level = digitalRead(PINS[i]);
  return IR_ACTIVE_LOW ? level == LOW : level == HIGH;
}

}  // namespace

void begin() {
  for (int i = 0; i < COUNT; i++) {
    if (i == HEAD && !IR_HEAD_INSTALLED) continue;   // its pin (16) is used by left odometry now
    // Pull toward the "obstacle" level so an unplugged sensor fails safe (reads as blocked).
    // GPIO34-39 have no internal pulls - an external pull-down is needed on GPIO35.
    if (PINS[i] >= 34) pinMode(PINS[i], INPUT);
    else pinMode(PINS[i], IR_ACTIVE_LOW ? INPUT_PULLDOWN : INPUT_PULLUP);
  }
  uint32_t now = millis();
  for (int i = 0; i < COUNT; i++) {
    raw[i] = stable[i] = readObstacle(i);
    rawSince[i] = now;
  }
}

void update(uint32_t now) {
  for (int i = 0; i < COUNT; i++) {
    bool r = readObstacle(i);
    if (r != raw[i]) {
      raw[i] = r;
      rawSince[i] = now;
    } else if (r != stable[i] && now - rawSince[i] >= IR_DEBOUNCE_MS) {
      stable[i] = r;
      dirty = true;
    }
  }

  if (dirty || now - lastReport >= IR_REPORT_INTERVAL_MS) {
    char msg[24];
    snprintf(msg, sizeof(msg), "IR %d %d %d %d", stable[0], stable[1], stable[2], stable[3]);
    pilink::send(msg, dirty);   // only log changes on USB, not the 2 Hz heartbeat
    dirty = false;
    lastReport = now;
  }
}

bool obstacle(int sensor) { return stable[sensor]; }

int firstObstacle() {
  for (int i = 0; i < COUNT; i++)
    if (stable[i]) return i;
  return -1;
}

const char* name(int sensor) { return NAMES[sensor]; }

}  // namespace ir
