// Pi <-> ESP32 line protocol.
//
// Commands are accepted from two places:
//   - Serial2 (GPIO22 RX / GPIO21 TX): the Pi. Replies go back on Serial2.
//   - Serial  (USB): a human typing in the serial monitor, for testing. Replies go to USB.
// Unsolicited messages (IR, LIFT_POS, POT, BLOCKED) always go to the Pi on Serial2 and are
// mirrored to the USB log with a "[tx]" prefix, so Pi traffic and debug text never mix.
#include <stdlib.h>
#include <string.h>
#include "config.h"
#include "modules.h"

namespace pilink {
namespace {

struct LineReader {
  char buf[LINK_LINE_MAX + 1];
  size_t len = 0;
  bool overflow = false;

  // Returns true when a complete line is ready in buf.
  bool feed(char c) {
    if (c == '\r') return false;
    if (c == '\n') {
      bool ok = !overflow && len > 0;
      buf[len] = '\0';
      len = 0;
      overflow = false;
      return ok;
    }
    if (len >= LINK_LINE_MAX) {
      overflow = true;   // drop the rest of this line
      return false;
    }
    buf[len++] = c;
    return false;
  }
};

LineReader piReader;
LineReader usbReader;

void reply(Source src, const char* msg) {
  if (src == Source::Pi) {
    Serial2.print(msg);
    Serial2.print('\n');
    Serial.printf("[tx] %s\n", msg);
  } else {
    Serial.println(msg);
  }
}

bool parseInt(const char* s, long* out) {
  if (!s) return false;
  char* end;
  long v = strtol(s, &end, 10);
  if (end == s || *end != '\0') return false;
  *out = v;
  return true;
}

void bad(Source src, const char* line, const char* why) {
  Serial.printf("[link] rejected '%s' from %s: %s\n", line, src == Source::Pi ? "pi" : "usb", why);
  if (src == Source::Usb) Serial.printf("ERR %s\n", why);
}

void handleLine(Source src, char* line, uint32_t now) {
  char original[LINK_LINE_MAX + 1];
  strncpy(original, line, sizeof(original));
  original[LINK_LINE_MAX] = '\0';

  char* save;
  char* cmd = strtok_r(line, " \t", &save);
  if (!cmd) return;
  char* a1 = strtok_r(nullptr, " \t", &save);
  char* a2 = strtok_r(nullptr, " \t", &save);
  char* extra = strtok_r(nullptr, " \t", &save);
  if (extra) return bad(src, original, "too many arguments");

  if (src == Source::Usb) Serial.printf("[rx usb] %s\n", original);

  long x, y;
  if (!strcasecmp(cmd, "M")) {
    if (!parseInt(a1, &x) || !parseInt(a2, &y)) return bad(src, original, "usage: M <left> <right>");
    motors::command((int)x, (int)y, now);
  } else if (!strcasecmp(cmd, "V")) {
    if (!parseInt(a1, &x) || !parseInt(a2, &y)) return bad(src, original, "usage: V <left mm/s> <right mm/s>");
    motors::velocity((int)x, (int)y, now);
  } else if (!strcasecmp(cmd, "STOP")) {
    if (a1) return bad(src, original, "usage: STOP");
    motors::stop();
  } else if (!strcasecmp(cmd, "LIFT")) {
    if (!parseInt(a1, &x)) return bad(src, original, "usage: LIFT <deg> [ms]");
    long ms = -1;
    if (a2 && (!parseInt(a2, &ms) || ms < 0)) return bad(src, original, "usage: LIFT <deg> [ms]");
    lift::moveTo((int)x, (int32_t)ms, now);
  } else if (!strcasecmp(cmd, "BATT")) {
    if (!parseInt(a1, &x) || !parseInt(a2, &y) || (y != 0 && y != 1))
      return bad(src, original, "usage: BATT <percent> <charging 0|1>");
    leds::setBattery((int)x, y == 1, now);
  } else if (!strcasecmp(cmd, "STATUS")) {
    if (!a1 || a2 || !leds::setStatus(a1))
      return bad(src, original, "usage: STATUS <idle|listening|thinking|recording|armed>");
  } else if (!strcasecmp(cmd, "ODOM_RESET")) {
    if (a1) return bad(src, original, "usage: ODOM_RESET");
    odom::reset();
  } else if (!strcasecmp(cmd, "PING")) {
    reply(src, "PONG");
  } else {
    bad(src, original, "unknown command");
  }
}

}  // namespace

void begin() {
  Serial2.begin(LINK_BAUD, SERIAL_8N1, PIN_LINK_RX, PIN_LINK_TX);
}

void update(uint32_t now) {
  while (Serial2.available()) {
    if (piReader.feed((char)Serial2.read())) handleLine(Source::Pi, piReader.buf, now);
  }
  while (Serial.available()) {
    if (usbReader.feed((char)Serial.read())) handleLine(Source::Usb, usbReader.buf, now);
  }
}

void send(const char* msg, bool echo) {
  Serial2.print(msg);
  Serial2.print('\n');
  if (echo) Serial.printf("[tx] %s\n", msg);
}

}  // namespace pilink
