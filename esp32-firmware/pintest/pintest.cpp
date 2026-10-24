// Perfboard pin test: drives the LED pins HIGH so wiring can be checked with
// the LEDs (or a multimeter: ~3.3 V on each). Not the robot firmware - flash
// the normal build back with:  pio run -e esp32dev -t upload
#include <Arduino.h>

// Battery LEDs green / yellow / red, white status LED (GPIO12 is a strapping
// pin: driving it HIGH after boot is fine, only a pull-up at power-on is not).
const int PINS_HIGH[] = {26, 27, 14, 12};
// Motor PWM / DIR held LOW so the motor controllers never see floating inputs.
const int PINS_LOW[] = {17, 13, 18, 23};

void setup() {
  for (int p : PINS_LOW) { pinMode(p, OUTPUT); digitalWrite(p, LOW); }
  for (int p : PINS_HIGH) { pinMode(p, OUTPUT); digitalWrite(p, HIGH); }
  Serial.begin(115200);
}

void loop() {
  Serial.print("[pintest] HIGH:");
  for (int p : PINS_HIGH) { Serial.print(" GPIO"); Serial.print(p); Serial.print(digitalRead(p) ? "=1" : "=0"); }
  Serial.println("  (motor pins held LOW)");
  delay(2000);
}
