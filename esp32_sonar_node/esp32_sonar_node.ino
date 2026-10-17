/*
  RESURGENCE FLEET - ESP32 ULTRASONIC SENSOR NODE (USB serial to Raspberry Pi)
  ===========================================================================
  Reads 4x HC-SR04 and prints one line ~10 times a second over USB:

      SONAR,<front>,<left>,<right>,<rear>      (cm, 250 = nothing in range)

  The Pi (pi_node.py --sonar-source esp32) reads these lines and sends them to
  the dashboard together with its GPS, LiDAR and IMU data.

  Wiring (ESP32 GPIO numbers):
      Sensor   TRIG    ECHO (through 1k/2k divider!)
      Front    5       18
      Left     19      21
      Right    22      23
      Rear     25      26
  VCC of every HC-SR04 -> ESP32 5V/VIN pin, GND -> ESP32 GND.
  ECHO outputs 5V and the ESP32 is 3.3V only:
      ECHO -> 1k -> ESP32 pin,   ESP32 pin -> 2k -> GND

  No extra libraries needed. Board: "ESP32 Dev Module". Serial monitor: 115200.
*/

const int NUM = 4;
const char* NAMES[NUM] = {"front", "left", "right", "rear"};
const int TRIG[NUM] = {5, 19, 22, 25};
const int ECHO[NUM] = {18, 21, 23, 26};

const int NO_ECHO_CM = 250;          // same value the dashboard already uses
const unsigned long TIMEOUT_US = 25000;  // ~4 m round trip
const unsigned long PERIOD_MS = 100;     // 10 lines per second

int readCm(int trig, int echo) {
  digitalWrite(trig, LOW);
  delayMicroseconds(3);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);
  unsigned long us = pulseIn(echo, HIGH, TIMEOUT_US);
  if (us == 0) return NO_ECHO_CM;
  int cm = (int)(us * 0.0343 / 2.0);
  if (cm < 2) cm = 2;
  if (cm >= NO_ECHO_CM) cm = NO_ECHO_CM;
  return cm;
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < NUM; i++) {
    pinMode(TRIG[i], OUTPUT);
    digitalWrite(TRIG[i], LOW);
    pinMode(ECHO[i], INPUT);
  }
  delay(200);
  Serial.println("# RESURGENCE ESP32 SONAR NODE ready (front,left,right,rear cm)");
}

void loop() {
  unsigned long t0 = millis();
  int d[NUM];
  // One sensor at a time with a short gap so they don't hear each other's echo.
  for (int i = 0; i < NUM; i++) {
    d[i] = readCm(TRIG[i], ECHO[i]);
    delay(15);
  }
  Serial.printf("SONAR,%d,%d,%d,%d\n", d[0], d[1], d[2], d[3]);
  unsigned long used = millis() - t0;
  if (used < PERIOD_MS) delay(PERIOD_MS - used);
}
