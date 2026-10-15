/*
 ==============================================================================
  RESURGENCE FLEET CONTROL ROOM — ARDUINO UNO R4 WiFi NODE (V2.2 PORT)

  Ported from the ESP32 V2.2 node. Same behaviour, same JSON payload.

  Wiring (UNO R4 WiFi, 5V logic):
  1. OLED SSD1306 128x64 (I2C):  SDA -> A4 (or SDA header), SCL -> A5 (or SCL header)
                                 VCC -> 5V, GND -> GND
  2. Ultrasonic HC-SR04 x3 (5V, connect directly, no level shifting needed):
     - Front:  TRIG=D2, ECHO=D3
     - Left:   TRIG=D4, ECHO=D5
     - Right:  TRIG=D6, ECHO=D7
  3. GPS NEO-6M / NEO-8M on Serial1 (pins D0/D1):
     - GPS TX -> D0 (RX1)
     - GPS RX -> leave unconnected (we never send to the GPS; the module's RX
       is 3.3V logic, so this avoids feeding it 5V)
     - Fallback coordinates for indoor testing at VNR VJIET campus
  4. Buzzer: D8     Status LED: D9
  5. Wi-Fi telemetry: HTTP(S) POST to Resurgence Server /update every 1.2 s

  Libraries: TinyGPSPlus, Adafruit GFX, Adafruit SSD1306 (Library Manager).
  WiFiS3 ships with the "Arduino UNO R4 Boards" core.
 ==============================================================================
*/

#include <WiFiS3.h>
#include <TinyGPS++.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ==============================================================================
// 1. CONFIGURATION
// ==============================================================================
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// https:// for Railway/Vercel/Render, or "http://<laptop-ip>:5000/update" on LAN.
const char* SERVER_URL    = "https://YOUR-APP-NAME.up.railway.app/update";

const char* VEHICLE_ID    = "TRUCK_01";
const char* API_KEY       = "PASTE_YOUR_HARDWARE_API_KEY_HERE"; // "" = none
const char* NODE_NAME     = "ESP32_BACKUP"; // kept so the server logic is unchanged

// ==============================================================================
// 2. PIN DEFINITIONS
// ==============================================================================
#define TRIG_PIN 2
#define ECHO_PIN 3

#define HAS_SIDE_SENSORS true
#define TRIG_LEFT_PIN    4
#define ECHO_LEFT_PIN    5
#define TRIG_RIGHT_PIN   6
#define ECHO_RIGHT_PIN   7

#define BUZZER_PIN 8
#define LED_PIN    9

#define gpsSerial Serial1   // D0 = RX1, D1 = TX1
#define GPS_BAUD  9600

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
bool oledReady = false;
uint8_t detectedOledAddr = 0x3C;

// ==============================================================================
// 3. GLOBAL OBJECTS & TIMING
// ==============================================================================
TinyGPSPlus gps;

WiFiClient    plainClient;
WiFiSSLClient sslClient;

bool     urlIsHttps = false;
String   urlHost;
uint16_t urlPort = 80;
String   urlPath;
bool     urlValid = false;

unsigned long lastSendTime = 0;
const unsigned long SEND_INTERVAL_MS = 1200;

unsigned long lastOledTime = 0;
const unsigned long OLED_INTERVAL_MS = 150;

unsigned long lastDiagTime = 0;

unsigned long lastReconnectTime = 0;
const unsigned long RECONNECT_INTERVAL_MS = 10000; // WiFi.begin() blocks, so don't spam it

const unsigned long HTTP_TIMEOUT_MS = 3000;

float fallbackLat = 17.5389;
float fallbackLng = 78.3846;

// ==============================================================================
// 4. HELPERS
// ==============================================================================
String hex2(uint8_t v) {
  String s = String(v, HEX);
  s.toUpperCase();
  if (s.length() < 2) s = "0" + s;
  return "0x" + s;
}

uint8_t scanI2CAddress() {
  Serial.println("[I2C] Scanning I2C bus on A4 (SDA) / A5 (SCL)...");
  uint8_t foundAddr = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.println("[I2C] Device detected at address " + hex2(addr));
      if (addr == 0x3C || addr == 0x3D) foundAddr = addr;
    }
  }
  return foundAddr;
}

// Splits SERVER_URL into scheme, host, port and path (no HTTPClient on the R4).
bool parseUrl(const char* url) {
  String u = url;
  if (u.startsWith("https://"))      { urlIsHttps = true;  u = u.substring(8); urlPort = 443; }
  else if (u.startsWith("http://"))  { urlIsHttps = false; u = u.substring(7); urlPort = 80;  }
  else return false;

  int slash = u.indexOf('/');
  String hostPort = (slash >= 0) ? u.substring(0, slash) : u;
  urlPath = (slash >= 0) ? u.substring(slash) : "/";

  int colon = hostPort.indexOf(':');
  if (colon >= 0) {
    urlHost = hostPort.substring(0, colon);
    urlPort = hostPort.substring(colon + 1).toInt();
  } else {
    urlHost = hostPort;
  }
  return urlHost.length() > 0;
}

// Minimal HTTP/1.1 POST. Returns HTTP status code, or
// -1 = could not connect, -2 = no response before timeout.
int postJson(const String& body, String& response) {
  Client* client = urlIsHttps ? (Client*)&sslClient : (Client*)&plainClient;
  client->setTimeout(HTTP_TIMEOUT_MS);

  if (!client->connect(urlHost.c_str(), urlPort)) return -1;

  client->print("POST "); client->print(urlPath); client->println(" HTTP/1.1");
  client->print("Host: "); client->println(urlHost);
  client->println("Content-Type: application/json");
  if (strlen(API_KEY) > 0) { client->print("X-API-Key: "); client->println(API_KEY); }
  client->print("Content-Length: "); client->println(body.length());
  client->println("Connection: close");
  client->println();
  client->print(body);

  unsigned long start = millis();
  while (!client->available()) {
    if (millis() - start > HTTP_TIMEOUT_MS) { client->stop(); return -2; }
    delay(5);
  }

  // Status line, e.g. "HTTP/1.1 200 OK"
  String statusLine = client->readStringUntil('\n');
  int code = 0;
  int sp = statusLine.indexOf(' ');
  if (sp > 0) code = statusLine.substring(sp + 1, sp + 4).toInt();

  // Skip headers until the blank line
  while (client->connected() || client->available()) {
    String line = client->readStringUntil('\n');
    if (line.length() == 0 || line == "\r") break;
  }

  // Read up to 200 chars of body for logging
  response = "";
  start = millis();
  while ((client->connected() || client->available()) &&
         millis() - start < HTTP_TIMEOUT_MS && response.length() < 200) {
    while (client->available() && response.length() < 200) response += (char)client->read();
  }

  client->stop();
  return code;
}

void connectWiFi() {
  Serial.print("[WiFi] Connecting to: ");
  Serial.println(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(400);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    delay(500); // give DHCP a moment so localIP() isn't 0.0.0.0
    Serial.println("\n[WiFi] CONNECTED!");
    Serial.print("[WiFi] IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n[WiFi WARNING] Connection failed. Will retry from loop.");
  }
}

// ==============================================================================
// 5. SENSOR READING & LOGIC (unchanged)
// ==============================================================================
int readUltrasonicDistance(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH, 18000); // ~3 m max
  if (duration == 0) return 250; // clear
  int distanceCm = duration * 0.034 / 2;
  return constrain(distanceCm, 2, 300);
}

String decideAction(int front, int left, int right) {
  if (front < 15) {
    if (left > right && left > 40) return "TURN LEFT";
    if (right > left && right > 40) return "TURN RIGHT";
    return "STOP";
  }
  if (front < 40) {
    if (left > right && left > 40) return "TURN LEFT";
    if (right > left && right > 40) return "TURN RIGHT";
    return "SLOW DOWN";
  }
  return "CLEAR";
}

// ==============================================================================
// 6. OLED REFRESH
// ==============================================================================
void refreshOled(String action, int front, bool hasFix, int sats, float lat, float lng) {
  if (!oledReady) return;

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  display.setCursor(0, 0);
  display.println("RESURGENCE: TRUCK 1");
  display.drawLine(0, 9, SCREEN_WIDTH, 9, SSD1306_WHITE);

  display.setCursor(0, 13);
  display.print("CMD: ");
  display.println(action);

  display.setCursor(0, 23);
  display.print("Front: ");
  display.print(front);
  display.println(" cm");

  display.setCursor(0, 34);
  display.print("GPS: ");
  if (hasFix) {
    display.print("LOCKED (");
    display.print(sats);
    display.println(" sats)");
  } else {
    display.print("SEARCHING (");
    display.print(sats);
    display.println(")");
  }

  display.setCursor(0, 45);
  display.print("Lat: ");
  display.println(lat, 5);

  display.setCursor(0, 55);
  display.print("Lng: ");
  display.println(lng, 5);

  display.display();
}

// ==============================================================================
// 7. LOCAL HARDWARE ALERTS (unchanged)
// ==============================================================================
void updateLocalAlerts(String action) {
  if (action == "STOP") {
    digitalWrite(LED_PIN, HIGH);
    digitalWrite(BUZZER_PIN, HIGH);
  } else if (action == "SLOW DOWN" || action == "TURN LEFT" || action == "TURN RIGHT") {
    digitalWrite(LED_PIN, HIGH);
    digitalWrite(BUZZER_PIN, (millis() / 300) % 2 == 0 ? HIGH : LOW);
  } else {
    digitalWrite(LED_PIN, LOW);
    digitalWrite(BUZZER_PIN, LOW);
  }
}

// ==============================================================================
// 8. SETUP
// ==============================================================================
void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 3000) {} // wait briefly for USB serial

  Serial.println("\n========================================================");
  Serial.println(" RESURGENCE UNO R4 WiFi TELEMETRY NODE (TRUCK 1) - V2.2");
  Serial.println("========================================================");

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  if (HAS_SIDE_SENSORS) {
    pinMode(TRIG_LEFT_PIN, OUTPUT);
    pinMode(ECHO_LEFT_PIN, INPUT);
    pinMode(TRIG_RIGHT_PIN, OUTPUT);
    pinMode(ECHO_RIGHT_PIN, INPUT);
  }

  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  // I2C on the fixed A4/A5 bus (no custom pins on the R4)
  Wire.begin();
  Wire.setClock(100000);
  delay(120);

  uint8_t oledAddr = scanI2CAddress();
  if (oledAddr == 0) oledAddr = 0x3C;

  // periphBegin=false: Wire is already started above
  if (display.begin(SSD1306_SWITCHCAPVCC, oledAddr, false, false)) {
    oledReady = true;
    detectedOledAddr = oledAddr;
  } else {
    uint8_t altAddr = (oledAddr == 0x3C) ? 0x3D : 0x3C;
    Serial.println("[OLED] " + hex2(oledAddr) + " failed. Trying alternate addr " + hex2(altAddr) + "...");
    if (display.begin(SSD1306_SWITCHCAPVCC, altAddr, false, false)) {
      oledReady = true;
      detectedOledAddr = altAddr;
    }
  }

  if (oledReady) {
    display.dim(false);
    display.clearDisplay();
    display.fillScreen(SSD1306_WHITE); // 400 ms self-test flash
    display.display();
    delay(400);

    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("RESURGENCE: TRUCK 1");
    display.setCursor(0, 18);
    display.println("OLED DISPLAY: ACTIVE");
    display.setCursor(0, 32);
    display.print("I2C Bus: ");
    display.print(hex2(detectedOledAddr));
    display.println(" (100kHz)");
    display.setCursor(0, 48);
    display.println("Connecting WiFi...");
    display.display();
    Serial.println("[OLED SUCCESS] Display active on addr " + hex2(detectedOledAddr) + " (A4/A5)");
  } else {
    Serial.println("[OLED WARNING] SSD1306 not detected on A4/A5.");
    Serial.println("  1. Check OLED VCC -> 5V and GND -> GND.");
    Serial.println("  2. Check OLED SDA -> A4 (or SDA), SCL -> A5 (or SCL).");
  }

  gpsSerial.begin(GPS_BAUD);
  Serial.println("[GPS] Initialized Serial1 (GPS TX -> D0)");

  urlValid = parseUrl(SERVER_URL);
  if (!urlValid) Serial.println("[HTTP WARNING] SERVER_URL must start with http:// or https://");

  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("[WiFi ERROR] Cannot talk to the WiFi module. Telemetry disabled.");
  } else {
    connectWiFi();
  }

  Serial.println("\n--- Telemetry Node Ready. Starting Main Loop ---");
}

// ==============================================================================
// 9. MAIN LOOP
// ==============================================================================
void loop() {
  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  int distFront = readUltrasonicDistance(TRIG_PIN, ECHO_PIN);
  int distLeft = 140;
  int distRight = 140;
  if (HAS_SIDE_SENSORS) {
    distLeft = readUltrasonicDistance(TRIG_LEFT_PIN, ECHO_LEFT_PIN);
    distRight = readUltrasonicDistance(TRIG_RIGHT_PIN, ECHO_RIGHT_PIN);
  }

  String action = decideAction(distFront, distLeft, distRight);
  updateLocalAlerts(action);

  bool gpsValid = false;
  float currentLat = fallbackLat;
  float currentLng = fallbackLng;
  float speedKmh = 0.0;
  int satellites = 0;

  if (gps.satellites.isValid()) satellites = gps.satellites.value();

  if (gps.location.isValid() && gps.location.age() < 3000) {
    gpsValid = true;
    currentLat = gps.location.lat();
    currentLng = gps.location.lng();
    speedKmh = gps.speed.kmph();
  } else {
    fallbackLat += ((float)random(-2, 3)) * 0.00001;
    fallbackLng += ((float)random(-2, 3)) * 0.00001;
    currentLat = fallbackLat;
    currentLng = fallbackLng;
    speedKmh = (action == "STOP") ? 0.0 : ((action == "CLEAR") ? 20.0 : 10.0);
  }

  if (millis() - lastOledTime >= OLED_INTERVAL_MS) {
    lastOledTime = millis();
    refreshOled(action, distFront, gpsValid, satellites, currentLat, currentLng);
  }

  if (millis() - lastDiagTime >= 2500) {
    lastDiagTime = millis();
    Serial.println("----------------------------------------------------------------");
    Serial.println("[GPS DIAGNOSTIC] Chars Processed: " + String(gps.charsProcessed()) +
                   " | Valid Checksums: " + String(gps.passedChecksum()) +
                   " | Failed: " + String(gps.failedChecksum()));
    Serial.println("[GPS STATUS] Satellites: " + String(satellites) +
                   " | Lock: " + String(gpsValid ? "3D_FIX_ACTIVE" : "NO_FIX_SEARCHING") +
                   " | Lat: " + String(currentLat, 6) + " | Lng: " + String(currentLng, 6));
    if (!gpsValid) {
      Serial.println("  -> Tip: If satellites=0, the GPS is still searching.");
      Serial.println("          PPS LED solid/off = searching, blinking = 3D lock.");
    }
  }

  if (millis() - lastSendTime >= SEND_INTERVAL_MS) {
    lastSendTime = millis();

    if (WiFi.status() == WL_CONNECTED && urlValid) {
      String jsonPayload = "{";
      jsonPayload += "\"vehicle_id\":\"" + String(VEHICLE_ID) + "\",";
      jsonPayload += "\"node\":\"" + String(NODE_NAME) + "\",";
      jsonPayload += "\"dist_front\":" + String(distFront) + ",";
      jsonPayload += "\"dist_left\":" + String(distLeft) + ",";
      jsonPayload += "\"dist_right\":" + String(distRight) + ",";
      jsonPayload += "\"action\":\"" + action + "\",";
      jsonPayload += "\"lat\":" + String(currentLat, 6) + ",";
      jsonPayload += "\"lng\":" + String(currentLng, 6) + ",";
      jsonPayload += "\"speed\":" + String(speedKmh, 1) + ",";
      jsonPayload += "\"gps_valid\":" + String(gpsValid ? "true" : "false");
      jsonPayload += "}";

      String resp;
      int httpCode = postJson(jsonPayload, resp);

      if (httpCode > 0) {
        Serial.println("[POST " + String(httpCode) + "] Front=" + String(distFront) +
                       "cm | Act=" + action + " | Resp: " + resp);
      } else if (httpCode == -1) {
        Serial.println("[POST ERROR] Could not connect to " + urlHost + ":" + String(urlPort));
      } else {
        Serial.println("[POST ERROR] No response before timeout");
      }
    } else if (WiFi.status() != WL_CONNECTED &&
               millis() - lastReconnectTime >= RECONNECT_INTERVAL_MS) {
      lastReconnectTime = millis();
      Serial.println("[WiFi] Lost connection. Reconnecting...");
      WiFi.disconnect();
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    }
  }
}
