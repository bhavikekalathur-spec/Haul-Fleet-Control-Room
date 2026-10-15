/*
 ==============================================================================
  RESURGENCE FLEET CONTROL ROOM — ESP32 HARDWARE PROTOTYPE NODE (V2.2 REVISED)
  
  Key Features:
  1. OLED Display (SSD1306 128x64 on SDA=13, SCL=32):
     - Critical Fix: periphBegin=false prevents Adafruit_SSD1306 from reverting
       Wire pins back to default GPIO 21 & GPIO 22 (which conflict with sensors!)
     - Auto-fallback for both 0x3C and 0x3D I2C addresses
     - 100kHz standard mode for high noise immunity with internal pull-ups
     - 400ms white-flash hardware self-test on boot to verify OLED pixels & power
     - Throttled refresh rate (150ms) to prevent I2C bus lockup
  2. Ultrasonic Collision Avoidance (HC-SR04 x3):
     - Front:  TRIG=5,  ECHO=18
     - Left:   TRIG=19, ECHO=21
     - Right:  TRIG=22, ECHO=23
  3. GPS Module (NEO-6M / NEO-8M via HardwareSerial 2):
     - RX2 = GPIO 16 (connected to GPS TX)
     - TX2 = GPIO 17 (connected to GPS RX)
     - Fallback coordinates for indoor testing at VNR VJIET campus (17.5389, 78.3846)
  4. Local Annunciation:
     - Buzzer: GPIO 25
     - Status LED: GPIO 26
  5. Wi-Fi Telemetry:
     - HTTP POST to Resurgence Server /update endpoint every 1.2 seconds
 ==============================================================================
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>   // needed for https:// (Railway/Vercel/Render, etc.)
#include <TinyGPS++.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ==============================================================================
// 1. CONFIGURATION
// ==============================================================================
const char* WIFI_SSID     = "YOUR_WIFI_SSID";       // <-- your phone hotspot / router name
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";   // <-- its password

// Your deployed website's /update endpoint. Use https:// for Railway/Vercel/Render.
// For a laptop on the same LAN instead, use "http://<laptop-ip>:5000/update".
const char* SERVER_URL    = "https://YOUR-APP-NAME.up.railway.app/update";

const char* VEHICLE_ID    = "TRUCK_01"; // Binds to Truck 1 in Resurgence
const char* API_KEY       = "PASTE_YOUR_HARDWARE_API_KEY_HERE"; // must equal the website's HARDWARE_API_KEY ("" = none set)

// ==============================================================================
// 2. PIN DEFINITIONS
// ==============================================================================
#define TRIG_PIN 5
#define ECHO_PIN 18

#define GPS_RX_PIN 16 // GPS module TX -> ESP32 RX2 (GPIO 16)
#define GPS_TX_PIN 17 // GPS module RX -> ESP32 TX2 (GPIO 17)

#define HAS_SIDE_SENSORS true
#define TRIG_LEFT_PIN    19
#define ECHO_LEFT_PIN    21
#define TRIG_RIGHT_PIN   22
#define ECHO_RIGHT_PIN   23

#define BUZZER_PIN 25
#define LED_PIN    26

// OLED display configuration (I2C on custom pins)
#define OLED_SDA 13
#define OLED_SCL 32
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
HardwareSerial gpsSerial(2); // Serial2 on ESP32

unsigned long lastSendTime = 0;
const unsigned long SEND_INTERVAL_MS = 1200; // Send telemetry every 1.2 seconds

unsigned long lastOledTime = 0;
const unsigned long OLED_INTERVAL_MS = 150;  // Update OLED at 6-7 FPS max (prevents I2C lockup)

unsigned long lastDiagTime = 0;

// VNR VJIET Campus coordinates (used when indoors where satellites are blocked)
float fallbackLat = 17.5389;
float fallbackLng = 78.3846;

// ==============================================================================
// 4. I2C SCANNER (AUTO-DETECTS 0x3C OR 0x3D)
// ==============================================================================
uint8_t scanI2CAddress() {
  Serial.println("[I2C] Scanning I2C bus on SDA=13, SCL=32...");
  uint8_t foundAddr = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    uint8_t error = Wire.endTransmission();
    if (error == 0) {
      Serial.printf("[I2C] Device detected at address 0x%02X\n", addr);
      if (addr == 0x3C || addr == 0x3D) {
        foundAddr = addr;
      }
    }
  }
  return foundAddr;
}

// ==============================================================================
// 5. SENSOR READING & LOGIC
// ==============================================================================
int readUltrasonicDistance(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH, 18000); // 18ms timeout (~3m max, prevents loop stalls)
  if (duration == 0) {
    return 250; // Clear
  }
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
// 6. OLED REFRESH (THROTTLED TO PREVENT BUS LOCKUP)
// ==============================================================================
void refreshOled(String action, int front, bool hasFix, int sats, float lat, float lng) {
  if (!oledReady) return;

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  // Header
  display.setCursor(0, 0);
  display.println("RESURGENCE: TRUCK 1");
  display.drawLine(0, 9, SCREEN_WIDTH, 9, SSD1306_WHITE);

  // Status & Front Obstacle
  display.setCursor(0, 13);
  display.print("CMD: ");
  display.println(action);

  display.setCursor(0, 23);
  display.print("Front: ");
  display.print(front);
  display.println(" cm");

  // GPS Satellite Status
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

  // Live Coordinates
  display.setCursor(0, 45);
  display.print("Lat: ");
  display.println(lat, 5);

  display.setCursor(0, 55);
  display.print("Lng: ");
  display.println(lng, 5);

  display.display();
}

// ==============================================================================
// 7. LOCAL HARDWARE ALERTS (LED + BUZZER)
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
  delay(1000);

  Serial.println("\n========================================================");
  Serial.println(" RESURGENCE ESP32 TELEMETRY NODE (TRUCK 1) - V2.2");
  Serial.println("========================================================");

  // Configure Ultrasonic Pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  if (HAS_SIDE_SENSORS) {
    pinMode(TRIG_LEFT_PIN, OUTPUT);
    pinMode(ECHO_LEFT_PIN, INPUT);
    pinMode(TRIG_RIGHT_PIN, OUTPUT);
    pinMode(ECHO_RIGHT_PIN, INPUT);
  }

  // Configure LED + Buzzer
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  // Initialize I2C with internal pullups on GPIO 13 and GPIO 32
  pinMode(OLED_SDA, INPUT_PULLUP);
  pinMode(OLED_SCL, INPUT_PULLUP);
  Wire.begin(OLED_SDA, OLED_SCL, 100000); // 100kHz standard mode for noise immunity
  Wire.setClock(100000);
  delay(120);

  // Scan for I2C devices
  uint8_t oledAddr = scanI2CAddress();
  if (oledAddr == 0) oledAddr = 0x3C; // Default to 0x3C if not detected in scan

  // CRITICAL FIX: The 4th argument 'periphBegin = false' tells Adafruit_SSD1306
  // NOT to call wire->begin() internally, which would otherwise reset the I2C
  // pins back to default GPIO 21 & GPIO 22 (conflicting with ultrasonic sensors)!
  if (display.begin(SSD1306_SWITCHCAPVCC, oledAddr, false, false)) {
    oledReady = true;
    detectedOledAddr = oledAddr;
  } else {
    // Try alternate address 0x3D
    uint8_t altAddr = (oledAddr == 0x3C) ? 0x3D : 0x3C;
    Serial.printf("[OLED] 0x%02X failed. Trying alternate addr 0x%02X...\n", oledAddr, altAddr);
    if (display.begin(SSD1306_SWITCHCAPVCC, altAddr, false, false)) {
      oledReady = true;
      detectedOledAddr = altAddr;
    }
  }

  if (oledReady) {
    display.dim(false); // Max brightness
    display.clearDisplay();
    // Hardware Self-Test: Flash screen white for 400ms so user can physically see pixels ignite
    display.fillScreen(SSD1306_WHITE);
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
    display.printf("I2C Bus: 0x%02X (100kHz)\n", detectedOledAddr);
    display.setCursor(0, 48);
    display.println("Connecting WiFi...");
    display.display();
    Serial.printf("[OLED SUCCESS] Display active on addr 0x%02X (SDA=13, SCL=32)\n", detectedOledAddr);
  } else {
    oledReady = false;
    Serial.println("[OLED WARNING] SSD1306 display not detected on SDA=13, SCL=32.");
    Serial.println("  1. Ensure OLED VCC is plugged into 5V / VIN (not 3.3V) due to onboard 662K regulator.");
    Serial.println("  2. Verify connections: OLED SCL -> GPIO 32, OLED SDA -> GPIO 13.");
    Serial.println("  3. Ensure OLED GND is connected to ESP32 GND.");
  }

  // Initialize GPS Serial port (9600 baud standard for NEO-6M / NEO-8M)
  gpsSerial.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  Serial.println("[GPS] Initialized Serial2 on RX2=GPIO16, TX2=GPIO17");

  // Connect to Wi-Fi
  Serial.print("[WiFi] Connecting to: ");
  Serial.println(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int wifiAttempts = 0;
  while (WiFi.status() != WL_CONNECTED && wifiAttempts < 20) {
    delay(400);
    Serial.print(".");
    wifiAttempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WiFi] CONNECTED!");
    Serial.print("[WiFi] IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n[WiFi WARNING] WiFi connection failed or still connecting in background.");
  }

  Serial.println("\n--- Telemetry Node Ready. Starting Main Loop ---");
}

// ==============================================================================
// 9. MAIN LOOP
// ==============================================================================
void loop() {
  // CRITICAL: Continually feed TinyGPS++ with any incoming GPS NMEA sentences
  while (gpsSerial.available() > 0) {
    char c = gpsSerial.read();
    gps.encode(c);
  }

  // Read Ultrasonic Sensors
  int distFront = readUltrasonicDistance(TRIG_PIN, ECHO_PIN);
  int distLeft = 140;
  int distRight = 140;

  if (HAS_SIDE_SENSORS) {
    distLeft = readUltrasonicDistance(TRIG_LEFT_PIN, ECHO_LEFT_PIN);
    distRight = readUltrasonicDistance(TRIG_RIGHT_PIN, ECHO_RIGHT_PIN);
  }

  String action = decideAction(distFront, distLeft, distRight);

  // Update physical LED & Buzzer
  updateLocalAlerts(action);

  // Read GPS coordinates & Satellites
  bool gpsValid = false;
  float currentLat = fallbackLat;
  float currentLng = fallbackLng;
  float speedKmh = 0.0;
  int satellites = 0;

  if (gps.satellites.isValid()) {
    satellites = gps.satellites.value();
  }

  if (gps.location.isValid() && gps.location.age() < 3000) {
    gpsValid = true;
    currentLat = gps.location.lat();
    currentLng = gps.location.lng();
    speedKmh = gps.speed.kmph();
  } else {
    // Indoor mode: simulate slight movement around VNR VJIET campus
    fallbackLat += ((float)random(-2, 3)) * 0.00001;
    fallbackLng += ((float)random(-2, 3)) * 0.00001;
    currentLat = fallbackLat;
    currentLng = fallbackLng;
    speedKmh = (action == "STOP") ? 0.0 : ((action == "CLEAR") ? 20.0 : 10.0);
  }

  // Update OLED at controlled 150ms interval (prevents I2C bus starvation)
  if (millis() - lastOledTime >= OLED_INTERVAL_MS) {
    lastOledTime = millis();
    refreshOled(action, distFront, gpsValid, satellites, currentLat, currentLng);
  }

  // Periodic Serial Diagnostic Printer (every 2.5s)
  if (millis() - lastDiagTime >= 2500) {
    lastDiagTime = millis();
    Serial.println("----------------------------------------------------------------");
    Serial.printf("[GPS DIAGNOSTIC] Chars Processed: %u | Valid Checksums: %u | Failed: %u\n",
                  gps.charsProcessed(), gps.passedChecksum(), gps.failedChecksum());
    Serial.printf("[GPS STATUS] Satellites: %d | Lock: %s | Lat: %.6f | Lng: %.6f\n",
                  satellites, gpsValid ? "3D_FIX_ACTIVE" : "NO_FIX_SEARCHING", currentLat, currentLng);
    if (!gpsValid) {
      Serial.println("  -> Tip: If satellites=0, the NEO-6M is still searching for satellites.");
      Serial.println("          Check if the small red/green PPS LED on the GPS module is blinking.");
      Serial.println("          Solid/OFF = Searching for sky view. Blinking = 3D Lock Acquired.");
    }
  }

  // Send Telemetry HTTP POST to Dashboard Server (every 1.2s)
  if (millis() - lastSendTime >= SEND_INTERVAL_MS) {
    lastSendTime = millis();

    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      WiFiClientSecure secureClient;   // only used for https:// URLs
      bool isHttps = (strncmp(SERVER_URL, "https://", 8) == 0);

      if (isHttps) {
        // Skips certificate validation. Simple and fine for a student/hobby project;
        // for production you would pin the server's CA certificate instead.
        secureClient.setInsecure();
        http.begin(secureClient, SERVER_URL);
      } else {
        http.begin(SERVER_URL);        // plain http:// (e.g. a laptop on the LAN)
      }
      http.addHeader("Content-Type", "application/json");
      if (strlen(API_KEY) > 0) http.addHeader("X-API-Key", API_KEY);

      String jsonPayload = "{";
      jsonPayload += "\"vehicle_id\":\"" + String(VEHICLE_ID) + "\",";
      jsonPayload += "\"node\":\"ESP32_BACKUP\",";  // Pi is PRIMARY; server uses this only if the Pi goes silent
      jsonPayload += "\"dist_front\":" + String(distFront) + ",";
      jsonPayload += "\"dist_left\":" + String(distLeft) + ",";
      jsonPayload += "\"dist_right\":" + String(distRight) + ",";
      jsonPayload += "\"action\":\"" + action + "\",";
      jsonPayload += "\"lat\":" + String(currentLat, 6) + ",";
      jsonPayload += "\"lng\":" + String(currentLng, 6) + ",";
      jsonPayload += "\"speed\":" + String(speedKmh, 1) + ",";
      jsonPayload += "\"gps_valid\":" + String(gpsValid ? "true" : "false");
      jsonPayload += "}";

      int httpCode = http.POST(jsonPayload);

      if (httpCode > 0) {
        String resp = http.getString();
        Serial.printf("[POST 200] Front=%dcm | Act=%s | Resp: %s\n", distFront, action.c_str(), resp.c_str());
      } else {
        Serial.printf("[POST ERROR] Code: %d (%s)\n", httpCode, http.errorToString(httpCode).c_str());
      }
      http.end();
    } else {
      Serial.println("[WiFi] Lost connection. Reconnecting...");
      WiFi.reconnect();
    }
  }
}
