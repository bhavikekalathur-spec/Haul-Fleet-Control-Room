#!/usr/bin/env python3
"""
Resurgence Fleet - Raspberry Pi hardware node
=============================================
Reads real sensors on the Pi and POSTs them to the dashboard's /update route
(the same route your ESP32 firmware uses), so the Pi shows up as a REAL_HARDWARE
truck on the dashboard.

  GPS (NEO-6M/8M/F9P over UART)  -> lat, lng, speed, heading, gps_valid
  3x HC-SR04 ultrasonic (GPIO)   -> dist_front / dist_left / dist_right  (cm)
  LiDAR (TF-Luna/TFmini/RPLIDAR) -> lidar {front_cm,...}   (--lidar tfluna|rplidar)
  IMU (BNO055/MPU6050, I2C)       -> imu {pitch,roll,accel_g,...}   (--imu)
  Pi camera / USB webcam         -> /api/camera/upload  (optional, --camera)

The Pi identifies itself as node "PI" (PRIMARY). An ESP32 running the updated
firmware posts as "ESP32_BACKUP"; the server switches to it automatically if the
Pi goes silent for 4 s, and switches back when the Pi returns.

Usage
-----
  python3 pi_node.py --server http://192.168.1.50:5000 --vehicle TRUCK_01
  python3 pi_node.py --sim ...            # no hardware needed, fake data
  python3 pi_node.py --no-sonar ...       # skip ultrasonic (e.g. GPS-only test)
  python3 pi_node.py --camera ...         # also stream camera frames
"""
import argparse
import math
import os
import random
import threading
import time

import requests

from pi_sensors import RPLidar, SimIMU, SimLidar, TFLuna, make_imu

# ---------------------------------------------------------------- defaults --
# Every value can be overridden with a CLI flag or an environment variable.
DEFAULT_SERVER = os.environ.get("HAUL_SERVER", "http://192.168.1.50:5000")
DEFAULT_VEHICLE = os.environ.get("HAUL_VEHICLE", "TRUCK_01")
DEFAULT_API_KEY = os.environ.get("HAUL_API_KEY", "")  # must match server HARDWARE_API_KEY

GPS_PORT = os.environ.get("GPS_PORT", "/dev/serial0")
GPS_BAUD = int(os.environ.get("GPS_BAUD", "9600"))  # NEO-6M = 9600, ZED-F9P = 38400/115200

# BCM GPIO numbers (not physical pin numbers).
# NOTE: HC-SR04 ECHO outputs 5V. Use a 1k/2k voltage divider before the Pi pin.
SONAR_PINS = {
    "front": (23, 24),  # (TRIG, ECHO)
    "left": (17, 27),
    "right": (5, 6),
    "rear": (16, 26),   # pin 36 TRIG, pin 37 ECHO
}
NO_ECHO_CM = 250  # value sent when nothing is in range (matches ESP32 firmware)

# Fallback position used when GPS has no fix (indoors). Dashboard ignores 0,0.
FALLBACK_LAT = float(os.environ.get("FALLBACK_LAT", "18.5849"))
FALLBACK_LNG = float(os.environ.get("FALLBACK_LNG", "81.22557"))

SEND_INTERVAL_S = 1.0     # server marks the link LINK_TIMEOUT after 12 s of silence
CAMERA_INTERVAL_S = 2.0   # server runs dehazing per frame, don't go faster


# --------------------------------------------------------------------- GPS --
class GPSReader(threading.Thread):
    """Background thread parsing NMEA $xxRMC / $xxGGA sentences."""

    def __init__(self, port, baud):
        super().__init__(daemon=True)
        self.port, self.baud = port, baud
        self.lat = self.lng = None
        self.speed_kmh = 0.0
        self.heading = None
        self.valid = False
        self.sats = 0
        self.last_fix = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def _deg(value, hemi):
        if not value:
            return None
        dot = value.index(".")
        deg = float(value[: dot - 2])
        minutes = float(value[dot - 2:])
        out = deg + minutes / 60.0
        return -out if hemi in ("S", "W") else out

    @staticmethod
    def _checksum_ok(line):
        try:
            body, cs = line[1:].split("*")
            calc = 0
            for ch in body:
                calc ^= ord(ch)
            return calc == int(cs[:2], 16)
        except Exception:
            return False

    def _parse(self, line):
        if not line.startswith("$") or not self._checksum_ok(line):
            return
        f = line.split("*")[0].split(",")
        kind = f[0][3:]
        with self._lock:
            if kind == "RMC" and len(f) > 8:
                if f[2] == "A":  # A = valid fix
                    self.lat = self._deg(f[3], f[4])
                    self.lng = self._deg(f[5], f[6])
                    self.speed_kmh = float(f[7] or 0) * 1.852  # knots -> km/h
                    if f[8] and self.speed_kmh > 1.0:  # course is noise when stationary
                        self.heading = float(f[8])
                    self.valid = True
                    self.last_fix = time.time()
                else:
                    self.valid = False
            elif kind == "GGA" and len(f) > 7 and f[7]:
                self.sats = int(f[7])

    def run(self):
        import serial  # pip install pyserial
        while True:
            try:
                with serial.Serial(self.port, self.baud, timeout=1) as ser:
                    print(f"[GPS] listening on {self.port} @ {self.baud}")
                    while True:
                        raw = ser.readline().decode("ascii", errors="ignore").strip()
                        if raw:
                            self._parse(raw)
            except Exception as e:
                print(f"[GPS] serial error: {e} - retrying in 3 s")
                time.sleep(3)

    def snapshot(self):
        with self._lock:
            fresh = self.valid and (time.time() - self.last_fix) < 3.0
            return {
                "lat": self.lat if fresh else None,
                "lng": self.lng if fresh else None,
                "speed": self.speed_kmh if fresh else 0.0,
                "heading": self.heading,
                "valid": fresh,
                "sats": self.sats,
            }


class SimGPS:
    """Drives a small circle so you can test the dashboard without hardware."""

    def __init__(self):
        self.t0 = time.time()

    def snapshot(self):
        a = (time.time() - self.t0) * 0.15
        return {
            "lat": FALLBACK_LAT + 0.0006 * math.sin(a),
            "lng": FALLBACK_LNG + 0.0006 * math.cos(a),
            "speed": 12.0 + random.uniform(-1, 1),
            "heading": (math.degrees(a) + 90) % 360,
            "valid": True,
            "sats": 9,
        }


# ------------------------------------------------------------ ultrasonics --
class Sonars:
    def __init__(self, enabled=True):
        self.sensors = {}
        if not enabled:
            return
        from gpiozero import DistanceSensor  # pip install gpiozero lgpio
        for name, (trig, echo) in SONAR_PINS.items():
            self.sensors[name] = DistanceSensor(
                echo=echo, trigger=trig, max_distance=2.5, queue_len=5
            )
        print(f"[SONAR] {', '.join(self.sensors)} ready")

    def read_cm(self):
        out = {}
        for name in ("front", "left", "right", "rear"):
            s = self.sensors.get(name)
            if s is None:
                out[name] = NO_ECHO_CM
                continue
            cm = int(s.distance * 100)
            out[name] = NO_ECHO_CM if cm >= 245 else max(cm, 2)
        return out


class EspSonars(threading.Thread):
    """4x HC-SR04 read by an ESP32 (esp32_sonar_node.ino) and sent over USB as
    lines like:  SONAR,front,left,right,rear   (cm, 250 = nothing in range)."""

    STALE_S = 1.5   # older than this -> report "no echo" instead of a frozen value

    def __init__(self, port="auto", baud=115200, exclude=()):
        super().__init__(daemon=True)
        self.port, self.baud = port, baud
        self.exclude = {os.path.realpath(p) for p in exclude if p}
        self._lock = threading.Lock()
        self._d = None
        self._ts = 0.0
        self.start()

    def _candidates(self):
        import glob
        if self.port != "auto":
            return [self.port]
        seen, out = set(), []
        for p in sorted(glob.glob("/dev/serial/by-id/*")) + sorted(
                glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")):
            rp = os.path.realpath(p)
            if rp in seen or rp in self.exclude:
                continue
            seen.add(rp)
            out.append(p)
        return out

    def run(self):
        import serial
        while True:
            for port in self._candidates():
                try:
                    with serial.Serial(port, self.baud, timeout=1) as ser:
                        ok_until = time.time() + 4      # ESP32 reboots when the port opens
                        announced = False
                        while True:
                            ln = ser.readline().decode("ascii", errors="ignore").strip()
                            if ln.startswith("SONAR,"):
                                try:
                                    f, l, r, b = (int(x) for x in ln.split(",")[1:5])
                                except ValueError:
                                    continue
                                if not announced:
                                    print(f"[SONAR] ESP32 on {port}: front, left, right, rear")
                                    announced = True
                                with self._lock:
                                    self._d = {"front": f, "left": l, "right": r, "rear": b}
                                    self._ts = time.time()
                                ok_until = time.time() + 4
                            elif time.time() > ok_until:
                                break       # not the ESP32 (or it stopped talking)
                except Exception:
                    pass
            print("[SONAR] no ESP32 sonar data yet (check USB cable / --esp-port) - retrying in 3 s")
            time.sleep(3)

    def read_cm(self):
        with self._lock:
            if self._d is None or time.time() - self._ts > self.STALE_S:
                return {k: NO_ECHO_CM for k in ("front", "left", "right", "rear")}
            return dict(self._d)


class SimSonars:
    def __init__(self):
        self.t0 = time.time()

    def read_cm(self):
        t = time.time() - self.t0
        # An obstacle slowly approaches then clears every ~40 s
        front = 250 - int(230 * max(0.0, math.sin(t / 40 * 2 * math.pi)))
        return {"front": front, "left": 140, "right": 155}


# ------------------------------------------------------------------ camera --
def camera_loop(server, vehicle, index, stop):
    import cv2
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"[CAM] cannot open camera {index}")
        return
    print(f"[CAM] streaming frames to {server}/api/camera/upload")
    while not stop.is_set():
        ok, frame = cap.read()
        if ok:
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok2:
                try:
                    requests.post(
                        f"{server}/api/camera/upload",
                        files={"image": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                        data={"vehicle_id": vehicle},
                        timeout=6,
                    )
                except requests.RequestException as e:
                    print(f"[CAM] upload failed: {e}")
        stop.wait(CAMERA_INTERVAL_S)
    cap.release()


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description="Raspberry Pi telemetry node")
    ap.add_argument("--server", default=DEFAULT_SERVER, help="Dashboard base URL")
    ap.add_argument("--vehicle", default=DEFAULT_VEHICLE)
    ap.add_argument("--api-key", default=DEFAULT_API_KEY,
                    help="shared secret (server env HARDWARE_API_KEY)")
    ap.add_argument("--sim", action="store_true", help="fake GPS + sonar, no hardware")
    ap.add_argument("--no-gps", action="store_true", help="use fallback coordinates")
    ap.add_argument("--no-sonar", action="store_true", help="skip ultrasonic sensors")
    ap.add_argument("--sonar-source", choices=["esp32", "gpio"],
                    default=os.environ.get("SONAR_SOURCE", "gpio"),
                    help="gpio = 4 sonars on Pi pins (default), esp32 = 4 sonars on an ESP32 over USB")
    ap.add_argument("--esp-port", default=os.environ.get("ESP_PORT", "auto"),
                    help="ESP32 serial port, e.g. /dev/serial/by-id/usb-... (auto = find it)")
    ap.add_argument("--lidar", choices=["none", "tfluna", "rplidar"], default="none")
    ap.add_argument("--lidar-port", default=os.environ.get("LIDAR_PORT", "/dev/ttyUSB0"))
    ap.add_argument("--lidar-baud", type=int, default=115200)
    ap.add_argument("--lidar-offset", type=float, default=0.0,
                    help="RPLIDAR: degrees to rotate so 0 = straight ahead")
    ap.add_argument("--imu", action="store_true", help="enable the IMU on I2C bus 1")
    ap.add_argument("--imu-type", choices=["bno055", "mpu6050"],
                    default=os.environ.get("IMU_TYPE", "bno055"))
    ap.add_argument("--node", default=os.environ.get("HAUL_NODE", "PI"),
                    help="node name sent to the server (PI = primary)")
    ap.add_argument("--camera", action="store_true", help="stream camera frames")
    ap.add_argument("--camera-index", type=int, default=0)
    args = ap.parse_args()
    server = args.server.rstrip("/")

    gps = SimGPS() if args.sim else None
    if gps is None and not args.no_gps:
        gps = GPSReader(GPS_PORT, GPS_BAUD)
        gps.start()
    if args.sim:
        sonar = SimSonars()
    elif args.no_sonar:
        sonar = Sonars(enabled=False)
    elif args.sonar_source == "esp32":
        excl = [args.lidar_port] if args.lidar != "none" else []
        sonar = EspSonars(args.esp_port, exclude=excl)
    else:
        sonar = Sonars(enabled=True)

    # ---- optional LiDAR / IMU (a failure never stops telemetry) ----------------
    lidar = imu = None
    try:
        if args.sim:
            lidar, imu = SimLidar(), SimIMU()
        else:
            if args.lidar == "tfluna":
                lidar = TFLuna(args.lidar_port, args.lidar_baud)
            elif args.lidar == "rplidar":
                lidar = RPLidar(args.lidar_port, args.lidar_offset)
            if lidar:
                lidar.start()
            if args.imu:
                imu = make_imu(args.imu_type)
                imu.start()
    except Exception as e:
        print(f"[WARN] extra sensor init failed: {e} (continuing without)")

    stop = threading.Event()
    if args.camera:
        threading.Thread(
            target=camera_loop,
            args=(server, args.vehicle, args.camera_index, stop),
            daemon=True,
        ).start()

    session = requests.Session()
    if args.api_key:
        session.headers["X-API-Key"] = args.api_key
    sent = failed = 0
    print(f"[NODE] {args.vehicle} -> {server}/update every {SEND_INTERVAL_S}s")
    try:
        while True:
            t0 = time.time()
            d = sonar.read_cm()
            g = gps.snapshot() if gps else {"lat": None, "lng": None, "speed": 0.0,
                                            "heading": None, "valid": False, "sats": 0}
            payload = {
                "vehicle_id": args.vehicle,
                "dist_front": d["front"],
                "dist_left": d["left"],
                "dist_right": d["right"],
                "lat": g["lat"] if g["lat"] is not None else FALLBACK_LAT,
                "lng": g["lng"] if g["lng"] is not None else FALLBACK_LNG,
                "speed": round(g["speed"], 1),
                "gps_valid": g["valid"],
            }
            if "rear" in d:
                payload["dist_rear"] = d["rear"]
            if g["heading"] is not None:
                payload["heading"] = round(g["heading"], 1)
            payload["node"] = args.node
            if lidar:
                payload["lidar"] = lidar.snapshot()
            if imu:
                payload["imu"] = imu.snapshot()

            try:
                r = session.post(f"{server}/update", json=payload, timeout=3)
                r.raise_for_status()
                sent += 1
                if sent % 5 == 1:
                    j = r.json()
                    lz = payload.get("lidar", {}).get("front_cm")
                    iu = payload.get("imu", {})
                    print(f"[OK] #{sent} sonar F/L/R/B={d['front']}/{d['left']}/{d['right']}/{d.get('rear', '-')}cm lidar={lz}cm "
                          f"imu={iu.get('pitch')}/{iu.get('roll')}deg {iu.get('accel_g')}g gps={'FIX' if g['valid'] else 'none'}"
                          f"/{g['sats']}sat spd={payload['speed']} zone={j.get('zone')} "
                          f"risk={j.get('risk_score')}")
            except requests.RequestException as e:
                failed += 1
                code = getattr(getattr(e, "response", None), "status_code", None)
                hint = "  <- wrong/missing API key (--api-key)" if code == 401 else ""
                print(f"[FAIL] #{failed} {e}{hint}")

            time.sleep(max(0.0, SEND_INTERVAL_S - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\n[NODE] stopped")
        stop.set()


if __name__ == "__main__":
    main()
