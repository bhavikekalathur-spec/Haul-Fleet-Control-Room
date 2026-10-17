#!/usr/bin/env python3
"""
Wiring checks - test ONE sensor at a time while you build.

  python3 hw_check.py imu
  python3 hw_check.py gps
  python3 hw_check.py sonar                       (4 sonars on Pi pins)
  python3 hw_check.py sonar --sonar-source esp32  (4 sonars on an ESP32 over USB)
  python3 hw_check.py lidar --type tfluna --port /dev/ttyUSB0
  python3 hw_check.py camera
"""
import argparse
import sys
import time


def check_imu(kind):
    from pi_sensors import make_imu
    m = make_imu(kind)
    m.start()
    print("IMU: keep still 2 s (calibrating), then TILT the board left/right/forward.")
    print("PASS = pitch/roll change when you tilt, accel_g ~ 1.00 at rest.\n")
    time.sleep(3.5)
    for _ in range(20):
        d = m.snapshot()
        if not d.get("ok"):
            print("  no data yet - see 'IMU' section of the guide (i2cdetect -y 1)")
        else:
            print(f"  pitch={d['pitch']:6.1f}  roll={d['roll']:6.1f}  g={d['accel_g']:.2f}  temp={d['temp_c']}C")
        time.sleep(0.5)


def check_gps():
    import serial
    from pi_node import GPS_BAUD, GPS_PORT, GPSReader
    print(f"GPS: raw serial from {GPS_PORT} @ {GPS_BAUD} for 5 s ...")
    lines = 0
    try:
        with serial.Serial(GPS_PORT, GPS_BAUD, timeout=1) as ser:
            end = time.time() + 5
            while time.time() < end:
                ln = ser.readline().decode("ascii", errors="ignore").strip()
                if ln:
                    lines += 1
                    if lines <= 4:
                        print("  ", ln)
    except Exception as e:
        print("FAIL: cannot open serial port:", e)
        return
    if lines == 0:
        print("FAIL: no data. Check TX/RX crossover, GPS power LED, serial enabled in raspi-config.")
        return
    print(f"PASS: wiring OK ({lines} NMEA lines). Now waiting for a satellite fix (needs sky view)...\n")
    g = GPSReader(GPS_PORT, GPS_BAUD)
    g.start()
    for i in range(180):
        s = g.snapshot()
        print(f"  [{i:3d}s] fix={'YES' if s['valid'] else 'no '}  sats={s['sats']}  lat={s['lat']}  lng={s['lng']}")
        if s["valid"]:
            print("PASS: GPS has a fix.")
            return
        time.sleep(1)
    print("No fix after 3 min - move the antenna outdoors/near a window.")


def check_sonar(source, esp_port):
    from pi_node import EspSonars, Sonars
    s = EspSonars(esp_port) if source == "esp32" else Sonars(enabled=True)
    if source == "esp32":
        time.sleep(4)   # ESP32 restarts when the port opens
    print("SONAR: hold your hand ~20 cm in front of ONE sensor at a time.")
    print("PASS = only that sensor's number drops. 250 = nothing in range.\n")
    for _ in range(30):
        d = s.read_cm()
        print(f"  front={d['front']:4d}  left={d['left']:4d}  right={d['right']:4d}"
              f"  rear={d.get('rear', '-')!s:>4}  cm")
        time.sleep(0.4)


def check_lidar(kind, port):
    from pi_sensors import RPLidar, TFLuna
    l = TFLuna(port) if kind == "tfluna" else RPLidar(port)
    l.start()
    print(f"LIDAR ({kind} on {port}): point it at a wall, then move your hand in front.\n")
    for _ in range(25):
        d = l.snapshot()
        print("  ", {k: d.get(k) for k in ("ok", "front_cm", "left_cm", "right_cm", "strength")})
        time.sleep(0.4)


def check_camera(index):
    import cv2
    cap = cv2.VideoCapture(index)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    cap.release()
    if not ok:
        print("FAIL: camera not readable at index", index)
        return
    cv2.imwrite("camera_test.jpg", frame)
    print(f"PASS: captured {frame.shape[1]}x{frame.shape[0]} -> camera_test.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sensor", choices=["imu", "gps", "sonar", "lidar", "camera"])
    ap.add_argument("--type", choices=["tfluna", "rplidar"], default="tfluna")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--sonar-source", choices=["esp32", "gpio"], default="gpio")
    ap.add_argument("--esp-port", default="auto")
    ap.add_argument("--imu-type", choices=["bno055", "mpu6050"], default="bno055")
    ap.add_argument("--camera-index", type=int, default=0)
    a = ap.parse_args()
    try:
        {"imu": lambda: check_imu(a.imu_type), "gps": check_gps, "sonar": lambda: check_sonar(a.sonar_source, a.esp_port),
         "lidar": lambda: check_lidar(a.type, a.port),
         "camera": lambda: check_camera(a.camera_index)}[a.sensor]()
    except KeyboardInterrupt:
        sys.exit(0)
