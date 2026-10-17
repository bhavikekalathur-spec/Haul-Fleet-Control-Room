"""
LiDAR + IMU drivers for pi_node.py
==================================
LiDAR   TFLuna    - Benewake TF-Luna / TFmini / TFmini-S (UART, single-point, cm)
        RPLidar   - Slamtec RPLIDAR A1/A2/C1 (USB, 360 deg scanner)  [pip install rplidar-roboticia]
IMU     BNO055    - I2C 0x28/0x29 (default, --imu-type bno055)
        MPU6050   - I2C 0x68 (also works with MPU6500/9250 accel+gyro)  [pip install smbus2]

Every driver runs in its own thread and exposes snapshot() -> dict, so a slow or
failed sensor can never block the telemetry loop.

IMU axis convention (mount the board like this, or edit _remap()):
    X = forward, Y = left, Z = up.   Flat and stationary -> accel = (0, 0, +1 g)
"""
import math
import random
import threading
import time


class _Worker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._data = {"ok": False}

    def _set(self, d):
        with self._lock:
            self._data = d

    def snapshot(self):
        with self._lock:
            return dict(self._data)


# ------------------------------------------------------------------ LiDAR --
class TFLuna(_Worker):
    """9-byte frames: 59 59 DistL DistH AmpL AmpH TempL TempH Checksum."""

    def __init__(self, port="/dev/ttyUSB0", baud=115200):
        super().__init__()
        self.port, self.baud = port, baud

    def run(self):
        import serial
        while True:
            try:
                with serial.Serial(self.port, self.baud, timeout=1) as ser:
                    print(f"[LIDAR] TF-Luna on {self.port} @ {self.baud}")
                    buf = bytearray()
                    while True:
                        buf += ser.read(ser.in_waiting or 9)
                        while len(buf) >= 9:
                            if buf[0] != 0x59 or buf[1] != 0x59:
                                del buf[0]
                                continue
                            frame = bytes(buf[:9])
                            del buf[:9]
                            if (sum(frame[:8]) & 0xFF) != frame[8]:
                                continue
                            dist = frame[2] | (frame[3] << 8)
                            amp = frame[4] | (frame[5] << 8)
                            # amp < 100 or 0xFFFF = unreliable / saturated reading
                            good = 100 <= amp < 0xFFFF and 0 < dist <= 800
                            self._set({"sensor": "TF-Luna", "ok": good,
                                       "front_cm": dist if good else None,
                                       "strength": amp, "ts": time.time()})
            except Exception as e:
                self._set({"sensor": "TF-Luna", "ok": False})
                print(f"[LIDAR] {e} - retrying in 3 s")
                time.sleep(3)


class RPLidar(_Worker):
    """Reduces each 360 deg scan to nearest obstacle in front / left / right sectors.

    RPLIDAR angles are clockwise from the connector cable direction. Set
    `front_offset_deg` so that 0 = straight ahead for how you mounted it.
    """

    def __init__(self, port="/dev/ttyUSB0", front_offset_deg=0.0, half_fov=20.0, min_cm=20.0):
        super().__init__()
        self.port, self.off, self.half = port, front_offset_deg, half_fov
        # Ignore anything closer than min_cm: the scanner's blind zone and the
        # truck's own body/mast would otherwise cause false STOPs.
        self.min_cm = min_cm

    def _sector_min(self, scan, centre):
        best = None
        for _q, ang, mm in scan:
            if mm <= 0 or mm / 10.0 < self.min_cm:
                continue
            d = abs(((ang - self.off - centre + 180) % 360) - 180)
            if d <= self.half:
                cm = mm / 10.0
                best = cm if best is None else min(best, cm)
        return None if best is None else int(best)

    def run(self):
        from rplidar import RPLidar as _R
        while True:
            lidar = None
            try:
                lidar = _R(self.port)
                print(f"[LIDAR] RPLIDAR on {self.port}: {lidar.get_info().get('model')}")
                for scan in lidar.iter_scans(max_buf_meas=5000):
                    self._set({"sensor": "RPLIDAR", "ok": True,
                               "front_cm": self._sector_min(scan, 0),
                               "right_cm": self._sector_min(scan, 90),
                               "left_cm": self._sector_min(scan, 270),
                               "points": len(scan), "ts": time.time()})
            except Exception as e:
                self._set({"sensor": "RPLIDAR", "ok": False})
                print(f"[LIDAR] {e} - retrying in 3 s")
                time.sleep(3)
            finally:
                try:
                    if lidar:
                        lidar.stop()
                        lidar.stop_motor()
                        lidar.disconnect()
                except Exception:
                    pass


class SimLidar:
    def __init__(self):
        self.t0 = time.time()

    def snapshot(self):
        t = time.time() - self.t0
        d = 600 - int(560 * max(0.0, math.sin(t / 40 * 2 * math.pi)))
        return {"sensor": "SIM-LiDAR", "ok": True, "front_cm": max(d, 30),
                "strength": 900, "ts": time.time()}


# -------------------------------------------------------------------- IMU --
class MPU6050(_Worker):
    ADDR = 0x68
    ACC_LSB = 16384.0   # +-2 g
    GYR_LSB = 131.0     # +-250 deg/s

    def __init__(self, bus=1, addr=0x68, rate_hz=50):
        super().__init__()
        self.bus_id, self.addr, self.dt = bus, addr, 1.0 / rate_hz
        self._peak_g = 1.0
        self._peak_lock = threading.Lock()

    @staticmethod
    def _s16(hi, lo):
        v = (hi << 8) | lo
        return v - 65536 if v & 0x8000 else v

    def _read(self, bus):
        r = bus.read_i2c_block_data(self.addr, 0x3B, 14)
        ax, ay, az = (self._s16(r[i], r[i + 1]) / self.ACC_LSB for i in (0, 2, 4))
        temp = self._s16(r[6], r[7]) / 340.0 + 36.53
        gx, gy, gz = (self._s16(r[i], r[i + 1]) / self.GYR_LSB for i in (8, 10, 12))
        return ax, ay, az, gx, gy, gz, temp

    def run(self):
        from smbus2 import SMBus
        while True:
            try:
                with SMBus(self.bus_id) as bus:
                    bus.write_byte_data(self.addr, 0x6B, 0x00)  # wake from sleep
                    time.sleep(0.1)
                    # gyro bias: average 100 samples (keep the truck still at boot)
                    gb = [0.0, 0.0, 0.0]
                    for _ in range(100):
                        s = self._read(bus)
                        gb = [gb[i] + s[3 + i] / 100 for i in range(3)]
                        time.sleep(0.005)
                    print(f"[IMU] MPU6050 ready @ 0x{self.addr:02X}")
                    pitch = roll = 0.0
                    last = time.time()
                    while True:
                        ax, ay, az, gx, gy, gz, temp = self._read(bus)
                        gx, gy, gz = gx - gb[0], gy - gb[1], gz - gb[2]
                        now = time.time()
                        dt, last = now - last, now
                        # complementary filter: gyro (smooth) + accel (drift-free)
                        acc_pitch = math.degrees(math.atan2(-ax, math.hypot(ay, az)))
                        acc_roll = math.degrees(math.atan2(ay, az))
                        pitch = 0.96 * (pitch + gy * dt) + 0.04 * acc_pitch
                        roll = 0.96 * (roll + gx * dt) + 0.04 * acc_roll
                        g_total = math.sqrt(ax * ax + ay * ay + az * az)
                        with self._peak_lock:
                            self._peak_g = max(self._peak_g, g_total)
                        self._set({"ok": True, "ax": round(ax, 3), "ay": round(ay, 3),
                                   "az": round(az, 3), "gx": round(gx, 1),
                                   "gy": round(gy, 1), "gz": round(gz, 1),
                                   "pitch": round(pitch, 1), "roll": round(roll, 1),
                                   "temp_c": round(temp, 1), "ts": now})
                        time.sleep(self.dt)
            except Exception as e:
                self._set({"ok": False})
                print(f"[IMU] {e} - retrying in 3 s")
                time.sleep(3)

    def snapshot(self):
        d = super().snapshot()
        # Report the PEAK g since the last snapshot so a brief shock between
        # 1 s telemetry posts is never missed.
        with self._peak_lock:
            d["accel_g"] = round(self._peak_g, 2)
            self._peak_g = 1.0
        return d


class BNO055(_Worker):
    """Bosch BNO055 9-axis IMU (Adafruit / 7SEMI / CJMCU boards) over I2C.

    Default address 0x28 (ADD/ADR pin low) or 0x29 (ADD high) - auto-detected.
    Runs in IMUPLUS mode (accel + gyro fusion, magnetometer off): pitch/roll are
    stable and nothing needs a figure-8 calibration, and nearby motors/steel
    cannot disturb it. Same snapshot() keys as MPU6050, so the server needs no
    changes.
    """
    CHIP_ID = 0xA0
    REG_CHIP_ID, REG_PAGE, REG_UNIT_SEL = 0x00, 0x07, 0x3B
    REG_OPR_MODE, REG_PWR_MODE, REG_SYS_TRIGGER = 0x3D, 0x3E, 0x3F
    MODE_CONFIG, MODE_IMUPLUS = 0x00, 0x08

    def __init__(self, bus=1, addr=None, rate_hz=50):
        super().__init__()
        self.bus_id, self.addr, self.dt = bus, addr, 1.0 / rate_hz
        self._peak_g = 1.0
        self._peak_lock = threading.Lock()

    @staticmethod
    def _s16(lo, hi):
        v = (hi << 8) | lo
        return v - 65536 if v & 0x8000 else v

    def _find(self, bus):
        for a in ([self.addr] if self.addr else [0x28, 0x29]):
            try:
                if bus.read_byte_data(a, self.REG_CHIP_ID) == self.CHIP_ID:
                    return a
            except OSError:
                pass
        raise OSError("BNO055 not found at 0x28/0x29 - check wiring (sudo i2cdetect -y 1)")

    def _setup(self, bus, a):
        bus.write_byte_data(a, self.REG_OPR_MODE, self.MODE_CONFIG)
        time.sleep(0.03)
        bus.write_byte_data(a, self.REG_SYS_TRIGGER, 0x20)   # reset
        time.sleep(0.7)
        for _ in range(20):                                  # wait for it to boot
            try:
                if bus.read_byte_data(a, self.REG_CHIP_ID) == self.CHIP_ID:
                    break
            except OSError:
                pass
            time.sleep(0.05)
        bus.write_byte_data(a, self.REG_PWR_MODE, 0x00)      # normal power
        bus.write_byte_data(a, self.REG_PAGE, 0x00)
        bus.write_byte_data(a, self.REG_UNIT_SEL, 0x00)      # m/s2, dps, degrees, C
        bus.write_byte_data(a, self.REG_SYS_TRIGGER, 0x00)
        time.sleep(0.01)
        bus.write_byte_data(a, self.REG_OPR_MODE, self.MODE_IMUPLUS)
        time.sleep(0.03)

    def _read(self, bus, a):
        r = bus.read_i2c_block_data(a, 0x08, 24)   # accel, mag, gyro, euler
        ax, ay, az = (self._s16(r[i], r[i + 1]) / 100.0 / 9.80665 for i in (0, 2, 4))
        gx, gy, gz = (self._s16(r[i], r[i + 1]) / 16.0 for i in (12, 14, 16))
        heading, roll, pitch = (self._s16(r[i], r[i + 1]) / 16.0 for i in (18, 20, 22))
        t = bus.read_i2c_block_data(a, 0x34, 2)    # temp, calib status
        temp = t[0] - 256 if t[0] > 127 else t[0]
        return ax, ay, az, gx, gy, gz, heading, roll, pitch, temp, t[1]

    def run(self):
        from smbus2 import SMBus
        while True:
            try:
                with SMBus(self.bus_id) as bus:
                    a = self._find(bus)
                    self._setup(bus, a)
                    print(f"[IMU] BNO055 ready @ 0x{a:02X} (IMUPLUS mode)")
                    while True:
                        ax, ay, az, gx, gy, gz, hd, roll, pitch, temp, cal = self._read(bus, a)
                        now = time.time()
                        g_total = math.sqrt(ax * ax + ay * ay + az * az)
                        with self._peak_lock:
                            self._peak_g = max(self._peak_g, g_total)
                        self._set({"ok": True, "sensor": "BNO055",
                                   "ax": round(ax, 3), "ay": round(ay, 3), "az": round(az, 3),
                                   "gx": round(gx, 1), "gy": round(gy, 1), "gz": round(gz, 1),
                                   "pitch": round(pitch, 1), "roll": round(roll, 1),
                                   "heading": round(hd, 1), "temp_c": temp,
                                   "calib": {"sys": cal >> 6 & 3, "gyro": cal >> 4 & 3,
                                             "accel": cal >> 2 & 3},
                                   "ts": now})
                        time.sleep(self.dt)
            except Exception as e:
                self._set({"ok": False})
                print(f"[IMU] {e} - retrying in 3 s")
                time.sleep(3)

    def snapshot(self):
        d = super().snapshot()
        with self._peak_lock:
            d["accel_g"] = round(self._peak_g, 2)
            self._peak_g = 1.0
        return d


def make_imu(kind="bno055"):
    return MPU6050() if kind == "mpu6050" else BNO055()


class SimIMU:
    def __init__(self):
        self.t0 = time.time()

    def snapshot(self):
        t = time.time() - self.t0
        return {"ok": True, "ax": 0.02, "ay": 0.01, "az": 1.0, "gx": 0.0, "gy": 0.0,
                "gz": 0.0, "pitch": round(6 * math.sin(t / 7), 1),
                "roll": round(3 * math.sin(t / 5), 1),
                "accel_g": round(1.0 + abs(random.gauss(0, 0.04)), 2), "temp_c": 31.0}
