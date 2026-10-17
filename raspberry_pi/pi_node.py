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
    """HC-SR04 sensors read directly with lgpio (works on the Pi 5).

    A background thread fires the sensors one after another and keeps the latest
    reading of each. A sensor that is unplugged or miswired simply reports 250
    ("nothing in range") instead of blocking the others.
    """
    TIMEOUT_S = 0.03      # ~5 m round trip - longer than any real echo
    GAP_S = 0.015         # pause between sensors so they don't hear each other

    def __init__(self, enabled=True):
        self._latest = {n: NO_ECHO_CM for n in ("front", "left", "right", "rear")}
        self._hist = {n: [] for n in SONAR_PINS}
        self._lock = threading.Lock()
        self._noecho = {n: 0 for n in SONAR_PINS}
        if not enabled:
            return
        import lgpio  # sudo apt install python3-lgpio
        self.lg = lgpio
        self.h = self._open_chip()
        for name, (trig, echo) in SONAR_PINS.items():
            lgpio.gpio_claim_output(self.h, trig, 0)
            lgpio.gpio_claim_input(self.h, echo)
        print(f"[SONAR] {', '.join(SONAR_PINS)} ready (lgpio)")
        threading.Thread(target=self._loop, daemon=True).start()

    def _open_chip(self):
        lg = self.lg
        for chip in (0, 4):           # Pi 5: RP1 is gpiochip0 on new kernels, 4 on old
            try:
                h = lg.gpiochip_open(chip)
            except Exception:
                continue
            try:
                info = lg.gpio_get_chip_info(h)
                label = str(info[-1]).lower()
                if "rp1" in label or chip == 4:
                    return h
            except Exception:
                return h
            lg.gpiochip_close(h)
        return lg.gpiochip_open(0)

    def _ping(self, trig, echo):
        lg, h, now = self.lg, self.h, time.perf_counter
        lg.gpio_write(h, trig, 1)
        time.sleep(0.00001)
        lg.gpio_write(h, trig, 0)
        t_end = now() + self.TIMEOUT_S
        while lg.gpio_read(h, echo) == 0:          # wait for the echo pulse to start
            if now() > t_end:
                return None
        start = now()
        t_end = start + self.TIMEOUT_S
        while lg.gpio_read(h, echo) == 1:          # measure how long it stays high
            if now() > t_end:
                return None
        return (now() - start) * 34300.0 / 2.0     # cm

    def _loop(self):
        while True:
            for name, (trig, echo) in SONAR_PINS.items():
                try:
                    cm = self._ping(trig, echo)
                except Exception:
                    cm = None
                with self._lock:
                    if cm is None or cm >= 245:
                        self._noecho[name] += 1
                        if self._noecho[name] >= 3:       # 3 misses in a row -> nothing there
                            self._hist[name].clear()
                            self._latest[name] = NO_ECHO_CM
                    else:
                        self._noecho[name] = 0
                        hist = self._hist[name]
                        hist.append(max(int(cm), 2))
                        del hist[:-3]
                        self._latest[name] = sorted(hist)[len(hist) // 2]   # median of 3
                time.sleep(self.GAP_S)

    def read_cm(self):
        with self._lock:
            return dict(self._latest)


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
    lidar = imu = ÔıÀşKw{¬ub	êè·[	HâC–ÂY¨×.´7]•æ»ŞÃ‡$TxÕAp…Ë´"fká:ùáZhÇô-ÍX~–Ÿ?=¯µ†ı? ñX *J
CŠÜ ¤@âÓå}Ÿh·ú~'e˜Åu[²ĞÄ	‚'§\š¸-çÈ†8m²+?ßó€@y;]~lLøáæ÷ªF¢²Ìà£ÙnE<SÕ°Ğƒv/Z8$OOÅôÓme¡Ëòï™Ìúm<SpÖn•1}^
»Æ4®±©«$
ûH„_ã×ä±é“v‘(Fõ-ÉTj§rT²£Ë8¾ñ(˜ëé·P»‰'œLĞÌ7|OFŒ.qSû‹îÙ@ÙyÁ’ó®“¬æÙ63lbd¿ï4 eWpøòúC bZ¸ıĞ­êÃÅÚO\OašÊ-„q;‘Ú¤½ˆG»èØ1SÂæ$$gy®Ÿ5ş#‰¥a©2÷_¼ERÀ6ş #–«¼J‚	gïƒÜ½ÛÂwDHÌßz]»zå× (wU0™ıîb§è™Ç8b"•´átô+kÏ¦~¸ŠôQiTVIöŞíÈËš¾WA!ì‘ ŠÜŠ10ŸF+¯ıyLµ¨aÓ?3í*d–r ßÄ|»òäã1I¶İ$>B®…A¡ĞY>„k ‘®Ÿ§üYZÌÆĞ·øÅÀ;gO{i„ï.qe•#%pâ†ºúYÔ6yµ©ó´ç—æÃBlxëÌ-tF–ZJSáÿÕÙZ­˜ñŒJÏ²=j-†8Î-Ó Q§hro²×ğ8¤…wµ²è?	GÖ*~âI Å°ª ¨5Š+úŸ\_”çRTL=ÍŞ÷oóË€ÊlA;˜án€kÕÒ.~¢w±ô%=wOD¢ıôœèƒ³b8áàÂmùHÖ«A]¡Ş×\–±l‡
>¹	}›&ƒÆÉÅ#“ÑZ©(ƒd¼_&û6ätV[®ÈÉÑš÷Äb?hNÙm‹îŒud7'*q@‚µûûĞsºPàì0ë²Ò%œM¯'E®r]1…É¾ªø™S™ƒèË½Ÿn"Œ$<Á,q$^¦Êv`³½Óÿ99x­±¦òã˜çaFmÚäÍ{G4^ERCûÚØø©]Uî9ÂßUküÊ"Îc@™œ€àÌ„jwÖ! s¾õ»!x´ç>«CÙ+ÜæF¡g´¥2¬:‹‰şş[›E	—hˆ¨:¶ßŸ)‚dÍÊ"1ÕU¨Š‡k½ı"ô7Fp2ví@­üV“é!·cšåïÃÏıpIt¯N\
ÑÖ¸XßÍ•x@G.*~RYáØâİx·Åò—£ç{2<mÀ`ú,kPe¦Ğ[³§§Ä9õ'™Ií¸ò¿†…Œ>H•h)
…Ó{4¤Ş«ŒHºJ”×0ñÆé%†cv¯=1°î–/d-SqZöûá¤m[´¼òÑîÌÄxKSNÃÔÃÕ‰KInÛu\~êÖ¢,	v÷4–üLf¨é;Ã7c€‘ñ"+¡n¨óP½çVÏ7\,—	³²NÅë‚~¹(Ø‹“Š?ä/ _}C´
eÜ>±7â+Æ’}¡}À»àÖğ$jm¢…õ¡UfUôMùßOÇÊ8ºX@ƒèÛübúDvÙ¨zc®åé_I@G™Ãû§ÖŒ°8\7âşœÒa—'€_‚P%Àëw#¨¾o¼"=‚7r˜½ÉÊ‘>Ÿ«´$ù0¡S\¯+èL‹½ÓDíqárÊ³´ßaƒ+UÚÑ@­tŠÊ&ijC>öàølÈõÉW4›Ğò"Ox™YqmîüîçU®G ¶’Í»ä¬ØÌA3RwõÜ#j™gqTŒÔË«†(¦ş·,EÌ‰92i³‰;š0ßZºd;¯­¤%¨ÿG…ıçØFµæ1yŠB¿Ñ!r [š Ní…ÄV×bdÏıî¸ÃûÏ8\qtjRjYJÍàâóõ•½l.ïß{÷UÀ¥.@· ±ÊRw‹§bè0Ök•IĞÇª>…ß5´>¡I(”+òz¨¤‡"I¹7hì&½Ó¾Å†¦Z—=ôd‚JQûñ=IZ3¢ÃáaœÖ–Ä\-–àüxvÃÜAc´yŞé+Yd»Æî¢éøûÕLgqn„Ñ;[€T%N÷ñºÄL£|…ç“ã\Áİ+dB6¡0Å.>º~| ¯	ÙŸ%²‹¦äêŒ,_¸²9(-³“OÎÆWQ™}oŒ
 ğ±ò6GÂ©Í¡—ØÖ5Rmgëò8txƒ-JmôˆÕçOÚL7­ÓŒÿí‹ûZr@‘¡+®B0>Ùç¯´bµiõÉ…ör×ÈjrWà¾ ´@ë8+ÊPjß'ÏŠUœÖÊü™\q®§I8Ã½YÛcèADéSzü$¶åvŸä#74Ô¼½†‚¨ø#"Cqş‚ia­;_ÚÀ—aÌGw\ØÌæ…»«y ùš¯Uá%î³;0™¤º"Dbû‰tıq2&ÃdEƒ\îşÑ¿N«É Ä›Ñg>[ÜlØŠ’R1_¬GFú3Íı¨Ğm¨°OçâqòdGîxß(9t%·³Ïw‰¦¸Ò,€õŒV˜jíÊTšoË™!=Å‚Z·OdÀêû—{¸7. ‡½•ò¹¨âW&"Yàcò5éIgAü>ÂŞv…=ÖĞˆ¢\kÚœIÃ§-@ñLÊJ.ß=»±U†éRõÓñÍh£ójlà¤Tª¡d5+´6>Ã“”´xÁw-Ùè–‹Ö‹á.IZ|ÎŒLìQ÷¾o€ğ™;I9nQŒ³Õ²¦¢¦-,ôëm²Ätç	–Jò~3ÕxÅa6Øy©R++—G\ÍçÜ„N±½Ä
ï%Ñ}Jº[ÆYû“ Æq(Rød_÷gîä¥¡¯O•>%ôÇ 0ƒbŸº80|m(ãÖzİ…¡ßBÁ³ûŸ^9@Í`,7hÿ¦Œ:Ù"ƒŒbp½™Õ"®‡äR·{Ø¾åEÍÉ@ÚGr9ç'
¦mœX˜xëıòP¯‹qÎû½œ*îÊ9µdqkVÄ$sÉNŸ!÷[è„hÑSÖ®øæ1C´4…Cs%Ú[ì¯a	Òº¬M0­şÊ½»nU7 <k"w™ô¨ÌË™Óˆ‚"¶—U$)îvï\EFpÖşNÃ‰±ÑI2ã2égû­cÜ©“v«ü^€F?õım!àŠÈ¾j1š]Êd‚Â@ßĞüU¨ucß'¥¸:Età›‹Ï²x+šªç¡!øÙ´V]F>íH%À/×¯{}éºØv0·Š•â’
YÀ4.e«…•7mÄ>òN…UÌ[òğSÑI¢°qº/û§èîĞMdkç=~m†,£xñ‰<òJÛßRÃ@Ø¤‘~ÍÓ4áGhf'ÃV¸Œx†™¡´óú³áëe9Z¹[,-Ä¦–NC+€ŞÜ¡;Œâ´L)}>÷{¢c1?n” ‹ÕÆfÊ~öù@Å¤ÇU²Xß	S»\K$õçà¼…j+î	é}–cÆİ¨v±x7ü
*Ô\_2KÖä`uÃ“ÅêI(—,ƒ§³‚8õ—OPôñ½¡n7HP"míÏ¨Ö¿§Å…]8O>ZIª™Ğòøzp§àåú²ÛïkDeĞE‚${u®À'#»·‚¼1Ğ_‘YÈÀâšş•?a„.míxrÃ^L˜´ûÓ©0²Z±¯8áã‘-–F§-Èæ†$Wl=viyJÓöóñS¤™ŠÙË´Ìhn+FÓ<¬ËÅ¬3A~şT	[’Ş²	q~çîô\CĞá+æOk´‰*;„ €Ö(µ÷s·?L!TŸ9Ë¢kõ ÕÎjŠnœæ	8ıyƒƒ¯G–ô
ØOX;¼@¤6¡š#Ö·©måÃèÆÕ\b}‡bw
"ıı±p]äh×_:¿Â(Ÿ H“Í.S¾í±Ù¿ÌrFÉHóæœPll'RyP§Íóëõ²@Å”8û—ª-Œ25§7`ÖbxIÙ*w˜®èİd‘C¼ûøîÅKZ„4¹$a&®ÚS»­ö‡1¤Ap½”ŞúÆàïqceÊ1œÅŸ)O${=ZSŞ¢ĞèŒ%]şº×EN„Â2ëH‰¹øèÜ¡gbgóYwVÆı« ¼ 4Ÿ6»f¡#ÌÃ>©w‘İ	"‰Bƒ™Û|–î~ãU,oŸMğ¸Î ÏºQŠtè²*!ğ- š¢µíŒ?VUJ~ıeÕôF7ëá1’tkŠÀ—ËßØAdŠ6T/©Ş¨}*bI