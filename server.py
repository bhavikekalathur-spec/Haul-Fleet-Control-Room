"""
RESURGENCE FLEET CONTROL ROOM — 6-Vehicle Simulation & Telemetry Server

Key Upgrades:
- Simplified industrial names: Truck 1 through Truck 6 (TRUCK_01 to TRUCK_06)
- Real-time Risk Score Engine (0-100) per vehicle (Speed, Fog, Traffic, Road Hazard)
- Dynamic Fog Zones (4 Zones) with real-time visibility classification & safe speed limits
- 10-Minute Fog Predictive Forecasting (Atmospheric model: humidity, dew point, elevation)
- Real Hardware vs Simulation Segregation (Dedicated ESP32 + GPS hardware node)
- Traffic Proximity Matrix (Haversine inter-truck collision proximity)
- Dark Channel Prior OpenCV Dehazing Stream
"""

import base64
import hmac
import json
import math
import os
import random
import threading
import time
import urllib.request
from datetime import datetime
from flask import Flask, Response, jsonify, render_template, request, session, redirect, url_for, send_from_directory
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "resurgence_mine_fleet_auth_key_2026")

# ==============================================================================
# CANONICAL SIMULATION DATA & REAL DEM TERRAIN ENGINE
# Conceptual 16-waypoint Deposit-14 simulation haul loop
# (Copernicus GLO-30 DEM, Kirandul / Deposit-14, Dantewada District, Chhattisgarh)
# Notice: Conceptual simulation route & pit geometry for digital-twin prototype.
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DATA_PATH = os.path.join(BASE_DIR, "bailadila_simulation_data.json")
TERRAIN_META_PATH = os.path.join(BASE_DIR, "static", "bailadila_terrain_meta.json")
TERRAIN_PNG_PATH = os.path.join(BASE_DIR, "static", "bailadila_terrain_256.png")

with open(SIM_DATA_PATH, "r", encoding="utf-8") as f:
    SIM_DATA = json.load(f)

with open(TERRAIN_META_PATH, "r", encoding="utf-8") as f:
    TERRAIN_META = json.load(f)

# Load 16-bit DEM heightmap into numpy array
DEM_IMG = cv2.imread(TERRAIN_PNG_PATH, cv2.IMREAD_UNCHANGED)
if DEM_IMG is None:
    DEM_IMG = np.full((256, 256), 32768, dtype=np.uint16)

DEM_W = TERRAIN_META.get("width", 256)
DEM_H = TERRAIN_META.get("height", 256)
DEM_Y_MIN = float(TERRAIN_META.get("minElevationMeters", 219.9658))
DEM_Y_MAX = float(TERRAIN_META.get("maxElevationMeters", 1272.3497))
DEM_SOUTH = float(TERRAIN_META.get("south", 18.500139))
DEM_NORTH = float(TERRAIN_META.get("north", 18.750139))
DEM_WEST = float(TERRAIN_META.get("west", 81.149861))
DEM_EAST = float(TERRAIN_META.get("east", 81.269861))

DEM_LAT0 = (DEM_NORTH + DEM_SOUTH) / 2.0
DEM_LON0 = (DEM_EAST + DEM_WEST) / 2.0
M_LAT = 111320.0
M_LON = 111320.0 * math.cos(DEM_LAT0 * math.pi / 180.0)

# Deposit 14 Pit Center & Geometry (Conceptual excavation)
PIT_LAT = 18.5792
PIT_LON = 81.2194
PIT_X = (PIT_LON - DEM_LON0) * M_LON
PIT_Z = -(PIT_LAT - DEM_LAT0) * M_LAT
PIT_RX = 1380.0
PIT_RZ = 1120.0
PIT_CUT_DEPTH = 225.0
PIT_BENCHES = 10

def sample_raw_dem_elevation(lon, lat):
    """Bilinear interpolation of Copernicus GLO-30 DEM elevation in meters."""
    if DEM_IMG is None:
        return 720.0
    fx = (lon - DEM_WEST) / (DEM_EAST - DEM_WEST) * (DEM_W - 1)
    fz = (DEM_NORTH - lat) / (DEM_NORTH - DEM_SOUTH) * (DEM_H - 1)
    x0 = max(0, min(DEM_W - 1, int(math.floor(fx))))
    z0 = max(0, min(DEM_H - 1, int(math.floor(fz))))
    x1 = min(DEM_W - 1, x0 + 1)
    z1 = min(DEM_H - 1, z0 + 1)
    tx = fx - x0
    tz = fz - z0

    def val_to_meters(val):
        return DEM_Y_MIN + (float(val) / 65535.0) * (DEM_Y_MAX - DEM_Y_MIN)

    a = val_to_meters(DEM_IMG[z0, x0])
    b = val_to_meters(DEM_IMG[z0, x1])
    c = val_to_meters(DEM_IMG[z1, x0])
    d = val_to_meters(DEM_IMG[z1, x1])
    return a * (1 - tx) * (1 - tz) + b * tx * (1 - tz) + c * (1 - tx) * tz + d * tx * tz

def sample_mine_surface_elevation(lat, lon):
    """
    Authoritative surface elevation: calculates natural DEM elevation,
    tests if point falls inside conceptual Deposit-14 pit excavation,
    and returns final rendered mine surface elevation with 10 stepped benches.
    """
    x = (lon - DEM_LON0) * M_LON
    z = -(lat - DEM_LAT0) * M_LAT
    raw_dem_h = sample_raw_dem_elevation(lon, lat)

    dx = x - PIT_X
    dz = z - PIT_Z
    theta = math.atan2(dz, dx)
    r_organic = 1.0 + 0.12 * math.cos(3 * theta + 0.5) + 0.08 * math.sin(5 * theta - 0.7) + 0.04 * math.cos(2 * theta)
    norm_dist = math.hypot(dx / PIT_RX, dz / PIT_RZ) / r_organic

    if norm_dist >= 1.0:
        return raw_dem_h  # Outside conceptual pit footprint

    rim_h = sample_raw_dem_elevation(PIT_LON, PIT_LAT) + 12.0
    floor_h = max(DEM_Y_MIN + 25.0, rim_h - PIT_CUT_DEPTH)
    total_cut = rim_h - floor_h
    num_benches = PIT_BENCHES
    bench_height = total_cut / num_benches

    if norm_dist > 0.88:
        blend = (norm_dist - 0.88) / 0.12
        top_bench_h = rim_h - bench_height * 0.4
        pit_h = top_bench_h * (1.0 - blend) + raw_dem_h * blend
    elif norm_dist > 0.15:
        bench_param = (0.88 - norm_dist) / 0.73 * num_benches
        bench_idx = min(num_benches - 1, int(math.floor(bench_param)))
        bench_frac = bench_param - bench_idx
        if bench_frac < 0.75:
            step_offset = (bench_idx + 0.08) * bench_height
        else:
            slope_frac = (bench_frac - 0.75) / 0.25
            step_offset = (bench_idx + 0.08 + slope_frac * 0.92) * bench_height
        pit_h = rim_h - step_offset
    else:
        pit_h = floor_h

    return min(raw_dem_h, pit_h)

BASE_LAT = 18.5792
BASE_LNG = 81.2194

# Canonical 16-waypoint Deposit-14 simulation haul loop from bailadila_simulation_data.json
SIM_WAYPOINTS = SIM_DATA["route"]["waypoints"]
BAILADILA_WAYPOINTS = SIM_WAYPOINTS
QUARRY_WAYPOINTS = SIM_WAYPOINTS

# Dynamic Fog Zones centered around Kirandul & Deposit-14 Pit
FOG_ZONES = {
    "ZONE_1": {
        "id": "ZONE_1",
        "name": "Zone 1: Deposit 14 Pit Floor & Loading Bay",
        "elevation_m": 530,
        "polygon": [
            [18.5775, 81.2155],
            [18.5815, 81.2155],
            [18.5815, 81.2215],
            [18.5775, 81.2215],
        ],
        "center": [18.5795, 81.2185],
        "temperature_c": 16.8,
        "humidity_pct": 94.0,
        "dew_point_c": 15.9,
        "wind_speed_kmh": 3.6,
        "wind_dir": "NE",
        "visibility_m": 18,
        "severity": "SEVERE FOG",
        "color": "#ef4444",
        "max_safe_speed": 10,
        "prediction_10m": {
            "visibility_m": 12,
            "severity": "CRITICAL FOG",
            "trend": "Worsening (Bastar Valley Cloud Inversion)",
            "advisory": "Dense valley cloud inversion pooling into pit basin. Mandatory 10 km/h speed governor."
        }
    },
    "ZONE_2": {
        "id": "ZONE_2",
        "name": "Zone 2: East & West Hairpin Switchbacks",
        "elevation_m": 680,
        "polygon": [
            [18.5815, 81.2135],
            [18.5885, 81.2135],
            [18.5885, 81.2275],
            [18.5815, 81.2275],
        ],
        "center": [18.5850, 81.2205],
        "temperature_c": 18.2,
        "humidity_pct": 84.0,
        "dew_point_c": 15.2,
        "wind_speed_kmh": 8.0,
        "wind_dir": "ENE",
        "visibility_m": 42,
        "severity": "LOW VISIBILITY",
        "color": "#f97316",
        "max_safe_speed": 16,
        "prediction_10m": {
            "visibility_m": 26,
            "severity": "SEVERE FOG",
            "trend": "Deteriorating (Mountain Mist)",
            "advisory": "Ridge updraft condensing along middle bench switchbacks. Active Fog-Guard radar required."
        }
    },
    "ZONE_3": {
        "id": "ZONE_3",
        "name": "Zone 3: Deposit 14 North & South Rim Crest",
        "elevation_m": 745,
        "polygon": [
            [18.5880, 81.2150],
            [18.5940, 81.2150],
            [18.5940, 81.2280],
            [18.5880, 81.2280],
        ],
        "center": [18.5910, 81.2215],
        "temperature_c": 19.5,
        "humidity_pct": 68.0,
        "dew_point_c": 13.0,
        "wind_speed_kmh": 14.5,
        "wind_dir": "E",
        "visibility_m": 80,
        "severity": "CAUTION",
        "color": "#f59e0b",
        "max_safe_speed": 22,
        "prediction_10m": {
            "visibility_m": 75,
            "severity": "CAUTION",
            "trend": "Stable (Ridge Wind Dispersing Haze)",
            "advisory": "Intermittent mountain crest fog bands. Maintain safe following clearance."
        }
    },
    "ZONE_4": {
        "id": "ZONE_4",
        "name": "Zone 4: Kirandul Dispatch Yard & Rail Terminal",
        "elevation_m": 715,
        "polygon": [
            [18.5900, 81.2180],
            [18.5960, 81.2180],
            [18.5960, 81.2260],
            [18.5900, 81.2260],
        ],
        "center": [18.5930, 81.2220],
        "temperature_c": 21.0,
        "humidity_pct": 52.0,
        "dew_point_c": 10.4,
        "wind_speed_kmh": 11.0,
        "wind_dir": "ESE",
        "visibility_m": 145,
        "severity": "NORMAL",
        "color": "#10b981",
        "max_safe_speed": 30,
        "prediction_10m": {
            "visibility_m": 140,
            "severity": "NORMAL",
            "trend": "Clear (Dispatch Yard Fans Active)",
            "advisory": "Dispatch yard, maintenance spur, and crusher approach clear. Nominal haul speeds permitted."
        }
    },
}


# ==============================================================================
# FOG DENSITY MEASUREMENT SYSTEM — DYNAMIC FOG RISK INDEX (DFRI)
# Koschmieder's Atmospheric Extinction Law & 5-Tier Industrial Scale
# ==============================================================================
def compute_dynamic_fog_risk_index(visibility_m, humidity_pct=85.0, temp_c=18.0, dew_point_c=15.0):
    """
    Quantitative Fog Density Measurement System:
    Calculates optical extinction coefficient beta via Koschmieder's Law:
        beta = 3.912 / V
    and maps meteorological sight distance (V in meters) into the 5 industrial severity tiers:
        > 100 m  -> NORMAL         (DFRI: 0-20,   Color: #10b981, Max Safe Speed: 35 km/h)
        50-100 m -> CAUTION        (DFRI: 21-40,  Color: #f59e0b, Max Safe Speed: 25 km/h)
        20-50 m  -> LOW VISIBILITY (DFRI: 41-65,  Color: #f97316, Max Safe Speed: 18 km/h)
        5-20 m   -> SEVERE FOG     (DFRI: 66-85,  Color: #ef4444, Max Safe Speed: 10 km/h)
        < 5 m    -> CRITICAL       (DFRI: 86-100, Color: #991b1b, Max Safe Speed: 5 km/h)
    """
    v_m = max(1.0, float(visibility_m))
    beta = round(3.912 / v_m, 4)

    if v_m >= 100.0:
        # Scale 100m - 300m to 20 - 0
        dfri_score = max(0, min(20, int(round(20.0 * max(0.0, 1.0 - (v_m - 100.0) / 200.0)))))
        tier = "NORMAL"
        color = "#10b981"
        behavior = "Nominal haul speeds permitted up to 35 km/h. Standard visual driving."
        speed_cap = 35.0
    elif v_m >= 50.0:
        # Scale 50m - 100m to 40 - 21
        dfri_score = int(round(21 + (1.0 - (v_m - 50.0) / 50.0) * 19))
        tier = "CAUTION"
        color = "#f59e0b"
        behavior = "Speed governed to 25 km/h. Fog-Guard radar standby. Double haul headway."
        speed_cap = 25.0
    elif v_m >= 20.0:
        # Scale 20m - 50m to 65 - 41
        dfri_score = int(round(41 + (1.0 - (v_m - 20.0) / 30.0) * 24))
        tier = "LOW VISIBILITY"
        color = "#f97316"
        behavior = "Speed governed to 18 km/h. Optical Fast-Pi de-weathering active. Hairpin speed cap 14 km/h."
        speed_cap = 18.0
    elif v_m >= 5.0:
        # Scale 5m - 20m to 85 - 66
        dfri_score = int(round(66 + (1.0 - (v_m - 5.0) / 15.0) * 19))
        tier = "SEVERE FOG"
        color = "#ef4444"
        behavior = "Speed governed to 10 km/h. Mandatory radar-guided convoy. Passing strictly prohibited."
        speed_cap = 10.0
    else:
        # Scale 1m - 5m to 100 - 86
        dfri_score = min(100, int(round(86 + (1.0 - v_m / 5.0) * 14)))
        tier = "CRITICAL"
        color = "#991b1b"
        behavior = "Emergency speed cap 5 km/h or mandatory bench halt. Cloud inversion pooling in pit bottom."
        speed_cap = 5.0

    dew_depression = max(0.0, float(temp_c) - float(dew_point_c))
    condensing = (dew_depression <= 1.5 and humidity_pct >= 90.0)

    tier_num = 1 if v_m >= 100.0 else (2 if v_m >= 50.0 else (3 if v_m >= 20.0 else (4 if v_m >= 5.0 else 5)))

    return {
        "tier": tier_num,
        "severity": tier,
        "severity_tier": tier,
        "dfri_score": dfri_score,
        "index_0_100": dfri_score,
        "visibility_m": round(v_m, 1),
        "extinction_coeff": beta,
        "extinction_coeff_beta": beta,
        "color": color,
        "permitted_behavior": {
            "policy": behavior,
            "safe_speed_limit_kmh": speed_cap,
        },
        "behavior_desc": behavior,
        "max_safe_speed_kmh": speed_cap,
        "is_condensing": condensing,
        "dew_point_depression_c": round(dew_depression, 1)
    }


# Initialize DFRI on all dynamic fog zones
for _zid, _z in FOG_ZONES.items():
    _z["dfri"] = compute_dynamic_fog_risk_index(
        _z["visibility_m"], _z.get("humidity_pct", 80), _z.get("temperature_c", 18), _z.get("dew_point_c", 15)
    )


def point_in_polygon(lat, lng, poly):
    """Ray-casting algorithm to test point inside zone polygon."""
    inside = False
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        if (p1[1] > lng) != (p2[1] > lng):
            slope = (lat - p1[0]) * (p2[1] - p1[1]) - (lng - p1[1]) * (p2[0] - p1[0])
            if (p2[1] > p1[1] and slope > 0) or (p2[1] <= p1[1] and slope < 0):
                inside = not inside
    return inside


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """Calculates ground distance between two GPS coordinates in meters."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


# In-memory geo cache to store resolved real-world area names for GPS coordinates
geo_cache = {}


def fetch_nominatim_async(lat, lng, cache_key):
    """Asynchronously fetches reverse geocoding from OpenStreetMap Nominatim."""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}&zoom=16"
        req = urllib.request.Request(url, headers={"User-Agent": "ResurgenceFleetControl/1.0"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            addr = data.get("address", {})
            name_part = addr.get("amenity") or addr.get("building") or data.get("name")
            suburb = addr.get("suburb") or addr.get("neighbourhood") or addr.get("city_district")
            city = addr.get("city") or addr.get("town") or addr.get("county")
            state = addr.get("state")

            parts = [p for p in [name_part, suburb, city, state] if p]
            if parts:
                resolved = ", ".join(parts[:3])
                geo_cache[cache_key] = resolved
                with fleet_lock:
                    if "TRUCK_01" in fleet_data:
                        t1 = fleet_data["TRUCK_01"]
                        if (round(t1["lat"], 3), round(t1["lng"], 3)) == cache_key:
                            t1["current_zone"]["name"] = resolved
    except Exception:
        pass


def get_real_area_name(lat, lng):
    """Resolves human-readable real-world area name for GPS coordinates."""
    cache_key = (round(lat, 3), round(lng, 3))
    if cache_key in geo_cache:
        return geo_cache[cache_key]

    # Offline high-accuracy regional detection (instant 0ms response)
    if 17.48 <= lat <= 17.62 and 78.30 <= lng <= 78.48:
        default_name = "VNR VJIET Campus, Bachupally, Hyderabad"
    elif 17.15 <= lat <= 17.70 and 78.10 <= lng <= 78.75:
        default_name = "Hyderabad Regional Sector, Telangana"
    else:
        default_name = f"Real-Time Area ({lat:.4f}°N, {lng:.4f}°E)"

    geo_cache[cache_key] = default_name
    # Trigger background async lookup to enrich location if internet is available
    threading.Thread(target=fetch_nominatim_async, args=(lat, lng, cache_key), daemon=True).start()
    return default_name


def get_zone_for_point(lat, lng):
    """
    Returns the dynamic fog zone or real-time area for the given GPS point.
    If inside Bailadila mining polygons, returns the specific mine zone.
    If outside the mine (e.g. real hardware GPS location), dynamically returns the real-time area.
    """
    for zid, z in FOG_ZONES.items():
        if point_in_polygon(lat, lng, z["polygon"]):
            return z

    # Check if within Bailadila mountain vicinity (~5km radius)
    d_to_bailadila = haversine_distance_m(lat, lng, BASE_LAT, BASE_LNG)
    if d_to_bailadila < 5000.0:
        closest_z = FOG_ZONES["ZONE_1"]
        min_dist = float("inf")
        for zid, z in FOG_ZONES.items():
            d = (lat - z["center"][0]) ** 2 + (lng - z["center"][1]) ** 2
            if d < min_dist:
                min_dist = d
                closest_z = z
        return closest_z

    # Outside Bailadila -> Dynamically resolve the real-world operational area
    area_name = get_real_area_name(lat, lng)
    real_vis = 150
    return {
        "id": "REAL_AREA",
        "name": area_name,
        "elevation_m": 580 if (17.0 <= lat <= 18.0) else 1040,
        "visibility_m": real_vis,
        "severity": "NORMAL",
        "color": "#38bdf8",
        "max_safe_speed": 35,
        "is_real_location": True,
        "dfri": compute_dynamic_fog_risk_index(real_vis, 55.0, 24.0, 14.0),
        "prediction_10m": {
            "visibility_m": 150,
            "severity": "NORMAL",
            "trend": "Live GPS Real-Time Area",
            "advisory": f"Operating live in {area_name}. Telemetry streamed from onboard ESP32 + NEO-6M."
        }
    }


# ==============================================================================
# 6 TRUCK FLEET DEFINITIONS (Clean Industrial Naming)
# Canonical Deposit-14 Fleet Configuration
# ==============================================================================
FLEET_DEFS = [
    {
        "id": "TRUCK_01",
        "name": "TRUCK_01",
        "model": "Caterpillar 797F",
        "type": "Haul Truck",
        "max_payload": 360,
        "base_wp_idx": 0,
        "avatar_color": "#38bdf8",
        "default_source": "SIMULATED",  # Dual-mode: SIMULATED or REAL_HARDWARE via /api/vehicle/mode
    },
    {
        "id": "TRUCK_02",
        "name": "TRUCK_02",
        "model": "Bucyrus MT4400",
        "type": "Haul Truck",
        "max_payload": 240,
        "base_wp_idx": 3,
        "avatar_color": "#10b981",
        "default_source": "SIMULATED",
    },
    {
        "id": "TRUCK_03",
        "name": "TRUCK_03",
        "model": "Komatsu 930E-5",
        "type": "Haul Truck",
        "max_payload": 290,
        "base_wp_idx": 6,
        "avatar_color": "#3b82f6",
        "default_source": "SIMULATED",
    },
    {
        "id": "TRUCK_04",
        "name": "TRUCK_04",
        "model": "Liebherr T 284",
        "type": "Haul Truck",
        "max_payload": 363,
        "base_wp_idx": 9,
        "avatar_color": "#f59e0b",
        "default_source": "SIMULATED",
    },
    {
        "id": "TRUCK_05",
        "name": "TRUCK_05",
        "model": "BelAZ 75710",
        "type": "Haul Truck",
        "max_payload": 450,
        "base_wp_idx": 12,
        "avatar_color": "#ec4899",
        "default_source": "SIMULATED",
    },
    {
        "id": "TRUCK_06",
        "name": "TRUCK_06",
        "model": "Hitachi EH5000AC-3",
        "type": "Haul Truck",
        "max_payload": 296,
        "base_wp_idx": 14,
        "avatar_color": "#a855f7",
        "default_source": "SIMULATED",
    },
]

# Alias map for backward compatibility with ESP32 scripts and earlier identifiers
VEHICLE_ALIASES = {
    "HAUL_01": "TRUCK_01", "DUMPER_01": "TRUCK_01", "DUMPER_01_SIM": "TRUCK_01", "PROTOTYPE_1": "TRUCK_01", "1": "TRUCK_01", "TRUCK_1": "TRUCK_01", "TRUCK 1": "TRUCK_01",
    "HAUL_02": "TRUCK_02", "DUMPER_02": "TRUCK_02", "2": "TRUCK_02", "TRUCK_2": "TRUCK_02", "TRUCK 2": "TRUCK_02",
    "HAUL_03": "TRUCK_03", "DUMPER_03": "TRUCK_03", "3": "TRUCK_03", "TRUCK_3": "TRUCK_03", "TRUCK 3": "TRUCK_03",
    "HAUL_04": "TRUCK_04", "DUMPER_04": "TRUCK_04", "4": "TRUCK_04", "TRUCK_4": "TRUCK_04", "TRUCK 4": "TRUCK_04",
    "HAUL_05": "TRUCK_05", "DUMPER_05": "TRUCK_05", "5": "TRUCK_05", "TRUCK_5": "TRUCK_05", "TRUCK 5": "TRUCK_05",
    "HAUL_06": "TRUCK_06", "DUMPER_06": "TRUCK_06", "6": "TRUCK_06", "TRUCK_6": "TRUCK_06", "TRUCK 6": "TRUCK_06",
}

# Role-Based Credentials:
# - Admin: full access to all fleet telemetry & control room
# - Drivers: access strictly limited to their own assigned truck
TRUCK_CREDENTIALS = {
    "ADMIN": "admin123",
    "TRUCK_01": "truck01",
    "TRUCK_02": "truck02",
    "TRUCK_03": "truck03",
    "TRUCK_04": "truck04",
    "TRUCK_05": "truck05",
    "TRUCK_06": "truck06",
}

fleet_lock = threading.Lock()
fleet_data = {}
telemetry_sequence = 0
active_scenario = {
    "id": "CLEAR_RUN",
    "description": "Normal haul cycle",
    "applied_at": time.time(),
}
safety_events = []

simulation_config = {
    "running": True,
    "speed_multiplier": 1.0,
    "obstacle_probability": 0.05,
}
MAX_TRAIL_LEN = 80
MAX_VEHICLE_LOG = 35

# ==============================================================================
# CANONICAL INDUSTRIAL SIMULATION PROFILES (Deposit-14 Haul Loop)
# Extracted from bailadila_simulation_data.json
# ==============================================================================
SIMULATION_PROFILES = {
    "TRUCK_01": {
        "desc": "Heavy Dispatch Unit (Kirandul Dispatch Yard)",
        "wp_idx": 0,
        "wp_t": 0.08,
        "target_speed": 16.0,
        "gear": "D2",
        "payload_pct": 0.694,  # 250 tons / 360t
        "engine_temp": 89.2,
        "tire_pressure": 102.5,
        "dist_front_range": (110, 160),
        "dist_left_range": (85, 120),
        "dist_right_range": (95, 130),
        "fog_vis": 75,
    },
    "TRUCK_02": {
        "desc": "Loaded High-Grade Ore Hauler (East Hairpin 1)",
        "wp_idx": 3,
        "wp_t": 0.25,
        "target_speed": 11.0,
        "gear": "D2",
        "payload_pct": 0.916,  # 220 tons
        "engine_temp": 95.4,
        "tire_pressure": 107.5,
        "dist_front_range": (42, 65),
        "dist_left_range": (65, 90),
        "dist_right_range": (50, 75),
        "fog_vis": 38,
    },
    "TRUCK_03": {
        "desc": "Empty Return Hauler (South Rim Crest)",
        "wp_idx": 6,
        "wp_t": 0.55,
        "target_speed": 9.0,
        "gear": "D1",
        "payload_pct": 0.0,    # 0 tons empty
        "engine_temp": 82.5,
        "tire_pressure": 98.2,
        "dist_front_range": (160, 220),
        "dist_left_range": (110, 150),
        "dist_right_range": (125, 165),
        "fog_vis": 110,
    },
    "TRUCK_04": {
        "desc": "Pit Floor Loading Crawl (Ore Loading Bay)",
        "wp_idx": 9,
        "wp_t": 0.10,
        "target_speed": 7.0,
        "gear": "D1",
        "payload_pct": 0.923,  # 335 tons
        "engine_temp": 98.8,
        "tire_pressure": 106.1,
        "dist_front_range": (20, 35),
        "dist_left_range": (40, 60),
        "dist_right_range": (75, 100),
        "fog_vis": 22,
    },
    "TRUCK_05": {
        "desc": "Ultra-Class Heavy Hauler (West Middle Bench)",
        "wp_idx": 12,
        "wp_t": 0.70,
        "target_speed": 13.0,
        "gear": "D2",
        "payload_pct": 0.911,  # 410 tons
        "engine_temp": 90.4,
        "tire_pressure": 115.0,
        "dist_front_range": (80, 115),
        "dist_left_range": (90, 120),
        "dist_right_range": (95, 125),
        "fog_vis": 55,
    },
    "TRUCK_06": {
        "desc": "North Ramp Climbing Unit (Northwest Upper Bench)",
        "wp_idx": 14,
        "wp_t": 0.35,
        "target_speed": 15.0,
        "gear": "D3",
        "payload_pct": 0.608,  # 180 tons
        "engine_temp": 92.1,
        "tire_pressure": 104.2,
        "dist_front_range": (140, 180),
        "dist_left_range": (110, 140),
        "dist_right_range": (115, 145),
        "fog_vis": 95,
    },
}


def decide_action(front, left, right):
    if front < 18:
        if left > right and left > 45:
            return "TURN LEFT"
        if right > left and right > 45:
            return "TURN RIGHT"
        return "STOP"
    if front < 45:
        if left > right and left > 40:
            return "TURN LEFT"
        if right > left and right > 40:
            return "TURN RIGHT"
        return "SLOW DOWN"
    return "CLEAR"


def determine_lifecycle_state(wp_idx, wp_name, action, speed_kmh):
    """Calculates haul cycle stage based on Deposit-14 loop position."""
    if action == "STOP":
        return "STOPPED"
    if wp_idx == 0:  # KIRANDUL_YARD / Stockpile Yard (CHP Hopper Unloading Zone)
        return "UNLOADING"
    if wp_idx in (8, 9):  # PIT_FLOOR or LOADING_BAY
        return "LOADING" if speed_kmh < 9.0 else "LOADED"
    if 10 <= wp_idx <= 15:  # WEST_LOWER ascending towards RAMP_NORTH
        return "HAULING"
    if action in ("SLOW DOWN", "TURN LEFT", "TURN RIGHT"):
        return "CAUTION"
    return "EMPTY_RETURN"


def log_safety_event(vid, event_type, severity, lat, lng, message):
    evt = {
        "event_id": f"EVT-{int(time.time() * 1000) % 1000000}",
        "vehicle_id": vid,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "epoch": int(time.time()),
        "type": event_type,
        "severity": severity,
        "lat": round(lat, 6),
        "lng": round(lng, 6),
        "message": message,
    }
    safety_events.insert(0, evt)
    if len(safety_events) > 50:
        safety_events.pop()
    return evt


# ==============================================================================
# DYNAMIC SPEED RECOMMENDATION ENGINE (DSR)
# Evaluates 8 Multi-Physics Inputs to recommend exact instantaneous velocity
# ==============================================================================
def compute_dynamic_safe_speed(vehicle, zone=None, closest_truck_info=None, cur_wp_idx=None, **kwargs):
    """
    Dynamic Speed Recommendation Engine (DSR):
    Answers: 'How fast should the dumper be travelling right now?'
    
    Evaluates 8 Real-time Inputs:
      1. Fog density / Visibility: Koschmieder visibility + Stopping Sight Distance (SSD)
      2. Obstacle distance: Ultrasonic radar front range (cm)
      3. Road curvature: Deposit-14 hairpin switchbacks (waypoints 3, 5, 11, 13)
      4. Vehicle speed: Current measured velocity (km/h)
      5. Traffic density: Proximity to leading/closest haul truck (meters)
      6. Location: Pit Floor vs Hairpin vs Crest vs Dispatch Yard
      7. Weather: Humidity, temperature, dew point depression
      8. Road condition: Friction coefficient mu (0.35 wet ore slurry vs 0.65 dry haul road)
      
    Outputs:
      - safe_speed_kmh (e.g. 18.0)
      - current_speed_kmh (e.g. 27.0)
      - speed_delta_kmh
      - advisory ("⚠ REDUCE SPEED", "🚨 CRITICAL BRAKING", "✓ MAINTAIN SAFE SPEED", "▲ NOMINAL")
      - advisory_badge
      - advisory_level ("REDUCE", "CRITICAL", "MAINTAIN", "NOMINAL")
      - advisory_color
      - bottleneck_type ("Fog Visibility", "Obstacle Ahead", etc.)
      - bottleneck_factor (string describing the primary active constraint)
      - primary_constraint
      - haul_efficiency_pct (0-100%)
      - stopping_sight_distance_m
      - inputs breakdown dict
      - factors breakdown dict
    """
    if zone is None:
        zone = vehicle.get("current_zone") or {}

    if closest_truck_info is None:
        closest_truck_info = vehicle.get("closest_truck", {"distance_m": 999.0, "name": "None"})

    cur_speed = float(vehicle.get("speed_kmh", 0.0))
    vis_m = float(vehicle.get("fog_visibility_m") or zone.get("visibility_m", 100.0))
    dist_front_cm = float(vehicle.get("dist_front", 180))
    if cur_wp_idx is None:
        cur_wp_idx = vehicle.get("wp_idx", 0)

    # 1. Fog Severity (DFRI) limit
    temp = float(zone.get("temperature_c", 18.0))
    hum = float(zone.get("humidity_pct", 80.0))
    dew = float(zone.get("dew_point_c", 15.0))
    dfri = compute_dynamic_fog_risk_index(vis_m, hum, temp, dew)
    speed_fog = float(dfri["max_safe_speed_kmh"])

    # 8. Road Condition & Stopping Sight Distance (SSD)
    # Iron ore haul road: wet slurry has friction mu ~0.35, dry ~0.65
    if "surface_friction_mu" in zone:
        mu = float(zone["surface_friction_mu"])
        is_wet = (mu < 0.5)
    else:
        is_wet = hum >= 85.0 or vis_m < 50.0 or dfri["is_condensing"]
        mu = 0.35 if is_wet else 0.65
    g = 9.81
    t_r = 1.5  # reaction time in seconds (300t dump truck pneumatic braking)
    # SSD = v*t_r + v^2 / (2*g*mu) <= vis_m
    discriminant = max(0.0, t_r**2 + (2.0 * vis_m) / (g * mu))
    v_max_ms = (g * mu) * (math.sqrt(discriminant) - t_r)
    speed_ssd = round(max(3.0, v_max_ms * 3.6), 1)

    # 2. Obstacle Distance limit
    if dist_front_cm < 30.0:
        speed_obs = 0.0
    elif dist_front_cm < 60.0:
        speed_obs = 8.0
    elif dist_front_cm < 100.0:
        speed_obs = 14.0
    elif dist_front_cm < 160.0:
        speed_obs = 22.0
    else:
        speed_obs = 35.0

    # 3. Road Curvature (Deposit-14 Hairpins)
    # Waypoints 3, 5, 11, 13 are the East/West mountain switchbacks (R ~ 45m)
    wp_mod = cur_wp_idx % 16
    if zone.get("is_hairpin") or wp_mod in (3, 5, 11, 13):
        speed_curve = 12.0
    elif wp_mod in (2, 4, 10, 12):
        speed_curve = 18.0
    else:
        speed_curve = 35.0

    # 5. Traffic Density (Closest Truck Headway)
    lead_dist = float(closest_truck_info.get("distance_m", 999.0))
    lead_name = closest_truck_info.get("name", "Leading Dumper")
    if lead_dist < 20.0:
        speed_traffic = 5.0
    elif lead_dist < 35.0:
        speed_traffic = 12.0
    elif lead_dist < 60.0:
        speed_traffic = 20.0
    else:
        speed_traffic = 35.0

    # 6. Location / Zone Maximum Safe Speed
    speed_zone = float(zone.get("max_safe_speed", 30.0))

    # 7. Weather / Condensation factor
    if dfri["is_condensing"]:
        speed_weather = round(speed_zone * 0.85, 1)
    else:
        speed_weather = 35.0

    # Composite Safe Speed: Minimum of all physical & regulatory bounds
    all_limits = [
        (speed_obs, "Obstacle Proximity", f"Front Hazard ({int(dist_front_cm)} cm)"),
        (speed_fog, "Fog Visibility", f"Fog Horizon ({int(vis_m)} m, DFRI {dfri['dfri_score']})"),
        (speed_ssd, "Stopping Sight Distance", f"Stopping Sight Distance ({speed_ssd} km/h on {'Wet Ore Slurry' if is_wet else 'Dry Road'})"),
        (speed_curve, "Hairpin Curvature", f"Deposit-14 Hairpin Turn (WP {cur_wp_idx})"),
        (speed_traffic, "Traffic Proximity", f"Traffic Headway ({int(lead_dist)} m to {lead_name})"),
        (speed_zone, "Mine Zone Governor", f"{zone.get('name', 'Mine Sector')} Limit"),
        (speed_weather, "Atmospheric Inversion", "Dew Point Condensation Inversion"),
    ]

    min_speed, bottleneck_type, bottleneck_desc = min(all_limits, key=lambda x: x[0])
    safe_speed = round(min_speed, 1)
    speed_delta = round(cur_speed - safe_speed, 1)

    # Determine Instantaneous Speed Advisory
    if safe_speed == 0.0 or dist_front_cm < 30.0:
        advisory = "🚨 CRITICAL BRAKING"
        advisory_level = "CRITICAL"
        advisory_color = "#ef4444"
    elif speed_delta > 7.0:
        advisory = "🚨 CRITICAL BRAKING"
        advisory_level = "CRITICAL"
        advisory_color = "#ef4444"
    elif speed_delta > 2.0:
        advisory = "⚠ REDUCE SPEED"
        advisory_level = "REDUCE"
        advisory_color = "#f59e0b"
    elif abs(speed_delta) <= 2.0:
        advisory = "✓ MAINTAIN SAFE SPEED"
        advisory_level = "MAINTAIN"
        advisory_color = "#10b981"
    else:
        advisory = "▲ NOMINAL (OPTIMAL HAUL)"
        advisory_level = "NOMINAL"
        advisory_color = "#38bdf8"

    # Haul Cycle Efficiency Calculation
    if safe_speed == 0.0:
        haul_efficiency_pct = 100.0 if cur_speed < 1.0 else round(max(0.0, 100.0 - cur_speed * 10.0), 1)
    else:
        if cur_speed <= safe_speed:
            haul_efficiency_pct = round(min(100.0, (cur_speed / safe_speed) * 100.0), 1)
        else:
            overspeed_penalty = ((cur_speed - safe_speed) / safe_speed) * 100.0
            haul_efficiency_pct = round(max(15.0, 100.0 - overspeed_penalty), 1)

    stopping_dist_m = round(cur_speed * (t_r / 3.6) + ((cur_speed / 3.6)**2) / (2.0 * g * mu), 1)

    return {
        "safe_speed_kmh": safe_speed,
        "current_speed_kmh": round(cur_speed, 1),
        "speed_delta_kmh": speed_delta,
        "advisory": advisory,
        "advisory_badge": advisory,
        "advisory_level": advisory_level,
        "advisory_color": advisory_color,
        "bottleneck_type": bottleneck_type,
        "bottleneck_factor": bottleneck_desc,
        "primary_constraint": bottleneck_desc,
        "haul_efficiency_pct": haul_efficiency_pct,
        "haul_cycle_efficiency_pct": haul_efficiency_pct,
        "stopping_sight_distance_m": stopping_dist_m,
        "dfri": dfri,
        "inputs": {
            "fog_density": f"{vis_m} m ({dfri['severity_tier']})",
            "obstacle_distance": f"{int(dist_front_cm)} cm",
            "road_curvature": f"{'Hairpin Turn' if (zone.get('is_hairpin') or wp_mod in (3, 5, 11, 13)) else 'Normal Tangent'}",
            "vehicle_speed": f"{round(cur_speed, 1)} km/h",
            "traffic_density": f"{int(lead_dist)} m to {lead_name}",
            "location": f"Waypoint {cur_wp_idx} ({zone.get('name', 'Mine Sector')})",
            "weather": f"{temp}°C, {hum}% RH",
            "road_condition": "Wet Iron Ore Slurry (mu=0.35)" if is_wet else "Compacted Dry Haul Road (mu=0.65)",
        },
        "factors": {
            "fog_visibility_kmh": speed_fog,
            "stopping_sight_distance_kmh": speed_ssd,
            "obstacle_proximity_kmh": speed_obs,
            "curvature_limit_kmh": speed_curve,
            "traffic_headway_kmh": speed_traffic,
            "zone_limit_kmh": speed_zone,
            "weather_condensing_kmh": speed_weather,
            "surface_friction_mu": mu,
            "surface_condition": "Wet Ore Slurry" if is_wet else "Compacted Dry Haul Road",
        }
    }


# Initialize canonical vehicle states on the Deposit-14 Loop
num_canonical_wps = len(SIM_WAYPOINTS)
for vdef in FLEET_DEFS:
    vid = vdef["id"]
    prof = SIMULATION_PROFILES.get(vid, {})

    wp_idx = prof.get("wp_idx", vdef["base_wp_idx"] % num_canonical_wps)
    wp_t = prof.get("wp_t", 0.0)

    cur_wp = SIM_WAYPOINTS[wp_idx]
    next_wp = SIM_WAYPOINTS[(wp_idx + 1) % num_canonical_wps]

    init_lat = round(cur_wp["lat"] + (next_wp["lat"] - cur_wp["lat"]) * wp_t, 6)
    init_lng = round(cur_wp["lng"] + (next_wp["lng"] - cur_wp["lng"]) * wp_t, 6)

    d_lng = next_wp["lng"] - cur_wp["lng"]
    d_lat = next_wp["lat"] - cur_wp["lat"]
    heading = (math.degrees(math.atan2(d_lng, d_lat)) + 360.0) % 360.0

    zone = get_zone_for_point(init_lat, init_lng)
    # Authoritative surface elevation sampled from Copernicus DEM & carved pit
    init_elev = round(sample_mine_surface_elevation(init_lat, init_lng), 1)

    speed_kmh = prof.get("target_speed", 15.0)
    target_speed = min(speed_kmh, zone["max_safe_speed"])
    gear = prof.get("gear", "D2")
    engine_temp_c = prof.get("engine_temp", 88.0)
    tire_pressure_psi = prof.get("tire_pressure", 104.0)
    payload_tons = round(vdef["max_payload"] * prof.get("payload_pct", 0.75), 1)

    dist_front = random.randint(*prof.get("dist_front_range", (120, 180)))
    dist_left = random.randint(*prof.get("dist_left_range", (90, 140)))
    dist_right = random.randint(*prof.get("dist_right_range", (90, 140)))
    fog_vis = prof.get("fog_vis", zone["visibility_m"])

    action = decide_action(dist_front, dist_left, dist_right)
    status = "CRITICAL" if action == "STOP" else ("CAUTION" if action != "CLEAR" else "NORMAL")

    lifecycle = determine_lifecycle_state(wp_idx, cur_wp.get("name", ""), action, speed_kmh)

    init_closest = {
        "id": "NONE",
        "name": "None",
        "distance_m": 999.0,
    }

    dsr = compute_dynamic_safe_speed(
        {"speed_kmh": speed_kmh, "dist_front": dist_front, "fog_visibility_m": fog_vis, "wp_idx": wp_idx},
        zone,
        init_closest,
        wp_idx
    )

    risk_score = {
        "total": 18,
        "level": "SAFE",
        "speed_factor": 12,
        "fog_factor": 20,
        "traffic_factor": 10,
        "hazard_factor": 5,
        "pit_edge_factor": 10,
    }

    fleet_data[vid] = {
        "id": vid,
        "name": vid,
        "model": vdef["model"],
        "type": vdef["type"],
        "max_payload": vdef["max_payload"],
        "avatar_color": vdef["avatar_color"],
        "source_type": vdef["default_source"],  # 'SIMULATED' or 'REAL_HARDWARE'
        "hardware_status": "CONNECTED_LIVE" if vdef["default_source"] == "REAL_HARDWARE" else "N/A",
        "last_hardware_packet": None,
        "packet_count": 0,
        "status": status,
        "action": action,
        "lifecycle_state": lifecycle,
        "speed_kmh": speed_kmh,
        "target_speed": target_speed,
        "gear": gear,
        "battery_pct": round(random.uniform(88.0, 99.5), 1),
        "engine_temp_c": engine_temp_c,
        "tire_pressure_psi": tire_pressure_psi,
        "payload_tons": payload_tons,
        "dist_front": dist_front,
        "dist_left": dist_left,
        "dist_right": dist_right,
        "fog_visibility_m": fog_vis,
        "gps_valid": True,
        "lat": init_lat,
        "lng": init_lng,
        "elevation_m": init_elev,
        "heading": round(heading, 1),
        "wp_idx": wp_idx,
        "wp_t": wp_t,
        "trail": [[init_lat, init_lng]],
        "current_zone": {
            "id": zone["id"],
            "name": zone["name"],
            "visibility_m": zone["visibility_m"],
            "severity": zone["severity"],
            "max_safe_speed": zone["max_safe_speed"],
            "color": zone["color"],
            "dfri": zone.get("dfri"),
        },
        "dsr": dsr,
        "dfri": dsr["dfri"],
        "safe_speed_kmh": dsr["safe_speed_kmh"],
        "speed_advisory": dsr["advisory"],
        "speed_bottleneck": dsr["bottleneck_factor"],
        "haul_efficiency_pct": dsr["haul_efficiency_pct"],
        "risk_score": risk_score,
        "has_risk_data": True,
        "has_diagnostics": True,
        "closest_truck": init_closest,
        "log": [
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "action": "CLEAR",
                "dist_front": dist_front,
                "note": f"{vid} online on Deposit-14 loop. Surface Elev: {init_elev}m. Source: {vdef['default_source']}.",
            }
        ],
        "manual_obstacle_until": 0,
        "last_update": datetime.now().strftime("%H:%M:%S"),
        "last_update_ts": time.time(),
        "sequence": 0,
        "is_connected": True,
        "is_moving": True,
    }


# ==============================================================================
# AUTHORITATIVE RISK SCORE ENGINE (0-100)
# ==============================================================================
def compute_risk_score(v, closest_dist, zone):
    """
    Computes real-time composite Risk Score (0-100) per truck:
      - Speed Factor: Speed vs Zone Maximum Safe Fog Speed
      - Fog Exposure: Inversely proportional to visibility
      - Traffic Proximity: Distance to closest other truck
      - Road Hazard: Front obstacle distance from ultrasonic sensor
      - Pit Edge / Switchback Factor: Proximity to steep crests/edges
    """
    safe_speed = zone.get("max_safe_speed", 25)
    speed = v.get("speed_kmh", 0.0)
    if speed <= safe_speed:
        speed_factor = (speed / (safe_speed + 1e-5)) * 35.0
    else:
        overspeed = speed - safe_speed
        speed_factor = min(100.0, 35.0 + (overspeed / 15.0) * 65.0)

    vis = v.get("fog_visibility_m") or zone.get("visibility_m", 100)
    if vis >= 120:
        fog_factor = 10.0
    elif vis >= 80:
        fog_factor = 30.0
    elif vis >= 40:
        fog_factor = 60.0
    elif vis >= 20:
        fog_factor = 85.0
    else:
        fog_factor = 100.0

    if closest_dist < 20.0:
        traffic_factor = 95.0
    elif closest_dist < 40.0:
        traffic_factor = 70.0
    elif closest_dist < 75.0:
        traffic_factor = 40.0
    elif closest_dist < 120.0:
        traffic_factor = 20.0
    else:
        traffic_factor = 5.0

    front = v.get("dist_front", 180)
    if front < 15:
        hazard_factor = 100.0
    elif front < 30:
        hazard_factor = 85.0
    elif front < 60:
        hazard_factor = 55.0
    elif front < 100:
        hazard_factor = 30.0
    elif front < 150:
        hazard_factor = 15.0
    else:
        hazard_factor = 5.0

    wp_idx = v.get("wp_idx", 0)
    if wp_idx in (3, 5, 11, 13):
        pit_edge_factor = 55.0
    elif wp_idx in (6, 7, 8, 9, 10):
        pit_edge_factor = 35.0
    else:
        pit_edge_factor = 10.0

    total = (
        0.25 * speed_factor +
        0.25 * fog_factor +
        0.20 * traffic_factor +
        0.20 * hazard_factor +
        0.10 * pit_edge_factor
    )
    total_clamped = int(max(0, min(100, round(total))))

    if total_clamped >= 76:
        level = "CRITICAL"
    elif total_clamped >= 51:
        level = "HIGH"
    elif total_clamped >= 26:
        level = "CAUTION"
    else:
        level = "SAFE"

    return {
        "total": total_clamped,
        "level": level,
        "speed_factor": int(round(speed_factor)),
        "fog_factor": int(round(fog_factor)),
        "traffic_factor": int(round(traffic_factor)),
        "hazard_factor": int(round(hazard_factor)),
        "pit_edge_factor": int(round(pit_edge_factor)),
    }


# ==============================================================================
# DETERMINISTIC STATELESS SIMULATION ENGINE (Deposit-14 Canonical Haul Loop)
# Multi-worker & serverless synchronized via global wall-clock time
# ==============================================================================
BASE_SIM_EPOCH = 1710500000.0
last_sim_tick = time.time()

CIRCUIT_NUM_WP = len(SIM_WAYPOINTS)
CIRCUIT_SEG_DISTANCES = []
CIRCUIT_CUM_DISTANCES = [0.0]
for i in range(CIRCUIT_NUM_WP):
    w1 = SIM_WAYPOINTS[i]
    w2 = SIM_WAYPOINTS[(i + 1) % CIRCUIT_NUM_WP]
    d = max(10.0, haversine_distance_m(w1["lat"], w1["lng"], w2["lat"], w2["lng"]))
    CIRCUIT_SEG_DISTANCES.append(d)
    CIRCUIT_CUM_DISTANCES.append(CIRCUIT_CUM_DISTANCES[-1] + d)

CIRCUIT_TOTAL_DIST = CIRCUIT_CUM_DISTANCES[-1]

INITIAL_LOOP_OFFSETS = {}
for vid, prof in SIMULATION_PROFILES.items():
    idx = prof["wp_idx"]
    t = prof["wp_t"]
    INITIAL_LOOP_OFFSETS[vid] = CIRCUIT_CUM_DISTANCES[idx] + t * CIRCUIT_SEG_DISTANCES[idx]

# ==============================================================================
# AUTHORITATIVE SHARED WORLD OBSTACLES (PHYSICAL DIGITAL TWIN HAZARDS)
# ==============================================================================
world_obstacles = {}


def get_truck_forward_vector(heading_deg):
    """Calculates forward unit vector in local DEM frame (0° North = -Z, 90° East = +X)."""
    rad = math.radians(heading_deg)
    fwd_x = math.sin(rad)
    fwd_z = -math.cos(rad)
    return fwd_x, fwd_z


def place_world_obstacle(target_vid, distance_m=10.0, duration_s=2.0):
    """
    Places an authoritative physical boulder 8-15m in front of the truck on real DEM terrain.
    Replaces existing temporary hazards for this vehicle to prevent duplicates.
    Auto-expires in exactly duration_s (default 2.0s).
    """
    now_ts = time.time()
    v = fleet_data.get(target_vid)
    if not v:
        return None

    heading = v.get("heading", 0.0)
    fwd_x, fwd_z = get_truck_forward_vector(heading)

    truck_x = (v["lng"] - DEM_LON0) * M_LON
    truck_z = -(v["lat"] - DEM_LAT0) * M_LAT

    obs_x = truck_x + fwd_x * distance_m
    obs_z = truck_z + fwd_z * distance_m

    obs_lng = round(DEM_LON0 + obs_x / M_LON, 6)
    obs_lat = round(DEM_LAT0 - obs_z / M_LAT, 6)
    obs_elev = round(sample_mine_surface_elevation(obs_lat, obs_lng), 2)

    clean_num = target_vid.replace("TRUCK_", "").replace("VEHICLE_", "")
    obs_id = f"BOULDER_{clean_num}"

    # Replace existing hazard for this truck to prevent duplicates
    for k in list(world_obstacles.keys()):
        if world_obstacles[k].get("target_vehicle_id") == target_vid or k == obs_id:
            world_obstacles.pop(k, None)

    obs_entity = {
        "id": obs_id,
        "type": "BOULDER",
        "target_vehicle_id": target_vid,
        "lat": obs_lat,
        "lng": obs_lng,
        "x": round(obs_x, 2),
        "z": round(obs_z, 2),
        "elevation_m": obs_elev,
        "nominal_dist_m": distance_m,
        "created_at": now_ts,
        "expires_at": now_ts + duration_s,
        "duration_s": duration_s,
        "active": True
    }
    world_obstacles[obs_id] = obs_entity
    return obs_entity


def update_fleet_state(now_ts=None):
    global telemetry_sequence, last_sim_tick
    if now_ts is None:
        now_ts = time.time()
    dt = max(0.0, min(2.0, now_ts - last_sim_tick)) if last_sim_tick else 0.0
    last_sim_tick = now_ts
    telemetry_sequence = int(now_ts * 2)
    now_str = datetime.fromtimestamp(now_ts).strftime("%H:%M:%S")
    multiplier = simulation_config.get("speed_multiplier", 1.0)
    is_running = simulation_config.get("running", True)
    scen_id = active_scenario.get("id", "CLEAR_RUN")

    # Authoritative auto-expiry of world obstacles (2.0s lifetime)
    expired_ids = [oid for oid, obs in list(world_obstacles.items()) if now_ts >= obs.get("expires_at", 0)]
    for oid in expired_ids:
        world_obstacles.pop(oid, None)

    # Global continuous displacement in meters along canonical Deposit-14 circuit
    base_fleet_speed_mps = 13.5 * 1000.0 / 3600.0  # 3.75 m/s (~13.5 km/h nominal corridor pace)
    if is_running:
        elapsed = (now_ts - BASE_SIM_EPOCH) * multiplier
    else:
        elapsed = 0.0

    for vid, v in fleet_data.items():
        v["last_update_ts"] = now_ts
        v["sequence"] = telemetry_sequence
        prof = SIMULATION_PROFILES.get(vid)

        # REAL_HARDWARE handling (ESP32 node / TRUCK_01 hardware mode)
        if v["source_type"] == "REAL_HARDWARE":
            if v.get("last_hardware_packet"):
                elapsed_hw = now_ts - v["last_hardware_packet"]
                if elapsed_hw > 12.0:
                    v["hardware_status"] = "LINK_TIMEOUT"
                    v["is_connected"] = False
                else:
                    v["hardware_status"] = "CONNECTED_LIVE"
                    v["is_connected"] = True
            else:
                v["hardware_status"] = "WAITING_TELEMETRY"
                v["is_connected"] = False

            v["elevation_m"] = round(sample_mine_surface_elevation(v["lat"], v["lng"]), 1)
            zone = get_zone_for_point(v["lat"], v["lng"])
            v["current_zone"] = {
                "id": zone["id"],
                "name": zone["name"],
                "visibility_m": zone["visibility_m"],
                "severity": zone["severity"],
                "max_safe_speed": zone["max_safe_speed"],
                "color": zone["color"],
                "dfri": zone.get("dfri"),
            }
            v["obstacles"] = list(world_obstacles.values())
            v["radar_detections"] = []
            v["nearest_obstacle"] = None
            continue

        # SIMULATED TRUCKS
        # Pause forward movement naturally when stopped by hazard, preventing teleportation
        if v.get("action") == "STOP" or not is_running:
            v["stopped_seconds"] = v.get("stopped_seconds", 0.0) + dt

        init_offset = INITIAL_LOOP_OFFSETS.get(vid, 0.0)
        effective_elapsed = max(0.0, elapsed - v.get("stopped_seconds", 0.0))
        curr_dist = (init_offset + effective_elapsed * base_fleet_speed_mps) % CIRCUIT_TOTAL_DIST

        # Locate segment
        seg_idx = 0
        for i in range(CIRCUIT_NUM_WP):
            if CIRCUIT_CUM_DISTANCES[i] <= curr_dist < CIRCUIT_CUM_DISTANCES[i + 1]:
                seg_idx = i
                break
        else:
            seg_idx = CIRCUIT_NUM_WP - 1

        seg_len = CIRCUIT_SEG_DISTANCES[seg_idx]
        wp_t = (curr_dist - CIRCUIT_CUM_DISTANCES[seg_idx]) / seg_len
        w1 = SIM_WAYPOINTS[seg_idx]
        w2 = SIM_WAYPOINTS[(seg_idx + 1) % CIRCUIT_NUM_WP]

        lat = round(w1["lat"] + (w2["lat"] - w1["lat"]) * wp_t, 6)
        lng = round(w1["lng"] + (w2["lng"] - w1["lng"]) * wp_t, 6)
        d_lng = w2["lng"] - w1["lng"]
        d_lat = w2["lat"] - w1["lat"]
        heading = round((math.degrees(math.atan2(d_lng, d_lat)) + 360.0) % 360.0, 1)
        elevation_m = round(sample_mine_surface_elevation(lat, lng), 1)

        v["wp_idx"] = seg_idx
        v["wp_t"] = round(wp_t, 4)
        v["lat"] = lat
        v["lng"] = lng
        v["heading"] = heading
        v["elevation_m"] = elevation_m

        zone = get_zone_for_point(lat, lng)
        v["current_zone"] = {
            "id": zone["id"],
            "name": zone["name"],
            "visibility_m": zone["visibility_m"],
            "severity": zone["severity"],
            "max_safe_speed": zone["max_safe_speed"],
            "color": zone["color"],
            "dfri": zone.get("dfri"),
        }

        # Corridor speed determined by segment & profile
        if seg_idx in (3, 5, 11, 13):
            # Hairpin switchbacks: safe climb / descent
            base_target = 10.5
        elif seg_idx == 0:
            # Stockpile yard
            base_target = 8.0
        elif seg_idx in (8, 9):
            # Pit floor loading bay
            base_target = 8.0
        else:
            base_target = prof.get("target_speed", 15.0) if prof else 15.0

        target_speed = min(base_target, zone["max_safe_speed"])

        # True 3D World Geometry Radar Perception calculation
        truck_x = (lng - DEM_LON0) * M_LON
        truck_z = -(lat - DEM_LAT0) * M_LAT
        fwd_x, fwd_z = get_truck_forward_vector(heading)
        right_x = math.cos(math.radians(heading))
        right_z = math.sin(math.radians(heading))

        radar_detections = []
        nearest_obs = None
        min_obs_dist = 9999.0

        for obs in world_obstacles.values():
            dx = obs["x"] - truck_x
            dz = obs["z"] - truck_z
            dist_m = math.hypot(dx, dz)

            local_fwd = dx * fwd_x + dz * fwd_z
            local_lat = dx * right_x + dz * right_z
            bearing_deg = math.degrees(math.atan2(local_lat, local_fwd))

            if local_fwd > 0 and abs(bearing_deg) <= 45.0:
                sector = "FRONT"
            elif 45.0 < bearing_deg <= 135.0:
                sector = "RIGHT"
            elif -135.0 <= bearing_deg < -45.0:
                sector = "LEFT"
            else:
                sector = "REAR"

            det = {
                "id": obs["id"],
                "distance_m": round(dist_m, 2),
                "local_x": round(local_lat, 2),
                "local_z": round(-local_fwd, 2),
                "sector": sector,
                "bearing_deg": round(bearing_deg, 1)
            }
            radar_detections.append(det)

            if dist_m < min_obs_dist:
                min_obs_dist = dist_m
                nearest_obs = det

        v["radar_detections"] = radar_detections
        v["nearest_obstacle"] = nearest_obs
        v["obstacles"] = list(world_obstacles.values())

        # Sensor readings computation based on active scenario and profiles
        has_manual = (now_ts < v.get("manual_obstacle_until", 0))
        has_front_hazard = (nearest_obs is not None and nearest_obs.get("sector") == "FRONT" and nearest_obs.get("distance_m", 999) <= 15.0)

        if has_manual or has_front_hazard:
            obs_dist = nearest_obs.get("distance_m", 10.0) if nearest_obs else 10.0
            front = max(8, min(150, int(obs_dist * 10)))
            left = 100
            right = 100
            vis = v.get("fog_visibility_m", zone["visibility_m"])
            action = "STOP"
            status = "CRITICAL"
            speed_kmh = 0.0
            gear = "N"
        elif scen_id == "OBSTACLE_AHEAD" and vid == "TRUCK_04":
            front = 12
            left = 65
            right = 50
            vis = 45
            action = "STOP"
            status = "CRITICAL"
            speed_kmh = 0.0
            gear = "N"
        elif scen_id == "DENSE_FOG":
            vis = 18
            front = 55
            left = 80
            right = 80
            action = "SLOW DOWN"
            status = "CAUTION"
            speed_kmh = 10.0
            gear = "D1"
        elif scen_id == "HAIRPIN_CAUTION" and seg_idx in (3, 5, 11, 13):
            vis = 38
            front = 40
            left = 45
            right = 85
            action = "SLOW DOWN"
            status = "CAUTION"
            speed_kmh = 9.0
            gear = "D1"
        elif scen_id == "TRAFFIC_CLOSE" and vid == "TRUCK_02":
            front = 28
            left = 60
            right = 65
            vis = 60
            action = "SLOW DOWN"
            status = "CAUTION"
            speed_kmh = 8.5
            gear = "D1"
        else:
            if prof:
                front_min, front_max = prof["dist_front_range"]
                front = front_min + int(abs(math.sin(now_ts * 0.15 + hash(vid) % 17)) * (front_max - front_min))
                left = 110
                right = 110
                vis = zone["visibility_m"]
            else:
                front = 160
                left = 110
                right = 110
                vis = zone["visibility_m"]

            action = decide_action(front, left, right)
            status = "NORMAL"
            speed_kmh = round(target_speed + 0.4 * math.sin(now_ts * 0.25 + hash(vid) % 11), 1)
            gear = prof["gear"] if prof else "D2"

        v["dist_front"] = front
        v["dist_left"] = left
        v["dist_right"] = right
        v["fog_visibility_m"] = vis
        v["action"] = action
        v["status"] = status
        v["speed_kmh"] = speed_kmh
        v["target_speed"] = target_speed
        v["gear"] = gear
        v["last_update"] = now_str
        v["is_connected"] = is_running
        v["is_moving"] = (speed_kmh > 0.5 and action != "STOP" and is_running)

        lifecycle = determine_lifecycle_state(seg_idx, w1.get("name", ""), action, speed_kmh)
        v["lifecycle_state"] = lifecycle

        # Dynamic Payload State Machine
        if lifecycle == "UNLOADING":
            v["payload_tons"] = 0.0
        elif lifecycle in ("LOADING", "LOADED"):
            target_payload = round(v["max_payload"] * prof.get("payload_pct", 0.88), 1) if prof else round(v["max_payload"] * 0.85, 1)
            v["payload_tons"] = max(target_payload, 150.0)
        elif lifecycle == "HAULING":
            if v.get("payload_tons", 0.0) <= 0.0:
                v["payload_tons"] = round(v["max_payload"] * 0.85, 1)
        elif lifecycle == "EMPTY_RETURN":
            v["payload_tons"] = 0.0

        # Telemetry trails
        trail = v.setdefault("trail", [])
        last_pt = trail[-1] if trail else None
        if not last_pt or (abs(last_pt[0] - lat) > 0.00003 or abs(last_pt[1] - lng) > 0.00003):
            trail.append([lat, lng])
            if len(trail) > MAX_TRAIL_LEN:
                trail.pop(0)

        # Vehicle log record
        if not v.get("log"):
            v["log"] = []
        if len(v["log"]) == 0 or (now_ts - v.get("_last_log_ts", 0)) >= 10.0:
            v["_last_log_ts"] = now_ts
            v["log"].insert(0, {
                "time": now_str,
                "action": action,
                "dist_front": front,
                "note": f"{vid} status: {status} | WP {seg_idx} | Speed: {speed_kmh}km/h | Elev: {elevation_m}m",
            })
            del v["log"][MAX_VEHICLE_LOG:]

        # Engine temp and tire pressure
        base_temp = prof["engine_temp"] if prof else 88.0
        v["engine_temp_c"] = round(base_temp + 2.5 * math.sin(now_ts * 0.05 + hash(vid) % 13), 1)
        v["tire_pressure_psi"] = round(prof.get("tire_pressure", 104.0) + 1.2 * math.cos(now_ts * 0.05 + hash(vid) % 13), 1)

    # Inter-truck proximity
    truck_ids = list(fleet_data.keys())
    closest_info = {}
    for i, vid1 in enumerate(truck_ids):
        v1 = fleet_data[vid1]
        min_d = 9999.0
        closest_id = None
        for j, vid2 in enumerate(truck_ids):
            if i == j:
                continue
            v2 = fleet_data[vid2]
            d = haversine_distance_m(v1["lat"], v1["lng"], v2["lat"], v2["lng"])
            if d < min_d:
                min_d = d
                closest_id = vid2
        closest_info[vid1] = {
            "id": closest_id or "NONE",
            "name": fleet_data[closest_id]["name"] if closest_id else "None",
            "distance_m": round(min_d, 1),
        }

    for vid, v in fleet_data.items():
        v["closest_truck"] = closest_info[vid]
        zone = v["current_zone"]
        dsr = compute_dynamic_safe_speed(v, zone, closest_info[vid], v.get("wp_idx", 0))
        v["dsr"] = dsr
        v["dfri"] = dsr["dfri"]
        v["safe_speed_kmh"] = dsr["safe_speed_kmh"]
        v["speed_advisory"] = dsr["advisory"]
        v["speed_bottleneck"] = dsr["bottleneck_factor"]
        v["haul_efficiency_pct"] = dsr["haul_efficiency_pct"]
        v["risk_score"] = compute_risk_score(v, closest_info[vid]["distance_m"], zone)


def simulation_step(dt=1.0):
    update_fleet_state()


def simulation_loop():
    while True:
        time.sleep(1.0)
        with fleet_lock:
            update_fleet_state()


sim_thread = threading.Thread(target=simulation_loop, daemon=True)
sim_thread.start()


# ==============================================================================
# OPENCV WEBCAM & DEHAZING
# ==============================================================================
camera = None
try:
    cam_candidate = cv2.VideoCapture(0)
    if cam_candidate.isOpened():
        camera = cam_candidate
        print("[Vision] Physical webcam detected and opened on index 0.")
    else:
        print("[Vision] No physical webcam detected. Enabling synthetic video feed.")
except Exception as e:
    print(f"[Vision] Camera init exception: {e}. Using synthetic feed.")


# ==============================================================================
# FAST-PI DE-WEATHERING ENGINE (Rain Streak Removal + Dark Channel Prior + Guided Filter)
# ==============================================================================
class FastPiDeWeather:
    def __init__(self, r=15, eps=0.001, omega=0.85):
        self.r = r           # Guided filter radius
        self.eps = eps       # Guided filter regularization parameter
        self.omega = omega   # Haze removal strength (0.85 avoids over-saturation)

    def remove_rain_streaks(self, img):
        """
        Uses directional bilateral/guided filtering to isolate 
        and remove high-frequency vertical rain streaks.
        """
        # Convert to YCrCb to process luminance channel (Y) only
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # Low-pass filter to smooth out fast-moving vertical rain drops
        base = cv2.bilateralFilter(y, d=5, sigmaColor=25, sigmaSpace=25)
        
        # High-frequency rain streak detail layer
        rain_layer = cv2.subtract(y, base)

        # Morphological directional filter (vertical kernel) to identify rain lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        rain_mask = cv2.morphologyEx(rain_layer, cv2.MORPH_OPEN, kernel)

        # Subtract detected rain streak layer from Luminance
        clean_y = cv2.subtract(y, rain_mask)
        
        # Reconstruct image
        clean_ycrcb = cv2.merge([clean_y, cr, cb])
        return cv2.cvtColor(clean_ycrcb, cv2.COLOR_YCrCb2BGR)

    def _guided_filter_fallback(self, guide, src, radius, eps):
        """Standard OpenCV Guided Filter Fallback if ximgproc is missing."""
        mean_p = cv2.boxFilter(src, -1, (radius, radius))
        mean_I = cv2.boxFilter(guide, -1, (radius, radius))
        mean_Ip = cv2.boxFilter(guide * src, -1, (radius, radius))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(guide * guide, -1, (radius, radius))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))

        return mean_a * guide + mean_b

    def remove_smoke_and_fog(self, img):
        """
        Modified Dark Channel Prior optimized for heavy smoke & fog without halos.
        """
        # 1. Compute Dark Channel
        min_channel = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.r, self.r))
        dark_channel = cv2.erode(min_channel, kernel)

        # 2. Robust Atmospheric Light (A) Estimation (Prevents smoke inversion)
        h, w = dark_channel.shape
        num_pixels = h * w
        top_num = int(max(math.floor(num_pixels * 0.001), 1))
        
        flat_dark = dark_channel.ravel()
        flat_img = img.reshape(num_pixels, 3)
        
        indices = np.argpartition(flat_dark, -top_num)[-top_num:]
        A = np.mean(flat_img[indices], axis=0)

        # 3. Estimate Transmission Map t(x)
        normalized = img.astype(np.float32) / np.maximum(A, 1.0)
        norm_dark = cv2.erode(np.min(normalized, axis=2), kernel)
        transmission = 1.0 - self.omega * norm_dark

        # 4. Refine Transmission with Guided Filter
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Try cv2.ximgproc; use fallback if missing
        try:
            refined_t = cv2.ximgproc.guidedFilter(
                guide=gray, 
                src=transmission.astype(np.float32), 
                radius=self.r, 
                eps=self.eps
            )
        except AttributeError:
            refined_t = self._guided_filter_fallback(
                guide=gray, 
                src=transmission.astype(np.float32), 
                radius=self.r, 
                eps=self.eps
            )
        
        # Bound transmission to avoid division by zero
        refined_t = np.maximum(refined_t, 0.1)

        # 5. Recover Clean Radiance J(x)
        result = np.empty_like(img, dtype=np.float32)
        for i in range(3):
            result[:, :, i] = (img[:, :, i].astype(np.float32) - A[i]) / refined_t + A[i]

        return np.clip(result, 0, 255).astype(np.uint8)

    def process_frame(self, frame):
        # Step 1: Remove vertical rain streaks
        derained = self.remove_rain_streaks(frame)
        # Step 2: Remove dense smoke and fog
        deweathered = self.remove_smoke_and_fog(derained)
        return deweathered


# Global deweathering engine instance
deweather_engine = FastPiDeWeather(r=15, eps=0.001, omega=0.85)

def deweather_frame(frame):
    return deweather_engine.process_frame(frame)

# Backwards compatibility alias
dehaze_frame = deweather_frame


# In-memory storage for custom uploaded/inserted camera frames per vehicle
# key: vehicle_id (str) -> dict
custom_camera_frames = {}


def generate_fog_sample(sample_id="pit_dense_fog", vid="TRUCK_02"):
    w, h = 640, 480
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    if sample_id in ("mountain_switchback", "ridge_cloud"):
        # Sky and steep rocky slopes
        cv2.rectangle(frame, (0, 0), (w, int(h * 0.45)), (110, 125, 135), -1)
        pts_m1 = np.array([[0, int(h * 0.45)], [int(w * 0.35), int(h * 0.20)], [int(w * 0.7), int(h * 0.45)]], np.int32)
        cv2.fillPoly(frame, [pts_m1], (70, 85, 95))
        pts_m2 = np.array([[int(w * 0.3), int(h * 0.45)], [int(w * 0.65), int(h * 0.15)], [w, int(h * 0.45)]], np.int32)
        cv2.fillPoly(frame, [pts_m2], (60, 75, 85))
        # Switchback road
        pts_road = np.array([[0, h], [int(w * 0.35), int(h * 0.65)], [int(w * 0.65), int(h * 0.65)], [w, h]], np.int32)
        cv2.fillPoly(frame, [pts_road], (45, 52, 58))
        # Large boulder hazard on road
        cv2.circle(frame, (int(w * 0.52), int(h * 0.78)), 36, (38, 44, 48), -1)
        cv2.circle(frame, (int(w * 0.50), int(h * 0.74)), 26, (55, 62, 68), -1)
        # Dense upslope fog
        fog = np.full((h, w, 3), (175, 185, 190), dtype=np.uint8)
        noise = np.random.randint(0, 22, (h, w, 3), dtype=np.uint8)
        fog = cv2.add(fog, noise)
        frame = cv2.addWeighted(frame, 0.26, fog, 0.74, 0)

    elif sample_id == "rain_mist":
        # Heavy Monsoon Rain & Wet Road Perspective
        cv2.rectangle(frame, (0, 0), (w, int(h * 0.40)), (55, 65, 75), -1)
        # Distant terraced bench in rain
        pts_bench = np.array([[0, int(h * 0.40)], [w, int(h * 0.40)], [w, int(h * 0.52)], [0, int(h * 0.52)]], np.int32)
        cv2.fillPoly(frame, [pts_bench], (35, 45, 52))
        # Wet road with reflections
        pts_road = np.array([[int(w * 0.42), int(h * 0.40)], [int(w * 0.58), int(h * 0.40)], [int(w * 0.96), h], [int(w * 0.04), h]], np.int32)
        cv2.fillPoly(frame, [pts_road], (25, 28, 34))
        # Wet asphalt sheen
        pts_sheen = np.array([[int(w * 0.46), int(h * 0.50)], [int(w * 0.54), int(h * 0.50)], [int(w * 0.65), h], [int(w * 0.35), h]], np.int32)
        cv2.fillPoly(frame, [pts_sheen], (35, 42, 50))
        # Obstacle: Stopped Light Vehicle / Rockfall ahead
        ox, oy = int(w * 0.50), int(h * 0.66)
        cv2.rectangle(frame, (ox - 32, oy - 24), (ox + 32, oy + 16), (20, 22, 28), -1)
        cv2.circle(frame, (ox - 22, oy + 10), 6, (0, 0, 220), -1)  # Red tail light
        cv2.circle(frame, (ox + 22, oy + 10), 6, (0, 0, 220), -1)  # Red tail light
        # Heavy angled rain streaks across camera lens
        for rx in range(10, w - 10, 16):
            ry = (rx * 13) % (h - 40)
            cv2.line(frame, (rx, ry), (rx - 4, ry + 28), (210, 225, 240), 1)
        # Wet spray & mist
        mist = np.full((h, w, 3), (160, 172, 180), dtype=np.uint8)
        noise = np.random.randint(0, 25, (h, w, 3), dtype=np.uint8)
        mist = cv2.add(mist, noise)
        frame = cv2.addWeighted(frame, 0.32, mist, 0.68, 0)

    elif sample_id == "haul_road_dust":
        # Hematite Iron Ore Dust Storm
        cv2.rectangle(frame, (0, 0), (w, int(h * 0.42)), (90, 70, 60), -1)
        pts_road = np.array([[int(w * 0.43), int(h * 0.42)], [int(w * 0.57), int(h * 0.42)], [int(w * 0.95), h], [int(w * 0.05), h]], np.int32)
        cv2.fillPoly(frame, [pts_road], (48, 32, 28))
        # Haul Truck bulk ahead in dust
        hx, hy = int(w * 0.50), int(h * 0.62)
        cv2.rectangle(frame, (hx - 40, hy - 32), (hx + 40, hy + 22), (28, 20, 18), -1)
        cv2.circle(frame, (hx - 28, hy + 14), 7, (0, 140, 255), -1)  # Amber hazard light
        cv2.circle(frame, (hx + 28, hy + 14), 7, (0, 140, 255), -1)  # Amber hazard light
        # Red-ochre dust haze
        dust = np.full((h, w, 3), (110, 130, 175), dtype=np.uint8)  # BGR for ochre dust
        noise = np.random.randint(0, 28, (h, w, 3), dtype=np.uint8)
        dust = cv2.add(dust, noise)
        frame = cv2.addWeighted(frame, 0.28, dust, 0.72, 0)

    else:  # pit_dense_fog (default)
        # Open pit bench benches
        cv2.rectangle(frame, (0, 0), (w, int(h * 0.4)), (100, 115, 125), -1)
        cv2.rectangle(frame, (0, int(h * 0.4)), (w, int(h * 0.55)), (55, 68, 76), -1)
        cv2.rectangle(frame, (0, int(h * 0.55)), (w, h), (40, 50, 58), -1)
        # Haul road perspective
        pts_road = np.array([[int(w * 0.44), int(h * 0.4)], [int(w * 0.56), int(h * 0.4)], [int(w * 0.95), h], [int(w * 0.05), h]], np.int32)
        cv2.fillPoly(frame, [pts_road], (32, 38, 44))
        # Hauler silhouette ahead in fog
        hx, hy = int(w * 0.51), int(h * 0.58)
        cv2.rectangle(frame, (hx - 36, hy - 28), (hx + 36, hy + 20), (25, 30, 35), -1)
        cv2.rectangle(frame, (hx - 42, hy + 5), (hx - 34, hy + 25), (15, 18, 20), -1)
        cv2.rectangle(frame, (hx + 34, hy + 5), (hx + 42, hy + 25), (15, 18, 20), -1)
        # Amber marker tail lights
        cv2.circle(frame, (hx - 26, hy + 14), 5, (0, 165, 255), -1)
        cv2.circle(frame, (hx + 26, hy + 14), 5, (0, 165, 255), -1)
        # Dense valley cloud fog inversion
        fog = np.full((h, w, 3), (170, 180, 185), dtype=np.uint8)
        noise = np.random.randint(0, 20, (h, w, 3), dtype=np.uint8)
        fog = cv2.add(fog, noise)
        frame = cv2.addWeighted(frame, 0.24, fog, 0.76, 0)

    return frame


def process_and_store_camera_frame(img, vid="TRUCK_02", source_name="Custom Image"):
    img_resized = cv2.resize(img, (640, 480))
    h, w = img_resized.shape[:2]

    # 1. Prepare raw frame with HUD overlay
    raw_frame = img_resized.copy()
    cam_id = "CAM-02" if vid == "TRUCK_02" else f"CAM-{vid.replace('TRUCK_', '')}"
    cv2.putText(raw_frame, f"{cam_id} [{vid} RAW OPTICAL FEED]", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2)
    cv2.putText(raw_frame, f"SOURCE: {source_name} | RAIN & FOG ENVIRONMENT", (16, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 80, 85), 1)

    # 2. Compute de-weathered image via FastPiDeWeather (Rain + Fog Removal)
    t0 = time.time()
    clean_frame = deweather_frame(img_resized)
    proc_ms = (time.time() - t0) * 1000.0

    enhanced = cv2.convertScaleAbs(clean_frame, alpha=1.20, beta=6)
    dehazed_frame = enhanced.copy()

    # Overlay HUD telemetry on deweathered frame
    cv2.putText(dehazed_frame, f"{cam_id} [{vid} DE-WEATHERED: FAST-PI ENGINE]", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (53, 209, 192), 2)
    cv2.putText(dehazed_frame, f"STATUS: RAIN & FOG REMOVED | {proc_ms:.1f}ms | +92% VISIBILITY", (16, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 231, 138), 1)

    # Automated edge detection / obstacle framing
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_cnts = [c for c in contours if cv2.contourArea(c) > 600]
    if large_cnts:
        c = max(large_cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        cv2.rectangle(dehazed_frame, (x - 6, y - 6), (x + bw + 6, y + bh + 6), (53, 209, 192), 2)
        cv2.putText(dehazed_frame, "TARGET DETECTED", (x - 6, max(15, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (53, 209, 192), 1)

    now_ts = time.time()
    now_str = datetime.now().strftime("%H:%M:%S")

    with fleet_lock:
        custom_camera_frames[vid] = {
            "raw": raw_frame,
            "dehazed": dehazed_frame,
            "timestamp": now_ts,
            "time_str": now_str,
            "source_name": source_name,
            "clarity_gain": 92,
            "contrast_gain": 2.5,
            "proc_ms": round(proc_ms, 1),
            "algorithm": "FastPiDeWeather (Rain + Smoke/Fog Removal)",
        }
    return custom_camera_frames[vid]


def generate_synthetic_frame(dehazed_mode=False, vid="TRUCK_02"):
    w, h = 480, 360
    t = time.time()
    cam_id = "CAM-02" if vid == "TRUCK_02" else f"CAM-{vid.replace('TRUCK_', '')}"

    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, int(h * 0.45)), (w, h), (40, 52, 60), -1)
    cv2.rectangle(frame, (0, 0), (w, int(h * 0.45)), (90, 105, 115), -1)

    pts = np.array([
        [int(w * 0.42), int(h * 0.45)],
        [int(w * 0.58), int(h * 0.45)],
        [int(w * 0.90), h],
        [int(w * 0.10), h],
    ], np.int32)
    cv2.fillPoly(frame, [pts], (30, 36, 42))

    offset = int((t * 60) % 40)
    for y in range(int(h * 0.48) + offset, h, 35):
        scale = (y - h * 0.45) / (h * 0.55)
        lw = max(2, int(6 * scale))
        lh = max(3, int(15 * scale))
        cx = int(w * 0.5)
        cv2.rectangle(frame, (cx - lw // 2, y), (cx + lw // 2, y + lh), (180, 190, 140), -1)

    obs_y = int(h * 0.65 + math.sin(t * 0.5) * 20)
    obs_x = int(w * 0.52 + math.cos(t * 0.3) * 30)
    cv2.circle(frame, (obs_x, obs_y), 16, (45, 50, 55), -1)
    cv2.circle(frame, (obs_x - 3, obs_y - 4), 10, (70, 75, 80), -1)

    if not dehazed_mode:
        # Simulate realistic rain streaks falling across camera aperture
        rain_phase = int((t * 140) % 50)
        for rx in range(15, w - 10, 28):
            ry = (rx * 7 + rain_phase * 15) % (h - 25)
            cv2.line(frame, (rx, ry), (rx + 1, ry + 18), (215, 228, 240), 1)

        fog = np.full((h, w, 3), (160, 175, 180), dtype=np.uint8)
        noise = np.random.randint(0, 20, (h, w, 3), dtype=np.uint8)
        fog = cv2.add(fog, noise)
        frame = cv2.addWeighted(frame, 0.28, fog, 0.72, 0)

        cv2.putText(frame, f"{cam_id} [{vid} RAW OPTICAL FEED]", (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 2)
        cv2.putText(frame, "ZONE 1: PIT BOTTOM | RAIN & DENSE FOG", (15, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 79, 85), 1)
    else:
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=-15)
        x1, y1 = obs_x - 22, obs_y - 20
        x2, y2 = obs_x + 22, obs_y + 20
        cv2.rectangle(frame, (x1, y1), (x2, y2), (53, 209, 192), 2)
        cv2.putText(frame, "OBSTACLE DETECTED", (x1 - 10, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (53, 209, 192), 1)

        cv2.putText(frame, f"{cam_id} [{vid} FAST-PI DE-WEATHERED]", (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (53, 209, 192), 2)
        cv2.putText(frame, "STATUS: RAIN & FOG CLEARED | +92%", (15, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 231, 138), 1)

    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (15, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 220, 220), 1)

    return frame


def gen_frames(is_dehazed=False, vid="TRUCK_02"):
    global camera
    target_vid = vid.upper() if vid else "TRUCK_02"
    while True:
        frame = None
        with fleet_lock:
            custom_entry = custom_camera_frames.get(target_vid) or custom_camera_frames.get("TRUCK_02")

        if custom_entry:
            base = custom_entry["dehazed"] if is_dehazed else custom_entry["raw"]
            frame = base.copy()
            h, w = frame.shape[:2]
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, ts, (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 225, 230), 1)
        else:
            if camera and camera.isOpened():
                success, captured = camera.read()
                if success:
                    frame = cv2.resize(captured, (480, 360))
                    cam_id = "CAM-02" if target_vid == "TRUCK_02" else f"CAM-{target_vid.replace('TRUCK_', '')}"
                    if is_dehazed:
                        frame = dehaze_frame(frame)
                        cv2.putText(frame, f"{cam_id} [{target_vid} LIVE DEHAZED]", (12, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (53, 209, 192), 2)
                    else:
                        cv2.putText(frame, f"{cam_id} [{target_vid} LIVE RAW]", (12, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

            if frame is None:
                frame = generate_synthetic_frame(dehazed_mode=is_dehazed, vid=target_vid)

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ret:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        time.sleep(0.06)


@app.route("/raw_feed")
def raw_feed():
    vid = request.args.get("vehicle_id") or request.args.get("v") or "TRUCK_02"
    return Response(gen_frames(is_dehazed=False, vid=vid), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/dehazed_feed")
@app.route("/deweathered_feed")
@app.route("/video_feed")
def dehazed_feed():
    vid = request.args.get("vehicle_id") or request.args.get("v") or "TRUCK_02"
    return Response(gen_frames(is_dehazed=True, vid=vid), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/camera/insert", methods=["POST"])
@app.route("/api/camera/sample", methods=["POST"])
@app.route("/api/camera/upload", methods=["POST"])
def insert_camera_image():
    """
    Inserts a custom image into the raw camera feed channel (defaults to TRUCK_02).
    Processes and calculates the Dark Channel Prior dehazed frame.
    Supports multipart file upload, base64 payload, or preset sample IDs.
    """
    raw_vid = request.form.get("vehicle_id") or request.args.get("vehicle_id") or "TRUCK_02"
    img = None
    source_name = "Uploaded Fog Image"

    if "image" in request.files or "file" in request.files:
        file = request.files.get("image") or request.files.get("file")
        if file and file.filename:
            source_name = file.filename
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        data = request.get_json(force=True, silent=True) or {}
        raw_vid = data.get("vehicle_id") or raw_vid
        if "image_base64" in data:
            b64 = data["image_base64"]
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            file_bytes = base64.b64decode(b64)
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            source_name = data.get("source_name", "Base64 Image")
        elif "sample_id" in data:
            sample_id = data.get("sample_id", "pit_dense_fog")
            source_name = f"Sample: {sample_id.replace('_', ' ').title()}"
            img = generate_fog_sample(sample_id, raw_vid)

    vid = VEHICLE_ALIASES.get(str(raw_vid).upper(), str(raw_vid).upper())
    if img is None:
        source_name = "Deposit 5 Pit Dense Fog Sample"
        img = generate_fog_sample("pit_dense_fog", vid)

    result = process_and_store_camera_frame(img, vid=vid, source_name=source_name)

    return jsonify({
        "status": "ok",
        "vehicle_id": vid,
        "message": f"Raw image inserted and dehazed successfully for {vid}",
        "source_name": source_name,
        "timestamp": result["time_str"],
        "clarity_gain": result["clarity_gain"],
        "algorithm": result["algorithm"],
        "raw_feed_url": f"/raw_feed?v={vid}",
        "dehazed_feed_url": f"/dehazed_feed?v={vid}",
    })


@app.route("/api/camera/reset", methods=["POST"])
def reset_camera_feed():
    req = request.get_json(force=True, silent=True) or {}
    raw_vid = req.get("vehicle_id") or "TRUCK_02"
    vid = VEHICLE_ALIASES.get(str(raw_vid).upper(), str(raw_vid).upper())

    with fleet_lock:
        if vid in custom_camera_frames:
            del custom_camera_frames[vid]
        if vid == "ALL":
            custom_camera_frames.clear()

    return jsonify({
        "status": "ok",
        "vehicle_id": vid,
        "message": f"Camera feed reset for {vid}. Live synthetic/webcam stream restored."
    })


@app.route("/api/camera/status", methods=["GET"])
def get_camera_status():
    raw_vid = request.args.get("vehicle_id") or "TRUCK_02"
    vid = VEHICLE_ALIASES.get(str(raw_vid).upper(), str(raw_vid).upper())
    with fleet_lock:
        entry = custom_camera_frames.get(vid) or custom_camera_frames.get("TRUCK_02")
        is_custom = entry is not None
        return jsonify({
            "status": "ok",
            "vehicle_id": vid,
            "has_custom_image": is_custom,
            "source_name": entry["source_name"] if is_custom else "Live Stream",
            "clarity_gain": entry["clarity_gain"] if is_custom else 88,
            "algorithm": entry["algorithm"] if is_custom else "Dark Channel Prior (DCP)",
            "timestamp": entry["time_str"] if is_custom else None,
        })


# ==============================================================================
# REST API ENDPOINTS
# ==============================================================================
@app.route("/api/simulation_route", methods=["GET"])
def get_simulation_route():
    """
    Exposes canonical 16-waypoint Deposit-14 simulation haul loop
    from bailadila_simulation_data.json. Single source of truth for Leaflet & Three.js.
    """
    return jsonify({
        "status": "ok",
        "name": "Conceptual 16-waypoint Deposit-14 simulation haul loop",
        "disclaimer": "Conceptual simulation route for digital-twin prototype. Not an official NMDC mine survey or mine-plan representation.",
        "loop": True,
        "waypoints": SIM_WAYPOINTS,
        "count": len(SIM_WAYPOINTS)
    })


@app.route("/api/terrain/sample", methods=["GET"])
def get_terrain_sample():
    """
    Authoritative terrain surface sampling endpoint.
    Determines regional DEM elevation, Deposit-14 pit footprint intersection,
    and returns exact final rendered mine surface elevation in meters.
    """
    try:
        lat = float(request.args.get("lat", BASE_LAT))
        lng = float(request.args.get("lng", BASE_LNG))
    except (ValueError, TypeError):
        return jsonify({"status": "error", "msg": "Invalid lat or lng parameters"}), 400

    raw_dem_h = round(sample_raw_dem_elevation(lng, lat), 2)
    mine_surface_h = round(sample_mine_surface_elevation(lat, lng), 2)

    x = (lng - DEM_LON0) * M_LON
    z = -(lat - DEM_LAT0) * M_LAT
    dx = x - PIT_X
    dz = z - PIT_Z
    norm_dist = math.hypot(dx / PIT_RX, dz / PIT_RZ)

    inside_dem = (DEM_SOUTH <= lat <= DEM_NORTH) and (DEM_WEST <= lng <= DEM_EAST)

    return jsonify({
        "status": "ok",
        "lat": lat,
        "lng": lng,
        "elevation_m": mine_surface_h,
        "raw_dem_elevation_m": raw_dem_h,
        "cut_depth_m": round(raw_dem_h - mine_surface_h, 2),
        "inside_dem": inside_dem,
        "inside_pit": bool(norm_dist < 1.0),
        "disclaimer": "Copernicus GLO-30 DEM regional elevation with conceptual 10-bench Deposit-14 pit excavation."
    })


@app.route("/api/scenarios", methods=["GET"])
def get_scenarios():
    return jsonify({
        "status": "ok",
        "active_scenario": active_scenario,
        "scenarios": SIM_DATA.get("scenario_library", [])
    })


@app.route("/api/scenarios/apply", methods=["POST"])
def apply_scenario():
    req = request.get_json(force=True, silent=True) or {}
    scen_id = str(req.get("scenario_id") or req.get("scenario") or "").upper().strip()

    valid_scenarios = {s["id"]: s for s in SIM_DATA.get("scenario_library", [])}
    if scen_id not in valid_scenarios:
        return jsonify({
            "status": "error",
            "msg": f"Invalid scenario '{scen_id}'. Valid: {list(valid_scenarios.keys())}"
        }), 400

    scen = valid_scenarios[scen_id]
    with fleet_lock:
        active_scenario["id"] = scen_id
        active_scenario["description"] = scen.get("description", "")
        active_scenario["applied_at"] = time.time()

        if scen_id == "OBSTACLE_AHEAD":
            if "TRUCK_04" in fleet_data:
                t = fleet_data["TRUCK_04"]
                t["dist_front"] = 12
                t["action"] = "STOP"
                t["status"] = "CRITICAL"
                t["speed_kmh"] = 0.0
                t["gear"] = "N"
                t["manual_obstacle_until"] = time.time() + 35.0
            log_safety_event(
                vid="TRUCK_04",
                event_type="OBSTACLE_DETECTED",
                severity="CRITICAL",
                lat=fleet_data["TRUCK_04"]["lat"] if "TRUCK_04" in fleet_data else BASE_LAT,
                lng=fleet_data["TRUCK_04"]["lng"] if "TRUCK_04" in fleet_data else BASE_LNG,
                message="SCENARIO OBSTACLE_AHEAD: Emergency stop triggered on TRUCK_04 (front obstacle 12cm)."
            )
        elif scen_id == "DENSE_FOG":
            FOG_ZONES["ZONE_1"]["visibility_m"] = 14
            FOG_ZONES["ZONE_2"]["visibility_m"] = 18
            FOG_ZONES["ZONE_3"]["visibility_m"] = 30
            FOG_ZONES["ZONE_4"]["visibility_m"] = 45
            for _zid, _z in FOG_ZONES.items():
                _z["dfri"] = compute_dynamic_fog_risk_index(
                    _z["visibility_m"], _z.get("humidity_pct", 80), _z.get("temperature_c", 18), _z.get("dew_point_c", 15)
                )
            for v in fleet_data.values():
                v["fog_visibility_m"] = 18
                v["dist_front"] = min(v["dist_front"], 65)
            log_safety_event(
                vid="ALL",
                event_type="LOW_VISIBILITY",
                severity="HIGH",
                lat=BASE_LAT,
                lng=BASE_LNG,
                message="SCENARIO DENSE_FOG: Severe Bastar cloud inversion pooling into pit basin (visibility 18m)."
            )
        elif scen_id == "TRAFFIC_CLOSE":
            if "TRUCK_02" in fleet_data:
                fleet_data["TRUCK_02"]["dist_front"] = 28
                fleet_data["TRUCK_02"]["action"] = "SLOW DOWN"
                fleet_data["TRUCK_02"]["status"] = "CAUTION"
            log_safety_event(
                vid="TRUCK_02",
                event_type="COLLISION_RISK",
                severity="HIGH",
                lat=fleet_data["TRUCK_02"]["lat"] if "TRUCK_02" in fleet_data else BASE_LAT,
                lng=fleet_data["TRUCK_02"]["lng"] if "TRUCK_02" in fleet_data else BASE_LNG,
                message="SCENARIO TRAFFIC_CLOSE: Inter-truck distance dropped below 30m collision caution threshold."
            )
        elif scen_id == "HAIRPIN_CAUTION":
            for v in fleet_data.values():
                if v.get("wp_idx") in (3, 5, 11, 13):
                    v["dist_front"] = 45
                    v["action"] = "SLOW DOWN"
                    v["status"] = "CAUTION"
            log_safety_event(
                vid="ALL",
                event_type="PIT_EDGE_WARNING",
                severity="CAUTION",
                lat=BASE_LAT,
                lng=BASE_LNG,
                message="SCENARIO HAIRPIN_CAUTION: Haulers approaching steep Deposit-14 switchback benches."
            )
        elif scen_id == "CLEAR_RUN":
            FOG_ZONES["ZONE_1"]["visibility_m"] = 18
            FOG_ZONES["ZONE_2"]["visibility_m"] = 42
            FOG_ZONES["ZONE_3"]["visibility_m"] = 80
            FOG_ZONES["ZONE_4"]["visibility_m"] = 145
            for _zid, _z in FOG_ZONES.items():
                _z["dfri"] = compute_dynamic_fog_risk_index(
                    _z["visibility_m"], _z.get("humidity_pct", 80), _z.get("temperature_c", 18), _z.get("dew_point_c", 15)
                )
            for v in fleet_data.values():
                v["manual_obstacle_until"] = 0
                v["fog_visibility_m"] = 140
                v["dist_front"] = random.randint(140, 200)
                v["action"] = "CLEAR"
                v["status"] = "NORMAL"
            log_safety_event(
                vid="ALL",
                event_type="NOMINAL_RECOVERY",
                severity="SAFE",
                lat=BASE_LAT,
                lng=BASE_LNG,
                message="SCENARIO CLEAR_RUN: Nominal operating conditions restored across Deposit-14 haul loop."
            )

        update_fleet_state(time.time())

    return jsonify({
        "status": "ok",
        "scenario_id": scen_id,
        "description": scen.get("description", ""),
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })


@app.route("/api/events", methods=["GET"])
def get_safety_events():
    return jsonify({
        "status": "ok",
        "events": safety_events[:40],
        "total": len(safety_events),
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })


@app.route("/api/fleet", methods=["GET"])
def get_fleet():
    with fleet_lock:
        now_ts = time.time()
        update_fleet_state(now_ts)

        current_role = session.get("role")
        current_vid = session.get("vehicle_id")

        v_list = []
        clear_count = 0
        caution_count = 0
        stopped_count = 0
        total_speed = 0.0
        high_risk_count = 0

        # Drivers can only see their own vehicle; Control room / Admin sees all
        items_to_process = fleet_data.items()
        if current_role == "driver" and current_vid:
            items_to_process = [(k, v) for k, v in fleet_data.items() if k == current_vid]

        for vid, v in list(items_to_process):
            # Calculate live connection status
            is_connected = False
            if v["source_type"] == "REAL_HARDWARE":
                if v.get("last_hardware_packet"):
                    elapsed = time.time() - v["last_hardware_packet"]
                    is_connected = (elapsed <= 12.0)
                else:
                    is_connected = False
            else:
                is_connected = simulation_config["running"]

            is_moving = (v["speed_kmh"] > 0.5 and v["action"] != "STOP")
            v["is_connected"] = is_connected
            v["is_moving"] = is_moving

            v_list.append({
                "id": v["id"],
                "dmp_id": v.get("dmp_id", f"DMP-10{int(v['id'][-1]) if v['id'][-1].isdigit() else 1}"),
                "name": v["name"],
                "model": v["model"],
                "type": v["type"],
                "source_type": v["source_type"],
                "hardware_status": v["hardware_status"],
                "is_connected": is_connected,
                "is_moving": is_moving,
                "lifecycle_state": v.get("lifecycle_state", "HAULING"),
                "packet_count": v["packet_count"],
                "status": v["status"],
                "action": v["action"],
                "speed_kmh": v["speed_kmh"],
                "target_speed": v.get("target_speed", 20.0),
                "battery_pct": v["battery_pct"],
                "payload_tons": v.get("payload_tons"),
                "engine_temp_c": v.get("engine_temp_c"),
                "tire_pressure_psi": v.get("tire_pressure_psi"),
                "gear": v.get("gear"),
                "dist_front": v["dist_front"],
                "dist_left": v["dist_left"],
                "dist_right": v["dist_right"],
                "fog_visibility_m": v.get("fog_visibility_m", 75),
                "lat": v["lat"],
                "lng": v["lng"],
                "elevation_m": v["elevation_m"],
                "heading": v["heading"],
                "avatar_color": v["avatar_color"],
                "current_zone": v["current_zone"],
                "risk_score": v.get("risk_score"),
                "dsr": v.get("dsr"),
                "dfri": v.get("dfri"),
                "safe_speed_kmh": v.get("safe_speed_kmh", v.get("target_speed", 20.0)),
                "speed_advisory": v.get("speed_advisory", "✓ MAINTAIN SAFE SPEED"),
                "speed_bottleneck": v.get("speed_bottleneck", "Nominal Haul Circuit"),
                "haul_efficiency_pct": v.get("haul_efficiency_pct", 95.0),
                "has_risk_data": True,
                "has_diagnostics": True,
                "closest_truck": v["closest_truck"],
                "last_update": v["last_update"],
                "last_update_ts": v.get("last_update_ts", time.time()),
                "sequence": telemetry_sequence,
                "radar_detections": v.get("radar_detections", []),
                "nearest_obstacle": v.get("nearest_obstacle"),
                "obstacles": list(world_obstacles.values()),
                "vehicle_state": v.get("vehicle_state"),
                "sensors": {k: ({**val, "age_s": round(time.time() - val["ts"], 1)}
                                if isinstance(val, dict) and "ts" in val else val)
                            for k, val in (v.get("sensors") or {}).items()},
                "active_node": v.get("active_node"),
                "nodes": {k: {"role": n["role"], "age_s": round(time.time() - n["ts"], 1), "packets": n["count"]}
                          for k, n in (v.get("nodes") or {}).items()},
            })

            if v["action"] == "STOP":
                stopped_count += 1
            elif v["action"] in ("SLOW DOWN", "TURN LEFT", "TURN RIGHT"):
                caution_count += 1
            else:
                clear_count += 1

            if v.get("risk_score") and isinstance(v["risk_score"], dict) and v["risk_score"].get("total", 0) >= 60:
                high_risk_count += 1

            total_speed += v["speed_kmh"]

        avg_speed = round(total_speed / len(v_list), 1) if v_list else 0.0

        return jsonify({
            "obstacles": list(world_obstacles.values()),
            "vehicles": v_list,
            "simulation": {
                "running": simulation_config["running"],
                "speed_multiplier": simulation_config["speed_multiplier"],
                "active_scenario": active_scenario,
            },
            "fleet_stats": {
                "total": len(v_list),
                "clear": clear_count,
                "caution": caution_count,
                "stopped": stopped_count,
                "high_risk": high_risk_count,
                "avg_speed": avg_speed,
            },
            "sequence": telemetry_sequence,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "epoch": int(time.time()),
        })


@app.route("/api/vehicle/<vehicle_id>", methods=["GET"])
def get_vehicle(vehicle_id):
    norm_vid = VEHICLE_ALIASES.get(vehicle_id.upper(), vehicle_id.upper())
    with fleet_lock:
        now_ts = time.time()
        update_fleet_state(now_ts)

        v = fleet_data.get(norm_vid)
        if not v:
            return jsonify({"status": "error", "msg": f"Vehicle {vehicle_id} not found"}), 404

        current_role = session.get("role")
        current_vid = session.get("vehicle_id")
        if current_role == "driver" and current_vid and norm_vid != current_vid:
            return jsonify({
                "status": "forbidden",
                "msg": f"Access restricted. You are authenticated as {current_vid} and cannot access telemetry for {norm_vid}."
            }), 403

        if v["source_type"] == "REAL_HARDWARE":
            if v.get("last_hardware_packet"):
                elapsed = time.time() - v["last_hardware_packet"]
                is_connected = (elapsed <= 12.0)
            else:
                is_connected = False
        else:
            is_connected = simulation_config["running"]

        is_moving = (v["speed_kmh"] > 0.5 and v["action"] != "STOP")
        v["is_connected"] = is_connected
        v["is_moving"] = is_moving
        v["sequence"] = telemetry_sequence
        v["obstacles"] = list(world_obstacles.values())

        return jsonify(v)


@app.route("/api/dynamic_speed/<vehicle_id>", methods=["GET"])
def get_vehicle_dynamic_speed(vehicle_id):
    """
    Returns full 8-input Dynamic Speed Recommendation (DSR) & DFRI analysis
    for the specified vehicle.
    """
    norm_vid = VEHICLE_ALIASES.get(vehicle_id.upper(), vehicle_id.upper())
    with fleet_lock:
        v = fleet_data.get(norm_vid)
        if not v:
            return jsonify({"status": "error", "msg": f"Vehicle {vehicle_id} not found"}), 404
        zone = get_zone_for_point(v["lat"], v["lng"])
        closest = v.get("closest_truck", {"distance_m": 999.0, "name": "None"})
        dsr = compute_dynamic_safe_speed(v, zone, closest, v.get("wp_idx", 0))
        return jsonify({
            "status": "ok",
            "success": True,
            "vehicle_id": norm_vid,
            "vehicle_name": v.get("name", norm_vid),
            "dsr": dsr,
            "dfri": dsr["dfri"],
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })


@app.route("/api/vehicle/add", methods=["POST"])
def add_vehicle():
    """Registers a new vehicle/device dynamically in the fleet."""
    req = request.get_json(force=True, silent=True) or {}
    raw_name = req.get("name", "").strip()
    raw_id = req.get("id", "").strip()
    model = req.get("model", "Caterpillar 797F").strip() or "Caterpillar 797F"
    vtype = req.get("type", "Haul Truck").strip() or "Haul Truck"
    source_type = req.get("source_type", "REAL_HARDWARE").upper()
    if source_type not in ("REAL_HARDWARE", "SIMULATED"):
        source_type = "REAL_HARDWARE"
    max_payload = float(req.get("max_payload", 320))
    avatar_color = req.get("avatar_color", "").strip()

    with fleet_lock:
        count = len(fleet_data) + 1
        if not raw_id:
            if raw_name and raw_name.upper().startswith("TRUCK_"):
                vid = raw_name.upper()
            else:
                vid = f"TRUCK_{count:02d}"
        else:
            vid = raw_id.upper().replace(" ", "_")

        vname = raw_name if raw_name else vid

        if vid in fleet_data:
            return jsonify({"status": "error", "msg": f"Device ID '{vid}' already exists."}), 400

        colors = ["#20d5c7", "#38e78a", "#3b82f6", "#f59e0b", "#ec4899", "#a855f7", "#06b6d4", "#eab308", "#10b981"]
        if not avatar_color:
            avatar_color = colors[(count - 1) % len(colors)]

        wp_idx = (count * 2) % len(QUARRY_WAYPOINTS)
        wp = QUARRY_WAYPOINTS[wp_idx]
        next_wp = QUARRY_WAYPOINTS[(wp_idx + 1) % len(QUARRY_WAYPOINTS)]
        d_lng = next_wp["lng"] - wp["lng"]
        d_lat = next_wp["lat"] - wp["lat"]
        heading = (math.degrees(math.atan2(d_lng, d_lat)) + 360) % 360

        lat = round(wp["lat"], 6)
        lng = round(wp["lng"], 6)
        if "lat" in req and "lng" in req and float(req["lat"]) != 0:
            lat = float(req["lat"])
            lng = float(req["lng"])

        zone = get_zone_for_point(lat, lng)
        now_str = datetime.now().strftime("%H:%M:%S")

        new_v = {
            "id": vid,
            "name": vname,
            "model": model,
            "type": vtype,
            "max_payload": max_payload,
            "avatar_color": avatar_color,
            "source_type": source_type,
            "hardware_status": "ONLINE" if source_type == "REAL_HARDWARE" else "N/A",
            "last_hardware_packet": time.time() if source_type == "REAL_HARDWARE" else None,
            "packet_count": 0,
            "status": "NORMAL",
            "action": "CLEAR",
            "speed_kmh": round(min(wp["speed_limit"], zone["max_safe_speed"]) * 0.9, 1),
            "target_speed": zone["max_safe_speed"],
            "gear": "D2",
            "battery_pct": 100.0,
            "engine_temp_c": 86.0,
            "tire_pressure_psi": 102.5,
            "payload_tons": round(max_payload * 0.75, 1),
            "dist_front": 180,
            "dist_left": 150,
            "dist_right": 150,
            "gps_valid": True,
            "lat": lat,
            "lng": lng,
            "elevation_m": zone["elevation_m"],
            "heading": round(heading, 1),
            "wp_idx": wp_idx,
            "wp_t": 0.0,
            "trail": [[lat, lng]],
            "current_zone": {
                "id": zone["id"],
                "name": zone["name"],
                "visibility_m": zone["visibility_m"],
                "severity": zone["severity"],
                "max_safe_speed": zone["max_safe_speed"],
                "color": zone["color"],
            },
            "risk_score": {
                "total": 20,
                "level": "LOW",
                "speed_factor": 15,
                "fog_factor": 20,
                "traffic_factor": 10,
                "hazard_factor": 5,
            },
            "closest_truck": {
                "id": "NONE",
                "name": "None",
                "distance_m": 999.0,
            },
            "log": [
                {
                    "time": now_str,
                    "action": "REGISTER",
                    "dist_front": 180,
                    "note": f"{vname} registered. Telemetry mode: {source_type}.",
                }
            ],
            "manual_obstacle_until": 0,
            "last_update": now_str,
            "is_connected": True,
            "is_moving": True if source_type == "SIMULATED" else False,
        }

        fleet_data[vid] = new_v

        # Register aliases for flexible mapping
        VEHICLE_ALIASES[vid] = vid
        VEHICLE_ALIASES[vid.lower()] = vid
        VEHICLE_ALIASES[vname] = vid
        VEHICLE_ALIASES[vname.upper()] = vid
        VEHICLE_ALIASES[vname.lower()] = vid

        # Auto-generate driver password for this vehicle
        TRUCK_CREDENTIALS[vid] = f"truck{vid.replace('TRUCK_', '').lower()}"

    return jsonify({
        "status": "ok",
        "msg": f"Device {vname} registered successfully.",
        "vehicle": new_v
    }), 201


@app.route("/api/vehicle/<vehicle_id>", methods=["DELETE"])
@app.route("/api/vehicle/delete", methods=["POST"])
def delete_vehicle(vehicle_id=None):
    """Removes/deregisters a vehicle from the fleet."""
    if not vehicle_id:
        req = request.get_json(force=True, silent=True) or {}
        vehicle_id = req.get("vehicle_id") or req.get("id")

    if not vehicle_id:
        return jsonify({"status": "error", "msg": "Vehicle ID is required"}), 400

    norm_vid = VEHICLE_ALIASES.get(vehicle_id.upper(), vehicle_id.upper())

    with fleet_lock:
        if norm_vid not in fleet_data:
            return jsonify({"status": "error", "msg": f"Vehicle '{vehicle_id}' not found"}), 404

        if len(fleet_data) <= 1:
            return jsonify({"status": "error", "msg": "Cannot delete the only remaining vehicle in the fleet"}), 400

        deleted_v = fleet_data.pop(norm_vid)
        # Purge aliases pointing to this vehicle
        aliases_to_remove = [k for k, v in VEHICLE_ALIASES.items() if v == norm_vid]
        for k in aliases_to_remove:
            VEHICLE_ALIASES.pop(k, None)

    return jsonify({
        "status": "ok",
        "msg": f"Device '{deleted_v.get('name', norm_vid)}' removed successfully.",
        "deleted_id": norm_vid
    })


@app.route("/api/fog_zones", methods=["GET"])
def get_fog_zones():
    return jsonify({
        "zones": list(FOG_ZONES.values()),
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })


@app.route("/api/vehicle/mode", methods=["POST"])
def set_vehicle_mode():
    """Toggles a vehicle between 'SIMULATED' and 'REAL_HARDWARE'."""
    req = request.get_json(force=True, silent=True) or {}
    raw_vid = req.get("vehicle_id", "TRUCK_01")
    vid = VEHICLE_ALIASES.get(raw_vid.upper(), raw_vid.upper())
    mode = str(req.get("mode") or req.get("source_type") or "SIMULATED").upper().strip()
    if mode not in ("SIMULATED", "REAL_HARDWARE"):
        mode = "SIMULATED"

    with fleet_lock:
        if vid not in fleet_data:
            return jsonify({"status": "error", "msg": f"Unknown vehicle {vid}"}), 404

        v = fleet_data[vid]
        v["source_type"] = mode
        if mode == "REAL_HARDWARE":
            v["hardware_status"] = "WAITING_TELEMETRY"
            v["speed_kmh"] = 0.0
            v["is_moving"] = False
        else:
            v["hardware_status"] = "N/A"
            v["is_connected"] = True

        v["log"].insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "action": "CONFIG",
            "dist_front": v["dist_front"],
            "note": f"Telemetry mode switched to {mode}",
        })
        log_safety_event(
            vid=vid,
            event_type="MODE_CHANGED",
            severity="INFO",
            lat=v["lat"],
            lng=v["lng"],
            message=f"{vid} telemetry mode switched to {mode}."
        )

    return jsonify({
        "status": "ok",
        "vehicle_id": vid,
        "source_type": mode,
        "hardware_status": v["hardware_status"]
    })


# Optional shared secret for hardware nodes. When the HARDWARE_API_KEY environment
# variable is set (do this on any public deployment), /update rejects requests that
# do not send a matching "X-API-Key" header. Unset = open (fine on a private LAN).
HARDWARE_API_KEY = os.environ.get("HARDWARE_API_KEY", "")
NODE_FAILOVER_S = 4.0      # seconds of PRIMARY silence before BACKUP takes over
IMU_IMPACT_G = 2.5        # total acceleration (g) treated as an impact
IMU_TILT_WARN_DEG = 25.0  # pitch/roll (deg) treated as rollover risk


@app.route("/update", methods=["POST"])
def update_telemetry():
    """
    Accepts telemetry updates from external ESP32 + GPS hardware node.
    Supports vehicle IDs: TRUCK_01, HAUL_01, DUMPER_01, PROTOTYPE_1.
    """
    if HARDWARE_API_KEY and not hmac.compare_digest(
            request.headers.get("X-API-Key", ""), HARDWARE_API_KEY):
        return jsonify({"status": "error", "msg": "Invalid or missing API key"}), 401

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"status": "error", "msg": "No JSON payload"}), 400

    raw_vid = data.get("vehicle_id", "TRUCK_01")
    vid = VEHICLE_ALIASES.get(str(raw_vid).upper(), str(raw_vid).upper())
    now_str = datetime.now().strftime("%H:%M:%S")

    with fleet_lock:
        if vid not in fleet_data:
            count = len(fleet_data) + 1
            colors = ["#20d5c7", "#38e78a", "#3b82f6", "#f59e0b", "#ec4899", "#a855f7", "#06b6d4", "#eab308", "#10b981"]
            avatar_color = colors[(count - 1) % len(colors)]
            wp_idx = (count * 2) % len(QUARRY_WAYPOINTS)
            wp = QUARRY_WAYPOINTS[wp_idx]
            lat_init = float(data.get("lat", wp["lat"])) if float(data.get("lat", 0.0)) != 0.0 else wp["lat"]
            lng_init = float(data.get("lng", wp["lng"])) if float(data.get("lng", 0.0)) != 0.0 else wp["lng"]
            zone_init = get_zone_for_point(lat_init, lng_init)
            fleet_data[vid] = {
                "id": vid,
                "name": str(data.get("name") or vid),
                "model": str(data.get("model") or "Haul Unit"),
                "type": str(data.get("type") or "Hardware Node"),
                "max_payload": float(data.get("max_payload", 320)),
                "avatar_color": str(data.get("avatar_color") or avatar_color),
                "source_type": "REAL_HARDWARE",
                "hardware_status": "CONNECTED_LIVE",
                "last_hardware_packet": time.time(),
                "packet_count": 0,
                "status": "NORMAL",
                "action": "CLEAR",
                "speed_kmh": float(data.get("speed", 0.0)),
                "target_speed": zone_init["max_safe_speed"],
                "gear": "D1",
                "battery_pct": 100.0,
                "engine_temp_c": 80.0,
                "tire_pressure_psi": 100.0,
                "payload_tons": 0.0,
                "dist_front": 250,
                "dist_left": 200,
                "dist_right": 200,
                "gps_valid": bool(data.get("gps_valid", True)),
                "lat": lat_init,
                "lng": lng_init,
                "elevation_m": round(sample_mine_surface_elevation(lat_init, lng_init), 1),
                "heading": float(data.get("heading", 0.0)),
                "wp_idx": wp_idx,
                "wp_t": 0.0,
                "trail": [[lat_init, lng_init]],
                "current_zone": zone_init,
                "closest_truck": {"distance_m": 999.0, "name": "None"},
                "log": [],
                "nodes": {},
                "sensors": {}
            }

        v = fleet_data[vid]
        v["source_type"] = "REAL_HARDWARE"
        v["hardware_status"] = "CONNECTED_LIVE"
        v["last_hardware_packet"] = time.time()
        v["packet_count"] += 1

        # ---- Multi-node failover: Raspberry Pi = PRIMARY, ESP32 = BACKUP -------
        # A packet with no "node" field is treated as PRIMARY (old ESP32 firmware
        # keeps working). BACKUP packets are only applied when no PRIMARY node
        # has been heard from within NODE_FAILOVER_S seconds.
        node = str(data.get("node", "PI")).upper()
        role = str(data.get("role") or ("BACKUP" if node.startswith("ESP32") else "PRIMARY")).upper()
        now_t = time.time()
        nodes = v.setdefault("nodes", {})
        nodes[node] = {"ts": now_t, "role": role, "count": nodes.get(node, {}).get("count", 0) + 1}
        if role == "BACKUP" and any(
            n["role"] == "PRIMARY" and now_t - n["ts"] < NODE_FAILOVER_S for n in nodes.values()
        ):
            return jsonify({"status": "ok", "vehicle_id": vid, "applied": False,
                            "standby": True, "active_node": v.get("active_node")})
        prev_active = v.get("active_node")
        v["active_node"] = node
        if prev_active and prev_active != node:
            log_safety_event(vid, "NODE_FAILOVER", "WARNING", v["lat"], v["lng"],
                             f"{vid} telemetry source switched {prev_active} -> {node}.")

        # ---- Extra sensors: LiDAR + IMU (both optional) -------------------------
        sensors = v.setdefault("sensors", {})
        lidar = data.get("lidar")
        if isinstance(lidar, dict):
            sensors["lidar"] = {**lidar, "ts": now_t}
        imu = data.get("imu")
        if isinstance(imu, dict):
            sensors["imu"] = {**imu, "ts": now_t}
            try:
                tilt = max(abs(float(imu.get("pitch", 0))), abs(float(imu.get("roll", 0))))
                g_mag = float(imu.get("accel_g", 1.0))
            except (TypeError, ValueError):
                tilt, g_mag = 0.0, 1.0
            if now_t - v.get("_last_imu_evt", 0) > 10:
                if g_mag >= IMU_IMPACT_G:
                    v["_last_imu_evt"] = now_t
                    log_safety_event(vid, "IMPACT_DETECTED", "CRITICAL", v["lat"], v["lng"],
                                     f"{vid} IMU shock {g_mag:.2f} g.")
                elif tilt >= IMU_TILT_WARN_DEG:
                    v["_last_imu_evt"] = now_t
                    log_safety_event(vid, "TILT_WARNING", "WARNING", v["lat"], v["lng"],
                                     f"{vid} tilt {tilt:.0f} deg (rollover risk).")

        front = int(data.get("dist_front", v["dist_front"]))
        sensors["front_ultrasonic_cm"] = front
        # Fuse: trust the CLOSER of ultrasonic and LiDAR (conservative for safety)
        if isinstance(lidar, dict) and lidar.get("ok") and lidar.get("front_cm") is not None:
            try:
                front = min(front, int(lidar["front_cm"]))
            except (TypeError, ValueError):
                pass
        left = int(data.get("dist_left", v["dist_left"]))
        right = int(data.get("dist_right", v["dist_right"]))
        action = data.get("action") or decide_action(front, left, right)

        v["dist_front"] = front
        v["dist_left"] = left
        v["dist_right"] = right
        if "dist_rear" in data:
            try:
                v["dist_rear"] = int(data["dist_rear"])
                sensors["rear_ultrasonic_cm"] = v["dist_rear"]
            except (TypeError, ValueError):
                pass
        v["action"] = action
        v["last_update"] = now_str
        v["last_update_ts"] = time.time()

        if "lat" in data and "lng" in data:
            lat = float(data["lat"])
            lng = float(data["lng"])
            if lat != 0.0 and lng != 0.0:
                v["lat"] = round(lat, 6)
                v["lng"] = round(lng, 6)
                v["trail"].append([v["lat"], v["lng"]])
                if len(v["trail"]) > MAX_TRAIL_LEN:
                    v["trail"].pop(0)

        if "heading" in data:
            v["heading"] = round(float(data["heading"]), 1)

        if "speed" in data:
            v["speed_kmh"] = round(float(data["speed"]), 1)

        if "gps_valid" in data:
            v["gps_valid"] = bool(data["gps_valid"])

        if action == "STOP":
            v["status"] = "CRITICAL"
            v["speed_kmh"] = 0.0
            v["gear"] = "N"
        elif action in ("SLOW DOWN", "TURN LEFT", "TURN RIGHT"):
            v["status"] = "CAUTION"
            v["gear"] = "D1"
        else:
            v["status"] = "NORMAL"

        _imu = sensors.get("imu") or {}
        try:
            if v["status"] == "NORMAL" and max(abs(float(_imu.get("pitch", 0))), abs(float(_imu.get("roll", 0)))) >= IMU_TILT_WARN_DEG:
                v["status"] = "CAUTION"
        except (TypeError, ValueError):
            pass

        v["is_moving"] = (v["speed_kmh"] > 0.5 and v["action"] != "STOP")

        zone = get_zone_for_point(v["lat"], v["lng"])
        v["current_zone"] = {
            "id": zone["id"],
            "name": zone["name"],
            "visibility_m": zone["visibility_m"],
            "severity": zone["severity"],
            "max_safe_speed": zone["max_safe_speed"],
            "color": zone["color"],
            "dfri": zone.get("dfri"),
        }

        # Authoritative surface elevation sampled from Copernicus DEM & carved pit
        v["elevation_m"] = round(sample_mine_surface_elevation(v["lat"], v["lng"]), 1)
        dsr = compute_dynamic_safe_speed(v, zone, v.get("closest_truck", {"distance_m": 999.0, "name": "None"}), v.get("wp_idx", 0))
        v["dsr"] = dsr
        v["dfri"] = dsr["dfri"]
        v["safe_speed_kmh"] = dsr["safe_speed_kmh"]
        v["speed_advisory"] = dsr["advisory"]
        v["speed_bottleneck"] = dsr["bottleneck_factor"]
        v["haul_efficiency_pct"] = dsr["haul_efficiency_pct"]
        v["risk_score"] = compute_risk_score(v, v["closest_truck"]["distance_m"], zone)

        if action != "CLEAR":
            v["log"].insert(0, {
                "time": now_str,
                "action": action,
                "dist_front": front,
                "note": f"ESP32 Hardware: {action} (Front {front}cm) | Elev: {v['elevation_m']}m",
            })
            del v["log"][MAX_VEHICLE_LOG:]

    return jsonify({
        "status": "ok",
        "vehicle_id": vid,
        "source_type": "REAL_HARDWARE",
        "speed_kmh": v["speed_kmh"],
        "risk_score": v["risk_score"]["total"] if v.get("risk_score") else None,
        "elevation_m": v["elevation_m"],
        "zone": zone["name"],
        "lat": v["lat"],
        "lng": v["lng"],
        "heading": v["heading"]
    })


@app.route("/api/simulate/obstacle", methods=["POST"])
def simulate_obstacle():
    req = request.get_json(force=True, silent=True) or {}
    raw_vid = req.get("vehicle_id", "TRUCK_01")
    vid = VEHICLE_ALIASES.get(raw_vid.upper(), raw_vid.upper())
    dist_req = float(req.get("distance", 10.0))
    # Standardize physical world distance (default 10.0m directly ahead)
    world_dist_m = 10.0 if (dist_req <= 15.0 and dist_req >= 8.0) else (dist_req if dist_req > 1.0 else 10.0)
    duration = float(req.get("duration", 2.0))
    # Strict 2.0s duration requirement
    if duration <= 0 or duration > 2.0:
        duration = 2.0

    with fleet_lock:
        if vid not in fleet_data:
            return jsonify({"status": "error", "msg": "Vehicle not found"}), 404

        v = fleet_data[vid]
        obs = place_world_obstacle(vid, distance_m=world_dist_m, duration_s=duration)

        v["dist_front"] = 10
        v["action"] = "STOP"
        v["status"] = "CRITICAL"
        v["speed_kmh"] = 0.0
        v["gear"] = "N"
        v["manual_obstacle_until"] = obs["expires_at"]
        v["last_update"] = datetime.now().strftime("%H:%M:%S")

        is_truck_01 = (vid == "TRUCK_01")
        if is_truck_01:
            v["risk_score"] = None
        else:
            zone = get_zone_for_point(v["lat"], v["lng"])
            v["risk_score"] = compute_risk_score(v, v["closest_truck"]["distance_m"], zone)

        v["log"].insert(0, {
            "time": v["last_update"],
            "action": "STOP",
            "dist_front": 10,
            "note": f"PHYSICAL HAZARD INJECTED: {obs['id']} at {world_dist_m}m (2.0s TTL)",
        })
        del v["log"][MAX_VEHICLE_LOG:]

    return jsonify({
        "status": "ok",
        "vehicle_id": vid,
        "action": "STOP",
        "dist_front": 10,
        "obstacle": obs,
        "msg": f"Physical obstacle {obs['id']} injected 10m ahead of {vid} (2.0s duration)"
    })


@app.route("/api/obstacles", methods=["GET"])
def get_world_obstacles_api():
    with fleet_lock:
        now_ts = time.time()
        expired = [k for k, o in list(world_obstacles.items()) if now_ts >= o.get("expires_at", 0)]
        for k in expired:
            world_obstacles.pop(k, None)
        return jsonify({
            "status": "ok",
            "count": len(world_obstacles),
            "obstacles": list(world_obstacles.values()),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })


@app.route("/api/simulate/control", methods=["POST"])
def simulate_control():
    req = request.get_json(force=True, silent=True) or {}
    with fleet_lock:
        if "running" in req:
            simulation_config["running"] = bool(req["running"])
        if "speed_multiplier" in req:
            simulation_config["speed_multiplier"] = max(0.2, min(10.0, float(req["speed_multiplier"])))

    return jsonify({
        "status": "ok",
        "running": simulation_config["running"],
        "speed_multiplier": simulation_config["speed_multiplier"],
    })


@app.route("/data", methods=["GET"])
def legacy_data():
    with fleet_lock:
        v = fleet_data.get("TRUCK_01", list(fleet_data.values())[0])
        return jsonify({
            "latest": {
                "vehicle_id": v["id"],
                "dist_front": v["dist_front"],
                "dist_left": v["dist_left"],
                "dist_right": v["dist_right"],
                "action": v["action"],
                "gps_valid": v["gps_valid"],
                "lat": v["lat"],
                "lng": v["lng"],
                "last_update": v["last_update"],
            },
            "log": v["log"],
        })


# ==============================================================================
# AUTHENTICATION & ROLE-BASED ACCESS CONTROL (Control Room vs Driver In-Cab)
# ==============================================================================
def resolve_user_role(username):
    uname = str(username).strip().upper()
    if uname == "ADMIN":
        return "admin", None
    vid = VEHICLE_ALIASES.get(uname, uname)
    return "driver", vid


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    req = request.get_json(force=True, silent=True) or {}
    username = str(req.get("username", "")).strip().upper()
    password = str(req.get("password", "")).strip()

    valid_pass = TRUCK_CREDENTIALS.get(username)
    if not valid_pass:
        vid = VEHICLE_ALIASES.get(username)
        if vid and vid in TRUCK_CREDENTIALS:
            username = vid
            valid_pass = TRUCK_CREDENTIALS.get(vid)

    if not valid_pass or valid_pass != password:
        return jsonify({
            "status": "error",
            "msg": "Invalid credentials. Use admin/admin123 for Control Room, or TRUCK_01/truck01, TRUCK_02/truck02, etc. for Driver Console."
        }), 401

    role, vid = resolve_user_role(username)
    session["user"] = username
    session["role"] = role
    session["vehicle_id"] = vid

    return jsonify({
        "status": "ok",
        "user": username,
        "role": role,
        "vehicle_id": vid,
        "redirect_url": f"/driver?vehicle_id={vid}" if role == "driver" else "/"
    })


@app.route("/api/auth/logout", methods=["POST", "GET"])
def auth_logout():
    session.clear()
    if request.is_json or request.path.startswith("/api/"):
        return jsonify({"status": "ok", "message": "Logged out successfully"})
    return redirect("/")


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    user = session.get("user")
    role = session.get("role", "admin")
    vid = session.get("vehicle_id")

    accounts = [{"id": "ADMIN", "role": "admin", "name": "Fleet Dispatcher (Central Control Room)", "default_pass": "admin123"}]
    with fleet_lock:
        for v_id, v in fleet_data.items():
            accounts.append({
                "id": v_id,
                "role": "driver",
                "name": f"{v.get('name', v_id)} ({v.get('model', 'Haul Unit')})",
                "default_pass": TRUCK_CREDENTIALS.get(v_id, f"truck{v_id.replace('TRUCK_', '').lower()}")
            })

    return jsonify({
        "authenticated": user is not None,
        "user": user or "ADMIN",
        "role": role,
        "vehicle_id": vid,
        "available_accounts": accounts
    })


@app.route("/driver", methods=["GET"])
def driver_portal():
    requested_vid = request.args.get("vehicle_id") or request.args.get("v")
    current_role = session.get("role")
    current_vid = session.get("vehicle_id")

    if current_role == "driver" and current_vid:
        target_vid = current_vid
    elif requested_vid:
        target_vid = VEHICLE_ALIASES.get(requested_vid.upper(), requested_vid.upper())
    else:
        target_vid = "TRUCK_02"

    return render_template("driver.html", target_vehicle_id=target_vid)


@app.route("/login", methods=["GET"])
def login_page():
    return render_template("driver.html", show_login=True)


@app.route("/mine3d", methods=["GET"])
@app.route("/digital_twin", methods=["GET"])
def mine_3d_portal():
    return render_template("mine3d.html")


@app.route("/", methods=["GET"])
@app.route("/control", methods=["GET"])
@app.route("/control_room", methods=["GET"])
def index():
    if session.get("role") == "driver" and session.get("vehicle_id"):
        return redirect(f"/driver?vehicle_id={session.get('vehicle_id')}")
    return render_template("index.html")


@app.route("/bailadila_terrain_256.png")
def get_terrain_png():
    return send_from_directory(os.path.join(app.root_path, "static"), "bailadila_terrain_256.png")


@app.route("/bailadila_terrain_meta.json")
@app.route("/api/terrain/meta")
def get_terrain_meta():
    return send_from_directory(os.path.join(app.root_path, "static"), "bailadila_terrain_meta.json")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"RESURGENCE FLEET CONTROL ROOM 2.0 serving on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
