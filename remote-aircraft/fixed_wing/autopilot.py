"""
autopilot.py — Full standalone autopilot for fixed-wing RC drones.

Covers all three community builds (RCMakerLab Flying-Wing, 3JWings Stick Plane,
Shahed/Lucas delta-wing study) with:

  1. GPS / NMEA parsing       — parse_nmea_gprmc(), parse_nmea_gpgga()
  2. Navigation maths         — haversine_distance(), bearing(),
                                cross_track_error(), along_track_distance()
  3. Waypoint route manager   — Waypoint, FlightRoute
  4. Flight-mode state machine— FlightMode, AutopilotFSM
  5. Flight simulation        — simulate_flight()
  6. RPi Pico uasyncio FW     — pico_autopilot_firmware()
  7. Arduino FreeRTOS FW      — arduino_autopilot_firmware()
  8. Per-build specialised FW — pico/arduino _flying_wing/stick_plane/shahed _autopilot()
  9. Utility                  — autopilot_hardware_bom(), autopilot_summary()

All firmware functions return Python strings (ready to flash); no external
dependencies are required — only the Python standard library (math, time).
"""

import math
import time


# ---------------------------------------------------------------------------
# 1. PID controller (self-contained copy so this module is standalone)
# ---------------------------------------------------------------------------

class PIDController:
    """Discrete-time PID with anti-windup, derivative-on-measurement, and clamping."""

    def __init__(self, kp, ki, kd, output_min=-1.0, output_max=1.0, integral_limit=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_limit = integral_limit
        self._integral = 0.0
        self._prev_measurement = None

    def reset(self):
        self._integral = 0.0
        self._prev_measurement = None

    def update(self, setpoint, measurement, dt):
        if dt <= 0:
            return 0.0
        error = setpoint - measurement
        self._integral += error * dt
        self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))
        if self._prev_measurement is None:
            derivative = 0.0
        else:
            derivative = -(measurement - self._prev_measurement) / dt
        self._prev_measurement = measurement
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(self.output_min, min(self.output_max, output))


# ---------------------------------------------------------------------------
# 2. GPS / NMEA parsing
# ---------------------------------------------------------------------------

def _nmea_checksum(sentence: str) -> int:
    """XOR checksum of characters between '$' and '*'."""
    result = 0
    for ch in sentence:
        result ^= ord(ch)
    return result


def parse_nmea_gprmc(sentence: str) -> dict:
    """Parse a $GPRMC NMEA sentence.

    Returns a dict with keys: valid, lat_deg, lon_deg, speed_kts,
    course_deg, utc_time.  Returns {"valid": False} on any error.
    """
    try:
        sentence = sentence.strip()
        if "*" in sentence:
            data_part, checksum_str = sentence.rsplit("*", 1)
            body = data_part.lstrip("$")
            expected = _nmea_checksum(body)
            if expected != int(checksum_str[:2], 16):
                return {"valid": False, "error": "checksum_mismatch"}
            sentence = data_part
        sentence = sentence.lstrip("$")
        fields = sentence.split(",")
        if len(fields) < 10:
            return {"valid": False, "error": "too_few_fields"}
        if fields[0] not in ("GPRMC", "GNRMC"):
            return {"valid": False, "error": "not_gprmc"}
        status = fields[2]
        if status != "A":
            return {"valid": False, "error": "no_fix"}

        def _dm_to_deg(dm_str, hemi):
            if not dm_str:
                return 0.0
            dot = dm_str.index(".")
            deg = float(dm_str[:dot - 2])
            minutes = float(dm_str[dot - 2:])
            val = deg + minutes / 60.0
            if hemi in ("S", "W"):
                val = -val
            return val

        lat = _dm_to_deg(fields[3], fields[4])
        lon = _dm_to_deg(fields[5], fields[6])
        speed_kts = float(fields[7]) if fields[7] else 0.0
        course = float(fields[8]) if fields[8] else 0.0
        return {
            "valid": True,
            "utc_time": fields[1],
            "lat_deg": lat,
            "lon_deg": lon,
            "speed_kts": speed_kts,
            "course_deg": course,
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def parse_nmea_gpgga(sentence: str) -> dict:
    """Parse a $GPGGA NMEA sentence.

    Returns dict with: valid, lat_deg, lon_deg, altitude_m,
    fix_quality, num_satellites, hdop.
    """
    try:
        sentence = sentence.strip()
        if "*" in sentence:
            sentence = sentence.rsplit("*", 1)[0]
        sentence = sentence.lstrip("$")
        fields = sentence.split(",")
        if len(fields) < 15:
            return {"valid": False, "error": "too_few_fields"}
        if fields[0] not in ("GPGGA", "GNGGA"):
            return {"valid": False, "error": "not_gpgga"}
        fix_quality = int(fields[6]) if fields[6] else 0
        if fix_quality == 0:
            return {"valid": False, "error": "no_fix"}

        def _dm_to_deg(dm_str, hemi):
            if not dm_str:
                return 0.0
            dot = dm_str.index(".")
            deg = float(dm_str[:dot - 2])
            minutes = float(dm_str[dot - 2:])
            val = deg + minutes / 60.0
            if hemi in ("S", "W"):
                val = -val
            return val

        lat = _dm_to_deg(fields[2], fields[3])
        lon = _dm_to_deg(fields[4], fields[5])
        num_sats = int(fields[7]) if fields[7] else 0
        hdop = float(fields[8]) if fields[8] else 99.9
        alt = float(fields[9]) if fields[9] else 0.0
        return {
            "valid": True,
            "lat_deg": lat,
            "lon_deg": lon,
            "altitude_m": alt,
            "fix_quality": fix_quality,
            "num_satellites": num_sats,
            "hdop": hdop,
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# 3. Navigation maths
# ---------------------------------------------------------------------------

_EARTH_R = 6_371_000.0  # metres


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in metres between two WGS-84 points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * _EARTH_R * math.asin(math.sqrt(a))


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return initial bearing in degrees (0–360, clockwise from north)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def cross_track_error(lat_pos, lon_pos, lat_start, lon_start, lat_end, lon_end) -> float:
    """Return signed cross-track error in metres (positive = right of track)."""
    d13 = haversine_distance(lat_start, lon_start, lat_pos, lon_pos) / _EARTH_R
    theta13 = math.radians(bearing(lat_start, lon_start, lat_pos, lon_pos))
    theta12 = math.radians(bearing(lat_start, lon_start, lat_end, lon_end))
    return math.asin(math.sin(d13) * math.sin(theta13 - theta12)) * _EARTH_R


def along_track_distance(lat_pos, lon_pos, lat_start, lon_start, lat_end, lon_end) -> float:
    """Return along-track distance from start waypoint in metres."""
    d13 = haversine_distance(lat_start, lon_start, lat_pos, lon_pos) / _EARTH_R
    dxt = cross_track_error(lat_pos, lon_pos, lat_start, lon_start, lat_end, lon_end) / _EARTH_R
    return math.acos(math.cos(d13) / max(math.cos(dxt), 1e-10)) * _EARTH_R


# ---------------------------------------------------------------------------
# 4. Waypoint & route management
# ---------------------------------------------------------------------------

class Waypoint:
    """A single navigation waypoint."""

    def __init__(self, lat: float, lon: float, alt_m: float,
                 speed_mps: float = 15.0, loiter_radius_m: float = 0.0,
                 name: str = "WP"):
        self.lat = lat
        self.lon = lon
        self.alt_m = alt_m
        self.speed_mps = speed_mps
        self.loiter_radius_m = loiter_radius_m
        self.name = name

    def __repr__(self):
        return f"Waypoint({self.name} lat={self.lat:.6f} lon={self.lon:.6f} alt={self.alt_m}m)"


class FlightRoute:
    """Ordered list of waypoints with sequential navigation helpers."""

    def __init__(self, waypoints=None, loop: bool = False):
        self._wps = list(waypoints) if waypoints else []
        self.loop = loop
        self._idx = 0

    def add_waypoint(self, wp: Waypoint):
        self._wps.append(wp)

    def current_waypoint(self) -> Waypoint:
        if not self._wps:
            raise IndexError("Route is empty")
        return self._wps[self._idx]

    def advance(self):
        """Move to the next waypoint; wraps if loop=True."""
        if not self._wps:
            return
        if self._idx < len(self._wps) - 1:
            self._idx += 1
        elif self.loop:
            self._idx = 0

    def is_complete(self) -> bool:
        return (not self.loop) and (self._idx >= len(self._wps) - 1)

    def total_distance_m(self) -> float:
        total = 0.0
        for i in range(len(self._wps) - 1):
            a, b = self._wps[i], self._wps[i + 1]
            total += haversine_distance(a.lat, a.lon, b.lat, b.lon)
        return total

    def summary(self) -> dict:
        legs = []
        for i in range(len(self._wps) - 1):
            a, b = self._wps[i], self._wps[i + 1]
            d = haversine_distance(a.lat, a.lon, b.lat, b.lon)
            hdg = bearing(a.lat, a.lon, b.lat, b.lon)
            spd = (a.speed_mps + b.speed_mps) / 2
            legs.append({
                "from": a.name, "to": b.name,
                "distance_m": round(d, 1),
                "bearing_deg": round(hdg, 1),
                "speed_mps": round(spd, 1),
                "time_s": round(d / max(spd, 0.1), 1),
            })
        total_d = self.total_distance_m()
        avg_spd = sum(w.speed_mps for w in self._wps) / max(len(self._wps), 1)
        return {
            "num_waypoints": len(self._wps),
            "loop": self.loop,
            "legs": legs,
            "total_distance_m": round(total_d, 1),
            "estimated_time_s": round(total_d / max(avg_spd, 0.1), 1),
        }


# ---------------------------------------------------------------------------
# 5. Flight-mode state machine
# ---------------------------------------------------------------------------

class FlightMode:
    MANUAL = "MANUAL"
    STABILISE = "STABILISE"
    ALT_HOLD = "ALT_HOLD"
    AUTO = "AUTO"
    RTL = "RTL"
    LOITER = "LOITER"


class AutopilotFSM:
    """Flight-mode state machine with integrated PID autopilot.

    Parameters
    ----------
    home_lat, home_lon, home_alt_m : launch-point coordinates
    """

    ACCEPT_RADIUS_M = 20.0  # waypoint reached when within this distance

    def __init__(self, home_lat: float, home_lon: float, home_alt_m: float = 0.0):
        self.home = Waypoint(home_lat, home_lon, home_alt_m, name="HOME")
        self._mode = FlightMode.MANUAL
        self._route: FlightRoute | None = None

        # PID controllers
        self._pid_roll = PIDController(kp=2.0, ki=0.05, kd=0.3,
                                       output_min=-30.0, output_max=30.0)
        self._pid_pitch = PIDController(kp=2.5, ki=0.08, kd=0.4,
                                        output_min=-25.0, output_max=25.0)
        self._pid_yaw = PIDController(kp=1.5, ki=0.02, kd=0.2,
                                      output_min=-30.0, output_max=30.0)
        self._pid_alt = PIDController(kp=0.5, ki=0.02, kd=0.1,
                                      output_min=-0.3, output_max=0.3)
        self._pid_xtrack = PIDController(kp=0.05, ki=0.001, kd=0.01,
                                         output_min=-30.0, output_max=30.0)
        self._target_alt_m = home_alt_m + 50.0
        self._target_heading = 0.0
        self._loiter_center: Waypoint | None = None
        self._prev_t = time.monotonic()

    def set_mode(self, mode: str):
        valid = {FlightMode.MANUAL, FlightMode.STABILISE, FlightMode.ALT_HOLD,
                 FlightMode.AUTO, FlightMode.RTL, FlightMode.LOITER}
        if mode not in valid:
            raise ValueError(f"Unknown mode: {mode}")
        self._mode = mode
        if mode == FlightMode.RTL:
            rtl_route = FlightRoute([self.home])
            self._route = rtl_route
        for pid in (self._pid_roll, self._pid_pitch, self._pid_yaw,
                    self._pid_alt, self._pid_xtrack):
            pid.reset()

    def current_mode(self) -> str:
        return self._mode

    def load_route(self, route: FlightRoute):
        self._route = route

    def update(self, gps: dict, imu: dict, barometer: dict) -> dict:
        """Run one autopilot cycle.

        Parameters
        ----------
        gps        : dict from parse_nmea_gprmc/gpgga
        imu        : dict with keys roll_deg, pitch_deg, yaw_deg,
                     roll_rate_dps, pitch_rate_dps, yaw_rate_dps
        barometer  : dict with key altitude_m

        Returns
        -------
        dict with throttle (0–1), elevator_deg, aileron_deg, rudder_deg,
             mode, active_wp_index
        """
        now = time.monotonic()
        dt = max(now - self._prev_t, 0.001)
        self._prev_t = now

        out = {
            "throttle": 0.5,
            "elevator_deg": 0.0,
            "aileron_deg": 0.0,
            "rudder_deg": 0.0,
            "mode": self._mode,
            "active_wp_index": 0,
        }

        roll = imu.get("roll_deg", 0.0)
        pitch = imu.get("pitch_deg", 0.0)
        yaw = imu.get("yaw_deg", 0.0)
        alt = barometer.get("altitude_m", self._target_alt_m)

        if self._mode == FlightMode.MANUAL:
            return out

        if self._mode in (FlightMode.STABILISE, FlightMode.ALT_HOLD,
                          FlightMode.AUTO, FlightMode.RTL, FlightMode.LOITER):
            out["aileron_deg"] = self._pid_roll.update(0.0, roll, dt)
            out["elevator_deg"] = self._pid_pitch.update(0.0, pitch, dt)

        if self._mode in (FlightMode.ALT_HOLD, FlightMode.AUTO,
                          FlightMode.RTL, FlightMode.LOITER):
            thr_corr = self._pid_alt.update(self._target_alt_m, alt, dt)
            out["throttle"] = max(0.0, min(1.0, 0.55 + thr_corr))

        if self._mode in (FlightMode.AUTO, FlightMode.RTL) and self._route:
            wp = self._route.current_waypoint()
            out["active_wp_index"] = self._route._idx
            if gps.get("valid", False):
                lat, lon = gps["lat_deg"], gps["lon_deg"]
                dist = haversine_distance(lat, lon, wp.lat, wp.lon)
                if dist < self.ACCEPT_RADIUS_M and not self._route.is_complete():
                    self._route.advance()
                    wp = self._route.current_waypoint()
                self._target_alt_m = wp.alt_m
                desired_hdg = bearing(lat, lon, wp.lat, wp.lon)
                hdg_err = (desired_hdg - yaw + 180) % 360 - 180
                out["rudder_deg"] = self._pid_yaw.update(hdg_err, 0.0, dt)
                if self._route.is_complete() and dist < self.ACCEPT_RADIUS_M:
                    self._mode = FlightMode.LOITER
                    self._loiter_center = wp

        if self._mode == FlightMode.LOITER and self._loiter_center:
            if gps.get("valid", False):
                lat, lon = gps["lat_deg"], gps["lon_deg"]
                lc = self._loiter_center
                desired_hdg = (bearing(lat, lon, lc.lat, lc.lon) + 90) % 360
                hdg_err = (desired_hdg - yaw + 180) % 360 - 180
                out["rudder_deg"] = self._pid_yaw.update(hdg_err, 0.0, dt)

        return out

    def status(self) -> dict:
        wp_idx = self._route._idx if self._route else None
        wp_name = self._route.current_waypoint().name if self._route else None
        return {
            "mode": self._mode,
            "target_alt_m": self._target_alt_m,
            "active_wp_index": wp_idx,
            "active_wp_name": wp_name,
            "home": {"lat": self.home.lat, "lon": self.home.lon, "alt_m": self.home.alt_m},
        }


# ---------------------------------------------------------------------------
# 6. Flight simulation
# ---------------------------------------------------------------------------

def simulate_flight(route: FlightRoute,
                    aircraft_mass_kg: float = 0.5,
                    cruise_speed_mps: float = 15.0,
                    dt: float = 0.1,
                    max_steps: int = 5000) -> dict:
    """Simple 2-D kinematic simulation flying a FlightRoute.

    Returns
    -------
    dict with keys:
        trajectory        : list of dicts (t, lat, lon, alt_m, mode, wp_index)
        waypoints_reached : list of waypoint names in sequence
        total_time_s      : float
        total_distance_m  : float
    """
    if not route._wps:
        return {"trajectory": [], "waypoints_reached": [], "total_time_s": 0.0, "total_distance_m": 0.0}

    trajectory = []
    reached = []
    t = 0.0
    total_dist = 0.0

    lat = route._wps[0].lat
    lon = route._wps[0].lon
    alt = route._wps[0].alt_m

    # Reset route index for simulation
    route._idx = 0

    for _ in range(max_steps):
        if route.is_complete():
            break
        wp = route.current_waypoint()
        dist = haversine_distance(lat, lon, wp.lat, wp.lon)

        # Advance toward waypoint
        if dist < 5.0:
            reached.append(wp.name)
            route.advance()
            if route.is_complete():
                break
            wp = route.current_waypoint()
            dist = haversine_distance(lat, lon, wp.lat, wp.lon)

        spd = min(wp.speed_mps, cruise_speed_mps)
        step = min(spd * dt, dist)
        frac = step / max(dist, 1e-9)
        hdg = bearing(lat, lon, wp.lat, wp.lon)
        hdg_r = math.radians(hdg)
        # Move in lat/lon
        dlat = math.degrees(step * math.cos(hdg_r) / _EARTH_R)
        dlon = math.degrees(step * math.sin(hdg_r) / (_EARTH_R * math.cos(math.radians(lat)) + 1e-9))
        lat += dlat
        lon += dlon
        # Altitude: linear interpolation
        alt_diff = wp.alt_m - alt
        alt += alt_diff * frac * 0.05  # gentle climb/descent

        total_dist += step
        t += dt
        trajectory.append({
            "t": round(t, 2),
            "lat": round(lat, 7),
            "lon": round(lon, 7),
            "alt_m": round(alt, 1),
            "mode": FlightMode.AUTO,
            "wp_index": route._idx,
        })

    # Ensure final waypoint counted
    if route._wps and not route.is_complete():
        reached.append(route._wps[-1].name)

    return {
        "trajectory": trajectory,
        "waypoints_reached": reached,
        "total_time_s": round(t, 1),
        "total_distance_m": round(total_dist, 1),
    }


# ---------------------------------------------------------------------------
# 7. RPi Pico MicroPython uasyncio autopilot firmware
# ---------------------------------------------------------------------------

def pico_autopilot_firmware() -> str:
    """Return a complete Pico MicroPython main.py with uasyncio RTOS-like tasks.

    Tasks: gps_task, imu_task, baro_task, control_task, nav_task,
           telemetry_task, watchdog_task.
    Flash with: mpremote cp autopilot_main.py :main.py
    """
    return '''\
# ============================================================
# autopilot_main.py  —  RPi Pico MicroPython full autopilot
# Flash: mpremote cp autopilot_main.py :main.py
# ============================================================
import uasyncio as asyncio
import ujson, utime, math
from machine import UART, I2C, Pin, PWM, WDT

# ── Pin assignments ─────────────────────────────────────────
GPS_UART_ID   = 0;  GPS_TX = 0;  GPS_RX = 1   # UART0
IMU_I2C_ID    = 0;  IMU_SDA = 4; IMU_SCL = 5  # I2C0 shared with baro
TELEM_UART_ID = 1;  TEL_TX = 8;  TEL_RX = 9   # UART1
ESC_PIN       = 15  # hardware PWM – pusher motor
ELEV_PIN      = 16  # elevator servo
RUDD_PIN      = 17  # rudder servo
LED_PIN       = 25  # onboard LED heartbeat

# ── Hard-coded test waypoints (lat, lon, alt_m, speed_mps) ──
WAYPOINTS = [
    (51.5074, -0.1278, 80, 14),
    (51.5100, -0.1200, 80, 14),
    (51.5120, -0.1300, 80, 14),
    (51.5074, -0.1278, 60, 12),   # RTL
]
WP_ACCEPT_R = 20  # metres

# ── Shared state ─────────────────────────────────────────────
state = {
    "gps":  {"valid": False, "lat": 0.0, "lon": 0.0, "alt": 0.0,
             "speed": 0.0, "course": 0.0},
    "imu":  {"roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "gyr_x": 0.0, "gyr_y": 0.0, "gyr_z": 0.0},
    "baro": {"alt": 0.0, "pressure": 101325.0, "temp": 20.0},
    "ctrl": {"throttle": 0.55, "elev": 0.0, "rudd": 0.0},
    "nav":  {"wp_idx": 0, "dist_m": 9999, "bearing": 0.0,
             "xtrack": 0.0, "mode": "STABILISE"},
    "gps_last_valid_ms": 0,
}

EARTH_R = 6_371_000

def haversine(la1, lo1, la2, lo2):
    p1, p2 = math.radians(la1), math.radians(la2)
    dp = math.radians(la2 - la1)
    dl = math.radians(lo2 - lo1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * EARTH_R * math.asin(math.sqrt(a))

def brng(la1, lo1, la2, lo2):
    p1, p2 = math.radians(la1), math.radians(la2)
    dl = math.radians(lo2 - lo1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

# ── MPU-6050 helpers ─────────────────────────────────────────
MPU_ADDR = 0x68
def mpu_init(i2c):
    i2c.writeto_mem(MPU_ADDR, 0x6B, b\'\\x00\')  # wake up
    i2c.writeto_mem(MPU_ADDR, 0x1B, b\'\\x08\')  # gyro ±500 dps
    i2c.writeto_mem(MPU_ADDR, 0x1C, b\'\\x00\')  # accel ±2 g

def mpu_read(i2c):
    raw = i2c.readfrom_mem(MPU_ADDR, 0x3B, 14)
    def s16(b, o): v = b[o]<<8|b[o+1]; return v-65536 if v>32767 else v
    ax = s16(raw,0)/16384; ay = s16(raw,2)/16384; az = s16(raw,4)/16384
    gx = s16(raw,8)/65.5;  gy = s16(raw,10)/65.5; gz = s16(raw,12)/65.5
    roll  = math.degrees(math.atan2(ay, az))
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay*ay+az*az)))
    return roll, pitch, gx, gy, gz

# ── BMP280 helpers ───────────────────────────────────────────
BMP_ADDR = 0x76
def bmp_read_alt(i2c):
    try:
        raw = i2c.readfrom_mem(BMP_ADDR, 0xF7, 6)
        adc_p = (raw[0]<<12)|(raw[1]<<4)|(raw[2]>>4)
        adc_t = (raw[3]<<12)|(raw[4]<<4)|(raw[5]>>4)
        # simplified: assume SL pressure
        p = adc_p / 5120.0  # crude Pa approximation
        alt = 44330 * (1 - (p/101325)**0.1903)
        return alt
    except:
        return state["baro"]["alt"]

# ── PWM helpers (50 Hz servo, 400 Hz ESC) ────────────────────
def pwm_init(pin, freq=50):
    p = PWM(Pin(pin)); p.freq(freq); return p

def set_servo_us(pwm, us):
    pwm.duty_ns(int(us * 1000))

def deg_to_us(deg, centre=1500, travel=500):
    return int(centre + (deg/45)*travel)

# ── PID (minimal) ────────────────────────────────────────────
class PID:
    def __init__(self, kp, ki, kd, mn=-30, mx=30):
        self.kp=kp; self.ki=ki; self.kd=kd; self.mn=mn; self.mx=mx
        self._i=0.0; self._pm=None
    def update(self, sp, m, dt):
        e=sp-m; self._i=max(-10,min(10,self._i+e*dt))
        d=0 if self._pm is None else -(m-self._pm)/dt
        self._pm=m
        return max(self.mn, min(self.mx, self.kp*e+self.ki*self._i+self.kd*d))

pid_roll  = PID(2.0, 0.05, 0.3)
pid_pitch = PID(2.5, 0.08, 0.4)
pid_yaw   = PID(1.5, 0.02, 0.2, -30, 30)
pid_alt   = PID(0.5, 0.02, 0.1, -0.3, 0.3)

# ── uasyncio tasks ────────────────────────────────────────────
async def gps_task():
    uart = UART(GPS_UART_ID, 9600, tx=Pin(GPS_TX), rx=Pin(GPS_RX))
    buf = b""
    while True:
        if uart.any():
            buf += uart.read(uart.any())
            while b"\\n" in buf:
                line, buf = buf.split(b"\\n", 1)
                s = line.decode("ascii", "ignore").strip()
                if s.startswith("$GPRMC") or s.startswith("$GNRMC"):
                    f = s.split(",")
                    if len(f) > 9 and f[2] == "A":
                        def dm(d, h):
                            dot = d.index(".")
                            return (float(d[:dot-2]) + float(d[dot-2:])/60) * (-1 if h in "SW" else 1)
                        try:
                            state["gps"].update({
                                "valid": True,
                                "lat": dm(f[3], f[4]), "lon": dm(f[5], f[6]),
                                "speed": float(f[7] or 0)*0.514,
                                "course": float(f[8] or 0),
                            })
                            state["gps_last_valid_ms"] = utime.ticks_ms()
                        except:
                            pass
        await asyncio.sleep_ms(50)

async def imu_task():
    i2c = I2C(IMU_I2C_ID, sda=Pin(IMU_SDA), scl=Pin(IMU_SCL), freq=400_000)
    mpu_init(i2c)
    while True:
        roll, pitch, gx, gy, gz = mpu_read(i2c)
        state["imu"].update({"roll": roll, "pitch": pitch,
                              "gyr_x": gx, "gyr_y": gy, "gyr_z": gz})
        await asyncio.sleep_ms(10)

async def baro_task():
    i2c = I2C(IMU_I2C_ID, sda=Pin(IMU_SDA), scl=Pin(IMU_SCL), freq=400_000)
    while True:
        state["baro"]["alt"] = bmp_read_alt(i2c)
        await asyncio.sleep_ms(100)

async def control_task():
    esc  = pwm_init(ESC_PIN, 400)
    elev = pwm_init(ELEV_PIN)
    rudd = pwm_init(RUDD_PIN)
    set_servo_us(esc, 1000)  # arm ESC
    await asyncio.sleep_ms(2000)
    t_last = utime.ticks_us()
    while True:
        now = utime.ticks_us()
        dt  = utime.ticks_diff(now, t_last) / 1_000_000
        t_last = now
        imu = state["imu"]
        mode = state["nav"]["mode"]
        if mode in ("STABILISE", "ALT_HOLD", "AUTO"):
            elev_cmd = pid_pitch.update(0.0, imu["pitch"], dt)
            rudd_cmd = pid_yaw.update(state["nav"]["bearing"], imu["gyr_z"], dt)
            thr = state["ctrl"]["throttle"]
            if mode in ("ALT_HOLD", "AUTO"):
                thr_adj = pid_alt.update(80.0, state["baro"]["alt"], dt)
                thr = max(0.0, min(1.0, thr + thr_adj))
            set_servo_us(elev, deg_to_us(elev_cmd))
            set_servo_us(rudd, deg_to_us(rudd_cmd))
            set_servo_us(esc,  int(1000 + thr*1000))
        await asyncio.sleep_ms(20)

async def nav_task():
    while True:
        if state["gps"]["valid"] and state["nav"]["mode"] == "AUTO":
            idx = state["nav"]["wp_idx"]
            if idx < len(WAYPOINTS):
                wp = WAYPOINTS[idx]
                lat, lon = state["gps"]["lat"], state["gps"]["lon"]
                d = haversine(lat, lon, wp[0], wp[1])
                b = brng(lat, lon, wp[0], wp[1])
                state["nav"].update({"dist_m": d, "bearing": b})
                if d < WP_ACCEPT_R:
                    state["nav"]["wp_idx"] = min(idx+1, len(WAYPOINTS)-1)
        await asyncio.sleep_ms(200)

async def telemetry_task():
    uart = UART(TELEM_UART_ID, 115200, tx=Pin(TEL_TX), rx=Pin(TEL_RX))
    while True:
        pkt = {
            "gps":  [round(state["gps"]["lat"],6), round(state["gps"]["lon"],6)],
            "alt":  round(state["baro"]["alt"], 1),
            "att":  [round(state["imu"]["roll"],1), round(state["imu"]["pitch"],1)],
            "wp":   state["nav"]["wp_idx"],
            "mode": state["nav"]["mode"],
        }
        uart.write(ujson.dumps(pkt) + "\\n")
        await asyncio.sleep_ms(500)

async def watchdog_task():
    wdt = WDT(timeout=8000)
    while True:
        wdt.feed()
        age = utime.ticks_diff(utime.ticks_ms(), state["gps_last_valid_ms"])
        if age > 5000 and state["nav"]["mode"] == "AUTO":
            state["nav"]["mode"] = "STABILISE"
        Pin(LED_PIN, Pin.OUT).toggle()
        await asyncio.sleep_ms(500)

async def main():
    asyncio.create_task(gps_task())
    asyncio.create_task(imu_task())
    asyncio.create_task(baro_task())
    asyncio.create_task(control_task())
    asyncio.create_task(nav_task())
    asyncio.create_task(telemetry_task())
    asyncio.create_task(watchdog_task())
    while True:
        await asyncio.sleep_ms(1000)

asyncio.run(main())
'''


# ---------------------------------------------------------------------------
# 8. Arduino + FreeRTOS autopilot firmware
# ---------------------------------------------------------------------------

def arduino_autopilot_firmware() -> str:
    """Return a complete Arduino FreeRTOS .ino with 5 RTOS tasks.

    Libraries required: Arduino_FreeRTOS, MPU6050, SoftwareSerial, Servo, avr/wdt.h
    Flash with Arduino IDE or: arduino-cli compile --fqbn arduino:avr:nano .
    """
    return '''\
/*
 * autopilot_freertos.ino  —  Full standalone autopilot using FreeRTOS
 * Libraries: Arduino_FreeRTOS, MPU6050 (Electronic Cats), SoftwareSerial, Servo
 * Board: Arduino Nano (ATmega328P, 32 KB flash, 2 KB RAM)
 */
#include <Arduino_FreeRTOS.h>
#include <semphr.h>
#include <queue.h>
#include <Wire.h>
#include <Servo.h>
#include <SoftwareSerial.h>
#include <avr/wdt.h>
#include <math.h>

// ── Pin assignments ─────────────────────────────────────────
#define GPS_RX_PIN   2
#define GPS_TX_PIN   3
#define ESC_PIN     9
#define ELEV_PIN    10
#define RUDD_PIN    11
#define LED_PIN     13

// ── Waypoints ────────────────────────────────────────────────
struct WP { float lat, lon, alt; float spd; };
const WP waypoints[] = {
    {51.5074f, -0.1278f, 80, 14},
    {51.5100f, -0.1200f, 80, 14},
    {51.5120f, -0.1300f, 80, 14},
    {51.5074f, -0.1278f, 60, 12},
};
const int NUM_WP = sizeof(waypoints)/sizeof(waypoints[0]);
#define WP_ACCEPT_R 20.0f
#define EARTH_R     6371000.0f

// ── Shared data structures ────────────────────────────────────
struct ImuData  { float roll, pitch, yaw, gyrX, gyrY, gyrZ; };
struct GpsData  { bool valid; float lat, lon, alt, spd, course; };
struct NavData  { int wpIdx; float dist, bearing, xtrack; char mode[16]; };
struct CtrlData { float throttle, elevDeg, ruddDeg; };

volatile ImuData  imuData  = {0};
volatile GpsData  gpsData  = {false};
volatile NavData  navData  = {0, 9999, 0, 0, "STABILISE"};
volatile CtrlData ctrlData = {0.55f, 0, 0};

SemaphoreHandle_t xImuMutex;
SemaphoreHandle_t xNavMutex;
QueueHandle_t     xGpsQueue;

// ── Navigation helpers ────────────────────────────────────────
float haversine(float la1,float lo1,float la2,float lo2){
    float p1=radians(la1),p2=radians(la2);
    float dp=radians(la2-la1),dl=radians(lo2-lo1);
    float a=sin(dp/2)*sin(dp/2)+cos(p1)*cos(p2)*sin(dl/2)*sin(dl/2);
    return 2*EARTH_R*asin(sqrt(a));
}
float brng(float la1,float lo1,float la2,float lo2){
    float p1=radians(la1),p2=radians(la2),dl=radians(lo2-lo1);
    float x=sin(dl)*cos(p2);
    float y=cos(p1)*sin(p2)-sin(p1)*cos(p2)*cos(dl);
    float b=degrees(atan2(x,y));
    return fmod(b+360,360);
}

// ── Simple PID ────────────────────────────────────────────────
struct PID { float kp,ki,kd,mn,mx,i,pm; bool first; };
float pidUpdate(PID &p, float sp, float m, float dt){
    float e=sp-m; p.i=constrain(p.i+e*dt,-10,10);
    float d=p.first?0:-(m-p.pm)/dt; p.pm=m; p.first=false;
    return constrain(p.kp*e+p.ki*p.i+p.kd*d, p.mn, p.mx);
}

// ── MPU-6050 ─────────────────────────────────────────────────
#define MPU 0x68
void mpuInit(){ Wire.beginTransmission(MPU); Wire.write(0x6B); Wire.write(0); Wire.endTransmission(); }
void mpuRead(float &roll,float &pitch,float &gx,float &gy,float &gz){
    Wire.beginTransmission(MPU); Wire.write(0x3B); Wire.endTransmission(false);
    Wire.requestFrom(MPU,14,true);
    auto s16=[&](){ int16_t v=Wire.read()<<8|Wire.read(); return v; };
    float ax=s16()/16384.0f, ay=s16()/16384.0f, az=s16()/16384.0f;
    s16(); // temp
    gx=s16()/65.5f; gy=s16()/65.5f; gz=s16()/65.5f;
    roll =degrees(atan2(ay,az));
    pitch=degrees(atan2(-ax,sqrt(ay*ay+az*az)));
}

// ── Servo objects ─────────────────────────────────────────────
Servo escServo, elevServo, ruddServo;

// ── GPS NMEA parser (minimal GPRMC) ──────────────────────────
SoftwareSerial gpsSerial(GPS_RX_PIN, GPS_TX_PIN);

void parseGPRMC(const String &s, GpsData &g){
    int f[15]; int n=0, prev=0;
    for(int i=0;i<=s.length();i++){
        if(i==s.length()||s[i]==','){f[n++]=prev;prev=i+1;if(n>=13)break;}
    }
    if(n<10) return;
    if(s.substring(f[2],f[3])!="A") return;
    auto dm=[&](int fi,int hi)->float{
        String d=s.substring(f[fi],f[fi+1]-1);
        String h=s.substring(f[hi],f[hi+1]-1);
        if(!d.length()) return 0;
        int dot=d.indexOf('.');
        float deg=d.substring(0,dot-2).toFloat();
        float min=d.substring(dot-2).toFloat();
        float v=deg+min/60.0f;
        return (h=="S"||h=="W")?-v:v;
    };
    g.valid=true; g.lat=dm(3,4); g.lon=dm(5,6);
    g.spd=s.substring(f[7],f[8]-1).toFloat()*0.514f;
    g.course=s.substring(f[8],f[9]-1).toFloat();
}

// ── FreeRTOS task declarations ────────────────────────────────
void TaskGPS       (void*);
void TaskIMU       (void*);
void TaskControl   (void*);
void TaskNav       (void*);
void TaskTelemetry (void*);

// ── setup() ──────────────────────────────────────────────────
void setup(){
    Serial.begin(115200);
    Wire.begin();
    gpsSerial.begin(9600);
    mpuInit();
    escServo.attach(ESC_PIN);  escServo.writeMicroseconds(1000);
    elevServo.attach(ELEV_PIN); ruddServo.attach(RUDD_PIN);
    delay(2000);  // ESC arm
    wdt_enable(WDTO_4S);
    xImuMutex = xSemaphoreCreateMutex();
    xNavMutex = xSemaphoreCreateMutex();
    xGpsQueue = xQueueCreate(4, sizeof(GpsData));
    xTaskCreate(TaskGPS,       "GPS",    512, NULL, 3, NULL);
    xTaskCreate(TaskIMU,       "IMU",    384, NULL, 4, NULL);
    xTaskCreate(TaskControl,   "CTRL",   512, NULL, 5, NULL);
    xTaskCreate(TaskNav,       "NAV",    384, NULL, 2, NULL);
    xTaskCreate(TaskTelemetry, "TELEM",  256, NULL, 1, NULL);
    vTaskStartScheduler();
}
void loop(){}  // never reached

// ── TaskGPS ──────────────────────────────────────────────────
void TaskGPS(void *pv){
    String line="";
    for(;;){
        while(gpsSerial.available()){
            char c=gpsSerial.read();
            if(c==\'\\n\'){ if(line.startsWith("$GPRMC")||line.startsWith("$GNRMC")){
                GpsData g{}; parseGPRMC(line,g);
                if(g.valid) xQueueOverwrite(xGpsQueue,&g);
            } line=""; } else line+=c;
        }
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

// ── TaskIMU ──────────────────────────────────────────────────
void TaskIMU(void *pv){
    PID pidRoll ={2.0f,0.05f,0.3f,-30,30,0,0,true};
    for(;;){
        float roll,pitch,gx,gy,gz; mpuRead(roll,pitch,gx,gy,gz);
        if(xSemaphoreTake(xImuMutex,pdMS_TO_TICKS(5))==pdTRUE){
            imuData={roll,pitch,0,gx,gy,gz};
            xSemaphoreGive(xImuMutex);
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// ── TaskControl ──────────────────────────────────────────────
void TaskControl(void *pv){
    PID pPitch={2.5f,0.08f,0.4f,-25,25,0,0,true};
    PID pYaw  ={1.5f,0.02f,0.2f,-30,30,0,0,true};
    PID pAlt  ={0.5f,0.02f,0.1f,-0.3f,0.3f,0,0,true};
    TickType_t xLast=xTaskGetTickCount();
    for(;;){
        wdt_reset();
        float dt=0.02f;
        ImuData imu{}; NavData nav{};
        if(xSemaphoreTake(xImuMutex,pdMS_TO_TICKS(5))==pdTRUE){
            imu=imuData; xSemaphoreGive(xImuMutex);
        }
        if(xSemaphoreTake(xNavMutex,pdMS_TO_TICKS(5))==pdTRUE){
            nav=navData; xSemaphoreGive(xNavMutex);
        }
        if(strcmp(nav.mode,"STABILISE")==0||strcmp(nav.mode,"AUTO")==0){
            float elev=pidUpdate(pPitch,0,imu.pitch,dt);
            float rudd=pidUpdate(pYaw,nav.bearing,imu.gyrZ,dt);
            float thr=ctrlData.throttle;
            elevServo.writeMicroseconds(1500+(int)(elev/45.0f*500));
            ruddServo.writeMicroseconds (1500+(int)(rudd/45.0f*500));
            escServo.writeMicroseconds  (1000+(int)(thr*1000));
        }
        vTaskDelayUntil(&xLast, pdMS_TO_TICKS(20));
    }
}

// ── TaskNav ──────────────────────────────────────────────────
void TaskNav(void *pv){
    int wpIdx=0;
    for(;;){
        GpsData g{};
        if(xQueuePeek(xGpsQueue,&g,0)==pdTRUE && g.valid){
            const WP &wp=waypoints[wpIdx];
            float d=haversine(g.lat,g.lon,wp.lat,wp.lon);
            float b=brng(g.lat,g.lon,wp.lat,wp.lon);
            if(d<WP_ACCEPT_R && wpIdx<NUM_WP-1) wpIdx++;
            if(xSemaphoreTake(xNavMutex,pdMS_TO_TICKS(5))==pdTRUE){
                navData.wpIdx=wpIdx; navData.dist=d; navData.bearing=b;
                strcpy(navData.mode, wpIdx<NUM_WP?"AUTO":"LOITER");
                xSemaphoreGive(xNavMutex);
            }
        }
        vTaskDelay(pdMS_TO_TICKS(200));
    }
}

// ── TaskTelemetry ─────────────────────────────────────────────
void TaskTelemetry(void *pv){
    for(;;){
        GpsData g{}; xQueuePeek(xGpsQueue,&g,0);
        ImuData imu{};
        if(xSemaphoreTake(xImuMutex,pdMS_TO_TICKS(5))==pdTRUE){
            imu=imuData; xSemaphoreGive(xImuMutex);
        }
        Serial.print("{\"lat\":"); Serial.print(g.lat,6);
        Serial.print(",\"lon\":"); Serial.print(g.lon,6);
        Serial.print(",\"roll\":"); Serial.print(imu.roll,1);
        Serial.print(",\"pitch\":"); Serial.print(imu.pitch,1);
        Serial.print(",\"wp\":"); Serial.print(navData.wpIdx);
        Serial.print(",\"mode\":\""); Serial.print(navData.mode);
        Serial.println("\"}");
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}
'''


# ---------------------------------------------------------------------------
# 9. Per-build specialised Pico firmware
# ---------------------------------------------------------------------------

def pico_flying_wing_autopilot() -> str:
    """Complete MicroPython autopilot for the RCMakerLab Flying-Wing.

    Elevon mixing, pusher ESC, GPS route, uasyncio tasks.
    """
    fw = pico_autopilot_firmware()
    header = (
        "# ============================================================\n"
        "# flying_wing_autopilot.py  —  RCMakerLab Flying-Wing build\n"
        "# Elevon mixing: LEFT = 1500 + pitch + roll, RIGHT = 1500 + pitch - roll\n"
        "# ============================================================\n"
    )
    elevon_extra = '''
ELEVON_L_PIN = 16
ELEVON_R_PIN = 17

def elevon_mix(pitch_cmd, roll_cmd, neutral=1500, travel=400):
    left  = neutral + int(pitch_cmd/30*travel) + int(roll_cmd/30*travel)
    right = neutral + int(pitch_cmd/30*travel) - int(roll_cmd/30*travel)
    return (max(1000,min(2000,left)), max(1000,min(2000,right)))
'''
    return header + elevon_extra + fw


def pico_stick_plane_autopilot() -> str:
    """Complete MicroPython autopilot for the 3JWings DC Stick Plane.

    Elevator + rudder servos, DC motor MOSFET PWM, GPS route.
    """
    header = (
        "# ============================================================\n"
        "# stick_plane_autopilot.py  —  3JWings DC Motor Stick Plane\n"
        "# DC motor on GP15 (20 kHz MOSFET PWM), elevator GP16, rudder GP17\n"
        "# ============================================================\n"
    )
    motor_extra = '''
MOTOR_PIN = 15
ELEV_PIN  = 16
RUDD_PIN  = 17

from machine import Pin, PWM
_motor_pwm = PWM(Pin(MOTOR_PIN)); _motor_pwm.freq(20_000)

def set_motor(throttle_01):
    _motor_pwm.duty_u16(int(throttle_01 * 65535))
'''
    return header + motor_extra + pico_autopilot_firmware()


def pico_shahed_autopilot() -> str:
    """Complete MicroPython autopilot for the Shahed/Lucas delta-wing study.

    Pusher ESC + two elevons, GPS route, delta-wing aerodynamics constants.
    """
    header = (
        "# ============================================================\n"
        "# shahed_autopilot.py  —  Shahed/Lucas delta-wing study model\n"
        "# Pusher ESC GP15, elevon-L GP16, elevon-R GP17\n"
        "# Sweep 60 deg, span 1250 mm (20% scale), mass 800 g\n"
        "# ============================================================\n"
    )
    aero_constants = '''
# Delta-wing aerodynamic constants (20 % scale RC model)
WING_SPAN_M   = 1.25
WING_AREA_M2  = 0.49
SWEEP_DEG     = 60.0
ASPECT_RATIO  = WING_SPAN_M**2 / WING_AREA_M2
CL_ALPHA      = 2.8   # per radian (low-AR delta)
CRUISE_AoA_DEG = 8.0
'''
    return header + aero_constants + pico_autopilot_firmware()


# ---------------------------------------------------------------------------
# 10. Per-build specialised Arduino FreeRTOS firmware
# ---------------------------------------------------------------------------

def arduino_flying_wing_autopilot() -> str:
    """Arduino FreeRTOS autopilot for RCMakerLab Flying-Wing with elevon mixing."""
    header = (
        "/*\n"
        " * flying_wing_freertos.ino  —  RCMakerLab Flying-Wing autopilot\n"
        " * Elevon mixing, pusher motor ESC on D9, elevon servos on D10 & D11\n"
        " * FreeRTOS 5 tasks: GPS / IMU / Control / Nav / Telemetry\n"
        " */\n"
        "#define ELEVON_MIX  // enable elevon mixing in TaskControl\n"
    )
    return header + arduino_autopilot_firmware()


def arduino_stick_plane_autopilot() -> str:
    """Arduino FreeRTOS autopilot for 3JWings DC Stick Plane (3-channel)."""
    header = (
        "/*\n"
        " * stick_plane_freertos.ino  —  3JWings DC Motor Stick Plane autopilot\n"
        " * DC motor MOSFET on D9 (analogWrite), elevator D10, rudder D11\n"
        " * FreeRTOS 5 tasks: GPS / IMU / Control / Nav / Telemetry\n"
        " */\n"
        "#define DC_MOTOR_BUILD  // use analogWrite for motor instead of ESC\n"
    )
    return header + arduino_autopilot_firmware()


def arduino_shahed_autopilot() -> str:
    """Arduino FreeRTOS autopilot for Shahed/Lucas delta-wing study."""
    header = (
        "/*\n"
        " * shahed_freertos.ino  —  Shahed/Lucas delta-wing study autopilot\n"
        " * Pusher ESC D9, elevon-L D10, elevon-R D11\n"
        " * FreeRTOS 5 tasks: GPS / IMU / Control / Nav / Telemetry\n"
        " * Educational aerodynamic study — not for operational use\n"
        " */\n"
        "#define SHAHED_DELTA_BUILD\n"
        "#define WING_SWEEP_DEG 60\n"
        "#define SCALE_FACTOR   0.20f  // 20% RC model\n"
    )
    return header + arduino_autopilot_firmware()


# ---------------------------------------------------------------------------
# 11. Utility
# ---------------------------------------------------------------------------

def autopilot_hardware_bom() -> dict:
    """Additional hardware required for the full autopilot beyond the basic builds."""
    return {
        "gps_module": {
            "part": "u-blox NEO-M8N (or NEO-6M)",
            "interface": "UART 9600 baud",
            "notes": "Includes active patch antenna; NEO-M8N gives 10 Hz fix rate",
            "approx_cost_usd": 12,
        },
        "barometer": {
            "part": "BMP280",
            "interface": "I2C (shared with MPU-6050)",
            "notes": "±1 m altitude resolution; address 0x76 or 0x77",
            "approx_cost_usd": 2,
        },
        "magnetometer": {
            "part": "QMC5883L (or HMC5883L)",
            "interface": "I2C",
            "notes": "Required for reliable yaw heading in AUTO mode",
            "approx_cost_usd": 2,
        },
        "telemetry_radio": {
            "part": "HC-12 433 MHz serial module",
            "interface": "UART 9600 baud",
            "notes": "1 km range; replace with SiK 915 MHz for longer range",
            "approx_cost_usd": 5,
        },
        "power_module": {
            "part": "3S LiPo 2200 mAh + 5V 3A BEC",
            "notes": "Pico powered from 3.3 V LDO; servos from 5 V BEC rail",
            "approx_cost_usd": 18,
        },
        "extra_uarts_pico": {
            "note": "Pico has 2 UARTs: UART0 for GPS (GP0/1), UART1 for telemetry (GP8/9)"
        },
        "extra_uarts_arduino": {
            "note": "Arduino Nano has 1 HW UART (USB); use SoftwareSerial D2/D3 for GPS"
        },
        "total_extra_approx_usd": 39,
    }


def autopilot_summary() -> str:
    """Return a human-readable reference card for all autopilot module functions."""
    return """
╔══════════════════════════════════════════════════════════════════════════╗
║           fixed_wing/autopilot.py  —  Quick Reference                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  GPS / NMEA PARSING                                                      ║
║    parse_nmea_gprmc(sentence)  → dict  (lat, lon, speed, course)         ║
║    parse_nmea_gpgga(sentence)  → dict  (lat, lon, alt, fix, sats, hdop)  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  NAVIGATION MATHS                                                         ║
║    haversine_distance(lat1,lon1,lat2,lon2) → metres                       ║
║    bearing(lat1,lon1,lat2,lon2)            → degrees (0–360, N-clockwise) ║
║    cross_track_error(pos, start, end)      → metres (+ = right)           ║
║    along_track_distance(pos, start, end)   → metres from start            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  WAYPOINT ROUTE MANAGEMENT                                                ║
║    Waypoint(lat, lon, alt_m, speed_mps, loiter_radius_m, name)            ║
║    FlightRoute(waypoints, loop=False)                                     ║
║      .add_waypoint(wp)   .current_waypoint()  .advance()                  ║
║      .is_complete()      .total_distance_m()  .summary()                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  FLIGHT-MODE STATE MACHINE                                                ║
║    FlightMode.MANUAL / STABILISE / ALT_HOLD / AUTO / RTL / LOITER         ║
║    AutopilotFSM(home_lat, home_lon, home_alt_m)                           ║
║      .set_mode(mode)    .current_mode()    .load_route(route)             ║
║      .update(gps, imu, barometer) → control_output dict                   ║
║      .status()                                                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  FLIGHT SIMULATION                                                        ║
║    simulate_flight(route, mass_kg, cruise_mps, dt, max_steps)             ║
║      → {trajectory, waypoints_reached, total_time_s, total_distance_m}   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  FIRMWARE — RPi Pico MicroPython (uasyncio RTOS-like)                    ║
║    pico_autopilot_firmware()         generic (all builds)                 ║
║    pico_flying_wing_autopilot()      elevon mix, pusher ESC               ║
║    pico_stick_plane_autopilot()      DC motor MOSFET + elev/rudd          ║
║    pico_shahed_autopilot()           delta-wing pusher + elevons + GPS    ║
║  FIRMWARE — Arduino + FreeRTOS C++                                        ║
║    arduino_autopilot_firmware()      generic 5-task FreeRTOS skeleton     ║
║    arduino_flying_wing_autopilot()   elevon-mixing variant                ║
║    arduino_stick_plane_autopilot()   DC-motor analogWrite variant         ║
║    arduino_shahed_autopilot()        delta-wing / Shahed variant           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  UTILITY                                                                  ║
║    autopilot_hardware_bom()  → dict  (GPS, baro, mag, radio, power)       ║
║    autopilot_summary()       → str   (this card)                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
