"""
esp_blast_flight_controller.py — ESP-BLAST Rocket Drone firmware skeleton
Project:          ESP-BLAST (Smallest ESP32 Brushless Rocket Drone)
Source:           https://www.instructables.com/Build-the-Smallest-ESP32-Brushless-Rocket-Drone-ES/
Language:         MicroPython (ESP32-S3)
Firmware base:    ESP-FC (https://github.com/rtlopez/esp-fc) — use for production flights
                  This file is an educational reference skeleton.

Hardware wiring (ESP32-S3 custom PCB):
  GPIO12 → Motor 1 ESC  (Front-Right, CW)
  GPIO13 → Motor 2 ESC  (Rear-Left,   CW)
  GPIO14 → Motor 3 ESC  (Front-Left,  CCW)
  GPIO15 → Motor 4 ESC  (Rear-Right,  CCW)
  GPIO16 → Camera tilt servo
  GPIO17 → Buzzer (active, 5 V)
  I2C (SDA=GPIO4, SCL=GPIO5) → IMU (MPU-6500) + Barometer (BMP280)
  UART1 (TX=GPIO8, RX=GPIO9) → GPS (NMEA, 9600 baud)
  UART2 (TX=GPIO6, RX=GPIO7) → OSD board (MSP telemetry)
  Wi-Fi (ESP-NOW) → RC transmitter (peer-to-peer, no router required)

Flash instructions:
  1. Install MicroPython for ESP32-S3:
     https://micropython.org/download/ESP32_GENERIC_S3/
  2. Copy this file to the board as main.py (use Thonny or ampy)
  3. Power on — it runs automatically

NOTE: For real flights use ESP-FC (PlatformIO / C++) — this skeleton is
      provided for learning purposes only.
"""

import asyncio
import struct
import time
import network
import espnow
from machine import I2C, Pin, PWM, UART

# ---------------------------------------------------------------------------
# Configuration — adjust to match your build
# ---------------------------------------------------------------------------

# Motor GPIO pins (quad-X layout, top view)
MOTOR_PINS = [12, 13, 14, 15]   # FR(CW), RL(CW), FL(CCW), RR(CCW)
TILT_PIN   = 16                  # Camera tilt servo
BUZZER_PIN = 17

# I2C for IMU + barometer
I2C_SDA = 4
I2C_SCL = 5

# UART1 — GPS
GPS_TX = 8
GPS_RX = 9
GPS_BAUD = 9600

# UART2 — OSD (MSP)
OSD_TX = 6
OSD_RX = 7
OSD_BAUD = 115200

# PWM parameters (DSHOT would require bitbanging — this uses analogue PWM)
PWM_FREQ_HZ = 500               # Higher frequency for better ESC response
PWM_MIN_US  = 1_000             # Minimum throttle (arm/idle)
PWM_MAX_US  = 2_000             # Maximum throttle
PWM_MID_US  = 1_500             # Centre for servos

# Attitude loop rate
LOOP_HZ = 1_000                 # 1 kHz inner loop (reduce if unstable)

# ESP-NOW peer MAC address of the RC transmitter (replace with your TX MAC)
TX_PEER_MAC = b'\xFF\xFF\xFF\xFF\xFF\xFF'  # broadcast placeholder

# PID gains — tune these values on your specific airframe
PID_ROLL  = dict(kp=0.65, ki=0.02, kd=0.18)
PID_PITCH = dict(kp=0.70, ki=0.02, kd=0.20)
PID_YAW   = dict(kp=0.55, ki=0.01, kd=0.00)


# ---------------------------------------------------------------------------
# PWM helpers
# ---------------------------------------------------------------------------

def init_motor_pwm(pin_number: int) -> PWM:
    """Initialise a PWM output for an ESC (500 Hz, armed at 1000 µs)."""
    pwm = PWM(Pin(pin_number, Pin.OUT), freq=PWM_FREQ_HZ)
    _set_us(pwm, PWM_MIN_US)
    return pwm


def _set_us(pwm: PWM, pulse_us: int) -> None:
    """Set PWM pulse width in microseconds (clamped to valid range)."""
    pulse_us = max(PWM_MIN_US, min(PWM_MAX_US, pulse_us))
    # duty_ns available on ESP32 MicroPython builds
    pwm.duty_ns(pulse_us * 1_000)


# ---------------------------------------------------------------------------
# PID controller
# ---------------------------------------------------------------------------

class PID:
    """Simple PID with integral anti-windup."""

    def __init__(self, kp: float, ki: float, kd: float,
                 i_limit: float = 250.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._i_limit = i_limit
        self._integral = 0.0
        self._last_err  = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._last_err  = 0.0

    def update(self, setpoint: float, measured: float, dt: float) -> float:
        if dt <= 0:
            return 0.0
        err = setpoint - measured
        self._integral = max(-self._i_limit,
                             min(self._i_limit,
                                 self._integral + err * dt))
        derivative = (err - self._last_err) / dt
        self._last_err = err
        return self.kp * err + self.ki * self._integral + self.kd * derivative


# ---------------------------------------------------------------------------
# Minimal MPU-6500 driver (SPI or I2C — I2C version shown)
# ---------------------------------------------------------------------------

_MPU_ADDR      = 0x68
_REG_PWR_MGMT  = 0x6B
_REG_GYRO_CFG  = 0x1B
_REG_ACCEL_CFG = 0x1C
_REG_ACCEL_OUT = 0x3B
_REG_GYRO_OUT  = 0x43

class MPU6500:
    """Minimal MPU-6500 I2C driver (gyro + accelerometer)."""

    GYRO_SCALE  = 131.0   # LSB / (°/s) at ±250 °/s
    ACCEL_SCALE = 16384.0 # LSB / g  at ±2 g

    def __init__(self, i2c: I2C, addr: int = _MPU_ADDR) -> None:
        self._i2c  = i2c
        self._addr = addr
        # Wake from sleep
        self._write(_REG_PWR_MGMT, 0x01)     # clock source = gyro X
        # Configure gyro ±250 °/s, accel ±2 g
        self._write(_REG_GYRO_CFG,  0x00)
        self._write(_REG_ACCEL_CFG, 0x00)

    def _write(self, reg: int, val: int) -> None:
        self._i2c.writeto_mem(self._addr, reg, bytes([val]))

    def _read_word(self, reg: int) -> int:
        raw = self._i2c.readfrom_mem(self._addr, reg, 2)
        val = struct.unpack('>h', raw)[0]
        return val

    def read_gyro_dps(self) -> tuple:
        """Return (roll_rate, pitch_rate, yaw_rate) in °/s."""
        gx = self._read_word(_REG_GYRO_OUT)
        gy = self._read_word(_REG_GYRO_OUT + 2)
        gz = self._read_word(_REG_GYRO_OUT + 4)
        s  = self.GYRO_SCALE
        return gx / s, gy / s, gz / s

    def read_accel_g(self) -> tuple:
        """Return (ax, ay, az) in g."""
        ax = self._read_word(_REG_ACCEL_OUT)
        ay = self._read_word(_REG_ACCEL_OUT + 2)
        az = self._read_word(_REG_ACCEL_OUT + 4)
        s  = self.ACCEL_SCALE
        return ax / s, ay / s, az / s


# ---------------------------------------------------------------------------
# Motor mixer — quad-X
# ---------------------------------------------------------------------------

def mix_quad_x(throttle: float, roll: float,
               pitch: float, yaw: float) -> list:
    """
    Quad-X mixer.  All inputs in the range [0, 1000] (throttle) or
    [-500, 500] (roll / pitch / yaw).
    Returns list of 4 motor outputs clamped to [0, 1000].

    Motor numbering (top view):
        Front-Right (M1 CW)    Front-Left (M3 CCW)
        Rear-Left   (M2 CW)    Rear-Right (M4 CCW)
    """
    m1 = throttle + roll - pitch - yaw   # FR  CW
    m2 = throttle - roll + pitch - yaw   # RL  CW
    m3 = throttle + roll + pitch + yaw   # FL  CCW
    m4 = throttle - roll - pitch + yaw   # RR  CCW
    return [max(0, min(1000, int(v))) for v in [m1, m2, m3, m4]]


# ---------------------------------------------------------------------------
# Arming state machine
# ---------------------------------------------------------------------------

class ArmFSM:
    DISARMED = 'DISARMED'
    ARMING   = 'ARMING'
    ARMED    = 'ARMED'
    FAILSAFE = 'FAILSAFE'

    _ARM_HOLD_MS = 2_000   # time to hold low-throttle + yaw-right to arm

    def __init__(self) -> None:
        self.state = self.DISARMED
        self._start = 0

    def request_arm(self) -> None:
        if self.state == self.DISARMED:
            self.state  = self.ARMING
            self._start = time.ticks_ms()
            print('[ARM] Arming…')

    def disarm(self) -> None:
        self.state = self.DISARMED
        print('[ARM] Disarmed.')

    def failsafe(self) -> None:
        self.state = self.FAILSAFE
        print('[ARM] FAILSAFE activated.')

    def tick(self) -> None:
        if self.state == self.ARMING:
            elapsed = time.ticks_diff(time.ticks_ms(), self._start)
            if elapsed >= self._ARM_HOLD_MS:
                self.state = self.ARMED
                print('[ARM] ARMED.')

    @property
    def is_armed(self) -> bool:
        return self.state == self.ARMED


# ---------------------------------------------------------------------------
# ESP-NOW RC receiver
# ---------------------------------------------------------------------------

class EspNowReceiver:
    """
    Receive RC control packets from the transmitter via ESP-NOW.

    Packet layout (8 bytes, little-endian):
      uint16 throttle  (1000–2000 µs)
      int16  roll      (-500 to +500)
      int16  pitch     (-500 to +500)
      int16  yaw       (-500 to +500)
    """
    _FMT = '<Hhhh'
    _PKT_SIZE = struct.calcsize(_FMT)

    def __init__(self) -> None:
        sta = network.WLAN(network.STA_IF)
        sta.active(True)
        self._en = espnow.ESPNow()
        self._en.active(True)
        try:
            self._en.add_peer(TX_PEER_MAC)
        except Exception:
            pass  # peer may already be registered

        # Default: throttle idle, all axes centred
        self.throttle: float = 0.0
        self.roll:     float = 0.0
        self.pitch:    float = 0.0
        self.yaw:      float = 0.0
        self.last_rx_ms: int  = time.ticks_ms()

    def poll(self) -> bool:
        """Non-blocking poll; returns True if a new packet was received."""
        try:
            host, msg = self._en.irecv(0)
        except Exception:
            return False

        if msg is None or len(msg) != self._PKT_SIZE:
            return False

        thr, rol, pit, yaw = struct.unpack(self._FMT, msg)
        self.throttle    = float(thr - 1000)           # map 1000–2000 → 0–1000
        # Scale stick deflection (-500..+500) to physical units:
        #   roll / pitch → target angle in degrees  (±45 °)
        #   yaw          → target yaw rate in °/s   (±200 °/s)
        self.roll        = float(rol) * (45.0 / 500.0)
        self.pitch       = float(pit) * (45.0 / 500.0)
        self.yaw         = float(yaw) * (200.0 / 500.0)
        self.last_rx_ms  = time.ticks_ms()
        return True

    @property
    def link_age_ms(self) -> int:
        """Milliseconds since the last valid RC packet."""
        return time.ticks_diff(time.ticks_ms(), self.last_rx_ms)


# ---------------------------------------------------------------------------
# Complementary filter for roll/pitch estimation from gyro + accel
# ---------------------------------------------------------------------------

def complementary_filter(roll: float, pitch: float,
                         gx: float, gy: float,
                         ax: float, ay: float, az: float,
                         dt: float,
                         alpha: float = 0.98) -> tuple:
    """
    Fuse gyroscope integration with accelerometer levelling.
    alpha = 0.98 means 98% gyro, 2% accel per step.
    Returns updated (roll, pitch) in degrees.
    """
    import math
    accel_roll  = math.atan2(ay, az) * 57.2958
    accel_pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az)) * 57.2958

    roll  = alpha * (roll  + gx * dt) + (1 - alpha) * accel_roll
    pitch = alpha * (pitch + gy * dt) + (1 - alpha) * accel_pitch
    return roll, pitch


# ---------------------------------------------------------------------------
# Attitude (inner) loop coroutine
# ---------------------------------------------------------------------------

async def attitude_loop(imu: MPU6500, motors: list,
                        arm: ArmFSM, rc: EspNowReceiver,
                        pids: dict) -> None:
    """High-rate attitude PID control loop (target: LOOP_HZ)."""
    dt = 1.0 / LOOP_HZ
    roll_est = pitch_est = 0.0
    t_last = time.ticks_us()

    while True:
        now = time.ticks_us()
        actual_dt = time.ticks_diff(now, t_last) * 1e-6
        t_last = now

        gx, gy, gz = imu.read_gyro_dps()
        ax, ay, az  = imu.read_accel_g()

        roll_est, pitch_est = complementary_filter(
            roll_est, pitch_est, gx, gy, ax, ay, az, actual_dt)

        arm.tick()
        rc.poll()

        if arm.is_armed:
            # Check for RC link loss → failsafe
            if rc.link_age_ms > 1_000:
                arm.failsafe()

            # roll / pitch: angle mode  (setpoint = target angle °, measured = estimated angle °)
            # yaw:          rate  mode  (setpoint = target yaw rate °/s, measured = gyro gz °/s)
            roll_out  = pids['roll'].update(rc.roll,   roll_est,  actual_dt)
            pitch_out = pids['pitch'].update(rc.pitch, pitch_est, actual_dt)
            yaw_out   = pids['yaw'].update(rc.yaw,    gz,        actual_dt)

            motor_vals = mix_quad_x(
                rc.throttle, roll_out, pitch_out, yaw_out)
            for i, pwm in enumerate(motors):
                _set_us(pwm, PWM_MIN_US + motor_vals[i])
        else:
            # All motors to minimum (disarmed / failsafe)
            for pwm in motors:
                _set_us(pwm, PWM_MIN_US)

        # Yield to other coroutines
        await asyncio.sleep_ms(max(1, int(1000 / LOOP_HZ)))


# ---------------------------------------------------------------------------
# MSP (MultiWii Serial Protocol) — minimal encoder for OSD telemetry
# ---------------------------------------------------------------------------

_MSP_HEADER = b'$M<'

def _msp_encode(cmd: int, payload: bytes) -> bytes:
    """Encode a minimal MSP v1 frame."""
    data = bytes([len(payload), cmd]) + payload
    crc  = 0
    for b in data:
        crc ^= b
    return _MSP_HEADER + data + bytes([crc])

def msp_attitude(roll_deg: float, pitch_deg: float,
                 yaw_deg: float) -> bytes:
    """MSP_ATTITUDE (cmd=108) — sent to OSD board."""
    payload = struct.pack('<hhH',
        int(roll_deg  * 10),    # decidegrees
        int(pitch_deg * 10),
        int(yaw_deg % 360))
    return _msp_encode(108, payload)


async def osd_loop(uart_osd: UART, imu: MPU6500) -> None:
    """10 Hz OSD telemetry output (MSP_ATTITUDE)."""
    roll = pitch = yaw = 0.0
    while True:
        frame = msp_attitude(roll, pitch, yaw)
        uart_osd.write(frame)
        await asyncio.sleep_ms(100)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    print('[BOOT] ESP-BLAST firmware starting…')

    # --- Hardware initialisation ---
    i2c      = I2C(0, scl=Pin(I2C_SCL), sda=Pin(I2C_SDA), freq=400_000)
    uart_osd = UART(2, baudrate=OSD_BAUD,
                    tx=Pin(OSD_TX), rx=Pin(OSD_RX))
    buzzer   = Pin(BUZZER_PIN, Pin.OUT, value=0)
    tilt_pwm = PWM(Pin(TILT_PIN), freq=50)
    _set_us(tilt_pwm, PWM_MID_US)  # camera centred

    imu    = MPU6500(i2c)
    motors = [init_motor_pwm(p) for p in MOTOR_PINS]
    rc     = EspNowReceiver()
    arm    = ArmFSM()

    pids = {
        'roll':  PID(**PID_ROLL),
        'pitch': PID(**PID_PITCH),
        'yaw':   PID(**PID_YAW),
    }

    # --- ESC arming sequence: hold 1000 µs for 3 s ---
    print('[BOOT] ESC arming sequence…')
    for _ in range(3):
        buzzer.value(1)
        await asyncio.sleep_ms(150)
        buzzer.value(0)
        await asyncio.sleep_ms(150)
    await asyncio.sleep_ms(2_000)
    print('[BOOT] ESCs armed — auto-arming flight controller…')
    arm.request_arm()

    # --- Launch coroutines ---
    await asyncio.gather(
        attitude_loop(imu, motors, arm, rc, pids),
        osd_loop(uart_osd, imu),
    )


asyncio.run(main())
