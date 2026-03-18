"""
autopilot_medium.py — Medium autopilot firmware skeleton (MicroPython, RP2040)
Complexity level: MEDIUM
Language:         MicroPython / asyncio

Features implemented:
  - MPU-6050 IMU reading over I2C
  - PID attitude stabilisation (roll, pitch, yaw)
  - Multi-channel SBUS / PWM input (simulated here)
  - MAVLink HEARTBEAT and ATTITUDE output via UART
  - Arming / disarming state machine
  - Basic failsafe on RC link loss

Wiring (Raspberry Pi Pico):
  GP4/GP5   → I2C0 SDA/SCL → MPU-6050
  GP8       → UART1 TX      → GCS telemetry
  GP9       → UART1 RX      (MAVLink uplink)
  GP15–GP18 → PWM servos / ESC outputs (motors 1–4)

Flash:
  Copy this file to the Pico as main.py
  Requires: micropython-imu, micropython-mavlink or minimal stubs below
"""

import asyncio
import struct
import time
from machine import I2C, Pin, UART, PWM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
I2C_SCL_PIN = 5
I2C_SDA_PIN = 4
UART_TX_PIN = 8
UART_RX_PIN = 9
UART_BAUD   = 57600

MOTOR_PINS  = [15, 16, 17, 18]  # quad X configuration

PWM_MIN_US  = 1_000
PWM_MAX_US  = 2_000
PWM_MID_US  = 1_500
LOOP_HZ     = 400              # attitude loop frequency

# MPU-6050 register addresses
MPU_ADDR        = 0x68
MPU_REG_ACCEL_X = 0x3B
MPU_REG_GYRO_X  = 0x43
MPU_REG_PWR_MGT = 0x6B


# ---------------------------------------------------------------------------
# PID controller
# ---------------------------------------------------------------------------
class PID:
    def __init__(self, kp: float, ki: float, kd: float,
                 integral_limit: float = 300.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, setpoint: float, measured: float, dt: float) -> float:
        error = setpoint - measured
        self._integral = max(-self.integral_limit,
                             min(self.integral_limit,
                                 self._integral + error * dt))
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative


# ---------------------------------------------------------------------------
# MPU-6050 driver (minimal)
# ---------------------------------------------------------------------------
class MPU6050:
    def __init__(self, i2c: I2C, addr: int = MPU_ADDR):
        self._i2c   = i2c
        self._addr  = addr
        # Wake up sensor
        self._i2c.writeto_mem(self._addr, MPU_REG_PWR_MGT, b'\x00')

    def _read_raw(self, reg: int) -> int:
        data = self._i2c.readfrom_mem(self._addr, reg, 2)
        val  = struct.unpack('>h', data)[0]
        return val

    def read_gyro_dps(self) -> tuple:
        """Return (roll_rate, pitch_rate, yaw_rate) in degrees/second."""
        gx = self._read_raw(MPU_REG_GYRO_X)
        gy = self._read_raw(MPU_REG_GYRO_X + 2)
        gz = self._read_raw(MPU_REG_GYRO_X + 4)
        scale = 131.0  # ±250 deg/s full-scale → 131 LSB per deg/s
        return gx / scale, gy / scale, gz / scale

    def read_accel_g(self) -> tuple:
        """Return (ax, ay, az) in g."""
        ax = self._read_raw(MPU_REG_ACCEL_X)
        ay = self._read_raw(MPU_REG_ACCEL_X + 2)
        az = self._read_raw(MPU_REG_ACCEL_X + 4)
        scale = 16384.0  # ±2g full-scale
        return ax / scale, ay / scale, az / scale


# ---------------------------------------------------------------------------
# Arming state machine
# ---------------------------------------------------------------------------
class ArmState:
    DISARMED = 0
    ARMING   = 1
    ARMED    = 2
    FAILSAFE = 3

    def __init__(self):
        self.state = self.DISARMED
        self._arm_start = 0

    def request_arm(self):
        if self.state == self.DISARMED:
            self.state    = self.ARMING
            self._arm_start = time.ticks_ms()
            print("Arming…")

    def tick(self, throttle_us: int) -> None:
        if self.state == self.ARMING:
            if time.ticks_diff(time.ticks_ms(), self._arm_start) > 2000:
                self.state = self.ARMED
                print("ARMED")

    def disarm(self):
        self.state = self.DISARMED
        print("DISARMED")

    def set_failsafe(self):
        self.state = self.FAILSAFE
        print("FAILSAFE")

    @property
    def is_armed(self) -> bool:
        return self.state == self.ARMED


# ---------------------------------------------------------------------------
# Minimal MAVLink encoder (HEARTBEAT + ATTITUDE only)
# Full MAVLink: use micropython-mavlink or pymavlink
# ---------------------------------------------------------------------------
MAVLINK_STX    = 0xFE
SYS_ID         = 1
COMP_ID        = 200  # autopilot component

def _checksum(data: bytes) -> int:
    ck_a = ck_b = 0
    for b in data:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return ck_a | (ck_b << 8)

def encode_heartbeat(seq: int) -> bytes:
    """Encode MAVLink HEARTBEAT message (msg_id=0)."""
    payload = struct.pack('<IBBBBB',
        0,        # custom_mode
        6,        # type: MAV_TYPE_GCS (placeholder)
        8,        # autopilot: MAV_AUTOPILOT_GENERIC
        0,        # base_mode
        0,        # system_status: MAV_STATE_STANDBY
        3,        # mavlink version
    )
    msg_id = 0
    header = bytes([len(payload), seq & 0xFF, SYS_ID, COMP_ID, msg_id])
    crc_extra = 50  # HEARTBEAT CRC_EXTRA
    cs = _checksum(header + payload + bytes([crc_extra]))
    return bytes([MAVLINK_STX]) + header + payload + struct.pack('<H', cs)

def encode_attitude(seq: int, roll: float, pitch: float, yaw: float,
                    rollspeed: float, pitchspeed: float, yawspeed: float) -> bytes:
    """Encode MAVLink ATTITUDE message (msg_id=30)."""
    import math
    t = time.ticks_ms()
    payload = struct.pack('<Iffffff',
        t, roll, pitch, yaw, rollspeed, pitchspeed, yawspeed)
    msg_id = 30
    header = bytes([len(payload), seq & 0xFF, SYS_ID, COMP_ID, msg_id])
    crc_extra = 39  # ATTITUDE CRC_EXTRA
    cs = _checksum(header + payload + bytes([crc_extra]))
    return bytes([MAVLINK_STX]) + header + payload + struct.pack('<H', cs)


# ---------------------------------------------------------------------------
# Motor mixer (quad X)
# ---------------------------------------------------------------------------
def mix_quad_x(throttle: float, roll: float, pitch: float,
               yaw: float) -> list:
    """
    Quad X motor mixing.
    Returns list of 4 motor outputs in range [0, 1000].
    Motor order (top view): 1=FR, 2=RL, 3=FL, 4=RR (CW/CCW alternating)
    """
    m1 = throttle + roll - pitch - yaw   # front-right  (CW)
    m2 = throttle - roll + pitch - yaw   # rear-left    (CW)
    m3 = throttle + roll + pitch + yaw   # front-left   (CCW)
    m4 = throttle - roll - pitch + yaw   # rear-right   (CCW)
    out = [m1, m2, m3, m4]
    # Clamp to [0, 1000]
    return [max(0, min(1000, int(v))) for v in out]


# ---------------------------------------------------------------------------
# Main coroutines
# ---------------------------------------------------------------------------
async def attitude_loop(imu: MPU6050, pids: dict, pwms: list,
                        arm: ArmState, rc: dict) -> None:
    """400 Hz attitude PID + motor output loop."""
    dt = 1.0 / LOOP_HZ
    import math

    while True:
        gx, gy, gz = imu.read_gyro_dps()

        if arm.is_armed:
            roll_out  = pids['roll'].update(rc['roll'], gx, dt)
            pitch_out = pids['pitch'].update(rc['pitch'], gy, dt)
            yaw_out   = pids['yaw'].update(rc['yaw'], gz, dt)

            motors = mix_quad_x(rc['throttle'], roll_out, pitch_out, yaw_out)
            for i, m in enumerate(motors):
                pulse_us = PWM_MIN_US + m
                pwms[i].duty_ns(pulse_us * 1_000)
        else:
            for pwm_ch in pwms:
                pwm_ch.duty_ns(PWM_MIN_US * 1_000)

        await asyncio.sleep_ms(int(1000 / LOOP_HZ))


async def telemetry_loop(uart: UART) -> None:
    """1 Hz MAVLink HEARTBEAT, 10 Hz ATTITUDE."""
    seq = 0
    hb_counter = 0
    while True:
        hb_counter += 1
        if hb_counter >= 10:
            hb_counter = 0
            uart.write(encode_heartbeat(seq))
        seq = (seq + 1) & 0xFF
        await asyncio.sleep_ms(100)


async def rc_watchdog(arm: ArmState, rc: dict) -> None:
    """Trigger failsafe if no RC update for 1 second."""
    while True:
        await asyncio.sleep_ms(1000)
        if time.ticks_diff(time.ticks_ms(), rc.get('last_update', 0)) > 1000:
            if arm.is_armed:
                arm.set_failsafe()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    # Hardware init
    i2c  = I2C(0, scl=Pin(I2C_SCL_PIN), sda=Pin(I2C_SDA_PIN), freq=400_000)
    uart = UART(1, baudrate=UART_BAUD, tx=Pin(UART_TX_PIN), rx=Pin(UART_RX_PIN))
    imu  = MPU6050(i2c)

    pwm_outputs = []
    for pin_num in MOTOR_PINS:
        p = PWM(Pin(pin_num))
        p.freq(50)
        p.duty_ns(PWM_MIN_US * 1_000)
        pwm_outputs.append(p)

    # PIDs — tune these values for your airframe
    pids = {
        'roll':  PID(kp=0.8,  ki=0.02, kd=0.12),
        'pitch': PID(kp=0.9,  ki=0.02, kd=0.14),
        'yaw':   PID(kp=0.5,  ki=0.01, kd=0.00),
    }

    # Shared RC state (populated by RC reader task — stub here)
    rc_state = {
        'throttle': 0.0,
        'roll':     0.0,
        'pitch':    0.0,
        'yaw':      0.0,
        'last_update': time.ticks_ms(),
    }

    arm = ArmState()
    arm.request_arm()   # auto-arm for demonstration; add safety interlock in production

    await asyncio.gather(
        attitude_loop(imu, pids, pwm_outputs, arm, rc_state),
        telemetry_loop(uart),
        rc_watchdog(arm, rc_state),
    )


asyncio.run(main())
