//! autopilot_medium.zig — Medium autopilot firmware skeleton (Zig, ARM Cortex-M)
//! Complexity level: MEDIUM
//! Language:         Zig 0.13+
//!
//! Features:
//!   - Comptime-configurable PID controller (zero heap allocation)
//!   - MPU-6050 gyro reading (I2C register read stubs)
//!   - Quad-X motor mixer
//!   - MAVLink HEARTBEAT encoder (no dynamic allocation)
//!   - Arming state machine
//!
//! Build (Cortex-M4 with FPU — STM32F401):
//!   zig build-exe autopilot_medium.zig \
//!       -target thumb-freestanding-eabihf \
//!       -mcpu cortex_m4+vfp4 \
//!       -O ReleaseSafe \
//!       --name autopilot_medium
//!
//! For RP2040 (Cortex-M0+, no FPU):
//!   zig build-exe autopilot_medium.zig \
//!       -target thumb-freestanding-none \
//!       -mcpu cortex_m0plus \
//!       -O ReleaseSmall \
//!       --name autopilot_medium

const std = @import("std");

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PWM_MIN_US: u32 = 1_000;
const PWM_MAX_US: u32 = 2_000;
const PWM_MID_US: u32 = 1_500;
const LOOP_PERIOD_US: u32 = 2_500; // 400 Hz
const NUM_MOTORS: usize = 4;

// ---------------------------------------------------------------------------
// PID controller — comptime configurable, zero allocation
// ---------------------------------------------------------------------------

/// PID gains bundle (comptime constant or runtime mutable).
const PidGains = struct {
    kp: f32,
    ki: f32,
    kd: f32,
    integral_limit: f32 = 300.0,
};

/// Runtime PID state.
const PidState = struct {
    integral:   f32 = 0.0,
    prev_error: f32 = 0.0,

    pub fn reset(self: *PidState) void {
        self.integral   = 0.0;
        self.prev_error = 0.0;
    }

    pub fn update(self: *PidState, gains: PidGains,
                  setpoint: f32, measured: f32, dt: f32) f32 {
        const err = setpoint - measured;
        self.integral = std.math.clamp(
            self.integral + err * dt,
            -gains.integral_limit, gains.integral_limit,
        );
        const deriv = if (dt > 0.0) (err - self.prev_error) / dt else 0.0;
        self.prev_error = err;
        return gains.kp * err + gains.ki * self.integral + gains.kd * deriv;
    }
};

// ---------------------------------------------------------------------------
// IMU abstraction (stub — replace with I2C HAL calls)
// ---------------------------------------------------------------------------

const ImuSample = struct {
    gx: f32, // roll rate  (deg/s)
    gy: f32, // pitch rate (deg/s)
    gz: f32, // yaw rate   (deg/s)
};

/// Read gyro from MPU-6050.  Stub: returns zeroes — replace with I2C read.
fn readImu() ImuSample {
    // Real implementation:
    //   1. i2c_write(addr=0x68, reg=0x43)
    //   2. i2c_read(6 bytes) → two's-complement 16-bit big-endian per axis
    //   3. divide by 131.0 for ±250 deg/s scale
    return ImuSample{ .gx = 0.0, .gy = 0.0, .gz = 0.0 };
}

// ---------------------------------------------------------------------------
// RC input (stub — replace with SBUS UART decoder)
// ---------------------------------------------------------------------------

const RcInput = struct {
    throttle: f32, // 0.0 – 1000.0  (maps to µs offset from 1000)
    roll:     f32, // deg/s setpoint
    pitch:    f32,
    yaw:      f32,
};

fn readRc() RcInput {
    return RcInput{ .throttle = 0.0, .roll = 0.0, .pitch = 0.0, .yaw = 0.0 };
}

// ---------------------------------------------------------------------------
// Quad-X mixer
// ---------------------------------------------------------------------------

fn mixQuadX(thr: f32, roll: f32, pitch: f32, yaw: f32) [NUM_MOTORS]u32 {
    const raw = [NUM_MOTORS]f32{
        thr + roll - pitch - yaw, // front-right  CW
        thr - roll + pitch - yaw, // rear-left    CW
        thr + roll + pitch + yaw, // front-left   CCW
        thr - roll - pitch + yaw, // rear-right   CCW
    };
    var out: [NUM_MOTORS]u32 = undefined;
    for (raw, 0..) |v, i| {
        const us = @as(u32, @intFromFloat(std.math.clamp(1000.0 + v,
            @as(f32, @floatFromInt(PWM_MIN_US)),
            @as(f32, @floatFromInt(PWM_MAX_US)))));
        out[i] = us;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Motor PWM output (stub register addresses — replace with MCU HAL)
// ---------------------------------------------------------------------------

const MOTOR_REGS = [NUM_MOTORS]u32{
    0x40000034, // TIM2 CCR1
    0x40000038, // TIM2 CCR2
    0x4000003C, // TIM2 CCR3
    0x40000040, // TIM2 CCR4
};

fn motorSet(idx: usize, pulse_us: u32) void {
    const reg = @as(*volatile u32, @ptrFromInt(MOTOR_REGS[idx]));
    reg.* = std.math.clamp(pulse_us, PWM_MIN_US, PWM_MAX_US);
}

fn motorSetAll(pulse_us: u32) void {
    for (0..NUM_MOTORS) |i| motorSet(i, pulse_us);
}

// ---------------------------------------------------------------------------
// Arming state machine
// ---------------------------------------------------------------------------

const ArmState = enum { Disarmed, Armed, Failsafe };

// ---------------------------------------------------------------------------
// MAVLink HEARTBEAT encoder (no allocator)
// ---------------------------------------------------------------------------

const MAVLINK_STX: u8 = 0xFE;
const SYS_ID: u8      = 1;
const COMP_ID: u8     = 1;

fn mavlinkCrc(data: []const u8) u16 {
    var crc: u16 = 0xFFFF;
    for (data) |b| {
        var tmp: u8 = b ^ @as(u8, @truncate(crc & 0xFF));
        tmp ^= tmp << 4;
        crc = (crc >> 8) ^
              (@as(u16, tmp) << 8) ^
              (@as(u16, tmp) << 3) ^
              (@as(u16, tmp) >> 4);
    }
    return crc;
}

/// Encode a HEARTBEAT message into a fixed-size buffer.
/// Returns slice of the buffer that was written.
fn encodeHeartbeat(seq: u8, buf: *[19]u8) []u8 {
    // payload (9 bytes)
    buf[0] = 0; buf[1] = 0; buf[2] = 0; buf[3] = 0; // custom_mode
    buf[4] = 6; // MAV_TYPE_GCS
    buf[5] = 8; // MAV_AUTOPILOT_GENERIC
    buf[6] = 0; // base_mode
    buf[7] = 0; // system_status
    buf[8] = 3; // mavlink version

    // header: len, seq, sysid, compid, msgid
    const payload_len: u8 = 9;
    const header = [_]u8{ payload_len, seq, SYS_ID, COMP_ID, 0 };

    // CRC over header + payload + CRC_EXTRA(50)
    var crc_data: [15]u8 = undefined;
    @memcpy(crc_data[0..5], &header);
    @memcpy(crc_data[5..14], buf[0..9]);
    crc_data[14] = 50; // CRC_EXTRA for HEARTBEAT
    const cs = mavlinkCrc(&crc_data);

    // Assemble final packet into buf (reuse since payload already there)
    var pkt: [19]u8 = undefined;
    pkt[0] = MAVLINK_STX;
    @memcpy(pkt[1..6], &header);
    @memcpy(pkt[6..15], buf[0..9]);
    pkt[15] = @as(u8, @truncate(cs & 0xFF));
    pkt[16] = @as(u8, @truncate(cs >> 8));
    @memcpy(buf, &pkt);
    return buf[0..17];
}

// ---------------------------------------------------------------------------
// UART transmit stub — replace with MMIO USART DR write
// ---------------------------------------------------------------------------
fn uartWrite(data: []const u8) void {
    _ = data; // In real firmware: write each byte to USART->DR with TXE wait
}

// ---------------------------------------------------------------------------
// Busy-wait delay stub
// NOTE: This stub uses an approximate nop-count (84 cycles/µs at 84 MHz).
// Compiler optimisations and pipeline effects make raw nop-counts unreliable.
// Replace with SysTick (g_tick_ms) or a hardware timer CCR compare in
// production firmware.
// ---------------------------------------------------------------------------
fn delayUs(us: u32) void {
    var i: u32 = 0;
    while (i < us * 42) : (i += 1) {
        asm volatile ("nop");
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

export fn main() noreturn {
    // ESC arming: 2 s at minimum throttle
    motorSetAll(PWM_MIN_US);
    delayUs(2_000_000);

    // PID gains
    const rollGains  = PidGains{ .kp = 0.8, .ki = 0.02, .kd = 0.12 };
    const pitchGains = PidGains{ .kp = 0.9, .ki = 0.02, .kd = 0.14 };
    const yawGains   = PidGains{ .kp = 0.5, .ki = 0.01, .kd = 0.00 };

    var rollPID  = PidState{};
    var pitchPID = PidState{};
    var yawPID   = PidState{};

    const dt: f32 = @as(f32, @floatFromInt(LOOP_PERIOD_US)) / 1_000_000.0;

    var seq: u8   = 0;
    var hb_ticks: u32 = 0;
    var hb_buf: [19]u8 = undefined;

    var arm = ArmState.Disarmed;
    _ = arm; // arm/disarm logic: add RC gesture or MAVLink command handler

    while (true) {
        const imu = readImu();
        const rc  = readRc();

        const rollOut  = rollPID.update(rollGains,  rc.roll,  imu.gx, dt);
        const pitchOut = pitchPID.update(pitchGains, rc.pitch, imu.gy, dt);
        const yawOut   = yawPID.update(yawGains,   rc.yaw,  imu.gz, dt);

        const motorUs = mixQuadX(rc.throttle, rollOut, pitchOut, yawOut);
        for (motorUs, 0..) |us, i| motorSet(i, us);

        // Heartbeat at ~1 Hz (400 ticks × 2500 µs = 1 000 000 µs)
        hb_ticks += 1;
        if (hb_ticks >= 400) {
            hb_ticks = 0;
            const pkt = encodeHeartbeat(seq, &hb_buf);
            uartWrite(pkt);
            seq +%= 1;
        }

        delayUs(LOOP_PERIOD_US);
    }
}
