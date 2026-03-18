//! rc_basic.zig — Simple RC firmware skeleton (Zig, ARM Cortex-M bare-metal)
//! Complexity level: SIMPLE
//! Language:         Zig 0.13+
//!
//! Target: ARM Cortex-M0+ (RP2040 / Pico)
//!
//! Build:
//!   zig build-exe rc_basic.zig \
//!       -target thumb-freestanding-none \
//!       -mcpu cortex_m0plus \
//!       -O ReleaseSmall \
//!       --name rc_basic
//!
//! For a full RP2040 HAL, pair this skeleton with:
//!   https://github.com/ZigEmbeddedGroup/microzig  (rp2040 board support)
//!
//! Wiring (hardware-agnostic — adapt register addresses to your MCU):
//!   PWM0A (GPIO0) → ESC throttle
//!   PWM0B (GPIO1) → Servo aileron
//!   PWM1A (GPIO2) → Servo elevator

const std = @import("std");

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Servo / ESC pulse widths in microseconds.
const PWM_MIN_US: u32 = 1_000;
const PWM_MAX_US: u32 = 2_000;
const PWM_MID_US: u32 = 1_500;

/// PWM period for 50 Hz = 20 000 µs.
const PWM_PERIOD_US: u32 = 20_000;

// ---------------------------------------------------------------------------
// Minimal RP2040 PWM register map (simplified placeholders).
// Replace with microzig rp2040 HAL in a real project.
// ---------------------------------------------------------------------------

/// PWM slice base addresses (RP2040 TRM §4.5).
const PWM_BASE: u32 = 0x40050000;
const PWM_SLICE_SIZE: u32 = 0x14;

const PwmSliceRegs = extern struct {
    csr:  u32, // control/status
    div:  u32, // clock divider
    ctr:  u32, // counter
    cc:   u32, // channel compare (A in low 16, B in high 16)
    top:  u32, // wrap value
};

/// Return a pointer to the register block for a PWM slice.
fn pwmSlice(slice: u32) *volatile PwmSliceRegs {
    const addr = PWM_BASE + slice * PWM_SLICE_SIZE;
    return @as(*volatile PwmSliceRegs, @ptrFromInt(addr));
}

// ---------------------------------------------------------------------------
// PWM helpers
// ---------------------------------------------------------------------------

/// Initialise a PWM slice for 50 Hz servo output.
/// Assumes system clock = 125 MHz.  Divider 125 → 1 MHz tick → top = 20000.
fn pwmInit(slice: u32) void {
    const regs = pwmSlice(slice);
    regs.div = 125 << 4; // integer divider = 125, fractional = 0
    regs.top = PWM_PERIOD_US - 1; // wrap at 20000 (20 ms)
    regs.cc  = (PWM_MID_US << 16) | PWM_MID_US; // both channels centred
    regs.csr = 1; // enable slice
}

/// Bitmask helpers for the RP2040 PWM CC register.
/// Channel A occupies bits [15:0]; Channel B occupies bits [31:16].
const CHANNEL_A_MASK: u32 = 0x0000_FFFF;
const CHANNEL_B_MASK: u32 = 0xFFFF_0000;

/// Set pulse width in microseconds on channel A (low 16 bits of CC).
fn setPulseA(slice: u32, pulse_us: u32) void {
    const clamped = std.math.clamp(pulse_us, PWM_MIN_US, PWM_MAX_US);
    const regs = pwmSlice(slice);
    // Preserve channel B (high 16 bits), update channel A (low 16 bits)
    regs.cc = (regs.cc & CHANNEL_B_MASK) | (clamped & CHANNEL_A_MASK);
}

/// Set pulse width in microseconds on channel B (high 16 bits of CC).
fn setPulseB(slice: u32, pulse_us: u32) void {
    const clamped = std.math.clamp(pulse_us, PWM_MIN_US, PWM_MAX_US);
    const regs = pwmSlice(slice);
    regs.cc = (regs.cc & 0x0000_FFFF) | (clamped << 16);
}

// ---------------------------------------------------------------------------
// Simulated RC input
// Replace with a real UART SBUS decoder or PIO PWM reader.
// ---------------------------------------------------------------------------

const RcChannels = struct {
    throttle: u32,
    aileron:  u32,
    elevator: u32,
};

/// Returns neutral / safe RC values (replace with real reader).
fn readRC() RcChannels {
    return .{
        .throttle = PWM_MIN_US,
        .aileron  = PWM_MID_US,
        .elevator = PWM_MID_US,
    };
}

// ---------------------------------------------------------------------------
// Busy-wait delay (replace with SysTick for production)
// ---------------------------------------------------------------------------

fn delayMs(ms: u32) void {
    // NOTE: This busy-wait is a skeleton stub only.
    // The nop count is approximate (125 MHz / ~4 cycles per iteration).
    // On real hardware use a SysTick-based delay or a hardware timer;
    // compiler optimisations and pipeline effects make raw nop-counts unreliable.
    var i: u32 = 0;
    while (i < ms * 31_250) : (i += 1) {
        asm volatile ("nop");
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

export fn main() noreturn {
    // Initialise three PWM slices
    pwmInit(0); // ESC throttle on slice 0-A
    pwmInit(1); // Aileron servo on slice 1-A
    pwmInit(2); // Elevator servo on slice 2-A

    // ESC arming: hold minimum throttle for 2 seconds
    setPulseA(0, PWM_MIN_US);
    delayMs(2_000);

    while (true) {
        const rc = readRC();

        // Pass-through — no stabilisation at simple level
        setPulseA(0, rc.throttle);
        setPulseA(1, rc.aileron);
        setPulseA(2, rc.elevator);

        delayMs(20); // 50 Hz loop
    }
}
