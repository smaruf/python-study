# 05 — Multi-Language Firmware Guide

Each language brings different trade-offs for embedded flight-control firmware.
This guide explains when to choose Python, TinyGo, Zig, or basic-C and how to
structure each for RC-aircraft and autonomous drones.

---

## 1. Language Comparison Matrix

| Criterion | MicroPython | TinyGo | Zig | basic-C |
|---|---|---|---|---|
| **Abstraction level** | High | Medium-high | Medium | Low |
| **Memory overhead** | High (GC) | Medium (GC opt.) | None (manual) | None (manual) |
| **Real-time suitability** | Moderate | Good | Excellent | Excellent |
| **Compile time** | Interpreted | Fast | Medium | Fast |
| **Binary size** | Large | Medium | Small | Smallest |
| **Safety (compile-time)** | Dynamic | Type-safe | Memory-safe | Unsafe |
| **Concurrency model** | asyncio | goroutines | async + threads | ISR + RTOS tasks |
| **Platforms** | RP2040, ESP32, STM32 | RP2040, AVR, ARM | ARM Cortex-M | AVR, STM32, ARM |
| **Ideal complexity tier** | Simple / Medium | Simple / Medium | Medium / Production | All tiers |
| **Toolchain complexity** | Very low | Low | Medium | Low–medium |

---

## 2. Python (MicroPython / CircuitPython)

### 2.1 When to use
- Rapid prototyping of sensor drivers
- Simple RC receivers on RP2040 or ESP32
- Companion computer scripts (Dronekit / MAVSDK on full Linux Python)
- Educational demonstrations

### 2.2 Hardware targets
| Board | Python runtime | Notes |
|---|---|---|
| Raspberry Pi Pico (RP2040) | MicroPython | Dual-core, best for simple/medium |
| ESP32 | MicroPython | WiFi/BT on-chip, good for telemetry |
| Circuit Playground Express | CircuitPython | Beginners, safe, USB drag-drop flash |
| Raspberry Pi (full OS) | CPython | Full Dronekit, MAVSDK, OpenCV |

### 2.3 Code patterns

**Reading a servo with PWM (MicroPython / RP2040):**
```python
from machine import Pin, PWM

# 50 Hz servo, pulse 1000–2000 µs
servo = PWM(Pin(15))
servo.freq(50)

def set_angle(degrees):
    # map 0–180° → 1000–2000 µs → duty in ns
    pulse_ns = int(1_000_000 + degrees / 180 * 1_000_000)
    servo.duty_ns(pulse_ns)

set_angle(90)  # centred
```

**Async multi-channel loop:**
```python
import asyncio
from machine import Pin, PWM

async def read_channel(pin_num: int, queue: asyncio.Queue):
    """Measure PWM pulse width from RC receiver."""
    pin = Pin(pin_num, Pin.IN)
    while True:
        # simple polling (replace with interrupt for accuracy)
        await asyncio.sleep_ms(20)
        await queue.put(pin.value())

async def main():
    q = asyncio.Queue()
    asyncio.create_task(read_channel(10, q))
    while True:
        val = await q.get()
        print(f"channel: {val}")

asyncio.run(main())
```

### 2.4 Limitations
- Garbage collector can cause latency spikes (use `gc.collect()` strategically)
- PWM jitter higher than bare-C (±20 µs typical on RP2040)
- No direct DMA/DSHOT support in standard MicroPython (use PIO on RP2040)

### 2.5 Dronekit (Linux Python — companion computer)
```python
from dronekit import connect, VehicleMode, LocationGlobalRelative

vehicle = connect("/dev/ttyAMA0", baud=57600, wait_ready=True)
vehicle.mode = VehicleMode("GUIDED")
vehicle.armed = True

target = LocationGlobalRelative(lat=37.7749, lon=-122.4194, alt=20)
vehicle.simple_goto(target)
```

---

## 3. TinyGo

### 3.1 When to use
- Goroutine-based concurrent sensor reading
- Memory-constrained systems where Go ergonomics are desired
- When DSHOT or CAN output can be built as a Go package
- Cross-compilation targets: RP2040, Arduino, STM32

### 3.2 Toolchain setup
```bash
# Install TinyGo (Linux / macOS)
brew install tinygo          # macOS
snap install tinygo          # Linux snap

# Flash to Raspberry Pi Pico
tinygo flash -target=pico main.go

# Flash to Arduino Nano
tinygo flash -target=arduino-nano main.go
```

### 3.3 Code patterns

**PWM servo driver:**
```go
package main

import (
    "machine"
    "time"
)

func main() {
    // RP2040: GPIO15 → servo signal
    pwm := machine.PWM7
    pwm.Configure(machine.PWMConfig{Period: 20_000_000}) // 50 Hz (20 ms)

    ch, _ := pwm.Channel(machine.GPIO15)

    for {
        // centred position: ~1500 µs
        pwm.Set(ch, pwm.Top()/13) // approx 1500 µs out of 20000 µs
        time.Sleep(time.Second)
    }
}
```

**Goroutine per sensor:**
```go
package main

import (
    "machine"
    "time"
)

var rollCh = make(chan float32, 4)
var pitchCh = make(chan float32, 4)

func readIMU(i2c *machine.I2C) {
    for {
        roll, pitch := readMPU6050(i2c)
        rollCh <- roll
        pitchCh <- pitch
        time.Sleep(10 * time.Millisecond)
    }
}

func pidLoop() {
    for {
        roll := <-rollCh
        pitch := <-pitchCh
        // PID math here
        _ = roll
        _ = pitch
        time.Sleep(10 * time.Millisecond)
    }
}

func main() {
    i2c := machine.I2C0
    i2c.Configure(machine.I2CConfig{Frequency: 400_000})

    go readIMU(i2c)
    pidLoop()
}
```

### 3.4 Limitations
- Garbage collector still present (use `-gc=leaking` or `-gc=conservative` for RT)
- No `unsafe` pointer arithmetic by default (safe, but limits hardware access)
- Smaller community vs C for embedded compared to mainstream Go
- Some packages not compatible with `tinygo` build constraints

---

## 4. Zig

### 4.1 When to use
- Production / medium builds requiring zero-cost abstraction
- Safety-critical code where C's undefined behaviour is a risk
- Cross-compilation with `comptime` for hardware-specific optimisation
- Replacing C while keeping the same binary size

### 4.2 Toolchain setup
```bash
# Install Zig (Linux)
wget https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz
tar -xf zig-linux-x86_64-0.13.0.tar.xz
export PATH="$PWD/zig-linux-x86_64-0.13.0:$PATH"

# Cross-compile for ARM Cortex-M0+ (RP2040)
zig build-exe main.zig \
    -target thumb-freestanding-none \
    -mcpu cortex_m0plus \
    -O ReleaseSmall \
    --name firmware
```

### 4.3 Code patterns

**Comptime HAL abstraction:**
```zig
const std = @import("std");

// Compile-time pin configuration
const PwmConfig = struct {
    pin: u8,
    period_us: u32,
};

fn initPwm(comptime config: PwmConfig) void {
    // Hardware register access (simplified)
    const base_addr: u32 = 0x40050000 + @as(u32, config.pin) * 0x14;
    const reg = @as(*volatile u32, @ptrFromInt(base_addr));
    reg.* = config.period_us;
}

pub fn main() !void {
    comptime var servo_cfg = PwmConfig{ .pin = 15, .period_us = 20_000 };
    initPwm(servo_cfg);
}
```

**Zero-allocation PID controller:**
```zig
const Pid = struct {
    kp: f32,
    ki: f32,
    kd: f32,
    integral: f32 = 0,
    prev_error: f32 = 0,

    pub fn update(self: *Pid, setpoint: f32, measurement: f32, dt: f32) f32 {
        const error = setpoint - measurement;
        self.integral += error * dt;
        const derivative = (error - self.prev_error) / dt;
        self.prev_error = error;
        return self.kp * error + self.ki * self.integral + self.kd * derivative;
    }
};

pub fn main() void {
    var roll_pid = Pid{ .kp = 0.8, .ki = 0.02, .kd = 0.12 };
    const output = roll_pid.update(0.0, 5.0, 0.01);
    _ = output;
}
```

### 4.4 Advantages over C
- No undefined behaviour in safe Zig (explicit unreachable, no silent wrap)
- `comptime` replaces error-prone `#define` macros
- Built-in test runner (`zig test`)
- Single binary, no dynamic linker on bare-metal

---

## 5. Basic C

### 5.1 When to use
- Any complexity tier — C is universal for embedded
- When existing libraries (ArduPilot, PX4, NuttX) are C/C++
- Bare-metal on 8-bit AVR where other languages don't fit
- When binary size or cycle-count is the primary constraint

### 5.2 Toolchain setup
```bash
# AVR (Arduino Uno/Nano)
sudo apt install gcc-avr avr-libc avrdude

# ARM Cortex-M (STM32, RP2040)
sudo apt install gcc-arm-none-eabi

# Compile for AVR
avr-gcc -mmcu=atmega328p -DF_CPU=16000000UL -Os \
    -o firmware.elf main.c
avr-objcopy -O ihex firmware.elf firmware.hex
avrdude -c arduino -p m328p -P /dev/ttyUSB0 -U flash:w:firmware.hex
```

### 5.3 Code patterns

**Interrupt-driven PWM input (AVR):**
```c
#include <avr/io.h>
#include <avr/interrupt.h>
#include <stdint.h>

volatile uint16_t pwm_width_us = 1500; // default: centre

ISR(TIMER1_CAPT_vect) {
    static uint16_t rise_time = 0;
    if (TCCR1B & (1 << ICES1)) {          // rising edge
        rise_time = ICR1;
        TCCR1B &= ~(1 << ICES1);          // switch to falling
    } else {                               // falling edge
        pwm_width_us = ICR1 - rise_time;  // 2 MHz timer → 0.5 µs per tick
        pwm_width_us /= 2;                // convert to µs
        TCCR1B |= (1 << ICES1);           // switch back to rising
    }
}

void pwm_init(void) {
    TCCR1B = (1 << ICNC1) | (1 << ICES1) | (1 << CS11); // prescaler 8 → 2 MHz
    TIMSK1 = (1 << ICIE1);
    sei();
}
```

**Simple PID in C:**
```c
typedef struct {
    float kp, ki, kd;
    float integral;
    float prev_error;
} PidController;

float pid_update(PidController *pid, float setpoint, float measured, float dt) {
    float error = setpoint - measured;
    pid->integral += error * dt;
    float derivative = (error - pid->prev_error) / dt;
    pid->prev_error = error;
    return pid->kp * error + pid->ki * pid->integral + pid->kd * derivative;
}
```

**Main control loop with 400 Hz rate:**
```c
#include <util/delay.h>

int main(void) {
    pwm_init();
    PidController roll_pid = {.kp=0.8, .ki=0.02, .kd=0.12};
    float output;

    while (1) {
        float roll_measured = read_imu_roll();
        float roll_setpoint = map_pwm_to_angle(pwm_width_us);
        output = pid_update(&roll_pid, roll_setpoint, roll_measured, 0.0025f);
        set_motor_output(output);
        _delay_us(2500); // 400 Hz
    }
}
```

### 5.4 Safety practices in C
- Always check array bounds explicitly
- Use `volatile` for all ISR-shared variables
- Use `static` for module-private globals
- Never use `gets()` / unbound `scanf`
- Prefer `stdint.h` types (`uint16_t`) over `int`
- Compile with `-Wall -Wextra -Werror`

---

## 6. Language Selection by Use Case

| Use case | Recommended | Alternative |
|---|---|---|
| First prototype, fast iteration | MicroPython | Python (full OS) |
| Educational demo | MicroPython | CircuitPython |
| Concurrent sensor reading | TinyGo | MicroPython asyncio |
| Production stabilisation loop | C (bare-metal) | Zig |
| Safety-critical with DO-178C | C (MISRA-C:2023) | Ada |
| Full autonomous stack | C++ (ArduPilot/PX4) | Python companion |
| Zero-overhead abstraction | Zig | C |
| Companion AI inference | Python (CPython) | Rust |
| Cross-language FFI | C header + bindings | Zig C-interop |

---

## 7. Cross-Language Integration

### 7.1 Python calling C via ctypes
```python
import ctypes

# Load a compiled shared library for fast PID computation
lib = ctypes.CDLL("./pid.so")
lib.pid_update.restype = ctypes.c_float
lib.pid_update.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # pid struct
    ctypes.c_float,                   # setpoint
    ctypes.c_float,                   # measured
    ctypes.c_float,                   # dt
]
```

### 7.2 Zig calling C headers
```zig
const c = @cImport({
    @cInclude("mavlink/mavlink.h");
});

pub fn sendHeartbeat(chan: u8) void {
    var msg: c.mavlink_message_t = undefined;
    c.mavlink_msg_heartbeat_pack(1, 200, &msg,
        c.MAV_TYPE_QUADROTOR,
        c.MAV_AUTOPILOT_GENERIC,
        0, 0, 0);
}
```

### 7.3 TinyGo calling C assembly routines
```go
// extern.go
package hal

// DShot600Send sends a DSHOT packet via PIO; implemented in pio_dshot.S
//
//go:linkname dshotSend _dshot_send
func dshotSend(value uint16)
```

---

## 8. References

- MicroPython documentation: <https://docs.micropython.org>
- TinyGo target list: <https://tinygo.org/docs/reference/microcontrollers/>
- Zig embedded guide (zig-embedded-group): <https://github.com/ZigEmbeddedGroup>
- MISRA-C:2023 guidelines: <https://misra.org.uk>
- MAVSDK Python: <https://mavsdk.mavlink.io/main/en/python/>
- Dronekit Python: <https://dronekit.io>
