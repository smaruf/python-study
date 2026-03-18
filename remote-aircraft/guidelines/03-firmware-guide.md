# 03 — Firmware Guide

Firmware sits between hardware and software. This guide covers the full spectrum —
from a single blinking LED on a microcontroller to a proprietary multi-RTOS
redundant avionics suite.

---

## 1. Firmware Spectrum

```
Simple ──────────────────────────────────────────► Proprietary
  │                │                 │                   │
Bare C blink   MicroPython       ArduPilot /        Custom RTOS /
LED + PWM      stabilisation     PX4 full stack     FPGA + HSM
```

---

## 2. Simple Firmware (Level 1)

### 2.1 What it does
- Reads PWM signal from RC receiver (one channel at a time)
- Maps PWM to servo position or motor speed
- No sensor fusion, no failsafe, no telemetry

### 2.2 Supported platforms

| Platform | Language | Flash size | RAM |
|---|---|---|---|
| Arduino Uno / Nano | C/C++ | 32 KB | 2 KB |
| RP2040 (Pico) | MicroPython / C | 2 MB | 264 KB |
| ATtiny85 | bare-C | 8 KB | 512 B |
| RP2040 | TinyGo | 2 MB | 264 KB |
| RP2040 | Zig | 2 MB | 264 KB |

### 2.3 Key concepts
- **PWM input** — pulse width 1000–2000 µs represents stick position
- **PWM output** — servo library maps 0–180° to pulse width
- **Single loop** — no RTOS, everything in `while True` / `loop()`

### 2.4 Skeleton firmware (see `../firmware/simple/`)

| File | Language | Description |
|---|---|---|
| `rc_basic.py` | MicroPython | Pico reads SBUS, drives servos |
| `rc_basic.c` | C (bare-metal) | Atmel AVR, interrupt-driven PWM |
| `rc_basic.go` | TinyGo | RP2040, goroutine per channel |
| `rc_basic.zig` | Zig | RP2040, comptime HAL abstraction |

---

## 3. Medium Firmware (Level 2)

### 3.1 What it adds over Level 1
- IMU (accelerometer + gyroscope) reading
- PID attitude stabilisation loop (roll, pitch, yaw)
- SBUS/CRSF multi-channel decoding
- MAVLink heartbeat and attitude message over UART
- Arming/disarming logic
- Configurable mixer (quad X, +, fixed-wing)
- Basic failsafe (return-to-home on link loss)

### 3.2 Supported platforms

| Platform | Language | Flash | RAM | Notes |
|---|---|---|---|---|
| STM32F4 (Blackpill) | C | 1 MB | 192 KB | Bare-metal with HAL |
| STM32H7 (Pixhawk6) | C | 2 MB | 1 MB | NuttX RTOS |
| RP2040 | MicroPython | 2 MB | 264 KB | asyncio for parallel tasks |
| RP2040 | TinyGo | 2 MB | 264 KB | goroutines for concurrency |
| RP2040 | Zig | 2 MB | 264 KB | comptime, no heap |

### 3.3 PID Controller Structure

```
                  ┌──────────┐
Setpoint ──(+)──► │  PID     │──► actuator_output
          -(─)    │ P·e      │
          error   │ I·∫e dt  │
              ▲   │ D·de/dt  │
              │   └──────────┘
           sensor
```

**PID gains starting points (tune with Ziegler-Nichols or autotune):**
| Axis | Kp | Ki | Kd |
|---|---|---|---|
| Roll | 0.8 | 0.02 | 0.12 |
| Pitch | 0.9 | 0.02 | 0.14 |
| Yaw | 0.5 | 0.01 | 0.00 |

### 3.4 MAVLink Integration

```
Baud rate: 57600 (over UART) or 115200 (USB serial)
Messages used at Level 2:
  HEARTBEAT         — 1 Hz
  ATTITUDE          — 50 Hz
  RC_CHANNELS_RAW   — 50 Hz
  VFR_HUD           — 10 Hz
  STATUSTEXT        — on events
```

### 3.5 Skeleton firmware (see `../firmware/medium/`)

| File | Language | Description |
|---|---|---|
| `autopilot_medium.py` | MicroPython | IMU + PID + MAVLink on RP2040 |
| `autopilot_medium.c` | C | STM32F4, NuttX-free, HAL drivers |
| `autopilot_medium.go` | TinyGo | RP2040, goroutines, DSHOT output |
| `autopilot_medium.zig` | Zig | RP2040, zero-allocation PID |

---

## 4. Production Firmware (Level 3)

### 4.1 Open-source stacks

#### ArduPilot
- Supports: multi-rotor, fixed-wing, VTOL, boat, submarine, rover
- Hardware: Cube Orange+, Pixhawk 4/6, Matek H7
- Language: C++17 on NuttX RTOS
- Key features:
  - Triple-redundant IMU voting
  - EKF3 (Extended Kalman Filter) for state estimation
  - Auto-mission with geofence and rally points
  - Scripting via Lua for on-board custom logic
  - Terrain following via SRTM tiles
  - Parachute deployment, fence breach, battery failsafe

```
ArduPilot firmware build (Linux):
  git clone https://github.com/ArduPilot/ardupilot.git
  cd ardupilot
  Tools/environment_install/install-prereqs-ubuntu.sh -y
  ./waf configure --board CubeOrange
  ./waf copter
```

#### PX4
- Supports: multi-rotor, fixed-wing, VTOL
- Hardware: Pixhawk 4/5X/6X, Holybro
- Language: C++ on NuttX + uORB message bus
- Key features:
  - Modular micro-publisher/subscriber (uORB)
  - Hardware-in-the-loop (HITL) simulation
  - ROS2 integration via Micro-XRCE-DDS
  - QGroundControl GCS integration

```
PX4 firmware build:
  git clone https://github.com/PX4/PX4-Autopilot.git
  cd PX4-Autopilot
  make px4_fmu-v6x_default     # Pixhawk 6X
```

#### Betaflight (multi-rotor racing / freestyle)
- Language: C on STM32 (F4/F7/H7)
- Features: blackbox logging, RPM filtering, bidirectional DSHOT
- Not suitable for autonomous missions

### 4.2 Proprietary and Military Stacks

| Stack | Developer | Highlights |
|---|---|---|
| Kestrel (formerly Piccolo) | Cloud Cap Technology | DO-178C certified |
| Micropilot MP21283G | MicroPilot | Redundant, STANAG 4586 |
| Aurora Flight Sciences FCS | Boeing subsidiary | Fixed-wing UCAV |
| Skynode (Auterion OS) | Auterion | Enterprise PX4 + LTE + OTA |
| DJI N3/A3 | DJI | Proprietary, SDK available |
| Pixhawk + encryption module | Various | NuttX + TPM2 + HSM |

### 4.3 Production Firmware Architecture

```
┌─────────────────────────────────────────────────┐
│               Flight Management Computer        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Sensor   │  │  State   │  │  Mission /   │  │
│  │ Fusion   │  │  Estimat.│  │  Navigation  │  │
│  │ (EKF3)   │  │ (EKF3)   │  │  (Waypoints) │  │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  │
│       └─────────────┼────────────────┘          │
│              ┌──────▼──────┐                     │
│              │  Attitude   │                     │
│              │  Controller │                     │
│              │  (PID/MPC)  │                     │
│              └──────┬──────┘                     │
│                     │                            │
│    ┌────────────────▼──────────────────┐         │
│    │         Actuator Output           │         │
│    │  Motor × N   Servo × M   Payload  │         │
│    └───────────────────────────────────┘         │
└─────────────────────────────────────────────────┘
        │ MAVLink / DroneCAN / UAVCAN
        ▼
  Ground Control Station (GCS)
  Mission Planner / QGroundControl / Custom
```

### 4.4 Redundancy Requirements (Production)

| System | Redundancy level | Failover action |
|---|---|---|
| IMU | Triple voting | Swap to 2-of-3 majority |
| GNSS | Dual + RTK | Fall back to dead reckoning |
| Power | Dual supply + diode-ORing | Switch automatically |
| Communication | Primary + satellite backup | Mission continuation |
| FC | Hot standby secondary | Bumpless switchover |
| Airspeed sensor | Dual (fixed-wing) | Averaged or failsafe |

### 4.5 Skeleton firmware (see `../firmware/production/`)

| File | Language | Description |
|---|---|---|
| `autonomous_drone.py` | Python | Dronekit / MAVSDK mission runner |
| `autonomous_drone.c` | C | Minimal bare-metal production stack |
| `autonomous_drone.go` | TinyGo | Goroutine-per-subsystem design |
| `autonomous_drone.zig` | Zig | Comptime dispatch, no allocator |

---

## 5. Firmware Update & Signing

### 5.1 Over-the-Air (OTA) Update Flow

```
CI pipeline
  └─ build firmware (signed with developer key)
       └─ upload to update server
            └─ drone polls update server (on boot / hourly)
                 └─ verify signature (RSA-PSS SHA-256)
                      └─ flash & reboot
```

### 5.2 Signing Tools

| Tool | Algorithm | Use |
|---|---|---|
| `imgtool` (MCUboot) | RSA-2048 / EC-P256 | Open-source bootloader signing |
| ARM TrustZone + TF-M | ECC-P256 | Cortex-M33 secure boot |
| HSM (ATECC608) | ECC-P256 | Hardware key storage |
| Custom TPM module | RSA-2048 | Military-grade chain-of-trust |

---

## 6. Development Tools & Simulators

| Tool | Purpose |
|---|---|
| ArduPilot SITL | Software-in-the-loop simulation |
| Gazebo + PX4 | Full 3D physics simulation |
| HITL (Hardware-in-the-loop) | Real FC, simulated sensors |
| Renode | Multi-node embedded simulation |
| OpenPilot/LibrePilot | Alternative open FC firmware |
| Betaflight Configurator | Blackbox analyser, PID tuning |
| Mission Planner | GCS for ArduPilot |
| QGroundControl | GCS for PX4 + ArduPilot |

---

## 7. References

- ArduPilot developer wiki: <https://ardupilot.org/dev/>
- PX4 developer guide: <https://docs.px4.io/main/en/>
- Betaflight wiki: <https://betaflight.com/docs>
- MAVLink protocol: <https://mavlink.io/en/>
- MCUboot (OTA): <https://docs.mcuboot.com>
- MAVSDK Python: <https://mavsdk.mavlink.io/main/en/python/>
