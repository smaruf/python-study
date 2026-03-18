[← Back to Remote Aircraft Design System](../README.md)

# RC-Aircraft & Autonomous Drone Guidelines

This directory contains comprehensive guidelines for designing, building, and programming
**full RC-aircraft** and **fully autonomous drones** — from hobby-level builds to
production-grade systems used in logistics and defence.

---

## Table of Contents

| # | Guide | Description |
|---|-------|-------------|
| 01 | [Structural Implementation](01-structural-implementation.md) | Airframes, materials, load-paths, structural analysis |
| 02 | [Active & Passive Components](02-active-passive-components.md) | Motors, ESCs, servos, batteries, sensors, and passive structural parts |
| 03 | [Firmware Guide](03-firmware-guide.md) | From bare-metal blink to full ArduPilot / PX4 stacks |
| 04 | [Complexity Levels](04-complexity-levels.md) | Simple hobbyist → Medium prosumer → Production (war / supply) |
| 05 | [Multi-Language Guide](05-multilang-guide.md) | Python, TinyGo, Zig, and basic-C firmware approaches |
| 06 | [Autonomous Drone Guide](06-autonomous-drone-guide.md) | Sensor fusion, path planning, GCS, and failsafe strategies |

---

## Quick-Start Firmware Examples

Pre-built firmware skeletons are provided in the `../firmware/` tree:

```
firmware/
├── simple/          # Bare-metal / MicroPython single-loop RC
│   ├── rc_basic.py        (MicroPython / CircuitPython)
│   ├── rc_basic.c         (bare-metal C, no RTOS)
│   ├── rc_basic.go        (TinyGo for RP2040 / AVR)
│   └── rc_basic.zig       (Zig for ARM Cortex-M)
├── medium/          # PID stabilisation, telemetry, MAVLink
│   ├── autopilot_medium.py
│   ├── autopilot_medium.c
│   ├── autopilot_medium.go
│   └── autopilot_medium.zig
└── production/      # Full autonomous stack (mission planner, GCS, redundancy)
    ├── autonomous_drone.py
    ├── autonomous_drone.c
    ├── autonomous_drone.go
    └── autonomous_drone.zig
```

---

## Scope

| Vehicle type | Coverage |
|---|---|
| Fixed-wing RC plane | Full |
| Multi-rotor (quad / hex / octo) | Full |
| VTOL / hybrid | Outline |
| Autonomous UAV (survey / supply) | Full |
| Combat / loitering munition | Architecture only |

> **Safety & Legal notice** — Always comply with local aviation authority regulations
> (FAA Part 107, EASA UAS, CAA, etc.) before flying any unmanned aircraft.  
> Production-level military systems are subject to ITAR / EAR export controls.

---

## Contributing

Follow the existing code style of the repository. Add firmware examples under
`firmware/<level>/<filename>.<ext>` and update this README table accordingly.
