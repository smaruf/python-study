[← Back to Remote Aircraft Design System](../../README.md)

# ESP-BLAST — Smallest ESP32 Brushless Rocket Drone

**Source:** [Instructables — Build the Smallest ESP32 Brushless Rocket Drone (ESP-BLAST)](https://www.instructables.com/Build-the-Smallest-ESP32-Brushless-Rocket-Drone-ES/)  
**Author:** Max Imagination  
**Type:** Micro quadcopter (rocket-style vertical launch & high-speed dash)

---

## Overview

The **ESP-BLAST** is a tiny, high-performance quadcopter that launches vertically like a
rocket and can reach speeds of **100+ km/h**.  The entire flight controller is built around
an **ESP32-S3** module — no separate F4/F7 flight controller board needed — running the
open-source **ESP-FC** firmware.

| Metric | Value |
|--------|-------|
| Top speed | ~108 km/h (tested) |
| All-up weight | ~136 g (with battery) |
| Flight time | 2–8 min (throttle-dependent) |
| Control range (ESP-NOW) | ~200 m |
| Frame material | 3D-printed PETG |
| Propellers | 2.5″ tri-blade (2040) |
| Battery | 3S 450 mAh LiPo (XT30) |
| Cost (approx.) | ~US $155 (excl. goggles / transmitter) |

---

## Key Features

- **Rocket-style airframe** — vertical take-off transitions into horizontal high-speed flight
- **ESP32-S3 flight controller** — onboard Wi-Fi / Bluetooth, no separate FC required
- **ESP-NOW radio link** — ultra-low-latency peer-to-peer wireless at ~200 m
- **FPV camera with motorised tilt** — transitions between vertical and horizontal FPV views
- **OSD telemetry overlay** — MWOSD-compatible for real-time battery, altitude, speed data
- **GPS speed tracking** — measures actual top speed during flight
- **Betaflight-compatible PID tuning** — configure rates and PIDs via Betaflight Configurator

---

## Bill of Materials

See [bom.md](bom.md) for the full itemised parts list with quantities and purchase links.

**Summary:**

| Category | Items |
|----------|-------|
| Flight controller | ESP32-S3 module + custom quadcopter PCB |
| Motors | 4× 1104 brushless motor |
| ESCs | 4× Micro 8 A brushless ESC |
| Propellers | 4× 2040 tri-blade (2× CW, 2× CCW) |
| Battery | 3S 450 mAh LiPo (XT30) |
| Camera | FPV micro camera + tilt servo/actuator |
| Sensors | IMU (accel/gyro/mag), barometer, GPS |
| Frame | 3D-printed PETG parts |
| Misc | Buck converter, buzzer, LEDs, wires, connectors |

---

## Firmware

### ESP-FC (recommended)

Open-source ESP32 flight-controller firmware purpose-built for this class of aircraft.

- **Repository:** <https://github.com/rtlopez/esp-fc>
- **Build tool:** PlatformIO
- **Tuning:** Betaflight Configurator (connect via USB or Wi-Fi)
- **Configuration video:** <https://youtu.be/QTmitUFotik>

#### Quick firmware flash steps

```bash
# 1. Clone ESP-FC
git clone https://github.com/rtlopez/esp-fc.git
cd esp-fc

# 2. Install PlatformIO (if not already installed)
pip install platformio

# 3. Build and flash (USB connection to ESP32-S3)
pio run -e esp32s3 -t upload

# 4. Open Betaflight Configurator → Connect → tune PIDs
```

### Firmware skeletons (this repository)

Lightweight firmware skeletons provided in the `firmware/` directory for
educational reference and customisation:

| File | Language | Description |
|------|----------|-------------|
| [`esp_blast_flight_controller.py`](firmware/esp_blast_flight_controller.py) | MicroPython | ESP32 attitude PID + ESP-NOW RC link |
| [`esp_blast_flight_controller.c`](firmware/esp_blast_flight_controller.c) | C (ESP-IDF) | Bare-metal attitude control loop |

### MWOSD (On-Screen Display)

- **Repository:** <https://github.com/ShikOfTheRa/scarab-osd>
- Connect the OSD UART to the ESP32 UART2 TX/RX pins

---

## 3D Print Files

- **STL / STEP / GCODE:** [Cults3D — ESP-BLAST design files](https://cults3d.com/en/3d-model/game/esp-blast-an-esp32-mini-rocket-drone-100km-h-3d-design-files)
- **Recommended material:** PETG (better heat resistance and impact strength than PLA)
- **Infill:** 40 % for structural parts, 20 % for fairings

---

## PCB & Wiring

- **Gerbers, schematics, wiring diagrams:** [Google Drive (Max Imagination)](https://drive.google.com/drive/folders/1vqXdgHj-5dydJ4V21HF_LdW4lPgcUf8g?usp=sharing)
- The custom PCB integrates ESP32-S3, IMU, barometer, voltage regulator, and ESC connectors on a single board sized to fit the printed airframe.

### Wiring Overview

```
                    ┌──────────────────────────────┐
                    │        ESP32-S3 PCB           │
                    │                               │
  RC receiver ─────►  UART0 (ESP-NOW via Wi-Fi)    │
  GPS module  ─────►  UART1 (NMEA @ 9600 baud)    │
  OSD board   ─────►  UART2 (MSP telemetry)        │
  IMU (MPU-6500)───►  SPI / I2C                    │
  Barometer   ─────►  I2C                           │
                    │                               │
                    │  GPIO12 ──► Motor 1 ESC (FR)  │
                    │  GPIO13 ──► Motor 2 ESC (RL)  │
                    │  GPIO14 ──► Motor 3 ESC (FL)  │
                    │  GPIO15 ──► Motor 4 ESC (RR)  │
                    │  GPIO16 ──► Camera tilt servo  │
                    │  GPIO17 ──► Buzzer             │
                    │                               │
  3S LiPo ─────────►  XT30 → Buck 5 V → VCC rail   │
                    └──────────────────────────────┘
```

### Motor Layout (top view, quad-X)

```
           Front
    3(CCW)    1(CW)
      │    ╲╱    │
      │    /\    │
    2(CW)    4(CCW)
           Rear
```

| Motor | Position | Direction | GPIO |
|-------|----------|-----------|------|
| M1 | Front-Right | CW  | GPIO12 |
| M2 | Rear-Left   | CW  | GPIO13 |
| M3 | Front-Left  | CCW | GPIO14 |
| M4 | Rear-Right  | CCW | GPIO15 |

---

## Build Steps

### 1. 3D Print the Airframe

1. Download STL files from Cults3D (link above).
2. Print in PETG at 0.2 mm layer height, 40 % infill for structural parts.
3. Parts to print: nose cone, tail fairing, 4× motor arm, central body, battery tray, camera mount.
4. Post-process: remove supports, test-fit all parts dry before assembly.

### 2. Solder the PCB

1. Download Gerber files from the Google Drive link above.
2. Order PCB from JLCPCB / PCBWay (standard 1.6 mm FR4, 2-layer).
3. Solder SMD components per schematic: ESP32-S3 module, IMU, barometer, buck converter, capacitors.
4. Solder through-hole connectors: XT30, motor ESC connectors, UART header pins.

### 3. Install Motors & ESCs

1. Press-fit each 1104 motor into its printed arm mount.
2. Solder the three motor phase wires to the micro ESC (any order — swap two wires if motor spins wrong direction).
3. Connect each ESC signal wire to the respective GPIO on the PCB.
4. Verify motor directions: Front-Right and Rear-Left spin CW; Front-Left and Rear-Right spin CCW.

### 4. Install Camera & Tilt Mechanism

1. Mount the FPV micro camera to the tilt bracket.
2. Connect the tilt servo to GPIO16.
3. Route the camera video signal to the OSD board, then OSD video out to the video transmitter.

### 5. Flash Firmware

Follow the [Firmware](#firmware) section above to flash ESP-FC.

### 6. Configure in Betaflight Configurator

1. Connect via USB.
2. Set **Craft name** and **Board alignment** (check motor spin direction in Motors tab).
3. Tune PID values (suggested starting point below).
4. Set **Rates** and **Expo** to preference.
5. Configure failsafe: motor stop + disarm after 1 s signal loss.

**Suggested starting PIDs (tune on bench first):**

| Axis | P | I | D |
|------|---|---|---|
| Roll | 45 | 60 | 25 |
| Pitch | 50 | 65 | 28 |
| Yaw | 55 | 70 | 0 |

### 7. Range-Test & Test Flight

1. Verify arming sequence works on the bench (props off).
2. Test camera tilt servo travel.
3. Do a low hover at 20 % throttle outdoors.
4. Gradually increase throttle and test directional control.
5. Use the GPS overlay in OSD to record top speed.

---

## Safety & Legal

> ⚠️ **This drone exceeds 100 km/h.** Always fly in open, unpopulated areas well clear of
> people, animals, and obstacles.  Check local regulations (FAA Part 107, EASA UAS, CAA,
> etc.) before every flight.  A drone of this weight and speed can cause serious injury.

- Never fly over crowds or near airports.
- Check battery health before every flight (3S LiPo — do not fly below 3.5 V/cell).
- Use a LiPo-safe bag for storage and charging.

---

## Resources

| Resource | Link |
|----------|------|
| Instructables project page | <https://www.instructables.com/Build-the-Smallest-ESP32-Brushless-Rocket-Drone-ES/> |
| YouTube build walkthrough | <https://www.youtube.com/watch?v=pUi1T12QYAU> |
| Firmware & RC setup video | <https://youtu.be/QTmitUFotik> |
| ESP-FC firmware repo | <https://github.com/rtlopez/esp-fc> |
| MWOSD OSD firmware | <https://github.com/ShikOfTheRa/scarab-osd> |
| 3D design files (Cults3D) | <https://cults3d.com/en/3d-model/game/esp-blast-an-esp32-mini-rocket-drone-100km-h-3d-design-files> |
| PCB & wiring files | <https://drive.google.com/drive/folders/1vqXdgHj-5dydJ4V21HF_LdW4lPgcUf8g?usp=sharing> |
| Elektor Labs write-up | <https://www.elektormagazine.com/labs/build-a-mini-esp32-rocket-drone-that-flies-100kmh-esp-blast> |
