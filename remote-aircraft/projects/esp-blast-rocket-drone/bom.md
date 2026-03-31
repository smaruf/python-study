[← Back to ESP-BLAST Project](README.md)

# Bill of Materials — ESP-BLAST Rocket Drone

All prices are approximate USD at time of writing (2024–2025).  
Purchase links are suggestions only — substitute equivalent parts as needed.

---

## Flight Controller

| Qty | Part | Notes | Est. Price |
|-----|------|-------|-----------|
| 1 | ESP32-S3 module (e.g. ESP32-S3-WROOM-1) | Core flight controller | $4–6 |
| 1 | Custom quadcopter PCB | Gerbers available in project files | $5–10 (fabrication) |

---

## Motors & ESCs

| Qty | Part | Notes | Est. Price |
|-----|------|-------|-----------|
| 4 | 1104 brushless motor (5000–7500 KV) | Tiny, high-KV micro motors | $6–8 each |
| 4 | Micro 8 A brushless ESC (BLHeli-S / BLHeli_32) | One per motor | $4–7 each |

---

## Propellers

| Qty | Part | Notes | Est. Price |
|-----|------|-------|-----------|
| 4+ | 2040 tri-blade props (2.5″) | 2× CW + 2× CCW; buy spares | $5–8 per pack |

---

## Battery

| Qty | Part | Notes | Est. Price |
|-----|------|-------|-----------|
| 1–2 | 3S 450 mAh LiPo (XT30) | ~11.1 V nominal; 75C+ recommended | $10–15 each |
| 1 | XT30 male pigtail | For PCB battery connector | $1–2 |

---

## FPV System

| Qty | Part | Notes | Est. Price |
|-----|------|-------|-----------|
| 1 | Micro FPV camera (e.g. RunCam Nano / Caddx Ant) | NTSC or PAL | $15–25 |
| 1 | Micro video transmitter (e.g. 25/100/200 mW 5.8 GHz) | Check local band regulations | $15–30 |
| 1 | Micro linear actuator or servo (tilt mechanism) | For camera tilt transition | $5–10 |

---

## On-Screen Display

| Qty | Part | Notes | Est. Price |
|-----|------|-------|-----------|
| 1 | MWOSD-compatible OSD board (e.g. MinimOSD / AT7456E) | Firmware: MWOSD | $5–10 |

---

## Sensors

| Qty | Part | Notes | Est. Price |
|-----|------|-------|-----------|
| 1 | IMU (MPU-6500 or ICM-20689 — typically on PCB) | Accel + gyro + mag | included on PCB |
| 1 | Barometer (BMP280 or MS5611 — typically on PCB) | Altitude hold | included on PCB |
| 1 | GPS module (e.g. BN-220 / M8N) | NMEA 9600 baud UART | $15–25 |

---

## Power & Wiring

| Qty | Part | Notes | Est. Price |
|-----|------|-------|-----------|
| 1 | Buck converter 3S→5 V (e.g. Pololu D24V10F5) | Powers ESP32 + 5 V rail | $3–5 |
| 1 roll | 30 AWG silicone wire | Signal / sensor wiring | $5–8 |
| 1 pack | 28 AWG silicone wire | Motor phase wires | $5–8 |
| 1 | Buzzer (active, 5 V) | Lost-drone alarm | $1–2 |
| 2–4 | Status LEDs (red / green) | Arm state indicator | < $1 |
| 1 | Electrolytic capacitor 35 V 470–1000 µF | Battery noise decoupling | < $1 |
| 1 pack | JST-SH 1.0 mm connectors | Sensor breakouts | $3–5 |

---

## Frame (3D Printed)

| Qty | Part | Material | Notes |
|-----|------|----------|-------|
| 1 | Central body / PCB mount | PETG | Main structural hub |
| 1 | Nose cone | PETG | Aerodynamic fairing |
| 1 | Tail fairing | PETG | Closes the airframe rear |
| 4 | Motor arm | PETG | Press-fit motor mount |
| 1 | Battery tray | PETG | Sliding/clip retention |
| 1 | Camera tilt bracket | PETG | Holds camera + actuator |
| — | Misc hardware | — | M2 screws, nuts, heat inserts |

> **Print settings:** 0.2 mm layer height, 40 % gyroid infill for structural parts,
> 20 % for fairings. Use a 0.4 mm nozzle at 240–245 °C for PETG.

---

## Tools Required

- Soldering iron + solder (fine tip, < 1 mm)
- Hot air rework station (for SMD PCB assembly)
- 3D printer (FDM, PETG-capable)
- Multimeter
- LiPo charger (balance charger, 3S compatible)
- USB-C cable (for ESP32 flashing)
- Betaflight Configurator (PC/Mac/Linux)
- PlatformIO (for ESP-FC firmware compilation)

---

## Total Estimated Cost

| Category | Approx. Cost |
|----------|-------------|
| Electronics (FC, motors, ESCs, sensors) | $70–95 |
| FPV system | $30–55 |
| Battery (×2) | $20–30 |
| PCB fabrication | $5–10 |
| Frame filament + hardware | $5–10 |
| **Total** | **~$130–200** |

> Costs exclude FPV goggles (~$100–500+) and RC transmitter (~$50–200+).
