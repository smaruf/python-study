# 02 — Active & Passive Components

Understanding the distinction between active (powered, moving, or sensing) and passive
(structural, static, non-powered) components is essential when designing, sourcing,
and debugging any RC-aircraft or autonomous drone.

---

## 1. Taxonomy Overview

```
Components
├── Active
│   ├── Actuators  (motors, servos, ESCs, valves)
│   ├── Sensors    (IMU, GPS, barometer, optical flow, LiDAR, camera)
│   ├── Computing  (flight controller, companion computer, RC receiver)
│   └── Power      (battery, BEC/UBEC, power module, capacitors bank)
└── Passive
    ├── Structural (frame, spars, ribs, skin)
    ├── Electrical (wire, connectors, solder joints, PCB traces)
    ├── Aerodynamic (fixed fairings, covers, prop guards)
    └── Thermal    (heatsinks, thermal pads, conformal coating)
```

---

## 2. Active Components

### 2.1 Brushless DC Motors (BLDC)

| Parameter | Hobby (simple) | Prosumer (medium) | Production |
|---|---|---|---|
| KV rating | 2300–2700 KV (5" prop) | 400–900 KV (10–15" prop) | 100–300 KV (18"+ prop) |
| Motor size | 2207, 2306 | 4114, 5010 | 6215, 8015 |
| Efficiency | 75–82% | 82–88% | 88–94% |
| Cooling | Air (passive) | Air + heat-pipe | Liquid or active air |
| Example | EMAX ECO 2207 | T-Motor MN5010 | T-Motor U15 |

**KV selection rule:**
```
Optimal_KV ≈ 1900 / (prop_diameter_inches × prop_pitch_inches)^0.5
```

**Motor thrust-to-weight targeting:**
- Agile / acro: 6:1 thrust-to-weight (TWR)
- Stable cruise: 3:1 TWR
- Heavy-lift / cargo: 2:1 TWR

### 2.2 Servos (Fixed-Wing & VTOL)

| Class | Torque | Speed | Protocol | Use |
|---|---|---|---|---|
| Micro analog | 1.5–2.5 kg·cm | 0.12 s/60° | PWM | Small trainers |
| Standard analog | 4–8 kg·cm | 0.10 s/60° | PWM | Medium RC planes |
| Standard digital | 6–10 kg·cm | 0.07 s/60° | PWM/PPM | Performance RC |
| Coreless digital | 8–15 kg·cm | 0.05 s/60° | PWM | Precision surfaces |
| High-voltage (HV) | 15–30 kg·cm | 0.04 s/60° | PWM/CAN | Production UAV |
| CAN/serial bus | Up to 60 kg·cm | 0.03 s/60° | CAN/Serial | Military grade |

**Servo sizing (control surface):**
```
Required_torque = hinge_moment × safety_factor
Hinge_moment = 0.5 × rho × V² × c² × b × CH
```
where `c` = chord, `b` = surface span, `CH` = hinge moment coefficient (~0.015).

### 2.3 Electronic Speed Controllers (ESCs)

| Feature | Simple | Medium | Production |
|---|---|---|---|
| Protocol | PWM 1000–2000 µs | DSHOT300/600 | DSHOT1200 / CAN |
| BLHeli firmware | BLHeli_S | BLHeli_32 | Custom / Kfend / Myxa |
| Current rating | 20–40 A | 40–80 A | 80–300 A |
| Telemetry | None | UART one-wire | CAN telemetry, logging |
| Motor brake | Basic | Configurable | Dynamic braking |
| FOC (sinusoidal) | No | Optional | Yes (quiet, efficient) |

**ESC current derating:** select ESC rated for 1.5× the motor's peak current draw.

### 2.4 Power Systems (Batteries)

| Chemistry | Energy density | C-rate | Cycle life | Use case |
|---|---|---|---|---|
| LiPo (standard) | 150–200 Wh/kg | 30–100 C | 200–400 | Hobby, racing |
| LiHV (4.35 V/cell) | 160–220 Wh/kg | 25–75 C | 150–300 | Acro, performance |
| Li-ion (18650/21700) | 200–260 Wh/kg | 5–15 C | 500–1000 | Long-endurance UAV |
| Li-S (emerging) | 350–400 Wh/kg | 1–5 C | 200 | Research / future |
| Hydrogen fuel cell | 800–1500 Wh/kg | — | 3000+ | Military endurance |

**Battery capacity calculation:**
```
Capacity_mAh = (AUW_g × hover_time_min × 1.1) / (efficiency × volts_per_cell × cells)
```

**Power module selection:** rate BEC/UBEC at 20% above peak load current.

### 2.5 Sensors

#### Inertial Measurement Unit (IMU)

| Sensor | DOF | Noise density | Common chips |
|---|---|---|---|
| Accelerometer | 3-axis | 80–300 µg/√Hz | MPU-6000, ICM-42688-P |
| Gyroscope | 3-axis | 0.003–0.015 °/s/√Hz | ICM-42688-P, BMI088 |
| Magnetometer | 3-axis | 0.3–2 µT/√Hz | QMC5883L, LIS3MDL |
| Combined IMU | 6/9 DOF | — | ICM-42688-P (production) |

#### GPS / GNSS

| System | Accuracy (CEP) | Constellations | Update rate |
|---|---|---|---|
| uBlox M8N | 2.5 m | GPS + GLONASS | 10 Hz |
| uBlox M9N | 1.5 m | GPS + GAL + BDS | 25 Hz |
| uBlox F9P (RTK) | 1 cm | All 4 + SBAS | 20 Hz |
| Septentrio (military) | 5 mm | All + encryption | 100 Hz |

#### Barometer

| Chip | Altitude resolution | Range |
|---|---|---|
| BMP280 | 10 cm | 0–9000 m |
| MS5611 | 10 cm | 0–10000 m |
| SPL06-001 | 5 cm | 0–10000 m |

#### Optical Flow / LiDAR (indoor + precision landing)

| Sensor | Range | Use |
|---|---|---|
| PMW3901 | 0.08–∞ m | Indoor hover / velocity estimate |
| TF-Luna LiDAR | 0.2–8 m | Low-altitude hold |
| TF-Mini Plus | 0.1–12 m | Terrain following |
| Garmin LIDAR-Lite v3 | 1–40 m | Outdoor altitude hold |

### 2.6 Flight Controllers (FC)

| Level | Example FC | MCU | FC Software |
|---|---|---|---|
| Simple | OMNIBUSF4 | STM32F4 | Betaflight |
| Medium | Pixhawk 4 / 6C | STM32H7 | ArduPilot / PX4 |
| Production | Cube Orange+ | STM32H7 dual | ArduPilot (triple IMU) |
| Military | Custom FPGA/ARM | Xilinx/Zynq | Proprietary RTOS |

### 2.7 Companion Computers

| Level | Device | OS | Role |
|---|---|---|---|
| Simple | None | — | FC only |
| Medium | Raspberry Pi Zero 2W | Linux | Vision, logging |
| Medium | Orange Pi 5 | Linux | Neural inference |
| Production | NVIDIA Jetson Orin NX | Linux | Full autonomy stack |
| Military | Rockchip RK3588 (hardened) | Linux RT | All roles |

### 2.8 RC Receivers & Radio Links

| Protocol | Latency | Range | Use |
|---|---|---|---|
| IBUS (FlySky) | ~6 ms | 1–2 km | Budget hobby |
| SBUS (FrSky) | 9 ms | 1–2 km | Standard hobby |
| CRSF (TBS Crossfire) | <1 ms | 10–30 km | Long-range FPV |
| ExpressLRS (ELRS) | <1 ms | 5–50 km | Open-source LRS |
| DroneCAN / MAVLink | ~5 ms | Any (via 4G/Sat) | Autonomous systems |
| Military encrypted | <2 ms | Line-of-sight / satellite | COMSEC link |

---

## 3. Passive Components

### 3.1 Structural Passive Parts

| Part | Material | Function |
|---|---|---|
| Main spar | CFRP tube or box | Carries span-wise bending |
| Wing ribs | Laser-cut plywood / 3D-printed | Maintains aerofoil shape |
| Fuselage skin | GFRP / foam / coroplast | Aerodynamic shell, load transfer |
| Motor mount plate | CF sheet or aluminium | Transfers thrust to airframe |
| Landing skids | Nylon / CFRP tube | Impact absorption |
| Prop guards | TPU / PETG | Passive prop protection |

### 3.2 Electrical Passive Parts

| Part | Spec (typical) | Notes |
|---|---|---|
| Power wire (main) | 12–14 AWG silicone | Low resistance, high flexibility |
| Signal wire | 26–28 AWG | Servo/sensor wiring |
| XT30 connector | 30 A | Small quads |
| XT60 connector | 60 A | Standard quads |
| XT90 connector | 90 A | Heavy-lift |
| AS150 / AS250 | 150–250 A | Production heavy |
| PCB/PDB | Custom copper pours | Power distribution |
| Capacitor (bulk) | 2–4 × 1000 µF 35 V | Reduces ESC voltage spikes |

### 3.3 Thermal Passive Parts

| Part | Material | Function |
|---|---|---|
| Motor heatsink fin | Aluminium | Dissipates winding heat |
| ESC heatsink pad | Copper / aluminium | ESC FET cooling |
| Thermal interface pad | Silicone, 4–8 W/m·K | Bridges chip to heatsink |
| Conformal coating | Acrylic / silicone | Moisture, dust protection |
| FC dampening foam | Silicone open-cell | Vibration + thermal isolation |

### 3.4 Aerodynamic Passive Parts

| Part | Notes |
|---|---|
| Spinner / nose cone | Reduces frontal drag |
| Wheel fairings | Reduces landing-gear drag |
| Wing root fillets | Reduces junction interference drag |
| Canopy | Protects electronics, maintains aero |
| Prop washer / hub | Secure prop attachment |

---

## 4. Bill of Materials (BOM) Templates

### 4.1 Simple RC Fixed-Wing BOM

| # | Component | Type | Qty | Notes |
|---|---|---|---|---|
| 1 | Foam board 1000×700 mm | Passive structural | 2 | Wing + fuselage |
| 2 | Brushless motor (2212 1400KV) | Active actuator | 1 | Puller configuration |
| 3 | 30 A ESC | Active power | 1 | BLHeli_S |
| 4 | 9g servo | Active actuator | 3 | Aileron × 2, elevator |
| 5 | 6g servo | Active actuator | 1 | Rudder |
| 6 | RC receiver (SBUS) | Active sensor/link | 1 | 6-channel min |
| 7 | 3S 2200 mAh LiPo | Active power | 1 | 30 C min |
| 8 | 8×4.5" APC propeller | Passive aero | 1 | Puller |
| 9 | XT60 connector pair | Passive elec | 1 | Battery lead |
| 10 | 3 mm carbon spar | Passive structural | 1 m | Main wing spar |

### 4.2 Medium Autonomous Quad BOM

| # | Component | Type | Qty | Notes |
|---|---|---|---|---|
| 1 | 5" CF frame (220 mm) | Passive structural | 1 | Stack-mount compatible |
| 2 | BLDC motor (2207 2450KV) | Active actuator | 4 | Quad configuration |
| 3 | 40 A ESC (DSHOT600) | Active power | 4 | BLHeli_32 |
| 4 | Pixhawk 6C FC | Active computing | 1 | With ArduPilot |
| 5 | GPS module (M9N) | Active sensor | 1 | With compass |
| 6 | CRSF receiver | Active link | 1 | Long-range RC |
| 7 | 4S 3000 mAh Li-ion | Active power | 1 | 10 C, long endurance |
| 8 | 5" tri-blade props | Passive aero | 2 pairs | CW + CCW |
| 9 | 30×30 ESC stack mount | Passive structural | 1 | Vibration isolated |
| 10 | RPi Zero 2W + camera | Active computing | 1 | Companion computer |

### 4.3 Production Cargo Drone BOM

| # | Component | Type | Qty | Notes |
|---|---|---|---|---|
| 1 | Hexacopter CFRP frame (900 mm) | Passive structural | 1 | Monocoque arms |
| 2 | T-Motor MN5010 360KV | Active actuator | 6 | Coaxial option available |
| 3 | 80 A FOC ESC (CAN) | Active power | 6 | Current telemetry |
| 4 | Cube Orange+ FC | Active computing | 1 | Triple redundant IMU |
| 5 | uBlox F9P (RTK) | Active sensor | 1 + base | cm-level accuracy |
| 6 | Lightware LW20 LiDAR | Active sensor | 2 | Terrain follow |
| 7 | NVIDIA Jetson Orin NX | Active computing | 1 | Full autonomy |
| 8 | 6S 16 000 mAh Li-ion pack | Active power | 2 | Parallel hot-swap |
| 9 | Parachute system | Passive safety | 1 | Ballistic deploy |
| 10 | 4G LTE + satellite link | Active link | 1 | Redundant C2 |

---

## 5. Component Integration Rules

1. **Power sequencing** — power FC before arming motors; power down motors before FC.
2. **Signal wire routing** — keep signal wires away from power wires (>10 mm separation or twisted pairs).
3. **Vibration isolation** — always mount FC on dampeners; hard-mount ESC stack.
4. **Connector torque locking** — use locking collars on XT90/AS150 for production.
5. **Redundancy** — production builds require dual IMU, dual GNSS, dual power modules.
6. **EMI shielding** — wrap GPS module in copper foil tape connected to ground plane.
7. **Thermal budget** — ensure total heat dissipation ≤ thermal design power (TDP) of enclosure.

---

## 6. References

- Betaflight motor and ESC setup guide: <https://betaflight.com/docs/tuning/motors>
- ArduPilot copter hardware selection: <https://ardupilot.org/copter/docs/choosing-a-flight-controller.html>
- T-Motor product selector: <https://store.tmotor.com>
- SpeedyBee FC & ESC stack guide: <https://www.speedybee.com>
