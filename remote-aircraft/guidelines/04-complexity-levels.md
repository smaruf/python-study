# 04 — Complexity Levels

This guide maps aircraft builds onto three complexity tiers —
**Simple**, **Medium**, and **Production** — and includes
supply-chain / war-grade considerations for the top tier.

---

## 1. Tier Overview

| Dimension | Simple | Medium | Production |
|---|---|---|---|
| **Operator skill** | Beginner | Intermediate | Expert team |
| **Build time** | Hours | Days–weeks | Months–years |
| **Budget** | $30–$200 | $300–$2 000 | $5 000–$500 000+ |
| **Autonomy** | Manual RC | Assisted / stabilised | Fully autonomous |
| **Regulations** | Sub-250 g or visual LOS | Part 107 / EASA A2 | BVLOS, SORA, NATO STANAG |
| **Certification** | None | Optional | DO-178C / DO-254 / MIL-STD |
| **Redundancy** | None | Single backup | Triple-redundant all-up |
| **Endurance** | 5–15 min | 20–60 min | 1–24 h+ |
| **Payload** | None / camera | 200 g–2 kg | 2–100 kg+ |
| **Use cases** | Learning, fun | Mapping, inspection | Supply, ISR, combat |

---

## 2. Simple Level

### 2.1 Profile
A beginner-accessible build using off-the-shelf parts, foam or 3D-printed airframe,
and basic manual RC control. No sensor fusion, no autonomous modes.

### 2.2 Fixed-Wing Example — "Trainer 800"

```
Wingspan:       800 mm
AUW:            350–450 g
Motor:          2212 1400KV brushless
Battery:        3S 1500 mAh LiPo
Receiver:       6-ch SBUS
FC:             None (receiver → servos direct, ESC direct)
Prop:           8×4.5"
Est. flight:    10–12 min
Build material: EPO foam, CF spar rod
```

**Wiring diagram:**
```
Battery (3S XT60)
    │
    ├──► ESC ──► Motor
    │      └──► BEC (5V)
    │               │
    │        ┌──────┴───────────────┐
    │        │ RC Receiver (SBUS)   │
    │        │ CH1: aileron servo   │
    │        │ CH2: elevator servo  │
    │        │ CH3: throttle (ESC)  │
    │        │ CH4: rudder servo    │
    │        └──────────────────────┘
```

### 2.3 Multirotor Example — "Mini Quad 65mm"

```
Frame:     65 mm brushless whoop
Motors:    1102 18000KV × 4
ESCs:      Integrated AIO board
FC:        F4 with betaflight (no GPS)
Battery:   1S 450 mAh LiPo
Receiver:  ELRS 2.4 GHz nano
```

### 2.4 Firmware at this level
- Betaflight on F4 AIO
- No GPS, no barometer, no position hold
- Manual / Acro / Angle mode only
- See `../firmware/simple/` for skeleton implementations

### 2.5 Build checklist — Simple

- [ ] Frame built and all joints secured
- [ ] Motor direction correct (quad: alternating CW/CCW)
- [ ] ESC calibrated (full throttle on power-up for range calibration)
- [ ] Receiver bound to transmitter, all channels moving correctly
- [ ] CG within 25–30% MAC (mean aerodynamic chord) for fixed-wing
- [ ] Range-check (transmitter low power) at 30 m — no glitching
- [ ] First flight: low throttle, small inputs, in open field

---

## 3. Medium Level

### 3.1 Profile
Prosumer or advanced hobbyist build with GPS, barometer, and a flight controller
running ArduPilot or PX4. Capable of autonomous waypoint missions, return-to-home,
and terrain following. Suitable for aerial photography, mapping, and light inspection.

### 3.2 Fixed-Wing Example — "Mapper 1500"

```
Wingspan:       1500 mm
AUW:            1.8 kg
Motor:          3520 400KV
Battery:        4S 5000 mAh Li-ion
FC:             Pixhawk 4 + ArduPlane
GPS:            uBlox M9N with compass
Airspeed:       MS4525 differential
Telemetry:      SiK 433 MHz radio (GCS link)
Camera:         Sony RX0 II for photogrammetry
FPV:            RunCam Phoenix 2 + 5.8 GHz
Prop:           11×5.5"
Est. flight:    45–60 min
```

**Auto-mission flow:**
```
Pre-arm checks ──► Takeoff mode ──► Auto (waypoints) ──► RTL ──► Land
```

### 3.3 Multirotor Example — "Survey Hex"

```
Frame:          X550 hexacopter
Motors:         4114 400KV × 6
ESCs:           40 A BLHeli_32 × 6
FC:             Cube Orange (ArduCopter)
GPS:            Here3+ CAN (dual)
LiDAR:          TF-Mini for terrain follow
Payload:        Sony A6000 mapping camera
Battery:        6S 10 000 mAh Li-ion
Endurance:      35 min with payload
```

### 3.4 Firmware at this level
- ArduPilot / PX4 full stack
- Waypoint missions, geofence, loiter, RTH
- Basic computer vision via companion Raspberry Pi Zero 2W
- Telemetry via MAVLink to Mission Planner / QGroundControl
- See `../firmware/medium/` for PID + MAVLink skeleton

### 3.5 Build checklist — Medium

- [ ] All sensors calibrated (compass, accelerometer, barometer)
- [ ] Motor/ESC output verified with motor test in GCS
- [ ] Failsafe: RC loss → RTH; battery critical → Land
- [ ] Geofence configured and tested on bench
- [ ] MAVLink telemetry confirmed on GCS before maiden
- [ ] SITL simulation run with mission plan
- [ ] Pre-flight checklist completed (ArduPilot wiki format)
- [ ] Log analysis after first flight (check EKF innovations, vibration levels)

---

## 4. Production Level

### 4.1 Profile
Systems designed for real-world operational use: supply delivery, ISR, search-and-rescue,
combat loitering munitions. These require certified hardware, redundant systems, regulatory
approval (BVLOS), rigorous testing, and in some cases military standards compliance.

### 4.2 Sub-types

#### Supply / Logistics Drone

```
Mission:        Last-mile delivery (medical, emergency supply)
Example:        Zipline, Wing, Matternet
AUW:            3–15 kg
Payload:        0.5–5 kg
Range:          10–100 km
Endurance:      30–90 min
Cruise speed:   60–120 km/h
Airframe:       Fixed-wing VTOL or hex with long-range wing
FC:             Cube Orange+ / custom with DO-178C
GNSS:           RTK dual constellation
Link:           4G LTE primary, satellite backup
Certification:  FAA BVLOS waiver / EASA STS-02
Recovery:       Parachute + foam belly landing
```

**Key design considerations for supply:**
- Payload quick-release mechanism (mechanical + electronic)
- Cold-chain payload box (temperature-logged)
- BVLOS communication with ATC integration (UTM/U-Space)
- Automated loading dock integration (RFID, barcode scan)

#### ISR / Surveillance Drone

```
Mission:        Intelligence, Surveillance, Reconnaissance
AUW:            2–50 kg
Payload:        EO/IR gimbal camera, synthetic aperture radar (SAR)
Endurance:      4–24 h
Airframe:       Fixed-wing or hybrid VTOL
FC:             Military-hardened with encrypted C2
Link:           Encrypted spread-spectrum + satellite
Comms security: AES-256, FHSS, ECCM
Survivability:  Low-acoustic, low-thermal signature
```

#### Loitering Munition (Architecture Only)

> **Note:** This section covers architecture only, for educational understanding of
> autonomous system design patterns. No targeting or weapons integration is described.

```
Architecture pattern:
  Navigation:  GPS + INS + optical terrain matching
  Terminal:    EO/IR seeker or radar homing
  Failsafe:    Self-destruct / mission abort on link loss
  Propulsion:  Small turbine or pusher brushless
  Airframe:    Folded storage, pop-out wings
  C2:          Encrypted one-way fire-and-forget or two-way abort
```

Standard reference: STANAG 4586 (NATO interoperable UAS C2 interface).

### 4.3 Production Firmware Requirements

| Requirement | Standard | Implementation |
|---|---|---|
| Software safety | DO-178C Level B/A | C code with MC/DC coverage |
| Hardware safety | DO-254 Level B/A | FPGA / custom ASIC |
| Cybersecurity | NIST SP 800-82 | Encrypted C2, OTA signing |
| EMI/EMC | MIL-STD-461G | Shielded enclosures, filtered power |
| Environmental | MIL-STD-810H | Temp, vibration, altitude, humidity |
| Interoperability | STANAG 4586 | NATO CUCS interface |
| Navigation | ITAR-free GNSS receiver | Novatel OEM7 or Septentrio |

### 4.4 Production Build Process

```
1. System Requirements Review (SRR)
2. Preliminary Design Review (PDR)
3. Critical Design Review (CDR)
4. Build → Integration → Unit Test
5. Subsystem Functional Test
6. System Integration Test (SIT)
7. Environmental Qualification (MIL-STD-810H / RTCA DO-160)
8. Software Qualification Test (DO-178C)
9. Airworthiness Review
10. First Flight (controlled) → Acceptance testing
11. Operational deployment
12. Maintenance & sustaining engineering
```

### 4.5 Build checklist — Production

- [ ] SRR, PDR, CDR completed and signed off
- [ ] Firmware built with verified compiler (LDRA, Polyspace, or equivalent)
- [ ] All sensors calibrated on certified test bench
- [ ] Triple IMU voting tested with single-failure injection
- [ ] GNSS spoofing detection enabled (multi-constellation crosscheck)
- [ ] C2 link encryption verified (AES-256 minimum)
- [ ] OTA update signing key managed in HSM
- [ ] Environmental testing passed (temp −20 to +55°C, humidity, vibration)
- [ ] Parachute / recovery system tested with inert drop
- [ ] Full operational scenario simulation in SITL / HITL
- [ ] Regulatory certification obtained (BVLOS waiver / type certificate)
- [ ] Operator training completed

---

## 5. Decision Matrix: Choosing Your Level

Answer these questions to select the appropriate tier:

| Question | Simple | Medium | Production |
|---|---|---|---|
| Budget < $500? | Yes | Maybe | No |
| Team < 2 people? | Yes | Maybe | No |
| Needs GPS hold / RTH? | No | Yes | Yes |
| Payload > 500 g? | No | Maybe | Yes |
| BVLOS or night flight? | No | No | Yes |
| Needs data encryption? | No | Optional | Required |
| Certification required? | No | No | Yes |
| Mission-critical (no loss tolerable)? | No | Maybe | Yes |

---

## 6. References

- FAA UAS Integration roadmap: <https://www.faa.gov/uas>
- EASA UAS regulation: <https://www.easa.europa.eu/en/domains/drones>
- NATO STANAG 4586: <https://standards.nato.int>
- DO-178C overview: <https://www.rtca.org>
- Zipline delivery system: <https://www.flyzipline.com>
- Wing aviation: <https://wing.com>
