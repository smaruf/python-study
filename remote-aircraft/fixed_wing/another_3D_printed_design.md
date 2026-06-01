[← Back to Remote Aircraft Design System](../README.md)

# Another 3D-Printed Design — 3JWings DC Motor Stick Plane

This document turns the original placeholder reference into a complete build and generation guide for the
**3JWings DC Motor RC Stick Plane** already modeled in this repository.

- **Reference build:** 3JWings — *"Make Rc Plane With dc Motor | DIY Rc Stick Plane"* (Jan 10, 2025)
- **Build type:** beginner 3-channel 3D-printed fixed-wing aircraft
- **Repository implementation:** [`../fixed_wing/community_builds.py`](community_builds.py)
- **Generator script:** [`../examples/generate_3d_printed_stick_plane.py`](../examples/generate_3d_printed_stick_plane.py)
- **Original external STL link:** <https://shorturl.at/ZFULq>

---

## Overview

This aircraft is a **small, light, trainer-style RC stick plane** built around:

- a **DC motor** in tractor configuration
- **rudder + elevator + throttle** control
- a **fully 3D-printed airframe**
- a **light carbon spar**
- a **1S or 2S power system**

The repository includes:

1. a complete design data model
2. aerodynamic estimates
3. 3D-print specifications
4. Arduino radio/control sketches
5. a Python STL generator for printable parts

---

## Main Specifications

| Parameter | Value |
|---|---:|
| Wingspan | 609 mm |
| Mean chord | 120 mm |
| Wing area | 73,080 mm² |
| Wing area | 730.8 cm² |
| Aspect ratio | 5.08 |
| All-up weight | 116 g |
| CG from leading edge | 33 mm |
| CG as % MAC | 27.5% |
| Estimated cruise speed | 5.4 m/s |
| Estimated stall speed | 3.7 m/s |
| Fuselage length | 480 mm |
| Stability class | VERY STABLE (trainer) |

---

## Airframe Geometry

### Wing

- **Configuration:** straight wing, flat-bottom printable section
- **Total span:** 609 mm
- **Semi-span:** 304.5 mm per half
- **Chord:** 120 mm
- **Maximum thickness:** 14 mm
- **Spar slot:** 4 mm
- **Recommended spar:** 4 mm carbon rod or 3 mm bamboo skewer

### Fuselage

The design is split into three printable sections:

1. **Fuselage nose**
   - length: ~80 mm
   - outer diameter: 42 mm
   - wall: 1.2 mm
   - motor shaft hole: 2.0 mm
2. **Fuselage mid**
   - length: 200 mm
   - cross-section: 42 × 38 mm
   - wall: 1.2 mm
   - intended to carry wing saddle and electronics
3. **Fuselage tail**
   - length: 200 mm
   - cross-section: 18 × 18 mm tapering to 12 × 12 mm
   - wall: 1.2 mm
   - intended to support the tail group

### Tail

- **Horizontal stabilizer area:** 14,616 mm²
- **Horizontal stabilizer span:** ~209 mm
- **Horizontal stabilizer chord:** 70 mm
- **Vertical fin area:** 7,308 mm²
- **Vertical fin height:** 80 mm
- **Vertical fin chord:** ~91 mm

---

## Aerodynamics and Stability

### Wing Loading

- **Wing loading:** 0.1588 g/cm²

This is a light-loading aircraft, which supports:

- low launch speed
- gentle handling
- short landing roll
- trainer-style flight

### Speed Estimates

- **Cruise:** ~5.4 m/s
- **Stall:** ~3.7 m/s

These values are theoretical estimates from the repository model and are suitable for build planning and
initial trimming.

### Stability

- **Tail volume coefficient:** ~0.696
- **Assessment:** **VERY STABLE (trainer)**

This suggests forgiving pitch behavior and makes the aircraft suitable for beginners.

---

## Control System

### Channel Layout

| Channel | Function |
|---|---|
| CH1 | throttle |
| CH2 | elevator |
| CH3 | rudder |

### Power and Propulsion

| Item | Value |
|---|---|
| Motor type | coreless / brushed DC motor |
| Voltage | 3.7 V (1S) or 7.4 V (2S LiPo) |
| Equivalent speed | 12,000–16,000 RPM/V direct drive |
| Current range | 3–8 A |
| Shaft diameter | 1.5 mm |
| Mount type | nose tractor |
| Propeller | 6×3 on 1S, 5×3 on 2S |

### Servos and Radio

- **Servos:** 2× 3.7 g micro servos
- **Receiver:** 4-channel micro receiver or Arduino Nano + NRF24L01
- **Mixing:** none required; this is a standard rudder/elevator layout

### Control Surface Sizing

| Surface | Size |
|---|---:|
| Elevator chord | 24.5 mm |
| Elevator span | 208.8 mm |
| Rudder height | 40.0 mm |
| Rudder chord | 91.3 mm |

### PID Guidance in Repository

| Axis | Kp | Ki | Kd |
|---|---:|---:|---:|
| Elevator / pitch | 1.20 | 0.03 | 0.08 |
| Rudder / yaw | 0.90 | 0.02 | 0.05 |

The repository includes reference Arduino sketches for the transmitter and receiver in
[`community_builds.py`](community_builds.py).

---

## 3D Printing Specification

### Printable Parts

| Part | Qty | Material | Infill | Notes |
|---|---:|---|---:|---|
| fuselage_nose | 1 | PLA or PETG | 40% | motor mount section |
| fuselage_mid | 1 | PLA | 15% | centre body with wing saddle |
| fuselage_tail | 1 | PLA | 15% | tapered tail boom section |
| wing_half | 2 | PLA | 10% | flat-bottom semi-wing |
| horizontal_stabilizer | 1 | PLA | 10% | flat plate |
| vertical_fin | 1 | PLA | 10% | flat plate |

### Recommended Print Setup

| Setting | Recommendation |
|---|---|
| Nozzle | 0.4 mm |
| Layer height | 0.2 mm |
| Walls | 2–3 perimeters |
| Top/bottom | 3–4 layers |
| Infill | use part-specific values above |
| Cooling | moderate for PLA |
| Supports | only where slicer preview shows overhang risk |
| Adhesion | skirt or brim for thin tail parts |

### Material Notes

- **PLA:** easiest to print and lightest practical option for this build
- **PETG:** better heat resistance for nose or motor-adjacent parts
- **Avoid heavy settings:** excess infill or thick walls will move weight upward quickly on a 116 g design

### Weight Target

The modeled all-up weight is **116 g**, so keep printed parts light:

- use the listed infill values
- avoid unnecessary supports
- avoid oversized glue joints
- check printed part mass before final assembly

---

## Python Generator

This repository now includes a dedicated generator:

```bash
cd /tmp/workspace/smaruf/python-study/remote-aircraft
PYTHONPATH=. python examples/generate_3d_printed_stick_plane.py
```

What it does:

1. loads the stick-plane design data
2. prints a complete build summary
3. prints every part specification
4. generates STL files when CadQuery is installed
5. falls back to specification-only mode when CadQuery is unavailable

Generated STL files are written to:

```text
output/community_builds/stick_plane/
```

Expected generated files:

- `wing_half.stl`
- `fuselage_nose.stl`
- `fuselage_mid.stl`
- `fuselage_tail.stl`
- `horizontal_stabilizer.stl`
- `vertical_fin.stl`

---

## Assembly Guide

### 1. Print the Parts

Print:

- 2× wing halves
- 1× fuselage nose
- 1× fuselage mid
- 1× fuselage tail
- 1× horizontal stabilizer
- 1× vertical fin

### 2. Prepare the Wing

1. clean the spar slot
2. dry-fit the spar
3. glue both wing halves symmetrically
4. verify incidence is equal on both sides

### 3. Assemble the Fuselage

1. join nose to fuselage mid
2. join fuselage mid to fuselage tail
3. keep the centreline straight while adhesive cures
4. test-fit wing and tail before final bonding

### 4. Install the Tail

1. centre the horizontal stabilizer
2. align the vertical fin at 90°
3. verify no twist before glue sets

### 5. Install Electronics

1. mount motor in the nose
2. install elevator and rudder servos
3. route pushrods without binding
4. install receiver / controller
5. install battery so CG can be adjusted

### 6. Balance the Aircraft

- target **33 mm from the leading edge**
- if tail-heavy, move battery forward or add nose ballast
- verify CG with the final propeller and battery installed

---

## Electronics and Hardware Checklist

- DC brushed motor
- propeller matched to voltage
- 1S 300–500 mAh LiPo or light 2S setup
- 2× micro servos
- receiver or Arduino Nano + NRF24L01
- MOSFET or brushed ESC for throttle
- pushrods and control horns
- 4 mm carbon rod or equivalent wing spar
- CA glue / epoxy

---

## Flight and Tuning Notes

- keep the maiden flight in calm air
- start with conservative throws
- use rudder to initiate turns and elevator to hold the arc
- if pitch feels sluggish, increase elevator Kp slightly
- if the model feels tail-heavy, correct CG before changing trim
- 3-channel aircraft should be trimmed patiently; they do not mask imbalance well

Repository tuning notes:

1. CG at 33 mm from LE is critical
2. 6×3 prop on 1S is the gentle baseline setup
3. this is a 3-channel design with no ailerons
4. increase elevator proportional gain if pitch response is slow

---

## Repository Commands

### Run the full analysis

```bash
cd /tmp/workspace/smaruf/python-study/remote-aircraft
PYTHONPATH=. python examples/community_builds_analysis.py
```

### Generate the dedicated 3D-print build

```bash
cd /tmp/workspace/smaruf/python-study/remote-aircraft
PYTHONPATH=. python examples/generate_3d_printed_stick_plane.py
```

### Run all example scripts

```bash
cd /tmp/workspace/smaruf/python-study/remote-aircraft
bash examples/run_all.sh
```

---

## Source Mapping

This document is based on repository code in:

- [`community_builds.py`](community_builds.py) — `stick_plane_dc_design()`
- [`community_builds.py`](community_builds.py) — `stick_plane_dc_arduino_sketch()`
- [`community_builds.py`](community_builds.py) — `generate_stick_plane_stl()`
- [`../examples/generate_3d_printed_stick_plane.py`](../examples/generate_3d_printed_stick_plane.py)

---

## External References

- Original placeholder source: <https://www.instructables.com/3D-Printed-UAV-Flying-3D-Printed/>
- Repository build reference: 3JWings — *Make Rc Plane With dc Motor | DIY Rc Stick Plane*
- External STL link noted by the repository model: <https://shorturl.at/ZFULq>
