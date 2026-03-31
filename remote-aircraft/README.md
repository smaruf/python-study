[← Back to Python Study Repository](../README.md)

# Remote Aircraft: FPV Drone & Fixed-Wing Design System

**Professional parametric CAD system + comprehensive course for designing, building, and flying FPV drones and fixed-wing aircraft.**

This is a complete, production-ready repository combining:
- 🐍 Python-based parametric CAD (CadQuery)
- 🎓 Hands-on multirotor and fixed-wing courses
- 📐 Engineering analysis tools (structural & aerodynamic)
- 🖨️ 3D printing + hybrid construction methods
- ✈️ Flight-ready designs for multirotors and fixed-wing UAVs

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to repository
cd remote-aircraft

# Install dependencies (for STL generation)
pip install -r requirements.txt
```

**Note:** CadQuery installation can be tricky. See [USAGE.md](USAGE.md) for detailed installation instructions.

### 2. Wind Tunnel Simulation (NEW! 🌪️)

Test and optimize your aircraft designs with comprehensive aerodynamic analysis:

```bash
# Interactive mode - guided design input
python aircraft_designer_cli.py --interactive

# Quick analysis - immediate results
python aircraft_designer_cli.py -w 1000 -c 150 --weight 1000

# Batch mode - process multiple designs
python aircraft_designer_cli.py --batch design.json --output results.json

# Or use the GUI wind tunnel
python airframe_designer.py
# Then click "🌪️ Wind Tunnel" button in any designer
```

**Features:**
- Lift, drag, and moment calculations
- Stall speed analysis
- Stability evaluation
- Performance optimization (best L/D)
- Angle of attack sweep analysis
- Trim condition calculation

See [WIND_TUNNEL_GUIDE.md](WIND_TUNNEL_GUIDE.md) for complete documentation.

### 3. Use the GUI Designer

```bash
# Launch the Airframe Designer GUI
python airframe_designer.py
```

This opens a graphical interface where you can:
- Design Fixed Wing Aircraft or Gliders
- Enter custom parameters
- **Run wind tunnel simulations** (NEW! 🌪️)
- Generate foamboard cutting templates
- Create 3D print specifications

See [AIRFRAME_DESIGNER_README.md](AIRFRAME_DESIGNER_README.md) for complete GUI documentation.

### 4. Generate Parts Programmatically

```bash
# Generate all default parts (if CadQuery installed)
python export_all.py

# Or run analysis examples (no CadQuery required)
PYTHONPATH=. python examples/weight_calc.py
PYTHONPATH=. python examples/stress_analysis.py

# Fixed-wing aircraft analysis
PYTHONPATH=. python examples/fixed_wing_analysis.py

# Advanced wing types analysis (NEW! ✈️)
PYTHONPATH=. python examples/wing_types_analysis.py
```

### 5. Community Builds (NEW! 🛸)

Replicate three real YouTube RC builds with full Python analysis, PID control system, foamboard/3D-print plans, and Arduino C++ sketches:

```bash
# Full analysis: Flying-Wing RC + DC Stick Plane + Shahed study
PYTHONPATH=. python examples/community_builds_analysis.py
```

```python
from fixed_wing.community_builds import (
    flying_wing_rc_design,        # RCMakerLab Oct 2025
    stick_plane_dc_design,        # 3JWings Jan 2025
    shahed_drone_design,          # educational study
    flying_wing_rc_arduino_sketch,
    flying_wing_rc_foamboard_plan,
    elevon_mix,
    PIDController,
    simulate_step_response,
)
```

### 5b. Microcontroller Firmware & Component Hardware (NEW! 🔌)

Full Raspberry Pi Pico (MicroPython) and Arduino Nano (C++) firmware for all three builds, plus component-level BOMs and ASCII wiring diagrams:

```bash
# Full hardware + firmware analysis
PYTHONPATH=. python examples/microcontroller_analysis.py
```

```python
from fixed_wing.microcontroller_firmware import (
    # Component BOMs (every individual part with value + notes)
    component_bom_flying_wing,
    component_bom_stick_plane,
    component_bom_shahed_study,

    # ASCII wiring diagrams (Arduino and RPi Pico)
    wiring_diagram_arduino,
    wiring_diagram_pico,

    # Raspberry Pi Pico MicroPython firmware (TX + RX, ready to flash)
    pico_flying_wing_firmware,
    pico_stick_plane_firmware,
    pico_shahed_study_firmware,

    # Arduino C++ firmware with MPU-6050 IMU (TX + RX, ready to flash)
    arduino_flying_wing_full,
    arduino_stick_plane_full,
    arduino_shahed_study_full,
)

# Get Pico firmware and print (copy to main.py → flash with Thonny)
fw = pico_flying_wing_firmware()
print(fw["transmitter_firmware"])   # TX Pico
print(fw["receiver_firmware"])       # RX Pico

# Get Arduino firmware and print (copy to .ino → flash with Arduino IDE)
fw = arduino_flying_wing_full()
print(fw["transmitter_sketch"])     # TX Nano
print(fw["receiver_sketch"])        # RX Nano
```

### 6. Community Projects (NEW! 🛸)

Real-world maker projects from the community — full documentation, BOM, and firmware skeletons:

| Project | Type | Speed / Weight |
|---------|------|---------------|
| [ESP-BLAST Rocket Drone](projects/esp-blast-rocket-drone/README.md) | Micro quad | 100+ km/h, ~136 g |

See [`projects/README.md`](projects/README.md) for the full project index.

### 7. Start the Course

See [`course/README.md`](course/README.md) for the complete 1-week practical course.

---

## 📦 Repository Structure

```
remote-aircraft/
├── README.md                    # This file
├── USAGE.md                     # Detailed usage examples
├── AIRFRAME_DESIGNER_README.md  # GUI Designer documentation
├── WIND_TUNNEL_GUIDE.md         # Wind tunnel simulation guide (NEW! 🌪️)
├── airframe_designer.py         # GUI application for aircraft design
├── aircraft_designer_cli.py     # CLI tool for design & simulation (NEW! 🌪️)
├── wind_tunnel.py               # Wind tunnel simulation engine (NEW! 🌪️)
├── wind_tunnel_window.py        # Wind tunnel GUI window (NEW! 🌪️)
├── requirements.txt             # Python dependencies
├── materials.py                 # Material properties database
├── export_all.py               # Generate all default parts
│
├── parts/                      # Parametric component designs
│   ├── motor_mount.py          # Motor mounting plates
│   ├── arm.py                  # Quadcopter arms
│   ├── camera_mount.py         # FPV camera mounts
│   └── battery_tray.py         # Battery holders
│
├── frames/                     # Complete frame assemblies
│   └── quad_frame.py           # Quadcopter frame generator
│
├── analysis/                   # Engineering calculations
│   ├── weight.py               # Weight calculations
│   ├── cg.py                   # Center of gravity
│   └── stress.py               # Stress analysis
│
├── fixed_wing/                 # Fixed-Wing Aircraft Design
│   ├── wing_rib.py            # Parametric wing ribs
│   ├── spar.py                # Spar design & load calcs
│   ├── fuselage.py            # Fuselage sections
│   ├── tail.py                # Tail components
│   └── loads.py               # Aerodynamic loads
│
├── course/                     # Practical Courses
│   ├── README.md               # Complete 7-day curriculum
│   ├── foam-board-templates.md # Build templates
│   ├── 3d-printing-guide.md    # Print settings & materials
│   ├── electronics-wiring-guide.md # Wiring & configuration
│   └── fixed-wing-design.md    # Fixed-wing design guide
│
├── examples/                   # Runnable examples
│   ├── weight_calc.py          # Weight & CG calculator
│   ├── stress_analysis.py      # Stress analysis
│   ├── generate_motor_mounts.py # Custom motor mounts
│   ├── fixed_wing_analysis.py  # Fixed-wing load analysis
│   └── generate_fixed_wing.py  # Generate fixed-wing STLs
│
├── guidelines/                 # Engineering & Build Guidelines
│
├── projects/                   # Community Projects (NEW! 🛸)
│   ├── README.md               # Project index
│   └── esp-blast-rocket-drone/ # ESP32 100 km/h rocket drone
│       ├── README.md           # Overview, specs, build guide
│       ├── bom.md              # Full Bill of Materials
│       └── firmware/           # Firmware skeletons (MicroPython + C)
│
└── output/                     # Generated STL files
    └── *.stl
```

### Sub-directory Documentation

- **[Course](course/README.md)**: Complete 7-day FPV and fixed-wing curriculum
- **[Examples](examples/README.md)**: Runnable example scripts and analysis tools
- **[Guidelines](guidelines/README.md)**: Engineering and build guidelines for RC aircraft
- **[Projects](projects/README.md)**: Real-world community project builds

---

## ✨ Features

### 🌪️ Wind Tunnel Simulation (NEW!)
- **Comprehensive Aerodynamic Analysis**: Lift, drag, moment calculations
- **Performance Prediction**: Stall speed, cruise speed, best glide ratio
- **Stability Evaluation**: Longitudinal static stability analysis
- **Pressure Distribution**: Visualize airflow over wing surfaces
- **CLI & GUI Interfaces**: Command-line tool and integrated GUI
- **Batch Processing**: Analyze multiple designs automatically
- **Design Optimization**: Automatic recommendations based on analysis
- **Export Results**: Save simulation data for documentation

### 🖥️ GUI Airframe Designer
- **Interactive Design**: User-friendly graphical interface
- **Fixed Wing Aircraft**: Complete parametric design with motor
- **Gliders**: Optimized for unpowered flight performance
- **Wind Tunnel Integration**: Run simulations directly from GUI (NEW! 🌪️)
- **Dual Output**: Generate both foamboard templates and 3D print specs
- **Material Selection**: Choose from PLA, PETG, Nylon, or CF-Nylon
- **Design Summary**: Automatic performance calculations and recommendations
- **No Dependencies**: Uses standard Python Tkinter (no extra install needed)

### Parametric CAD Design

#### Multirotor Components
- **Motor Mounts**: Customizable for any motor size (1507 to 2810+)
- **Quadcopter Arms**: Various lengths and cross-sections
- **Camera Mounts**: Adjustable tilt angles
- **Battery Trays**: Sized for different battery capacities
- **Complete Frames**: Full quadcopter assemblies

#### Fixed-Wing Components (NEW! ✈️)
- **Wing Ribs**: Parametric airfoil profiles (Clark-Y, symmetric)
- **Fuselage Sections**: Modular semi-monocoque shells
- **Wing Mount Plates**: Reinforced connection components
- **Tail Components**: Horizontal & vertical stabilizers
- **Tail Boom Mounts**: Carbon tube integration
- **Complete UAV Design**: 1200mm wingspan electric aircraft

#### Advanced Wing Types (NEW! ✈️)
- **Delta Wings**: Triangular wings for high-speed flight and aerobatics
- **Flying Wings**: Tailless design with maximum efficiency
- **Canard Configurations**: Forward wing design with inherent stall resistance
- **Oblique Wings**: Variable sweep for speed range optimization
- **Flying Pancake**: Circular wing design for fun and demonstrations
- **Comprehensive Analysis**: Theoretical background and performance calculations
- **Construction Guides**: Detailed building principles for each type

### Engineering Analysis

#### Structural Analysis
- **Weight Calculator**: Compute part weights with different materials
- **Center of Gravity**: Find CG position for stability
- **Stress Analysis**: Calculate bending stress under load
- **Material Comparison**: Compare PLA, PETG, Nylon, CF-Nylon

#### Aerodynamic Analysis (NEW! ✈️)
- **Lift Calculations**: Wing loading and lift distribution
- **Flight Loads**: Bending moments and safety factors
- **Cruise Speed**: Performance estimation
- **Tail Sizing**: Stability analysis and recommendations
- **Spar Selection**: Structural requirements and recommendations

### 1-Week Practical Course

A complete hands-on curriculum covering:

#### **Day 1**: Introduction to FPV and Gliders
- Flight principles and aerodynamics
- FPV system basics
- Build a simple chuck glider

#### **Day 2**: Foam-Board Design Fundamentals
- Cutting and shaping techniques
- Build FPV-ready glider
- Wing and control surface design

#### **Day 3**: 3D Printing for Aircraft Parts
- CAD design with CadQuery
- Material selection (PLA, PETG, Nylon)
- Slicing and print settings

#### **Day 4**: Motor and Electronics Integration
- Power systems and wiring
- Flight controller setup
- Build 3D printed quad frame

#### **Day 5**: Assembly and Weight Optimization
- Advanced weight analysis
- CG calculation and adjustment
- Hollow arms and optimization

#### **Day 6**: Flight Testing and Tuning
- Pre-flight safety checks
- Betaflight configuration
- First flights and PID tuning

#### **Day 7**: Advanced Designs and Customization
- Wing design optimization
- Long-range configurations
- Final project builds

See [`course/README.md`](course/README.md) for full details.

### Fixed-Wing Aircraft Design (NEW! ✈️)

A comprehensive guide to designing real fixed-wing UAVs with Python and 3D printing.

**Key Topics:**
- **Wing Structure**: Ribs, spars, and airfoil design
- **Load Analysis**: Aerodynamic forces and structural requirements
- **Fuselage Design**: Semi-monocoque shell construction
- **Tail Design**: Stability and control surfaces
- **Hybrid Construction**: Combining 3D printing with traditional materials

**Design Target:**
- 1200mm wingspan electric UAV
- 1.2-1.5kg takeoff weight
- Clark-Y airfoil profile
- 15-20 m/s cruise speed

See [`course/fixed-wing-design.md`](course/fixed-wing-design.md) for the complete guide.

---

## 💡 Usage Examples

### Multirotor Examples

### Example 1: Generate Custom Motor Mount

```python
import cadquery as cq
from parts.motor_mount import motor_mount

# For 2306 motor
mount = motor_mount(
    motor_diameter=30,
    thickness=6,
    bolt_circle=16,
    bolt_hole=3
)

cq.exporters.export(mount, "output/motor_mount_2306.stl")
```

### Example 2: Calculate Weight and CG

```python
from analysis.weight import part_weight
from analysis.cg import center_of_gravity
from materials import NYLON

# Arm weight
arm_volume = 150 * 16 * 12  # mm³
weight = part_weight(arm_volume, NYLON)
print(f"Arm weight: {weight:.2f}g")

# Calculate CG
masses = [120, 150, 80]  # Battery, frame, motors
positions = [75, 75, 140]  # Positions from front
cg = center_of_gravity(masses, positions)
print(f"CG: {cg:.1f}mm from front")
```

### Example 3: Stress Analysis

```python
from analysis.stress import bending_stress

# Arm under crash load
force = 500  # grams
length = 150  # mm
inertia = (16 * 12**3) / 12  # Rectangular beam

stress = bending_stress(force, length, inertia)
print(f"Bending stress: {stress:.2f} g/mm²")
```

### Fixed-Wing Examples (NEW! ✈️)

### Example 4: Fixed-Wing Load Analysis

```python
from fixed_wing.loads import calculate_flight_loads

# Analyze a 1200mm wingspan UAV
loads = calculate_flight_loads(
    weight=1400,      # grams
    wingspan=1200,    # mm
    chord=180         # mm
)

print(f"Cruise speed: {loads['estimated_cruise_speed_ms']:.1f} m/s")
print(f"Wing loading: {loads['wing_loading_g_cm2']:.4f} g/cm²")
print(f"Lift per wing: {loads['lift_per_wing_g']:.1f} g")
```

### Example 5: Generate Wing Ribs

```python
import cadquery as cq
from fixed_wing.wing_rib import wing_rib

# Clark-Y airfoil rib
rib = wing_rib(
    chord=180,        # mm
    thickness=6,      # mm
    spar_slot=10      # mm
)

cq.exporters.export(rib, "output/wing_rib.stl")
```

### Example 6: Spar Recommendation

```python
from fixed_wing.spar import recommend_spar_type, wing_bending_load

# Calculate bending load
load = wing_bending_load(weight=1400, span=1200)
print(f"Bending moment: {load:.1f} g·mm")

# Get recommendation
spar = recommend_spar_type(wingspan=1200, weight=1400)
print(f"Recommended: {spar['type']}")
# Output: "Carbon tube (6-8mm OD)"
```

### Example 7: Advanced Wing Types (NEW! ✈️)

```python
from fixed_wing.wing_types import (
    delta_wing_design,
    flying_wing_design,
    canard_design,
    flying_pancake_design
)

# Analyze a delta wing for high-speed flight
delta = delta_wing_design(
    root_chord=400,
    wingspan=1000,
    sweep_angle=45
)
print(f"Delta wing aspect ratio: {delta['geometry']['aspect_ratio']:.2f}")

# Design a flying wing for efficiency
flying = flying_wing_design(
    center_chord=350,
    wingspan=1200,
    sweep_angle=25
)
print(f"Flying wing L/D gain: {flying['aerodynamics']['efficiency_gain']:.0%}")

# Analyze a canard configuration
canard = canard_design(
    main_wing_chord=200,
    main_wingspan=1000,
    canard_chord=80,
    canard_span=400
)
print(f"Canard stall safety: {canard['aerodynamics']['stall_safety']}")

# Design a fun flying pancake
pancake = flying_pancake_design(diameter=600)
print(f"Pancake stability: {pancake['aerodynamics']['flight_characteristics']}")
```

See [USAGE.md](USAGE.md) for more examples.

---

## 🎯 What You Can Build

### Multirotor Builds

#### Beginner
- **5" Quad**: Standard freestyle/racing quad
- **3" Micro**: Indoor/outdoor fun
- **FPV Glider**: 800mm foam-board with FPV system

#### Intermediate
- **7" Long-Range**: Extended flight time, GPS
- **Stretch-X Frame**: Better camera view
- **Custom Designs**: Fully parametric, your specs

#### Advanced
- **Competition Builds**: Racing or freestyle optimized
- **Hybrid Designs**: Custom geometries and materials

### Fixed-Wing Builds (NEW! ✈️)

#### Beginner
- **Chuck Glider**: Simple hand-launch glider
- **Small Sport Plane**: 800mm wingspan, basic trainer
- **FPV Glider**: Long flight times, stable platform

#### Intermediate
- **Electric UAV**: 1200mm wingspan (design target)
- **Camera Platform**: Aerial photography/FPV
- **Long-Range Explorer**: GPS navigation, extended range

#### Advanced
- **Delta Wing**: High-speed aerobatic platform
- **Flying Wing**: Maximum efficiency FPV platform
- **Canard Configuration**: Stall-resistant design
- **Oblique Wing**: Variable sweep experimental
- **Flying Pancake**: Fun circular wing design
- **RCMakerLab Flying-Wing**: Full foam build with Arduino NRF24L01 RC 🆕
- **3JWings DC Stick Plane**: 3D-printed beginner plane (609 mm, 116 g) 🆕
- **Shahed / Lucas Study Model**: 55° delta scaled to RC (educational) 🆕
- **Custom Airframes**: Fully parametric designs
- **Competition Models**: Optimised for performance

---

## 🖨️ 3D Printing Guide

### Recommended Settings

**For PETG (general parts):**
```
Nozzle: 0.4mm
Layer height: 0.2mm
Infill: 30-50% Gyroid
Temperature: 240°C
Bed: 80°C
```

**For Nylon (strength):**
```
Nozzle: 0.6mm hardened
Layer height: 0.28mm
Infill: 30% Gyroid
Temperature: 255°C
Bed: 85°C
Enclosure: Recommended
```

See [`course/3d-printing-guide.md`](course/3d-printing-guide.md) for complete material guide.

### Fixed-Wing Construction (Hybrid Approach) ✈️

**❌ Don't do this:**
- Full printed wings (too heavy and flexible)
- Printed spars for large aircraft (will fail)
- Heavy tail components (CG problems)

**✅ Best practice (Hybrid Construction):**

**Wings:**
```
1. Print ribs only (PLA/PETG, 30% infill)
2. Use carbon tube spar (6-8mm OD)
3. Cover with foam/balsa + heat-shrink film
```

**Fuselage:**
```
Material: Nylon/PETG
Wall thickness: 2mm
Infill: 30%
Reinforcement: Double thickness at wing mount
```

**Tail:**
```
Boom: Carbon tube (8mm) - NOT printed
Mounts: Print brackets only (Nylon)
Stabilizers: Foam core + film covering
```

See [`course/fixed-wing-design.md`](course/fixed-wing-design.md) for complete build guide.

---

## 🎓 Course Materials

All course materials are included:

- **[Main Course](course/README.md)**: Complete 7-day FPV curriculum
- **[Foam-Board Templates](course/foam-board-templates.md)**: Cut patterns and assembly
- **[3D Printing Guide](course/3d-printing-guide.md)**: Materials, settings, troubleshooting
- **[Electronics Wiring](course/electronics-wiring-guide.md)**: Complete wiring diagrams
- **[Fixed-Wing Design](course/fixed-wing-design.md)**: Complete fixed-wing UAV design guide
- **[Wing Types Guide](course/wing-types-guide.md)**: Traditional and advanced wing designs (straight, swept, delta, flying-wing, canard, oblique, pancake) (NEW! ✈️)

---

## 📊 Technical Specifications

### Multirotor Specs

#### Supported Motor Sizes
- 1507 (Tiny Whoop)
- 2204 (4" racing)
- 2306 (5" freestyle)
- 2806 (7" long range)
- Custom sizes via parameters

#### Frame Sizes
- 120mm arms (3" micro)
- 150mm arms (5" standard)
- 180mm arms (7" long range)
- Custom sizes programmable

### Fixed-Wing Specs (NEW! ✈️)

#### Design Target UAV
- Wingspan: 800-1500mm (parametric)
- Wing chord: 150-200mm
- Airfoil: Clark-Y (simplified for printing)
- Cruise speed: 15-20 m/s
- Takeoff weight: 1.0-2.0 kg

#### Supported Components
- Wing ribs (multiple chord sizes)
- Fuselage sections (modular)
- Wing mount plates (reinforced)
- Horizontal stabilizers
- Vertical stabilizers
- Tail boom mounts (for carbon tube)

### Materials Database
- PLA: 1.24 g/cm³
- PETG: 1.27 g/cm³
- Nylon: 1.15 g/cm³
- CF-Nylon: 1.20 g/cm³

---

## 🔧 Requirements

### Software
- Python 3.9+
- CadQuery (for STL generation)
- Betaflight Configurator (for flight controller)
- Slicer software (Cura, PrusaSlicer, etc.)

### Hardware (for builds)
- 3D printer or foam-board
- Soldering iron
- Basic hand tools
- FPV electronics (see course materials)

---

## 📚 Learning Path

### Complete Beginner
1. Read [course/README.md](course/README.md) introduction
2. Build Day 1 chuck glider
3. Run `examples/weight_calc.py` to understand analysis
4. Progress through course days 2-7

### Some FPV Experience
1. Review [course/3d-printing-guide.md](course/3d-printing-guide.md)
2. Generate parts with `export_all.py`
3. Build 5" quad following Days 4-6
4. Customize designs using examples

### Experienced Builder
1. Review CAD code in `parts/` and `frames/`
2. Modify parameters for custom builds
3. Use analysis tools for optimization
4. Design completely new parts

---

## 🤝 Contributing

Improvements welcome! Areas of interest:
- Additional part designs
- Course material improvements
- More analysis tools
- Build documentation
- Example projects

---

## 📝 License

This project is open source. Feel free to use, modify, and share.

---

## 🎯 Project Goals

This repository provides:

✅ **Complete parametric CAD system** for multirotors and fixed-wing aircraft  
✅ **Production-ready parts** that actually fly  
✅ **Engineering analysis** for safe, optimized builds (structural + aerodynamic)  
✅ **Comprehensive courses** from zero to flight (FPV + fixed-wing)  
✅ **Multiple construction methods** (3D printing, foam-board, hybrid)  
✅ **Real aerospace engineering** adapted for solo builders  
✅ **Real-world tested** designs and techniques

---

## 🚁 Ready to Build?

### For Multirotors:
1. **Start the course**: [`course/README.md`](course/README.md)
2. **Generate parts**: `python export_all.py`
3. **Run examples**: `PYTHONPATH=. python examples/weight_calc.py`
4. **Build and fly!**

### For Fixed-Wing: ✈️
1. **Read the guide**: [`course/fixed-wing-design.md`](course/fixed-wing-design.md)
2. **Analyze your design**: `PYTHONPATH=. python examples/fixed_wing_analysis.py`
3. **Generate components**: `PYTHONPATH=. python examples/generate_fixed_wing.py`
4. **Build hybrid** (3D print + carbon + foam/film)
5. **Test and fly!**

Happy building! ✈️

---

## 🛸 Notable Aircraft References

### Shahed / Lucas Drone

The **Shahed** (also referred to as **Lucas** or **Geranium** in some operational contexts) is a delta-wing loitering munition / kamikaze UAV. It is widely studied in the aeromodelling community for its simple yet effective flying-wing aerodynamics.

**Key characteristics:**
- **Wing type**: Delta / Flying-Wing (tailless)
- **Propulsion**: Pusher propeller driven by a small internal-combustion or electric engine
- **Navigation**: GPS-waypoint guidance with IR/optical terminal homing
- **Construction**: Largely plywood/composite frame — translates well to foam-board DIY builds
- **Relevance to this repo**: Its planform is essentially a swept delta wing and can be modelled using the `delta_wing_design()` and `flying_wing_design()` functions already present in `fixed_wing/wing_types.py`

**DIY / study approximation with this toolkit:**

```python
from fixed_wing.wing_types import delta_wing_design

# Approximate Shahed-136 planform (scaled down for RC)
shahed_approx = delta_wing_design(
    root_chord=400,    # mm — scaled-down centre chord
    wingspan=900,      # mm — scaled-down span
    sweep_angle=55     # degrees — characteristic high-sweep delta
)
print(f"Aspect ratio : {shahed_approx['geometry']['aspect_ratio']:.2f}")
print(f"Wing area    : {shahed_approx['geometry']['wing_area_mm2']:.0f} mm²")
```

**Full implementation** — geometry, aerodynamics, PID control system, foamboard cut plan, and STL generation:

```python
from fixed_wing.community_builds import (
    shahed_drone_design,
    generate_shahed_study_stl,   # requires CadQuery
)

study = shahed_drone_design(scale_factor=0.20)
rc    = study["rc_study_model"]
ctrl  = study["control_system"]

print(f"RC model span  : {rc['wingspan_mm']} mm")
print(f"RC model AUW   : {rc['auw_grams']} g")
print(f"CG from LE     : {rc['cg_from_le_mm']} mm")
print(f"Cruise speed   : {rc['cruise_speed_ms']} m/s")
print(f"Elevon chord   : {ctrl['elevon_chord_mm']} mm")
print(f"PID Pitch Kp   : {ctrl['pid_pitch']['kp']}")

# Same elevon-mixing Arduino sketch as the flying-wing (change address string)
from fixed_wing.community_builds import flying_wing_rc_arduino_sketch
sketches = flying_wing_rc_arduino_sketch()
print(sketches["receiver_sketch"])
```

> ⚠️ **Note**: This reference is provided purely for educational aerodynamics study. Always comply with local laws and regulations when building and flying any UAV.

---

## 🎬 Recommended Build Videos

### 1. Build a Flying-Wing with Simple Materials — DIY RC Plane & Remote Control

> **Channel**: RCMakerLab (491K subscribers)  
> **Published**: Oct 9, 2025  
> **Views**: 118,594  
> **YouTube**: Search *"RCMakerLab Build Flying-Wing Simple Materials"* on YouTube

**Description:**  
Demonstrates how to build a delta-wing type RC model airplane and a handmade remote control using Arduino. Uses the most basic materials possible. Delta-wing (flying-wing) models are preferred for their simple construction and excellent stability.

**Build tags:** `#rcplane` `#diyrc` `#homemadercplane`

**Chapter timestamps:**
| Time  | Topic |
|-------|-------|
| 00:00 | RC Flight-Wing overview |
| 00:06 | Material selection for RC Flying-Wing |
| 00:26 | Building the flying-wing |
| 05:25 | RC Servo assembly |
| 07:52 | Making the Remote Control Circuit (Arduino) |
| 10:57 | Cardboard motor mount |
| 11:14 | Brushless motor assembly |
| 12:08 | ESC and Radio setup |
| 14:50 | Center of Gravity (CG) |
| 14:52 | First flight of RC Delta-Wing |

**Electronics / Parts list:**
| Component | Notes |
|-----------|-------|
| 2205 2300KV Motor (CW) | Brushless motor |
| 30A BL ESC | Electronic Speed Controller |
| 5050 or 5045 3-Blade Propeller | |
| MG90S Servo | |
| NRF24L01+PA+LNA 100mW (E01-ML01DP5) | Receiver module |
| Arduino Nano V3 (Micro connector) | Flight / RC controller |
| GT-24 NRF24L01+PA+LNA (With Antenna) | Transmitter module |
| 2× PS4 Analogue Joystick (10K) | Transmitter sticks |
| 2× Toggle switch | |
| LM1117 3.3V voltage regulator | |
| Capacitors 10µF (×2), 100µF (×3), 100nF (×5) | Filtering |
| JST 2-Pin connector | |
| 6 mm insulation styrofoam (×2 plates) | Airframe body |
| 3 mm kraft foamboard (×1 plate) | Airframe structure |

**Reference links:**
- Gerber files, dimensions & code: https://www.rcpano.net/2025/09/30/making-delta-wing-rc-plane/
- Full flight video: *Handmade RC Flying-Wing FULL Flight 1*

**Python implementation:**

```python
# Full design + aerodynamics + control system + foamboard plan + Arduino sketch
from fixed_wing.community_builds import (
    flying_wing_rc_design,
    flying_wing_rc_arduino_sketch,
    flying_wing_rc_foamboard_plan,
    generate_flying_wing_rc_stl,   # requires CadQuery
    elevon_mix,
    PIDController,
    simulate_step_response,
)

design = flying_wing_rc_design()
print(f"Cruise speed : {design['aerodynamics']['cruise_speed_ms']} m/s")
print(f"CG from LE   : {design['aerodynamics']['cg_from_le_mm']} mm")
print(f"Stall speed  : {design['aerodynamics']['stall_speed_ms']} m/s")

# Get full Arduino TX + RX sketch as strings
sketches = flying_wing_rc_arduino_sketch()
print(sketches["transmitter_sketch"])   # copy to .ino and flash

# Foamboard cutting plan with ASCII layout
plan = flying_wing_rc_foamboard_plan()
print(plan["ascii_layout"])

# Simulate PID step response before flying
pid = PIDController(kp=1.80, ki=0.05, kd=0.12)
history = simulate_step_response(pid, target=30)   # 30° bank target
```

Run the full analysis:

```bash
PYTHONPATH=. python examples/community_builds_analysis.py
```

### 2. Make RC Plane With DC Motor — DIY RC Stick Plane

> **Channel**: 3JWings (235K subscribers)  
> **Published**: Jan 10, 2025  
> **Views**: 278,098  
> **YouTube**: Search *"3JWings Make Rc Plane With dc Motor DIY Rc Stick Plane"* on YouTube  
> **Tags:** `#rcplane` `#3dprinting`

**Description:**  
A beginner-friendly 3D-printed RC plane build. The simplest 3D-printed plane you can build at home — flight control is straightforward, making it ideal for newcomers to aeromodelling.

**Plane Specifications:**
| Parameter | Value |
|-----------|-------|
| Wingspan | 609 mm |
| All-up weight | 116 g (including battery) |
| CG position | 33 mm from leading edge |
| Propulsion | DC motor |

**Files & Resources:**
- STL files / Plans: https://shorturl.at/ZFULq
- Electronics setup guide: *DIY Electronics Setup For Mini RC Plane*
- 3D Printer used: Creality Ender 3 V3 Plus — https://shorturl.at/XMSSY

**Python implementation:**

```python
# Full design + aerodynamics + 3-channel control system + 3D print specs + Arduino sketch
from fixed_wing.community_builds import (
    stick_plane_dc_design,
    stick_plane_dc_arduino_sketch,
    generate_stick_plane_stl,   # requires CadQuery
)

design = stick_plane_dc_design()
print(f"Wingspan     : {design['geometry']['wingspan_mm']} mm")
print(f"AUW          : {design['aerodynamics']['auw_grams']} g")
print(f"CG from LE   : {design['aerodynamics']['cg_from_le_mm']} mm  "
      f"({design['aerodynamics']['cg_pct_mac']} % MAC)")
print(f"Cruise speed : {design['aerodynamics']['cruise_speed_ms']} m/s")
print(f"Stability    : {design['aerodynamics']['stability_assessment']}")

# 3D print specs for every part
for part, info in design["build_specification"]["print_specs"].items():
    print(part, info)

# Arduino sketch for 3-channel DC motor control
sketches = stick_plane_dc_arduino_sketch()
print(sketches["transmitter_sketch"])   # copy to .ino and flash
```

---

## 📞 Resources

- **CadQuery Docs**: https://cadquery.readthedocs.io/
- **Betaflight**: https://betaflight.com/
- **r/Multicopter**: Reddit FPV community
- **Flite Test**: Foam-board design tutorials
- **Joshua Bardwell**: FPV YouTube channel

---

**Note**: See [README_ORIGINAL.md](README_ORIGINAL.md) for the original design specification that this repository implements.
