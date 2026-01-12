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

### 2. Generate Your First Parts

```bash
# Generate all default parts (if CadQuery installed)
python export_all.py

# Or run analysis examples (no CadQuery required)
PYTHONPATH=. python examples/weight_calc.py
PYTHONPATH=. python examples/stress_analysis.py

# Fixed-wing aircraft analysis
PYTHONPATH=. python examples/fixed_wing_analysis.py
```

### 3. Start the Course

See [`course/README.md`](course/README.md) for the complete 1-week practical course.

---

## 📦 Repository Structure

```
remote-aircraft/
├── README.md                    # This file
├── USAGE.md                     # Detailed usage examples
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
└── output/                     # Generated STL files
    └── *.stl
```

---

## ✨ Features

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
- **Flying Wing**: High-speed FPV platform
- **Custom Airframes**: Fully parametric designs
- **Competition Models**: Optimized for performance

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

## 📞 Resources

- **CadQuery Docs**: https://cadquery.readthedocs.io/
- **Betaflight**: https://betaflight.com/
- **r/Multicopter**: Reddit FPV community
- **Flite Test**: Foam-board design tutorials
- **Joshua Bardwell**: FPV YouTube channel

---

**Note**: See [README_ORIGINAL.md](README_ORIGINAL.md) for the original design specification that this repository implements.
