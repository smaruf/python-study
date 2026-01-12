# Remote Aircraft: FPV Drone & Glider Design System

**Professional parametric CAD system + comprehensive 1-week practical course for designing, building, and flying FPV drones and gliders.**

This is a complete, production-ready repository combining:
- 🐍 Python-based parametric CAD (CadQuery)
- 🎓 1-week hands-on FPV course
- 📐 Engineering analysis tools
- 🖨️ 3D printing + foam-board construction
- ✈️ Flight-ready designs

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

### 2. Use the GUI Designer (New!)

```bash
# Launch the Airframe Designer GUI
python airframe_designer.py
```

This opens a graphical interface where you can:
- Design Fixed Wing Aircraft or Gliders
- Enter custom parameters
- Generate foamboard cutting templates
- Create 3D print specifications

See [AIRFRAME_DESIGNER_README.md](AIRFRAME_DESIGNER_README.md) for complete GUI documentation.

### 3. Generate Parts Programmatically

```bash
# Generate all default parts (if CadQuery installed)
python export_all.py

# Or run analysis examples (no CadQuery required)
PYTHONPATH=. python examples/weight_calc.py
PYTHONPATH=. python examples/stress_analysis.py
```

### 4. Start the Course

See [`course/README.md`](course/README.md) for the complete 1-week practical course.

---

## 📦 Repository Structure

```
remote-aircraft/
├── README.md                    # This file
├── USAGE.md                     # Detailed usage examples
├── AIRFRAME_DESIGNER_README.md  # GUI Designer documentation
├── airframe_designer.py         # GUI application for aircraft design
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
├── course/                     # 1-Week Practical Course
│   ├── README.md               # Complete 7-day curriculum
│   ├── foam-board-templates.md # Build templates
│   ├── 3d-printing-guide.md    # Print settings & materials
│   └── electronics-wiring-guide.md # Wiring & configuration
│
├── examples/                   # Runnable examples
│   ├── weight_calc.py          # Weight & CG calculator
│   ├── stress_analysis.py      # Stress analysis
│   └── generate_motor_mounts.py # Custom motor mounts
│
└── output/                     # Generated STL files
    └── *.stl
```

---

## ✨ Features

### 🖥️ GUI Airframe Designer (New!)
- **Interactive Design**: User-friendly graphical interface
- **Fixed Wing Aircraft**: Complete parametric design with motor
- **Gliders**: Optimized for unpowered flight performance
- **Dual Output**: Generate both foamboard templates and 3D print specs
- **Material Selection**: Choose from PLA, PETG, Nylon, or CF-Nylon
- **Design Summary**: Automatic performance calculations and recommendations
- **No Dependencies**: Uses standard Python Tkinter (no extra install needed)

### Parametric CAD Design
- **Motor Mounts**: Customizable for any motor size (1507 to 2810+)
- **Quadcopter Arms**: Various lengths and cross-sections
- **Camera Mounts**: Adjustable tilt angles
- **Battery Trays**: Sized for different battery capacities
- **Complete Frames**: Full quadcopter assemblies

### Engineering Analysis
- **Weight Calculator**: Compute part weights with different materials
- **Center of Gravity**: Find CG position for stability
- **Stress Analysis**: Calculate bending stress under load
- **Material Comparison**: Compare PLA, PETG, Nylon, CF-Nylon

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

---

## 💡 Usage Examples

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

See [USAGE.md](USAGE.md) for more examples.

---

## 🎯 What You Can Build

### Beginner Builds
- **Chuck Glider**: Simple hand-launch glider (Day 1)
- **FPV Glider**: 800mm foam-board with FPV system (Day 2)
- **5" Quad**: Standard freestyle/racing quad (Days 3-6)

### Intermediate Builds
- **7" Long-Range**: Extended flight time, GPS
- **Stretch-X Frame**: Better camera view
- **Hybrid Wing**: Multirotor with wings

### Advanced Builds
- **Flying Wing**: High-speed FPV platform
- **Custom Designs**: Fully parametric, your specs
- **Competition Builds**: Racing or freestyle optimized

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

---

## 🎓 Course Materials

All course materials are included:

- **[Main Course](course/README.md)**: Complete 7-day curriculum
- **[Foam-Board Templates](course/foam-board-templates.md)**: Cut patterns and assembly
- **[3D Printing Guide](course/3d-printing-guide.md)**: Materials, settings, troubleshooting
- **[Electronics Wiring](course/electronics-wiring-guide.md)**: Complete wiring diagrams

---

## 📊 Technical Specifications

### Supported Motor Sizes
- 1507 (Tiny Whoop)
- 2204 (4" racing)
- 2306 (5" freestyle)
- 2806 (7" long range)
- Custom sizes via parameters

### Frame Sizes
- 120mm arms (3" micro)
- 150mm arms (5" standard)
- 180mm arms (7" long range)
- Custom sizes programmable

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

✅ **Complete parametric CAD system** for drone/glider design
✅ **Production-ready parts** that actually fly
✅ **Engineering analysis** for safe, optimized builds
✅ **Comprehensive course** from zero to flight
✅ **Both 3D printing and foam-board** construction methods
✅ **Real-world tested** designs and techniques

---

## 🚁 Ready to Build?

1. **Start the course**: [`course/README.md`](course/README.md)
2. **Generate parts**: `python export_all.py`
3. **Run examples**: `PYTHONPATH=. python examples/weight_calc.py`
4. **Build and fly!**

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
