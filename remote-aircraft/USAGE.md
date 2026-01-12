# Usage Examples and Quick Start Guide

## Quick Start

### 1. Install Dependencies

```bash
cd remote-aircraft
pip install -r requirements.txt
```

**Note:** CadQuery can be tricky to install. If you have issues:

**Option A: Using conda (recommended)**
```bash
conda create -n cadquery
conda activate cadquery
conda install -c conda-forge -c cadquery cadquery=master
pip install numpy
```

**Option B: Using CQ-Editor (easiest for beginners)**
- Download from: https://github.com/CadQuery/CQ-editor/releases
- Provides GUI environment + all dependencies
- Works on Windows, Mac, Linux

### 2. Generate Your First Part

```bash
# Generate all default parts
python export_all.py
```

This will create STL files in the `output/` directory:
- `motor_mount.stl`
- `quad_frame_120.stl`
- `quad_frame_150.stl`
- `quad_frame_180.stl`

### 3. View STL Files

Use any STL viewer:
- **Online**: [3D Viewer Online](https://3dviewer.net/)
- **Desktop**: Cura, PrusaSlicer, Meshmixer
- **CAD**: FreeCAD, Fusion 360

---

## Examples

### Example 1: Custom Motor Mount

Create a motor mount for a specific motor:

```python
from parts.motor_mount import motor_mount
import cadquery as cq

# For a 2306 motor (30mm diameter)
mount = motor_mount(
    motor_diameter=30,
    thickness=6,
    bolt_circle=16,
    bolt_hole=3,
    shaft_hole=6
)

# Export to STL
cq.exporters.export(mount, "output/motor_mount_2306.stl")

print("Motor mount created!")
```

**Run it:**
```bash
python -c "from parts.motor_mount import motor_mount; import cadquery as cq; cq.exporters.export(motor_mount(motor_diameter=30, thickness=6), 'output/motor_mount_2306.stl')"
```

---

### Example 2: Calculate Weight and CG

```python
from analysis.weight import part_weight
from analysis.cg import center_of_gravity
from materials import PETG, NYLON

# Calculate weight of frame parts
motor_mount_volume = 3.14 * (14**2) * 5  # Approximate cylinder
mount_weight = part_weight(motor_mount_volume, PETG)
print(f"Motor mount weight: {mount_weight:.2f}g")

# Calculate arm weight
arm_volume = 150 * 16 * 12  # mm³
arm_weight = part_weight(arm_volume, NYLON)
print(f"One arm weight: {arm_weight:.2f}g")
print(f"Four arms weight: {arm_weight * 4:.2f}g")

# Calculate quad center of gravity
components = {
    "Camera": (15, 20),      # (weight_g, position_mm)
    "Battery": (120, 75),
    "FC": (10, 80),
    "VTX": (8, 85),
    "Motors": (120, 140),     # 4x 30g motors
}

masses = [w for w, _ in components.values()]
positions = [p for _, p in components.values()]

cg = center_of_gravity(masses, positions)
print(f"\nCenter of Gravity: {cg:.1f}mm from front")
print(f"Recommended CG: 70-80mm (for 150mm arms)")
```

**Run it:**
```bash
python examples/weight_calc.py
```

---

### Example 3: Custom Quad Size

Generate a 7" long-range quad frame:

```python
from frames.quad_frame import quad_frame
import cadquery as cq

# Generate 7" frame (180mm arms)
frame_7inch = quad_frame(arm_length=180)

# Export
cq.exporters.export(
    frame_7inch,
    "output/quad_frame_7inch.stl"
)

print("7-inch frame created!")
print("Supports: 7\" props")
print("Motor size: 2806-2810")
print("Battery: 4S 2200-3000mAh")
```

---

### Example 4: Parametric Design Exploration

Create multiple variants at once:

```python
import cadquery as cq
from parts.motor_mount import motor_mount

# Generate mounts for different motor sizes
motor_sizes = {
    "1507": 15,  # Tiny whoop
    "2204": 22,  # 4" quad
    "2306": 23,  # 5" quad
    "2806": 28,  # 7" quad
}

for name, diameter in motor_sizes.items():
    mount = motor_mount(
        motor_diameter=diameter,
        thickness=5 if diameter < 25 else 6,
        bolt_circle=12 if diameter < 20 else 16,
    )
    
    filename = f"output/motor_mount_{name}.stl"
    cq.exporters.export(mount, filename)
    print(f"Created: {filename}")
```

---

### Example 5: Using CQ-Editor (Interactive)

If you installed CQ-Editor:

1. **Open CQ-Editor**
2. **Create new file** or open existing (e.g., `parts/motor_mount.py`)
3. **Edit parameters** at the bottom:
   ```python
   # Show the part
   show_object(motor_mount(motor_diameter=28, thickness=5))
   ```
4. **Press F5** to render
5. **Adjust parameters** and re-render in real-time
6. **Export** when satisfied

---

## Advanced Examples

### Example 6: Hollow Arms for Weight Savings

```python
import cadquery as cq

def hollow_arm(length=150, width=16, height=12, wall=3):
    """Create hollow arm for weight reduction"""
    outer = (
        cq.Workplane("XY")
        .rect(width, height)
        .extrude(length)
    )
    
    inner = (
        cq.Workplane("XY")
        .rect(width - 2*wall, height - 2*wall)
        .extrude(length)
    )
    
    hollow = outer.cut(inner).edges("|Z").fillet(2)
    return hollow

# Compare weights
from analysis.weight import part_weight
from materials import NYLON

solid_volume = 150 * 16 * 12
hollow_volume = solid_volume - (150 * 10 * 6)  # Approximate

solid_weight = part_weight(solid_volume, NYLON)
hollow_weight = part_weight(hollow_volume, NYLON)

print(f"Solid arm: {solid_weight:.2f}g")
print(f"Hollow arm: {hollow_weight:.2f}g")
print(f"Weight savings: {solid_weight - hollow_weight:.2f}g ({100*(1-hollow_weight/solid_weight):.1f}%)")

# Export
arm = hollow_arm()
cq.exporters.export(arm, "output/arm_hollow.stl")
```

---

### Example 7: Camera Mount with Tilt Angle

```python
import cadquery as cq

def adjustable_camera_mount(width=20, height=20, thickness=3, angle=25):
    """Camera mount with adjustable tilt"""
    base = (
        cq.Workplane("XY")
        .rect(width + 10, thickness + 5)
        .extrude(5)
        .edges("|Z")
        .fillet(1)
    )
    
    mount = (
        cq.Workplane("XY")
        .workplane(offset=5)
        .rect(width, thickness)
        .extrude(height)
        .edges("|Z")
        .fillet(1)
        .rotate((0,0,0), (1,0,0), angle)  # Tilt forward
    )
    
    return base.union(mount)

# Create mounts at different angles
for angle in [0, 15, 25, 35, 45]:
    mount = adjustable_camera_mount(angle=angle)
    cq.exporters.export(
        mount,
        f"output/camera_mount_{angle}deg.stl"
    )
    print(f"Created camera mount at {angle}° tilt")
```

---

## Integration with Slicer

### PrusaSlicer / Cura Settings

After generating STL files, import into your slicer:

**For PETG (frame parts):**
```
Material: PETG
Nozzle: 0.4mm (or 0.6mm for faster prints)
Layer height: 0.2mm (0.28mm with 0.6mm nozzle)
Infill: 30-50% Gyroid
Perimeters: 4
Top/bottom layers: 5
Print speed: 50mm/s
Supports: Usually not needed
```

**For Nylon (high-strength parts):**
```
Material: Nylon
Nozzle: 0.6mm hardened steel
Layer height: 0.28mm
Infill: 30% Gyroid
Perimeters: 3-4
Dry filament before printing!
Enclosure recommended
```

---

## Troubleshooting

### "Module not found: cadquery"
```bash
# Try conda install
conda install -c conda-forge -c cadquery cadquery

# Or use CQ-Editor (includes everything)
```

### "Can't export STL"
```python
# Make sure you're using the exporter correctly
import cadquery as cq

part = motor_mount()  # Your part function
cq.exporters.export(part, "output/part.stl")  # Note: exporters, not exporter
```

### "Parts look wrong in slicer"
- Check units: Our parts are in millimeters
- Verify scale is 1:1 in slicer
- Check orientation (may need to rotate)

---

## Learning Resources

### CadQuery Documentation
- **Official Docs**: https://cadquery.readthedocs.io/
- **Examples**: https://github.com/CadQuery/cadquery/tree/master/examples
- **Forum**: https://github.com/CadQuery/cadquery/discussions

### FPV Design
- See `course/README.md` for complete 1-week course
- See `course/foam-board-templates.md` for build templates
- See `course/3d-printing-guide.md` for material selection

### Community
- **r/Multicopter**: Reddit community
- **Flite Test Forums**: Foam-board designs
- **IntFPV**: FPV racing community

---

## Project Structure Reference

```
remote-aircraft/
├── README.md                    ← Overview and philosophy
├── requirements.txt             ← Python dependencies
├── materials.py                 ← Material properties
├── export_all.py               ← Generate all default parts
│
├── parts/                      ← Individual component designs
│   ├── motor_mount.py
│   ├── arm.py
│   ├── camera_mount.py
│   └── battery_tray.py
│
├── frames/                     ← Complete frame assemblies
│   └── quad_frame.py
│
├── analysis/                   ← Engineering calculations
│   ├── weight.py
│   ├── cg.py
│   └── stress.py
│
├── course/                     ← 1-week practical course
│   ├── README.md               ← Main course content
│   ├── foam-board-templates.md
│   ├── 3d-printing-guide.md
│   └── electronics-wiring-guide.md
│
└── output/                     ← Generated STL files (ignored by git)
    └── *.stl
```

---

## Next Steps

1. **Try the examples above** - Start with Example 1
2. **Modify parameters** - Change sizes to fit your needs
3. **Print a part** - Start with motor_mount.stl
4. **Follow the course** - See `course/README.md`
5. **Build your quad** - Apply what you learned!

Happy building! 🚁
