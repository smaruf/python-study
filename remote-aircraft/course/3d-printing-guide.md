# 3D Printing Guide for FPV Parts

## Material Selection Guide

### PLA (Polylactic Acid)
**Best for**: Prototypes, non-stressed parts, learning

| Property | Value |
|----------|-------|
| Strength | Medium |
| Flexibility | Low |
| Temperature resistance | Low (60°C) |
| Ease of printing | Very Easy |
| Cost | Low |
| UV resistance | Poor |

**Use cases:**
- Camera mounts (indoor/covered)
- Battery trays (light loads)
- Antenna holders
- Prototyping parts

**Print settings:**
```
Nozzle temperature: 200-220°C
Bed temperature: 60°C
Print speed: 60 mm/s
Retraction: 5mm at 45 mm/s
```

---

### PETG (Polyethylene Terephthalate Glycol)
**Best for**: Flight parts, outdoor use, durable components

| Property | Value |
|----------|-------|
| Strength | High |
| Flexibility | Medium |
| Temperature resistance | Medium (80°C) |
| Ease of printing | Medium |
| Cost | Medium |
| UV resistance | Good |

**Use cases:**
- Motor mounts
- Frame parts
- Camera protection
- Battery trays (high loads)
- Landing gear

**Print settings:**
```
Nozzle temperature: 230-250°C
Bed temperature: 80°C
Print speed: 50 mm/s
Retraction: 3mm at 40 mm/s
Fan: 50% after first layer
```

**Tips:**
- Clean bed with isopropyl alcohol
- Slight stringing is normal
- First layer crucial for adhesion

---

### Nylon (Polyamide)
**Best for**: High-stress parts, arms, structural components

| Property | Value |
|----------|-------|
| Strength | Very High |
| Flexibility | High |
| Temperature resistance | High (100°C) |
| Ease of printing | Hard |
| Cost | High |
| UV resistance | Excellent |

**Use cases:**
- Quad arms
- Motor mounts (high vibration)
- Landing gear
- Impact zones

**Print settings:**
```
Nozzle temperature: 245-265°C
Bed temperature: 70-90°C
Print speed: 30-40 mm/s
Enclosure: Required
Drying: 4+ hours at 70°C before printing
```

**Tips:**
- Nylon absorbs moisture - MUST dry before use
- Enclosed printer recommended
- Warping common - use brim/raft
- Very strong layer adhesion

---

### CF-Nylon (Carbon Fiber Nylon)
**Best for**: Maximum strength, racing quads

| Property | Value |
|----------|-------|
| Strength | Extreme |
| Flexibility | Medium-Low |
| Temperature resistance | Very High (120°C) |
| Ease of printing | Very Hard |
| Cost | Very High |
| UV resistance | Excellent |

**Use cases:**
- Racing quad arms
- High-impact areas
- Load-bearing structures

**Print settings:**
```
Nozzle temperature: 260-280°C
Bed temperature: 90-100°C
Print speed: 30 mm/s
Nozzle: Hardened steel (0.6mm recommended)
```

**Tips:**
- Requires hardened nozzle (brass wears out)
- Very abrasive
- Excellent strength-to-weight ratio
- Minimal warping vs pure Nylon

---

## Part-Specific Settings

### Motor Mounts

**Critical requirements:**
- High temperature resistance
- Vibration dampening
- Strong bolt-hole threads

**Recommended material:** PETG or Nylon

**Settings:**
```
Layer height: 0.2mm
Infill: 50% Gyroid
Perimeters: 4
Top/bottom layers: 5
Orientation: Flat (layers perpendicular to motor thrust)
Supports: None needed for provided designs
```

**Post-processing:**
- Drill out bolt holes to exact size (3.0mm for M3)
- Tap threads if using metal screws repeatedly
- Consider heat-set inserts for frequent disassembly

---

### Quadcopter Arms

**Critical requirements:**
- Maximum strength in bending
- Minimum weight
- Crash resistance

**Recommended material:** Nylon or CF-Nylon

**Settings:**
```
Layer height: 0.28mm (with 0.6mm nozzle)
Infill: 30% Gyroid
Perimeters: 3
Orientation: Flat (layers parallel to ground)
Supports: None
```

**Why this works:**
```
Crash force typically from below/side
Layers oriented to resist bending
Gyroid infill provides omnidirectional strength
Thicker layers with larger nozzle = faster + stronger
```

**Advanced: Hollow Arms**
```python
# Modify parts/arm.py
def hollow_arm(
    length=150,
    width=16,
    height=12,
    wall_thickness=3
):
    outer = (
        cq.Workplane("XY")
        .rect(width, height)
        .extrude(length)
    )
    
    inner = (
        cq.Workplane("XY")
        .rect(width - 2*wall_thickness, height - 2*wall_thickness)
        .extrude(length)
    )
    
    return outer.cut(inner).edges("|Z").fillet(2)
```

Weight savings: 40-50%
Strength reduction: ~20% (still adequate)

---

### Camera Mounts

**Critical requirements:**
- Vibration dampening
- Adjustable angle
- Crash protection

**Recommended material:** PLA or PETG

**Settings:**
```
Layer height: 0.2mm
Infill: 30% Grid
Perimeters: 3
Orientation: Upright (to allow angle adjustment)
Supports: Minimal (remove carefully)
```

**Design tip:**
Add TPU dampening interface between camera and mount:
```python
def camera_mount_with_damper(width=20, height=20, thickness=3):
    # Main mount in PETG
    mount = camera_mount(width, height, thickness)
    
    # TPU insert (print separately)
    damper = (
        cq.Workplane("XY")
        .rect(width - 2, thickness - 1)
        .extrude(height - 2)
    )
    return mount, damper
```

---

### Battery Trays

**Critical requirements:**
- Secure battery retention
- Weight optimization
- Easy battery changes

**Recommended material:** PETG

**Settings:**
```
Layer height: 0.2mm
Infill: 20% Honeycomb (or 0% with thick perimeters)
Perimeters: 4
Top/bottom layers: 4
Orientation: Flat
```

**Design variations:**
```python
# Add strap slots
def battery_tray_with_straps(length=100, width=35, wall=2):
    tray = battery_tray(length, width, wall)
    
    # Cut slots for battery strap
    slot_width = 25
    tray = (
        tray
        .faces(">Z")
        .workplane()
        .rect(slot_width, 4)
        .cutThruAll()
        .workplane(offset=length-10)
        .rect(slot_width, 4)
        .cutThruAll()
    )
    return tray
```

---

## Print Orientation Guide

**Rule of thumb:** Orient parts so layer lines resist primary force direction

### Examples:

**Motor Mount:**
```
✓ CORRECT:          ✗ WRONG:
   Flat                Upright
   ____               |
  |____|              |
  
Layers ⊥ to thrust   Layers ∥ to thrust
Strong               Weak (delamination risk)
```

**Arm:**
```
✓ CORRECT:          ✗ WRONG:
   Flat                Vertical
   ________           |
                      |
                      
Layers resist bend   Layers split easily
```

**Camera Mount:**
```
✓ CORRECT:          ~ OK:
   Upright             Angled
      |               /
      |              /
      
Good finish         Needs supports
Easy removal        Better strength
```

---

## Infill Patterns Explained

### Gyroid
**Properties:**
- Omnidirectional strength
- No orientation dependency
- Self-supporting (no pillars)
- Slightly slower to slice

**Best for:** Arms, mounts, structural parts

### Grid
**Properties:**
- Fast to print
- Good vertical strength
- Directional (stronger in XY)
- Very common

**Best for:** General parts, prototypes

### Honeycomb
**Properties:**
- Excellent stiffness
- Lightweight
- Good for thin parts
- Slower to print

**Best for:** Panels, battery trays, covers

### Concentric
**Properties:**
- Flexible in Z-axis
- Strong in XY
- Good for organic shapes

**Best for:** Curved parts, flexible mounts

---

## Post-Processing Techniques

### 1. Hole Cleanup
```
Problem: Printed holes often too small or rough
Solution:
  - Drill with correct size bit (3mm for M3, etc.)
  - Use reamer for precision
  - Hand ream with drill bit for perfect fit
```

### 2. Heat-Set Inserts
```
Equipment: Soldering iron, heat-set brass inserts

Process:
  1. Print hole 0.1mm smaller than insert OD
  2. Heat soldering iron to 200-250°C
  3. Place insert on hole
  4. Apply gentle pressure
  5. Let cool completely
  
Benefits:
  - Much stronger than plastic threads
  - Allows repeated screw installation
  - Professional results
```

### 3. Annealing (PLA only)
```
Process:
  1. Print part with 50% infill minimum
  2. Heat oven to 90°C
  3. Place part in oven for 30 minutes
  4. Let cool slowly in oven
  
Results:
  - +20% strength
  - +50% temperature resistance
  - -5% dimensional accuracy (shrinks slightly)
  
Use for: Motor mounts, high-temp areas
```

### 4. Vapor Smoothing (ABS only)
```
Not recommended for flight parts (weakens structure)
Only for aesthetic parts or low-stress components
```

---

## Troubleshooting Common Issues

### Warping
**Symptoms:** Corners lift from bed

**Solutions:**
- Increase bed temperature (+5-10°C)
- Use brim (5-10mm)
- Ensure bed level
- Clean bed thoroughly
- Use glue stick or hairspray for adhesion
- Enclose printer (for Nylon/ABS)

---

### Stringing
**Symptoms:** Fine strings between parts

**Solutions:**
- Increase retraction distance (+1mm)
- Increase retraction speed (+5mm/s)
- Lower temperature (-5°C)
- Enable "combing" in slicer
- Dry filament (especially Nylon)

---

### Layer Delamination
**Symptoms:** Layers separate under stress

**Solutions:**
- Increase nozzle temperature (+5-10°C)
- Decrease print speed (-10mm/s)
- Increase flow rate (+2-5%)
- Check for partial clogs
- Dry filament

---

### Poor Bridging
**Symptoms:** Sagging between supports

**Solutions:**
- Increase fan speed (except first layer)
- Decrease temperature (-5°C)
- Slow down bridge speed
- Reduce bridge distance in design
- Add supports if necessary

---

## Pre-Flight Print Checklist

Before installing any printed part:

- [ ] No visible layer delamination
- [ ] All bolt holes clear and sized correctly
- [ ] No cracks or weak points
- [ ] Proper dimensional accuracy (±0.2mm)
- [ ] Smooth surfaces (no rough edges that catch wires)
- [ ] Adequate infill visible in cross-section
- [ ] No warping or distortion
- [ ] Heat-set inserts properly installed (if used)
- [ ] Test fit with actual components

**Stress test:**
- Apply realistic forces by hand
- Check for flex/cracking
- Simulate crash impact (drop test)

---

## Part Library Expansion

Create custom parts for your needs:

### Antenna Mount
```python
import cadquery as cq

def sma_antenna_mount(thickness=3, diameter=8):
    """Mount for SMA antenna connector"""
    return (
        cq.Workplane("XY")
        .circle(diameter)
        .extrude(thickness)
        .faces(">Z")
        .workplane()
        .hole(6.2)  # SMA connector hole
        .faces(">Z")
        .workplane()
        .rect(15, 3)
        .cutThruAll()  # Mounting slot
    )
```

### Landing Gear
```python
def simple_landing_gear(height=40, width=60):
    """Basic landing gear legs"""
    return (
        cq.Workplane("XY")
        .rect(10, 10)
        .workplane(offset=height)
        .rect(width, 10)
        .loft()
        .edges("|Z")
        .fillet(2)
    )
```

### GoPro Mount
```python
def gopro_mount(angle=30):
    """GoPro session mount"""
    base = (
        cq.Workplane("XY")
        .rect(40, 30)
        .extrude(5)
    )
    
    # GoPro fingers (simplified)
    fingers = (
        cq.Workplane("XY")
        .workplane(offset=5)
        .rect(35, 4)
        .extrude(10)
        .faces(">Z")
        .workplane()
        .rect(10, 4)
        .cutThruAll()
    )
    
    return base.union(fingers).rotate((0,0,0), (1,0,0), angle)
```

---

## Cost Analysis

### Material Cost per Part

Based on average filament prices:

| Material | Price/kg | Motor Mount | Arm (150mm) | Frame (4 arms) |
|----------|----------|-------------|-------------|----------------|
| PLA | $20 | $0.15 | $0.50 | $2.00 |
| PETG | $25 | $0.20 | $0.65 | $2.60 |
| Nylon | $40 | $0.30 | $1.00 | $4.00 |
| CF-Nylon | $60 | $0.45 | $1.50 | $6.00 |

**Complete 5" Quad Frame Cost:**
- Material: $4-6
- Print time: 8-12 hours
- Electricity: ~$0.50

**vs Commercial Frame:**
- Carbon fiber frame: $30-80
- Savings: $25-75 per build

**Break-even:** 1-2 builds (considering printer cost amortization)

---

## Summary: Best Practices

1. **Choose right material for application**
   - PLA: Prototypes, low-stress
   - PETG: General flight parts
   - Nylon: High-stress, impacts
   - CF-Nylon: Racing, maximum strength

2. **Orientation matters more than infill**
   - Orient layers to resist primary forces
   - Use 30-50% infill for most parts

3. **Post-process for professional results**
   - Clean holes with drill bits
   - Use heat-set inserts for repeated assembly
   - Anneal PLA for strength

4. **Test before flight**
   - Visual inspection
   - Stress test by hand
   - Verify dimensions

5. **Iterate and improve**
   - First print is rarely perfect
   - Adjust parameters for your printer
   - Keep notes on successful settings

---

**Happy printing! 🖨️**
