# Fixed-Wing Aircraft Design Guide

**Complete guide to designing and building fixed-wing UAVs using Python and 3D printing.**

This module provides professional tools for fixed-wing aircraft design, following aerospace engineering principles scaled for solo builders.

---

## 📐 Design Philosophy

### Key Differences from Multirotors

| Multirotor | Fixed-Wing |
|------------|------------|
| Thrust holds it up | **Lift holds it up** |
| Arms take bending | **Wings take bending + torsion** |
| Symmetric load | **Highly asymmetric load** |

### Golden Rules

> **Wings are beams.**  
> **Fuselage is a shell.**  
> **Tail is a lever.**

---

## 🎯 Design Target: Small Electric UAV

### Baseline Specifications

| Parameter | Value |
|-----------|-------|
| Wingspan | 1200 mm |
| Wing chord | 180 mm |
| Airfoil | Clark-Y |
| Cruise speed | 15–20 m/s |
| Takeoff weight | 1.2–1.5 kg |

---

## 🧩 Core Structural Elements

### Wing Structure

- **Spar**: Takes bending loads (primary structure)
- **Ribs**: Maintain airfoil shape
- **Skin**: Provides torsion resistance and aerodynamic surface

### Fuselage

- Semi-monocoque shell construction
- Local reinforcements at critical points:
  - Wing mount (90% of failures occur here!)
  - Landing gear attachment
  - Motor mount

### Tail

- Horizontal stabilizer (elevator)
- Vertical stabilizer (rudder)
- Light but stiff construction
- **Critical**: Tail weight affects CG position

---

## 🔧 Module Overview

### `wing_rib.py`

Generate parametric wing ribs with airfoil profiles.

```python
from fixed_wing.wing_rib import wing_rib, wing_rib_simple

# Clark-Y airfoil (realistic)
rib = wing_rib(chord=180, thickness=6, spar_slot=10)

# Simplified symmetric (easier to print)
rib_simple = wing_rib_simple(chord=180, thickness=6, spar_slot=10)
```

**Features:**
- Clark-Y airfoil profile (simplified for printing)
- Symmetric airfoil option
- Integrated spar slot at 30% chord
- Fully parametric dimensions

### `spar.py`

Spar design and structural calculations.

```python
from fixed_wing.spar import (
    wing_bending_load,
    spar_stress,
    recommend_spar_type
)

# Calculate bending load
load = wing_bending_load(weight=1400, span=1200)

# Get spar recommendation
recommendation = recommend_spar_type(wingspan=1200, weight=1400)
print(recommendation['type'])
# Output: "Carbon tube (6-8mm OD)"
```

**Spar Selection Rules:**

| Aircraft Size | Weight | Recommendation |
|---------------|--------|----------------|
| < 1000mm | < 1kg | Printed spar acceptable (test first!) |
| 1000-1500mm | 1-2kg | Carbon tube preferred (6-8mm) |
| > 1500mm | > 2kg | Carbon tube required (8-10mm) |

**Critical Note:** If spar fails → aircraft fails!

### `fuselage.py`

Modular fuselage sections and mounting components.

```python
from fixed_wing.fuselage import (
    fuselage_section,
    fuselage_bulkhead,
    wing_mount_plate
)

# Standard section
section = fuselage_section(radius=35, length=80, wall=2)

# Reinforced section for wing mount
reinforced = fuselage_section(
    radius=35, 
    length=80, 
    wall=2, 
    reinforced=True
)

# Wing mount plate (critical component!)
mount = wing_mount_plate(
    width=60, 
    height=40, 
    thickness=6,
    bolt_spacing=20
)
```

**Engineering Notes:**
- Semi-monocoque design: shell carries structural loads
- Modular sections for easy printing
- **Double wall thickness** at wing mount
- No sharp transitions (stress concentrators)

### `loads.py`

Aerodynamic load calculations and analysis.

```python
from fixed_wing.loads import calculate_flight_loads

# Comprehensive load analysis
loads = calculate_flight_loads(
    weight=1400,      # grams
    wingspan=1200,    # mm
    chord=180         # mm
)

print(f"Cruise speed: {loads['estimated_cruise_speed_ms']:.1f} m/s")
print(f"Wing loading: {loads['wing_loading_g_cm2']:.4f} g/cm²")
print(f"Lift per wing: {loads['lift_per_wing_g']:.1f} g")
```

**Includes:**
- Lift calculations (with 2× safety factor for gusts)
- Wing loading analysis
- Cruise speed estimation
- Tail sizing recommendations
- Stability analysis (tail volume coefficient)

### `tail.py`

Tail components for stability and control.

```python
from fixed_wing.tail import (
    horizontal_stabilizer,
    vertical_stabilizer,
    tail_boom_mount
)

# Horizontal stabilizer
h_stab = horizontal_stabilizer(span=400, chord=100)

# Vertical stabilizer
v_stab = vertical_stabilizer(height=120, chord=100)

# Tail boom mount (for carbon tube)
mount = tail_boom_mount(boom_diameter=8)
```

**Design Rules:**
- Tail area: ~20-30% of wing area
- Moment arm: 2-3× wing chord
- **Minimize weight** - directly affects CG position
- Use carbon tube for tail boom, NOT printed

---

## 📊 Usage Examples

### Example 1: Complete Load Analysis

```bash
PYTHONPATH=. python examples/fixed_wing_analysis.py
```

This will output:
- Wing loading and classification
- Lift requirements with safety factors
- Spar recommendations
- Tail sizing and stability analysis
- Build recommendations

### Example 2: Generate STL Files

```bash
PYTHONPATH=. python examples/generate_fixed_wing.py
```

Generates:
- Wing ribs (multiple sizes and profiles)
- Fuselage sections (standard and reinforced)
- Wing mount plates
- Tail components
- Tail boom mounts

All STL files saved to `output/fixed_wing/`

---

## 🏗️ Build Strategy: HYBRID CONSTRUCTION

### ❌ Don't Do This

- **Full printed wings** → Too heavy and flexible
- **Printed spar for large aircraft** → Will fail
- **Heavy tail components** → CG problems

### ✅ Best Practice

**Wings:**
1. Print ribs only (PLA/PETG, 30% infill)
2. Use carbon tube spar (6-8mm OD)
3. Cover with:
   - Foam core (depron/EPP)
   - Balsa sheeting
   - Heat-shrink film covering

**Fuselage:**
1. Print in Nylon/PETG (2mm walls)
2. Reinforce wing mount (double thickness)
3. Modular sections for printing

**Tail:**
1. Carbon tube boom (8mm) - NOT printed
2. Print mounting brackets only
3. Foam/balsa stabilizers + film covering

---

## ⚠️ Critical Failure Points

### Wing-Fuselage Connection (90% of failures!)

**Requirements:**
- Metal or carbon spar pass-through
- Double wall thickness
- No sharp corners or transitions
- Test with 2× expected loads

**Reinforcement:**
```python
# Use reinforced fuselage section
section = fuselage_section(
    radius=35,
    length=80,
    wall=2,
    reinforced=True  # Critical!
)

# Plus dedicated wing mount plate
mount = wing_mount_plate(thickness=6)  # Thick!
```

---

## 🎓 Engineering Principles

### Beam Theory (Wings)

Wings are beams subject to bending:

```
Stress (σ) = M × y / I
```

Where:
- M = bending moment
- y = distance from neutral axis
- I = moment of inertia

**Implication:** Spar depth matters more than width!

### Shell Theory (Fuselage)

Thin-walled structures carry loads through shell action:
- Tension/compression in the shell
- Local reinforcement at concentrated loads
- Avoid stress concentrations

### Stability (Tail)

Tail volume coefficient determines stability:

```
V_h = (S_tail × L_tail) / (S_wing × C_wing)
```

**Typical values:**
- Trainers: 0.6-0.7 (very stable)
- Sport: 0.4-0.5 (balanced)
- Aerobatic: 0.3-0.4 (agile)

---

## 🔬 Material Selection

| Component | Material | Why |
|-----------|----------|-----|
| Wing ribs | PLA/PETG | Adequate strength, easy to print |
| Fuselage | Nylon/PETG | Impact resistance, flexibility |
| Wing mounts | CF-Nylon | Maximum strength where it matters |
| Tail mounts | Nylon | Strength + light weight |
| **Spar** | **Carbon tube** | **Best strength-to-weight** |
| **Tail boom** | **Carbon tube** | **Lightweight + stiff** |

---

## 📈 What You Can Build Now

### Beginner
- Small electric glider (800mm wingspan)
- 3D printed ribs + foam core + film
- Great for learning aerodynamics

### Intermediate
- Sport UAV (1200mm wingspan)
- Full parametric design
- FPV capability

### Advanced
- Long-range platform (1500mm+)
- Optimized for efficiency
- Custom airfoils and structures

---

## 🚀 Next Steps

1. **Run the analysis:**
   ```bash
   PYTHONPATH=. python examples/fixed_wing_analysis.py
   ```

2. **Generate components:**
   ```bash
   PYTHONPATH=. python examples/generate_fixed_wing.py
   ```

3. **Customize parameters** in the code for your specific design

4. **Print and test** incrementally

5. **Build hybrid** (print + traditional materials)

---

## 📚 Additional Resources

### Airfoil Data
- **UIUC Airfoil Database**: Thousands of airfoil coordinates
- **Airfoil Tools**: Online analysis and visualization

### Structural Analysis
- **Beam calculators**: For spar sizing
- **FEA tools**: For complex stress analysis

### RC Aircraft Communities
- **RC Groups**: Forums and build logs
- **Flite Test**: Foam-board aircraft techniques
- **Experimental Aircraft Association**: General aviation principles

---

## 🎯 Summary

You now have:

✅ **Real aerospace engineering tools**  
✅ **Parametric design capability**  
✅ **Structural analysis methods**  
✅ **Practical build strategies**  
✅ **Print-optimized components**

This is **actual aircraft engineering**, adapted for solo builders using modern tools.

**Remember:** Hybrid construction wins. Print what makes sense, use traditional materials where they excel.

---

## ⚡ Quick Reference

```python
# Generate wing rib
from fixed_wing.wing_rib import wing_rib
rib = wing_rib(chord=180, thickness=6, spar_slot=10)

# Calculate loads
from fixed_wing.loads import calculate_flight_loads
loads = calculate_flight_loads(weight=1400, wingspan=1200, chord=180)

# Get spar recommendation
from fixed_wing.spar import recommend_spar_type
spar = recommend_spar_type(wingspan=1200, weight=1400)

# Generate fuselage
from fixed_wing.fuselage import fuselage_section
fuse = fuselage_section(radius=35, length=80, wall=2, reinforced=True)
```

---

**Ready to design real aircraft!** ✈️
