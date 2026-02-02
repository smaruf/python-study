# Advanced Wing Types - Design Guide

## Overview

This guide covers advanced wing configurations beyond conventional designs. Each configuration offers unique advantages and challenges, suitable for different applications and builder experience levels.

**Wing Types Covered:**
1. Delta Wing
2. Flying Wing (Tailless)
3. Canard Configuration
4. Oblique Wing (Variable Sweep)
5. Flying Pancake (Circular Wing)

---

## 1. Delta Wing

### Description

Delta wings are triangular wings with swept leading edges, commonly used in high-speed aircraft. The triangular planform provides excellent structural efficiency and unique aerodynamic properties.

### Theoretical Background

**Geometry:**
- Triangular planform with swept leading edges
- Low aspect ratio (typically 2-4)
- Taper to point or small tip chord
- Leading edge sweep angle: 40-50° typical

**Aerodynamics:**
- Generates lift through conventional flow at low angles of attack
- Creates leading edge vortices at high angles (vortex lift)
- Sweep reduces effective aspect ratio
- Better high-speed performance than conventional wings
- Delayed shock wave formation at transonic speeds

**Lift Mechanisms:**
1. **Conventional Lift** (low α): Standard pressure distribution
2. **Vortex Lift** (high α): Leading edge vortices create suction over wing
3. **Combined Lift**: Both mechanisms work together at moderate α

### Design Parameters

```python
from fixed_wing.wing_types import delta_wing_design

design = delta_wing_design(
    root_chord=400,      # Root chord in mm
    wingspan=1000,       # Total span in mm
    sweep_angle=45,      # Leading edge sweep (degrees)
    thickness_ratio=0.08 # Airfoil thickness ratio
)
```

**Key Parameters:**
- **Root Chord**: 300-500mm for small UAVs
- **Sweep Angle**: 40-50° (more sweep = higher speed capability)
- **Aspect Ratio**: 2-4 (low AR is characteristic of deltas)
- **Thickness Ratio**: 6-10% (thinner for speed, thicker for strength)

### Stability & Control

**Pitch Stability:**
- Tailless design (no horizontal stabilizer)
- Center of lift moves aft with increasing angle of attack
- Requires careful CG placement: 25-30% of root chord
- Elevons provide pitch control

**Roll Stability:**
- Good roll damping due to sweep
- Elevons differential for roll control

**Yaw Stability:**
- Vertical stabilizer required (or split elevons as rudders)
- Sweep provides some directional stability

**Critical Design Point:**
- CG location: 25-30% of root chord from leading edge
- Too far forward: Heavy pitch-up tendency
- Too far aft: Unstable in pitch

### Control Surfaces

**Elevons:**
- Combined elevator + aileron
- Located on trailing edge
- Chord: 25-30% of root chord
- Span: 40-50% of wingspan per side
- Mixing required: Pitch = both up/down, Roll = differential

**Optional Rudders:**
- Split elevons can act as rudders
- Or dedicated vertical surfaces at tips

### Construction Principles

**Option 1: Foam Core with Composite Skin**
```
1. Hot-wire cut foam core (EPP or EPS)
2. Laminate with fiberglass or carbon fiber
3. Embedded spar at centerline
4. Torsion box from LE to 60% chord
```

**Option 2: 3D Printed Ribs with Spar**
```
1. Print ribs perpendicular to flight direction
2. Carbon fiber spar through centerline
3. Foam leading edge (EPP for impact resistance)
4. Covering: Film, fabric, or thin composite
```

**Structural Elements:**
- **Main Spar**: Along centerline or dual spars
- **Ribs**: Perpendicular to flight direction, spaced 40-60mm
- **Torsion Box**: Critical for rigidity, LE to 60% chord
- **Skin**: Composite or heat-shrink film

**Material Recommendations:**
- Spar: Carbon fiber tube 8-12mm diameter
- Ribs: PLA/PETG 3D printed or plywood
- Skin: Fiberglass (2-3 layers) or heat-shrink film
- Leading Edge: EPP foam (impact resistant)

### Performance Characteristics

**Advantages:**
- ✅ Excellent high-speed performance
- ✅ Strong structure (bending loads in-plane)
- ✅ Simple tailless design
- ✅ Good for aerobatics
- ✅ Large internal volume for equipment

**Disadvantages:**
- ❌ Higher stall speed than conventional wings
- ❌ Less efficient at low speeds
- ❌ Critical CG location
- ❌ Requires elevon mixing

**Typical Performance:**
- Stall speed: ~30% higher than conventional wing
- L/D ratio: 8-12 (moderate)
- Best flight regime: Medium to high speed
- Maneuverability: Excellent

### Build Tips

1. **CG Location Critical**: Test with glide tests, adjust ballast
2. **Leading Edge Strength**: Reinforce for durability
3. **Torsion Rigidity**: Ensure wing doesn't twist under load
4. **Elevon Linkage**: Use ball links, minimize slop
5. **Flight Testing**: Start with conservative CG (more forward)

### Example Aircraft

**Small UAV Specifications:**
- Root chord: 400mm
- Wingspan: 1000mm
- Sweep angle: 45°
- Weight: 800-1200g
- Wing loading: 30-50 g/dm²
- Power: 200-400W

---

## 2. Flying Wing (Tailless)

### Description

Flying wings eliminate the fuselage and tail, integrating all components within the wing envelope. This provides maximum aerodynamic efficiency and minimum drag.

### Theoretical Background

**Geometry:**
- All components within wing profile
- Swept or straight leading edge
- Tapered planform
- Reflex airfoil essential

**Aerodynamics:**
- Highest lift-to-drag ratio of any configuration
- All surfaces generate lift (no tail download)
- Reflex airfoil provides pitch stability
- Sweep provides directional stability

**Stability Without a Tail:**
- Conventional airfoils generate nose-down pitching moment
- Reflex airfoils have upturned trailing edge
- Reflex creates nose-up moment, balancing the wing
- CG at neutral point (25-30% MAC)

### Design Parameters

```python
from fixed_wing.wing_types import flying_wing_design

design = flying_wing_design(
    center_chord=350,    # Center section chord in mm
    wingspan=1200,       # Total span in mm
    sweep_angle=25,      # Sweep angle (degrees)
    wing_twist=-2        # Washout angle (degrees, negative)
)
```

**Key Parameters:**
- **Center Chord**: 300-400mm (houses electronics)
- **Sweep Angle**: 20-30° (for stability)
- **Washout**: -2° to -3° (prevents tip stalling)
- **Aspect Ratio**: 8-12 (higher than delta)

### Airfoil Requirements

**Critical: Must Use Reflex Airfoil**

Recommended airfoils:
- **Eppler E325**: Classic flying wing airfoil
- **MH-45**: Good performance, stable
- **S8036**: General purpose reflex
- **AG38**: High efficiency

**Characteristics:**
- Upturned trailing edge (2-4° reflex)
- Creates positive pitching moment
- Reduces efficiency slightly (~5%) vs conventional
- Essential for pitch stability

**DO NOT use**:
- Clark-Y (no reflex, will dive)
- NACA 4-digit (conventional, unstable)
- Flat bottom airfoils

### Stability & Control

**Pitch Stability:**
- Provided by reflex airfoil ONLY
- CG at 25-30% MAC (very critical!)
- CG tolerance: ±5mm (much tighter than conventional)
- Test flights required to fine-tune

**Roll Stability:**
- Sweep provides some dihedral effect
- Consider 2-3° geometric dihedral
- Washout helps roll damping

**Yaw Stability:**
- Sweep provides weathercock stability
- Optional: Winglets with rudders
- Vertical fins at wing tips

**Critical Design Point:**
- CG location is CRITICAL - ±5mm tolerance
- Balance with adjustable battery position
- Use CG calculator before first flight

### Control Surfaces

**Elevons:**
- Combined elevator + aileron function
- Chord: 25-30% of local chord
- Span: 40-50% of wingspan per side
- Mixing: Pitch = both, Roll = differential

**Optional Winglets:**
- Improve directional stability
- Can include rudders
- Reduce tip vortices (efficiency gain)

### Construction Principles

**Recommended: Hot-Wire Foam Core**

```
1. Cut foam core with hot wire (CNC or templates)
2. Install main spar at 30-35% chord
3. Carve electronics bay in center section
4. Apply composite skin (1-2 layers fiberglass)
5. Install elevon servos and linkages
```

**Alternative: 3D Printed + Foam**

```
1. Print center section with electronics bay
2. Foam outer panels
3. Carbon spar through entire wing
4. Composite or film covering
```

**Structural Elements:**
- **Main Spar**: Through entire wing at 30-35% chord
- **Torsion Box**: LE to spar, critical for stiffness
- **Electronics Bay**: Center section, 150-200mm wide
- **Skin**: Composite or heat-shrink film

**Material Recommendations:**
- Core: EPP or EPO foam (crash resistant)
- Spar: Carbon fiber tube or pultruded rod
- Skin: Fiberglass (lightweight) or heat-shrink film
- Center section: Can be 3D printed (PETG/Nylon)

### Performance Characteristics

**Advantages:**
- ✅ Highest efficiency (L/D 15-20+)
- ✅ Minimum drag (no fuselage/tail)
- ✅ Excellent for long-range FPV
- ✅ Sleek appearance
- ✅ Best endurance for given battery

**Disadvantages:**
- ❌ Critical CG location (±5mm)
- ❌ Requires reflex airfoil
- ❌ More difficult to build accurately
- ❌ Less stable than conventional
- ❌ Requires careful setup

**Typical Performance:**
- L/D ratio: 15-20 (excellent)
- Stall speed: Low (large area)
- Best flight regime: Cruise, soaring
- Maneuverability: Moderate

### Build Tips

1. **Airfoil Accuracy**: Use templates or CNC foam cutting
2. **CG Testing**: Balance before covering, test glide
3. **Washout**: Build in -2° to -3° twist
4. **Battery Position**: Make adjustable for CG tuning
5. **First Flight**: Have experienced pilot test, calm conditions

### Example Aircraft

**FPV Flying Wing Specifications:**
- Center chord: 350mm
- Wingspan: 1200mm
- Sweep angle: 25°
- Airfoil: Eppler E325 or MH-45
- Weight: 800-1000g
- Wing loading: 20-30 g/dm²
- Power: 150-250W
- Flight time: 30-60 minutes

---

## 3. Canard Configuration

### Description

Canard aircraft feature a small wing (canard) ahead of the main wing. Both surfaces generate lift, and the configuration offers inherent safety through stall resistance.

### Theoretical Background

**Geometry:**
- Small forward wing (canard) ahead of CG
- Main wing behind CG
- Fuselage connects both surfaces
- Canard area typically 15-25% of main wing

**Aerodynamics:**
- Both surfaces generate positive lift (more efficient than tail)
- Canard operates at higher angle of attack than main wing
- Canard stalls first → nose drops → automatic recovery
- Inherently stall-resistant

**Lift Distribution:**
- Canard: ~20-30% of total lift
- Main wing: ~70-80% of total lift
- Total lift > weight for flight
- Trim drag exists (similar to tail)

### Design Parameters

```python
from fixed_wing.wing_types import canard_design

design = canard_design(
    main_wing_chord=200,     # Main wing chord in mm
    main_wingspan=1000,      # Main wing span in mm
    canard_chord=80,         # Canard chord in mm
    canard_span=400,         # Canard span in mm
    canard_position=150      # Distance ahead of main wing in mm
)
```

**Key Ratios:**
- **Canard Area / Main Area**: 0.15-0.25
- **Canard Volume Coefficient**: 0.04-0.08
- **Moment Arm**: 1-2× main wing chord
- **Canard Incidence**: 2-4° higher than main wing

### Stability & Control

**Pitch Stability:**
- Canard provides positive pitch stability
- Stall-proof: Canard stalls before main wing
- Nose drops automatically if approaching stall
- Inherently safe design

**Roll Stability:**
- Main wing has ailerons for roll control
- Consider dihedral on main wing
- Canard can have anhedral (negative dihedral)

**Yaw Stability:**
- Vertical fin typically on fuselage
- Or on main wing tips
- Less critical than conventional (short fuselage)

**Critical Design Point:**
- Canard MUST stall before main wing
- Achieve by: Higher incidence, different airfoil, or smaller area
- CG at 15-25% of main wing MAC

### Control Surfaces

**Canard Elevators:**
- Pitch control surfaces on canard
- Chord: 25-30% of canard chord
- Full span or partial span

**Main Wing Ailerons:**
- Roll control on main wing
- Chord: 25% of main wing chord
- Span: 30-40% of main wing span per side

**Rudder (Optional):**
- On fuselage vertical fin
- Or split surfaces on wing tips

### Construction Principles

**Fuselage:**
```
1. Strong boom connecting canard and main wing
2. Carbon tube (15-20mm) or 3D printed monocoque
3. Reinforced mounting points for both wings
4. Length: Canard position + main wing position + tail boom
```

**Canard:**
```
1. Flat-bottom or high-lift airfoil
2. 3D printed ribs or foam core
3. Carbon spar through fuselage
4. Higher incidence than main wing (+2-4°)
```

**Main Wing:**
```
1. Conventional airfoil (Clark-Y, NACA 4412)
2. Standard construction (ribs + spar)
3. Ailerons on trailing edge
4. Removable for transport
```

**Material Recommendations:**
- Fuselage boom: Carbon fiber tube or 3D printed PETG/Nylon
- Wing spars: Carbon fiber tube
- Ribs: 3D printed PLA/PETG or plywood
- Covering: Heat-shrink film or light composite

### Performance Characteristics

**Advantages:**
- ✅ Stall-proof (canard stalls first)
- ✅ Both surfaces generate lift (~5% more efficient)
- ✅ Excellent visibility (canard doesn't block view)
- ✅ Wide CG range
- ✅ Great for trainers and general flying

**Disadvantages:**
- ❌ More complex than conventional
- ❌ Requires precise incidence angles
- ❌ Some trim drag (like tail)
- ❌ Longer fuselage (more weight)

**Typical Performance:**
- L/D ratio: 10-14 (good)
- Stall speed: Moderate
- Best flight regime: General sport flying
- Maneuverability: Good
- Safety: Excellent (stall-resistant)

### Build Tips

1. **Incidence Angles**: Canard 2-4° higher than main wing
2. **Stall Testing**: Verify canard stalls first (bench test)
3. **CG Location**: 15-25% of main wing MAC
4. **Propulsion**: Pusher configuration recommended
5. **Transport**: Make canard removable

### Example Aircraft

**Sport Canard UAV Specifications:**
- Main wing: 200mm chord × 1000mm span
- Canard: 80mm chord × 400mm span
- Canard position: 150mm ahead of main wing
- Weight: 900-1200g
- Wing loading: 30-40 g/dm²
- Power: 200-300W
- Propulsion: Pusher

---

## 4. Oblique Wing (Variable Sweep)

### Description

Oblique wings pivot to achieve variable sweep, with one wing sweeping forward and the other backward. This configuration optimizes performance across a wide speed range.

### Theoretical Background

**Geometry:**
- Wing pivots about central axis
- Symmetric wing swept asymmetrically
- One side forward-swept, other aft-swept
- Pivot point typically at wing center

**Aerodynamics:**
- Variable sweep optimizes for different speeds
- Straight (0°): Best low-speed lift
- Swept (30-45°): Reduced wave drag at high speed
- Asymmetric lift distribution when swept
- Requires trim compensation

**Speed Range Optimization:**
- 0°: Maximum lift, low speed (takeoff, landing)
- 20°: Cruise efficiency
- 30-45°: High-speed, minimum drag
- 60°+: Supersonic (experimental, not practical for models)

### Design Parameters

```python
from fixed_wing.wing_types import oblique_wing_design

# Analyze at different sweep angles
for sweep in [0, 20, 30, 45]:
    design = oblique_wing_design(
        wingspan=1000,          # Total span in mm
        chord=180,              # Constant chord in mm
        sweep_angle=sweep,      # Current sweep angle
        pivot_position=0.5      # Pivot at center (0.5)
    )
```

**Key Parameters:**
- **Wingspan**: 800-1200mm for small UAVs
- **Chord**: Constant chord recommended
- **Sweep Range**: 0-45° typical for models
- **Pivot Position**: 0.5 (center) typical

### Stability & Control

**Pitch Stability:**
- Changes with sweep angle
- Requires horizontal stabilizer (not shown in basic design)
- Trim changes with sweep angle

**Roll Stability:**
- Asymmetric lift creates rolling moment
- Forward-swept side generates more lift
- Requires aileron trim compensation
- ~15% lift difference at 30° sweep

**Yaw Stability:**
- Yaw-roll coupling when swept
- More pronounced at higher sweep angles
- Requires coordinated control

**Critical Design Points:**
- CG shifts slightly with sweep angle
- Control mixing required for trim
- Fly-by-wire strongly recommended

### Control Surfaces

**Ailerons:**
- Must function at all sweep angles
- Mixing required for trim compensation
- Differential throw helps with yaw

**Elevator & Rudder:**
- Conventional tail surfaces
- Trim adjustment for each sweep angle
- Consider programmable radio

**Sweep Actuation:**
- Servo-driven pivot mechanism
- Slow movement (10-20 seconds for full range)
- Position feedback sensor
- Locking mechanism at preset angles

### Construction Principles

**Pivot Mechanism:**
```
1. High-strength bearing at fuselage centerline
2. Servo with reduction gearbox (high torque)
3. Position sensor (potentiometer or encoder)
4. Locking pins at 0°, 20°, 30°, 45° positions
5. Emergency lock mechanism
```

**Wing Structure:**
```
1. Symmetric airfoil (works in both sweep directions)
2. Constant chord (simplifies design)
3. Carbon fiber spar through pivot point
4. Minimal taper (avoid tip loading)
5. Strong spar-to-pivot connection
```

**Fuselage:**
```
1. Cylindrical or streamlined body
2. Wide enough for wing pivot range
3. Reinforced pivot mounting (critical)
4. Electronics for sweep control
5. Battery forward for CG
```

**Material Recommendations:**
- Pivot bearing: Metal ball bearing (high quality)
- Wing spar: Carbon fiber tube (stiff and strong)
- Fuselage: 3D printed Nylon or PETG
- Wing ribs: PLA/PETG
- Covering: Light film (minimizes pivot torque)

### Performance Characteristics

**Advantages:**
- ✅ Optimized for wide speed range
- ✅ Unique and interesting design
- ✅ Educational / demonstration value
- ✅ Variable performance envelope

**Disadvantages:**
- ❌ Very high complexity
- ❌ Requires fly-by-wire control
- ❌ Asymmetric trim required
- ❌ Heavy pivot mechanism
- ❌ Not for beginners!

**Typical Performance:**
| Sweep | Speed Factor | Efficiency | Best For |
|-------|-------------|------------|----------|
| 0° | 1.0× | 100% | Takeoff, landing, slow flight |
| 20° | 1.1× | 95% | Cruise |
| 30° | 1.2× | 90% | High-speed cruise |
| 45° | 1.35× | 85% | Maximum speed |

### Build Tips

1. **Start Simple**: Consider fixed oblique wing first (no pivot)
2. **Strong Pivot**: Use quality bearings, over-engineer
3. **Slow Actuation**: Never sweep quickly in flight
4. **Locking Mechanism**: Prevent unwanted rotation
5. **Fly-by-Wire**: Use flight controller for mixing
6. **Test Glide**: Test as glider at each sweep angle first

### Example Aircraft

**Experimental Oblique Wing UAV:**
- Wingspan: 1000mm
- Chord: 180mm (constant)
- Sweep range: 0-45°
- Weight: 1200-1500g (pivot adds weight)
- Wing loading: Variable with sweep
- Power: 300-500W
- **Recommended for**: Experienced builders only!

---

## 5. Flying Pancake (Circular Wing)

### Description

The Flying Pancake features a circular or nearly-circular wing planform. Based on the historic Vought V-173 design, it's a fun and unusual configuration with docile handling.

### Theoretical Background

**Geometry:**
- Circular or elliptical planform
- Very low aspect ratio (<2)
- Large wing area relative to span
- Center cutout for fuselage/propulsion

**Aerodynamics:**
- High induced drag (low aspect ratio)
- Excellent low-speed handling
- Gentle stall characteristics
- Tip vortices meet at rear (reduced interference)
- Can fly at extreme angles of attack

**Historical Context:**
- Vought V-173 "Flying Pancake" (1942)
- Charles Zimmerman design
- Proved concept worked
- Led to XF5U-1 fighter prototype
- Unique appearance and performance

### Design Parameters

```python
from fixed_wing.wing_types import flying_pancake_design

design = flying_pancake_design(
    diameter=600,           # Wing diameter in mm
    thickness_ratio=0.12,   # Airfoil thickness
    center_cutout=120       # Center fuselage cutout in mm
)
```

**Key Parameters:**
- **Diameter**: 500-800mm for small models
- **Aspect Ratio**: ~1.0-1.5 (very low)
- **Thickness Ratio**: 12-15% (thick for structure)
- **Center Cutout**: 100-150mm for electronics

### Stability & Control

**Pitch Stability:**
- Very stable (large area, short moment arms)
- Low wing loading
- Gentle stall (progressive, not abrupt)

**Roll Stability:**
- Excellent damping
- Short effective wingspan

**Yaw Stability:**
- Stable due to large side area
- Short body reduces weather-vaning

**Control Characteristics:**
- Docile and forgiving
- High drag = safe flight envelope
- Difficult to overstress

### Control Surfaces

**Elevons (4-6 around perimeter):**
- Located on rear 120-150° arc
- Chord: 25-30% of local chord
- 4 independent surfaces typical
- Mixing required for pitch/roll/yaw

**Perimeter Control:**
- Distribute around rear half
- Can act as elevators, ailerons, rudders
- Requires electronic mixing

### Construction Principles

**Option 1: Radial Rib Structure**

```
1. Create circular template
2. Cut radial ribs (12 ribs every 30°)
3. Ring spar at 60-70% radius
4. Center hub (3D printed or plywood)
5. Plywood or foam skin
```

**Option 2: Foam Core**

```
1. Cut circular blank from foam
2. Carve airfoil profile
3. Center cavity for electronics
4. Fiberglass or carbon skin
5. Perimeter reinforcement
```

**Structural Elements:**
- **Radial Ribs**: 12 ribs, every 30°
- **Ring Spar**: At 60-70% radius for rigidity
- **Center Hub**: 3D printed, houses electronics
- **Skin**: Plywood (lightweight), foam, or composite

**Material Recommendations:**
- Ribs: 3D printed PLA/PETG or laser-cut plywood
- Ring spar: Plywood or carbon fiber ring
- Skin: 2mm plywood, foam, or composite
- Center hub: 3D printed PETG/Nylon
- Covering: Film or fabric over framework

### Propulsion Options

**Option 1: Twin Tip Motors** (Historical)
- Motors at opposite wing tips
- Reduces tip vortices
- Authentic to V-173 design
- Complex mechanically

**Option 2: Single Pusher** (Practical)
- Rear-mounted motor
- Simpler installation
- Less weight, less complexity
- Still effective

**Power Requirements:**
- High due to drag (1.5-2× conventional)
- Large props for static thrust
- Low pitch props (slow flight)

### Performance Characteristics

**Advantages:**
- ✅ Extremely stable and docile
- ✅ Gentle stall (progressive)
- ✅ Can fly at very high angles of attack
- ✅ Fun and unique appearance
- ✅ Great conversation starter
- ✅ Excellent for demonstrations

**Disadvantages:**
- ❌ High drag (low efficiency)
- ❌ Low speed (high drag)
- ❌ High power requirement
- ❌ Not practical for long range
- ❌ Heavy compared to conventional

**Typical Performance:**
- L/D ratio: 5-8 (poor)
- Stall speed: 30% lower than conventional (large area)
- Best flight regime: Slow, stable flight
- Maneuverability: Good (responsive, stable)
- Efficiency: 40% less than conventional

### Build Tips

1. **Accuracy**: Cut perfect circle with trammel or CNC
2. **Symmetry**: Balance carefully (round shape critical)
3. **Power**: Use larger motor and battery
4. **Props**: Large diameter, low pitch
5. **Testing**: Expect unusual flight characteristics
6. **Fun Factor**: Embrace the novelty!

### Example Aircraft

**Fun Flying Pancake Specifications:**
- Diameter: 600mm
- Center cutout: 120mm
- Wing area: ~2600 cm²
- Weight: 700-900g
- Wing loading: 25-35 g/dm²
- Power: 250-400W (high for size)
- Propulsion: Single pusher or twin tip motors
- **Purpose**: Fun flying, demonstrations, education

---

## Comparison Summary

| Wing Type | Complexity | Stability | Speed | Efficiency | Build Difficulty | Best For |
|-----------|------------|-----------|-------|------------|------------------|----------|
| **Delta Wing** | Medium | Good | High | Medium | Medium | Speed, aerobatics |
| **Flying Wing** | High | Medium | Med-High | Excellent | High | Efficiency, FPV long-range |
| **Canard** | Medium | Excellent | Medium | Good | Medium | General flying, trainers |
| **Oblique Wing** | Very High | Medium | Variable | Variable | Very High | Experimentation |
| **Flying Pancake** | Low-Med | Excellent | Low | Poor | Medium | Fun, demonstrations |

---

## Recommendations by Experience Level

### Beginner
**Start with**: Canard or simple Delta Wing
- Inherently stable
- Forgiving flight characteristics
- Good learning platforms
- Conventional control

### Intermediate
**Try**: Flying Wing or Delta Wing
- Requires more precision
- Rewarding performance
- Moderate complexity
- Unique flight experience

### Advanced
**Experiment with**: Oblique Wing or complex Flying Wing
- High complexity
- Requires advanced skills
- Educational value
- Cutting-edge designs

### Just for Fun
**Build**: Flying Pancake
- Unique and interesting
- Forgiving despite low efficiency
- Great conversation piece
- Fun project

---

## Construction Resources

### Tools Needed
1. **3D Printer** (optional): For ribs, hubs, components
2. **Hot Wire Cutter** (foam wings): CNC or manual
3. **Carbon Fiber Tubes**: Various diameters for spars
4. **Composite Materials**: Fiberglass cloth, epoxy resin
5. **Covering Materials**: Heat-shrink film or composite
6. **Electronics**: Flight controller, servos, motor, ESC
7. **Hand Tools**: Knives, sandpaper, adhesives

### Materials

**Structural:**
- Carbon fiber tubes: 6-12mm diameter
- Plywood: 2-3mm aircraft plywood
- Foam: EPP or EPO foam sheets
- Fiberglass: 25-50 g/m² cloth

**Electronics:**
- Flight controller: With mixing capability
- Servos: 9g for control surfaces
- Motor: 200-500W depending on design
- ESC: Appropriate for motor
- Battery: LiPo 3S-4S

---

## Next Steps

1. **Choose a design** based on your experience and goals
2. **Study the theory** - understand before building
3. **Calculate parameters** using provided functions
4. **Generate models** (if using CadQuery)
5. **Build carefully** - precision matters
6. **Test thoroughly** - glide tests before power
7. **Fly safely** - start conservative

**Code Examples:**
```bash
# Analyze all wing types
PYTHONPATH=. python examples/wing_types_analysis.py

# In Python
from fixed_wing.wing_types import (
    delta_wing_design,
    flying_wing_design,
    canard_design,
    oblique_wing_design,
    flying_pancake_design,
    compare_wing_types
)

# Choose your design and build!
```

---

**Happy building and safe flying!** ✈️
