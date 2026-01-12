# Foam-Board Build Templates

## Basic Glider Template (Day 1)

### Materials Needed:
- 1 sheet 5mm foam-board (20" × 30")
- Hot glue gun + glue sticks
- Hobby knife with fresh blade
- Metal ruler (24")
- Pencil
- Small weights (coins or washers)

### Cutting Plan

```
FOAM-BOARD LAYOUT (20" × 30" sheet)
┌─────────────────────────────────────────────────────┐
│                                                     │
│  WING (24" × 6")                                    │
│  ┌────────────────────────────────────────┐        │
│  │                                        │        │
│  └────────────────────────────────────────┘        │
│                                                     │
│  FUSELAGE (16" × 1.5")                              │
│  ┌──────────────────────────────┐                  │
│  └──────────────────────────────┘                  │
│                                                     │
│  HORIZONTAL STABILIZER (8" × 2.5")                  │
│  ┌────────────────┐                                │
│  └────────────────┘                                │
│                                                     │
│  VERTICAL STABILIZER (4" × 3")                      │
│  ┌──────┐                                          │
│  │      │                                          │
│  └──────┘                                          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Step-by-Step Assembly:

**1. Cut Wing (24" × 6")**
   - Mark centerline
   - Add 5° dihedral: Score center, bend upward
   - Reinforce with tape on underside

**2. Cut Fuselage (16" × 1.5")**
   - Fold foam to create box section
   - Create 1.5" × 1.5" square tube
   - Seal edges with hot glue

**3. Wing Attachment**
   - Position wing 6" from nose
   - Hot glue to fuselage top
   - Reinforce with popsicle stick spar

**4. Tail Assembly**
   - Cut horizontal stabilizer (8" × 2.5")
   - Cut vertical stabilizer (4" × 3")
   - Form T-tail configuration
   - Attach 14" from nose

**5. Balance Point**
   - CG should be 6-7" from nose (25% of wing chord)
   - Add coins/washers to nose if tail-heavy
   - Test balance by holding at CG

### Flight Testing Guide:

**Launch Technique:**
1. Hold at fuselage, level
2. Throw straight and level (not upward)
3. Medium speed, smooth release
4. Observe flight path

**Trimming:**
- **Dives immediately**: CG too far forward → remove weight
- **Stalls immediately**: CG too far back → add weight
- **Turns left/right**: Adjust vertical stabilizer angle
- **Oscillates**: CG slightly too far back

---

## FPV Glider Template (Day 2)

### Materials:
- 2 sheets 5mm foam-board
- 3mm carbon fiber rod (2 pieces, 24")
- FPV camera (board camera with housing)
- 2× 9g servos
- 3S 1000mAh LiPo
- Flight controller (optional for stabilization)

### Wing Construction:

**Airfoil: Clark-Y Profile**

```
Cross-section (scaled):
    _______________
   /               \___
  /                    \__
 /________________________\
 
Chord: 7"
Max thickness: 10% at 30% chord
```

**Cutting the Airfoil:**

1. **Create Template**
   ```
   Print template at 100% scale
   Trace onto foam-board
   Cut 4 ribs exactly
   ```

2. **Build Wing Structure**
   ```
   Rib spacing: 8" apart
   Spar position: 33% back from leading edge
   
   Layout:
   Leading Edge ──┬── Rib 1 ──┬── Rib 2 ──┬── Rib 3 ──┬── Rib 4 ──┬── Trailing Edge
                  8"         8"         8"         8"
   
   Total span: 32"
   ```

3. **Assembly**
   - Cut leading edge strip (1" wide)
   - Insert carbon spar through ribs
   - Glue ribs to spar
   - Attach leading edge
   - Skin wing with 3mm foam sheet

### Fuselage with Electronics Bay:

```
TOP VIEW:
┌─────────┬──────────────┬─────────┬─────────┐
│ Camera  │   Battery    │  FC/VTX │  Servo  │
│  Bay    │     Bay      │   Bay   │  Tray   │
└─────────┴──────────────┴─────────┴─────────┘
0"        4"            12"       16"       20"

SIDE VIEW:
     Camera (15° tilt)
        ↓
┌───┐  ┌───────────────┬─────────┐
│   │  │               │         │
└───┘  └───────────────┴─────────┘
       Battery          FC/Servos
```

**Component Placement:**
- Camera: Front, 15° upward tilt
- Battery: Middle, removable with velcro
- FC/VTX: Rear, protected
- Servos: Rear, accessible for control horns

### Control Surface Installation:

**Elevator:**
```
Cut line: 75% back from leading edge
Width: Full span
Hinge: Clear tape on bottom
Travel: ±15° (30° total)
```

**Rudder:**
```
Height: 3" × 2"
Hinge: Clear tape on side
Travel: ±20° (40° total)
```

### Weight Budget:
```
Foam structure:    80g
Carbon rods:       15g
Camera:            15g
VTX:              8g
Battery:           80g
Servos (2x):      20g
Wiring:           12g
─────────────────────
TOTAL:            230g

Wing area: ~150 in² (9.7 dm²)
Wing loading: 23.7 g/dm² (good for gliding)
```

---

## Build Tips and Tricks:

### Foam Working Techniques:

1. **Clean Cuts**
   - Use fresh blade
   - Multiple light passes better than one heavy cut
   - Score and snap for straight lines

2. **Beveling**
   - 45° cuts for control surfaces
   - Sand lightly for smooth finish
   - Test fit before gluing

3. **Reinforcement**
   - Hot glue + popsicle stick = strong joints
   - Carbon fiber for high-stress areas
   - Strapping tape for wing leading edges

4. **Painting** (optional)
   - Water-based only (solvent melts foam)
   - Acrylic craft paint works well
   - Thin coats to avoid weight

### Common Mistakes:

❌ **Too much glue**: Adds weight, makes repairs harder
✅ Use minimal glue, just enough to bond

❌ **CG too far back**: Unstable, dangerous
✅ Always check CG before first flight

❌ **Weak joints**: Parts separate in flight
✅ Reinforce high-stress areas (wing/fuselage)

❌ **Battery loose**: Shifts CG in flight
✅ Secure with velcro or strap

---

## Quick Reference: Dimensions

### Glider Sizing Rules:
```
Wingspan-to-chord ratio: 4:1 to 6:1
Tail volume coefficient: 0.5-0.7
CG position: 25-30% MAC (mean aerodynamic chord)
Dihedral angle: 3-7° for stability
```

### Servo Throws:
```
Elevator: ±15° (30° total)
Rudder: ±20° (40° total)
Ailerons (if used): ±20° (40° total)
```

### Battery Selection:
```
Flight time = (Battery mAh × 0.8) / Average Current (mA)

Example:
1000mAh × 0.8 = 800mAh usable
Average draw: 2A (2000mA) for FPV gear
Flight time: 800 / 2000 = 0.4 hours = 24 minutes

Add motor power for powered flight
```

---

These templates provide practical, tested dimensions for successful builds!
