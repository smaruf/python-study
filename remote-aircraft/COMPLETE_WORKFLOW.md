# Complete Workflow Example: Building a Fixed Wing Trainer

This document shows a complete end-to-end workflow from design to flying aircraft using the Airframe Designer GUI.

## Project Goal
Build a stable, foam-board fixed-wing trainer aircraft suitable for beginners, with 3D printed reinforcement parts.

---

## Phase 1: Design (5 minutes)

### Step 1: Launch Designer
```bash
cd remote-aircraft
python airframe_designer.py
```

### Step 2: Select Aircraft Type
Click: **✈️ Fixed Wing Aircraft**

### Step 3: Enter Design Parameters

**Wing Dimensions:**
- Wing Span: 1000 mm (1 meter - manageable size)
- Wing Chord: 200 mm (5:1 aspect ratio - stable)
- Wing Thickness: 15% (forgiving airfoil)
- Dihedral Angle: 5° (self-stabilizing)

*Rationale: Large wing area for slow flight, high dihedral for stability*

**Fuselage:**
- Length: 800 mm (0.8x wingspan - proportional)
- Width: 60 mm (fits standard electronics)
- Height: 80 mm (room for battery)

*Rationale: Box fuselage is easy to build with foamboard*

**Tail Surfaces:**
- H-Stab Span: 400 mm (40% of wingspan)
- H-Stab Chord: 100 mm (50% of wing chord)
- V-Stab Height: 150 mm (15% of wingspan)
- V-Stab Chord: 120 mm

*Rationale: Large tail surfaces for stability and control authority*

**Propulsion:**
- Motor Diameter: 22 mm (2204 motor - appropriate for size)
- Motor Length: 28 mm
- Propeller Diameter: 7 inches (slow-fly prop)

*Rationale: Moderate power for gentle flying*

### Step 4: Select Build Options
- Material: **PETG** (weather-resistant for outdoor flying)
- ✓ 3D Print Files
- ✓ Foamboard Templates

### Step 5: Generate
- Click **"Generate Design"**
- Select directory: `~/Documents/trainer_v1`
- Click **"Select Folder"**

### Results Generated:
```
trainer_v1/
├── fixed_wing_design_summary.txt
├── foamboard_templates.txt
└── 3d_print_parts.txt
```

---

## Phase 2: Review Design (10 minutes)

### Open Design Summary
```bash
cat ~/Documents/trainer_v1/fixed_wing_design_summary.txt
```

**Key Specifications:**
- Wing Area: 200 cm² (excellent for slow flight)
- Aspect Ratio: 5.0 (stable, not too sensitive)
- Wing Loading: ~10 g/dm² (very light - flies slow)

**Verification Checklist:**
- ✓ Tail is 30-40% of wing size
- ✓ Fuselage proportional to wingspan
- ✓ Wing loading appropriate for trainer
- ✓ All dimensions buildable with standard foamboard

---

## Phase 3: Prepare Materials (1 day)

### Foamboard Needs
From `foamboard_templates.txt`:
- 2x sheets 5mm foamboard (800 x 600 mm)
- Hot glue gun + glue sticks
- Sharp hobby knife
- Metal ruler
- Cutting mat
- Sandpaper (medium grit)

### 3D Printing Queue
From `3d_print_parts.txt`:

**Part 1: Motor Mount**
- File: motor_mount_2204.stl (would be generated with CadQuery)
- Material: PETG
- Settings: 0.2mm, 30% infill, 240°C
- Time: ~1.5 hours

**Part 2: Wing Center Joiner**
- Dimensions: 160 x 20 x 3 mm
- Material: PETG
- Quantity: 1
- Time: ~2 hours

**Part 3: Servo Mounts (x3)**
- Type: Standard 9g servo mount
- Material: PETG
- Quantity: 3 (2 wing, 1 tail)
- Time: ~1 hour each

**Part 4: Landing Gear Mounts (x2)**
- Material: PETG
- Quantity: 2
- Time: ~1 hour each

**Total Print Time:** ~8.5 hours

### Electronics List
- Flight Controller: CC3D or similar
- ESC: 20A
- Motor: 2204 2300KV
- Servos: 3x 9g micro servos
- Receiver: 2.4GHz (match transmitter)
- Battery: 3S 1000mAh LiPo
- Propeller: 7x4.5 slow-fly

---

## Phase 4: Build Foamboard Structure (4 hours)

### Step 1: Cut Wing Panels (45 min)
From templates:
- Cut 2x wing halves: 500 x 200 mm
- Round leading edge with sandpaper
- Taper trailing edge thin
- Score centerline for dihedral fold

### Step 2: Build Fuselage (1 hour)
- Cut 2x sides: 800 x 80 mm
- Cut 2x top/bottom: 800 x 60 mm
- Glue box together
- Reinforce corners with hot glue

### Step 3: Tail Surfaces (30 min)
- Cut H-stab: 400 x 100 mm
- Cut V-stab: 150 x 120 mm
- Add control surface cuts (elevator, rudder)
- Bevel hinge lines

### Step 4: Assembly (1 hour)
1. Join wing halves with 3D printed center joiner at 5° dihedral
2. Attach wing to fuselage (use CA glue + activator)
3. Mount H-stab to tail
4. Mount V-stab perpendicular to H-stab
5. Install 3D printed motor mount at nose

### Step 5: Finishing (45 min)
- Sand all edges smooth
- Fill gaps with hot glue
- Add reinforcement tape at stress points
- Paint/decorate (optional)

---

## Phase 5: Electronics Installation (2 hours)

### Motor and ESC
1. Mount motor to 3D printed motor mount
2. Solder ESC to motor wires
3. Route ESC wires through fuselage

### Servos
1. Install servos in 3D printed mounts
2. Mount 2x in wing for ailerons
3. Mount 1x in tail for elevator
4. Install rudder servo in fuselage

### Flight Controller
1. Mount FC in center of fuselage
2. Connect ESC signal wire
3. Connect all servo signal wires
4. Install receiver
5. Bind to transmitter

### Battery and Final
1. Install battery tray
2. Route wiring neatly
3. Secure all components with velcro
4. Connect battery (test briefly)
5. Perform control surface checks

---

## Phase 6: Pre-Flight Setup (1 hour)

### Center of Gravity Check
From design summary: CG should be at 25-30% of wing chord from leading edge.
- Calculate: 200mm × 0.275 = 55mm from leading edge
- Mark CG location on wing
- Hold at CG point - should balance level
- Adjust battery position forward/back as needed

### Control Surface Setup
- Set all surfaces to neutral
- Check movement directions:
  * Elevator: stick back = up
  * Rudder: stick right = right
  * Ailerons: stick right = right down, left up
- Set throws:
  * Elevator: ±15mm
  * Rudder: ±20mm
  * Ailerons: ±10mm

### Weight Check
- Target weight: ~200g (per design)
- Actual weight: ___g (measure)
- If over: remove excess glue, lighter battery
- If under: add nose weight for CG

### Range Check
1. Turn on transmitter
2. Turn on aircraft
3. Walk 30m away
4. Test all controls
5. Should have full control at 30m

---

## Phase 7: First Flight (Weather Dependent)

### Pre-Flight Checks
- ✓ CG correct (balances at 55mm from leading edge)
- ✓ Controls move correctly
- ✓ Battery fully charged
- ✓ Propeller secure
- ✓ All wiring secure
- ✓ Range check passed

### Flight Test Conditions
- Wind: <10 km/h (calm day)
- Space: Large open field
- Help: Experienced pilot recommended
- Time: Morning (calm air)

### Launch and Test
1. **Hand Launch**
   - Point into any wind
   - Throttle 75%
   - Gentle throw at slight up-angle
   - Expect immediate climb

2. **Trim for Level Flight**
   - Reduce throttle to 50%
   - Adjust elevator trim for level
   - Adjust rudder trim for straight
   - Note: High dihedral self-levels

3. **Gentle Turns**
   - Small aileron inputs
   - Aircraft should turn smoothly
   - Releases to level automatically

4. **Landing Pattern**
   - Reduce throttle to 30%
   - Gentle descent
   - Aim for soft grass
   - Slight up-elevator just before touchdown

### Expected Performance
- Stall speed: ~15 km/h (very slow)
- Cruise speed: ~25 km/h
- Flight time: ~8-10 minutes (3S 1000mAh)
- Stability: Very high (trainer-level)
- Responsiveness: Gentle (good for learning)

---

## Phase 8: Tuning and Iteration

### After First Flights
Record observations:
- CG position: Too far forward/back?
- Control throws: Too much/little?
- Power: Too much/little?
- Stability: Need more/less?

### Common Adjustments

**If nose-heavy:**
- Move battery back 10mm
- Re-check CG

**If tail-heavy:**
- Move battery forward
- Add nose weight if needed

**If insufficient climb:**
- Reduce wing loading
- Larger propeller
- Higher KV motor

**If too fast/twitchy:**
- Reduce control throws
- Larger wing area in v2
- More dihedral

### Version 2 Design
Based on test flights, re-run Designer with adjusted parameters:
- More wingspan? (slower, more stable)
- Less dihedral? (more responsive)
- Different motor? (more/less power)

---

## Results

### Build Statistics
- Design time: 5 minutes
- Print time: 8.5 hours
- Build time: 6 hours
- Setup time: 1 hour
- **Total:** ~16 hours (over 2-3 days)

### Cost Estimate
- Foamboard: $5
- Electronics: $60
- 3D print filament: $3
- Misc supplies: $7
- **Total:** ~$75

### Achievement Unlocked! 🏆
You now have:
- ✓ Custom-designed aircraft
- ✓ Hand-built structure
- ✓ Flight-tested performance
- ✓ Knowledge for future designs
- ✓ Awesome flying machine!

---

## Next Projects

Now that you've mastered the process:

1. **Design a Glider** (easier - no motor!)
2. **Scale Up** to 1500mm wingspan for longer flights
3. **Try Aerobatic** design with lower dihedral
4. **Experiment** with different wing shapes
5. **Share** your design with others!

---

## Troubleshooting Reference

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Won't fly | CG too far back | Move battery forward |
| Nosedives | CG too far forward | Move battery back |
| Hard to turn | Control throws too small | Increase servo travel |
| Twitchy | Control throws too large | Reduce servo travel |
| Poor glide | Too heavy | Reduce weight |
| Spins | Wing not aligned | Check wing angle |
| Rolls left/right | Unequal dihedral | Rebuild wing joint |

---

**Happy Flying!** ✈️

*Remember: Start with the Designer GUI, but don't be afraid to iterate and improve based on real-world testing!*

