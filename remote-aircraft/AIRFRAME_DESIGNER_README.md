# Airframe Designer GUI

A graphical user interface for designing Fixed Wing Aircraft and Gliders with parametric inputs. Generate both foamboard cutting templates and 3D printable parts specifications.

## Features

### ✈️ Fixed Wing Aircraft Designer
- Complete parametric design for fixed-wing aircraft
- Configurable wing, fuselage, tail, and propulsion parameters
- Automatic performance calculations (wing loading, aspect ratio)
- Generate foamboard cutting templates
- Generate 3D printable parts specifications

### 🪂 Glider Designer
- Optimized for unpowered glider designs
- Tapered wing configuration support
- High aspect ratio wings for efficient gliding
- Foamboard construction templates
- 3D printable reinforcement parts

## Quick Start

### Installation

No additional dependencies required beyond standard Python packages! The GUI uses Tkinter which comes with Python.

```bash
cd remote-aircraft
python airframe_designer.py
```

### Usage

1. **Launch the application:**
   ```bash
   python airframe_designer.py
   ```

2. **Select aircraft type:**
   - Choose "Fixed Wing Aircraft" for powered planes
   - Choose "Glider" for unpowered sailplanes

3. **Enter parameters:**
   - Wing dimensions (span, chord, thickness)
   - Fuselage dimensions
   - Tail surface sizes
   - Motor specifications (fixed wing only)
   - Select 3D print material
   - Choose output options

4. **Generate design:**
   - Click "Generate Design"
   - Select output directory
   - Review generated files

## Output Files

### Design Summary
- Complete specifications
- Performance estimates
- Build recommendations

### Foamboard Templates
- Cutting dimensions for all parts
- Assembly instructions
- Material requirements

### 3D Print Parts
- Parts list with dimensions
- Print settings for selected material
- Assembly notes

## Parameters Guide

### Wing Parameters

**Wing Span**: Total width of the aircraft from tip to tip
- Fixed Wing: 800-1200mm typical
- Glider: 1000-1500mm for better glide

**Wing Chord**: Width of wing from leading to trailing edge
- Affects wing area and lift
- Larger chord = more lift, less speed

**Wing Thickness**: Airfoil thickness as percentage
- 10-12%: Fast, efficient
- 12-15%: General purpose
- 15%+: Slow, stable trainer

**Dihedral Angle**: Upward angle of wings
- 0-3°: Aerobatic, less stable
- 3-5°: General purpose
- 5-7°: Very stable, trainer

### Fuselage Parameters

**Length**: Overall fuselage length
- Should be 0.7-0.9x wing span
- Longer = more stable

**Width/Height**: Cross-section dimensions
- Must fit electronics
- Smaller = less drag

### Tail Parameters

**Horizontal Stabilizer**: Rear wing for pitch control
- Span: ~30-40% of wing span
- Chord: ~40-50% of wing chord

**Vertical Stabilizer**: Tail fin for yaw control
- Height: ~15-20% of wing span
- Should extend above and below fuselage

### Motor Parameters (Fixed Wing Only)

**Motor Diameter**: Stator diameter in mm
- 22mm: Small parkflyer
- 28mm: General purpose
- 30mm+: Larger models

**Propeller Diameter**: Prop size in inches
- Smaller prop = higher speed
- Larger prop = more thrust at low speed

## Material Selection

### PLA
- **Pros**: Easy to print, cheap, stiff
- **Cons**: Brittle, low temperature resistance
- **Best for**: Indoor models, light loads

### PETG
- **Pros**: Strong, flexible, weather resistant
- **Cons**: Slightly harder to print
- **Best for**: General purpose outdoor flying

### Nylon
- **Pros**: Very strong, excellent flexibility
- **Cons**: Difficult to print, absorbs moisture
- **Best for**: High-stress parts, crash resistance

### CF-Nylon
- **Pros**: Maximum strength-to-weight
- **Cons**: Expensive, requires hardened nozzle
- **Best for**: Racing, competition builds

## Design Tips

### Fixed Wing Aircraft
1. **Balance Point (CG)**: Should be 25-30% back from wing leading edge
2. **Wing Loading**: Keep under 30 g/dm² for trainers
3. **Thrust-to-Weight**: Aim for 1.5:1 minimum, 2:1+ for aerobatics
4. **Tail Volume**: H-stab should be 25-30% of wing area

### Gliders
1. **Balance Point (CG)**: 25-33% of mean aerodynamic chord
2. **Wing Loading**: Lower is better - aim for <20 g/dm²
3. **Aspect Ratio**: Higher = better glide (8:1 to 12:1)
4. **Launch Method**: Hand launch or high-start winch

## Example Designs

### 800mm Trainer (Fixed Wing)
- Wing Span: 800mm
- Wing Chord: 180mm
- Dihedral: 5°
- Motor: 2204 or 2206
- Propeller: 7-8 inches

### 1200mm Glider
- Wing Span: 1200mm
- Root Chord: 220mm
- Tip Chord: 150mm
- Dihedral: 5°
- Weight target: <150g

## Building Process

### Foamboard Construction
1. Print templates (actual size)
2. Transfer to 5mm foamboard
3. Cut carefully with sharp knife
4. Sand edges smooth
5. Glue parts together
6. Reinforce with 3D printed parts
7. Install electronics
8. Balance and test

### 3D Printing
1. Generate parts specifications
2. Slice with recommended settings
3. Print reinforcement parts
4. Install into foamboard structure
5. Add electronics
6. Final assembly and balance

## Troubleshooting

### CG Too Far Forward
- Move battery backward
- Add weight to tail
- Reduce nose weight

### CG Too Far Back
- Move battery forward
- Add nose weight
- Check motor/prop weight

### Poor Glide (Gliders)
- Check CG position
- Reduce weight
- Increase wing area
- Improve wing surface finish

### Insufficient Thrust (Fixed Wing)
- Check motor/prop combination
- Reduce weight
- Increase battery voltage
- Use larger propeller

## Advanced Features

Future enhancements could include:
- Actual STL file generation (requires CadQuery)
- Visual 3D preview
- Automated CG calculation with component placement
- Airfoil selection and design
- Performance prediction graphs
- Export to CAD formats

## Support

For issues or questions:
1. Check parameters are reasonable
2. Verify output files are generated
3. Review design summary for warnings
4. Consult `course/README.md` for building guidance
5. See `USAGE.md` for more examples

## License

Part of the remote-aircraft learning repository. Open source and free to use.

---

**Happy Building!** ✈️🪂

