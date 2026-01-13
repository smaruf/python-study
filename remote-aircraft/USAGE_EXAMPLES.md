# Airframe Designer - Usage Examples

This guide provides step-by-step examples of using the Airframe Designer GUI.

## Example 1: Design a Simple Fixed Wing Trainer

### Objective
Create a stable, easy-to-fly fixed wing trainer aircraft using foamboard construction.

### Steps

1. **Launch the Application**
   ```bash
   cd remote-aircraft
   python airframe_designer.py
   ```

2. **Select Aircraft Type**
   - Click on "✈️ Fixed Wing Aircraft"

3. **Enter Parameters**
   
   **Wing Dimensions:**
   - Wing Span: 1000 mm (good size for beginners)
   - Wing Chord: 200 mm (stable flight characteristics)
   - Wing Thickness: 15% (forgiving airfoil)
   - Dihedral Angle: 5° (self-stabilizing)

   **Fuselage:**
   - Length: 800 mm (proportional to wingspan)
   - Width: 60 mm (room for electronics)
   - Height: 80 mm (battery compartment)

   **Tail Surfaces:**
   - H-Stab Span: 400 mm (40% of wingspan)
   - H-Stab Chord: 100 mm (50% of wing chord)
   - V-Stab Height: 150 mm (15% of wingspan)
   - V-Stab Chord: 120 mm

   **Propulsion:**
   - Motor Diameter: 28 mm (2206 or 2306 motor)
   - Motor Length: 30 mm
   - Propeller Diameter: 8 inches

4. **Build Options**
   - Material: PETG (good outdoor durability)
   - ✓ Generate 3D Print Files
   - ✓ Generate Foamboard Templates

5. **Generate Design**
   - Click "Generate Design"
   - Select output directory (e.g., ~/Documents/trainer_v1)
   - Click "Select Folder"

6. **Review Output**
   - `fixed_wing_design_summary.txt` - Complete specifications
   - `foamboard_templates.txt` - Cutting guide
   - `3d_print_parts.txt` - Parts to print

### Expected Results
- Wing Area: 200 cm²
- Aspect Ratio: 5.0 (good for stability)
- Wing Loading: ~10 g/dm² (light and forgiving)
- Design files ready for construction!

---

## Example 2: High-Performance Glider

### Objective
Create an efficient glider with excellent glide performance for thermal soaring.

### Steps

1. **Launch the Application**
   ```bash
   python airframe_designer.py
   ```

2. **Select Aircraft Type**
   - Click on "🪂 Glider"

3. **Enter Parameters**
   
   **Wing Dimensions:**
   - Wing Span: 1500 mm (longer span = better glide)
   - Root Chord: 200 mm
   - Tip Chord: 120 mm (tapered for efficiency)
   - Wing Thickness: 12% (efficient airfoil)
   - Dihedral Angle: 4° (lateral stability)

   **Fuselage:**
   - Length: 750 mm (sleek design)
   - Width: 45 mm (minimal drag)
   - Height: 55 mm (streamlined)

   **Tail Surfaces:**
   - H-Stab Span: 450 mm
   - H-Stab Chord: 80 mm
   - V-Stab Height: 140 mm
   - V-Stab Chord: 90 mm

4. **Build Options**
   - Material: PLA (lightweight for gliders)
   - ✓ Generate 3D Print Files
   - ✓ Generate Foamboard Templates

5. **Generate Design**
   - Click "Generate Design"
   - Choose output directory
   - Review generated files

### Expected Results
- Wing Area: 240 cm²
- Aspect Ratio: 9.38 (excellent for gliding)
- Wing Loading: ~0.63 g/dm² (very light)
- Estimated Glide Ratio: 7.5:1 (good performance)

---

## Example 3: Aerobatic Fixed Wing

### Objective
Fast, agile fixed wing for aerobatic maneuvers and racing.

### Steps

1. **Select Fixed Wing Aircraft**

2. **Enter Parameters**
   
   **Wing Dimensions:**
   - Wing Span: 800 mm (shorter for agility)
   - Wing Chord: 180 mm
   - Wing Thickness: 10% (fast, symmetrical airfoil)
   - Dihedral Angle: 1° (minimal self-leveling)

   **Fuselage:**
   - Length: 700 mm
   - Width: 55 mm
   - Height: 70 mm

   **Tail Surfaces:**
   - H-Stab Span: 320 mm
   - H-Stab Chord: 90 mm
   - V-Stab Height: 120 mm
   - V-Stab Chord: 100 mm

   **Propulsion:**
   - Motor Diameter: 30 mm (2806 or larger)
   - Motor Length: 32 mm
   - Propeller Diameter: 9 inches (high thrust)

3. **Build Options**
   - Material: CF_NYLON (maximum strength)
   - ✓ Generate 3D Print Files
   - ✓ Generate Foamboard Templates

### Expected Results
- Wing Area: 144 cm²
- Aspect Ratio: 4.44 (maneuverable)
- Wing Loading: ~13.9 g/dm² (fast flying)
- High thrust-to-weight for vertical performance

---

## Example 4: Long-Duration Glider

### Objective
Maximum endurance glider for extended flying sessions.

### Parameters

**Wing:**
- Span: 2000 mm (very long for efficiency)
- Root Chord: 250 mm
- Tip Chord: 180 mm
- Thickness: 14%
- Dihedral: 6° (very stable)

**Fuselage:**
- Length: 900 mm
- Width: 50 mm
- Height: 60 mm

**Tail:**
- H-Stab: 500 x 90 mm
- V-Stab: 160 x 100 mm

**Material:** PLA (lightest option)

### Expected Results
- Wing Area: 430 cm²
- Aspect Ratio: 9.30
- Wing Loading: ~0.35 g/dm²
- Estimated Glide Ratio: 7.4:1
- Excellent thermal performance!

---

## Tips for Best Results

### Design Phase
1. **Start Conservative**: Use default values first
2. **Check Ratios**: Ensure tail is 30-40% of wing size
3. **Balance**: Fuselage length should be 0.7-0.9x wingspan
4. **CG Planning**: Note weight estimates in summary

### Material Selection
- **PLA**: Best for gliders (lightweight)
- **PETG**: Best all-around for general use
- **Nylon**: Best for crash resistance
- **CF-Nylon**: Best for racing/competition

### Building Process
1. **Review all files** before starting construction
2. **Cut carefully** following foamboard templates
3. **Print parts** with recommended settings
4. **Dry-fit** before gluing
5. **Balance carefully** - CG is critical!
6. **Test glide** before powered flight

### Common Issues

**Design won't generate?**
- Check all fields have valid numbers
- Ensure values are reasonable (no negative numbers)
- Try default values first

**Poor glide performance?**
- Check CG position (should be 25-33% back)
- Reduce weight
- Sand wing surfaces smooth
- Check wing alignment

**Unstable flight?**
- Increase dihedral
- Move CG forward
- Increase tail size
- Add more wing area

---

## File Output Reference

### Design Summary File
Contains:
- All input parameters
- Calculated specifications
- Performance estimates
- Build recommendations

### Foamboard Templates File
Contains:
- Part dimensions
- Cutting instructions
- Material requirements
- Assembly steps

### 3D Print Parts File
Contains:
- Parts list with sizes
- Material-specific print settings
- Quantities needed
- Assembly notes

---

## Next Steps After Design

1. **Review Summary**
   - Check performance estimates
   - Verify all dimensions
   - Note any warnings

2. **Prepare Materials**
   - Get foamboard (5mm recommended)
   - Start 3D prints
   - Gather tools

3. **Build**
   - Follow templates
   - Install 3D printed reinforcements
   - Add electronics

4. **Test**
   - Check CG
   - Do glide test
   - Adjust as needed

5. **Fly!**
   - Start with gentle launches
   - Trim for level flight
   - Enjoy your custom design!

---

## Advanced Customization

For users familiar with Python, you can:
- Modify default values in the code
- Add custom calculations
- Create preset designs
- Export to other formats

See `airframe_designer.py` source code for implementation details.

---

**Have fun designing and building!** ✈️🪂

