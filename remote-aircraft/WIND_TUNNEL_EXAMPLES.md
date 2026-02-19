# Wind Tunnel Simulation - Usage Examples

This document demonstrates real-world usage of the wind tunnel simulation tools with practical examples.

## Example 1: Designing a Beginner Trainer

**Goal**: Create a stable, easy-to-fly trainer aircraft suitable for beginners.

**Requirements**:
- Low stall speed (< 10 m/s)
- Good stability
- Forgiving flight characteristics
- Easy to build

**Design Approach**:
```bash
python aircraft_designer_cli.py -w 1000 -c 200 --weight 1200 --cruise 12
```

**Results**:
- Stall Speed: 8.5 m/s ✓ (Good - easy hand launch)
- Best L/D: 11.2 (Adequate efficiency)
- Aspect Ratio: 5.0 (Sturdy, not fragile)

**Recommendations**:
1. ✓ Low stall speed makes it beginner-friendly
2. ✓ Low aspect ratio = more durable
3. Consider adding dihedral for roll stability
4. Use Clark-Y airfoil for predictable behavior

---

## Example 2: High-Performance Thermal Glider

**Goal**: Design an efficient glider for thermal soaring and long flight times.

**Requirements**:
- Maximum L/D ratio
- Light weight
- High aspect ratio wings
- Excellent glide performance

**Design Approach**:
```bash
python aircraft_designer_cli.py -w 1500 -c 150 --weight 900 --cruise 15
```

**Results**:
- Stall Speed: 6.6 m/s ✓ (Excellent - stays aloft easily)
- Best L/D: 15.8 ✓ (Outstanding glide ratio)
- Aspect Ratio: 10.0 (High efficiency)
- Glide Ratio: 1:15.8 (travels 15.8m for every 1m of altitude loss)

**Recommendations**:
1. ✓ High L/D perfect for thermal soaring
2. ✓ Low stall speed enables tight thermal circles
3. ⚠ High aspect ratio - build carefully to avoid wing twist
4. Consider carbon fiber spar for strength

---

## Example 3: Fast Sport/Aerobatic Plane

**Goal**: Design a fast, agile aircraft for sport flying and basic aerobatics.

**Requirements**:
- High cruise speed
- Good roll rate (low aspect ratio)
- Robust structure
- Responsive controls

**Design Approach**:
```bash
python aircraft_designer_cli.py -w 900 -c 180 --weight 1500 --cruise 20
```

**Results**:
- Stall Speed: 10.6 m/s (Higher - requires larger field)
- Best L/D: 11.2 (Less efficient but adequate)
- Aspect Ratio: 5.0 (Good for aerobatics)
- Cruise Speed: 20 m/s (Fast and exciting)

**Recommendations**:
1. ✓ Low aspect ratio = excellent roll rate
2. ⚠ Higher stall speed requires experienced pilot
3. ✓ Robust design can handle aerobatic loads
4. Consider symmetric airfoil for inverted flight

---

## Example 4: Comparing Airfoil Types

### Clark-Y (Cambered) Airfoil
```bash
python aircraft_designer_cli.py -w 1200 -c 180 --weight 1400 --airfoil clark_y --cruise 15
```

**Characteristics**:
- CL at 0°: 0.30 (positive lift even at zero angle)
- Better slow flight performance
- Self-stabilizing tendency
- Good for trainers and sport planes

### Symmetric Airfoil
```bash
python aircraft_designer_cli.py -w 1200 -c 180 --weight 1400 --airfoil symmetric --cruise 15
```

**Characteristics**:
- CL at 0°: 0.00 (no lift at zero angle)
- Equal performance upright and inverted
- Required for aerobatic aircraft
- Less drag in high-speed flight

**Comparison**:
- **Clark-Y**: Better for trainers, gliders, sport flying
- **Symmetric**: Required for aerobatics, slightly more efficient at high speed

---

## Example 5: Batch Analysis for Design Optimization

Create multiple design files and analyze them all:

**design_light.json**:
```json
{
  "design_params": {
    "wingspan": 1200,
    "chord": 150,
    "weight": 800,
    "airfoil_type": "clark_y"
  },
  "cruise_speed": 12.0
}
```

**design_standard.json**:
```json
{
  "design_params": {
    "wingspan": 1200,
    "chord": 180,
    "weight": 1400,
    "airfoil_type": "clark_y"
  },
  "cruise_speed": 15.0
}
```

**design_heavy.json**:
```json
{
  "design_params": {
    "wingspan": 1200,
    "chord": 200,
    "weight": 2000,
    "airfoil_type": "clark_y"
  },
  "cruise_speed": 18.0
}
```

Run batch analysis:
```bash
python aircraft_designer_cli.py --batch design_light.json -o results_light.json
python aircraft_designer_cli.py --batch design_standard.json -o results_standard.json
python aircraft_designer_cli.py --batch design_heavy.json -o results_heavy.json
```

Compare the results to find the optimal design.

---

## Example 6: Wing Loading Analysis

**Scenario**: Compare different wing loadings to understand performance trade-offs.

### Low Wing Loading (Glider)
```bash
python aircraft_designer_cli.py -w 1500 -c 150 --weight 900 --cruise 12
# Wing loading: 900g / 2250cm² = 0.40 g/cm²
```
- Result: Stall speed ~6.6 m/s, Best L/D ~15.8
- Use case: Thermal soaring, slow flight

### Medium Wing Loading (Sport)
```bash
python aircraft_designer_cli.py -w 1200 -c 180 --weight 1400 --cruise 15
# Wing loading: 1400g / 2160cm² = 0.65 g/cm²
```
- Result: Stall speed ~8.6 m/s, Best L/D ~12.9
- Use case: Sport flying, general purpose

### High Wing Loading (Fast)
```bash
python aircraft_designer_cli.py -w 1000 -c 180 --weight 1800 --cruise 20
# Wing loading: 1800g / 1800cm² = 1.00 g/cm²
```
- Result: Stall speed ~12.2 m/s, Best L/D ~10.8
- Use case: Fast sport, wind penetration

---

## Example 7: Aspect Ratio Effects

### Low Aspect Ratio (AR = 4)
```bash
python aircraft_designer_cli.py -w 800 -c 200 --weight 1200 --cruise 15
# AR = 800²/160000 = 4.0
```
- More drag (induced drag)
- Better roll rate
- More robust structure
- Good for aerobatics

### Medium Aspect Ratio (AR = 6.7)
```bash
python aircraft_designer_cli.py -w 1000 -c 150 --weight 1000 --cruise 15
# AR = 1000²/150000 = 6.7
```
- Balanced performance
- Good efficiency
- Reasonable strength
- All-around design

### High Aspect Ratio (AR = 10)
```bash
python aircraft_designer_cli.py -w 1500 -c 150 --weight 900 --cruise 15
# AR = 1500²/225000 = 10.0
```
- Excellent L/D ratio
- Lower induced drag
- More fragile (requires careful construction)
- Best for gliders

---

## Example 8: Interactive Design Session

Use interactive mode to experiment:

```bash
python aircraft_designer_cli.py --interactive
```

**Sample Session**:
```
Wingspan (mm) [1000]: 1100
Wing chord (mm) [150]: 160
Aircraft weight (g) [1000]: 1100
Airfoil type (clark_y/symmetric) [clark_y]: clark_y
Cruise speed (m/s) [15]: 14

🔬 Running wind tunnel simulation...

[Results displayed...]

Save results to file? (filename or n) [n]: my_design.json
✓ Results saved to my_design.json
```

---

## Example 9: Understanding Stability

The simulation evaluates longitudinal static stability. Here's how to interpret:

**Stable Design** (Static Margin > 5%):
```
Status: ✓ STABLE
Static Margin: 8.2%
Assessment: Stable
```
- Self-correcting in pitch
- Good for beginners
- Less responsive (more damped)

**Marginally Stable** (Static Margin 1-5%):
```
Status: ✗ UNSTABLE
Static Margin: 2.3%
Assessment: Unstable or marginally stable
```
- May require active pilot input
- Recommended: Increase tail size or move CG forward

**Unstable** (Static Margin < 1%):
```
Status: ✗ UNSTABLE
Static Margin: 0.5%
Assessment: Unstable or marginally stable
```
- Requires experienced pilot or flight controller
- Very responsive (good for aerobatics)
- **Not recommended for beginners**

---

## Example 10: Real-World Validation

Compare simulation with actual RC aircraft:

**Simulated Design**:
```bash
python aircraft_designer_cli.py -w 1200 -c 180 --weight 1400 --cruise 15
```
- Stall Speed: 8.6 m/s (31 km/h)
- Best L/D: 12.9

**Actual HobbyKing Bixler 2** (similar size):
- Measured stall: ~30 km/h ✓ (matches within 10%)
- Reported glide ratio: ~12:1 ✓ (matches well)

The simulation provides realistic preliminary estimates!

---

## Tips for Best Results

1. **Start Conservative**: Use proven designs as baselines
2. **Iterate**: Small changes, test each variation
3. **Weight Estimation**: 
   - Foam + electronics: ~0.8× wingspan (mm) in grams
   - Balsa construction: ~1.2× wingspan
   - 3D printed: ~1.5× wingspan
4. **Validate Assumptions**: Compare with similar existing aircraft
5. **Consider Build Method**: Simulation assumes perfect construction
6. **Safety Factor**: Actual performance may vary ±15%

---

## Common Scenarios Quick Reference

| Scenario | Wingspan | Chord | Weight | AR | Use |
|----------|----------|-------|--------|----|-----|
| Park Flyer | 800mm | 150mm | 600g | 5.3 | Indoor/small field |
| Trainer | 1000mm | 200mm | 1200g | 5.0 | Beginner learning |
| Sport | 1200mm | 180mm | 1400g | 6.7 | General flying |
| Glider | 1500mm | 150mm | 900g | 10.0 | Thermal soaring |
| Aerobatic | 900mm | 180mm | 1500g | 5.0 | 3D flying |
| FPV Cruiser | 1300mm | 200mm | 1800g | 6.5 | Long range FPV |

---

## Integration with GUI

All these analyses can also be performed through the GUI:

1. Launch: `python airframe_designer.py`
2. Choose "Fixed Wing Aircraft" or "Glider"
3. Enter your design parameters
4. Click "🌪️ Wind Tunnel"
5. View comprehensive analysis with visual feedback
6. Save results for documentation

---

## Next Steps

After simulation:
1. **Validate Critical Values**: Check stall speed, stability
2. **Adjust Design**: Iterate based on recommendations
3. **Build and Test**: Start with foam prototype
4. **Measure Performance**: Compare actual vs predicted
5. **Refine**: Update weight estimates for future designs

For more information, see [WIND_TUNNEL_GUIDE.md](WIND_TUNNEL_GUIDE.md).
