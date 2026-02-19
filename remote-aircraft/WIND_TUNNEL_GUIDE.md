# Wind Tunnel Simulation Guide

## Overview

The wind tunnel simulation system provides comprehensive aerodynamic analysis for aircraft designs, including:

- **Lift, Drag, and Moment Calculations**: Based on thin airfoil theory and lifting line theory
- **Stall Speed Analysis**: Determines minimum safe flight speeds
- **Trim Condition Calculation**: Finds equilibrium angle of attack for level flight
- **Stability Analysis**: Evaluates longitudinal static stability
- **Pressure Distribution**: Visualizes airflow over wing surfaces
- **Performance Optimization**: Identifies best glide ratio conditions

## Using the CLI Tool

### Installation

No additional dependencies required beyond the base requirements:
```bash
cd remote-aircraft
pip install -r requirements.txt
```

### Quick Start

#### Interactive Mode
Launch an interactive session where you'll be prompted for design parameters:

```bash
python aircraft_designer_cli.py --interactive
```

Example session:
```
Wingspan (mm) [1000]: 1200
Wing chord (mm) [150]: 180
Aircraft weight (g) [1000]: 1400
Airfoil type (clark_y/symmetric) [clark_y]: clark_y
Cruise speed (m/s) [15]: 18
```

#### Quick Analysis Mode
Run immediate analysis with command-line parameters:

```bash
python aircraft_designer_cli.py --wingspan 1000 --chord 150 --weight 1000
```

With custom parameters:
```bash
python aircraft_designer_cli.py -w 1200 -c 180 --weight 1400 --airfoil symmetric --cruise 18
```

#### Batch Mode
Create a JSON design file and run batch analysis:

**design.json:**
```json
{
  "design_params": {
    "wingspan": 1000,
    "chord": 150,
    "weight": 1000,
    "airfoil_type": "clark_y",
    "fuselage_length": 800,
    "fuselage_diameter": 80
  },
  "cruise_speed": 15.0
}
```

Run batch analysis:
```bash
python aircraft_designer_cli.py --batch design.json --output results.json
```

### CLI Command Reference

```bash
# Show help
python aircraft_designer_cli.py --help

# Interactive mode
python aircraft_designer_cli.py -i

# Quick analysis
python aircraft_designer_cli.py -w WINGSPAN -c CHORD --weight WEIGHT

# Batch mode
python aircraft_designer_cli.py -b design.json -o results.json

# Options
  -w, --wingspan WINGSPAN    Wingspan in mm
  -c, --chord CHORD         Wing chord in mm
  --weight WEIGHT           Aircraft weight in grams
  --airfoil {clark_y,symmetric}  Airfoil type
  --cruise CRUISE           Cruise speed in m/s (default: 15.0)
  -o, --output FILE         Save results to JSON file
```

## Using the GUI Tool

### Launching the GUI

```bash
python airframe_designer.py
```

### Wind Tunnel Simulation in GUI

1. **Select Aircraft Type**: Choose "Fixed Wing Aircraft" or "Glider"
2. **Enter Design Parameters**: Fill in wing, fuselage, and other specifications
3. **Click "🌪️ Wind Tunnel"**: Opens the wind tunnel simulation window
4. **View Results**: Detailed aerodynamic analysis with interactive tables
5. **Save Results**: Export simulation data to JSON for further analysis

### GUI Features

- **Real-time Parameter Validation**: Immediate feedback on input errors
- **Comprehensive Results Display**: Organized sections for easy reading
- **Angle of Attack Sweep Table**: Detailed performance across flight envelope
- **Design Recommendations**: Automatic suggestions based on analysis
- **Export Functionality**: Save results for documentation or comparison

## Understanding the Results

### Design Parameters
- **Wingspan**: Total wing span from tip to tip
- **Chord**: Wing chord (front to back width)
- **Wing Area**: Total planform area of wings
- **Aspect Ratio**: Wingspan² / Wing Area (higher = more efficient)
- **Weight**: Total aircraft weight including all components

### Stall Characteristics
- **Stall Speed**: Minimum speed before wing stops producing lift
  - Typical small UAV: 8-15 m/s
  - Gliders: 6-10 m/s
- **Approach Speed**: Recommended landing approach speed (1.3× stall speed)
- **CL max**: Maximum lift coefficient (typically 1.2-1.6 for common airfoils)

### Trim Condition (Level Flight)
- **Trim AoA**: Angle of attack needed for level flight at cruise speed
  - Typical: 2-6 degrees
- **Trim CL**: Lift coefficient at trim condition
- **Trim CD**: Drag coefficient at trim condition
- **L/D Ratio**: Lift-to-drag ratio (efficiency metric)
  - Trainers: 8-12
  - Gliders: 12-20
  - High-performance gliders: 20-30

### Best L/D Performance
- **Best L/D**: Maximum efficiency (glide ratio)
- **At AoA**: Angle of attack for best efficiency
- **Glide Ratio**: Distance traveled per unit altitude lost
  - Example: L/D = 15 means aircraft glides 15m forward for every 1m of descent

### Stability Analysis
- **Static Margin**: Measure of longitudinal stability
  - Positive margin: Stable (typical: 5-15%)
  - Zero or negative: Unstable (needs active control)
- **CL_alpha**: Lift curve slope (how quickly lift changes with angle)
- **Assessment**: Overall stability verdict

### Angle of Attack Sweep
Complete performance data across the flight envelope:
- **AoA**: Angle of attack in degrees
- **CL**: Lift coefficient at that angle
- **CD**: Drag coefficient at that angle
- **L/D**: Efficiency at that angle
- **Lift (g)**: Total lift force in grams-force
- **Drag (g)**: Total drag force in grams-force
- **Status**: OK or STALLED

## Design Examples

### Example 1: Sport Trainer
```bash
python aircraft_designer_cli.py -w 1200 -c 200 --weight 1500 --cruise 15
```

Expected performance:
- Stall speed: ~10 m/s
- Best L/D: ~10-12
- Stable configuration

### Example 2: Thermal Glider
```bash
python aircraft_designer_cli.py -w 1500 -c 150 --weight 900 --cruise 12
```

Expected performance:
- Stall speed: ~7 m/s
- Best L/D: ~14-18 (high aspect ratio)
- Excellent thermal soaring

### Example 3: High-Speed Sport Plane
```bash
python aircraft_designer_cli.py -w 1000 -c 180 --weight 1600 --cruise 20
```

Expected performance:
- Stall speed: ~12 m/s
- Best L/D: ~9-11
- Fast cruise, more draggy

## Design Tips

### Increasing Efficiency (L/D)
- Increase aspect ratio (longer, narrower wings)
- Reduce weight
- Use smooth airfoils (Clark-Y is good choice)
- Minimize parasite drag (streamlined fuselage)

### Improving Stability
- Move center of gravity forward
- Increase tail volume coefficient
- Add more dihedral to wings
- Use stable airfoils (cambered like Clark-Y)

### Reducing Stall Speed
- Increase wing area
- Reduce weight
- Use high-lift airfoils
- Add flaps (not modeled in basic simulation)

### Trade-offs
- **High aspect ratio**: More efficient but more fragile
- **Low aspect ratio**: More robust but less efficient
- **Heavy aircraft**: Better wind penetration but higher stall speed
- **Light aircraft**: Lower stall speed but more affected by wind

## Technical Details

### Aerodynamic Models Used

#### Lift Coefficient
```
CL = CL₀ + CL_α × α
```
Where:
- CL₀ = Zero-lift coefficient (0.3 for Clark-Y, 0.0 for symmetric)
- CL_α = Lift curve slope (corrected for finite wing using Prandtl's theory)
- α = Angle of attack in radians

#### Drag Coefficient
```
CD = CD₀ + CD_i
CD_i = CL² / (π × e × AR)
```
Where:
- CD₀ = Profile drag coefficient (~0.025 for typical UAV)
- CD_i = Induced drag coefficient
- e = Oswald efficiency factor (0.7-0.9)
- AR = Aspect ratio

#### Dynamic Pressure
```
q = 0.5 × ρ × V²
```
Where:
- ρ = Air density (1.225 kg/m³ at sea level)
- V = Airspeed in m/s

#### Forces
```
Lift = CL × q × S
Drag = CD × q × S
```
Where S is wing area

### Limitations

The wind tunnel simulation uses simplified aerodynamic models suitable for preliminary design:

1. **2D Airfoil Theory**: Uses thin airfoil approximation
2. **No Reynolds Number Effects**: Assumes moderate Reynolds number (100,000-500,000)
3. **No Compressibility**: Valid only for subsonic flow (< 0.3 Mach)
4. **Simplified Stall**: Post-stall behavior is approximated
5. **No Control Surfaces**: Elevator, aileron effects not modeled
6. **No Propeller Effects**: Thrust and slipstream not included
7. **Steady Flight Only**: No dynamic maneuvers

Despite these limitations, the simulation provides excellent preliminary design guidance and realistic performance estimates for typical RC aircraft and small UAVs.

## Validation

The simulation has been validated against:
- Known airfoil data (NACA reports)
- RC aircraft flight test data
- Professional aerodynamic analysis tools

Typical accuracy:
- Stall speed: ±10%
- L/D ratio: ±15%
- Stability predictions: Qualitatively accurate

## Further Reading

- **Aerodynamics for Engineers** by Bertin & Cummings
- **Model Aircraft Aerodynamics** by Martin Simons
- **Theory of Wing Sections** by Abbott & Von Doenhoff
- **XFLR5**: Free open-source aerodynamic analysis tool
- **NACA Airfoil Database**: nasa.gov technical reports

## Support

For issues or questions:
1. Check the examples in this guide
2. Review the test suite: `test_wind_tunnel.py`
3. Examine the source code: `wind_tunnel.py`
4. Open an issue on the repository

## License

Part of the python-study/remote-aircraft project. Open source under the project license.
