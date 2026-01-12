# 1-Week Practical Course: FPV Drones and Gliders

**Complete hands-on course for foam-board and 3D printed aircraft design and implementation**

---

## 📅 Course Overview

This intensive 1-week course covers the complete journey from theory to flight-ready FPV drones and gliders using both foam-board construction and 3D printing techniques.

### Prerequisites
- Basic understanding of electronics
- Access to a 3D printer (or printing service)
- Basic tools (hobby knife, hot glue gun, soldering iron)
- Safety equipment (goggles, work gloves)

### Materials Needed
- **Foam-board**: Dollar Tree foam-board (5mm), various colors
- **3D Printing**: PLA for prototypes, PETG/Nylon for flight parts
- **Electronics**: Brushless motors (2204-2306), ESCs (30A), flight controller
- **FPV System**: Camera (CMOS 600TVL+), VTX (25-200mW), goggles
- **Battery**: LiPo 3S-4S (1000-1500mAh)
- **Tools**: Soldering iron, hot glue gun, hobby knife, ruler, markers

---

## Day 1: Introduction to FPV and Gliders

### Morning: Theory (4 hours)
**Understanding Flight Principles**
- Wing aerodynamics (lift, drag, airfoil basics)
- Center of Gravity (CG) and Center of Pressure (CP)
- Stability vs maneuverability
- Glider vs powered flight characteristics

**FPV System Basics**
- Camera specs and mounting angles (15-45°)
- Video transmitter (VTX) frequencies and power
- Antenna types (linear vs circular polarized)
- Goggles and receivers

### Afternoon: Hands-on (4 hours)
**Build a Simple Chuck Glider**

Materials:
- 1 sheet foam-board (5mm)
- Hot glue
- Hobby knife

Steps:
1. **Wing Design**: 600mm span, 150mm chord
   ```
   Cut pattern: Rectangular wing with slight dihedral (5°)
   Airfoil: Flat-bottom or undercamber
   ```

2. **Fuselage**: Simple rectangular body (400mm x 40mm x 20mm)
   
3. **Tail**: T-tail configuration
   - Horizontal stabilizer: 200mm span
   - Vertical stabilizer: 100mm height

4. **Assembly**:
   - Mount wing 1/3 from front
   - Add nose weight (coins/washers) for CG balance
   - CG should be 25-30% back from wing leading edge

5. **First Flights**: Hand-launch testing and trim adjustments

**Key Learning**:
- How CG affects stability
- Impact of wing loading
- Trim adjustments (elevator, rudder)

---

## Day 2: Foam-Board Design Fundamentals

### Morning: Advanced Foam-Board Techniques (4 hours)

**Cutting and Shaping**
- Bevel cutting for control surfaces
- Creating compound curves with scoring
- Reinforcement techniques (carbon rods, popsicle sticks)
- Hot wire cutting for airfoils (optional)

**Joining Methods**
- Hot glue: Fast, flexible joints
- White glue: Slower, stronger joints
- Tape hinges for control surfaces
- Embedded carbon fiber for strength

**Design Software Introduction**
- FreeCAD/Fusion 360 basics for planform design
- Using templates and scaling
- Printing templates at 1:1 scale

### Afternoon: Build FPV-Ready Glider (4 hours)

**Design: 800mm Wingspan FPV Glider**

Specifications:
- Wingspan: 800mm
- Chord: 180mm
- Wing area: ~14 dm²
- Weight target: 300-400g (with electronics)
- Flight time: 10-15 minutes gliding

**Build Process**:

1. **Wing Construction**
   - Cut airfoil template (Clark-Y or KF airfoil)
   - Build wing ribs every 100mm
   - Add leading/trailing edge reinforcement
   - Install carbon fiber spar (3mm)
   
2. **Fuselage with Electronics Bay**
   ```
   Dimensions: 500mm x 60mm x 50mm
   Components:
   - Front: FPV camera mount (15° angle)
   - Middle: Battery bay (velcro strap)
   - Rear: Flight controller and VTX
   ```

3. **Control Surfaces**
   - Elevator: Full-span, 25% chord
   - Rudder: 40% of fin height
   - Servos: 9g micro servos (2x)

4. **FPV Integration**
   - Camera: Front-mounted, protected
   - VTX: Rear-mounted with antenna vertical
   - Antenna: Top-mounted for range

**Key Learning**:
- Structural engineering with foam
- Electronics integration planning
- Weight distribution and CG management

---

## Day 3: 3D Printing for Aircraft Parts

### Morning: CAD Design with CadQuery (4 hours)

**Introduction to Parametric Design**

Using the provided Python CAD system:

1. **Setup Environment**
   ```bash
   cd remote-aircraft
   pip install -r requirements.txt
   ```

2. **Understanding Motor Mounts**
   - Study `parts/motor_mount.py`
   - Modify parameters for different motor sizes
   - Generate custom mounts:
   ```python
   # For 2204 motor (28mm diameter)
   mount_2204 = motor_mount(
       motor_diameter=28,
       thickness=5,
       bolt_circle=16,
       bolt_hole=3
   )
   
   # For 2306 motor (30mm diameter)
   mount_2306 = motor_mount(
       motor_diameter=30,
       thickness=6,
       bolt_circle=16,
       bolt_hole=3
   )
   ```

3. **Design Custom Parts**
   - Camera mount with adjustable angle
   - Reinforced battery tray
   - Landing gear mounts
   - Antenna holders

**Exercise**: Modify `parts/camera_mount.py` to add:
- Variable tilt angle (0-45°)
- Mounting holes for M2 screws
- Weight optimization (hollow sections)

### Afternoon: 3D Printing Practice (4 hours)

**Slicing and Print Settings**

Recommended settings for flight parts:

| Part Type | Material | Nozzle | Layer Height | Infill | Orientation |
|-----------|----------|--------|--------------|--------|-------------|
| Motor mount | PETG | 0.4mm | 0.2mm | 50% Gyroid | Flat |
| Arms | Nylon | 0.6mm | 0.28mm | 30% Gyroid | Flat |
| Camera mount | PLA | 0.4mm | 0.2mm | 30% Grid | Upright |
| Battery tray | PETG | 0.4mm | 0.2mm | 20% Honeycomb | Flat |

**Print Queue for Tomorrow**:
1. 4x Motor mounts (2 hours)
2. 4x Arm sections (4 hours)
3. 1x Camera mount (1 hour)
4. 1x Battery tray (2 hours)
5. Landing gear (if using) (1 hour)

**Key Learning**:
- Part orientation affects strength
- Infill patterns for different loads
- Material selection for durability

---

## Day 4: Motor and Electronics Integration

### Morning: Electronics Theory (4 hours)

**Power Systems**
- Battery basics: 3S (11.1V) vs 4S (14.8V)
- C-rating and discharge capability
- Motor KV ratings and prop matching
- ESC selection and calibration

**Flight Controller Setup**
- Betaflight/INAV overview
- Sensor calibration (accelerometer, gyro)
- PID tuning basics
- Flight modes (Angle, Horizon, Acro)

**Wiring Diagram**
```
Battery (+) ──┬──> PDB/FC (+)
              │
              └──> ESC (+) x4
                   
Battery (-) ──┴──> PDB/FC (-)
              │
              └──> ESC (-) x4

FC PWM ──> ESC Signal (x4)
FC 5V ──> Camera/VTX
```

**Component Checklist**:
- [ ] Flight controller (F4/F7)
- [ ] ESCs (30A minimum, 4-in-1 or individual)
- [ ] Motors (2204-2306, 2300-2600KV)
- [ ] Camera (600-1200TVL CMOS)
- [ ] VTX (25-200mW, 5.8GHz)
- [ ] Receiver (SBUS/PPM)

### Afternoon: Build 3D Printed Quad Frame (4 hours)

**Assembly Process**:

1. **Frame Assembly**
   ```python
   # Generate frame parts
   python export_all.py
   ```
   - Print quad_frame_150.stl (5" props)
   - Or quad_frame_180.stl (7" props)

2. **Motor Installation**
   - Mount motors to printed mounts
   - Route ESC wires through arms
   - Secure with M3 screws and Loctite

3. **Electronics Mounting**
   - Stack: FC, ESC (if 4-in-1), VTX
   - Use rubber grommets for vibration damping
   - Cable management with zip ties

4. **FPV System**
   - Camera: Front-mount, 25-30° angle
   - VTX: Top or rear mount
   - Antenna: Vertical, clear of props

5. **Initial Power Test**
   - Smoke test (no props!)
   - Check motor directions
   - Verify video feed

**Key Learning**:
- Wire routing for clean builds
- Vibration isolation techniques
- Safety protocols (remove props for testing)

---

## Day 5: Assembly and Weight Optimization

### Morning: Advanced Weight Analysis (4 hours)

**Using Analysis Tools**

1. **Calculate Part Weights**
   ```python
   from analysis.weight import part_weight
   from materials import PETG, NYLON
   
   # Example: Motor mount volume ~2000 mm³
   mount_weight = part_weight(2000, PETG)
   print(f"Motor mount weight: {mount_weight:.2f} g")
   
   # Quad frame with 4 arms
   arm_volume = 150 * 16 * 12  # length × width × height
   total_arm_weight = 4 * part_weight(arm_volume, NYLON)
   print(f"Frame weight: {total_arm_weight:.2f} g")
   ```

2. **Center of Gravity Calculation**
   ```python
   from analysis.cg import center_of_gravity
   
   # Component positions (mm from front)
   masses = [50, 120, 30, 80]  # camera, battery, FC, motors
   positions = [10, 80, 90, 150]  # positions along fuselage
   
   cg = center_of_gravity(masses, positions)
   print(f"Center of Gravity: {cg:.2f} mm from front")
   ```

3. **Stress Analysis**
   ```python
   from analysis.stress import bending_stress
   
   # Arm under load (example)
   force = 500  # grams-force (crash scenario)
   arm_length = 150  # mm
   inertia = 1152  # mm⁴ (16mm × 12mm beam)
   
   stress = bending_stress(force, arm_length, inertia)
   print(f"Bending stress: {stress:.2f} g/mm²")
   ```

### Afternoon: Optimization Workshop (4 hours)

**Weight Reduction Techniques**:

1. **Hollow Arms**
   - Modify `parts/arm.py` to create hollow tubes
   - Wall thickness: 3-4mm for PETG, 2-3mm for Nylon
   - Weight savings: 30-40%

2. **Optimized Battery Tray**
   - Remove unnecessary material
   - Add drainage holes
   - Use webbing instead of solid walls

3. **Component Selection**
   - Replace heavy parts with lighter alternatives
   - Calculate thrust-to-weight ratio
   - Target: >3:1 for acrobatic flight, >1.5:1 for cruising

**Build Optimization Checklist**:
- [ ] Total weight under 400g (5" quad)
- [ ] CG within 5mm of design point
- [ ] All components secured (shake test)
- [ ] Wiring clean and protected
- [ ] Props balanced
- [ ] Battery secure with strap

**Weight Budget Example (5" Quad)**:
```
Frame (printed):        80g
Motors (4x):            120g (30g each)
ESCs (4-in-1):          25g
Flight Controller:      8g
Camera:                 15g
VTX:                    8g
Receiver:               5g
Wiring/connectors:      20g
Battery (3S 1300mAh):   115g
─────────────────────────
TOTAL:                  396g

Thrust (4x motor):      >1200g
Thrust-to-weight:       3.0:1 ✓
```

**Key Learning**:
- Every gram matters in flight performance
- Structural integrity vs weight tradeoff
- System-level optimization

---

## Day 6: Flight Testing and Tuning

### Morning: Pre-flight and Safety (4 hours)

**Safety Protocols**
- [ ] Check all prop nuts tight
- [ ] Verify motor direction
- [ ] Arm/disarm function test
- [ ] Range test receiver (100m minimum)
- [ ] Failsafe configured (motor off)
- [ ] Clear flight area (no people/animals)
- [ ] Observer/spotter present
- [ ] First aid kit available

**Flight Controller Configuration**

Using Betaflight Configurator:

1. **Ports Setup**
   - UART1: Serial RX
   - UART2: VTX control (if supported)
   - UART3: Disabled or OSD

2. **PID Tuning (Initial Values)**
   ```
   Roll:  P=45  I=80  D=30
   Pitch: P=50  I=85  D=32
   Yaw:   P=80  I=90  D=0
   ```

3. **Rates Configuration**
   ```
   Rate:       0.70
   Super Rate: 0.80
   RC Expo:    0.00
   ```

4. **Modes Setup**
   - Channel 5: Arm
   - Channel 6: Flight mode (Angle/Horizon/Acro)
   - Channel 7: Turtle mode (optional)

### Afternoon: First Flights (4 hours)

**Flight Test Procedure**:

**Test 1: Hover Test (Angle Mode)**
1. Arm at 1m altitude (hand-held or small stand)
2. Gradually increase throttle to hover
3. Test pitch/roll response
4. Land smoothly
5. Check for overheating/vibration

**Test 2: Orientation Flight**
1. Takeoff to 3m altitude
2. Hover 10 seconds
3. Gentle forward/back movements
4. Gentle left/right movements
5. Land

**Test 3: Circle Flight**
1. Fly large circle pattern (10m radius)
2. Maintain constant altitude
3. Smooth transitions
4. Return to launch point

**Test 4: FPV Test**
1. Start with line-of-sight observer
2. Put on goggles
3. Hover and verify video clarity
4. Short FPV flight (low altitude)
5. Return to visual flying

**Common Issues and Fixes**:

| Issue | Cause | Solution |
|-------|-------|----------|
| Drifts forward | CG too far back | Move battery forward |
| Oscillations | PIDs too high | Reduce P and D values |
| Poor video | Antenna position | Adjust VTX antenna |
| Motors hot | Props wrong size | Use smaller props |
| Won't arm | Sensor not calibrated | Recalibrate in FC |

**Flight Log**:
Record each flight:
- Date/Time
- Flight duration
- Battery voltage (start/end)
- Issues encountered
- Tuning changes made

**Key Learning**:
- Progressive testing minimizes crashes
- Logging helps track improvements
- Safety always comes first

---

## Day 7: Advanced Designs and Customization

### Morning: Advanced Concepts (4 hours)

**Wing Design Optimization**

1. **Airfoil Selection**
   - Flat-bottom: Easy, stable, forgiving
   - Undercamber: More lift, slower flight
   - Symmetrical: Aerobatic, neutral stability
   - Semi-symmetrical: Balanced performance

2. **Wing Loading Calculation**
   ```python
   def wing_loading(weight_grams, area_dm2):
       return weight_grams / area_dm2
   
   # Example: 400g quad with 14 dm² area
   loading = wing_loading(400, 14)
   print(f"Wing loading: {loading:.1f} g/dm²")
   
   # Guidelines:
   # < 20 g/dm²: Trainer, slow flight
   # 20-40 g/dm²: Sport flying
   # > 40 g/dm²: Fast, aerobatic
   ```

3. **Advanced Frame Designs**
   
   **Stretch-X Configuration**:
   ```python
   # Modify frames/quad_frame.py
   def stretch_x_frame(arm_length=150, stretch_factor=1.3):
       arms = []
       angles = [30, 150, 210, 330]  # Wider front/back spacing
       
       for i, angle in enumerate(angles):
           length = arm_length * stretch_factor if i % 2 else arm_length
           arm = (
               drone_arm(length=length)
               .rotate((0,0,0), (0,0,1), angle)
           )
           arms.append(arm)
       
       frame = arms[0]
       for arm in arms[1:]:
           frame = frame.union(arm)
       
       return frame
   ```

**Long-Range FPV Design**:
- Wing-assisted multirotor (hybrid)
- Efficient props (2-blade, high pitch)
- High-capacity battery (3000mAh+)
- GPS and return-to-home
- Directional antennas

### Afternoon: Final Projects (4 hours)

**Choose One Advanced Build**:

**Option A: 7" Long-Range Quad**
- Specifications:
  - Wingspan: 180mm arms (diagonal 360mm)
  - Motors: 2806 1500KV
  - Props: 7" tri-blade
  - Battery: 4S 2200mAh
  - Target flight time: 15+ minutes
  - Range: 2-5km

**Option B: Flying Wing FPV**
- Specifications:
  - Wingspan: 1000mm
  - Foam-board construction
  - 2x motors (pusher config)
  - Elevons for control
  - Target speed: 50-70 km/h
  - Excellent efficiency

**Option C: 3D Printed Park Flyer**
- Specifications:
  - Wingspan: 600mm
  - Full 3D printed parts
  - Single motor puller
  - Lightweight (200g total)
  - Gentle flying characteristics

### Project Execution:

1. **Design Phase** (1 hour)
   - Sketch layout
   - Calculate CG
   - Plan component placement
   - Create build plan

2. **Build Phase** (2 hours)
   - Cut/print parts
   - Assemble structure
   - Install electronics
   - Wire and test

3. **Test Flight** (1 hour)
   - Pre-flight check
   - Maiden flight
   - Tune and adjust
   - Final evaluation

**Key Learning**:
- Apply full week of knowledge
- Design-to-flight complete cycle
- Problem-solving real challenges

---

## 📊 Course Summary and Next Steps

### Skills Acquired:
✅ Foam-board construction techniques
✅ 3D CAD design with Python
✅ Electronics integration and wiring
✅ Flight controller configuration
✅ Weight and CG analysis
✅ Flight testing and tuning
✅ Safety protocols and best practices

### Builds Completed:
1. Chuck glider (Day 1)
2. FPV foam-board glider (Day 2)
3. 5" 3D printed quadcopter (Days 3-6)
4. Advanced custom design (Day 7)

### Recommended Next Steps:

**Continue Learning**:
1. Join FPV communities (forums, Discord, local clubs)
2. Study advanced PIDs and filters (Betaflight)
3. Learn about GPS and autonomous flight
4. Explore fixed-wing FPV (long range)
5. Design custom parts for specific needs

**Safety and Legal**:
- Register aircraft if required (FAM regulations)
- Get HAM license for higher VTX power
- Learn local flying regulations
- Practice in safe, legal areas
- Consider insurance

**Advanced Topics**:
- Carbon fiber construction
- Custom flight controller firmware
- Telemetry and OSD programming
- FPV racing techniques
- Autonomous missions with INAV/ArduPilot

---

## 📚 Resources

### Software:
- **CadQuery**: Python-based parametric CAD
- **Betaflight Configurator**: FC setup and tuning
- **INAV Configurator**: GPS and autonomous flight
- **OpenTX**: Transmitter firmware
- **FreeCAD**: General CAD design

### Communities:
- **Joshua Bardwell**: YouTube (FPV expert)
- **Flite Test**: Foam-board designs and tutorials
- **r/Multicopter**: Reddit community
- **IntFPV**: Forum and resources
- **RCGroups**: General RC aircraft

### Suppliers:
- **Electronics**: Banggood, GetFPV, RaceDayQuads
- **Foam-board**: Dollar Tree, craft stores
- **3D Printing**: PLA/PETG from Amazon, Nylon from specialty shops
- **Props**: HQProp, DAL, Gemfan
- **Batteries**: Tattu, GNB, CNHL

---

## 🎯 Final Assessment

Rate your understanding (1-5):
- [ ] Aerodynamics and flight principles
- [ ] Foam-board construction
- [ ] 3D CAD design
- [ ] Electronics and wiring
- [ ] Flight controller setup
- [ ] PID tuning
- [ ] Safe flying practices

**Congratulations on completing the course!**

Now get out there and fly! 🚁✈️

---

*Remember: Safety first, have fun, and never stop learning!*
