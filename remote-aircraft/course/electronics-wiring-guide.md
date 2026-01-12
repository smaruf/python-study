# Electronics Wiring and Setup Guide

## Component Overview

### Essential Components for FPV Quad

#### 1. Flight Controller (FC)
**Popular choices:**
- **F4**: Budget-friendly, 4-8 UARTs
- **F7**: More processing power, better filtering
- **F405/F722**: Most common for 5" quads

**Key features:**
- Gyroscope: MPU6000 or ICM42688 (best)
- Barometer: Optional (for altitude hold)
- OSD: Built-in on most modern FCs
- Blackbox: SD card or flash memory

#### 2. Electronic Speed Controllers (ESC)
**Types:**
- **Individual**: 4 separate ESCs, more wiring
- **4-in-1**: One board, cleaner build, less weight

**Specifications:**
- **Current rating**: 30A minimum for 5" (40A safer)
- **Firmware**: BLHeli_32 or BLHeli_S
- **Protocol**: DShot300/600/1200 (modern standard)

#### 3. Motors
**Naming convention:** 2306 2400KV
- **2306**: 23mm stator diameter, 06mm stator height
- **2400KV**: RPM per volt (lower KV = more torque, higher KV = more speed)

**Sizing guide:**
```
Quad size → Motor size → KV range → Prop size
3" quad  → 1408-1507  → 3000-4000KV → 3-4"
5" quad  → 2204-2306  → 2300-2700KV → 5-6"
7" quad  → 2806-2810  → 1300-1900KV → 7-8"
```

#### 4. FPV Camera
**Types:**
- **Analog**: 600-1200 TVL, low latency (<10ms)
- **Digital (DJI/Walksnail)**: HD quality, higher latency (~28ms)

**For beginners:** Analog (cheaper, more forgiving)

**Specifications:**
- Sensor: CMOS (most common)
- Format: NTSC (60fps) or PAL (50fps)
- Voltage: 5V (from FC) or 12V (direct from battery)

#### 5. Video Transmitter (VTX)
**Power levels:**
- 25mW: Indoor/short range (<100m)
- 200mW: Standard outdoor (300-500m)
- 600mW+: Long range (requires HAM license)

**Features:**
- SmartAudio/Tramp: Change channel from transmitter
- PitMode: Low power for bench testing
- Frequency: 5.8GHz (5645-5945MHz)

#### 6. Receiver (RX)
**Protocols:**
- **FrSky**: D8, D16, ACCESS
- **TBS**: Crossfire (long range)
- **ExpressLRS**: Open source, long range, low latency
- **Spektrum**: DSM2/DSMX

**Connection types:**
- SBUS: Single wire, most common
- PPM: Single wire, older
- PWM: Multiple wires, obsolete

#### 7. Battery (LiPo)
**Cell count:**
- **3S** (11.1V nominal): Lighter, longer flight time
- **4S** (14.8V nominal): More power, faster

**Capacity vs weight:**
```
Capacity → Weight → Flight time (5" quad)
850mAh  → 85g    → 3-4 min (racing)
1300mAh → 115g   → 4-5 min (freestyle)
1800mAh → 155g   → 5-6 min (cruising)
```

**C-rating:** Discharge rate
```
Example: 1300mAh 75C battery
Max current: 1.3A × 75 = 97.5A
```

---

## Wiring Diagrams

### Basic 5" Quad Wiring

```
                    Battery (XT60 connector)
                         │
                    ┌────┴────┐
                    │         │
                 (+)│         │(-)
                    │   PDB   │  ← Power Distribution Board
                    │         │     (or FC with integrated PDB)
                 (+)│         │(-)
                    │         │
        ┌───────────┼─────────┼────────────┐
        │           │         │            │
    Motor 1     Motor 2   Motor 3      Motor 4
    [ESC 1]     [ESC 2]   [ESC 3]      [ESC 4]
        │           │         │            │
        └───────────┴─────────┴────────────┘
                Signal wires (4) to FC
                         │
                         ▼
              ┌──────────────────────┐
              │   Flight Controller   │
              │  ┌─────────────────┐ │
              │  │ UART1: RX       │ │ ← Receiver (SBUS)
              │  │ UART2: VTX      │ │ ← SmartAudio
              │  │ 5V:    Camera   │ │ ← FPV Camera
              │  │ 5V:    RX       │ │ ← Receiver power
              │  │ 9V:    VTX      │ │ ← VTX power (if needed)
              │  └─────────────────┘ │
              └──────────────────────┘
```

### 4-in-1 ESC Stack

```
Layer view (bottom to top):

┌─────────────────┐
│   VTX (top)     │ ← Video Transmitter
├─────────────────┤
│   FC (middle)   │ ← Flight Controller
├─────────────────┤
│  4-in-1 ESC     │ ← ESC stack
│  (bottom)       │
└─────────────────┘
        │
        │ Motor wires (4×3 = 12 wires)
        └──────────┐
                   ▼
        [M1] [M2] [M3] [M4]

Stack mounting:
- M3 standoffs between layers (typically 20mm)
- Rubber grommets for vibration isolation
- FC in middle for best gyro performance
```

### Wire Gauge Reference

| Current | Wire Gauge (AWG) | Typical Use |
|---------|------------------|-------------|
| 0-3A | 26-28 AWG | Signal wires, camera, VTX |
| 3-10A | 22-24 AWG | 5V/12V power, small motors |
| 10-30A | 18-20 AWG | ESC to motor |
| 30-60A | 14-16 AWG | Battery to PDB/ESC |
| 60A+ | 12 AWG | High-power battery leads |

---

## Step-by-Step Build Process

### Phase 1: Frame Assembly (Day 4, Part 1)

1. **Install Motors on Arms**
   ```
   - Place motor on mount
   - Insert 4× M3×8mm screws
   - Tighten in star pattern
   - Apply threadlocker (optional but recommended)
   ```

2. **Route ESC Wires**
   ```
   If using 4-in-1 ESC:
   - Run motor wires through arms
   - Leave 10-20mm slack at motor
   - Zip tie to arm for strain relief
   
   If using individual ESCs:
   - Mount ESC on arm (use double-sided tape or zip ties)
   - Keep away from carbon fiber (electrical noise)
   - Heatshrink over connections
   ```

3. **Assemble Frame**
   ```
   - Attach arms to center plate
   - Use M3 screws (typically 10-12mm)
   - Check motor rotation clearance
   - Tighten evenly
   ```

### Phase 2: Electronics Installation (Day 4, Part 2)

**Tools needed:**
- Soldering iron (60-80W)
- Solder (60/40 or lead-free)
- Wire strippers
- Helping hands/vice
- Flux (optional but helpful)
- Multimeter

**Safety:**
- Work in ventilated area
- Don't inhale solder fumes
- Unplug battery when not testing
- Double-check polarity before powering

#### Step 1: Solder Motor Wires to ESC

```
4-in-1 ESC layout:
┌─────────────────┐
│  M1   M2   M3   M4  │
│  ABC  ABC  ABC  ABC │ ← A, B, C pads for each motor
│                     │
│    (+) Battery (-)  │ ← Main power pads
│                     │
│      Signal pad     │ ← Connect to FC
└─────────────────────┘

Process:
1. Tin all pads on ESC (apply small amount of solder)
2. Tin motor wire ends
3. Cut motor wires to appropriate length (leave 10mm slack)
4. Solder each wire to corresponding pad
5. Test continuity with multimeter
```

**Motor wire colors:**
- Usually: Red, Black, Blue (or Yellow)
- **Order doesn't matter initially** - we'll swap if motor spins wrong direction

#### Step 2: Connect Battery Lead

```
XT60 connector:
   ___
  |   |  ← Yellow (female on battery)
  | + |     Red (male on ESC)
  |___|

Wiring:
1. Cut battery lead to length (~80mm)
2. Solder red (+) to ESC positive pad
3. Solder black (-) to ESC negative pad
4. Add capacitor (optional but recommended):
   - 470µF-1000µF, 35V+ rating
   - Solder across +/- pads
   - Reduces electrical noise
```

#### Step 3: Mount Stack

```
Standoff configuration:
   FC
   ║ 20mm standoff
   ║
  ESC

Assembly:
1. Install rubber grommets on FC mounting holes
2. Screw standoffs into ESC (M3×20mm)
3. Place FC on standoffs
4. Secure with M3 screws (hand-tight, don't over-torque)
```

#### Step 4: Connect Signal Wire (ESC → FC)

```
Most 4-in-1 ESCs use single wire harness:

ESC connector:
┌───────────┐
│ GND       │ ← Ground
│ M1        │ ← Motor 1 signal
│ M2        │ ← Motor 2 signal
│ M3        │ ← Motor 3 signal
│ M4        │ ← Motor 4 signal
└───────────┘

Connects to FC motor pads:
- Check FC diagram for motor pad layout
- Usually labeled M1-M4 or S1-S4
- Plug in connector or solder individual wires
```

#### Step 5: Wire Camera and VTX

**Camera wiring:**
```
Camera:         FC:
Yellow (video) → CAM (video in)
Red (+)        → 5V
Black (-)      → GND
```

**VTX wiring:**
```
VTX:           FC:
Video in       → CAM (video out/OSD out)
+ (power)      → VBAT or 9V (check VTX spec)
- (ground)     → GND
SmartAudio     → TX2 (UART2 transmit pad)
```

**Antenna safety:**
⚠️ **NEVER power VTX without antenna connected**
- Will burn out transmitter instantly
- Attach antenna BEFORE first power-up
- Use proper RP-SMA connector

#### Step 6: Wire Receiver

**SBUS wiring (most common):**
```
Receiver:      FC:
S (signal)  →  RX1 (UART1 receive pad)
+ (power)   →  5V
- (ground)  →  GND
```

**Alternative: ExpressLRS**
```
ELRS RX:       FC:
TX          →  RX1
RX          →  TX1
5V          →  5V
GND         →  GND
```

#### Step 7: Battery Mounting

```
Methods:
1. Velcro strap (most common)
   - Battery strap through frame
   - Velcro pad on battery and frame
   - Quick battery changes
   
2. Battery pad + strap
   - Foam or rubber pad on frame
   - Protects battery and frame
   - Better vibration isolation
   
3. Printed battery tray (our design)
   - Slide battery into tray
   - Secure with strap
   - Keeps battery centered
```

---

## Flight Controller Configuration

### Betaflight Setup (Day 6, Part 1)

**Download Betaflight Configurator:**
- Available for Windows, Mac, Linux
- https://github.com/betaflight/betaflight-configurator

**Connect to FC:**
1. Plug in USB cable (FC to computer)
2. Open Betaflight Configurator
3. Select COM port
4. Click "Connect"

### Tab-by-Tab Configuration:

#### 1. Setup Tab

**Calibrate Accelerometer:**
```
1. Place quad on level surface
2. Click "Calibrate Accelerometer"
3. Wait for completion
4. Don't move quad during calibration
```

**Configure Board Alignment:**
```
If FC is rotated on frame:
- Yaw: 0°, 45°, 90°, etc.
- Usually leave at 0° if FC arrow points forward
```

#### 2. Ports Tab

**Enable UART for peripherals:**
```
UART1: Serial RX (for receiver)
UART2: Peripherals → VTX (SmartAudio/Tramp)
UART3: Disabled (or GPS if used)

Example:
┌──────┬────────────┬──────────────┐
│ UART │ Serial RX  │ Peripherals  │
├──────┼────────────┼──────────────┤
│ 1    │ ✓ ON       │              │
│ 2    │            │ ✓ VTX        │
│ 3    │            │              │
└──────┴────────────┴──────────────┘
```

**Save and Reboot**

#### 3. Configuration Tab

**ESC/Motor:**
```
ESC Protocol: DSHOT600 (or DSHOT300 if issues)
Motor Poles: 14 (most common for 5" motors)
```

**Board and Sensor:**
```
Accelerometer: ON
Barometer: OFF (not needed for acro flying)
Magnetometer: OFF (causes issues, not needed)
```

**Receiver:**
```
Receiver: Serial-based receiver
Protocol: SBUS (or your receiver type)
Channel Map: AETR1234 (Aileron, Elevator, Throttle, Rudder)
```

**Other Features:**
```
✓ OSD (for on-screen display)
✓ AIRMODE (for better control at low throttle)
✓ ANTI_GRAVITY (reduces wobble during throttle changes)
```

**Save and Reboot**

#### 4. Receiver Tab

**Verify Receiver Connection:**
```
1. Turn on transmitter
2. Move sticks
3. Check bars move in Configurator:
   - Roll (aileron): Right stick left/right
   - Pitch (elevator): Right stick up/down
   - Throttle: Left stick up/down
   - Yaw (rudder): Left stick left/right
```

**Set endpoints:**
```
All channels should show ~1000-2000
If not:
- Adjust endpoints on transmitter
- Ensure full stick deflection
```

#### 5. Modes Tab

**Configure Flight Modes:**

| Mode | Channel | Range | Description |
|------|---------|-------|-------------|
| ARM | AUX1 (Ch5) | 1700-2100 | Enables motors |
| ANGLE | AUX2 (Ch6) | 1000-1300 | Self-leveling |
| HORIZON | AUX2 (Ch6) | 1300-1700 | Mixed mode |
| (none) | AUX2 (Ch6) | 1700-2100 | Acro (full manual) |

**Setup:**
```
1. Click "Add Range"
2. Select mode (e.g., "ARM")
3. Select channel (e.g., AUX1)
4. Adjust range slider to match switch position
5. Test by moving switch and checking yellow highlight
```

**Recommended for beginners:**
- ARM on switch (required)
- ANGLE mode on 3-position switch (learn to fly)
- HORIZON in middle (when comfortable)
- Acro when ready (advanced)

#### 6. OSD Tab

**Enable OSD Elements:**
```
✓ Craft name
✓ Battery voltage
✓ Current draw
✓ mAh used
✓ Flight time
✓ Artificial horizon (if using ANGLE mode)
✓ Warnings
```

**Layout:**
```
Drag elements on preview screen
Position where they don't obstruct view
Save
```

#### 7. Motors Tab

⚠️ **SAFETY: Remove props before testing motors!**

**Check Motor Direction:**
```
Quad layout (top view):
     Front
      ↑
  M1  │  M2
  CCW │  CW     → CCW = Counter-clockwise
  ────┼────      CW = Clockwise
  CW  │  CCW
  M3  │  M4

1. Enable motor control (check box)
2. Slowly increase motor 1 slider
3. Verify motor spins CCW (counter-clockwise)
4. Repeat for all motors
```

**If motor spins wrong direction:**
```
Method 1 (BLHeli_32 ESCs):
- Use BLHeli Configurator
- Connect to ESC
- Check "Motor Direction Reversed"
- Write settings

Method 2 (swap any two motor wires):
- Disconnect battery
- Swap any two wires on motor (e.g., A↔B)
- Test again
```

#### 8. CLI Tab (Command Line)

**Useful commands:**
```
status              → Show FC status
diff all            → Show all changes from defaults
dump               → Show all settings
set name = MyQuad  → Set craft name
save               → Save and reboot
```

---

## First Power-Up Checklist

**Before connecting battery:**
- [ ] All solder joints checked (no bridges)
- [ ] Correct polarity on all connections
- [ ] No bare wires touching each other
- [ ] VTX antenna connected
- [ ] Props REMOVED
- [ ] Fire extinguisher nearby (just in case!)

**First connection:**
```
1. Connect battery (use smoke stopper if available)
2. Immediately check:
   - Any smoke? → DISCONNECT
   - Any hot components? → DISCONNECT
   - Beeps from ESCs? → Good!
   - FC LEDs on? → Good!
3. If all good, check:
   - Video feed in goggles
   - Receiver binding (if not done)
   - USB still works (for Betaflight)
```

---

## Troubleshooting Common Issues

### No Video in Goggles

**Check:**
1. VTX powered? (LED should be on)
2. Antenna connected?
3. Correct frequency/channel?
4. Goggles on same frequency?
5. Camera powered? (5V to camera)
6. Wire connections (video signal path)

**Test:**
- Connect camera directly to goggles (bypass FC OSD)
- If video appears → FC OSD problem
- If no video → Camera or VTX problem

---

### Motors Won't Arm

**Common causes:**
```
Check in Betaflight CLI: "status"

Possible errors:
- "RXLOSS": No receiver signal
  → Check receiver wiring
  → Check transmitter on and bound
  
- "ACCEL": Accelerometer not calibrated
  → Calibrate in Setup tab
  
- "ANGLE": Quad not level
  → Place on level surface
  
- "FAILSAFE": Failsafe active
  → Check receiver connection
  → Ensure throttle at minimum
```

---

### Motor Spins Wrong Direction

**Fix:**
```
Swap any two motor wires (A↔B or B↔C or A↔C)
All combinations work, just swap two
Retest
```

---

### Quad Flips on Takeoff

**Cause:** Motor order or rotation wrong

**Check:**
1. Motor order matches Betaflight layout
2. Each motor rotates correct direction
3. Props installed correct direction
4. Props on correct motors (normal vs reverse)

---

## Advanced Wiring: Telemetry and GPS

### Telemetry Setup (FrSky S.Port)

```
FrSky RX:      FC:
S.Port      →  TX1 (UART1 transmit) + RX1 (with inverter)

In Betaflight Ports tab:
UART1: Serial RX + Telemetry
Save and reboot

Transmitter will display:
- Battery voltage
- RSSI
- Flight time
```

### GPS Setup (Long Range)

```
GPS Module:    FC:
TX          →  RX3 (UART3)
RX          →  TX3
5V          →  5V
GND         →  GND

In Betaflight:
Ports: UART3 → GPS
Configuration: Enable GPS and Magnetometer
Save and reboot

Modes: Enable GPS RESCUE mode (return to home)
```

---

## Cable Management Tips

**Clean build = better performance:**

1. **Route wires along frame**
   - Use zip ties every 30-40mm
   - Avoid wires near props
   - Keep away from hot motors

2. **Use correct lengths**
   - Too long: tangled mess, snag on crashes
   - Too short: strain on solder joints
   - Just right: 10-20mm slack

3. **Protect exposed connections**
   - Heat shrink over solder joints
   - Electrical tape as last resort
   - Keep wires away from carbon (conducts electricity)

4. **Label wires** (for future repairs)
   - Small tags on connector bundles
   - Helps when rebuilding after crash

---

## Summary: Build Sequence

**Day 4 Schedule:**
```
Hour 1-2: Frame assembly, motor installation
Hour 3-4: Soldering ESC and battery leads
Hour 5-6: Stack mounting, camera/VTX wiring
Hour 7-8: Receiver connection, initial testing
```

**Day 6 Schedule:**
```
Hour 1-2: Betaflight configuration
Hour 3-4: Motor direction testing, ARM setup
Hour 5-6: First hover test (props ON, outdoors)
Hour 7-8: Tuning and adjustments
```

---

Ready to wire your build? Follow these guides carefully and double-check every connection! ⚡
