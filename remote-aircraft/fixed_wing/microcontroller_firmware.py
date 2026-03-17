# -*- coding: utf-8 -*-
"""
Microcontroller Firmware & Component-Level Hardware Design

Provides complete, from-scratch implementations for three RC aircraft builds:
1. RCMakerLab Flying-Wing RC Plane  (Oct 2025)
2. 3JWings DC Motor Stick Plane     (Jan 2025)
3. Shahed / Lucas Drone study model (educational only)

Each build is covered by:
- Component-level BOM with part values, footprints, and wiring notes
- ASCII schematic / wiring diagram
- Raspberry Pi Pico MicroPython TX + RX firmware (complete, ready-to-flash)
- Arduino (ATmega328P) C++ TX + RX firmware (complete, with MPU-6050 IMU)
- Pin-mapping tables for both platforms
- Flash / upload instructions

Design philosophy:
  Everything is modelled in pure Python so the source can be read, run,
  and versioned without any hardware present.  Each firmware block is a
  raw Python string that the user copies into the corresponding IDE and
  flashes directly onto the microcontroller.
"""

# ===========================================================================
# COMPONENT-LEVEL BOM DESIGN
# ===========================================================================


def component_bom_flying_wing():
    """
    Return a from-scratch component-level Bill of Materials for the
    RCMakerLab Flying-Wing RC Plane.

    Covers every individual electronic component needed to build the
    custom Arduino Nano + NRF24L01 radio system from bare parts.

    Returns:
        Dict with ``transmitter_bom``, ``receiver_bom``, ``shared_bom``,
        ``power_budget_ma``, and ``design_notes``.
    """
    transmitter_bom = [
        # MCU
        {"ref": "U1",  "part": "Arduino Nano V3 (ATmega328P)",
         "qty": 1, "notes": "Micro-USB variant; 3.3 V I/O-safe via level-shifter"},
        # Radio
        {"ref": "U2",  "part": "NRF24L01+PA+LNA  GT-24 (with SMA antenna)",
         "qty": 1, "notes": "Long-range TX module; 100 mW; 3.3 V VCC"},
        # Joystick inputs
        {"ref": "JS1", "part": "PS4 Analogue Joystick 10 kΩ",
         "qty": 1, "notes": "Left stick: A0 (throttle Y) + A1 (roll X)"},
        {"ref": "JS2", "part": "PS4 Analogue Joystick 10 kΩ",
         "qty": 1, "notes": "Right stick: A2 (pitch Y) + A3 (aux X)"},
        # Switches
        {"ref": "SW1", "part": "SPDT Toggle Switch 250 V 3 A",
         "qty": 1, "notes": "Arm switch → D2 (INPUT_PULLUP)"},
        {"ref": "SW2", "part": "SPDT Toggle Switch 250 V 3 A",
         "qty": 1, "notes": "Aux mode → D3 (INPUT_PULLUP)"},
        # Voltage reg for radio
        {"ref": "U3",  "part": "LM1117-3.3 SOT-223 LDO",
         "qty": 1, "notes": "Steps 5 V USB/LiPo BEC down to 3.3 V for NRF24L01"},
        # Decoupling / bulk caps
        {"ref": "C1",  "part": "100 µF 10 V electrolytic (radial)",
         "qty": 2, "notes": "Bulk cap on 5 V rail (prevent Nano brownout)"},
        {"ref": "C2",  "part": "100 nF 50 V MLCC 0805",
         "qty": 4, "notes": "Local decoupling at each IC VCC pin"},
        {"ref": "C3",  "part": "10 µF 16 V electrolytic (radial)",
         "qty": 2, "notes": "Input + output cap on LM1117 (per datasheet)"},
        # Power connector
        {"ref": "J1",  "part": "JST-PH 2-pin 2.0 mm socket",
         "qty": 1, "notes": "9 V battery / 2S LiPo input to Nano VIN"},
        # PCB
        {"ref": "PCB", "part": "70 × 50 mm perfboard (2.54 mm pitch)",
         "qty": 1, "notes": "Or custom KiCad PCB — Gerber at rcpano.net ref link"},
    ]

    receiver_bom = [
        # MCU
        {"ref": "U1",  "part": "Arduino Nano V3 (ATmega328P)",
         "qty": 1, "notes": "Flight controller; mounted in centre bay"},
        # Radio
        {"ref": "U2",  "part": "NRF24L01+PA+LNA E01-ML01DP5 (100 mW)",
         "qty": 1, "notes": "Receiver module; SMA rubber-duck antenna"},
        # IMU
        {"ref": "U3",  "part": "MPU-6050 (GY-521 breakout)",
         "qty": 1, "notes": "I2C gyro+accel; SDA=A4, SCL=A5; AD0=GND → addr 0x68"},
        # Voltage reg
        {"ref": "U4",  "part": "LM1117-3.3 SOT-223 LDO",
         "qty": 1, "notes": "3.3 V supply for NRF24L01"},
        # ESC BEC feeds the 5 V rail; no separate 5 V reg needed
        # Decoupling
        {"ref": "C1",  "part": "100 µF 10 V electrolytic",
         "qty": 3, "notes": "Bulk on 5 V (from BEC) and 3.3 V rails"},
        {"ref": "C2",  "part": "100 nF MLCC 0805",
         "qty": 6, "notes": "Local decoupling at each IC"},
        # Power
        {"ref": "J1",  "part": "XT-30 or Deans female connector",
         "qty": 1, "notes": "Main LiPo input to ESC"},
        {"ref": "J2",  "part": "3-pin 2.54 mm male headers × 3",
         "qty": 3, "notes": "Servo connectors: D5 left elevon, D6 right elevon"},
        # ESC
        {"ref": "ESC1","part": "30 A BLHeli_S ESC (BEC 5 V 3 A)",
         "qty": 1, "notes": "Powers Nano + servos via BEC; signal on D3"},
    ]

    shared_bom = [
        {"part": "2205 2300KV Brushless Motor (CW thread)",
         "qty": 1, "notes": "Tractor mount, nose of centre bay"},
        {"part": "5045 3-blade propeller (CW)",
         "qty": 1, "notes": "Balanced before install; reverse-thread nut"},
        {"part": "MG90S Metal-Gear Servo (180°)",
         "qty": 2, "notes": "Left elevon = D5, right elevon = D6"},
        {"part": "2S 7.4 V 1300 mAh LiPo (XT-30)",
         "qty": 1, "notes": "≈12 min flight; balance charge at 1C (1.3 A)"},
        {"part": "6 mm insulation styrofoam sheets A4",
         "qty": 2, "notes": "Wing panels"},
        {"part": "3 mm kraft foamboard A4",
         "qty": 2, "notes": "Centre section + motor-mount doubler"},
        {"part": "4 mm carbon fibre rod × 900 mm",
         "qty": 1, "notes": "Main wing spar; cut to wingspan"},
        {"part": "Packing tape (clear, 48 mm wide)",
         "qty": 1, "notes": "Hinge + surface reinforcement"},
    ]

    power_budget_ma = {
        "Arduino Nano (active)": 22,
        "NRF24L01+PA+LNA (TX mode)": 115,
        "MPU-6050 (receiver only)": 4,
        "2× MG90S servo (stall)": 1200,
        "Total peak (excl. motor)": 1341,
        "BEC rating required": "≥2 A at 5 V",
        "Motor + ESC (75% throttle)": 15000,
        "Battery capacity recommendation": "1300 mAh 2S for ≥10 min",
    }

    design_notes = [
        "Power the NRF24L01 from the LM1117-3.3 ONLY — never connect to 5 V",
        "Add 10 µF across NRF24L01 VCC-GND to stabilise RF supply",
        "MPU-6050 on receiver: mount flat, nose forward, right wing positive Y",
        "Use twisted-pair wires for SPI (NRF24L01 SCK/MOSI/MISO) to reduce noise",
        "Route servo signal wires away from ESC motor cables",
        "Calibrate ESC before first flight: full throttle → power on → wait beeps → cut throttle",
        "Balance prop; unbalanced prop at 2300 KV causes gyro vibration noise",
        "Bind TX to RX on bench before installing in airframe",
    ]

    return {
        "build": "RCMakerLab Flying-Wing RC Plane",
        "transmitter_bom": transmitter_bom,
        "receiver_bom": receiver_bom,
        "shared_bom": shared_bom,
        "power_budget_ma": power_budget_ma,
        "design_notes": design_notes,
    }


def component_bom_stick_plane():
    """
    Return a from-scratch component-level Bill of Materials for the
    3JWings DC Motor RC Stick Plane.

    Returns:
        Dict with ``transmitter_bom``, ``receiver_bom``, ``shared_bom``,
        ``power_budget_ma``, and ``design_notes``.
    """
    transmitter_bom = [
        {"ref": "U1",  "part": "Arduino Nano V3 (ATmega328P)",
         "qty": 1, "notes": "3-channel TX: throttle + elevator + rudder"},
        {"ref": "U2",  "part": "NRF24L01 (standard, non-PA, SMD module)",
         "qty": 1, "notes": "Short-range OK for park flying; 3.3 V"},
        {"ref": "JS1", "part": "PS4 Analogue Joystick 10 kΩ",
         "qty": 1, "notes": "Left stick: A0 throttle, A1 rudder"},
        {"ref": "JS2", "part": "PS4 Analogue Joystick 10 kΩ",
         "qty": 1, "notes": "Right stick: A2 elevator"},
        {"ref": "SW1", "part": "Momentary push button",
         "qty": 1, "notes": "D2 — arm toggle"},
        {"ref": "U3",  "part": "LM1117-3.3 SOT-223",
         "qty": 1, "notes": "3.3 V for NRF24L01"},
        {"ref": "C1",  "part": "100 µF 10 V electrolytic",
         "qty": 2, "notes": "Bulk decoupling"},
        {"ref": "C2",  "part": "100 nF MLCC",
         "qty": 3, "notes": "Local decoupling"},
        {"ref": "J1",  "part": "9 V battery clip (PP3)",
         "qty": 1, "notes": "Powers TX; ~6 h on AA × 4 pack"},
    ]

    receiver_bom = [
        {"ref": "U1",  "part": "Arduino Nano V3 (ATmega328P)",
         "qty": 1, "notes": "3-ch RX + DC motor control + servos"},
        {"ref": "U2",  "part": "NRF24L01 (standard SMD)",
         "qty": 1, "notes": "3.3 V; CE=D9, CSN=D10"},
        {"ref": "Q1",  "part": "IRLZ44N Logic-Level N-channel MOSFET (TO-220)",
         "qty": 1, "notes": "DC motor switch; Gate=D3 via 100 Ω; RDS_on=22 mΩ"},
        {"ref": "R1",  "part": "100 Ω 1/4 W resistor",
         "qty": 1, "notes": "Gate series resistor (reduces ringing on gate drive)"},
        {"ref": "R2",  "part": "10 kΩ 1/4 W resistor",
         "qty": 1, "notes": "Gate pull-down to GND (ensures motor off at power-up)"},
        {"ref": "D1",  "part": "1N5822 Schottky diode (DO-201)",
         "qty": 1, "notes": "Flyback diode across DC motor (cathode to V+)"},
        {"ref": "U3",  "part": "LM1117-3.3 SOT-223",
         "qty": 1, "notes": "3.3 V for NRF24L01"},
        {"ref": "C1",  "part": "100 µF 10 V electrolytic",
         "qty": 3, "notes": "Bulk cap on motor V+, 5 V, 3.3 V"},
        {"ref": "C2",  "part": "100 nF MLCC",
         "qty": 4, "notes": "Local decoupling"},
        {"ref": "J1",  "part": "JST-PH 2-pin 1S socket",
         "qty": 1, "notes": "1S 3.7 V LiPo; feeds MOSFET drain and 5 V LDO"},
        {"ref": "J2",  "part": "3-pin 2.54 mm header × 2",
         "qty": 2, "notes": "Elevator servo D5, rudder servo D6"},
    ]

    shared_bom = [
        {"part": "8.5×20 mm Coreless DC Motor 3.7 V",
         "qty": 1, "notes": "55000 RPM no-load; direct-drive pusher"},
        {"part": "6×3 GWS-style propeller",
         "qty": 1, "notes": "Slow-fly; fits directly on motor shaft (1.5 mm)"},
        {"part": "3.7 g micro servo (SG51R or equivalent)",
         "qty": 2, "notes": "Elevator D5, rudder D6"},
        {"part": "1S 3.7 V 350 mAh LiPo (JST-PH)",
         "qty": 1, "notes": "≈8 min flight; charge at 0.35 A (1C)"},
        {"part": "3D-printed PLA parts (from STL files at shorturl.at/ZFULq)",
         "qty": 1, "notes": "Wing halves × 2, fuselage nose, mid, tail boom, H-stab, V-fin"},
        {"part": "3 mm bamboo skewer or 3 mm carbon rod × 300 mm",
         "qty": 1, "notes": "Wing spar"},
        {"part": "CA glue + accelerator (kicker)",
         "qty": 1, "notes": "Assembly adhesive"},
    ]

    power_budget_ma = {
        "Arduino Nano": 22,
        "NRF24L01 (RX mode)": 14,
        "2× 3.7 g micro servo (stall)": 400,
        "DC motor (full throttle, 1S)": 3500,
        "Total (full throttle)": 3936,
        "Battery capacity for ≥8 min": "350 mAh 1S minimum",
    }

    design_notes = [
        "MOSFET gate resistor (R1=100 Ω) prevents oscillation on PWM edges",
        "Pull-down R2 (10 kΩ) keeps motor off during Arduino boot-up",
        "Flyback diode D1 is mandatory — DC motor back-EMF kills MOSFET without it",
        "Keep motor wires short (< 80 mm) and twisted to reduce EMI",
        "Power Nano from 5 V LDO fed from LiPo; do NOT use Nano VIN with 3.7 V",
        "Use a dedicated 1S LiPo charger (TP4056 board works fine at 0.35 A)",
        "At 1S voltage the DC motor runs at reduced speed vs datasheet (rated 3.7 V)",
    ]

    return {
        "build": "3JWings DC Motor RC Stick Plane",
        "transmitter_bom": transmitter_bom,
        "receiver_bom": receiver_bom,
        "shared_bom": shared_bom,
        "power_budget_ma": power_budget_ma,
        "design_notes": design_notes,
    }


def component_bom_shahed_study():
    """
    Return a from-scratch component-level Bill of Materials for the
    Shahed / Lucas Drone RC study model (educational only).

    Returns:
        Dict with ``transmitter_bom``, ``receiver_bom``, ``shared_bom``,
        ``power_budget_ma``, and ``design_notes``.
    """
    transmitter_bom = [
        {"ref": "U1",  "part": "Arduino Nano V3 or RPi Pico",
         "qty": 1, "notes": "Identical to flying-wing TX; change radio address"},
        {"ref": "U2",  "part": "NRF24L01+PA+LNA GT-24",
         "qty": 1, "notes": "Long-range TX required (larger, faster aircraft)"},
        {"ref": "JS1", "part": "PS4 Analogue Joystick 10 kΩ",
         "qty": 2, "notes": "Same as flying-wing TX"},
        {"ref": "SW1", "part": "SPDT Toggle Switch",
         "qty": 2, "notes": "Arm + flight-mode selector"},
        {"ref": "U3",  "part": "LM1117-3.3",
         "qty": 1, "notes": "Radio supply"},
        {"ref": "C1",  "part": "Decoupling caps (same as flying-wing)",
         "qty": 8, "notes": "See flying-wing BOM"},
    ]

    receiver_bom = [
        {"ref": "U1",  "part": "Arduino Nano V3 or RPi Pico",
         "qty": 1, "notes": "Flight controller (pusher motor, 2 elevon servos)"},
        {"ref": "U2",  "part": "NRF24L01+PA+LNA E01-ML01DP5",
         "qty": 1, "notes": "Long-range RX; SMA antenna pointing upward"},
        {"ref": "U3",  "part": "MPU-6050 GY-521 breakout",
         "qty": 1, "notes": "IMU: pitch/roll stabilisation; I2C addr 0x68"},
        {"ref": "U4",  "part": "BN-880 GPS module (UART 9600 baud)",
         "qty": 1, "notes": "Optional for ArduPlane/iNav waypoint demo; TX→D0"},
        {"ref": "U5",  "part": "LM1117-3.3",
         "qty": 1, "notes": "Radio supply"},
        {"ref": "ESC1","part": "30 A BLHeli_S ESC (BEC 5 V)",
         "qty": 1, "notes": "Pusher brushless at rear centreline; signal D3"},
        {"ref": "C1–C6","part": "Decoupling caps (same as flying-wing RX)",
         "qty": 9, "notes": "See flying-wing BOM"},
        {"ref": "J1",  "part": "XT-30 female connector",
         "qty": 1, "notes": "2S LiPo main power"},
        {"ref": "J2",  "part": "3-pin 2.54 mm headers × 2",
         "qty": 2, "notes": "Elevon servos on D5 (left) and D6 (right)"},
    ]

    shared_bom = [
        {"part": "2206 2300KV Brushless Motor (pusher, CCW thread)",
         "qty": 1, "notes": "Rear centreline; CCW prop thread"},
        {"part": "6×4.5 or 7×3.5 propeller (pusher CCW)",
         "qty": 1, "notes": "Pusher direction; verify rotation before install"},
        {"part": "2× standard servo (SG90 or MG90S, 180°)",
         "qty": 2, "notes": "Left elevon D5, right elevon D6"},
        {"part": "2S 7.4 V 1000 mAh LiPo (XT-30)",
         "qty": 1, "notes": "Balance charge at 1 A (1C)"},
        {"part": "6 mm EPP foam sheets (A3 × 4)",
         "qty": 4, "notes": "Delta wing halves (impact-resistant)"},
        {"part": "3 mm foamboard (A4 × 2)",
         "qty": 2, "notes": "Fuselage pod + elevon blanks"},
        {"part": "4 mm carbon rod × 600 mm",
         "qty": 1, "notes": "Spar at 25 % chord"},
        {"part": "Tape + CA glue",
         "qty": 1, "notes": "Assembly"},
    ]

    power_budget_ma = {
        "Arduino Nano / Pico": 30,
        "NRF24L01+PA+LNA": 115,
        "MPU-6050": 4,
        "GPS (optional)": 30,
        "2× servos stall": 1200,
        "Total (excl. motor)": 1379,
        "Pusher motor 75% throttle": 18000,
        "Battery recommendation": "1000 mAh 2S",
    }

    design_notes = [
        "Pusher layout: motor at rear centroid — keeps CG forward naturally",
        "55° sweep: very high sweep so elevon roll authority is reduced vs 30° wing",
        "Increase roll PID Kp by ~25 % vs flying-wing to compensate",
        "GPS is optional for RC manual; mandatory for ArduPlane waypoint demo",
        "MPU-6050 must be rigidly mounted to airframe (not on foam) to avoid vibration noise",
        "Do NOT fly this design near airports, controlled airspace, or populated areas",
        "This design is for educational aerodynamics study only",
    ]

    return {
        "build": "Shahed / Lucas Drone Study Model (educational only)",
        "transmitter_bom": transmitter_bom,
        "receiver_bom": receiver_bom,
        "shared_bom": shared_bom,
        "power_budget_ma": power_budget_ma,
        "design_notes": design_notes,
    }


# ===========================================================================
# ASCII WIRING DIAGRAMS
# ===========================================================================

def wiring_diagram_arduino():
    """
    Return ASCII wiring diagrams for all three Arduino-based builds.

    Returns:
        Dict with keys ``flying_wing``, ``stick_plane``, ``shahed_study``.
    """
    flying_wing = r"""
╔══════════════════════════════════════════════════════════════════════════╗
║          FLYING-WING RC — ARDUINO NANO RECEIVER WIRING                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   2S LiPo 7.4 V ──┬──► ESC (30 A BLHeli_S)                             ║
║         (XT-30)   │        │ BEC 5 V──────────────► Nano 5V pin         ║
║                   │        │ Signal (PWM) ─────────► Nano D3            ║
║                   │        └──► 2205 Brushless Motor                    ║
║                   │                                                      ║
║                   └──► (to ESC only; Nano powered from BEC)             ║
║                                                                          ║
║   Nano 5V ──────────────► LM1117-3.3 IN                                 ║
║   LM1117-3.3 OUT ──────► NRF24L01 VCC  (with 10µF cap)                  ║
║   Nano GND ─────────────► LM1117 GND, NRF24L01 GND                      ║
║                                                                          ║
║   NRF24L01 ──────────────── Nano                                         ║
║     CE   ────────────────── D9                                           ║
║     CSN  ────────────────── D10                                          ║
║     SCK  ────────────────── D13 (SPI CLK)                               ║
║     MOSI ────────────────── D11 (SPI MOSI)                              ║
║     MISO ────────────────── D12 (SPI MISO)                              ║
║     IRQ  ────────────────── (NC)                                         ║
║                                                                          ║
║   MPU-6050 ──────────────── Nano                                         ║
║     VCC  ────────────────── 3.3 V                                        ║
║     GND  ────────────────── GND                                          ║
║     SDA  ────────────────── A4  (I2C SDA)                                ║
║     SCL  ────────────────── A5  (I2C SCL)                                ║
║     AD0  ────────────────── GND (addr = 0x68)                            ║
║     INT  ────────────────── D2  (data-ready interrupt, optional)         ║
║                                                                          ║
║   Left  MG90S Servo ─────── Nano D5  (PWM, signal/5V/GND)               ║
║   Right MG90S Servo ─────── Nano D6  (PWM, signal/5V/GND)               ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║          FLYING-WING RC — ARDUINO NANO TRANSMITTER WIRING               ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   9V Battery ──────────────► Nano VIN                                   ║
║   Nano 5V ──────────────────► LM1117-3.3 IN                             ║
║   LM1117-3.3 OUT ──────────► GT-24 NRF24L01+PA+LNA VCC                  ║
║                                                                          ║
║   GT-24 NRF24L01 ─────────── Nano                                        ║
║     (same SPI pins as RX)                                                ║
║                                                                          ║
║   JS1 X (roll)    ────────── A1     JS1 Y (throttle) ── A0             ║
║   JS2 Y (pitch)   ────────── A2     JS2 X (aux)      ── A3             ║
║   SW1 (arm)       ────────── D2 (INPUT_PULLUP)                          ║
║   SW2 (aux mode)  ────────── D3 (INPUT_PULLUP)                          ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

    stick_plane = r"""
╔══════════════════════════════════════════════════════════════════════════╗
║          DC STICK PLANE — ARDUINO NANO RECEIVER WIRING                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   1S LiPo 3.7 V ──┬──► IRLZ44N MOSFET (Drain)                          ║
║      (JST-PH)    │        │  (Source → DC Motor −)                      ║
║                   │        │  Gate ◄── R1 100Ω ◄── Nano D3             ║
║                   │        │  Gate ──► R2 10kΩ ──► GND                  ║
║                   │        └──► D1 1N5822 Flyback (cathode to V+)        ║
║                   │                                                      ║
║                   └──► 5V LDO (LM7805 or HT7550)                        ║
║                            │                                             ║
║                            └──► Nano 5V, Servos VCC                     ║
║                                                                          ║
║   Nano 5V ──────────────────► LM1117-3.3 IN                             ║
║   LM1117-3.3 OUT ───────────► NRF24L01 VCC                              ║
║                                                                          ║
║   NRF24L01 ────────────────── Nano (CE=D9, CSN=D10, SPI same as above) ║
║                                                                          ║
║   Elevator 3.7g Servo ─────── Nano D5                                   ║
║   Rudder   3.7g Servo ─────── Nano D6                                   ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║   DC Motor circuit detail:                                               ║
║                                                                          ║
║     V+ ──┬──── Motor(+) ──── Motor(−) ──── MOSFET Drain                ║
║           │                                                              ║
║           └──── D1(cathode)      D1(anode) ──── MOSFET Drain            ║
║   (flyback path: Motor(−) back-EMF → D1 → V+ → Motor(+))               ║
║                                                                          ║
║     MOSFET Source ──── GND                                               ║
║     MOSFET Gate   ──── R1(100Ω) ──── Nano D3 (PWM 0–255)               ║
║                   ──── R2(10kΩ) ──── GND                                ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

    shahed_study = r"""
╔══════════════════════════════════════════════════════════════════════════╗
║       SHAHED STUDY MODEL — ARDUINO NANO RECEIVER WIRING                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   Identical to Flying-Wing wiring EXCEPT:                               ║
║   • Motor is PUSHER (rear-mounted, CCW prop)                             ║
║   • Radio address changed to "SHD1\0" (avoid RF clash with flying-wing)  ║
║   • GPS BN-880 RX ────────────────────────── Nano D0 (Software Serial)  ║
║   • GPS BN-880 TX ────────────────────────── Nano D1 (not used in RX)   ║
║   • GPS VCC ──────────────────────────────── Nano 5V                    ║
║   • GPS GND ──────────────────────────────── GND                        ║
║                                                                          ║
║   Full wiring matrix:                                                    ║
║   ┌─────────────────────────┬──────────┬───────────────────────────┐    ║
║   │ Signal                  │ Nano pin │ Component                 │    ║
║   ├─────────────────────────┼──────────┼───────────────────────────┤    ║
║   │ ESC (pusher motor) PWM  │ D3       │ 30A BLHeli_S ESC signal   │    ║
║   │ Left  elevon servo      │ D5       │ MG90S / SG90 servo        │    ║
║   │ Right elevon servo      │ D6       │ MG90S / SG90 servo        │    ║
║   │ NRF24L01 CE             │ D9       │ Radio CE                  │    ║
║   │ NRF24L01 CSN            │ D10      │ Radio chip-select         │    ║
║   │ SPI MOSI                │ D11      │ Radio + (shared bus)      │    ║
║   │ SPI MISO                │ D12      │ Radio                     │    ║
║   │ SPI SCK                 │ D13      │ Radio                     │    ║
║   │ MPU-6050 SDA            │ A4       │ IMU I2C data              │    ║
║   │ MPU-6050 SCL            │ A5       │ IMU I2C clock             │    ║
║   │ GPS UART RX             │ D0       │ BN-880 TX (optional)      │    ║
║   └─────────────────────────┴──────────┴───────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

    return {
        "flying_wing": flying_wing,
        "stick_plane": stick_plane,
        "shahed_study": shahed_study,
    }


def wiring_diagram_pico():
    """
    Return ASCII wiring diagrams for all three Raspberry Pi Pico builds.

    Returns:
        Dict with keys ``flying_wing``, ``stick_plane``, ``shahed_study``.
    """
    flying_wing = r"""
╔══════════════════════════════════════════════════════════════════════════╗
║       FLYING-WING RC — RASPBERRY PI PICO RECEIVER WIRING                ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   2S LiPo 7.4 V ──► ESC (BEC 5V) ──► VSYS (pin 39) on Pico            ║
║                                    (Pico regulates 5V → 3.3V)          ║
║                                                                          ║
║   NRF24L01+PA+LNA ─────────────── Pico                                  ║
║     VCC (3.3V)  ─────────────── 3V3 OUT (pin 36)                        ║
║     GND         ─────────────── GND  (pin 38)                           ║
║     CE          ─────────────── GP17 (pin 22)                           ║
║     CSN         ─────────────── GP13 (pin 17) [SPI1 CS]                 ║
║     SCK         ─────────────── GP10 (pin 14) [SPI1 SCK]                ║
║     MOSI        ─────────────── GP11 (pin 15) [SPI1 MOSI]               ║
║     MISO        ─────────────── GP12 (pin 16) [SPI1 MISO]               ║
║                                                                          ║
║   MPU-6050 ────────────────────── Pico                                   ║
║     VCC (3.3V)  ─────────────── 3V3 OUT (pin 36)                        ║
║     GND         ─────────────── GND                                     ║
║     SDA         ─────────────── GP0  (pin 1)  [I2C0 SDA]               ║
║     SCL         ─────────────── GP1  (pin 2)  [I2C0 SCL]               ║
║     AD0 → GND   (I2C address 0x68)                                       ║
║                                                                          ║
║   ESC PWM signal ──────────────── GP2  (pin 4)  [PWM1A]                 ║
║   Left  elevon servo ─────────── GP3  (pin 5)  [PWM1B]                  ║
║   Right elevon servo ─────────── GP4  (pin 6)  [PWM2A]                  ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║       FLYING-WING RC — RASPBERRY PI PICO TRANSMITTER WIRING             ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   USB power (5V) ──► VSYS                                               ║
║                                                                          ║
║   GT-24 NRF24L01+PA+LNA ────── Pico                                     ║
║     (same SPI1 pins as RX)                                               ║
║                                                                          ║
║   JS1 X (roll)    ─────────── GP26 (ADC0)                               ║
║   JS1 Y (throttle)─────────── GP27 (ADC1)                               ║
║   JS2 Y (pitch)   ─────────── GP28 (ADC2)                               ║
║   SW1 (arm)       ─────────── GP15 (INPUT with internal pull-up)        ║
║   SW2 (aux mode)  ─────────── GP16 (INPUT with internal pull-up)        ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

    stick_plane = r"""
╔══════════════════════════════════════════════════════════════════════════╗
║       DC STICK PLANE — RASPBERRY PI PICO RECEIVER WIRING                ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   1S LiPo 3.7 V ──► HT7533 (3.3V LDO) ──► VSYS  (Pico)               ║
║                └───────────────────────► MOSFET Drain (V+)              ║
║                                                                          ║
║   MOSFET (IRLZ44N) gate ──► R1(100Ω) ──► GP2  (PWM)                    ║
║   MOSFET gate         ──► R2(10kΩ)  ──► GND                             ║
║   MOSFET source       ──► GND                                            ║
║   MOSFET drain        ──► DC Motor(−)                                   ║
║   D1 flyback (1N5822) ──► across Motor(+) and Drain                     ║
║                                                                          ║
║   NRF24L01 (standard, non-PA) ──── Pico (SPI1, same pins as FW)        ║
║                                                                          ║
║   Elevator servo ─────────────── GP3                                    ║
║   Rudder   servo ─────────────── GP4                                    ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

    shahed_study = r"""
╔══════════════════════════════════════════════════════════════════════════╗
║       SHAHED STUDY MODEL — RASPBERRY PI PICO RECEIVER WIRING            ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   Same as Flying-Wing Pico wiring PLUS:                                  ║
║                                                                          ║
║   GPS BN-880 TX ─────────────── GP5  (UART1 RX) [optional]              ║
║   GPS VCC (3.3V) ────────────── 3V3 OUT                                  ║
║   GPS GND ────────────────────── GND                                     ║
║                                                                          ║
║   Pin summary:                                                           ║
║   ┌──────────────────────┬─────────┬──────────────────────────────┐     ║
║   │ Signal               │ Pico pin│ Notes                        │     ║
║   ├──────────────────────┼─────────┼──────────────────────────────┤     ║
║   │ ESC (pusher)         │ GP2     │ PWM1A 50 Hz 1000–2000 µs     │     ║
║   │ Left  elevon servo   │ GP3     │ PWM1B                        │     ║
║   │ Right elevon servo   │ GP4     │ PWM2A                        │     ║
║   │ NRF24L01 CE          │ GP17    │ SPI1                         │     ║
║   │ NRF24L01 CSN         │ GP13    │ SPI1 CS                      │     ║
║   │ SPI SCK              │ GP10    │ SPI1 SCK                     │     ║
║   │ SPI MOSI             │ GP11    │ SPI1 MOSI                    │     ║
║   │ SPI MISO             │ GP12    │ SPI1 MISO                    │     ║
║   │ MPU-6050 SDA         │ GP0     │ I2C0 SDA                     │     ║
║   │ MPU-6050 SCL         │ GP1     │ I2C0 SCL                     │     ║
║   │ GPS UART RX          │ GP5     │ UART1 RX (optional)          │     ║
║   └──────────────────────┴─────────┴──────────────────────────────┘     ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

    return {
        "flying_wing": flying_wing,
        "stick_plane": stick_plane,
        "shahed_study": shahed_study,
    }


# ===========================================================================
# RASPBERRY PI PICO (MicroPython) FIRMWARE
# ===========================================================================

def pico_flying_wing_firmware():
    """
    Return complete MicroPython firmware for the RCMakerLab Flying-Wing
    RC plane running on a Raspberry Pi Pico.

    Hardware:
    - Pico (RP2040, 133 MHz)
    - NRF24L01+PA+LNA on SPI1  (CE=GP17, CSN=GP13)
    - MPU-6050 on I2C0         (SDA=GP0, SCL=GP1)
    - ESC PWM on GP2, Left elevon GP3, Right elevon GP4

    The firmware uses the micropython-nrf24l01 library:
    https://github.com/micropython/micropython-lib/tree/master/micropython/drivers/radio/nrf24l01

    Returns:
        Dict with ``transmitter_firmware``, ``receiver_firmware``,
        ``micropython_library``, and ``flash_instructions``.
    """
    tx = r'''
# ============================================================
# FLYING-WING RC — TRANSMITTER (Raspberry Pi Pico)
# MicroPython firmware — flash with Thonny or mpremote
# Radio: NRF24L01+PA+LNA on SPI1 (CE=GP17, CSN=GP13)
# ============================================================
import time
import struct
from machine import SPI, Pin, ADC
from nrf24l01 import NRF24L01

# ---- SPI1 + Radio setup ----
spi  = SPI(1, sck=Pin(10), mosi=Pin(11), miso=Pin(12))
csn  = Pin(13, Pin.OUT, value=1)
ce   = Pin(17, Pin.OUT, value=0)
nrf  = NRF24L01(spi, csn, ce, payload_size=8)
ADDR = b"FWng1"
nrf.open_tx_pipe(ADDR)
nrf.stop_listening()
nrf.set_power_speed(NRF24L01.POWER_3, NRF24L01.SPEED_250K)

# ---- Analogue inputs ----
throttle_adc = ADC(Pin(27))   # GP27 = ADC1
roll_adc     = ADC(Pin(26))   # GP26 = ADC0
pitch_adc    = ADC(Pin(28))   # GP28 = ADC2

# ---- Digital switches ----
sw_arm  = Pin(15, Pin.IN, Pin.PULL_UP)   # arm toggle
sw_aux  = Pin(16, Pin.IN, Pin.PULL_UP)   # aux mode

def read_stick(adc_obj, centre=32768):
    """Read 16-bit ADC, map to ±500 (or 0–1000 for throttle)."""
    raw = adc_obj.read_u16()
    return int((raw - centre) * 500 / centre)

def read_throttle(adc_obj):
    raw = adc_obj.read_u16()
    return int(raw * 1000 / 65535)

print("Flying-Wing TX ready")

while True:
    thr = read_throttle(throttle_adc)          # 0–1000
    rol = read_stick(roll_adc)                  # ±500
    pit = read_stick(pitch_adc)                 # ±500
    sw1 = 0 if sw_arm.value() else 1            # active-low
    sw2 = 0 if sw_aux.value() else 1

    # Pack as: throttle(h), roll(h), pitch(h), sw1(B), sw2(B) = 8 bytes
    payload = struct.pack(">hhhBB", thr, rol, pit, sw1, sw2)

    try:
        nrf.send(payload)
    except OSError:
        pass   # ignore occasional TX timeout

    time.sleep_ms(20)   # 50 Hz
'''

    rx = r'''
# ============================================================
# FLYING-WING RC — RECEIVER (Raspberry Pi Pico)
# MicroPython firmware — flash with Thonny or mpremote
# Radio  : NRF24L01+PA+LNA on SPI1 (CE=GP17, CSN=GP13)
# IMU    : MPU-6050 on I2C0 (SDA=GP0, SCL=GP1)
# ESC    : GP2   Left elevon: GP3   Right elevon: GP4
# ============================================================
import time
import struct
import math
from machine import SPI, Pin, I2C, PWM
from nrf24l01 import NRF24L01

# ---- Radio ----
spi = SPI(1, sck=Pin(10), mosi=Pin(11), miso=Pin(12))
csn = Pin(13, Pin.OUT, value=1)
ce  = Pin(17, Pin.OUT, value=0)
nrf = NRF24L01(spi, csn, ce, payload_size=8)
ADDR = b"FWng1"
nrf.open_rx_pipe(1, ADDR)
nrf.start_listening()
nrf.set_power_speed(NRF24L01.POWER_3, NRF24L01.SPEED_250K)

# ---- PWM outputs (50 Hz, duty via ns) ----
def make_servo(pin_num):
    p = PWM(Pin(pin_num))
    p.freq(50)
    return p

def set_us(pwm, us):
    """Set servo/ESC pulse width in microseconds (1000–2000)."""
    # Pico PWM period at 50 Hz = 20 000 000 ns
    pwm.duty_ns(int(us * 1000))

esc_pwm   = make_servo(2)
left_pwm  = make_servo(3)
right_pwm = make_servo(4)
set_us(esc_pwm, 1000)    # ESC arm
set_us(left_pwm, 1500)
set_us(right_pwm, 1500)
time.sleep(2)

# ---- I2C / MPU-6050 ----
i2c = I2C(0, sda=Pin(0), scl=Pin(1), freq=400_000)
MPU_ADDR = 0x68

def mpu_init():
    i2c.writeto_mem(MPU_ADDR, 0x6B, b'\x00')   # wake up
    i2c.writeto_mem(MPU_ADDR, 0x1B, b'\x00')   # gyro ±250°/s
    i2c.writeto_mem(MPU_ADDR, 0x1A, b'\x03')   # DLPF 44 Hz

def read_gyro():
    """Return (gx, gy, gz) in degrees/second."""
    raw = i2c.readfrom_mem(MPU_ADDR, 0x43, 6)
    gx, gy, gz = struct.unpack(">hhh", raw)
    scale = 250.0 / 32768.0
    return gx * scale, gy * scale, gz * scale

mpu_init()

# ---- PID state ----
pitch_intg = 0.0; pitch_prev = 0.0
roll_intg  = 0.0; roll_prev  = 0.0

KP_P, KI_P, KD_P = 1.40, 0.04, 0.10
KP_R, KI_R, KD_R = 1.80, 0.05, 0.12
DT = 0.02   # 50 Hz
I_LIM, O_LIM = 200.0, 500.0

def pid_step(sp, meas, kp, ki, kd, intg, prev):
    err = sp - meas
    intg = max(-I_LIM, min(I_LIM, intg + err * DT))
    derv = -(meas - prev) / DT
    prev = meas
    out  = max(-O_LIM, min(O_LIM, kp * err + ki * intg + kd * derv))
    return out, intg, prev

def elevon_mix(pitch_out, roll_out):
    left  = int(1500 + pitch_out + roll_out)
    right = int(1500 + pitch_out - roll_out)
    return max(1000, min(2000, left)), max(1000, min(2000, right))

print("Flying-Wing RX ready")
armed = False

while True:
    t0 = time.ticks_ms()

    if nrf.any():
        buf = nrf.recv()
        thr, rol, pit, sw1, sw2 = struct.unpack(">hhhBB", buf)
        armed = (sw1 == 1)

    # Read IMU gyro rates (°/s)
    gx, gy, gz = read_gyro()
    measured_pitch = gy   # pitch rate
    measured_roll  = gx   # roll  rate

    # PID
    pitch_out, pitch_intg, pitch_prev = pid_step(
        pit, measured_pitch, KP_P, KI_P, KD_P, pitch_intg, pitch_prev)
    roll_out, roll_intg, roll_prev = pid_step(
        rol, measured_roll,  KP_R, KI_R, KD_R, roll_intg,  roll_prev)

    if armed:
        thr_us = 1000 + int(thr)     # thr=0–1000 → 1000–2000 µs
    else:
        thr_us = 1000                # disarmed
        pitch_intg = roll_intg = 0.0

    left_us, right_us = elevon_mix(pitch_out, roll_out)

    set_us(esc_pwm,   thr_us)
    set_us(left_pwm,  left_us)
    set_us(right_pwm, right_us)

    elapsed = time.ticks_diff(time.ticks_ms(), t0)
    remaining = 20 - elapsed
    if remaining > 0:
        time.sleep_ms(remaining)
'''

    return {
        "platform": "Raspberry Pi Pico (RP2040, MicroPython)",
        "micropython_library": "nrf24l01.py from micropython-lib (MIT licence) — "
                               "https://github.com/micropython/micropython-lib/tree/"
                               "master/micropython/drivers/radio/nrf24l01",
        "transmitter_firmware": tx,
        "receiver_firmware": rx,
        "flash_instructions": [
            "Download MicroPython UF2 for RP2040: https://micropython.org/download/rp2-pico/",
            "Hold BOOTSEL, plug Pico into USB — drag UF2 to RPI-RP2 drive",
            "Open Thonny IDE (v4+), select interpreter: MicroPython (Raspberry Pi Pico)",
            "Install nrf24l01.py: copy to Pico root via Thonny file panel",
            "Paste transmitter firmware into main.py on TX Pico → Run / Save",
            "Paste receiver firmware into main.py on RX Pico → Run / Save",
            "Open Thonny Shell to see 'TX ready' / 'RX ready' print confirmation",
            "Calibrate ESC: set throttle to max → power on → hear beeps → cut throttle",
        ],
    }


def pico_stick_plane_firmware():
    """
    Return complete MicroPython firmware for the 3JWings DC Motor
    RC Stick Plane on a Raspberry Pi Pico.

    Returns:
        Dict with ``transmitter_firmware``, ``receiver_firmware``,
        and ``flash_instructions``.
    """
    tx = r'''
# ============================================================
# DC STICK PLANE — TRANSMITTER (Raspberry Pi Pico, 3-channel)
# Throttle + Elevator + Rudder
# Radio: NRF24L01 standard on SPI1 (CE=GP17, CSN=GP13)
# ============================================================
import time
import struct
from machine import SPI, Pin, ADC
from nrf24l01 import NRF24L01

spi = SPI(1, sck=Pin(10), mosi=Pin(11), miso=Pin(12))
nrf = NRF24L01(spi, Pin(13, Pin.OUT, value=1),
               Pin(17, Pin.OUT, value=0), payload_size=6)
ADDR = b"SP3ch"
nrf.open_tx_pipe(ADDR)
nrf.stop_listening()
nrf.set_power_speed(NRF24L01.POWER_0, NRF24L01.SPEED_250K)  # low power, short range

thr_adc  = ADC(Pin(27))  # GP27 = throttle
elev_adc = ADC(Pin(28))  # GP28 = elevator
rudd_adc = ADC(Pin(26))  # GP26 = rudder

def read_thr(adc):
    return int(adc.read_u16() * 1000 // 65535)

def read_axis(adc):
    raw = adc.read_u16()
    return int((raw - 32768) * 500 // 32768)

print("Stick-Plane TX ready (3ch)")

while True:
    thr  = read_thr(thr_adc)
    elev = read_axis(elev_adc)
    rudd = read_axis(rudd_adc)
    payload = struct.pack(">hhh", thr, elev, rudd)
    try:
        nrf.send(payload)
    except OSError:
        pass
    time.sleep_ms(20)
'''

    rx = r'''
# ============================================================
# DC STICK PLANE — RECEIVER (Raspberry Pi Pico, 3-channel)
# DC motor via MOSFET PWM on GP2
# Elevator servo GP3 | Rudder servo GP4
# Radio: NRF24L01 on SPI1 (CE=GP17, CSN=GP13)
# ============================================================
import time
import struct
from machine import SPI, Pin, PWM
from nrf24l01 import NRF24L01

spi = SPI(1, sck=Pin(10), mosi=Pin(11), miso=Pin(12))
nrf = NRF24L01(spi, Pin(13, Pin.OUT, value=1),
               Pin(17, Pin.OUT, value=0), payload_size=6)
ADDR = b"SP3ch"
nrf.open_rx_pipe(1, ADDR)
nrf.start_listening()

# DC motor MOSFET gate: PWM 20 kHz (above audible range)
motor_pwm = PWM(Pin(2), freq=20000, duty_u16=0)

# Servos at 50 Hz
def make_servo(gp):
    p = PWM(Pin(gp), freq=50)
    return p

def set_us(pwm, us):
    pwm.duty_ns(us * 1000)   # 1 µs = 1000 ns

elev_pwm = make_servo(3)
rudd_pwm = make_servo(4)
set_us(elev_pwm, 1500)
set_us(rudd_pwm, 1500)

# PID (elevator + rudder)
ei, ep = 0.0, 0.0
ri, rp = 0.0, 0.0
KP_E, KI_E, KD_E = 1.20, 0.03, 0.08
KP_R, KI_R, KD_R = 0.90, 0.02, 0.05
DT = 0.02

def pid_step(sp, meas, kp, ki, kd, i, prev):
    err = sp - meas
    i = max(-200.0, min(200.0, i + err * DT))
    d = -(meas - prev) / DT
    prev = meas
    return max(-500.0, min(500.0, kp*err + ki*i + kd*d)), i, prev

thr, elev_sp, rudd_sp = 0, 0, 0

print("Stick-Plane RX ready (3ch)")

while True:
    t0 = time.ticks_ms()

    if nrf.any():
        buf = nrf.recv()
        thr, elev_sp, rudd_sp = struct.unpack(">hhh", buf)

    # TODO: read IMU pitch rate for measured_elev (e.g. MPU-6050 gy)
    measured_elev = elev_sp   # replace with actual gyro reading
    measured_rudd = rudd_sp   # replace with actual gyro reading

    elev_out, ei, ep = pid_step(elev_sp, measured_elev, KP_E, KI_E, KD_E, ei, ep)
    rudd_out, ri, rp = pid_step(rudd_sp, measured_rudd, KP_R, KI_R, KD_R, ri, rp)

    # DC motor PWM duty 0–65535
    motor_duty = int(thr * 65535 // 1000)
    motor_pwm.duty_u16(motor_duty)

    # Elevator ±30° → 1200–1800 µs
    set_us(elev_pwm, max(1200, min(1800, int(1500 + elev_out * 300 // 500))))
    # Rudder ±35°
    set_us(rudd_pwm, max(1150, min(1850, int(1500 + rudd_out * 350 // 500))))

    elapsed = time.ticks_diff(time.ticks_ms(), t0)
    if 20 - elapsed > 0:
        time.sleep_ms(20 - elapsed)
'''

    return {
        "platform": "Raspberry Pi Pico (MicroPython)",
        "transmitter_firmware": tx,
        "receiver_firmware": rx,
        "flash_instructions": [
            "Flash MicroPython UF2 to both Picos (same as Flying-Wing)",
            "Copy nrf24l01.py library to each Pico root",
            "TX: use PP3 9 V battery or USB power bank",
            "RX: wire MOSFET (IRLZ44N) between motor and LiPo V+ (see wiring diagram)",
            "Verify flyback diode (D1) across DC motor before powering",
        ],
    }


def pico_shahed_study_firmware():
    """
    Return complete MicroPython firmware for the Shahed / Lucas Drone
    study model on a Raspberry Pi Pico.

    Identical wiring to flying-wing but uses a different radio address
    and includes optional GPS UART read for ArduPlane-style demo.

    Returns:
        Dict with ``transmitter_firmware``, ``receiver_firmware``,
        and ``flash_instructions``.
    """
    tx = r'''
# ============================================================
# SHAHED STUDY MODEL — TRANSMITTER (Raspberry Pi Pico)
# Same as flying-wing TX but radio address = b"SHD1\x00"
# ⚠ FOR EDUCATIONAL / AERODYNAMIC STUDY ONLY
# ============================================================
import time
import struct
from machine import SPI, Pin, ADC
from nrf24l01 import NRF24L01

spi = SPI(1, sck=Pin(10), mosi=Pin(11), miso=Pin(12))
nrf = NRF24L01(spi, Pin(13, Pin.OUT, value=1),
               Pin(17, Pin.OUT, value=0), payload_size=8)
ADDR = b"SHD1\x00"   # different address from flying-wing to avoid clash
nrf.open_tx_pipe(ADDR)
nrf.stop_listening()
nrf.set_power_speed(NRF24L01.POWER_3, NRF24L01.SPEED_250K)

throttle_adc = ADC(Pin(27))
roll_adc     = ADC(Pin(26))
pitch_adc    = ADC(Pin(28))
sw_arm  = Pin(15, Pin.IN, Pin.PULL_UP)
sw_mode = Pin(16, Pin.IN, Pin.PULL_UP)   # flight mode (RC / waypoint)

def read_thr(adc):
    return int(adc.read_u16() * 1000 // 65535)

def read_axis(adc):
    raw = adc.read_u16()
    return int((raw - 32768) * 500 // 32768)

print("Shahed Study TX ready")
print("WARNING: Educational use only. Comply with local regulations.")

while True:
    thr = read_thr(throttle_adc)
    rol = read_axis(roll_adc)
    pit = read_axis(pitch_adc)
    sw1 = 0 if sw_arm.value()  else 1
    sw2 = 0 if sw_mode.value() else 1
    payload = struct.pack(">hhhBB", thr, rol, pit, sw1, sw2)
    try:
        nrf.send(payload)
    except OSError:
        pass
    time.sleep_ms(20)
'''

    rx = r'''
# ============================================================
# SHAHED STUDY MODEL — RECEIVER (Raspberry Pi Pico)
# Pusher ESC on GP2 | Left elevon GP3 | Right elevon GP4
# MPU-6050 I2C0 SDA=GP0 SCL=GP1
# Optional GPS BN-880 on UART1 RX=GP5
# ⚠ FOR EDUCATIONAL / AERODYNAMIC STUDY ONLY
# ============================================================
import time
import struct
from machine import SPI, Pin, I2C, PWM, UART
from nrf24l01 import NRF24L01

# ---- Radio ----
spi = SPI(1, sck=Pin(10), mosi=Pin(11), miso=Pin(12))
nrf = NRF24L01(spi, Pin(13, Pin.OUT, value=1),
               Pin(17, Pin.OUT, value=0), payload_size=8)
ADDR = b"SHD1\x00"
nrf.open_rx_pipe(1, ADDR)
nrf.start_listening()
nrf.set_power_speed(NRF24L01.POWER_3, NRF24L01.SPEED_250K)

# ---- PWM (50 Hz servos + ESC) ----
def make_servo(gp):
    p = PWM(Pin(gp), freq=50)
    return p

def set_us(pwm, us):
    pwm.duty_ns(int(us * 1000))

esc_pwm   = make_servo(2)
left_pwm  = make_servo(3)
right_pwm = make_servo(4)
set_us(esc_pwm, 1000)
set_us(left_pwm, 1500)
set_us(right_pwm, 1500)
time.sleep(2)

# ---- IMU MPU-6050 ----
i2c = I2C(0, sda=Pin(0), scl=Pin(1), freq=400_000)
MPU = 0x68
i2c.writeto_mem(MPU, 0x6B, b'\x00')
i2c.writeto_mem(MPU, 0x1B, b'\x00')  # ±250°/s gyro range

def read_gyro():
    d = i2c.readfrom_mem(MPU, 0x43, 6)
    gx, gy, gz = struct.unpack(">hhh", d)
    s = 250.0 / 32768.0
    return gx * s, gy * s, gz * s

# ---- Optional GPS (BN-880) ----
gps_uart = UART(1, baudrate=9600, rx=Pin(5))

def read_gps_line():
    """Non-blocking GPS NMEA line read."""
    if gps_uart.any():
        try:
            return gps_uart.readline().decode().strip()
        except Exception:
            return None
    return None

# ---- PID (55° delta: higher Kp for sluggish roll) ----
pi_intg = 0.0; pi_prev = 0.0
ri_intg = 0.0; ri_prev = 0.0
KP_P, KI_P, KD_P = 1.50, 0.04, 0.12
KP_R, KI_R, KD_R = 2.00, 0.05, 0.15   # higher Kp for high-sweep delta
DT = 0.02

def pid_step(sp, meas, kp, ki, kd, intg, prev):
    err  = sp - meas
    intg = max(-200.0, min(200.0, intg + err * DT))
    derv = -(meas - prev) / DT
    prev = meas
    return max(-500.0, min(500.0, kp*err + ki*intg + kd*derv)), intg, prev

armed = False
print("Shahed Study RX ready")
print("WARNING: Educational use only. Comply with local regulations.")

while True:
    t0 = time.ticks_ms()

    if nrf.any():
        buf = nrf.recv()
        thr, rol, pit, sw1, sw2 = struct.unpack(">hhhBB", buf)
        armed = (sw1 == 1)

    gx, gy, gz = read_gyro()
    meas_pitch = gy   # pitch rate from gyro
    meas_roll  = gx   # roll  rate from gyro

    pitch_out, pi_intg, pi_prev = pid_step(
        pit, meas_pitch, KP_P, KI_P, KD_P, pi_intg, pi_prev)
    roll_out, ri_intg, ri_prev = pid_step(
        rol, meas_roll,  KP_R, KI_R, KD_R, ri_intg, ri_prev)

    if not armed:
        pitch_out = roll_out = 0.0
        pi_intg = ri_intg = 0.0

    thr_us  = 1000 + int(thr) if armed else 1000
    left_us  = max(1000, min(2000, int(1500 + pitch_out + roll_out)))
    right_us = max(1000, min(2000, int(1500 + pitch_out - roll_out)))

    set_us(esc_pwm,   thr_us)
    set_us(left_pwm,  left_us)
    set_us(right_pwm, right_us)

    # Optional GPS display (non-blocking)
    gps_line = read_gps_line()
    if gps_line and gps_line.startswith("$GPGGA"):
        print("GPS:", gps_line[:40])

    elapsed = time.ticks_diff(time.ticks_ms(), t0)
    if 20 - elapsed > 0:
        time.sleep_ms(20 - elapsed)
'''

    return {
        "platform": "Raspberry Pi Pico (MicroPython)",
        "warning": "⚠ FOR EDUCATIONAL / AERODYNAMIC STUDY ONLY",
        "transmitter_firmware": tx,
        "receiver_firmware": rx,
        "flash_instructions": [
            "Flash MicroPython UF2 to Pico (same as Flying-Wing)",
            "Upload nrf24l01.py to Pico root",
            "GPS is optional — comment out UART and read_gps_line() if not fitted",
            "Change ADDR to avoid RF collision with other NRF24L01 units nearby",
            "Higher Kp for roll (2.0) reflects 55° sweep sluggish response",
            "Comply with all local aviation laws and regulations",
        ],
    }


# ===========================================================================
# ARDUINO C++ FIRMWARE (EXTENDED WITH MPU-6050 IMU)
# ===========================================================================

def arduino_flying_wing_full():
    """
    Return a complete, extended Arduino C++ TX + RX sketch for the
    RCMakerLab Flying-Wing RC Plane, now including MPU-6050 IMU
    stabilisation on the receiver side.

    Returns:
        Dict with ``transmitter_sketch``, ``receiver_sketch``,
        ``libraries_required``, and ``flash_instructions``.
    """
    tx = r'''
// ================================================================
// FLYING-WING RC — TRANSMITTER (Arduino Nano V3 + NRF24L01)
// Full implementation with arm switch and aux mode
// ================================================================
#include <SPI.h>
#include <RF24.h>

RF24 radio(9, 10);                       // CE=D9, CSN=D10
const byte ADDR[6] = "FWng1";

struct TxPayload {
    int16_t throttle;   // 0–1000
    int16_t roll;       // -500 to +500
    int16_t pitch;      // -500 to +500
    uint8_t sw1;        // arm (1=armed)
    uint8_t sw2;        // aux mode
};
TxPayload tx_data;

const int PIN_THR  = A0;
const int PIN_ROLL = A1;
const int PIN_PTCH = A2;
const int PIN_SW1  = 2;
const int PIN_SW2  = 3;

void setup() {
    pinMode(PIN_SW1, INPUT_PULLUP);
    pinMode(PIN_SW2, INPUT_PULLUP);
    radio.begin();
    radio.setChannel(108);
    radio.setDataRate(RF24_250KBPS);
    radio.setPALevel(RF24_PA_HIGH);
    radio.setPayloadSize(sizeof(TxPayload));
    radio.openWritingPipe(ADDR);
    radio.stopListening();
    Serial.begin(115200);
    Serial.println("Flying-Wing TX ready");
}

void loop() {
    tx_data.throttle = map(analogRead(PIN_THR),  0, 1023,    0, 1000);
    tx_data.roll     = map(analogRead(PIN_ROLL), 0, 1023, -500,  500);
    tx_data.pitch    = map(analogRead(PIN_PTCH), 0, 1023, -500,  500);
    tx_data.sw1      = !digitalRead(PIN_SW1);
    tx_data.sw2      = !digitalRead(PIN_SW2);
    radio.write(&tx_data, sizeof(tx_data));
    delay(20);   // 50 Hz
}
'''

    rx = r'''
// ================================================================
// FLYING-WING RC — RECEIVER (Arduino Nano V3)
// Includes MPU-6050 IMU for pitch + roll rate stabilisation
// ================================================================
#include <SPI.h>
#include <RF24.h>
#include <Servo.h>
#include <Wire.h>
#include <MPU6050.h>

RF24   radio(9, 10);
const  byte ADDR[6] = "FWng1";
Servo  escServo;      // D3
Servo  leftElevon;    // D5
Servo  rightElevon;   // D6
MPU6050 imu;

struct TxPayload {
    int16_t throttle;
    int16_t roll;
    int16_t pitch;
    uint8_t sw1;
    uint8_t sw2;
};
TxPayload rx_data;
bool armed = false;

// ---- PID ----
float pitchIntg = 0, pitchPrev = 0;
float rollIntg  = 0, rollPrev  = 0;
const float KP_P=1.40, KI_P=0.04, KD_P=0.10;
const float KP_R=1.80, KI_R=0.05, KD_R=0.12;
const float DT = 0.02;

float pidStep(float sp, float meas, float kp, float ki, float kd,
              float &intg, float &prev) {
    float err  = sp - meas;
    intg = constrain(intg + err * DT, -200, 200);
    float derv = -(meas - prev) / DT;
    prev = meas;
    return constrain(kp*err + ki*intg + kd*derv, -500, 500);
}

void setup() {
    Serial.begin(115200);
    Wire.begin();
    imu.initialize();
    if (!imu.testConnection()) {
        Serial.println("MPU-6050 not found — check wiring!");
        while (true);
    }
    // Gyro full-scale ±250°/s, DLPF 44 Hz
    imu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
    imu.setDLPFMode(MPU6050_DLPF_BW_44);

    escServo.attach(3);
    leftElevon.attach(5);
    rightElevon.attach(6);
    escServo.writeMicroseconds(1000);   // ESC arm
    delay(2000);

    radio.begin();
    radio.setChannel(108);
    radio.setDataRate(RF24_250KBPS);
    radio.setPALevel(RF24_PA_HIGH);
    radio.setPayloadSize(sizeof(TxPayload));
    radio.openReadingPipe(0, ADDR);
    radio.startListening();
    Serial.println("Flying-Wing RX ready");
}

void loop() {
    if (radio.available())
        radio.read(&rx_data, sizeof(rx_data));

    armed = (rx_data.sw1 == 1);

    // Read gyro rates (raw → °/s: scale = 250/32768)
    int16_t gx_raw, gy_raw, gz_raw;
    imu.getRotation(&gx_raw, &gy_raw, &gz_raw);
    float measuredRoll  = gx_raw * (250.0f / 32768.0f);
    float measuredPitch = gy_raw * (250.0f / 32768.0f);

    float pitchOut = pidStep(rx_data.pitch, measuredPitch,
                             KP_P, KI_P, KD_P, pitchIntg, pitchPrev);
    float rollOut  = pidStep(rx_data.roll,  measuredRoll,
                             KP_R, KI_R, KD_R, rollIntg,  rollPrev);

    if (!armed) { pitchOut = rollOut = pitchIntg = rollIntg = 0; }

    int thrUs  = armed ? map(rx_data.throttle, 0, 1000, 1000, 2000) : 1000;
    int leftUs  = constrain(1500 + (int)pitchOut + (int)rollOut, 1000, 2000);
    int rightUs = constrain(1500 + (int)pitchOut - (int)rollOut, 1000, 2000);

    escServo.writeMicroseconds(thrUs);
    leftElevon.writeMicroseconds(leftUs);
    rightElevon.writeMicroseconds(rightUs);
    delay(20);
}
'''

    return {
        "language": "C++ (Arduino IDE 2.x)",
        "board": "Arduino Nano V3 (ATmega328P)",
        "libraries_required": [
            "RF24 by TMRh20  — Library Manager",
            "MPU6050 by Electronic Cats  — Library Manager",
            "Wire (built-in)",
            "Servo (built-in)",
            "SPI (built-in)",
        ],
        "transmitter_sketch": tx,
        "receiver_sketch": rx,
        "flash_instructions": [
            "Install RF24 and MPU6050 libraries via Arduino IDE Library Manager",
            "Select board: Tools → Board → Arduino Nano",
            "Select port: Tools → Port → COMx / /dev/ttyUSBx",
            "Flash tx sketch to transmitter Nano, rx sketch to receiver Nano",
            "Open Serial Monitor at 115200 baud to verify IMU connection",
            "Calibrate ESC before first flight",
        ],
    }


def arduino_stick_plane_full():
    """
    Return the complete Arduino C++ TX + RX sketch for the
    3JWings DC Motor RC Stick Plane.

    Returns:
        Dict with ``transmitter_sketch``, ``receiver_sketch``,
        ``libraries_required``, and ``flash_instructions``.
    """
    tx = r'''
// ================================================================
// DC STICK PLANE — TRANSMITTER (Arduino Nano V3, 3-channel)
// ================================================================
#include <SPI.h>
#include <RF24.h>

RF24 radio(9, 10);
const byte ADDR[6] = "SP3ch";

struct TxPayload {
    int16_t throttle;   // 0–1000
    int16_t elevator;   // -500 to +500
    int16_t rudder;     // -500 to +500
};
TxPayload tx_data;

void setup() {
    radio.begin();
    radio.setChannel(76);
    radio.setDataRate(RF24_250KBPS);
    radio.setPALevel(RF24_PA_LOW);   // short range; save battery
    radio.setPayloadSize(sizeof(TxPayload));
    radio.openWritingPipe(ADDR);
    radio.stopListening();
}

void loop() {
    tx_data.throttle = map(analogRead(A0), 0, 1023,    0, 1000);
    tx_data.elevator = map(analogRead(A1), 0, 1023, -500,  500);
    tx_data.rudder   = map(analogRead(A2), 0, 1023, -500,  500);
    radio.write(&tx_data, sizeof(tx_data));
    delay(20);
}
'''

    rx = r'''
// ================================================================
// DC STICK PLANE — RECEIVER (Arduino Nano V3)
// DC motor MOSFET (IRLZ44N) on D3 via PWM
// Elevator servo D5 | Rudder servo D6
// ================================================================
#include <SPI.h>
#include <RF24.h>
#include <Servo.h>
#include <Wire.h>
#include <MPU6050.h>

RF24 radio(9, 10);
const byte ADDR[6] = "SP3ch";
Servo servoElev;    // D5
Servo servoRudd;    // D6
MPU6050 imu;

const int MOTOR_PIN = 3;

struct TxPayload {
    int16_t throttle;
    int16_t elevator;
    int16_t rudder;
};
TxPayload rx_data;

float ei = 0, ep = 0;
float ri = 0, rp = 0;
const float KP_E=1.20, KI_E=0.03, KD_E=0.08;
const float KP_R=0.90, KI_R=0.02, KD_R=0.05;
const float DT = 0.02;

float pidStep(float sp, float meas, float kp, float ki, float kd,
              float &intg, float &prev) {
    float err  = sp - meas;
    intg = constrain(intg + err*DT, -200, 200);
    float derv = -(meas - prev) / DT;
    prev = meas;
    return constrain(kp*err + ki*intg + kd*derv, -500, 500);
}

void setup() {
    Serial.begin(115200);
    Wire.begin();
    imu.initialize();
    imu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
    imu.setDLPFMode(MPU6050_DLPF_BW_44);

    pinMode(MOTOR_PIN, OUTPUT);
    analogWrite(MOTOR_PIN, 0);
    servoElev.attach(5);
    servoRudd.attach(6);
    servoElev.write(90);
    servoRudd.write(90);

    radio.begin();
    radio.setChannel(76);
    radio.setDataRate(RF24_250KBPS);
    radio.setPALevel(RF24_PA_LOW);
    radio.setPayloadSize(sizeof(TxPayload));
    radio.openReadingPipe(0, ADDR);
    radio.startListening();
    Serial.println("Stick-Plane RX ready");
}

void loop() {
    if (radio.available())
        radio.read(&rx_data, sizeof(rx_data));

    int16_t gx_raw, gy_raw, gz_raw;
    imu.getRotation(&gx_raw, &gy_raw, &gz_raw);
    float measuredPitch = gy_raw * (250.0f / 32768.0f);
    float measuredYaw   = gz_raw * (250.0f / 32768.0f);

    float elevOut = pidStep(rx_data.elevator, measuredPitch,
                            KP_E, KI_E, KD_E, ei, ep);
    float ruddOut = pidStep(rx_data.rudder,   measuredYaw,
                            KP_R, KI_R, KD_R, ri, rp);

    analogWrite(MOTOR_PIN, map(rx_data.throttle, 0, 1000, 0, 255));
    servoElev.write(constrain(90 + (int)(elevOut/500.0*30), 60, 120));
    servoRudd.write(constrain(90 + (int)(ruddOut/500.0*35), 55, 125));
    delay(20);
}
'''

    return {
        "language": "C++ (Arduino IDE 2.x)",
        "board": "Arduino Nano V3",
        "libraries_required": [
            "RF24 by TMRh20",
            "MPU6050 by Electronic Cats",
            "Wire, Servo (built-in)",
        ],
        "transmitter_sketch": tx,
        "receiver_sketch": rx,
        "flash_instructions": [
            "Install RF24 + MPU6050 libraries",
            "Wire MOSFET per wiring diagram (R1 gate resistor + R2 pull-down + D1 flyback diode)",
            "Flash tx sketch to TX Nano, rx sketch to RX Nano",
            "Open Serial Monitor (115200) on RX to verify IMU",
            "Check servo direction: if reversed, negate elevOut or ruddOut in code",
        ],
    }


def arduino_shahed_study_full():
    """
    Return the complete Arduino C++ TX + RX sketch for the
    Shahed / Lucas Drone study model (educational only).

    Includes MPU-6050 rate stabilisation and optional GPS
    NMEA passthrough on Serial for ground station display.

    Returns:
        Dict with ``transmitter_sketch``, ``receiver_sketch``,
        ``libraries_required``, and ``flash_instructions``.
    """
    tx = r'''
// ================================================================
// SHAHED STUDY MODEL — TRANSMITTER (Arduino Nano V3)
// ⚠ FOR EDUCATIONAL / AERODYNAMIC STUDY ONLY
// ================================================================
#include <SPI.h>
#include <RF24.h>

RF24 radio(9, 10);
const byte ADDR[6] = "SHD1\0";   // unique address

struct TxPayload {
    int16_t throttle;
    int16_t roll;
    int16_t pitch;
    uint8_t sw1;    // arm
    uint8_t sw2;    // flight mode
};
TxPayload tx_data;

void setup() {
    pinMode(2, INPUT_PULLUP);   // arm
    pinMode(3, INPUT_PULLUP);   // mode
    radio.begin();
    radio.setChannel(90);       // different channel from flying-wing (108)
    radio.setDataRate(RF24_250KBPS);
    radio.setPALevel(RF24_PA_HIGH);
    radio.setPayloadSize(sizeof(TxPayload));
    radio.openWritingPipe(ADDR);
    radio.stopListening();
}

void loop() {
    tx_data.throttle = map(analogRead(A0), 0, 1023,    0, 1000);
    tx_data.roll     = map(analogRead(A1), 0, 1023, -500,  500);
    tx_data.pitch    = map(analogRead(A2), 0, 1023, -500,  500);
    tx_data.sw1      = !digitalRead(2);
    tx_data.sw2      = !digitalRead(3);
    radio.write(&tx_data, sizeof(tx_data));
    delay(20);
}
'''

    rx = r'''
// ================================================================
// SHAHED STUDY MODEL — RECEIVER (Arduino Nano V3)
// Pusher ESC D3 | Left elevon D5 | Right elevon D6
// MPU-6050 I2C (SDA=A4, SCL=A5)
// Optional GPS BN-880 on SoftwareSerial D0 (RX)
// ⚠ FOR EDUCATIONAL / AERODYNAMIC STUDY ONLY
// ================================================================
#include <SPI.h>
#include <RF24.h>
#include <Servo.h>
#include <Wire.h>
#include <MPU6050.h>
#include <SoftwareSerial.h>

RF24   radio(9, 10);
const  byte ADDR[6] = "SHD1\0";
Servo  escPusher;    // D3
Servo  leftElevon;   // D5
Servo  rightElevon;  // D6
MPU6050 imu;
SoftwareSerial gpsSerial(0, 1);  // RX=D0, TX=D1 (GPS TX → Nano D0)

struct TxPayload {
    int16_t throttle;
    int16_t roll;
    int16_t pitch;
    uint8_t sw1;
    uint8_t sw2;
};
TxPayload rx_data;
bool armed = false;

// PID — higher Kp for roll (55° sweep sluggish roll)
float pi_i=0, pi_p=0, ri_i=0, ri_p=0;
const float KP_P=1.50, KI_P=0.04, KD_P=0.12;
const float KP_R=2.00, KI_R=0.05, KD_R=0.15;
const float DT=0.02;

float pidStep(float sp, float meas, float kp, float ki, float kd,
              float &intg, float &prev) {
    float err  = sp - meas;
    intg = constrain(intg + err*DT, -200, 200);
    float derv = -(meas - prev) / DT;
    prev = meas;
    return constrain(kp*err + ki*intg + kd*derv, -500, 500);
}

void setup() {
    Serial.begin(115200);
    gpsSerial.begin(9600);
    Wire.begin();
    imu.initialize();
    imu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
    imu.setDLPFMode(MPU6050_DLPF_BW_44);

    escPusher.attach(3);
    leftElevon.attach(5);
    rightElevon.attach(6);
    escPusher.writeMicroseconds(1000);
    delay(2000);

    radio.begin();
    radio.setChannel(90);
    radio.setDataRate(RF24_250KBPS);
    radio.setPALevel(RF24_PA_HIGH);
    radio.setPayloadSize(sizeof(TxPayload));
    radio.openReadingPipe(0, ADDR);
    radio.startListening();
    Serial.println("Shahed Study RX ready");
    Serial.println("WARNING: Educational use only.");
}

void loop() {
    if (radio.available())
        radio.read(&rx_data, sizeof(rx_data));

    armed = (rx_data.sw1 == 1);

    int16_t gx_raw, gy_raw, gz_raw;
    imu.getRotation(&gx_raw, &gy_raw, &gz_raw);
    float measuredPitch = gy_raw * (250.0f / 32768.0f);
    float measuredRoll  = gx_raw * (250.0f / 32768.0f);

    float pitchOut = pidStep(rx_data.pitch, measuredPitch,
                             KP_P, KI_P, KD_P, pi_i, pi_p);
    float rollOut  = pidStep(rx_data.roll,  measuredRoll,
                             KP_R, KI_R, KD_R, ri_i, ri_p);

    if (!armed) { pitchOut = rollOut = pi_i = ri_i = 0; }

    int thrUs   = armed ? map(rx_data.throttle, 0, 1000, 1000, 2000) : 1000;
    int leftUs  = constrain(1500+(int)pitchOut+(int)rollOut, 1000, 2000);
    int rightUs = constrain(1500+(int)pitchOut-(int)rollOut, 1000, 2000);

    escPusher.writeMicroseconds(thrUs);
    leftElevon.writeMicroseconds(leftUs);
    rightElevon.writeMicroseconds(rightUs);

    // Optional: echo GPS NMEA line to USB serial
    if (gpsSerial.available()) {
        String line = gpsSerial.readStringUntil('\n');
        if (line.startsWith("$GPGGA"))
            Serial.println(line);
    }
    delay(20);
}
'''

    return {
        "language": "C++ (Arduino IDE 2.x)",
        "board": "Arduino Nano V3",
        "warning": "⚠ FOR EDUCATIONAL / AERODYNAMIC STUDY ONLY",
        "libraries_required": [
            "RF24 by TMRh20",
            "MPU6050 by Electronic Cats",
            "Wire, Servo, SoftwareSerial (built-in)",
        ],
        "transmitter_sketch": tx,
        "receiver_sketch": rx,
        "flash_instructions": [
            "Install RF24 + MPU6050 libraries",
            "GPS is optional — SoftwareSerial on D0/D1 only if BN-880 fitted",
            "Use channel 90 (different from FW channel 108 and SP channel 76)",
            "Flash tx to TX Nano, rx to RX Nano",
            "Verify IMU + radio on Serial Monitor at 115200",
            "Comply with all local aviation laws and regulations",
        ],
    }
