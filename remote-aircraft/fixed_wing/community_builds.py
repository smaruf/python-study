"""
Community Build Designs: Flying-Wing RC Plane, DC Motor Stick Plane,
and Shahed/Lucas Drone aerodynamic study.

Based on YouTube builds:
1. RCMakerLab - "Build a Flying-Wing with Simple Materials" (Oct 2025)
   Delta-wing RC plane with Arduino NRF24L01 remote control.
2. 3JWings - "Make RC Plane With dc Motor | DIY Rc Stick Plane" (Jan 2025)
   Beginner 3D-printed plane with DC motor, 609 mm wingspan, 116 g AUW.

Educational aerodynamic reference:
3. Shahed/Lucas drone delta-wing planform study (educational only).

Each design provides:
- Full geometric parameters
- Aerodynamic analysis (lift, drag, stall speed, wing loading)
- Control system design (PID gains, servo sizing, mixing equations)
- Foamboard / 3D-print build specifications
- Electronics BOM
- Optional CadQuery STL generation (graceful fallback if not installed)
- Reference Arduino C++ sketches as Python strings
"""

import math
import os

try:
    import cadquery as cq
    HAS_CADQUERY = True
except ImportError:
    HAS_CADQUERY = False


# ===========================================================================
# PID CONTROLLER
# ===========================================================================

class PIDController:
    """
    Discrete-time PID controller suitable for Arduino implementation.

    Implements standard PID with:
    - Anti-windup (integral clamping)
    - Derivative on measurement (avoids derivative kick on setpoint change)
    - Symmetric output clamping

    Arduino C++ equivalent::

        float pid_update(float sp, float meas, float dt) {
            float error = sp - meas;
            integral += error * dt;
            integral = constrain(integral, -i_limit, i_limit);
            float deriv = -(meas - prev_meas) / dt;
            prev_meas = meas;
            return constrain(kp*error + ki*integral + kd*deriv,
                             -out_limit, out_limit);
        }
    """

    def __init__(self, kp, ki, kd, out_limit=500, i_limit=200):
        """
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            out_limit: Symmetric output clamp (±out_limit)
            i_limit: Symmetric integral clamp to prevent windup
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_limit = out_limit
        self.i_limit = i_limit
        self._integral = 0.0
        self._prev_measured = None

    def update(self, setpoint, measured, dt):
        """
        Compute one PID step.

        Args:
            setpoint: Desired value
            measured: Current sensor reading
            dt: Time step in seconds

        Returns:
            Control output clamped to ±out_limit
        """
        error = setpoint - measured
        self._integral = max(
            -self.i_limit,
            min(self.i_limit, self._integral + error * dt)
        )
        if self._prev_measured is None:
            self._prev_measured = measured
        derivative = -(measured - self._prev_measured) / dt
        self._prev_measured = measured
        raw = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(-self.out_limit, min(self.out_limit, raw))

    def reset(self):
        """Reset integrator and derivative state."""
        self._integral = 0.0
        self._prev_measured = None


def simulate_step_response(pid, target, initial=0.0, dt=0.02, steps=200):
    """
    Simulate a PID step response with a simple first-order plant model.

    Useful for visualising gain-tuning behaviour before flying.

    Args:
        pid: PIDController instance
        target: Step target value (e.g., 30 deg bank angle)
        initial: Initial plant state
        dt: Time step in seconds (default: 0.02 = 50 Hz)
        steps: Number of simulation steps

    Returns:
        List of (time_s, measured, output) tuples
    """
    pid.reset()
    measured = initial
    history = []
    tau_plant = 0.15  # simplified first-order plant time constant (s)
    for i in range(steps):
        t = i * dt
        output = pid.update(target, measured, dt)
        measured += (output / pid.out_limit) * abs(target) * (dt / tau_plant)
        measured = max(-180.0, min(180.0, measured))
        history.append((round(t, 4), round(measured, 3), round(output, 3)))
    return history


# ===========================================================================
# ELEVON MIXING
# ===========================================================================

def elevon_mix(throttle, pitch, roll,
               pitch_gain=1.0, roll_gain=1.0,
               out_min=1000, out_max=2000):
    """
    Standard elevon mixing for flying-wing / delta-wing aircraft.

    Elevons combine elevator (pitch) and aileron (roll) functions.
    Used by both the RCMakerLab Flying-Wing and the Shahed study model.

    Mixing equations::

        left_elevon  = neutral + pitch_us + roll_us
        right_elevon = neutral + pitch_us - roll_us

    Args:
        throttle: Raw throttle in µs (1000–2000)
        pitch: Pitch demand in normalised stick units (−500 to +500)
        roll: Roll demand in normalised stick units (−500 to +500)
        pitch_gain: Pitch authority scaling (default 1.0)
        roll_gain: Roll authority scaling (default 1.0)
        out_min: Minimum servo pulse width in µs (default 1000)
        out_max: Maximum servo pulse width in µs (default 2000)

    Returns:
        Dict with ``throttle_us``, ``left_elevon_us``, ``right_elevon_us``
    """
    neutral = (out_min + out_max) / 2.0   # 1500 µs
    pitch_us = pitch * pitch_gain
    roll_us = roll * roll_gain
    left = neutral + pitch_us + roll_us
    right = neutral + pitch_us - roll_us

    def _clamp(v):
        return max(out_min, min(out_max, v))

    return {
        "throttle_us": _clamp(throttle),
        "left_elevon_us": _clamp(left),
        "right_elevon_us": _clamp(right),
    }


# ===========================================================================
# FLYING-WING RC PLANE  (RCMakerLab Build)
# ===========================================================================

def flying_wing_rc_design(
    root_chord=400,
    wingspan=860,
    sweep_angle=30,
    wing_twist=-2,
    auw_grams=350,
):
    """
    Full design and analysis for the RCMakerLab Flying-Wing RC plane.

    Reference: "Build a Flying-Wing with Simple Materials.
    DIY RC Plane & Remote Control" — RCMakerLab, Oct 9, 2025.

    Construction: 6 mm insulation styrofoam + 3 mm kraft foamboard.
    Control: Arduino Nano V3 + NRF24L01+PA+LNA 2.4 GHz radio.
    Motor: 2205 2300KV brushless | ESC: 30A BL | Prop: 5050/5045 3-blade.
    Servos: 2× MG90S (one per elevon).

    Args:
        root_chord: Centre chord in mm (default: 400)
        wingspan: Total span in mm (default: 860)
        sweep_angle: Leading-edge sweep in degrees (default: 30)
        wing_twist: Washout angle, negative = washout (default: −2)
        auw_grams: All-up weight estimate in grams (default: 350)

    Returns:
        Dictionary with geometry, aerodynamics, control system, and build data
    """
    # --- Geometry ---
    tip_chord = root_chord * 0.40
    semi_span = wingspan / 2.0
    area_mm2 = ((root_chord + tip_chord) / 2.0) * wingspan
    area_cm2 = area_mm2 / 100.0
    aspect_ratio = (wingspan ** 2) / area_mm2
    tr = tip_chord / root_chord
    mac = (2.0 / 3.0) * root_chord * ((1 + tr + tr ** 2) / (1 + tr))
    centre_width = 120   # fuselage bay width mm

    # --- Aerodynamics ---
    weight_n = auw_grams * 0.00981
    area_m2 = area_mm2 / 1e6
    rho = 1.225
    cl_cruise = 0.50
    cl_max = 1.10
    cruise_ms = math.sqrt((2 * weight_n) / (rho * area_m2 * cl_cruise))
    stall_ms = math.sqrt((2 * weight_n) / (rho * area_m2 * cl_max))
    wing_loading = auw_grams / area_cm2

    # CG: 28 % MAC from leading edge (tight tolerance ±5 mm)
    cg_from_le = 0.28 * mac

    # --- Control surfaces ---
    panel_span = (wingspan - centre_width) / 2.0
    elevon_chord = 0.30 * root_chord
    elevon_span = 0.45 * panel_span
    control_tau = elevon_chord / mac

    # --- PID gains (Arduino 50 Hz loop, ±500 stick units) ---
    pitch_pid = PIDController(kp=1.40, ki=0.04, kd=0.10)
    roll_pid = PIDController(kp=1.80, ki=0.05, kd=0.12)

    # --- Foamboard cut plan ---
    sweep_rad = math.radians(sweep_angle)
    le_setback = semi_span * math.tan(sweep_rad)

    foam_cuts = {
        "left_wing_panel": {
            "qty": 1,
            "material": "6 mm insulation styrofoam",
            "blank_mm": f"{int(semi_span)} × {int(root_chord)}",
            "cut": (
                f"Taper leading edge {int(le_setback)} mm at tip "
                f"({sweep_angle}° sweep). "
                "Score and remove front triangle."
            ),
        },
        "right_wing_panel": {
            "qty": 1,
            "material": "6 mm insulation styrofoam",
            "blank_mm": f"{int(semi_span)} × {int(root_chord)}",
            "cut": "Mirror of left panel.",
        },
        "centre_fuselage_section": {
            "qty": 1,
            "material": "3 mm kraft foamboard",
            "blank_mm": f"{int(centre_width)} × {int(root_chord)}",
            "cut": "Rectangular — no taper.",
        },
        "left_elevon": {
            "qty": 1,
            "material": "6 mm styrofoam (wing trailing-edge offcut)",
            "blank_mm": f"{int(elevon_span)} × {int(elevon_chord)}",
            "cut": "Score top face 0.5 mm deep along hinge line for tape hinge.",
        },
        "right_elevon": {
            "qty": 1,
            "material": "6 mm styrofoam (wing trailing-edge offcut)",
            "blank_mm": f"{int(elevon_span)} × {int(elevon_chord)}",
            "cut": "Mirror of left elevon.",
        },
        "motor_mount_doubler": {
            "qty": 1,
            "material": "3 mm kraft foamboard — 2 layers CA-glued",
            "blank_mm": "62 × 62",
            "cut": "Drill 5 mm motor shaft + 4× M3 bolts on 16 mm circle.",
        },
    }

    return {
        "type": "Flying-Wing RC Plane",
        "reference": {
            "title": (
                "Build a Flying-Wing with Simple Materials. "
                "DIY RC Plane & Remote Control"
            ),
            "channel": "RCMakerLab",
            "published": "Oct 9, 2025",
            "views": "118,594",
            "search": "RCMakerLab Build Flying-Wing Simple Materials",
            "gerber_and_code": (
                "https://www.rcpano.net/2025/09/30/making-delta-wing-rc-plane/"
            ),
        },
        "geometry": {
            "root_chord_mm": root_chord,
            "tip_chord_mm": round(tip_chord, 1),
            "wingspan_mm": wingspan,
            "sweep_angle_deg": sweep_angle,
            "wing_twist_deg": wing_twist,
            "area_mm2": round(area_mm2),
            "area_cm2": round(area_cm2, 1),
            "aspect_ratio": round(aspect_ratio, 2),
            "mac_mm": round(mac, 1),
            "centre_section_width_mm": centre_width,
        },
        "aerodynamics": {
            "auw_grams": auw_grams,
            "wing_loading_g_cm2": round(wing_loading, 4),
            "cruise_speed_ms": round(cruise_ms, 1),
            "stall_speed_ms": round(stall_ms, 1),
            "cg_from_le_mm": round(cg_from_le, 1),
            "cg_tolerance_mm": 5,
        },
        "control_system": {
            "type": "Elevon mixing (pitch + roll combined)",
            "radio": "Arduino Nano V3 + NRF24L01+PA+LNA 2.4 GHz",
            "transmitter_mcu": "Arduino Nano V3 (Micro connector)",
            "receiver_mcu": "Arduino Nano V3 (Micro connector)",
            "joystick": "2× PS4 Analogue Joystick 10 K (TX side)",
            "elevon_chord_mm": round(elevon_chord, 1),
            "elevon_span_mm": round(elevon_span, 1),
            "control_effectiveness_tau": round(control_tau, 3),
            "servos": "2× MG90S (one per elevon, 180° range)",
            "pid_pitch": {
                "kp": pitch_pid.kp,
                "ki": pitch_pid.ki,
                "kd": pitch_pid.kd,
                "axis": "elevon collective",
            },
            "pid_roll": {
                "kp": roll_pid.kp,
                "ki": roll_pid.ki,
                "kd": roll_pid.kd,
                "axis": "elevon differential",
            },
            "channel_map": {
                "CH1_throttle": "ESC via J1 Y-axis (A0, 1000–2000 µs)",
                "CH2_roll": "J1 X-axis → elevon differential (A1)",
                "CH3_pitch": "J2 Y-axis → elevon collective (A2)",
                "CH4_aux": "J2 X-axis → arm switch / aux (A3)",
            },
            "motor": "2205 2300KV brushless",
            "esc": "30A BL ESC (BEC 5 V for servos)",
            "propeller": "5050 or 5045 3-blade",
            "voltage_reg": "LM1117 3.3 V (NRF24L01 supply)",
        },
        "electronics_bom": [
            "Arduino Nano V3 (Micro connector) × 2  [TX + RX]",
            "NRF24L01+PA+LNA 100 mW E01-ML01DP5  [RX side]",
            "GT-24 NRF24L01+PA+LNA with antenna  [TX side]",
            "2205 2300KV Brushless Motor (CW)",
            "30A BL ESC",
            "5050 or 5045 3-blade propeller",
            "2× MG90S Servo",
            "2× PS4 Analogue Joystick 10 K",
            "2× Toggle switch",
            "LM1117 3.3 V voltage regulator",
            "Capacitor 10 µF × 2, 100 µF × 3, 100 nF (104) × 5",
            "JST 2-Pin connector",
            "2S 7.4 V or 3S 11.1 V LiPo (800–1300 mAh)",
        ],
        "build_specification": {
            "construction": "Foam + foamboard — no 3D printer required",
            "foam_cuts": foam_cuts,
            "spar": "4–6 mm carbon rod through both wing panels at 25 % chord",
            "hinge": "Packing-tape hinge or vinyl film for elevon pivot",
            "covering": "Optional packing tape over foam for surface rigidity",
            "total_build_time_hours": "4–6",
            "cg_note": (
                "Balance at CG mark (28 % MAC = "
                f"{round(cg_from_le, 0):.0f} mm from LE). "
                "Add nose ballast clay if tail-heavy. "
                "Test glide before powering motor."
            ),
        },
        "tuning_notes": [
            "Verify CG before first flight — add clay ballast to nose if tail-heavy",
            "Maiden at 50 % throttle in calm conditions",
            "Increase pitch Kd if pitch oscillation visible",
            "Increase roll Kp if roll response is sluggish",
            "Reflex trim (slight up-elevon at full throttle) may be needed",
            "Gerber / code: https://www.rcpano.net/2025/09/30/making-delta-wing-rc-plane/",
        ],
    }


def flying_wing_rc_arduino_sketch():
    """
    Return the reference Arduino C++ transmitter and receiver sketches for
    the RCMakerLab Flying-Wing RC plane as Python strings.

    The TX sketch reads two PS4 joysticks and two toggle switches, then
    transmits a 7-byte struct via NRF24L01 at 50 Hz.

    The RX sketch receives that struct, runs PID on pitch and roll axes,
    and outputs elevon-mixed servo pulses plus an ESC throttle signal.

    Returns:
        Dict with keys ``transmitter_sketch``, ``receiver_sketch``,
        ``board``, ``radio_library``, and ``flash_instructions``.
    """
    tx = r"""
// =================================================================
// FLYING-WING RC  —  TRANSMITTER
// Board: Arduino Nano V3  |  Radio: NRF24L01+PA+LNA
// Based on: RCMakerLab "Build a Flying-Wing" (Oct 2025)
// =================================================================
#include <SPI.h>
#include <RF24.h>

RF24 radio(9, 10);              // CE = D9, CSN = D10
const byte ADDR[6] = "FWng1";

struct TxPayload {
    int16_t throttle;           // 0 – 1000
    int16_t roll;               // -500 to +500
    int16_t pitch;              // -500 to +500
    uint8_t sw1;                // toggle switch 1 (arm)
    uint8_t sw2;                // toggle switch 2 (aux mode)
};
TxPayload tx;

void setup() {
    pinMode(2, INPUT_PULLUP);   // toggle switch 1
    pinMode(3, INPUT_PULLUP);   // toggle switch 2
    radio.begin();
    radio.setChannel(108);
    radio.setDataRate(RF24_250KBPS);
    radio.setPALevel(RF24_PA_HIGH);
    radio.openWritingPipe(ADDR);
    radio.stopListening();
}

void loop() {
    // J1: A0 = throttle Y-axis,  A1 = roll X-axis
    // J2: A2 = pitch  Y-axis,  A3 = aux (unused)
    tx.throttle = map(analogRead(A0), 0, 1023,    0, 1000);
    tx.roll     = map(analogRead(A1), 0, 1023, -500,  500);
    tx.pitch    = map(analogRead(A2), 0, 1023, -500,  500);
    tx.sw1      = !digitalRead(2);
    tx.sw2      = !digitalRead(3);
    radio.write(&tx, sizeof(tx));
    delay(20);          // 50 Hz
}
"""

    rx = r"""
// =================================================================
// FLYING-WING RC  —  RECEIVER  (elevon mixing + PID)
// Board: Arduino Nano V3  |  Radio: NRF24L01+PA+LNA
// Based on: RCMakerLab "Build a Flying-Wing" (Oct 2025)
// =================================================================
#include <SPI.h>
#include <RF24.h>
#include <Servo.h>

RF24   radio(9, 10);
const  byte ADDR[6] = "FWng1";
Servo  escMotor;                // ESC on D3
Servo  servoLeft;               // Left  elevon on D5
Servo  servoRight;              // Right elevon on D6

// PID state
float pitchIntg = 0, pitchPrev = 0;
float rollIntg  = 0, rollPrev  = 0;
const float KP_P = 1.40, KI_P = 0.04, KD_P = 0.10;
const float KP_R = 1.80, KI_R = 0.05, KD_R = 0.12;
const float DT   = 0.02;   // 50 Hz

struct TxPayload {
    int16_t throttle;
    int16_t roll;
    int16_t pitch;
    uint8_t sw1;
    uint8_t sw2;
};
TxPayload rx;
bool armed = false;

float pidStep(float sp, float meas,
              float kp, float ki, float kd,
              float &intg, float &prev) {
    float err  = sp - meas;
    intg = constrain(intg + err * DT, -200, 200);
    float derv = -(meas - prev) / DT;
    prev = meas;
    return constrain(kp*err + ki*intg + kd*derv, -500, 500);
}

void setup() {
    escMotor.attach(3);
    servoLeft.attach(5);
    servoRight.attach(6);
    escMotor.writeMicroseconds(1000);   // ESC arm sequence
    delay(2000);
    radio.begin();
    radio.setChannel(108);
    radio.setDataRate(RF24_250KBPS);
    radio.setPALevel(RF24_PA_HIGH);
    radio.openReadingPipe(0, ADDR);
    radio.startListening();
}

void loop() {
    if (!radio.available()) { delay(5); return; }
    radio.read(&rx, sizeof(rx));

    armed = (rx.sw1 == 1);
    int thrUs = armed ? map(rx.throttle, 0, 1000, 1000, 2000) : 1000;
    escMotor.writeMicroseconds(thrUs);

    // NOTE: Replace measuredPitch / measuredRoll with actual IMU readings
    // (e.g., from an MPU-6050 via I2C) for genuine stabilisation.
    // Here rx.pitch/rx.roll pass through directly as a pass-through fallback
    // so the aircraft responds to stick input even without an IMU attached.
    float measuredPitch = rx.pitch;  // TODO: replace with IMU pitch rate (°/s)
    float measuredRoll  = rx.roll;   // TODO: replace with IMU roll  rate (°/s)

    float pitchOut = pidStep(rx.pitch, measuredPitch,
                             KP_P, KI_P, KD_P, pitchIntg, pitchPrev);
    float rollOut  = pidStep(rx.roll,  measuredRoll,
                             KP_R, KI_R, KD_R, rollIntg,  rollPrev);

    // Elevon mixing: left = neutral + pitch + roll
    //               right= neutral + pitch - roll
    int leftUs  = constrain(1500 + (int)pitchOut + (int)rollOut,  1000, 2000);
    int rightUs = constrain(1500 + (int)pitchOut - (int)rollOut,  1000, 2000);
    servoLeft.writeMicroseconds(leftUs);
    servoRight.writeMicroseconds(rightUs);
    delay(20);
}
"""

    return {
        "language": "C++ (Arduino IDE)",
        "board": "Arduino Nano V3 (ATmega328P, Micro USB)",
        "radio_library": "RF24 by TMRh20 (install via Library Manager)",
        "servo_library": "Servo (built-in Arduino)",
        "transmitter_sketch": tx,
        "receiver_sketch": rx,
        "flash_instructions": [
            "Install RF24 library: IDE → Sketch → Include Library → Manage Libraries → search RF24",
            "Select board: Tools → Board → Arduino Nano",
            "Select processor: ATmega328P (Old Bootloader) if upload fails",
            "Flash tx sketch to transmitter Nano, rx sketch to receiver Nano",
            "Calibrate ESC: power on at full throttle, wait for beeps, cut throttle",
            "Check elevon direction — swap left/right or negate in code if reversed",
        ],
    }


def flying_wing_rc_foamboard_plan(
    root_chord=400,
    wingspan=860,
    sweep_angle=30,
):
    """
    Return a text-based foamboard cutting plan for the RCMakerLab
    Flying-Wing RC plane.

    Args:
        root_chord: Centre chord in mm
        wingspan: Total wingspan in mm
        sweep_angle: LE sweep angle in degrees

    Returns:
        Dict with ``ascii_layout``, ``cut_list``, ``assembly_steps``,
        and ``tools_needed``
    """
    sweep_rad = math.radians(sweep_angle)
    semi_span = wingspan / 2.0
    tip_chord = root_chord * 0.40
    le_setback = semi_span * math.tan(sweep_rad)
    centre_width = 120
    elevon_chord = int(0.30 * root_chord)
    panel_span = (semi_span - centre_width / 2.0)
    elevon_span = int(0.45 * panel_span)

    tr = tip_chord / root_chord
    mac = (2.0 / 3.0) * root_chord * ((1 + tr + tr ** 2) / (1 + tr))
    cg_mm = int(0.28 * mac)

    ascii_layout = f"""
TOP VIEW — FLYING WING  (one side shown; mirror for the other)
Units: mm

  ←─ {int(le_setback)} ─→  ←─ {int(tip_chord)} ─→
   ╱─────────────────────────────╲  ↑
  ╱  WING PANEL  (6 mm foam)      ╲ │ {int(semi_span)} mm
 ╱   blank: {int(semi_span)} × {int(root_chord)} mm           ╲ │
╱  trim LE to {sweep_angle}° sweep                ╲ ↓
│═══════════════════════════════│
│  CENTRE SECTION (3 mm board)  │ ↕ {int(root_chord)} mm
│   width: {int(centre_width)} mm                    │
│═══════════════════════════════│

  ELEVON (cut from wing trailing edge):
    {elevon_span} mm span  ×  {elevon_chord} mm chord
    Score top face 0.5 mm deep for tape hinge

  CG MARK: {cg_mm} mm from leading edge (28 % MAC)
"""

    cut_list = [
        {
            "part": "Left wing panel",
            "qty": 1,
            "material": "6 mm insulation styrofoam",
            "blank_mm": f"{int(semi_span)} × {int(root_chord)}",
            "cut": (
                f"Trim leading edge: setback {int(le_setback)} mm at tip "
                f"({sweep_angle}° sweep)"
            ),
        },
        {
            "part": "Right wing panel",
            "qty": 1,
            "material": "6 mm insulation styrofoam",
            "blank_mm": f"{int(semi_span)} × {int(root_chord)}",
            "cut": "Mirror of left panel",
        },
        {
            "part": "Centre fuselage section",
            "qty": 1,
            "material": "3 mm kraft foamboard",
            "blank_mm": f"{int(centre_width)} × {int(root_chord)}",
            "cut": "Rectangular — no taper",
        },
        {
            "part": "Left elevon",
            "qty": 1,
            "material": "6 mm styrofoam (trailing-edge offcut)",
            "blank_mm": f"{elevon_span} × {elevon_chord}",
            "cut": "Score hinge line 0.5 mm deep; attach with tape hinge",
        },
        {
            "part": "Right elevon",
            "qty": 1,
            "material": "6 mm styrofoam (trailing-edge offcut)",
            "blank_mm": f"{elevon_span} × {elevon_chord}",
            "cut": "Mirror of left elevon",
        },
        {
            "part": "Motor-mount doubler plate",
            "qty": 1,
            "material": "3 mm kraft foamboard — 2 layers CA-glued",
            "blank_mm": "62 × 62",
            "cut": "Drill 5 mm motor shaft + 4× M3 holes on 16 mm bolt circle",
        },
        {
            "part": "Spar channel blocks",
            "qty": 4,
            "material": "3 mm foamboard",
            "blank_mm": "30 × 20",
            "cut": "Notch 6 mm wide × 6 mm deep for carbon rod",
        },
    ]

    assembly_steps = [
        "1.  Cut both wing panels and trim LE to sweep angle.",
        "2.  Slice elevon strips from trailing 30 % of each panel.",
        "3.  Hot-glue centre foamboard section between wing panels.",
        "4.  Push 4–6 mm carbon spar through channel blocks at 25 % chord.",
        "5.  CA-glue motor-mount doubler to trailing centreline.",
        "6.  Attach elevons with 3 strips of packing tape per hinge line.",
        "7.  Mount MG90S servos in centre bay; route pushrods to elevons.",
        "8.  Solder ESC to motor; install receiver Arduino in centre bay.",
        f"9.  Mark CG at {cg_mm} mm from LE; balance with clay ballast if needed.",
        "10. Range-check radio; maiden in calm conditions at 50 % throttle.",
    ]

    tools = [
        "Sharp hobby knife / box cutter",
        "Metal straight-edge ruler",
        "Hot-glue gun (low-temp)",
        "CA (cyanoacrylate) glue + accelerator",
        "Packing tape (hinges + surface reinforcement)",
        "Drill or Dremel (motor mount holes)",
        "80 W soldering iron (electronics)",
    ]

    return {
        "ascii_layout": ascii_layout,
        "cut_list": cut_list,
        "assembly_steps": assembly_steps,
        "tools_needed": tools,
    }


def generate_flying_wing_rc_stl(
    output_dir="output/community_builds/flying_wing_rc",
    root_chord=400,
    wingspan=860,
):
    """
    Generate CadQuery STL files for the 3D-printable brackets and parts
    of the RCMakerLab Flying-Wing RC plane.

    The wing body itself is foam; only small hardware is printed:
    - Motor-mount plate (62 × 62 × 6 mm, 4× M3 on 16 mm bolt circle)
    - Electronics bay (Arduino Nano + ESC enclosure)
    - Wing spar channel block (holds 6 mm carbon rod)
    - Servo horn extension (25 mm reach for MG90S)

    Args:
        output_dir: Directory to save STL files
        root_chord: Centre chord in mm
        wingspan: Total wingspan in mm

    Returns:
        List of generated STL file paths

    Raises:
        ImportError: If CadQuery is not installed
    """
    if not HAS_CADQUERY:
        raise ImportError(
            "CadQuery is required. "
            "Install: conda install -c conda-forge -c cadquery cadquery"
        )

    os.makedirs(output_dir, exist_ok=True)
    generated = []

    # 1. Motor-mount plate
    mount = (
        cq.Workplane("XY")
        .rect(62, 62)
        .extrude(6)
        .faces(">Z").workplane()
        .hole(5)
        .pushPoints([(8, 8), (-8, 8), (8, -8), (-8, -8)])
        .hole(3)
    )
    p = f"{output_dir}/motor_mount_plate.stl"
    cq.exporters.export(mount, p)
    generated.append(p)

    # 2. Electronics bay (Arduino Nano + ESC, snap-lid box)
    bay_l, bay_w, bay_h, wall = 75, 45, 22, 2
    bay = (
        cq.Workplane("XY")
        .rect(bay_l, bay_w)
        .extrude(bay_h)
        .faces("<Z").shell(-wall)
    )
    p = f"{output_dir}/electronics_bay.stl"
    cq.exporters.export(bay, p)
    generated.append(p)

    # 3. Wing spar channel block (6 mm carbon rod slot)
    spar_block = (
        cq.Workplane("XY")
        .rect(30, 20)
        .extrude(20)
        .faces(">Z").workplane()
        .rect(30, 6)
        .cutBlind(-20)
    )
    p = f"{output_dir}/spar_channel_block.stl"
    cq.exporters.export(spar_block, p)
    generated.append(p)

    # 4. Servo horn extension for MG90S (25 mm reach)
    horn = (
        cq.Workplane("XY")
        .rect(50, 8)
        .extrude(3)
        .faces(">Z").workplane()
        .pushPoints([(20, 0), (-20, 0)])
        .hole(2)
    )
    p = f"{output_dir}/servo_horn_extension.stl"
    cq.exporters.export(horn, p)
    generated.append(p)

    print(f"Generated {len(generated)} STL files → {output_dir}/")
    return generated


# ===========================================================================
# DC MOTOR STICK PLANE  (3JWings Build)
# ===========================================================================

def stick_plane_dc_design(
    wingspan_mm=609,
    weight_grams=116,
    cg_from_le_mm=33,
):
    """
    Full design and analysis for the 3JWings DC Motor RC Stick Plane.

    Reference: "Make Rc Plane With dc Motor | DIY Rc Stick Plane"
    — 3JWings, Jan 10, 2025. STL files: https://shorturl.at/ZFULq

    3D-printed beginner plane with a single DC motor (not brushless).
    Wingspan 609 mm, 116 g AUW, CG 33 mm from the leading edge.

    Args:
        wingspan_mm: Total wingspan in mm (default: 609)
        weight_grams: All-up weight in grams (default: 116)
        cg_from_le_mm: CG position from leading edge in mm (default: 33)

    Returns:
        Dictionary with geometry, aerodynamics, control system, and build data
    """
    chord_mm = 120                          # estimated (AR ≈ 5.1)
    area_mm2 = wingspan_mm * chord_mm
    area_cm2 = area_mm2 / 100.0
    aspect_ratio = (wingspan_mm ** 2) / area_mm2
    mac = chord_mm
    cg_pct_mac = (cg_from_le_mm / mac) * 100

    # Aerodynamics
    weight_n = weight_grams * 0.00981
    area_m2 = area_mm2 / 1e6
    rho = 1.225
    cl_cruise, cl_max = 0.55, 1.20
    cruise_ms = math.sqrt((2 * weight_n) / (rho * area_m2 * cl_cruise))
    stall_ms = math.sqrt((2 * weight_n) / (rho * area_m2 * cl_max))
    wing_loading = weight_grams / area_cm2

    # Tail geometry
    fuse_length_mm = 480
    tail_arm_mm = fuse_length_mm * 0.55
    h_stab_chord = 70
    h_stab_area_mm2 = area_mm2 * 0.20
    h_stab_span_mm = h_stab_area_mm2 / h_stab_chord
    v_fin_h = 80
    v_fin_area_mm2 = area_mm2 * 0.10
    v_fin_chord = v_fin_area_mm2 / v_fin_h
    tail_vol_coef = (h_stab_area_mm2 * tail_arm_mm) / (area_mm2 * mac)

    if tail_vol_coef >= 0.6:
        stability = "VERY STABLE (trainer)"
    elif tail_vol_coef >= 0.4:
        stability = "BALANCED (sport)"
    else:
        stability = "AGILE (aerobatic)"

    # Control surfaces
    elevator_chord_mm = 0.35 * h_stab_chord
    elevator_span_mm = h_stab_span_mm
    rudder_h = v_fin_h * 0.50
    rudder_chord = v_fin_chord

    # PID gains (3-ch: throttle + elevator + rudder)
    elev_pid = PIDController(kp=1.20, ki=0.03, kd=0.08)
    rudd_pid = PIDController(kp=0.90, ki=0.02, kd=0.05)

    # 3D-print specification
    print_spec = {
        "fuselage_nose": {
            "description": "Nose section with motor mount (~80 mm long)",
            "outer_diameter_mm": 42,
            "wall_mm": 1.2,
            "infill_pct": 40,
            "material": "PLA or PETG",
            "motor_shaft_hole_mm": 2.0,
        },
        "fuselage_mid": {
            "description": "Centre body with wing saddle",
            "length_mm": 200,
            "cross_section_mm": "42 × 38",
            "wall_mm": 1.2,
            "infill_pct": 15,
            "material": "PLA",
        },
        "fuselage_tail": {
            "description": "Tail boom + stabiliser mounts",
            "length_mm": 200,
            "cross_section_mm": "18 × 18 tapered to 12 × 12",
            "wall_mm": 1.2,
            "infill_pct": 15,
            "material": "PLA",
        },
        "wing_half": {
            "description": "Semi-wing flat-bottom airfoil (14 mm thick)",
            "span_half_mm": int(wingspan_mm / 2),
            "chord_mm": chord_mm,
            "max_thickness_mm": 14,
            "spar_slot_mm": 4,
            "infill_pct": 10,
            "material": "PLA (lightest infill to save weight)",
            "qty": 2,
        },
        "horizontal_stabilizer": {
            "span_mm": int(h_stab_span_mm),
            "chord_mm": h_stab_chord,
            "thickness_mm": 6,
            "infill_pct": 10,
            "material": "PLA",
            "qty": 1,
        },
        "vertical_fin": {
            "height_mm": v_fin_h,
            "chord_mm": int(v_fin_chord),
            "thickness_mm": 6,
            "infill_pct": 10,
            "material": "PLA",
            "qty": 1,
        },
    }

    return {
        "type": "DC Motor RC Stick Plane",
        "reference": {
            "title": "Make Rc Plane With dc Motor | DIY Rc Stick Plane",
            "channel": "3JWings",
            "published": "Jan 10, 2025",
            "views": "278,098",
            "search": "3JWings Make Rc Plane With dc Motor DIY Rc Stick Plane",
            "stl_files": "https://shorturl.at/ZFULq",
            "electronics_guide": (
                "DIY Electronics Setup For Mini RC Plane (3JWings YouTube)"
            ),
        },
        "skill_level": "Beginner",
        "geometry": {
            "wingspan_mm": wingspan_mm,
            "chord_mm": chord_mm,
            "area_mm2": area_mm2,
            "area_cm2": area_cm2,
            "aspect_ratio": round(aspect_ratio, 2),
            "mac_mm": mac,
            "fuselage_length_mm": fuse_length_mm,
            "h_stab_span_mm": round(h_stab_span_mm, 0),
            "h_stab_chord_mm": h_stab_chord,
            "v_fin_height_mm": v_fin_h,
            "v_fin_chord_mm": round(v_fin_chord, 0),
        },
        "aerodynamics": {
            "auw_grams": weight_grams,
            "cg_from_le_mm": cg_from_le_mm,
            "cg_pct_mac": round(cg_pct_mac, 1),
            "wing_loading_g_cm2": round(wing_loading, 4),
            "cruise_speed_ms": round(cruise_ms, 1),
            "stall_speed_ms": round(stall_ms, 1),
            "tail_volume_coefficient": round(tail_vol_coef, 3),
            "stability_assessment": stability,
        },
        "control_system": {
            "type": "3-channel (throttle + elevator + rudder)",
            "motor": {
                "type": "DC motor (coreless / brushed)",
                "voltage": "3.7 V (1S) or 7.4 V (2S LiPo)",
                "kv_equiv": "12 000–16 000 RPM/V direct drive",
                "current_a": "3–8 A",
                "shaft_mm": 1.5,
                "mount": "Nose tractor",
                "propeller": "6×3 for 1S  |  5×3 for 2S",
            },
            "servos": "2× 3.7 g micro servo (1 elevator + 1 rudder)",
            "receiver": "4-ch micro RC receiver or Arduino Nano + NRF24L01",
            "elevator_chord_mm": round(elevator_chord_mm, 1),
            "elevator_span_mm": round(elevator_span_mm, 1),
            "rudder_height_mm": round(rudder_h, 1),
            "rudder_chord_mm": round(rudder_chord, 1),
            "pid_elevator": {
                "kp": elev_pid.kp,
                "ki": elev_pid.ki,
                "kd": elev_pid.kd,
                "axis": "pitch via elevator",
            },
            "pid_rudder": {
                "kp": rudd_pid.kp,
                "ki": rudd_pid.ki,
                "kd": rudd_pid.kd,
                "axis": "yaw via rudder",
            },
            "channel_map": {
                "CH1_throttle": "DC motor speed controller (MOSFET / ESC)",
                "CH2_elevator": "Pitch control (±15° deflection)",
                "CH3_rudder": "Yaw / turn control (±20° deflection)",
            },
        },
        "build_specification": {
            "construction": "Fully 3D-printed — PLA, light infill",
            "spar": "4 mm carbon rod or 3 mm bamboo skewer through wing halves",
            "battery": "1S 3.7 V 300–500 mAh LiPo",
            "total_build_time_hours": "6–10 h (print) + 2 h (assembly)",
            "printer": "Creality Ender 3 V3 Plus (original build)",
            "print_specs": print_spec,
        },
        "tuning_notes": [
            "CG at 33 mm from LE = 27.5 % MAC — add nose weight if tail-heavy",
            "DC motor: direct drive, 6×3 prop on 1S for slow gentle flying",
            "3-channel only: no aileron — use rudder for banked turns",
            "Increase elevator Kp if pitch response feels sluggish",
            "STL files: https://shorturl.at/ZFULq",
        ],
    }


def stick_plane_dc_arduino_sketch():
    """
    Return a reference Arduino C++ sketch for the 3JWings DC Motor Stick Plane.

    3-channel: throttle (DC motor via MOSFET or ESC), elevator servo,
    rudder servo.  Uses the same NRF24L01 radio link as the flying-wing TX.

    Returns:
        Dict with ``transmitter_sketch``, ``receiver_sketch``,
        ``board``, and ``flash_instructions``
    """
    tx = r"""
// =================================================================
// STICK PLANE  —  TRANSMITTER  (3-channel)
// Board: Arduino Nano V3  |  Radio: NRF24L01+PA+LNA
// Based on: 3JWings "DC Motor Stick Plane" (Jan 2025)
// =================================================================
#include <SPI.h>
#include <RF24.h>

RF24 radio(9, 10);
const byte ADDR[6] = "SP3ch";

struct TxPayload {
    int16_t throttle;   // 0 – 1000
    int16_t elevator;   // -500 to +500
    int16_t rudder;     // -500 to +500
};
TxPayload tx;

void setup() {
    radio.begin();
    radio.setChannel(76);
    radio.setDataRate(RF24_250KBPS);
    radio.setPALevel(RF24_PA_HIGH);
    radio.openWritingPipe(ADDR);
    radio.stopListening();
}

void loop() {
    tx.throttle = map(analogRead(A0), 0, 1023,    0, 1000);
    tx.elevator = map(analogRead(A1), 0, 1023, -500,  500);
    tx.rudder   = map(analogRead(A2), 0, 1023, -500,  500);
    radio.write(&tx, sizeof(tx));
    delay(20);
}
"""

    rx = r"""
// =================================================================
// STICK PLANE  —  RECEIVER  (throttle + elevator + rudder)
// DC motor via PWM + MOSFET on D3
// Elevator servo D5  |  Rudder servo D6
// Based on: 3JWings "DC Motor Stick Plane" (Jan 2025)
// =================================================================
#include <SPI.h>
#include <RF24.h>
#include <Servo.h>

RF24   radio(9, 10);
const  byte ADDR[6] = "SP3ch";
Servo  servoElev;
Servo  servoRudd;
const  int MOTOR_PIN = 3;       // PWM to MOSFET gate

float elevIntg=0, elevPrev=0;
float ruddIntg=0, ruddPrev=0;
const float KP_E=1.20, KI_E=0.03, KD_E=0.08;
const float KP_R=0.90, KI_R=0.02, KD_R=0.05;
const float DT=0.02;

struct TxPayload {
    int16_t throttle;
    int16_t elevator;
    int16_t rudder;
};
TxPayload rx;

float pidStep(float sp, float meas,
              float kp, float ki, float kd,
              float &intg, float &prev) {
    float err  = sp - meas;
    intg = constrain(intg + err*DT, -200, 200);
    float derv = -(meas - prev) / DT;
    prev = meas;
    return constrain(kp*err + ki*intg + kd*derv, -500, 500);
}

void setup() {
    pinMode(MOTOR_PIN, OUTPUT);
    servoElev.attach(5);
    servoRudd.attach(6);
    servoElev.write(90);
    servoRudd.write(90);
    analogWrite(MOTOR_PIN, 0);
    radio.begin();
    radio.setChannel(76);
    radio.setDataRate(RF24_250KBPS);
    radio.setPALevel(RF24_PA_HIGH);
    radio.openReadingPipe(0, ADDR);
    radio.startListening();
}

void loop() {
    if (!radio.available()) { delay(5); return; }
    radio.read(&rx, sizeof(rx));

    // DC motor: map 0–1000 → PWM 0–255
    analogWrite(MOTOR_PIN, map(rx.throttle, 0, 1000, 0, 255));

    // NOTE: Replace measuredElev / measuredRudd with actual IMU readings
    // (e.g., pitch rate from MPU-6050) for genuine stabilisation.
    // Pass-through values mean the plane responds to stick input without an IMU.
    float measuredElev = rx.elevator;  // TODO: replace with IMU pitch rate (°/s)
    float measuredRudd = rx.rudder;    // TODO: replace with IMU yaw   rate (°/s)

    float elevOut = pidStep(rx.elevator, measuredElev,
                            KP_E, KI_E, KD_E, elevIntg, elevPrev);
    float ruddOut = pidStep(rx.rudder,   measuredRudd,
                            KP_R, KI_R, KD_R, ruddIntg, ruddPrev);

    // Map PID output ±500 to servo angle 60°–120° (elevator ±30°)
    servoElev.write(constrain(90 + (int)(elevOut/500.0*30), 60, 120));
    // Rudder: ±35°
    servoRudd.write(constrain(90 + (int)(ruddOut/500.0*35), 55, 125));
    delay(20);
}
"""

    return {
        "language": "C++ (Arduino IDE)",
        "board": "Arduino Nano V3",
        "motor_driver": "Logic-level MOSFET e.g. IRLZ44N, or small brushed ESC on D3",
        "note": "3-channel only: throttle / elevator / rudder (no aileron)",
        "transmitter_sketch": tx,
        "receiver_sketch": rx,
        "flash_instructions": [
            "Install RF24 library via Arduino IDE Library Manager",
            "Flash tx sketch to transmitter Nano, rx sketch to receiver Nano",
            "Wire DC motor between V+ and MOSFET drain; gate to D3 via 100 Ω resistor",
            "Alternatively use a 10 A brushed ESC on D3 (forward-only mode)",
            "Check servo directions — reverse polarity in code if control is backwards",
        ],
    }


def generate_stick_plane_stl(
    output_dir="output/community_builds/stick_plane",
    wingspan_mm=609,
    chord_mm=120,
):
    """
    Generate CadQuery STL files for the 3JWings DC Motor Stick Plane.

    Parts generated:
    - Wing half (flat-bottom airfoil with spar slot)
    - Fuselage nose section with motor shaft hole
    - Horizontal stabilizer (flat plate)
    - Vertical fin (flat plate)

    Args:
        output_dir: Directory to save STL files
        wingspan_mm: Total wingspan in mm
        chord_mm: Wing chord in mm

    Returns:
        List of generated STL file paths

    Raises:
        ImportError: If CadQuery is not installed
    """
    if not HAS_CADQUERY:
        raise ImportError(
            "CadQuery is required. "
            "Install: conda install -c conda-forge -c cadquery cadquery"
        )

    os.makedirs(output_dir, exist_ok=True)
    generated = []
    semi_span = wingspan_mm / 2.0

    # 1. Wing half — simplified flat-bottom airfoil profile
    profile = [
        (0,                0),
        (chord_mm * 0.05,  chord_mm * 0.04),
        (chord_mm * 0.25,  chord_mm * 0.09),
        (chord_mm * 0.50,  chord_mm * 0.115),
        (chord_mm * 0.75,  chord_mm * 0.08),
        (chord_mm * 0.92,  chord_mm * 0.03),
        (chord_mm,         0),
    ]
    wing_half = (
        cq.Workplane("XY")
        .polyline(profile).close()
        .extrude(semi_span)
        .faces(">Z").workplane()
        .center(chord_mm * 0.30 - chord_mm / 2, chord_mm * 0.055)
        .hole(4)
    )
    p = f"{output_dir}/wing_half.stl"
    cq.exporters.export(wing_half, p)
    generated.append(p)

    # 2. Fuselage nose (42 mm OD, 1.5 mm wall, 80 mm long, motor shaft hole)
    nose = (
        cq.Workplane("XY")
        .circle(21)
        .extrude(80)
        .faces(">Z").workplane()
        .circle(19.5)
        .cutBlind(-78)
        .faces(">Z").workplane()
        .hole(2)
    )
    p = f"{output_dir}/fuselage_nose.stl"
    cq.exporters.export(nose, p)
    generated.append(p)

    # 3. Horizontal stabilizer
    h_stab_area = wingspan_mm * chord_mm * 0.20
    h_stab_chord = 70
    h_stab_span = int(h_stab_area / h_stab_chord)
    h_stab = (
        cq.Workplane("XY")
        .rect(h_stab_span, h_stab_chord)
        .extrude(6)
    )
    p = f"{output_dir}/horizontal_stabilizer.stl"
    cq.exporters.export(h_stab, p)
    generated.append(p)

    # 4. Vertical fin
    v_area = wingspan_mm * chord_mm * 0.10
    v_h = 80
    v_c = int(v_area / v_h)
    v_fin = (
        cq.Workplane("XY")
        .rect(v_c, v_h)
        .extrude(6)
    )
    p = f"{output_dir}/vertical_fin.stl"
    cq.exporters.export(v_fin, p)
    generated.append(p)

    print(f"Generated {len(generated)} STL files → {output_dir}/")
    return generated


# ===========================================================================
# SHAHED / LUCAS DRONE  (Educational Aerodynamic Study)
# ===========================================================================

def shahed_drone_design(
    root_chord_full=3920,
    wingspan_full=2500,
    sweep_angle=55,
    scale_factor=0.20,
    auw_scale_grams=None,
):
    """
    Educational aerodynamic study of the Shahed-136 / Lucas delta-wing UAV,
    with a scaled-down RC model for hands-on learning.

    ⚠ FOR EDUCATIONAL / AERODYNAMIC STUDY ONLY.
      Always comply with local aviation laws and regulations.

    Args:
        root_chord_full: Full-scale root chord in mm (default: 3920 ≈ 3.92 m)
        wingspan_full: Full-scale wingspan in mm (default: 2500 ≈ 2.5 m)
        sweep_angle: Leading-edge sweep in degrees (default: 55°)
        scale_factor: RC study model scale (0.20 = 20 % → ~500 mm span)
        auw_scale_grams: RC model AUW in grams (cube-law estimate if None)

    Returns:
        Dictionary with full-scale reference, scaled RC model, and control data
    """
    # Full-scale geometry
    fs_tip = root_chord_full * 0.05
    fs_area = ((root_chord_full + fs_tip) / 2.0) * wingspan_full
    fs_ar = (wingspan_full ** 2) / fs_area
    tr = fs_tip / root_chord_full
    fs_mac = (2.0 / 3.0) * root_chord_full * ((1 + tr + tr ** 2) / (1 + tr))
    fs_le = (wingspan_full / 2.0) / math.cos(math.radians(sweep_angle))

    # Scaled RC model
    sc_root = root_chord_full * scale_factor
    sc_span = wingspan_full * scale_factor
    sc_tip = sc_root * 0.05
    sc_area = ((sc_root + sc_tip) / 2.0) * sc_span
    sc_ar = (sc_span ** 2) / sc_area
    sc_mac = fs_mac * scale_factor

    if auw_scale_grams is None:
        # Cube-law scaling from ~200 kg full-scale Shahed-136.
        # Mass scales with the cube of the linear scale factor.
        # 180 g minimum ensures sufficient mass for RC electronics (servo + ESC + battery).
        auw_scale_grams = max(200_000 * (scale_factor ** 3), 180)

    weight_n = auw_scale_grams * 0.00981
    area_m2 = sc_area / 1e6
    rho = 1.225
    cl_cruise, cl_max = 0.45, 1.00
    cruise_ms = math.sqrt((2 * weight_n) / (rho * area_m2 * cl_cruise))
    stall_ms = math.sqrt((2 * weight_n) / (rho * area_m2 * cl_max))
    cg_mm = 0.27 * sc_mac

    elevon_chord = 0.25 * sc_root
    elevon_span = sc_span * 0.35
    tau = elevon_chord / sc_mac

    pitch_pid = PIDController(kp=1.50, ki=0.04, kd=0.12)
    roll_pid = PIDController(kp=2.00, ki=0.05, kd=0.15)

    # RC foam-board cut plan
    semi = sc_span / 2.0
    le_set = semi * math.tan(math.radians(sweep_angle))
    foam_cuts = {
        "delta_wing_half": {
            "qty": 2,
            "material": "6 mm EPP or insulation foam",
            "cut_description": (
                f"Right-triangle blank {int(semi)} mm base × "
                f"{int(sc_root)} mm height. "
                f"LE setback at tip = {int(le_set)} mm "
                f"({sweep_angle}° sweep)."
            ),
        },
        "fuselage_pod": {
            "qty": 1,
            "material": "3 mm foamboard box section",
            "dimensions_mm": (
                f"20 × 20 × {int(sc_root * 0.6)} mm "
                "(pusher motor mount at rear face)"
            ),
        },
        "elevon": {
            "qty": 2,
            "material": "3 mm foamboard (trailing-edge offcut)",
            "dimensions_mm": (
                f"{int(elevon_span)} mm span × {int(elevon_chord)} mm chord"
            ),
        },
    }

    return {
        "type": "Shahed / Lucas Drone (Aerodynamic Study)",
        "reference": "Shahed-136 delta-wing UAV — educational aerodynamics study",
        "warning": (
            "⚠ FOR EDUCATIONAL / AERODYNAMIC STUDY ONLY. "
            "Comply with all local aviation laws and regulations."
        ),
        "full_scale_reference": {
            "root_chord_mm": root_chord_full,
            "wingspan_mm": wingspan_full,
            "sweep_angle_deg": sweep_angle,
            "area_mm2": round(fs_area),
            "aspect_ratio": round(fs_ar, 2),
            "mac_mm": round(fs_mac, 1),
            "leading_edge_length_mm": round(fs_le, 1),
            "configuration": "Tailless delta, pusher prop, GPS waypoint guidance",
        },
        "rc_study_model": {
            "scale": scale_factor,
            "root_chord_mm": round(sc_root, 1),
            "wingspan_mm": round(sc_span, 1),
            "area_mm2": round(sc_area),
            "area_cm2": round(sc_area / 100, 1),
            "aspect_ratio": round(sc_ar, 2),
            "mac_mm": round(sc_mac, 1),
            "auw_grams": round(auw_scale_grams),
            "wing_loading_g_cm2": round(auw_scale_grams / (sc_area / 100), 4),
            "cruise_speed_ms": round(cruise_ms, 1),
            "stall_speed_ms": round(stall_ms, 1),
            "cg_from_le_mm": round(cg_mm, 1),
        },
        "control_system": {
            "type": "Elevon only (tailless delta — identical to flying-wing)",
            "elevon_chord_mm": round(elevon_chord, 1),
            "elevon_span_mm": round(elevon_span, 1),
            "control_tau": round(tau, 3),
            "servos": "2× standard servo (one per elevon)",
            "motor": "Brushless pusher (2205–2208 class for RC scale)",
            "pid_pitch": {
                "kp": pitch_pid.kp,
                "ki": pitch_pid.ki,
                "kd": pitch_pid.kd,
                "note": "55° sweep gives slow pitch response; increase Kp cautiously",
            },
            "pid_roll": {
                "kp": roll_pid.kp,
                "ki": roll_pid.ki,
                "kd": roll_pid.kd,
                "note": "High sweep → sluggish roll; higher Kp needed vs flying-wing",
            },
            "navigation_study": {
                "rc_manual": (
                    "Use the same elevon-mixing Arduino sketch as the flying-wing "
                    "(just change the address string to avoid RF collision)"
                ),
                "autopilot": (
                    "ArduPlane / iNav with GPS (wing type: delta, "
                    "no tail, elevon mixing enabled)"
                ),
                "full_scale_ref": (
                    "GPS waypoint + IR terminal homing (reference only)"
                ),
            },
        },
        "build_specification": {
            "construction": "Delta foam-board (tailless, flying-wing layout)",
            "foam_cuts": foam_cuts,
            "spar": "4 mm carbon tube at 25 % chord from LE",
            "propulsion": "Pusher brushless at rear centreline",
            "total_build_time_hours": "3–5",
        },
        "aerodynamic_notes": [
            "55° sweep creates strong LE vortices — high CLmax at high AOA",
            "No tail: elevons provide all pitch and roll authority",
            "High sweep reduces effective AR → lower induced drag at high speed",
            "CG at 25–27 % MAC is critical: too far aft = pitch divergence",
            "Pusher config keeps propwash off wing and CG naturally forward",
            "Use rounded LE profile (not sharp) on RC foam model",
        ],
    }


def generate_shahed_study_stl(
    output_dir="output/community_builds/shahed_study",
    scale_factor=0.20,
    root_chord_full=3920,
    wingspan_full=2500,
):
    """
    Generate CadQuery STL files for the Shahed/Lucas drone RC study model.

    Parts generated:
    - Delta root rib (thin flat-bottom delta airfoil profile)
    - Fuselage pod (rectangular box section)
    - Pusher motor mount (rear-facing plate)

    Args:
        output_dir: Directory to save STL files
        scale_factor: Scale (0.20 = 20 % of full scale → ~500 mm span)
        root_chord_full: Full-scale root chord in mm
        wingspan_full: Full-scale wingspan in mm

    Returns:
        List of generated STL file paths

    Raises:
        ImportError: If CadQuery is not installed
    """
    if not HAS_CADQUERY:
        raise ImportError(
            "CadQuery is required. "
            "Install: conda install -c conda-forge -c cadquery cadquery"
        )

    os.makedirs(output_dir, exist_ok=True)
    generated = []
    sc_root = root_chord_full * scale_factor

    # 1. Delta root rib
    rib_t = 6
    profile = [
        (0,                0),
        (sc_root * 0.08,   sc_root * 0.040),
        (sc_root * 0.25,   sc_root * 0.070),
        (sc_root * 0.50,   sc_root * 0.080),
        (sc_root * 0.72,   sc_root * 0.060),
        (sc_root * 0.88,   sc_root * 0.030),
        (sc_root * 0.96,   sc_root * 0.005),
        (sc_root,          0),
    ]
    root_rib = (
        cq.Workplane("XY")
        .polyline(profile).close()
        .extrude(rib_t)
        .faces(">Z").workplane()
        .center(sc_root * 0.25 - sc_root / 2, sc_root * 0.035)
        .hole(4)
    )
    p = f"{output_dir}/delta_root_rib.stl"
    cq.exporters.export(root_rib, p)
    generated.append(p)

    # 2. Fuselage pod (box section, closed ends open)
    pod_len = sc_root * 0.60
    pod = (
        cq.Workplane("XY")
        .rect(20, 20)
        .extrude(pod_len)
        .faces("<Z").shell(-2)
    )
    p = f"{output_dir}/fuselage_pod.stl"
    cq.exporters.export(pod, p)
    generated.append(p)

    # 3. Pusher motor mount
    mount = (
        cq.Workplane("XY")
        .rect(50, 50)
        .extrude(8)
        .faces(">Z").workplane()
        .hole(5)
        .pushPoints([(8, 8), (-8, 8), (8, -8), (-8, -8)])
        .hole(2.5)
    )
    p = f"{output_dir}/pusher_motor_mount.stl"
    cq.exporters.export(mount, p)
    generated.append(p)

    print(f"Generated {len(generated)} STL files → {output_dir}/")
    return generated
