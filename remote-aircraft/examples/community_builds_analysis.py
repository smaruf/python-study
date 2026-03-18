"""
Community Builds — Full Design & Analysis

Runs complete design, aerodynamic, control-system, and build analysis
for all three community-build aircraft:

1. RCMakerLab Flying-Wing RC Plane (Oct 2025)
   Delta-wing type with Arduino NRF24L01 remote control.

2. 3JWings DC Motor Stick Plane (Jan 2025)
   Beginner-friendly 3D-printed plane, 609 mm wingspan, 116 g AUW.

3. Shahed / Lucas Drone (educational aerodynamic study)
   Delta-wing UAV planform scaled to an RC study model.

Run with:
    PYTHONPATH=. python examples/community_builds_analysis.py

Optional STL generation (requires CadQuery):
    Uncomment the generate_*_stl() calls at the bottom of main().
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fixed_wing.community_builds import (
    PIDController,
    simulate_step_response,
    elevon_mix,
    flying_wing_rc_design,
    flying_wing_rc_arduino_sketch,
    flying_wing_rc_foamboard_plan,
    generate_flying_wing_rc_stl,
    stick_plane_dc_design,
    stick_plane_dc_arduino_sketch,
    generate_stick_plane_stl,
    shahed_drone_design,
    generate_shahed_study_stl,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(title, width=80, char="="):
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)
    print()


def _sub(title):
    print(f"\n--- {title} ---")


def _kv(label, value, indent=2):
    pad = " " * indent
    if isinstance(value, float):
        print(f"{pad}{label}: {value:.3f}")
    else:
        print(f"{pad}{label}: {value}")


def _tree(d, indent=2):
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            _tree(v, indent + 2)
        elif isinstance(v, list):
            print(f"{pad}{k}:")
            for item in v:
                if isinstance(item, dict):
                    _tree(item, indent + 4)
                else:
                    print(f"{' ' * (indent + 2)}- {item}")
        elif isinstance(v, float):
            print(f"{pad}{k}: {v:.3f}")
        else:
            print(f"{pad}{k}: {v}")


# ---------------------------------------------------------------------------
# 1. Flying-Wing RC Plane
# ---------------------------------------------------------------------------

def analyse_flying_wing():
    _header("FLYING-WING RC PLANE  (RCMakerLab, Oct 2025)")
    d = flying_wing_rc_design()

    _sub("Reference")
    for k, v in d["reference"].items():
        _kv(k, v)

    _sub("Geometry")
    _tree(d["geometry"])

    _sub("Aerodynamics")
    _tree(d["aerodynamics"])

    _sub("Control System")
    ctrl = d["control_system"]
    for k, v in ctrl.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                print(f"    {sk}: {sv}")
        elif isinstance(v, list):
            print(f"  {k}:")
            for item in v:
                print(f"    - {item}")
        else:
            print(f"  {k}: {v}")

    _sub("Electronics BOM")
    for item in d["electronics_bom"]:
        print(f"  • {item}")

    _sub("Build Specification — Foam Cuts")
    spec = d["build_specification"]
    print(f"  Construction  : {spec['construction']}")
    print(f"  Spar          : {spec['spar']}")
    print(f"  Build time    : {spec['total_build_time_hours']}")
    print(f"  CG note       : {spec['cg_note']}")
    print()
    for part, info in spec["foam_cuts"].items():
        print(f"  [{info['qty']}×] {part}")
        print(f"       Material : {info['material']}")
        print(f"       Blank    : {info.get('blank_mm', '')}")
        print(f"       Cut      : {info.get('cut', '')}")

    _sub("Tuning Notes")
    for note in d["tuning_notes"]:
        print(f"  ► {note}")


# ---------------------------------------------------------------------------
# 2. Elevon Mixing Demonstration
# ---------------------------------------------------------------------------

def demo_elevon_mixing():
    _header("ELEVON MIXING  (flying-wing & delta-wing)")
    print("  Input range: throttle 1000–2000 µs, pitch/roll ±500 stick units")
    print()

    cases = [
        ("Neutral",                 1500,     0,     0),
        ("Full throttle, level",    2000,     0,     0),
        ("Pull up  (pitch +)",      1500,   300,     0),
        ("Push down (pitch −)",     1500,  -300,     0),
        ("Roll right",              1500,     0,   300),
        ("Roll left",               1500,     0,  -300),
        ("Climb + bank right",      1700,   200,   200),
        ("Dive  + bank left",       1400,  -200,  -200),
    ]

    hdr = f"  {'Case':<28} {'Throttle':>10} {'Left elv':>10} {'Right elv':>11}"
    print(hdr)
    print("  " + "-" * 62)
    for label, thr, pit, rol in cases:
        out = elevon_mix(thr, pit, rol)
        print(
            f"  {label:<28}"
            f" {out['throttle_us']:>10}"
            f" {out['left_elevon_us']:>10}"
            f" {out['right_elevon_us']:>11}"
        )


# ---------------------------------------------------------------------------
# 3. PID Step-Response Simulation
# ---------------------------------------------------------------------------

def demo_pid_response():
    _header("PID STEP-RESPONSE SIMULATION  (roll axis, flying-wing)")

    target = 30.0
    pid = PIDController(kp=1.80, ki=0.05, kd=0.12, out_limit=500)
    history = simulate_step_response(pid, target=target, dt=0.02, steps=200)

    print(f"  Target bank: {target}°  |  Kp=1.80  Ki=0.05  Kd=0.12")
    print()
    print(f"  {'Time (s)':>9}  {'Measured (°)':>14}  {'PID output':>12}")
    print("  " + "-" * 42)
    for t, meas, out in history[::10]:
        print(f"  {t:>9.2f}  {meas:>14.3f}  {out:>12.3f}")

    final = history[-1][1]
    peak = max(h[1] for h in history)
    overshoot = peak - target
    settle_t = next(
        (h[0] for h in reversed(history) if abs(h[1] - target) > 0.5 * 2),
        history[-1][0],
    )
    print()
    print(f"  Final value : {final:.2f}°  (target {target}°)")
    print(f"  Peak overshoot: {overshoot:.2f}°")
    print(f"  Approx settle time: {settle_t:.2f} s  (within ±1°)")
    print()
    print("  Tip: increase Kd to reduce overshoot; increase Kp for faster response")


# ---------------------------------------------------------------------------
# 4. DC Motor Stick Plane
# ---------------------------------------------------------------------------

def analyse_stick_plane():
    _header("DC MOTOR RC STICK PLANE  (3JWings, Jan 2025)")
    d = stick_plane_dc_design()

    _sub("Reference")
    for k, v in d["reference"].items():
        _kv(k, v)

    _sub("Geometry")
    _tree(d["geometry"])

    _sub("Aerodynamics")
    _tree(d["aerodynamics"])

    _sub("Control System")
    ctrl = d["control_system"]
    print(f"  Type    : {ctrl['type']}")
    mot = ctrl["motor"]
    print(f"  Motor   : {mot['type']}  |  {mot['voltage']}  |  prop {mot['propeller']}")
    print(f"  Servos  : {ctrl['servos']}")
    print(f"  PID Elevator : Kp={ctrl['pid_elevator']['kp']}  Ki={ctrl['pid_elevator']['ki']}  Kd={ctrl['pid_elevator']['kd']}")
    print(f"  PID Rudder   : Kp={ctrl['pid_rudder']['kp']}  Ki={ctrl['pid_rudder']['ki']}  Kd={ctrl['pid_rudder']['kd']}")
    print(f"  Channel map:")
    for ch, desc in ctrl["channel_map"].items():
        print(f"    {ch}: {desc}")

    _sub("3D Print Specifications")
    for part, info in d["build_specification"]["print_specs"].items():
        print(f"  {part}:")
        for k, v in info.items():
            print(f"    {k}: {v}")

    _sub("Tuning Notes")
    for note in d["tuning_notes"]:
        print(f"  ► {note}")


# ---------------------------------------------------------------------------
# 5. Shahed / Lucas Drone Study
# ---------------------------------------------------------------------------

def analyse_shahed():
    _header("SHAHED / LUCAS DRONE  (Educational Aerodynamic Study)")
    d = shahed_drone_design()

    print(f"  {d['warning']}")
    print()

    _sub("Full-Scale Reference Geometry")
    _tree(d["full_scale_reference"])

    _sub("RC Study Model  (20 % scale)")
    _tree(d["rc_study_model"])

    _sub("Control System")
    ctrl = d["control_system"]
    print(f"  Type        : {ctrl['type']}")
    print(f"  Elevon chord: {ctrl['elevon_chord_mm']} mm")
    print(f"  Elevon span : {ctrl['elevon_span_mm']} mm")
    print(f"  Control τ   : {ctrl['control_tau']}")
    print(f"  PID Pitch  : Kp={ctrl['pid_pitch']['kp']}  Ki={ctrl['pid_pitch']['ki']}  Kd={ctrl['pid_pitch']['kd']}")
    print(f"             → {ctrl['pid_pitch']['note']}")
    print(f"  PID Roll   : Kp={ctrl['pid_roll']['kp']}  Ki={ctrl['pid_roll']['ki']}  Kd={ctrl['pid_roll']['kd']}")
    print(f"             → {ctrl['pid_roll']['note']}")
    print(f"  Navigation study:")
    for k, v in ctrl["navigation_study"].items():
        print(f"    {k}: {v}")

    _sub("RC Foam-Board Cut Plan")
    for part, info in d["build_specification"]["foam_cuts"].items():
        print(f"  {part} ({info['qty']}×): {info['material']}")
        if "cut_description" in info:
            print(f"    → {info['cut_description']}")
        if "dimensions_mm" in info:
            print(f"    → {info['dimensions_mm']}")

    _sub("Aerodynamic Notes")
    for note in d["aerodynamic_notes"]:
        print(f"  ► {note}")


# ---------------------------------------------------------------------------
# 6. Arduino Sketch Summary
# ---------------------------------------------------------------------------

def show_arduino_summary():
    _header("ARDUINO SKETCHES SUMMARY")

    fw = flying_wing_rc_arduino_sketch()
    sp = stick_plane_dc_arduino_sketch()

    print("Flying-Wing RC Plane")
    print(f"  Board         : {fw['board']}")
    print(f"  Radio library : {fw['radio_library']}")
    print("  Flash steps:")
    for step in fw["flash_instructions"]:
        print(f"    - {step}")

    print()
    print("DC Motor Stick Plane")
    print(f"  Board         : {sp['board']}")
    print(f"  Motor driver  : {sp['motor_driver']}")
    print("  Flash steps:")
    for step in sp["flash_instructions"]:
        print(f"    - {step}")

    print()
    print("  Full C++ sketch source is available as Python strings in the dict keys")
    print("  'transmitter_sketch' and 'receiver_sketch' returned by each function.")
    print("  Copy the string into a .ino file and flash from Arduino IDE.")


# ---------------------------------------------------------------------------
# 7. Foamboard Cutting Plan
# ---------------------------------------------------------------------------

def show_foamboard_plan():
    _header("FOAMBOARD CUTTING PLAN  (Flying-Wing RC Plane)")
    plan = flying_wing_rc_foamboard_plan()

    print(plan["ascii_layout"])

    print("Detailed Cut List:")
    for item in plan["cut_list"]:
        print(f"  [{item['qty']}×] {item['part']}")
        print(f"        Material : {item['material']}")
        print(f"        Blank    : {item['blank_mm']}")
        print(f"        Cut      : {item['cut']}")

    print()
    print("Assembly Steps:")
    for step in plan["assembly_steps"]:
        print(f"  {step}")

    print()
    print("Tools Needed:")
    for tool in plan["tools_needed"]:
        print(f"  • {tool}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print()
    print("█" * 80)
    print("  COMMUNITY BUILDS — FULL DESIGN, CONTROL SYSTEM & BUILD ANALYSIS")
    print("  1. Flying-Wing RC Plane  (RCMakerLab, Oct 2025)")
    print("  2. DC Motor Stick Plane  (3JWings, Jan 2025)")
    print("  3. Shahed / Lucas Drone  (educational aerodynamic study)")
    print("█" * 80)

    analyse_flying_wing()
    demo_elevon_mixing()
    demo_pid_response()
    analyse_stick_plane()
    analyse_shahed()
    show_arduino_summary()
    show_foamboard_plan()

    _header("ANALYSIS COMPLETE")
    print("  Next steps:")
    print("  1. Build the Flying-Wing from foam (see foamboard plan above)")
    print("  2. Print stick-plane parts from STL files: https://shorturl.at/ZFULq")
    print("  3. Flash Arduino sketches via flying_wing_rc_arduino_sketch()")
    print("     and stick_plane_dc_arduino_sketch() — see 'transmitter_sketch' key")
    print()
    print("  STL generation (requires CadQuery):")
    print("    from fixed_wing.community_builds import (")
    print("        generate_flying_wing_rc_stl,")
    print("        generate_stick_plane_stl,")
    print("        generate_shahed_study_stl,")
    print("    )")
    print("    generate_flying_wing_rc_stl()")
    print("    generate_stick_plane_stl()")
    print("    generate_shahed_study_stl()")
    print()


if __name__ == "__main__":
    main()
