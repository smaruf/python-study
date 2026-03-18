"""
Fixed-Wing Aircraft Design Components

Modules for designing fixed-wing aircraft structures:
- wing_rib: Parametric wing rib generator
- spar: Spar design and load calculations
- fuselage: Fuselage section generator
- tail: Tail components design
- loads: Flight load calculations
- wing_types: Advanced wing configurations (delta, flying-wing, canard, oblique, pancake)
- community_builds: Full designs for three real YouTube builds —
    RCMakerLab Flying-Wing RC Plane (Oct 2025),
    3JWings DC Motor Stick Plane (Jan 2025),
    Shahed/Lucas Drone aerodynamic study.
  Each build includes: geometry, aerodynamics, PID control system,
  foamboard/3D-print build specs, electronics BOM, and Arduino C++ sketches.
- microcontroller_firmware: Raspberry Pi Pico (MicroPython) and Arduino (C++) firmware
    for all three builds, plus component-level BOMs and ASCII wiring diagrams.
"""
