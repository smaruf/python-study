"""
Generate 3D-printable parts for the 3JWings DC Motor Stick Plane.

Runs with or without CadQuery:
- without CadQuery: prints the full part specification
- with CadQuery: also exports STL files
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fixed_wing.community_builds import (
    HAS_CADQUERY,
    generate_stick_plane_stl,
    stick_plane_dc_design,
)


def _print_header(title):
    print("=" * 72)
    print(title)
    print("=" * 72)


def _print_specs(design):
    geometry = design["geometry"]
    aero = design["aerodynamics"]
    build = design["build_specification"]

    print(f"Wingspan        : {geometry['wingspan_mm']} mm")
    print(f"Chord           : {geometry['chord_mm']} mm")
    print(f"AUW             : {aero['auw_grams']} g")
    print(f"CG from LE      : {aero['cg_from_le_mm']} mm")
    print(f"Cruise speed    : {aero['cruise_speed_ms']} m/s")
    print(f"Stall speed     : {aero['stall_speed_ms']} m/s")
    print(f"Construction    : {build['construction']}")
    print(f"Spar            : {build['spar']}")
    print(f"Printer         : {build['printer']}")
    print()
    print("Printable parts:")
    for part_name, spec in build["print_specs"].items():
        print(f"  - {part_name}")
        for key, value in spec.items():
            print(f"      {key}: {value}")


def main():
    design = stick_plane_dc_design()
    output_dir = "output/community_builds/stick_plane"

    _print_header("3JWINGS DC MOTOR STICK PLANE — 3D PRINT GENERATOR")
    _print_specs(design)
    print()

    if not HAS_CADQUERY:
        print("CadQuery not installed.")
        print("Install with:")
        print("  conda install -c conda-forge -c cadquery cadquery")
        print()
        print("No STL files were generated, but the full print specification is shown above.")
        return

    print(f"Generating STL files in: {output_dir}")
    generated = generate_stick_plane_stl(
        output_dir=output_dir,
        wingspan_mm=design["geometry"]["wingspan_mm"],
        chord_mm=design["geometry"]["chord_mm"],
    )
    print()
    print(f"Generated {len(generated)} STL files:")
    for path in generated:
        print(f"  ✓ {path}")
    print()
    print("Next steps:")
    print("  1. Slice using the documented infill and material per part.")
    print("  2. Dry-fit wing spar, fuselage sections, and tail parts.")
    print("  3. Balance the finished aircraft at 33 mm from the leading edge.")


if __name__ == "__main__":
    main()
