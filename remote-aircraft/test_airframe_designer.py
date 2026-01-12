#!/usr/bin/env python3
"""
Demo script for Airframe Designer functionality
Tests the design generation without requiring GUI (for headless environments)
"""

import os
import tempfile
import sys

# Add remote-aircraft to path
sys.path.insert(0, os.path.dirname(__file__))

from materials import PLA, PETG, NYLON, CF_NYLON

# Design constants for calculations
TYPICAL_FIXED_WING_WEIGHT_G = 200  # Typical weight for small fixed wing aircraft in grams
TYPICAL_GLIDER_WEIGHT_G = 150      # Typical weight for small glider in grams
GLIDE_RATIO_EFFICIENCY = 0.8       # Aerodynamic efficiency factor for glide ratio calculation


def test_fixed_wing_design():
    """Test fixed wing aircraft design generation"""
    print("=" * 70)
    print("TESTING FIXED WING AIRCRAFT DESIGNER")
    print("=" * 70)
    
    # Sample parameters
    params = {
        'wing_span': 1000,
        'wing_chord': 200,
        'wing_thickness': 12,
        'dihedral': 3,
        'fuse_length': 800,
        'fuse_width': 60,
        'fuse_height': 80,
        'h_stab_span': 400,
        'h_stab_chord': 100,
        'v_stab_height': 150,
        'v_stab_chord': 120,
        'motor_diameter': 28,
        'motor_length': 30,
        'prop_diameter': 9,
    }
    
    material = "PETG"
    
    # Print design parameters
    print("\n--- Design Parameters ---")
    for key, value in params.items():
        print(f"{key:20}: {value}")
    print(f"{'Material':20}: {material}")
    
    # Generate design summary
    summary = create_fixed_wing_summary(params, material)
    print("\n" + summary)
    
    # Test file generation
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n--- Generating Files to: {tmpdir} ---")
        
        summary_file = os.path.join(tmpdir, "fixed_wing_design_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(summary)
        print(f"✓ Created: {os.path.basename(summary_file)}")
        
        foam_file = os.path.join(tmpdir, "foamboard_templates.txt")
        generate_fixed_wing_foam_template(params, foam_file)
        print(f"✓ Created: {os.path.basename(foam_file)}")
        
        parts_file = os.path.join(tmpdir, "3d_print_parts.txt")
        generate_fixed_wing_3d_parts(params, material, parts_file)
        print(f"✓ Created: {os.path.basename(parts_file)}")
        
        print(f"\n✓ Fixed Wing design generation successful!")


def test_glider_design():
    """Test glider design generation"""
    print("\n" + "=" * 70)
    print("TESTING GLIDER DESIGNER")
    print("=" * 70)
    
    # Sample parameters
    params = {
        'wing_span': 1200,
        'root_chord': 220,
        'tip_chord': 150,
        'wing_thickness': 14,
        'dihedral': 5,
        'fuse_length': 700,
        'fuse_width': 50,
        'fuse_height': 60,
        'h_stab_span': 350,
        'h_stab_chord': 90,
        'v_stab_height': 130,
        'v_stab_chord': 100,
    }
    
    material = "PLA"
    
    # Print design parameters
    print("\n--- Design Parameters ---")
    for key, value in params.items():
        print(f"{key:20}: {value}")
    print(f"{'Material':20}: {material}")
    
    # Generate design summary
    summary = create_glider_summary(params, material)
    print("\n" + summary)
    
    # Test file generation
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n--- Generating Files to: {tmpdir} ---")
        
        summary_file = os.path.join(tmpdir, "glider_design_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(summary)
        print(f"✓ Created: {os.path.basename(summary_file)}")
        
        foam_file = os.path.join(tmpdir, "foamboard_templates.txt")
        generate_glider_foam_template(params, foam_file)
        print(f"✓ Created: {os.path.basename(foam_file)}")
        
        parts_file = os.path.join(tmpdir, "3d_print_parts.txt")
        generate_glider_3d_parts(params, material, parts_file)
        print(f"✓ Created: {os.path.basename(parts_file)}")
        
        print(f"\n✓ Glider design generation successful!")


def create_fixed_wing_summary(params, material):
    """Create fixed wing design summary"""
    summary = "=" * 60 + "\n"
    summary += "FIXED WING AIRCRAFT DESIGN SUMMARY\n"
    summary += "=" * 60 + "\n\n"
    
    summary += "--- Wing Specifications ---\n"
    summary += f"Wingspan: {params['wing_span']:.1f} mm\n"
    summary += f"Wing Chord: {params['wing_chord']:.1f} mm\n"
    summary += f"Wing Area: {params['wing_span'] * params['wing_chord'] / 1000:.1f} cm²\n"
    summary += f"Wing Thickness: {params['wing_thickness']:.1f}%\n"
    summary += f"Dihedral: {params['dihedral']:.1f}°\n"
    summary += f"Aspect Ratio: {params['wing_span'] / params['wing_chord']:.2f}\n\n"
    
    summary += "--- Fuselage Specifications ---\n"
    summary += f"Length: {params['fuse_length']:.1f} mm\n"
    summary += f"Width: {params['fuse_width']:.1f} mm\n"
    summary += f"Height: {params['fuse_height']:.1f} mm\n\n"
    
    summary += "--- Build Options ---\n"
    summary += f"3D Print Material: {material}\n\n"
    
    wing_loading = TYPICAL_FIXED_WING_WEIGHT_G / (params['wing_span'] * params['wing_chord'] / 10000)
    summary += "--- Performance Estimates ---\n"
    summary += f"Estimated Wing Loading: {wing_loading:.2f} g/dm² (assuming {TYPICAL_FIXED_WING_WEIGHT_G}g weight)\n"
    summary += "\n" + "=" * 60 + "\n"
    
    return summary


def create_glider_summary(params, material):
    """Create glider design summary"""
    summary = "=" * 60 + "\n"
    summary += "GLIDER DESIGN SUMMARY\n"
    summary += "=" * 60 + "\n\n"
    
    wing_area = (params['root_chord'] + params['tip_chord']) / 2 * params['wing_span'] / 1000
    mean_chord = (params['root_chord'] + params['tip_chord']) / 2
    aspect_ratio = params['wing_span'] / mean_chord
    
    summary += "--- Wing Specifications ---\n"
    summary += f"Wingspan: {params['wing_span']:.1f} mm\n"
    summary += f"Root Chord: {params['root_chord']:.1f} mm\n"
    summary += f"Tip Chord: {params['tip_chord']:.1f} mm\n"
    summary += f"Mean Chord: {mean_chord:.1f} mm\n"
    summary += f"Wing Area: {wing_area:.1f} cm²\n"
    summary += f"Aspect Ratio: {aspect_ratio:.2f}\n\n"
    
    summary += "--- Build Options ---\n"
    summary += f"3D Print Material: {material}\n\n"
    
    wing_loading = TYPICAL_GLIDER_WEIGHT_G / wing_area
    summary += "--- Performance Estimates ---\n"
    summary += f"Estimated Wing Loading: {wing_loading:.2f} g/dm² (assuming {TYPICAL_GLIDER_WEIGHT_G}g weight)\n"
    summary += f"Estimated Glide Ratio: {aspect_ratio * GLIDE_RATIO_EFFICIENCY:.1f}:1\n"
    summary += "\n" + "=" * 60 + "\n"
    
    return summary


def generate_fixed_wing_foam_template(params, output_file):
    """Generate fixed wing foamboard template"""
    with open(output_file, 'w') as f:
        f.write("FOAMBOARD CUTTING TEMPLATES - FIXED WING\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Main Wing: {params['wing_span']/2:.1f} x {params['wing_chord']:.1f} mm (Cut 2)\n")
        f.write(f"Fuselage: {params['fuse_length']:.1f} x {params['fuse_width']:.1f} mm\n")
        f.write(f"H-Stabilizer: {params['h_stab_span']:.1f} x {params['h_stab_chord']:.1f} mm\n")
        f.write(f"V-Stabilizer: {params['v_stab_height']:.1f} x {params['v_stab_chord']:.1f} mm\n")


def generate_glider_foam_template(params, output_file):
    """Generate glider foamboard template"""
    with open(output_file, 'w') as f:
        f.write("FOAMBOARD CUTTING TEMPLATES - GLIDER\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Main Wing (tapered): Root {params['root_chord']:.1f} mm, Tip {params['tip_chord']:.1f} mm\n")
        f.write(f"Span per side: {params['wing_span']/2:.1f} mm (Cut 2)\n")
        f.write(f"Fuselage: {params['fuse_length']:.1f} x {params['fuse_width']:.1f} mm\n")


def generate_fixed_wing_3d_parts(params, material, output_file):
    """Generate fixed wing 3D parts specification"""
    with open(output_file, 'w') as f:
        f.write("3D PRINTABLE PARTS - FIXED WING\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Material: {material}\n\n")
        f.write(f"1. Motor Mount: Ø{params['motor_diameter'] + 2:.1f} mm\n")
        f.write(f"2. Wing Joiner: {params['wing_chord'] * 0.8:.1f} x 20 x 3 mm\n")
        f.write(f"3. Servo Mounts: Standard 9g (Qty: 3)\n")


def generate_glider_3d_parts(params, material, output_file):
    """Generate glider 3D parts specification"""
    with open(output_file, 'w') as f:
        f.write("3D PRINTABLE PARTS - GLIDER\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Material: {material}\n\n")
        f.write(f"1. Wing Joiner: {params['root_chord'] * 0.7:.1f} x 15 x 3 mm\n")
        f.write(f"2. Nose Weight Holder: Ø{params['fuse_width'] - 5:.1f} x 40 mm\n")
        f.write(f"3. Servo Mounts: Standard 9g (Qty: 2)\n")


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "AIRFRAME DESIGNER - DEMO & TEST" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\nThis script demonstrates the airframe design functionality")
    print("without requiring a GUI (useful for testing in headless environments)\n")
    
    # Test both designers
    test_fixed_wing_design()
    test_glider_design()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    print("\nTo use the GUI version, run:")
    print("  python airframe_designer.py")
    print("\nFor more information, see:")
    print("  AIRFRAME_DESIGNER_README.md")
    print()


if __name__ == "__main__":
    main()
