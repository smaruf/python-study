#!/usr/bin/env python3
"""
Verification script for wing_types module
Tests the wing type design functions to ensure they work correctly
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fixed_wing.wing_types import (
    straight_wing_design,
    backward_swept_wing_design,
    forward_swept_wing_design,
    delta_wing_design,
    flying_wing_design,
    canard_design,
    oblique_wing_design,
    flying_pancake_design,
    compare_wing_types
)


def test_straight_wing():
    """Test straight wing design function"""
    print("Testing Straight Wing Design...")
    
    design = straight_wing_design(
        wingspan=1200,
        chord=200,
        taper_ratio=0.7,
        dihedral=3
    )
    
    # Basic validation
    assert design['type'] == 'Straight Wing', "Type mismatch"
    assert design['geometry']['wingspan_mm'] == 1200, "Wingspan mismatch"
    assert design['geometry']['aspect_ratio'] > 0, "Invalid aspect ratio"
    assert design['geometry']['taper_ratio'] == 0.7, "Taper ratio mismatch"
    
    print("  ✓ Straight wing design generation successful")
    print(f"  - Aspect Ratio: {design['geometry']['aspect_ratio']:.2f}")
    print(f"  - Wing Area: {design['geometry']['wing_area_mm2']/1000:.1f} cm²")
    return True


def test_backward_swept_wing():
    """Test backward swept wing design function"""
    print("\nTesting Backward Swept Wing Design...")
    
    design = backward_swept_wing_design(
        wingspan=1200,
        chord=200,
        sweep_angle=25,
        taper_ratio=0.6
    )
    
    # Basic validation
    assert design['type'] == 'Backward Swept Wing', "Type mismatch"
    assert design['geometry']['sweep_angle_deg'] == 25, "Sweep angle mismatch"
    assert design['aerodynamics']['required_washout_deg'] < 0, "Washout should be negative"
    assert design['structure']['washout_required'] == True, "Washout required"
    
    print("  ✓ Backward swept wing design generation successful")
    print(f"  - Aspect Ratio: {design['geometry']['aspect_ratio']:.2f}")
    print(f"  - Required Washout: {design['aerodynamics']['required_washout_deg']:.1f}°")
    return True


def test_forward_swept_wing():
    """Test forward swept wing design function"""
    print("\nTesting Forward Swept Wing Design...")
    
    design = forward_swept_wing_design(
        wingspan=1200,
        chord=200,
        sweep_angle=25,
        taper_ratio=0.6
    )
    
    # Basic validation
    assert design['type'] == 'Forward Swept Wing', "Type mismatch"
    assert design['geometry']['sweep_angle_deg'] == -25, "Sweep should be negative"
    assert design['aerodynamics']['root_stall_first'] == True, "Root should stall first"
    assert design['structure']['stiffness_requirement'] > 2, "Should require high stiffness"
    
    print("  ✓ Forward swept wing design generation successful")
    print(f"  - Aspect Ratio: {design['geometry']['aspect_ratio']:.2f}")
    print(f"  - Stiffness Requirement: {design['structure']['stiffness_requirement']:.1f}×")
    return True


def test_delta_wing():
    """Test delta wing design function"""
    print("Testing Delta Wing Design...")
    
    design = delta_wing_design(
        root_chord=400,
        wingspan=1000,
        sweep_angle=45,
        thickness_ratio=0.08
    )
    
    # Basic validation
    assert design['type'] == 'Delta Wing', "Type mismatch"
    assert design['geometry']['root_chord_mm'] == 400, "Root chord mismatch"
    assert design['geometry']['wingspan_mm'] == 1000, "Wingspan mismatch"
    assert design['geometry']['aspect_ratio'] > 0, "Invalid aspect ratio"
    assert design['geometry']['area_mm2'] > 0, "Invalid area"
    
    print("  ✓ Delta wing design generation successful")
    print(f"  - Aspect Ratio: {design['geometry']['aspect_ratio']:.2f}")
    print(f"  - Wing Area: {design['geometry']['area_mm2']/1000:.1f} cm²")
    return True


def test_flying_wing():
    """Test flying wing design function"""
    print("\nTesting Flying Wing Design...")
    
    design = flying_wing_design(
        center_chord=350,
        wingspan=1200,
        sweep_angle=25,
        wing_twist=-2
    )
    
    # Basic validation
    assert design['type'] == 'Flying Wing', "Type mismatch"
    assert design['geometry']['center_chord_mm'] == 350, "Center chord mismatch"
    assert design['geometry']['wingspan_mm'] == 1200, "Wingspan mismatch"
    assert design['aerodynamics']['efficiency_gain'] > 1.0, "Invalid efficiency gain"
    assert design['aerodynamics']['cg_tolerance_mm'] == 5, "CG tolerance mismatch"
    
    print("  ✓ Flying wing design generation successful")
    print(f"  - Aspect Ratio: {design['geometry']['aspect_ratio']:.2f}")
    print(f"  - Efficiency Gain: {design['aerodynamics']['efficiency_gain']:.0%}")
    return True


def test_canard():
    """Test canard design function"""
    print("\nTesting Canard Design...")
    
    design = canard_design(
        main_wing_chord=200,
        main_wingspan=1000,
        canard_chord=80,
        canard_span=400,
        canard_position=150
    )
    
    # Basic validation
    assert design['type'] == 'Canard', "Type mismatch"
    assert design['main_wing']['chord_mm'] == 200, "Main wing chord mismatch"
    assert design['canard']['chord_mm'] == 80, "Canard chord mismatch"
    assert design['configuration']['area_ratio'] > 0, "Invalid area ratio"
    assert design['aerodynamics']['stall_safety'] == True, "Stall safety should be True"
    
    print("  ✓ Canard design generation successful")
    print(f"  - Area Ratio: {design['configuration']['area_ratio']:.2f}")
    print(f"  - Canard Volume: {design['configuration']['canard_volume_coefficient']:.3f}")
    return True


def test_oblique_wing():
    """Test oblique wing design function"""
    print("\nTesting Oblique Wing Design...")
    
    # Test at different sweep angles
    for sweep in [0, 20, 30, 45]:
        design = oblique_wing_design(
            wingspan=1000,
            chord=180,
            sweep_angle=sweep,
            pivot_position=0.5
        )
        
        # Basic validation
        assert design['type'] == 'Oblique Wing', "Type mismatch"
        assert design['geometry']['wingspan_mm'] == 1000, "Wingspan mismatch"
        assert design['geometry']['current_sweep_deg'] == sweep, "Sweep angle mismatch"
        assert design['geometry']['effective_span_mm'] > 0, "Invalid effective span"
        assert f"{sweep}_deg" in design['performance_envelope'], "Performance envelope missing"
    
    print("  ✓ Oblique wing design generation successful")
    print("  - Tested sweep angles: 0°, 20°, 30°, 45°")
    return True


def test_flying_pancake():
    """Test flying pancake design function"""
    print("\nTesting Flying Pancake Design...")
    
    design = flying_pancake_design(
        diameter=600,
        thickness_ratio=0.12,
        center_cutout=120
    )
    
    # Basic validation
    assert design['type'] == 'Flying Pancake (Circular Wing)', "Type mismatch"
    assert design['geometry']['diameter_mm'] == 600, "Diameter mismatch"
    assert design['geometry']['aspect_ratio'] < 2, "Aspect ratio should be low for pancake"
    assert design['structure']['num_radial_ribs'] == 12, "Radial ribs mismatch"
    assert design['aerodynamics']['drag_coefficient'] > 0, "Invalid drag coefficient"
    
    print("  ✓ Flying pancake design generation successful")
    print(f"  - Aspect Ratio: {design['geometry']['aspect_ratio']:.2f}")
    print(f"  - Wing Area: {design['geometry']['wing_area_mm2']/1000:.1f} cm²")
    return True


def test_comparison():
    """Test wing type comparison function"""
    print("\nTesting Wing Type Comparison...")
    
    purposes = ["general", "speed", "efficiency", "aerobatic", "fun"]
    
    for purpose in purposes:
        comparison = compare_wing_types(
            weight=1400,
            target_speed=15,
            purpose=purpose
        )
        
        # Basic validation
        assert 'recommendation' in comparison, "Missing recommendation"
        assert 'all_comparisons' in comparison, "Missing comparisons"
        assert 'primary' in comparison['recommendation'], "Missing primary recommendation"
        assert len(comparison['all_comparisons']) == 8, "Should have 8 wing types (3 traditional + 5 advanced)"
    
    print("  ✓ Wing type comparison successful")
    print("  - Tested purposes: general, speed, efficiency, aerobatic, fun")
    return True


def test_edge_cases():
    """Test edge cases and parameter validation"""
    print("\nTesting Edge Cases...")
    
    # Test straight wing with rectangular planform (taper=1.0)
    design = straight_wing_design(
        wingspan=1000,
        chord=180,
        taper_ratio=1.0,
        dihedral=0
    )
    assert design['geometry']['tip_chord_mm'] == design['geometry']['root_chord_mm'], "Rectangular wing"
    
    # Test with minimum values
    design = delta_wing_design(
        root_chord=100,
        wingspan=300,
        sweep_angle=30,
        thickness_ratio=0.05
    )
    assert design['geometry']['area_mm2'] > 0, "Should handle small values"
    
    # Test with large values
    design = flying_wing_design(
        center_chord=500,
        wingspan=2000,
        sweep_angle=35,
        wing_twist=-3
    )
    assert design['geometry']['area_mm2'] > 0, "Should handle large values"
    
    # Test oblique wing at 0 degrees (straight)
    design = oblique_wing_design(
        wingspan=1000,
        chord=180,
        sweep_angle=0,
        pivot_position=0.5
    )
    assert design['asymmetry']['lift_asymmetry_percent'] == 0, "No asymmetry at 0 degrees"
    
    print("  ✓ Edge cases handled correctly")
    return True


def main():
    """Run all verification tests"""
    print("\n" + "=" * 70)
    print("  WING TYPES MODULE VERIFICATION")
    print("=" * 70 + "\n")
    
    tests = [
        ("Straight Wing", test_straight_wing),
        ("Backward Swept Wing", test_backward_swept_wing),
        ("Forward Swept Wing", test_forward_swept_wing),
        ("Delta Wing", test_delta_wing),
        ("Flying Wing", test_flying_wing),
        ("Canard", test_canard),
        ("Oblique Wing", test_oblique_wing),
        ("Flying Pancake", test_flying_pancake),
        ("Comparison", test_comparison),
        ("Edge Cases", test_edge_cases),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"  ✗ {name} test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {name} test error: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"  VERIFICATION COMPLETE: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✓ All wing types module tests passed!")
        print("\nYou can now:")
        print("  1. Run analysis: PYTHONPATH=. python examples/wing_types_analysis.py")
        print("  2. Use in code: from fixed_wing.wing_types import delta_wing_design")
        print("  3. Read docs: course/wing-types-guide.md")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
