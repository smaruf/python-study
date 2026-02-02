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
    delta_wing_design,
    flying_wing_design,
    canard_design,
    oblique_wing_design,
    flying_pancake_design,
    compare_wing_types
)


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
        assert len(comparison['all_comparisons']) == 5, "Should have 5 wing types"
    
    print("  ✓ Wing type comparison successful")
    print("  - Tested purposes: general, speed, efficiency, aerobatic, fun")
    return True


def test_edge_cases():
    """Test edge cases and parameter validation"""
    print("\nTesting Edge Cases...")
    
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
        print("  3. Read docs: course/advanced-wing-types.md")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
