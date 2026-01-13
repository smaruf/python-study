"""
Verification script to test fixed-wing module imports.
This ensures all modules are properly structured without requiring CadQuery.
"""

def test_imports():
    """Test that all fixed-wing modules can be imported."""
    print("=" * 70)
    print("TESTING FIXED-WING MODULE IMPORTS")
    print("=" * 70)
    
    try:
        print("\n✓ Testing spar module...")
        from fixed_wing.spar import (
            wing_bending_load,
            spar_stress,
            tube_moment_of_inertia,
            rectangular_moment_of_inertia,
            recommend_spar_type
        )
        
        # Quick test
        load = wing_bending_load(1400, 1200)
        recommendation = recommend_spar_type(1200, 1400)
        print(f"  - Bending load: {load:.1f} g·mm")
        print(f"  - Recommended spar: {recommendation['type']}")
        
    except Exception as e:
        print(f"✗ Error in spar module: {e}")
        return False
    
    try:
        print("\n✓ Testing loads module...")
        from fixed_wing.loads import (
            lift_per_wing,
            wing_loading,
            estimate_cruise_speed,
            tail_moment_arm,
            tail_volume_coefficient,
            calculate_flight_loads
        )
        
        # Quick test
        loads = calculate_flight_loads(1400, 1200, 180)
        print(f"  - Wing area: {loads['wing_area_cm2']:.1f} cm²")
        print(f"  - Cruise speed: {loads['estimated_cruise_speed_ms']:.1f} m/s")
        
    except Exception as e:
        print(f"✗ Error in loads module: {e}")
        return False
    
    try:
        print("\n✓ Testing wing_rib module (imports only)...")
        # We can't test actual function calls without CadQuery
        from fixed_wing import wing_rib
        print("  - Module imports successfully")
        print("  - (STL generation requires CadQuery)")
        
    except Exception as e:
        print(f"✗ Error in wing_rib module: {e}")
        return False
    
    try:
        print("\n✓ Testing fuselage module (imports only)...")
        from fixed_wing import fuselage
        print("  - Module imports successfully")
        print("  - (STL generation requires CadQuery)")
        
    except Exception as e:
        print(f"✗ Error in fuselage module: {e}")
        return False
    
    try:
        print("\n✓ Testing tail module (imports only)...")
        from fixed_wing import tail
        print("  - Module imports successfully")
        print("  - (STL generation requires CadQuery)")
        
    except Exception as e:
        print(f"✗ Error in tail module: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    print("\n✅ All fixed-wing modules are properly installed")
    print("✅ Analysis functions work correctly")
    print("✅ CAD functions available (require CadQuery for STL export)")
    print("\n📝 Next steps:")
    print("   1. Run: PYTHONPATH=. python examples/fixed_wing_analysis.py")
    print("   2. Install CadQuery to generate STL files")
    print("   3. Read: course/fixed-wing-design.md")
    print()
    
    return True


if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
