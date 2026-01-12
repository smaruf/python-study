"""
Fixed-Wing Aircraft Load Analysis Example

Demonstrates flight load calculations for a small electric UAV.
"""

from fixed_wing.loads import (
    lift_per_wing,
    wing_loading,
    estimate_cruise_speed,
    tail_volume_coefficient,
    calculate_flight_loads
)
from fixed_wing.spar import (
    wing_bending_load,
    recommend_spar_type
)


def main():
    print("=" * 70)
    print("FIXED-WING AIRCRAFT LOAD ANALYSIS")
    print("=" * 70)
    
    # Design parameters for small electric UAV
    print("\n--- UAV Design Parameters ---")
    
    wingspan = 1200  # mm
    chord = 180      # mm
    weight = 1400    # grams (1.4 kg)
    
    print(f"Wingspan: {wingspan}mm ({wingspan/10:.1f}cm)")
    print(f"Wing chord: {chord}mm")
    print(f"Takeoff weight: {weight}g ({weight/1000:.2f}kg)")
    
    # Calculate comprehensive loads
    print("\n" + "=" * 70)
    print("FLIGHT LOAD ANALYSIS")
    print("=" * 70)
    
    loads = calculate_flight_loads(
        weight=weight,
        wingspan=wingspan,
        chord=chord
    )
    
    print(f"\nWing area: {loads['wing_area_cm2']:.1f} cm²")
    print(f"Wing loading: {loads['wing_loading_g_cm2']:.4f} g/cm²")
    
    print("\n--- Lift Calculations (with 2× safety factor) ---")
    print(f"Total lift required: {loads['lift_total_g']:.1f}g")
    print(f"Lift per wing: {loads['lift_per_wing_g']:.1f}g")
    
    print("\n--- Performance Estimate ---")
    print(f"Estimated cruise speed: {loads['estimated_cruise_speed_ms']:.1f} m/s")
    print(f"                       = {loads['estimated_cruise_speed_ms'] * 3.6:.1f} km/h")
    
    # Wing loading classification
    print("\n--- Wing Loading Classification ---")
    wl = loads['wing_loading_g_cm2']
    if wl < 0.03:
        print("✓ VERY LOW - Excellent glider characteristics")
    elif wl < 0.05:
        print("✓ LOW - Good for sport flying and FPV")
    elif wl < 0.08:
        print("⚠ MODERATE - Fast cruise, needs speed to fly")
    else:
        print("⚠ HIGH - Racing plane, high stall speed")
    
    # Spar analysis
    print("\n" + "=" * 70)
    print("SPAR DESIGN ANALYSIS")
    print("=" * 70)
    
    bending = wing_bending_load(weight, wingspan)
    print(f"\nWing bending moment: {bending:.1f} g·mm")
    print(f"                   = {bending/1000:.2f} g·m")
    
    spar_recommendation = recommend_spar_type(wingspan, weight)
    print("\n--- Spar Recommendation ---")
    print(f"Type: {spar_recommendation['type']}")
    print(f"Notes: {spar_recommendation['notes']}")
    print(f"Alternative: {spar_recommendation['alternative']}")
    
    # Tail design
    print("\n" + "=" * 70)
    print("TAIL DESIGN RECOMMENDATIONS")
    print("=" * 70)
    
    print(f"\nRecommended tail moment arm: {loads['recommended_tail_arm_mm']:.0f}mm")
    print(f"Recommended tail area: {loads['recommended_tail_area_mm2']/100:.1f} cm²")
    
    # Calculate tail volume coefficient for recommended design
    tail_chord = 100  # mm
    tail_span = loads['recommended_tail_area_mm2'] / tail_chord
    
    tail_vol_coef = tail_volume_coefficient(
        tail_area=loads['recommended_tail_area_mm2'],
        tail_arm=loads['recommended_tail_arm_mm'],
        wing_area=loads['wing_area_mm2'],
        wing_chord=chord
    )
    
    print(f"\nHorizontal tail volume coefficient: {tail_vol_coef:.2f}")
    print("\n--- Stability Assessment ---")
    if tail_vol_coef >= 0.6:
        print("✓ VERY STABLE - Trainer aircraft")
    elif tail_vol_coef >= 0.4:
        print("✓ BALANCED - Sport aircraft (recommended)")
    else:
        print("⚠ AGILE - Aerobatic aircraft (less stable)")
    
    # Weight impact
    print("\n" + "=" * 70)
    print("TAIL WEIGHT CONSIDERATIONS")
    print("=" * 70)
    
    print("\n⚠ CRITICAL: Tail weight heavily affects CG position")
    print("   - Keep tail components as light as possible")
    print("   - Use carbon tube for tail boom (not printed)")
    print("   - Print ribs only, use foam/film covering")
    print("   - Excessive tail weight moves CG backward (unstable)")
    
    # Build recommendations
    print("\n" + "=" * 70)
    print("BUILD RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n✓ Wing Construction:")
    print("  - Print ribs only (6mm thick)")
    print(f"  - Use {spar_recommendation['type']}")
    print("  - Cover with foam or balsa + heat-shrink film")
    print("  - Full printed wing = TOO HEAVY & FLEXIBLE")
    
    print("\n✓ Fuselage:")
    print("  - Nylon or PETG, 2mm wall thickness")
    print("  - Reinforce wing mount area (double thickness)")
    print("  - Keep it modular for easy printing")
    
    print("\n✓ Tail:")
    print("  - Carbon tube boom (8mm recommended)")
    print("  - Print mounting brackets only")
    print("  - Lightweight stabilizers (foam core + film)")
    
    print("\n⚠ FAILURE WARNING:")
    print("  90% of fixed-wing failures occur at:")
    print("  → Wing-fuselage connection")
    print("  → Use metal or carbon spar pass-through")
    print("  → No sharp transitions in structure")
    print("  → Test with 2× expected loads before flight")
    
    print("\n" + "=" * 70)
    print("READY TO BUILD!")
    print("=" * 70)
    print("\nThis UAV design is realistic and achievable.")
    print("Follow hybrid construction (print + traditional) for best results.")
    print("\n")


if __name__ == "__main__":
    main()
