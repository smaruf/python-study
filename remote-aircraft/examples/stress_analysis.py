"""
Stress analysis for quad arms under load

Calculate bending stress on arms during crashes or hard maneuvers.
"""

from analysis.stress import bending_stress
import math

def main():
    print("=" * 60)
    print("ARM STRESS ANALYSIS")
    print("=" * 60)
    
    # Arm specifications
    arm_length = 150  # mm
    arm_width = 16    # mm
    arm_height = 12   # mm
    
    # Rectangular beam moment of inertia: I = (width * height³) / 12
    inertia = (arm_width * (arm_height ** 3)) / 12
    
    print(f"\nArm Specifications:")
    print(f"  Length: {arm_length}mm")
    print(f"  Width: {arm_width}mm")
    print(f"  Height: {arm_height}mm")
    print(f"  Moment of Inertia: {inertia:.1f} mm⁴")
    
    # Test scenarios
    scenarios = {
        "Normal Flight (1.5g)": {
            "force": 150,  # grams (assuming 100g motor + prop)
            "description": "Typical flight maneuvers"
        },
        "Hard Turn (3g)": {
            "force": 300,
            "description": "Aggressive flying"
        },
        "Crash (5g)": {
            "force": 500,
            "description": "Minor impact"
        },
        "Hard Crash (10g)": {
            "force": 1000,
            "description": "Major impact"
        },
    }
    
    print("\n" + "=" * 60)
    print("STRESS ANALYSIS UNDER VARIOUS LOADS")
    print("=" * 60)
    
    # Material properties (approximate)
    materials = {
        "PLA": 50,      # MPa tensile strength
        "PETG": 50,     # MPa
        "Nylon": 75,    # MPa
        "CF-Nylon": 100 # MPa
    }
    
    print()
    for scenario_name, scenario in scenarios.items():
        force = scenario["force"]
        description = scenario["description"]
        
        # Calculate bending stress at the base of the arm
        # Using simplified beam formula: σ = (M * c) / I
        # where M = force * length, c = height/2
        
        # For our bending_stress function: stress = (force * length) / inertia
        stress = bending_stress(force, arm_length, inertia)
        
        # Convert to MPa (assuming force in grams, dimensions in mm)
        # 1 g/mm² = 0.00981 MPa (approximately)
        stress_mpa = stress * 0.00981
        
        print(f"--- {scenario_name} ---")
        print(f"  Force: {force}g ({description})")
        print(f"  Bending stress: {stress:.2f} g/mm²")
        print(f"  Bending stress: {stress_mpa:.2f} MPa")
        
        # Check against material strengths
        print(f"\n  Material Safety Factors:")
        for material, strength in materials.items():
            safety_factor = strength / stress_mpa if stress_mpa > 0 else float('inf')
            status = "✓ SAFE" if safety_factor >= 2 else ("⚠ MARGINAL" if safety_factor >= 1 else "✗ FAIL")
            print(f"    {material:<12} SF = {safety_factor:>5.2f}  {status}")
        
        print()
    
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("""
1. For normal flying (PLA acceptable, but not recommended)
   - Use PETG minimum for outdoor use
   - Better UV resistance than PLA

2. For freestyle/aggressive flying
   - Use Nylon minimum
   - Better impact resistance
   - More flexible (absorbs shock)

3. For racing/competition
   - Use CF-Nylon for maximum strength
   - Highest strength-to-weight ratio
   - Most expensive but worth it

4. Design considerations:
   - Print arms FLAT (layers parallel to ground)
   - Use 30%+ infill (Gyroid pattern)
   - 3-4 perimeters for strength
   - Fillet edges to reduce stress concentration
    """)
    
    print("\n" + "=" * 60)
    print("WEIGHT OPTIMIZATION")
    print("=" * 60)
    
    # Solid vs hollow comparison
    solid_volume = arm_length * arm_width * arm_height
    wall_thickness = 3  # mm
    hollow_volume = solid_volume - (arm_length * (arm_width - 2*wall_thickness) * (arm_height - 2*wall_thickness))
    
    weight_reduction = (1 - hollow_volume / solid_volume) * 100
    
    # Approximate strength reduction (hollow beam)
    # Very simplified - actual calculation is complex
    I_outer = (arm_width * (arm_height ** 3)) / 12
    I_inner = ((arm_width - 2*wall_thickness) * ((arm_height - 2*wall_thickness) ** 3)) / 12
    I_hollow = I_outer - I_inner
    
    strength_reduction = (1 - I_hollow / I_outer) * 100
    
    print(f"\nSolid Arm:")
    print(f"  Volume: {solid_volume:.0f} mm³")
    print(f"  Moment of Inertia: {I_outer:.1f} mm⁴")
    
    print(f"\nHollow Arm ({wall_thickness}mm wall):")
    print(f"  Volume: {hollow_volume:.0f} mm³")
    print(f"  Moment of Inertia: {I_hollow:.1f} mm⁴")
    
    print(f"\nComparison:")
    print(f"  Weight reduction: {weight_reduction:.1f}%")
    print(f"  Strength reduction: {strength_reduction:.1f}%")
    print(f"  ✓ Good tradeoff for weight savings!")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
