"""
Wing Types Analysis Example

This script demonstrates the analysis and design of various wing
configurations including:

Traditional Wing Types:
- Straight wings (rectangular/tapered)
- Backward swept wings
- Forward swept wings

Advanced Wing Types:
- Delta wings
- Flying wings
- Canard configurations
- Oblique wings
- Flying pancake (circular wing)

Run with: PYTHONPATH=. python examples/wing_types_analysis.py
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


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_dict(data, indent=0):
    """Print dictionary in a readable format."""
    for key, value in data.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_dict(value, indent + 1)
        elif isinstance(value, list):
            print("  " * indent + f"{key}:")
            for item in value:
                print("  " * (indent + 1) + f"- {item}")
        elif isinstance(value, float):
            print("  " * indent + f"{key}: {value:.2f}")
        else:
            print("  " * indent + f"{key}: {value}")


def analyze_straight_wing():
    """Analyze straight wing design."""
    print_section("STRAIGHT WING ANALYSIS")
    
    # Design parameters for a classic trainer
    design = straight_wing_design(
        wingspan=1200,
        chord=200,
        taper_ratio=0.7,
        dihedral=3,
        thickness_ratio=0.12
    )
    
    print_dict(design)
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("   - Most common and traditional wing type")
    print("   - Excellent low-speed characteristics")
    print("   - Simple to design and build")
    print("   - Predictable and stable flight")
    print("   - Wide CG range (forgiving)")
    print("   - Best for trainers and sport aircraft")


def analyze_backward_swept_wing():
    """Analyze backward swept wing design."""
    print_section("BACKWARD SWEPT WING ANALYSIS")
    
    # Design parameters for a high-speed sport plane
    design = backward_swept_wing_design(
        wingspan=1200,
        chord=200,
        sweep_angle=25,
        taper_ratio=0.6,
        thickness_ratio=0.10
    )
    
    print_dict(design)
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("   - Excellent high-speed performance")
    print("   - CRITICAL: Must have washout to prevent tip stalling")
    print("   - 15% speed increase vs straight wing")
    print("   - More complex to build (requires accurate twist)")
    print("   - Wing fences or vortex generators recommended")
    print("   - Best for high-speed sport and scale jets")


def analyze_forward_swept_wing():
    """Analyze forward swept wing design."""
    print_section("FORWARD SWEPT WING ANALYSIS")
    
    # Design parameters for an advanced experimental design
    design = forward_swept_wing_design(
        wingspan=1200,
        chord=200,
        sweep_angle=25,
        taper_ratio=0.6,
        thickness_ratio=0.10
    )
    
    print_dict(design)
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("   - ADVANCED DESIGN - Not for beginners!")
    print("   - Excellent stall safety (root stalls first)")
    print("   - 25% better maneuverability than backward sweep")
    print("   - CRITICAL: Requires carbon fiber structure (very stiff)")
    print("   - Expensive to build, structural testing essential")
    print("   - Famous examples: Grumman X-29, Su-47")


def analyze_delta_wing():
    """Analyze delta wing design."""
    print_section("DELTA WING ANALYSIS")
    
    # Design parameters for a small delta wing UAV
    design = delta_wing_design(
        root_chord=400,
        wingspan=1000,
        sweep_angle=45,
        thickness_ratio=0.08
    )
    
    print_dict(design)
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("   - Excellent for high-speed flight and aerobatics")
    print("   - No tail needed (elevons control pitch and roll)")
    print("   - Leading edge vortices enhance lift at high angles")
    print("   - Requires precise CG placement (25-30% MAC)")
    print("   - Great structural efficiency")


def analyze_flying_wing():
    """Analyze flying wing design."""
    print_section("FLYING WING ANALYSIS")
    
    # Design parameters for a FPV flying wing
    design = flying_wing_design(
        center_chord=350,
        wingspan=1200,
        sweep_angle=25,
        wing_twist=-2
    )
    
    print_dict(design)
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("   - Highest aerodynamic efficiency (20% better L/D)")
    print("   - CRITICAL CG placement (±5mm tolerance!)")
    print("   - Requires reflex airfoil (Eppler E325, MH-45)")
    print("   - Best for long-range FPV and soaring")
    print("   - Washout prevents tip stalling")


def analyze_canard():
    """Analyze canard configuration."""
    print_section("CANARD CONFIGURATION ANALYSIS")
    
    # Design parameters for a canard UAV
    design = canard_design(
        main_wing_chord=200,
        main_wingspan=1000,
        canard_chord=80,
        canard_span=400,
        canard_position=150
    )
    
    print_dict(design)
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("   - Inherently stall-resistant (canard stalls first)")
    print("   - Both surfaces generate lift (5% more efficient)")
    print("   - Excellent visibility and CG range")
    print("   - Pusher prop configuration recommended")
    print("   - Great for trainers and general flying")


def analyze_oblique_wing():
    """Analyze oblique wing design."""
    print_section("OBLIQUE WING ANALYSIS")
    
    # Analyze at different sweep angles
    print("⚙️  Analyzing oblique wing at different sweep angles:\n")
    
    for sweep in [0, 20, 30, 45]:
        design = oblique_wing_design(
            wingspan=1000,
            chord=180,
            sweep_angle=sweep,
            pivot_position=0.5
        )
        
        print(f"\n📐 Sweep Angle: {sweep}°")
        print(f"   Effective Span: {design['geometry']['effective_span_mm']:.1f} mm")
        print(f"   Aspect Ratio: {design['geometry']['aspect_ratio']:.2f}")
        print(f"   Lift Asymmetry: {design['asymmetry']['lift_asymmetry_percent']:.1f}%")
        
        perf_key = f"{sweep}_deg"
        if perf_key in design['performance_envelope']:
            perf = design['performance_envelope'][perf_key]
            print(f"   Speed Factor: {perf['speed_factor']:.2f}×")
            print(f"   Efficiency: {perf['efficiency']:.0%}")
            print(f"   Best For: {perf['description']}")
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("   - EXPERIMENTAL DESIGN - High complexity!")
    print("   - Variable sweep optimizes for speed range")
    print("   - Asymmetric configuration requires careful trim")
    print("   - Strong pivot mechanism essential")
    print("   - Fly-by-wire control recommended")
    print("   - Start with fixed oblique angle if new to concept")


def analyze_flying_pancake():
    """Analyze flying pancake design."""
    print_section("FLYING PANCAKE ANALYSIS")
    
    # Design parameters for a fun flying pancake
    design = flying_pancake_design(
        diameter=600,
        thickness_ratio=0.12,
        center_cutout=120
    )
    
    print_dict(design)
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("   - FUN EXPERIMENTAL DESIGN!")
    print("   - Based on historic Vought V-173 'Flying Pancake'")
    print("   - Very stable and docile handling")
    print("   - High drag = low efficiency (need more power)")
    print("   - Great conversation starter and demo aircraft")
    print("   - Can fly at extreme angles of attack")
    print("   - Consider twin tip motors for authentic look")


def compare_all_wings():
    """Compare all wing types and provide recommendations."""
    print_section("WING TYPE COMPARISON & RECOMMENDATIONS")
    
    purposes = ["general", "speed", "efficiency", "aerobatic", "fun"]
    
    for purpose in purposes:
        comparison = compare_wing_types(
            weight=1400,
            target_speed=15,
            purpose=purpose
        )
        
        print(f"\n🎯 For {purpose.upper()} flying:")
        rec = comparison['recommendation']
        print(f"   Primary Choice: {rec['primary']}")
        print(f"   Alternative: {rec['alternative']}")
        print(f"   Reason: {rec['reason']}")
    
    print("\n\n📊 DETAILED COMPARISON:")
    print("\n{:<20} {:<12} {:<20} {:<12} {:<12} {:<15} {:<30}".format(
        "Wing Type", "Complexity", "Stability", "Speed", "Efficiency", "Build Diff", "Best For"
    ))
    print("-" * 140)
    
    comparison = compare_wing_types()
    for wing_type, specs in comparison['all_comparisons'].items():
        print("{:<20} {:<12} {:<20} {:<12} {:<12} {:<15} {:<30}".format(
            wing_type,
            specs['complexity'],
            specs['stability'],
            specs['speed'],
            specs['efficiency'],
            specs['build_difficulty'],
            specs['best_for']
        ))
    
    print("\n\n💡 GENERAL NOTES:")
    for note in comparison['notes']:
        print(f"   - {note}")


def main():
    """Run all wing type analyses."""
    print("\n" + "=" * 80)
    print("  WING TYPES - DESIGN & ANALYSIS")
    print("  Traditional and advanced wing configurations")
    print("=" * 80)
    
    # Analyze traditional wing types first
    print("\n" + "█" * 80)
    print("  TRADITIONAL WING TYPES")
    print("█" * 80)
    analyze_straight_wing()
    analyze_backward_swept_wing()
    analyze_forward_swept_wing()
    
    # Analyze advanced wing types
    print("\n" + "█" * 80)
    print("  ADVANCED WING TYPES")
    print("█" * 80)
    analyze_delta_wing()
    analyze_flying_wing()
    analyze_canard()
    analyze_oblique_wing()
    analyze_flying_pancake()
    
    # Compare all types
    compare_all_wings()
    
    print("\n\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n📚 NEXT STEPS:")
    print("   1. Choose a wing type based on your goals and skill level")
    print("   2. BEGINNERS: Start with Straight Wing (easiest)")
    print("   3. INTERMEDIATE: Try Backward Swept or Delta Wing")
    print("   4. ADVANCED: Experiment with Forward Swept or Flying Wing")
    print("   5. Review the theoretical background and notes")
    print("   6. Use the construction principles as a guide")
    print("\n💻 GENERATE MODELS:")
    print("   - Modify examples/generate_fixed_wing.py to add wing type ribs")
    print("   - Use generate_delta_wing_ribs() or generate_flying_pancake_ribs()")
    print("   - Requires CadQuery for STL generation")
    print("\n✈️  Happy building!\n")


if __name__ == "__main__":
    main()
