"""
Generate custom motor mounts for different motor sizes

This example shows how to create parametric parts for various applications.
"""

try:
    import cadquery as cq
    from parts.motor_mount import motor_mount
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False
    print("Warning: CadQuery not available. Will show calculations only.")

def main():
    print("=" * 60)
    print("CUSTOM MOTOR MOUNT GENERATOR")
    print("=" * 60)
    
    # Define motor specifications
    motor_specs = {
        "1507 (Tiny Whoop)": {
            "diameter": 15,
            "thickness": 4,
            "bolt_circle": 9,
            "bolt_hole": 1.5,
            "shaft_hole": 4,
        },
        "2204 (4\" Racing)": {
            "diameter": 22,
            "thickness": 5,
            "bolt_circle": 12,
            "bolt_hole": 2,
            "shaft_hole": 5,
        },
        "2306 (5\" Freestyle)": {
            "diameter": 23,
            "thickness": 5,
            "bolt_circle": 16,
            "bolt_hole": 3,
            "shaft_hole": 6,
        },
        "2806 (7\" Long Range)": {
            "diameter": 28,
            "thickness": 6,
            "bolt_circle": 16,
            "bolt_hole": 3,
            "shaft_hole": 6,
        },
    }
    
    print("\nGenerating motor mounts for various motor sizes...")
    print()
    
    for name, specs in motor_specs.items():
        print(f"--- {name} ---")
        print(f"  Motor diameter: {specs['diameter']}mm")
        print(f"  Mount thickness: {specs['thickness']}mm")
        print(f"  Bolt circle: {specs['bolt_circle']}mm")
        print(f"  Bolt holes: M{specs['bolt_hole']}")
        print(f"  Shaft hole: {specs['shaft_hole']}mm")
        
        if CADQUERY_AVAILABLE:
            # Generate the mount
            mount = motor_mount(
                motor_diameter=specs['diameter'],
                thickness=specs['thickness'],
                bolt_circle=specs['bolt_circle'],
                bolt_hole=specs['bolt_hole'],
                shaft_hole=specs['shaft_hole'],
            )
            
            # Create filename
            filename = f"output/motor_mount_{name.split()[0]}.stl"
            
            # Export
            cq.exporters.export(mount, filename)
            print(f"  ✓ Generated: {filename}")
        else:
            print(f"  ⚠ Would generate: output/motor_mount_{name.split()[0]}.stl")
        
        print()
    
    print("=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    
    if CADQUERY_AVAILABLE:
        print("\nSTL files saved in output/ directory")
        print("Import into your slicer and print!")
    else:
        print("\nTo generate actual STL files, install CadQuery:")
        print("  conda install -c conda-forge -c cadquery cadquery")
        print("  or download CQ-Editor from:")
        print("  https://github.com/CadQuery/CQ-editor/releases")

if __name__ == "__main__":
    main()
