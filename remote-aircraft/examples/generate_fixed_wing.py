"""
Generate Fixed-Wing Components

Example script to generate STL files for fixed-wing aircraft components.
Requires CadQuery to be installed.
"""

import os
import sys

try:
    import cadquery as cq
    from fixed_wing.wing_rib import wing_rib, wing_rib_simple
    from fixed_wing.fuselage import (
        fuselage_section,
        fuselage_bulkhead,
        wing_mount_plate
    )
    from fixed_wing.tail import (
        horizontal_stabilizer,
        vertical_stabilizer,
        tail_boom_mount
    )
except ImportError as e:
    print(f"Error: {e}")
    print("\nCadQuery not installed or import error.")
    print("\nTo install CadQuery:")
    print("  conda install -c conda-forge -c cadquery cadquery")
    print("  or download CQ-Editor from:")
    print("  https://github.com/CadQuery/CQ-editor/releases")
    sys.exit(1)


def main():
    # Create output directory
    output_dir = "output/fixed_wing"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING FIXED-WING AIRCRAFT COMPONENTS")
    print("=" * 70)
    
    # Generate wing ribs
    print("\n--- Wing Ribs ---")
    
    print("  Generating: wing_rib_clark_y.stl")
    rib = wing_rib(chord=180, thickness=6, spar_slot=10)
    cq.exporters.export(rib, f"{output_dir}/wing_rib_clark_y.stl")
    
    print("  Generating: wing_rib_simple.stl")
    rib_simple = wing_rib_simple(chord=180, thickness=6, spar_slot=10)
    cq.exporters.export(rib_simple, f"{output_dir}/wing_rib_simple.stl")
    
    # Generate multiple rib sizes
    for chord_size in [150, 180, 200]:
        filename = f"wing_rib_{chord_size}mm.stl"
        print(f"  Generating: {filename}")
        rib = wing_rib(chord=chord_size, thickness=6, spar_slot=10)
        cq.exporters.export(rib, f"{output_dir}/{filename}")
    
    # Generate fuselage components
    print("\n--- Fuselage Components ---")
    
    print("  Generating: fuselage_section_standard.stl")
    fuse = fuselage_section(radius=35, length=80, wall=2, reinforced=False)
    cq.exporters.export(fuse, f"{output_dir}/fuselage_section_standard.stl")
    
    print("  Generating: fuselage_section_reinforced.stl")
    fuse_reinforced = fuselage_section(radius=35, length=80, wall=2, reinforced=True)
    cq.exporters.export(fuse_reinforced, f"{output_dir}/fuselage_section_reinforced.stl")
    
    print("  Generating: fuselage_bulkhead.stl")
    bulkhead = fuselage_bulkhead(radius=35, thickness=4, center_hole=10)
    cq.exporters.export(bulkhead, f"{output_dir}/fuselage_bulkhead.stl")
    
    print("  Generating: wing_mount_plate.stl")
    mount = wing_mount_plate(width=60, height=40, thickness=6, bolt_spacing=20)
    cq.exporters.export(mount, f"{output_dir}/wing_mount_plate.stl")
    
    # Generate tail components
    print("\n--- Tail Components ---")
    
    print("  Generating: horizontal_stabilizer.stl")
    h_stab = horizontal_stabilizer(span=400, chord=100, thickness=4)
    cq.exporters.export(h_stab, f"{output_dir}/horizontal_stabilizer.stl")
    
    print("  Generating: vertical_stabilizer.stl")
    v_stab = vertical_stabilizer(height=120, chord=100, thickness=4)
    cq.exporters.export(v_stab, f"{output_dir}/vertical_stabilizer.stl")
    
    print("  Generating: tail_boom_mount.stl")
    boom_mount = tail_boom_mount(boom_diameter=8, mount_length=30, wall_thickness=3)
    cq.exporters.export(boom_mount, f"{output_dir}/tail_boom_mount.stl")
    
    # Summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nSTL files saved in: {output_dir}/")
    print("\n--- Generated Components ---")
    print("  Wing Ribs:")
    print("    - Clark-Y airfoil (realistic)")
    print("    - Simple symmetric (easier to print)")
    print("    - Multiple chord sizes (150, 180, 200mm)")
    print("\n  Fuselage:")
    print("    - Standard sections (modular)")
    print("    - Reinforced sections (wing mount areas)")
    print("    - Bulkheads (internal support)")
    print("    - Wing mount plate (critical connection)")
    print("\n  Tail:")
    print("    - Horizontal stabilizer")
    print("    - Vertical stabilizer")
    print("    - Tail boom mount (for carbon tube)")
    
    print("\n--- Printing Recommendations ---")
    print("  Wing ribs: PLA/PETG, 30% infill, 0.2mm layers")
    print("  Fuselage: Nylon/PETG, 30% infill, 0.2mm layers")
    print("  Mounts: Nylon/CF-Nylon, 50% infill, 0.28mm layers")
    print("  Tail mounts: Nylon, 40% infill, 0.2mm layers")
    
    print("\n--- Assembly Notes ---")
    print("  ⚠ Use carbon tube spar (6-8mm) - NOT printed")
    print("  ⚠ Cover wings with foam/balsa + heat-shrink")
    print("  ⚠ Full printed wings = too heavy and flexible")
    print("  ⚠ Reinforce wing-fuselage connection heavily")
    print("  ⚠ Use carbon tube for tail boom (8mm)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
