"""
3D Printer Export Examples
This script demonstrates how to convert 3D diagrams to 3D printer formats (STL).
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from da3 import DA3


def example_1_basic_stl_export():
    """Example 1: Basic STL export for single shapes"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic STL Export")
    print("=" * 70)
    
    da3 = DA3(output_dir='./stl_examples')
    
    print("\n1. Exporting a surface mesh...")
    da3.export_surface_stl('example_surface.stl')
    
    print("\n2. Exporting a torus (donut shape)...")
    da3.export_torus_stl('example_torus.stl')
    
    print("\n3. Exporting a sphere...")
    da3.export_sphere_stl('example_sphere.stl')
    
    print("\n4. Exporting a helix tube (spring)...")
    da3.export_helix_stl('example_helix.stl')
    
    print("\n✓ Basic STL export complete!")


def example_2_batch_export():
    """Example 2: Batch export all shapes"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Batch Export All Shapes")
    print("=" * 70)
    
    da3 = DA3(output_dir='./stl_batch')
    
    print("\nExporting all shapes at once...")
    da3.export_all_stl()
    
    print("\n✓ Batch export complete!")


def example_3_combined_workflow():
    """Example 3: Create both visualizations and STL files"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Combined Workflow (Visualization + STL)")
    print("=" * 70)
    
    da3 = DA3(output_dir='./combined_output')
    
    print("\nStep 1: Creating visualization plots...")
    da3.surface_plot(filename='viz_surface.png')
    da3.scatter_plot(filename='viz_scatter.png')
    
    print("\nStep 2: Exporting to STL format...")
    da3.export_surface_stl('model_surface.stl')
    da3.export_sphere_stl('model_sphere.stl')
    
    print("\nStep 3: Printing summary...")
    da3.print_summary()
    
    print("\n✓ Combined workflow complete!")


def example_4_direct_module_usage():
    """Example 4: Using the 3d_printer_export module directly"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Direct Module Usage")
    print("=" * 70)
    
    # Load module dynamically
    import importlib.util
    module_path = os.path.join(current_dir, '3d_printer_export.py')
    spec = importlib.util.spec_from_file_location('printer_export', module_path)
    printer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(printer_module)
    
    print("\nCreating STL files directly from module...")
    
    # Custom parameters
    printer_module.create_surface_mesh_stl(filename='custom_surface.stl')
    printer_module.create_torus_stl(filename='custom_torus.stl', R=4, r=1.5)
    printer_module.create_sphere_stl(filename='custom_sphere.stl', radius=3)
    printer_module.create_helix_tube_stl(filename='custom_helix.stl', 
                                         tube_radius=0.3, 
                                         helix_radius=2, 
                                         height=8)
    
    print("\n✓ Direct module usage complete!")


def print_3d_printing_instructions():
    """Print instructions for using the STL files"""
    print("\n" + "=" * 70)
    print("HOW TO USE STL FILES FOR 3D PRINTING")
    print("=" * 70)
    
    print("""
The generated STL files can be used with 3D printers. Follow these steps:

1. OPEN THE STL FILE:
   - Use slicing software like Cura, PrusaSlicer, or Simplify3D
   - File > Open > Select your .stl file

2. CONFIGURE PRINT SETTINGS:
   - Layer height: 0.1-0.2mm (finer for smooth surfaces)
   - Infill: 10-20% (depending on strength needs)
   - Support: Add if overhangs > 45 degrees
   - Print speed: 50-60 mm/s

3. SLICE AND EXPORT:
   - Click "Slice" to generate G-code
   - Save to SD card or send directly to printer

4. 3D PRINT:
   - Load filament (PLA recommended for beginners)
   - Heat bed and nozzle to appropriate temperatures
   - Start print and monitor first layer adhesion

RECOMMENDED SOFTWARE:
   • Cura (free, beginner-friendly)
   • PrusaSlicer (free, advanced features)
   • Simplify3D (paid, professional)
   • MeshLab (free, for viewing/editing STL files)

FILE FORMATS:
   • STL - Standard format for 3D printing
   • All files use centimeter (cm) units
   • Manifold meshes (watertight, printable)

TIPS:
   • Scale models if needed in slicing software
   • Check for errors using "Repair" tools in slicer
   • Test small objects first before large prints
    """)


def main():
    """Run all examples"""
    print("=" * 70)
    print("3D PRINTER EXPORT EXAMPLES FOR DA3 LIBRARY")
    print("=" * 70)
    print("\nThis script demonstrates how to convert 3D diagrams")
    print("to STL format for 3D printing.\n")
    
    # Run examples
    example_1_basic_stl_export()
    example_2_batch_export()
    example_3_combined_workflow()
    example_4_direct_module_usage()
    
    # Print instructions
    print_3d_printing_instructions()
    
    print("\n" + "=" * 70)
    print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nCheck the following directories for output:")
    print("  • ./stl_examples/")
    print("  • ./stl_batch/")
    print("  • ./combined_output/")
    print("=" * 70)


if __name__ == "__main__":
    main()
