"""
Physical Output Examples - 2D Plotter and 3D Printer
This script demonstrates converting visual data to physical formats.
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from da3 import DA3


def example_workflow():
    """Complete workflow: Visual -> 2D Plotter -> 3D Printer"""
    print("=" * 70)
    print("COMPLETE WORKFLOW: Visual Data to Physical Output")
    print("=" * 70)
    
    da3 = DA3(output_dir='./physical_output')
    
    print("\nStep 1: Create visual data (PNG images)")
    print("-" * 70)
    da3.surface_plot(filename='visual_surface.png')
    da3.scatter_plot(filename='visual_scatter.png')
    
    print("\nStep 2: Export to 2D plotter format (SVG)")
    print("-" * 70)
    da3.export_contour_svg(filename='plotter_contour.svg', plot_type='contour')
    da3.export_parametric_curve_svg(filename='plotter_spiral.svg', curve_type='spiral')
    da3.export_pattern_svg(filename='plotter_hexagon.svg', pattern_type='hexagon')
    
    print("\nStep 3: Export to 3D printer format (STL)")
    print("-" * 70)
    try:
        da3.export_surface_stl(filename='printer_surface.stl')
        da3.export_torus_stl(filename='printer_torus.stl')
    except:
        print("Note: 3D printer export requires numpy-stl library")
    
    print("\nStep 4: Summary")
    print("-" * 70)
    da3.print_summary()
    
    print("\n" + "=" * 70)
    print("✓ Complete workflow finished!")
    print("=" * 70)


if __name__ == "__main__":
    example_workflow()
