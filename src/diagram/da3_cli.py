#!/usr/bin/env python3
"""
DA3 Command-Line Interface
A comprehensive CLI tool for generating 3D visualizations, 2D plotter outputs, and 3D printer files.
"""

import argparse
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from da3 import DA3


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description='DA3 - Data Analytics 3D: Complete visualization and physical output tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Create a single surface plot
  %(prog)s --surface --output plots/

  # Create all visualization types
  %(prog)s --all --output plots/

  # Export to 3D printer (STL format)
  %(prog)s --export-stl --surface --output models/

  # Export to 2D plotter (SVG format)
  %(prog)s --export-svg --spiral --output plotter/

  # Complete workflow: visual + 2D + 3D
  %(prog)s --complete-workflow --output complete/

  # Custom parameters for torus
  %(prog)s --export-stl --torus --torus-R 4 --torus-r 1.5 --output custom/
        ''')
    
    # Output options
    parser.add_argument('-o', '--output', default='./output',
                       help='Output directory for generated files (default: ./output)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    
    # Visualization options (PNG)
    viz_group = parser.add_argument_group('Visualization Options (PNG)')
    viz_group.add_argument('--surface', action='store_true',
                          help='Create surface plot')
    viz_group.add_argument('--parametric-surface', action='store_true',
                          help='Create parametric surface (torus)')
    viz_group.add_argument('--scatter', action='store_true',
                          help='Create scatter plot')
    viz_group.add_argument('--cluster-scatter', action='store_true',
                          help='Create cluster scatter plot')
    viz_group.add_argument('--wireframe', action='store_true',
                          help='Create wireframe plot')
    viz_group.add_argument('--sphere', action='store_true',
                          help='Create sphere wireframe')
    viz_group.add_argument('--line', action='store_true',
                          help='Create line plot (helix)')
    viz_group.add_argument('--spiral', action='store_true',
                          help='Create spiral plot')
    viz_group.add_argument('--lissajous', action='store_true',
                          help='Create Lissajous curve')
    viz_group.add_argument('--all', action='store_true',
                          help='Create all visualization types')
    
    # 2D Plotter export options (SVG)
    svg_group = parser.add_argument_group('2D Plotter Export (SVG)')
    svg_group.add_argument('--export-svg', action='store_true',
                          help='Export to SVG format for 2D plotters')
    svg_group.add_argument('--contour', action='store_true',
                          help='Export contour plot (SVG)')
    svg_group.add_argument('--svg-spiral', action='store_true',
                          help='Export spiral curve (SVG)')
    svg_group.add_argument('--svg-lissajous', action='store_true',
                          help='Export Lissajous curve (SVG)')
    svg_group.add_argument('--pattern-grid', action='store_true',
                          help='Export grid pattern (SVG)')
    svg_group.add_argument('--pattern-hexagon', action='store_true',
                          help='Export hexagon pattern (SVG)')
    svg_group.add_argument('--text', type=str, metavar='TEXT',
                          help='Export text (SVG)')
    svg_group.add_argument('--all-svg', action='store_true',
                          help='Export all SVG patterns')
    
    # 3D Printer export options (STL)
    stl_group = parser.add_argument_group('3D Printer Export (STL)')
    stl_group.add_argument('--export-stl', action='store_true',
                          help='Export to STL format for 3D printing')
    stl_group.add_argument('--stl-surface', action='store_true',
                          help='Export surface mesh (STL)')
    stl_group.add_argument('--torus', action='store_true',
                          help='Export torus (STL)')
    stl_group.add_argument('--stl-sphere', action='store_true',
                          help='Export sphere (STL)')
    stl_group.add_argument('--helix', action='store_true',
                          help='Export helix tube (STL)')
    stl_group.add_argument('--all-stl', action='store_true',
                          help='Export all STL shapes')
    
    # Custom parameters
    param_group = parser.add_argument_group('Custom Parameters')
    param_group.add_argument('--torus-R', type=float, default=3,
                            help='Torus major radius (default: 3)')
    param_group.add_argument('--torus-r', type=float, default=1,
                            help='Torus minor radius (default: 1)')
    param_group.add_argument('--sphere-radius', type=float, default=2,
                            help='Sphere radius (default: 2)')
    param_group.add_argument('--text-size', type=int, default=72,
                            help='Text font size for SVG (default: 72)')
    
    # Workflow options
    workflow_group = parser.add_argument_group('Workflow Options')
    workflow_group.add_argument('--complete-workflow', action='store_true',
                                help='Run complete workflow: PNG + SVG + STL')
    workflow_group.add_argument('--summary', action='store_true',
                               help='Print summary of created files')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Initialize DA3
    da3 = DA3(output_dir=args.output)
    
    if args.verbose:
        print(f"DA3 CLI - Output directory: {args.output}")
        print("=" * 70)
    
    # Track if any action was performed
    action_performed = False
    
    # Handle complete workflow
    if args.complete_workflow:
        if args.verbose:
            print("\n[COMPLETE WORKFLOW]")
            print("Creating: PNG visualizations + SVG patterns + STL models")
        
        da3.surface_plot()
        da3.scatter_plot()
        
        try:
            da3.export_contour_svg()
            da3.export_parametric_curve_svg(curve_type='spiral')
        except:
            print("Note: 2D plotter export may not be available")
        
        try:
            da3.export_surface_stl()
            da3.export_torus_stl()
        except:
            print("Note: 3D printer export requires numpy-stl library")
        
        action_performed = True
    
    # Handle visualization requests
    if args.all:
        if args.verbose:
            print("\n[CREATING ALL VISUALIZATIONS]")
        da3.create_all_plots()
        action_performed = True
    else:
        if args.surface:
            if args.verbose:
                print("\n[Creating surface plot]")
            da3.surface_plot()
            action_performed = True
        
        if args.parametric_surface:
            if args.verbose:
                print("\n[Creating parametric surface]")
            da3.parametric_surface()
            action_performed = True
        
        if args.scatter:
            if args.verbose:
                print("\n[Creating scatter plot]")
            da3.scatter_plot()
            action_performed = True
        
        if args.cluster_scatter:
            if args.verbose:
                print("\n[Creating cluster scatter plot]")
            da3.cluster_scatter()
            action_performed = True
        
        if args.wireframe:
            if args.verbose:
                print("\n[Creating wireframe plot]")
            da3.wireframe_plot()
            action_performed = True
        
        if args.sphere:
            if args.verbose:
                print("\n[Creating sphere wireframe]")
            da3.sphere_wireframe()
            action_performed = True
        
        if args.line:
            if args.verbose:
                print("\n[Creating line plot]")
            da3.line_plot()
            action_performed = True
        
        if args.spiral:
            if args.verbose:
                print("\n[Creating spiral plot]")
            da3.spiral_plot()
            action_performed = True
        
        if args.lissajous:
            if args.verbose:
                print("\n[Creating Lissajous curve]")
            da3.lissajous_curve()
            action_performed = True
    
    # Handle 2D plotter (SVG) exports
    if args.export_svg or args.all_svg:
        try:
            if args.all_svg:
                if args.verbose:
                    print("\n[EXPORTING ALL SVG PATTERNS]")
                da3.export_all_svg()
                action_performed = True
            else:
                if args.contour:
                    if args.verbose:
                        print("\n[Exporting contour to SVG]")
                    da3.export_contour_svg()
                    action_performed = True
                
                if args.svg_spiral:
                    if args.verbose:
                        print("\n[Exporting spiral to SVG]")
                    da3.export_parametric_curve_svg(curve_type='spiral')
                    action_performed = True
                
                if args.svg_lissajous:
                    if args.verbose:
                        print("\n[Exporting Lissajous to SVG]")
                    da3.export_parametric_curve_svg(curve_type='lissajous')
                    action_performed = True
                
                if args.pattern_grid:
                    if args.verbose:
                        print("\n[Exporting grid pattern to SVG]")
                    da3.export_pattern_svg(pattern_type='grid')
                    action_performed = True
                
                if args.pattern_hexagon:
                    if args.verbose:
                        print("\n[Exporting hexagon pattern to SVG]")
                    da3.export_pattern_svg(pattern_type='hexagon')
                    action_performed = True
                
                if args.text:
                    if args.verbose:
                        print(f"\n[Exporting text '{args.text}' to SVG]")
                    da3.export_text_svg(text=args.text, font_size=args.text_size)
                    action_performed = True
        except Exception as e:
            print(f"Error: 2D plotter export failed: {e}")
    
    # Handle 3D printer (STL) exports
    if args.export_stl or args.all_stl:
        try:
            if args.all_stl:
                if args.verbose:
                    print("\n[EXPORTING ALL STL MODELS]")
                da3.export_all_stl()
                action_performed = True
            else:
                if args.stl_surface:
                    if args.verbose:
                        print("\n[Exporting surface to STL]")
                    da3.export_surface_stl()
                    action_performed = True
                
                if args.torus:
                    if args.verbose:
                        print(f"\n[Exporting torus to STL (R={args.torus_R}, r={args.torus_r})]")
                    # Note: Would need to add custom parameters to export methods
                    da3.export_torus_stl()
                    action_performed = True
                
                if args.stl_sphere:
                    if args.verbose:
                        print(f"\n[Exporting sphere to STL (radius={args.sphere_radius})]")
                    da3.export_sphere_stl()
                    action_performed = True
                
                if args.helix:
                    if args.verbose:
                        print("\n[Exporting helix to STL]")
                    da3.export_helix_stl()
                    action_performed = True
        except Exception as e:
            print(f"Error: 3D printer export failed: {e}")
            print("Note: Install numpy-stl with: pip install numpy-stl")
    
    # Print summary if requested or if verbose
    if args.summary or args.verbose:
        if action_performed:
            print("\n" + "=" * 70)
            da3.print_summary()
    
    # If no action was performed, show help
    if not action_performed:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
