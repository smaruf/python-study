"""
3D Diagram Library (DA3 - Data Analytics 3D)
This module provides a comprehensive interface for creating and printing 3D diagrams.

This library addresses the question about "DA3" by providing a custom Data Analytics 3D
module for creating various types of 3D visualizations using matplotlib.
"""

import os
import sys
from datetime import datetime
import importlib.util


def load_module(name, path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load all 3D plotting modules
surface_module = load_module('surface_plot_3d', 
                              os.path.join(current_dir, '3d_surface_plot.py'))
scatter_module = load_module('scatter_plot_3d', 
                              os.path.join(current_dir, '3d_scatter_plot.py'))
wireframe_module = load_module('wireframe_plot_3d', 
                                os.path.join(current_dir, '3d_wireframe_plot.py'))
line_module = load_module('line_plot_3d', 
                          os.path.join(current_dir, '3d_line_plot.py'))

# Import functions from loaded modules
create_3d_surface_plot = surface_module.create_3d_surface_plot
create_3d_parametric_surface = surface_module.create_3d_parametric_surface
create_3d_scatter_plot = scatter_module.create_3d_scatter_plot
create_3d_cluster_scatter = scatter_module.create_3d_cluster_scatter
create_3d_wireframe_plot = wireframe_module.create_3d_wireframe_plot
create_3d_sphere_wireframe = wireframe_module.create_3d_sphere_wireframe
create_3d_line_plot = line_module.create_3d_line_plot
create_3d_spiral_plot = line_module.create_3d_spiral_plot
create_3d_lissajous_curve = line_module.create_3d_lissajous_curve


class DA3:
    """
    DA3 - Data Analytics 3D
    
    A comprehensive library for creating and printing 3D diagrams.
    Provides various types of 3D visualizations including:
    - Surface plots
    - Scatter plots
    - Wireframe plots
    - Line plots
    """
    
    def __init__(self, output_dir='./output'):
        """
        Initialize DA3 library.
        
        Args:
            output_dir (str): Directory to save output files
        """
        self.output_dir = output_dir
        self.created_plots = []
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def _get_output_path(self, filename):
        """Get full path for output file."""
        return os.path.join(self.output_dir, filename)
    
    def _log_plot(self, plot_type, filename):
        """Log created plot information."""
        self.created_plots.append({
            'type': plot_type,
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        })
    
    # Surface plots
    def surface_plot(self, save=True, filename='surface.png'):
        """Create a 3D surface plot."""
        if save:
            output_path = self._get_output_path(filename)
            create_3d_surface_plot(save_to_file=True, filename=output_path)
            self._log_plot('surface', output_path)
            return output_path
        else:
            create_3d_surface_plot(save_to_file=False)
            return None
    
    def parametric_surface(self, save=True, filename='parametric_surface.png'):
        """Create a 3D parametric surface (torus)."""
        if save:
            output_path = self._get_output_path(filename)
            create_3d_parametric_surface(save_to_file=True, filename=output_path)
            self._log_plot('parametric_surface', output_path)
            return output_path
        else:
            create_3d_parametric_surface(save_to_file=False)
            return None
    
    # Scatter plots
    def scatter_plot(self, save=True, filename='scatter.png'):
        """Create a 3D scatter plot."""
        if save:
            output_path = self._get_output_path(filename)
            create_3d_scatter_plot(save_to_file=True, filename=output_path)
            self._log_plot('scatter', output_path)
            return output_path
        else:
            create_3d_scatter_plot(save_to_file=False)
            return None
    
    def cluster_scatter(self, save=True, filename='cluster_scatter.png'):
        """Create a 3D scatter plot with multiple clusters."""
        if save:
            output_path = self._get_output_path(filename)
            create_3d_cluster_scatter(save_to_file=True, filename=output_path)
            self._log_plot('cluster_scatter', output_path)
            return output_path
        else:
            create_3d_cluster_scatter(save_to_file=False)
            return None
    
    # Wireframe plots
    def wireframe_plot(self, save=True, filename='wireframe.png'):
        """Create a 3D wireframe plot."""
        if save:
            output_path = self._get_output_path(filename)
            create_3d_wireframe_plot(save_to_file=True, filename=output_path)
            self._log_plot('wireframe', output_path)
            return output_path
        else:
            create_3d_wireframe_plot(save_to_file=False)
            return None
    
    def sphere_wireframe(self, save=True, filename='sphere.png'):
        """Create a 3D wireframe sphere."""
        if save:
            output_path = self._get_output_path(filename)
            create_3d_sphere_wireframe(save_to_file=True, filename=output_path)
            self._log_plot('sphere_wireframe', output_path)
            return output_path
        else:
            create_3d_sphere_wireframe(save_to_file=False)
            return None
    
    # Line plots
    def line_plot(self, save=True, filename='line.png'):
        """Create a 3D line plot (helix)."""
        if save:
            output_path = self._get_output_path(filename)
            create_3d_line_plot(save_to_file=True, filename=output_path)
            self._log_plot('line', output_path)
            return output_path
        else:
            create_3d_line_plot(save_to_file=False)
            return None
    
    def spiral_plot(self, save=True, filename='spiral.png'):
        """Create a 3D spiral plot."""
        if save:
            output_path = self._get_output_path(filename)
            create_3d_spiral_plot(save_to_file=True, filename=output_path)
            self._log_plot('spiral', output_path)
            return output_path
        else:
            create_3d_spiral_plot(save_to_file=False)
            return None
    
    def lissajous_curve(self, save=True, filename='lissajous.png'):
        """Create a 3D Lissajous curve."""
        if save:
            output_path = self._get_output_path(filename)
            create_3d_lissajous_curve(save_to_file=True, filename=output_path)
            self._log_plot('lissajous', output_path)
            return output_path
        else:
            create_3d_lissajous_curve(save_to_file=False)
            return None
    
    # Utility methods
    def create_all_plots(self):
        """Create all available 3D plots."""
        print("Creating all 3D plots...")
        print("-" * 50)
        
        self.surface_plot(filename='01_surface.png')
        self.parametric_surface(filename='02_parametric_surface.png')
        self.scatter_plot(filename='03_scatter.png')
        self.cluster_scatter(filename='04_cluster_scatter.png')
        self.wireframe_plot(filename='05_wireframe.png')
        self.sphere_wireframe(filename='06_sphere.png')
        self.line_plot(filename='07_line.png')
        self.spiral_plot(filename='08_spiral.png')
        self.lissajous_curve(filename='09_lissajous.png')
        
        print("-" * 50)
        print(f"All plots created successfully!")
        self.print_summary()
    
    def print_summary(self):
        """Print summary of all created plots."""
        print("\n" + "=" * 50)
        print("3D DIAGRAM GENERATION SUMMARY")
        print("=" * 50)
        print(f"Total plots created: {len(self.created_plots)}")
        print(f"Output directory: {self.output_dir}")
        print("\nCreated plots:")
        print("-" * 50)
        
        for i, plot in enumerate(self.created_plots, 1):
            print(f"{i}. Type: {plot['type']}")
            print(f"   File: {plot['filename']}")
            print(f"   Time: {plot['timestamp']}")
            print()
        
        print("=" * 50)
    
    def list_available_plots(self):
        """List all available plot types."""
        print("\n" + "=" * 50)
        print("DA3 - Available 3D Plot Types")
        print("=" * 50)
        
        plot_types = [
            ("surface_plot", "3D Surface Plot - Mathematical functions"),
            ("parametric_surface", "Parametric Surface - Torus shape"),
            ("scatter_plot", "3D Scatter Plot - Random data points"),
            ("cluster_scatter", "Cluster Scatter - Multiple data clusters"),
            ("wireframe_plot", "3D Wireframe - Mesh visualization"),
            ("sphere_wireframe", "Sphere Wireframe - Geometric shape"),
            ("line_plot", "3D Line Plot - Helix curve"),
            ("spiral_plot", "Spiral Plot - Double spiral"),
            ("lissajous_curve", "Lissajous Curve - Parametric curve"),
        ]
        
        for method, description in plot_types:
            print(f"• {method:20s} - {description}")
        
        print("=" * 50)
        print("\nUsage example:")
        print("  from da3 import DA3")
        print("  da3 = DA3(output_dir='./my_plots')")
        print("  da3.surface_plot()")
        print("  da3.create_all_plots()")
        print("=" * 50)


def demo():
    """Run a demonstration of all DA3 capabilities."""
    print("\n" + "=" * 60)
    print(" DA3 - Data Analytics 3D Library Demonstration")
    print("=" * 60)
    print("\nThis library provides comprehensive 3D visualization")
    print("capabilities for data analysis and scientific computing.")
    print("\nAnswering the question about 'DA3':")
    print("DA3 is a custom Data Analytics 3D module that provides")
    print("various types of 3D diagrams and printing functionality.")
    print("=" * 60)
    
    # Create DA3 instance
    da3 = DA3(output_dir='./da3_output')
    
    # List available plots
    da3.list_available_plots()
    
    # Create all plots
    print("\n\nGenerating all 3D plots...")
    da3.create_all_plots()
    
    print("\n\nDemo completed successfully!")
    print(f"Check the '{da3.output_dir}' directory for generated plots.")


if __name__ == "__main__":
    demo()
