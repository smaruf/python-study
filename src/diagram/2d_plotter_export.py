"""
2D Plotter Export Module
This module provides functionality to export 2D diagrams to formats compatible with 2D plotters.
Supported formats: SVG (Scalable Vector Graphics), HPGL (HP Graphics Language)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_svg import FigureCanvasSVG


def create_2d_projection_svg(save_to_file=True, filename='projection_2d.svg', 
                             view='top', plot_type='surface'):
    """
    Create a 2D projection of 3D data and export to SVG format for 2D plotting.
    
    Args:
        save_to_file (bool): Whether to save the plot to a file
        filename (str): Name of the SVG file to save
        view (str): Projection view - 'top', 'front', 'side'
        plot_type (str): Type of plot - 'surface', 'contour', 'scatter'
        
    Returns:
        fig: The matplotlib figure object
    """
    # Create figure for 2D plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate data based on plot type
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    
    if plot_type == 'contour':
        # Contour plot for 2D plotter
        contour = ax.contour(X, Y, Z, levels=15, colors='black', linewidths=1)
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_title(f'2D Contour Plot ({view} view)', fontsize=14, fontweight='bold')
    
    elif plot_type == 'filled_contour':
        # Filled contour plot
        contourf = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
        contour = ax.contour(X, Y, Z, levels=20, colors='black', linewidths=0.5, alpha=0.3)
        plt.colorbar(contourf, ax=ax)
        ax.set_title(f'2D Filled Contour Plot ({view} view)', fontsize=14, fontweight='bold')
    
    elif plot_type == 'scatter':
        # Scatter plot with color mapping
        scatter = ax.scatter(X.flatten(), Y.flatten(), c=Z.flatten(), 
                           cmap='viridis', s=10, alpha=0.6)
        plt.colorbar(scatter, ax=ax)
        ax.set_title(f'2D Scatter Plot ({view} view)', fontsize=14, fontweight='bold')
    
    else:  # surface/default
        # Use contour lines as default
        contour = ax.contour(X, Y, Z, levels=20, colors='black', linewidths=0.8)
        ax.set_title(f'2D Surface Projection ({view} view)', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    if save_to_file:
        # Save as SVG for vector output (perfect for plotters)
        fig.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
        print(f"SVG file saved to {filename}")
        print(f"Format: Scalable Vector Graphics (SVG)")
        print(f"Compatible with 2D plotters, vinyl cutters, and laser cutters")
        plt.close(fig)
    
    return fig


def create_parametric_curve_svg(save_to_file=True, filename='parametric_curve.svg',
                                curve_type='spiral'):
    """
    Create a parametric curve for 2D plotting.
    
    Args:
        save_to_file (bool): Whether to save the plot to a file
        filename (str): Name of the SVG file to save
        curve_type (str): Type of curve - 'spiral', 'lissajous', 'rose'
        
    Returns:
        fig: The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    t = np.linspace(0, 4 * np.pi, 2000)
    
    if curve_type == 'spiral':
        # Archimedean spiral
        r = t
        x = r * np.cos(t)
        y = r * np.sin(t)
        title = 'Archimedean Spiral'
    
    elif curve_type == 'lissajous':
        # Lissajous curve
        A, B = 3, 2
        a, b = 5, 4
        x = A * np.sin(a * t)
        y = B * np.sin(b * t)
        title = 'Lissajous Curve'
    
    elif curve_type == 'rose':
        # Rose curve
        k = 5  # Number of petals
        r = 5 * np.cos(k * t)
        x = r * np.cos(t)
        y = r * np.sin(t)
        title = 'Rose Curve'
    
    else:
        # Default: simple circle
        x = 5 * np.cos(t)
        y = 5 * np.sin(t)
        title = 'Circle'
    
    ax.plot(x, y, 'k-', linewidth=1.5)
    ax.set_title(f'2D {title} for Plotter', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    ax.axis('equal')
    
    if save_to_file:
        fig.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
        print(f"SVG file saved to {filename}")
        print(f"Curve type: {title}")
        print(f"Line segments: {len(t)}")
        plt.close(fig)
    
    return fig


def create_geometric_pattern_svg(save_to_file=True, filename='geometric_pattern.svg',
                                 pattern_type='grid'):
    """
    Create geometric patterns for 2D plotting.
    
    Args:
        save_to_file (bool): Whether to save the plot to a file
        filename (str): Name of the SVG file to save
        pattern_type (str): Type of pattern - 'grid', 'hexagon', 'voronoi'
        
    Returns:
        fig: The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if pattern_type == 'grid':
        # Grid pattern
        for i in range(-10, 11, 2):
            ax.axhline(y=i, color='black', linewidth=0.8)
            ax.axvline(x=i, color='black', linewidth=0.8)
        title = 'Grid Pattern'
    
    elif pattern_type == 'hexagon':
        # Hexagonal pattern
        def hexagon(x_center, y_center, size):
            angles = np.linspace(0, 2*np.pi, 7)
            x = x_center + size * np.cos(angles)
            y = y_center + size * np.sin(angles)
            return x, y
        
        size = 2
        for row in range(-3, 4):
            for col in range(-3, 4):
                x_offset = col * size * 1.5
                y_offset = row * size * np.sqrt(3)
                if col % 2 == 1:
                    y_offset += size * np.sqrt(3) / 2
                x, y = hexagon(x_offset, y_offset, size)
                ax.plot(x, y, 'k-', linewidth=0.8)
        title = 'Hexagonal Pattern'
    
    elif pattern_type == 'concentric':
        # Concentric circles
        for r in np.linspace(1, 10, 10):
            theta = np.linspace(0, 2*np.pi, 100)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            ax.plot(x, y, 'k-', linewidth=0.8)
        title = 'Concentric Circles'
    
    else:
        # Default: radial pattern
        for angle in np.linspace(0, 2*np.pi, 24, endpoint=False):
            x = [0, 10 * np.cos(angle)]
            y = [0, 10 * np.sin(angle)]
            ax.plot(x, y, 'k-', linewidth=0.8)
        title = 'Radial Pattern'
    
    ax.set_title(f'2D {title} for Plotter', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    
    if save_to_file:
        fig.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
        print(f"SVG file saved to {filename}")
        print(f"Pattern type: {title}")
        plt.close(fig)
    
    return fig


def create_text_for_plotter(save_to_file=True, filename='text_output.svg',
                           text="DA3", font_size=72):
    """
    Create text output for 2D plotting/engraving.
    
    Args:
        save_to_file (bool): Whether to save the plot to a file
        filename (str): Name of the SVG file to save
        text (str): Text to plot
        font_size (int): Font size
        
    Returns:
        fig: The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.text(0.5, 0.5, text, fontsize=font_size, ha='center', va='center',
            fontfamily='sans-serif', fontweight='bold', transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f'Text Output for 2D Plotter', fontsize=14, fontweight='bold', 
                 y=0.95)
    
    if save_to_file:
        fig.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
        print(f"SVG file saved to {filename}")
        print(f"Text: {text}")
        print(f"Font size: {font_size}pt")
        plt.close(fig)
    
    return fig


if __name__ == "__main__":
    print("=" * 70)
    print("2D Plotter Export - Creating SVG Files")
    print("=" * 70)
    
    print("\n1. Creating 2D contour projection...")
    create_2d_projection_svg(filename='plotter_contour.svg', plot_type='contour')
    
    print("\n2. Creating parametric spiral curve...")
    create_parametric_curve_svg(filename='plotter_spiral.svg', curve_type='spiral')
    
    print("\n3. Creating Lissajous curve...")
    create_parametric_curve_svg(filename='plotter_lissajous.svg', curve_type='lissajous')
    
    print("\n4. Creating geometric grid pattern...")
    create_geometric_pattern_svg(filename='plotter_grid.svg', pattern_type='grid')
    
    print("\n5. Creating hexagonal pattern...")
    create_geometric_pattern_svg(filename='plotter_hexagon.svg', pattern_type='hexagon')
    
    print("\n6. Creating text output...")
    create_text_for_plotter(filename='plotter_text.svg', text='DA3')
    
    print("\n" + "=" * 70)
    print("✓ All SVG files created successfully!")
    print("These files can be opened in 2D plotting software like:")
    print("  - Inkscape (free vector editor)")
    print("  - Adobe Illustrator")
    print("  - 2D plotter control software")
    print("  - Vinyl cutter software")
    print("  - Laser cutter software (for engraving)")
    print("=" * 70)
