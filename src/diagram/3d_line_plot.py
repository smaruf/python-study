"""
3D Line Plot Example
This module demonstrates creating 3D line plots using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_3d_line_plot(save_to_file=False, filename='3d_line.png'):
    """
    Create a 3D line plot and optionally save it to a file.
    
    Args:
        save_to_file (bool): Whether to save the plot to a file
        filename (str): Name of the file to save the plot to
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate data for a parametric curve
    t = np.linspace(0, 10, 1000)
    x = np.sin(t)
    y = np.cos(t)
    z = t
    
    # Create the 3D line plot
    ax.plot(x, y, z, 'b-', linewidth=2, label='Helix')
    
    # Add labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Line Plot: Helix Curve')
    ax.legend()
    
    if save_to_file:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    plt.show()
    return fig, ax


def create_3d_spiral_plot(save_to_file=False, filename='3d_spiral.png'):
    """
    Create a 3D spiral plot with multiple curves.
    
    Args:
        save_to_file (bool): Whether to save the plot to a file
        filename (str): Name of the file to save the plot to
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate data for multiple spirals
    t = np.linspace(0, 4 * np.pi, 1000)
    
    # Spiral 1
    x1 = t * np.cos(t)
    y1 = t * np.sin(t)
    z1 = t
    
    # Spiral 2 (rotated)
    x2 = t * np.cos(t + np.pi/4)
    y2 = t * np.sin(t + np.pi/4)
    z2 = t
    
    # Plot both spirals
    ax.plot(x1, y1, z1, 'r-', linewidth=2, label='Spiral 1', alpha=0.8)
    ax.plot(x2, y2, z2, 'g-', linewidth=2, label='Spiral 2', alpha=0.8)
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Line Plot: Double Spiral')
    ax.legend()
    
    if save_to_file:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    plt.show()
    return fig, ax


def create_3d_lissajous_curve(save_to_file=False, filename='3d_lissajous.png'):
    """
    Create a 3D Lissajous curve.
    
    Args:
        save_to_file (bool): Whether to save the plot to a file
        filename (str): Name of the file to save the plot to
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate Lissajous curve
    t = np.linspace(0, 2 * np.pi, 1000)
    
    # Parameters for Lissajous curve
    A, B, C = 1, 2, 3
    a, b, c = 3, 2, 1
    delta_x, delta_y, delta_z = 0, np.pi/2, np.pi/4
    
    x = A * np.sin(a * t + delta_x)
    y = B * np.sin(b * t + delta_y)
    z = C * np.sin(c * t + delta_z)
    
    # Color the line based on time parameter
    colors = plt.cm.viridis(t / (2 * np.pi))
    
    for i in range(len(t) - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], 
                color=colors[i], linewidth=2)
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Line Plot: Lissajous Curve')
    
    if save_to_file:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    plt.show()
    return fig, ax


if __name__ == "__main__":
    print("Creating 3D Line Plot (Helix)...")
    create_3d_line_plot(save_to_file=True, filename='3d_line.png')
    
    print("\nCreating 3D Double Spiral Plot...")
    create_3d_spiral_plot(save_to_file=True, filename='3d_spiral.png')
    
    print("\nCreating 3D Lissajous Curve...")
    create_3d_lissajous_curve(save_to_file=True, filename='3d_lissajous.png')
