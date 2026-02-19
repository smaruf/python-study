"""
3D Wireframe Plot Example
This module demonstrates creating 3D wireframe plots using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_3d_wireframe_plot(save_to_file=False, filename='3d_wireframe.png'):
    """
    Create a 3D wireframe plot and optionally save it to a file.
    
    Args:
        save_to_file (bool): Whether to save the plot to a file
        filename (str): Name of the file to save the plot to
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate data
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z values - create a 3D surface
    Z = np.cos(np.sqrt(X**2 + Y**2))
    
    # Create the wireframe plot
    ax.plot_wireframe(X, Y, Z, color='blue', alpha=0.7, linewidth=0.5)
    
    # Add labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Wireframe Plot: z = cos(√(x² + y²))')
    
    if save_to_file:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    plt.show()
    return fig, ax


def create_3d_sphere_wireframe(save_to_file=False, filename='3d_sphere.png'):
    """
    Create a 3D wireframe sphere.
    
    Args:
        save_to_file (bool): Whether to save the plot to a file
        filename (str): Name of the file to save the plot to
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate sphere data
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    U, V = np.meshgrid(u, v)
    
    # Parametric equations for sphere
    X = np.cos(U) * np.sin(V)
    Y = np.sin(U) * np.sin(V)
    Z = np.cos(V)
    
    # Create the wireframe
    ax.plot_wireframe(X, Y, Z, color='green', alpha=0.7, linewidth=0.8)
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Wireframe: Sphere')
    
    if save_to_file:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    plt.show()
    return fig, ax


if __name__ == "__main__":
    print("Creating 3D Wireframe Plot...")
    create_3d_wireframe_plot(save_to_file=True, filename='3d_wireframe.png')
    
    print("\nCreating 3D Sphere Wireframe...")
    create_3d_sphere_wireframe(save_to_file=True, filename='3d_sphere.png')
