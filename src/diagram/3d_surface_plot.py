"""
3D Surface Plot Example
This module demonstrates creating 3D surface plots using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt


def create_3d_surface_plot(save_to_file=False, filename='3d_surface.png'):
    """
    Create a 3D surface plot and optionally save it to a file.
    
    Args:
        save_to_file (bool): Whether to save the plot to a file
        filename (str): Name of the file to save the plot to
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate data
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z values - create a 3D surface
    Z = np.sin(np.sqrt(X**2 + Y**2))
    
    # Create the surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    # Add labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Surface Plot: z = sin(√(x² + y²))')
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    if save_to_file:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    plt.show()
    return fig, ax


def create_3d_parametric_surface(save_to_file=False, filename='3d_parametric.png'):
    """
    Create a 3D parametric surface (torus).
    
    Args:
        save_to_file (bool): Whether to save the plot to a file
        filename (str): Name of the file to save the plot to
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Parameters for torus
    R = 3  # Major radius
    r = 1  # Minor radius
    
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, 2 * np.pi, 100)
    U, V = np.meshgrid(u, v)
    
    # Parametric equations for torus
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    
    # Create the surface
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8)
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Parametric Surface: Torus')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    if save_to_file:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
        plt.close(fig)
    else:
        plt.show()
    
    return fig, ax


if __name__ == "__main__":
    print("Creating 3D Surface Plot...")
    create_3d_surface_plot(save_to_file=True, filename='3d_surface.png')
    
    print("\nCreating 3D Parametric Surface (Torus)...")
    create_3d_parametric_surface(save_to_file=True, filename='3d_parametric.png')
