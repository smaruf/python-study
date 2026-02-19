"""
3D Scatter Plot Example
This module demonstrates creating 3D scatter plots using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_3d_scatter_plot(save_to_file=False, filename='3d_scatter.png'):
    """
    Create a 3D scatter plot and optionally save it to a file.
    
    Args:
        save_to_file (bool): Whether to save the plot to a file
        filename (str): Name of the file to save the plot to
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate random data
    n = 200
    x = np.random.standard_normal(n)
    y = np.random.standard_normal(n)
    z = np.random.standard_normal(n)
    
    # Color based on distance from origin
    colors = np.sqrt(x**2 + y**2 + z**2)
    
    # Create the scatter plot
    scatter = ax.scatter(x, y, z, c=colors, cmap='rainbow', 
                        marker='o', s=50, alpha=0.6)
    
    # Add labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Scatter Plot: Random Data Points')
    
    # Add a color bar
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    
    if save_to_file:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    plt.show()
    return fig, ax


def create_3d_cluster_scatter(save_to_file=False, filename='3d_clusters.png'):
    """
    Create a 3D scatter plot with multiple clusters.
    
    Args:
        save_to_file (bool): Whether to save the plot to a file
        filename (str): Name of the file to save the plot to
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate clustered data
    n_points = 100
    
    # Cluster 1
    x1 = np.random.normal(0, 1, n_points)
    y1 = np.random.normal(0, 1, n_points)
    z1 = np.random.normal(0, 1, n_points)
    
    # Cluster 2
    x2 = np.random.normal(5, 1, n_points)
    y2 = np.random.normal(5, 1, n_points)
    z2 = np.random.normal(5, 1, n_points)
    
    # Cluster 3
    x3 = np.random.normal(-5, 1, n_points)
    y3 = np.random.normal(5, 1, n_points)
    z3 = np.random.normal(-5, 1, n_points)
    
    # Plot each cluster with different colors
    ax.scatter(x1, y1, z1, c='red', marker='o', s=50, alpha=0.6, label='Cluster 1')
    ax.scatter(x2, y2, z2, c='blue', marker='^', s=50, alpha=0.6, label='Cluster 2')
    ax.scatter(x3, y3, z3, c='green', marker='s', s=50, alpha=0.6, label='Cluster 3')
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Scatter Plot: Multiple Clusters')
    ax.legend()
    
    if save_to_file:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    plt.show()
    return fig, ax


if __name__ == "__main__":
    print("Creating 3D Scatter Plot...")
    create_3d_scatter_plot(save_to_file=True, filename='3d_scatter.png')
    
    print("\nCreating 3D Cluster Scatter Plot...")
    create_3d_cluster_scatter(save_to_file=True, filename='3d_clusters.png')
