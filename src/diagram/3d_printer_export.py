"""
3D Printer Export Module
This module provides functionality to export 3D diagrams to formats compatible with 3D printers.
Supported formats: STL (stereolithography), OBJ (Wavefront)
"""

import numpy as np
from stl import mesh


def create_surface_mesh_stl(save_to_file=True, filename='surface_mesh.stl'):
    """
    Create a 3D surface mesh and export to STL format for 3D printing.
    
    Args:
        save_to_file (bool): Whether to save the mesh to a file
        filename (str): Name of the STL file to save
        
    Returns:
        mesh.Mesh: The created mesh object
    """
    # Generate a mathematical surface for 3D printing
    # Using a scaled-down version suitable for printing
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    
    # Create a wave surface (scaled for printing in cm)
    Z = 0.5 * np.sin(np.sqrt(X**2 + Y**2))
    
    # Convert to vertices and faces for STL
    vertices = []
    faces = []
    
    # Create triangular faces from the grid
    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            # Get the four corners of the quad
            p1 = [X[i, j], Y[i, j], Z[i, j]]
            p2 = [X[i+1, j], Y[i+1, j], Z[i+1, j]]
            p3 = [X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]]
            p4 = [X[i, j+1], Y[i, j+1], Z[i, j+1]]
            
            # Add vertices
            v_idx = len(vertices)
            vertices.extend([p1, p2, p3, p4])
            
            # Create two triangles for the quad
            faces.append([v_idx, v_idx+1, v_idx+2])
            faces.append([v_idx, v_idx+2, v_idx+3])
    
    # Create the mesh
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Create mesh object
    surface_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            surface_mesh.vectors[i][j] = vertices[face[j], :]
    
    if save_to_file:
        surface_mesh.save(filename)
        print(f"STL mesh saved to {filename}")
        print(f"Mesh contains {len(faces)} triangular faces")
        print(f"Dimensions: X=[{vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}], "
              f"Y=[{vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}], "
              f"Z=[{vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f}] (in cm)")
    
    return surface_mesh


def create_torus_stl(save_to_file=True, filename='torus.stl', R=3, r=1):
    """
    Create a torus (donut shape) mesh for 3D printing.
    
    Args:
        save_to_file (bool): Whether to save the mesh to a file
        filename (str): Name of the STL file to save
        R (float): Major radius (in cm)
        r (float): Minor radius (in cm)
        
    Returns:
        mesh.Mesh: The created mesh object
    """
    # Generate torus vertices
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, 2 * np.pi, 30)
    U, V = np.meshgrid(u, v)
    
    # Parametric equations for torus
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    
    # Convert to vertices and faces
    vertices = []
    faces = []
    
    # Create triangular faces from the grid
    for i in range(len(u) - 1):
        for j in range(len(v) - 1):
            # Get the four corners of the quad
            p1 = [X[j, i], Y[j, i], Z[j, i]]
            p2 = [X[j+1, i], Y[j+1, i], Z[j+1, i]]
            p3 = [X[j+1, i+1], Y[j+1, i+1], Z[j+1, i+1]]
            p4 = [X[j, i+1], Y[j, i+1], Z[j, i+1]]
            
            # Add vertices
            v_idx = len(vertices)
            vertices.extend([p1, p2, p3, p4])
            
            # Create two triangles for the quad
            faces.append([v_idx, v_idx+1, v_idx+2])
            faces.append([v_idx, v_idx+2, v_idx+3])
    
    # Create the mesh
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Create mesh object
    torus_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            torus_mesh.vectors[i][j] = vertices[face[j], :]
    
    if save_to_file:
        torus_mesh.save(filename)
        print(f"STL torus saved to {filename}")
        print(f"Mesh contains {len(faces)} triangular faces")
        print(f"Major radius: {R} cm, Minor radius: {r} cm")
    
    return torus_mesh


def create_sphere_stl(save_to_file=True, filename='sphere.stl', radius=2):
    """
    Create a sphere mesh for 3D printing.
    
    Args:
        save_to_file (bool): Whether to save the mesh to a file
        filename (str): Name of the STL file to save
        radius (float): Sphere radius (in cm)
        
    Returns:
        mesh.Mesh: The created mesh object
    """
    # Generate sphere vertices
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 30)
    U, V = np.meshgrid(u, v)
    
    # Parametric equations for sphere
    X = radius * np.cos(U) * np.sin(V)
    Y = radius * np.sin(U) * np.sin(V)
    Z = radius * np.cos(V)
    
    # Convert to vertices and faces
    vertices = []
    faces = []
    
    # Create triangular faces from the grid
    for i in range(len(u) - 1):
        for j in range(len(v) - 1):
            # Get the four corners of the quad
            p1 = [X[j, i], Y[j, i], Z[j, i]]
            p2 = [X[j+1, i], Y[j+1, i], Z[j+1, i]]
            p3 = [X[j+1, i+1], Y[j+1, i+1], Z[j+1, i+1]]
            p4 = [X[j, i+1], Y[j, i+1], Z[j, i+1]]
            
            # Add vertices
            v_idx = len(vertices)
            vertices.extend([p1, p2, p3, p4])
            
            # Create two triangles for the quad
            faces.append([v_idx, v_idx+1, v_idx+2])
            faces.append([v_idx, v_idx+2, v_idx+3])
    
    # Create the mesh
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Create mesh object
    sphere_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            sphere_mesh.vectors[i][j] = vertices[face[j], :]
    
    if save_to_file:
        sphere_mesh.save(filename)
        print(f"STL sphere saved to {filename}")
        print(f"Mesh contains {len(faces)} triangular faces")
        print(f"Sphere radius: {radius} cm")
    
    return sphere_mesh


def create_helix_tube_stl(save_to_file=True, filename='helix_tube.stl', 
                          tube_radius=0.2, helix_radius=1.5, height=5):
    """
    Create a helix tube (spring shape) for 3D printing.
    
    Args:
        save_to_file (bool): Whether to save the mesh to a file
        filename (str): Name of the STL file to save
        tube_radius (float): Radius of the tube (in cm)
        helix_radius (float): Radius of the helix (in cm)
        height (float): Total height of the helix (in cm)
        
    Returns:
        mesh.Mesh: The created mesh object
    """
    # Generate helix path
    t = np.linspace(0, 4 * np.pi, 200)
    helix_x = helix_radius * np.cos(t)
    helix_y = helix_radius * np.sin(t)
    helix_z = height * t / (4 * np.pi)
    
    # Create tube around helix path
    vertices = []
    faces = []
    
    n_circle = 12  # Number of points around tube circumference
    theta = np.linspace(0, 2 * np.pi, n_circle)
    
    for i in range(len(t)):
        # Calculate tangent vector
        if i < len(t) - 1:
            tangent = np.array([
                helix_x[i+1] - helix_x[i],
                helix_y[i+1] - helix_y[i],
                helix_z[i+1] - helix_z[i]
            ])
        else:
            tangent = np.array([
                helix_x[i] - helix_x[i-1],
                helix_y[i] - helix_y[i-1],
                helix_z[i] - helix_z[i-1]
            ])
        tangent = tangent / np.linalg.norm(tangent)
        
        # Create perpendicular vectors
        if abs(tangent[2]) < 0.99:
            perp1 = np.cross(tangent, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(tangent, np.array([1, 0, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(tangent, perp1)
        
        # Add circle vertices around helix path
        for angle in theta:
            offset = tube_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertex = [
                helix_x[i] + offset[0],
                helix_y[i] + offset[1],
                helix_z[i] + offset[2]
            ]
            vertices.append(vertex)
    
    # Create faces connecting the circles
    for i in range(len(t) - 1):
        for j in range(n_circle):
            j_next = (j + 1) % n_circle
            
            v1 = i * n_circle + j
            v2 = i * n_circle + j_next
            v3 = (i + 1) * n_circle + j_next
            v4 = (i + 1) * n_circle + j
            
            # Two triangles per quad
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])
    
    # Create the mesh
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Create mesh object
    helix_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            helix_mesh.vectors[i][j] = vertices[face[j], :]
    
    if save_to_file:
        helix_mesh.save(filename)
        print(f"STL helix tube saved to {filename}")
        print(f"Mesh contains {len(faces)} triangular faces")
        print(f"Dimensions: Tube radius={tube_radius} cm, Helix radius={helix_radius} cm, Height={height} cm")
    
    return helix_mesh


if __name__ == "__main__":
    print("=" * 70)
    print("3D Printer Export - Creating STL Files")
    print("=" * 70)
    
    print("\n1. Creating surface mesh...")
    create_surface_mesh_stl(filename='printable_surface.stl')
    
    print("\n2. Creating torus (donut)...")
    create_torus_stl(filename='printable_torus.stl')
    
    print("\n3. Creating sphere...")
    create_sphere_stl(filename='printable_sphere.stl')
    
    print("\n4. Creating helix tube (spring)...")
    create_helix_tube_stl(filename='printable_helix.stl')
    
    print("\n" + "=" * 70)
    print("✓ All STL files created successfully!")
    print("These files can be opened in 3D printing software like:")
    print("  - Cura")
    print("  - PrusaSlicer")
    print("  - Simplify3D")
    print("  - MeshLab (for viewing)")
    print("=" * 70)
