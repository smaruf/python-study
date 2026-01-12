"""
Tail Design Components

Horizontal and vertical stabilizers for fixed-wing aircraft.
Light but stiff construction is critical.
"""

import cadquery as cq


def horizontal_stabilizer(
    span=400,
    chord=100,
    thickness=4,
    airfoil_thickness_ratio=0.10
):
    """
    Generate a horizontal stabilizer (elevator).
    
    Args:
        span: Stabilizer span in mm
        chord: Root chord in mm
        thickness: Structure thickness in mm
        airfoil_thickness_ratio: Max thickness as fraction of chord
    
    Returns:
        CadQuery workplane with horizontal stabilizer
    
    Design Notes:
        - Symmetric airfoil for neutral pitch
        - Lightweight construction critical
        - Usually 20-30% of wing area
    """
    
    max_thickness = chord * airfoil_thickness_ratio
    
    # Simple symmetric airfoil profile
    profile_points = [
        (0, 0),
        (chord * 0.3, max_thickness / 2),
        (chord * 0.7, max_thickness / 2),
        (chord, 0),
        (chord * 0.7, -max_thickness / 2),
        (chord * 0.3, -max_thickness / 2),
        (0, 0)
    ]
    
    # Create one half
    half = (
        cq.Workplane("XY")
        .polyline(profile_points)
        .close()
        .extrude(span / 2)
    )
    
    # Mirror for full stabilizer
    stabilizer = half.union(
        half.mirror("XZ")
    )
    
    return stabilizer


def vertical_stabilizer(
    height=120,
    chord=100,
    thickness=4
):
    """
    Generate a vertical stabilizer (rudder).
    
    Args:
        height: Stabilizer height in mm
        chord: Root chord in mm
        thickness: Structure thickness in mm
    
    Returns:
        CadQuery workplane with vertical stabilizer
    
    Design Notes:
        - Provides directional stability
        - Usually smaller than horizontal stabilizer
        - Symmetric airfoil section
    """
    
    # Simplified vertical stabilizer profile
    stabilizer = (
        cq.Workplane("XY")
        .polyline([
            (0, 0),
            (chord * 0.3, thickness / 2),
            (chord, thickness / 2),
            (chord, -thickness / 2),
            (chord * 0.3, -thickness / 2),
            (0, 0)
        ])
        .close()
        .extrude(height)
    )
    
    return stabilizer


def tail_boom_mount(
    boom_diameter=8,
    mount_length=30,
    wall_thickness=3
):
    """
    Generate a tail boom mounting bracket.
    
    Args:
        boom_diameter: Carbon tube diameter in mm
        mount_length: Length of mount in mm
        wall_thickness: Wall thickness in mm
    
    Returns:
        CadQuery workplane with boom mount
    
    Critical Component:
        - Secures tail to fuselage
        - Must be strong and lightweight
        - Use Nylon or CF-Nylon material
    """
    
    outer_radius = (boom_diameter / 2) + wall_thickness
    
    # Create tube clamp
    mount = (
        cq.Workplane("XY")
        .circle(outer_radius)
        .circle(boom_diameter / 2)
        .extrude(mount_length)
    )
    
    # Add mounting flanges
    flange = (
        cq.Workplane("XY")
        .rect(outer_radius * 3, outer_radius * 2)
        .extrude(wall_thickness)
    )
    
    mount = mount.union(flange)
    
    # Add bolt holes in flanges
    for x_offset in [-outer_radius * 1.2, outer_radius * 1.2]:
        mount = (
            mount.faces(">Z")
            .workplane()
            .center(x_offset, 0)
            .circle(1.5)  # M3 bolt hole
            .cutThruAll()
        )
    
    return mount
