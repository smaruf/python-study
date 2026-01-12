"""
Fuselage Design

Modular fuselage sections for fixed-wing aircraft.
Semi-monocoque shell construction optimized for 3D printing.
"""

import cadquery as cq


def fuselage_section(
    radius=35,
    length=80,
    wall=2,
    reinforced=False
):
    """
    Generate a cylindrical fuselage section.
    
    Args:
        radius: Outer radius in mm (default: 35mm)
        length: Section length in mm (default: 80mm)
        wall: Wall thickness in mm (default: 2mm)
        reinforced: Add internal ribs for strength (default: False)
    
    Returns:
        CadQuery workplane with fuselage section
    
    Engineering Notes:
        - Semi-monocoque design: shell carries loads
        - Reinforced sections for wing/motor mounts
        - Modular for easy printing and assembly
    """
    
    # Create outer cylinder
    outer = (
        cq.Workplane("XY")
        .circle(radius)
        .extrude(length)
    )
    
    # Create inner cavity
    inner = (
        cq.Workplane("XY")
        .workplane(offset=wall)
        .circle(radius - wall)
        .extrude(length - wall * 2)
    )
    
    section = outer.cut(inner)
    
    # Add reinforcement ribs if requested
    if reinforced:
        # Internal ribs at 1/3 and 2/3 length
        for position in [length/3, 2*length/3]:
            rib = (
                cq.Workplane("XY")
                .workplane(offset=position)
                .circle(radius - wall - 1)
                .circle(radius - wall - 2)
                .extrude(wall)
            )
            section = section.union(rib)
    
    return section


def fuselage_bulkhead(
    radius=35,
    thickness=4,
    center_hole=0
):
    """
    Generate a fuselage bulkhead for reinforcement.
    
    Args:
        radius: Bulkhead radius in mm
        thickness: Bulkhead thickness in mm
        center_hole: Center hole diameter in mm (0 for solid)
    
    Returns:
        CadQuery workplane with bulkhead
    
    Use Cases:
        - Wing attachment points
        - Motor mount reinforcement
        - Landing gear attachment
    """
    
    bulkhead = (
        cq.Workplane("XY")
        .circle(radius)
        .extrude(thickness)
    )
    
    if center_hole > 0:
        bulkhead = (
            bulkhead.faces(">Z")
            .workplane()
            .circle(center_hole / 2)
            .cutThruAll()
        )
    
    return bulkhead


def wing_mount_plate(
    width=60,
    height=40,
    thickness=6,
    bolt_spacing=20
):
    """
    Generate a wing-fuselage mounting plate.
    
    Args:
        width: Plate width in mm
        height: Plate height in mm
        thickness: Plate thickness in mm
        bolt_spacing: Spacing between mounting bolts in mm
    
    Returns:
        CadQuery workplane with mounting plate
    
    Critical Zone:
        90% of fixed-wing failures occur at wing-fuselage connection.
        Use double wall thickness and proper fasteners.
    """
    
    plate = (
        cq.Workplane("XY")
        .rect(width, height)
        .extrude(thickness)
    )
    
    # Add bolt holes in a square pattern
    bolt_hole_diameter = 3
    
    for x_offset in [-bolt_spacing/2, bolt_spacing/2]:
        for y_offset in [-bolt_spacing/2, bolt_spacing/2]:
            plate = (
                plate.faces(">Z")
                .workplane()
                .center(x_offset, y_offset)
                .circle(bolt_hole_diameter / 2)
                .cutThruAll()
            )
    
    # Add center slot for spar pass-through
    plate = (
        plate.faces(">Z")
        .workplane()
        .rect(bolt_spacing * 1.5, 8)
        .cutThruAll()
    )
    
    return plate
