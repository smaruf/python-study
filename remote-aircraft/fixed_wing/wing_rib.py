"""
Wing Rib Generator

Parametric wing rib design for fixed-wing aircraft.
Ribs define the airfoil shape and provide structure for the wing.
"""

import cadquery as cq


def wing_rib(
    chord=180,
    thickness=6,
    spar_slot=10,
    airfoil_type="clark-y"
):
    """
    Generate a parametric wing rib.
    
    Args:
        chord: Wing chord length in mm (default: 180mm)
        thickness: Rib thickness in mm (default: 6mm)
        spar_slot: Width of spar slot in mm (default: 10mm)
        airfoil_type: Airfoil profile type (default: "clark-y")
    
    Returns:
        CadQuery workplane with the rib geometry
    
    Engineering Notes:
        - Clark-Y is a popular airfoil for small UAVs
        - Spar slot provides structural connection
        - Simplified airfoil for printability
    """
    
    # Clark-Y simplified airfoil coordinates (relative to chord)
    # Upper surface
    upper_coords = [
        (0.00, 0.00),
        (0.05, 0.06),
        (0.20, 0.12),
        (0.40, 0.15),
        (0.60, 0.15),
        (0.80, 0.10),
        (0.95, 0.04),
        (1.00, 0.00),
    ]
    
    # Lower surface (flat bottom for Clark-Y)
    lower_coords = [
        (1.00, 0.00),
        (0.00, 0.00),
    ]
    
    # Convert to absolute coordinates
    points = []
    for x, y in upper_coords:
        points.append((x * chord, y * chord))
    
    for x, y in reversed(lower_coords[:-1]):  # Exclude duplicate start point
        points.append((x * chord, y * chord))
    
    # Create rib profile
    rib = (
        cq.Workplane("XY")
        .polyline(points)
        .close()
        .extrude(thickness)
    )
    
    # Add spar slot at 30% chord (standard location)
    spar_position = chord * 0.30
    spar_height = chord * 0.12  # Height at spar location
    
    rib = (
        rib.faces(">Z")
        .workplane()
        .center(spar_position - chord/2, spar_height/2)
        .rect(spar_slot, thickness + 1)
        .cutBlind(-thickness)
    )
    
    return rib


def wing_rib_simple(
    chord=180,
    thickness=6,
    spar_slot=10
):
    """
    Generate a simplified symmetric wing rib.
    
    Simpler airfoil for basic UAVs and easier printing.
    
    Args:
        chord: Wing chord length in mm
        thickness: Rib thickness in mm
        spar_slot: Width of spar slot in mm
    
    Returns:
        CadQuery workplane with the rib geometry
    """
    
    # Simplified symmetric airfoil
    rib = (
        cq.Workplane("XY")
        .polyline([
            (0, 0),
            (chord, 0),
            (chord * 0.95, chord * 0.12),
            (chord * 0.6, chord * 0.15),
            (chord * 0.2, chord * 0.12),
            (0, 0)
        ])
        .close()
        .extrude(thickness)
    )
    
    # Spar slot
    rib = (
        rib.faces(">Z")
        .workplane()
        .center(chord * 0.30 - chord/2, chord * 0.08)
        .rect(spar_slot, thickness + 1)
        .cutBlind(-thickness)
    )
    
    return rib
