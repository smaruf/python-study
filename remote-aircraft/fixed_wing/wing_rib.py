"""
Wing Rib Generator

Parametric wing rib design for fixed-wing aircraft.
Ribs define the airfoil shape and provide structure for the wing.
"""

try:
    import cadquery as cq
    HAS_CADQUERY = True
except ImportError:
    HAS_CADQUERY = False


def wing_rib(
    chord=180,
    thickness=6,
    spar_slot=10
):
    """
    Generate a parametric wing rib with Clark-Y airfoil.
    
    Args:
        chord: Wing chord length in mm (default: 180mm)
        thickness: Rib thickness in mm (default: 6mm)
        spar_slot: Width of spar slot in mm (default: 10mm)
    
    Returns:
        CadQuery workplane with the rib geometry
    
    Engineering Notes:
        - Clark-Y is a popular airfoil for small UAVs
        - Spar slot provides structural connection
        - Simplified airfoil for printability
    
    Raises:
        ImportError: If CadQuery is not installed
    """
    
    if not HAS_CADQUERY:
        raise ImportError(
            "CadQuery is required for STL generation. "
            "Install with: conda install -c conda-forge -c cadquery cadquery"
        )
    
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
    
    Raises:
        ImportError: If CadQuery is not installed
    """
    
    if not HAS_CADQUERY:
        raise ImportError(
            "CadQuery is required for STL generation. "
            "Install with: conda install -c conda-forge -c cadquery cadquery"
        )
    
    # Simplified symmetric airfoil coordinates (relative to chord)
    # These create a teardrop shape suitable for basic UAVs:
    # - Leading edge at 0% chord
    # - Maximum thickness (15%) at 60% chord
    # - Trailing edge at 100% chord
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
