"""
Spar Design and Load Calculations

Spar is the primary load-bearing structure in a wing.
It resists bending moments from aerodynamic loads.
"""


def wing_bending_load(weight, span):
    """
    Calculate simplified wing bending load.
    
    Args:
        weight: Aircraft weight in grams
        span: Wing span in mm
    
    Returns:
        Bending moment in g·mm
    
    Engineering Rule:
        For a simple wing, bending moment ≈ weight × span / 4
        This assumes uniform lift distribution (simplified)
    """
    return weight * span / 4


def spar_stress(bending_moment, distance_from_neutral, moment_of_inertia):
    """
    Calculate bending stress in spar using beam theory.
    
    Args:
        bending_moment: Bending moment in g·mm
        distance_from_neutral: Distance from neutral axis in mm
        moment_of_inertia: Second moment of area in mm⁴
    
    Returns:
        Stress in g/mm²
    
    Formula:
        σ = M × y / I
        Where:
            σ = bending stress
            M = bending moment
            y = distance from neutral axis
            I = second moment of area
    """
    return (bending_moment * distance_from_neutral) / moment_of_inertia


def tube_moment_of_inertia(outer_diameter, inner_diameter):
    """
    Calculate moment of inertia for a hollow circular tube.
    
    Args:
        outer_diameter: Outer diameter in mm
        inner_diameter: Inner diameter in mm
    
    Returns:
        Moment of inertia in mm⁴
    
    Formula:
        I = π/64 × (D⁴ - d⁴)
    """
    import math
    return (math.pi / 64) * (outer_diameter**4 - inner_diameter**4)


def rectangular_moment_of_inertia(width, height):
    """
    Calculate moment of inertia for a rectangular beam.
    
    Args:
        width: Width (base) in mm
        height: Height in mm
    
    Returns:
        Moment of inertia in mm⁴
    
    Formula:
        I = (width × height³) / 12
    """
    return (width * height**3) / 12


def recommend_spar_type(wingspan, weight):
    """
    Recommend spar type based on aircraft parameters.
    
    Args:
        wingspan: Wing span in mm
        weight: Aircraft weight in grams
    
    Returns:
        Dictionary with spar recommendation
    
    Rules of thumb:
        - Small UAVs (< 1000mm, < 1kg): Printed spar acceptable
        - Medium UAVs (1000-1500mm, 1-2kg): Carbon tube preferred
        - Large UAVs (> 1500mm, > 2kg): Carbon tube required
    """
    if wingspan < 1000 and weight < 1000:
        return {
            "type": "Printed spar (Nylon/CF-Nylon)",
            "notes": "Acceptable for small UAVs, add safety factor 2×",
            "alternative": "6mm carbon tube for better performance"
        }
    elif wingspan < 1500 and weight < 2000:
        return {
            "type": "Carbon tube (6-8mm OD)",
            "notes": "Recommended for reliability",
            "alternative": "Heavy-duty printed spar (test first)"
        }
    else:
        return {
            "type": "Carbon tube (8-10mm OD)",
            "notes": "Required for structural integrity",
            "alternative": "Aluminum tube (heavier but acceptable)"
        }
