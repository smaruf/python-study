"""
Flight Load Calculations

Aerodynamic load analysis for fixed-wing aircraft.
Essential for structural design and safety.
"""


def lift_per_wing(weight, safety_factor=2.0):
    """
    Calculate lift per wing with safety factor.
    
    Args:
        weight: Total aircraft weight in grams
        safety_factor: Safety factor for gusts (default: 2.0)
    
    Returns:
        Lift per wing in grams
    
    Engineering Notes:
        - Weight is distributed between two wings
        - 2× safety factor accounts for gusts and maneuvers
        - Actual load can spike during turns and gusts
    """
    return (weight / 2) * safety_factor


def wing_loading(weight, wing_area):
    """
    Calculate wing loading.
    
    Args:
        weight: Aircraft weight in grams
        wing_area: Total wing area in mm²
    
    Returns:
        Wing loading in g/mm²
    
    Typical Values:
        - Gliders: 0.0001 - 0.0003 g/mm²
        - Sport planes: 0.0003 - 0.0005 g/mm²
        - Racers: 0.0005 - 0.0008 g/mm²
    """
    return weight / wing_area


def estimate_cruise_speed(weight, wing_area, lift_coefficient=0.5):
    """
    Estimate cruise speed (simplified).
    
    Args:
        weight: Aircraft weight in grams
        wing_area: Wing area in mm²
        lift_coefficient: Lift coefficient (default: 0.5)
    
    Returns:
        Approximate cruise speed in m/s
    
    Simplified Formula:
        V ≈ sqrt(2 × W / (ρ × S × CL))
        Where:
            W = weight (converted to N)
            ρ = air density (~1.225 kg/m³)
            S = wing area (converted to m²)
            CL = lift coefficient
    """
    import math
    
    # Convert units
    weight_n = weight * 0.00981  # grams to Newtons
    wing_area_m2 = wing_area / 1_000_000  # mm² to m²
    rho = 1.225  # kg/m³ at sea level
    
    # Calculate velocity
    velocity = math.sqrt((2 * weight_n) / (rho * wing_area_m2 * lift_coefficient))
    
    return velocity


def tail_moment_arm(cg_position, tail_position):
    """
    Calculate tail moment arm.
    
    Args:
        cg_position: CG position from nose in mm
        tail_position: Tail aerodynamic center from nose in mm
    
    Returns:
        Moment arm in mm
    
    Engineering Note:
        Longer moment arm = more stable but heavier tail boom
        Typical: 2-3× wing chord length
    """
    return tail_position - cg_position


def tail_volume_coefficient(tail_area, tail_arm, wing_area, wing_chord):
    """
    Calculate horizontal tail volume coefficient.
    
    Args:
        tail_area: Horizontal tail area in mm²
        tail_arm: Distance from CG to tail AC in mm
        wing_area: Wing area in mm²
        wing_chord: Wing mean chord in mm
    
    Returns:
        Tail volume coefficient (dimensionless)
    
    Typical Values:
        - Trainers: 0.5 - 0.7 (very stable)
        - Sport planes: 0.4 - 0.5 (balanced)
        - Aerobatic: 0.3 - 0.4 (agile)
    """
    return (tail_area * tail_arm) / (wing_area * wing_chord)


def calculate_flight_loads(
    weight,
    wingspan,
    chord,
    cg_position=None
):
    """
    Comprehensive flight load analysis.
    
    Args:
        weight: Aircraft weight in grams
        wingspan: Wing span in mm
        chord: Wing chord in mm
        cg_position: CG position from nose in mm (optional)
    
    Returns:
        Dictionary with load analysis results
    """
    
    # Calculate wing area
    wing_area = wingspan * chord
    
    # Lift calculations
    lift_total = weight * 2  # With safety factor
    lift_per_w = lift_per_wing(weight)
    
    # Wing loading
    w_loading = wing_loading(weight, wing_area)
    
    # Cruise speed estimate
    cruise_speed = estimate_cruise_speed(weight, wing_area)
    
    # Recommended tail size
    tail_arm_recommended = chord * 2.5
    tail_area_recommended = wing_area * 0.3  # 30% of wing area
    
    results = {
        "wing_area_mm2": wing_area,
        "wing_area_cm2": wing_area / 100,
        "lift_total_g": lift_total,
        "lift_per_wing_g": lift_per_w,
        "wing_loading_g_mm2": w_loading,
        "wing_loading_g_cm2": w_loading * 100,
        "estimated_cruise_speed_ms": cruise_speed,
        "recommended_tail_arm_mm": tail_arm_recommended,
        "recommended_tail_area_mm2": tail_area_recommended,
    }
    
    return results
