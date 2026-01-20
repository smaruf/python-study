"""
Basic Aerodynamics Module

This module provides fundamental aerodynamic calculations for aircraft design,
flight dynamics, and performance analysis. It includes functions for calculating
lift, drag, and various aerodynamic coefficients.

Author: Python Study Repository
Date: 2026-01-19
"""

import math


def calculate_lift(
    velocity: float,
    air_density: float,
    wing_area: float,
    lift_coefficient: float
) -> float:
    """
    Calculate the lift force acting on a wing.
    
    Uses the lift equation: L = 0.5 * ρ * V² * S * CL
    
    Args:
        velocity (float): Airspeed in m/s
        air_density (float): Air density in kg/m³ (standard: 1.225 kg/m³ at sea level)
        wing_area (float): Wing planform area in m²
        lift_coefficient (float): Lift coefficient (CL), typically 0.2 to 1.5
    
    Returns:
        float: Lift force in Newtons (N)
    
    Example:
        >>> # Calculate lift for a small aircraft
        >>> lift = calculate_lift(25, 1.225, 10.0, 0.5)
        >>> print(f"Lift: {lift:.2f} N")
        Lift: 1914.06 N
    
    Engineering Notes:
        - Lift coefficient depends on angle of attack
        - Typical cruise CL: 0.3-0.5
        - Maximum CL before stall: 1.2-1.8
        - Standard sea level air density: 1.225 kg/m³
    """
    dynamic_pressure = 0.5 * air_density * velocity ** 2
    lift = dynamic_pressure * wing_area * lift_coefficient
    return lift


def calculate_drag(
    velocity: float,
    air_density: float,
    wing_area: float,
    drag_coefficient: float
) -> float:
    """
    Calculate the drag force acting on a body.
    
    Uses the drag equation: D = 0.5 * ρ * V² * S * CD
    
    Args:
        velocity (float): Airspeed in m/s
        air_density (float): Air density in kg/m³
        wing_area (float): Reference area in m²
        drag_coefficient (float): Drag coefficient (CD), typically 0.02 to 0.05 for aircraft
    
    Returns:
        float: Drag force in Newtons (N)
    
    Example:
        >>> # Calculate drag for a small aircraft
        >>> drag = calculate_drag(25, 1.225, 10.0, 0.03)
        >>> print(f"Drag: {drag:.2f} N")
        Drag: 114.84 N
    
    Engineering Notes:
        - Total drag = parasitic drag + induced drag
        - Parasitic drag increases with V²
        - Induced drag decreases with V²
        - Minimum total drag occurs at optimal cruise speed
    """
    dynamic_pressure = 0.5 * air_density * velocity ** 2
    drag = dynamic_pressure * wing_area * drag_coefficient
    return drag


def calculate_dynamic_pressure(velocity: float, air_density: float) -> float:
    """
    Calculate dynamic pressure (q).
    
    Formula: q = 0.5 * ρ * V²
    
    Args:
        velocity (float): Airspeed in m/s
        air_density (float): Air density in kg/m³
    
    Returns:
        float: Dynamic pressure in Pascals (Pa)
    
    Example:
        >>> q = calculate_dynamic_pressure(25, 1.225)
        >>> print(f"Dynamic pressure: {q:.2f} Pa")
        Dynamic pressure: 382.81 Pa
    
    Notes:
        - Dynamic pressure represents the kinetic energy per unit volume
        - Used in calculating aerodynamic forces
        - Also called q-bar (q̄) or velocity pressure
    """
    return 0.5 * air_density * velocity ** 2


def calculate_wing_loading(weight: float, wing_area: float) -> float:
    """
    Calculate wing loading (W/S).
    
    Args:
        weight (float): Aircraft weight in Newtons (N)
        wing_area (float): Wing planform area in m²
    
    Returns:
        float: Wing loading in N/m²
    
    Example:
        >>> # For a 1000 kg aircraft with 10 m² wing
        >>> wing_loading = calculate_wing_loading(9810, 10.0)
        >>> print(f"Wing loading: {wing_loading:.2f} N/m²")
        Wing loading: 981.00 N/m²
    
    Engineering Notes:
        - Higher wing loading = higher stall speed
        - Lower wing loading = better climb performance
        - Typical values:
            * Gliders: 200-400 N/m²
            * General aviation: 400-800 N/m²
            * Fighters: 2000-4000 N/m²
    """
    return weight / wing_area


def calculate_aspect_ratio(wingspan: float, wing_area: float) -> float:
    """
    Calculate wing aspect ratio (AR).
    
    Formula: AR = b² / S
    where b is wingspan and S is wing area
    
    Args:
        wingspan (float): Wing span in meters
        wing_area (float): Wing planform area in m²
    
    Returns:
        float: Aspect ratio (dimensionless)
    
    Example:
        >>> ar = calculate_aspect_ratio(12.0, 10.0)
        >>> print(f"Aspect ratio: {ar:.2f}")
        Aspect ratio: 14.40
    
    Engineering Notes:
        - Higher AR = better lift-to-drag ratio
        - Higher AR = more induced drag reduction
        - Typical values:
            * Gliders: 15-40
            * General aviation: 6-10
            * Fighters: 2-4
        - Trade-off: high AR wings have higher structural weight
    """
    return wingspan ** 2 / wing_area


def calculate_induced_drag_coefficient(
    lift_coefficient: float,
    aspect_ratio: float,
    efficiency_factor: float = 0.8
) -> float:
    """
    Calculate induced drag coefficient (CDi).
    
    Formula: CDi = CL² / (π * AR * e)
    
    Args:
        lift_coefficient (float): Lift coefficient (CL)
        aspect_ratio (float): Wing aspect ratio (AR)
        efficiency_factor (float): Oswald efficiency factor (e), typically 0.7-0.9
    
    Returns:
        float: Induced drag coefficient (dimensionless)
    
    Example:
        >>> cdi = calculate_induced_drag_coefficient(0.5, 10.0, 0.8)
        >>> print(f"Induced drag coefficient: {cdi:.4f}")
        Induced drag coefficient: 0.0100
    
    Engineering Notes:
        - Induced drag is caused by wingtip vortices
        - Proportional to CL²
        - Inversely proportional to aspect ratio
        - Oswald efficiency factor accounts for non-ideal lift distribution
        - Winglets can improve efficiency factor
    """
    return lift_coefficient ** 2 / (math.pi * aspect_ratio * efficiency_factor)


def calculate_stall_speed(
    weight: float,
    air_density: float,
    wing_area: float,
    max_lift_coefficient: float
) -> float:
    """
    Calculate stall speed (Vs).
    
    Formula: Vs = sqrt(2 * W / (ρ * S * CLmax))
    
    Args:
        weight (float): Aircraft weight in Newtons (N)
        air_density (float): Air density in kg/m³
        wing_area (float): Wing area in m²
        max_lift_coefficient (float): Maximum lift coefficient before stall (CLmax)
    
    Returns:
        float: Stall speed in m/s
    
    Example:
        >>> # For a 1000 kg aircraft
        >>> vs = calculate_stall_speed(9810, 1.225, 10.0, 1.4)
        >>> print(f"Stall speed: {vs:.2f} m/s ({vs*3.6:.2f} km/h)")
        Stall speed: 26.97 m/s (97.08 km/h)
    
    Safety Notes:
        - Always maintain speed above 1.3 * Vs for safety margin
        - Stall speed increases with weight
        - Stall speed increases at higher altitudes (lower air density)
        - Flaps increase CLmax, reducing stall speed
    """
    return math.sqrt(2 * weight / (air_density * wing_area * max_lift_coefficient))


def calculate_lift_to_drag_ratio(lift: float, drag: float) -> float:
    """
    Calculate lift-to-drag ratio (L/D).
    
    Args:
        lift (float): Lift force in Newtons
        drag (float): Drag force in Newtons
    
    Returns:
        float: L/D ratio (dimensionless)
    
    Example:
        >>> ld_ratio = calculate_lift_to_drag_ratio(9810, 490.5)
        >>> print(f"L/D ratio: {ld_ratio:.2f}")
        L/D ratio: 20.00
    
    Engineering Notes:
        - Higher L/D = better aerodynamic efficiency
        - Maximum L/D occurs at specific speed for each aircraft
        - Typical maximum L/D values:
            * Gliders: 40-60
            * General aviation: 10-15
            * Airliners: 15-20
            * Fighters: 8-12
        - Best range occurs at maximum L/D
    """
    return lift / drag


def reynolds_number(
    velocity: float,
    characteristic_length: float,
    kinematic_viscosity: float = 1.46e-5
) -> float:
    """
    Calculate Reynolds number (Re).
    
    Formula: Re = V * L / ν
    
    Args:
        velocity (float): Flow velocity in m/s
        characteristic_length (float): Characteristic length (e.g., chord) in meters
        kinematic_viscosity (float): Kinematic viscosity in m²/s
                                     (default: 1.46e-5 for air at 15°C)
    
    Returns:
        float: Reynolds number (dimensionless)
    
    Example:
        >>> re = reynolds_number(25, 1.5)
        >>> print(f"Reynolds number: {re:.2e}")
        Reynolds number: 2.57e+06
    
    Engineering Notes:
        - Indicates ratio of inertial to viscous forces
        - Laminar flow: Re < 500,000
        - Transitional: Re = 500,000 - 1,000,000
        - Turbulent flow: Re > 1,000,000
        - Affects drag coefficient and boundary layer behavior
        - Critical for airfoil selection and performance prediction
    """
    return velocity * characteristic_length / kinematic_viscosity


def calculate_glide_ratio(lift_to_drag_ratio: float) -> float:
    """
    Calculate glide ratio (horizontal distance per unit altitude).
    
    For unpowered flight, glide ratio equals L/D ratio.
    
    Args:
        lift_to_drag_ratio (float): L/D ratio
    
    Returns:
        float: Glide ratio (distance/altitude)
    
    Example:
        >>> glide = calculate_glide_ratio(15.0)
        >>> print(f"Glide ratio: 1:{glide:.1f}")
        Glide ratio: 1:15.0
        >>> print(f"From 1000m altitude, can glide {glide * 1000:.0f} meters")
        From 1000m altitude, can glide 15000 meters
    
    Notes:
        - Glide ratio = horizontal distance / vertical distance
        - Does not account for wind
        - Best glide speed is where L/D is maximum
    """
    return lift_to_drag_ratio


def mach_number(velocity: float, speed_of_sound: float = 343.0) -> float:
    """
    Calculate Mach number (M).
    
    Formula: M = V / a
    
    Args:
        velocity (float): Airspeed in m/s
        speed_of_sound (float): Speed of sound in m/s (default: 343 m/s at 20°C)
    
    Returns:
        float: Mach number (dimensionless)
    
    Example:
        >>> m = mach_number(100)
        >>> print(f"Mach number: {m:.3f}")
        Mach number: 0.292
    
    Flight Regimes:
        - Subsonic: M < 0.8
        - Transonic: M = 0.8 - 1.2
        - Supersonic: M = 1.2 - 5.0
        - Hypersonic: M > 5.0
    
    Notes:
        - Speed of sound varies with temperature
        - At sea level standard day (15°C): a = 340 m/s
        - At 11,000m (cruise altitude): a = 295 m/s
    """
    return velocity / speed_of_sound


if __name__ == "__main__":
    # Example usage and demonstrations
    print("=" * 60)
    print("BASIC AERODYNAMICS MODULE - EXAMPLES")
    print("=" * 60)
    
    # Example 1: Small aircraft at cruise
    print("\n1. SMALL AIRCRAFT AT CRUISE")
    print("-" * 40)
    velocity = 50  # m/s (180 km/h)
    air_density = 1.225  # kg/m³
    wing_area = 12.0  # m²
    wingspan = 12.0  # m
    weight = 10000  # N (approximately 1000 kg)
    
    CL = 0.4
    CD = 0.03
    
    lift = calculate_lift(velocity, air_density, wing_area, CL)
    drag = calculate_drag(velocity, air_density, wing_area, CD)
    ld_ratio = calculate_lift_to_drag_ratio(lift, drag)
    ar = calculate_aspect_ratio(wingspan, wing_area)
    
    print(f"Velocity: {velocity} m/s ({velocity * 3.6:.1f} km/h)")
    print(f"Wing area: {wing_area} m²")
    print(f"Wingspan: {wingspan} m")
    print(f"Lift: {lift:.2f} N")
    print(f"Drag: {drag:.2f} N")
    print(f"L/D ratio: {ld_ratio:.2f}")
    print(f"Aspect ratio: {ar:.2f}")
    
    # Example 2: Stall speed calculation
    print("\n2. STALL SPEED CALCULATION")
    print("-" * 40)
    CLmax = 1.4
    vs = calculate_stall_speed(weight, air_density, wing_area, CLmax)
    vs_safety = vs * 1.3
    
    print(f"Maximum CL: {CLmax}")
    print(f"Stall speed: {vs:.2f} m/s ({vs * 3.6:.1f} km/h)")
    print(f"Safe speed (1.3 × Vs): {vs_safety:.2f} m/s ({vs_safety * 3.6:.1f} km/h)")
    
    # Example 3: Glider performance
    print("\n3. GLIDER PERFORMANCE")
    print("-" * 40)
    glider_ld = 35.0
    altitude = 1000  # meters
    glide_distance = calculate_glide_ratio(glider_ld) * altitude
    
    print(f"L/D ratio: {glider_ld}")
    print(f"Altitude: {altitude} m")
    print(f"Maximum glide distance: {glide_distance / 1000:.1f} km")
    
    # Example 4: Reynolds number
    print("\n4. REYNOLDS NUMBER ANALYSIS")
    print("-" * 40)
    chord = 1.2  # m
    re = reynolds_number(velocity, chord)
    
    print(f"Chord length: {chord} m")
    print(f"Velocity: {velocity} m/s")
    print(f"Reynolds number: {re:.2e}")
    if re > 1e6:
        print("Flow regime: Turbulent")
    elif re > 5e5:
        print("Flow regime: Transitional")
    else:
        print("Flow regime: Laminar")
    
    # Example 5: Induced drag
    print("\n5. INDUCED DRAG ANALYSIS")
    print("-" * 40)
    CDi = calculate_induced_drag_coefficient(CL, ar, 0.8)
    induced_drag = calculate_drag(velocity, air_density, wing_area, CDi)
    
    print(f"Lift coefficient: {CL}")
    print(f"Aspect ratio: {ar}")
    print(f"Induced drag coefficient: {CDi:.4f}")
    print(f"Induced drag: {induced_drag:.2f} N")
    print(f"Parasitic drag: {drag - induced_drag:.2f} N")
    
    print("\n" + "=" * 60)
