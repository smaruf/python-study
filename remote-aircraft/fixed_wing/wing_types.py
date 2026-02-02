"""
Advanced Wing Type Designs and Analysis

This module provides detailed designs, theoretical analysis, and construction
principles for various wing configurations:

Traditional Wing Types:
- Straight-wing (rectangular/tapered)
- Backward swept wing
- Forward swept wing

Advanced Wing Types:
- Delta-wing
- Flying-wing
- Canard
- Oblique-wing
- Flying-pancake (round wing)

Each wing type includes:
- Geometric design parameters
- Aerodynamic analysis
- Stability considerations
- Construction principles
- Performance characteristics
"""

import math

try:
    import cadquery as cq
    HAS_CADQUERY = True
except ImportError:
    HAS_CADQUERY = False


# ============================================================================
# DELTA WING
# ============================================================================

def delta_wing_design(
    root_chord=400,
    wingspan=1000,
    sweep_angle=45,
    thickness_ratio=0.08
):
    """
    Delta wing design and analysis.
    
    Delta wings are triangular wings used in high-speed aircraft.
    They offer excellent high-speed performance and structural efficiency.
    
    Args:
        root_chord: Root chord length in mm (default: 400mm)
        wingspan: Total wingspan in mm (default: 1000mm)
        sweep_angle: Leading edge sweep angle in degrees (default: 45°)
        thickness_ratio: Airfoil thickness ratio (default: 0.08)
    
    Returns:
        Dictionary with design parameters and analysis
    
    Theoretical Background:
        - Delta wings have swept leading edges forming a triangular planform
        - Excellent structural efficiency (carries bending loads in plane of wing)
        - Generates lift via leading edge vortices at high angles of attack
        - Low aspect ratio reduces induced drag at high speeds
        
    Aerodynamic Characteristics:
        - Aspect Ratio (AR) = wingspan² / area (typically 2-4 for deltas)
        - Sweep reduces effective AR and delays shock wave formation
        - Leading edge vortices enhance lift at high alpha
        - Better high-speed performance than conventional wings
        
    Stability Considerations:
        - Center of lift moves aft with increasing angle of attack
        - Requires careful CG placement (typically 25-30% of root chord)
        - No horizontal tail needed (tailless design)
        - Elevons provide pitch and roll control
        
    Construction Principles:
        1. Wing Structure:
           - Main spar along centerline or dual spars
           - Ribs perpendicular to flight direction
           - Torsion box from leading edge to 60% chord
           
        2. Material Selection:
           - Foam core with carbon/fiberglass skin
           - 3D printed ribs with carbon spar
           - EPP foam for leading edge (impact resistance)
           
        3. Control Surfaces:
           - Elevons on trailing edge (pitch + roll)
           - Typically 25-30% of root chord
           - Consider split flaps for drag control
    """
    
    # Calculate geometric parameters
    sweep_rad = math.radians(sweep_angle)
    
    # Tip chord (delta wings taper to point or small tip)
    tip_chord = root_chord * 0.1  # 10% of root
    
    # Calculate semi-span
    semi_span = wingspan / 2
    
    # Wing area (trapezoidal approximation)
    area = ((root_chord + tip_chord) / 2) * wingspan
    
    # Aspect ratio
    aspect_ratio = (wingspan ** 2) / area
    
    # Mean aerodynamic chord (MAC)
    mac = (2/3) * root_chord * ((1 + tip_chord/root_chord + (tip_chord/root_chord)**2) / 
                                 (1 + tip_chord/root_chord))
    
    # Leading edge length
    leading_edge_length = semi_span / math.cos(sweep_rad)
    
    # Taper ratio
    taper_ratio = tip_chord / root_chord
    
    # Center of pressure (approximate, varies with AOA)
    cp_position = 0.40 * mac  # Typically 40-45% MAC for delta
    
    # Recommended CG position (% of MAC)
    cg_position = 0.30 * mac  # 25-30% MAC for stability
    
    # Elevon sizing
    elevon_chord = 0.28 * root_chord  # 28% of root chord
    elevon_span = wingspan * 0.40  # 40% of span each side
    
    # Structural considerations
    max_thickness = root_chord * thickness_ratio
    spar_depth = max_thickness * 0.6  # Spar typically 60% of thickness
    
    # Performance estimates
    stall_speed_factor = 1.3  # Delta wings stall ~30% higher than conventional
    
    return {
        "type": "Delta Wing",
        "geometry": {
            "root_chord_mm": root_chord,
            "tip_chord_mm": tip_chord,
            "wingspan_mm": wingspan,
            "sweep_angle_deg": sweep_angle,
            "area_mm2": area,
            "aspect_ratio": aspect_ratio,
            "taper_ratio": taper_ratio,
            "mac_mm": mac,
            "leading_edge_length_mm": leading_edge_length,
        },
        "aerodynamics": {
            "cp_position_mm": cp_position,
            "recommended_cg_mm": cg_position,
            "stall_speed_factor": stall_speed_factor,
            "vortex_lift_available": True,
        },
        "control_surfaces": {
            "elevon_chord_mm": elevon_chord,
            "elevon_span_mm": elevon_span,
            "surface_area_mm2": elevon_chord * elevon_span * 2,
        },
        "structure": {
            "max_thickness_mm": max_thickness,
            "spar_depth_mm": spar_depth,
            "construction_method": "Foam core with composite skin or 3D printed ribs",
        },
        "notes": [
            "Tailless design - no horizontal stabilizer needed",
            "Elevons provide combined pitch and roll control",
            "Leading edge vortices enhance lift at high AOA",
            "Excellent for high-speed flight and aerobatics",
            "Requires precise CG location for stability",
        ]
    }


# ============================================================================
# FLYING WING
# ============================================================================

def flying_wing_design(
    center_chord=350,
    wingspan=1200,
    sweep_angle=25,
    wing_twist=-2
):
    """
    Flying wing (tailless) design and analysis.
    
    Flying wings eliminate the fuselage and tail, providing maximum
    aerodynamic efficiency and reduced drag.
    
    Args:
        center_chord: Center section chord in mm (default: 350mm)
        wingspan: Total wingspan in mm (default: 1200mm)
        sweep_angle: Wing sweep angle in degrees (default: 25°)
        wing_twist: Washout angle in degrees (default: -2°, negative = washout)
    
    Returns:
        Dictionary with design parameters and analysis
    
    Theoretical Background:
        - All components contained within wing envelope
        - Highest lift-to-drag ratio of any configuration
        - Requires special airfoil with reflex camber for pitch stability
        - Sweep provides directional stability
        
    Aerodynamic Characteristics:
        - Reflex airfoil creates nose-up pitching moment
        - Washout (twist) prevents tip stalling
        - Sweep angle provides weathercock stability
        - Elevons control pitch and roll
        - Winglets or vertical fins for yaw stability
        
    Stability Considerations:
        - No tail for pitch stability - relies on reflexed airfoil
        - Sweep and dihedral provide lateral stability
        - CG must be precisely at neutral point (~25-30% MAC)
        - Narrow CG range (typically ±5mm)
        
    Construction Principles:
        1. Wing Structure:
           - Central "fuselage" section (150-200mm wide)
           - Tapered outer panels
           - Main spar through thickest section
           - Torsion box critical for rigidity
           
        2. Airfoil Selection:
           - Must have reflex camber (e.g., Eppler 325, MH-45)
           - Center section: 12-15% thick
           - Tip section: 8-10% thick
           - Washout: -2° to -3° (tips have less incidence)
           
        3. Electronics Bay:
           - Carved into center section
           - Battery position adjustable for CG tuning
           - Access hatch on top or bottom
           
        4. Control Surfaces:
           - Elevons: 25-30% chord, 40-50% span
           - Consider split flaps for speed control
           - Winglets with rudders optional
    """
    
    sweep_rad = math.radians(sweep_angle)
    
    # Wing geometry (assumed tapered planform)
    tip_chord = center_chord * 0.4  # 40% taper
    semi_span = wingspan / 2
    
    # Wing area
    area = ((center_chord + tip_chord) / 2) * wingspan
    
    # Aspect ratio
    aspect_ratio = (wingspan ** 2) / area
    
    # Mean aerodynamic chord
    mac = (2/3) * center_chord * ((1 + tip_chord/center_chord + (tip_chord/center_chord)**2) / 
                                   (1 + tip_chord/center_chord))
    
    # Center section width (fuselage replacement)
    center_width = 150  # mm, typical for small flying wing
    
    # CG location (very critical for flying wings!)
    cg_position = 0.28 * mac  # 25-30% MAC is typical
    cg_tolerance = 5  # ±5mm tolerance (very tight!)
    
    # Elevon sizing
    elevon_chord = 0.30 * center_chord
    elevon_span = (wingspan - center_width) * 0.45  # 45% of each panel
    
    # Reflex airfoil characteristics
    reflex_amount = 2.0  # degrees of reflex at trailing edge
    
    # Performance advantages
    drag_reduction = 0.15  # 15% less drag than conventional
    efficiency_gain = 1.20  # 20% better L/D ratio
    
    return {
        "type": "Flying Wing",
        "geometry": {
            "center_chord_mm": center_chord,
            "tip_chord_mm": tip_chord,
            "wingspan_mm": wingspan,
            "sweep_angle_deg": sweep_angle,
            "area_mm2": area,
            "aspect_ratio": aspect_ratio,
            "mac_mm": mac,
            "center_width_mm": center_width,
            "wing_twist_deg": wing_twist,
        },
        "aerodynamics": {
            "recommended_cg_mm": cg_position,
            "cg_tolerance_mm": cg_tolerance,
            "reflex_angle_deg": reflex_amount,
            "drag_reduction": drag_reduction,
            "efficiency_gain": efficiency_gain,
        },
        "control_surfaces": {
            "elevon_chord_mm": elevon_chord,
            "elevon_span_mm": elevon_span,
            "requires_mixing": True,
        },
        "structure": {
            "construction_method": "Hot-wire cut foam core with composite skin",
            "airfoil_requirement": "Reflex camber (e.g., Eppler E325, MH-45)",
            "spar_location": "30-35% chord through center section",
        },
        "notes": [
            "Requires reflex airfoil for pitch stability",
            "CG location is CRITICAL - must be precise",
            "Washout prevents tip stalling",
            "Highest aerodynamic efficiency",
            "Best for long-range FPV and soaring",
            "Consider winglets for directional stability",
        ]
    }


# ============================================================================
# CANARD
# ============================================================================

def canard_design(
    main_wing_chord=200,
    main_wingspan=1000,
    canard_chord=80,
    canard_span=400,
    canard_position=150
):
    """
    Canard configuration design and analysis.
    
    Canard aircraft have a small forward wing (canard) ahead of the main wing,
    providing inherent pitch stability and stall resistance.
    
    Args:
        main_wing_chord: Main wing chord in mm (default: 200mm)
        main_wingspan: Main wing span in mm (default: 1000mm)
        canard_chord: Canard chord in mm (default: 80mm)
        canard_span: Canard span in mm (default: 400mm)
        canard_position: Distance ahead of main wing in mm (default: 150mm)
    
    Returns:
        Dictionary with design parameters and analysis
    
    Theoretical Background:
        - Canard provides pitch control and stability
        - Stall-proof design: canard stalls before main wing
        - Both surfaces generate lift (more efficient than tail)
        - Forward CG relative to main wing
        
    Aerodynamic Characteristics:
        - Canard generates download in cruise (trim drag)
        - Must stall before main wing for safety
        - Typically has higher incidence angle than main wing
        - Total lift distributed between canard and main wing
        - Close-coupled canards improve stall characteristics
        
    Stability Considerations:
        - Static margin provided by forward surface
        - CG typically at 15-25% of main wing MAC
        - Canard must have lower stall angle than main wing
        - Surface area ratio typically 0.15-0.25 (canard/main)
        
    Construction Principles:
        1. Layout:
           - Canard ahead of CG, main wing behind
           - Typical spacing: 1-2× main wing chord
           - Fuselage connects both surfaces
           - Pusher prop common (keeps prop clear)
           
        2. Wing Design:
           - Canard: Flat bottom airfoil, high lift
           - Main wing: Conventional airfoil
           - Incidence: Canard 2-4° higher than main wing
           - Dihedral: Canard often has anhedral (negative)
           
        3. Structure:
           - Strong fuselage to connect wings
           - Canard mounting reinforced
           - Main wing carries majority of load
           - Consider removable canard for transport
           
        4. Control Surfaces:
           - Canard has elevators (pitch control)
           - Main wing has ailerons (roll control)
           - Optional rudder for yaw control
    """
    
    # Main wing parameters
    main_area = main_wingspan * main_wing_chord
    main_aspect_ratio = (main_wingspan ** 2) / main_area
    main_mac = main_wing_chord  # Simplified for rectangular wing
    
    # Canard parameters
    canard_area = canard_span * canard_chord
    canard_aspect_ratio = (canard_span ** 2) / canard_area
    
    # Area ratio (critical parameter)
    area_ratio = canard_area / main_area
    
    # Moment arm
    moment_arm = canard_position
    
    # Canard volume coefficient
    # V_c = (S_c × l_c) / (S_w × MAC_w)
    canard_volume = (canard_area * moment_arm) / (main_area * main_mac)
    
    # Typical values: 0.04 - 0.08 for canards
    
    # CG position relative to main wing
    cg_position = 0.20 * main_mac  # 15-25% of main wing MAC
    
    # Incidence angle difference (canard higher to stall first)
    incidence_diff = 3.0  # degrees
    
    # Lift distribution (approximate)
    canard_lift_fraction = 0.25  # Canard carries ~25% of lift
    main_lift_fraction = 0.75   # Main wing carries ~75%
    
    # Performance characteristics
    stall_safety = True  # Canard stalls first
    efficiency = 1.05    # 5% more efficient than tail (both surfaces lift)
    
    return {
        "type": "Canard",
        "main_wing": {
            "chord_mm": main_wing_chord,
            "span_mm": main_wingspan,
            "area_mm2": main_area,
            "aspect_ratio": main_aspect_ratio,
            "mac_mm": main_mac,
        },
        "canard": {
            "chord_mm": canard_chord,
            "span_mm": canard_span,
            "area_mm2": canard_area,
            "aspect_ratio": canard_aspect_ratio,
            "position_ahead_mm": canard_position,
        },
        "configuration": {
            "area_ratio": area_ratio,
            "canard_volume_coefficient": canard_volume,
            "moment_arm_mm": moment_arm,
            "cg_position_mm": cg_position,
            "incidence_difference_deg": incidence_diff,
        },
        "aerodynamics": {
            "canard_lift_fraction": canard_lift_fraction,
            "main_lift_fraction": main_lift_fraction,
            "stall_safety": stall_safety,
            "efficiency_vs_tail": efficiency,
        },
        "structure": {
            "construction_method": "Strong fuselage boom connecting surfaces",
            "canard_mounting": "Reinforced fuselage section",
            "propulsion": "Pusher configuration recommended",
        },
        "notes": [
            "Canard MUST stall before main wing for safety",
            "Both surfaces generate lift (more efficient)",
            "Excellent visibility and CG range",
            "Stall-resistant design",
            "Pusher prop keeps propeller clear",
            "Canard should have 2-4° more incidence than main wing",
        ]
    }


# ============================================================================
# OBLIQUE WING
# ============================================================================

def oblique_wing_design(
    wingspan=1000,
    chord=180,
    sweep_angle=0,
    pivot_position=0.5
):
    """
    Oblique wing (pivoting) design and analysis.
    
    Oblique wings pivot to achieve variable sweep, optimizing for different
    flight speeds. One wing sweeps forward, the other backward.
    
    Args:
        wingspan: Total wingspan in mm (default: 1000mm)
        chord: Wing chord in mm (default: 180mm)
        sweep_angle: Current sweep angle in degrees (default: 0° = straight)
        pivot_position: Pivot point as fraction of span (default: 0.5 = center)
    
    Returns:
        Dictionary with design parameters and analysis
    
    Theoretical Background:
        - Wing pivots about central axis
        - One side sweeps forward, other backward
        - Reduces wave drag at high speeds
        - Straight wing for low-speed flight
        - Asymmetric configuration in sweep
        
    Aerodynamic Characteristics:
        - Variable sweep optimizes for speed range
        - Straight (0°): Best low-speed performance
        - Swept (30-45°): Reduced drag at high speed
        - Asymmetric lift distribution when swept
        - Requires careful trim in swept configuration
        
    Stability Considerations:
        - Asymmetric configuration creates rolling moment
        - Requires aileron trim in swept mode
        - Yaw-roll coupling when swept
        - CG shifts with sweep angle
        - Flight control complexity increases
        
    Construction Principles:
        1. Pivot Mechanism:
           - Strong central pivot bearing
           - Servo-actuated rotation (slow movement)
           - Locking mechanism for each sweep angle
           - Reinforced pivot area
           
        2. Wing Structure:
           - Symmetric airfoil (works both directions)
           - Strong main spar through pivot point
           - Minimal taper (avoid tip loading)
           - Carbon fiber spar essential
           
        3. Control System:
           - Ailerons functional at all sweep angles
           - Fly-by-wire mixing for trim
           - Position feedback for sweep angle
           - Emergency lock in neutral position
           
        4. Fuselage:
           - Cylindrical or streamlined body
           - Wide enough for wing pivot range
           - Reinforced pivot mounting points
           - Consider retractable landing gear
    """
    
    sweep_rad = math.radians(sweep_angle)
    
    # Wing area (constant regardless of sweep)
    area = wingspan * chord
    
    # Effective span changes with sweep
    effective_span = wingspan * math.cos(sweep_rad)
    
    # Aspect ratio (changes with sweep)
    aspect_ratio = (effective_span ** 2) / area
    
    # Pivot location
    pivot_location = wingspan * pivot_position
    
    # Forward and aft sweep distances
    forward_sweep = pivot_location * math.sin(sweep_rad)
    aft_sweep = (wingspan - pivot_location) * math.sin(sweep_rad)
    
    # Asymmetric lift distribution
    # Forward-swept side generates more lift at same angle
    lift_asymmetry = math.sin(sweep_rad) * 0.15  # ~15% difference at 30°
    
    # Rolling moment coefficient
    rolling_moment = lift_asymmetry * wingspan / 2
    
    # Performance at different sweep angles
    performance = {
        "0_deg": {"speed_factor": 1.0, "efficiency": 1.0, "description": "Best low-speed"},
        "20_deg": {"speed_factor": 1.1, "efficiency": 0.95, "description": "Cruise"},
        "30_deg": {"speed_factor": 1.2, "efficiency": 0.90, "description": "High-speed"},
        "45_deg": {"speed_factor": 1.35, "efficiency": 0.85, "description": "Maximum speed"},
    }
    
    return {
        "type": "Oblique Wing",
        "geometry": {
            "wingspan_mm": wingspan,
            "chord_mm": chord,
            "area_mm2": area,
            "current_sweep_deg": sweep_angle,
            "effective_span_mm": effective_span,
            "aspect_ratio": aspect_ratio,
            "pivot_position": pivot_position,
        },
        "asymmetry": {
            "forward_sweep_mm": forward_sweep,
            "aft_sweep_mm": aft_sweep,
            "lift_asymmetry_percent": lift_asymmetry * 100,
            "rolling_moment": rolling_moment,
        },
        "performance_envelope": performance,
        "control_system": {
            "aileron_mixing_required": True,
            "trim_adjustment_required": True,
            "sweep_actuation": "Servo with position feedback",
            "emergency_lock": "Essential safety feature",
        },
        "structure": {
            "pivot_mechanism": "High-strength bearing with locking positions",
            "spar_type": "Carbon fiber tube through pivot",
            "construction_method": "Composite or 3D printed ribs with carbon spar",
            "airfoil": "Symmetric (must work in both sweep directions)",
        },
        "notes": [
            "EXPERIMENTAL DESIGN - High complexity",
            "Variable sweep optimizes for speed range",
            "Asymmetric configuration requires trim",
            "Strong pivot mechanism essential",
            "Best for speed range demonstrations",
            "Consider starting with fixed oblique angle",
            "Fly-by-wire control recommended",
        ]
    }


# ============================================================================
# FLYING PANCAKE (Round Wing)
# ============================================================================

def flying_pancake_design(
    diameter=600,
    thickness_ratio=0.12,
    center_cutout=120
):
    """
    Flying pancake (circular wing) design and analysis.
    
    A fun and unusual design featuring a circular or elliptical wing planform.
    Based on the historic Vought V-173 "Flying Pancake" design.
    
    Args:
        diameter: Wing diameter in mm (default: 600mm)
        thickness_ratio: Airfoil thickness ratio (default: 0.12)
        center_cutout: Center fuselage cutout diameter in mm (default: 120mm)
    
    Returns:
        Dictionary with design parameters and analysis
    
    Theoretical Background:
        - Circular or nearly circular planform
        - Very low aspect ratio (<2)
        - High induced drag but excellent maneuverability
        - Compact design with large wing area
        - Tip vortices meet at rear, reducing interference
        
    Aerodynamic Characteristics:
        - Low aspect ratio = high induced drag
        - Excellent slow-speed handling
        - Very stable due to large area and low wing loading
        - Gentle stall characteristics
        - High power required due to drag
        - Can fly at extreme angles of attack
        
    Stability Considerations:
        - Highly stable in all axes
        - Large wing area provides damping
        - Short moment arms (compact)
        - CG typically at geometric center
        - Docile flight characteristics
        
    Construction Principles:
        1. Wing Shape:
           - Circular or slightly elliptical planform
           - Constant or slightly varying chord
           - Thick airfoil (12-15%) for structure
           - Beveled leading edge recommended
           
        2. Structure:
           - Radial ribs from center to perimeter
           - Ring spar at 60-70% radius
           - Plywood or foam core
           - 3D printed center section
           
        3. Control Surfaces:
           - Elevons on rear 30% of perimeter
           - 4-6 control surfaces around perimeter
           - Central section houses electronics
           - Propellers at wing tips (original design)
           
        4. Power System:
           - Twin motors at wing tips (reduces tip vortex)
           - OR single pusher at rear
           - High power required (low efficiency)
           - Large props for static thrust
    """
    
    # Geometric parameters
    radius = diameter / 2
    
    # Wing area (circle minus center cutout)
    total_area = math.pi * (radius ** 2)
    cutout_area = math.pi * ((center_cutout / 2) ** 2)
    wing_area = total_area - cutout_area
    
    # Equivalent "wingspan" (diameter)
    wingspan = diameter
    
    # Aspect ratio (very low for circular wing)
    # AR = b² / S
    # Note: For a solid circle, AR ≈ 4/π ≈ 1.27 (theoretical)
    # With center cutout, the actual AR will be slightly higher
    aspect_ratio = (wingspan ** 2) / wing_area
    
    # Mean chord (approximate)
    mean_chord = wing_area / diameter
    
    # Perimeter (for control surface calculation)
    perimeter = math.pi * diameter
    
    # Control surface sizing (rear 120° arc)
    control_arc = 120  # degrees
    control_chord = mean_chord * 0.30
    control_length = (control_arc / 360) * perimeter
    
    # CG location (at geometric center)
    cg_x = 0  # Center
    cg_y = 0  # Center
    
    # Airfoil thickness
    max_thickness = mean_chord * thickness_ratio
    
    # Performance characteristics
    drag_coefficient = 0.12  # High due to low AR
    stall_speed_factor = 0.7  # 30% lower than conventional (large area)
    efficiency_factor = 0.6   # 40% less efficient (high drag)
    
    # Structural ribs (radial)
    num_ribs = 12  # Radial ribs every 30°
    
    return {
        "type": "Flying Pancake (Circular Wing)",
        "geometry": {
            "diameter_mm": diameter,
            "radius_mm": radius,
            "total_area_mm2": total_area,
            "wing_area_mm2": wing_area,
            "center_cutout_mm": center_cutout,
            "aspect_ratio": aspect_ratio,
            "mean_chord_mm": mean_chord,
            "perimeter_mm": perimeter,
        },
        "aerodynamics": {
            "drag_coefficient": drag_coefficient,
            "stall_speed_factor": stall_speed_factor,
            "efficiency_factor": efficiency_factor,
            "cg_position": {"x_mm": cg_x, "y_mm": cg_y},
            "flight_characteristics": "Stable, docile, but draggy",
        },
        "control_surfaces": {
            "control_arc_deg": control_arc,
            "control_chord_mm": control_chord,
            "control_length_mm": control_length,
            "number_of_elevons": 4,
            "description": "4 elevons around rear perimeter",
        },
        "structure": {
            "num_radial_ribs": num_ribs,
            "ring_spar_radius_mm": radius * 0.65,
            "max_thickness_mm": max_thickness,
            "construction_method": "Radial ribs with ring spar, foam or plywood core",
            "center_section": "3D printed hub for electronics",
        },
        "propulsion": {
            "recommended": "Twin tip-mounted motors (historical) OR single pusher",
            "power_requirement": "High (1.5-2× normal due to drag)",
            "prop_size": "Large diameter for static thrust",
        },
        "notes": [
            "FUN EXPERIMENTAL DESIGN - Historical curiosity",
            "Based on Vought V-173 'Flying Pancake'",
            "Very stable and easy to fly",
            "High drag = low efficiency",
            "Excellent for demonstrations and fun flying",
            "Can fly at extreme angles of attack",
            "Consider twin tip motors for authentic look",
            "Great conversation starter!",
        ]
    }


# ============================================================================
# CAD Generation Functions (Optional)
# ============================================================================

def generate_delta_wing_ribs(
    root_chord=400,
    tip_chord=40,
    num_ribs=8,
    thickness=6
):
    """
    Generate ribs for a delta wing (requires CadQuery).
    
    Args:
        root_chord: Root chord in mm
        tip_chord: Tip chord in mm
        num_ribs: Number of ribs to generate
        thickness: Rib thickness in mm
    
    Returns:
        List of CadQuery workplanes (one per rib)
    
    Raises:
        ImportError: If CadQuery is not installed
    """
    if not HAS_CADQUERY:
        raise ImportError("CadQuery required for STL generation")
    
    ribs = []
    
    for i in range(num_ribs):
        # Interpolate chord at this rib position
        fraction = i / (num_ribs - 1)
        chord = root_chord * (1 - fraction) + tip_chord * fraction
        
        # Simple symmetric airfoil
        rib = (
            cq.Workplane("XY")
            .polyline([
                (0, 0),
                (chord, 0),
                (chord * 0.95, chord * 0.08),
                (chord * 0.5, chord * 0.10),
                (chord * 0.05, chord * 0.08),
                (0, 0)
            ])
            .close()
            .extrude(thickness)
        )
        
        ribs.append(rib)
    
    return ribs


def generate_flying_pancake_ribs(
    diameter=600,
    num_ribs=12,
    thickness=6,
    airfoil_thickness=0.12
):
    """
    Generate radial ribs for a flying pancake (requires CadQuery).
    
    Args:
        diameter: Wing diameter in mm
        num_ribs: Number of radial ribs
        thickness: Rib thickness in mm
        airfoil_thickness: Airfoil thickness ratio
    
    Returns:
        List of CadQuery workplanes (one per rib)
    
    Raises:
        ImportError: If CadQuery is not installed
    """
    if not HAS_CADQUERY:
        raise ImportError("CadQuery required for STL generation")
    
    ribs = []
    radius = diameter / 2
    chord = radius  # Radial ribs extend from center to edge
    max_thick = chord * airfoil_thickness
    
    for i in range(num_ribs):
        # Radial rib (symmetric airfoil)
        rib = (
            cq.Workplane("XY")
            .polyline([
                (0, 0),
                (chord, 0),
                (chord * 0.95, max_thick * 0.5),
                (chord * 0.5, max_thick),
                (chord * 0.05, max_thick * 0.5),
                (0, 0)
            ])
            .close()
            .extrude(thickness)
        )
        
        ribs.append(rib)
    
    return ribs


# ============================================================================
# TRADITIONAL WING TYPES
# ============================================================================

def straight_wing_design(
    wingspan=1200,
    chord=200,
    taper_ratio=0.7,
    dihedral=3,
    thickness_ratio=0.12
):
    """
    Straight wing (rectangular or tapered) design and analysis.
    
    Straight wings are the most common and traditional wing configuration,
    used in most general aviation aircraft, trainers, and sport planes.
    
    Args:
        wingspan: Total wingspan in mm (default: 1200mm)
        chord: Root chord length in mm (default: 200mm)
        taper_ratio: Tip chord / root chord ratio (default: 0.7, 1.0 = rectangular)
        dihedral: Dihedral angle in degrees (default: 3°)
        thickness_ratio: Airfoil thickness ratio (default: 0.12)
    
    Returns:
        Dictionary with design parameters and analysis
    
    Theoretical Background:
        - Zero sweep angle (perpendicular to fuselage)
        - Most efficient at subsonic speeds
        - Simple to design and build
        - Excellent low-speed characteristics
        - Standard configuration for trainers
        
    Aerodynamic Characteristics:
        - Maximum lift coefficient at low speeds
        - Linear lift curve (predictable behavior)
        - Efficient wing loading distribution
        - Simple stall characteristics
        - Moderate induced drag
        
    Stability Considerations:
        - Dihedral provides roll stability
        - Conventional CG location (25-30% MAC)
        - Standard tail required for pitch control
        - Predictable and stable flight
        - Wide CG range (forgiving)
        
    Construction Principles:
        1. Wing Structure:
           - Main spar at 25-30% chord
           - Rear spar at 60-70% chord
           - Ribs perpendicular to span
           - Torsion box between spars
           
        2. Taper Options:
           - Rectangular (taper=1.0): Simplest, easier to build
           - Tapered (taper=0.6-0.8): More efficient, less tip loading
           - Elliptical (complex): Most efficient but hard to build
           
        3. Construction Methods:
           - Foam core with composite skin
           - Balsa/plywood built-up structure
           - 3D printed ribs with spar
           - Foam board for simple models
           
        4. Dihedral:
           - 2-4° typical for trainers
           - Increases roll stability
           - Too much = dutch roll tendency
    """
    
    # Calculate tip chord
    tip_chord = chord * taper_ratio
    
    # Mean chord (for tapered wing)
    mean_chord = (chord + tip_chord) / 2
    
    # Wing area
    wing_area = wingspan * mean_chord
    
    # Aspect ratio
    aspect_ratio = (wingspan ** 2) / wing_area
    
    # Mean aerodynamic chord (MAC)
    mac = (2/3) * chord * ((1 + taper_ratio + taper_ratio**2) / (1 + taper_ratio))
    
    # CG location (standard)
    cg_position = 0.275 * mac  # 25-30% MAC typical
    
    # Control surfaces
    aileron_chord = 0.25 * mean_chord
    aileron_span = wingspan * 0.35  # 35% of span per side
    
    elevator_chord = 0.30 * mac  # Horizontal tail sizing
    rudder_area = wing_area * 0.08  # 8% of wing area
    
    # Structural sizing
    max_thickness = chord * thickness_ratio
    spar_depth = max_thickness * 0.65
    
    # Dihedral effect
    dihedral_rad = math.radians(dihedral)
    effective_span_vertical = wingspan * math.sin(dihedral_rad)
    
    # Performance characteristics
    stall_speed_factor = 1.0  # Baseline (best low-speed performance)
    cruise_efficiency = 1.0   # Baseline efficiency
    
    return {
        "type": "Straight Wing",
        "geometry": {
            "wingspan_mm": wingspan,
            "root_chord_mm": chord,
            "tip_chord_mm": tip_chord,
            "mean_chord_mm": mean_chord,
            "mac_mm": mac,
            "wing_area_mm2": wing_area,
            "aspect_ratio": aspect_ratio,
            "taper_ratio": taper_ratio,
            "dihedral_deg": dihedral,
            "dihedral_height_mm": effective_span_vertical,
        },
        "aerodynamics": {
            "cg_position_mm": cg_position,
            "cg_range_percent_mac": "25-35%",
            "stall_speed_factor": stall_speed_factor,
            "cruise_efficiency": cruise_efficiency,
            "lift_distribution": "Relatively uniform (tapered) or tip-heavy (rectangular)",
        },
        "control_surfaces": {
            "aileron_chord_mm": aileron_chord,
            "aileron_span_mm": aileron_span,
            "elevator_chord_mm": elevator_chord,
            "rudder_area_mm2": rudder_area,
        },
        "structure": {
            "main_spar_location": "25-30% chord",
            "rear_spar_location": "60-70% chord",
            "max_thickness_mm": max_thickness,
            "spar_depth_mm": spar_depth,
            "construction_method": "Foam core, balsa build-up, or 3D printed ribs with spar",
        },
        "notes": [
            "Most common and traditional wing type",
            "Excellent low-speed characteristics",
            "Simple to design and build",
            "Predictable and stable flight",
            "Wide CG range (forgiving)",
            "Requires conventional tail surfaces",
            "Best for trainers and sport aircraft",
        ]
    }


def backward_swept_wing_design(
    wingspan=1200,
    chord=200,
    sweep_angle=25,
    taper_ratio=0.6,
    thickness_ratio=0.10
):
    """
    Backward swept wing design and analysis.
    
    Backward (aft) swept wings have leading edges angled backward from root to tip.
    Used in high-speed aircraft to delay shock wave formation and reduce drag.
    
    Args:
        wingspan: Total wingspan in mm (default: 1200mm)
        chord: Root chord length in mm (default: 200mm)
        sweep_angle: Sweep angle in degrees (default: 25°)
        taper_ratio: Tip chord / root chord ratio (default: 0.6)
        thickness_ratio: Airfoil thickness ratio (default: 0.10)
    
    Returns:
        Dictionary with design parameters and analysis
    
    Theoretical Background:
        - Leading edge swept backward (toward tail)
        - Delays shock wave formation at high speeds
        - Reduces effective Mach number on wing
        - Common in jets and high-speed aircraft
        - Sweep angle typically 20-45°
        
    Aerodynamic Characteristics:
        - Reduced drag at high speeds
        - Lower critical Mach number impact
        - Spanwise flow component
        - Tendency for tip stalling
        - Reduced lift curve slope
        
    Stability Considerations:
        - Natural pitch-up tendency at stall
        - Tip stalling tendency (dangerous)
        - Dutch roll mode possible
        - Requires washout to prevent tip stall
        - CG typically at 30-35% MAC
        
    Construction Principles:
        1. Wing Structure:
           - Spar perpendicular to fuselage (not wing)
           - Strong root attachment
           - Washout essential (-2° to -3°)
           - Torsion box critical for stiffness
           
        2. Stall Prevention:
           - MUST have washout (geometric twist)
           - Wing fences or vortex generators
           - Leading edge slots at tips
           - Root stall before tip essential
           
        3. Control Surfaces:
           - Outboard ailerons for roll
           - All-moving tail or large elevator
           - Yaw damper helpful
           
        4. Build Considerations:
           - More complex than straight wing
           - Requires accurate alignment
           - Washout must be built in
           - Strong wing-fuselage joint
    """
    
    sweep_rad = math.radians(sweep_angle)
    tip_chord = chord * taper_ratio
    
    # Effective span (perpendicular to flow)
    effective_span = wingspan * math.cos(sweep_rad)
    
    # Wing area (trapezoidal approximation)
    wing_area = ((chord + tip_chord) / 2) * wingspan
    
    # Aspect ratio (based on actual span)
    aspect_ratio = (wingspan ** 2) / wing_area
    
    # Effective aspect ratio (based on sweep)
    effective_ar = aspect_ratio * (math.cos(sweep_rad) ** 2)
    
    # Mean aerodynamic chord
    mac = (2/3) * chord * ((1 + taper_ratio + taper_ratio**2) / (1 + taper_ratio))
    
    # CG location (aft of straight wing due to sweep)
    cg_position = 0.32 * mac  # 30-35% MAC for swept wings
    
    # Quarter-chord sweep (aerodynamic reference)
    quarter_chord_sweep = math.degrees(math.atan(
        math.tan(sweep_rad) - (1 - taper_ratio) / (aspect_ratio * (1 + taper_ratio))
    ))
    
    # Control surfaces
    aileron_chord = 0.25 * tip_chord  # Outboard, smaller chord
    aileron_span = wingspan * 0.30  # 30% span per side (outboard)
    
    # Required washout
    washout_angle = -2.5  # degrees (negative = tips at lower angle)
    
    # Structural considerations
    max_thickness = chord * thickness_ratio
    spar_depth = max_thickness * 0.60
    
    # Performance characteristics
    high_speed_advantage = 1.15  # 15% speed increase vs straight wing
    low_speed_penalty = 0.92     # 8% higher stall speed
    
    return {
        "type": "Backward Swept Wing",
        "geometry": {
            "wingspan_mm": wingspan,
            "root_chord_mm": chord,
            "tip_chord_mm": tip_chord,
            "sweep_angle_deg": sweep_angle,
            "quarter_chord_sweep_deg": quarter_chord_sweep,
            "wing_area_mm2": wing_area,
            "aspect_ratio": aspect_ratio,
            "effective_aspect_ratio": effective_ar,
            "taper_ratio": taper_ratio,
            "mac_mm": mac,
        },
        "aerodynamics": {
            "cg_position_mm": cg_position,
            "required_washout_deg": washout_angle,
            "spanwise_flow": "Present - toward tips",
            "tip_stall_tendency": "HIGH - washout essential",
            "high_speed_advantage": high_speed_advantage,
            "low_speed_penalty": low_speed_penalty,
        },
        "control_surfaces": {
            "aileron_chord_mm": aileron_chord,
            "aileron_span_mm": aileron_span,
            "aileron_position": "Outboard (60-90% span)",
            "tail_required": "Yes - conventional or T-tail",
        },
        "structure": {
            "main_spar_orientation": "Perpendicular to fuselage centerline",
            "washout_required": True,
            "washout_amount_deg": washout_angle,
            "max_thickness_mm": max_thickness,
            "spar_depth_mm": spar_depth,
            "construction_method": "Foam core with twist jig, or CNC cut ribs at different angles",
        },
        "stability": {
            "pitch_up_tendency": "Present at stall",
            "dutch_roll_susceptibility": "Moderate to high",
            "stall_characteristics": "Must prevent tip stall with washout",
            "wing_fences_recommended": True,
        },
        "notes": [
            "CRITICAL: Must have washout to prevent tip stalling",
            "Excellent high-speed performance",
            "More complex to build than straight wing",
            "Requires accurate alignment and twist",
            "Wing fences or vortex generators recommended",
            "Best for high-speed sport and scale jets",
            "Not recommended for beginners",
        ]
    }


def forward_swept_wing_design(
    wingspan=1200,
    chord=200,
    sweep_angle=25,
    taper_ratio=0.6,
    thickness_ratio=0.10
):
    """
    Forward swept wing design and analysis.
    
    Forward swept wings have leading edges angled forward from root to tip.
    Unusual configuration with unique advantages but challenging to build.
    
    Args:
        wingspan: Total wingspan in mm (default: 1200mm)
        chord: Root chord length in mm (default: 200mm)
        sweep_angle: Forward sweep angle in degrees (default: 25°)
        taper_ratio: Tip chord / root chord ratio (default: 0.6)
        thickness_ratio: Airfoil thickness ratio (default: 0.10)
    
    Returns:
        Dictionary with design parameters and analysis
    
    Theoretical Background:
        - Leading edge swept forward (toward nose)
        - Promotes root stalling (safer than tip stall)
        - Superior maneuverability
        - Structural divergence tendency
        - Requires very stiff wing structure
        
    Aerodynamic Characteristics:
        - Natural root stall (safer than backward sweep)
        - Spanwise flow toward root
        - Excellent maneuverability
        - Good low-speed handling
        - Vortex lift at high angles
        
    Stability Considerations:
        - Root stalls first (SAFE stall behavior)
        - Natural washout effect from geometry
        - Better stall characteristics than backward sweep
        - Aeroelastic divergence risk (requires stiff structure)
        - CG typically at 28-32% MAC
        
    Construction Principles:
        1. Structural Requirements:
           - CRITICAL: Very stiff wing required
           - Torsional rigidity essential
           - Carbon fiber spar mandatory (no aluminum/wood)
           - Double spar or D-box torsion structure
           - Failure mode is catastrophic divergence
           
        2. Stiffness Requirements:
           - 2-3× stiffer than backward swept wing
           - Torsion box from LE to 70% chord
           - Carbon fiber skin recommended
           - No flexible construction methods
           
        3. Advantages:
           - Safe stall (root first)
           - Excellent maneuverability
           - Superior roll rate
           - Unique appearance
           
        4. Challenges:
           - Difficult to build stiff enough
           - Expensive materials required
           - Structural testing essential
           - Not for beginners!
    """
    
    sweep_rad = math.radians(sweep_angle)
    tip_chord = chord * taper_ratio
    
    # Effective span
    effective_span = wingspan * math.cos(sweep_rad)
    
    # Wing area
    wing_area = ((chord + tip_chord) / 2) * wingspan
    
    # Aspect ratio
    aspect_ratio = (wingspan ** 2) / wing_area
    effective_ar = aspect_ratio * (math.cos(sweep_rad) ** 2)
    
    # Mean aerodynamic chord
    mac = (2/3) * chord * ((1 + taper_ratio + taper_ratio**2) / (1 + taper_ratio))
    
    # CG location
    cg_position = 0.30 * mac  # 28-32% MAC
    
    # Quarter-chord sweep (negative for forward sweep)
    quarter_chord_sweep = -math.degrees(math.atan(
        math.tan(sweep_rad) + (1 - taper_ratio) / (aspect_ratio * (1 + taper_ratio))
    ))
    
    # Control surfaces
    aileron_chord = 0.25 * chord  # Inboard, larger chord
    aileron_span = wingspan * 0.35  # 35% span per side
    
    # Structural requirements (much stiffer than backward sweep)
    stiffness_requirement = 2.5  # 2.5× normal requirement
    max_thickness = chord * thickness_ratio
    spar_depth = max_thickness * 0.70  # Deeper spar needed
    
    # Torsion box requirement
    torsion_box_extent = 0.70  # 70% of chord (vs 60% for other wings)
    
    # Performance characteristics
    maneuverability_gain = 1.25  # 25% better roll rate
    stall_safety = "Excellent"   # Root stalls first
    
    return {
        "type": "Forward Swept Wing",
        "geometry": {
            "wingspan_mm": wingspan,
            "root_chord_mm": chord,
            "tip_chord_mm": tip_chord,
            "sweep_angle_deg": -sweep_angle,  # Negative indicates forward
            "quarter_chord_sweep_deg": quarter_chord_sweep,
            "wing_area_mm2": wing_area,
            "aspect_ratio": aspect_ratio,
            "effective_aspect_ratio": effective_ar,
            "taper_ratio": taper_ratio,
            "mac_mm": mac,
        },
        "aerodynamics": {
            "cg_position_mm": cg_position,
            "spanwise_flow": "Toward root (beneficial)",
            "stall_behavior": stall_safety,
            "root_stall_first": True,
            "maneuverability_gain": maneuverability_gain,
            "vortex_lift_available": True,
        },
        "control_surfaces": {
            "aileron_chord_mm": aileron_chord,
            "aileron_span_mm": aileron_span,
            "aileron_position": "Inboard (30-65% span)",
            "tail_required": "Yes - conventional",
        },
        "structure": {
            "stiffness_requirement": stiffness_requirement,
            "spar_material": "Carbon fiber REQUIRED (no aluminum or wood)",
            "torsion_box_extent": f"{torsion_box_extent:.0%} of chord",
            "skin_material": "Carbon fiber or carbon/glass hybrid",
            "max_thickness_mm": max_thickness,
            "spar_depth_mm": spar_depth,
            "construction_method": "Composite layup with carbon spar - very stiff construction only",
        },
        "stability": {
            "aeroelastic_divergence_risk": "HIGH - requires extreme stiffness",
            "stall_safety": "Excellent - root stalls first",
            "roll_rate": "Superior to backward sweep",
            "dutch_roll": "Low susceptibility",
        },
        "notes": [
            "ADVANCED DESIGN - Not for beginners!",
            "CRITICAL: Requires extremely stiff structure (carbon fiber)",
            "Root stalls first (safer than backward sweep)",
            "Excellent maneuverability and roll rate",
            "Expensive to build (carbon fiber required)",
            "Structural testing essential before flight",
            "Aeroelastic divergence can cause catastrophic failure",
            "Famous example: Grumman X-29, Su-47",
        ]
    }


# ============================================================================
# Comparison and Selection Helper
# ============================================================================

def compare_wing_types(weight=1400, target_speed=15, purpose="general"):
    """
    Compare all wing types and recommend the best for given requirements.
    
    Args:
        weight: Aircraft weight in grams
        target_speed: Target cruise speed in m/s
        purpose: Flight purpose ("general", "speed", "efficiency", "aerobatic", "fun")
    
    Returns:
        Dictionary with recommendations and comparison
    """
    
    recommendations = {
        "general": {
            "primary": "Straight Wing",
            "alternative": "Canard",
            "reason": "Versatile, stable, easy to build and fly"
        },
        "beginner": {
            "primary": "Straight Wing",
            "alternative": "Straight Wing (rectangular)",
            "reason": "Most forgiving, simple construction, wide CG range"
        },
        "speed": {
            "primary": "Backward Swept Wing",
            "alternative": "Delta Wing",
            "reason": "High-speed capability, reduced drag"
        },
        "efficiency": {
            "primary": "Flying Wing",
            "alternative": "Straight Wing (tapered)",
            "reason": "Maximum L/D ratio, efficient lift generation"
        },
        "aerobatic": {
            "primary": "Delta Wing",
            "alternative": "Forward Swept Wing",
            "reason": "High maneuverability, strong structure"
        },
        "fun": {
            "primary": "Flying Pancake",
            "alternative": "Forward Swept Wing",
            "reason": "Unique appearance, docile handling, conversation starter"
        }
    }
    
    comparison = {
        "Straight Wing": {
            "complexity": "Low",
            "stability": "Excellent",
            "speed": "Medium",
            "efficiency": "Good",
            "build_difficulty": "Easy",
            "best_for": "Trainers, sport aircraft, general flying"
        },
        "Backward Swept Wing": {
            "complexity": "Medium",
            "stability": "Good (requires washout)",
            "speed": "High",
            "efficiency": "Medium-Good",
            "build_difficulty": "Medium",
            "best_for": "High-speed sport, scale jets"
        },
        "Forward Swept Wing": {
            "complexity": "High",
            "stability": "Excellent (root stall)",
            "speed": "Medium-High",
            "efficiency": "Good",
            "build_difficulty": "Very High",
            "best_for": "Advanced aerobatics, experimentation"
        },
        "Delta Wing": {
            "complexity": "Medium",
            "stability": "Good (tailless)",
            "speed": "High",
            "efficiency": "Medium",
            "build_difficulty": "Medium",
            "best_for": "High-speed flight, aerobatics"
        },
        "Flying Wing": {
            "complexity": "High",
            "stability": "Medium (critical CG)",
            "speed": "Medium-High",
            "efficiency": "Excellent",
            "build_difficulty": "High",
            "best_for": "Long-range FPV, efficiency"
        },
        "Canard": {
            "complexity": "Medium",
            "stability": "Excellent",
            "speed": "Medium",
            "efficiency": "Good",
            "build_difficulty": "Medium",
            "best_for": "General flying, trainers"
        },
        "Oblique Wing": {
            "complexity": "Very High",
            "stability": "Medium (requires trim)",
            "speed": "Variable (0.8-1.4×)",
            "efficiency": "Variable",
            "build_difficulty": "Very High",
            "best_for": "Experimentation, demonstrations"
        },
        "Flying Pancake": {
            "complexity": "Low",
            "stability": "Excellent",
            "speed": "Low",
            "efficiency": "Poor",
            "build_difficulty": "Medium",
            "best_for": "Fun flying, demonstrations"
        }
    }
    
    selected = recommendations.get(purpose, recommendations["general"])
    
    return {
        "recommendation": selected,
        "all_comparisons": comparison,
        "notes": [
            "Consider your building skills and experience",
            "Start with simpler designs (Canard, Delta)",
            "Flying Wing requires precise CG placement",
            "Oblique Wing is experimental - not for beginners",
            "Flying Pancake is fun but inefficient"
        ]
    }
