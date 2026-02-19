"""
Wind Tunnel Simulation Module

Simulates aerodynamic behavior of aircraft designs including:
- Lift, drag, and moment calculations
- Pressure distribution
- Flow visualization data
- Stability derivatives
"""

import math
from typing import Dict, List, Tuple


class WindTunnelSimulation:
    """Wind tunnel simulation for aircraft designs"""
    
    # Constants
    AIR_DENSITY = 1.225  # kg/m³ at sea level
    GRAVITY = 9.81  # m/s²
    
    def __init__(self, design_params: Dict):
        """
        Initialize wind tunnel simulation with design parameters.
        
        Args:
            design_params: Dictionary containing:
                - wingspan: Wing span in mm
                - chord: Wing chord in mm
                - wing_area: Wing area in mm²
                - weight: Aircraft weight in grams
                - airfoil_type: Airfoil profile (e.g., 'clark_y', 'symmetric')
                - fuselage_length: Length in mm (optional)
                - fuselage_diameter: Diameter in mm (optional)
        """
        self.params = design_params
        self.wingspan = design_params.get('wingspan', 1000)
        self.chord = design_params.get('chord', 150)
        self.wing_area = design_params.get('wing_area', self.wingspan * self.chord)
        self.weight = design_params.get('weight', 1000)
        self.airfoil_type = design_params.get('airfoil_type', 'clark_y')
        
        # Airfoil-specific stall angles (degrees)
        self.stall_angle = 15 if self.airfoil_type == 'clark_y' else 12
        
        # Calculate aspect ratio
        self.aspect_ratio = (self.wingspan ** 2) / self.wing_area
        
    def calculate_lift_coefficient(self, angle_of_attack: float) -> float:
        """
        Calculate lift coefficient for given angle of attack.
        
        Args:
            angle_of_attack: Angle in degrees
            
        Returns:
            Lift coefficient (CL)
        """
        # Convert to radians
        aoa_rad = math.radians(angle_of_attack)
        
        # Base lift curve slope (per radian) - typical for subsonic flow
        # For Clark-Y: ~5.7 per radian, Symmetric: ~6.0 per radian
        if self.airfoil_type == 'clark_y':
            cl_alpha = 5.7
            cl_0 = 0.3  # Zero-lift angle for cambered airfoil
        else:  # symmetric
            cl_alpha = 6.0
            cl_0 = 0.0
        
        # Finite wing correction (Prandtl's lifting line theory)
        cl_alpha_corrected = cl_alpha / (1 + (cl_alpha / (math.pi * self.aspect_ratio)))
        
        # Calculate CL
        cl = cl_0 + cl_alpha_corrected * aoa_rad
        
        # Stall modeling (simplified)
        if angle_of_attack > self.stall_angle:
            # Post-stall CL drops significantly
            stall_factor = math.cos(math.radians(angle_of_attack - self.stall_angle))
            cl = cl * max(0.3, stall_factor)
        
        return cl
    
    def calculate_drag_coefficient(self, cl: float) -> float:
        """
        Calculate drag coefficient.
        
        Args:
            cl: Lift coefficient
            
        Returns:
            Drag coefficient (CD)
        """
        # Parasite drag coefficient (profile + interference)
        cd_0 = 0.025  # Typical for small UAV
        
        # Induced drag coefficient
        e = 0.8  # Oswald efficiency factor (0.7-0.9 for typical wings)
        cd_i = (cl ** 2) / (math.pi * e * self.aspect_ratio)
        
        # Total drag
        cd = cd_0 + cd_i
        
        return cd
    
    def calculate_moment_coefficient(self, cl: float, angle_of_attack: float) -> float:
        """
        Calculate pitching moment coefficient.
        
        Args:
            cl: Lift coefficient
            angle_of_attack: Angle in degrees
            
        Returns:
            Moment coefficient (CM) about quarter-chord
        """
        # Moment coefficient for cambered airfoils
        if self.airfoil_type == 'clark_y':
            cm_0 = -0.05  # Negative = nose-down moment
        else:
            cm_0 = 0.0  # Symmetric airfoil
        
        # Moment typically decreases slightly with CL
        cm = cm_0 - 0.01 * cl
        
        return cm
    
    def simulate_at_speed(self, speed_ms: float, angle_of_attack: float) -> Dict:
        """
        Run simulation at specified speed and angle of attack.
        
        Args:
            speed_ms: Airspeed in m/s
            angle_of_attack: Angle in degrees
            
        Returns:
            Dictionary with simulation results
        """
        # Convert wing area to m²
        wing_area_m2 = self.wing_area / 1_000_000
        
        # Calculate coefficients
        cl = self.calculate_lift_coefficient(angle_of_attack)
        cd = self.calculate_drag_coefficient(cl)
        cm = self.calculate_moment_coefficient(cl, angle_of_attack)
        
        # Calculate dynamic pressure
        q = 0.5 * self.AIR_DENSITY * (speed_ms ** 2)
        
        # Calculate forces (in Newtons)
        lift_n = cl * q * wing_area_m2
        drag_n = cd * q * wing_area_m2
        
        # Convert to grams-force
        lift_g = lift_n / self.GRAVITY * 1000
        drag_g = drag_n / self.GRAVITY * 1000
        
        # Calculate L/D ratio
        ld_ratio = cl / cd if cd > 0 else 0
        
        # Calculate pitching moment (N⋅m)
        moment_nm = cm * q * wing_area_m2 * (self.chord / 1000)
        
        results = {
            'speed_ms': speed_ms,
            'angle_of_attack': angle_of_attack,
            'cl': cl,
            'cd': cd,
            'cm': cm,
            'lift_n': lift_n,
            'drag_n': drag_n,
            'lift_g': lift_g,
            'drag_g': drag_g,
            'ld_ratio': ld_ratio,
            'moment_nm': moment_nm,
            'dynamic_pressure_pa': q,
            'stalled': angle_of_attack > self.stall_angle
        }
        
        return results
    
    def sweep_angle_of_attack(self, speed_ms: float, aoa_range: Tuple[float, float] = (-5, 20)) -> List[Dict]:
        """
        Sweep through angles of attack at constant speed.
        
        Args:
            speed_ms: Airspeed in m/s
            aoa_range: Tuple of (min_aoa, max_aoa) in degrees
            
        Returns:
            List of simulation results for each angle
        """
        results = []
        
        # Generate angles with 1-degree increments
        for aoa in range(int(aoa_range[0]), int(aoa_range[1]) + 1):
            result = self.simulate_at_speed(speed_ms, float(aoa))
            results.append(result)
        
        return results
    
    def sweep_speed(self, speed_range: Tuple[float, float], aoa: float = 5.0) -> List[Dict]:
        """
        Sweep through speeds at constant angle of attack.
        
        Args:
            speed_range: Tuple of (min_speed, max_speed) in m/s
            aoa: Angle of attack in degrees
            
        Returns:
            List of simulation results for each speed
        """
        results = []
        
        # Generate speeds with 2 m/s increments
        speed = speed_range[0]
        while speed <= speed_range[1]:
            result = self.simulate_at_speed(speed, aoa)
            results.append(result)
            speed += 2.0
        
        return results
    
    def calculate_pressure_distribution(self, angle_of_attack: float, num_points: int = 50) -> Dict:
        """
        Calculate pressure distribution over wing surface.
        
        Args:
            angle_of_attack: Angle in degrees
            num_points: Number of points along chord
            
        Returns:
            Dictionary with upper and lower surface pressure coefficients
        """
        # Simplified pressure distribution using thin airfoil theory
        cl = self.calculate_lift_coefficient(angle_of_attack)
        
        # Generate x positions along chord (0 to 1)
        x_positions = [i / (num_points - 1) for i in range(num_points)]
        
        upper_cp = []
        lower_cp = []
        
        for x in x_positions:
            # Simplified pressure coefficient
            # Upper surface: suction peak near leading edge
            if x < 0.5:
                cp_upper = -cl * (1 - 2 * x) - 0.5 * cl * math.sqrt(x)
            else:
                cp_upper = -cl * (1 - 2 * x) + 0.3 * cl * (x - 0.5)
            
            # Lower surface: positive pressure
            if x < 0.5:
                cp_lower = cl * (1 - 2 * x) + 0.3 * cl * math.sqrt(x)
            else:
                cp_lower = cl * (1 - 2 * x) - 0.2 * cl * (x - 0.5)
            
            upper_cp.append(cp_upper)
            lower_cp.append(cp_lower)
        
        return {
            'x_positions': x_positions,
            'upper_cp': upper_cp,
            'lower_cp': lower_cp,
            'cl': cl
        }
    
    def calculate_trim_condition(self, speed_ms: float, target_weight_g: float = None) -> Dict:
        """
        Calculate trim angle of attack for level flight.
        
        Args:
            speed_ms: Flight speed in m/s
            target_weight_g: Target weight to support (default: design weight)
            
        Returns:
            Dictionary with trim conditions
        """
        if target_weight_g is None:
            target_weight_g = self.weight
        
        # Binary search for trim angle
        aoa_min, aoa_max = -5.0, 15.0
        tolerance = 0.1  # grams
        
        for _ in range(20):  # Max iterations
            aoa_mid = (aoa_min + aoa_max) / 2
            result = self.simulate_at_speed(speed_ms, aoa_mid)
            
            lift_error = result['lift_g'] - target_weight_g
            
            if abs(lift_error) < tolerance:
                return {
                    'trim_aoa': aoa_mid,
                    'trim_cl': result['cl'],
                    'trim_cd': result['cd'],
                    'trim_ld': result['ld_ratio'],
                    'drag_g': result['drag_g'],
                    'converged': True
                }
            
            if lift_error > 0:
                aoa_max = aoa_mid
            else:
                aoa_min = aoa_mid
        
        # Didn't converge
        return {
            'trim_aoa': None,
            'converged': False,
            'message': 'Trim condition not found - check speed and weight'
        }
    
    def estimate_stall_speed(self, weight_g: float = None) -> Dict:
        """
        Estimate stall speed for given weight.
        
        Args:
            weight_g: Aircraft weight (default: design weight)
            
        Returns:
            Dictionary with stall speed information
        """
        if weight_g is None:
            weight_g = self.weight
        
        # Maximum lift coefficient (at stall)
        cl_max = self.calculate_lift_coefficient(14.0)  # Just before stall
        
        # Convert units
        weight_n = weight_g * self.GRAVITY / 1000
        wing_area_m2 = self.wing_area / 1_000_000
        
        # Stall speed: V_stall = sqrt(2 * W / (rho * S * CL_max))
        v_stall = math.sqrt((2 * weight_n) / (self.AIR_DENSITY * wing_area_m2 * cl_max))
        
        # 1.3 * V_stall is typical approach speed
        v_approach = v_stall * 1.3
        
        return {
            'stall_speed_ms': v_stall,
            'approach_speed_ms': v_approach,
            'cl_max': cl_max,
            'weight_g': weight_g
        }
    
    def analyze_stability(self, cruise_speed_ms: float) -> Dict:
        """
        Analyze longitudinal stability characteristics.
        
        Args:
            cruise_speed_ms: Cruise speed in m/s
            
        Returns:
            Dictionary with stability derivatives and metrics
        """
        # Calculate at cruise condition
        trim = self.calculate_trim_condition(cruise_speed_ms)
        
        if not trim.get('converged'):
            return {'stable': False, 'message': 'Could not establish trim'}
        
        aoa_trim = trim['trim_aoa']
        
        # Calculate lift curve slope (dCL/dα)
        delta_aoa = 1.0
        cl_plus = self.calculate_lift_coefficient(aoa_trim + delta_aoa)
        cl_minus = self.calculate_lift_coefficient(aoa_trim - delta_aoa)
        cl_alpha = (cl_plus - cl_minus) / (2 * delta_aoa * math.pi / 180)  # per radian
        
        # Calculate moment curve slope (dCM/dα)
        cm_plus = self.calculate_moment_coefficient(cl_plus, aoa_trim + delta_aoa)
        cm_minus = self.calculate_moment_coefficient(cl_minus, aoa_trim - delta_aoa)
        cm_alpha = (cm_plus - cm_minus) / (2 * delta_aoa * math.pi / 180)  # per radian
        
        # Static margin (should be positive for stability)
        # Simplified: SM = -dCM/dCL
        static_margin = -cm_alpha / cl_alpha if cl_alpha != 0 else 0
        
        # Stability assessment
        stable = static_margin > 0.05  # At least 5% static margin
        
        return {
            'stable': stable,
            'static_margin': static_margin,
            'cl_alpha': cl_alpha,
            'cm_alpha': cm_alpha,
            'trim_aoa': aoa_trim,
            'trim_cl': trim['trim_cl'],
            'assessment': 'Stable' if stable else 'Unstable or marginally stable'
        }


def run_comprehensive_analysis(design_params: Dict, cruise_speed: float = 15.0) -> Dict:
    """
    Run comprehensive wind tunnel analysis on a design.
    
    Args:
        design_params: Aircraft design parameters
        cruise_speed: Cruise speed in m/s
        
    Returns:
        Dictionary with complete analysis results
    """
    wt = WindTunnelSimulation(design_params)
    
    # Calculate key performance points
    stall = wt.estimate_stall_speed()
    trim = wt.calculate_trim_condition(cruise_speed)
    stability = wt.analyze_stability(cruise_speed)
    
    # Run angle of attack sweep
    aoa_sweep = wt.sweep_angle_of_attack(cruise_speed)
    
    # Find best L/D
    best_ld = max(aoa_sweep, key=lambda x: x['ld_ratio'])
    
    # Pressure distribution at cruise
    pressure = wt.calculate_pressure_distribution(trim.get('trim_aoa', 5.0) if trim.get('converged') else 5.0)
    
    return {
        'design_params': design_params,
        'cruise_speed_ms': cruise_speed,
        'stall_characteristics': stall,
        'trim_condition': trim,
        'stability_analysis': stability,
        'best_ld_condition': best_ld,
        'aoa_sweep_data': aoa_sweep,
        'pressure_distribution': pressure,
        'aspect_ratio': wt.aspect_ratio
    }
