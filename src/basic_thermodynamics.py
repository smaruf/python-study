"""
Basic Thermodynamics Module

This module provides fundamental thermodynamic calculations for heat transfer,
gas dynamics, and energy systems. It includes functions for ideal gas law,
heat transfer, thermodynamic cycles, and energy calculations.

Author: Python Study Repository
Date: 2026-01-19
"""

import math


# Constants
R_UNIVERSAL = 8.314  # J/(mol·K) - Universal gas constant
R_AIR = 287.05  # J/(kg·K) - Specific gas constant for air
GAMMA_AIR = 1.4  # Ratio of specific heats for air (Cp/Cv)
STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴) - Stefan-Boltzmann constant


def ideal_gas_law(pressure=None, volume=None, n_moles=None, temperature=None):
    """
    Calculate unknown variable using the Ideal Gas Law.
    
    Formula: PV = nRT
    where R = 8.314 J/(mol·K)
    
    Args:
        pressure (float, optional): Pressure in Pascals (Pa)
        volume (float, optional): Volume in cubic meters (m³)
        n_moles (float, optional): Amount of substance in moles (mol)
        temperature (float, optional): Temperature in Kelvin (K)
    
    Returns:
        float: The missing variable
    
    Example:
        >>> # Find pressure for 1 mole of gas at 300K in 0.024 m³
        >>> p = ideal_gas_law(volume=0.024, n_moles=1, temperature=300)
        >>> print(f"Pressure: {p:.2f} Pa")
        Pressure: 103925.00 Pa
        
        >>> # Find temperature for gas at 101325 Pa, 1 mole, 0.024 m³
        >>> t = ideal_gas_law(pressure=101325, volume=0.024, n_moles=1)
        >>> print(f"Temperature: {t:.2f} K")
        Temperature: 292.31 K
    
    Notes:
        - Provide exactly 3 of the 4 parameters
        - Assumes ideal gas behavior (accurate for most gases at low pressure)
        - Standard conditions: T=273.15K (0°C), P=101325 Pa (1 atm)
    """
    params = [pressure, volume, n_moles, temperature]
    none_count = sum(1 for p in params if p is None)
    
    if none_count != 1:
        raise ValueError("Provide exactly 3 of the 4 parameters")
    
    if pressure is None:
        return (n_moles * R_UNIVERSAL * temperature) / volume
    elif volume is None:
        return (n_moles * R_UNIVERSAL * temperature) / pressure
    elif n_moles is None:
        return (pressure * volume) / (R_UNIVERSAL * temperature)
    else:  # temperature is None
        return (pressure * volume) / (n_moles * R_UNIVERSAL)


def ideal_gas_density(pressure: float, temperature: float, R_specific: float = R_AIR) -> float:
    """
    Calculate gas density using ideal gas law.
    
    Formula: ρ = P / (R * T)
    
    Args:
        pressure (float): Pressure in Pascals (Pa)
        temperature (float): Temperature in Kelvin (K)
        R_specific (float): Specific gas constant in J/(kg·K) (default: air = 287.05)
    
    Returns:
        float: Density in kg/m³
    
    Example:
        >>> # Air density at sea level standard conditions
        >>> rho = ideal_gas_density(101325, 288.15)
        >>> print(f"Air density: {rho:.3f} kg/m³")
        Air density: 1.225 kg/m³
        
        >>> # Air density at 5000m altitude (lower pressure and temperature)
        >>> rho_5000 = ideal_gas_density(54000, 255.65)
        >>> print(f"Air density at 5000m: {rho_5000:.3f} kg/m³")
        Air density at 5000m: 0.736 kg/m³
    
    Notes:
        - Standard sea level: ρ = 1.225 kg/m³
        - Density decreases with altitude and temperature
        - Affects aircraft performance and engine efficiency
    """
    return pressure / (R_specific * temperature)


def heat_transfer_conduction(
    thermal_conductivity: float,
    area: float,
    temperature_diff: float,
    thickness: float
) -> float:
    """
    Calculate heat transfer by conduction (Fourier's Law).
    
    Formula: Q = k * A * ΔT / L
    
    Args:
        thermal_conductivity (float): Material thermal conductivity in W/(m·K)
        area (float): Cross-sectional area in m²
        temperature_diff (float): Temperature difference in K or °C
        thickness (float): Material thickness in meters
    
    Returns:
        float: Heat transfer rate in Watts (W)
    
    Example:
        >>> # Heat loss through a wall
        >>> # Concrete: k = 1.4 W/(m·K), wall: 10m², thickness: 0.2m, ΔT: 20K
        >>> q = heat_transfer_conduction(1.4, 10.0, 20.0, 0.2)
        >>> print(f"Heat transfer: {q:.2f} W")
        Heat transfer: 1400.00 W
    
    Thermal Conductivity Values:
        - Air: 0.026 W/(m·K)
        - Wood: 0.12 W/(m·K)
        - Concrete: 1.4 W/(m·K)
        - Steel: 50 W/(m·K)
        - Copper: 400 W/(m·K)
    """
    return thermal_conductivity * area * temperature_diff / thickness


def heat_transfer_convection(
    convection_coefficient: float,
    area: float,
    temperature_diff: float
) -> float:
    """
    Calculate heat transfer by convection (Newton's Law of Cooling).
    
    Formula: Q = h * A * ΔT
    
    Args:
        convection_coefficient (float): Convection heat transfer coefficient in W/(m²·K)
        area (float): Surface area in m²
        temperature_diff (float): Temperature difference between surface and fluid in K
    
    Returns:
        float: Heat transfer rate in Watts (W)
    
    Example:
        >>> # Heat loss from hot surface to air
        >>> # h = 25 W/(m²·K), area = 2 m², ΔT = 50K
        >>> q = heat_transfer_convection(25, 2.0, 50)
        >>> print(f"Heat transfer: {q:.2f} W")
        Heat transfer: 2500.00 W
    
    Typical Convection Coefficients:
        - Free convection (air): 5-25 W/(m²·K)
        - Forced convection (air): 10-200 W/(m²·K)
        - Forced convection (water): 50-10,000 W/(m²·K)
        - Boiling water: 3,000-100,000 W/(m²·K)
    """
    return convection_coefficient * area * temperature_diff


def heat_transfer_radiation(
    emissivity: float,
    area: float,
    surface_temp: float,
    ambient_temp: float
) -> float:
    """
    Calculate heat transfer by radiation (Stefan-Boltzmann Law).
    
    Formula: Q = ε * σ * A * (T_s⁴ - T_a⁴)
    
    Args:
        emissivity (float): Surface emissivity (0-1, dimensionless)
        area (float): Surface area in m²
        surface_temp (float): Surface temperature in Kelvin (K)
        ambient_temp (float): Ambient temperature in Kelvin (K)
    
    Returns:
        float: Heat transfer rate in Watts (W)
    
    Example:
        >>> # Radiation from a hot surface
        >>> # ε = 0.9, area = 1 m², T_surface = 400K, T_ambient = 300K
        >>> q = heat_transfer_radiation(0.9, 1.0, 400, 300)
        >>> print(f"Heat transfer: {q:.2f} W")
        Heat transfer: 648.52 W
    
    Emissivity Values:
        - Black body (ideal): 1.0
        - Black paint: 0.95
        - White paint: 0.90
        - Polished aluminum: 0.05
        - Oxidized steel: 0.80
    """
    return emissivity * STEFAN_BOLTZMANN * area * (surface_temp**4 - ambient_temp**4)


def specific_heat_capacity(heat_energy: float, mass: float, temp_change: float) -> float:
    """
    Calculate specific heat capacity.
    
    Formula: c = Q / (m * ΔT)
    
    Args:
        heat_energy (float): Heat energy in Joules (J)
        mass (float): Mass in kilograms (kg)
        temp_change (float): Temperature change in K or °C
    
    Returns:
        float: Specific heat capacity in J/(kg·K)
    
    Example:
        >>> # Heat 2 kg of water by 10K with 83,720 J
        >>> c = specific_heat_capacity(83720, 2.0, 10.0)
        >>> print(f"Specific heat: {c:.2f} J/(kg·K)")
        Specific heat: 4186.00 J/(kg·K)
    
    Common Specific Heat Values:
        - Water: 4,186 J/(kg·K)
        - Air: 1,005 J/(kg·K)
        - Aluminum: 900 J/(kg·K)
        - Steel: 500 J/(kg·K)
        - Copper: 385 J/(kg·K)
    """
    return heat_energy / (mass * temp_change)


def heat_energy(mass: float, specific_heat: float, temp_change: float) -> float:
    """
    Calculate heat energy required for temperature change.
    
    Formula: Q = m * c * ΔT
    
    Args:
        mass (float): Mass in kilograms (kg)
        specific_heat (float): Specific heat capacity in J/(kg·K)
        temp_change (float): Temperature change in K or °C
    
    Returns:
        float: Heat energy in Joules (J)
    
    Example:
        >>> # Heat 5 kg of water from 20°C to 100°C
        >>> q = heat_energy(5.0, 4186, 80)
        >>> print(f"Energy required: {q:.2f} J ({q/1000:.2f} kJ)")
        Energy required: 1674400.00 J (1674.40 kJ)
    
    Notes:
        - Energy required for phase change (melting, boiling) is additional
        - Latent heat of fusion (water): 334,000 J/kg
        - Latent heat of vaporization (water): 2,260,000 J/kg
    """
    return mass * specific_heat * temp_change


def carnot_efficiency(hot_temp: float, cold_temp: float) -> float:
    """
    Calculate maximum theoretical efficiency of a heat engine (Carnot efficiency).
    
    Formula: η = 1 - (T_cold / T_hot)
    
    Args:
        hot_temp (float): Hot reservoir temperature in Kelvin (K)
        cold_temp (float): Cold reservoir temperature in Kelvin (K)
    
    Returns:
        float: Efficiency (0-1, dimensionless)
    
    Example:
        >>> # Steam engine: T_hot = 500K, T_cold = 300K
        >>> eff = carnot_efficiency(500, 300)
        >>> print(f"Maximum efficiency: {eff:.1%}")
        Maximum efficiency: 40.0%
        
        >>> # Gas turbine: T_hot = 1500K, T_cold = 300K
        >>> eff_turbine = carnot_efficiency(1500, 300)
        >>> print(f"Maximum efficiency: {eff_turbine:.1%}")
        Maximum efficiency: 80.0%
    
    Notes:
        - Carnot efficiency is the theoretical maximum
        - Real engines achieve 50-70% of Carnot efficiency
        - Higher temperature difference = higher efficiency
        - Absolute zero cold reservoir would give 100% efficiency (impossible)
    """
    if hot_temp <= cold_temp:
        raise ValueError("Hot temperature must be greater than cold temperature")
    return 1 - (cold_temp / hot_temp)


def thermal_efficiency(work_output: float, heat_input: float) -> float:
    """
    Calculate actual thermal efficiency of a heat engine.
    
    Formula: η = W_out / Q_in
    
    Args:
        work_output (float): Useful work output in Joules (J)
        heat_input (float): Heat energy input in Joules (J)
    
    Returns:
        float: Efficiency (0-1, dimensionless)
    
    Example:
        >>> # Engine produces 5000 J work from 15000 J heat
        >>> eff = thermal_efficiency(5000, 15000)
        >>> print(f"Thermal efficiency: {eff:.1%}")
        Thermal efficiency: 33.3%
    
    Typical Efficiencies:
        - Steam turbines: 35-40%
        - Gasoline engines: 25-30%
        - Diesel engines: 35-45%
        - Gas turbines: 35-40%
        - Combined cycle: 50-60%
    """
    return work_output / heat_input


def coefficient_of_performance_cooling(cooling_effect: float, work_input: float) -> float:
    """
    Calculate Coefficient of Performance for cooling systems (refrigerators, AC).
    
    Formula: COP_cooling = Q_cold / W_in
    
    Args:
        cooling_effect (float): Heat removed from cold space in Joules (J)
        work_input (float): Work input to system in Joules (J)
    
    Returns:
        float: COP (dimensionless, can be > 1)
    
    Example:
        >>> # AC removes 10,000 J heat using 3,000 J work
        >>> cop = coefficient_of_performance_cooling(10000, 3000)
        >>> print(f"COP: {cop:.2f}")
        COP: 3.33
    
    Notes:
        - COP can be greater than 1 (unlike efficiency)
        - Higher COP = more efficient cooling
        - Typical values:
            * Room AC: 2.5-4.0
            * Industrial chillers: 3.5-6.0
            * Heat pumps: 3.0-5.0
    """
    return cooling_effect / work_input


def coefficient_of_performance_heating(heating_effect: float, work_input: float) -> float:
    """
    Calculate Coefficient of Performance for heating systems (heat pumps).
    
    Formula: COP_heating = Q_hot / W_in
    
    Args:
        heating_effect (float): Heat delivered to hot space in Joules (J)
        work_input (float): Work input to system in Joules (J)
    
    Returns:
        float: COP (dimensionless, typically 3-5)
    
    Example:
        >>> # Heat pump delivers 15,000 J heat using 3,000 J work
        >>> cop = coefficient_of_performance_heating(15000, 3000)
        >>> print(f"COP: {cop:.2f}")
        COP: 5.00
    
    Notes:
        - COP_heating = COP_cooling + 1
        - More efficient than direct electric heating
        - Efficiency decreases as outdoor temperature drops
    """
    return heating_effect / work_input


def enthalpy_change(mass: float, specific_heat: float, temp_change: float) -> float:
    """
    Calculate enthalpy change for constant pressure process.
    
    Formula: ΔH = m * Cp * ΔT
    
    Args:
        mass (float): Mass in kilograms (kg)
        specific_heat (float): Specific heat at constant pressure in J/(kg·K)
        temp_change (float): Temperature change in K or °C
    
    Returns:
        float: Enthalpy change in Joules (J)
    
    Example:
        >>> # Enthalpy change for 2 kg of air heated by 100K
        >>> # Cp for air = 1005 J/(kg·K)
        >>> dh = enthalpy_change(2.0, 1005, 100)
        >>> print(f"Enthalpy change: {dh:.2f} J ({dh/1000:.2f} kJ)")
        Enthalpy change: 201000.00 J (201.00 kJ)
    
    Notes:
        - Enthalpy represents total heat content
        - Used in thermodynamic cycles analysis
        - For ideal gas: ΔH = m * Cp * ΔT
    """
    return mass * specific_heat * temp_change


def isentropic_temperature_ratio(pressure_ratio: float, gamma: float = GAMMA_AIR) -> float:
    """
    Calculate temperature ratio for isentropic (reversible adiabatic) process.
    
    Formula: T2/T1 = (P2/P1)^((γ-1)/γ)
    
    Args:
        pressure_ratio (float): Pressure ratio P2/P1
        gamma (float): Ratio of specific heats Cp/Cv (default: 1.4 for air)
    
    Returns:
        float: Temperature ratio T2/T1
    
    Example:
        >>> # Air compressed from 1 to 10 atmospheres isentropically
        >>> temp_ratio = isentropic_temperature_ratio(10.0)
        >>> print(f"Temperature ratio: {temp_ratio:.3f}")
        Temperature ratio: 1.931
        >>> # If T1 = 300K, then T2 = 300 * 1.931 = 579K
    
    Applications:
        - Compressor outlet temperature prediction
        - Turbine expansion calculations
        - Nozzle flow analysis
    
    Notes:
        - Isentropic = ideal process (no entropy change)
        - Real processes are less efficient
        - γ = 1.4 for diatomic gases (air, N₂, O₂)
        - γ = 1.67 for monatomic gases (He, Ar)
    """
    return pressure_ratio ** ((gamma - 1) / gamma)


def isentropic_pressure_ratio(temperature_ratio: float, gamma: float = GAMMA_AIR) -> float:
    """
    Calculate pressure ratio for isentropic process from temperature ratio.
    
    Formula: P2/P1 = (T2/T1)^(γ/(γ-1))
    
    Args:
        temperature_ratio (float): Temperature ratio T2/T1
        gamma (float): Ratio of specific heats (default: 1.4 for air)
    
    Returns:
        float: Pressure ratio P2/P1
    
    Example:
        >>> # Temperature doubles in isentropic compression
        >>> press_ratio = isentropic_pressure_ratio(2.0)
        >>> print(f"Pressure ratio: {press_ratio:.3f}")
        Pressure ratio: 7.595
    """
    return temperature_ratio ** (gamma / (gamma - 1))


def speed_of_sound(temperature: float, gamma: float = GAMMA_AIR, R: float = R_AIR) -> float:
    """
    Calculate speed of sound in an ideal gas.
    
    Formula: a = sqrt(γ * R * T)
    
    Args:
        temperature (float): Temperature in Kelvin (K)
        gamma (float): Ratio of specific heats (default: 1.4 for air)
        R (float): Specific gas constant in J/(kg·K) (default: 287.05 for air)
    
    Returns:
        float: Speed of sound in m/s
    
    Example:
        >>> # Speed of sound at sea level (15°C = 288K)
        >>> a = speed_of_sound(288)
        >>> print(f"Speed of sound: {a:.2f} m/s")
        Speed of sound: 340.17 m/s
        
        >>> # At cruise altitude (-56.5°C = 216.65K)
        >>> a_cruise = speed_of_sound(216.65)
        >>> print(f"Speed of sound at altitude: {a_cruise:.2f} m/s")
        Speed of sound at altitude: 295.07 m/s
    
    Notes:
        - Speed of sound decreases with decreasing temperature
        - Important for Mach number calculations
        - In air at 20°C: approximately 343 m/s
    """
    return math.sqrt(gamma * R * temperature)


if __name__ == "__main__":
    # Example usage and demonstrations
    print("=" * 60)
    print("BASIC THERMODYNAMICS MODULE - EXAMPLES")
    print("=" * 60)
    
    # Example 1: Ideal Gas Law
    print("\n1. IDEAL GAS LAW")
    print("-" * 40)
    p = ideal_gas_law(volume=0.024, n_moles=1, temperature=300)
    print(f"Gas at 300K, 1 mole, 0.024 m³")
    print(f"Pressure: {p:.2f} Pa ({p/101325:.2f} atm)")
    
    # Example 2: Air density at different altitudes
    print("\n2. AIR DENSITY AT DIFFERENT ALTITUDES")
    print("-" * 40)
    altitudes = [
        (0, 101325, 288.15, "Sea level"),
        (5000, 54000, 255.65, "5000m"),
        (10000, 26500, 223.25, "10000m (cruise)")
    ]
    for alt, press, temp, name in altitudes:
        rho = ideal_gas_density(press, temp)
        print(f"{name:20s}: ρ = {rho:.3f} kg/m³")
    
    # Example 3: Heat transfer through a wall
    print("\n3. HEAT TRANSFER - CONDUCTION")
    print("-" * 40)
    q_cond = heat_transfer_conduction(
        thermal_conductivity=1.4,  # Concrete
        area=20.0,
        temperature_diff=25.0,
        thickness=0.2
    )
    print(f"Concrete wall: 20 m², 0.2m thick, ΔT = 25K")
    print(f"Heat loss: {q_cond:.2f} W ({q_cond/1000:.2f} kW)")
    
    # Example 4: Heat transfer by convection
    print("\n4. HEAT TRANSFER - CONVECTION")
    print("-" * 40)
    q_conv = heat_transfer_convection(
        convection_coefficient=25,
        area=2.0,
        temperature_diff=50
    )
    print(f"Hot surface: 2 m², h = 25 W/(m²·K), ΔT = 50K")
    print(f"Heat loss: {q_conv:.2f} W")
    
    # Example 5: Heating water
    print("\n5. ENERGY TO HEAT WATER")
    print("-" * 40)
    mass_water = 10.0  # kg
    temp_rise = 80  # K (from 20°C to 100°C)
    energy = heat_energy(mass_water, 4186, temp_rise)
    print(f"Mass: {mass_water} kg")
    print(f"Temperature rise: {temp_rise}°C")
    print(f"Energy required: {energy/1000:.2f} kJ")
    print(f"Time with 2 kW heater: {energy/2000:.1f} seconds ({energy/2000/60:.1f} minutes)")
    
    # Example 6: Heat engine efficiency
    print("\n6. HEAT ENGINE EFFICIENCY")
    print("-" * 40)
    T_hot = 800  # K
    T_cold = 300  # K
    carnot_eff = carnot_efficiency(T_hot, T_cold)
    actual_eff = carnot_eff * 0.6  # 60% of Carnot efficiency
    
    print(f"Hot reservoir: {T_hot} K")
    print(f"Cold reservoir: {T_cold} K")
    print(f"Carnot efficiency (max): {carnot_eff:.1%}")
    print(f"Actual efficiency (60% of Carnot): {actual_eff:.1%}")
    
    # Example 7: Air conditioner COP
    print("\n7. AIR CONDITIONER PERFORMANCE")
    print("-" * 40)
    cooling = 12000  # J
    work = 3500  # J
    cop = coefficient_of_performance_cooling(cooling, work)
    print(f"Cooling effect: {cooling} J")
    print(f"Work input: {work} J")
    print(f"COP: {cop:.2f}")
    print(f"For every 1 kW of electricity, removes {cop:.2f} kW of heat")
    
    # Example 8: Compressor temperature rise
    print("\n8. ISENTROPIC COMPRESSION")
    print("-" * 40)
    pressure_ratio = 8.0
    T1 = 300  # K
    temp_ratio = isentropic_temperature_ratio(pressure_ratio)
    T2 = T1 * temp_ratio
    
    print(f"Initial temperature: {T1} K ({T1-273.15:.1f}°C)")
    print(f"Pressure ratio: {pressure_ratio}")
    print(f"Final temperature: {T2:.1f} K ({T2-273.15:.1f}°C)")
    print(f"Temperature rise: {T2-T1:.1f} K")
    
    # Example 9: Speed of sound
    print("\n9. SPEED OF SOUND AT DIFFERENT TEMPERATURES")
    print("-" * 40)
    temps = [
        (273.15, "0°C (freezing)"),
        (288.15, "15°C (standard)"),
        (293.15, "20°C (room temp)"),
        (216.65, "-56.5°C (cruise alt)")
    ]
    for temp, desc in temps:
        a = speed_of_sound(temp)
        print(f"{desc:25s}: {a:.2f} m/s")
    
    print("\n" + "=" * 60)
