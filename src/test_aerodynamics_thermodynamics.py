"""
Unit Tests for Basic Aerodynamics and Thermodynamics Modules

This file contains comprehensive unit tests for the basic_aerodynamics and
basic_thermodynamics modules to ensure correctness of calculations.

Author: Python Study Repository
Date: 2026-01-19
"""

import unittest
import math
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import basic_aerodynamics as aero
import basic_thermodynamics as thermo


class TestAerodynamics(unittest.TestCase):
    """Test cases for basic aerodynamics module."""
    
    def test_calculate_lift(self):
        """Test lift calculation."""
        # Standard conditions
        lift = aero.calculate_lift(
            velocity=50,
            air_density=1.225,
            wing_area=10.0,
            lift_coefficient=0.5
        )
        # Expected: 0.5 * 1.225 * 50^2 * 10.0 * 0.5 = 7656.25
        self.assertAlmostEqual(lift, 7656.25, places=2)
    
    def test_calculate_drag(self):
        """Test drag calculation."""
        drag = aero.calculate_drag(
            velocity=50,
            air_density=1.225,
            wing_area=10.0,
            drag_coefficient=0.03
        )
        # Expected: 0.5 * 1.225 * 50^2 * 10.0 * 0.03 = 459.375
        self.assertAlmostEqual(drag, 459.375, places=2)
    
    def test_dynamic_pressure(self):
        """Test dynamic pressure calculation."""
        q = aero.calculate_dynamic_pressure(velocity=25, air_density=1.225)
        # Expected: 0.5 * 1.225 * 25^2 = 382.8125
        self.assertAlmostEqual(q, 382.8125, places=2)
    
    def test_wing_loading(self):
        """Test wing loading calculation."""
        loading = aero.calculate_wing_loading(weight=10000, wing_area=10.0)
        # Expected: 10000 / 10.0 = 1000
        self.assertEqual(loading, 1000.0)
    
    def test_aspect_ratio(self):
        """Test aspect ratio calculation."""
        ar = aero.calculate_aspect_ratio(wingspan=12.0, wing_area=10.0)
        # Expected: 12^2 / 10.0 = 14.4
        self.assertAlmostEqual(ar, 14.4, places=2)
    
    def test_induced_drag_coefficient(self):
        """Test induced drag coefficient calculation."""
        cdi = aero.calculate_induced_drag_coefficient(
            lift_coefficient=0.5,
            aspect_ratio=10.0,
            efficiency_factor=0.8
        )
        # Expected: 0.5^2 / (pi * 10.0 * 0.8) = 0.00995
        expected = 0.5**2 / (math.pi * 10.0 * 0.8)
        self.assertAlmostEqual(cdi, expected, places=5)
    
    def test_stall_speed(self):
        """Test stall speed calculation."""
        vs = aero.calculate_stall_speed(
            weight=10000,
            air_density=1.225,
            wing_area=10.0,
            max_lift_coefficient=1.4
        )
        # Expected: sqrt(2 * 10000 / (1.225 * 10.0 * 1.4))
        expected = math.sqrt(2 * 10000 / (1.225 * 10.0 * 1.4))
        self.assertAlmostEqual(vs, expected, places=2)
    
    def test_lift_to_drag_ratio(self):
        """Test L/D ratio calculation."""
        ld_ratio = aero.calculate_lift_to_drag_ratio(lift=10000, drag=500)
        self.assertEqual(ld_ratio, 20.0)
    
    def test_reynolds_number(self):
        """Test Reynolds number calculation."""
        re = aero.reynolds_number(
            velocity=25,
            characteristic_length=1.5,
            kinematic_viscosity=1.46e-5
        )
        # Expected: 25 * 1.5 / 1.46e-5
        expected = 25 * 1.5 / 1.46e-5
        self.assertAlmostEqual(re, expected, places=0)
    
    def test_glide_ratio(self):
        """Test glide ratio calculation."""
        glide = aero.calculate_glide_ratio(15.0)
        self.assertEqual(glide, 15.0)
    
    def test_mach_number(self):
        """Test Mach number calculation."""
        m = aero.mach_number(velocity=100, speed_of_sound=343)
        # Expected: 100 / 343 = 0.2915
        self.assertAlmostEqual(m, 100/343, places=4)


class TestThermodynamics(unittest.TestCase):
    """Test cases for basic thermodynamics module."""
    
    def test_ideal_gas_law_pressure(self):
        """Test ideal gas law for pressure calculation."""
        p = thermo.ideal_gas_law(volume=0.024, n_moles=1, temperature=300)
        # Expected: (1 * 8.314 * 300) / 0.024
        expected = (1 * 8.314 * 300) / 0.024
        self.assertAlmostEqual(p, expected, places=1)
    
    def test_ideal_gas_law_temperature(self):
        """Test ideal gas law for temperature calculation."""
        t = thermo.ideal_gas_law(pressure=101325, volume=0.024, n_moles=1)
        # Expected: (101325 * 0.024) / (1 * 8.314)
        expected = (101325 * 0.024) / (1 * 8.314)
        self.assertAlmostEqual(t, expected, places=1)
    
    def test_ideal_gas_law_invalid_params(self):
        """Test ideal gas law with invalid parameters."""
        with self.assertRaises(ValueError):
            # Providing only 2 parameters
            thermo.ideal_gas_law(volume=0.024, n_moles=1)
        
        with self.assertRaises(ValueError):
            # Providing all 4 parameters
            thermo.ideal_gas_law(pressure=101325, volume=0.024, n_moles=1, temperature=300)
    
    def test_ideal_gas_density(self):
        """Test air density calculation."""
        rho = thermo.ideal_gas_density(pressure=101325, temperature=288.15)
        # Expected: 101325 / (287.05 * 288.15) = ~1.225
        expected = 101325 / (287.05 * 288.15)
        self.assertAlmostEqual(rho, expected, places=3)
    
    def test_heat_transfer_conduction(self):
        """Test conduction heat transfer."""
        q = thermo.heat_transfer_conduction(
            thermal_conductivity=1.4,
            area=10.0,
            temperature_diff=20.0,
            thickness=0.2
        )
        # Expected: 1.4 * 10.0 * 20.0 / 0.2 = 1400
        self.assertEqual(q, 1400.0)
    
    def test_heat_transfer_convection(self):
        """Test convection heat transfer."""
        q = thermo.heat_transfer_convection(
            convection_coefficient=25,
            area=2.0,
            temperature_diff=50
        )
        # Expected: 25 * 2.0 * 50 = 2500
        self.assertEqual(q, 2500.0)
    
    def test_heat_transfer_radiation(self):
        """Test radiation heat transfer."""
        q = thermo.heat_transfer_radiation(
            emissivity=0.9,
            area=1.0,
            surface_temp=400,
            ambient_temp=300
        )
        # Expected: 0.9 * 5.67e-8 * 1.0 * (400^4 - 300^4)
        expected = 0.9 * 5.67e-8 * 1.0 * (400**4 - 300**4)
        self.assertAlmostEqual(q, expected, places=1)
    
    def test_specific_heat_capacity(self):
        """Test specific heat capacity calculation."""
        c = thermo.specific_heat_capacity(
            heat_energy=83720,
            mass=2.0,
            temp_change=10.0
        )
        # Expected: 83720 / (2.0 * 10.0) = 4186
        self.assertEqual(c, 4186.0)
    
    def test_heat_energy(self):
        """Test heat energy calculation."""
        q = thermo.heat_energy(mass=5.0, specific_heat=4186, temp_change=80)
        # Expected: 5.0 * 4186 * 80 = 1,674,400
        self.assertEqual(q, 1674400.0)
    
    def test_carnot_efficiency(self):
        """Test Carnot efficiency calculation."""
        eff = thermo.carnot_efficiency(hot_temp=500, cold_temp=300)
        # Expected: 1 - (300/500) = 0.4
        self.assertEqual(eff, 0.4)
    
    def test_carnot_efficiency_invalid(self):
        """Test Carnot efficiency with invalid temperatures."""
        with self.assertRaises(ValueError):
            thermo.carnot_efficiency(hot_temp=300, cold_temp=500)
    
    def test_thermal_efficiency(self):
        """Test thermal efficiency calculation."""
        eff = thermo.thermal_efficiency(work_output=5000, heat_input=15000)
        # Expected: 5000 / 15000 = 0.333...
        self.assertAlmostEqual(eff, 1/3, places=4)
    
    def test_cop_cooling(self):
        """Test COP for cooling."""
        cop = thermo.coefficient_of_performance_cooling(
            cooling_effect=10000,
            work_input=3000
        )
        # Expected: 10000 / 3000 = 3.333...
        self.assertAlmostEqual(cop, 10000/3000, places=2)
    
    def test_cop_heating(self):
        """Test COP for heating."""
        cop = thermo.coefficient_of_performance_heating(
            heating_effect=15000,
            work_input=3000
        )
        # Expected: 15000 / 3000 = 5.0
        self.assertEqual(cop, 5.0)
    
    def test_enthalpy_change(self):
        """Test enthalpy change calculation."""
        dh = thermo.enthalpy_change(mass=2.0, specific_heat=1005, temp_change=100)
        # Expected: 2.0 * 1005 * 100 = 201,000
        self.assertEqual(dh, 201000.0)
    
    def test_isentropic_temperature_ratio(self):
        """Test isentropic temperature ratio."""
        temp_ratio = thermo.isentropic_temperature_ratio(pressure_ratio=10.0)
        # Expected: 10^((1.4-1)/1.4) = 10^(0.2857) ≈ 1.931
        expected = 10.0 ** ((1.4 - 1) / 1.4)
        self.assertAlmostEqual(temp_ratio, expected, places=3)
    
    def test_isentropic_pressure_ratio(self):
        """Test isentropic pressure ratio."""
        press_ratio = thermo.isentropic_pressure_ratio(temperature_ratio=2.0)
        # Expected: 2^(1.4/(1.4-1)) = 2^3.5 ≈ 7.595
        expected = 2.0 ** (1.4 / (1.4 - 1))
        self.assertAlmostEqual(press_ratio, expected, places=3)
    
    def test_speed_of_sound(self):
        """Test speed of sound calculation."""
        a = thermo.speed_of_sound(temperature=288)
        # Expected: sqrt(1.4 * 287.05 * 288)
        expected = math.sqrt(1.4 * 287.05 * 288)
        self.assertAlmostEqual(a, expected, places=2)


class TestIntegration(unittest.TestCase):
    """Integration tests combining both modules."""
    
    def test_aircraft_performance(self):
        """Test complete aircraft performance calculation."""
        # Aircraft parameters
        weight = 10000  # N
        wing_area = 10.0  # m²
        wingspan = 12.0  # m
        velocity = 50  # m/s
        altitude_temp = 288.15  # K
        altitude_press = 101325  # Pa
        
        # Calculate air density
        rho = thermo.ideal_gas_density(altitude_press, altitude_temp)
        
        # Calculate lift
        CL = 0.5
        lift = aero.calculate_lift(velocity, rho, wing_area, CL)
        
        # Calculate drag
        CD = 0.03
        drag = aero.calculate_drag(velocity, rho, wing_area, CD)
        
        # Verify calculations are reasonable
        self.assertGreater(lift, 0)
        self.assertGreater(drag, 0)
        self.assertLess(drag, lift)  # L/D should be positive
        
        # Calculate L/D
        ld_ratio = aero.calculate_lift_to_drag_ratio(lift, drag)
        self.assertGreater(ld_ratio, 1)
    
    def test_engine_thermodynamics(self):
        """Test engine cycle thermodynamics."""
        # Compression
        T1 = 300  # K
        pressure_ratio = 8.0
        temp_ratio = thermo.isentropic_temperature_ratio(pressure_ratio)
        T2 = T1 * temp_ratio
        
        # Verify compression increases temperature
        self.assertGreater(T2, T1)
        
        # Calculate efficiency
        T_cold = 300  # K
        T_hot = 800  # K
        eff = thermo.carnot_efficiency(T_hot, T_cold)
        
        # Verify efficiency is between 0 and 1
        self.assertGreater(eff, 0)
        self.assertLess(eff, 1)


def run_tests_verbose():
    """Run tests with verbose output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestAerodynamics))
    suite.addTests(loader.loadTestsFromTestCase(TestThermodynamics))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("UNIT TESTS FOR AERODYNAMICS AND THERMODYNAMICS MODULES")
    print("=" * 60)
    print()
    
    result = run_tests_verbose()
    
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed.")
    
    print("=" * 60)
