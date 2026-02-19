"""
Test suite for wind tunnel simulation module
"""

import sys
import os

# Add remote-aircraft to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wind_tunnel import WindTunnelSimulation, run_comprehensive_analysis


def test_wind_tunnel_initialization():
    """Test WindTunnelSimulation initialization"""
    design_params = {
        'wingspan': 1000,
        'chord': 150,
        'wing_area': 150000,
        'weight': 1000,
        'airfoil_type': 'clark_y'
    }
    
    wt = WindTunnelSimulation(design_params)
    
    assert wt.wingspan == 1000, "Wingspan not set correctly"
    assert wt.chord == 150, "Chord not set correctly"
    assert wt.wing_area == 150000, "Wing area not set correctly"
    assert wt.weight == 1000, "Weight not set correctly"
    assert wt.aspect_ratio > 0, "Aspect ratio should be positive"
    
    print("✓ Initialization test passed")


def test_lift_coefficient_calculation():
    """Test lift coefficient calculation"""
    design_params = {
        'wingspan': 1000,
        'chord': 150,
        'wing_area': 150000,
        'weight': 1000,
        'airfoil_type': 'clark_y'
    }
    
    wt = WindTunnelSimulation(design_params)
    
    # Test at 0 degrees
    cl_0 = wt.calculate_lift_coefficient(0)
    assert cl_0 > 0, "Clark-Y should have positive CL at 0 degrees"
    
    # Test at positive angle
    cl_5 = wt.calculate_lift_coefficient(5)
    assert cl_5 > cl_0, "CL should increase with angle of attack"
    
    # Test symmetric airfoil
    design_params['airfoil_type'] = 'symmetric'
    wt_sym = WindTunnelSimulation(design_params)
    cl_sym_0 = wt_sym.calculate_lift_coefficient(0)
    assert abs(cl_sym_0) < 0.1, "Symmetric airfoil should have near-zero CL at 0 degrees"
    
    print("✓ Lift coefficient calculation test passed")


def test_drag_coefficient_calculation():
    """Test drag coefficient calculation"""
    design_params = {
        'wingspan': 1000,
        'chord': 150,
        'wing_area': 150000,
        'weight': 1000,
        'airfoil_type': 'clark_y'
    }
    
    wt = WindTunnelSimulation(design_params)
    
    # Test at low CL
    cd_low = wt.calculate_drag_coefficient(0.3)
    
    # Test at high CL
    cd_high = wt.calculate_drag_coefficient(1.0)
    
    assert cd_high > cd_low, "CD should increase with CL (induced drag)"
    assert cd_low > 0.02, "CD should include profile drag"
    
    print("✓ Drag coefficient calculation test passed")


def test_simulation_at_speed():
    """Test simulation at specific speed"""
    design_params = {
        'wingspan': 1000,
        'chord': 150,
        'wing_area': 150000,
        'weight': 1000,
        'airfoil_type': 'clark_y'
    }
    
    wt = WindTunnelSimulation(design_params)
    
    result = wt.simulate_at_speed(15.0, 5.0)
    
    assert 'cl' in result, "Result should contain CL"
    assert 'cd' in result, "Result should contain CD"
    assert 'lift_g' in result, "Result should contain lift in grams"
    assert 'drag_g' in result, "Result should contain drag in grams"
    assert 'ld_ratio' in result, "Result should contain L/D ratio"
    
    assert result['lift_g'] > 0, "Lift should be positive at positive AoA"
    assert result['drag_g'] > 0, "Drag should always be positive"
    assert result['ld_ratio'] > 0, "L/D ratio should be positive"
    
    print("✓ Simulation at speed test passed")


def test_angle_of_attack_sweep():
    """Test angle of attack sweep"""
    design_params = {
        'wingspan': 1000,
        'chord': 150,
        'wing_area': 150000,
        'weight': 1000,
        'airfoil_type': 'clark_y'
    }
    
    wt = WindTunnelSimulation(design_params)
    
    results = wt.sweep_angle_of_attack(15.0, (-5, 20))
    
    assert len(results) > 0, "Sweep should return results"
    assert len(results) == 26, "Should have results for angles -5 to 20 (26 points)"
    
    # Check that lift increases with angle (before stall)
    cl_values = [r['cl'] for r in results[:15]]  # Before stall
    for i in range(len(cl_values) - 1):
        assert cl_values[i+1] >= cl_values[i], "CL should increase with AoA before stall"
    
    print("✓ Angle of attack sweep test passed")


def test_stall_speed_estimation():
    """Test stall speed estimation"""
    design_params = {
        'wingspan': 1000,
        'chord': 150,
        'wing_area': 150000,
        'weight': 1000,
        'airfoil_type': 'clark_y'
    }
    
    wt = WindTunnelSimulation(design_params)
    
    stall = wt.estimate_stall_speed()
    
    assert 'stall_speed_ms' in stall, "Should return stall speed"
    assert 'approach_speed_ms' in stall, "Should return approach speed"
    assert stall['stall_speed_ms'] > 0, "Stall speed should be positive"
    assert stall['approach_speed_ms'] > stall['stall_speed_ms'], "Approach speed should be higher than stall speed"
    assert stall['approach_speed_ms'] / stall['stall_speed_ms'] > 1.2, "Approach speed should be at least 1.2x stall speed"
    
    print("✓ Stall speed estimation test passed")


def test_trim_condition():
    """Test trim condition calculation"""
    design_params = {
        'wingspan': 1000,
        'chord': 150,
        'wing_area': 150000,
        'weight': 1000,
        'airfoil_type': 'clark_y'
    }
    
    wt = WindTunnelSimulation(design_params)
    
    trim = wt.calculate_trim_condition(15.0)
    
    assert 'converged' in trim, "Should indicate convergence status"
    
    if trim['converged']:
        assert 'trim_aoa' in trim, "Should return trim angle of attack"
        assert 'trim_cl' in trim, "Should return trim CL"
        assert 'trim_ld' in trim, "Should return trim L/D"
        assert trim['trim_aoa'] >= -5 and trim['trim_aoa'] <= 15, "Trim AoA should be reasonable"
    
    print("✓ Trim condition test passed")


def test_stability_analysis():
    """Test stability analysis"""
    design_params = {
        'wingspan': 1000,
        'chord': 150,
        'wing_area': 150000,
        'weight': 1000,
        'airfoil_type': 'clark_y'
    }
    
    wt = WindTunnelSimulation(design_params)
    
    stability = wt.analyze_stability(15.0)
    
    assert 'stable' in stability, "Should indicate stability status"
    
    if stability['stable'] is not None:
        assert 'static_margin' in stability, "Should return static margin"
        assert 'cl_alpha' in stability, "Should return lift curve slope"
        assert 'assessment' in stability, "Should provide stability assessment"
    
    print("✓ Stability analysis test passed")


def test_comprehensive_analysis():
    """Test comprehensive analysis function"""
    design_params = {
        'wingspan': 1000,
        'chord': 150,
        'wing_area': 150000,
        'weight': 1000,
        'airfoil_type': 'clark_y'
    }
    
    results = run_comprehensive_analysis(design_params, cruise_speed=15.0)
    
    assert 'design_params' in results, "Should include design params"
    assert 'stall_characteristics' in results, "Should include stall characteristics"
    assert 'trim_condition' in results, "Should include trim condition"
    assert 'stability_analysis' in results, "Should include stability analysis"
    assert 'best_ld_condition' in results, "Should include best L/D"
    assert 'aoa_sweep_data' in results, "Should include AoA sweep data"
    assert 'pressure_distribution' in results, "Should include pressure distribution"
    
    print("✓ Comprehensive analysis test passed")


def test_realistic_values():
    """Test that simulation produces realistic values"""
    # Small glider design
    design_params = {
        'wingspan': 1200,
        'chord': 180,
        'wing_area': 1200 * 180,
        'weight': 800,
        'airfoil_type': 'clark_y'
    }
    
    results = run_comprehensive_analysis(design_params, cruise_speed=12.0)
    
    # Check stall speed is reasonable for a glider
    stall_speed = results['stall_characteristics']['stall_speed_ms']
    assert 5 < stall_speed < 15, f"Stall speed {stall_speed:.1f} m/s seems unrealistic for this design"
    
    # Check L/D ratio is reasonable
    best_ld = results['best_ld_condition']['ld_ratio']
    assert 5 < best_ld < 25, f"L/D ratio {best_ld:.1f} seems unrealistic"
    
    # Check cruise CL is reasonable
    if results['trim_condition'].get('converged'):
        cruise_cl = results['trim_condition']['trim_cl']
        assert 0.2 < cruise_cl < 1.2, f"Cruise CL {cruise_cl:.2f} seems unrealistic"
    
    print("✓ Realistic values test passed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Running Wind Tunnel Simulation Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_wind_tunnel_initialization,
        test_lift_coefficient_calculation,
        test_drag_coefficient_calculation,
        test_simulation_at_speed,
        test_angle_of_attack_sweep,
        test_stall_speed_estimation,
        test_trim_condition,
        test_stability_analysis,
        test_comprehensive_analysis,
        test_realistic_values
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
