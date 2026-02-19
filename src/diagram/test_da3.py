"""
Test script for DA3 - Data Analytics 3D library
This script validates that all 3D diagram functions work correctly.
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from da3 import DA3


def test_da3_basic():
    """Test basic DA3 functionality."""
    print("Testing DA3 basic functionality...")
    
    # Create test output directory
    test_dir = '/tmp/da3_test'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Create DA3 instance
    da3 = DA3(output_dir=test_dir)
    
    # Test individual plot creation
    print("  - Testing surface plot...")
    da3.surface_plot(filename='test_surface.png')
    assert os.path.exists(os.path.join(test_dir, 'test_surface.png'))
    
    print("  - Testing scatter plot...")
    da3.scatter_plot(filename='test_scatter.png')
    assert os.path.exists(os.path.join(test_dir, 'test_scatter.png'))
    
    print("  - Testing wireframe plot...")
    da3.wireframe_plot(filename='test_wireframe.png')
    assert os.path.exists(os.path.join(test_dir, 'test_wireframe.png'))
    
    print("  - Testing line plot...")
    da3.line_plot(filename='test_line.png')
    assert os.path.exists(os.path.join(test_dir, 'test_line.png'))
    
    # Verify summary tracking
    assert len(da3.created_plots) == 4
    
    print("✓ All basic tests passed!")
    return True


def test_da3_batch_creation():
    """Test batch plot creation."""
    print("\nTesting DA3 batch creation...")
    
    test_dir = '/tmp/da3_batch_test'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    da3 = DA3(output_dir=test_dir)
    
    # Create all plots
    print("  - Creating all plots...")
    da3.create_all_plots()
    
    # Verify all plots were created
    assert len(da3.created_plots) == 9
    
    # Verify files exist
    expected_files = [
        '01_surface.png', '02_parametric_surface.png',
        '03_scatter.png', '04_cluster_scatter.png',
        '05_wireframe.png', '06_sphere.png',
        '07_line.png', '08_spiral.png', '09_lissajous.png'
    ]
    
    for filename in expected_files:
        filepath = os.path.join(test_dir, filename)
        assert os.path.exists(filepath), f"File {filename} not found"
    
    print("✓ Batch creation test passed!")
    return True


def test_individual_modules():
    """Test individual plot modules."""
    print("\nTesting individual modules...")
    
    # Import modules using dynamic loading
    import importlib.util
    
    def load_and_test(module_name, func_name, filename):
        module_path = os.path.join(current_dir, f'{module_name}.py')
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        func = getattr(module, func_name)
        test_file = f'/tmp/{filename}'
        func(save_to_file=True, filename=test_file)
        assert os.path.exists(test_file), f"Module {module_name} failed to create plot"
        print(f"  ✓ {module_name} - {func_name}")
    
    load_and_test('3d_surface_plot', 'create_3d_surface_plot', 'mod_surface.png')
    load_and_test('3d_scatter_plot', 'create_3d_scatter_plot', 'mod_scatter.png')
    load_and_test('3d_wireframe_plot', 'create_3d_wireframe_plot', 'mod_wireframe.png')
    load_and_test('3d_line_plot', 'create_3d_line_plot', 'mod_line.png')
    
    print("✓ All individual module tests passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("DA3 - Data Analytics 3D Library Test Suite")
    print("=" * 60)
    
    try:
        test_da3_basic()
        test_da3_batch_creation()
        test_individual_modules()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        return 0
    
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
