#!/usr/bin/env python3
"""
Aircraft Designer CLI Tool

Command-line interface for aircraft design experimentation with wind tunnel simulation.
Allows batch processing and detailed analysis of wing, body, and engine parameters.
"""

import argparse
import json
import sys
from typing import Dict, Optional
from wind_tunnel import WindTunnelSimulation, run_comprehensive_analysis


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n--- {title} ---")


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with specified decimals."""
    return f"{value:.{decimals}f}"


def display_simulation_results(results: Dict):
    """Display simulation results in a formatted way."""
    
    print_header("Wind Tunnel Simulation Results")
    
    # Design parameters
    print_section("Design Parameters")
    params = results['design_params']
    print(f"  Wingspan:        {params['wingspan']} mm")
    print(f"  Chord:           {params['chord']} mm")
    print(f"  Wing Area:       {format_number(params['wing_area'] / 100)} cm²")
    print(f"  Weight:          {params['weight']} g")
    print(f"  Airfoil:         {params['airfoil_type']}")
    print(f"  Aspect Ratio:    {format_number(results['aspect_ratio'])}")
    
    # Stall characteristics
    print_section("Stall Characteristics")
    stall = results['stall_characteristics']
    print(f"  Stall Speed:     {format_number(stall['stall_speed_ms'])} m/s")
    print(f"  Approach Speed:  {format_number(stall['approach_speed_ms'])} m/s")
    print(f"  CL max:          {format_number(stall['cl_max'])}")
    
    # Trim condition
    print_section("Trim Condition (Level Flight)")
    trim = results['trim_condition']
    if trim.get('converged'):
        print(f"  Cruise Speed:    {format_number(results['cruise_speed_ms'])} m/s")
        print(f"  Trim AoA:        {format_number(trim['trim_aoa'])}°")
        print(f"  Trim CL:         {format_number(trim['trim_cl'])}")
        print(f"  Trim CD:         {format_number(trim['trim_cd'])}")
        print(f"  L/D Ratio:       {format_number(trim['trim_ld'])}")
        print(f"  Drag Force:      {format_number(trim['drag_g'])} g")
    else:
        print(f"  ⚠ {trim.get('message', 'Could not find trim condition')}")
    
    # Best L/D
    print_section("Best L/D Performance")
    best = results['best_ld_condition']
    print(f"  Best L/D:        {format_number(best['ld_ratio'])}")
    print(f"  At AoA:          {format_number(best['angle_of_attack'])}°")
    print(f"  CL:              {format_number(best['cl'])}")
    print(f"  CD:              {format_number(best['cd'])}")
    
    # Stability
    print_section("Stability Analysis")
    stability = results['stability_analysis']
    if stability.get('stable') is not None:
        status = "✓ STABLE" if stability['stable'] else "✗ UNSTABLE"
        print(f"  Status:          {status}")
        print(f"  Static Margin:   {format_number(stability['static_margin'] * 100)}%")
        print(f"  CL_alpha:        {format_number(stability['cl_alpha'])} /rad")
        print(f"  Assessment:      {stability['assessment']}")
    else:
        print(f"  ⚠ Could not analyze stability")
    
    print("\n" + "=" * 70 + "\n")


def display_aoa_sweep_table(results: Dict):
    """Display angle of attack sweep in table format."""
    print_header("Angle of Attack Sweep")
    
    print(f"  {'AoA (°)':>8} {'CL':>8} {'CD':>8} {'L/D':>8} {'Lift (g)':>10} {'Drag (g)':>10} {'Status':>10}")
    print("  " + "-" * 72)
    
    for data in results['aoa_sweep_data']:
        status = "STALLED" if data['stalled'] else "OK"
        print(f"  {data['angle_of_attack']:>8.1f} {data['cl']:>8.3f} {data['cd']:>8.4f} "
              f"{data['ld_ratio']:>8.1f} {data['lift_g']:>10.1f} {data['drag_g']:>10.2f} {status:>10}")
    
    print()


def save_results_to_file(results: Dict, filename: str):
    """Save results to JSON file."""
    try:
        # Convert results to JSON-serializable format
        output = {
            'design_params': results['design_params'],
            'cruise_speed_ms': results['cruise_speed_ms'],
            'aspect_ratio': results['aspect_ratio'],
            'stall_characteristics': results['stall_characteristics'],
            'trim_condition': results['trim_condition'],
            'stability_analysis': results['stability_analysis'],
            'best_ld_condition': results['best_ld_condition'],
            'aoa_sweep_data': results['aoa_sweep_data']
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Results saved to {filename}")
        return True
    except Exception as e:
        print(f"✗ Error saving results: {e}")
        return False


def interactive_mode():
    """Run in interactive mode to get design parameters."""
    print_header("Aircraft Design Experimentation Tool")
    print("\nInteractive Mode - Enter design parameters\n")
    
    try:
        # Get design parameters
        wingspan = float(input("Wingspan (mm) [1000]: ") or "1000")
        chord = float(input("Wing chord (mm) [150]: ") or "150")
        weight = float(input("Aircraft weight (g) [1000]: ") or "1000")
        
        airfoil_input = input("Airfoil type (clark_y/symmetric) [clark_y]: ").lower() or "clark_y"
        airfoil = 'clark_y' if 'clark' in airfoil_input else 'symmetric'
        
        cruise_speed = float(input("Cruise speed (m/s) [15]: ") or "15")
        
        # Optional: fuselage parameters
        add_fuselage = input("\nAdd fuselage parameters? (y/n) [n]: ").lower() == 'y'
        fuselage_length = None
        fuselage_diameter = None
        
        if add_fuselage:
            fuselage_length = float(input("Fuselage length (mm) [800]: ") or "800")
            fuselage_diameter = float(input("Fuselage diameter (mm) [80]: ") or "80")
        
        # Build design params
        design_params = {
            'wingspan': wingspan,
            'chord': chord,
            'wing_area': wingspan * chord,
            'weight': weight,
            'airfoil_type': airfoil
        }
        
        if fuselage_length:
            design_params['fuselage_length'] = fuselage_length
            design_params['fuselage_diameter'] = fuselage_diameter
        
        # Run analysis
        print("\n🔬 Running wind tunnel simulation...\n")
        results = run_comprehensive_analysis(design_params, cruise_speed)
        
        # Display results
        display_simulation_results(results)
        display_aoa_sweep_table(results)
        
        # Ask to save
        save_file = input("Save results to file? (filename or n) [n]: ")
        if save_file and save_file.lower() != 'n':
            save_results_to_file(results, save_file)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n✗ Cancelled by user")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


def batch_mode(design_file: str, output_file: Optional[str] = None):
    """Run in batch mode from design file."""
    try:
        with open(design_file, 'r') as f:
            design_data = json.load(f)
        
        design_params = design_data.get('design_params', {})
        cruise_speed = design_data.get('cruise_speed', 15.0)
        
        # Validate required parameters
        required = ['wingspan', 'chord', 'weight']
        for param in required:
            if param not in design_params:
                print(f"✗ Error: Missing required parameter '{param}' in design file")
                return 1
        
        # Set defaults
        design_params.setdefault('wing_area', design_params['wingspan'] * design_params['chord'])
        design_params.setdefault('airfoil_type', 'clark_y')
        
        print(f"📄 Loading design from {design_file}...")
        print("🔬 Running wind tunnel simulation...\n")
        
        results = run_comprehensive_analysis(design_params, cruise_speed)
        
        # Display results
        display_simulation_results(results)
        display_aoa_sweep_table(results)
        
        # Save if output file specified
        if output_file:
            save_results_to_file(results, output_file)
        
        return 0
        
    except FileNotFoundError:
        print(f"✗ Error: Design file '{design_file}' not found")
        return 1
    except json.JSONDecodeError:
        print(f"✗ Error: Invalid JSON in design file")
        return 1
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


def quick_analysis_mode(wingspan: float, chord: float, weight: float, 
                        airfoil: str = 'clark_y', cruise_speed: float = 15.0):
    """Run quick analysis with command-line parameters."""
    design_params = {
        'wingspan': wingspan,
        'chord': chord,
        'wing_area': wingspan * chord,
        'weight': weight,
        'airfoil_type': airfoil
    }
    
    print("🔬 Running wind tunnel simulation...\n")
    results = run_comprehensive_analysis(design_params, cruise_speed)
    
    display_simulation_results(results)
    display_aoa_sweep_table(results)
    
    return 0


def main():
    """Main entry point for CLI tool."""
    parser = argparse.ArgumentParser(
        description='Aircraft Design Experimentation Tool with Wind Tunnel Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python aircraft_designer_cli.py --interactive
  
  # Quick analysis
  python aircraft_designer_cli.py --wingspan 1000 --chord 150 --weight 1000
  
  # Batch mode from file
  python aircraft_designer_cli.py --batch design.json --output results.json
  
  # Quick with custom parameters
  python aircraft_designer_cli.py -w 1200 -c 180 --weight 1400 --airfoil symmetric --cruise 18

Design File Format (JSON):
  {
    "design_params": {
      "wingspan": 1000,
      "chord": 150,
      "weight": 1000,
      "airfoil_type": "clark_y"
    },
    "cruise_speed": 15.0
  }
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('-i', '--interactive', action='store_true',
                           help='Run in interactive mode')
    mode_group.add_argument('-b', '--batch', metavar='FILE',
                           help='Run in batch mode from design file')
    
    # Quick analysis parameters
    parser.add_argument('-w', '--wingspan', type=float,
                       help='Wingspan in mm')
    parser.add_argument('-c', '--chord', type=float,
                       help='Wing chord in mm')
    parser.add_argument('--weight', type=float,
                       help='Aircraft weight in grams')
    parser.add_argument('--airfoil', choices=['clark_y', 'symmetric'],
                       default='clark_y', help='Airfoil type (default: clark_y)')
    parser.add_argument('--cruise', type=float, default=15.0,
                       help='Cruise speed in m/s (default: 15.0)')
    
    # Output options
    parser.add_argument('-o', '--output', metavar='FILE',
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Determine mode and run
    if args.interactive:
        return interactive_mode()
    
    elif args.batch:
        return batch_mode(args.batch, args.output)
    
    elif args.wingspan and args.chord and args.weight:
        result = quick_analysis_mode(args.wingspan, args.chord, args.weight,
                                    args.airfoil, args.cruise)
        if args.output:
            # Need to regenerate results for saving
            design_params = {
                'wingspan': args.wingspan,
                'chord': args.chord,
                'wing_area': args.wingspan * args.chord,
                'weight': args.weight,
                'airfoil_type': args.airfoil
            }
            results = run_comprehensive_analysis(design_params, args.cruise)
            save_results_to_file(results, args.output)
        return result
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
