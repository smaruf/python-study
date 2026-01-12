"""
Weight and Center of Gravity Calculator

Calculate the weight of components and overall CG position.
"""

from analysis.weight import part_weight
from analysis.cg import center_of_gravity
from materials import PETG, NYLON, PLA

def main():
    print("=" * 60)
    print("WEIGHT AND CENTER OF GRAVITY CALCULATOR")
    print("=" * 60)
    
    # Calculate weight of frame parts
    print("\n--- Frame Component Weights ---")
    
    # Motor mount (approximate cylinder volume)
    motor_mount_volume = 3.14 * (14**2) * 5  # π * r² * h
    mount_weight = part_weight(motor_mount_volume, PETG)
    print(f"Motor mount (PETG): {mount_weight:.2f}g")
    print(f"  4x motor mounts: {mount_weight * 4:.2f}g")
    
    # Arm (rectangular volume)
    arm_volume = 150 * 16 * 12  # length × width × height mm³
    arm_weight = part_weight(arm_volume, NYLON)
    print(f"\nArm (Nylon): {arm_weight:.2f}g")
    print(f"  4x arms: {arm_weight * 4:.2f}g")
    
    # Battery tray
    tray_volume = (100 * 35 * 2) - (96 * 31 * 2)  # Outer - inner cavity
    tray_weight = part_weight(tray_volume, PETG)
    print(f"\nBattery tray (PETG): {tray_weight:.2f}g")
    
    # Camera mount
    camera_mount_volume = 20 * 20 * 3
    camera_mount_weight = part_weight(camera_mount_volume, PLA)
    print(f"Camera mount (PLA): {camera_mount_weight:.2f}g")
    
    # Total printed parts
    total_printed = (mount_weight * 4) + (arm_weight * 4) + tray_weight + camera_mount_weight
    print(f"\n--- Total Printed Parts: {total_printed:.2f}g ---")
    
    # Calculate complete quad center of gravity
    print("\n" + "=" * 60)
    print("CENTER OF GRAVITY CALCULATION (5\" Quad)")
    print("=" * 60)
    
    # Component definitions: (weight_g, position_mm_from_front)
    components = {
        "Camera": (15, 20),
        "Battery (3S 1300mAh)": (120, 75),
        "Flight Controller": (10, 80),
        "VTX": (8, 85),
        "4x Motors (30g each)": (120, 140),
        "4x ESCs (included in motors)": (0, 140),
        "Receiver": (5, 85),
        "Frame (printed)": (total_printed, 75),
    }
    
    print("\nComponent Breakdown:")
    print(f"{'Component':<30} {'Weight':<10} {'Position':<10}")
    print("-" * 60)
    
    total_weight = 0
    masses = []
    positions = []
    
    for name, (weight, position) in components.items():
        if weight > 0:  # Skip zero-weight components
            print(f"{name:<30} {weight:>6.1f}g    {position:>6.1f}mm")
            masses.append(weight)
            positions.append(position)
            total_weight += weight
    
    # Calculate CG
    cg = center_of_gravity(masses, positions)
    
    print("-" * 60)
    print(f"{'TOTAL WEIGHT':<30} {total_weight:>6.1f}g")
    print(f"\nCenter of Gravity: {cg:.1f}mm from front")
    print(f"Recommended CG: 70-80mm (for 150mm arms)")
    
    # CG Analysis
    print("\n--- CG Analysis ---")
    if 70 <= cg <= 80:
        print("✓ CG is OPTIMAL")
    elif 65 <= cg < 70:
        print("⚠ CG is slightly forward - stable but less agile")
    elif 80 < cg <= 85:
        print("⚠ CG is slightly back - more agile but less stable")
    else:
        print("✗ CG is OUT OF RANGE - adjust battery position!")
    
    # Thrust-to-weight calculation
    print("\n" + "=" * 60)
    print("THRUST-TO-WEIGHT RATIO")
    print("=" * 60)
    
    # Typical motor thrust (example: 2306 2400KV on 4S)
    thrust_per_motor = 800  # grams
    total_thrust = thrust_per_motor * 4
    
    thrust_to_weight = total_thrust / total_weight
    
    print(f"\nMotor: 2306 2400KV on 4S")
    print(f"Thrust per motor: ~{thrust_per_motor}g")
    print(f"Total thrust: {total_thrust}g")
    print(f"Total weight: {total_weight:.1f}g")
    print(f"\nThrust-to-weight ratio: {thrust_to_weight:.2f}:1")
    
    print("\n--- Performance Prediction ---")
    if thrust_to_weight >= 4:
        print("✓ EXCELLENT - Racing/high-performance")
    elif thrust_to_weight >= 3:
        print("✓ VERY GOOD - Freestyle/acrobatics")
    elif thrust_to_weight >= 2:
        print("✓ GOOD - General flying/cruising")
    elif thrust_to_weight >= 1.5:
        print("⚠ ADEQUATE - Gentle flying only")
    else:
        print("✗ INSUFFICIENT - Will not fly well!")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
