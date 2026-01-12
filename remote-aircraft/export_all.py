import os
import sys

try:
    import cadquery as cq
    from parts.motor_mount import motor_mount
    from frames.quad_frame import quad_frame
except ImportError as e:
    print("Error: CadQuery not installed.")
    print("\nTo install CadQuery:")
    print("  conda install -c conda-forge -c cadquery cadquery")
    print("  or download CQ-Editor from:")
    print("  https://github.com/CadQuery/CQ-editor/releases")
    sys.exit(1)

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

print("Generating STL files...")

# Export motor mount
print("  - motor_mount.stl")
cq.exporters.export(
    motor_mount(),
    "output/motor_mount.stl"
)

# Export quad frames
for arm_len in [120, 150, 180]:
    print(f"  - quad_frame_{arm_len}.stl")
    frame = quad_frame(arm_len)
    cq.exporters.export(
        frame,
        f"output/quad_frame_{arm_len}.stl"
    )

print("\nComplete! STL files saved in output/ directory")
