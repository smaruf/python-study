import cadquery as cq
from parts.motor_mount import motor_mount
from frames.quad_frame import quad_frame

cq.exporters.export(
    motor_mount(),
    "output/motor_mount.stl"
)

for arm_len in [120, 150, 180]:
    frame = quad_frame(arm_len)
    cq.exporters.export(
        frame,
        f"output/quad_frame_{arm_len}.stl"
    )
