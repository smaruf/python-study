import cadquery as cq
from parts.arm import drone_arm

def quad_frame(arm_length=150):
    arms = []

    for angle in [0, 90, 180, 270]:
        arm = (
            drone_arm(length=arm_length)
            .rotate((0,0,0), (0,0,1), angle)
        )
        arms.append(arm)

    frame = arms[0]
    for arm in arms[1:]:
        frame = frame.union(arm)

    return frame
