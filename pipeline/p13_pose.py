from .config import rotation_x, CAMERA_TILT_DEG

R = rotation_x(CAMERA_TILT_DEG)

def camera_to_world(P):
    return R @ P
