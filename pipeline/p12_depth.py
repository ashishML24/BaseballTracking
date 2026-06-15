from .config import FOCAL_LENGTH_MM, BALL_RADIUS_MM, PIXEL_SIZE_MM

def estimate_depth(radius_px):
    r_mm = radius_px * PIXEL_SIZE_MM
    return (FOCAL_LENGTH_MM * BALL_RADIUS_MM / r_mm) / 1000.0
