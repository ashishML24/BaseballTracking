import numpy as np
from .config import CX, CY, FOCAL_LENGTH_MM, PIXEL_SIZE_MM

def pixel_to_ray(u, v):
    fx = FOCAL_LENGTH_MM / PIXEL_SIZE_MM
    r = np.array([(u-CX)/fx, (v-CY)/fx, 1.0])
    return r / np.linalg.norm(r)
