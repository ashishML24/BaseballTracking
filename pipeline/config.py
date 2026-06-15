import numpy as np

FPS = 240
DT = 1.0 / FPS

FOCAL_LENGTH_MM = 8.0
PIXEL_SIZE_MM = 0.0048

IMG_WIDTH = 1280
IMG_HEIGHT = 1024
CX, CY = IMG_WIDTH / 2, IMG_HEIGHT / 2

BALL_RADIUS_M = 0.0373
BALL_RADIUS_MM = BALL_RADIUS_M * 1000

CAMERA_TILT_DEG = 10.0
G = np.array([0.0, -9.81, 0.0])

def rotation_x(theta_deg):
    t = np.deg2rad(theta_deg)
    return np.array([
        [1, 0, 0],
        [0, np.cos(t), -np.sin(t)],
        [0, np.sin(t),  np.cos(t)]
    ])
