import numpy as np
from .config import DT

def compute_velocity(P):
    v = (P[1:] - P[:-1]) / DT
    return v, v.mean(axis=0)
