import numpy as np
from .config import DT, G

def kalman_filter(P):
    x = np.zeros(6)
    x[:3] = P[0]

    F = np.eye(6)
    for i in range(3):
        F[i, i+3] = DT

    H = np.zeros((3,6))
    H[:,:3] = np.eye(3)

    Q = np.eye(6) * 1e-4
    Rm = np.eye(3) * 1e-3
    Pcov = np.eye(6)

    Pf = []

    for z in P:
        x = F @ x
        x[3:] += G * DT
        Pcov = F @ Pcov @ F.T + Q

        y = z - H @ x
        S = H @ Pcov @ H.T + Rm
        K = Pcov @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        Pcov = (np.eye(6) - K @ H) @ Pcov

        Pf.append(x[:3].copy())

    return np.array(Pf)
