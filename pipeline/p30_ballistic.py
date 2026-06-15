import numpy as np
from .config import DT, G
from scipy.stats import t

def fit_ballistic_with_ci(P, alpha=0.05):
    tvals = np.arange(len(P)) * DT
    A = np.column_stack([np.ones_like(tvals), tvals])

    p0, v0, v0_ci = [], [], []

    for axis in range(3):
        y = P[:,axis]
        if axis == 1:
            y = y - 0.5 * G[axis] * tvals**2

        coeff, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
        dof = len(y) - 2
        sigma2 = res[0] / dof if len(res) else 0.0
        cov = sigma2 * np.linalg.inv(A.T @ A)

        tval = t.ppf(1 - alpha/2, dof)
        ci = tval * np.sqrt(cov[1,1])

        p0.append(coeff[0])
        v0.append(coeff[1])
        v0_ci.append(ci)

    return np.array(p0), np.array(v0), np.array(v0_ci)
