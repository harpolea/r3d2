"""
all = compute_all(gamma, w)

Convert from the basic primitive variables w = (rho, v, eps) to the full
set all = (rho, v, eps, p, W, h, cs^2)

Should add error checking.
"""
import numpy as np

def compute_all( gamma, w ):

    rho = w[0]
    v = w[1]
    eps = w[2]
    p = (gamma - 1.) * rho * eps
    W_lorentz = 1. / np.sqrt( 1. - v**2)
    h = 1. + eps + p / rho
    cs2 = gamma * p / (rho * h)

    return rho, v, eps, p, W_lorentz, h, cs2
