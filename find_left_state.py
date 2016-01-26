"""
Given a (relativistic) Mach number and the initial data for the unshocked fluid on the right, this script will find the left state.
"""

import eos_defns
import SR1d
import numpy as np
from scipy.optimize import brentq

def find_left(q_r, M=1.):
    """
    Finds the left wave for a given Mach number and initial right state.

    Returns the left hand state and speed of the shock wave.
    """

    p_r = q_r.p
    c_s = q_r.cs

    print('c_s = {}'.format(c_s))

    # Limits for p_star guess.
    # If M < 1, this should be less than the rhs pressure;
    # if M > 1, should be higher
    if M < 1.:
        p_star_lims = [1.e-5 * p_r, p_r]
    else:
        p_star_lims = [p_r, 1.e4 * p_r]

    def find_p_star(p_star_guess, q_r, c_s):
        # wavenumber is 2 as nonlinear rhs wave
        wave_r = SR1d.Wave(q_r, p_star_guess, 2)

        # get wave speed
        v_s = wave_r.wave_speed[0]

        # calculate Mach number
        M_star = (v_s / np.sqrt(1. - v_s**2)) / (c_s / np.sqrt(1. - c_s**2))

        return M_star - M

    try:
        p_star = brentq(find_p_star, p_star_lims[0], p_star_lims[1], args=(q_r, c_s))
    except ValueError:
        # brentq will have failed due to the limits, so try
        # lowering/raising them
        if M < 1.:
            p_star_lims[0] *= 1.e-5
        else:
            p_star_lims[1] *= 1.e4

        p_star = brentq(find_p_star, p_star_lims[0], p_star_lims[1], args=(q_r, c_s))

    wave_r = SR1d.Wave(q_r, p_star, 2)

    return wave_r.q_l, wave_r.wave_speed[0]


if __name__ == "__main__":

    # standard 5/3 gamma law EoS
    gamma = 5./3.
    K = 1.
    eos = eos_defns.eos_gamma_law(gamma)

    # right state
    rho_r, v_r, vt_r = (0.001, 0.0, 0.0)
    eps_r = K * rho_r**(gamma - 1.) / (gamma - 1.)

    # initialise right state
    q_r = SR1d.State(rho_r, v_r, vt_r, eps_r, eos, label="R")

    q_l, v_s = find_left(q_r, M=20.)
    print('left state [rho,  v,  vt,  eps] = {}'.format(q_l.prim()))
    print('wave speed = {}'.format(v_s))
