"""
function w_star = SR1d_Find_p_star( gamma, w_l, w_r, p_star_0 )

Find the value of w_star that solves the Riemann problem.
"""
import numpy as np

def SR1d_Find_p_star( gamma, w_l, w_r, p_star_0 ):

    pmin = min(w_l[3], w_r[3], p_star_0)
    pmax = max(w_l[3], w_r[3], p_star_0)
    return fzero(SR1d_Find_Delta_v, 0.5*pmin, 2*pmax)

def SR1d_Find_Delta_v(p_s):

    v_star_l = SR1d_Find_v(w_l, p_s, -1)
    v_star_r = SR1d_Find_v(w_r, p_s,  1)

    return v_star_l - v_star_r


def SR1d_Find_v(known_state, p_s, lr_sign):

    w_star = SR1d_GetState(gamma, known_state, p_s, lr_sign)
    return  w_star(2)
