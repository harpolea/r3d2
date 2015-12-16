"""
function w = SR1d_GetState( gamma, known_state, p_star, lr_sign )

Given the known state and the pressure the other side of the wave,
compute all the state information
"""
import numpy as np

def SR1d_GetState( gamma, known_state, p_star, lr_sign ):

    rho_known = known_state[0]
    v_known   = known_state[1]
    eps_known = known_state[2]
    p_known   = known_state[3]
    h_known   = known_state[5]
    cs_known  = np.sqrt(known_state[6])
    e_known   = rho_known * (1. + eps_known)

    if (p_star > p_known): # Shock wave

        # Check the root of the quadratic
        a = 1. + (gamma - 1.) * (p_known - p_star) / (gamma * p_star)
        b = 1. - a
        c = h_known * (p_known - p_star) / rho_known - h_known**2
        if (c > b**2 / (4. * a)):
            error('Unphysical enthalpy')

        # Find quantities across the wave
        h_star = ( -b + sqrt( b**2 - 4. * a * c) ) / (2. * a)
        rho_star = gamma * p_star / ( (gamma - 1.) * (h_star - 1.) )
        eps_star = p_star / (rho_star * (gamma - 1.))
        e_star = rho_star + p_star / (gamma - 1.)

        v_12 = -lr_sign * \
            sqrt( (p_star - p_known) * (e_star - e_known) / \
            ( (e_known + p_star) * (e_star + p_known) ) )
        v_star = (v_known - v_12) / (1. - v_known * v_12)

    else: # Rarefaction wave

        rho_star = rho_known * (p_star / p_known)**(1. / gamma)
        eps_star = p_star / (rho_star * (gamma - 1.))
        h_star = 1. + eps_star + p_star / rho_star
        cs_star = np.sqrt(gamma * p_star / (h_star * rho_star))
        sqgm1 = np.sqrt(gamma - 1.)
        a = (1. + v_known) / (1. - v_known) * \
            ( ( sqgm1 + cs_known ) / ( sqgm1 - cs_known ) * \
            ( sqgm1 - cs_star  )  / ( sqgm1 + cs_star  ) )**(-lr_sign * \
            2. / sqgm1)

        v_star = (a - 1.) / (a + 1.)

    return rho_star, v_star, eps_star
