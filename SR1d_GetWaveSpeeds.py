import numpy as np

def SR1d_GetWaveSpeeds( l, s_l, s_r, r ):

    wave_speeds = np.zeros((5, 1))

    p_l   =   l[3]
    p_s_l = s_l[3]
    p_s_r = s_r[3]
    p_r   =   r[3]

    h_l   =   l[5]
    h_s_l = s_l[5]
    h_r   =   r[5]
    h_s_r = s_r[5]

    rho_l   =   l[0]
    rho_s_l = s_l[0]
    rho_r   =   r[0]
    rho_s_r = s_r[0]

    v_l   =   l[1]
    v_s_l = s_l[1]
    v_r   =   r[1]
    v_s_r = s_r[1]

    cs_l   = np.sqrt(  l[6])
    cs_s_l = np.sqrt(s_l[6])
    cs_r   = np.sqrt(  r[6])
    cs_s_r = np.sqrt(s_r[6])
    # Left wave
    if (p_s_l > p_l): # Shock
        w2 = l[4]**2
        j = -np.sqrt( (p_s_l - p_l) / (h_l / rho_l - h_s_l / rho_s_l) )
        a = j**2 + rho_l**2 * w2
        b = -v_l * rho_l**2 * w2
        wave_speeds[0] = (-b - j**2 * np.sqrt(1. + (rho_l / j)**2)) / a
        wave_speeds[1] = wave_speeds[0]
    else: # Rarefaction
        wave_speeds[0] = (v_l - cs_l) / (1. - v_l * cs_l)
        wave_speeds[1] = (v_s_l - cs_s_l) / (1. - v_s_l * cs_s_l)

    # Contact
    wave_speeds[2] = s_l[1]

    # Right wave
    if (p_s_r > p_r): # Shock
        w2 = r[4]**2
        j = np.sqrt( (p_s_r - p_r) / (h_r / rho_r - h_s_r / rho_s_r) )
        a = j**2 + rho_r**2 * w2
        b = -v_r * rho_r**2 * w2
        wave_speeds[3] = (-b + j**2 * np.sqrt(1. + (rho_r / j)**2)) / a
        wave_speeds[4] = wave_speeds[3]
    else: # Rarefaction
        wave_speeds[3] = (v_s_r - cs_s_r) / (1. - v_s_r * cs_s_r)
        wave_speeds[4] = (v_r - cs_r) / (1. - v_r * cs_r)

    return wave_speeds
