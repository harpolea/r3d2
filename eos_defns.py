"""
Equations of state.
"""

import numpy as np

def eos_gamma_law(gamma):

    p_from_rho_eps = lambda rho, eps : (gamma - 1.0) * rho * eps
    h_from_rho_eps = lambda rho, eps : 1.0 + gamma * eps
    cs_from_rho_eps = lambda rho, eps : \
    np.sqrt(gamma * p_from_rho_eps(rho, eps) / (rho * h_from_rho_eps(rho, eps)))
    h_from_rho_p = lambda rho, p : 1.0 + gamma / (gamma - 1.0) * p / rho

    eos = {'p_from_rho_eps' : p_from_rho_eps,
           'h_from_rho_eps' : h_from_rho_eps,
           'cs_from_rho_eps' : cs_from_rho_eps,
           'h_from_rho_p' : h_from_rho_p}

    return eos

def eos_gamma_law_react(gamma, q, Cv):

    p_from_rho_eps = lambda rho, eps : (gamma - 1.0) * rho * (eps - q)
    h_from_rho_eps = lambda rho, eps : 1.0 + gamma * eps + (1.0 - gamma) * q
    cs_from_rho_eps = lambda rho, eps : \
    np.sqrt(gamma * (gamma - 1.0) * (eps - q) / \
    (1.0 + gamma * eps + (1.0 - gamma) * q))
    h_from_rho_p = lambda rho, p : 1.0 + gamma / (gamma - 1.0) * p / rho + q
    t_from_rho_eps = lambda rho, eps : (eps - q) / Cv

    eos = {'p_from_rho_eps' : p_from_rho_eps,
           'h_from_rho_eps' : h_from_rho_eps,
           'cs_from_rho_eps' : cs_from_rho_eps,
           'h_from_rho_p' : h_from_rho_p,
           't_from_rho_eps' : t_from_rho_eps}

    return eos

def eos_polytrope_law(gamma, gamma_th, rho_transition, k):

    def p_from_rho_eps(rho, eps):
        if (rho < rho_transition):
            p_cold = k[0] * rho**gamma[0]
            eps_cold = p_cold / rho / (gamma[0] - 1.)
        else:
            p_cold = k[1] * rho**gamma[1]
            eps_cold = p_cold / rho / (gamma[1] - 1.) - \
                k[1] * rho_transition**(gamma[1] - 1.) + \
                k[0] * rho_transition**(gamma[0] - 1.)
        
        p_th = max(0.0, (gamma_th - 1.0) * rho * (eps - eps_cold))

        return p_cold + p_th

    def h_from_rho_eps(rho, eps):
        if (rho < rho_transition):
            p_cold = k[0] * rho**gamma[0]
            eps_cold = p_cold / rho / (gamma[0] - 1.0)
        else:
            p_cold = k[1] * rho**gamma[1]
            eps_cold = p_cold / rho / (gamma[1] - 1.0) - \
                k[1] * rho_transition**(gamma[1] - 1.0) + \
                k[0] * rho_transition**(gamma[0] - 1.0)

        p_th = max(0., (gamma_th - 1.) * rho * (eps - eps_cold))

        return 1. + eps_cold + eps + (p_cold + p_th)/ rho

    def cs_from_rho_eps(rho, eps):
        return np.sqrt(gamma * p_from_rho_eps(rho, eps) / (rho * h_from_rho_eps(rho, eps)))

    def h_from_rho_p(rho, p):
        if (rho < rho_transition):
            p_cold = k[0] * rho**gamma[0]
            eps_cold = p_cold / rho / (gamma[0] - 1.0)
        else:
            p_cold = k[1] * rho**gamma[1]
            eps_cold = p_cold / rho / (gamma[1] - 1.0) - \
                k[1] * rho_transition**(gamma[1] - 1.0) + \
                k[0] * rho_transition**(gamma[0] - 1.0)
        
        p_th = max(0.0, p - p_cold)
        eps= 0.0 * p_th #FIXME
        
        return 1.0 + eps_cold + eps + p / rho

    eos = {'p_from_rho_eps' : p_from_rho_eps,
           'h_from_rho_eps' : h_from_rho_eps,
           'cs_from_rho_eps' : cs_from_rho_eps,
           'h_from_rho_p' : h_from_rho_p}

    return eos

