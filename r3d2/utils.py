# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 09:48:54 2016

@author: ih3
"""

import numpy
from scipy.integrate import odeint
import wave

def plot_P_v(rp, ax, var_to_plot = "velocity"):
    """
    Plot the curves joining states within phase space for the Riemann Problem.
    
    Parameters
    ----------
    
    rp : RiemannProblem
        The Riemann Problem to be plotted
    ax : matplotlib axis
        The axis on which to plot
    var_to_plot : string
        The name of the variable to plot on the y axis
    """
    
    if var_to_plot == "velocity":
        var_index = 1
        var_invert = False
        var_name = r"$v$"
    elif var_to_plot == "volume":
        var_index = 0
        var_invert = True
        var_name = r"$p$"
    else:
        raise(ValueError, "var_to_plot ({}) not recognized".format(var_to_plot))
    p_min=min([rp.state_l.p, rp.state_r.p, rp.p_star])
    p_max=max([rp.state_l.p, rp.state_r.p, rp.p_star])
    ax.plot(rp.state_l.v, rp.state_l.p, 'ko', label=r"$U_L$")
    ax.plot(rp.state_r.v, rp.state_r.p, 'k^', label=r"$U_R$")
    ax.plot(rp.state_star_l.v, rp.p_star, 'k*', label=r"$U_*$")
    dp = max(0.1, p_max-p_min)
    dp_fraction = min(0.5, 5*p_min)*dp
    
    p_l_1 = numpy.linspace(p_min-0.1*dp_fraction, rp.state_l.p-1e-3*dp_fraction)
    v_l_1 = numpy.zeros_like(p_l_1)
    p_l_2 = numpy.linspace(rp.state_l.p+1e-3*dp_fraction, p_max+0.1*dp_fraction)
    v_l_2 = numpy.zeros_like(p_l_2)
    p_r_1 = numpy.linspace(p_min-0.1*dp_fraction, rp.state_r.p-1e-3*dp_fraction)
    v_r_1 = numpy.zeros_like(p_r_1)
    p_r_2 = numpy.linspace(rp.state_r.p+1e-3*dp_fraction, p_max+0.1*dp_fraction)
    v_r_2 = numpy.zeros_like(p_r_2)
    for i, p in enumerate(p_l_1):
        w_all = odeint(wave.rarefaction_dwdp,
                       numpy.array([rp.state_l.rho, rp.state_l.v, rp.state_l.eps]),
                       [rp.state_l.p, p], rtol = 1e-12, atol = 1e-10,
                       args=((rp.state_l, 0)))
        v_l_1[i] = w_all[-1, var_index]
    for i, p in enumerate(p_r_1):
        w_all = odeint(wave.rarefaction_dwdp,
                       numpy.array([rp.state_r.rho, rp.state_r.v, rp.state_r.eps]),
                       [rp.state_r.p, p], rtol = 1e-12, atol = 1e-10,
                       args=((rp.state_r, 2)))
        v_r_1[i] = w_all[-1, var_index]
    for i, p in enumerate(p_l_2):
        j2, rho, eps, diffp = wave.mass_flux_squared(rp.state_l, p,
                                                rp.state_l.eos)
        v_shock, q_end = wave.post_discontinuity_state(p, rp.state_l,
                                                  -1, "", j2,
                                                  rho, eps, diffp)
        v_l_2[i] = q_end.prim()[var_index]
    for i, p in enumerate(p_r_2):
        j2, rho, eps, diffp = wave.mass_flux_squared(rp.state_r, p,
                                                rp.state_r.eos)
        v_shock, q_end = wave.post_discontinuity_state(p, rp.state_r,
                                                  1, "", j2,
                                                  rho, eps, diffp)
        v_r_2[i] = q_end.prim()[var_index]
    
    if var_invert:
        v_l_1 = 1.0 / v_l_1
        v_r_1 = 1.0 / v_r_1
        v_l_1 = 1.0 / v_l_2
        v_r_1 = 1.0 / v_r_2
        
    ax.plot(v_l_1, p_l_1, '--', label=r"${\cal R}_{\leftarrow}$")
    ax.plot(v_l_2, p_l_2, '-', label=r"${\cal S}_{\leftarrow}$")
    ax.plot(v_r_1, p_r_1, '--', label=r"${\cal R}_{\rightarrow}$")
    ax.plot(v_r_2, p_r_2, '-', label=r"${\cal S}_{\rightarrow}$")
    ax.set_xlabel(var_name)
    ax.set_ylabel(r"$p$")
    ax.set_ylim(0, p_max+0.1*dp_fraction)
    ax.legend()