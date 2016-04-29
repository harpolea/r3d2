# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 09:48:54 2016

@author: ih3
"""

import numpy
from scipy.integrate import odeint
from scipy.optimize import brentq
from matplotlib.transforms import offset_copy
import matplotlib.transforms as mtrans
from . import wave

def rarefaction(ps, q_start, lr_sign):
    vs = numpy.zeros_like(ps)
    for i, p in enumerate(ps):
        w_all = odeint(wave.rarefaction_dwdp,
                       numpy.array([q_start.rho, q_start.v,
                       q_start.eps]),
                       [q_start.p, p], rtol = 1e-12,
                       atol = 1e-10,
                       args=((q_start, lr_sign+1)))
        vs[i] = w_all[-1, 1]
    return vs

def shock(ps, q_start, lr_sign):
    vs = numpy.zeros_like(ps)
    for i, p in enumerate(ps):
        j2, rho, eps, diffp = wave.mass_flux_squared(q_start,
                                    p, q_start.eos)
        v_shock, q_end = wave.post_discontinuity_state(p,
                                    q_start,
                                    lr_sign, "", j2,
                                    rho, eps, diffp)
        vs[i] = q_end.v
    return vs

def deflagration(ps, q_start, lr_sign):
    vs = numpy.zeros_like(ps)
    for i, p in enumerate(ps):
        j2, rho, eps, diffp = wave.mass_flux_squared(q_start,
                                    p, q_start.eos)
        v_deflagration, q_end = wave.post_discontinuity_state(p, q_start,
                                    lr_sign, "", j2,
                                    rho, eps, diffp)

        if (lr_sign*(q_end.wavespeed(lr_sign+1) - v_deflagration) < 0):
            p_cjdf = brentq(wave.deflagration_root,
                            (1.0+1e-9)*p,
                            (1.0-1e-9)*q_start.p,
                            args=(q_start, q_start.eos,
                            lr_sign+1, ""))
            j2, rho, eps, dp = wave.mass_flux_squared(q_start,
                            p_cjdf, q_start.eos)
            v_deflagration, q_end = wave.post_discontinuity_state(p_cjdf, q_start,
                                    lr_sign, "", j2,
                                    rho, eps, diffp)
        vs[i] = q_end.v
    return vs

def detonation(ps, q_start, lr_sign):
    vs = numpy.zeros_like(ps)
    for i, p in enumerate(ps):
        j2, rho, eps, diffp = wave.mass_flux_squared(q_start,
                                    p, q_start.eos)
        if j2 < 0:
            # The single detonation is unphysical - must be unstable weak
            # detonation. So skip the calculation and make sure the CJ
            # calculation runs
            q_unknown = deepcopy(q_start)
            v_detonation = q_unknown.wavespeed(lr_sign+1) + 1
        else:
            v_detonation, q_unknown = wave.post_discontinuity_state(p, q_start,
                                    lr_sign, "", j2,
                                    rho, eps, diffp,
                                    q_start.eos)

        if (lr_sign*(q_unknown.wavespeed(lr_sign+1) - v_detonation) < 0):
            pmin = (1.0+1e-9)*min(q_start.p, p)
            pmax = max(q_start.p, p)
            fmin = wave.deflagration_root(pmin, q_start, q_start.eos, lr_sign+1, "")
            fmax = wave.deflagration_root(pmax, q_start, q_start.eos, lr_sign+1, "")
            while fmin * fmax > 0:
                pmax *= 2.0
                fmax = wave.deflagration_root(pmax, q_start, q_start.eos, lr_sign+1, "")
            p_cjdt = brentq(wave.deflagration_root, pmin, pmax,
                            args=(q_start, q_start.eos, lr_sign+1, ""))
            j2, rho, eps, dp = wave.mass_flux_squared(q_start, p_cjdt, q_start.eos)
            v_detonation, q_unknown = wave.post_discontinuity_state(p_cjdt, q_start,
                                   lr_sign, "", j2,
                                   rho, eps, diffp,
                                   q_start.eos)
        vs[i] = q_unknown.v
    return vs

def find_pre_ignition(v_s, j2, rho, lr_sign):
    A = rho**2 / (j2 *
        (v_s - lr_sign * numpy.sqrt(1 + rho**2 / j2)))

    return -0.5 * A + numpy.sqrt(1 + A * v_s), \
           -0.5 * A - numpy.sqrt(1 + A * v_s)


def plot_P_v(rp, ax, fig, var_to_plot = "velocity"):
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

    # TODO: This won't work with volume at the moment

    if var_to_plot == "velocity":
        var_index = 1
        var_invert = False
        var_name = r"$v$"
    elif var_to_plot == "volume":
        var_index = 0
        var_invert = True
        var_name = r"$V$"
    else:
        raise(ValueError, "var_to_plot ({}) not recognized".format(var_to_plot))
    p_min=min([rp.state_l.p, rp.state_r.p, rp.p_star])
    p_max=max([rp.state_l.p, rp.state_r.p, rp.p_star])
    if var_to_plot == "velocity":
        ax.plot(rp.state_l.v, rp.state_l.p, 'ko', label=r"$U_L$")
        ax.plot(rp.state_r.v, rp.state_r.p, 'k^', label=r"$U_R$")
        ax.plot(rp.state_star_l.v, rp.p_star, 'k*', label=r"$U_*$")
        trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
        ax.text(rp.state_l.v, rp.state_l.p, r"$U_L$", transform=trans_offset, horizontalalignment='center',
                 verticalalignment='bottom')
        trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=20, y=-6, units='dots')
        ax.text(rp.state_r.v, rp.state_r.p, r"$U_R$", transform=trans_offset, horizontalalignment='center',
                 verticalalignment='bottom')
        trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=20, y=-6, units='dots')
        ax.text(rp.state_star_l.v, rp.p_star, r"$U_*$", transform=trans_offset, horizontalalignment='center',
                 verticalalignment='bottom')
    else:
        ax.plot(1.0/rp.state_l.rho, rp.state_l.p, 'ko', label=r"$U_L$")
        ax.plot(1.0/rp.state_r.rho, rp.state_r.p, 'k^', label=r"$U_R$")
        ax.plot(1.0/rp.state_star_l.rho, rp.p_star, 'k*', label=r"$U_{*_L}$")
        ax.plot(1.0/rp.state_star_r.rho, rp.p_star, 'k<', label=r"$U_{*_R}$")
    dp = max(0.1, p_max-p_min)
    dp_fraction = min(0.5, 5*p_min)*dp

    p_l_1 = numpy.linspace(p_min-0.1*dp_fraction, rp.state_l.p-1e-3*dp_fraction)
    p_l_2 = numpy.linspace(rp.state_l.p+1e-3*dp_fraction, p_max+0.2*dp_fraction)
    p_l_df = numpy.linspace(p_min-0.1*dp_fraction, rp.state_l.p-1e-3*dp_fraction) # want this to go to CJ rather than pmin
    v_l_df = numpy.zeros_like(p_l_df)
    p_l_dt = numpy.linspace(rp.state_l.p+1e-3*dp_fraction, p_max+0.2*dp_fraction) # want this to go from CJ rather than pmax
    v_l_dt = numpy.zeros_like(p_l_dt)
    p_r_1 = numpy.linspace(p_min-0.1*dp_fraction, rp.state_r.p-1e-3*dp_fraction)
    p_r_2 = numpy.linspace(rp.state_r.p+1e-3*dp_fraction, p_max+0.2*dp_fraction)
    p_r_df = numpy.linspace(p_min-0.1*dp_fraction, rp.state_r.p-1e-3*dp_fraction) # want this to go to CJ rather than pmin
    v_r_df = numpy.zeros_like(p_r_df)
    p_r_dt = numpy.linspace(rp.state_r.p+1e-3*dp_fraction, p_max+0.2*dp_fraction) # want this to go from CJ rather than pmax
    v_r_dt = numpy.zeros_like(p_r_dt)

    # rarefaction curves
    v_l_1 = rarefaction(p_l_1, rp.state_l, -1)
    v_r_1 = rarefaction(p_r_1, rp.state_r, 1)

    # shock curves
    v_l_2 = shock(p_l_2, rp.state_l, -1)
    v_r_2 = shock(p_r_2, rp.state_r, 1)

    plot_inert = numpy.ones(4, dtype=numpy.bool)
    # check to make sure there is actually a wave - turn off plotting if not
    if len(rp.waves[0].wave_sections) == 1 and rp.waves[0].wave_sections[0].trivial:
        plot_inert[:2] = False
    if len(rp.waves[-1].wave_sections) == 1 and rp.waves[-1].wave_sections[0].trivial:
        plot_inert[-2:] = False

    # deflagration & detonation curves
    plot_burning = numpy.zeros(4, dtype=numpy.bool)
    if len(rp.waves[0].wave_sections) == 2 or rp.waves[0].wave_sections[0].type == 'Deflagration' or \
        rp.waves[0].wave_sections[0].type == 'Detonation':
        plot_burning[0] = True
        plot_burning[2] = True
        for s in rp.waves[0].wave_sections:
            if s.type == 'Deflagration':
                CJ_v, CJ_p, CJ_rho, CJ_eps = s.q_end.v, s.q_end.p, s.q_end.rho, s.q_end.eps
                p_max=max([p_max, CJ_p])
                CJ_q = s.q_end
                ax.plot(CJ_v, CJ_p, 'ko', label=r"$CJDF$")
                trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
                ax.text(CJ_v, CJ_p, r"$CJ$", transform=trans_offset, horizontalalignment='center',
                verticalalignment='bottom')
                p_l_df = numpy.linspace(CJ_p+1e-3*dp_fraction, rp.state_l.p-1e-3*dp_fraction)
                p_l_dt = numpy.linspace(rp.state_l.p+1e-3*dp_fraction, p_max+0.2*dp_fraction)
                plot_inert[1] = False # get rid of left shock, as this is now a detonation
                # cut off rarefaction at CJ point
                p_l_1 = numpy.linspace(p_min-0.1*dp_fraction, CJ_p-1e-3*dp_fraction)
                v_l_1 = rarefaction(p_l_1, CJ_q, -1)
                v_l_df = deflagration(p_l_df, CJ_q, -1)

            elif s.type == 'Detonation':
                CJ_v, CJ_p, CJ_rho, CJ_eps = s.q_end.v, s.q_end.p, s.q_end.rho, s.q_end.eps
                p_max=max([p_max, CJ_p])
                CJ_q = s.q_end
                ax.plot(CJ_v, CJ_p, 'ko', label=r"$CJDT$")
                trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
                ax.text(CJ_v, CJ_p, r"$CJ$", transform=trans_offset, horizontalalignment='center',
                verticalalignment='bottom')
                p_l_dt = numpy.linspace(rp.state_l.p+1e-3*dp_fraction, CJ_p-1e-3*dp_fraction)
                p_l_df = numpy.linspace(p_min-0.1*dp_fraction, rp.state_l.p-1e-3*dp_fraction)
                plot_inert[0] = False # get rid of left rarefaction - it's now a deflagration
                # cut off shock at CJ point
                p_l_2 = numpy.linspace(CJ_p+1e-3*dp_fraction, p_max+0.2*dp_fraction)
                # redo shock
                v_l_2 = shock(p_l_2, CJ_q, -1)
                v_l_df = deflagration(p_l_df, rp.state_l, -1)

            if s.type == 'Detonation' or s.type == 'Deflagration':
                v_l_dt = detonation(p_l_dt, rp.state_l, -1)

    # right wave
    if len(rp.waves[-1].wave_sections) == 2 or rp.waves[-1].wave_sections[0].type == 'Deflagration' or\
        rp.waves[-1].wave_sections[0].type == 'Detonation':
        plot_burning[1] = True
        plot_burning[3] = True
        for s in rp.waves[-1].wave_sections:

            if s.type == 'Deflagration':
                CJ_v, CJ_p, CJ_rho, CJ_eps = s.q_end.v, s.q_end.p, s.q_end.rho, s.q_end.eps
                CJ_q = s.q_end
                p_max=max([p_max, CJ_p])
                ax.plot(CJ_v, CJ_p, 'ko', label=r"$CJDF$")
                trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
                ax.text(CJ_v, CJ_p, r"$CJ$", transform=trans_offset, horizontalalignment='center',
                verticalalignment='bottom')
                p_r_df = numpy.linspace(CJ_p+1e-3*dp_fraction, rp.state_r.p-1e-3*dp_fraction)
                p_r_dt = numpy.linspace(rp.state_r.p+1e-3*dp_fraction, p_max+0.2*dp_fraction)
                plot_inert[3] = False # get rid of right shock - now a deflagration
                # cut off rarefaction at CJ point
                p_r_1 = numpy.linspace(p_min-0.1*dp_fraction, CJ_p-1e-3*dp_fraction)
                v_r_1 = rarefaction(p_r_1, CJ_q, 1)
                v_r_df = deflagration(p_r_df, CJ_q, 1)

            elif s.type == 'Detonation':
                CJ_v, CJ_p, CJ_rho, CJ_eps = s.q_end.v, s.q_end.p, s.q_end.rho, s.q_end.eps
                CJ_q = s.q_end
                p_max=max([p_max, CJ_p])
                ax.plot(CJ_v, CJ_p, 'ko', label=r"$CJDT$")
                trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
                ax.text(CJ_v, CJ_p, r"$CJ$", transform=trans_offset, horizontalalignment='center',
                verticalalignment='bottom')
                p_r_dt = numpy.linspace(rp.state_r.p+1e-3*dp_fraction, CJ_p-1e-3*dp_fraction)
                p_r_df = numpy.linspace(p_min-0.1*dp_fraction, rp.state_r.p-1e-3*dp_fraction)
                plot_inert[2] = False # get rid of right rarefaction - it's now a deflagration
                # cut off shock at CJ point
                p_r_2 = numpy.linspace(CJ_p+1e-3*dp_fraction, p_max+0.2*dp_fraction)
                v_r_2 = shock(p_r_2, CJ_q, 1)
                v_r_df = rarefaction(p_r_df, rp.state_r, 1)

            if s.type == 'Detonation' or s.type == 'Deflagration':
                v_r_dt = detonation(p_r_dt, rp.state_r, 1)

    # put this here as going to change it for 3 wave case
    ax.set_ylim(0, p_max+0.2*dp_fraction)
    #plt.xlim(v_min-0.5*dv, v_max+0.5*dv)

    # now for weird 3 wave case
    if len(rp.waves[0].wave_sections) == 3:
        #ax.set_ylim(0, p_max+6*dp_fraction)

        # ignition points
        i_p, i_v = rp.waves[0].wave_sections[1].q_start.p, rp.waves[0].wave_sections[1].q_start.v
        ax.plot(i_v, i_p, 'ko')
        trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=15, y=-6, units='dots')
        ax.text(i_v, i_p, r"$i_S$", transform=trans_offset, horizontalalignment='center',
             verticalalignment='bottom')
        j2, rho, eps, diffp = wave.mass_flux_squared(
            rp.waves[0].wave_sections[1].q_start,
            rp.waves[0].wave_sections[1].q_end.p,
            rp.waves[0].wave_sections[1].q_start.eos)
        v1, v2 = find_pre_ignition(
            rp.waves[0].wave_sections[1].wavespeed,
            j2, rho, -1)
        # choose the one that is closest to other ignition velocity
        if (i_v - v1)**2 < (i_v - v2):
            i_v2 = v1
        else:
            i_v2 = v2

        for s in rp.waves[0].wave_sections:
            # CJ point
            CJ_v, CJ_p, CJ_rho, CJ_eps = s.q_end.v, s.q_end.p, s.q_end.rho, s.q_end.eps
            CJ_q = s.q_end

            p_max=max([p_max, CJ_p, i_p])
            ax.set_ylim(0, p_max+0.2*dp_fraction)

            if s.type == 'Deflagration':
                plot_burning[0] = True
                #CJ_v, CJ_p, CJ_rho, CJ_eps = s.q_end.v, s.q_end.p, s.q_end.rho, s.q_end.eps
                ax.plot(CJ_v, CJ_p, 'ko', label=r"$CJDF$")
                trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
                ax.text(CJ_v, CJ_p, r"$CJ$", transform=trans_offset, horizontalalignment='center',
             verticalalignment='bottom')
                # cut off rarefaction and shock at CJ point
                p_l_1 = numpy.linspace(p_min-0.1*dp_fraction, CJ_p-1e-6*dp_fraction)
                v_l_1 = rarefaction(p_l_1, CJ_q, -1)
                p_l_2 = numpy.linspace(rp.state_l.p+1e-6*dp_fraction, i_p)
                v_l_2 = shock(p_l_2, rp.state_l, -1)
                p_l_df = numpy.linspace(CJ_p+1e-6*dp_fraction, p_max+0.1*dp_fraction)
                v_l_df = rarefaction(p_l_df, CJ_q, -1)

            elif s.type == 'Detonation':
                plot_burning[2] = True
                #CJ_v, CJ_p, CJ_rho, CJ_eps = s.q_end.v, s.q_end.p, s.q_end.rho, s.q_end.eps
                ax.plot(CJ_v, CJ_p, 'ko', label=r"$CJDT$")
                trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
                ax.text(CJ_v, CJ_p, r"$CJ$", transform=trans_offset, horizontalalignment='center',
             verticalalignment='bottom')
                p_l_dt = numpy.linspace(rp.state_l.p+1e-3*dp_fraction, CJ_p+1e-6*dp_fraction)
                v_l_dt = detonation(p_l_dt, rp.state_l, -1)
                # cut off rarefaction and shock at CJ point
                p_l_1 = numpy.linspace(CJ_p+1e-6*dp_fraction, p_max+0.2*dp_fraction)
                v_l_1 = rarefaction(p_l_1, CJ_q, -1)
                p_l_2 = numpy.linspace(p_min-0.1*dp_fraction, CJ_p-1e-6*dp_fraction)
                v_l_2 = shock(p_l_2, CJ_q, -1)

            if s.type == 'Deflagration' or s.type == 'Detonation':
                distance = (i_v2 - v_l_df)**2
                # get approximate ignition point
                i_p2 = p_l_df[numpy.argmin(distance)]
                i_v2 = v_l_df[numpy.argmin(distance)]
                ax.plot(i_v2, i_p2, 'ko')
                trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
                ax.text(i_v2, i_p2, r"$i$", transform=trans_offset, horizontalalignment='center',
                     verticalalignment='bottom')
                # now cut off deflagration
                p_l_df = p_l_df[:numpy.argmin(distance)+1]
                v_l_df = v_l_df[:numpy.argmin(distance)+1]
                ax.plot([i_v, i_v2], [i_p, i_p2], '--k')

                p_max=max([p_max, i_p2])
                ax.set_ylim(0, p_max+0.2*dp_fraction)

    # right wave
    if len(rp.waves[-1].wave_sections) == 3:
         # ignition points
        i_p, i_v = rp.waves[-1].wave_sections[0].q_end.p, rp.waves[-1].wave_sections[0].q_end.v
        ax.plot(i_v, i_p, 'ko')
        trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=15, y=-6, units='dots')
        ax.text(i_v, i_p, r"$i$", transform=trans_offset, horizontalalignment='center',
             verticalalignment='bottom')
        j2, rho, eps, diffp = wave.mass_flux_squared(
            rp.waves[-1].wave_sections[1].q_start,
            rp.waves[-1].wave_sections[1].q_end.p,
            rp.waves[-1].wave_sections[1].q_start.eos)
        v1, v2 = find_pre_ignition(
            rp.waves[-1].wave_sections[1].wavespeed,
            j2, rho, 1)
        # choose the one that is closest to other ignition velocity
        if (i_v - v1)**2 < (i_v - v2):
            i_v2 = v1
        else:
            i_v2 = v2

        for s in rp.waves[-1].wave_sections:
            CJ_v, CJ_p, CJ_rho, CJ_eps = s.q_end.v, s.q_end.p, s.q_end.rho, s.q_end.eps
            CJ_q = s.q_end
            p_max=max([p_max, CJ_p, i_p])
            ax.set_ylim(0, p_max+0.2*dp_fraction)

            if s.type == 'Deflagration':
                plot_burning[1] = True
                #CJ_v, CJ_p, CJ_rho, CJ_eps = s.q_end.v, s.q_end.p, s.q_end.rho, s.q_end.eps
                ax.plot(CJ_v, CJ_p, 'ko', label=r"$CJDF$")
                trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
                ax.text(CJ_v, CJ_p, r"$CJ$", transform=trans_offset, horizontalalignment='center',
                verticalalignment='bottom')
                # cut off rarefaction at CJ point
                p_r_1 = numpy.linspace(p_min-0.1*dp_fraction, CJ_p-1e-3*dp_fraction)
                v_r_1 = rarefaction(p_r_1, CJ_q, 1)
                p_r_2 = numpy.linspace(rp.state_r.p+1e-3*dp_fraction, i_p)
                v_r_2 = shock(p_r_2, rp.state_r, 1)
                p_r_df = numpy.linspace(CJ_p+1e-6*dp_fraction, p_max+5*dp_fraction)
                v_r_df = deflagration(p_r_df, CJ_q, 1)

            elif s.type == 'Detonation':
                plot_burning[3] = True
                #CJ_v, CJ_p, CJ_rho, CJ_eps = s.q_end.v, s.q_end.p, s.q_end.rho, s.q_end.eps
                ax.plot(CJ_v, CJ_p, 'ko', label=r"$CJDT$")
                trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
                ax.text(CJ_v, CJ_p, r"$CJ$", transform=trans_offset, horizontalalignment='center',
                verticalalignment='bottom')
                # cut off shock at CJ point
                p_r_1 = numpy.linspace(rp.state_r.p+1e-3*dp_fraction, p_max+0.2*dp_fraction)
                v_r_1 = rarefaction(p_r_1, rp.state_r, 1)
                p_r_2 = numpy.linspace(p_min-0.1*dp_fraction, CJ_p-1e-3*dp_fraction)
                v_r_2 = shock(p_r_2, CJ_q, 1)
                p_r_dt = numpy.linspace(CJ_p+1e-6*dp_fraction, p_max+5*dp_fraction)
                v_r_dt = detonation(p_r_dt, CJ_q, 1)

            if s.type == 'Deflagration' or s.type == 'Detonation':
                # find point of intersection of the two curves
                distance = (i_v2 - v_r_df)**2
                # get approximate ignition point
                i_p2 = p_r_df[numpy.argmin(distance)]
                i_v2 = v_r_df[numpy.argmin(distance)]
                ax.plot(i_v2, i_p2, 'ko')
                trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
                ax.text(i_v2, i_p2, r"$i$", transform=trans_offset, horizontalalignment='center',
                verticalalignment='bottom')
                # now cut off deflagration
                p_r_df = p_r_df[:numpy.argmin(distance)+1]
                v_r_df = v_r_df[:numpy.argmin(distance)+1]
                ax.plot([i_v, i_v2], [i_p, i_p2], '--k')

                p_max=max([p_max, i_p2])
                ax.set_ylim(0, p_max+0.2*dp_fraction)


    if plot_inert[0]: ax.plot(v_l_1, p_l_1, '--', label=r"${\cal R}_{\leftarrow}$")
    if plot_inert[1]: ax.plot(v_l_2, p_l_2, '-', label=r"${\cal S}_{\leftarrow}$")
    if plot_inert[2]: ax.plot(v_r_1, p_r_1, '--', label=r"${\cal R}_{\rightarrow}$")
    if plot_inert[3]: ax.plot(v_r_2, p_r_2, '-', label=r"${\cal S}_{\rightarrow}$")
    if plot_burning[0]: ax.plot(v_l_df, p_l_df, '-', label=r"${\cal DF}_{\leftarrow}$")
    if plot_burning[1]: ax.plot(v_r_df, p_r_df, '-', label=r"${\cal DF}_{\rightarrow}$")
    if plot_burning[2]: ax.plot(v_l_dt, p_l_dt, '-', label=r"${\cal DT}_{\leftarrow}$")
    if plot_burning[3]: ax.plot(v_r_dt, p_r_dt, '-', label=r"${\cal DT}_{\rightarrow}$")


    #v_r_1[i] = w_all[-1, var_index]

    #v_r_2[i] = q_end.prim()[var_index]

    if var_invert:
        v_l_1 = 1.0 / v_l_1
        v_r_1 = 1.0 / v_r_1
        v_l_2 = 1.0 / v_l_2
        v_r_2 = 1.0 / v_r_2

    ax.set_xlabel(var_name)
    ax.set_ylabel(r"$p$")
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
