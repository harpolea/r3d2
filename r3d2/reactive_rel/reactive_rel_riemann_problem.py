# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:23:51 2016

@author: ih3
"""

import numpy
from matplotlib import pyplot
import matplotlib.transforms as mtrans
from IPython.core.pylabtools import print_figure

from r3d2 import RiemannProblem, utils
from reactive_rel_wave import ReactiveRelWave, ReactiveRelWaveSection


class ReactiveRelRiemannProblem(RiemannProblem):
    """
    This is a more general Riemann Problem class.

    Allows for different EOSs on both sides (as required for burning problems).
    Uses the State class.
    """

    def find_delta_v(self, p_star_guess):

        wave_l = ReactiveRelWave(self.state_l, p_star_guess, 0)
        wave_r = ReactiveRelWave(self.state_r, p_star_guess, 2)

        return wave_l.q_r.v - wave_r.q_l.v

    def make_waves(self):
        wave_l = ReactiveRelWave(self.state_l, self.p_star, 0)
        wave_r = ReactiveRelWave(self.state_r, self.p_star, 2)
        self.state_star_l = wave_l.q_r
        self.state_star_r = wave_r.q_l
        self.waves = [wave_l,
                      ReactiveRelWave(self.state_star_l, self.state_star_r, 1),
                      wave_r]

    def _figure_data(self, fig_format):
        fig, axs = pyplot.subplots(3,3)
        ax = axs[0,0]
        for w in self.waves[0], self.waves[2]:
            if len(w.wavespeed)==1:
                ax.plot([0, w.wavespeed[0]], [0, 1], 'k-', linewidth=3)
            elif len(w.wavespeed)==2:
                xi_end = numpy.linspace(w.wavespeed[0], w.wavespeed[1], 5)
                ax.fill_between([0, xi_end[0], xi_end[-1], 0],
                                [0, 1, 1, 0], color='k', alpha=0.1)
                for xi in xi_end:
                    ax.plot([0, xi], [0, 1], 'k-', linewidth=1)
        if len(self.waves[1].wavespeed):
            ax.plot([0, self.waves[1].wavespeed[0]], [0, 1], 'k--', linewidth=1)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$t$")
        ax.set_title("Characteristics")
        names = [r"$\rho$", r"$v$", r"$v_t$", r"$\epsilon$", r"$p$", r"$W$",
                 r"$h$", r"$c_s$"]
        xi = [-1.05]
        data = self.state_l.state()
        for wave in self.waves:
            xi_wave, data_wave = wave.plotting_data()
            xi = numpy.hstack((xi, xi_wave))
            data = numpy.vstack((data, data_wave))
        xi = numpy.hstack((xi, [1.05]))
        data = numpy.vstack((data, self.state_r.state()))
        for ax_j in range(3):
            for ax_i in range(3):
                if ax_i == 0 and ax_j == 0:
                    continue
                nvar = ax_i*3 + ax_j - 1
                axs[ax_i, ax_j].plot(xi, data[:, nvar])
                var_max = numpy.max(data[:, nvar])
                var_min = numpy.min(data[:, nvar])
                d_var = max(var_max - var_min,
                            0.01 * min(abs(var_min), abs(var_max)), 0.01)
                axs[ax_i, ax_j].set_xlim(-1.05, 1.05)
                axs[ax_i, ax_j].set_ylim(var_min - 0.05 * d_var,
                                         var_max + 0.05 * d_var)
                axs[ax_i, ax_j].set_xlabel(r"$\xi$")
                axs[ax_i, ax_j].set_ylabel(names[nvar])
        fig.tight_layout()
        data = print_figure(fig, fig_format)
        pyplot.close(fig)
        return data


    def _repr_latex_(self):
        s = r"$\begin{cases} "
        s += self.state_l.latex_string()
        s += r",\\ "
        s += self.state_r.latex_string()
        s += r", \end{cases} \quad \implies \quad "
        for wave in self.waves:
            s+= wave.name
        s += r", \quad p_* = {:.4f}, \quad".format(self.p_star)
        s += r"\begin{cases} "
        for wave in self.waves[:-1]:
            s += wave.latex_string() + r",\\ "
        s += self.waves[-1].latex_string()
        s += r", \end{cases} \quad \begin{cases} "
        s += self.state_star_l.latex_string()
        s += r",\\ "
        s += self.state_star_r.latex_string()
        s += r". \end{cases}$"
        return s

    def plot_P_v(self, ax, fig, var_to_plot = "velocity"):
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
            raise ValueError("var_to_plot ({}) not recognized".format(var_to_plot))
        p_min=min([self.state_l.p, self.state_r.p, self.p_star])
        p_max=max([self.state_l.p, self.state_r.p, self.p_star])

        if var_to_plot == "velocity":
            ax.plot(self.state_l.v, self.state_l.p, 'ko', label=r"$U_L$")
            ax.plot(self.state_r.v, self.state_r.p, 'k^', label=r"$U_R$")
            ax.plot(self.state_star_l.v, self.p_star, 'k*', label=r"$U_*$")
            trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
            ax.text(self.state_l.v, self.state_l.p, r"$U_L$", transform=trans_offset, horizontalalignment='center',
                     verticalalignment='bottom')
            trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=20, y=-6, units='dots')
            ax.text(self.state_r.v, self.state_r.p, r"$U_R$", transform=trans_offset, horizontalalignment='center',
                     verticalalignment='bottom')
            trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=20, y=-6, units='dots')
            ax.text(self.state_star_l.v, self.p_star, r"$U_*$", transform=trans_offset, horizontalalignment='center',
                     verticalalignment='bottom')
        else:
            ax.plot(1.0/self.state_l.rho, self.state_l.p, 'ko', label=r"$U_L$")
            ax.plot(1.0/self.state_r.rho, self.state_r.p, 'k^', label=r"$U_R$")
            ax.plot(1.0/self.state_star_l.rho, self.p_star, 'k*', label=r"$U_{*_L}$")
            ax.plot(1.0/self.state_star_r.rho, self.p_star, 'k<', label=r"$U_{*_R}$")
        dp = max(0.1, p_max-p_min)
        dp_fraction = min(0.5, 5*p_min)*dp

        p_l_1 = numpy.linspace(p_min-0.1*dp_fraction, self.state_l.p-1e-3*dp_fraction)
        p_l_2 = numpy.linspace(self.state_l.p+1e-3*dp_fraction, p_max+0.2*dp_fraction)
        p_l_df = numpy.linspace(p_min-0.1*dp_fraction, self.state_l.p-1e-3*dp_fraction) # want this to go to CJ rather than pmin
        v_l_df = numpy.zeros_like(p_l_df)
        p_l_dt = numpy.linspace(self.state_l.p+1e-3*dp_fraction, p_max+0.2*dp_fraction) # want this to go from CJ rather than pmax
        v_l_dt = numpy.zeros_like(p_l_dt)
        p_r_1 = numpy.linspace(p_min-0.1*dp_fraction, self.state_r.p-1e-3*dp_fraction)
        p_r_2 = numpy.linspace(self.state_r.p+1e-3*dp_fraction, p_max+0.2*dp_fraction)
        p_r_df = numpy.linspace(p_min-0.1*dp_fraction, self.state_r.p-1e-3*dp_fraction) # want this to go to CJ rather than pmin
        v_r_df = numpy.zeros_like(p_r_df)
        p_r_dt = numpy.linspace(self.state_r.p+1e-3*dp_fraction, p_max+0.2*dp_fraction) # want this to go from CJ rather than pmax
        v_r_dt = numpy.zeros_like(p_r_dt)

        # rarefaction curves
        v_l_1 = utils.rarefaction(p_l_1, self.state_l, -1)
        v_r_1 = utils.rarefaction(p_r_1, self.state_r, 1)

        # shock curves
        v_l_2 = utils.shock(p_l_2, self.state_l, -1)
        v_r_2 = utils.shock(p_r_2, self.state_r, 1)

        plot_inert = numpy.ones(4, dtype=numpy.bool)
        # check to make sure there is actually a wave - turn off plotting if not
        if len(self.waves[0].wave_sections) == 1 and self.waves[0].wave_sections[0].trivial:
            plot_inert[:2] = False
        if len(self.waves[-1].wave_sections) == 1 and self.waves[-1].wave_sections[0].trivial:
            plot_inert[-2:] = False

        # deflagration & detonation curves
        plot_burning = numpy.zeros(4, dtype=numpy.bool)
        if len(self.waves[0].wave_sections) == 2 or self.waves[0].wave_sections[0].type == 'Deflagration' or \
            self.waves[0].wave_sections[0].type == 'Detonation':
            plot_burning[0] = True
            plot_burning[2] = True
            for s in self.waves[0].wave_sections:
                if s.type == 'Deflagration':
                    CJ_v, CJ_p, CJ_rho, CJ_eps = s.q_end.v, s.q_end.p, s.q_end.rho, s.q_end.eps
                    p_max=max([p_max, CJ_p])
                    CJ_q = s.q_end
                    ax.plot(CJ_v, CJ_p, 'ko', label=r"$CJDF$")
                    trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
                    ax.text(CJ_v, CJ_p, r"$CJ$", transform=trans_offset, horizontalalignment='center',
                    verticalalignment='bottom')
                    p_l_df = numpy.linspace(CJ_p+1e-3*dp_fraction, self.state_l.p-1e-3*dp_fraction)
                    p_l_dt = numpy.linspace(self.state_l.p+1e-3*dp_fraction, p_max+0.2*dp_fraction)
                    plot_inert[1] = False # get rid of left shock, as this is now a detonation
                    # cut off rarefaction at CJ point
                    p_l_1 = numpy.linspace(p_min-0.1*dp_fraction, CJ_p-1e-3*dp_fraction)
                    v_l_1 = utils.rarefaction(p_l_1, CJ_q, -1)
                    v_l_df = utils.deflagration(p_l_df, CJ_q, -1)

                elif s.type == 'Detonation':
                    CJ_v, CJ_p, CJ_rho, CJ_eps = s.q_end.v, s.q_end.p, s.q_end.rho, s.q_end.eps
                    p_max=max([p_max, CJ_p])
                    CJ_q = s.q_end
                    ax.plot(CJ_v, CJ_p, 'ko', label=r"$CJDT$")
                    trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
                    ax.text(CJ_v, CJ_p, r"$CJ$", transform=trans_offset, horizontalalignment='center',
                    verticalalignment='bottom')
                    p_l_dt = numpy.linspace(self.state_l.p+1e-3*dp_fraction, CJ_p-1e-3*dp_fraction)
                    p_l_df = numpy.linspace(p_min-0.1*dp_fraction, self.state_l.p-1e-3*dp_fraction)
                    plot_inert[0] = False # get rid of left rarefaction - it's now a deflagration
                    # cut off shock at CJ point
                    p_l_2 = numpy.linspace(CJ_p+1e-3*dp_fraction, p_max+0.2*dp_fraction)
                    # redo shock
                    v_l_2 = utils.shock(p_l_2, CJ_q, -1)
                    v_l_df = utils.deflagration(p_l_df, self.state_l, -1)

                if s.type == 'Detonation' or s.type == 'Deflagration':
                    v_l_dt = utils.detonation(p_l_dt, self.state_l, -1)

        # right wave
        if len(self.waves[-1].wave_sections) == 2 or self.waves[-1].wave_sections[0].type == 'Deflagration' or\
            self.waves[-1].wave_sections[0].type == 'Detonation':
            plot_burning[1] = True
            plot_burning[3] = True
            for s in self.waves[-1].wave_sections:

                if s.type == 'Deflagration':
                    CJ_v, CJ_p, CJ_rho, CJ_eps = s.q_end.v, s.q_end.p, s.q_end.rho, s.q_end.eps
                    CJ_q = s.q_end
                    p_max=max([p_max, CJ_p])
                    ax.plot(CJ_v, CJ_p, 'ko', label=r"$CJDF$")
                    trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
                    ax.text(CJ_v, CJ_p, r"$CJ$", transform=trans_offset, horizontalalignment='center',
                    verticalalignment='bottom')
                    p_r_df = numpy.linspace(CJ_p+1e-3*dp_fraction, self.state_r.p-1e-3*dp_fraction)
                    p_r_dt = numpy.linspace(self.state_r.p+1e-3*dp_fraction, p_max+0.2*dp_fraction)
                    plot_inert[3] = False # get rid of right shock - now a deflagration
                    # cut off rarefaction at CJ point
                    p_r_1 = numpy.linspace(p_min-0.1*dp_fraction, CJ_p-1e-3*dp_fraction)
                    v_r_1 = utils.rarefaction(p_r_1, CJ_q, 1)
                    v_r_df = utils.deflagration(p_r_df, CJ_q, 1)

                elif s.type == 'Detonation':
                    CJ_v, CJ_p, CJ_rho, CJ_eps = s.q_end.v, s.q_end.p, s.q_end.rho, s.q_end.eps
                    CJ_q = s.q_end
                    p_max=max([p_max, CJ_p])
                    ax.plot(CJ_v, CJ_p, 'ko', label=r"$CJDT$")
                    trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
                    ax.text(CJ_v, CJ_p, r"$CJ$", transform=trans_offset, horizontalalignment='center',
                    verticalalignment='bottom')
                    p_r_dt = numpy.linspace(self.state_r.p+1e-3*dp_fraction, CJ_p-1e-3*dp_fraction)
                    p_r_df = numpy.linspace(p_min-0.1*dp_fraction, self.state_r.p-1e-3*dp_fraction)
                    plot_inert[2] = False # get rid of right rarefaction - it's now a deflagration
                    # cut off shock at CJ point
                    p_r_2 = numpy.linspace(CJ_p+1e-3*dp_fraction, p_max+0.2*dp_fraction)
                    v_r_2 = utils.shock(p_r_2, CJ_q, 1)
                    v_r_df = utils.rarefaction(p_r_df, self.state_r, 1)

                if s.type == 'Detonation' or s.type == 'Deflagration':
                    v_r_dt = utils.detonation(p_r_dt, self.state_r, 1)

        # put this here as going to change it for 3 wave case
        ax.set_ylim(0, p_max+0.2*dp_fraction)
        #plt.xlim(v_min-0.5*dv, v_max+0.5*dv)

        # now for weird 3 wave case
        if len(self.waves[0].wave_sections) == 3:
            #ax.set_ylim(0, p_max+6*dp_fraction)

            # ignition points
            i_p, i_v = self.waves[0].wave_sections[1].q_start.p, self.waves[0].wave_sections[1].q_start.v
            ax.plot(i_v, i_p, 'ko')
            trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=15, y=-6, units='dots')
            ax.text(i_v, i_p, r"$i_S$", transform=trans_offset, horizontalalignment='center',
                 verticalalignment='bottom')
            j2, rho, eps, diffp = ReactiveRelWaveSection.mass_flux_squared(
                self.waves[0].wave_sections[1].q_start,
                self.waves[0].wave_sections[1].q_end.p,
                self.waves[0].wave_sections[1].q_start.eos)
            v1, v2 = utils.find_pre_ignition(
                self.waves[0].wave_sections[1].wavespeed,
                j2, rho, -1)
            # choose the one that is closest to other ignition velocity
            if (i_v - v1)**2 < (i_v - v2):
                i_v2 = v1
            else:
                i_v2 = v2

            for s in self.waves[0].wave_sections:
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
                    v_l_1 = utils.rarefaction(p_l_1, CJ_q, -1)
                    p_l_2 = numpy.linspace(self.state_l.p+1e-6*dp_fraction, i_p)
                    v_l_2 = utils.shock(p_l_2, self.state_l, -1)
                    p_l_df = numpy.linspace(CJ_p+1e-6*dp_fraction, p_max+0.1*dp_fraction)
                    v_l_df = utils.rarefaction(p_l_df, CJ_q, -1)

                elif s.type == 'Detonation':
                    plot_burning[2] = True
                    #CJ_v, CJ_p, CJ_rho, CJ_eps = s.q_end.v, s.q_end.p, s.q_end.rho, s.q_end.eps
                    ax.plot(CJ_v, CJ_p, 'ko', label=r"$CJDT$")
                    trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
                    ax.text(CJ_v, CJ_p, r"$CJ$", transform=trans_offset, horizontalalignment='center',
                 verticalalignment='bottom')
                    p_l_dt = numpy.linspace(self.state_l.p+1e-3*dp_fraction, CJ_p+1e-6*dp_fraction)
                    v_l_dt = utils.detonation(p_l_dt, self.state_l, -1)
                    # cut off rarefaction and shock at CJ point
                    p_l_1 = numpy.linspace(CJ_p+1e-6*dp_fraction, p_max+0.2*dp_fraction)
                    v_l_1 = utils.rarefaction(p_l_1, CJ_q, -1)
                    p_l_2 = numpy.linspace(p_min-0.1*dp_fraction, CJ_p-1e-6*dp_fraction)
                    v_l_2 = utils.shock(p_l_2, CJ_q, -1)

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
        if len(self.waves[-1].wave_sections) == 3:
             # ignition points
            i_p, i_v = self.waves[-1].wave_sections[0].q_end.p, self.waves[-1].wave_sections[0].q_end.v
            ax.plot(i_v, i_p, 'ko')
            trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=15, y=-6, units='dots')
            ax.text(i_v, i_p, r"$i$", transform=trans_offset, horizontalalignment='center',
                 verticalalignment='bottom')
            j2, rho, eps, diffp = ReactiveRelWaveSection.mass_flux_squared(
                self.waves[-1].wave_sections[1].q_start,
                self.waves[-1].wave_sections[1].q_end.p,
                self.waves[-1].wave_sections[1].q_start.eos)
            v1, v2 = utils.find_pre_ignition(
                self.waves[-1].wave_sections[1].wavespeed,
                j2, rho, 1)
            # choose the one that is closest to other ignition velocity
            if (i_v - v1)**2 < (i_v - v2):
                i_v2 = v1
            else:
                i_v2 = v2

            for s in self.waves[-1].wave_sections:
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
                    v_r_1 = utils.rarefaction(p_r_1, CJ_q, 1)
                    p_r_2 = numpy.linspace(self.state_r.p+1e-3*dp_fraction, i_p)
                    v_r_2 = utils.shock(p_r_2, self.state_r, 1)
                    p_r_df = numpy.linspace(CJ_p+1e-6*dp_fraction, p_max+5*dp_fraction)
                    v_r_df = utils.deflagration(p_r_df, CJ_q, 1)

                elif s.type == 'Detonation':
                    plot_burning[3] = True
                    #CJ_v, CJ_p, CJ_rho, CJ_eps = s.q_end.v, s.q_end.p, s.q_end.rho, s.q_end.eps
                    ax.plot(CJ_v, CJ_p, 'ko', label=r"$CJDT$")
                    trans_offset = mtrans.offset_copy(ax.transData, fig=fig, x=-15, y=-6, units='dots')
                    ax.text(CJ_v, CJ_p, r"$CJ$", transform=trans_offset, horizontalalignment='center',
                    verticalalignment='bottom')
                    # cut off shock at CJ point
                    p_r_1 = numpy.linspace(self.state_r.p+1e-3*dp_fraction, p_max+0.2*dp_fraction)
                    v_r_1 = utils.rarefaction(p_r_1, self.state_r, 1)
                    p_r_2 = numpy.linspace(p_min-0.1*dp_fraction, CJ_p-1e-3*dp_fraction)
                    v_r_2 = utils.shock(p_r_2, CJ_q, 1)
                    p_r_dt = numpy.linspace(CJ_p+1e-6*dp_fraction, p_max+5*dp_fraction)
                    v_r_dt = utils.detonation(p_r_dt, CJ_q, 1)

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
