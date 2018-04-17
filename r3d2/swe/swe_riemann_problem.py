# -*- coding: utf-8 -*-

import numpy
from matplotlib import pyplot
from IPython.core.pylabtools import print_figure
from scipy.optimize import brentq, root

from r3d2 import RiemannProblem
from r3d2.swe.swe_wave import SWEWave, SWEShock, SWERarefaction


class SWERiemannProblem(RiemannProblem):
    """
    This is a more general Riemann Problem class.

    Allows for different EOSs on both sides.
    Uses the SWEState class.
    """

    def __init__(self, state_l, state_r):
        # Cache for plot
        self._png_data = None
        self._svg_data = None

        self.state_l = state_l
        self.state_r = state_r

        phimin = min(self.state_l.phi, self.state_r.phi)
        phimax = max(self.state_l.phi, self.state_r.phi)
        while self.find_delta_v(phimin) * self.find_delta_v(phimax) > 0.0:
            phimin /= 2.0
            phimax *= 2.0

        phimin_rootfind = 0.9*phimin
        phimax_rootfind = 1.1*phimax
        try:
            self.find_delta_v(phimin_rootfind)
        except ValueError:
            phimin_rootfind = phimin
        try:
            self.find_delta_v(phimax_rootfind)
        except ValueError:
            phimax_rootfind = phimax

        self.phi_star = brentq(self.find_delta_v, phimin_rootfind, phimax_rootfind)
        self.make_waves()

    def find_delta_v(self, phi_star):
        v_star_raref = SWERarefaction.rarefaction_solve(self.state_l, phi_star)[-1]
        V_s, v_star_shock = SWEShock.analytic_shock(self.state_r, phi_star)

        return v_star_raref - v_star_shock

    def make_waves(self):
        wave_l = SWEWave(self.state_l, self.phi_star, 0)
        wave_r = SWEWave(self.state_r, self.phi_star, 2)
        self.state_star_l = wave_l.q_r
        self.state_star_r = wave_r.q_l
        self.waves = [wave_l,
                      SWEWave(self.state_star_l, self.state_star_r, 1), wave_r]


    def _repr_latex_(self):
        s = r"$\begin{cases} "
        s += self.state_l.latex_string()
        s += r",\\ "
        s += self.state_r.latex_string()
        s += r", \end{cases} \quad \implies \quad "
        for wave in self.waves:
            s+= wave.name
        s += r", \quad \Phi_* = {:.4f}, \quad".format(self.phi_star)
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

    def _figure_data(self, fig_format):
        fig, axs = pyplot.subplots(1,3)
        ax = axs[0]
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
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$t$")
        ax.set_title("Characteristics")
        names = [r"$\Phi$", r"$v$"]
        xi = [-5.05]
        data = self.state_l.state()
        for wave in self.waves:
            xi_wave, data_wave = wave.plotting_data()
            xi = numpy.hstack((xi, xi_wave))
            data = numpy.vstack((data, data_wave))
        xi = numpy.hstack((xi, [5.05]))
        data = numpy.vstack((data, self.state_r.state()))
        for ax_i in range(3):
            if ax_i == 0:
                continue
            nvar = ax_i - 1
            axs[ax_i].plot(xi, data[:, nvar])
            var_max = numpy.max(data[:, nvar])
            var_min = numpy.min(data[:, nvar])
            d_var = max(var_max - var_min,
                        0.01 * min(abs(var_min), abs(var_max)), 0.01)
            axs[ax_i].set_xlim(xi[0], xi[-1])
            axs[ax_i].set_ylim(var_min - 0.05 * d_var,
                                    var_max + 0.05 * d_var)
            axs[ax_i].set_xlabel(r"$\xi$")
            axs[ax_i].set_ylabel(names[nvar])
        fig.tight_layout()
        data = print_figure(fig, fig_format)
        pyplot.close(fig)
        return data
