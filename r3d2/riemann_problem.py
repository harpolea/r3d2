# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:23:51 2016

@author: ih3
"""

import numpy
from scipy.optimize import brentq

from matplotlib import pyplot
from IPython.core.pylabtools import print_figure

from .wave import Wave


class RiemannProblem(object):
    """
    This is a more general Riemann Problem class.

    Allows for different EOSs on both sides (as required for burning problems).
    Uses the State class.
    """

    def __init__(self, state_l, state_r):
        """
        Constructor
        """

        # Cache for plot
        self._png_data = None
        self._svg_data = None

        self.state_l = state_l
        self.state_r = state_r

        def find_delta_v(p_star_guess):

            wave_l = Wave(self.state_l, p_star_guess, 0)
            wave_r = Wave(self.state_r, p_star_guess, 2)

            return wave_l.q_r.v - wave_r.q_l.v

        pmin = min(self.state_l.p, self.state_r.p)
        pmax = max(self.state_l.p, self.state_r.p)
        while find_delta_v(pmin) * find_delta_v(pmax) > 0.0:
            pmin /= 2.0
            pmax *= 2.0

        pmin_rootfind = 0.9*pmin
        pmax_rootfind = 1.1*pmax
        try:
            find_delta_v(pmin_rootfind)
        except ValueError:
            pmin_rootfind = pmin
        try:
            find_delta_v(pmax_rootfind)
        except ValueError:
            pmax_rootfind = pmax

        self.p_star = brentq(find_delta_v, pmin_rootfind, pmax_rootfind)
        wave_l = Wave(self.state_l, self.p_star, 0)
        wave_r = Wave(self.state_r, self.p_star, 2)
        self.state_star_l = wave_l.q_r
        self.state_star_r = wave_r.q_l
        self.waves = [wave_l,
                      Wave(self.state_star_l, self.state_star_r, 1), wave_r]

    def _figure_data(self, format):
        fig, axs = pyplot.subplots(3,3, figsize=(10,6))
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
        data = print_figure(fig, format)
        pyplot.close(fig)
        return data

    def _repr_png_(self):
        if self._png_data is None:
            self._png_data = self._figure_data('png')
        return self._png_data

    def _repr_svg_(self):
        if self._svg_data is None:
            self._svg_data = self._figure_data('svg')
        return self._svg_data

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
