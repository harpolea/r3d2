# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:23:51 2016

@author: ih3
"""

import numpy
from matplotlib import pyplot
from IPython.core.pylabtools import print_figure
from scipy.optimize import brentq
import matplotlib.transforms as mtrans

from .wave import Wave, WaveSection
from . import utils


class ShallowWaterRiemannProblem(RiemannProblem):
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



    def plot_P_v(self, ax, fig, var_to_plot="velocity"):
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

        pass
