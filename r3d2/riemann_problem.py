# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:23:51 2016

@author: ih3
"""
from abc import ABCMeta, abstractmethod
import numpy
from matplotlib import pyplot
from IPython.core.pylabtools import print_figure
from scipy.optimize import brentq

class RiemannProblem(metaclass=ABCMeta):
    """
    Abstract base Riemann Problem class.
    """

    def __init__(self, state_l, state_r, t_end=1.0):
        """
        Constructor

        Parameters
        ----------
        state_l : State
            left state
        state_r : State
            right state
        t_end : float, optional
            end time of simulation, default=1
        """

        # Cache for plot
        self._png_data = None
        self._svg_data = None

        self.state_l = state_l
        self.state_r = state_r

        self.t_end = t_end

        pmin = min(self.state_l.p, self.state_r.p)
        pmax = max(self.state_l.p, self.state_r.p)
        while self.find_delta_v(pmin) * self.find_delta_v(pmax) > 0.0:
            pmin /= 2.0
            pmax *= 2.0

        pmin_rootfind = 0.9*pmin
        pmax_rootfind = 1.1*pmax
        try:
            self.find_delta_v(pmin_rootfind)
        except ValueError:
            pmin_rootfind = pmin
        try:
            self.find_delta_v(pmax_rootfind)
        except ValueError:
            pmax_rootfind = pmax

        self.p_star = brentq(self.find_delta_v, pmin_rootfind, pmax_rootfind)

        # initialise
        self.state_star_l = None
        self.state_star_r = None
        self.waves = []

        # use concrete implementation
        self.make_waves()

    @abstractmethod
    def find_delta_v(self, p_star):
        """
        Find change in speed between left and right states given a guess for a variable in the star state.

        This is abstract so requires a concrete implementation.
        """
        pass

    @abstractmethod
    def make_waves(self):
        """
        Create the star states and waves. This is abstract so requires a concrete implementation.
        """
        pass

    @abstractmethod
    def _figure_data(self, fig_format):
        """
        Provides figure data for the _repr_png_ and _repr_svg_ functions. Must have a concrete implementation.

        Parameters
        ----------
        fig_format : string
            Provides format to the print_figure function. 
        """
        pass

    def _repr_png_(self):
        if self._png_data is None:
            self._png_data = self._figure_data('png')
        return self._png_data

    def _repr_svg_(self):
        if self._svg_data is None:
            self._svg_data = self._figure_data('svg')
        return self._svg_data

    @abstractmethod
    def _repr_latex_(self):
        """
        Force concrete implementation of latex representation of object.
        """
        pass
