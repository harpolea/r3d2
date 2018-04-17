# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:26:41 2016

@author: ih3
"""

import numpy
from r3d2 import State

class SWEState(State):
    """
    A state at a point. Initialized with the rest mass density, velocity, and
    specific internal energy, as well as an equation of state.
    """

    def __init__(self, phi, v, label=None):
        r"""
        Constructor

        Parameters
        ----------

        rho : scalar
            Rest mass density :math:`\rho_0`
        v : scalar
            Velocity component in the normal (:math:`x`) direction :math:`v_x`
        label : string
            Label for output purposes.
        """
        self.phi = phi
        self.v = v
        self.W_lorentz = 1.0 / numpy.sqrt(1.0 - self.v**2)

        self.label = label

    def prim(self):
        r"""
        Return the primitive variables :math:`\Phi, v`.
        """
        return numpy.array([self.phi, self.v])

    def state(self):
        r"""
        Return all variables :math:`\Phi, v, W`.
        """
        return numpy.array([self.phi, self.v, self.W_lorentz])

    def wavespeed(self, wavenumber):
        """
        Compute the wavespeed given the wave number (0 for the left wave,
        2 for the right wave).

        Parameters
        ----------

        wavenumber: scalar
            Wave number ([0,1,2]).
        """
        if wavenumber == 1:
            return self.v
        elif abs(wavenumber - 1) == 1:
            return 0.5 * (-self.v**3 + self.v + (self.v**2 - 1) *
                          numpy.sqrt(self.v**2 + 4 / self.phi))
        else:
            raise NotImplementedError("wavenumber must be 0, 1, 2")

    def latex_string(self):
        """
        Helper function to represent the state as a string.
        """
        s = r"\begin{pmatrix} \Phi \\ v \end{pmatrix}"
        if self.label:
            s += r"_{{{}}} ".format(self.label)
        s += "= "
        s += r"\begin{{pmatrix}} {} \\ {} \end{{pmatrix}}".format(self.phi, self.v)
        return s
