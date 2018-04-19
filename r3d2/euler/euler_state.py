# -*- coding: utf-8 -*-

import numpy
from r3d2 import State

class EulerState(State):
    """
    A state at a point. Initialized with the rest mass density, velocity, and
    specific internal energy, as well as an equation of state.
    """

    def __init__(self, rho, v, eps, eos, label=None):
        r"""
        Constructor

        Parameters
        ----------

        rho : scalar
            Rest mass density :math:`\rho_0`
        v : scalar
            Velocity component in the normal (:math:`x`) direction :math:`v_x`
        eps : scalar
            Specific internal energy :math:`\epsilon`
        eos : dictionary
            Equation of State
        label : string
            Label for output purposes.
        """
        self.rho = rho
        self.v = v
        self.eps = eps
        self.eos = eos
        if hasattr(self.eos, 'q'):
            self.q = self.eos.q
        else:
            self.q = None
        self.p = self.eos.p_from_rho_eps(rho, eps)
        self.h = self.eos.h_from_rho_eps(rho, eps)
        self.cs = self.eos.cs_from_rho_eps(rho, eps)
        self.label = label

    def prim(self):
        r"""
        Return the primitive variables :math:`\rho, v_x, \epsilon`.
        """
        return numpy.array([self.rho, self.v, self.eps])

    def state(self):
        r"""
        Return all variables :math:`\rho, v_x, \epsilon, p, h, c_s`.
        """
        return numpy.array([self.rho, self.v, self.eps, self.p,\
        self.h, self.cs])

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
            s = wavenumber - 1
            # term1 = self.v * (1.0 - self.cs**2)
            # term2 = (1.0 - self.v**2) * (1.0 - self.v**2)
            # term3 = 1.0 - (self.v**2) * self.cs**2
            # return (term1 + s * self.cs * numpy.sqrt(term2)) / term3
            return self.v + s * self.cs
        else:
            raise NotImplementedError("wavenumber must be 0, 1, 2")



    def latex_string(self):
        """
        Helper function to represent the state as a string.
        """
        if self.q:
            s = r"\begin{pmatrix} \rho \\ v_x \\ \epsilon \\ q \end{pmatrix}"
        else:
            s = r"\begin{pmatrix} \rho \\ v_x \\ \epsilon \end{pmatrix}"
        if self.label:
            s += r"_{{{}}} ".format(self.label)
        s += "= "
        if self.q:
            s += r"\begin{{pmatrix}} {:.4f} \\ {:.4f} \\ {:.4f} \\ {:.4f} \end{{pmatrix}}".format(\
            self.rho, self.v, self.eps, self.q)
        else:
            s += r"\begin{{pmatrix}} {:.4f} \\ {:.4f} \\ {:.4f} \end{{pmatrix}}".format(\
            self.rho, self.v, self.eps)
        return s
