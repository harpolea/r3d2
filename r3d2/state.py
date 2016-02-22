# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:26:41 2016

@author: ih3
"""

import numpy

class State(object):
    """
    A state at a point. Initialized with the rest mass density, velocity, and
    specific internal energy, as well as an equation of state.
    """

    def __init__(self, rho, v, vt, eps, eos, label=None):
        r"""
        Constructor

        Parameters
        ----------

        rho : scalar
            Rest mass density :math:`\rho_0`
        v : scalar
            Velocity component in the normal (:math:`x`) direction :math:`v_x`
        v_t : scalar
            Velocity component tangential to :math:`x` :math:`v_t`
        eps : scalar
            Specific internal energy :math:`\epsilon`
        eos : dictionary
            Equation of State
        label : string
            Label for output purposes.
        """
        self.rho = rho
        self.v = v
        self.vt = vt
        self.eps = eps
        self.eos = eos
        if 'q_available' in self.eos:
            self.q = self.eos['q_available']
        else:
            self.q = None
        self.W_lorentz = 1.0 / numpy.sqrt(1.0 - self.v**2 - self.vt**2)
        self.p = self.eos['p_from_rho_eps'](rho, eps)
        self.h = self.eos['h_from_rho_eps'](rho, eps)
        self.cs = self.eos['cs_from_rho_eps'](rho, eps)
        self.label = label

    def prim(self):
        r"""
        Return the primitive variables :math:`\rho, v_x, v_t, \epsilon`.
        """
        return numpy.array([self.rho, self.v, self.vt, self.eps])

    def state(self):
        r"""
        Return all variables :math:`\rho, v_x, v_t, \epsilon, p, W, h, c_s`.
        """
        return numpy.array([self.rho, self.v, self.vt, self.eps, self.p,\
        self.W_lorentz, self.h, self.cs])

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
            term1 = self.v * (1.0 - self.cs**2)
            term2 = (1.0 - self.v**2 - self.vt**2) * (1.0 - self.v**2 -
            self.vt**2 * self.cs**2)
            term3 = 1.0 - (self.v**2 + self.vt**2) * self.cs**2
            return (term1 + s * self.cs * numpy.sqrt(term2)) / term3
        else:
            raise NotImplementedError("wavenumber must be 0, 1, 2")

    def vt_from_known(self, rho, v, eps):
        r"""
        Computes tangential velocity across a wave.

        Parameters
        ----------

        self : State
            The known state
        rho : scalar
            Rest mass density in the unknown state across the wave
        v : scalar
            Normal velocity :math:`v_x` in the unknown state across the wave
        eps : scalar
            Specific internal energy in the unknown state across the wave

        Returns
        -------

        vt : scalar
            Tangential velocity :math:`v_t` in the unknown state
        """
        h = self.eos['h_from_rho_eps'](rho, eps)
        vt = self.h * self.W_lorentz * self.vt
        vt *= numpy.sqrt((1.0 - v**2)/
            (h**2 + (self.h * self.W_lorentz * self.vt)**2))
        return vt

    def latex_string(self):
        """
        Helper function to represent the state as a string.
        """
        if self.q:
            s = r"\begin{pmatrix} \rho \\ v_x \\ v_t \\ \epsilon \\ q \end{pmatrix}"
        else:
            s = r"\begin{pmatrix} \rho \\ v_x \\ v_t \\ \epsilon \end{pmatrix}"
        if self.label:
            s += r"_{{{}}} ".format(self.label)
        s += "= "
        if self.q:
            s += r"\begin{{pmatrix}} {:.4f} \\ {:.4f} \\ {:.4f} \\ {:.4f} \\ {:.4f} \end{{pmatrix}}".format(\
            self.rho, self.v, self.vt, self.eps, self.q)
        else:
            s += r"\begin{{pmatrix}} {:.4f} \\ {:.4f} \\ {:.4f} \\ {:.4f} \end{{pmatrix}}".format(\
            self.rho, self.v, self.vt, self.eps)
        return s

    def _repr_latex_(self):
        """
        IPython or Jupyter repr.
        """
        s = r"$" + self.latex_string() + r"$"
        return s
