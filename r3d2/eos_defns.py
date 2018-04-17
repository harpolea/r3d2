"""
Equations of state.
"""

import sys
from abc import ABCMeta#, abstractmethod
import numpy

class EOS(metaclass=ABCMeta):
    """
    Abstract base class for EOS
    """
    _fields = []

    def __init__(self, *args):
        """
        Initialise by setting all the arguments to be the variables
        listed in the _fields list
        """
        if len(args) > len(self._fields):
            raise TypeError(r'Expected {} arguments'.format(len(self._fields)))

        for name, value in zip(self._fields, args):
            setattr(self, name, value)

    def p_from_rho_eps(self, rho, eps):
        self.implementation_error()

    def h_from_rho_eps(self, rho, eps):
        self.implementation_error()

    def cs_from_rho_eps(self, rho, eps):
        self.implementation_error()

    def h_from_rho_p(self, rho, p):
        self.implementation_error()

    def rho_from_p_eps(self, p, eps):
        self.implementation_error()

    def t_from_rho_eps(self, rho, eps):
        self.implementation_error()

    def implementation_error(self):
        raise NotImplementedError(sys._getframe(1).f_code.co_name + " not implemented for " + self.__class__.__name__)

class Gamma_law(EOS):
    """
    Gamma law EOS

    Parameters
    ----------
    gamma : float
        adiabatic index
    """
    _fields = ['gamma']

    def p_from_rho_eps(self, rho, eps):
        return (self.gamma - 1.0) * rho * eps

    def h_from_rho_eps(self, rho, eps):
        return 1.0 + self.gamma * eps

    def cs_from_rho_eps(self, rho, eps):
        return numpy.sqrt(self.gamma * self.p_from_rho_eps(rho, eps) / (rho * self.h_from_rho_eps(rho, eps)))

    def h_from_rho_p(self, rho, p):
        return 1.0 + self.gamma / (self.gamma - 1.0) * p / rho

    def rho_from_p_eps(self, p, eps):
        return p / ((self.gamma - 1.0) * eps)

class Gamma_law_react(EOS):
    """
    Reactive gamma law EOS. Describes unreacted material.

    Parameters
    ----------
    gamma : float
        adiabatic index
    q : float
        specific heat release of reaction
    Cv : float
        heat capacity at constant volume
    t_i : float
        ignition temperature
    eos_inert : EOS
        equation of state of reacted material
    """
    _fields = ['gamma', 'q', 'Cv', 't_i', 'eos_inert']

    def p_from_rho_eps(self, rho, eps):
        return (self.gamma - 1.0) * rho * (eps - self.q)

    def h_from_rho_eps(self, rho, eps):
        return 1.0 + self.gamma * eps + (1.0 - self.gamma) * self.q

    def cs_from_rho_eps(self, rho, eps):
        return numpy.sqrt(self.gamma * (self.gamma - 1.0) * (eps - self.q) / \
        (1.0 + self.gamma * eps + (1.0 - self.gamma) * self.q))

    def h_from_rho_p(self, rho, p):
        return 1.0 + self.gamma / (self.gamma - 1.0) * p / rho + self.q

    def t_from_rho_eps(self, rho, eps):
        return (eps - self.q) / self.Cv

    # done for backwards compatibility
    def t_i_from_rho_eps(self, rho, eps):
        if self.t_i is None:
            return 2.5 * rho**(2./3.) / (eps-self.q)**(1./3.)
        else:
            return self.t_i

class Polytrope_law(EOS):
    """
    Polytropic EOS

    Parameters
    ----------
    gamma : float array
        adiabatic indices of cold and hot material
    gamma_th : float
        adiabatic index of ?
    rho_transition : float
        critical density at which material transitions
    k : float
        polytropic constant
    """
    _fields = ['gamma', 'gamma_th', 'rho_transition', 'k']

    def p_from_rho_eps(self, rho, eps):
        if rho < self.rho_transition:
            p_cold = self.k[0] * rho**self.gamma[0]
            eps_cold = p_cold / rho / (self.gamma[0] - 1.)
        else:
            p_cold = self.k[1] * rho**self.gamma[1]
            eps_cold = p_cold / rho / (self.gamma[1] - 1.) - \
                self.k[1] * self.rho_transition**(self.gamma[1] - 1.) + \
                self.k[0] * self.rho_transition**(self.gamma[0] - 1.)

        p_th = max(0.0, (self.gamma_th - 1.0) * rho * (eps - eps_cold))

        return p_cold + p_th

    def h_from_rho_eps(self, rho, eps):
        if rho < self.rho_transition:
            p_cold = self.k[0] * rho**self.gamma[0]
            eps_cold = p_cold / rho / (self.gamma[0] - 1.0)
        else:
            p_cold = self.k[1] * rho**self.gamma[1]
            eps_cold = p_cold / rho / (self.gamma[1] - 1.0) - \
                self.k[1] * self.rho_transition**(self.gamma[1] - 1.0) + \
                self.k[0] * self.rho_transition**(self.gamma[0] - 1.0)

        p_th = max(0., (self.gamma_th - 1.) * rho * (eps - eps_cold))

        return 1. + eps_cold + eps + (p_cold + p_th)/ rho

    def cs_from_rho_eps(self, rho, eps):
        return numpy.sqrt(self.gamma[0] * self.p_from_rho_eps(rho, eps) / (rho * self.h_from_rho_eps(rho, eps)))

    # TODO: fix
    def h_from_rho_p(self, rho, p):
        if rho < self.rho_transition:
            p_cold = self.k[0] * rho**self.gamma[0]
            eps_cold = p_cold / rho / (self.gamma[0] - 1.0)
        else:
            p_cold = self.k[1] * rho**self.gamma[1]
            eps_cold = p_cold / rho / (self.gamma[1] - 1.0) - \
                self.k[1] * self.rho_transition**(self.gamma[1] - 1.0) + \
                self.k[0] * self.rho_transition**(self.gamma[0] - 1.0)

        p_th = max(0.0, p - p_cold)
        eps = p_th / (self.gamma_th - 1.0) / rho + eps_cold

        return 1.0 + eps_cold + eps + p / rho
