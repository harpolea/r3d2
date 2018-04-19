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
