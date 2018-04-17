# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:26:41 2016

@author: ih3
"""
from abc import ABCMeta, abstractmethod
import numpy

class State(metaclass=ABCMeta):
    """
    Abstract base class for state at a point.
    """

    @abstractmethod
    def __init__(self, *args, label=None):
        r"""
        Force concrete implementation
        """
        pass

    @abstractmethod
    def prim(self):
        r"""
        Return the primitive variables.
        """
        pass

    @abstractmethod
    def state(self):
        r"""
        Return all variables.
        """
        pass

    @abstractmethod
    def wavespeed(self, wavenumber):
        """
        Compute the wavespeed given the wave number (0 for the left wave,
        2 for the right wave).
        """
        pass

    @abstractmethod
    def latex_string(self):
        """
        Helper function to represent the state as a string.
        """
        pass

    def _repr_latex_(self):
        """
        IPython or Jupyter repr.
        """
        s = r"$" + self.latex_string() + r"$"
        return s
