# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

class AbstractFactory(metaclass=ABCMeta):
    """
    Interface for operations that create abstract products.
    """

    @abstractmethod
    def riemann_problem(self, *args, **kwargs):
        pass

    @abstractmethod
    def state(self, *args, **kwargs):
        pass

    @abstractmethod
    def wave(self, *args, **kwargs):
        pass

    @abstractmethod
    def wavesection(self, *args, **kwargs):
        pass
