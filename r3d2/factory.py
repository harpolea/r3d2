# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from r3d2.reactive_rel import reactive_rel_wave, reactive_rel_riemann_problem, reactive_rel_state
from r3d2.swe import swe_wave, swe_state, swe_riemann_problem


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

class ReactiveRelFactory(AbstractFactory):
    """
    Create concrete reactive relativistic Riemann problem objects
    """

    def riemann_problem(self, state_l, state_r, t_end=1.0):
        return reactive_rel_riemann_problem.ReactiveRelRiemannProblem(state_l, state_r, t_end)

    def state(self, rho, v, vt, eps, eos, label=None):
        return reactive_rel_state.ReactiveRelState(rho, v, vt, eps, eos, label)

    def wave(self, q_known, unknown_value, wavenumber):
        return reactive_rel_wave.ReactiveRelWave(q_known, unknown_value, wavenumber)

    def wavesection(self, q_start, p_end, wavenumber):
        return reactive_rel_wave.ReactiveRelWaveSection(q_start, p_end, wavenumber)




class SWEFactory(AbstractFactory):
    """
    Create concrete shallow water Riemann problem objects
    """

    def riemann_problem(self, state_l, state_r, t_end=1.0):
        return swe_riemann_problem.SWERiemannProblem(state_l, state_r, t_end)

    def state(self, phi, v, label=None):
        return swe_state.SWEState(phi, v, label)

    def wave(self, q_known, unknown_value, wavenumber):
        return swe_wave.SWEWave(q_known, unknown_value, wavenumber)

    def wavesection(self, q_start, p_end, wavenumber):
        return swe_wave.SWEWaveSection(q_start, p_end, wavenumber)
