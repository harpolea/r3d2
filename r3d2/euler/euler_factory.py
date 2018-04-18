from r3d2 import AbstractFactory

from r3d2.euler import *

class EulerFactory(AbstractFactory):
    """
    Create concrete reactive relativistic Riemann problem objects
    """

    def riemann_problem(self, state_l, state_r, t_end=1.0):
        return euler_riemann_problem.EulerRiemannProblem(state_l, state_r, t_end)

    def state(self, rho, v, eps, eos, label=None):
        return euler_state.EulerState(rho, v, eps, eos, label)

    def wave(self, q_known, unknown_value, wavenumber):
        return euler_wave.EulerWave(q_known, unknown_value, wavenumber)

    def wavesection(self, q_start, p_end, wavenumber):
        return euler_wave.EulerWaveSection(q_start, p_end, wavenumber)
