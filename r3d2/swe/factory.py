from r3d2 import AbstractFactory
from r3d2.swe import swe_wave, swe_state, swe_riemann_problem

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
