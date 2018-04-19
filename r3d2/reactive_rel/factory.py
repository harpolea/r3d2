from r3d2 import AbstractFactory
from r3d2.reactive_rel import reactive_rel_wave, reactive_rel_riemann_problem, reactive_rel_state

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
