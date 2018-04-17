from r3d2 import Gamma_law, Gamma_law_react, ReactiveRelFactory
from numpy.testing import assert_allclose
from numpy import sqrt
from nose.tools import assert_raises

def test_basic_state():
    """
    Simple gamma law, inert state
    """

    eos = Gamma_law(5.0/3.0)
    f = ReactiveRelFactory()
    U = f.state(1.0, 0.0, 0.0, 1.5, eos, label="Test")
    state = U.state()
    state_correct = [1.0, 0.0, 0.0, 1.5, 1.0, 1.0, 3.5, sqrt(10.0/21.0)]
    assert_allclose(state, state_correct)
    string = r"\begin{pmatrix} \rho \\ v_x \\ v_t \\ \epsilon \end{pmatrix}_{Test}"
    string += r" = \begin{pmatrix} 1.0000 \\ 0.0000 \\ 0.0000 \\ 1.5000 \end{pmatrix}"
    assert U.latex_string() == string
    assert U._repr_latex_() == r"$" + string + r"$"
    assert_allclose(U.wavespeed(0), -sqrt(10.0/21.0))
    assert_allclose(U.wavespeed(1), 0.0)
    assert_allclose(U.wavespeed(2), +sqrt(10.0/21.0))
    assert_raises(NotImplementedError, U.wavespeed, 3)

def test_reactive_state():
    """
    Reactive EOS.
    """
    f = ReactiveRelFactory()
    eos = Gamma_law(5.0/3.0)
    eos_reactive = Gamma_law_react(5.0/3.0, 1.0, 1.0, 1.0, eos)
    U = f.state(1.0, 0.0, 0.0, 1.5, eos_reactive, label="Test")
    string = r"\begin{pmatrix} \rho \\ v_x \\ v_t \\ \epsilon \\ q \end{pmatrix}_{Test}"
    string += r" = \begin{pmatrix} 1.0000 \\ 0.0000 \\ 0.0000 \\ 1.5000 \\ 1.0000 \end{pmatrix}"
    assert U.latex_string() == string
