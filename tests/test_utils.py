from r3d2 import eos_defns, State, utils
from numpy.testing import assert_allclose

def test_find_left_state():
    """
    For setting up single shock initial data.
    """

    gamma = 5./3.
    K = 1.
    eos = eos_defns.eos_gamma_law(gamma)

    # right state
    rho_r, v_r, vt_r = (0.001, 0.0, 0.0)
    eps_r = K * rho_r**(gamma - 1.) / (gamma - 1.)

    # initialise right state
    q_r = State(rho_r, v_r, vt_r, eps_r, eos, label="R")

    q_l, v_s = utils.find_left(q_r, M=20.)

    q_l_expected = [0.0062076725292866553,  0.85384931817355381,  0.0,  0.96757276531735015]
    v_s_expected = 0.93199842492399299

    assert_allclose(q_l.prim(), q_l_expected)
    assert_allclose(v_s, v_s_expected)

def test_find_left_state_subsonic():
    """
    For setting up single shock initial data; subsonic case.
    """

    gamma = 5./3.
    K = 1.
    eos = eos_defns.eos_gamma_law(gamma)

    # right state
    rho_r, v_r, vt_r = (0.001, 0.0, 0.0)
    eps_r = K * rho_r**(gamma - 1.) / (gamma - 1.)

    # initialise right state
    q_r = State(rho_r, v_r, vt_r, eps_r, eos, label="R")

    q_l, v_s = utils.find_left(q_r, M=0.5)

    q_l_expected = [0.00066884219753554435, -0.048031759566342737,  0.0,  0.011472032502081871]
    v_s_expected = 0.064150029919561105

    assert_allclose(q_l.prim(), q_l_expected)
    assert_allclose(v_s, v_s_expected)
