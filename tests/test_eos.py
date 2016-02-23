from r3d2 import eos_defns
import numpy
from numpy.testing import assert_allclose

def test_eos_gamma_law():
    eos = eos_defns.eos_gamma_law(5.0/3.0)
    rho = 1.0
    eps = 1.0
    p_true = 2.0 / 3.0
    h_true = 8.0 / 3.0
    cs_true = numpy.sqrt(5.0 / 12.0)
    p = eos['p_from_rho_eps'](rho, eps)
    h = eos['h_from_rho_eps'](rho, eps)
    cs = eos['cs_from_rho_eps'](rho, eps)
    assert_allclose([p, h, cs], [p_true, h_true, cs_true], rtol=1.e-8)

def test_eos_gamma_law_react():
    eos_inert = eos_defns.eos_gamma_law(5.0/3.0)
    eos = eos_defns.eos_gamma_law_react(5.0/3.0, 0.5, 1.0, 10.0, eos_inert)
    rho = 1.0
    eps = 1.0
    p_true = 1.0 / 3.0
    h_true = 7.0 / 3.0
    cs_true = numpy.sqrt(5.0 / 21.0)
    t_true = 0.5
    p = eos['p_from_rho_eps'](rho, eps)
    h = eos['h_from_rho_eps'](rho, eps)
    cs = eos['cs_from_rho_eps'](rho, eps)
    t = eos['t_from_rho_eps'](rho, eps)
    assert_allclose([p, h, cs, t], [p_true, h_true, cs_true, t_true], rtol=1.e-8)

def test_eos_polytrope_law():
    eos = eos_defns.eos_polytrope_law([5.0/3.0, 5.0/3.0], 7.0/5.0, 1.5, [1.0, 1.0])
    rho = 1.0
    eps = 2.0
    p_true = 1.2
    h_true = 5.7
    cs_true = numpy.sqrt(20.0/57.0)
    p = eos['p_from_rho_eps'](rho, eps)
    h = eos['h_from_rho_eps'](rho, eps)
    cs = eos['cs_from_rho_eps'](rho, eps)
    assert_allclose([p, h, cs], [p_true, h_true, cs_true], rtol=1.e-8)
    h_from_rho_p = eos['h_from_rho_p'](rho, p)
    # TODO: fix this
    assert_allclose(h, h_from_rho_p, rtol=1.e-8)

def test_eos_polytrope_law_cold():
    """
    Tests polytrope law for a system with rho > rho_transition
    """
    eos = eos_defns.eos_polytrope_law([5.0/3.0, 7.0/5.0], 7.0/5.0, 0.5, [1.0, 1.0])
    rho = 1.0
    eps = 2.0
    p_true = 1.0
    h_true = 6.372102242
    p = eos['p_from_rho_eps'](rho, eps)
    h = eos['h_from_rho_eps'](rho, eps)
    assert_allclose([p, h], [p_true, h_true], rtol=1.e-8)
    h_from_rho_p = eos['h_from_rho_p'](rho, p)
    # TODO: fix this
    #assert_allclose(h, h_from_rho_p, rtol=1.e-8)
