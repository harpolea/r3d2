import eos_defns
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
    assert_allclose([p, h, cs], [p_true, h_true, cs_true], rtol=1e-8)
