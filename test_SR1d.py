import eos_defns
import SR1d
from numpy.testing import assert_allclose

def test_standard_sod():
    """
    Relativistic Sod test.
    
    Numbers are taken from the General Matlab code, so accuracy isn't perfect.
    """
    eos = eos_defns.eos_gamma_law(5.0/3.0)
    w_left = SR1d.State(1.0, 0.0, 1.5, eos, label="L")
    w_right = SR1d.State(0.125, 0.0, 1.2, eos, label="R")
    rp = SR1d.RP(w_left, w_right)
    p_star_matlab = 0.308909954203586
    assert_allclose(rp.p_star, p_star_matlab, rtol=1e-6)
    rarefaction_speeds_matlab = [-0.690065559342354, -0.277995552140227]
    assert_allclose(rp.waves[0].wave_speed, rarefaction_speeds_matlab, rtol=1e-6)
    shock_speed_matlab = 0.818591417744604
    assert_allclose(rp.waves[2].wave_speed, shock_speed_matlab, rtol=1e-6)
