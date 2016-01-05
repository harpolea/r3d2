import eos_defns
import SR1d
from numpy.testing import assert_allclose

def test_standard_sod():
    """
    Relativistic Sod test.
    
    Numbers are taken from the General Matlab code, so accuracy isn't perfect.
    """
    eos = eos_defns.eos_gamma_law(5.0/3.0)
    w_left = SR1d.State(1.0, 0.0, 0.0, 1.5, eos, label="L")
    w_right = SR1d.State(0.125, 0.0, 0.0, 1.2, eos, label="R")
    rp = SR1d.RP(w_left, w_right)
    p_star_matlab = 0.308909954203586
    assert_allclose(rp.p_star, p_star_matlab, rtol=1e-6)
    rarefaction_speeds_matlab = [-0.690065559342354, -0.277995552140227]
    assert_allclose(rp.waves[0].wave_speed, rarefaction_speeds_matlab, rtol=1e-6)
    shock_speed_matlab = 0.818591417744604
    assert_allclose(rp.waves[2].wave_speed, shock_speed_matlab, rtol=1e-6)

def test_bench_1():
    """
    Test Bench problem 1.
    
    Take from Marti & Muller's Living Review (section 6.3). See
    http://computastrophys.livingreviews.org/Articles/lrca-2015-3
    
    Uses Matlab code to test, so extreme accuracy given.
    """
    eos = eos_defns.eos_gamma_law(5.0/3.0)
    w_left = SR1d.State(10.0, 0.0, 0.0, 2.0, eos, label="L")
    w_right = SR1d.State(1.0, 0.0, 0.0, 1.5e-6, eos, label="R")
    rp = SR1d.RP(w_left, w_right)
    v_shock_ref = 0.828398034190528
    v_contact_ref = 0.714020700932637
    v_raref_ref = [-0.716114874039433, 0.167236293932105]
    p_star_ref = 1.447945155994138
    prim_star_l = [2.639295549545608, 0.714020700932637, 0.0, 0.822915695957254]
    prim_star_r = [5.070775964247521, 0.714020700932636, 0.0, 0.428320586297783]
    assert_allclose(rp.waves[0].wave_speed, v_raref_ref, rtol=1e-8)
    assert_allclose(rp.waves[1].wave_speed, v_contact_ref, rtol=1e-8)
    assert_allclose(rp.waves[2].wave_speed, v_shock_ref, rtol=1e-8)
    assert_allclose(rp.p_star, p_star_ref, rtol=1e-8)
    assert_allclose(rp.state_star_l.prim(), prim_star_l, rtol=1e-8)
    assert_allclose(rp.state_star_r.prim(), prim_star_r, rtol=1e-8)
    
def test_bench_2():
    """
    Test Bench problem 2.
    
    Take from Marti & Muller's Living Review (section 6.3). See
    http://computastrophys.livingreviews.org/Articles/lrca-2015-3
    
    Uses Matlab code to test, so extreme accuracy given.
    """
    eos = eos_defns.eos_gamma_law(5.0/3.0)
    w_left = SR1d.State(1.0, 0.0, 0.0, 1500.0, eos, label="L")
    w_right = SR1d.State(1.0, 0.0, 0.0, 1.5e-2, eos, label="R")
    rp = SR1d.RP(w_left, w_right)
    v_shock_ref = 0.986804253648698
    v_contact_ref = 0.960409611243646
    v_raref_ref = [-0.816333330585011, 0.668125119704241]
    p_star_ref = 18.597078678567343
    prim_star_l = [0.091551789392213, 0.960409611243646, 0.0, 304.6976820774594]
    prim_star_r = [10.415581582734182, 0.960409611243650, 0.0, 2.678258318680287]
    assert_allclose(rp.waves[0].wave_speed, v_raref_ref, rtol=1e-8)
    assert_allclose(rp.waves[1].wave_speed, v_contact_ref, rtol=1e-8)
    assert_allclose(rp.waves[2].wave_speed, v_shock_ref, rtol=1e-8)
    assert_allclose(rp.p_star, p_star_ref, rtol=1e-8)
    assert_allclose(rp.state_star_l.prim(), prim_star_l, rtol=1e-8)
    assert_allclose(rp.state_star_r.prim(), prim_star_r, rtol=1e-8)
    

def test_bench_3():
    """
    Test Bench problem 3.
    
    Take from Marti & Muller's Living Review (section 6.3). See
    http://computastrophys.livingreviews.org/Articles/lrca-2015-3
    
    """
    eos = eos_defns.eos_gamma_law(5.0/3.0)
    w_left = SR1d.State(1.0, 0.0, 0.0, 1500, eos, label="L")
    w_right = SR1d.State(1.0, 0.0, 0.99, 0.015, eos, label="R")
    rp = SR1d.RP(w_left, w_right)
    v_shock_ref = 0.927006
    v_contact_ref = 0.766706
    assert_allclose(rp.waves[2].wave_speed, v_shock_ref, rtol=1e-5)
    assert_allclose(rp.waves[1].wave_speed, v_contact_ref, rtol=1e-5)
    
def test_bench_4():
    """
    Test Bench problem 4.
    
    Take from Marti & Muller's Living Review (section 6.3). See
    http://computastrophys.livingreviews.org/Articles/lrca-2015-3
    
    Left and right states have been flipped so it complements the above
    Sod test.
    """
    eos = eos_defns.eos_gamma_law(5.0/3.0)
    w_left = SR1d.State(1.0, 0.0, 0.9, 0.015, eos, label="L")
    w_right = SR1d.State(1.0, 0.0, 0.9, 1500, eos, label="R")
    rp = SR1d.RP(w_left, w_right)
    v_shock_ref = -0.445008
    v_contact_ref = -0.319371
    p_star_ref = 0.90379102665871947
    assert_allclose(rp.waves[0].wave_speed, v_shock_ref, rtol=1e-5)
    assert_allclose(rp.waves[1].wave_speed, v_contact_ref, rtol=1e-5)
    assert_allclose(rp.p_star, p_star_ref, rtol=1e-4)
    