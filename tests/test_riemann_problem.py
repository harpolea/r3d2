from r3d2 import eos_defns, State, RiemannProblem
from numpy.testing import assert_allclose

def test_standard_sod():
    """
    Relativistic Sod test.
    
    Numbers are taken from the General Matlab code, so accuracy isn't perfect.
    """
    eos = eos_defns.eos_gamma_law(5.0/3.0)
    w_left = State(1.0, 0.0, 0.0, 1.5, eos, label="L")
    w_right = State(0.125, 0.0, 0.0, 1.2, eos, label="R")
    rp = RiemannProblem(w_left, w_right)
    p_star_matlab = 0.308909954203586
    assert_allclose(rp.p_star, p_star_matlab, rtol=1e-6)
    rarefaction_speeds_matlab = [-0.690065559342354, -0.277995552140227]
    assert_allclose(rp.waves[0].wavespeed, rarefaction_speeds_matlab, rtol=1e-6)
    shock_speed_matlab = 0.818591417744604
    assert_allclose(rp.waves[2].wavespeed, shock_speed_matlab, rtol=1e-6)

def test_bench_1():
    """
    Test Bench problem 1.
    
    Take from Marti & Muller's Living Review (section 6.3). See
    http://computastrophys.livingreviews.org/Articles/lrca-2015-3
    
    Uses Matlab code to test, so extreme accuracy given.
    """
    eos = eos_defns.eos_gamma_law(5.0/3.0)
    w_left = State(10.0, 0.0, 0.0, 2.0, eos, label="L")
    w_right = State(1.0, 0.0, 0.0, 1.5e-6, eos, label="R")
    rp = RiemannProblem(w_left, w_right)
    v_shock_ref = 0.828398034190528
    v_contact_ref = 0.714020700932637
    v_raref_ref = [-0.716114874039433, 0.167236293932105]
    p_star_ref = 1.447945155994138
    prim_star_l = [2.639295549545608, 0.714020700932637, 0.0, 0.822915695957254]
    prim_star_r = [5.070775964247521, 0.714020700932636, 0.0, 0.428320586297783]
    assert_allclose(rp.waves[0].wavespeed, v_raref_ref, rtol=1e-8)
    assert_allclose(rp.waves[1].wavespeed, v_contact_ref, rtol=1e-8)
    assert_allclose(rp.waves[2].wavespeed, v_shock_ref, rtol=1e-8)
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
    w_left = State(1.0, 0.0, 0.0, 1500.0, eos, label="L")
    w_right = State(1.0, 0.0, 0.0, 1.5e-2, eos, label="R")
    rp = RiemannProblem(w_left, w_right)
    v_shock_ref = 0.986804253648698
    v_contact_ref = 0.960409611243646
    v_raref_ref = [-0.816333330585011, 0.668125119704241]
    p_star_ref = 18.597078678567343
    prim_star_l = [0.091551789392213, 0.960409611243646, 0.0, 304.6976820774594]
    prim_star_r = [10.415581582734182, 0.960409611243650, 0.0, 2.678258318680287]
    assert_allclose(rp.waves[0].wavespeed, v_raref_ref, rtol=1e-8)
    assert_allclose(rp.waves[1].wavespeed, v_contact_ref, rtol=1e-8)
    assert_allclose(rp.waves[2].wavespeed, v_shock_ref, rtol=1e-8)
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
    w_left = State(1.0, 0.0, 0.0, 1500, eos, label="L")
    w_right = State(1.0, 0.0, 0.99, 0.015, eos, label="R")
    rp = RiemannProblem(w_left, w_right)
    v_shock_ref = 0.927006
    v_contact_ref = 0.766706
    assert_allclose(rp.waves[2].wavespeed, v_shock_ref, rtol=1e-5)
    assert_allclose(rp.waves[1].wavespeed, v_contact_ref, rtol=1e-5)
    
def test_bench_4():
    """
    Test Bench problem 4.
    
    Take from Marti & Muller's Living Review (section 6.3). See
    http://computastrophys.livingreviews.org/Articles/lrca-2015-3
    
    Left and right states have been flipped so it complements the above
    Sod test.
    """
    eos = eos_defns.eos_gamma_law(5.0/3.0)
    w_left = State(1.0, 0.0, 0.9, 0.015, eos, label="L")
    w_right = State(1.0, 0.0, 0.9, 1500, eos, label="R")
    rp = RiemannProblem(w_left, w_right)
    v_shock_ref = -0.445008
    v_contact_ref = -0.319371
    p_star_ref = 0.90379102665871947
    assert_allclose(rp.waves[0].wavespeed, v_shock_ref, rtol=1e-5)
    assert_allclose(rp.waves[1].wavespeed, v_contact_ref, rtol=1e-5)
    assert_allclose(rp.p_star, p_star_ref, rtol=1e-4)

def test_multi_gamma():
    """
    Test using different equations of state either side of the interface.
    
    This is essentially the strong test (Figure 3) of Millmore and Hawke
    """
    eos1 = eos_defns.eos_gamma_law(1.4)
    eos2 = eos_defns.eos_gamma_law(1.67)
    w_left = State(10.2384, 0.9411, 0.0, 50.0/0.4/10.23841, eos1, label="L")
    w_right = State(0.1379, 0.0, 0.0, 1.0/0.1379/0.67, eos2, label="R")
    rp = RiemannProblem(w_left, w_right)
    v_shock_ref = 0.989670551306888
    v_contact_ref = 0.949361020941429
    v_raref_ref = [0.774348025484414, 0.804130593636139]
    p_star_ref = 41.887171487985299
    prim_star_l = [9.022178190552809, 0.949361020941429, 0.0, 11.606723621310707]
    prim_star_r = [1.063740721273106, 0.949361020941430, 0.0, 58.771996925298758]
    assert_allclose(rp.waves[0].wavespeed, v_raref_ref, rtol=1e-6)
    assert_allclose(rp.waves[1].wavespeed, v_contact_ref, rtol=1e-6)
    assert_allclose(rp.waves[2].wavespeed, v_shock_ref, rtol=1e-6)
    assert_allclose(rp.p_star, p_star_ref, rtol=1e-6)
    assert_allclose(rp.state_star_l.prim(), prim_star_l, rtol=1e-6)
    assert_allclose(rp.state_star_r.prim(), prim_star_r, rtol=1e-6)
    
def test_detonation_wave():
    """
    A single detonation wave
    """
    eos = eos_defns.eos_gamma_law(5.0/3.0)
    eos_reactive = eos_defns.eos_gamma_law_react(5.0/3.0, 0.1, 1.0, 1.0, eos)
    U_reactive = State(5.0, 0.0, 0.0, 2.0, eos_reactive)
    U_burnt = State(8.113665227084942, -0.34940431910454606, 0.0, 
                    2.7730993786742353, eos)
    rp = RiemannProblem(U_reactive, U_burnt)
    assert(rp.waves[2].wave_sections[0].trivial)
    assert_allclose(rp.waves[0].wavespeed, -0.82680400067536064)
    
def test_deflagration_wave():
    """
    A single deflagration wave
    """
    eos = eos_defns.eos_gamma_law(5.0/3.0)
    eos_reactive = eos_defns.eos_gamma_law_react(5.0/3.0, 0.1, 1.0, 1.0, eos)
    U_reactive = State(5.0, 0.0, 0.0, 2.0, eos_reactive)
    U_burnt = State(0.10089486779791534, 0.97346270073482888, 0.0, 
                    0.14866950243842186, eos)
    rp = RiemannProblem(U_reactive, U_burnt)
    assert(rp.waves[2].wave_sections[0].trivial)
    wavespeed_deflagration = [-0.60970641412658788, 0.94395720523915128]
    assert_allclose(rp.waves[0].wavespeed, wavespeed_deflagration)
    
def test_trivial():
    """
    A trivial Riemann Problem
    """
    eos = eos_defns.eos_gamma_law(5.0/3.0)
    U = State(1.0, 0.0, 0.0, 1.0, eos)
    rp = RiemannProblem(U, U)
    for wave in rp.waves:
        assert(wave.wave_sections[0].trivial)
        assert(wave.name == "")

