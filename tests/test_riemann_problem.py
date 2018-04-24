from r3d2 import ReactiveRelFactory, SWEFactory, EulerFactory
from r3d2.reactive_rel import Gamma_law, Gamma_law_react
from r3d2 import euler
from numpy.testing import assert_allclose

def test_standard_sod():
    """
    Relativistic Sod test.

    Numbers are taken from the General Matlab code, so accuracy isn't perfect.
    """
    eos = Gamma_law(5.0/3.0)
    f = ReactiveRelFactory()
    w_left = f.state(1.0, 0.0, 0.0, 1.5, eos, label="L")
    w_right = f.state(0.125, 0.0, 0.0, 1.2, eos, label="R")
    rp = f.riemann_problem(w_left, w_right)
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
    eos = Gamma_law(5.0/3.0)
    f = ReactiveRelFactory()
    w_left = f.state(10.0, 0.0, 0.0, 2.0, eos, label="L")
    w_right = f.state(1.0, 0.0, 0.0, 1.5e-6, eos, label="R")
    rp = f.riemann_problem(w_left, w_right)
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
    eos = Gamma_law(5.0/3.0)
    f = ReactiveRelFactory()
    w_left = f.state(1.0, 0.0, 0.0, 1500.0, eos, label="L")
    w_right = f.state(1.0, 0.0, 0.0, 1.5e-2, eos, label="R")
    rp = f.riemann_problem(w_left, w_right)
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
    eos = Gamma_law(5.0/3.0)
    f = ReactiveRelFactory()
    w_left = f.state(1.0, 0.0, 0.0, 1500, eos, label="L")
    w_right = f.state(1.0, 0.0, 0.99, 0.015, eos, label="R")
    rp = f.riemann_problem(w_left, w_right)
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
    eos = Gamma_law(5.0/3.0)
    f = ReactiveRelFactory()
    w_left = f.state(1.0, 0.0, 0.9, 0.015, eos, label="L")
    w_right = f.state(1.0, 0.0, 0.9, 1500, eos, label="R")
    rp = f.riemann_problem(w_left, w_right)
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
    eos1 = Gamma_law(1.4)
    eos2 = Gamma_law(1.67)
    f = ReactiveRelFactory()
    w_left = f.state(10.2384, 0.9411, 0.0, 50.0/0.4/10.23841, eos1, label="L")
    w_right = f.state(0.1379, 0.0, 0.0, 1.0/0.1379/0.67, eos2, label="R")
    rp = f.riemann_problem(w_left, w_right)
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
    eos = Gamma_law(5.0/3.0)
    eos_reactive = Gamma_law_react(5.0/3.0, 0.1, 1.0, 1.0, eos)
    f = ReactiveRelFactory()
    U_reactive = f.state(5.0, 0.0, 0.0, 2.0, eos_reactive)
    U_burnt = f.state(8.113665227084942, -0.34940431910454606, 0.0,
                    2.7730993786742353, eos)
    rp = f.riemann_problem(U_reactive, U_burnt)
    assert(rp.waves[2].wave_sections[0].trivial)
    assert_allclose(rp.waves[0].wavespeed, -0.82680400067536064)

def test_cj_detonation_wave():
    """
    A single CJ detonation wave
    """
    eos = Gamma_law(5.0/3.0)
    eos_reactive = Gamma_law_react(5.0/3.0, 0.1, 1.0, 1.0, eos)
    f = ReactiveRelFactory()
    U_reactive = f.state(5.0, 0.0, 0.0, 2.0, eos_reactive)
    U_burnt = f.state(5.1558523350586452, -0.031145176327346744, 0.0,
                    2.0365206985013153, eos)
    rp = f.riemann_problem(U_reactive, U_burnt)
    assert(rp.waves[0].wave_sections[0].name == r"{\cal CJDT}_{\leftarrow}")
    assert(rp.waves[0].wave_sections[1].name == r"{\cal R}_{\leftarrow}")
    assert(rp.waves[2].wave_sections[0].trivial)
    wavespeed_cj_detonation = [-0.79738216287617047, -0.73237792243759536]
    assert_allclose(rp.waves[0].wavespeed, wavespeed_cj_detonation)

def test_deflagration_wave():
    """
    A single deflagration wave
    """
    eos = Gamma_law(5.0/3.0)
    eos_reactive = Gamma_law_react(5.0/3.0, 0.1, 1.0, 1.0, eos)
    f = ReactiveRelFactory()
    U_reactive = f.state(5.0, 0.0, 0.0, 2.0, eos_reactive)
    U_burnt = f.state(0.10089486779791534, 0.97346270073482888, 0.0,
                    0.14866950243842186, eos)
    rp = f.riemann_problem(U_reactive, U_burnt)
    assert(rp.waves[2].wave_sections[0].trivial)
    wavespeed_deflagration = [-0.60970641412658788, 0.94395720523915128]
    assert_allclose(rp.waves[0].wavespeed, wavespeed_deflagration)

def test_precursor_deflagration_wave():
    """
    A single deflagration wave with precursor shock
    """
    eos = Gamma_law(5.0/3.0)
    eos_reactive = Gamma_law_react(5.0/3.0, 0.1, 1.0, 1.0, eos)
    f = ReactiveRelFactory()
    U_reactive = f.state(0.5, 0.0, 0.0, 1.0, eos_reactive)
    U_burnt = f.state(0.24316548798524526, 0.39922932397353039, 0.0,
                    0.61686385086179807, eos)
    rp = f.riemann_problem(U_reactive, U_burnt)
    assert(rp.waves[0].wave_sections[0].name == r"{\cal S}_{\leftarrow}")
    assert(rp.waves[0].wave_sections[1].name == r"{\cal CJDF}_{\leftarrow}")
    assert(rp.waves[0].wave_sections[2].name == r"{\cal R}_{\leftarrow}")
    assert(rp.waves[2].wave_sections[0].trivial)
    wavespeed_deflagration = [-0.65807776007359042, -0.23714630045322399]
    assert_allclose(rp.waves[0].wavespeed, wavespeed_deflagration)

def test_swe():
    """
    Test shallow water problem.
    """
    f = SWEFactory()
    Ul = f.state(0.41, 0)
    Ur = f.state(0.01, 0)
    rp = f.riemann_problem(Ul, Ur, t_end=0.3)
    assert(rp.waves[0].wave_sections[0].name == r"{\cal R}_{\leftarrow}")
    assert(rp.waves[1].wave_sections[0].trivial)
    assert(rp.waves[2].wave_sections[0].name == r"{\cal S}_{\rightarrow}")
    wavespeeds = [-0.6403124237432849, 0.5943695015574617]
    assert_allclose([rp.waves[0].wavespeed[0], rp.waves[2].wavespeed[0]], wavespeeds)

def test_swe_reverse():
    """
    Test shallow water problem in reverse direction.
    """
    f = SWEFactory()
    Ul = f.state(0.01, 0)
    Ur = f.state(0.41, 0)
    rp = f.riemann_problem(Ul, Ur, t_end=0.3)
    assert(rp.waves[0].wave_sections[0].name == r"{\cal S}_{\leftarrow}")
    assert(rp.waves[1].wave_sections[0].trivial)
    assert(rp.waves[2].wave_sections[0].name == r"{\cal R}_{\rightarrow}")
    wavespeeds = [-0.5943695015574617, 0.6403124237432849]
    assert_allclose([rp.waves[0].wavespeed[0], rp.waves[2].wavespeed[1]], wavespeeds)

def test_swe_extreme():
    """
    Test shallow water problem.
    """
    f = SWEFactory()
    Ul = f.state(0.9999, 0)
    Ur = f.state(0.00001, 0)
    rp = f.riemann_problem(Ul, Ur, t_end=0.4)
    assert(rp.waves[0].wave_sections[0].name == r"{\cal R}_{\leftarrow}")
    assert(rp.waves[1].wave_sections[0].trivial)
    assert(rp.waves[2].wave_sections[0].name == r"{\cal S}_{\rightarrow}")
    wavespeeds = [-0.9999499987499375, 0.9190022758576769]
    assert_allclose([rp.waves[0].wavespeed[0], rp.waves[2].wavespeed[0]], wavespeeds)


def test_trivial():
    """
    A trivial Riemann Problem
    """
    eos = Gamma_law(5.0/3.0)
    f = ReactiveRelFactory()
    U = f.state(1.0, 0.0, 0.0, 1.0, eos)
    rp = f.riemann_problem(U, U)
    for wave in rp.waves:
        assert(wave.wave_sections[0].trivial)
        assert(wave.name == "")

def test_trivial_swe():
    """
    A trivial SWE Riemann Problem
    """
    f = SWEFactory()
    U = f.state(1.0, 0.0)
    rp = f.riemann_problem(U, U)
    for wave in rp.waves:
        assert(wave.wave_sections[0].trivial)
        assert(wave.name == "")

def test_trivial_euler():
    """
    A trivial Newtonian Riemann Problem
    """
    eos = euler.Gamma_law(5.0/3.0)
    f = EulerFactory()
    U = f.state(1.0, 0.0, 1.0, eos)
    rp = f.riemann_problem(U, U)
    for wave in rp.waves:
        assert(wave.wave_sections[0].trivial)
        assert(wave.name == "")
