"""
Set up range of initial data for the reactive Riemann problem and see if the
wave pattern changes with tangential velocity.
"""
from r3d2 import eos_defns, State, RiemannProblem
from itertools import combinations
import numpy as np

def check_wave_pattern(U_l, U_r, vt_side, vts=[-0.9,0.5,0.0,-0.5,0.9]):
    """
    Given an initial reactive state and burnt state, will run the
    reactive Riemann problem with the reactive state having a (given) range
    of tangential velocities. Shall print output to screen where the
    resulting wave patterns are different.
    """
    flat_patterns = make_flat_patterns(U_l, U_r, vts, vt_side)

    #print(flat_patterns)

    # generate list of pairs
    pairs = list(combinations(range(len(vts)), 2))

    # check to see if patterns match
    for i, j in pairs:
        if not flat_patterns[i] == flat_patterns[j]:
            print('vt = {}, {} are different'.format(vts[i], vts[j]))
            print(', '.join(flat_patterns[i]))
            print(', '.join(flat_patterns[j]))


def make_flat_patterns(U_l, U_r, vts, vt_side):
    """
    Save some code repetition. Given reactive and burnt states, produces a list of lists of wave patterns for a given list of tangential velocities.
    """
    eos_l = eos_defns.eos_gamma_law(5.0/3.0)
    eos_r = eos_defns.eos_gamma_law_react(5.0/3.0, 0.1, 1.0, 1.0, eos_l)

    wave_patterns = []
    if vt_side == 'l':
        rho_l = U_l.rho
        v_l = U_l.v
        eps_l = U_l.eps
        eos_l = U_l.eos
    else:
        rho_r = U_r.rho
        v_r = U_r.v
        eps_r = U_r.eps
        eos_r = U_r.eos

    for vt in vts:
        #print('vt = {}'.format(vt))
        # first change the vt
        if vt_side == 'l':
            U_l = State(rho_l, v_l, vt, eps_l, eos_l)
            #U_r = State(rho_r, v_r, 0.0, eps_r, eos_r)
        else:
            U_r = State(rho_r, v_r, vt, eps_r, eos_r)
        try:
            rp = RiemannProblem(U_l, U_r)
            wave_patterns.append(rp.waves)
        except:
            break

    # now check if all the wave patterns are the same

    # flatten patterns
    flat_patterns = []
    for i, p in enumerate(wave_patterns):
        flat_patterns.append([])
        for w in p:
            for s in w.wave_sections:
                if not s.trivial:
                    flat_patterns[i].append(s.type)

    return flat_patterns

def find_critical_vt(U_l, U_r, vt_side):
    """
    Pretty sure that only magnitude of vt matters. As the function that
    would be used for root finding is discontinuous, shall just use a very
    basic bisection method. If too coarse a grid of vt values is used in
    the initial pass, then a root may be missed if the wave pattern changes
    to a different pattern then back again.
    """
    vts = np.linspace(0., 0.999, num=100)
    tolerance = 1.e-4

    def bisect(vt0, vtend, tol=tolerance):
        """
        Can be super lazy and only evaluate RiemannProblem for vt0 and vthalf
        as if pattern(vt0) == pattern(vthalf), must be that
        pattern(vthalf) != pattern(vtend) unless something has gone really
        wrong.
        """
        maxIts = 100
        nIts = 0
        while (vtend-vt0) > tol and nIts < maxIts:
            vthalf = 0.5 * (vt0 + vtend)
            flat_patterns = make_flat_patterns(U_l, U_r, [vt0, vthalf], vt_side)

            if len(flat_patterns) > 1 and not flat_patterns[0] == flat_patterns[1]:
                vtend = vthalf
            else:
                vt0 = vthalf

            nIts += 1

        flat_patterns = make_flat_patterns(U_l, U_r, [vt0, vtend], vt_side)

        return 0.5 * (vt0 + vtend), flat_patterns

    # do a first pass to find where pattern changes
    flat_patterns = make_flat_patterns(U_l, U_r, vts, vt_side)
    #print(flat_patterns)

    critical_vts = []
    critical_patterns = []

    for i in range(len(flat_patterns) - 2):
        if not flat_patterns[i] == flat_patterns[i+1]:
            vt, pattern = bisect(vts[i], vts[i+1])
            critical_vts.append(vt)
            critical_patterns.append(pattern)

    if len(critical_vts) == 1:
        print('There is one critical tangential velocity ')
    else:
        print('There are {} critical tangential velocities '.format(len(critical_vts)))
    for i, v in enumerate(critical_vts):
        print('vt: {}, patterns: {} -> {}'.format(v, ', '.join(critical_patterns[i][0]), ', '.join(critical_patterns[i][1])))

    return critical_vts

def vary_rho(U_l, U_r, vt_side, rhos):

    critical_vts = np.zeros_like(rhos)

    if vt_side == 'l':
        v_l = U_l.v
        eps_l = U_l.eps
        eos_l = U_l.eos
    else:
        v_r = U_r.v
        eps_r = U_r.eps
        eos_r = U_r.eos

    for i, rho in enumerate(rhos):

        # first change the rho
        if vt_side == 'l':
            U_l = State(rho, v_l, 0.0, eps_l, eos_l)
            #U_r = State(rho_r, v_r, 0.0, eps_r, eos_r)
        else:
            U_r = State(rho, v_r, 0.0, eps_r, eos_r)

        crit_vt = find_critical_vt(U_l, U_r, vt_side)

        if len(crit_vt) == 0:
            critical_vts[i] = 0.0
        else:
            critical_vts[i] = crit_vt[0] # assume only one of them.
            # where it detects two, it's usually because it happens to be able to resolve it well there.

    return critical_vts

if __name__ == "__main__":
    eos = eos_defns.eos_gamma_law(5.0/3.0)
    eos_reactive = eos_defns.eos_gamma_law_react(5.0/3.0, 0.1, 1.0, 1.0, eos)
    #U_l = State(5.0, 0.0, 0.0, 2.0, eos_reactive)
    # detonation wave
    #U_r = State(8.113665227084942, -0.34940431910454606, 0.0, 2.7730993786742353, eos)
    # cj detonation wave
    #U_burnt = State(5.1558523350586452, -0.031145176327346744, 0.0,
    #                2.0365206985013153, eos)
    # deflagration
    #U_burnt = State(0.10089486779791534, 0.97346270073482888, 0.0,
    #                0.14866950243842186, eos)
    # deflagration with precursor shock
    #U_burnt = State(0.24316548798524526, 0.39922932397353039, 0.0,
    #                0.61686385086179807, eos)

    # FIXME: there is a really weird bug where this breaks if U_r has a
    # non-zero normal velocity and try to give U_l tangential velocity.
    #U_l = State(1.0, 0.0, 0.0, 1.6, eos)
    #U_r = State(0.125, 0.5, 0.0, 1.2, eos_reactive)

    U_l = State(5.0, 0.0, 0.0, 2.0, eos_reactive)
    U_r = State(8.0, -0.2, 0.0, 2.5, eos)

    #check_wave_pattern(U_l, U_r, 'r', vts=[-0.5,-0.1, 0.0, 0.1, 0.5, 0.87])
    #check_wave_pattern(U_l, U_r, 'l', vts=[0.5, 0.55])
    #check_wave_pattern(U_r, U_l, 'l')
    #critical_vts = find_critical_vt(U_l, U_r, 'l')
    #print('vt = {}'.format(critical_vts))

    rhos = np.linspace(0.5, 15.0, 40)
    critical_vts = vary_rho(U_l, U_r, 'l', rhos)
    for vt in critical_vts:
        print('vt = {}'.format(vt))
