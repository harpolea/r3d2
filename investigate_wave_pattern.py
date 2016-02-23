"""
Set up range of initial data for the reactive Riemann problem and see if the
wave pattern changes with tangential velocity.
"""
from r3d2 import eos_defns, State, RiemannProblem

def check_wave_pattern(U_reactive, U_burnt):
    """
    """
    wave_patterns = []
    vts = [-0.9, 0.0, 0.9]
    for vt in vts:
    # first change the vt
        U_reactive.vt = vt
        rp = RiemannProblem(U_reactive, U_burnt)
        wave_patterns.append(rp.waves)

    # now check if all the wave patterns are the same
    p1, p2, p3 = wave_patterns

    for i in range(len(p1)):
        w1 = p1[i].wave_sections
        w2 = p2[i].wave_sections
        w3 = p3[i].wave_sections
        if len(w1) != len(w2):
            print('vt = {}, {} are different'.format(vts[0], vts[1]))
        elif len(w1) != len(w3):
            print('vt = {}, {} are different'.format(vts[0], vts[2]))
        elif len(w3) != len(w2):
            print('vt = {}, {} are different'.format(vts[1], vts[2]))
        else:
            for i in range(len(w1)):
                if not w1[i].type == w2[i].type:
                    print('vt = {}, {} are different'.format(vts[0], vts[1]))
                if not w1[i].type == w3[i].type:
                    print('vt = {}, {} are different'.format(vts[0], vts[2]))
                if not w2[i].type == w3[i].type:
                    print('vt = {}, {} are different'.format(vts[1], vts[2]))


if __name__ == "__main__":
    eos = eos_defns.eos_gamma_law(5.0/3.0)
    eos_reactive = eos_defns.eos_gamma_law_react(5.0/3.0, 0.1, 1.0, 1.0, eos)
    U_reactive = State(5.0, 0.0, 0.0, 2.0, eos_reactive)
    U_burnt = State(8.113665227084942, -0.34940431910454606, 0.0,
                    2.7730993786742353, eos)

    check_wave_pattern(U_reactive, U_burnt)
