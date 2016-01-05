import numpy as np
from scipy.optimize import brentq, root
from scipy.integrate import odeint
from copy import deepcopy

import eos_defns

class State(object):

    def __init__(self, rho, v, vt, eps, eos, label=None):
        """
        Constructor
        """
        self.rho = rho
        self.v = v
        self.vt = vt
        self.eps = eps
        self.eos = eos
        self.W_lorentz = 1.0 / np.sqrt(1.0 - self.v**2 - self.vt**2)
        self.p = self.eos['p_from_rho_eps'](rho, eps)
        self.h = self.eos['h_from_rho_eps'](rho, eps)
        self.cs = self.eos['cs_from_rho_eps'](rho, eps)
        self.label = label

    def prim(self):
        return np.array([self.rho, self.v, self.vt, self.eps])

    def state(self):
        return np.array([self.rho, self.v, self.vt, self.eps, self.p,\
        self.W_lorentz, self.h, self.cs])

    def wavespeed(self, wavenumber):
        if wavenumber == 1:
            return self.v
        elif abs(wavenumber - 1) == 1:
            s = wavenumber - 1
            term1 = self.v * (1.0 - self.cs**2)
            term2 = (1.0 - self.v**2 - self.vt**2) * (1.0 - self.v**2 - 
            self.vt**2 * self.cs**2)
            term3 = 1.0 - (self.v**2 + self.vt**2) * self.cs**2
            return (term1 + s * self.cs * np.sqrt(term2)) / term3
        else:
            raise NotImplementedError("wavenumber must be 0, 1, 2")
            
    def vt_from_known(self, rho, v, eps):
        h = self.eos['h_from_rho_eps'](rho, eps)
        vt = self.h * self.W_lorentz * self.vt
        vt *= np.sqrt((1.0 - v**2)/
        (h**2 + (self.h * self.W_lorentz * self.vt)**2))
        return vt

    def latex_string(self):
        s = r"\begin{pmatrix} \rho \\ v_x \\ v_t \\ \epsilon \end{pmatrix}"
        if self.label:
            s += r"_{{{}}} = ".format(self.label)
        s += r"\begin{{pmatrix}} {:.4f} \\ {:.4f} \\ {:.4f} \\ {:.4f} \end{{pmatrix}}".format(\
        self.rho, self.v, self.vt, self.eps)
        return s

    def _repr_latex_(self):
        s = r"$" + self.latex_string() + r"$"
        return s

class Wave(object):

    def __init__(self, q_known, unknown_value, wavenumber):
        """
        Initialize a wave.
        
        There are two possibilities: the wave is linear (wavenumber = 1),
        which is a contact, where the known value is the left state and the
        unknown value the right state.
        
        The second possibility is that the wave is nonlinear (wavenumber = 0,2)
        which is either a shock or a rarefaction, where the known value is the
        left/right state (wavenumber = 0,2 respectively) and the unknown value
        the pressure in the star state.
        """
        self.trivial = False
        assert(wavenumber in [0, 1, 2]), "wavenumber must be 0, 1, 2"
        self.wavenumber = wavenumber
        if self.wavenumber == 1:
            self.type = "Contact"
            assert(isinstance(unknown_value, State)), "unknown_value must " \
            "be a State when wavenumber is 1"
            self.q_l = q_known
            self.q_r = unknown_value
            assert(np.allclose(self.q_l.v, self.q_r.v)), "For a contact, "\
            "wavespeeds must match across the wave"
            assert(np.allclose(self.q_l.p, self.q_r.p)), "For a contact, "\
            "pressure must match across the wave"
            if np.allclose(self.q_l.state(), self.q_r.state()):
                self.trivial = True
            self.wave_speed = np.array([self.q_l.v, self.q_r.v])
            self.name = r"{\cal C}"
        elif self.wavenumber == 0:
            self.q_l = deepcopy(q_known)
            if (self.q_l.p < unknown_value):
                self.solve_shock(q_known, unknown_value)
            else:
                self.solve_rarefaction(q_known, unknown_value)
        else:
            self.q_r = deepcopy(q_known)
            if (self.q_r.p < unknown_value):
                self.solve_shock(q_known, unknown_value)
            else:
                self.solve_rarefaction(q_known, unknown_value)
    
    def solve_shock(self, q_known, p_star):
        
        self.type = "Shock"
        lr_sign = self.wavenumber - 1

        def shock_root(rho_eps):
            rho, eps = rho_eps
            p = q_known.eos['p_from_rho_eps'](rho, eps)
            h = q_known.eos['h_from_rho_eps'](rho, eps)
            dw = np.zeros_like(rho_eps)
            dw[0] = p_star - p
            dw[1] = (h**2 - q_known.h**2) - \
            (h/rho + q_known.h/q_known.rho) * (p - q_known.p)
            return dw

        self.name = r"{\cal S}"
        if self.wavenumber == 0:
            label = r"\star_L"
            self.name += r"_{\leftarrow}"
        else:
            label = r"\star_R"
            self.name += r"_{\rightarrow}"

        if np.allclose(q_known.p, p_star):
            self.trivial = True
            q_unknown = State(q_known.rho, q_known.v, q_known.vt, q_known.eps, 
            q_known.eos, label)
            v_shock = q_known.wavespeed(self.wavenumber)
        else:
            rho_eps = root(shock_root, np.array([q_known.rho, q_known.eps]))
            rho, eps = rho_eps.x
            dp = p_star - q_known.p
            h = 1.0 + eps + p_star / rho
            dh2 = h**2 - q_known.h**2
            j = np.sqrt(-dp / (dh2 / dp - 2.0 * q_known.h / q_known.rho))
            v_shock = (q_known.rho**2 * q_known.W_lorentz**2 * q_known.v + \
            lr_sign * j**2 * \
            np.sqrt(1.0 + q_known.rho**2 * q_known.W_lorentz**2 * (1.0 - q_known.v**2) / j**2)) / \
            (q_known.rho**2 * q_known.W_lorentz**2 + j**2)
            W_lorentz_shock = 1.0 / np.sqrt(1.0 - v_shock**2)
            v = (q_known.h * q_known.W_lorentz * q_known.v + lr_sign * dp * W_lorentz_shock / j) / \
            (q_known.h * q_known.W_lorentz + dp * (1.0 / q_known.rho / q_known.W_lorentz + \
            lr_sign * q_known.v * W_lorentz_shock / j))
            vt = q_known.vt_from_known(rho, v, eps)
            q_unknown = State(rho, v, vt, eps, q_known.eos, label)
                        
        if self.wavenumber == 0:
            self.q_r = deepcopy(q_unknown)
        else:
            self.q_l = deepcopy(q_unknown)

        self.wave_speed = np.array([v_shock, v_shock])
        
    
    def solve_rarefaction(self, q_known, p_star):
        
        self.type = "Rarefaction"
        lr_sign = self.wavenumber - 1
        
        def rarefaction_dwdp(w, p):
            """
            There is a tricky point here that needs investigation. If
            the input p is used here, rather than local_state.p, then they
            can diverge (when vt is significant) leading to overflows of g. By
            using local_state we avoid the overflow, but it may mean the final
            state is not very accurate. Even this isn't enough to tackle the
            faster test bench 3 problem.
            """
            dwdp = np.zeros_like(w)
            rho, v, eps = w
            vt = q_known.vt_from_known(rho, v, eps)
            local_state = State(rho, v, vt, eps, q_known.eos)
            cs = local_state.cs
            h = local_state.h
            W_lorentz = local_state.W_lorentz
            xi = local_state.wavespeed(self.wavenumber)
            g = vt**2 * (xi**2 - 1.0) / (1.0 - xi * v)**2
            dwdp[0] = 1.0 / (h * cs**2)
            dwdp[1] = lr_sign / (rho * h * W_lorentz**2 * cs) / np.sqrt(1.0 + g)
            dwdp[2] = local_state.p / (rho**2 * h * cs**2)
            return dwdp

        self.name = r"{\cal R}"
        if self.wavenumber == 0:
            label = r"\star_L"
            self.name += r"_{\leftarrow}"
        else:
            label = r"\star_R"
            self.name += r"_{\rightarrow}"
            
        v_known = q_known.wavespeed(self.wavenumber)

        if np.allclose(q_known.p, p_star):
            self.trivial = True
            q_unknown = State(q_known.rho, q_known.v, q_known.vt, q_known.eps,
            q_known.eos, label)
            v_unknown = v_known
        else:
            w_all = odeint(rarefaction_dwdp, 
                           np.array([q_known.rho, q_known.v, q_known.eps]), 
                           [q_known.p, p_star], rtol = 1e-12, atol = 1e-10)
            q_unknown = State(w_all[-1, 0], w_all[-1, 1], 
                              q_known.vt_from_known(w_all[-1, 0], w_all[-1, 1], w_all[-1, 2]),
                              w_all[-1, 2], q_known.eos, label)
            v_unknown = q_unknown.wavespeed(self.wavenumber)
            
        self.wave_speed = []
        if self.wavenumber == 0:
            self.q_r = deepcopy(q_unknown)
            self.wave_speed = np.array([v_known, v_unknown])
        else:
            self.q_l = deepcopy(q_unknown)
            self.wave_speed = np.array([v_unknown, v_known])

    def latex_string(self):
        s = self.name
        s += r": \lambda^{{({})}}".format(self.wavenumber)
        if self.type == "Rarefaction":
            s += r"\in [{:.4f}, {:.4f}]".format(self.wave_speed[0], 
                         self.wave_speed[1])
        else:
            s += r"= {:.4f}".format(self.wave_speed[0])
        return s

    def _repr_latex_(self):
        s = r"$" + self.latex_string() + r"$"
        return s

class RP(object):
    """
    This is a more general Riemann Problem class.

    Allows for different EOSs on both sides (as required for burning problems).
    Uses the State class.
    """

    def __init__(self, state_l, state_r):
        """
        Constructor
        """
        self.state_l = state_l
        self.state_r = state_r
        
        def find_delta_v(p_star_guess):
            
            wave_l = Wave(self.state_l, p_star_guess, 0)
            wave_r = Wave(self.state_r, p_star_guess, 2)
            
            return wave_l.q_r.v - wave_r.q_l.v
        
        pmin = min(self.state_l.p, self.state_r.p)
        pmax = max(self.state_l.p, self.state_r.p)
        while find_delta_v(pmin) * find_delta_v(pmax) > 0.0:
            pmin /= 2.0
            pmax *= 2.0
            
        self.p_star = brentq(find_delta_v, 0.9*pmin, 1.1*pmax)
        wave_l = Wave(self.state_l, self.p_star, 0)
        wave_r = Wave(self.state_r, self.p_star, 2)
        self.state_star_l = wave_l.q_r
        self.state_star_r = wave_r.q_l
        self.waves = [wave_l, 
                      Wave(self.state_star_l, self.state_star_r, 1), wave_r]
        

    def _repr_latex_(self):
        s = r"$\begin{cases} "
        s += self.state_l.latex_string()
        s += r",\\ "
        s += self.state_r.latex_string()
        s += r", \end{cases} \quad \implies \quad "
        for wave in self.waves:
            s+= wave.name
        s += r", \quad p_* = {:.4f}, \quad".format(self.p_star)
        s += r"\begin{cases} "
        for wave in self.waves[:-1]:
            s += wave.latex_string() + r",\\ "
        s += self.waves[-1].latex_string()
        s += r", \end{cases} \quad \begin{cases} "
        s += self.state_star_l.latex_string()
        s += r",\\ "
        s += self.state_star_r.latex_string()
        s += r". \end{cases}$"
        return s

if __name__ == "__main__":
    
    eos = eos_defns.eos_gamma_law(5.0/3.0)
    w_left = State(1.0, 0.0, 0.0, 1.5, eos, label="L")
    w_right = State(0.125, 0.0, 0.0, 1.2, eos, label="R")
    w_left = State(1.0, 0.0, 0.9, 0.015, eos, label="L")
    w_right = State(1.0, 0.0, 0.9, 1500, eos, label="R")
    rp = RP(w_left, w_right)
    print(rp.p_star)
