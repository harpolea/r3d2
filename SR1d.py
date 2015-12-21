import numpy as np
from scipy.optimize import brentq, root
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from copy import deepcopy

class State(object):

    def __init__(self, rho, v, eps, eos, label=None):
        """
        Constructor
        """
        self.rho = rho
        self.v = v
        self.eps = eps
        self.eos = eos
        self.W_lorentz = 1.0 / np.sqrt(1.0 - self.v**2)
        self.p = self.eos['p_from_rho_eps'](rho, eps)
        self.h = self.eos['h_from_rho_eps'](rho, eps)
        self.cs = self.eos['cs_from_rho_eps'](rho, eps)
        self.label = label

    def prim(self):
        return np.array([self.rho, self.v, self.eps])

    def state(self):
        return np.array([self.rho, self.v, self.eps, self.p,\
        self.W_lorentz, self.h, self.cs])

    def wavespeed(self, wavenumber):
        if wavenumber == 1:
            return self.v
        elif abs(wavenumber - 1) == 1:
            s = wavenumber - 1
            return (self.v + s * self.cs) / (1.0 + s * self.v * self.cs)
        else:
            raise NotImplementedError("wavenumber must be 0, 1, 2")

    def latex_string(self):
        s = r"\begin{pmatrix} \rho \\ v \\ \epsilon \end{pmatrix}"
        if self.label:
            s += r"_{{{}}} = ".format(self.label)
        s += r"\begin{{pmatrix}} {:.4f} \\ {:.4f} \\ {:.4f} \end{{pmatrix}}".format(\
        self.rho, self.v, self.eps)
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
            q_unknown = State(q_known.rho, q_known.v, q_known.eps, \
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
            lr_sign * q_known.v * W_lorentz_shock / j));
            q_unknown = State(rho, v, eps, q_known.eos, label)
                        
        if self.wavenumber == 0:
            self.q_r = deepcopy(q_unknown)
        else:
            self.q_l = deepcopy(q_unknown)

        self.wave_speed = np.array([v_shock, v_shock])
        
    
    def solve_rarefaction(self, q_known, p_star):
        
        self.type = "Rarefaction"
        lr_sign = self.wavenumber - 1
        
        def rarefaction_dwdp(w, p):
            dwdp = np.zeros_like(w)
            rho, v, eps = w
            cs = q_known.eos['cs_from_rho_eps'](rho, eps)
            h = q_known.eos['h_from_rho_eps'](rho, eps)
            W_lorentz = 1.0 / np.sqrt(1.0 - v**2)
            dwdp[0] = 1.0 / (h * cs**2)
            dwdp[1] = lr_sign / (rho * h * W_lorentz**2 * cs)
            dwdp[2] = p / (rho**2 * h * cs**2)
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
            q_unknown = State(q_known.rho, q_known.v, q_known.eps, \
            q_known.eos, label)
            v_unknown = v_known
        else:
            w_all = odeint(rarefaction_dwdp, \
            np.array([q_known.rho, q_known.v, q_known.eps]), [q_known.p, p_star])
            q_unknown = State(w_all[-1, 0], w_all[-1, 1], w_all[-1, 2], 
                              q_known.eos, label)
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
            
        self.p_star = brentq(find_delta_v, 0.5*pmin, 2.0*pmax)
        wave_l = Wave(self.state_l, self.p_star, 0)
        wave_r = Wave(self.state_r, self.p_star, 2)
        self.state_star_l = wave_l.q_r
        self.state_star_r = wave_r.q_l
        self.waves = [wave_l, 
                      Wave(self.state_star_l, self.state_star_r, 1), wave_r]
        

    def _repr_latex_(self):
        s = r"$\begin{cases} "
        for wave in self.waves:
            s+= wave.name
        s += r",\\ "
        for wave in self.waves:
            s += wave.latex_string() + r",\\ "
        for state in self.state_l, self.state_star_l, self.state_star_r, self.state_r:
            s += state.latex_string()
            s += r",\\ "
        s += r"p_* = {:.4f}".format(self.p_star)
        s += r"\end{cases}$"
        return s

class SR1d(object):

    def __init__(self, t_end, w_l, w_r, gamma=5./3.):
        """
        Constructor, initialises variables and the full left and right states given the primitive variables.
        """
        self.t_end = t_end
        self.gamma = gamma
        self.x = np.linspace(0., 1.)
        self.xi = (self.x - 0.5) / t_end
        self.q = np.zeros((len(self.x), 7))

        self.q_sharper = np.zeros((len(self.x), 7))
        self.xi_sharper = self.xi


        # compute initial left and right states given
        # the primitive variables
        self.q_l = self.compute_state(w_l)
        self.q_r = self.compute_state(w_r)


    def compute_state(self, w):
        """
        Convert from the basic primitive variables
            w = (rho, v, eps)
        to the full set
            q = (rho, v, eps, p, W, h, cs^2)
        """
        rho, v, eps = w

        p = (self.gamma - 1.) * rho * eps
        W_lorentz = 1. / np.sqrt( 1. - v**2)
        h = 1. + eps + p / rho
        cs = np.sqrt(self.gamma * p / (rho * h))

        return np.array([rho, v, eps, p, W_lorentz, h, cs])


    def find_p_star(self, q_l, q_r, p_star_0):
        """
        Find the value of q_star that solves the Riemann problem.
        """

        pmin = min(q_l[3], q_r[3], p_star_0)
        pmax = max(q_l[3], q_r[3], p_star_0)

        def find_delta_v(p_s):

            q_star_l = self.get_state(q_l, p_s, -1)
            v_star_l = q_star_l[1]
            q_star_r = self.get_state(q_r, p_s, 1)
            v_star_r = q_star_r[1]

            return v_star_l - v_star_r

        return brentq(find_delta_v, 0.5*pmin, 2*pmax)


    def get_state(self, q_known, p_star, lr_sign):
        """
        Given the known state and the pressure the other side of the wave,
        compute all the state information
        """
        rho_known, v_known, eps_known, p_known, _, h_known, cs_known = q_known

        e_known = rho_known * (1. + eps_known)

        if (p_star > p_known): # Shock wave

            # Check the root of the quadratic
            a = 1. + (self.gamma - 1.) * (p_known - p_star) / (self.gamma * p_star)
            b = 1. - a
            c = h_known * (p_known - p_star) / rho_known - h_known**2

            if (c > b**2 / (4. * a)):
                raise ValueError('Unphysical enthalpy')

            # Find quantities across the wave
            h_star = ( -b + np.sqrt( b**2 - 4. * a * c) ) / (2. * a)
            rho_star = self.gamma * p_star / ( (self.gamma - 1.) * (h_star - 1.) )
            eps_star = p_star / (rho_star * (self.gamma - 1.))
            e_star = rho_star + p_star / (self.gamma - 1.)

            v_12 = -lr_sign * \
                np.sqrt( (p_star - p_known) * (e_star - e_known) / \
                ( (e_known + p_star) * (e_star + p_known) ) )
            v_star = (v_known - v_12) / (1. - v_known * v_12)

        else: # Rarefaction wave

            rho_star = rho_known * (p_star / p_known)**(1. / self.gamma)
            eps_star = p_star / (rho_star * (self.gamma - 1.))
            h_star = 1. + eps_star + p_star / rho_star
            cs_star = np.sqrt(self.gamma * p_star / (h_star * rho_star))
            sqgm1 = np.sqrt(self.gamma - 1.)
            a = (1. + v_known) / (1. - v_known) * \
                ( ( sqgm1 + cs_known ) / ( sqgm1 - cs_known ) * \
                ( sqgm1 - cs_star  )  / ( sqgm1 + cs_star  ) )**(-lr_sign * \
                2. / sqgm1)

            v_star = (a - 1.) / (a + 1.)

        return np.array([rho_star, v_star, eps_star])


    def get_wave_speeds(self, l, s_l, s_r, r):
        """
        Calculate wave speeds given states
        """
        wave_speeds = np.zeros((5, 1))

        rho_l, v_l, _, p_l, W_l, h_l, cs_l = l
        rho_s_l, v_s_l, _, p_s_l, _, h_s_l, cs_s_l = s_l
        rho_s_r, v_s_r, _, p_s_r, _, h_s_r, cs_s_r = s_r
        rho_r, v_r, _, p_r, W_r, h_r, cs_r = r

        # Left wave
        if (p_s_l > p_l): # Shock
            w2 = W_l**2
            j = -np.sqrt((p_s_l - p_l) / (h_l / rho_l - h_s_l / rho_s_l))
            a = j**2 + rho_l**2 * w2
            b = -v_l * rho_l**2 * w2
            wave_speeds[0] = (-b - j**2 * np.sqrt(1. + (rho_l / j)**2)) / a
            wave_speeds[1] = wave_speeds[0]
        else: # Rarefaction
            wave_speeds[0] = (v_l - cs_l) / (1. - v_l * cs_l)
            wave_speeds[1] = (v_s_l - cs_s_l) / (1. - v_s_l * cs_s_l)

        # Contact
        wave_speeds[2] = s_l[1]

        # Right wave
        if (p_s_r > p_r): # Shock
            w2 = W_r**2
            j = np.sqrt((p_s_r - p_r) / (h_r / rho_r - h_s_r / rho_s_r))
            a = j**2 + rho_r**2 * w2
            b = -v_r * rho_r**2 * w2
            wave_speeds[3] = (-b + j**2 * np.sqrt(1. + (rho_r / j)**2)) / a
            wave_speeds[4] = wave_speeds[3]
        else: # Rarefaction
            wave_speeds[3] = (v_s_r - cs_s_r) / (1. - v_s_r * cs_s_r)
            wave_speeds[4] = (v_r - cs_r) / (1. - v_r * cs_r)

        return wave_speeds


    def rarefaction(self, xi, q_known, lr_sign):
        """
        Compute the state inside a rarefaction wave.
        """
        rho_known, v_known, _, _, _, _, cs_known = q_known

        b = np.sqrt(self.gamma - 1.)
        c = (b + cs_known) / (b - cs_known)
        d = -lr_sign * b / 2.
        k = (1. + xi) / (1. - xi)
        l = c * k**d
        v_wave = ( (1. - v_known) / (1. + v_known) )**d

        def rarefaction_cs(cs_guess):
            return l * v_wave * (1. + lr_sign * cs_guess)**d * (cs_guess - b) + \
                (1. - lr_sign * cs_guess)**d * (cs_guess + b)

        cs = brentq(rarefaction_cs, 0., cs_known)

        v = (xi + lr_sign * cs) / (1. + lr_sign * xi * cs)
        rho = rho_known * ( ( cs**2 * ((self.gamma - 1.) - cs_known**2) ) /\
            ( cs_known**2 * ((self.gamma - 1.) - cs**2) ) )**(1. / (self.gamma - 1.))
        p = cs**2 * (self.gamma - 1.) * rho / (self.gamma *((self.gamma - 1.) - cs**2))
        eps = p / (rho * (self.gamma - 1.))

        return np.array([rho, v, eps])


    def riemann_solver(self):
        """
        Riemann solver for 1d SR hydro
        """
        # initial guess
        p_star_0 = 0.5 * (self.q_l[3] + self.q_r[3])

        ## Root find
        p_star = self.find_p_star(self.q_l, self.q_r, p_star_0)

        ## Compute final states, characteristic speeds etc.

        w_star_l = self.get_state(self.q_l, p_star, -1.)
        w_star_r = self.get_state(self.q_r, p_star,  1.)
        q_star_l = self.compute_state(w_star_l)
        q_star_r = self.compute_state(w_star_r)

        wave_speeds = self.get_wave_speeds(self.q_l, q_star_l,
                                           q_star_r, self.q_r)

        # characterise waves
        if (abs(wave_speeds[1] - wave_speeds[0]) < 1.e-10):
            print('Left wave is a shock, speed {}.'.format(wave_speeds[0]))
        else:
            print('Left wave is a rarefaction, speeds ({}, {}).'.format(wave_speeds[0], wave_speeds[1]))

        print('Contact wave has speed {}.'.format(wave_speeds[2]))
        if (abs(wave_speeds[4] - wave_speeds[3]) < 1.e-10):
            print('Right wave is a shock, speed {}.'.format(wave_speeds[3]))
        else:
            print('Right wave is a rarefaction, speeds ({}, {}).'.format(wave_speeds[3], wave_speeds[4]))

        # solve riemann problem
        

        # calculate sharper solution
        rarefaction_pts = 100

        xi_left = -0.5 / self.t_end
        xi_right = 0.5 / self.t_end

        q = self.q_sharper
        xi = self.xi_sharper
        xi = np.array([xi_left])

        # solve riemann problem
        if (xi < wave_speeds[0]):
            w = w_left
        elif (xi < wave_speeds[1]):
            w = self.rarefaction(xi, self.q_l, 1.)
        elif (xi < wave_speeds[2]):
            w = w_star_l
        elif (xi < wave_speeds[3]):
            w = w_star_r
        elif (xi < wave_speeds[4]):
            w = self.rarefaction(xi, self.q_r, -1.)
        else:
            w = w_right

        q = self.compute_state(w)

        if ((wave_speeds[0] > xi_left) and (wave_speeds[0] < xi_right)):
            xi = np.append(xi, wave_speeds[0])
            q = np.vstack((q, self.q_l))

        if ((wave_speeds[1] > xi_left) and (wave_speeds[1] < xi_right)):
            if (wave_speeds[1] > wave_speeds[0] + 1.e-10):
                xi = np.append(xi, np.linspace(xi[-1], wave_speeds[1], rarefaction_pts))
                for i in range(100):
                    w = self.rarefaction(xi[-1+i-rarefaction_pts], self.q_l, 1.)
                    q = np.vstack((q, self.compute_state(w)))

            else:
                xi = np.append(xi, wave_speeds[1])
                q = np.vstack((q, q_star_l))


        if ((wave_speeds[2] > xi_left) and (wave_speeds[2] < xi_right)):
            xi = np.append(xi, [wave_speeds[2], wave_speeds[2]])
            q = np.vstack((q, q_star_l, q_star_r))

        if ((wave_speeds[3] > xi_left) and (wave_speeds[3] < xi_right)):
            xi = np.append(xi, wave_speeds[3])
            q = np.vstack((q, q_star_r))

        if ((wave_speeds[4] > xi_left) and (wave_speeds[4] < xi_right)):
            if (wave_speeds[4] > wave_speeds[4] + 1.e-10):
                xi = np.append(xi, np.linspace(wave_speeds[3], wave_speeds[4], rarefaction_pts))
                for i in range(100):
                    w = self.rarefaction(xi[-1+i-rarefaction_pts], self.q_r, -1.)
                    q = np.vstack((q, self.compute_state(w)))

            else:
                xi = np.append(xi, wave_speeds[4])
                q = np.vstack((q, self.q_r))

        if ((wave_speeds[4] > xi_right) and wave_speeds[4] > wave_speeds[3] + 1.e-10):
            xi = np.append(xi, np.linspace(wave_speeds[3], xi_right, rarefaction_pts))
            for i in range(100):
                w = self.rarefaction(xi[-1+i-rarefaction_pts], self.q_r, -1.)
                q = np.vstack((q, self.compute_state(w)))

        self.q_sharper = q
        self.xi_sharper = xi


    def plot_results(self, filename=None):
        """
        Produce plots of the density, speed and pressure.
        """

        plt.clf()
        plt.rc("font", size=12)
        plt.figure(num=1, figsize=(20,7), dpi=100, facecolor='w')

        q = self.q
        x = self.x

        ax = plt.subplot(131)
        ax.set_xlabel("$x$")
        ax.set_ylabel(r"$\rho$")

        ax2 = plt.subplot(132)
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$v$")

        ax3 = plt.subplot(133)
        ax3.set_xlabel("$x$")
        ax3.set_ylabel("$p$")

        # plot the sharper solution
        q = self.q_sharper
        xi = self.xi_sharper

        x = xi * self.t_end + 0.5

        ax.plot(x, q[:, 0], 'bx--')
        ax2.plot(x, q[:, 1], 'rx--')
        ax3.plot(x, q[:, 3], 'gx--')

        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename)

        plt.show()

    def _repr_png_(self):
        """
        png plot of the density, speed and pressure.
        """

        plt.clf()
        plt.rc("font", size=12)
        plt.figure(num=1, figsize=(20,7), dpi=100, facecolor='w')

        q = self.q
        x = self.x

        ax = plt.subplot(131)
        ax.set_xlabel("$x$")
        ax.set_ylabel(r"$\rho$")

        ax2 = plt.subplot(132)
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$v$")

        ax3 = plt.subplot(133)
        ax3.set_xlabel("$x$")
        ax3.set_ylabel("$p$")

        # plot the sharper solution
        q = self.q_sharper
        xi = self.xi_sharper

        x = xi * self.t_end + 0.5

        ax.plot(x, q[:, 0], 'bx--')
        ax2.plot(x, q[:, 1], 'rx--')
        ax3.plot(x, q[:, 3], 'gx--')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    

    # initial data
    t_end = 0.4
    gamma = 5./3.
    w_left  = [10., 0., 2.   ]
    w_right = [ 1., 0., 1.e-5]

    # initialise
    sr1d = SR1d(t_end, w_left, w_right, gamma)

    # solve and plot
    sr1d.riemann_solver()

    sr1d.plot_results(filename="sr1d.png")
