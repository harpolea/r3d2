import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

def eos_gamma_law(gamma):

    p_from_rho_eps = lambda rho, eps : (gamma - 1.0) * rho * eps
    h_from_rho_eps = lambda rho, eps : 1.0 + gamma * eps
    cs_from_rho_eps = lambda rho, eps : \
    np.sqrt(gamma * p_from_rho_eps(rho, eps) / (rho * h_from_rho_eps(rho, eps)))

    eos = {'p_from_rho_eps' : p_from_rho_eps,
           'h_from_rho_eps' : h_from_rho_eps,
           'cs_from_rho_eps' : cs_from_rho_eps}

    return eos

def eos_multi_gamma_law(gamma, wave_i):

    p_from_rho_eps = lambda rho, eps : (gamma[wave_i] - 1.0) * rho * eps
    h_from_rho_eps = lambda rho, eps : 1.0 + gamma[wave_i] * eps
    cs_from_rho_eps = lambda rho, eps : \
    np.sqrt(gamma[wave_i] * p_from_rho_eps(rho, eps) / (rho * h_from_rho_eps(rho, eps)))

    eos = {'p_from_rho_eps' : p_from_rho_eps,
           'h_from_rho_eps' : h_from_rho_eps,
           'cs_from_rho_eps' : cs_from_rho_eps}

    return eos

def eos_polytrope_law(gamma, gamma_th, rho_transition, k):

    def p_from_rho_eps(rho, eps):
        if (rho < rho_transition):
            p_cold = k[0] * rho**gamma[0]
            eps_cold = p_cold / rho / (gamma[0] - 1.)
        else:
            p_cold = k[1] * rho**gamma[1]
            eps_cold = p_cold / rho / (gamma[1] - 1.) - \
                k[1] * rho_transition**(gamma[1] - 1.) + \
                k[0] * rho_transition**(gamma[0] - 1.)
        
        p_th = max(0.0, (gamma_th - 1.0) * rho * (eps - eps_cold))

        return p_cold + p_th

    def h_from_rho_eps(rho, eps):
        if (rho < rho_transition):
            p_cold = k[0] * rho**gamma[0]
            eps_cold = p_cold / rho / (gamma[0] - 1.0)
        else:
            p_cold = k[1] * rho**gamma[1]
            eps_cold = p_cold / rho / (gamma[1] - 1.0) - \
                k[1] * rho_transition**(gamma[1] - 1.0) + \
                k[0] * rho_transition**(gamma[0] - 1.0)

        p_th = max(0., (gamma_th - 1.) * rho * (eps - eps_cold))

        return 1. + eps_cold + eps + (p_cold + p_th)/ rho

    def cs_from_rho_eps(rho, eps):
        return np.sqrt(gamma * p_from_rho_eps(rho, eps) / (rho * h_from_rho_eps(rho, eps)))

    eos = {'p_from_rho_eps' : p_from_rho_eps,
           'h_from_rho_eps' : h_from_rho_eps,
           'cs_from_rho_eps' : cs_from_rho_eps}

    return eos


class State():

    def __init__(self, rho, v, eps, eos, label=None):
        """
        Constructor
        """
        self.rho = rho
        self.v = v
        self.eps = eps
        self.W_lorentz = 1.0 / np.sqrt(1.0 - self.v**2)
        self.p = eos['p_from_rho_eps'](rho, eps)
        self.h = eos['h_from_rho_eps'](rho, eps)
        self.cs = eos['cs_from_rho_eps'](rho, eps)
        self.label = label

    def prim(self):
        return np.array([self.rho, self.v, self.eps])

    def state(self):
        return np.array([self.rho, self.v, self.eps, self.p,\
        self.W_lorentz, self.h, self.cs])

    def _repr_latex_(self):
        s = r"$\begin{{pmatrix}} \rho \\ v \\ \epsilon \end{{pmatrix}}"
        if self.label:
            s += r"_{{{}}} = ".format(self.label)
        s += r"\begin{{pmatrix}} {:.4f} \\ {:.4f} \\ {:.4f} \end{{pmatrix}}$".format(\
        self.rho, self.v, self.eps)
        return s

class Wave():

    def __init__(self, q_l, q_r):
        # initialise wave with left and right states and speed. This defaults to the behaviour of a contact wave.
        self.q_l = q_l
        self.q_r = q_r
        self.v_l = q_l.v
        self.v_r = q_l.v


class Shock(Wave):
    """
    shock wave
    """
    def __init__(self, q_l, q_r, lr_sign):

        # wave going in opposite direction, switch states.
        # FIXME: this is kind of hacky? Must be a better way
        if lr_sign == -1:
            q_l, q_r = q_r, q_l

        Wave.__init__(self, q_l, q_r)

        w2 = q_l.W_lorentz**2
        j = -np.sqrt((q_l.p - q_r.p) / (q_r.h / q_r.rho - q_l.h / q_l.rho))
        a = j**2 + q_r.rho**2 * w2
        b = -q_r.v * q_r.rho**2 * w2

        # speed of wave
        self.v_l = (-b - j**2 * np.sqrt(1. + (q_r.rho / j)**2)) / a
        self.v_r = self.v_l

        # wave going in opposite direction, switch states back
        if lr_sign == -1:
            self.q_l, self.q_r = q_r, q_l


    def get_state(self):
        """
        Compute the state other side of the shock wave.
        """
        pass

class Rarefaction(Wave):
    """
    rarefaction wave
    """
    def __init__(self, q_l, q_r):

        Wave.__init__(self, q_l, q_r)

        # speed left of wave
        self.v_l = (q_l.v - q_l.cs) / (1. - q_l.v * q_l.cs)
        # speed right of wave
        self.v_r = (q_r.v - q_r.cs) / (1. - q_r.v * q_r.cs)

        self.state = self.get_state()



    def get_state(self):
        """
        Compute the state inside a rarefaction wave.
        """
        pass


class RP():
    """
    This is a more general Riemann Problem class.

    Allows for different EOSs on both sides (as required for burning problems).
    Uses the State class.
    """

    def __init__(self, state_l, state_r, eos_l, eos_r, gamma=5./3.):
        """
        Constructor
        """
        self.state_l = state_l
        self.state_r = state_r
        self.gamma = gamma

        self.p_star = self.find_pstar()
        self.state_star_l = self.get_state(self.state_l, self.p_star, eos_l, -1)
        self.state_star_r = self.get_state(self.state_r, self.p_star, eos_r, +1)
        self.wave_speeds = self.get_wave_speeds(self.state_star_l, self.state_star_r)



    def find_pstar(self, p_star_0=None):
        """
        Find the value of q_star that solves the Riemann problem.
        """
        pmin = min(self.state_l.p, self.state_r.p, p_star_0)
        pmax = max(self.state_l.p, self.state_r.p, p_star_0)

        def find_delta_v(p_s):

            q_star_l = self.get_state(self.state_l.p, p_s, -1)
            v_star_l = q_star_l.v
            q_star_r = self.get_state(self.state_r.p, p_s, 1)
            v_star_r = q_star_r.v

            return v_star_l - v_star_r

        return brentq(find_delta_v, 0.5*pmin, 2*pmax)

    def get_state(self, state_known, p_star, eos, lr_sign):
        """
        Given the known state and the pressure the other side of the wave,
        compute all the state information
        """

        if (p_star > state_known.p): # Shock wave

            # Check the root of the quadratic
            a = 1. + (self.gamma - 1.) * (state_known.p - p_star) / (self.gamma * p_star)
            b = 1. - a
            c = state_known.h * (state_known.p - p_star) / state_known.rho - state_known.h**2

            if (c > b**2 / (4. * a)):
                raise ValueError('Unphysical enthalpy')

            # Find quantities across the wave
            h_star = ( -b + np.sqrt( b**2 - 4. * a * c) ) / (2. * a)
            rho_star = self.gamma * p_star / ( (self.gamma - 1.) * (h_star - 1.) )
            eps_star = p_star / (rho_star * (self.gamma - 1.))
            e_star = rho_star + p_star / (self.gamma - 1.)

            v_12 = -lr_sign * \
                np.sqrt( (p_star - state_known.p) * (e_star - state_known.eps) / \
                ( (state_known.eps + p_star) * (e_star + state_known.p) ) )
            v_star = (state_known.v - v_12) / (1. - state_known.v * v_12)

        else: # Rarefaction wave

            rho_star = state_known.rho * (p_star / state_known.p)**(1. / self.gamma)
            eps_star = p_star / (rho_star * (self.gamma - 1.))
            h_star = 1. + eps_star + p_star / rho_star
            cs_star = np.sqrt(self.gamma * p_star / (h_star * rho_star))
            sqgm1 = np.sqrt(self.gamma - 1.)
            a = (1. + state_known.v) / (1. - state_known.v) * \
                ( ( sqgm1 + state_known.cs ) / ( sqgm1 - state_known.cs ) * \
                ( sqgm1 - cs_star  )  / ( sqgm1 + cs_star  ) )**(-lr_sign * \
                2. / sqgm1)

            v_star = (a - 1.) / (a + 1.)

        return State(rho_star, v_star, eps_star, eos)

    # FIXME: maybe instead of this, produce a 3-tuple of Waves (see next function)
    def get_wave_speeds(self, s_l, s_r):
        """
        Calculate wave speeds given states
        """
        wave_speeds = np.zeros((5, 1))

        l = self.state_l
        r = self.state_r

        # Left wave
        if (s_l.p > l.p): # Shock
            shock = Shock(l, s_l, -1)
            wave_speeds[:2] = shock.v_l
        else: # Rarefaction
            rarefaction = Rarefaction(l, s_l)
            wave_speeds[0] = rarefaction.v_l
            wave_speeds[1] = rarefaction.v_r

        # Contact
        wave_speeds[2] = s_l.v_l

        # Right wave
        if (s_r.p > r.p): # Shock
            shock = Shock(s_r, r, 1)
            wave_speeds[3:] = shock.v_l

        else: # Rarefaction
            rarefaction = Rarefaction(s_r)
            wave_speeds[3] = rarefaction.v_l
            wave_speeds[4] = rarefaction.v_r

        return wave_speeds

    def get_waves(self, s_l, s_r):
        """
        Returns tuple of (left, contact, right) Waves given the left and right states.
        """
        l = self.state_l
        r = self.state_r

        # Left wave
        if (s_l.p > l.p): # Shock
            l_wave = Shock(l, s_l, -1)
        else: # Rarefaction
            l_wave = Rarefaction(l, s_l)

        # Right wave
        if (s_r.p > r.p): # Shock
            r_wave = Shock(s_r, r, 1)
        else: # Rarefaction
            r_wave = Rarefaction(s_r)

        return l_wave, Wave(s_l, s_r), r_wave



class SR1d():

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
        if (s_r.p > r.p): # Shock
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

        xi = self.xi
        w = np.zeros((len(self.xi), 3))

        # solve riemann problem
        for i in range(len(xi)):

            if (xi[i] < wave_speeds[0]):
                w[i, :] = self.q_l[:3]
            elif (xi[i] < wave_speeds[1]):
                w[i, :] = self.rarefaction(xi[i], self.q_l, 1)
            elif (xi[i] < wave_speeds[2]):
                w[i, :] = w_star_l
            elif (xi[i] < wave_speeds[3]):
                w[i, :] = w_star_r
            elif (xi[i] < wave_speeds[4]):
                w[i, :] = self.rarefaction(xi[i], self.q_r, -1)
            else:
                w[i, :] = self.q_r[:3]

            self.q[i, :] = self.compute_state(self.w[i, :])


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
