"""
R3D2 is a Relativistic Reactive Riemann problem solver for Deflagrations and Detonations. It extends standard solutions of the relativistic Riemann Problem to include a reaction term.
"""

from __future__ import division
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import odeint
from copy import deepcopy

import eos_defns

from matplotlib import pyplot
from IPython.core.pylabtools import print_figure

class State(object):
    """
    A state at a point. Initialized with the rest mass density, velocity, and
    specific internal energy, as well as an equation of state.
    """

    def __init__(self, rho, v, vt, eps, eos, label=None):
        r"""
        Constructor

        Parameters
        ----------

        rho : scalar
            Rest mass density :math:`\rho_0`
        v : scalar
            Velocity component in the normal (:math:`x`) direction :math:`v_x`
        v_t : scalar
            Velocity component tangential to :math:`x` :math:`v_t`
        eps : scalar
            Specific internal energy :math:`\epsilon`
        eos : dictionary
            Equation of State
        label : string
            Label for output purposes.
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
        r"""
        Return the primitive variables :math:`\rho, v_x, v_t, \epsilon`.
        """
        return np.array([self.rho, self.v, self.vt, self.eps])

    def state(self):
        r"""
        Return all variables :math:`\rho, v_x, v_t, \epsilon, p, W, h, c_s`.
        """
        return np.array([self.rho, self.v, self.vt, self.eps, self.p,\
        self.W_lorentz, self.h, self.cs])

    def wavespeed(self, wavenumber):
        """
        Compute the wavespeed given the wave number (0 for the left wave,
        2 for the right wave).

        Parameters
        ----------

        wavenumber: scalar
            Wave number ([0,1,2]).
        """
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
        r"""
        Computes tangential velocity across a wave.

        Parameters
        ----------

        self : State
            The known state
        rho : scalar
            Rest mass density in the unknown state across the wave
        v : scalar
            Normal velocity :math:`v_x` in the unknown state across the wave
        eps : scalar
            Specific internal energy in the unknown state across the wave

        Returns
        -------

        vt : scalar
            Tangential velocity :math:`v_t` in the unknown state
        """
        h = self.eos['h_from_rho_eps'](rho, eps)
        vt = self.h * self.W_lorentz * self.vt
        vt *= np.sqrt((1.0 - v**2)/
            (h**2 + (self.h * self.W_lorentz * self.vt)**2))
        return vt

    def latex_string(self):
        """
        Helper function to represent the state as a string.
        """
        s = r"\begin{pmatrix} \rho \\ v_x \\ v_t \\ \epsilon \end{pmatrix}"
        if self.label:
            s += r"_{{{}}} = ".format(self.label)
        s += r"\begin{{pmatrix}} {:.4f} \\ {:.4f} \\ {:.4f} \\ {:.4f} \end{{pmatrix}}".format(\
        self.rho, self.v, self.vt, self.eps)
        return s

    def _repr_latex_(self):
        """
        IPython or Jupyter repr.
        """
        s = r"$" + self.latex_string() + r"$"
        return s

def rarefaction_dwdp(w, p, q_known, wavenumber):
    r"""
    There is a tricky point here that needs investigation. If
    the input p is used here, rather than local_state.p, then they
    can diverge (when :math:`v_t` is significant) leading to overflows of g. By
    using local_state we avoid the overflow, but it may mean the final
    state is not very accurate.
    """
    lr_sign = wavenumber - 1
    dwdp = np.zeros_like(w)
    rho, v, eps = w
    vt = q_known.vt_from_known(rho, v, eps)
    local_state = State(rho, v, vt, eps, q_known.eos)
    cs = local_state.cs
    h = local_state.h
    W_lorentz = local_state.W_lorentz
    xi = local_state.wavespeed(wavenumber)
    # NOTE: what is g?
    g = vt**2 * (xi**2 - 1.0) / (1.0 - xi * v)**2
    dwdp[0] = 1.0 / (h * cs**2)
    dwdp[1] = lr_sign / (rho * h * W_lorentz**2 * cs) / np.sqrt(1.0 + g)
    dwdp[2] = local_state.p / (rho**2 * h * cs**2)
    return dwdp

class Wave(object):

    def __init__(self, q_known, unknown_value, wavenumber, unknown_eos=None,
                 t_i=None):
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
        self.trivial = False # NOTE: what does this variable represent?
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
                if unknown_eos is not None:
                    assert(t_i is not None)
                    self.solve_detonation(q_known, unknown_value, unknown_eos, t_i)
                else:
                    self.solve_shock(q_known, unknown_value)
            else:
                if unknown_eos is not None:
                    assert(t_i is not None)
                    self.solve_deflagration(q_known, unknown_value, unknown_eos, t_i)
                else:
                    self.solve_rarefaction(q_known, unknown_value)
        else:
            self.q_r = deepcopy(q_known)
            if (self.q_r.p < unknown_value):
                if unknown_eos is not None:
                    assert(t_i is not None)
                    self.solve_detonation(q_known, unknown_value, unknown_eos, t_i)
                else:
                    self.solve_shock(q_known, unknown_value)
            else:
                if unknown_eos is not None:
                    assert(t_i is not None)
                    self.solve_deflagration(q_known, unknown_value, unknown_eos, t_i)
                else:
                    self.solve_rarefaction(q_known, unknown_value)

    def mass_flux_squared(self, q_known, p_star, unknown_eos=None):

        if unknown_eos is None:
            unknown_eos = q_known.eos

        def shock_root_rho(rho):
            h = unknown_eos['h_from_rho_p'](rho, p_star)
            return (h**2 - q_known.h**2) - \
            (h/rho + q_known.h/q_known.rho) * (p_star - q_known.p)

        if p_star >= q_known.p:
            # Shock
            min_rho = q_known.rho
            shock_root_min = shock_root_rho(min_rho)
            max_rho = np.sqrt(p_star/q_known.p) * q_known.rho
            shock_root_max = shock_root_rho(max_rho)
            while(shock_root_min * shock_root_max > 0.0):
                min_rho /= 1.1 # Not sure - could end up with unphysical root?
                max_rho *= 10.0
                shock_root_min = shock_root_rho(min_rho)
                shock_root_max = shock_root_rho(max_rho)
        else:
            # Deflagration
            max_rho = q_known.rho
            shock_root_max = shock_root_rho(max_rho)
            min_rho = np.sqrt(p_star/q_known.p) * q_known.rho
            shock_root_min = shock_root_rho(min_rho)
            while(shock_root_min * shock_root_max > 0.0):
                min_rho /= 10.0 # Not sure - could end up with unphysical root?
                max_rho *= 1.1
                shock_root_min = shock_root_rho(min_rho)
                shock_root_max = shock_root_rho(max_rho)
        rho = brentq(shock_root_rho, min_rho, max_rho)
        h = unknown_eos['h_from_rho_p'](rho, p_star)
        eps = h - 1.0 - p_star / rho
        dp = p_star - q_known.p
        dh2 = h**2 - q_known.h**2
        j2 = -dp / (dh2 / dp - 2.0 * q_known.h / q_known.rho)

        return j2, rho, eps, dp

    def solve_shock(self, q_known, p_star, unknown_eos=None):
        r"""
        In the case of a shock, finds the unknown state on one side of the wave and wavespeed given the known state on the wave's other side and :math:`p_*`.

        Parameters
        ----------

        self : Wave
            The wave, which has a known state on one side and an unknown
            state on the other side.
        q_known : State
            The known state on one side of the wave
        p_star : scalar
            Pressure in the region of unknown state
        unknown_eos : dictionary
            Equation of state in the unknown region
        """

        self.type = "Shock"
        lr_sign = self.wavenumber - 1
        if unknown_eos is None:
            unknown_eos = q_known.eos

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
            j2, rho, eps, dp = self.mass_flux_squared(q_known, p_star, unknown_eos)
            j = np.sqrt(j2)
            v_shock = (q_known.rho**2 * q_known.W_lorentz**2 * q_known.v + \
                lr_sign * j**2 * \
                np.sqrt(1.0 + q_known.rho**2 * q_known.W_lorentz**2 * (1.0 - q_known.v**2) / j**2)) / \
                (q_known.rho**2 * q_known.W_lorentz**2 + j**2)
            W_lorentz_shock = 1.0 / np.sqrt(1.0 - v_shock**2)
            v = (q_known.h * q_known.W_lorentz * q_known.v + lr_sign * dp * W_lorentz_shock / j) / \
                (q_known.h * q_known.W_lorentz + dp * (1.0 / q_known.rho / q_known.W_lorentz + \
                lr_sign * q_known.v * W_lorentz_shock / j))
            vt = q_known.vt_from_known(rho, v, eps)
            q_unknown = State(rho, v, vt, eps, unknown_eos, label)

        if self.wavenumber == 0:
            self.q_r = deepcopy(q_unknown)
        else:
            self.q_l = deepcopy(q_unknown)

        self.wave_speed = np.array([v_shock, v_shock])

    def solve_rarefaction(self, q_known, p_star, unknown_eos=None):
        r"""
        In the case of a rarefaction, finds the unknown state on one side of the wave and wavespeed given the known state on the wave's other side and :math:`p_*`.

        Parameters
        ----------

        self : Wave
            The wave, which has a known state on one side and an unknown
            state on the other side.
        q_known : State
            The known state on one side of the wave
        p_star : scalar
            Pressure in the region of unknown state
        unknown_eos : dictionary
            Equation of state in the unknown region
        """

        self.type = "Rarefaction"
        if unknown_eos is None:
            unknown_eos = q_known.eos

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
                           [q_known.p, p_star], rtol = 1e-12, atol = 1e-10,
                           args=((q_known, self.wavenumber)))
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

    def solve_deflagration(self, q_known, p_star, unknown_eos, t_i):
        r"""
        In the case of a deflagration, finds the unknown state on one side of the wave and wavespeed given the known state on the wave's other side and :math:`p_*`.

        Parameters
        ----------

        self : Wave
            The wave, which has a known state on one side and an unknown
            state on the other side.
        q_known : State
            The known state on one side of the wave
        p_star : scalar
            Pressure in the region of unknown state
        unknown_eos : dictionary
            Equation of state in the unknown region
        t_i : scalar
            temperature
            NOTE: what is this the temperature of??
        """

        self.type = "Deflagration"
        self.name = r"({\cal WDF})"
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
            q_precursor = deepcopy(q_unknown)
            v_deflagration = v_known
        else:
            lr_sign = self.wavenumber - 1
            p_min = p_star
            p_max = q_known.p

#            def mass_flux_deflagration(p_0_star):
#                w_0_star = odeint(rarefaction_dwdp,
#                       np.array([q_known.rho, q_known.v, q_known.eps]),
#                       [q_known.p, p_0_star], rtol = 1e-12, atol = 1e-10,
#                       args=((q_known, self.wavenumber)))
#                q_0_star_known = State(w_0_star[-1, 0], w_0_star[-1, 1],
#                            q_known.vt_from_known(w_0_star[-1, 0], w_0_star[-1, 1], w_0_star[-1, 2]),
#                            w_0_star[-1, 2], q_known.eos, label)
#                j2, rho, eps, dp = self.mass_flux_squared(q_0_star_known, p_star, unknown_eos)
#                return j2, rho, eps, dp, q_0_star_known
#
#            def mass_flux_root(p_0_star):
#                j2, _, _, _, _ = mass_flux_deflagration(p_0_star)
#                return j2

            def deflagration_root(p_0_star, q_precursor):
                j2, rho, eps, dp = self.mass_flux_squared(q_precursor, p_0_star, unknown_eos)
                j = np.sqrt(j2)
                v_deflagration = (q_precursor.rho**2 *
                    q_precursor.W_lorentz**2 * q_precursor.v + \
                    lr_sign * j**2 * \
                    np.sqrt(1.0 + q_precursor.rho**2 *
                    q_precursor.W_lorentz**2 *
                    (1.0 - q_precursor.v**2) / j**2)) / \
                    (q_precursor.rho**2 * q_precursor.W_lorentz**2 + j**2)
                W_lorentz_deflagration = 1.0 / np.sqrt(1.0 - v_deflagration**2)
                v = (q_precursor.h * q_precursor.W_lorentz *
                     q_precursor.v + lr_sign * dp *
                     W_lorentz_deflagration / j) / \
                    (q_precursor.h * q_precursor.W_lorentz + dp * (1.0 /
                     q_precursor.rho / q_precursor.W_lorentz + \
                     lr_sign * q_precursor.v *
                     W_lorentz_deflagration / j))
                vt = q_precursor.vt_from_known(rho, v, eps)
                q_unknown = State(rho, v, vt, eps, unknown_eos, label)

                return q_unknown.wavespeed(self.wavenumber) - v_deflagration

            def precursor_root(p_0_star):
                shock = Wave(q_known, p_0_star, self.wavenumber)
                if self.wavenumber == 0:
                    q_precursor = shock.q_r
                else:
                    q_precursor = shock.q_l
                t_precursor = q_precursor.eos['t_from_rho_eps'](
                                q_precursor.rho, q_precursor.eps)
                return t_precursor - t_i

            # First, find the precursor shock
            t_min = precursor_root(p_min)
            t_max = precursor_root(p_max)
            assert(t_min < 0)

            if t_max <= 0:
                p_max *= 2
                t_max = precursor_root(p_max)

            p_0_star = brentq(precursor_root, p_min, p_max)
            precursor_shock = Wave(q_known, p_0_star, self.wavenumber)

            if self.wavenumber == 0:
                q_precursor = precursor_shock.q_r
            else:
                q_precursor = precursor_shock.q_l

            # Next, find the deflagration discontinuity
            j2, rho, eps, dp = self.mass_flux_squared(q_precursor,
                                p_star, unknown_eos)
            j = np.sqrt(j2)
            v_deflagration = (q_precursor.rho**2 *
                q_precursor.W_lorentz**2 * q_precursor.v + \
                lr_sign * j**2 * \
                np.sqrt(1.0 + q_precursor.rho**2 *
                q_precursor.W_lorentz**2 *
                (1.0 - q_precursor.v**2) / j**2)) / \
                (q_precursor.rho**2 * q_precursor.W_lorentz**2 + j**2)
            W_lorentz_deflagration = 1.0 / np.sqrt(1.0 - v_deflagration**2)
            v = (q_precursor.h * q_precursor.W_lorentz * q_precursor.v +
                 lr_sign * dp * W_lorentz_deflagration / j) / \
                 (q_precursor.h * q_precursor.W_lorentz + dp * (1.0 /
                  q_precursor.rho / q_precursor.W_lorentz + \
                 lr_sign * q_precursor.v * W_lorentz_deflagration / j))
            vt = q_precursor.vt_from_known(rho, v, eps)
            q_unknown = State(rho, v, vt, eps, unknown_eos, label)

            # If the speed in the unknown state means the characteristics are
            # not going into the deflagration, then this is an unstable strong
            # deflagration
            if (lr_sign*(q_unknown.wavespeed(self.wavenumber) - v_deflagration) < 0):
                p_cjdf = brentq(deflagration_root, (1.0+1e-9)*p_star,
                                (1.0-1e-9)*p_0_star,
                                args=(q_precursor,))
                j2, rho, eps, dp = self.mass_flux_squared(q_precursor, p_cjdf, unknown_eos)
                j = np.sqrt(j2)
                v_deflagration = (q_precursor.rho**2 *
                    q_precursor.W_lorentz**2 * q_precursor.v + \
                    lr_sign * j**2 * \
                    np.sqrt(1.0 + q_precursor.rho**2 *
                    q_precursor.W_lorentz**2 *
                    (1.0 - q_precursor.v**2) / j**2)) / \
                    (q_precursor.rho**2 * q_precursor.W_lorentz**2 + j**2)
                W_lorentz_deflagration = 1.0 / np.sqrt(1.0 - v_deflagration**2)
                v = (q_precursor.h * q_precursor.W_lorentz *
                    q_precursor.v + lr_sign * dp * W_lorentz_deflagration / j) / \
                    (q_precursor.h * q_precursor.W_lorentz + dp * (1.0 /
                     q_precursor.rho / q_precursor.W_lorentz + \
                     lr_sign * q_precursor.v * W_lorentz_deflagration / j))
                vt = q_precursor.vt_from_known(rho, v, eps)
                q_cjdf = State(rho, v, vt, eps, unknown_eos, label)
                # Now, we have to attach using a rarefaction
                w_0_star = odeint(rarefaction_dwdp,
                       np.array([q_cjdf.rho, q_cjdf.v, q_cjdf.eps]),
                       [q_cjdf.p, p_star], rtol = 1e-12, atol = 1e-10,
                       args=((q_cjdf, self.wavenumber)))
                q_unknown = State(w_0_star[-1, 0], w_0_star[-1, 1],
                            q_cjdf.vt_from_known(w_0_star[-1, 0], w_0_star[-1, 1], w_0_star[-1, 2]),
                            w_0_star[-1, 2], unknown_eos, label)

            self.p_0_star = p_0_star
            self.q_0_star = q_precursor

        self.wave_speed = []
        if self.wavenumber == 0:
            self.q_r = deepcopy(q_unknown)
            self.wave_speed = np.array([v_known, v_deflagration, q_precursor.wavespeed])
        else:
            self.q_l = deepcopy(q_unknown)
            self.wave_speed = np.array([q_precursor.wavespeed, v_deflagration, v_known])

    def solve_detonation(self, q_known, p_star, unknown_eos):
        r"""
        In the case of a detonation, finds the unknown state on one side of the wave and wavespeed given the known state on the wave's other side and :math:`p_*`.

        Parameters
        ----------

        self : Wave
            The wave, which has a known state on one side and an unknown
            state on the other side.
        q_known : State
            The known state on one side of the wave
        p_star : scalar
            Pressure in the region of unknown state
        unknown_eos : dictionary
            Equation of state in the unknown region
        """

        self.type = "Detonation"
        lr_sign = self.wavenumber - 1
        if unknown_eos is None:
            unknown_eos = q_known.eos

        self.name = r"({\cal SDT})"
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
            raise(NotImplementedError, "Do this")

        if self.wavenumber == 0:
            self.q_r = deepcopy(q_unknown)
        else:
            self.q_l = deepcopy(q_unknown)

        self.wave_speed = np.array([v_shock, v_shock])

    def plotting_data(self):

        if self.type == "Rarefaction":
            if self.wavenumber == 0:
                p = np.linspace(self.q_l.p, self.q_r.p)
                w_all = odeint(rarefaction_dwdp,
                               np.array([self.q_l.rho, self.q_l.v, self.q_l.eps]),
                               p, rtol = 1e-12, atol = 1e-10,
                               args=(self.q_l,self.wavenumber))
            else:
                p = np.linspace(self.q_r.p, self.q_l.p)
                w_all = odeint(rarefaction_dwdp,
                               np.array([self.q_r.rho, self.q_r.v, self.q_r.eps]),
                               p, rtol = 1e-12, atol = 1e-10,
                               args=(self.q_r,self.wavenumber))
                p = p[-1::-1]
                w_all = w_all[-1::-1,:]
            data = np.zeros((len(p),8))
            xi = np.zeros_like(p)
            for i in range(len(p)):
                state = State(w_all[i,0], w_all[i,1],
                              self.q_l.vt_from_known(w_all[i,0], w_all[i,1], w_all[i,2]),
                              w_all[i, 2], self.q_l.eos)
                xi[i] = state.wavespeed(self.wavenumber)
                data[i,:] = state.state()
        elif self.type == "Shock":
            data = np.vstack((self.q_l.state(), self.q_r.state()))
            xi = np.array([self.q_l.wavespeed(self.wavenumber),
                           self.q_l.wavespeed(self.wavenumber)])
        elif self.type == "Deflagration":
            if self.wavenumber == 0:
                p = np.linspace(self.q_l.p, self.p_0_star)
                w_all = odeint(rarefaction_dwdp,
                               np.array([self.q_l.rho, self.q_l.v, self.q_l.eps]),
                               p, rtol = 1e-12, atol = 1e-10,
                               args=(self.q_l,self.wavenumber))
            else:
                p = np.linspace(self.q_r.p, self.p_0_star)
                w_all = odeint(rarefaction_dwdp,
                               np.array([self.q_r.rho, self.q_r.v, self.q_r.eps]),
                               p, rtol = 1e-12, atol = 1e-10,
                               args=(self.q_r,self.wavenumber))
                p = p[-1::-1]
                w_all = w_all[-1::-1,:]
            data = np.zeros((len(p)+1,8))
            xi = np.zeros(len(p)+1)
            if self.wavenumber == 0:
                for i in range(len(p)):
                    state = State(w_all[i,0], w_all[i,1],
                                  self.q_l.vt_from_known(w_all[i,0], w_all[i,1], w_all[i,2]),
                                  w_all[i, 2], self.q_l.eos)
                    xi[i] = state.wavespeed(self.wavenumber)
                    data[i,:] = state.state()
                xi[-1] = self.wave_speed[-1]
                data[-1,:] = self.q_r.state()
            else:
                xi[0] = self.wave_speed[0]
                data[0,:] = self.q_l.state()
                for i in range(len(p)):
                    state = State(w_all[i,0], w_all[i,1],
                                  self.q_r.vt_from_known(w_all[i,0], w_all[i,1], w_all[i,2]),
                                  w_all[i, 2], self.q_r.eos)
                    xi[i+1] = state.wavespeed(self.wavenumber)
                    data[i+1,:] = state.state()

        return xi, data

    def latex_string(self):
        s = self.name
        s += r": \lambda^{{({})}}".format(self.wavenumber)
        if self.type == "Rarefaction" or self.type == "Deflagration":
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

        # Cache for plot
        self._png_data = None
        self._svg_data = None

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

    def _figure_data(self, format):
        fig, axs = pyplot.subplots(3,3)
        ax = axs[0,0]
        for w in self.waves[0], self.waves[2]:
            if w.type == 'Rarefaction':
                xi_end = np.linspace(w.wave_speed[0], w.wave_speed[1], 5)
                ax.fill_between([0, xi_end[0], xi_end[-1], 0],
                                [0, 1, 1, 0], color='k', alpha=0.1)
                for xi in xi_end:
                    ax.plot([0, xi], [0, 1], 'k-', linewidth=1)
            else:
                ax.plot([0, w.wave_speed[0]], [0, 1], 'k-', linewidth=3)
        ax.plot([0, self.waves[1].wave_speed[0]], [0, 1], 'k--', linewidth=1)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$t$")
        ax.set_title("Characteristics")
        names = [r"$\rho$", r"$v$", r"$v_t$", r"$\epsilon$", r"$p$", r"$W$",
                 r"$h$", r"$c_s$"]
        xi = [-1.05]
        data = self.state_l.state()
        for wave in self.waves:
            xi_wave, data_wave = wave.plotting_data()
            xi = np.hstack((xi, xi_wave))
            data = np.vstack((data, data_wave))
        xi = np.hstack((xi, [1.05]))
        data = np.vstack((data, self.state_r.state()))
        for ax_j in range(3):
            for ax_i in range(3):
                if ax_i == 0 and ax_j == 0:
                    continue
                nvar = ax_i*3 + ax_j - 1
                axs[ax_i, ax_j].plot(xi, data[:, nvar])
                var_max = np.max(data[:, nvar])
                var_min = np.min(data[:, nvar])
                d_var = max(var_max - var_min,
                            0.01 * min(abs(var_min), abs(var_max)), 0.01)
                axs[ax_i, ax_j].set_xlim(-1.05, 1.05)
                axs[ax_i, ax_j].set_ylim(var_min - 0.05 * d_var,
                                         var_max + 0.05 * d_var)
                axs[ax_i, ax_j].set_xlabel(r"$\xi$")
                axs[ax_i, ax_j].set_ylabel(names[nvar])
        fig.tight_layout()
        data = print_figure(fig, format)
        pyplot.close(fig)
        return data

    def _repr_png_(self):
        if self._png_data is None:
            self._png_data = self._figure_data('png')
        return self._png_data

    def _repr_svg_(self):
        if self._svg_data is None:
            self._svg_data = self._figure_data('svg')
        return self._svg_data

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

#    eos = eos_defns.eos_gamma_law(5.0/3.0)
#    w_left = State(1.0, 0.0, 0.0, 1.5, eos, label="L")
#    w_right = State(0.125, 0.0, 0.0, 1.2, eos, label="R")
#    w_left = State(1.0, 0.0, 0.9, 0.015, eos, label="L")
#    w_right = State(1.0, 0.0, 0.9, 1500, eos, label="R")
#    w_left = State(10.0, 0.0, 0.0, 2.0, eos, label="L")
#    w_right = State(1.0, 0.0, 0.0, 1.5e-6, eos, label="R")
#    eos1 = eos_defns.eos_gamma_law(1.4)
#    eos2 = eos_defns.eos_gamma_law(1.67)
#    w_left = State(10.2384, 0.9411, 0.0, 50.0/0.4/10.23841, eos1, label="L")
#    w_right = State(0.1379, 0.0, 0.0, 1.0/0.1379/0.67, eos2, label="R")
#    rp = RP(w_left, w_right)
#    print(rp.p_star)

    q_burnt = 0.0
    q_unburnt = 0.1
    gamma = 5/3
    Cv = 1.0
    t_i = 2
    eos_burnt = eos_defns.eos_gamma_law_react(gamma, q_burnt, Cv)
    eos_unburnt = eos_defns.eos_gamma_law_react(gamma, q_unburnt, Cv)
    q_left = State(1, 0, 0, 2, eos_burnt)
    q_right = State(1, 0, 0, 2, eos_unburnt)
    w_right = Wave(q_right, 0.001*q_right.p, 2, eos_burnt, t_i)
