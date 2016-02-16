# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:28:53 2016

@author: ih3
"""

import numpy
from scipy.optimize import brentq
from scipy.integrate import odeint
from copy import deepcopy
from .state import State

def rarefaction_dwdp(w, p, q_known, wavenumber):
    r"""
    There is a tricky point here that needs investigation. If
    the input p is used here, rather than local_state.p, then they
    can diverge (when :math:`v_t` is significant) leading to overflows of g. By
    using local_state we avoid the overflow, but it may mean the final
    state is not very accurate.
    """
    lr_sign = wavenumber - 1
    dwdp = numpy.zeros_like(w)
    rho, v, eps = w
    vt = q_known.vt_from_known(rho, v, eps)
    local_state = State(rho, v, vt, eps, q_known.eos)
    cs = local_state.cs
    h = local_state.h
    W_lorentz = local_state.W_lorentz
    xi = local_state.wavespeed(wavenumber)
    # g quantifies the effect of tangential velocities: see the Living Review
    # and original Pons et al paper for details.
    g = vt**2 * (xi**2 - 1.0) / (1.0 - xi * v)**2
    dwdp[0] = 1.0 / (h * cs**2)
    dwdp[1] = lr_sign / (rho * h * W_lorentz**2 * cs) / numpy.sqrt(1.0 + g)
    dwdp[2] = local_state.p / (rho**2 * h * cs**2)
    return dwdp
    
def mass_flux_squared(q_start, p_end, unknown_eos=None):

    if unknown_eos is None:
        unknown_eos = q_start.eos

    def shock_root_rho(rho):
        h = unknown_eos['h_from_rho_p'](rho, p_end)
        return (h**2 - q_start.h**2) - \
        (h/rho + q_start.h/q_start.rho) * (p_end - q_start.p)

    if p_end >= q_start.p:
        # Shock
        min_rho = q_start.rho
        shock_root_min = shock_root_rho(min_rho)
        max_rho = numpy.sqrt(p_end/q_start.p) * q_start.rho
        shock_root_max = shock_root_rho(max_rho)
        while(shock_root_min * shock_root_max > 0.0):
            min_rho /= 1.1 # Not sure - could end up with unphysical root?
            max_rho *= 10.0
            shock_root_min = shock_root_rho(min_rho)
            shock_root_max = shock_root_rho(max_rho)
    else:
        # Deflagration
        max_rho = q_start.rho
        shock_root_max = shock_root_rho(max_rho)
        min_rho = numpy.sqrt(p_end/q_start.p) * q_start.rho
        shock_root_min = shock_root_rho(min_rho)
        while(shock_root_min * shock_root_max > 0.0):
            min_rho /= 10.0 # Not sure - could end up with unphysical root?
            max_rho *= 1.1
            shock_root_min = shock_root_rho(min_rho)
            shock_root_max = shock_root_rho(max_rho)
    rho = brentq(shock_root_rho, min_rho, max_rho)
    h = unknown_eos['h_from_rho_p'](rho, p_end)
    eps = h - 1.0 - p_end / rho
    dp = p_end - q_start.p
    dh2 = h**2 - q_start.h**2
    j2 = -dp / (dh2 / dp - 2.0 * q_start.h / q_start.rho)

    return j2, rho, eps, dp

def deflagration_root(p_0_star, q_precursor, unknown_eos, wavenumber, label):
    lr_sign = wavenumber - 1    
    j2, rho, eps, dp = mass_flux_squared(q_precursor, p_0_star, unknown_eos)
    j = numpy.sqrt(j2)
    v_deflagration = (q_precursor.rho**2 *
        q_precursor.W_lorentz**2 * q_precursor.v + \
        lr_sign * j**2 * \
        numpy.sqrt(1.0 + q_precursor.rho**2 *
        q_precursor.W_lorentz**2 *
        (1.0 - q_precursor.v**2) / j**2)) / \
        (q_precursor.rho**2 * q_precursor.W_lorentz**2 + j**2)
    W_lorentz_deflagration = 1.0 / numpy.sqrt(1.0 - v_deflagration**2)
    v = (q_precursor.h * q_precursor.W_lorentz *
         q_precursor.v + lr_sign * dp *
         W_lorentz_deflagration / j) / \
        (q_precursor.h * q_precursor.W_lorentz + dp * (1.0 /
         q_precursor.rho / q_precursor.W_lorentz + \
         lr_sign * q_precursor.v *
         W_lorentz_deflagration / j))
    vt = q_precursor.vt_from_known(rho, v, eps)
    q_unknown = State(rho, v, vt, eps, unknown_eos, label)

    return q_unknown.wavespeed(wavenumber) - v_deflagration

def precursor_root(p_0_star, q_known, t_i, wavenumber):
    shock = Shock(q_known, p_0_star, wavenumber)
    if wavenumber == 0:
        q_precursor = shock.q_r
    else:
        q_precursor = shock.q_l
    t_precursor = q_precursor.eos['t_from_rho_eps'](
                    q_precursor.rho, q_precursor.eps)
    return t_precursor - t_i

class WaveSection(object):
    
    def __init__(self, q_start, p_end, wavenumber):
        """
        A part of a wave. For a single shock or rarefaction, this will be the
        complete wave. For a deflagration or detonation, it may form part of
        the full wave.
        """
        self.trivial = False
        assert(wavenumber in [0, 1, 2]), "wavenumber must be 0, 1, 2"
        self.wavenumber = wavenumber
        
    def latex_string(self):
        if self.trivial:
            return ""
        else:
            s = self.name
            s += r": \lambda^{{({})}}".format(self.wavenumber)
            if len(self.wavespeed) > 1:
                s += r"\in [{:.4f}, {:.4f}]".format(self.wavespeed[0],
                             self.wavespeed[-1])
            else:
                s += r"= {:.4f}".format(self.wavespeed[0])
            return s

    def _repr_latex_(self):
        s = r"$" + self.latex_string() + r"$"
        return s

class Contact(WaveSection):
    
    def __init__(self, q_start, q_end, wavenumber):
        """
        A contact.
        """
        
        self.trivial = False
        assert(wavenumber in [1]), "wavenumber for a Contact must be 1"
        self.type = "Contact"
        self.wavenumber = wavenumber
        self.q_start = deepcopy(q_start)
        self.q_end = deepcopy(q_end)

        self.name = r"{\cal C}"
        
        self.wavespeed = [q_start.v]
        
        assert(numpy.allclose(q_start.v, q_end.v)), "Velocities of states "\
        "must match for a contact"
        assert(numpy.allclose(q_start.p, q_end.p)), "Pressures of states "\
        "must match for a contact"
        assert(numpy.allclose(q_start.wavespeed(wavenumber),
                              q_end.wavespeed(wavenumber))), "Wavespeeds of "\
        "states must match for a contact"
        
    def plotting_data(self):
        
        data = numpy.vstack((self.q_start.state(), self.q_end.state()))
        xi = numpy.array([self.q_start.wavespeed(self.wavenumber),
                       self.q_start.wavespeed(self.wavenumber)])
        
        return xi, data
        
class Rarefaction(WaveSection):
    
    def __init__(self, q_start, p_end, wavenumber):
        """
        A rarefaction.
        """
        
        self.trivial = False
        assert(wavenumber in [0, 2]), "wavenumber for a Rarefaction "\
        "must be in 0, 2"
        assert(q_start.p >= p_end), "For a rarefaction, p_start >= p_end"
        self.type = "Rarefaction"
        self.wavenumber = wavenumber
        self.q_start = deepcopy(q_start)

        self.name = r"{\cal R}"
        if self.wavenumber == 0:
            label = r"\star_L"
            self.name += r"_{\leftarrow}"
        else:
            label = r"\star_R"
            self.name += r"_{\rightarrow}"

        v_known = q_start.wavespeed(self.wavenumber)

        if numpy.allclose(q_start.p, p_end):
            self.trivial = True
            self.q_end = State(q_start.rho, q_start.v, q_start.vt, q_start.eps,
            q_start.eos, label)
            v_unknown = v_known
        else:
            w_all = odeint(rarefaction_dwdp,
                           numpy.array([q_start.rho, q_start.v, q_start.eps]),
                           [q_start.p, p_end], rtol = 1e-12, atol = 1e-10,
                           args=((q_start, self.wavenumber)))
            self.q_end = State(w_all[-1, 0], w_all[-1, 1],
                              q_start.vt_from_known(w_all[-1, 0], w_all[-1, 1], w_all[-1, 2]),
                              w_all[-1, 2], q_start.eos, label)
            v_unknown = self.q_end.wavespeed(self.wavenumber)

        self.wavespeed = []
        if self.wavenumber == 0:
            self.wavespeed = numpy.array([v_known, v_unknown])
        else:
            self.wavespeed = numpy.array([v_unknown, v_known])
    
    def plotting_data(self):
        p = numpy.linspace(self.q_start.p, self.q_end.p)
        w_all = odeint(rarefaction_dwdp,
                       numpy.array([self.q_start.rho, self.q_start.v, self.q_start.eps]),
                           p, rtol = 1e-12, atol = 1e-10,
                           args=(self.q_start, self.wavenumber))
        data = numpy.zeros((len(p),8))
        xi = numpy.zeros_like(p)
        for i in range(len(p)):
            state = State(w_all[i,0], w_all[i,1],
                          self.q_start.vt_from_known(w_all[i,0], w_all[i,1], w_all[i,2]),
                          w_all[i, 2], self.q_start.eos)
            xi[i] = state.wavespeed(self.wavenumber)
            data[i,:] = state.state()
        
        return xi, data

class Shock(WaveSection):
    
    def __init__(self, q_start, p_end, wavenumber):
        """
        A shock.
        """

        self.trivial = False
        assert(wavenumber in [0, 2]), "wavenumber for a Shock "\
        "must be in 0, 2"
        assert(q_start.p <= p_end), "For a shock, p_start <= p_end"
        self.type = "Shock"
        self.wavenumber = wavenumber
        lr_sign = self.wavenumber - 1
        self.q_start = deepcopy(q_start)

        self.name = r"{\cal S}"
        if self.wavenumber == 0:
            label = r"\star_L"
            self.name += r"_{\leftarrow}"
        else:
            label = r"\star_R"
            self.name += r"_{\rightarrow}"

        if numpy.allclose(q_start.p, p_end):
            self.trivial = True
            self.q_end = State(q_start.rho, q_start.v, q_start.vt, q_start.eps,
            q_start.eos, label)
            v_shock = q_start.wavespeed(self.wavenumber)
        else:
            j2, rho, eps, dp = mass_flux_squared(q_start, p_end, 
                                                      q_start.eos)
            j = numpy.sqrt(j2)
            v_shock = (q_start.rho**2 * q_start.W_lorentz**2 * q_start.v + \
                lr_sign * j**2 * \
                numpy.sqrt(1.0 + q_start.rho**2 * q_start.W_lorentz**2 * (1.0 - q_start.v**2) / j**2)) / \
                (q_start.rho**2 * q_start.W_lorentz**2 + j**2)
            W_lorentz_shock = 1.0 / numpy.sqrt(1.0 - v_shock**2)
            v = (q_start.h * q_start.W_lorentz * q_start.v + lr_sign * dp * W_lorentz_shock / j) / \
                (q_start.h * q_start.W_lorentz + dp * (1.0 / q_start.rho / q_start.W_lorentz + \
                lr_sign * q_start.v * W_lorentz_shock / j))
            vt = q_start.vt_from_known(rho, v, eps)
            self.q_end = State(rho, v, vt, eps, q_start.eos, label)

        self.wavespeed = numpy.array([v_shock, v_shock])
        
    def plotting_data(self):
        
        data = numpy.vstack((self.q_start.state(), self.q_end.state()))
        xi = numpy.array([self.q_start.wavespeed(self.wavenumber),
                       self.q_start.wavespeed(self.wavenumber)])
        
        return xi, data

class Deflagration(WaveSection):
    
    def __init__(self, q_start, p_end, wavenumber, eos_end, t_i):
        """
        A deflagration.
        """
        
        self.trivial = False
        assert(wavenumber in [0, 2]), "wavenumber for a Deflagration "\
        "must be in 0, 2"
        assert(q_start.p >= p_end), "For a deflagration, p_start >= p_end"
        t_start = q_start.eos['t_from_rho_eps'](q_start.rho, q_start.eps)
        assert(t_start >= t_i), "For a deflagration, temperature of start "\
        "state must be at least the ignition temperature"
        self.type = "Deflagration"
        self.wavenumber = wavenumber
        lr_sign = self.wavenumber - 1
        self.q_start = deepcopy(q_start)

        self.name = r"{\cal WDF}"
        if self.wavenumber == 0:
            label = r"\star_L"
            self.name += r"_{\leftarrow}"
        else:
            label = r"\star_R"
            self.name += r"_{\rightarrow}"

        v_known = q_start.wavespeed(self.wavenumber)

        if numpy.allclose(q_start.p, p_end):
            self.trivial = True
            self.q_end = State(q_start.rho, q_start.v, q_start.vt, q_start.eps,
            eos_end, label)
            v_deflagration = v_known
        else:
            # This is a single deflagration, so the start state must be at the
            # reaction temperature already.
            j2, rho, eps, dp = mass_flux_squared(q_start, p_end, eos_end)
            j = numpy.sqrt(j2)
            v_deflagration = (q_start.rho**2 *
                q_start.W_lorentz**2 * q_start.v + \
                lr_sign * j**2 * \
                numpy.sqrt(1.0 + q_start.rho**2 *
                q_start.W_lorentz**2 *
                (1.0 - q_start.v**2) / j**2)) / \
                (q_start.rho**2 * q_start.W_lorentz**2 + j**2)
            W_lorentz_deflagration = 1.0 / numpy.sqrt(1.0 - v_deflagration**2)
            v = (q_start.h * q_start.W_lorentz * q_start.v +
                 lr_sign * dp * W_lorentz_deflagration / j) / \
                 (q_start.h * q_start.W_lorentz + dp * (1.0 /
                  q_start.rho / q_start.W_lorentz + \
                 lr_sign * q_start.v * W_lorentz_deflagration / j))
            vt = q_start.vt_from_known(rho, v, eps)
            q_unknown = State(rho, v, vt, eps, eos_end, label)

            # If the speed in the unknown state means the characteristics are
            # not going into the deflagration, then this is an unstable strong
            # deflagration
            if (lr_sign*(q_unknown.wavespeed(self.wavenumber) - v_deflagration) < 0):
                p_cjdf = brentq(deflagration_root, (1.0+1e-9)*p_end,
                                (1.0-1e-9)*q_start.p,
                                args=(q_start, eos_end, self.wavenumber, t_i))
                j2, rho, eps, dp = self.mass_flux_squared(q_start, p_cjdf, eos_end)
                j = numpy.sqrt(j2)
                v_deflagration = (q_start.rho**2 *
                    q_start.W_lorentz**2 * q_start.v + \
                    lr_sign * j**2 * \
                    numpy.sqrt(1.0 + q_start.rho**2 *
                    q_start.W_lorentz**2 *
                    (1.0 - q_start.v**2) / j**2)) / \
                    (q_start.rho**2 * q_start.W_lorentz**2 + j**2)
                W_lorentz_deflagration = 1.0 / numpy.sqrt(1.0 - v_deflagration**2)
                v = (q_start.h * q_start.W_lorentz *
                    q_start.v + lr_sign * dp * W_lorentz_deflagration / j) / \
                    (q_start.h * q_start.W_lorentz + dp * (1.0 /
                     q_start.rho / q_start.W_lorentz + \
                     lr_sign * q_start.v * W_lorentz_deflagration / j))
                vt = q_start.vt_from_known(rho, v, eps)
                q_unknown = State(rho, v, vt, eps, eos_end, label)
                self.name = r"{\cal CJDF}"
                if self.wavenumber == 0:
                    label = r"\star_L"
                    self.name += r"_{\leftarrow}"
                else:
                    label = r"\star_R"
                    self.name += r"_{\rightarrow}"

            self.q_end = deepcopy(q_unknown)


        self.wavespeed = numpy.array([v_deflagration, v_deflagration])
    
    def plotting_data(self):
        
        data = numpy.vstack((self.q_start.state(), self.q_end.state()))
        xi = numpy.array([self.q_start.wavespeed(self.wavenumber),
                       self.q_start.wavespeed(self.wavenumber)])
        
        return xi, data

class Detonation(WaveSection):
    
    def __init__(self, q_start, p_end, wavenumber, eos_end, t_i):
        """
        A detonation.
        """
        
        self.trivial = False
        assert(wavenumber in [0, 2]), "wavenumber for a Detonation "\
        "must be in 0, 2"
        assert(q_start.p <= p_end), "For a detonation, p_start <= p_end"
        t_start = q_start.eos['t_from_rho_eps'](q_start.rho, q_start.eps)
        assert(t_start >= t_i), "For a detonation, temperature of start "\
        "state must be at least the ignition temperature"
        self.type = "Detonation"
        self.wavenumber = wavenumber
        lr_sign = self.wavenumber - 1
        self.q_start = deepcopy(q_start)

        self.name = r"{\cal SDT}"
        if self.wavenumber == 0:
            label = r"\star_L"
            self.name += r"_{\leftarrow}"
        else:
            label = r"\star_R"
            self.name += r"_{\rightarrow}"

        v_known = q_start.wavespeed(self.wavenumber)

        if numpy.allclose(q_start.p, p_end):
            self.trivial = True
            self.q_end = State(q_start.rho, q_start.v, q_start.vt, q_start.eps,
            eos_end, label)
            v_detonation = v_known
        else:
            # This is a single detonation, so the start state must be at the
            # reaction temperature already.
            j2, rho, eps, dp = mass_flux_squared(q_start, p_end, eos_end)
            j = numpy.sqrt(j2)
            v_detonation = (q_start.rho**2 *
                q_start.W_lorentz**2 * q_start.v + \
                lr_sign * j**2 * \
                numpy.sqrt(1.0 + q_start.rho**2 *
                q_start.W_lorentz**2 *
                (1.0 - q_start.v**2) / j**2)) / \
                (q_start.rho**2 * q_start.W_lorentz**2 + j**2)
            W_lorentz_detonation = 1.0 / numpy.sqrt(1.0 - v_detonation**2)
            v = (q_start.h * q_start.W_lorentz * q_start.v +
                 lr_sign * dp * W_lorentz_detonation / j) / \
                 (q_start.h * q_start.W_lorentz + dp * (1.0 /
                  q_start.rho / q_start.W_lorentz + \
                 lr_sign * q_start.v * W_lorentz_detonation / j))
            vt = q_start.vt_from_known(rho, v, eps)
            q_unknown = State(rho, v, vt, eps, eos_end, label)

            # If the speed in the unknown state means the characteristics are
            # not going into the detonation, then this is an unstable weak
            # detonation
            if (lr_sign*(q_unknown.wavespeed(self.wavenumber) - v_detonation) < 0):
                # NOTE: if this truly is the same function, then this whole
                # set of code should be refactored, as it's essentially the
                # same for deflagration and detonation.
                p_cjdt = brentq(deflagration_root, (1.0+1e-9)*p_end,
                                (1.0-1e-9)*q_start.p,
                                args=(q_start, eos_end, self.wavenumber, t_i))
                j2, rho, eps, dp = self.mass_flux_squared(q_start, p_cjdt, eos_end)
                j = numpy.sqrt(j2)
                v_detonation = (q_start.rho**2 *
                    q_start.W_lorentz**2 * q_start.v + \
                    lr_sign * j**2 * \
                    numpy.sqrt(1.0 + q_start.rho**2 *
                    q_start.W_lorentz**2 *
                    (1.0 - q_start.v**2) / j**2)) / \
                    (q_start.rho**2 * q_start.W_lorentz**2 + j**2)
                W_lorentz_detonation = 1.0 / numpy.sqrt(1.0 - v_detonation**2)
                v = (q_start.h * q_start.W_lorentz *
                    q_start.v + lr_sign * dp * W_lorentz_detonation / j) / \
                    (q_start.h * q_start.W_lorentz + dp * (1.0 /
                     q_start.rho / q_start.W_lorentz + \
                     lr_sign * q_start.v * W_lorentz_detonation / j))
                vt = q_start.vt_from_known(rho, v, eps)
                q_unknown = State(rho, v, vt, eps, eos_end, label)
                self.name = r"{\cal CJDT}"
                if self.wavenumber == 0:
                    label = r"\star_L"
                    self.name += r"_{\leftarrow}"
                else:
                    label = r"\star_R"
                    self.name += r"_{\rightarrow}"

            self.q_end = deepcopy(q_unknown)


        self.wavespeed = numpy.array([v_detonation, v_detonation])
    
    def plotting_data(self):
        
        data = numpy.vstack((self.q_start.state(), self.q_end.state()))
        xi = numpy.array([self.q_start.wavespeed(self.wavenumber),
                       self.q_start.wavespeed(self.wavenumber)])
        
        return xi, data

def build_inert_wave_section(q_known, unknown_value, wavenumber):
    """
    Object factory for the WaveSection; non-reactive case
    """
    
    if wavenumber == 1:
        return Contact(q_known, unknown_value, wavenumber)
    elif q_known.p < unknown_value:
        return Shock(q_known, unknown_value, wavenumber)
    else:
        return Rarefaction(q_known, unknown_value, wavenumber)
        
def build_reactive_wave_section(q_known, unknown_value, wavenumber, 
                                unknown_eos, t_i):
    """
    Object factory for the WaveSection; reactive case
    """
    
    if wavenumber == 1:
        return Contact(q_known, unknown_value, wavenumber)
    else:
        wavesections = []
        if q_known.p < unknown_value:
            # The detonation wave
            detonation = Detonation(q_known, unknown_value, wavenumber,
                                        unknown_eos, t_i)
            wavesections.append(detonation)
            q_next = deepcopy(detonation.q_end)
            # Finally, was it a CJ detonation?
            if q_next.p > unknown_value:
                rarefaction = Rarefaction(q_next, unknown_value, wavenumber)
                wavesections.append(rarefaction)
        else:
            t_known = q_known.eos['t_from_rho_eps'](q_known.rho, q_known.eps)
            if t_known < t_i: # Need a precursor shock
                p_min = unknown_value
                p_max = q_known.p
                t_min = precursor_root(p_min, q_known, t_i, wavenumber)
                t_max = precursor_root(p_max, q_known, t_i, wavenumber)
                assert(t_min < 0)
    
                if t_max <= 0:
                    p_max *= 2
                    t_max = precursor_root(p_max)
    
                p_0_star = brentq(precursor_root, p_min, p_max,
                                  args=(q_known, t_i, wavenumber))
                precursor_shock = Shock(q_known, p_0_star, wavenumber)
                wavesections.append(precursor_shock)
                q_next = precursor_shock.q_end
            else: # No precursor shock
                q_next = deepcopy(q_known)
            # Next, the deflagration wave
            deflagration = Deflagration(q_next, unknown_value, wavenumber,
                                        unknown_eos, t_i)
            wavesections.append(deflagration)
            q_next = deepcopy(deflagration.q_end)
            # Finally, was it a CJ deflagration?
            if q_next.p > unknown_value:
                rarefaction = Rarefaction(q_next, unknown_value, wavenumber)
                wavesections.append(rarefaction)

        return wavesections
        

class Wave(object):  
    
    def __init__(self, q_known, unknown_value, wavenumber, unknown_eos=None,
                 t_i=None):
        """
        A wave.
        
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
            temperature at which the state starts to react
        """
        
        self.wavenumber = wavenumber
        self.wave_sections = []
        
        if q_known.q is None:
            waves = build_inert_wave_section(q_known, unknown_value, 
                                             wavenumber)
            self.wave_sections.append(waves)
        else:
            waves = build_reactive_wave_section(q_known, unknown_value,
                                                wavenumber, unknown_eos, t_i)
            self.wave_sections.append(waves)

        if wavenumber == 0:
            self.q_l = deepcopy(q_known)
            if self.wave_sections:
                self.q_r = self.wave_sections[-1].q_end
            else:
                self.q_r = deepcopy(self.q_l)
        elif wavenumber == 1:
            self.q_l = deepcopy(q_known)
            self.q_r = deepcopy(q_known)
        else:
            self.q_r = deepcopy(q_known)
            if self.wave_sections:
                self.q_l = self.wave_sections[-1].q_end
            else:
                self.q_l = deepcopy(self.q_r)
    
    def plotting_data(self):
        
        xi_wave = numpy.zeros((0,))
        data_wave = numpy.zeros((8,0))
        for wavesection in self.wave_sections:
            xi_section, data_section = wavesection.plotting_data()
            xi_wave = numpy.hstack((xi_wave, xi_section))
            data_wave = numpy.vstack((data_wave, data_section))
        
        return xi_wave, data_wave
    
    def latex_string(self):
        if len(self.wave_sections)==0:
            return ""
        elif len(self.wave_sections)==1:
            s = self.wave_sections[0].latex_string()
        else:
            s = r"\left("
            for wavesection in self.wave_sections:
                s += self.name
            s += r"\right) "
            s += r": \lambda^{{({})}}".format(self.wavenumber)
            s += r"\in [{:.4f}, {:.4f}]".format(self.wave_sections[0].wavespeed[0],
                             self.wave_sections[-1].wavespeed[-1])
        return s
        
    def _repr_latex_(self):
        s = r"$" + self.latex_string() + r"$"
        return s


#class Wave(object):
#
#    def __init__(self, q_known, unknown_value, wavenumber, unknown_eos=None,
#                 t_i=None):
#        """
#        Initialize a wave.
#
#        There are two possibilities: the wave is linear (wavenumber = 1),
#        which is a contact, where the known value is the left state and the
#        unknown value the right state.
#
#        The second possibility is that the wave is nonlinear (wavenumber = 0,2)
#        which is either a shock or a rarefaction, where the known value is the
#        left/right state (wavenumber = 0,2 respectively) and the unknown value
#        the pressure in the star state.
#        """
#        # The wave is trivial if no quantites jump across it
#        # This should be used to simplify output, especially for 
#        # Riemann Problems with eg only a single shock.
#        self.trivial = False 
#        assert(wavenumber in [0, 1, 2]), "wavenumber must be 0, 1, 2"
#        self.wavenumber = wavenumber
#        if self.wavenumber == 1:
#            self.type = "Contact"
#            assert(isinstance(unknown_value, State)), "unknown_value must " \
#            "be a State when wavenumber is 1"
#            self.q_l = q_known
#            self.q_r = unknown_value
#            assert(numpy.allclose(self.q_l.v, self.q_r.v)), "For a contact, "\
#            "wavespeeds must match across the wave"
#            assert(numpy.allclose(self.q_l.p, self.q_r.p)), "For a contact, "\
#            "pressure must match across the wave"
#            if numpy.allclose(self.q_l.state(), self.q_r.state()):
#                self.trivial = True
#            self.wave_speed = numpy.array([self.q_l.v, self.q_r.v])
#            self.name = r"{\cal C}"
#        elif self.wavenumber == 0:
#            self.q_l = deepcopy(q_known)
#            if (self.q_l.p < unknown_value):
#                if unknown_eos is not None:
#                    assert(t_i is not None)
#                    self.solve_detonation(q_known, unknown_value, unknown_eos, t_i)
#                else:
#                    self.solve_shock(q_known, unknown_value)
#            else:
#                if unknown_eos is not None:
#                    assert(t_i is not None)
#                    self.solve_deflagration(q_known, unknown_value, unknown_eos, t_i)
#                else:
#                    self.solve_rarefaction(q_known, unknown_value)
#        else:
#            self.q_r = deepcopy(q_known)
#            if (self.q_r.p < unknown_value):
#                if unknown_eos is not None:
#                    assert(t_i is not None)
#                    self.solve_detonation(q_known, unknown_value, unknown_eos, t_i)
#                else:
#                    self.solve_shock(q_known, unknown_value)
#            else:
#                if unknown_eos is not None:
#                    assert(t_i is not None)
#                    self.solve_deflagration(q_known, unknown_value, unknown_eos, t_i)
#                else:
#                    self.solve_rarefaction(q_known, unknown_value)
#
#    def mass_flux_squared(self, q_known, p_star, unknown_eos=None):
#
#        if unknown_eos is None:
#            unknown_eos = q_known.eos
#
#        def shock_root_rho(rho):
#            h = unknown_eos['h_from_rho_p'](rho, p_star)
#            return (h**2 - q_known.h**2) - \
#            (h/rho + q_known.h/q_known.rho) * (p_star - q_known.p)
#
#        if p_star >= q_known.p:
#            # Shock
#            min_rho = q_known.rho
#            shock_root_min = shock_root_rho(min_rho)
#            max_rho = numpy.sqrt(p_star/q_known.p) * q_known.rho
#            shock_root_max = shock_root_rho(max_rho)
#            while(shock_root_min * shock_root_max > 0.0):
#                min_rho /= 1.1 # Not sure - could end up with unphysical root?
#                max_rho *= 10.0
#                shock_root_min = shock_root_rho(min_rho)
#                shock_root_max = shock_root_rho(max_rho)
#        else:
#            # Deflagration
#            max_rho = q_known.rho
#            shock_root_max = shock_root_rho(max_rho)
#            min_rho = numpy.sqrt(p_star/q_known.p) * q_known.rho
#            shock_root_min = shock_root_rho(min_rho)
#            while(shock_root_min * shock_root_max > 0.0):
#                min_rho /= 10.0 # Not sure - could end up with unphysical root?
#                max_rho *= 1.1
#                shock_root_min = shock_root_rho(min_rho)
#                shock_root_max = shock_root_rho(max_rho)
#        rho = brentq(shock_root_rho, min_rho, max_rho)
#        h = unknown_eos['h_from_rho_p'](rho, p_star)
#        eps = h - 1.0 - p_star / rho
#        dp = p_star - q_known.p
#        dh2 = h**2 - q_known.h**2
#        j2 = -dp / (dh2 / dp - 2.0 * q_known.h / q_known.rho)
#
#        return j2, rho, eps, dp
#
#    def solve_shock(self, q_known, p_star, unknown_eos=None):
#        r"""
#        In the case of a shock, finds the unknown state on one side of the wave and wavespeed given the known state on the wave's other side and :math:`p_*`.
#
#        Parameters
#        ----------
#
#        self : Wave
#            The wave, which has a known state on one side and an unknown
#            state on the other side.
#        q_known : State
#            The known state on one side of the wave
#        p_star : scalar
#            Pressure in the region of unknown state
#        unknown_eos : dictionary
#            Equation of state in the unknown region
#        """
#
#        self.type = "Shock"
#        lr_sign = self.wavenumber - 1
#        if unknown_eos is None:
#            unknown_eos = q_known.eos
#
#        self.name = r"{\cal S}"
#        if self.wavenumber == 0:
#            label = r"\star_L"
#            self.name += r"_{\leftarrow}"
#        else:
#            label = r"\star_R"
#            self.name += r"_{\rightarrow}"
#
#        if numpy.allclose(q_known.p, p_star):
#            self.trivial = True
#            q_unknown = State(q_known.rho, q_known.v, q_known.vt, q_known.eps,
#            q_known.eos, label)
#            v_shock = q_known.wavespeed(self.wavenumber)
#        else:
#            j2, rho, eps, dp = self.mass_flux_squared(q_known, p_star, unknown_eos)
#            j = numpy.sqrt(j2)
#            v_shock = (q_known.rho**2 * q_known.W_lorentz**2 * q_known.v + \
#                lr_sign * j**2 * \
#                numpy.sqrt(1.0 + q_known.rho**2 * q_known.W_lorentz**2 * (1.0 - q_known.v**2) / j**2)) / \
#                (q_known.rho**2 * q_known.W_lorentz**2 + j**2)
#            W_lorentz_shock = 1.0 / numpy.sqrt(1.0 - v_shock**2)
#            v = (q_known.h * q_known.W_lorentz * q_known.v + lr_sign * dp * W_lorentz_shock / j) / \
#                (q_known.h * q_known.W_lorentz + dp * (1.0 / q_known.rho / q_known.W_lorentz + \
#                lr_sign * q_known.v * W_lorentz_shock / j))
#            vt = q_known.vt_from_known(rho, v, eps)
#            q_unknown = State(rho, v, vt, eps, unknown_eos, label)
#
#        if self.wavenumber == 0:
#            self.q_r = deepcopy(q_unknown)
#        else:
#            self.q_l = deepcopy(q_unknown)
#
#        self.wave_speed = numpy.array([v_shock, v_shock])
#
#    def solve_rarefaction(self, q_known, p_star, unknown_eos=None):
#        r"""
#        In the case of a rarefaction, finds the unknown state on one side of the wave and wavespeed given the known state on the wave's other side and :math:`p_*`.
#
#        Parameters
#        ----------
#
#        self : Wave
#            The wave, which has a known state on one side and an unknown
#            state on the other side.
#        q_known : State
#            The known state on one side of the wave
#        p_star : scalar
#            Pressure in the region of unknown state
#        unknown_eos : dictionary
#            Equation of state in the unknown region
#        """
#
#        self.type = "Rarefaction"
#        if unknown_eos is None:
#            unknown_eos = q_known.eos
#
#        self.name = r"{\cal R}"
#        if self.wavenumber == 0:
#            label = r"\star_L"
#            self.name += r"_{\leftarrow}"
#        else:
#            label = r"\star_R"
#            self.name += r"_{\rightarrow}"
#
#        v_known = q_known.wavespeed(self.wavenumber)
#
#        if numpy.allclose(q_known.p, p_star):
#            self.trivial = True
#            q_unknown = State(q_known.rho, q_known.v, q_known.vt, q_known.eps,
#            q_known.eos, label)
#            v_unknown = v_known
#        else:
#            w_all = odeint(rarefaction_dwdp,
#                           numpy.array([q_known.rho, q_known.v, q_known.eps]),
#                           [q_known.p, p_star], rtol = 1e-12, atol = 1e-10,
#                           args=((q_known, self.wavenumber)))
#            q_unknown = State(w_all[-1, 0], w_all[-1, 1],
#                              q_known.vt_from_known(w_all[-1, 0], w_all[-1, 1], w_all[-1, 2]),
#                              w_all[-1, 2], q_known.eos, label)
#            v_unknown = q_unknown.wavespeed(self.wavenumber)
#
#        self.wave_speed = []
#        if self.wavenumber == 0:
#            self.q_r = deepcopy(q_unknown)
#            self.wave_speed = numpy.array([v_known, v_unknown])
#        else:
#            self.q_l = deepcopy(q_unknown)
#            self.wave_speed = numpy.array([v_unknown, v_known])
#
#    def solve_deflagration(self, q_known, p_star, unknown_eos, t_i):
#        r"""
#        In the case of a deflagration, finds the unknown state on one side of the wave and wavespeed given the known state on the wave's other side and :math:`p_*`.
#
#        Parameters
#        ----------
#
#        self : Wave
#            The wave, which has a known state on one side and an unknown
#            state on the other side.
#        q_known : State
#            The known state on one side of the wave
#        p_star : scalar
#            Pressure in the region of unknown state
#        unknown_eos : dictionary
#            Equation of state in the unknown region
#        t_i : scalar
#            temperature
#            NOTE: what is this the temperature of??
#        """
#
#        self.type = "Deflagration"
#        self.name = r"({\cal WDF})"
#        if self.wavenumber == 0:
#            label = r"\star_L"
#            self.name += r"_{\leftarrow}"
#        else:
#            label = r"\star_R"
#            self.name += r"_{\rightarrow}"
#
#        v_known = q_known.wavespeed(self.wavenumber)
#
#        if numpy.allclose(q_known.p, p_star):
#            self.trivial = True
#            q_unknown = State(q_known.rho, q_known.v, q_known.vt, q_known.eps,
#            q_known.eos, label)
#            q_precursor = deepcopy(q_unknown)
#            v_deflagration = v_known
#        else:
#            lr_sign = self.wavenumber - 1
#            p_min = p_star
#            p_max = q_known.p
#
#            def deflagration_root(p_0_star, q_precursor):
#                j2, rho, eps, dp = self.mass_flux_squared(q_precursor, p_0_star, unknown_eos)
#                j = numpy.sqrt(j2)
#                v_deflagration = (q_precursor.rho**2 *
#                    q_precursor.W_lorentz**2 * q_precursor.v + \
#                    lr_sign * j**2 * \
#                    numpy.sqrt(1.0 + q_precursor.rho**2 *
#                    q_precursor.W_lorentz**2 *
#                    (1.0 - q_precursor.v**2) / j**2)) / \
#                    (q_precursor.rho**2 * q_precursor.W_lorentz**2 + j**2)
#                W_lorentz_deflagration = 1.0 / numpy.sqrt(1.0 - v_deflagration**2)
#                v = (q_precursor.h * q_precursor.W_lorentz *
#                     q_precursor.v + lr_sign * dp *
#                     W_lorentz_deflagration / j) / \
#                    (q_precursor.h * q_precursor.W_lorentz + dp * (1.0 /
#                     q_precursor.rho / q_precursor.W_lorentz + \
#                     lr_sign * q_precursor.v *
#                     W_lorentz_deflagration / j))
#                vt = q_precursor.vt_from_known(rho, v, eps)
#                q_unknown = State(rho, v, vt, eps, unknown_eos, label)
#
#                return q_unknown.wavespeed(self.wavenumber) - v_deflagration
#
#            def precursor_root(p_0_star):
#                shock = Wave(q_known, p_0_star, self.wavenumber)
#                if self.wavenumber == 0:
#                    q_precursor = shock.q_r
#                else:
#                    q_precursor = shock.q_l
#                t_precursor = q_precursor.eos['t_from_rho_eps'](
#                                q_precursor.rho, q_precursor.eps)
#                return t_precursor - t_i
#
#            # First, find the precursor shock
#            t_min = precursor_root(p_min)
#            t_max = precursor_root(p_max)
#            assert(t_min < 0)
#
#            if t_max <= 0:
#                p_max *= 2
#                t_max = precursor_root(p_max)
#
#            p_0_star = brentq(precursor_root, p_min, p_max)
#            precursor_shock = Wave(q_known, p_0_star, self.wavenumber)
#
#            if self.wavenumber == 0:
#                q_precursor = precursor_shock.q_r
#            else:
#                q_precursor = precursor_shock.q_l
#
#            # Next, find the deflagration discontinuity
#            j2, rho, eps, dp = self.mass_flux_squared(q_precursor,
#                                p_star, unknown_eos)
#            j = numpy.sqrt(j2)
#            v_deflagration = (q_precursor.rho**2 *
#                q_precursor.W_lorentz**2 * q_precursor.v + \
#                lr_sign * j**2 * \
#                numpy.sqrt(1.0 + q_precursor.rho**2 *
#                q_precursor.W_lorentz**2 *
#                (1.0 - q_precursor.v**2) / j**2)) / \
#                (q_precursor.rho**2 * q_precursor.W_lorentz**2 + j**2)
#            W_lorentz_deflagration = 1.0 / numpy.sqrt(1.0 - v_deflagration**2)
#            v = (q_precursor.h * q_precursor.W_lorentz * q_precursor.v +
#                 lr_sign * dp * W_lorentz_deflagration / j) / \
#                 (q_precursor.h * q_precursor.W_lorentz + dp * (1.0 /
#                  q_precursor.rho / q_precursor.W_lorentz + \
#                 lr_sign * q_precursor.v * W_lorentz_deflagration / j))
#            vt = q_precursor.vt_from_known(rho, v, eps)
#            q_unknown = State(rho, v, vt, eps, unknown_eos, label)
#
#            # If the speed in the unknown state means the characteristics are
#            # not going into the deflagration, then this is an unstable strong
#            # deflagration
#            if (lr_sign*(q_unknown.wavespeed(self.wavenumber) - v_deflagration) < 0):
#                p_cjdf = brentq(deflagration_root, (1.0+1e-9)*p_star,
#                                (1.0-1e-9)*p_0_star,
#                                args=(q_precursor,))
#                j2, rho, eps, dp = self.mass_flux_squared(q_precursor, p_cjdf, unknown_eos)
#                j = numpy.sqrt(j2)
#                v_deflagration = (q_precursor.rho**2 *
#                    q_precursor.W_lorentz**2 * q_precursor.v + \
#                    lr_sign * j**2 * \
#                    numpy.sqrt(1.0 + q_precursor.rho**2 *
#                    q_precursor.W_lorentz**2 *
#                    (1.0 - q_precursor.v**2) / j**2)) / \
#                    (q_precursor.rho**2 * q_precursor.W_lorentz**2 + j**2)
#                W_lorentz_deflagration = 1.0 / numpy.sqrt(1.0 - v_deflagration**2)
#                v = (q_precursor.h * q_precursor.W_lorentz *
#                    q_precursor.v + lr_sign * dp * W_lorentz_deflagration / j) / \
#                    (q_precursor.h * q_precursor.W_lorentz + dp * (1.0 /
#                     q_precursor.rho / q_precursor.W_lorentz + \
#                     lr_sign * q_precursor.v * W_lorentz_deflagration / j))
#                vt = q_precursor.vt_from_known(rho, v, eps)
#                q_cjdf = State(rho, v, vt, eps, unknown_eos, label)
#                # Now, we have to attach using a rarefaction
#                w_0_star = odeint(rarefaction_dwdp,
#                       numpy.array([q_cjdf.rho, q_cjdf.v, q_cjdf.eps]),
#                       [q_cjdf.p, p_star], rtol = 1e-12, atol = 1e-10,
#                       args=((q_cjdf, self.wavenumber)))
#                q_unknown = State(w_0_star[-1, 0], w_0_star[-1, 1],
#                            q_cjdf.vt_from_known(w_0_star[-1, 0], w_0_star[-1, 1], w_0_star[-1, 2]),
#                            w_0_star[-1, 2], unknown_eos, label)
#
#            self.p_0_star = p_0_star
#            self.q_0_star = q_precursor
#
#        self.wave_speed = []
#        if self.wavenumber == 0:
#            self.q_r = deepcopy(q_unknown)
#            self.wave_speed = numpy.array([v_known, v_deflagration, q_precursor.wavespeed])
#        else:
#            self.q_l = deepcopy(q_unknown)
#            self.wave_speed = numpy.array([q_precursor.wavespeed, v_deflagration, v_known])
#
#    def solve_detonation(self, q_known, p_star, unknown_eos):
#        r"""
#        In the case of a detonation, finds the unknown state on one side of the wave and wavespeed given the known state on the wave's other side and :math:`p_*`.
#
#        Parameters
#        ----------
#
#        self : Wave
#            The wave, which has a known state on one side and an unknown
#            state on the other side.
#        q_known : State
#            The known state on one side of the wave
#        p_star : scalar
#            Pressure in the region of unknown state
#        unknown_eos : dictionary
#            Equation of state in the unknown region
#        """
#
#        self.type = "Detonation"
#        lr_sign = self.wavenumber - 1
#        if unknown_eos is None:
#            unknown_eos = q_known.eos
#
#        self.name = r"({\cal SDT})"
#        if self.wavenumber == 0:
#            label = r"\star_L"
#            self.name += r"_{\leftarrow}"
#        else:
#            label = r"\star_R"
#            self.name += r"_{\rightarrow}"
#
#        if numpy.allclose(q_known.p, p_star):
#            self.trivial = True
#            q_unknown = State(q_known.rho, q_known.v, q_known.vt, q_known.eps,
#            q_known.eos, label)
#            v_shock = q_known.wavespeed(self.wavenumber)
#        else:
#            raise(NotImplementedError, "Do this")
#
#        if self.wavenumber == 0:
#            self.q_r = deepcopy(q_unknown)
#        else:
#            self.q_l = deepcopy(q_unknown)
#
#        self.wave_speed = numpy.array([v_shock, v_shock])
#
#    def plotting_data(self):
#
#        if self.type == "Rarefaction":
#            if self.wavenumber == 0:
#                p = numpy.linspace(self.q_l.p, self.q_r.p)
#                w_all = odeint(rarefaction_dwdp,
#                               numpy.array([self.q_l.rho, self.q_l.v, self.q_l.eps]),
#                               p, rtol = 1e-12, atol = 1e-10,
#                               args=(self.q_l,self.wavenumber))
#            else:
#                p = numpy.linspace(self.q_r.p, self.q_l.p)
#                w_all = odeint(rarefaction_dwdp,
#                               numpy.array([self.q_r.rho, self.q_r.v, self.q_r.eps]),
#                               p, rtol = 1e-12, atol = 1e-10,
#                               args=(self.q_r,self.wavenumber))
#                p = p[-1::-1]
#                w_all = w_all[-1::-1,:]
#            data = numpy.zeros((len(p),8))
#            xi = numpy.zeros_like(p)
#            for i in range(len(p)):
#                state = State(w_all[i,0], w_all[i,1],
#                              self.q_l.vt_from_known(w_all[i,0], w_all[i,1], w_all[i,2]),
#                              w_all[i, 2], self.q_l.eos)
#                xi[i] = state.wavespeed(self.wavenumber)
#                data[i,:] = state.state()
#        elif self.type == "Shock":
#            data = numpy.vstack((self.q_l.state(), self.q_r.state()))
#            xi = numpy.array([self.q_l.wavespeed(self.wavenumber),
#                           self.q_l.wavespeed(self.wavenumber)])
#        elif self.type == "Deflagration":
#            if self.wavenumber == 0:
#                p = numpy.linspace(self.q_l.p, self.p_0_star)
#                w_all = odeint(rarefaction_dwdp,
#                               numpy.array([self.q_l.rho, self.q_l.v, self.q_l.eps]),
#                               p, rtol = 1e-12, atol = 1e-10,
#                               args=(self.q_l,self.wavenumber))
#            else:
#                p = numpy.linspace(self.q_r.p, self.p_0_star)
#                w_all = odeint(rarefaction_dwdp,
#                               numpy.array([self.q_r.rho, self.q_r.v, self.q_r.eps]),
#                               p, rtol = 1e-12, atol = 1e-10,
#                               args=(self.q_r,self.wavenumber))
#                p = p[-1::-1]
#                w_all = w_all[-1::-1,:]
#            data = numpy.zeros((len(p)+1,8))
#            xi = numpy.zeros(len(p)+1)
#            if self.wavenumber == 0:
#                for i in range(len(p)):
#                    state = State(w_all[i,0], w_all[i,1],
#                                  self.q_l.vt_from_known(w_all[i,0], w_all[i,1], w_all[i,2]),
#                                  w_all[i, 2], self.q_l.eos)
#                    xi[i] = state.wavespeed(self.wavenumber)
#                    data[i,:] = state.state()
#                xi[-1] = self.wave_speed[-1]
#                data[-1,:] = self.q_r.state()
#            else:
#                xi[0] = self.wave_speed[0]
#                data[0,:] = self.q_l.state()
#                for i in range(len(p)):
#                    state = State(w_all[i,0], w_all[i,1],
#                                  self.q_r.vt_from_known(w_all[i,0], w_all[i,1], w_all[i,2]),
#                                  w_all[i, 2], self.q_r.eos)
#                    xi[i+1] = state.wavespeed(self.wavenumber)
#                    data[i+1,:] = state.state()
#
#        return xi, data
#
#    def latex_string(self):
#        s = self.name
#        s += r": \lambda^{{({})}}".format(self.wavenumber)
#        if self.type == "Rarefaction" or self.type == "Deflagration":
#            s += r"\in [{:.4f}, {:.4f}]".format(self.wave_speed[0],
#                         self.wave_speed[1])
#        else:
#            s += r"= {:.4f}".format(self.wave_speed[0])
#        return s
#
#    def _repr_latex_(self):
#        s = r"$" + self.latex_string() + r"$"
#        return s
