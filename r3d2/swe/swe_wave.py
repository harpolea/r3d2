# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:28:53 2016

@author: ih3
"""

from copy import deepcopy
from abc import ABCMeta#, abstractmethod
import numpy
from scipy.optimize import root
from scipy.integrate import odeint
from r3d2.wave import Wave
from swe_state import SWEState



# NOTE: all subclasses begin with initialising type, name, wavenumber etc.
#       Can avoid some repeated code by passing these as arguments to
#       superclass constructer and calling that.

# NOTE: To avoid more repeated code: wave speed calculation appears to
#       consist of one or two main parts - a shock wave bit and a burning
#       wave bit. The code for these two sections is almost identical for
#       all subclasses of WaveSection.
#       Could therefore make define functions calculate_shock_speed and
#       calculate_burning_speed for WaveSection class, which are then
#       called by its subclasses



class WaveSection(metaclass=ABCMeta):
    """
    Abstract base class for wave sections
    """

    def __init__(self, q_start, p_end, wavenumber):
        """
        A part of a wave. For a single shock or rarefaction, this will be the
        complete wave. For a deflagration or detonation, it may form part of
        the full wave.
        """
        # NOTE: what does self.trivial mean?
        self.trivial = False
        assert(wavenumber in [0, 1, 2]), "wavenumber must be 0, 1, 2"
        self.wavenumber = wavenumber
        self.name = None
        self.q_start = None
        self.q_end = None
        self.wavespeed = []
        self.type = ""

    def latex_string(self):
        if self.trivial:
            return ""
        else:
            s = deepcopy(self.name)
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

    def __repr__(self):
        return self.type

    def plotting_data(self):

        if self.trivial:
            data = numpy.zeros((0,8))
            xi = numpy.zeros((0,))
        else:
            data = numpy.vstack((self.q_start.state(), self.q_end.state()))
            xi = numpy.array([self.wavespeed[0], self.wavespeed[0]])

        return xi, data


# NOTE: this class has a different signature to all other subclasses of
#       WaveSection (q_end rather than p_end). Might be more consistent
#       to use the same signature for all subclasses - all could
#       take argument q_end and access variable q_end.p.
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

        if numpy.allclose(q_start.state(), q_end.state()):
            self.trivial = True
            self.name = ""

        assert(numpy.allclose(q_start.v, q_end.v)), "Velocities of states "\
        "must match for a contact"
        assert(numpy.allclose(q_start.p, q_end.p)), "Pressures of states "\
        "must match for a contact"
        assert(numpy.allclose(q_start.wavespeed(wavenumber),
                              q_end.wavespeed(wavenumber))), "Wavespeeds of "\
        "states must match for a contact"

class Rarefaction(WaveSection):

    def __init__(self, q_start, phi_end, wavenumber):
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

        self.wavespeed = []

        if numpy.allclose(q_start.phi, phi_end):
            self.trivial = True
            self.q_end = SWEState(q_start.phi, q_start.v, label=label)
            v_unknown = v_known
            self.name = ""
        else:
            phi_points = numpy.linspace(q_start.phi, phi_end)
            w_all = odeint(self.raref, q_start.v, phi_points,
                           rtol=1e-12, atol=1e-10)
            self.q_end = SWEState(w_all[-1, 0], w_all[-1, 1], label=label)
            v_unknown = self.q_end.wavespeed(self.wavenumber)
            if self.wavenumber == 0:
                self.wavespeed = numpy.array([v_known, v_unknown])
            else:
                self.wavespeed = numpy.array([v_unknown, v_known])

    @staticmethod
    def raref(v, phi):
        return 0.5 * (-v**3 + v + (v**2 - 1) * numpy.sqrt(v**2 + 4 / phi))


    def plotting_data(self):
        # TODO: make the number of points in the rarefaction plot a parameter
        if self.trivial:
            xi = numpy.zeros((0,))
            data = numpy.zeros((0,8))
        else:
            p = numpy.linspace(self.q_start.p, self.q_end.p, 500)
            w_all = odeint(self.rarefaction_dwdp,
                           numpy.array([self.q_start.phi, self.q_start.v, self.q_start.eps]),
                               p, rtol = 1e-12, atol = 1e-10,
                               args=(self.q_start, self.wavenumber))
            data = numpy.zeros((len(p),8))
            xi = numpy.zeros_like(p)
            for i in range(len(p)):
                state = SWEState(w_all[i,0], w_all[i,1])
                xi[i] = state.wavespeed(self.wavenumber)
                data[i,:] = state.state()

        return xi, data

class Shock(WaveSection):

    def __init__(self, q_start, phi_end, wavenumber):
        """
        A shock.
        """

        self.trivial = False
        assert(wavenumber in [0, 2]), "wavenumber for a Shock "\
        "must be in 0, 2"
        # As we use the Shock code for deflagration checks, we can't apply
        # this check
        #assert(q_start.p <= p_end), "For a shock, p_start <= p_end"
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

        if numpy.allclose(q_start.phi, phi_end):
            self.trivial = True
            self.q_end = SWEState(q_start.phi, q_start.v, label=label)
            v_shock = q_start.wavespeed(self.wavenumber)
            self.name = ""
        else:
            self.q_end = root(self.shock_residual, [q_start.v, v_star], args=(q_start, phi_end))

        self.wavespeed = [v_shock]

    @staticmethod
    def shock_residual(guess, w, phi_end):
        Vs, v_star = guess
        phi, v = w
        W = 1 / numpy.sqrt(1 - v**2)
        W_star = 1 / numpy.sqrt(1 - v_star**2)
        q = numpy.array([phi * W, phi * W**2 * v])
        q_star = numpy.array([phi_end * W_star, phi_end * W_star**2 * v_star])
        f = numpy.array([phi * W * v, phi * W**2 * v**2 + (phi**2) / 2])
        f_star = numpy.array([phi_end * W_star * v_star, phi_end * W_star**2 * v_star**2 + (phi_end**2) / 2])
        residual = Vs * (q - q_star) - (f - f_star)
        return residual

    # @staticmethod
    # def phi_residual(phi_star, wl, wr, Vs_hard_guess):
    #     # Solve across the rarefaction
    #     v_star_raref = rarefaction(wl, phi_star)[-1]
    #     # Guess the shock speed
    #     if Vs_hard_guess is None:
    #         Vs_guess = 0.9*evals_p_sp(phi_star, v_star_raref)
    #     else:
    #         Vs_guess = Vs_hard_guess
    #     #print('guess', evals_p_sp(phi_star, v_star_raref), evals_p_sp(*wr), Vs_guess)
    #     shock_result = root(shock_residual, [Vs_guess, v_star_raref], args=(wr, phi_star))
    #     #print(shock_result)
    #     v_star_shock = shock_result.x[1]
    #
    #     return v_star_raref - v_star_shock


class SWEWave(Wave):

    def __init__(self, q_known, unknown_value, wavenumber):
        """
        A wave.

        Parameters
        ----------

        self : Wave
            The wave, which has a known state on one side and an unknown
            state on the other side.
        q_known : State
            The known state on one side of the wave
        unknown_value : scalar
            Pressure in the region of unknown state
        wavenumber : scalar
            characterises direction of travel of wave
        """

        # NOTE: it's not so clear what wavenumber is - change to something like a wavedirection variable which can be left/right/static?
        self.wavenumber = wavenumber
        self.wave_sections = []
        self.wavespeed = []

        waves = self.build_inert_wave_section(q_known, unknown_value,
                                         wavenumber)
        for sections in waves:
            self.wave_sections.append(sections)

        self.name = self.wave_sections_latex_string()
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

        minspeed = 10
        maxspeed = -10
        if self.wave_sections:
            for wavesection in self.wave_sections:
                for speed in wavesection.wavespeed:
                    minspeed = min(speed, minspeed)
                    maxspeed = max(speed, maxspeed)
        self.wavespeed.append(minspeed)
        if not numpy.allclose(minspeed, maxspeed):
            self.wavespeed.append(maxspeed)

        self.trivial = True
        if self.wave_sections:
            for wavesection in self.wave_sections:
                if not wavesection.trivial:
                    self.trivial = False
        if self.trivial:
            self.wavespeed = []

    @staticmethod
    def build_inert_wave_section(q_known, unknown_value, wavenumber):
        """
        Object factory for the WaveSection; non-reactive case
        """

        if wavenumber == 1:
            return [Contact(q_known, unknown_value, wavenumber)]
        elif q_known.p < unknown_value:
            return [Shock(q_known, unknown_value, wavenumber)]
        else:
            return [Rarefaction(q_known, unknown_value, wavenumber)]

    @staticmethod
    def precursor_root(p_0_star, q_known, wavenumber):
        shock = Shock(q_known, p_0_star, wavenumber)
        q_precursor = shock.q_end
        t_precursor = q_precursor.eos.t_from_phi_eps(
                        q_precursor.phi, q_precursor.eps)
        t_i = q_precursor.eos.t_i_from_phi_eps(q_precursor.phi, q_precursor.eps)
        return t_precursor - t_i

    def plotting_data(self):

        xi_wave = numpy.zeros((0,))
        data_wave = numpy.zeros((0,8))
        for wavesection in self.wave_sections:
            xi_section, data_section = wavesection.plotting_data()
            xi_wave = numpy.hstack((xi_wave, xi_section))
            data_wave = numpy.vstack((data_wave, data_section))

        if self.wavenumber == 2:
            xi_wave = xi_wave[-1::-1]
            data_wave = data_wave[-1::-1,:]

        return xi_wave, data_wave

    def wave_sections_latex_string(self):
        names = []
        sections = deepcopy(self.wave_sections)
        if self.wavenumber == 2:
            sections.reverse()
        for sec in sections:
            if not sec.trivial:
                names.append(sec.name)
        s = ""
        if len(names)==1:
            s = names[0]
        elif len(names)>1:
            s = r"\left("
            for n in names:
                s += n
            s += r"\right) "
        return s

    def latex_string(self):
        s = self.wave_sections_latex_string()
        speeds = []
        sections = deepcopy(self.wave_sections)
        if self.wavenumber == 2:
            sections.reverse()
        for sec in sections:
            if not sec.trivial:
                for speed in sec.wavespeed:
                    speeds.append(speed)
        if len(speeds) == 0:
            return ""
        elif len(speeds) == 1:
            s += r": \lambda^{{({})}}".format(self.wavenumber)
            s += r"= {:.4f}".format(speeds[0])
        else:
            s += r": \lambda^{{({})}}".format(self.wavenumber)
            s += r"\in [{:.4f}, {:.4f}]".format(min(speeds), max(speeds))

        return s

    def _repr_latex_(self):
        s = r"$" + self.latex_string() + r"$"
        return s
