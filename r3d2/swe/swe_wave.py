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
from r3d2.wave import Wave, WaveSection
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



class SWEWaveSection(WaveSection):
    """
    Class for wave sections
    """



# NOTE: this class has a different signature to all other subclasses of
#       WaveSection (q_end rather than p_end). Might be more consistent
#       to use the same signature for all subclasses - all could
#       take argument q_end and access variable q_end.p.
class SWEContact(SWEWaveSection):

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
        assert(numpy.allclose(q_start.phi, q_end.phi)), "Pressures of states "\
        "must match for a contact"
        assert(numpy.allclose(q_start.wavespeed(wavenumber),
                              q_end.wavespeed(wavenumber))), "Wavespeeds of "\
        "states must match for a contact"

class SWERarefaction(SWEWaveSection):

    def __init__(self, q_start, phi_end, wavenumber):
        """
        A rarefaction.
        """

        self.trivial = False
        assert(wavenumber in [0, 2]), "wavenumber for a Rarefaction "\
        "must be in 0, 2"
        assert(q_start.phi >= phi_end), "For a rarefaction, phi_start >= phi_end"
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
            v_star = self.rarefaction_solve(q_start, phi_end, len(phi_points))[-1]

            self.q_end = SWEState(phi_end, v_star, label=label)
            v_unknown = self.q_end.wavespeed(self.wavenumber)

            #print(f'v_unknown = {v_unknown}, v_known = {v_known}')
            if self.wavenumber == 0:
                self.wavespeed = numpy.array([v_known, v_unknown])
            else:
                self.wavespeed = numpy.array([v_unknown, v_known])

    @staticmethod
    def raref(v, phi):
        return 0.5 * (-v**3 + v + (v**2 - 1) * numpy.sqrt(v**2 + 4 / phi))

    @staticmethod
    def eval(phi, v):
        return -0.5 * numpy.sqrt(phi) * (v - 1) * (v + 1) * \
            numpy.sqrt(phi * v**2 + 4) + 0.5 * v * (phi * v**2 - phi + 2)

    @staticmethod
    def rarefaction_solve(q, phi_star, n_phi_vals=2):
        phi, v = q.prim()
        phi_points = numpy.linspace(phi, phi_star, n_phi_vals)
        # lam_l = SWERarefaction.eval(phi, v)
        v_raref = odeint(SWERarefaction.raref, v, phi_points)
        return v_raref


    def plotting_data(self, n_points=500):
        """
        Parameters
        ----------
        n_points: integer
            number of points in the rarefaction plot
        """
        if self.trivial:
            xi = numpy.zeros((0,))
            data = numpy.zeros((0,3))
        else:
            phi_points = numpy.linspace(self.q_start.phi, self.q_end.phi, n_points)

            v_points = self.rarefaction_solve(self.q_start, self.q_end.phi, len(phi_points))
            #self.q_end = SWEState(self.q_end.phi, v_end)
            data = numpy.zeros((len(phi_points),3))
            xi = numpy.zeros_like(phi_points)
            for i in range(len(phi_points)):
                state = SWEState(phi_points[i], v_points[i])
                xi[i] = state.wavespeed(self.wavenumber)
                data[i,:] = state.state()

        return xi, data

class SWEShock(SWEWaveSection):

    def __init__(self, q_start, phi_end, wavenumber):
        """
        A shock.
        """

        self.trivial = False
        assert(wavenumber in [0, 2]), "wavenumber for a Shock "\
        "must be in 0, 2"
        assert(q_start.phi <= phi_end), "For a shock, phi_start <= phi_end"
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
            v_shock, v_star = self.analytic_shock(q_start, phi_end)
            self.q_end = SWEState(phi_end, v_star, label=label)

        self.wavespeed = [v_shock]


    @staticmethod
    def analytic_shock(q, phi_star):
        phi, v = q.prim()
        w_bar = numpy.sqrt(1 + phi_star / phi * (phi_star + phi) / 2)
        v_bar = -numpy.sqrt(1 - 1 / w_bar**2)
        V_s = (v - v_bar) / (1 - v * v_bar)
        #print('shock1', w_bar, v_bar, V_s)
        Wv_star_bar = phi * w_bar * v_bar / phi_star
        w_star_bar = numpy.sqrt(1 + Wv_star_bar**2)
        v_star_bar = -numpy.sqrt(1 - 1 / w_star_bar**2)
        v_star = (v_star_bar + V_s) / (1 + v_star_bar * V_s)
        return V_s, v_star




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

        super().__init__(self, q_known, unknown_value, wavenumber)

        waves = self.build_wave_section(q_known, unknown_value,
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
    def build_wave_section(q_known, unknown_value, wavenumber):
        """
        Object factory for the WaveSection
        """

        if wavenumber == 1:
            return [SWEContact(q_known, unknown_value, wavenumber)]
        elif q_known.phi < unknown_value:
            return [SWEShock(q_known, unknown_value, wavenumber)]
        else:
            return [SWERarefaction(q_known, unknown_value, wavenumber)]
