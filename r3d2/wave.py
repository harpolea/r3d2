# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:28:53 2016

@author: ih3
"""
from abc import ABCMeta#, abstractmethod
from copy import deepcopy
import numpy
from scipy.optimize import brentq
from scipy.integrate import odeint
from .state import State


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

    def plotting_data(self, t_end):

        if self.trivial:
            data = numpy.zeros((0,len(self.q_start.state())))
            xi = numpy.zeros((0,))
        else:
            data = numpy.vstack((self.q_start.state(), self.q_end.state()))
            xi = numpy.array([self.wavespeed[0], self.wavespeed[0]]) * t_end

        return xi, data



class Wave(object):

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

        # initialise
        self.q_l = deepcopy(q_known)
        self.q_r = deepcopy(q_known)


    def plotting_data(self, t_end):

        xi_wave = numpy.zeros((0,))
        data_wave = numpy.zeros((0,len(self.q_l.state())))
        for wavesection in self.wave_sections:
            xi_section, data_section = wavesection.plotting_data(t_end)
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
