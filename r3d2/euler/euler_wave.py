# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy
from scipy.optimize import brentq
from scipy.integrate import odeint
from r3d2 import Wave, WaveSection
from r3d2.euler.euler_state import EulerState

class EulerWaveSection(WaveSection):
    """
    Class for wave sections
    """
    #
    # @staticmethod
    # def deflagration_root(p_0_star, q_precursor, unknown_eos, wavenumber, label):
    #     lr_sign = wavenumber - 1
    #     j2, rho, eps, dp = EulerWaveSection.mass_flux_squared(q_precursor, p_0_star, unknown_eos)
    #     if j2 < 0:
    #         return 10.0 # Unphysical part of Crussard curve, return a random number
    #     j = numpy.sqrt(j2)
    #     v_deflagration = (q_precursor.rho**2 *
    #         q_precursor.v + \
    #         lr_sign * j**2 * \
    #         numpy.sqrt(1.0 + q_precursor.rho**2 *
    #         (1.0 - q_precursor.v**2) / j**2)) / \
    #         (q_precursor.rho**2 + j**2)
    #     v = (q_precursor.h *
    #          q_precursor.v + lr_sign * dp / j) / \
    #         (q_precursor.h * + dp * (1.0 /
    #          q_precursor.rho + \
    #          lr_sign * q_precursor.v / j))
    #     q_unknown = EulerState(rho, v, eps, unknown_eos, label)
    #
    #     return q_unknown.wavespeed(wavenumber) - v_deflagration

    @staticmethod
    def post_discontinuity_state(p_star, q_start, lr_sign, label, eos_end=None):
        if eos_end is None:
            eos_end = q_start.eos

        gamma = q_start.eos.gamma

        A = 2 / (gamma+1) / q_start.rho
        B = (gamma-1) * q_start.p / (gamma+1)
        f = (p_star - q_start.p) * numpy.sqrt(A / (p_star + B))
        Q = numpy.sqrt((p_star + B) / A)

        v = q_start.v + lr_sign * f

        rho = q_start.rho * (p_star / q_start.p + (gamma-1)/(gamma+1)) / ((gamma-1)/(gamma+1) * p_star / q_start.p + 1)
        eps = q_start.eos.eps_from_rho_p(rho, p_star)

        v_shock = q_start.v + lr_sign * Q / q_start.rho

        q_end = EulerState(rho, v, eps, eos_end, label=label)
        return v_shock, q_end

# NOTE: this class has a different signature to all other subclasses of
#       WaveSection (q_end rather than p_end). Might be more consistent
#       to use the same signature for all subclasses - all could
#       take argument q_end and access variable q_end.p.
class Contact(EulerWaveSection):

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
        # print(f'p_start = {q_start.p}, p_end = {q_end.p}')
        assert(numpy.allclose(q_start.p, q_end.p)), "Pressures of states "\
        "must match for a contact"
        assert(numpy.allclose(q_start.wavespeed(wavenumber),
                              q_end.wavespeed(wavenumber))), "Wavespeeds of "\
        "states must match for a contact"

class Rarefaction(EulerWaveSection):

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

        self.wavespeed = []

        if numpy.allclose(q_start.p, p_end):
            self.trivial = True
            self.q_end = EulerState(q_start.rho, q_start.v, q_start.eps,
                                    q_start.eos, label=label)
            v_unknown = v_known
            self.name = ""
        else:
            w_all = self.rarefaction_w(q_start, [q_start.p, p_end],
                                       self.wavenumber)
            self.q_end = EulerState(w_all[-1, 0], w_all[-1, 1],
                                    w_all[-1, 2], q_start.eos, label=label)
            v_unknown = self.q_end.wavespeed(self.wavenumber)
            if self.wavenumber == 0:
                self.wavespeed = numpy.array([v_known, v_unknown])
            else:
                self.wavespeed = numpy.array([v_unknown, v_known])

    @staticmethod
    def rarefaction_w(q_known, ps, wavenumber):
        r"""
        There is a tricky point here that needs investigation. If
        the input p is used here, rather than local_state.p, then they
        can diverge (when :math:`v_t` is significant) leading to overflows of g. By
        using local_state we avoid the overflow, but it may mean the final
        state is not very accurate.

        Parameters
        ----------

        w : tuple
            primitive state (rho, v, eps)
        p : scalar
            pressure (required by odeint, but not used: see note above)
        q_known : State
            Known state
        wavenumber : scalar
            Wave number
        """
        lr_sign = wavenumber - 1
        ws = numpy.zeros((len(ps), len(q_known.prim())))
        #xi_start = q_known.v - q_known.cs
        ws[0, :] = q_known.prim()
        gamma = q_known.eos.gamma
        for p, w in zip(ps[1:], ws[1:]):
            xi = q_known.v - lr_sign * \
                 ((p/q_known.p)**((gamma-1)/(2*gamma)) - 
                 2/(gamma+1)) * (gamma + 1) * q_known.cs / (gamma -1)
            rho = q_known.rho * (2 / (gamma + 1) - lr_sign *
                (gamma-1) / ((gamma+1)*q_known.cs) *
                (q_known.v - xi))**(2 / (gamma-1))
            v = 2 / (gamma + 1) * (- lr_sign * q_known.cs +
                0.5 * (gamma-1) * q_known.v + xi)
            # v = 2 * q_known.cs / (gamma - 1) * ((p_star / q_start.p)**(0.5*(gamma-1)/gamma) - 1)
            eps = q_known.eos.eps_from_rho_p(rho, p)

            w[:] = [rho, v, eps]

        return ws

    def plotting_data(self, t_end):
        # TODO: make the number of points in the rarefaction plot a parameter
        if self.trivial:
            xi = numpy.zeros((0,))
            data = numpy.zeros((0,len(self.q_start.state())))
        else:
            p = numpy.linspace(self.q_start.p, self.q_end.p, 500)
            w_all = self.rarefaction_w(self.q_start, p,
                                       self.wavenumber)
            data = numpy.zeros((len(p),len(self.q_start.state())))
            xi = numpy.zeros_like(p)
            for i in range(len(p)):
                state = EulerState(w_all[i,0], w_all[i,1],
                              w_all[i, 2], self.q_start.eos)
                xi[i] = state.wavespeed(self.wavenumber)
                data[i,:] = state.state()

        return xi, data

class Shock(EulerWaveSection):

    def __init__(self, q_start, p_end, wavenumber):
        """
        A shock.
        """

        self.trivial = False
        assert(wavenumber in [0, 2]), "wavenumber for a Shock "\
        "must be in 0, 2"
        # As we use the Shock code for deflagration checks, we can't apply
        # this check
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
            self.q_end = EulerState(q_start.rho, q_start.v, q_start.eps,
                                    q_start.eos, label=label)
            v_shock = q_start.wavespeed(self.wavenumber)
            self.name = ""
        else:
            v_shock, self.q_end = self.post_discontinuity_state(p_end,
                                    q_start,
                                    lr_sign, label)

        self.wavespeed = [v_shock]
#
# # TODO: Check that q is correctly initialized across each wave in det, defl.
# class Deflagration(EulerWaveSection):
#
#     def __init__(self, q_start, p_end, wavenumber):
#         """
#         A deflagration.
#         """
#
#         eos_end = q_start.eos.eos_inert
#         t_i = q_start.eos.t_i_from_rho_eps(q_start.rho, q_start.eps)
#
#         self.trivial = False
#         assert(wavenumber in [0, 2]), "wavenumber for a Deflagration "\
#         "must be in 0, 2"
#         assert(q_start.p >= p_end), "For a deflagration, p_start >= p_end"
# #        t_start = q_start.eos['t_from_rho_eps'](q_start.rho, q_start.eps)
# #        assert(t_start >= t_i), "For a deflagration, temperature of start "\
# #        "state must be at least the ignition temperature"
#         # TODO The above check should be true, but the root-find sometimes just
#         # misses. numpy allclose type check?
#         self.type = "Deflagration"
#         self.wavenumber = wavenumber
#         lr_sign = self.wavenumber - 1
#         self.q_start = deepcopy(q_start)
#
#         self.name = r"{\cal WDF}"
#         if self.wavenumber == 0:
#             label = r"\star_L"
#             self.name += r"_{\leftarrow}"
#         else:
#             label = r"\star_R"
#             self.name += r"_{\rightarrow}"
#
#         v_known = q_start.wavespeed(self.wavenumber)
#
#         if numpy.allclose(q_start.p, p_end):
#             self.trivial = True
#             self.q_end = EulerState(q_start.rho, q_start.v, q_start.eps,
#                                     eos_end, label=label)
#             v_deflagration = v_known
#             self.name = ""
#         else:
#             # This is a single deflagration, so the start state must be at the
#             # reaction temperature already.
#             v_deflagration, q_unknown = self.post_discontinuity_state(p_end, q_start,
#                                     lr_sign, label)
#
#             # If the speed in the unknown state means the characteristics are
#             # not going into the deflagration, then this is an unstable strong
#             # deflagration
#             if (lr_sign*(q_unknown.wavespeed(self.wavenumber) - v_deflagration) < 0):
#                 p_cjdf = brentq(self.deflagration_root, (1.0+1e-9)*p_end,
#                                 (1.0-1e-9)*q_start.p,
#                                 args=(q_start, eos_end, self.wavenumber, label))
#                 v_deflagration, q_unknown = self.post_discontinuity_state(p_cjdf, q_start,
#                                         lr_sign, label)
#                 self.name = r"{\cal CJDF}"
#                 if self.wavenumber == 0:
#                     label = r"\star_L"
#                     self.name += r"_{\leftarrow}"
#                 else:
#                     label = r"\star_R"
#                     self.name += r"_{\rightarrow}"
#
#             self.q_end = deepcopy(q_unknown)
#
#         self.wavespeed = [v_deflagration]
#
# class Detonation(EulerWaveSection):
#
#     def __init__(self, q_start, p_end, wavenumber):
#         """
#         A detonation.
#         """
#
#         eos_end = q_start.eos.eos_inert
#         t_i = q_start.eos.t_i_from_rho_eps(q_start.rho, q_start.eps)
#
#         self.trivial = False
#         assert(wavenumber in [0, 2]), "wavenumber for a Detonation "\
#         "must be in 0, 2"
#         assert(q_start.p <= p_end), "For a detonation, p_start <= p_end"
#         #t_start = q_start.eos['t_from_rho_eps'](q_start.rho, q_start.eps)
#         #assert(t_start >= t_i), "For a detonation, temperature of start "\
#         #"state must be at least the ignition temperature"
#         self.type = "Detonation"
#         self.wavenumber = wavenumber
#         lr_sign = self.wavenumber - 1
#         self.q_start = deepcopy(q_start)
#
#         self.name = r"{\cal SDT}"
#         if self.wavenumber == 0:
#             label = r"\star_L"
#             self.name += r"_{\leftarrow}"
#         else:
#             label = r"\star_R"
#             self.name += r"_{\rightarrow}"
#
#         v_known = q_start.wavespeed(self.wavenumber)
#
#         if numpy.allclose(q_start.p, p_end):
#             self.trivial = True
#             self.q_end = EulerState(q_start.rho, q_start.v, q_start.eps,
#                                     eos_end, label=label)
#             v_detonation = v_known
#             self.name = ""
#         else:
#             # This is a single detonation, so the start state must be at the
#             # reaction temperature already.
#             j2, rho, eps, dp = self.mass_flux_squared(q_start, p_end, eos_end)
#             if j2 < 0:
#                 # The single detonation is unphysical - must be unstable weak
#                 # detonation. So skip the calculation and make sure the CJ
#                 # calculation runs
# #                print("Should be a CJ detonation")
#                 q_unknown = deepcopy(q_start)
#                 v_detonation = q_unknown.wavespeed(self.wavenumber) + lr_sign
#             else:
#                 v_detonation, q_unknown = self.post_discontinuity_state(p_end, q_start,
#                                         lr_sign, label, j2,
#                                         rho, eps, dp,
#                                         eos_end)
#
#             # If the speed in the unknown state means the characteristics are
#             # not going into the detonation, then this is an unstable weak
#             # detonation
#             if lr_sign*(q_unknown.wavespeed(self.wavenumber) - v_detonation) < 0:
#                 pmin = (1.0+1e-9)*min(q_start.p, p_end)
#                 pmax = max(q_start.p, p_end)
#                 fmin = self.deflagration_root(pmin, q_start, eos_end, self.wavenumber, label)
#                 fmax = self.deflagration_root(pmax, q_start, eos_end, self.wavenumber, label)
#                 while fmin * fmax > 0:
#                     pmax *= 2.0
#                     fmax = self.deflagration_root(pmax, q_start, eos_end, self.wavenumber, label)
#                 p_cjdt = brentq(self.deflagration_root, pmin, pmax,
#                                 args=(q_start, eos_end, self.wavenumber, label))
#                 j2, rho, eps, dp = self.mass_flux_squared(q_start, p_cjdt, eos_end)
#                 v_detonation, q_unknown = self.post_discontinuity_state(p_cjdt, q_start,
#                                        lr_sign, label, j2,
#                                        rho, eps, dp,
#                                        eos_end)
#                 self.name = r"{\cal CJDT}"
#                 if self.wavenumber == 0:
#                     label = r"\star_L"
#                     self.name += r"_{\leftarrow}"
#                 else:
#                     label = r"\star_R"
#                     self.name += r"_{\rightarrow}"
#
#             self.q_end = deepcopy(q_unknown)
#
#         self.wavespeed = numpy.array([v_detonation])



class EulerWave(Wave):

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

        super().__init__(q_known, unknown_value, wavenumber)

        if not hasattr(q_known.eos, 'q'):
            waves = self.build_inert_wave_section(q_known, unknown_value,
                                                  wavenumber)
            for sections in waves:
                self.wave_sections.append(sections)
        else:
            waves = self.build_reactive_wave_section(q_known, unknown_value,
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
    #
    # @staticmethod
    # def precursor_root(p_0_star, q_known, wavenumber):
    #     shock = Shock(q_known, p_0_star, wavenumber)
    #     q_precursor = shock.q_end
    #     t_precursor = q_precursor.eos.t_from_rho_eps(
    #         q_precursor.rho, q_precursor.eps)
    #     t_i = q_precursor.eos.t_i_from_rho_eps(q_precursor.rho, q_precursor.eps)
    #     return t_precursor - t_i
    #
    # @staticmethod
    # def build_reactive_wave_section(q_known, unknown_value, wavenumber):
    #     """
    #     Object factory for the WaveSection; reactive case
    #     """
    #
    #     t_i = q_known.eos.t_i_from_rho_eps(q_known.rho, q_known.eps)
    #
    #     if wavenumber == 1:
    #         return Contact(q_known, unknown_value, wavenumber)
    #     else:
    #         wavesections = []
    #         if q_known.p < unknown_value:
    #             # The detonation wave
    #             detonation = Detonation(q_known, unknown_value, wavenumber)
    #             wavesections.append(detonation)
    #             q_next = deepcopy(detonation.q_end)
    #             # Finally, was it a CJ detonation?
    #             if q_next.p > unknown_value:
    #                 rarefaction = Rarefaction(q_next, unknown_value, wavenumber)
    #                 wavesections.append(rarefaction)
    #         else:
    #             t_known = q_known.eos.t_from_rho_eps(q_known.rho, q_known.eps)
    #             t_i = q_known.eos.t_i_from_rho_eps(q_known.rho, q_known.eps)
    #             if t_known < t_i: # Need a precursor shock
    #                 p_min = unknown_value
    #                 p_max = q_known.p
    #                 t_min = EulerWave.precursor_root(p_min, q_known, wavenumber)
    #                 t_max = EulerWave.precursor_root(p_max, q_known, wavenumber)
    #                 assert t_min < 0
    #
    #                 if t_max <= 0:
    #                     p_max *= 2
    #                     t_max = EulerWave.precursor_root(p_max, q_known,
    #                                                      wavenumber)
    #
    #                 p_0_star = brentq(EulerWave.precursor_root, p_min, p_max,
    #                                   args=(q_known, wavenumber))
    #                 precursor_shock = Shock(q_known, p_0_star, wavenumber)
    #                 wavesections.append(precursor_shock)
    #                 q_next = precursor_shock.q_end
    #                 q_next.q = q_known.q # No reaction across inert precursor
    #                 q_next.eos = q_known.eos
    #             else: # No precursor shock
    #                 q_next = deepcopy(q_known)
    #             # Next, the deflagration wave
    #             deflagration = Deflagration(q_next, unknown_value, wavenumber)
    #             wavesections.append(deflagration)
    #             q_next = deepcopy(deflagration.q_end)
    #             # Finally, was it a CJ deflagration?
    #             if q_next.p > unknown_value:
    #                 rarefaction = Rarefaction(q_next, unknown_value, wavenumber)
    #                 wavesections.append(rarefaction)
    #
    #         return wavesections
