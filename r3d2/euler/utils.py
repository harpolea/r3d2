# -*- coding: utf-8 -*-

import numpy
from scipy.integrate import odeint
from scipy.optimize import brentq
from matplotlib.transforms import offset_copy
import matplotlib.transforms as mtrans
from r3d2.euler.euler_wave import EulerWaveSection, Rarefaction

def rarefaction(ps, q_start, lr_sign):
    vs = numpy.zeros_like(ps)
    for i, p in enumerate(ps):
        w_all = Rarefaction.rarefaction_w(q_start, [q_start.p, p],
                                   lr_sign+1)
        vs[i] = w_all[-1, 1]
    return vs

def shock(ps, q_start, lr_sign):
    vs = numpy.zeros_like(ps)
    for i, p in enumerate(ps):
        _, q_end = EulerWaveSection.post_discontinuity_state(p,
                                    q_start,
                                    lr_sign, "")
        vs[i] = q_end.v
    return vs 
#
# def deflagration(ps, q_start, lr_sign):
#     vs = numpy.zeros_like(ps)
#     for i, p in enumerate(ps):
#         j2, rho, eps, diffp = EulerWaveSection.mass_flux_squared(q_start,
#                                     p, q_start.eos)
#         v_deflagration, q_end = EulerWaveSection.post_discontinuity_state(p, q_start,
#                                     lr_sign, "", j2,
#                                     rho, eps, diffp)
#
#         if (lr_sign*(q_end.wavespeed(lr_sign+1) - v_deflagration) < 0):
#             p_cjdf = brentq(EulerWaveSection.deflagration_root,
#                             (1.0+1e-9)*p,
#                             (1.0-1e-9)*q_start.p,
#                             args=(q_start, q_start.eos,
#                             lr_sign+1, ""))
#             j2, rho, eps, dp = EulerWaveSection.mass_flux_squared(q_start,
#                             p_cjdf, q_start.eos)
#             v_deflagration, q_end = EulerWaveSection.post_discontinuity_state(p_cjdf, q_start,
#                                     lr_sign, "", j2,
#                                     rho, eps, diffp)
#         vs[i] = q_end.v
#     return vs
#
# def detonation(ps, q_start, lr_sign):
#     vs = numpy.zeros_like(ps)
#     for i, p in enumerate(ps):
#         j2, rho, eps, diffp = EulerWaveSection.mass_flux_squared(q_start,
#                                     p, q_start.eos)
#         if j2 < 0:
#             # The single detonation is unphysical - must be unstable weak
#             # detonation. So skip the calculation and make sure the CJ
#             # calculation runs
#             q_unknown = q_start[:]
#             v_detonation = q_unknown.wavespeed(lr_sign+1) + 1
#         else:
#             v_detonation, q_unknown = EulerWaveSection.post_discontinuity_state(p, q_start,
#                                     lr_sign, "", j2,
#                                     rho, eps, diffp,
#                                     q_start.eos)
#
#         if (lr_sign*(q_unknown.wavespeed(lr_sign+1) - v_detonation) < 0):
#             pmin = (1.0+1e-9)*min(q_start.p, p)
#             pmax = max(q_start.p, p)
#             fmin = EulerWaveSection.deflagration_root(pmin, q_start, q_start.eos, lr_sign+1, "")
#             fmax = EulerWaveSection.deflagration_root(pmax, q_start, q_start.eos, lr_sign+1, "")
#             while fmin * fmax > 0:
#                 pmax *= 2.0
#                 fmax = EulerWaveSection.deflagration_root(pmax, q_start, q_start.eos, lr_sign+1, "")
#             p_cjdt = brentq(EulerWaveSection.deflagration_root, pmin, pmax,
#                             args=(q_start, q_start.eos, lr_sign+1, ""))
#             j2, rho, eps, dp = EulerWaveSection.mass_flux_squared(q_start, p_cjdt, q_start.eos)
#             v_detonation, q_unknown = EulerWaveSection.post_discontinuity_state(p_cjdt, q_start,
#                                    lr_sign, "", j2,
#                                    rho, eps, diffp,
#                                    q_start.eos)
#         vs[i] = q_unknown.v
#     return vs
#
# def find_pre_ignition(v_s, j2, rho, lr_sign):
#     A = rho**2 / (j2 *
#         (v_s - lr_sign * numpy.sqrt(1 + rho**2 / j2)))
#
#     return -0.5 * A + numpy.sqrt(1 + A * v_s), \
#            -0.5 * A - numpy.sqrt(1 + A * v_s)
