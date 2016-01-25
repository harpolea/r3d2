*********
eos_defns
*********

This is a set of pre-defined equations of state (EOS) for use with the Riemann Solver.

In general an equation of state means specifying the internal energy in terms of two thermodynamic variables, such as the volume and the temperature. All other quantities are derived from the Maxwell relations.

For our purposes, an EOS is a Python *dictionary* containing the essential relations needed.

Functions needed
================

1. ``p_from_rho_eps(rho, eps)``. :math:`p(\rho_0, \epsilon)`. Pressure given rest mass density and specific internal energy.
2. ``h_from_rho_eps(rho, eps)``. :math:`h(\rho_0, \epsilon)`. Specific enthalpy given rest mass density and specific internal energy.
3. ``cs_from_rho_eps(rho, eps)``. :math:`c_s (\rho_0, \epsilon)`. Speed of sound given rest mass density and specific internal energy.
4. ``h_from_rho_p(rho, p)``. :math:`h(\rho_0, p)`. Specific enthalpy given rest mass density and pressure.
5. ``t_from_rho_eps(rho, eps)``. :math:`T(\rho_0, \epsilon)`. Temperature given rest mass density and specific internal energy.

Provided EOS
============

1. ``eos_gamma_law(gamma)``. Standard :math:`\gamma`-law EOS where :math:`e(V, T) = C_V T` and so :math:`p(\rho_0, \epsilon) = (\gamma - 1) \rho_0 \epsilon`.
2. ``eos_gamma_law_react(gamma, q, Cv)``. Reactive EOS where :math:`e(V, T) = C_V T + q` and so :math:`p(\rho_0, \epsilon) = (\gamma - 1) \rho_0 (\epsilon - q)`, where :math:`q` is the chemical binding energy.
