---
title: 'R3D2: Relativistic Reactive Riemann problem solver for Deflagrations and Detonations'
tags:
  - hydrodynamics
  - Riemann problem
  - relativity
  - reactions
authors:
 - name: Alice Harpole
   orcid: 0000-0002-1530-781X
   affiliation: University of Southampton
 - name: Ian Hawke
   orcid: 0000-0003-4805-0309
   affiliation: University of Southampton
date: 6 May 2016
bibliography: paper.bib
---

# Summary

This code extends standard exact solutions of the relativistic Riemann Problem to include a reaction term. It builds on existing solutions for the inert relativistic Riemann problem, as described by [@Marti2015], and of the non-relativistic reactive Riemann problem, as given by [@Zhang1989].

Models of ideal hydrodynamics, where there is no viscosity or dissipation, can have solutions with *discontinuities* such as shocks. A simple case is the Riemann Problem, where two constant states are separated by a barrier. After the barrier is removed the solution develops, with *waves* (such as shocks and rarefactions) separating constant states.

The Riemann Problem has three main uses. Efficient, often approximate, solvers are an integral part of many modern hydrodynamic evolution codes. Second, the exact solution is a standard test for such codes. Finally, the solver can illustrate features of discontinuous solutions in more complex scenarios.

In Newtonian hydrodynamics, the Riemann problem is one-dimensional: the solution depends only on the normal component of any vector quantities in the initial conditions. However, in relativistic systems, the Lorentz factor introduces a coupling between the normal and tangential components. As found by [@Rezzolla2002], for high enough tangential velocities, the solution will smoothly transition from one wave pattern to another while maintaining the initial states otherwise unmodified. This code allows such transitions to be investigated in both inert and reactive systems.

# References
