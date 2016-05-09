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

In Newtonian hydrodynamics, the Riemann problem is one-dimensional: the solution depends only on the normal component of any vector quantities in the initial conditions. However, in relativistic systems, the Lorentz factor introduces a coupling between the normal and tangential components. As found by [@Rezzolla2002], for high enough tangential velocities, the solution will smoothly transition from one wave pattern to another while maintaining the initial states otherwise unmodified. This code allows such transitions to be investigated in both inert and reactive systems.

# References
