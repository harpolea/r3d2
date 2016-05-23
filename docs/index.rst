.. R3D2 documentation master file, created by
   sphinx-quickstart on Mon Jan 25 10:49:16 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to R3D2's documentation!
================================

R3D2 (Relativistic Reactive Riemann problem solver for Deflagrations and Detonations) solves the Riemann problem for the *relativistic* Euler equation. It also includes the option to include reaction terms, for "infinitely" fast reactions, leading to deflagrations and detonations.

Models of ideal hydrodynamics, where there is no viscosity or dissipation, can have solutions with *discontinuities* such as shocks. A simple case is the Riemann Problem, where two constant states are separated by a barrier. After the barrier is removed the solution develops, with *waves* (such as shocks and rarefactions) separating constant states.

The Riemann Problem has three main uses. Efficient, often approximate, solvers are an integral part of many modern hydrodynamic evolution codes. Second, the exact solution is a standard test for such codes. Finally, the solver can illustrate features of discontinuous solutions in more complex scenarios.

This code is intended for exploring possible solutions and relativistic effects, or for comparing against a compressible code with reactive sources. It is optimized for use with Jupyter notebooks. It is **not** intended for use within a HRSC evolution code: the performance is far too poor, and the assumptions made to extreme.

Installation
------------

A standard::

    python setup.py install

or::

    pip install r3d2

should work.

Usage
-----

Import the equations of state, State class, and Riemann Problem class:
::

    >>> from r3d2 import eos_defns, State, RiemannProblem

Set up an equation of state:
::

    >>> eos = eos_defn.eos_gamma_law(5.0/3.0)

Set up the left and right states:
::

    >>> U_L = State(rho=1.0, v=0.0, vt=0.0, eps=1.5, eos=eos)
    >>> U_R = State(rho=0.125, v=0.0, vt=0.0, eps=1.2, eos=eos)

Solve the Riemann Problem:
::

    >>> rp = RiemannProblem(U_L, U_R)

The output can be examined for details of the solution and its wave structure. For example, the three waves are each built of wave *sections*, which can be examined to check their type, via e.g.

>>> rp.waves[0].wave_sections

and its speed (or the range of speeds) can be examined via

>>> rp.waves[0].wavespeed

The states that the waves separate can be found via, e.g.,

>>> rp.waves[0].q_r

and the detailed values via

>>> rp.waves[0].q_r.state()

However, the classes are optimized for display in a Jupyter notebook. See the documentation for more detail.

Contents
--------

.. toctree::
   :maxdepth: 2

   riemann_problem
   eos_defns.rst
   states
   waves
   tangential_change
   p_v_plots

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
