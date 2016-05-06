.. R3D2 documentation master file, created by
   sphinx-quickstart on Mon Jan 25 10:49:16 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to R3D2's documentation!
================================

R3D2 (Relativistic Reactive Riemann problem solver for Deflagrations and Detonations) solves the Riemann problem for the *relativistic* Euler equation. It also includes the option to include reaction terms, for "infinitely" fast reactions, leading to deflagrations and detonations.

This code is intended for exploring possible solutions and relativistic effects, or for comparing against a compressible code with reactive sources. It is optimized for use with Jupyter notebooks. It is **not** intended for use within a HRSC code: the performance is far too poor, and the assumptions made to extreme.

Contents:

.. toctree::
   :maxdepth: 2

   riemann_problem
   eos_defns.rst
   states
   waves
   tangential_change

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
