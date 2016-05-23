R3D2
====

.. image:: https://travis-ci.org/harpolea/r3d2.svg?branch=master
    :target: https://travis-ci.org/harpolea/r3d2
.. image:: https://readthedocs.org/projects/r3d2/badge/?version=latest
    :target: http://r3d2.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://codecov.io/github/harpolea/r3d2/coverage.svg?branch=master
    :target: https://codecov.io/github/harpolea/r3d2?branch=master
.. image:: https://zenodo.org/badge/21891/harpolea/r3d2.svg
    :target: https://zenodo.org/badge/latestdoi/21891/harpolea/r3d2

Relativistic Reactive Riemann problem solver for Deflagrations and Detonations
------------------------------------------------------------------------------

This extends standard solutions of the relativistic Riemann Problem to include a reaction term.

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

    >>> eos = eos_defns.eos_gamma_law(5.0/3.0)

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

Documentation
-------------

The documentation is available at `<http://r3d2.readthedocs.org>`_ .

Contributing
------------

Please open a pull request at `<https://github.com/harpolea/r3d2/pulls>`_ .

Support
-------

Please open an issue at `<https://github.com/harpolea/r3d2/issues>`_ .
