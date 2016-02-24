R3D2
====

.. image:: https://travis-ci.org/harpolea/r3d2.svg?branch=master
    :target: https://travis-ci.org/harpolea/r3d2
.. image:: http://readthedocs.org/projects/r3d2/badge/?version=latest
    :target: http://r3d2.readthedocs.org/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://codecov.io/github/harpolea/r3d2/coverage.svg?branch=master
    :target: https://codecov.io/github/harpolea/r3d2?branch=master

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

    >>> eos = eos_defn.eos_gamma_law(5.0/3.0)

Set up the left and right states:
::

    >>> U_L = State(rho=1.0, v=0.0, vt=0.0, eps=1.5, eos=eos)
    >>> U_R = State(rho=0.125, v=0.0, vt=0.0, eps=1.2, eos=eos)

Solve the Riemann Problem:
::

    >>> rp = RiemannProblem(U_L, U_R)

The output can be examined for details of the solution and its wave structure. However, the classes are optimized for display in a Jupyter notebook. See the documentation for more detail.

Documentation
-------------

The documentation is available at `<http://r3d2.readthedocs.org>`_ .
