************************************
List of classes and functions
************************************

This is a list of the classes and functions included in ``SR1d.py``.

Modules
========

.. automodule:: SR1d


Classes
================

.. autoclass:: SR1d.State
    :members: __init__, prim, state, wavespeed, vt_from_known, latex_string, _repr_latex_

.. autoclass:: SR1d.Wave
    :members: __init__, mass_flux_squared, solve_shock, solve_rarefaction, solve_deflagration, solve_detonation, plotting_data, latex_string, _repr_latex_

.. autoclass:: SR1d.RP
    :members: __init__, _figure_data, _repr_png_, _repr_svg_, _repr_latex_


Functions
================

.. autofunction:: SR1d.rarefaction_dwdp

.. autofunction:: find_left_state.find_left
