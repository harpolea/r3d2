****************
Riemann Problems
****************

The code solves Riemann Problems for the relativistic Euler equations

.. math::

  \partial_t \begin{pmatrix} D \\ S_x \\ S_t \\ \tau \end{pmatrix} + \partial_x \begin{pmatrix} S_x \\ S_x v_x + p \\ S_t v_x \\ (\tau + p) v_x \end{pmatrix} = 0.

For further details on this system of equations, see the `Living Review of Martí and Müller <http://computastrophys.livingreviews.org/Articles/lrca-2015-3/>`_, particularly `section 3.1 <http://computastrophys.livingreviews.org/Articles/lrca-2015-3/articlese3.html#x6-190003.1>`_ for the equations and `section 8.5 <http://computastrophys.livingreviews.org/Articles/lrca-2015-3/articlese8.html#x11-1150008.5>`_ for the solution of the Riemann Problem.

The initial data is piecewise constant: two states :math:`w_{L, R}` are specified, each in terms of `w = (\rho_0, v_x, v_t, \epsilon)`, (the specific rest mass density, normal (x) and tangential (t) velocity components, and the specific internal energy). At :math:`t=0` the data is set by :math:`w_L` for :math:`x<0` and :math:`w_R` for :math:`x>0`. Each state has associated with it an equation of state (EOS) to close the set of equations: the EOS does not need to be the same for each state.
