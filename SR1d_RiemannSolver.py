"""
Riemann solver for 1d SR hydro

This is a practice run for doing the SR elastic Riemann solver
"""
import numpy as np
import matplotlib.pyplot as plt

# Define the states
# The order is   w = (rho, v, eps)
# We compute   q = (rho, v, eps, p, W, h, cs**2)

t_end = 0.4

# Whisky case "Simple" (or blast wave 1 from the Living Review)
gamma = 5./3.
w_left  = [10., 0., 2.   ]
w_right = [ 1., 0., 1.e-5]
# Blast wave 2 from the Living Review
# gamma = 5/3
# w_left  = [ 1 0 1500]
# w_right = [ 1 0 0.015]
# Test 7 from Riemann1d.f90
# gamma = 5/3
# w_left = [1.0139158702550264 6.57962037903012369e-3 1.0309221446370552e-1]
# w_right = [1 0 0.1]

q_left  = compute_q(gamma, w_left )
q_right = compute_q(gamma, w_right)

## Initial guess
# At present this is not reqy used a better guess will be required for
# elasticity.

p_star_0 = 0.5 * (q_left[3] + q_right[3])

## Root find

p_star = SR1d_Find_p_star(gamma, q_left, q_right, p_star_0)

## Compute final states, characteristic speeds etc.

w_star_l = SR1d_GetState(gamma, q_left , p_star, -1.)
w_star_r = SR1d_GetState(gamma, q_right, p_star,  1.)
q_star_l = compute_q(gamma, w_star_l)
q_star_r = compute_q(gamma, w_star_r)

wave_speeds = SR1d_GetWaveSpeeds(q_left, q_star_l, \
    q_star_r, q_right)
if (abs(wave_speeds[1] - wave_speeds[0]) < 1.e-10):
    print('Left wave is a shock, speed #g.\n', wave_speeds[0])
else:
    print('Left wave is a rarefaction, speeds (#g, #g).\n',
        wave_speeds[0], wave_speeds[1])

print('Contact wave has speed #g.\n', wave_speeds[2])
if (abs(wave_speeds[4] - wave_speeds[3]) < 1.e-10):
    print('Right wave is a shock, speed #g.\n', wave_speeds[3])
else:
    print('Right wave is a rarefaction, speeds (#g, #g).\n',
        wave_speeds[3], wave_speeds[4])


## Produce a plot
# Work on the domain [0, 1] assuming the discontinuity is at 0.5.
# Characteristic variable is \xi = (x - 1/2) / t.
x = np.linspace(0., 1.)
xi = (x - 0.5) / t_end
w = np.zeros((length(x), 3))
q = np.zeros((length(x), 7))

for i in range(len(xi)):

    if (xi(i) < wave_speeds[0]):
        w[i, :] = w_left
    elif (xi(i) < wave_speeds[1]):
        w[i, :] = SR1d_Rarefaction(gamma, xi(i), q_left, 1)
    elif (xi(i) < wave_speeds[2]):
        w[i, :] = w_star_l
    elif (xi(i) < wave_speeds[3]):
        w[i, :] = w_star_r
    elif (xi(i) < wave_speeds[4]):
        w[i, :] = SR1d_Rarefaction(gamma, xi(i), q_right, -1)
    else:
        w[i, :] = w_right

    q[i, :] = compute_q(gamma, w[i, :])


# plotting
plt.clf()
plt.rc("font", size=12)
fig, axes = plt.subplots(nrows=1, ncols=3, num=1)

ax = axes.flat[0]
ax.plot(x, w[:, 0])
ax.xlabel('$x$')
ax.ylabel('$\rho$')

ax = axes.flat[1]
ax.plot(x, w[:, 1])
ax.xlabel('$x$')
ax.ylabel('$v$')

ax = axes.flat[2]
ax.plot(x, q[:, 3])
ax.xlabel('$x$')
ax.ylabel('$p$')

## Sharper plot

rarefaction_pts = 100

xi_left = -0.5 / t_end
xi_right = 0.5 / t_end

xi = xi_left
if (xi < wave_speeds[0]):
    w = w_left
elif (xi < wave_speeds[1]):
    w = SR1d_Rarefaction(gamma, xi(i), q_left, 1.)
elif (xi < wave_speeds[2]):
    w = w_star_l
elif (xi < wave_speeds[3]):
    w = w_star_r
elif (xi < wave_speeds[4]):
    w = SR1d_Rarefaction(gamma, xi(i), q_right, -1.)
else:
    w = w_right

q = compute_q(gamma, w)

if ((wave_speeds[0] > xi_left) and (wave_speeds[0] < xi_right)):
    xi = [xi, wave_speeds[0]]
    q = [q, q_left]

if ((wave_speeds[1] > xi_left) and (wave_speeds[1] < xi_right)):
    if (wave_speeds[1] > wave_speeds[0] + 1.e-10):
        xi = [xi, np.linspace(xi(end), wave_speeds[1], rarefaction_pts)]
        for i in range(100):
            w = SR1d_Rarefaction(gamma, xi(end+i-rarefaction_pts), q_left, 1.)
            q = [q, compute_q(gamma, w)]

    else:
        xi = [xi, wave_speeds[1]]
        q = [q, q_star_l]


if ((wave_speeds[2] > xi_left) and (wave_speeds[2] < xi_right)):
    xi = [xi, wave_speeds[2], wave_speeds[2]]
    q = [q, q_star_l, q_star_r]

if ((wave_speeds[3] > xi_left) and (wave_speeds[3] < xi_right)):
    xi = [xi, wave_speeds[3]]
    q = [q, q_star_r]

if ((wave_speeds[4] > xi_left) and (wave_speeds[4] < xi_right)):
    if (wave_speeds[4] > wave_speeds[4] + 1.e-10):
        xi = [xi, np.linspace(wave_speeds[3], wave_speeds[4], rarefaction_pts)]
        for i in range(100):
            w = SR1d_Rarefaction(gamma, xi(end+i-rarefaction_pts), q_right, -1.)
            q = [q, compute_q(gamma, w)]

    else:
        xi = [xi, wave_speeds[4]]
        q = [q, q_right]

if ((wave_speeds[4] > xi_right) and wave_speeds[4] > wave_speeds[3] + 1.e-10):
    xi = [xi, np.linspace(wave_speeds[3], xi_right, rarefaction_pts)]
    for i in range(100):
        w = SR1d_Rarefaction(gamma, xi(end+i-rarefaction_pts), q_right, -1.)
        q = [q, compute_q(gamma, w)]

# plotting
x = xi * t_end + 0.5
x_right = xi_right * t_end + 0.5

fig, axes = plt.subplots(nrows=1, ncols=3, num=1)

ax = axes.flat[0]
ax.plot(x, q[:, 0], 'bx', x_right, q_right[:, 0], 'k-')
ax.xlabel('$x$')
ax.ylabel('$\rho$')

ax = axes.flat[1]
ax.plot(x, q[:, 1], 'bx', x_right, q_right[:, 1], 'k-')
ax.xlabel('$x$')
ax.ylabel('$v$')

ax = axes.flat[2]
ax.plot(x, q[:, 3], 'bx', x_right, q_right[:, 3], 'k-')
ax.xlabel('$x$')
ax.ylabel('$p$')
