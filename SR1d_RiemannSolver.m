%% Riemann solver for 1d SR hydro
%
% This is a practice run for doing the SR elastic Riemann solver

clear;
close all;
clc

%% Define the states
% The order is   w = (rho, v, eps)
% We compute   all = (rho, v, eps, p, W, h, cs^2)

t_end = 0.4;

% Whisky case "Simple" (or blast wave 1 from the Living Review)
gamma = 5/3;
w_left  = [10 0 2   ];
w_right = [ 1 0 1e-5];
% Blast wave 2 from the Living Review
% gamma = 5/3;
% w_left  = [ 1 0 1500];
% w_right = [ 1 0 0.015];
% Test 7 from Riemann1d.f90
% gamma = 5/3;
% w_left = [1.0139158702550264 6.57962037903012369e-3 1.0309221446370552e-1];
% w_right = [1 0 0.1];

all_left  = compute_all(gamma, w_left );
all_right = compute_all(gamma, w_right);

%% Initial guess
% At present this is not really used; a better guess will be required for
% elasticity.

p_star_0 = 0.5 * (all_left(4) + all_right(4));

%% Root find

p_star = SR1d_Find_p_star(gamma, all_left, all_right, p_star_0);

%% Compute final states, characteristic speeds etc.

w_star_l = SR1d_GetState(gamma, all_left , p_star, -1);
w_star_r = SR1d_GetState(gamma, all_right, p_star,  1);
all_star_l = compute_all(gamma, w_star_l);
all_star_r = compute_all(gamma, w_star_r);

wave_speeds = SR1d_GetWaveSpeeds(all_left, all_star_l, ...
    all_star_r, all_right);
if (abs(wave_speeds(2) - wave_speeds(1)) < 1e-10)
    fprintf('Left wave is a shock, speed %g.\n', wave_speeds(1));
else
    fprintf('Left wave is a rarefaction, speeds (%g, %g).\n', ...
        wave_speeds(1), wave_speeds(2));
end
fprintf('Contact wave has speed %g.\n', wave_speeds(3));
if (abs(wave_speeds(5) - wave_speeds(4)) < 1e-10)
    fprintf('Right wave is a shock, speed %g.\n', wave_speeds(4));
else
    fprintf('Right wave is a rarefaction, speeds (%g, %g).\n', ...
        wave_speeds(4), wave_speeds(5));
end

%% Produce a plot
% Work on the domain [0, 1] assuming the discontinuity is at 0.5.
% Characteristic variable is \xi = (x - 1/2) / t.
x = linspace(0, 1);
xi = (x - 0.5) / t_end;
w = zeros(length(x), 3);
all = zeros(length(x), 7);

for i = 1:length(xi)

    if (xi(i) < wave_speeds(1))
        w(i, :) = w_left;
    elseif (xi(i) < wave_speeds(2))
        w(i, :) = SR1d_Rarefaction(gamma, xi(i), all_left, 1);
    elseif (xi(i) < wave_speeds(3))
        w(i, :) = w_star_l;
    elseif (xi(i) < wave_speeds(4))
        w(i, :) = w_star_r;
    elseif (xi(i) < wave_speeds(5))
        w(i, :) = SR1d_Rarefaction(gamma, xi(i), all_right, -1);
    else
        w(i, :) = w_right;
    end
    all(i, :) = compute_all(gamma, w(i, :));
end
subplot(1,3,1);plot(x, w(:, 1)); xlabel('x'); ylabel('\rho');
subplot(1,3,2);plot(x, w(:, 2)); xlabel('x'); ylabel('v');
subplot(1,3,3);plot(x, all(:, 4)); xlabel('x'); ylabel('p');

%% Sharper plot

rarefaction_pts = 100;

xi_left = -0.5 / t_end;
xi_right = 0.5 / t_end;
clearvars xi all;
xi = xi_left;
if (xi < wave_speeds(1))
    w = w_left;
elseif (xi < wave_speeds(2))
    w = SR1d_Rarefaction(gamma, xi(i), all_left, 1);
elseif (xi < wave_speeds(3))
    w = w_star_l;
elseif (xi < wave_speeds(4))
    w = w_star_r;
elseif (xi < wave_speeds(5))
    w = SR1d_Rarefaction(gamma, xi(i), all_right, -1);
else
    w = w_right;
end
all = compute_all(gamma, w);

if ((wave_speeds(1) > xi_left) && (wave_speeds(1) < xi_right))
    xi = [xi wave_speeds(1)];
    all = [all; all_left];
end
if ((wave_speeds(2) > xi_left) && (wave_speeds(2) < xi_right))
    if (wave_speeds(2) > wave_speeds(1) + 1e-10)
        xi = [xi linspace(xi(end), wave_speeds(2), rarefaction_pts)];
        for i = 1:100
           w = SR1d_Rarefaction(gamma, xi(end+i-rarefaction_pts), all_left, 1);
           all = [all; compute_all(gamma, w)];
        end
    else
        xi = [xi wave_speeds(2)];
        all = [all; all_star_l];
    end
end
if ((wave_speeds(3) > xi_left) && (wave_speeds(3) < xi_right))
    xi = [xi wave_speeds(3) wave_speeds(3)];
    all = [all; all_star_l; all_star_r];
end
if ((wave_speeds(4) > xi_left) && (wave_speeds(4) < xi_right))
    xi = [xi wave_speeds(4)];
    all = [all; all_star_r];
end
if ((wave_speeds(5) > xi_left) && (wave_speeds(5) < xi_right))
    if (wave_speeds(5) > wave_speeds(5) + 1e-10)
        xi = [xi linspace(wave_speeds(4), wave_speeds(5), rarefaction_pts)];
        for i = 1:100
            w = SR1d_Rarefaction(gamma, xi(end+i-rarefaction_pts), all_right, -1);
           all = [all; compute_all(gamma, w)];
        end
    else
        xi = [xi wave_speeds(5)];
        all = [all; all_right];
    end
end
if ((wave_speeds(5) > xi_right) && wave_speeds(5) > wave_speeds(4) + 1e-10)
    xi = [xi linspace(wave_speeds(4), xi_right, rarefaction_pts)];
    for i = 1:100
        w = SR1d_Rarefaction(gamma, xi(end+i-rarefaction_pts), all_right, -1);
        all = [all; compute_all(gamma, w)];
    end
end
xi = [xi xi_right];
x = (xi') * t_end + 0.5;
all = [all; all_right];
subplot(1,3,1);plot(x, all(:, 1), 'bx-'); xlabel('x'); ylabel('\rho');
subplot(1,3,2);plot(x, all(:, 2), 'bx-'); xlabel('x'); ylabel('v');
subplot(1,3,3);plot(x, all(:, 4), 'bx-'); xlabel('x'); ylabel('p');
