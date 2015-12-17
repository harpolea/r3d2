%% Riemann solver for 1d SR hydro
%
% This is a general 1d (no tangential velocity) Riemann solver
% Modify the EOS function to get more generality.

clear;
close all;
clc

%% Define the states
% The order is   w = (rho, v, eps)
% We compute   all = (rho, v, eps, p, W, h, cs^2)

t_end = 0.4;

eos_options.eos = 'Gamma Law';
eos_options.gamma = 5/3;
% Blast wave 1 from the Living Review
% w_left  = [10; 0; 2   ];
% w_right = [ 1; 0; 1e-5];
% Blast wave 2 from the Living Review
% w_left  = [ 1; 0; 1500 ];
% w_right = [ 1; 0; 0.015];
% Relativistic Sod test
w_left  = [ 1; 0; 1.5 ];
w_right = [ 0.125; 0; 0.1*1.5/0.125];

% eos_options.eos = 'Piecewise Polytrope';
% eos_options.gamma = [1.3367 2.4];
% eos_options.rho_transition = 4e-4;
% eos_options.k = [1.2217 5000];
% eos_options.gamma_th = 5/3;
% w_left  = [1e-3; 0; 1.5];
% w_right = [2e-3; 0; 1.5];

eos_options.eos = 'Multi Gamma Law';
eos_options.gamma = [1.4 1.67];
% Mild shock-interface test from STM+IH
% Note potential typo in left pressure!
w_left  = [ 1.3346; 0.1837; 1.57 / 1.3346 / 0.4 ];
w_right = [ 0.1379; 0; 1 / 0.1379 / 0.67];

[~, ~, P_l, ~, ~, ~, ~, ~] = EOS(w_left(1) , w_left(3) , 1, eos_options);
[~, ~, P_r, ~, ~, ~, ~, ~] = EOS(w_right(1), w_right(3), 2, eos_options);


%% Initial guess
% At present this is not really used; a better guess will be required for
% elasticity.

p_star_0 = 0.5 * (P_l + P_r);

%% Root find

p_star = SR1d_Find_p_star(w_left, w_right, p_star_0, eos_options);

%% Plot result using P.

%Start
x = -0.5;
w = w_left;

% Left wave and edge of contact
[xinew wnew] = SR1d_FindSolution(w_left, p_star, -1, eos_options);
x = [x xinew * t_end];
w = [w wnew];
left_pts = length(x);
% Right wave and edge of contact
[xinew wnew] = SR1d_FindSolution(w_right, p_star, 1, eos_options);
x = [x xinew * t_end];
w = [w wnew];

% End
x = [x 0.5];
w = [w w_right];

% Now plot

subplot(1,3,1);plot(x, w(1, :), 'bx-'); xlabel('x'); ylabel('\rho');
subplot(1,3,2);plot(x, w(2, :), 'bx-'); xlabel('x'); ylabel('v');
subplot(1,3,3);plot(x, w(3, :), 'bx-'); xlabel('x'); ylabel('\epsilon');

% More plots

auxl = zeros(5, length(x));
for i = 1:length(x)
    if (i <= left_pts)
        [auxl(1, i), auxl(2, i), auxl(3, i), ~, ~, ~, ~, ~] = ...
            EOS(w(1, i), w(3, i), 1, eos_options);
        lambda = SR1d_GetCharSpeeds(w(:, i), 1, eos_options);
    else
        [auxl(1, i), auxl(2, i), auxl(3, i), ~, ~, ~, ~, ~] = ...
            EOS(w(1, i), w(3, i), 2, eos_options);
        lambda = SR1d_GetCharSpeeds(w(:, i), 2, eos_options);
    end
    auxl(4, i) = lambda(1);
    auxl(5, i) = lambda(3);
end
figure
subplot(2,2,1);plot(x, auxl(1, :), 'bx-'); xlabel('x'); ylabel('c_s');
subplot(2,2,2);plot(x, auxl(2, :), 'bx-'); xlabel('x'); ylabel('h');
subplot(2,2,3);plot(x, auxl(3, :), 'bx-'); xlabel('x'); ylabel('P');
subplot(2,2,4);plot(x, auxl(4, :), 'bx-',x, auxl(5, :), 'go--'); xlabel('x'); ylabel('\lambda_{\pm}');




