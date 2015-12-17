% all = compute_all(gamma, w)
%
% Convert from the basic primitive variables w = (rho, v, eps) to the full
% set all = (rho, v, eps, p, W, h, cs^2)
%
% Should add error checking.

function all = compute_all( gamma, w )

rho = w(1); v = w(2); eps = w(3);
p = (gamma - 1) * rho * eps;
W_lorentz = 1 / sqrt( 1 - v^2 );
h = 1 + eps + p / rho;
cs2 = gamma * p / (rho * h);

all = [rho v eps p W_lorentz h cs2];

end

