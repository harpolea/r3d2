% function w = SR1d_Rarefaction( gamma, xi, w_known, lr_sign )
%
% Compute the state inside a rarefaction wave.
function w = SR1d_Rarefaction( gamma, xi, known, lr_sign )

rho_known = known(1);
v_known = known(2);
%p_known = known(4);
cs_known = sqrt(known(7));
b = sqrt(gamma - 1);
c = (b + cs_known) / (b - cs_known);
d = -lr_sign * b / 2;
k = (1 + xi) / (1 - xi);
l = c * (k^d);
v_wave = ( (1 - v_known) / (1 + v_known) )^d;

cs = fzero(@SR1d_Rarefaction_cs, cs_known);

v = (xi + lr_sign * cs) / (1 + lr_sign * xi * cs);
rho = rho_known * ( ( cs^2 * ((gamma - 1) - cs_known^2) ) / ...
    ( cs_known^2 * ((gamma - 1) - cs^2) ) )^(1 / (gamma - 1));
p = cs^2 * (gamma - 1) * rho / (gamma *((gamma - 1) - cs^2));
eps = p / (rho * (gamma - 1));

w = [rho v eps];

    function fcs = SR1d_Rarefaction_cs(cs_guess)
        fcs = l * v_wave * (1 + lr_sign * cs_guess)^d * (cs_guess - b) + ...
            (1 - lr_sign * cs_guess)^d * (cs_guess + b);
    end
end

