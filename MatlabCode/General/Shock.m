function [v_shock, w_end] = Shock(w_start, P_end, sign, eos_options)

if (sign == -1)
  wave_p = 1;
else
  wave_p = 2;
end

w_end = zeros(size(w_start));

[~, h_start, P_start, ~, ~, ~, ~, ~] = EOS(w_start(1), w_start(3), ...
                                           wave_p, eos_options);

if (P_end - P_start < 1e-10)
    
    w_end = w_start;
    v_shock = w_start(2);
    
else
    
    rho_start = w_start(1);
    v_start = w_start(2);
    Wlorentz_start = 1 ./ sqrt(1 - v_start.^2);
    
    fsolve_options = optimset('Display', 'off');
    
    rhoeps = fsolve(@Shock_root, [w_start(1) w_start(3)], fsolve_options);
    
    w_end(1) = rhoeps(1);
    w_end(3) = rhoeps(2);
    
    dP = P_end - P_start;
    h_end = 1 + w_end(3) + P_end / w_end(1);
    dh2 = h_end^2 - h_start^2;
    
    j = sqrt( -dP / (dh2 / dP - 2 * h_start / rho_start) );
    
    v_shock = (rho_start^2*Wlorentz_start^2*v_start + sign*j^2* ...
        sqrt(1 + rho_start^2*Wlorentz_start^2*(1-v_start^2)/j^2)) / ...
        (rho_start^2*Wlorentz_start^2+j^2);
    
    Wlorentz_shock = 1 / sqrt(1 - v_shock^2);
    
    w_end(2) = (h_start * Wlorentz_start * v_start + sign * dP * Wlorentz_shock / j) / ...
        (h_start * Wlorentz_start + dP * (1 / rho_start / Wlorentz_start + ...
        sign * v_start * Wlorentz_shock / j));
    
    w_end(3) = h_end - 1 - P_end / w_end(1);
end

    function dw = Shock_root(w0)
        [~, h0, P0, ~, ~, ~, ~, ~] = EOS(w0(1), w0(2), wave_p, eos_options);
        dw(1) = P_end - P0;
        dw(2) = (h0^2 - h_start^2) - (h0/w0(1) + h_start/w_start(1))*(P0 - P_start);
    end
end