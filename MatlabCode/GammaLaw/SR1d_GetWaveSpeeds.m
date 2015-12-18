function wave_speeds = SR1d_GetWaveSpeeds( l, s_l, s_r, r )

wave_speeds = zeros(5, 1);

p_l   =   l(4);
p_s_l = s_l(4);
p_s_r = s_r(4);
p_r   =   r(4);

h_l   =   l(6);
h_s_l = s_l(6);
h_r   =   r(6);
h_s_r = s_r(6);

rho_l   =   l(1);
rho_s_l = s_l(1);
rho_r   =   r(1);
rho_s_r = s_r(1);

v_l   =   l(2);
v_s_l = s_l(2);
v_r   =   r(2);
v_s_r = s_r(2);

cs_l   = sqrt(  l(7));
cs_s_l = sqrt(s_l(7));
cs_r   = sqrt(  r(7));
cs_s_r = sqrt(s_r(7));
% Left wave
if (p_s_l > p_l) % Shock
    w2 = l(5)^2;
    j = -sqrt( (p_s_l - p_l) / (h_l / rho_l - h_s_l / rho_s_l) );
    a = j^2 + rho_l^2 * w2;
    b = -v_l * rho_l^2 * w2;
    wave_speeds(1) = (-b - j^2 * sqrt(1 + (rho_l / j)^2)) / a;
    wave_speeds(2) = wave_speeds(1);
else % Rarefaction
    wave_speeds(1) = (v_l - cs_l) / (1 - v_l * cs_l);
    wave_speeds(2) = (v_s_l - cs_s_l) / (1 - v_s_l * cs_s_l);
end

% Contact
wave_speeds(3) = s_l(2);

% Right wave
if (p_s_r > p_r) % Shock
    w2 = r(5)^2;
    j = sqrt( (p_s_r - p_r) / (h_r / rho_r - h_s_r / rho_s_r) );
    a = j^2 + rho_r^2 * w2;
    b = -v_r * rho_r^2 * w2;
    wave_speeds(4) = (-b + j^2 * sqrt(1 + (rho_r / j)^2)) / a;
    wave_speeds(5) = wave_speeds(4);
else % Rarefaction
    wave_speeds(4) = (v_s_r - cs_s_r) / (1 - v_s_r * cs_s_r);
    wave_speeds(5) = (v_r - cs_r) / (1 - v_r * cs_r);
end

end

