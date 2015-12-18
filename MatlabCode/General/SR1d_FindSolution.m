function [xi wnew] = SR1d_FindSolution(w, p_star, lr_sign, eos_options)

if (lr_sign == -1)
  wave_p = 1;
else
  wave_p = 2;
end

if (lr_sign == -1)
    lr_string = 'Left';
else
    lr_string = 'Right';
end

% First find the new states and speeds.

w_star = SR1d_GetState(w , p_star, lr_sign, eos_options);

[cs     , h     , p, ~, ~, ~, ~, ~] = EOS(w(1)     , w(3)     , ...
                                          wave_p, eos_options);
[cs_star, h_star, ~, ~, ~, ~, ~, ~] = EOS(w_star(1), w_star(3), ...
                                          wave_p, eos_options);

rho      =      w(1);
v        =      w(2);
v_star   = w_star(2);

if (p_star > p) % Shock
    
    if (lr_sign == -1)
        xi = [];
        wnew = [];
    else
        xi = v_star;
        wnew = w_star;
    end
    
    W2 = 1/(1-v^2);
    dP = p_star - p;
    dh2 = h_star^2 - h^2;
    
    j = sqrt( -dP / (dh2 / dP - 2 * h / rho) );
    
    v_shock = (rho^2*W2*v + lr_sign*j^2* ...
        sqrt(1 + rho^2*W2*(1-v^2)/j^2)) / ...
        (rho^2*W2+j^2);
    fprintf('%s wave is a shock, speed %g.\n', lr_string, v_shock);
    
    if (lr_sign == -1)
        xi = v_shock;
        wnew = w;
        xi = [xi v_shock];
        wnew = [wnew w_star];
        xi = [xi v_star];
        wnew = [wnew w_star];
    else
        xi = [xi v_shock];
        wnew = [wnew w_star];
        xi = [xi v_shock];
        wnew = [wnew w];
    end
        
    
else % Rarefaction
    
    if (lr_sign == -1)
        xi = [];
        wnew = [];
    else
        xi = v_star;
        wnew = w_star;
    end
    
    wave_speed      = (v      + lr_sign * cs     ) / ...
        (1 + lr_sign * v      * cs     );
    wave_speed_star = (v_star + lr_sign * cs_star) / ...
        (1 + lr_sign * v_star * cs_star);
    
    if (lr_sign == -1)
        fprintf('%s wave is a rarefaction, speeds (%g, %g).\n', ...
            lr_string, wave_speed     , wave_speed_star);
    else
        fprintf('%s wave is a rarefaction, speeds (%g, %g).\n', ...
            lr_string, wave_speed_star, wave_speed     );
    end
    
    solivp = Rarefaction(w, p_star, lr_sign, eos_options);
    p_points = linspace(p, p_star);
    w_raref = deval(solivp, p_points);
    xi_raref = zeros(size(p_points));
    for i = 1:length(p_points)
        lambda = SR1d_GetCharSpeeds(w_raref(:, i), wave_p, eos_options);
        xi_raref(i) = lambda(2 + lr_sign);
    end
    if (lr_sign == 1) % Transpose solution for plotting
        xi_raref = xi_raref(end:-1:1);
        w_raref = w_raref(:,end:-1:1);
    end
    xi = [xi xi_raref];
    wnew = [wnew w_raref];
    if (lr_sign == -1)
        xi = [xi v_star];
        wnew = [wnew w_star];
    end
    
end

end