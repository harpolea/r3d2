% The EOS for a standard ideal gas
function [cs, h, P, dPdrho, dPdeps, d2Pd2rho, d2Pdrhoeps, d2Pd2eps] = ...
    EOS(rho, eps, wave_p, options)

if (strcmp(options.eos, 'Gamma Law'))
    
    gamma = options.gamma;
    
    P = (gamma - 1) * rho.*eps;
    h = 1 + eps + P./rho;
    
    dPdeps = (gamma - 1) * rho;
    dPdrho = (gamma - 1) * eps;
    cs = sqrt((dPdrho + P.*dPdeps./rho.^2)./h);
    
    d2Pd2rho = 0;
    d2Pdrhoeps = (gamma - 1);
    d2Pd2eps = 0;

elseif (strcmp(options.eos, 'Multi Gamma Law'))
    
    gamma = options.gamma;
    
    P = (gamma(wave_p) - 1) * rho.*eps;
    h = 1 + eps + P./rho;
    
    dPdeps = (gamma(wave_p) - 1) * rho;
    dPdrho = (gamma(wave_p) - 1) * eps;
    cs = sqrt((dPdrho + P.*dPdeps./rho.^2)./h);
    
    d2Pd2rho = 0;
    d2Pdrhoeps = (gamma(wave_p) - 1);
    d2Pd2eps = 0;
    
elseif (strcmp(options.eos, 'Piecewise Polytrope'))
    
    gamma = options.gamma;
    gamma_th = options.gamma_th;
    rho_transition = options.rho_transition;
    k = options.k;
    
    if (rho < rho_transition)
        p_cold = k(1) * rho^gamma(1);
        eps_cold = p_cold / rho / (gamma(1) - 1);
    else
        p_cold = k(2) * rho^gamma(2);
        eps_cold = p_cold / rho / (gamma(2) - 1) - ...
            k(2) * rho_transition^(gamma(2) - 1) + ...
            k(1) * rho_transition^(gamma(1) - 1);
    end
    
    p_th = max(0, (gamma_th - 1) * rho * (eps - eps_cold));
    
    P = p_cold + p_th;
    
    dPdrho = max(0,(gamma_th - 1) * (eps - eps_cold - p_cold / rho));
    if (rho < rho_transition)
        dPdrho = dPdrho + k(1) * gamma(1) * rho^(gamma(1)-1);
        d2Pd2rho = 0;
    else
        dPdrho = dPdrho + k(2) * gamma(2) * rho^(gamma(2)-1);
        d2Pd2rho = 0;
    end
    
    dPdeps = (gamma_th - 1) * rho;
    d2Pd2eps = 0;
    d2Pdrhoeps = (gamma_th - 1);
    
    h = 1 + eps + P / rho;
    cs = sqrt((dPdrho + P.*dPdeps./rho.^2)./h);
     
else
    
    error('Equation of state %s unrecognized!', eos.options);
    
end

end