function sol_ivp = Rarefaction(w_start, P_end, lr_sign, eos_options)

if (lr_sign == -1)
  wave_p = 1;
else
  wave_p = 2;
end

[~, ~, P_start, ~, ~, ~, ~, ~] = EOS(w_start(1), w_start(3), ...
                                     wave_p, eos_options);

interval = [P_start P_end];

ivp_options = odeset('Events', @Rarefaction_IVP_events);

sol_ivp = ode45(@Rarefaction_IVP, interval, w_start, ivp_options);

    function dwdp = Rarefaction_IVP(P, w)
        
        dwdp = zeros(3, 1);
        rho = w(1); v = w(2); epsilon = w(3);
        
        [cs, h, ~, ~, ~, ~, ~, ~] = EOS(rho, epsilon, wave_p, ...
                                        eos_options);
        
        Wlorentz = 1./sqrt(1 - v.^2);
        
        dwdp(1) = 1./(h.*cs.^2);
        dwdp(2) = lr_sign./(rho.*h.*Wlorentz.^2.*cs);
        dwdp(3) = P./(rho.^2.*h.*cs.^2);
               
    end

    function [value,isterminal,direction] = Rarefaction_IVP_events(P, w)
        
        rho = w(1); epsilon = w(3);
        
        [cs, h, ~, dPdrho, dPdeps, d2Pd2rho, d2Pdrhoeps, d2Pd2eps] = ...
            EOS(rho, epsilon, wave_p, eos_options);
        
        value = d2Pd2rho.*rho.^4 - ...
            rho.^2.*(cs.^2.*rho - dPdeps).*(dPdrho + P.*dPdeps./rho.^2) + ...
            2*P.*d2Pdrhoeps.*rho.^2 - 2*rho.*P.*dPdeps + ...
            2*cs.^2.*rho.^3.*(1+epsilon) - 2*cs.^4.*rho.^3.*h + ...
            P.^2.*d2Pd2eps;
        isterminal = 0;
        direction = 0;
        
    end
end