% function w = SR1d_GetState( known_state, p_star, lr_sign, eos_options )
%
% Given the known state and the pressure the other side of the wave, 
% compute all the state information
function w = SR1d_GetState( known_state, p_star, lr_sign, eos_options )

if (lr_sign == -1)
  wave_p = 1;
else
  wave_p = 2;
end

[~, ~, p_known, ~, ~, ~, ~, ~] = EOS(known_state(1), known_state(3), ...
                                     wave_p, eos_options);

if (p_star >= p_known) % Shock wave
    
    [~, w] = Shock(known_state, p_star, lr_sign, eos_options);
    
else % Rarefaction wave
    
    sol_ivp = Rarefaction(known_state, p_star, lr_sign, eos_options);
    w = sol_ivp.y(:, end);
    
end

end

