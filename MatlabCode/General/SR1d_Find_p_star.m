% function w_star = SR1d_Find_p_star( w_l, w_r, p_star_0, eos_options )
%
% Find the value of w_star that solves the Riemann problem.
function p_star = SR1d_Find_p_star( w_l, w_r, p_star_0, eos_options )

[~, ~, P_l, ~, ~, ~, ~, ~] = EOS(w_l(1), w_l(3), 1, eos_options);
[~, ~, P_r, ~, ~, ~, ~, ~] = EOS(w_r(1), w_r(3), 2, eos_options);

fz_options = optimset('Display', 'off');

pmin = min([P_l P_r p_star_0]);
pmax = max([P_l P_r p_star_0]);
p_star = fzero(@SR1d_Find_Delta_v, [0.5*pmin 2.0*pmax], fz_options);

    function Delta_v = SR1d_Find_Delta_v(p_s)

        v_star_l = SR1d_Find_v(w_l, p_s, -1);
        v_star_r = SR1d_Find_v(w_r, p_s,  1);
                
        Delta_v = v_star_l - v_star_r;
        
    end

    function v_star = SR1d_Find_v(known_state, p_s, lr_sign)
         
        w_star = SR1d_GetState(known_state, p_s, lr_sign, eos_options);
        v_star = w_star(2);
        
    end

end

