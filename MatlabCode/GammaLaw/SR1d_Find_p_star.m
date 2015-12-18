% function w_star = SR1d_Find_p_star( gamma, w_l, w_r, p_star_0 )
%
% Find the value of w_star that solves the Riemann problem.
function p_star = SR1d_Find_p_star( gamma, w_l, w_r, p_star_0 )

pmin = min([w_l(4) w_r(4) p_star_0]); pmax = max([w_l(4) w_r(4) p_star_0]);
p_star = fzero(@SR1d_Find_Delta_v, [0.5*pmin 2*pmax]);

    function Delta_v = SR1d_Find_Delta_v(p_s)

        v_star_l = SR1d_Find_v(w_l, p_s, -1);
        v_star_r = SR1d_Find_v(w_r, p_s,  1);
                
        Delta_v = v_star_l - v_star_r;
        
    end

    function v_star = SR1d_Find_v(known_state, p_s, lr_sign)
        
        w_star = SR1d_GetState(gamma, known_state, p_s, lr_sign);
        v_star = w_star(2);
        
    end

end

