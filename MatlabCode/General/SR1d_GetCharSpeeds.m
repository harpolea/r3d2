function lambda = SR1d_GetCharSpeeds(w, wave_p, eos_options)

[cs, ~, ~, ~, ~, ~, ~, ~] = EOS(w(1), w(3), wave_p, eos_options);

lambda(1) = (w(2) - cs) / (1 - w(2) * cs);
lambda(2) = w(2);
lambda(3) = (w(2) + cs) / (1 + w(2) * cs);

end