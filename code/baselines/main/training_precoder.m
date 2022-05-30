function F = training_precoder(N_t, N_s, Nbit_t)
j = sqrt(-1);
angles_t = linspace(0,2*pi,2^Nbit_t + 1);
angle_index = reshape(randsample(length(angles_t)-1,N_t*N_s,true),[N_t,N_s]);
F = (1/sqrt(N_t))*exp(j*angles_t(angle_index));
end