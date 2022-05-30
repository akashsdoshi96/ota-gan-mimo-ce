function W = training_combiner(N_r, N_s, Nbit_r)
j = sqrt(-1);
angles_r = linspace(0,2*pi,2^Nbit_r + 1);
angle_index = reshape(randsample(length(angles_r)-1,N_r*N_s,true),[N_r,N_s]);
W = (1/sqrt(N_r))*exp(j*angles_r(angle_index))';
end