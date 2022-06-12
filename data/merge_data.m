N_r = 16;
N_t = 64;
nSamples = 6000;
hest = zeros(N_r,N_t,nSamples);
for m = 1:6000
    fname = sprintf('data_16x64_A/hA%d.mat',m);
    hest_samples = load(fname);
    hest(:,:,m) = reshape(hest_samples.a,[N_r,N_t,1]);
end
save('H_16x64_MIMO_CDL_A_ULA_clean.mat','hest')