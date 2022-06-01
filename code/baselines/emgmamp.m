clc; clear;
addpath('EMGMAMP')
addpath('main')
H_ex = load('../../data/H_16x64_MIMO_CDL_A_ULA_test.mat').hest;
N_t = 64;
N_r = 16;
N_s = 16;
N_p_vec = [25];
SNR_vec = -15:5:15;
nrepeat = 50;
ntest = 20;
j = sqrt(-1);
qpsk_constellation = (1/sqrt(2))*[1+1j,1-1j,-1+1j,-1-1j];
identity = eye(N_r);
A1 = dftmtx(N_t)/sqrt(N_t); 
A2 = dftmtx(N_r)/sqrt(N_r); 
nmse_arr = zeros(length(SNR_vec),ntest);
Nbit_r = 2;
Nbit_t = 6;

for b = 1:length(N_p_vec)
    N_p = N_p_vec(b);
    for a = 1:nrepeat
        disp(a);
        pilot_sequence_ind = randi([1,4],N_s,N_p);
        pilot_sequence = qpsk_constellation(pilot_sequence_ind);
        precoder_training = training_precoder(N_t,N_s,Nbit_t);
        W = training_combiner(N_r,N_s,Nbit_r);       
        A = kron(transpose(precoder_training*pilot_sequence),W);
        A_H = A';
        A_tx = kron(transpose(precoder_training*pilot_sequence),identity);
        A_sp = kron(transpose((A1')*precoder_training*pilot_sequence),W*A2);
        for i = 1:ntest
            vec_H_single = reshape(H_ex(:,:,i),N_r*N_t,1);
            signal = A_tx*vec_H_single;
            E_s = signal.*conj(signal);
            noise = (randn(N_r*N_p,1)+j*randn(N_r*N_p,1));
            for k = 1:length(SNR_vec)
                SNRdB = SNR_vec(k);           % SNR in dB.
                std_dev = (1/(10^(SNRdB/20)))*sqrt(E_s);
                noise_matrix = (1/sqrt(2))*W*reshape(std_dev.*noise,[N_r,N_p]);
                y = A*vec_H_single + reshape(noise_matrix,[N_r*N_p,1]);
                [xhat, EMfin] = EMGMAMP(y,A_sp);
                H_est = reshape(A2*reshape(xhat,N_r,N_t)*A1',N_r*N_t,1);
                % Display the MSE
                nmseGAMP = (norm(vec_H_single-H_est)/norm(vec_H_single))^2;
                nmse_arr(k,i) = nmse_arr(k,i) + nmseGAMP; 
            end
        end
    end
end
nmse_arr = nmse_arr/(nrepeat);
