import torch
import scipy.io as sio
import numpy as np
N_t = 64
N_r = 16
nclient = 4
from .helper import *

class loader(object):
    def __init__(self):
        self.__load_dataset()

    def __load_dataset(self):
        H_org = sio.loadmat("data/H_16x64_MIMO_CDL_A_ULA_clean.mat")
        H_ex = H_org['hest']
        H_extracted = np.transpose(H_ex,(2,1,0))
        dft_basis = sio.loadmat("data/dft_basis.mat")
        A_T = dft_basis['A1']/np.sqrt(N_t)
        A_R = dft_basis['A2']/np.sqrt(N_r)
        for i in range(H_ex.shape[2]):
            H_extracted[i] = np.transpose(np.matmul(np.matmul(A_R.conj().T,H_extracted[i].T,dtype='complex64'),A_T))

        img_np_real = np.real(H_extracted)
        img_np_imag = np.imag(H_extracted)

        mu_real = np.mean(img_np_real,axis=0)
        mu_imag = np.mean(img_np_imag,axis=0)
        std_real = np.std(img_np_real,axis=0)
        std_imag = np.std(img_np_imag,axis=0)

        img_np_real = (img_np_real - mu_real)/std_real
        img_np_imag = (img_np_imag - mu_imag)/std_imag

        img_np = np.zeros((H_ex.shape[2],2,N_t,N_r))
        img_np[:,0,:,:] = img_np_real
        img_np[:,1,:,:] = img_np_imag
        self.train_CDL_A = img_np
        self.mu_real = mu_real
        self.mu_imag = mu_imag
        self.std_real = std_real
        self.std_imag = std_imag
        #A path loss realization from TR 38.901 InH-Office for determining UE noise variance in
        #FedAmbGAN training.
        ls_path_loss = np.array([-71.27021014, -76.15337504, -72.64004475, -73.47226972])
        self.channel_coeff = 10**(ls_path_loss/20)
