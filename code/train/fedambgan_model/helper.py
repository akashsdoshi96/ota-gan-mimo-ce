import torch
import torch.nn
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data
from torch.nn import functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import sys
import scipy.io as sio
import pdb
dtype = torch.cuda.FloatTensor
lambda_gp = 10
batch_size = 50

#Wireless Parameters
noise_var = 10**(-11.5) 
lr=5e-5
N_t = 64
N_r = 16
latent_dim = 65
length = int(N_t/4)
breadth = int(N_r/4)
n_client = 4
n_d = 20

dft_basis = sio.loadmat("data/dft_basis.mat")
A = np.load('data/A_mat_1024.npy')
FS_T = np.load('data/FS_T_1024.npy')
W_arr = np.load('data/W_1024.npy')

A_T = dft_basis['A1']/np.sqrt(N_t)
A_R = dft_basis['A2']/np.sqrt(N_r)

alpha = 0.25
N_p = int(alpha*N_t)
M = N_p*N_r
qpsk_constellation = (1/np.sqrt(2))*np.array([1+1j,1-1j,-1+1j,-1-1j])
identity = np.identity(N_p)
N_s = N_r
Nbit_t = 6
Nbit_r = 2
angles_t = np.linspace(0,2*np.pi,2**Nbit_t,endpoint=False)
angles_r = np.linspace(0,2*np.pi,2**Nbit_r,endpoint=False)
    
A_T_R = np.kron(A_T.conj(),A_R)
A_R_T = np.kron(np.transpose(A_T),np.matrix(A_R).getH())
A_T_R_real = dtype(np.real(A_T_R))
A_T_R_imag = dtype(np.imag(A_T_R))

#Construct B and tx_kron
B = np.zeros((N_t*N_r,N_t*N_r),dtype='complex64')
tx_kron = np.zeros((N_t*N_r,N_t*N_r),dtype='complex64')
for i in range(4):
    B[N_p*N_r*i:N_p*N_r*(i+1),N_p*N_r*i:N_p*N_r*(i+1)] = np.kron(identity,W_arr[i])
    tx_kron[N_p*N_r*i:N_p*N_r*(i+1),:] = np.kron(FS_T[i],np.identity(N_r))
A_real = dtype(np.real(tx_kron))
A_imag = dtype(np.imag(tx_kron))

A_mat = np.matmul(A_R_T,np.matmul(np.linalg.inv(A),B))
A_mat_real = dtype(np.real(A_mat))
A_mat_imag = dtype(np.imag(A_mat))

def training_precoder(N_t,N_s):
    angle_index = np.random.choice(len(angles_t),(N_t,N_s))
    return (1/np.sqrt(N_t))*np.exp(1j*angles_t[angle_index])

def training_combiner(N_r,N_s):
    angle_index = np.random.choice(len(angles_r),(N_r,N_s))
    W = (1/np.sqrt(N_r))*np.exp(1j*angles_r[angle_index])
    return np.matrix(W).getH()

class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# custom con2d, because pytorch don't have "padding='same'" option.
def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):

    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows - input_rows)
    padding_rows = max(0, (out_rows - 1) * stride[0] + (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    # same for padding_cols
    input_cols = input.size(3)
    filter_cols = weight.size(3)
    effective_filter_size_cols = (filter_cols - 1) * dilation[0] + 1
    out_cols = (input_cols + stride[0] - 1) // stride[0]
    padding_needed = max(0, (out_cols - 1) * stride[0] + effective_filter_size_cols - input_cols)
    padding_cols = max(0, (out_cols - 1) * stride[0] + (filter_cols - 1) * dilation[0] + 1 - input_cols)
    cols_odd = (padding_cols % 2 != 0)
    if rows_odd or cols_odd:
        input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])
    return F.conv2d(input, weight, bias, stride, padding=(padding_rows // 2, padding_cols // 2),dilation=dilation, groups=groups)

#Generator Architecture (BS)
def generator(mb_size): 
    model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128*length*breadth),
            torch.nn.ReLU(),
            View([mb_size,128,length,breadth]),
            torch.nn.Upsample(scale_factor=2),
            Conv2d(128,128,4,bias=False),
            torch.nn.BatchNorm2d(128,momentum=0.8),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2),
            Conv2d(128,128,4,bias=False),
            torch.nn.BatchNorm2d(128,momentum=0.8),
            torch.nn.ReLU(),
            Conv2d(128,2,4,bias=False),)
    return model

#Critic Architecture (UE)
def critic():
    model = torch.nn.Sequential(
            Conv2d(2,16,3,stride=2),
            torch.nn.LeakyReLU(0.2,inplace=True),
            torch.nn.Dropout(0.25),
            Conv2d(16,32,3,stride=2),
            torch.nn.ZeroPad2d(padding=(0,1,0,1)),
            torch.nn.LeakyReLU(0.2,inplace=True),
            torch.nn.Dropout(0.25),
            Conv2d(32,64,3,stride=2),
            torch.nn.LeakyReLU(0.2,inplace=True),
            torch.nn.Dropout(0.25),
            Conv2d(64,128,3,stride=1),
            torch.nn.LeakyReLU(0.2,inplace=True),
            torch.nn.Dropout(0.25),
            torch.nn.Flatten(),
            torch.nn.Linear(3456,1),)
    return model        
    
class View(torch.nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)