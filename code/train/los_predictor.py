from helper import *

#Wireless Parameters
N_t = 64
N_r = 16
mb_size = 200

parser = argparse.ArgumentParser()
parser.add_argument('--train_models',type=str, default='ABCDE', choices=['ABCDE','BD'])
config = parser.parse_args()
    
dft_basis = sio.loadmat("data/dft_basis.mat")
A_T = dft_basis['A1']/np.sqrt(N_t)
A_R = dft_basis['A2']/np.sqrt(N_r)
A = np.load('data/A_mat_1024.npy')
A_inv = np.linalg.inv(A)
FS_T = np.load('data/FS_T_1024.npy')
W_arr = np.load('data/W_1024.npy')

def fft_op(H_extracted):
    for i in range(H_extracted.shape[0]):
        H_extracted[i] = np.transpose(np.matmul(np.matmul(A_R.conj().T,H_extracted[i].T,dtype='complex64'),A_T))
    return H_extracted
 
H_org_B = sio.loadmat("data/H_16x64_MIMO_CDL_B_ULA_clean.mat")
H_ex_B = H_org_B['hest']
H_extracted_B = np.transpose(copy.deepcopy(H_ex_B),(2,1,0))

H_org_D = sio.loadmat("data/H_16x64_MIMO_CDL_D_ULA_clean.mat")
H_ex_D = H_org_D['hest']
H_extracted_D = np.transpose(copy.deepcopy(H_ex_D),(2,1,0))

if config.train_models == 'ABCDE':
    H_org_A = sio.loadmat("data/H_16x64_MIMO_CDL_A_ULA_clean.mat")
    H_ex_A = H_org_A['hest']
    H_extracted_A = np.transpose(copy.deepcopy(H_ex_A),(2,1,0))

    H_org_C = sio.loadmat("data/H_16x64_MIMO_CDL_C_ULA_clean.mat")
    H_ex_C = H_org_C['hest']
    H_extracted_C = np.transpose(copy.deepcopy(H_ex_C),(2,1,0))

    H_org_E = sio.loadmat("data/H_16x64_MIMO_CDL_E_ULA_clean.mat")
    H_ex_E = H_org_E['hest']
    H_extracted_E = np.transpose(copy.deepcopy(H_ex_E),(2,1,0))

    H_extracted = np.concatenate([H_extracted_A,H_extracted_B,H_extracted_C,H_extracted_D,H_extracted_E],axis=0)
    size = int(H_extracted.shape[0]/5)
    CDL_NLOS = np.zeros((3*size,1))
    CDL_LOS = np.ones((2*size,1))
    CDL = np.concatenate((CDL_NLOS,CDL_LOS),axis=0)
else:
    H_extracted = np.concatenate([H_extracted_B,H_extracted_D],axis=0)
    size = int(H_extracted.shape[0]/2)
    CDL_NLOS = np.zeros((size,1))
    CDL_LOS = np.ones((size,1))
    CDL = np.concatenate((CDL_NLOS,CDL_LOS),axis=0)

LOS_NN = torch.nn.Sequential(
    Conv2d(2,16,3,stride=2),
    torch.nn.LeakyReLU(0.2,inplace=True),
    torch.nn.Dropout(0.25),
    Conv2d(16,32,3,stride=2),
    torch.nn.ZeroPad2d(padding=(0,1,0,1)),
    torch.nn.BatchNorm2d(32,momentum=0.8),
    torch.nn.LeakyReLU(0.2,inplace=True),
    torch.nn.Dropout(0.25),
    Conv2d(32,64,3,stride=2),
    torch.nn.BatchNorm2d(64,momentum=0.8),
    torch.nn.LeakyReLU(0.2,inplace=True),
    torch.nn.Dropout(0.25),
    Conv2d(64,128,3,stride=1),
    torch.nn.BatchNorm2d(128,momentum=0.8),
    torch.nn.LeakyReLU(0.2,inplace=True),
    torch.nn.Dropout(0.25),
    torch.nn.Flatten(),
    torch.nn.Linear(3456,1),
    torch.nn.Sigmoid(),
)
LOS_NN = LOS_NN.type(dtype) 
    
alpha = 0.25
N_p = int(alpha*N_t)
M = N_p*N_r
qpsk_constellation = (1/np.sqrt(2))*np.array([1+1j,1-1j,-1+1j,-1-1j])
identity = np.identity(N_p)
A_T_R = np.kron(A_T.conj(),A_R)
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
A_mat = np.matmul(A_inv,B)

SNR_vec = range(-15,20,5)
loss = torch.nn.BCELoss()
lr = 3e-4
LOS_NN_solver = optim.Adam(LOS_NN.parameters(), lr=lr)
n_iter = 20000

def generate_dataset(X, SNR):   
    tx_message = np.reshape(np.matmul(tx_kron,np.transpose(np.reshape(X,(mb_size,N_t*N_r)))),(mb_size,4*N_p,N_r))
    std_dev = (1/(10**(SNR/20)))*np.abs(tx_message)
    noise_matrix = (1/np.sqrt(2))*(np.multiply(std_dev,np.random.randn(mb_size,4*N_p,N_r) + 1j*np.random.randn(mb_size,4*N_p,N_r)))
    corr_noise = np.matmul(A_mat,np.transpose(np.reshape(noise_matrix,(mb_size,4*N_p*N_r))))
    corr_noise = np.reshape(np.expand_dims(np.transpose(corr_noise),2),(mb_size,N_t,N_r))
    s1_dft = np.matmul(np.transpose(X + corr_noise,(0,2,1)),A_T)
    s2_dft = np.matmul(A_R.conj(),s1_dft)
    return np.transpose(s2_dft,(0,2,1))
       
for it in range(n_iter):
    idx = np.random.randint(0, H_extracted.shape[0], mb_size)
    X = H_extracted[idx] 
    Y = torch.from_numpy(CDL[idx]).float().type(dtype)
    SNR_id = np.random.randint(0, len(SNR_vec))
    H_LS = generate_dataset(X,SNR_vec[SNR_id])
    H_LS_t = torch.zeros(mb_size,2,N_t,N_r).type(dtype)
    H_LS_t[:,0,:,:] = torch.from_numpy(np.real(H_LS)).float().type(dtype)
    H_LS_t[:,1,:,:] = torch.from_numpy(np.imag(H_LS)).float().type(dtype)
    soft_labels = LOS_NN(H_LS_t)
    output = loss(soft_labels,Y)
    output.backward()
    LOS_NN_solver.step()
    LOS_NN.zero_grad()
    if it%50 == 0:
        print('Iter-{}; BCELoss: {}'.format(it, output.cpu().data.numpy()))
    if it%500 == 0:
        torch.save(LOS_NN.state_dict(),'results/pcgan/los_predictor/LOS_NN%d.pt'%it)

torch.save(LOS_NN.state_dict(),'results/pcgan/los_predictor/LOS_NN_%s.pt'%config.train_models)
    