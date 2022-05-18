from helper import *
noise_add = True
reset_optim_D = False

#Wireless Parameters
N_t = 64
N_r = 16
latent_dim = 65

parser = argparse.ArgumentParser()
parser.add_argument('--snr',type=int, default=40, choices=[10,20,30,40])
config = parser.parse_args()
    
H_org = sio.loadmat("data/H_16x64_MIMO_CDL_A_ULA_clean.mat")
H_ex = H_org['hest']
H_extracted = np.transpose(H_ex,(2,1,0))
dft_basis = sio.loadmat("data/dft_basis.mat")
A = np.load('data/A_mat_1024.npy')
FS_T = np.load('data/FS_T_1024.npy')
W_arr = np.load('data/W_1024.npy')

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

#Construct B and tx_kron
B = np.zeros((N_t*N_r,N_t*N_r),dtype='complex64')
tx_kron = np.zeros((N_t*N_r,N_t*N_r),dtype='complex64')
for i in range(4):
    B[N_p*N_r*i:N_p*N_r*(i+1),N_p*N_r*i:N_p*N_r*(i+1)] = np.kron(identity,W_arr[i])
    tx_kron[N_p*N_r*i:N_p*N_r*(i+1),:] = np.kron(FS_T[i],np.identity(N_r))
A_real = dtype(np.real(tx_kron))
A_imag = dtype(np.imag(tx_kron))

A_T = dft_basis['A1']/np.sqrt(N_t)
A_R = dft_basis['A2']/np.sqrt(N_r)
for i in range(H_ex.shape[2]):
    H_extracted[i] = np.transpose(np.matmul(np.matmul(A_R.conj().T,H_extracted[i].T,dtype='complex64'),A_T))

mb_size = 200
cnt = 1
lr = 5e-5
SNR = config.snr

img_np_real = np.real(H_extracted)
img_np_imag = np.imag(H_extracted)

mu_real = np.mean(img_np_real,axis=0)
mu_imag = np.mean(img_np_imag,axis=0)
std_real = np.std(img_np_real,axis=0)
std_imag = np.std(img_np_imag,axis=0)

std_real_tensor = dtype(np.tile(np.expand_dims(std_real,0),(mb_size,1,1)))
std_imag_tensor = dtype(np.tile(np.expand_dims(std_imag,0),(mb_size,1,1)))

img_np_real = (img_np_real - mu_real)/std_real
img_np_imag = (img_np_imag - mu_imag)/std_imag

img_np = np.zeros((H_ex.shape[2],2,N_t,N_r))
img_np[:,0,:,:] = img_np_real
img_np[:,1,:,:] = img_np_imag
X_train = img_np

length = int(N_t/4)
breadth = int(N_r/4)

G = torch.nn.Sequential(
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
    Conv2d(128,2,4,bias=False),
)
G = G.type(dtype) 


D = torch.nn.Sequential(
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
    torch.nn.Linear(3456,1), #1536
)
D = D.type(dtype)

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = dtype(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(dtype(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = torch.reshape(gradients,(gradients.size(0), -1))
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def reset_grad():
    G.zero_grad()
    D.zero_grad()
    
A_T_R = np.kron(A_T.conj(),A_R)
A_R_T = np.kron(np.transpose(A_T),np.matrix(A_R).getH())
A_T_R_real = dtype(np.real(A_T_R))
A_T_R_imag = dtype(np.imag(A_T_R))

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

def generate_rx_signal(X, G_sample, X_eval, SNR):
    GS = torch.zeros(mb_size,2,4*N_p,N_r).type(dtype)
    G_sample_arr = torch.zeros(mb_size,2,N_t,N_r).type(dtype)
    G1 = torch.zeros(mb_size,2,N_t,N_r).type(dtype)
    XS = torch.zeros(mb_size,2,4*N_p,N_r).type(dtype)
    X_sample_arr = torch.zeros(mb_size,2,N_t,N_r).type(dtype)
    X1 = torch.zeros(mb_size,2,N_t,N_r).type(dtype)
    
    #G_sample
    G1[:,0,:,:] = dtype(std_real)*G_sample[:,0,:,:] + dtype(mu_real)
    G1[:,1,:,:] = dtype(std_imag)*G_sample[:,1,:,:] + dtype(mu_imag)
    G_sample_real = torch.mm(A_T_R_real,torch.transpose(G1[:,0,:,:].view(mb_size,N_t*N_r),0,1)) - torch.mm(A_T_R_imag,torch.transpose(G1[:,1,:,:].view(mb_size,N_t*N_r),0,1))
    G_sample_imag = torch.mm(A_T_R_real,torch.transpose(G1[:,1,:,:].view(mb_size,N_t*N_r),0,1)) + torch.mm(A_T_R_imag,torch.transpose(G1[:,0,:,:].view(mb_size,N_t*N_r),0,1))
    GS_real = torch.transpose(torch.mm(A_real,G_sample_real) - torch.mm(A_imag,G_sample_imag),0,1)
    GS_imag = torch.transpose(torch.mm(A_real,G_sample_imag) + torch.mm(A_imag,G_sample_real),0,1)
    GS[:,0,:,:] = torch.reshape(GS_real,(mb_size,4*N_p,N_r))
    GS[:,1,:,:] = torch.reshape(GS_imag,(mb_size,4*N_p,N_r))
    if noise_add:
        E_s = torch.sum(torch.mul(GS,GS),1)
        std_dev = (1/(10**(SNR/20)))*torch.unsqueeze(torch.sqrt(E_s),1).repeat(1,2,1,1)
        noise_matrix = (1/np.sqrt(2))*torch.mul(std_dev,torch.randn(mb_size,2,4*N_p,N_r).type(dtype))
        channel_noise_matrix = torch.zeros(mb_size,2,N_t,N_r).type(dtype)
        corr_noise_real = torch.mm(A_mat_real,torch.transpose(noise_matrix[:,0,:,:].view(mb_size,4*N_p*N_r),0,1)) - torch.mm(A_mat_imag,torch.transpose(noise_matrix[:,1,:,:].view(mb_size,4*N_p*N_r),0,1))
        corr_noise_imag = torch.mm(A_mat_real,torch.transpose(noise_matrix[:,1,:,:].view(mb_size,4*N_p*N_r),0,1)) + torch.mm(A_mat_imag,torch.transpose(noise_matrix[:,0,:,:].view(mb_size,4*N_p*N_r),0,1))
        channel_noise_matrix[:,0,:,:] = torch.div(torch.reshape(corr_noise_real,(mb_size,N_t,N_r)),std_real_tensor)
        channel_noise_matrix[:,1,:,:] = torch.div(torch.reshape(corr_noise_imag,(mb_size,N_t,N_r)),std_imag_tensor)
        G_sample_arr = G_sample + channel_noise_matrix
    
    #X
    if X_eval:
        X1[:,0,:,:] = dtype(std_real)*X[:,0,:,:] + dtype(mu_real)
        X1[:,1,:,:] = dtype(std_imag)*X[:,1,:,:] + dtype(mu_imag)
        X_real = torch.mm(A_T_R_real,torch.transpose(X1[:,0,:,:].view(mb_size,N_t*N_r),0,1)) - torch.mm(A_T_R_imag,torch.transpose(X1[:,1,:,:].view(mb_size,N_t*N_r),0,1))
        X_imag = torch.mm(A_T_R_real,torch.transpose(X1[:,1,:,:].view(mb_size,N_t*N_r),0,1)) + torch.mm(A_T_R_imag,torch.transpose(X1[:,0,:,:].view(mb_size,N_t*N_r),0,1))
        XS_real = torch.transpose(torch.mm(A_real,X_real) - torch.mm(A_imag,X_imag),0,1)
        XS_imag = torch.transpose(torch.mm(A_real,X_imag) + torch.mm(A_imag,X_real),0,1)
        XS[:,0,:,:] =  torch.reshape(XS_real,(mb_size,4*N_p,N_r))
        XS[:,1,:,:] =  torch.reshape(XS_imag,(mb_size,4*N_p,N_r))
        if noise_add:
            E_s = torch.sum(torch.mul(XS,XS),1)
            std_dev = (1/(10**(SNR/20)))*torch.unsqueeze(torch.sqrt(E_s),1).repeat(1,2,1,1)
            noise_matrix = (1/np.sqrt(2))*torch.mul(std_dev,torch.randn(mb_size,2,4*N_p,N_r).type(dtype))
            channel_noise_matrix = torch.zeros(mb_size,2,N_t,N_r).type(dtype)
            corr_noise_real = torch.mm(A_mat_real,torch.transpose(noise_matrix[:,0,:,:].view(mb_size,4*N_p*N_r),0,1)) - torch.mm(A_mat_imag,torch.transpose(noise_matrix[:,1,:,:].view(mb_size,4*N_p*N_r),0,1))
            corr_noise_imag = torch.mm(A_mat_real,torch.transpose(noise_matrix[:,1,:,:].view(mb_size,4*N_p*N_r),0,1)) + torch.mm(A_mat_imag,torch.transpose(noise_matrix[:,0,:,:].view(mb_size,4*N_p*N_r),0,1))
            channel_noise_matrix[:,0,:,:] = torch.div(torch.reshape(corr_noise_real,(mb_size,N_t,N_r)),std_real_tensor)
            channel_noise_matrix[:,1,:,:] = torch.div(torch.reshape(corr_noise_imag,(mb_size,N_t,N_r)),std_imag_tensor)
            X_sample_arr = X + channel_noise_matrix
    
    return X_sample_arr, G_sample_arr

G_solver = optim.RMSprop(G.parameters(), lr=lr)
D_solver = optim.RMSprop(D.parameters(), lr=lr)
lambda_gp = 10

for it in range(0,60000): 
    if reset_optim_D:
        D_solver = optim.RMSprop(D.parameters(), lr=lr)
    for _ in range(5):
        # Sample data
        z = Variable(torch.randn(mb_size, latent_dim)).type(dtype)
        idx = np.random.choice(X_train.shape[0], mb_size, replace = False)
        X = X_train[idx]
        X = Variable(torch.from_numpy(X).float()).type(dtype)

        # Dicriminator forward-loss-backward-update
        G_sample = G(z)
        #Generate y_X and y_G_sample
        y_X, y_G_sample = generate_rx_signal(X, G_sample, True, SNR) 
        
        D_real = D(y_X)
        D_fake = D(y_G_sample)
        gradient_penalty = compute_gradient_penalty(D,y_X,y_G_sample)
        D_loss = -(torch.mean(D_real) - torch.mean(D_fake)) + lambda_gp*gradient_penalty
        D_loss.backward()
        D_solver.step()

        # Housekeeping - reset gradient
        reset_grad()

    # Generator forward-loss-backward-update
    idx = np.random.choice(X_train.shape[0], mb_size, replace = False)
    X = X_train[idx]
    X = Variable(torch.from_numpy(X)).type(dtype)
    z = Variable(torch.randn(mb_size, latent_dim)).type(dtype)

    G_sample = G(z)
    _, y_G_sample = generate_rx_signal(X, G_sample, False, SNR) 
    D_fake = D(y_G_sample)

    G_loss = -torch.mean(D_fake)

    G_loss.backward()
    G_solver.step()

    # Housekeeping - reset gradient
    reset_grad()
    
    if it%2000 == 0:
        torch.save(G.state_dict(),'results/pilot_gan/U_1/snr_%d/generator%d.pt'%(SNR,it))

    # Print and plot every now and then
    if it % 50 == 0:
        print('Iter-{}; D_loss: {}; G_loss: {}'
              .format(it, D_loss.cpu().data.numpy(), G_loss.cpu().data.numpy()))
    
    G_loss_val = G_loss.cpu().data.numpy()
    D_loss_val = D_loss.cpu().data.numpy()
    if math.isnan(G_loss_val):
        pdb.set_trace()

torch.save(G.state_dict(),'results/pilot_gan/U_1/snr_%d/amb_generator_LS_CDL_ld_35_FFT.pt'%SNR)