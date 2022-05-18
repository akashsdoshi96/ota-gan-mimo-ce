from helper import *
noise_add = True
gp_flag = False
reset_optim_D = False

#Wireless Parameters
N_t = 64
N_r = 16
latent_dim = 65

parser = argparse.ArgumentParser()
parser.add_argument('--snr',type=int, default=30, choices=[20,30])
config = parser.parse_args()
    
dft_basis = sio.loadmat("data/dft_basis.mat")
A_T = dft_basis['A1']/np.sqrt(N_t)
A_R = dft_basis['A2']/np.sqrt(N_r)
    
def fft_op(H_extracted):
    for i in range(H_extracted.shape[0]):
        H_extracted[i] = np.transpose(np.matmul(np.matmul(A_R.conj().T,H_extracted[i].T,dtype='complex64'),A_T))
    return H_extracted
    
H_org_A = sio.loadmat("data/H_16x64_MIMO_CDL_A_ULA_clean.mat")
H_ex_A = H_org_A['hest']
H_extracted_A = np.transpose(copy.deepcopy(H_ex_A),(2,1,0))
H_extracted_A = fft_op(H_extracted_A)

H_org_B = sio.loadmat("data/H_16x64_MIMO_CDL_B_ULA_clean.mat")
H_ex_B = H_org_B['hest']
H_extracted_B = np.transpose(copy.deepcopy(H_ex_B),(2,1,0))
H_extracted_B = fft_op(H_extracted_B)

H_org_C = sio.loadmat("data/H_16x64_MIMO_CDL_C_ULA_clean.mat")
H_ex_C = H_org_C['hest']
H_extracted_C = np.transpose(copy.deepcopy(H_ex_C),(2,1,0))
H_extracted_C = fft_op(H_extracted_C)

H_org_D = sio.loadmat("data/H_16x64_MIMO_CDL_D_ULA_clean.mat")
H_ex_D = H_org_D['hest']
H_extracted_D = np.transpose(copy.deepcopy(H_ex_D),(2,1,0))
H_extracted_D = fft_op(H_extracted_D)

H_org_E = sio.loadmat("data/H_16x64_MIMO_CDL_E_ULA_clean.mat")
H_ex_E = H_org_E['hest']
H_extracted_E = np.transpose(copy.deepcopy(H_ex_E),(2,1,0))
H_extracted_E = fft_op(H_extracted_E)

H_extracted = np.concatenate([H_extracted_A,H_extracted_B,H_extracted_C,H_extracted_D,H_extracted_E],axis=0)

mb_size = 200
cnt = 0
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

img_np = np.zeros((H_extracted.shape[0],2,N_t,N_r))
img_np[:,0,:,:] = img_np_real
img_np[:,1,:,:] = img_np_imag
X_train = img_np

size = int(X_train.shape[0]/5)

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
LOS_NN.load_state_dict(torch.load('results/pcgan/los_predictor/LOS_NN_ABCDE.pt'))
LOS_NN.eval()

length = int(N_t/4)
breadth = int(N_r/4)

class Generator(nn.Module):
    def __init__(self,mb_size):
        super(Generator,self).__init__()
        self.mb_size = mb_size
        self.embedding = nn.Embedding(2,10)
        self.linear_e = nn.Linear(10,length*breadth)
        self.view_e = View([mb_size,1,length,breadth])
        self.linear_g1 = nn.Linear(latent_dim, 127*length*breadth)
        self.relu = nn.ReLU()
        self.view_g = View([mb_size,127,length,breadth])
        self.upsample_g = nn.Upsample(scale_factor=2)
        self.batchnorm_g1 = nn.BatchNorm2d(128,momentum=0.8)
        self.batchnorm_g2 = nn.BatchNorm2d(128,momentum=0.8)
        self.batchnorm_g3 = nn.BatchNorm2d(128,momentum=0.8)
        self.conv2d_g1 = Conv2d(128,128,4,bias=False)
        self.conv2d_g2 = Conv2d(128,128,4,bias=False)
        self.conv2d_g4 = Conv2d(128,128,4,bias=False)
        self.conv2d_g3 = Conv2d(128,2,4,bias=False)
        self.concat = Concatenate(1)
        
    def forward(self,z,c):
        c_e = self.embedding(c)
        c_l = self.linear_e(c_e)
        c_v = self.view_e(c_l)
        
        z1 = self.linear_g1(z)
        z1 = self.relu(z1)
        z1_v = self.view_g(z1)
        
        z_c = self.concat([z1_v,c_v])
        z_c_1 = self.upsample_g(z_c)
        z_c_1 = self.conv2d_g1(z_c_1)
        z_c_1 = self.batchnorm_g1(z_c_1)
        z_c_1 = self.relu(z_c_1)
        z_c_2 = self.upsample_g(z_c_1)
        z_c_2 = self.conv2d_g2(z_c_2)
        z_c_2 = self.batchnorm_g2(z_c_2)
        z_c_2 = self.relu(z_c_2)
        z_c_2 = self.conv2d_g4(z_c_2)
        z_c_2 = self.batchnorm_g3(z_c_2)
        z_c_2 = self.relu(z_c_2)
        output = self.conv2d_g3(z_c_2)
        
        return output

G = Generator(mb_size).type(dtype)

class Discriminator(nn.Module):
    def __init__(self,mb_size):
        super(Discriminator,self).__init__()
        self.embedding = nn.Embedding(2,10)
        self.linear_e = nn.Linear(10,N_t*N_r)
        self.view_e = View([mb_size,1,N_t,N_r])
        self.concat = Concatenate(1)
        self.conv2d_1 = Conv2d(3,16,3,stride=2)
        self.leakyrelu = nn.LeakyReLU(0.2,inplace=True)
        self.dropout = nn.Dropout(0.25)
        self.zeropad = nn.ZeroPad2d(padding=(0,1,0,1))
        self.conv2d_2 = Conv2d(16,32,3,stride=2)
        self.conv2d_3 = Conv2d(32,64,3,stride=2)
        self.conv2d_4 = Conv2d(64,128,3,stride=1)
        if gp_flag == False:
            self.batchnorm_1 = nn.BatchNorm2d(32,momentum=0.8)
            self.batchnorm_2 = nn.BatchNorm2d(64,momentum=0.8)
            self.batchnorm_3 = nn.BatchNorm2d(128,momentum=0.8)
        self.flatten = nn.Flatten()
        self.linear_d = nn.Linear(3456,1)
        
    def forward(self,x,c):
        c_e = self.embedding(c)
        c_l = self.linear_e(c_e)
        c_v = self.view_e(c_l)
        
        x_c = self.concat([x,c_v])
        x_c_1 = self.conv2d_1(x_c)
        x_c_1 = self.leakyrelu(x_c_1)
        x_c_1 = self.dropout(x_c_1)
        x_c_2 = self.conv2d_2(x_c_1)
        x_c_2 = self.zeropad(x_c_2)
        if gp_flag == False:
            x_c_2 = self.batchnorm_1(x_c_2)
        x_c_2 = self.leakyrelu(x_c_2)
        x_c_2 = self.dropout(x_c_2)
        x_c_3 = self.conv2d_3(x_c_2)
        if gp_flag == False:
            x_c_3 = self.batchnorm_2(x_c_3)
        x_c_3 = self.leakyrelu(x_c_3)
        x_c_3 = self.dropout(x_c_3)
        x_c_4 = self.conv2d_4(x_c_3)
        if gp_flag == False:
            x_c_4 = self.batchnorm_3(x_c_4)
        x_c_4 = self.leakyrelu(x_c_4)
        x_c_4 = self.dropout(x_c_4)
        x_c_4 = self.flatten(x_c_4)
        output = self.linear_d(x_c_4)
        
        return output

D = Discriminator(mb_size).type(dtype)

def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = dtype(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates,labels)
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
    
alpha = 0.25
N_p = int(alpha*N_t)
M = N_p*N_r
qpsk_constellation = (1/np.sqrt(2))*np.array([1+1j,1-1j,-1+1j,-1-1j])
identity = np.identity(N_p)
A_T_R = np.kron(A_T.conj(),A_R)
A_R_T = np.kron(np.transpose(A_T),np.matrix(A_R).getH())
A_T_R_real = dtype(np.real(A_T_R))
A_T_R_imag = dtype(np.imag(A_T_R))
N_s = N_r
A = np.load('data/A_mat_1024.npy')
A_inv = np.linalg.inv(A)
FS_T = np.load('data/FS_T_1024.npy')
W_arr = np.load('data/W_1024.npy')

#Construct B and tx_kron
B = np.zeros((N_t*N_r,N_t*N_r),dtype='complex64')
tx_kron = np.zeros((N_t*N_r,N_t*N_r),dtype='complex64')
for i in range(4):
    B[N_p*N_r*i:N_p*N_r*(i+1),N_p*N_r*i:N_p*N_r*(i+1)] = np.kron(identity,W_arr[i])
    tx_kron[N_p*N_r*i:N_p*N_r*(i+1),:] = np.kron(FS_T[i],np.identity(N_r))
A_real = dtype(np.real(tx_kron))
A_imag = dtype(np.imag(tx_kron))
A_mat = np.matmul(A_R_T,np.matmul(A_inv,B))
A_mat_real = dtype(np.real(A_mat))
A_mat_imag = dtype(np.imag(A_mat))

def generate_rx_signal(X, LOS_predictor, SNR):
    XS = torch.zeros(mb_size,2,4*N_p,N_r).type(dtype)
    X_sample_arr = torch.zeros(mb_size,2,N_t,N_r).type(dtype)
    X1 = torch.zeros(mb_size,2,N_t,N_r).type(dtype)
    output_arr = np.zeros((mb_size,1))
    
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
          
    if LOS_predictor:
        X_sample_arr_los = torch.zeros(mb_size,2,N_t,N_r).type(dtype)
        X_sample_arr_los[:,0,:,:] = X_sample_arr[:,0,:,:] + dtype(mu_real)
        X_sample_arr_los[:,1,:,:] = X_sample_arr[:,1,:,:] + dtype(mu_imag)
        output = LOS_NN(X_sample_arr_los).data.cpu().numpy()
        output_arr = (np.sign(2*output - 1) + 1)/2
        output_arr = output_arr.astype(int)
    
    return X_sample_arr, output_arr

G_solver = optim.RMSprop(G.parameters(), lr=lr)
D_solver = optim.RMSprop(D.parameters(), lr=lr)
lambda_gp = 10

for it in range(0,100000):
    if reset_optim_D:
        D_solver = optim.RMSprop(D.parameters(), lr=lr)
    for _ in range(5):
        # Sample data
        z = Variable(torch.randn(mb_size, latent_dim)).type(dtype)
        idx = np.random.choice(X_train.shape[0], mb_size, replace = False)
        X = X_train[idx]
        X = torch.from_numpy(X).float().type(dtype)
        y_X, output_arr = generate_rx_signal(X, True, SNR)
        CDL_mb_d = torch.from_numpy(output_arr).long().type(dtype_long)

        #Generate y_X and y_G_sample              
        if gp_flag == False:
            CDL_mb_g = torch.from_numpy(np.random.randint(0,2,mb_size)).long().type(dtype_long)
            G_sample = G(z,CDL_mb_g)
            y_G_sample, _ = generate_rx_signal(G_sample, False, SNR)
            D_fake = D(y_G_sample,CDL_mb_g)
        else:
            G_sample = G(z,CDL_mb_d)
            y_G_sample, _ = generate_rx_signal(G_sample, False, SNR)
            D_fake = D(y_G_sample,CDL_mb_d)
        
        # Dicriminator forward-loss-backward-update
        D_real = D(y_X,CDL_mb_d)
        D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
        if gp_flag:
            gradient_penalty = compute_gradient_penalty(D,y_X,y_G_sample, CDL_mb_d)
            D_loss += lambda_gp*gradient_penalty

        D_loss.backward()
        D_solver.step()
        
        # Weight clipping
        if gp_flag == False:
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

        # Housekeeping - reset gradient
        reset_grad()

    # Generator forward-loss-backward-update
    z = Variable(torch.randn(mb_size, latent_dim)).type(dtype)
    CDL_mb = torch.from_numpy(np.random.randint(0,2,mb_size)).long().type(dtype_long)
    
    G_sample = G(z,CDL_mb)
    y_G_sample, _ = generate_rx_signal(G_sample, False, SNR)
    D_fake = D(y_G_sample,CDL_mb)

    G_loss = -torch.mean(D_fake)

    G_loss.backward()
    G_solver.step()

    # Housekeeping - reset gradient
    reset_grad()
    
    if it%4000 == 0 or (it%400 == 0 and it < 4000):
        torch.save(G.state_dict(),'results/pcgan/snr_%d/generator%d.pt'%(SNR,it))

    # Print and plot every now and then
    if it % 50 == 0:
        print('Iter-{}; D_loss: {}; G_loss: {}'
              .format(it, D_loss.cpu().data.numpy(), G_loss.cpu().data.numpy()))
    
    G_loss_val = G_loss.cpu().data.numpy()
    D_loss_val = D_loss.cpu().data.numpy()
    if math.isnan(G_loss_val):
        pdb.set_trace()

torch.save(G.state_dict(),'results/pcgan/snr_%d/generator_CDL_ABCDE_ld_35_FFT.pt'%SNR)