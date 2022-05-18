from helper import *
gp_flag = False
lambda_gp = 10
reset_optim_D = True

#Wireless Parameters
N_t = 64
N_r = 16
latent_dim = 65
    
dft_basis = sio.loadmat("data/dft_basis.mat")
A_T = dft_basis['A1']/np.sqrt(N_t)
A_R = dft_basis['A2']/np.sqrt(N_r)

def normalize_ms(H_extracted):
    img_np_real = np.real(H_extracted)
    img_np_imag = np.imag(H_extracted)
    mu_real = np.mean(img_np_real,axis=0)
    mu_imag = np.mean(img_np_imag,axis=0)
    std_real = np.std(img_np_real,axis=0)
    std_imag = np.std(img_np_imag,axis=0)

    img_np_real = (img_np_real - mu_real)/std_real
    img_np_imag = (img_np_imag - mu_imag)/std_imag
    img_np = np.zeros((img_np_real.shape[0],2,N_t,N_r))
    img_np[:,0,:,:] = img_np_real
    img_np[:,1,:,:] = img_np_imag
    X_train = img_np
    return X_train

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

X_train = normalize_ms(H_extracted)

size = int(X_train.shape[0]/5)
CDL_NLOS = np.ones((3*size,1))
CDL_LOS = 2*np.ones((2*size,1))
CDL = np.concatenate((CDL_NLOS,CDL_LOS),axis=0)

mb_size = 200
cnt = 0
lr = 5e-5

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

def reset_grad():
    G.zero_grad()
    D.zero_grad()
    
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

G_solver = optim.RMSprop(G.parameters(), lr=lr)
D_solver = optim.RMSprop(D.parameters(), lr=lr)

for it in range(100000):
    if reset_optim_D:
        D_solver = optim.RMSprop(D.parameters(), lr=lr)
    for _ in range(5):
        # Sample data
        z = Variable(torch.randn(mb_size, latent_dim)).type(dtype)
        idx = np.random.choice(X_train.shape[0], mb_size, replace=False)
        X = X_train[idx]
        CDL_mb_d = CDL[idx,0] - 1
        X = torch.from_numpy(X).float().type(dtype)
        CDL_mb_d = torch.from_numpy(CDL_mb_d).long().type(dtype_long)
        CDL_mb_g = torch.from_numpy(np.random.randint(0,2,mb_size)).long().type(dtype_long)
        
        # Dicriminator forward-loss-backward-update
        D_real = D(X,CDL_mb_d)
        G_sample = G(z,CDL_mb_g)
        D_fake = D(G_sample,CDL_mb_g)

        D_loss = -(torch.mean(D_real) - torch.mean(D_fake)) 
        if gp_flag:
            gradient_penalty = compute_gradient_penalty(D, X, G_sample, CDL_mb_d)
            D_loss += lambda_gp * gradient_penalty
            
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
    D_fake = D(G_sample,CDL_mb)

    G_loss = -torch.mean(D_fake)

    G_loss.backward()
    G_solver.step()

    # Housekeeping - reset gradient
    reset_grad()
    
    if it%4000 == 0 or (it%400 == 0 and it < 4000):
        torch.save(G.state_dict(),'results/cwgan/generator%d.pt'%it)

    # Print and plot every now and then
    if it % 50 == 0:
        print('Iter-{}; D_loss: {}; G_loss: {}'
              .format(it, D_loss.cpu().data.numpy(), G_loss.cpu().data.numpy()))

torch.save(G.state_dict(),'results/cwgan/generator_CDL_ABCDE_ld_35_FFT.pt')