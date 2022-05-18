from helper import *
reset_optim_D = True

#Wireless Parameters
N_t = 64
N_r = 16
latent_dim = 65
lambda_gp = 10

parser = argparse.ArgumentParser()
parser.add_argument('--CDL_model',type=str, default='A', choices=['A','B','C','D','E'])
config = parser.parse_args()

H_org = sio.loadmat('data/H_16x64_MIMO_CDL_%s_ULA_clean.mat'%config.CDL_model)
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
X_train = img_np

mb_size = 200
cnt = 0
lr = 5e-5

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
    torch.nn.Linear(3456,1),
)
D = D.type(dtype)

def reset_grad():
    G.zero_grad()
    D.zero_grad()

G_solver = optim.RMSprop(G.parameters(), lr=lr)
D_solver = optim.RMSprop(D.parameters(), lr=lr)

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

pdb.set_trace()
for it in range(60000):
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
        D_real = D(X)
        D_fake = D(G_sample)
        gradient_penalty = compute_gradient_penalty(D,X,G_sample)
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
    D_fake = D(G_sample)

    G_loss = -torch.mean(D_fake)

    G_loss.backward()
    G_solver.step()

    # Housekeeping - reset gradient
    reset_grad()
    if it%2000 == 0:
        torch.save(G.state_dict(),'results/wgan_gp/CDL_%s/generator%d.pt'%(config.CDL_model,it))

    # Print and plot every now and then
    if it % 50 == 0:
        print('Iter-{}; D_loss: {}; G_loss: {}'
              .format(it, D_loss.cpu().data.numpy(), G_loss.cpu().data.numpy()))

torch.save(G.state_dict(),'results/wgan_gp/CDL_%s/generator_CDL_ld_35_FFT.pt'%config.CDL_model)