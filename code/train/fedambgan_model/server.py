from .helper import *
batch_size_server = 800 #For n_d = 20. Use 200 for n_d = 5.

class server(object):
    def __init__(self, size, dataset):
        self.size = size
        # data loader
        self.X_train = dataset.train_CDL_A
        self.dataset = dataset
        self.path_G = 'results/fed_pilot_gan/cache/global_G_state_%d.pkl'%n_d
        self.path_D = 'results/fed_pilot_gan/cache/global_D_state_%d.pkl'%n_d
        self.model_G, self.model_D = self.__init_server()
        self.G_solver = optim.RMSprop(self.model_G.parameters(), lr=5e-5)
        self.D_solver = optim.SGD(self.model_D.parameters(), lr=5e-5)

    def __init_server(self,mb_size=batch_size_server):
        G = generator(mb_size).type(dtype)
        D = critic().type(dtype) 
        torch.save(G.state_dict(), self.path_G)
        torch.save(D.state_dict(), self.path_D)
        torch.save(G.state_dict(), 'results/fed_pilot_gan/cache/checkpoints/n_d_%d/global_G_state0.pkl'%n_d)
        return G, D
    
    def return_datasets(self):
        id_sample = np.random.choice(self.X_train.shape[0], 2000, replace = False) #2000 = D x n_client
        X_train = np.zeros((4,500,2,N_t,N_r)) #D = 500
        for i in range(4):
            X_train[i] = self.X_train[id_sample[500*i:500*(i+1)]]
        return X_train
    
    def __load_grads(self):
        grads_info = []
        for s in range(self.size):
            grads_info.append(torch.load('results/fed_pilot_gan/cache/grads_D_{}_%d.pkl'%n_d.format(s)))
        return grads_info
    
    def generate_rx_signal(self, G_sample, noise_var, stats):
        G_sample_arr = torch.zeros(batch_size_server,2,N_t,N_r).type(dtype)
        G1 = torch.zeros(batch_size_server,2,N_t,N_r).type(dtype)
        std_real_tensor = dtype(np.tile(np.expand_dims(stats.std_real,0),(batch_size_server,1,1)))
        std_imag_tensor = dtype(np.tile(np.expand_dims(stats.std_imag,0),(batch_size_server,1,1)))

        #G_sample
        #Normalize output of G
        G1[:,0,:,:] = dtype(stats.std_real)*G_sample[:,0,:,:] + dtype(stats.mu_real)
        G1[:,1,:,:] = dtype(stats.std_imag)*G_sample[:,1,:,:] + dtype(stats.mu_imag)
        #Convert from beamspace to spatial domain
        G_sample_real = torch.mm(A_T_R_real,torch.transpose(G1[:,0,:,:].view(batch_size_server,N_t*N_r),0,1)) - torch.mm(A_T_R_imag,torch.transpose(G1[:,1,:,:].view(batch_size_server,N_t*N_r),0,1))
        G_sample_imag = torch.mm(A_T_R_real,torch.transpose(G1[:,1,:,:].view(batch_size_server,N_t*N_r),0,1)) + torch.mm(A_T_R_imag,torch.transpose(G1[:,0,:,:].view(batch_size_server,N_t*N_r),0,1))
        std_dev = np.sqrt(noise_var)
        #Add noise of variance determined by TR 38.901 InH-Office path losses
        noise_matrix = (std_dev/(np.sqrt(2)*np.mean(stats.channel_coeff)))*torch.randn(batch_size_server,2,4*N_p,N_r).type(dtype)
        channel_noise_matrix = torch.zeros(batch_size_server,2,N_t,N_r).type(dtype)
        #Correlate noise to model LS channel estimation. Refer (10) and (11) in paper
        corr_noise_real = torch.mm(A_mat_real,torch.transpose(noise_matrix[:,0,:,:].view(batch_size_server,4*N_p*N_r),0,1)) - torch.mm(A_mat_imag,torch.transpose(noise_matrix[:,1,:,:].view(batch_size_server,4*N_p*N_r),0,1))
        corr_noise_imag = torch.mm(A_mat_real,torch.transpose(noise_matrix[:,1,:,:].view(batch_size_server,4*N_p*N_r),0,1)) + torch.mm(A_mat_imag,torch.transpose(noise_matrix[:,0,:,:].view(batch_size_server,4*N_p*N_r),0,1))
        channel_noise_matrix[:,0,:,:] = torch.div(torch.reshape(corr_noise_real,(batch_size_server,N_t,N_r)),std_real_tensor)
        channel_noise_matrix[:,1,:,:] = torch.div(torch.reshape(corr_noise_imag,(batch_size_server,N_t,N_r)),std_imag_tensor)
        #Supply normalized samples to D. Hence channel_noise_matrix is normalized by std_dev above.
        G_sample_arr = G_sample + channel_noise_matrix
        
        return G_sample_arr

    @staticmethod
    def __average_grads(grads_info):
        total_grads = {} 
        n_total_samples = 0
        for info in grads_info:
            n_samples = info['n_samples']
            for k, v in info['named_grads'].items():
                if k not in total_grads:
                    total_grads[k] = v
                total_grads[k] += v * n_samples
            n_total_samples += n_samples
        gradients = {}
        for k, v in total_grads.items():
            gradients[k] = torch.div(v, n_total_samples)
        return gradients

    def __step(self, gradients, iteration):
        self.model_D.train()
        self.D_solver.zero_grad()
        for k, v in self.model_D.named_parameters():
            v.grad = gradients[k]
        #Update global discriminator
        self.D_solver.step()
        z = Variable(torch.randn(batch_size_server, latent_dim)).type(dtype)
        #Generate simulated channels
        G_sample = self.model_G(z)
        #Add correlated noise to convert simulated channels to LS estimates
        y_G_sample = self.generate_rx_signal(G_sample, noise_var, self.dataset)
        D_fake = self.model_D(y_G_sample)
        G_loss = -torch.mean(D_fake)
        G_loss.backward()
        #Update global generator
        self.G_solver.step()
        if iteration%20 == 0:
            print('Iter-{}; G_loss: {};'.format(iteration, G_loss.cpu().data.numpy()))
        self.model_G.zero_grad()
        self.model_D.zero_grad()

    def aggregate(self,epoch,iteration):
        #Load critic gradients from UE
        grads_info = self.__load_grads()
        #Average critic gradients
        gradients = self.__average_grads(grads_info)
        #Update global discriminator and train global generator
        self.__step(gradients,iteration)
        torch.save(self.model_G.state_dict(), 'results/fed_pilot_gan/cache/global_G_state_%d.pkl'%n_d)
        torch.save(self.model_D.state_dict(), 'results/fed_pilot_gan/cache/global_D_state_%d.pkl'%n_d)
        
        if epoch%10 == 0 and (iteration+1)%25 == 0:
            torch.save(self.model_G.state_dict(), 'results/fed_pilot_gan/cache/checkpoints/n_d_%d/global_G_state%d.pkl'%(n_d,epoch))