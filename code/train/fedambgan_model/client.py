from .helper import *
batch_size = 50 #UE batch size. Does not change in paper.

class client(object):
    def __init__(self, rank):
        # rank
        self.rank = rank

    def __init_client(self):
        G = generator(batch_size).type(dtype)
        D = critic().type(dtype)  
        return G,D
    
    #@staticmethod
    def __load_global_model(self):
        global_G_state = torch.load('results/fed_pilot_gan/cache/global_G_state_%d.pkl'%n_d)
        global_D_state = torch.load('results/fed_pilot_gan/cache/global_D_state_%d.pkl'%n_d)
        model_G, model_D = self.__init_client()
        model_G.load_state_dict(global_G_state)
        model_D.load_state_dict(global_D_state)
        return model_G, model_D
    
    def generate_rx_signal(self, X, G_sample, X_eval, noise_var, stats, nc):
        G_sample_arr = torch.zeros(batch_size,2,N_t,N_r).type(dtype)
        X_sample_arr = torch.zeros(batch_size,2,N_t,N_r).type(dtype)
        std_real_tensor = dtype(np.tile(np.expand_dims(stats.std_real,0),(batch_size,1,1)))
        std_imag_tensor = dtype(np.tile(np.expand_dims(stats.std_imag,0),(batch_size,1,1)))
        std_dev = np.sqrt(noise_var)

        #G_sample
        noise_matrix = (std_dev/(np.sqrt(2)*stats.channel_coeff[nc]))*torch.randn(batch_size,2,4*N_p,N_r).type(dtype)
        channel_noise_matrix = torch.zeros(batch_size,2,N_t,N_r).type(dtype)
        corr_noise_real = torch.mm(A_mat_real,torch.transpose(noise_matrix[:,0,:,:].view(batch_size,4*N_p*N_r),0,1)) - torch.mm(A_mat_imag,torch.transpose(noise_matrix[:,1,:,:].view(batch_size,4*N_p*N_r),0,1))
        corr_noise_imag = torch.mm(A_mat_real,torch.transpose(noise_matrix[:,1,:,:].view(batch_size,4*N_p*N_r),0,1)) + torch.mm(A_mat_imag,torch.transpose(noise_matrix[:,0,:,:].view(batch_size,4*N_p*N_r),0,1))
        channel_noise_matrix[:,0,:,:] = torch.div(torch.reshape(corr_noise_real,(batch_size,N_t,N_r)),std_real_tensor)
        channel_noise_matrix[:,1,:,:] = torch.div(torch.reshape(corr_noise_imag,(batch_size,N_t,N_r)),std_imag_tensor)
        G_sample_arr = G_sample + channel_noise_matrix
        
        #X
        noise_matrix = (std_dev/(np.sqrt(2)*stats.channel_coeff[nc]))*torch.randn(batch_size,2,4*N_p,N_r).type(dtype) 
        channel_noise_matrix = torch.zeros(batch_size,2,N_t,N_r).type(dtype)
        corr_noise_real = torch.mm(A_mat_real,torch.transpose(noise_matrix[:,0,:,:].view(batch_size,4*N_p*N_r),0,1)) - torch.mm(A_mat_imag,torch.transpose(noise_matrix[:,1,:,:].view(batch_size,4*N_p*N_r),0,1))
        corr_noise_imag = torch.mm(A_mat_real,torch.transpose(noise_matrix[:,1,:,:].view(batch_size,4*N_p*N_r),0,1)) + torch.mm(A_mat_imag,torch.transpose(noise_matrix[:,0,:,:].view(batch_size,4*N_p*N_r),0,1))
        channel_noise_matrix[:,0,:,:] = torch.div(torch.reshape(corr_noise_real,(batch_size,N_t,N_r)),std_real_tensor)
        channel_noise_matrix[:,1,:,:] = torch.div(torch.reshape(corr_noise_imag,(batch_size,N_t,N_r)),std_imag_tensor)
        X_sample_arr = X + channel_noise_matrix

        return X_sample_arr, G_sample_arr
    
    def compute_gradient_penalty(self,D, real_samples, fake_samples):
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

    def __train(self, model_G, model_D, X_train, stats, it, nc):
        start_D = True
        D_solver = optim.RMSprop(model_D.parameters(), lr=lr)
        alpha = D_solver.param_groups[0]['alpha']
        epsilon = D_solver.param_groups[0]['eps']
        grads_D = {'n_samples': batch_size, 'named_grads': {}}
        for _ in range(n_d):
            z = Variable(torch.randn(batch_size, latent_dim)).type(dtype)
            G_sample = model_G(z)
            idx = np.random.choice(500, batch_size, replace = False)
            X = X_train[idx]
            X = Variable(torch.from_numpy(X).float()).type(dtype)
            y_X, y_G_sample = self.generate_rx_signal(X, G_sample, True, noise_var, stats, nc)
            D_fake = model_D(y_G_sample)
            D_real = model_D(y_X)
            D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
            gradient_penalty = self.compute_gradient_penalty(model_D,y_X,y_G_sample)
            D_loss += lambda_gp*gradient_penalty
            D_loss.backward()
            D_solver.step()
            #Save gradients used for RMSProp updates i.e. with momentum applied
            for name, param in model_D.named_parameters():
                gradient = torch.div(param.grad,torch.sqrt(D_solver.state[param]['square_avg']) + epsilon)
                if start_D:
                    grads_D['named_grads'][name] = gradient
                else:
                    grads_D['named_grads'][name] += gradient
            model_D.zero_grad()
            model_G.zero_grad()
            start_D = False
        if it % 20 == 0:
            print('Iter-{}; D_loss: {};'.format(it, D_loss.cpu().data.numpy()))
        return grads_D

    def run(self, X_train, stats, e, nc):
        model_G, model_D = self.__load_global_model()
        grads = self.__train(model_G, model_D, X_train, stats, e, nc)
        torch.save(grads, 'results/fed_pilot_gan/cache/grads_D_{}_%d.pkl'%n_d.format(self.rank))