import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0),input.size(1), -1)

class SinkhornNet(nn.Module):

    def __init__(self, patch_dim=10,latent_dim=16, K=12, d=2,n_samples=5, noise_factor=1.0, temp=1.0, n_iters=20,sigQ=5):
        super(SinkhornNet, self).__init__()

        self.latent_dim = latent_dim

        self.logSigQ = torch.nn.Parameter(sigQ*torch.ones(1))

        self.K = K
        self.d = d
        self.n_samples = n_samples
        self.noise_factor = noise_factor
        self.temp = temp
        self.n_iters = n_iters

        self.sinknet = nn.Sequential(nn.LogSoftmax(dim=1))

        self.encoder = nn.Sequential(
              Flatten(),
              nn.Linear(2*patch_dim*patch_dim*2,latent_dim),
              nn.ReLU(),
              nn.Linear(latent_dim, K),
          )


    def forward(self,im,z):

        latent = self.encoder(im)

        log_alpha = self.sinknet(latent)

        soft_perms_inf, log_alpha_w_noise = self.gumbel_sinkhorn(log_alpha)

        P = self.inv_soft_pers_flattened(soft_perms_inf,self.K)
        return torch.transpose(P,2,3), latent

    def inv_soft_pers_flattened(self,soft_perms_inf,n_numbers):
        inv_soft_perms = torch.transpose(soft_perms_inf, 2, 3)
        inv_soft_perms = torch.transpose(inv_soft_perms, 0, 1)

        return inv_soft_perms

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).float()
        return -torch.log(eps - torch.log(U + eps))

    def gumbel_sinkhorn(self,log_alpha):

        n = log_alpha.size()[1]
        log_alpha = log_alpha.view(-1, n, n)
        batch_size = log_alpha.size()[0]

        log_alpha_w_noise = log_alpha.repeat(self.n_samples, 1, 1)

        if self.noise_factor == 0:
            noise = 0.0
        else:
            noise = self.sample_gumbel([self.n_samples*batch_size, n, n])*self.noise_factor

        log_alpha_w_noise = log_alpha_w_noise + noise
        log_alpha_w_noise = log_alpha_w_noise / self.temp

        my_log_alpha_w_noise = log_alpha_w_noise.clone()

        sink = self.sinkhorn(my_log_alpha_w_noise)

        sink = sink.view(self.n_samples, batch_size, n, n)
        sink = torch.transpose(sink, 1, 0)
        log_alpha_w_noise = log_alpha_w_noise.view(self.n_samples, batch_size, n, n)
        log_alpha_w_noise = torch.transpose(log_alpha_w_noise, 1, 0)

        return sink, log_alpha_w_noise

    def sinkhorn(self,log_alpha, n_iters = 20):

        n = log_alpha.size()[1]
        log_alpha = log_alpha.view(-1, n, n)

        for i in range(n_iters):
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(-1, n, 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(-1, 1, n)
        return torch.exp(log_alpha)


def get_projection(P,z,x,k=16,dim=2,hard=False):

    if hard:
        row_ind, col_ind = linear_sum_assignment(1-P.detach().numpy())
        P = torch.zeros_like(P)
        P[row_ind,col_ind] = 1
        
    # Some magic to scale it up to k*d dims
    H = torch.zeros(dim*k,dim*k*2)
    H[0:k,0:k] = P[0:z.shape[0],:]
    H[k:,k:2*k] = P[0:z.shape[0],:]

    return H

def forward(Ps,Z,state_init,cov_init,k=10,d=2,sigR=0.1,sigQ=0.1,hard=False,dt=0.01):
    
    cov = [cov_init.clone()]
    state = [state_init.clone()]
    cov_pred = [torch.empty_like(cov_init).copy_(cov_init)]
    state_pred = [torch.empty_like(state_init).copy_(state_init)]
    Ht_list = []
    
    Q = torch.zeros(2*d*k,2*d*k)
    Q[d*k:,d*k:] = sigQ*torch.eye(d*k)
    
    # RW motion model
    F = torch.eye(2*d*k)
    for j in range(d*k):
        F[j,d*k+j] = dt
    
    for i in range(1,Z.shape[0]+1):
        
        state_pred.append(torch.matmul(F,state[i-1]))
        cov_pred.append(torch.matmul(torch.matmul(F,cov[i-1]),F.T) + Q)

        Ht = get_projection(Ps[i-1,:,:],Z[i-1],state_pred[i],k=k,dim=d,hard=hard)
        
        Ht_list.append(Ht)
        R = sigR*torch.eye(Ht.shape[0])
        S = torch.matmul(torch.matmul(Ht,cov_pred[i]),Ht.T) + R
        K = torch.matmul(torch.matmul(cov_pred[i],Ht.T),torch.inverse(S))

        state.append(state_pred[i] + torch.matmul(K,(Z[i-1].reshape(-1,1)- (torch.matmul(Ht,state_pred[i])))))
        cov.append(torch.matmul((torch.eye(2*d*k) - torch.matmul(K,Ht)),cov_pred[i]))
    return state,cov,state_pred,cov_pred,Ht_list
    
def backward(state,cov,state_pred,cov_pred,k=10,d=2,dt=0.01):
    
    # RW motion model
    F = torch.eye(2*d*k)
    for j in range(d*k):
        F[j,d*k+j] = dt
        
    for i in range(len(state)-2,1,-1):
    
        C = torch.matmul(torch.matmul(cov[i],F.T),torch.inverse(cov_pred[i+1]))
        state[i] =  state[i] + torch.matmul(C,(state[i+1] - state_pred[i+1]))
        cov[i] =  cov[i] + torch.matmul(torch.matmul(C,(cov[i+1] - cov_pred[i+1])),C.T)

    return state, cov


def ll(z,cov,meas,sigR):
    R = sigR*torch.eye(cov.shape[0])
    return torch.distributions.MultivariateNormal(meas.reshape(-1,),torch.unsqueeze(cov+R,dim=0)).log_prob(z.reshape(1,-1))
