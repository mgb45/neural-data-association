import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from IPython import display

from envs import CollectData
from datahandler import ShuffleDataset

from models import SinkhornNet, forward, backward, ll

import argparse
import uuid
import csv

parser = argparse.ArgumentParser(description='Run a nda experiment.')
parser.add_argument('-K',type=int,help='Number of targets.',default=4)
parser.add_argument('-pd',type=int,help='Dimensionality of patch cropped around measurement.',default=10)
parser.add_argument('-ssl',type=int,help='Lengths of subsequences used for training.',default=50)
parser.add_argument('-batch_size',type=int,help='Training batch size.',default=8)
parser.add_argument('-epochs',type=int,help='Training epochs.',default=5)
parser.add_argument('-sigR',type=float,help='Measurement noise.',default=5)
parser.add_argument('-sigInit',type=float,help='Initial state uncertainty.',default=300)
parser.add_argument('-lr',type=float,help='Learning rate.',default=1e-3)

args = parser.parse_args()
K = args.K # Number of targets

patch_dim = args.pd # patch size of window around detections
l = args.ssl # subsequence length
batch_size = args.batch_size #batch size for training updates
epochs = args.epochs

sigR = args.sigR
sigInit = args.sigInit

lr = args.lr

id = str(uuid.uuid4())

with open('./logs/'+id+'.csv','w') as f:
    for key in vars(args).keys():
        f.write("%s,%s\n"%(key,vars(args)[key]))

# Collect some training data (100 motion sequences, 100 steps long)
dg = CollectData(K=K)

d = dg.positions[0][0].shape[-1] # Target state dimensionality

# Define a sinkhorn net for data association and optimizer
sn = SinkhornNet(latent_dim=8, K=K, d=d, n_samples=1, noise_factor=1.0, temp=1.0, n_iters=25, sigQ=5)
optimizer = torch.optim.Adam(sn.parameters(),lr=lr)

# Training loop
    
# Initialise with high uncertainty
state_init = torch.zeros(K*2*2,1)
cov_init = sigInit*torch.eye(K*2*2)

# Repeat over multiple epochs
for epoch in range(epochs):
    
    for n in range(len(dg.ims)):

        #Sample a new training sequence task
        dset = ShuffleDataset(dg.ims[n],dg.positions[n],l=l,K=K,d=patch_dim)
        sampler = DataLoader(dset, batch_size=1, shuffle=True)

        loss_bb = []
        #Train for 1 epoch on a subsequence drawn from this
        for ims_batch, patches_batch, meas_batch, _ in sampler:

                meas_batch = torch.transpose(torch.squeeze(meas_batch),1,2).reshape(l,-1)

                batch_losses_b = []

                # Predict permutations
                Ps,latents = sn(patches_batch.reshape(-1,K,2*patch_dim,2*patch_dim),meas_batch)
                loss_b = []

                for P in Ps:

                    # Get covariance params
                    sigQ = torch.exp(sn.logSigQ)

                    # Forward-Backward RTS
                    state,cov,state_pred,cov_pred,H_list = forward(P,meas_batch,state_init,cov_init,k=K,d=2,sigR=sigR,sigQ=sigQ)
                    state,cov = backward(state,cov,state_pred,cov_pred,k=K,d=2)

                    # Marginal Likelihood
                    cov_z = [torch.matmul(torch.matmul(H_list[i],cov[i+1]),H_list[i].T) for i in range(len(H_list))]
                    z_x = [torch.matmul(H_list[i],state[i+1]) for i in range(len(H_list))]
                    lls = [ll(z_x[i],cov_z[i],torch.squeeze(meas_batch)[i],sigR=sigR) for i in range(len(H_list))]

                    #Append losses
                    loss_b.append(torch.sum(-torch.stack(lls)))
                    
                loss_bb.append(torch.mean(torch.stack(loss_b)))

                # Step after batch size updates
                if len(loss_bb) >= batch_size:
                    loss = torch.mean(torch.stack(loss_bb))
                    loss.backward()
                    optimizer.step()

                    optimizer.zero_grad()
                    
                    loss_bb = []

    torch.save(sn.state_dict(),'./logs/'+id+'.npy')

