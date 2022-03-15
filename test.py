import torch
import numpy as np
import random

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

import csv
import glob

from torch.utils.data import Dataset, DataLoader
from envs import CollectData
from datahandler import ShuffleDataset

from models import SinkhornNet, forward, backward, ll

files = glob.glob('./logs/*.csv')

print('Testing models:',files)

for f in files:
    with open(f,'r') as infile:

        reader = csv.reader(infile)
        param_dict = {rows[0]:rows[1] for rows in reader}

        K = int(param_dict['K'])
        pd = int(param_dict['pd'])
        l = int(param_dict['ssl'])

        dg = CollectData(K=K)

        d = dg.positions[0][0].shape[-1] # Target state dimensionality
        
        sn = SinkhornNet(latent_dim=8, K=K, d=d, n_samples=1, noise_factor=1.0, temp=1.0, n_iters=25, sigQ=5)
        sn.load_state_dict(torch.load(f[:-4]+'.npy'))

        x = np.zeros((K,K)) 
        for rep in range(len(dg.ims)):
            
            test_dset = ShuffleDataset(dg.ims[rep],dg.positions[rep],l=l,K=K,d=pd)
            test_sampler = DataLoader(test_dset, batch_size=1, shuffle=False)

            # Check association with true labels - ideally we should see a 1-1 mapping between learned labels and actual labels    
            ims_batch, patches_batch, meas_batch,bins_batch = next(iter(test_sampler))
                
            meas_batch = torch.transpose(torch.squeeze(meas_batch),1,2).reshape(l,-1)
                    
            logits,_ = sn(patches_batch.reshape(-1,K,2*pd,2*pd),meas_batch)
            bins_pred = torch.argmax(torch.squeeze(logits),dim=2)
            
            for i in range(bins_batch.shape[1]):
                x[bins_pred[i,:],torch.squeeze(bins_batch)[i,:]] = x[bins_pred[i,:],torch.squeeze(bins_batch)[i,:]] + 1

        np.savetxt(f[:-4]+'_result.txt',x)
        print("Accuracy = ",100*np.sum(np.max(x,axis=1))/np.sum(x),' K =',str(K))


