import torch
import numpy as np

class ShuffleDataset(torch.utils.data.Dataset):

    def __init__(self, ims, pos, K=3, l=50,d=10,width=100):
        super(ShuffleDataset, self).__init__()
        
        self.K = K
        self.l = l
        self.d = d
        self.w = width
        self.ims = ims
        self.pos = pos

    def __getitem__(self, index): 
        
        ims_batch = []
        meas_batch = []
        bins_batch = []
        
        for i in range(index,index+self.l):

            im = self.ims[i]
            patches = np.zeros((self.K,2*self.d,2*self.d))
            for j in range(self.K): 
                x = int(self.pos[i][j,0])
                y = int(self.pos[i][j,1])
                pt = im[max(x-self.d,0):min(x+self.d,self.w),max(y-self.d,0):min(y+self.d,self.w)]
                
                patches[j,0:pt.shape[0],0:pt.shape[1]] = pt
            
            idx = np.arange(self.K)
            np.random.shuffle(idx)
  
            meas_batch.append(torch.from_numpy(self.pos[i][idx,:]).float())
            ims_batch.append(torch.from_numpy(patches[idx,:,:]))
            original_ims = np.array(self.ims[index:index+self.l])
        return torch.from_numpy(original_ims).float(), torch.stack(ims_batch).float(), torch.stack(meas_batch).float()

    def __len__(self):
        return len(self.ims)-self.l
