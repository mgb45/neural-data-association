import numpy as np


# Setup a toy environment
class SimpleEnv():

    def __init__(self,N=100,K=3,w=5,dt=0.1):
        
        self.dt = dt
        self.N = N
        self.K = K
        self.w = w
        self.reset()
        
    def reset(self):
        
        self.pos = np.random.randint(0,self.N,(self.K,2))
        self.vel = np.zeros((self.K,2))
        
    def step(self,u):
        
        # Simple dynamics model
        self.vel = self.vel + self.dt*u
        self.pos = self.pos + self.vel*self.dt
        
        self.pos = np.clip(self.pos,self.w,self.N-self.w)
        
        self.vel[self.pos>=(self.N-self.w)] = -self.vel[self.pos>=(self.N-self.w)]
        self.vel[self.pos<=(self.w)] = -self.vel[self.pos<=(self.w)]
        
    def render(self):
        
        image = np.zeros((self.N,self.N))
        
        pos = self.pos.astype(int)
        for k in range(self.K):
            image[pos[k,0]-self.w:pos[k,0]+self.w,pos[k,1]-self.w:pos[k,1]+self.w] = (k+1)/self.K
            
        return image, self.pos, self.vel

class CollectData:
    
    def __init__(self,Nsteps=100, reps=100,K=3):
        
        self.Nsteps = Nsteps
        self.reps = reps
        self.K = K
        
        self.ims = []
        self.positions = []
        
        self.collect()
    
    def collect(self):
        env = SimpleEnv(K=self.K)

        for rep in range(self.reps):
            env.reset()

            acc = 100*np.random.randn(self.K,2)

            pos_list = []
            im_list = []

            for step in range(self.Nsteps):
                acc = acc*0.9 + 0.1*np.random.randn(self.K,2)

                env.step(acc)
                image,pos,vel = env.render()

                pos_list.append(pos)
                im_list.append(image)
                
            self.ims.append(im_list)
            self.positions.append(pos_list)
