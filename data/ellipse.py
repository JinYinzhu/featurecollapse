import numpy as np
import torch
import os

class Ellipse():
    def __init__(self, data_path, device = 'cpu'):
        circle = np.load(os.path.join(data_path,'X_0.npy'))[:500]
        line = np.load(os.path.join(data_path,'X_1.npy'))[:500]
        X = np.concatenate([circle,line],0)
        X = np.expand_dims(X,1).astype(np.float32)
        self.X = torch.Tensor(X).to(device)
        self.rv = None
        self.device = device
        
    def xpos(self, x):
        x = x.squeeze()
        xidx = torch.arange(x.shape[1]).float().unsqueeze(1).to(self.device)
        h_sum = torch.sum(torch.matmul(x,xidx))
        scaler = torch.sum(x)
        mu = h_sum/scaler
        return mu
    
    def ypos(self, x):
        x = x.squeeze()
        yidx = torch.arange(x.shape[0]).float().unsqueeze(0).to(self.device)
        v_sum = torch.sum(torch.matmul(yidx,x))
        scaler = torch.sum(x)
        mu = v_sum/scaler
        return mu
    
    def size(self, x):
        return torch.sum(x)
    
    def ar(self, x):
        #log aspect ratio
        x = x.squeeze()
        yidx = torch.arange(x.shape[0]).float().unsqueeze(0).to(self.device)
        xidx = torch.arange(x.shape[1]).float().unsqueeze(1).to(self.device)
        v_sum = torch.sum(torch.matmul(yidx,x))
        h_sum = torch.sum(torch.matmul(x,xidx))
        scaler = torch.sum(x)
        mu = (v_sum/scaler,h_sum/scaler)
        S_v = torch.sum(torch.matmul((yidx-mu[0])**2,x))/scaler          
        S_h = torch.sum(torch.matmul(x,(xidx-mu[1])**2))/scaler
        tmp = (xidx-mu[1]).repeat(1,x.shape[0]).T*(yidx-mu[0]).repeat(x.shape[1],1).T
        S_vh = torch.sum(tmp*x)/scaler
        trace = S_v+S_h
        tolerance = 1e-8
        det = S_v*S_h-S_vh**2
        disc = trace**2-4*det
        eval1 = 0.5*(trace+torch.sqrt(disc))
        eval2 = 0.5*(trace-torch.sqrt(disc))
        logar = torch.log(eval1+tolerance)-torch.log(eval2+tolerance)
        return logar
    
    def set_rv(self, rv_value=None):
        if rv_value is None:
            self.rv = np.random.normal(0,1,(28*28,))
        elif len(rv_value.reshape(-1))==28*28:
            self.rv = rv_value.reshape(-1)
        else:
            raise Exception("Incorrect random vector size")
        self.rv = torch.Tensor(self.rv).to(self.device)
    
    def RLF(self, x):
        if self.rv is None:
            raise Exception("Random vector not set")
        xt = x.reshape(-1)
        return torch.sum(xt.dot(self.rv))