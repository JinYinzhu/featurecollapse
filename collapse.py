import numpy as np
import torch

class FeatureCollapser():
    
    def __init__(self, f, vae, baseline_val=None):
        #f: a function that maps a data point to its feature value, pytorch differentiable
        #vae: a vae trained on the data
        #baseline_val: the baseline feature value to collapse to
        
        self.f = f
        self.vae = vae
        self.baseline_val = baseline_val
    
    def fit(self, X):
        #this is to set the baseline value as the mean feature values of the given dataset
        f_val_sum = 0
        for x in X:
            f_val_sum += self.f(x).detach().item()
        self.baseline_val = f_val_sum/len(X)
    
    def transform(self, X, step_size, max_step=1000):
        #X: list of data points to collapse its feature dimension to the baseline value
        #step_size: the step size for the integration to the baseline set,
        #           larger step size will result in inaccurate results,
        #           smaller step size will take longer time
        #max_step: max step for integration,
        #          if this number is reached before hittibg baseline value,
        #          an exception will be raised
        
        if self.baseline_val is None:
            raise Exception("Baseline value is not set")
        
        X_new = []
        cnt = 0
        leng = len(X)
        points = [int(leng/10*(i+1)) for i in range(10)]
        for x in X:
            cnt += 1
            if cnt in points:
                print(str(cnt)+'/'+str(leng), end=' ')
            z = self.vae.encode(x.unsqueeze(0))
            zi = z.detach().clone()
            zi.requires_grad = True
            xi = self.vae.decode(zi)
            f0 = self.f(xi).detach().item()
            
            for i in range(max_step):
                fx = self.f(xi)
                if (fx-self.baseline_val)*(f0-self.baseline_val)<=0:
                    break
                fx.backward()
                gradz = zi.grad.data.detach()
                
                newz = zi - step_size*gradz*torch.sign(fx-self.baseline_val)
                zi = newz.detach().clone()
                zi.requires_grad = True
                xi = self.vae.decode(zi)
             
            if i==max_step-1:
                raise Exception("Maximum step reached, try larger step size or larger maximum step\n"\
                               +"It could also indicate your feature function is not differentiable everywhere")
            
            X_new.append(xi.detach())
        
        return X_new
    
    def fit_transform(self, X, step_size, max_step=1000):
        self.fit(X)
        X_new = self.transform(X, step_size, max_step)
        
        return X_new
        
