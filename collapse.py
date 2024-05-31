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
        for x in X:
            xi = x.detach().clone()
            f0 = self.f(xi).detach().item()
            xi.requires_grad = True
            
            for i in range(max_step):
                fx = self.f(xi)
                if (fx-self.baseline_val)*(f0-self.baseline_val)<=0:
                    break
                fx.backward()
                gradx = xi.grad.data.detach()
                
                newx = xi - step_size*gradx*torch.sign(fx-self.baseline_val)
                xi = newx.detach().clone()
                xi.requires_grad = True
             
            if i==max_step-1:
                raise Exception("Maximum step reached, try larger step size or larger maximum step\n"\
                               +"It could also indicate your feature function is not differentiable everywhere")
            
            X_new.append(xi.detach())
        
        return X_new
    
    def fit_transform(self, X, step_size, max_step=1000):
        self.fit(X)
        X_new = self.transform(X, step_size, max_step)
        
        return X_new
        
