import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self,latent_dim):
        super(VAE,self).__init__()
        
        self.conv1 = nn.Conv2d(1,32,kernel_size=2,stride=2,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=2,stride=2,padding=1)
        self.conv3 = nn.Conv2d(64,64,kernel_size=2,stride=2,padding=1)
        self.conv4 = nn.Conv2d(64,64,kernel_size=2,stride=2,padding=1)
        
        self.elu = nn.ELU()
        self.flatten = nn.Flatten()
        
        self.linear1 = nn.Linear(576,256)
        self.linear2 = nn.Linear(256,256)
        
        self.linear3 = nn.Linear(256,latent_dim)
        self.linear4 = nn.Linear(256,latent_dim)
        
        self.linear5 = nn.Linear(latent_dim,256)
        self.linear6 = nn.Linear(256,576)
        
        self.convT1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2,padding=1,output_padding=1)
        self.convT2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2,padding=1)
        self.convT3 = nn.ConvTranspose2d(64,32,kernel_size=2,stride=2,padding=1,output_padding=1)
        self.convT4 = nn.ConvTranspose2d(32,1,kernel_size=2,stride=2,padding=1)
        
        self.sigmoid = nn.Sigmoid()
    
    def reparameterize(self,mu,logvar):
        epsilon = torch.randn(mu.shape).to(mu.device)
        return mu + epsilon*torch.exp(logvar/2)
    
    def encode2(self,x):
        #used for training
        h = self.elu(self.conv1(x))
        h = self.elu(self.conv2(h))
        h = self.elu(self.conv3(h))
        h = self.elu(self.conv4(h))
        h = self.flatten(h)
        h = self.elu(self.linear1(h))
        h = self.elu(self.linear2(h))
        
        mu = self.linear3(h)
        logvar = self.linear4(h)
        z = self.reparameterize(mu,logvar)
        
        return z, mu, logvar
    
    def encode(self,x):
        h = self.elu(self.conv1(x))
        h = self.elu(self.conv2(h))
        h = self.elu(self.conv3(h))
        h = self.elu(self.conv4(h))
        h = self.flatten(h)
        h = self.elu(self.linear1(h))
        h = self.elu(self.linear2(h))
        
        z = self.linear3(h)
        
        return z
    
    def decode(self,z):
        h = self.elu(self.linear5(z))
        h = self.elu(self.linear6(h))
        h = h.reshape(-1,64,3,3)
        h = self.elu(self.convT1(h))
        h = self.elu(self.convT2(h))
        h = self.elu(self.convT3(h))
        y = self.sigmoid(self.convT4(h))
        
        return y
    
    def forward(self,x):
        z, mu, logvar = self.encode2(x)
        y = self.decode(z)
        return y, mu, logvar, z