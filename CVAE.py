# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 22:27:26 2026

@author: yzyja
"""
# Avoid OpenMP conflict on Windows (multiple OpenMP runtimes loaded)
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils
import torch.distributions
import torchvision
import lightning.pytorch as pl
import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Use GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('.',
                                   transform=torchvision.transforms.ToTensor(),
                                   download=True),
        batch_size=128,
        shuffle=True)

class CondVariationalEncoder(nn.Module):
    def __init__(self,latent_dims,n_classes):
        super().__init__()
        self.linear1 = nn.Linear(784 + n_classes, 512) #map to hidden state
        self.linear2 = nn.Linear(512, latent_dims) #map to mean
        self.linear3 = nn.Linear(512, latent_dims) #map to std
        
        self.N = torch.distributions.Normal(
            torch.tensor(0., device=device),
            torch.tensor(1., device=device)
            )
  
        self.kl = 0
    def forward(self, x, y):
        x = torch.flatten(x,start_dim=1)
        # x = x.view(-1, 1*28*28)
        
        x = functional.relu(self.linear1(torch.cat((x,y),dim=1)))
        
        mu = self.linear2(x)
        
        sigma = torch.exp(self.linear3(x))
        
        z = mu + sigma * self.N.sample(mu.shape)
        
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
class CondVariationDecoder(nn.Module):
    
    def __init__(self, latent_dims, n_classes):
        super().__init__()
        self.linear1 = nn.Linear(latent_dims + n_classes,512)
        self.linear2 = nn.Linear(512,784)
        
    def forward(self,z,y):
        z = functional.relu(self.linear1(torch.cat((z,y), dim=1)))
        z = torch.sigmoid(self.linear2(z))
        
        return z.reshape((-1,1,28,28))
    
    
    
class CondVariationalAutoencoder(nn.Module):
    
    def __init__(self,latent_dims, n_classes):
        super().__init__()
        self.encoder = CondVariationalEncoder(latent_dims, n_classes)
        self.decoder = CondVariationDecoder(latent_dims, n_classes)
    
    def forward(self, x,y):
        z = self.encoder(x,y)
        return self.decoder(z,y)
    
    
class CVAEModel(pl.LightningModule):
    def __init__(self, latent_dims, n_classes):
        super().__init__()
        self.cvae = CondVariationalAutoencoder(latent_dims, n_classes)
        self.n_classes = n_classes
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.02)
    def training_step(self,batch):
        x, y, = batch
        y_oh = functional.one_hot(y, num_classes = self.n_classes)
        
        x_hat = self.cvae(x,y_oh)
        loss = ((x-x_hat)**2).sum() + self.cvae.encoder.kl
        
        self.log('Training loss', loss, on_step = False, on_epoch=True, logger=False, prog_bar = True)
        
        return loss
    
    
    
    def train(self,data, max_epoch=10):
        for _ in range(max_epoch):
            for batch in tqdm.tqdm(data, desc=f'Training....epoch {_}'):
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            

def plot_reconstructed(autoencoder, r0=(-3, 3), r1=(-3, 3),
                       n=8, number=2, device='cuda'):
    # Define plot array:
    fig, axs = plt.subplots(n, n)

    # Loop over a grid in the latent space
    for i, a in enumerate(np.linspace(*r1, n)):
        for j, b in enumerate(np.linspace(*r0, n)):

            z = torch.Tensor([[a, b]]).to(device)
            # One-hot encoding of the integer
            y = functional.one_hot(torch.tensor([number]),
                                   num_classes=10).to(device)
            # Forwarding the data through the decoder
            x_hat = autoencoder.decoder(z, y)

            x_hat = x_hat.reshape(28, 28).detach().cpu().numpy()
            axs[i, j].imshow(x_hat)
            axs[i, j].axis('off')
    plt.show()

latent_dims = 2
model = CVAEModel(latent_dims, 10)


# trainer = pl.Trainer(devices=1, accelerator='gpu' if torch.cuda.is_available() else 'cpu', max_epochs=10)
# trainer.fit(model,data)

model = model.to(device)
model.train(data)
plot_reconstructed(model.cvae, number=8, device=device)
def plot_latent_cvae(autoencoder, data, num_batches=100, device='cpu'):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device),
                                torch.nn.functional.one_hot(torch.tensor(y),
                                                            num_classes=10).to(device))
        z = z.detach().cpu().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break

model = model.to(device)
plot_latent_cvae(model.cvae, data, device=device)

plt.figure()
z = torch.normal(0,1,(1,2)).to(device)
# One-hot encoding of the integer
y = functional.one_hot(torch.tensor([1]),
                       num_classes=10).to(device)
# Forwarding the data through the decoder
x_hat = model.cvae.decoder(z, y)

x_hat = x_hat.reshape(28, 28).detach().cpu().numpy()
plt.imshow(x_hat)
plt.axis('off')