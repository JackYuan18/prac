# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 21:32:21 2026

@author: yzyja
"""

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('.',
                                   transform=torchvision.transforms.ToTensor(),
                                   download=True),
        batch_size=128,
        shuffle=True)

class AutoEncoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        
        self.encoder = Encoder(input_dim, hidden_dim, dropout)
        self.decoder = Decoder(hidden_dim,input_dim, dropout)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.02)
        self.criterion = nn.MSELoss()
        self.dropout = dropout
    def train_AE(self, data, max_epochs=200):
        pbar = tqdm.tqdm(range(max_epochs), desc="Training")

        for _ in pbar:
            epoch_loss = 0.0
            n_batches = 0
            for batch in data:
                batch = batch.to(device)
                z = self.encoder(batch)
                recon = self.decoder(z)
                
                loss = self.criterion(batch,recon)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / n_batches
        pbar.set_postfix(loss=avg_loss)

        
class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(            
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),        
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(dropout),        
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(dropout),        
            nn.Linear(16,hidden_dim),
            nn.ReLU()
        )    
    def forward(self, x):
        
        
        return self.network(x)
    
class Decoder(nn.Module):
    
    def __init__(self, hidden_dim,output_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),        
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Dropout(dropout),        
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Dropout(dropout),        
            nn.Linear(64,output_dim),
            nn.ReLU()
        # self.dropout4 = nn.Dropout(0.2)
        )
    def forward(self, x):
        
        
        return self.network(x)
    
    
label = 5

batch_size = 128
X = torch.empty(batch_size, 28*28, device=device)
for batch in data:
    x,y = batch
    x = x[y==label]
    x = torch.flatten(x, start_dim = 1, end_dim = 3)
    X = torch.cat((X,x))


class_data = [X[i:i+batch_size,:]for i in range(0,len(X),batch_size)]

train_data = class_data[:-5]
test_data = class_data[-5:]    

model = AutoEncoder(X.shape[-1],4,0.2)
model = model.to(device)

model.train()
model.train_AE(train_data)

model.eval()

test = test_data[0][5]

test_data = torch.cat(test_data)
z = model.encoder(test_data)
recon = model.decoder(z)

test_loss = model.criterion(test_data,recon).detach().cpu().numpy()

print(f'Testing loss: {test_loss: .2f}')



z = model.encoder(test)
recon  = model.decoder(z)

example_loss = model.criterion(test, recon).detach().cpu().numpy()

print(f'Example loss: {example_loss}')
recon_img = recon.reshape(28,28).detach().cpu().numpy()
test_img = test.reshape(28,28).detach().cpu().numpy()


plt.figure()
plt.imshow(test_img)

plt.figure()
plt.imshow(recon_img)









