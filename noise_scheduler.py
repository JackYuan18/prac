#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 14:34:24 2026

@author: zyuan
"""
import torch
import matplotlib.pyplot as plt
device = 'cpu'
class noise_scheduler:
    def __init__(self, T=1000, beta0=1e-3, betaT=0.02):
        self.T = 1000
        self.beta0 = beta0
        self.betaT = betaT
        self.betas = torch.linspace(1e-4, 0.02, T, device = device)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim = 0)
    
    
        

    def add_noise(self, x, t, noise):
        """Add noise to images based on alpha blending."""
        
        alpha_t = self.alpha_cumprod[t]
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        noisy = x * alpha_t + noise * (1-alpha_t)
        return noisy
    
    
data = load_flower_dataset(create_dataloaders=True, batch_size=8, image_size=256)
train_load = data['train_loader']

x,l = next(iter(train_load))
x = x[0]

fig,ax = plt.subplots(2,1, figsize = (12,5))
ax[0].imshow(x.permute(1,2,0))

noise = torch.randn_like(x)
ns = noise_scheduler()
xn = ns.add_noise(x, 500, noise)
ax[1].imshow(xn.squeeze().permute(1,2,0))