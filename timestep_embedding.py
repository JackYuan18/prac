# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 21:25:15 2026

@author: yzyja
"""
import torch
import math
import matplotlib.pyplot as plt

def timestep_embedding(timesteps, embed_dim):
    """
    timesteps: (B,)
    returns: (B, dim)
    
    """
    
    half_dim = embed_dim//2
    freqs = torch.exp( - math.log(10000) * torch.arange(half_dim, dtype = torch.float32) / half_dim).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim = -1)
    
    return embedding

if __name__=='__main__':
    timesteps = torch.randint(0, 1000, (8,))
    embed_dim = 10
    
    half_dim = embed_dim//2
    freqs = torch.exp( - math.log(10000) * torch.arange(half_dim, dtype = torch.float32) / half_dim).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim = -1)
    
