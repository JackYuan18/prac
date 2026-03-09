# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 18:07:21 2026

@author: yzyja
"""

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
torch.manual_seed(0)

#data generation

class Linear(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(x_dim,1), requires_grad = True)
        self.b = nn.Parameter(torch.randn(x_dim,1), requires_grad = True)
        
    def forward(self, x):
        return x * self.w + self.b
    

class LinearNN(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.net = nn.Linear(x_dim, 1, bias=True)
        
    def forward(self, x):
        return self.net(x)

w_true = 3
b_true = 2

x = torch.linspace(0,10,100)
noise = torch.randn_like(x)
y_obs = w_true * x + b_true + noise

y_obs = y_obs.unsqueeze(0)

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

loss_fn  = torch.nn.MSELoss()
lr = 0.001

model = Linear(1)
optim = torch.optim.Adam(model.parameters(), lr = 0.001)

for _ in range(1000):
    optim.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y_obs)
    
    
    loss.backward()
    optim.step()

# print(model.w.item(), model.b.item())
# w = model.w.item()
# b = model.b.item()
y_pred = model(x).detach().squeeze()
plt.scatter(x, y_obs)
plt.plot(x, y_pred, color = 'red')
"""


for _ in range(100):
    
    
    y = w * x + b
    loss = loss_fn(y,y_obs)
    
    loss.backward()
    with torch.no_grad():    
        w -= lr * w.grad
        b -= lr * b.grad
    
    w.grad.zero_()
    b.grad.zero_()

print(w,b)
"""        
# w = w.detach()
# b = b.detach()

# plt.scatter(x, y_obs)
# plt.plot(x, w*x+b, color = 'red')


