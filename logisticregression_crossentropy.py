# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 23:08:59 2026

@author: yzyja
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
class DNN(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.x_dim = x_dim
        
        self.net = nn.Linear(x_dim, 2, bias=True)                 
            
        
        
    def forward(self,x):
        return self.net(x)

    

# def loss_fn(p,y):
#     return torch.mean(-y*torch.log(p) - (1-y)*torch.log(p))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__=='__main__':
    
    torch.manual_seed(0)
    
    X1 = torch.randn((100,1),device = dev)+5
    X2 = torch.randn((100,1), device = dev)+0
    
    Y1 = torch.ones((X1.shape[0],), device = dev, dtype = torch.long)
    Y2 = torch.zeros((X1.shape[0],), device = dev, dtype = torch.long)
    
    plt.scatter(X1.cpu(),Y1.cpu())
    plt.scatter(X2.cpu(),Y2.cpu())
    
    
    net = DNN(X1.shape[1]).to(dev)
    optim = torch.optim.Adam(net.parameters(), lr = 0.0001)
    X = torch.cat((X1,X2), dim = 0)
    Y = torch.cat((Y1,Y2), dim  = 0)
    
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    net.train()
    for i in range(5000):
        optim.zero_grad()
        
        logits = net(X)
        # prob = torch.softmax(logits, dim =1)
        
        loss = loss_fn(logits,Y)
        losses.append(loss.detach().cpu().numpy())
        loss.backward()
        optim.step()
    net.eval()
    with torch.no_grad():
        X_test = torch.linspace(-5, 10, 1000).to(dev).view(-1,1)
        logits = net(X_test)
        prob = nn.functional.softmax(logits, dim =1)
    
    plt.figure()
    plt.plot(X_test.cpu(), prob[:,1].cpu())
    plt.scatter(X1.cpu(),Y1.cpu())
    plt.scatter(X2.cpu(),Y2.cpu())
    plt.figure()
    plt.plot(losses)
        
            
    
    
    
    
    
    
    
    