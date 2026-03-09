# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 11:45:32 2026

@author: yzyja
"""
import numpy as np

    
def ReLU(z):
    a=  np.ones(z.shape)
    a[z<=0] = 0
    a[z>0] = 1
    a = a*z
    return a
    

class MLP:
    
    def __init__(self):
        self.W1 = np.random.normal(0,1, (2,2))
        self.W2 = np.random.normal(0,1, (2,2))
    
        self.W1 = np.array([[1,2], [3,-4]])
        self.W2 = np.array([[4], [2]])
    def forward(self,x):
        """
        x: (1,2)
        
        """
        self.z1 = x @ self.W1
        self.a1 = ReLU(self.z1)
        self.pred = self.a1 @ self.W2
        return self.pred
    
    
    def backward(self, y):
        self.loss = (self.pred - y)**2
        self.dw2 = 2*(self.pred - y)*self.a1
        
        dlossda1 = 2*(self.pred - y)*self.W2
        da1dz1 = (self.z1>0).astype(np.float32)
        dz1dw1 = x
        
        self.dw1 = dz1dw1 * da1dz1 * dlossda1        
    
    def step(self):
        self.W1 = self.W1 - 0.001* self.dw1
        self.W2 = self.W2 - 0.001* self.dw2
        
        


x = np.array([[1,2]])

nn = MLP()

y = nn.forward(x)
print(f'W1: {nn.W1}')
print(f'W2: {nn.W2}')
print(f'z1: {nn.z1}')
print(f'a1: {nn.a1}')
print(f'output: {y}')

truth = np.array([[1,2]])
nn.backward(truth)

nn.step()
print(f'W1: {nn.W1}')
print(f'W2: {nn.W2}')
print(f'z1: {nn.z1}')
print(f'a1: {nn.a1}')
print(f'output: {y}')
    
        