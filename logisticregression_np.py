# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 23:08:59 2026

@author: yzyja
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
class logisticregression:
    def __init__(self, lr = 0.001, iterations = 1000):
        self.lr = lr
        self.iterations = iterations
        self.weight = np.random.normal(0,0.1)
        self.bias = np.random.normal(0,0.1)
        self.cost_history = []
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def prob(self,x):
        return self.sigmoid(x*self.weight+self.bias)
    
    def pred(self,x):
        prob = self.prob(x)
        return np.int(prob>0.5)
    
    def cost(self, h,y):
        return np.mean( - y*np.log(h)-(1-y)*np.log(h))
    
    def fit(self, X,Y):
        for _ in range(self.iterations):
            z = X * self.weight + self.bias
            h = self.sigmoid(z)
            
            dw = np.mean(X.T @ (h - Y))
            db = np.mean(h - Y)
            
            self.weight = self.weight - self.lr * dw
            self.bias -= self.lr * db
            
            self.cost_history.append(self.cost(h,Y))
            
    
def loss_fn(p,y):
    return torch.mean(-y*torch.log(p) - (1-y)*torch.log(p))

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__=='__main__':
    
    np.random.seed(0)
    
    X1 = np.random.normal(0,1,(100,1))+5
    X2 = np.random.normal(0,1,(100,1))+0
    
    Y1 = np.ones((X1.shape[0],1))
    Y2 = np.zeros((X1.shape[0],1))
    
    plt.scatter(X1,Y1)
    plt.scatter(X2,Y2)
    
    
    
    X = np.vstack((X1,X2))
    Y = np.vstack((Y1,Y2))
    
    # loss_fn = nn.CrossEntropyLoss()
    model=  logisticregression()
    model.fit(X, Y)
    
    X_test = np.linspace(-5, 10, 1000)
    X_test = np.reshape(X_test,(-1,1))
    prob = model.prob(X_test)
        # prob = nn.functional.softmax(logits, dim =1)
    
    plt.figure()
    plt.plot(X_test, prob)
    plt.scatter(X1,Y1)
    plt.scatter(X2,Y2)
    plt.figure()
    plt.plot(model.cost_history)
        
            
    
    
    
    
    
    
    
    