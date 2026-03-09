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

data_train = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('.',
                                   transform=torchvision.transforms.ToTensor(),
                                   train=True,
                                   download=True
                                ),
        batch_size=128,

        shuffle=True)
data_test = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('.',
                                   transform=torchvision.transforms.ToTensor(),
                                   train=False,
                                   download=True),
        batch_size=128,

        shuffle=True)

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, num_classes)         
            
            
            )
    def forward(self, x):
        logits = self.net(x)
        prob = torch.nn.functional.softmax(logits, dim = -1)
        return logits, prob

class CNN(nn.Module):
    
    def __init__(self, inchannels=1, num_classes = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inchannels, 8, kernel_size = 3, padding = 1 ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride =2),
            nn.Dropout(),
            nn.Conv2d(8, 16, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size =2, stride = 2),
            nn.Dropout()
                 
            )        
        self.fc = nn.Linear(16*7*7, num_classes)
        

    
    def forward(self, x):
        h = self.net(x)
        
        logits = self.fc(h.flatten(start_dim = 1))
        prob = torch.softmax(logits, dim  = -1)
        return logits, prob
    


cnn = MLP(28*28).to(device)
optim = torch.optim.Adam(cnn.parameters(), lr = 0.0001)
epochs = 10
loss_fn = torch.nn.CrossEntropyLoss()
losses = []
cnn.train()
for _ in tqdm.tqdm(range(epochs)):
    for x,y in data_train:
        x = x.to(device)
        y = y.to(device)
        
        x = x.flatten(start_dim = 1)
        optim.zero_grad()
        logits, prob = cnn(x)
        
        loss = loss_fn(logits, y)
        
        losses.append(loss.detach().cpu().numpy())
        loss.backward()
        optim.step()

plt.plot(losses)
cnn.eval()
num_correct = 0
num_sample = 0



# x_test,y_test = next(iter(data_test))
# idx = 20
# x = x_test[idx]
# y = y_test[idx]
# plt.imshow(x.permute(1,2,0))
# _, prob = cnn(x.unsqueeze(0))
# pred = torch.max(prob, dim = -1).indices
# print(pred,y)

with torch.no_grad():
    for x, y in data_test:
        x = x.to(device)
        y = y.to(device)
        
        x = x.flatten(start_dim=1)
        _, prob = cnn(x)
        pred = torch.max(prob, dim = -1).indices
        
        num_correct += torch.sum(pred==y)
        num_sample += y.shape[0]


    accuracy = num_correct/num_sample

print(accuracy)        
        
        
        

        
        
        
        
        
