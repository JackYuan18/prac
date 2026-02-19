#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 13:15:21 2026

@author: zyuan
"""
import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class UNet(nn.Module):
    
    def __init__(self,in_channel=3, out_channel=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channel, 32, kernel_size = 5, padding =2),
            nn.Conv2d(32, 64, kernel_size = 5, padding = 2),
            nn.Conv2d(64,64, kernel_size = 5, padding = 2)          
            ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size = 5, padding =2),
            nn.Conv2d(64, 32, kernel_size = 5, padding = 2),
            nn.Conv2d(32,out_channel, kernel_size = 5, padding = 2)          
            ])
        
        self.act = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor = 2)
    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))
            if i< 2:
                h.append(x)
                x = self.downscale(x)
        
        for i, l in enumerate(self.up_layers):
            if i>0:
                x = self.upscale(x)
                x += h.pop()
            x = self.act(l(x))
        return x

def load_flower_dataset(data_dir='/home/zyuan/.cache/kagglehub/datasets/nunenuh/pytorch-challange-flower-dataset/versions/3',
                        batch_size=32, image_size=224, create_dataloaders=True):
    """
    Load the flower dataset from the specified directory.
    
    Args:
        data_dir: Path to the dataset directory
        batch_size: Batch size for DataLoaders (default: 32)
        image_size: Target image size for transforms (default: 224)
        create_dataloaders: If True, return DataLoaders; if False, return only datasets (default: True)
    
    Returns:
        If create_dataloaders=True:
            dict with keys: 'train_loader', 'valid_loader', 'test_loader', 'datasets', 'cat_to_name', 'num_classes'
        If create_dataloaders=False:
            dict with keys: 'datasets', 'cat_to_name', 'num_classes'
    """
    dataset_path = os.path.join(data_dir, 'dataset')
    
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    valid_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    # Minimal transform: resize to same size and convert PIL to tensor (needed for DataLoader)
    minimal_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to same size for batching
        transforms.ToTensor()
    ])
    
    image_datasets = {
        'train': datasets.ImageFolder(
            root=os.path.join(dataset_path, 'train'),
            transform=minimal_transform
        ),
        'valid': datasets.ImageFolder(
            root=os.path.join(dataset_path, 'valid'),
            transform=minimal_transform
        )
    }
    
    # Test set may not have class folders (unlabeled test set)
    test_path = os.path.join(dataset_path, 'test')
    if os.path.exists(test_path):
        # Check if test directory has class folders or just images
        test_subdirs = [d for d in os.listdir(test_path) 
                        if os.path.isdir(os.path.join(test_path, d)) and d.isdigit()]
        if test_subdirs:
            # Test set has class folders
            image_datasets['test'] = datasets.ImageFolder(
                root=test_path,
                transform=minimal_transform
            )
        # If test set has no class folders, skip it (unlabeled test set)
    
    # Load category to name mapping
    cat_to_name_path = os.path.join(data_dir, 'cat_to_name.json')
    with open(cat_to_name_path, 'r') as f:
        cat_to_name = json.load(f)
    
    # Get number of classes
    num_classes = len(image_datasets['train'].classes)
    
    result = {
        'datasets': image_datasets,
        'cat_to_name': cat_to_name,
        'num_classes': num_classes
    }
    
    # Create DataLoaders if requested
    if create_dataloaders:
        dataloaders = {
            'train': DataLoader(
                image_datasets['train'],
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Set to 0 to avoid multiprocessing issues
                pin_memory=False
            ),
            'valid': DataLoader(
                image_datasets['valid'],
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # Set to 0 to avoid multiprocessing issues
                pin_memory=False
            )
        }
        result['train_loader'] = dataloaders['train']
        result['valid_loader'] = dataloaders['valid']
        
        # Only create test loader if test dataset exists 
        if 'test' in image_datasets:
            dataloaders['test'] = DataLoader(
                image_datasets['test'],
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # Set to 0 to avoid multiprocessing issues
                pin_memory=False
            )
            result['test_loader'] = dataloaders['test']
    
    return result

data = load_flower_dataset(create_dataloaders=True)
train_load = data['train_loader']
valid_load = data['valid_loader']
cat_to_name = data['cat_to_name']
images, labels = next(iter(train_load))
num_samples = len(images)
img = images[:5]
# img = img.permute(0,2, 3, 1)
# plt.imshow(img)

def add_noise(x, alpha):
    noise = torch.rand_like(x)
    alpha = alpha.view(-1,1,1,1)
    noisy = x*(1-alpha) + noise*alpha
    return noisy

fig, ax = plt.subplots(2,1)
ax[0].set_title('Input')
ax[0].imshow(torchvision.utils.make_grid(img).permute(1,2,0))

alpha = torch.linspace(0,1, img.shape[0])
x_noise = add_noise(img,alpha)
ax[1].set_title('Noisy input')
ax[1].imshow(torchvision.utils.make_grid(x_noise).permute(1,2,0))





# Visualize samples from training dataloader
def visualize_samples(dataloader, cat_to_name, num_samples=8, figsize=(12, 12)):
    """
    Display a grid of sample images from the dataloader.
    
    Args:
        dataloader: DataLoader to sample from
        cat_to_name: Dictionary mapping category numbers to names
        num_samples: Number of samples to display (default: 8)
        figsize: Figure size (default: (12, 12))
    """
    # Get a batch
    images, labels = next(iter(dataloader))
    
    # Limit to available samples
    num_samples = min(num_samples, len(images))
    
    # Calculate grid dimensions
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Get class names from dataset
    dataset = dataloader.dataset
    class_names = [cat_to_name.get(str(cls), f"Class {cls}") for cls in dataset.classes]
    
    for idx in range(num_samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Handle tensor (from ToTensor transform)
        img = images[idx]
        if isinstance(img, torch.Tensor):
            # Convert tensor to numpy (C, H, W) -> (H, W, C)
            img = img.permute(1, 2, 0).numpy()
            # ToTensor scales to [0, 1], so just clip to be safe
            img = np.clip(img, 0, 1)
        else:
            # Fallback for PIL Image (shouldn't happen with ToTensor)
            img = np.array(img)
            if img.max() > 1:
                img = img / 255.0
        
        ax.imshow(img)
        ax.axis('off')
        
        # Get label and class name
        label_idx = labels[idx].item()
        class_idx = dataset.classes[label_idx]
        class_name = cat_to_name.get(str(class_idx), f"Class {class_idx}")
        ax.set_title(f"Label: {label_idx}\n{class_name}", fontsize=10)
    
    # Hide extra subplots
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize samples
# print("Visualizing samples from training dataloader...")
# visualize_samples(train_load, cat_to_name, num_samples=8)


