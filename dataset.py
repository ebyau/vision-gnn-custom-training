# =============================================================================
# Title       : DataLoader Script
# Description : This script provides a function to create data loaders for training 
#               and validation datasets using the PyTorch framework.
# Author      : Brian Ebiyau (bebyau@gmail.com)
# Date        : 2024-06-15
# Version     : 1.0.0
# License     : MIT License
# =============================================================================


import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_data_loaders(train_dir, val_dir, batch_size=8):
    # Define the image preprocessing and augmentation pipeline for training
    train_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Define the image preprocessing pipeline for validation
    val_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the custom dataset with the respective transformations
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_preprocess)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_preprocess)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Example usage
#train_loader, val_loader = get_data_loaders('train', 'val', batch_size=8)
