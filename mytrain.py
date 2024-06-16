# -*- coding: utf-8 -*-
# =============================================================================
# Title       : Vision Transformer Training Script
# Description : This script trains a Vision Transformer (ViG) model on a specified dataset.
#               It includes functions for loading pretrained weights, training the model,
#               and evaluating its performance.
# Author      : Brian Ebiyau (bebyau@gmail.com)
# Date        : 2024-06-15
# Version     : 1.0.0
# License     : MIT License
# =============================================================================

"""
Module Title: Vision Transformer Training Script
Module Description:
    This module trains a Vision Transformer (ViG) model using a given dataset.
    It provides functionalities to:
    - Load pretrained weights into the model.
    - Train the model and track training and validation metrics.
    - Save the best performing model based on validation loss.

Functions:
    load_pretrained_weights(model, pretrained_model_path, device)
        Loads pretrained weights into the model from a specified path.

    train_model(model, train_loader, val_loader, num_epochs, lr, save_path)
        Trains the model using the training and validation data loaders.

Usage:
    From the command line, set the data directory and pretrained model path, and run the script:
    python this_script.py

Examples:
    train_loader, val_loader = get_dataloader(data_dir)
    model = vig_b_224_gelu()
    model = load_pretrained_weights(model, pretrained_model_path)
    train_model(model, train_loader, val_loader, num_epochs=10, save_path='best_model.pth')
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from vig import vig_ti_224_gelu, vig_s_224_gelu, vig_b_224_gelu
from mydataset import get_dataloader
from tqdm.auto import tqdm
from torchmetrics import Accuracy
import matplotlib.pyplot as plt



device="cuda" if torch.cuda.is_available() else "cpu"


# Calculate accuracy (a classification metric)
# def accuracy_fn(y_true, y_pred):
#     """Calculates accuracy between truth labels and predictions.

#     Args:
#         y_true (torch.Tensor): Truth labels for predictions.
#         y_pred (torch.Tensor): Predictions to be compared to predictions.

#     Returns:
#         [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
#     """
#     correct = torch.eq(y_true, y_pred).sum().item()
#     acc = (correct / len(y_pred)) * 100
#     return acc

# Initialize the accuracy function for multiclass classification
accuracy_fn = Accuracy(task='multiclass', num_classes=4).to(device)

def load_pretrained_weights(model, pretrained_model_path, device=device):
    """
    Loads pretrained weights into the model from a specified path.

    Args:
        model (torch.nn.Module): The model into which the weights will be loaded.
        pretrained_model_path (str): Path to the pretrained model weights.
        device (torch.device, optional): Device to load the model onto (default is CUDA if available, otherwise CPU).

    Returns:
        model (torch.nn.Module): The model with loaded weights.
    """
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    print(f"Loaded Pretrained Model: {pretrained_model_path}")
    return model

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, save_path='model.pth'):
    """
    Trains the model using the training and validation data loaders.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        num_epochs (int, optional): Number of epochs to train the model (default is 10).
        lr (float, optional): Learning rate for the optimizer (default is 0.001).
        save_path (str, optional): Path to save the best performing model (default is 'model.pth').

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize lists to store loss and accuracy
    training_loss = []
    validation_loss = []
    training_acc = []
    validation_acc = []
    
    # Define the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        
        # Training loop
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs} - Training', unit='batch') as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                #print(labels)
                #print(outputs)
                #print(torch.argmax)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                running_acc += accuracy_fn(outputs.argmax(dim=1), labels)
                pbar.update(1)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)
        training_loss.append(epoch_loss)
        training_acc.append(epoch_acc)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.5f}')
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with tqdm(total=len(val_loader), desc=f'Epoch {epoch+1}/{num_epochs} - Validation', unit='batch') as pbar:
            with torch.inference_mode():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_acc += accuracy_fn(outputs.argmax(dim=1), labels)
                    pbar.update(1)
            
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_acc / len(val_loader.dataset)
        validation_loss.append(val_loss)
        validation_acc.append(val_acc)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
        
        # # Save the best model based on validation loss
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), save_path)
        #     print(f'Saving best model with loss {best_val_loss:.4f}')

    print('Training complete')
    
    # Plotting the training and validation loss
    plt.plot(range(1, num_epochs + 1), training_loss, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Plotting the training and validation accuracy
    plt.plot(range(1, num_epochs + 1), training_acc, label='Training acc')
    plt.plot(range(1, num_epochs + 1), validation_acc, label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    device="cuda" if torch.cuda.is_available() else "cpu"
    
    # Set the data directory and pretrained model path
    data_dir = 'D:/Summer 2024/Vision GNN/malaria_dataset'
    pretrained_model_path = 'ViG Checkpoint/vig_b_82.6.pth'
    
    # Get the data loaders for training and validation datasets
    train_loader, val_loader = get_dataloader(data_dir)
    
    # Load the pretrained ViG model
    model = vig_b_224_gelu()
    model = load_pretrained_weights(model, pretrained_model_path,device=device)
    
    # Modify the final classification layer to match the number of classes
    model.prediction = nn.Sequential(
        nn.Conv2d(model.prediction[0].in_channels, 1024, kernel_size=(1, 1), stride=(1, 1)),
        nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.GELU(approximate='none'),
        nn.Dropout(0.5),
        nn.Conv2d(1024, 4, kernel_size=1, stride=1)
    )
    
    #print(model)
    
    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=10, save_path='best_model.pth')
