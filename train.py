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
import matplotlib.pyplot as plt
import wandb
from load_dataset import get_data_loaders
import argparse




def train_and_validate(model, train_loader, val_loader, config):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)

    # Move the model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    # Training loop
    for epoch in range(config.num_epochs):
        if early_stop:
            print("Early stopping")
            break

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} - Training")
        
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            train_loader_tqdm.set_postfix(loss=f"{running_loss/len(train_loader):.4f}", accuracy=f"{100 * correct_train / total_train:.2f}")
        
        train_accuracy = 100 * correct_train / total_train
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")
        
        # Log training metrics to wandb
        wandb.log({"epoch": epoch+1, "train_loss": running_loss/len(train_loader), "train_accuracy": train_accuracy})
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} - Validation")
        
        with torch.no_grad():
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_loader_tqdm.set_postfix(loss=f"{val_loss/len(val_loader):.4f}", accuracy=f"{100 * correct_val / total_val:.2f}")
        
        val_accuracy = 100 * correct_val / total_val
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # Log validation metrics to wandb
        wandb.log({"epoch": epoch+1, "val_loss": val_loss/len(val_loader), "val_accuracy": val_accuracy})
        
        # Step the scheduler
        scheduler.step()
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print("Early stopping")
                early_stop = True


def main():
    parser = argparse.ArgumentParser(description="Train a Vision GNN model.")
    parser.add_argument('--train_data', type=str, required=True, help='Path to the training dataset')
    parser.add_argument('--val_data', type=str, required=True, help='Path to the validation dataset')
    parser.add_argument('--model', type=str, required=True, choices=['vig_ti_224_gelu', 'vig_s_224_gelu', 'vig_b_224_gelu'], help='Model type to use')
    parser.add_argument('--model_ckpt', type=str, required=True, help='Path to the model checkpoint')
    args = parser.parse_args()

    # Define hyperparameters
    hyperparameters = {
        "learning_rate": 0.001,
        "batch_size": 8,
        "num_epochs": 10,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "scheduler_step_size": 5,
        "scheduler_gamma": 0.1,
        "patience": 5  # Early stopping patience
    }

    # Initialize wandb with hyperparameters
    wandb.init(project="finetuning-gnn-malaria", config=hyperparameters)

    # Access hyperparameters from wandb.config
    config = wandb.config

    # Model selection
    if args.model == 'vig_ti_224_gelu':
        model = vig_ti_224_gelu()
    elif args.model == 'vig_s_224_gelu':
        model = vig_s_224_gelu()
    elif args.model == 'vig_b_224_gelu':
        model = vig_b_224_gelu()

    # Load model checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device))

    # Modify the final layer to output 4 classes
    model.prediction = nn.Sequential(
        nn.Conv2d(640, 1024, kernel_size=(1, 1), stride=(1, 1)),
        nn.BatchNorm2d(1024),
        nn.GELU(),
        nn.Dropout(p=0.5),
        nn.Conv2d(1024, 4, kernel_size=(1, 1), stride=(1, 1))
    )

    # Assume train_loader and val_loader are defined elsewhere
    train_loader, val_loader = get_dataloaders(args.train_data, args.val_data, config.batch_size)

    # Train and validate the model
    train_and_validate(model, train_loader, val_loader, config)

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
