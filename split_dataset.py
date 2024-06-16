# -*- coding: utf-8 -*-
# =============================================================================
# Title       : Dataset Splitting Script
# Description : This script splits a dataset into training and validation sets.
#               It randomly shuffles the images within each class and distributes
#               them according to the specified validation ratio.
# Author      : Brian Ebiyau (bebyau@gmail.com)
# Date        : 2024-06-15
# Version     : 1.0.0
# License     : MIT License
# =============================================================================

"""
Module Title: Dataset Splitting Script
Module Description:
    This module splits a dataset into training and validation sets.
    It ensures that images are randomly shuffled and then distributed
    between the training and validation sets based on a given ratio.

Functions:
    split_dataset(data_dir, output_dir, val_ratio=0.2)
        Splits the dataset into training and validation sets.

Usage:
    From the command line, specify the data directory and output directory, and run the script:
    python this_script.py

Examples:
    split_dataset('path/to/data', 'path/to/output', val_ratio=0.2)
"""

import os
import shutil
import random

def split_dataset(data_dir, output_dir, val_ratio=0.2):
    """
    Splits the dataset into training and validation sets.

    Args:
        data_dir (str): Path to the dataset directory containing class subdirectories.
        output_dir (str): Path to the output directory where the split dataset will be saved.
        val_ratio (float, optional): Ratio of validation set size to the total dataset size (default is 0.2).

    Returns:
        None
    """
    random.seed(42)  # Set seed for reproducibility
    
    # Get the list of class directories
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        images = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
        
        random.shuffle(images)  # Shuffle images to ensure random split
        split_idx = int(len(images) * (1 - val_ratio))  # Compute the split index
        
        # Split images into training and validation sets
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Create directories for training and validation sets
        train_cls_dir = os.path.join(output_dir, 'train', cls)
        val_cls_dir = os.path.join(output_dir, 'val', cls)
        
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)
        
        # Copy training images to the training directory
        for img in train_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(train_cls_dir, img))
        
        # Copy validation images to the validation directory
        for img in val_images:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(val_cls_dir, img))
        
        # Print the number of images in training and validation sets for each class
        print(f"Class {cls}: {len(train_images)} images for training, {len(val_images)} images for validation")

if __name__ == "__main__":
    # Define the data directory and output directory
    data_dir = 'D:/Summer 2024/Vision GNN/malaria_dataset'  
    output_dir = 'D:/Summer 2024/Vision GNN/malaria_dataset' 
    
    # Split the dataset
    split_dataset(data_dir, output_dir)
