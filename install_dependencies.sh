#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Anaconda or Miniconda and try again."
    exit
fi

#echo "Installing the dependencies using pip..."
pip install -r requirements.txt

# download the checkpoints
wget https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/vig/vig_b_82.6.pth

mv vig_b_82.6.pth checkpoints




