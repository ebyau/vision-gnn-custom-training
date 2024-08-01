# Vision GNN Training

This project finetunes the ViG (Vision GNN) model from the paper [Vision GNN: An Image is Worth Graph of Nodes](https://arxiv.org/abs/2206.00272)  on a custom dataset.   
The official repo can be found <a href="https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch" target="_blank">HERE</a>.


## Setup
1. Setup Environment
```
# download and Install MiniConda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
```
conda deactivate # exit from base env
conda create -n gnn python=3.10
conda activate gnn
```
Clone Repo
```
git clone https://github.com/ebyau/vision-gnn-custom-training.git
```
Install Dependencies
```
bash install_dependencies.sh
```

2. Prepare your dataset:
Organize your dataset into train and val directories
Each class should have its subdirectory containing the respective images.
Alternatively, you could run the `split_dataset.py` script to split your dataset into train and validation subsets.


## Training
Before training, ensure you update the `data_dir` and `pretrained_model_path` variables to point to the correct path for dataset and pretrained model weights.
```
python mytrain.py
```

## Results

After training, the best model is saved to best_model.pth. Plot of accuracy and loss is made for the training and validation subset.

