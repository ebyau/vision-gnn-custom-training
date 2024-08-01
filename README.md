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




## Training
Before training, ensure you update the `data_dir` and `pretrained_model_path` variables to point to the correct path for dataset and pretrained model weights.
```
python train.py --train_data /path/to/train-data --val_data /path/to/val_data --model [vig_ti_224_gelu, vig_s_224_gelu, vig_b_224_gelu] --model_ckpt /path/to/checkpoint_file
```

## Results
Results are automatically logged unto wandb

