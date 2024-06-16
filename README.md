# Vision GNN Training

This project finetunes the ViG (Vision GNN) from the paper [Vision GNN: An Image is Worth Graph of Nodes](https://arxiv.org/abs/2206.00272) model using a custom dataset. 

## Project Structure

- `mydataset.py`: Contains the dataloader implementation for loading the custom dataset.
- `gcn_lib`: Contains the implementation of the GCN layers and related utilities.
- `vig.py`: Implementation of Vig model.
- `mytrain.py`: Script to train the model.
- `requirements.txt`: Libraries needed for the projects.
- `ViG Checkpoint`: Folder containing checkpoints for ViG and ViG pyramid models.
- `pyramid_vig.py`: Implementation for Vig Pyramid models
- `malaria_dataset/`: Directory containing your dataset, each class in its folder.
- `split_datatset`: Script to split dataset into train and validation subsets

## Requirements
- `torchvision`
- `timm`
- `tqdm`
- `python 3.10 or higher`

You can install the required packages using:

```bash
pip install torch torchvision timm tqdm

```

## Setup
1. Clone Repository
```
git clone https://github.com/ebyau/vision-gnn-custom-training.git
cd vision-gnn-custom-training
```

2. Prepare your dataset:
Organize your dataset into train and val directories within path_to_your_custom_dataset/.
Each class should have its own subdirectory containing the respective images.
Alternatively, you could run the `split_dataset.py` script to split you dataset into train and validation subsets.

3. Place the pre-trained weights file (pretrained_weights.pth) in the `ViG Checkpoint`. The [official Vision GNN repo](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch)  contains the pretrained weights for various models.


## Training
Before training, ensure you update the `data_dir` and `pretrained_model_path` variables to point to the correct path for dataset ab=nd pretrained model weights.
```
python mytrain.py
```

## Results

After training, the best model is saved to best_model.pth. Plot of accuracy and loss is made for training and validation subset.

