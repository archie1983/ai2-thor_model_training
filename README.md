# Diffusion Approach for World Modelling in Navigation Context

This project provides a diffusion model for multiple continuous label like spatial information (x, y, z, orientation).
This model integrates CCDM's continuous label processing methodology into the DiT framework.
DiT repository at https://github.com/facebookresearch/DiT.
CCDM repository at https://github.com/UBCDingXin/CCDM.

## Key Modifications
- **Replaced the original label embedding layer** in DiT to handle continuous numerical labels (e.g., coordinates and orientations).
- **Adapted the attention mechanism** to process spatial metadata efficiently.

## Dataset
The dataset follows WebDataset format, where each shard contains xxxx.png image files paired with corresponding xxxx.json label files.
In dataset folder, there is a small dataset which have 88 images (11 * 8).

## Installation
We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate DiT
```

## Sampling
```bash
python sample.py --model DiT-S/8 --image-size 256 --ckpt /path/to/model.pt
```

## Training
```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --model DiT-S/8 --data-path /path/to/dataset/train
```

## Others
[`command.txt`](command.txt) contains some command and information which might be useful.