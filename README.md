# Moiré Photo Restoration using Multi-Scale CNN (PyTorch)

This repository provides an **PyTorch implementation** of the method described in the paper: *Moiré Photo Restoration Using Multiresolution Convolutional Neural Networks (IEEE TIP 2018)*.

This repo is based on the original repository [ZhengJun-AI/MoirePhotoRestoration-MCNN](https://github.com/ZhengJun-AI/MoirePhotoRestoration-MCNN).

## Requirements

- Python 3.8+
- PyTorch >= 1.7.0
- torchvision
- numpy
- PIL (Pillow)
- tqdm

Install dependencies:
```bash
pip install torch torchvision numpy pillow tqdm

```

## Data Preparation 

Dataset: https://huggingface.co/datasets/zxbsmk/TIP-2018

```plain text
dataset/
├── trainData/
│   ├── source/  # Moiré images
│   └── target/  # Clean ground truth images
└── testData/
    ├── source/
    └── target/
```

## Usage

1. train

```python
python train.py --dataset ../dataset/TIP-2018-clean/trainData --batchsize 64 --save ./model
```

2. test

```python
python test.py --dataset ../dataset/TIP-2018-clean/testData --batchsize 1 --model ./model/moire_best_weights.pth
```

## Performance

| Metric | Dataset | Value | Note |
| :--- | :--- | :--- | :--- |
| **PSNR (Y-Channel)** | TIP-2018 | **24.22 dB** | Calculated on Luminance (Y) channel |
| *Reference (Paper)* | *TIP-2018* | *26.77 dB* | *Performance in paper* |