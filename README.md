# Feature Pyramid Networks (FPN) for Object Detection - Reproduction

**Disclaimer:** This project is a reproduction of the FPN paper for the course COMP3314: Introduction to Machine Learning at the University of Hong Kong. Experiments were conducted on the university's GPU farm. Details specific to the university's computing environment are intentionally omitted.

## Paper Details

- **Title**: Feature Pyramid Networks for Object Detection
- **Authors**: Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie
- **Conference**: CVPR 2017
- **Original Paper**: [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)

## Full Reproduction Guide

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n fpn python=3.10 -y
conda activate fpn

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install other dependencies
pip install opencv-python matplotlib pycocotools ninja
```

### 2. Dataset Setup

```bash
# Create directory structure
mkdir -p ~/data/coco
cd ~/data/coco

# Download COCO 2017 dataset (this takes time)
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract files
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

### 3. Training

Run the training script:

```bash
# Run training in background (survives disconnections)
nohup python train_fpn.py > training.log 2>&1 &

# Monitor training progress
tail -f training.log
```

Sample expected training output:

```bash
[11/08 10:14:26 d2.utils.events]:  eta: 5:56:07  iter: 339  total_loss: 0.9507  loss_cls: 0.4049  loss_box_reg: 0.2223  loss_rpn_cls: 0.1361  loss_rpn_loc: 0.1279    time: 0.2416  last_time: 0.2381  data_time: 0.0091  last_data_time: 0.0092   lr: 0.0067932  max_mem: 4212M
```

Useful commands for monitoring:

```bash
# Check checkpoints are being saved
ls -la fpn_output/

# Expected output:
...
-rw-rw-r-- 333264340 Nov  8 16:23 model_0043999.pth
-rw-rw-r-- 333264340 Nov  8 16:31 model_0045999.pth
-rw-rw-r-- 333264340 Nov  8 16:39 model_0047999.pth
...

# Check GPU usage
nvidia-smi

# Check training process
ps aux | grep python | grep train_fpn
```

### 4. Evaluation and Results

Run the evaluation script:

```bash
python eval_fpn.py
```

Results should look something like the following:

```bash
Loading and preparing results...
DONE (t=0.39s)
creating index...
index created!
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ...
 ...
```

## Results

Full results found in txt files. below is relevant values to compare with table 5 of paper:

Differences due to:

- Hardware: reproduction run on NVIDIA RTX 4080

## Reproduced Results (wip keep above)

- **COCO AP**: 0.277 (Paper: 0.339) - 82% reproduced
- **AP@0.5**: 0.458 (Paper: 0.569)
- **Training**: 90,000 iterations on COCO 2017
- **Architecture**: Faster R-CNN + FPN with ResNet-50

## Challenges
