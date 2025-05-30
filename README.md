# DGIQA: Depth-guided Feature Attention and Refinement for IQA

This repository contains the implementation of **DGIQA**, a no-reference image quality assessment model that fuses RGB and depth features via Transformer–CNN bridges and depth-guided cross attention and refinement.  

Paper:  
> **DGIQA: Depth-guided Feature Attention and Refinement for Generalizable Image Quality Assessment**  
> Vaishnav Ramesh, Junliang Liu, Haining Wang, Md Jahidul Islam.   

---

## Table of Contents

- [Overview](#overview)  
- [Citation](#citation)  
- [Installation](#installation)  
- [Pretrained Weights](#pretrained-weights)
- [Inference](#single-image-inference)  
- [Scripts and Utilities](#scripts-and-utilities)  
- [Acknowledgements](#acknowledgements)
- [Authors](#authors)  

---

## Overview

### Abstract
A long-held challenge in no-reference image quality assessment (NR-IQA) learning from human subjective perception is the lack of objective generalization to unseen natural distortions. To address this, we integrate a novel DepthGuided cross-attention and refinement (Depth-CAR) mechanism, which distills scene depth and spatial features into a structure-aware representation for improved NR-IQA. This brings in the knowledge of object saliency and relative contrast of the scene for more discriminative feature learning. Additionally, we introduce the idea of TCB (TransformerCNN Bridge) to fuse high-level global contextual dependencies from a transformer backbone with local spatial features captured by a set of hierarchical CNN (convolutional neural network) layers. We implement TCB and Depth-CAR as multimodal attention-based projection functions to select the most informative features, which also improve training time and inference efficiency. Experimental results demonstrate that our proposed DGIQA model achieves state-ofthe-art (SOTA) performance on both synthetic and authentic benchmark datasets. More importantly, DGIQA outperforms SOTA models on cross-dataset evaluations as well as in assessing natural image distortions such as low-light effects, hazy conditions, and lens flares.

<p align="center">
  <img src="data/model.png" width="75%" alt="Model Architecture">
</p>

## Citation


## Installation

```bash
git clone https://github.com/uf-robopi/DGIQA.git
cd DGIQA
conda create -n dgiqa python=3.10 -y
conda activate dgiqa
conda install pytorch torchvision cudatoolkit=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Pretrained Weights

Pretrained models are available for:

- KADID10K  
- KonIQ-10k  

Weights can be downloaded from [Dropbox]() . 

Use with `--model_weights` in inference scripts.

## Inference

Compute a quality score for one image:

```bash
python single_image_inference.py --img_path /path/to/image.png  --model_weights pretrained_weights/DGIQA_Koniq10k.pth --encoder vitl --crops 20
```

* `--img_path`: path to your RGB image
* `--model_weights`: path to pretrained DGIQA `.pth` file
* `--encoder`: DepthAnything encoder variant (`vits`, `vitb`, `vitl`)
* `--crops`: number of random 224×224 crops (default: 20)

**Output**: averaged MOS score in \[0,1].

---

## Scripts and Utilities

* **single\_image\_inference.py**
  Reads an image, generates its depth map, extracts random crops, and averages model predictions.

* **depth\_utils.py**
  Defines `DepthGenerator`, a wrapper around DepthAnything for depth-map generation.

## Acknowledgements

We thank the authors of [DepthAnything](https://github.com/LiheYoung/Depth-Anything.git) for their excellent open-source code and pretrained models, which we use in this project for depth map generation and as a component of our inference pipeline.

## Authors

* **Vaishnav Ramesh**
* **Junliang Liu**
* **Haining Wang**
* **Md Jahidul Islam**