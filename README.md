# DGIQA: Depth-guided Feature Attention and Refinement for Generalizable Image Quality Assessment

This repository contains the implementation of **DGIQA**, a no-reference image quality assessment model that fuses RGB and depth features via Transformer–CNN bridges(TCB) and depth-guided cross attention and refinement(DCAR). Our approach leverages depth information to help the model prioritize salient and closer objects in the scene, significantly enhancing its generalization. DGIQA achieves SOTA perfromance in benchmark datasets and excels in assessing unseen distortions.

**Paper**:  
> [**DGIQA: Depth-guided Feature Attention and Refinement for Generalizable Image Quality Assessment**](https://arxiv.org/abs/2505.24002)  
> Vaishnav Ramesh, Junliang Liu, Haining Wang, Md Jahidul Islam.   


<p align="center">
  <img src="data/model.png" width="75%" alt="Model Architecture">
</p>

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

Weights can be downloaded from [Dropbox](https://www.dropbox.com/scl/fo/j0b2989bw643rs62zhybc/ABOjncdXtyySxsYm4DFoDIs?rlkey=stkaxq09czt51bsdayn96dgwt&st=1qwdjss0&dl=0) . 

Use with `--model_weights` in inference scripts.

## Inference

Compute a quality score for one image:

```bash
python single_image_inference.py --img /path/to/image.png  --weights /path/to/weights.pth --encoder vitl --crops 20
```

* `--img`: path to your RGB image
* `--weights`: path to pretrained DGIQA `.pth` file
* `--encoder`: DepthAnything encoder variant (`vits`, `vitb`, `vitl`)
* `--crops`: number of random 224×224 crops (default: 20)

**Output**: averaged MOS score in \[0,1].

## Citation

```bibtex 
@misc{ramesh2025dgiqadepthguidedfeatureattention,
      title={DGIQA: Depth-guided Feature Attention and Refinement for Generalizable Image Quality Assessment}, 
      author={Vaishnav Ramesh and Junliang Liu and Haining Wang and Md Jahidul Islam},
      year={2025},
      eprint={2505.24002},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.24002}, 
}
```

## Acknowledgements

We thank the authors of [DepthAnything](https://github.com/LiheYoung/Depth-Anything.git) for their excellent open-source code and pretrained models, which we use in this project for depth map generation and as a component of our inference pipeline.
