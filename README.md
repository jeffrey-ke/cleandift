<h2 align="center">🧹CleanDIFT: Diffusion Features without Noise</h2>
<div align="center"> 
  <a href="https://nickstracke.dev/" target="_blank">Nick Stracke</a><sup>*</sup> · 
  <a href="https://stefan-baumann.eu/" target="_blank">Stefan A. Baumann</a><sup>*</sup> · 
  <a href="https://bsky.app/profile/koljabauer.bsky.social" target="_blank">Kolja Bauer</a><sup>*</sup> · 
  <a href="https://ffundel.de/" target="_blank">Frank Fundel</a> · 
  <a href="https://ommer-lab.com/people/ommer/" target="_blank">Björn Ommer</a>
</div>
<p align="center"> 
  <b>CompVis @ LMU Munich</b><br/>CVPR 2025 (Oral)
</p>

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://compvis.github.io/cleandift/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/pdf/2412.03439)
[![Weights](https://img.shields.io/badge/HuggingFace-Weights-orange)](https://huggingface.co/CompVis/cleandift)

This repository contains the official implementation of the paper "CleanDIFT: Diffusion Features without Noise".

We propose CleanDIFT, a novel method to extract noise-free, timestep-independent features by enabling diffusion models to work directly with clean input images. Our approach is efficient, training on a single GPU in just 30 minutes.

![teaser](./docs/static/images/teaser_fig.jpg)

## 🚀 Usage

### Setup

Clone the repo, then create the uv environment (pulls pinned `vision-core` from GitHub automatically):

```bash
cd cleandift
uv sync
```

### Training

Run training with the project venv:

```bash
uv run cleandift-train
# or: uv run python train.py
```

**COYO-style data** (`./data` with `filename.jpg` + `filename.json` containing `caption`):

```bash
uv run cleandift-train --config-name sd21_feature_extractor
```

**Isaac ObsMask renders** (read via `ObsMask.deserialize` — no jpg+json export):

```bash
# one umbrella (parent of render*/)
uv run cleandift-train --config-name sd21_isaac_obsmask \
  data.dataset_dir=/path/to/expanded-refseg

# several umbrellas concatenated (3750 frames across expanded-refseg + mixed-persp + shelf-optflow)
uv run cleandift-train --config-name sd21_isaac_obsmask_multi
# or override:
uv run cleandift-train --config-name sd21_isaac_obsmask \
  data.dataset_dir=null \
  'data.dataset_dirs=[/data/user/jeffk/datasets/expanded-refseg,/data/user/jeffk/datasets/mixed-persp]'
```

### Feature Extraction

For feature extraction, please refer to one of the notebooks at [`notebooks`](https://github.com/CompVis/cleandift/tree/main/notebooks). We demonstrate how to extract features and use them for semantic correspondence detection and depth prediction.

Our checkpoints are fully compatible with the `diffusers` library. If you already have a pipeline using SD 1.5 or SD 2.1 from `diffusers`, you can simply replace the U-Net state dict:

```python
from diffusers import UNet2DConditionModel
from huggingface_hub import hf_hub_download

unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="unet")
ckpt_pth = hf_hub_download(repo_id="CompVis/cleandift", filename="cleandift_sd21_unet.safetensors")
state_dict = load_file(ckpt_pth)
unet.load_state_dict(state_dict, strict=True)
```

#### Combination with Telling Left From Right (TLFR)
Our checkpoints can also be combined with more advanced feature extraction methods than [DIFT](https://diffusionfeatures.github.io/), such as [TLFR](https://telling-left-from-right.github.io/). We provide an adapted version of their codebase that can be used to reproduce our state-of-the-art zero-shot semantic correspondence results in the [`geoaware-sc-eval`](https://github.com/CompVis/cleandift/tree/geoaware-sc-eval) branch.

## 🎓 Citation

If you use this codebase or otherwise found our work valuable, please cite our paper:

```bibtex
@inproceedings{stracke2025cleandift,
    title={CleanDIFT: Diffusion Features without Noise}, 
    author={Nick Stracke and Stefan Andreas Baumann and Kolja Bauer and Frank Fundel and Björn Ommer},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}
}
```
