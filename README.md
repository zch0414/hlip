# HLIP
> Official PyTorch implementation of the following paper:\
> Towards Scalable Language-Image Pre-training for 3D Medical Imaging\
> University of Michigan\
> [![arXiv](https://img.shields.io/badge/arXiv%20paper-2505.21862-b31b1b.svg)](https://arxiv.org/abs/2505.21862)&nbsp;


## Overview
<p align="center"><img src="https://github.com/Zch0414/hlip/blob/master/docs/github.png" width=96% height=96% class="center"></p>

We propose **H**ierarchical attention for **L**anguage-**I**mage **P**re-training (**HLIP**), inspired by the natural hierarchy of radiology data: slice, scan, and study. With this lightweight attention mechanism, HLIP can be trained directly on uncurated clinical datasets, enabling scalable language-image pre-training in 3D medical imaging. For real-world clinical use, HLIP can be applied to studies containing either a single scan (e.g., chest CT) or multiple scans (e.g., brain MRI).

## Updates
- **(2025-06)** Complete the initiation of HLIP repository.
- **(2025-05)** Release HLIP models trained on chest CT and brain MRI, feel free to try our demos.

## Getting Started

### Install 
[open-clip](https://github.com/mlfoundations/open_clip/tree/main)
```bash
python3 -m venv env
source env/bin/activate
pip install -U pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
git clone git@github.com:mlfoundations/open_clip.git
cd open_clip
make install
make install-training
```

### Model Card
| Data | Attention | Patch Size | Model |
| -------- | -------- | -------- | -------- |
| CT-RATE-20K | <code>slice</code> + <code>scan</code> | <code>8, 24, 24</code> | [ViT-Base](https://drive.google.com/file/d/1muu7L9H3KaL3nq3fNtN8kKF1eDK3R5Z4/view?usp=drive_link) |
| BrainMRI220K | <code>scan</code> + <code>study</code> | <code>16, 16, 16</code> | [ViT-Base](https://drive.google.com/file/d/1uUdcE0TYx3K2YU7FQMfwb2FsFQjQcGil/view?usp=drive_link) |
| BrainMRI220K | <code>scan</code> + <code>study</code> | <code>8, 16, 16</code> | [ViT-Base](https://drive.google.com/file/d/12BwJvd6IEZynXM8jkled0ND7t11iuySj/view?usp=drive_link) |
| BrainMRI220K | <code>slice</code> + <code>scan</code> + <code>study</code> | <code>8, 16, 16</code> | [ViT-Base](https://drive.google.com/file/d/1FgOS3W6LhnhH4gJlbASPopUEXChcjeqy/view?usp=drive_link) |
| HeadCT240K | <code>scan</code> + <code>study</code> | <code>8, 16, 16</code> | [ViT-Base](https://drive.google.com/file/d/1rfoz-kzF0iwaMQ-4MuR7F4NlTjtPIZa7/view?usp=drive_link) |

### Demo
Chest CT: an example from the external Rad-ChestCT dataset.
```bash
python inference_rad_chestct.py \
  --model vit_base_singlescan_h2_token1176 \
  --resume /path/to/vit_base_chestct_h2_token1176.pt \
  --data ../../docs/tst32751/tst32751.pt
```

Brain MRI: an example from the external BraTS23 dataset.
```bash
python inference_pub_brain_5.py \
  --model vit_base_multiscan_h2_token1176 \
  --resume /path/to/vit_base_brainmri_h2_token1176.pt \
  --patch-size 8 16 16 \
  --num-slices 72 \
  --data ../../docs/BraTS-GLI-00459-000/
```
Visualizing the activation with <code>--interpret</code>.

### Evaluation
CT-RATE
```bash
python zeroshot_ct_rate.py \
  --model vit_base_singlescan_h2_token2744 \
  --resume /path/to/vit_base_chestct_h2_token2744.pt \
  --ct-rate-root /data/ct_rate/valid/ \
  --zeroshot-template volume
```

Rad-ChestCT
```bash
python zeroshot_rad_chestct.py \
  --model vit_base_singlescan_h2_token2744 \
  --resume /path/to/vit_base_chestct_h2_token2744.pt \
  --rad-chestct-root /data/rad_chestct/ \
  --zeroshot-template volume
```

Brain MRI
```bash
python pub_brain_5_embed.py \
  --model vit_base_multiscan_h2_token1176 \
  --resume /path/to/vit_base_brainmri_h2_token1176.pt \
  --num-slices 144
```
```bash
python zeroshot_pub_brain_5.py \
  --model vit_base_multiscan_h2_token1176 \
  --resume /path/to/vit_base_brainmri_h2_token1176.pt \
  --num-slices 144 \
  --zeroshot_prompt prompt \
  --zeroshot_template template
```
As there are ~18K studies in the Pub-Brain-5 dataset, evaluation may take ~30 minutes. We first extract the embedding for each study, followed by zero-shot classification. This procedure supports researchers interested in prompt engineering. 

<code>--num-slices</code> is set to 144 during evaluation, though we use a fixed input size of <code>48, 224, 224</code>. We found that HLIP can directly transfer and benefit from higher-resolution inputs at test time.

### Training

Our training implementation is closely aligned with [open-clip](https://github.com/mlfoundations/open_clip/tree/main), allowing us to leverage features such as <code>patch dropout</code> and <code>siglip</code>. Below, we provide a training code demo for chest CT. Training on CT-RATE for 20 epochs takes ~6 hours using a node with 4 A40 GPUs.

```bash
torchrun --rdzv_endpoint=localhost:29500 --nproc_per_node 4 main.py \
  --json-root ../../data/ct_rate/files/ --data-root /path/to/data/ct_rate/ \
  --train-data raw_annotation --input-info -1150 350 crop \
  --zeroshot-ct-rate ../../data/ct_rate/metafiles/valid_labels.csv --zeroshot-template volume \
  --zeroshot-frequency 1 \
  --save-frequency 1 \
  --report-to wandb \
  --wandb-project-name chest_ct \
  --warmup 377 \
  --batch-size 16 \
  --accum-batch 1 \
  --lr=1e-5 \
  --wd=0.2 \
  --epochs=20 \
  --precision amp \
  --workers 4 \
  --grad-checkpointing \
  --model vit_base_singlescan_h2_token2744 \
  --use-cxr-bert \
  --lock-text
```

Use the following commands for <code>patch dropout</code>:
```bash
  --force-patch-dropout 0.5 \
  --beta2 0.95
```

Use the following commands for <code>siglip</code>:
```bash
  --siglip
```

## Citation
If you find this repository helpful, please consider citing:
```bib
@article{zhao2025towards,
  title={Towards Scalable Language-Image Pre-training for 3D Medical Imaging},
  author={Zhao, Chenhui and Lyu, Yiwei and Chowdury, Asadur and Harake, Edward and Kondepudi, Akhil and Rao, Akshay and Hou, Xinhai and Lee, Honglak and Hollon, Todd},
  journal={arXiv preprint arXiv:2505.21862},
  year={2025}
}
```
