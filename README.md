# HVLFormer

### **Segmenting Visuals With Querying Words: Language Anchors For Semi-Supervised Image Segmentation**
> Numair Nadeem et al.\
> Project status: code scaffold adapted from TQDM; paper, checkpoints, and final splits coming soon.

#### [[`Project Page`](https://numnz.github.io/HVLFormer/)] [[`Paper`](#)] [[`Checkpoints`](#)]

## Overview

This repository implements **HVLFormer**, a vision-language segmentation
framework for semi-supervised image segmentation. HVLFormer builds on the
Textual Query-Driven Mask Transformer (TQDM) codebase and adds:

- **HTQG**: Hierarchical Textual Query Generation with dataset-aware
  learnable prompts and semantic relevance estimation.
- **PTRM**: Pixel-Text Refinement Module for bidirectional spatial
  conditioning between image features and textual queries.
- **CMCR**: Cross-View and Modal Consistency Regularisation for unlabeled
  images under weak and strong augmentations.

## Environment

### Requirements

```bash
conda create -n hvlformer python=3.9 numpy=1.26.4
conda activate hvlformer
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install xformers==0.0.20
pip install mmcv-full==1.5.3
```

### Pre-trained VLM Models

Download the pre-trained vision-language models and save them in `./pretrained`.

| Model | Type | Link |
|-----|-----|:-----:|
| CLIP | `ViT-B-16.pt` | [official repo](https://github.com/openai/CLIP) |
| SigLIP2 | ViT family | [model family](https://huggingface.co/google) |
| EVA02-CLIP | `EVA02_CLIP_L_336_psz14_s6B` | [official repo](https://github.com/baaivision/EVA/tree/master/EVA-CLIP#eva-02-clip-series) |

## Checkpoints

Checkpoints will be released after paper review.

| Model | Pretrained | Training split | Config | Link |
|-----|-----|-----|-----|:-----:|
| `hvlformer-clip-vit-b-voc` | CLIP ViT-B/16 | VOC semi-supervised | [`config`](configs/hvlformer/hvlformer_clip_vit-b_voc_80e_512.py) | Coming soon |
| `hvlformer-eva02-clip-vit-l` | EVA02-CLIP ViT-L/14 | VOC/COCO/ADE20K/Cityscapes | Coming soon | Coming soon |
| `hvlformer-siglip2` | SigLIP2 | VOC/COCO/ADE20K/Cityscapes | Coming soon | Coming soon |

## Datasets

We benchmark on Pascal VOC, COCO, ADE20K, and Cityscapes. Following SemiVL, each
training iteration uses a mixed batch of eight labeled and eight unlabeled
images. Inputs are cropped to `512 x 512`, except Cityscapes, where `801 x 801`
is used.

Before training, edit the dataset roots and semi-supervised splits in the config
files under `configs/_base_/datasets` or add your dataset split file:

```python
data_root = '[YOUR_DATA_FOLDER_ROOT]'
labeled_split = '[LABELED_SPLIT_FILE]'
unlabeled_split = '[UNLABELED_SPLIT_FILE]'
```

## Train

```bash
bash dist_train.sh configs/hvlformer/hvlformer_clip_vit-b_voc_80e_512.py 2
```

The paper setting uses AdamW, initial learning rate `1e-4`, polynomial decay
with power `0.9`, and linear warm-up for `1.5k` iterations. Training schedules
are 80, 10, 40, and 240 epochs for VOC, COCO, ADE20K, and Cityscapes.

## Test

```bash
bash dist_test.sh configs/hvlformer/hvlformer_clip_vit-b_voc_80e_512.py \
  work_dirs/hvlformer_clip_vit-b_voc_80e_512/epoch_last.pth \
  2 --eval mIoU
```

## The Most Relevant Files

- [`configs/hvlformer/*`](configs/hvlformer) - HVLFormer training configs.
- [`mmseg/models/decode_heads/hvlformer_head.py`](mmseg/models/decode_heads/hvlformer_head.py) - HTQG, PTRM, and HVLFormer decode head.
- [`mmseg/models/losses/cmcr_loss.py`](mmseg/models/losses/cmcr_loss.py) - CMCR consistency loss for original, weak, and strong views.
- [`mmseg/models/decode_heads/tqdm_head.py`](mmseg/models/decode_heads/tqdm_head.py) - Original TQDM textual query Mask2Former head.
- [`mmseg/models/plugins/tqdm_msdeformattn_pixel_decoder.py`](mmseg/models/plugins/tqdm_msdeformattn_pixel_decoder.py) - DN-DETR-style text-aware pixel decoder inherited from TQDM.
- [`models/segmentors/hvlformer_clip.py`](models/segmentors/hvlformer_clip.py) - CLIP-backed HVLFormer segmentor wrapper.

## Citation

If you find this code helpful, please cite:

```bibtex
@misc{nadeem2026segmenting,
  title={Segmenting Visuals With Querying Words: Language Anchors For Semi-Supervised Image Segmentation},
  author={Nadeem, Numair and others},
  year={2026}
}
```

This project is based on the excellent
[TQDM](https://github.com/ByeongHyunPak/tqdm) repository.
