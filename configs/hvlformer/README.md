# HVLFormer Configs

This folder contains HVLFormer-style configs adapted from the original TQDM
configuration layout.

Current starter config:

- `hvlformer_clip_vit-b_voc_80e_512.py`: CLIP ViT-B/16 backbone, 8-token
  learnable prompting, 3 hierarchical textual queries per class, DN-DETR-style
  6-layer pixel decoder, 9-layer Mask2Former decoder, and AdamW with polynomial
  decay plus 1.5k warm-up iterations.

Before training, replace the inherited dataset base with the exact
semi-supervised split for VOC, COCO, ADE20K, or Cityscapes.
