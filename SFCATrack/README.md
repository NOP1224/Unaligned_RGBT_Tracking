# SFCATrack: Spatial-Feature Collaborative Alignment Tracker

SFCATrack is an unaligned UAV RGBT tracking framework for visible-thermal tracking with spatially misaligned RGB and TIR inputs. The method follows an OSTrack-style dual-stream tracking backbone and introduces collaborative image-level and feature-level alignment to improve multimodal correspondence under UAV-induced displacement, resolution difference, and nonlinear deformation.

## Method Overview

<p align="center">
  <img src="SFCATrack/asserts/SFCATrack.jpg" width="100%">
</p>


The tracking pipeline contains three main parts:

1. **Image-level spatial alignment: MSEE**
   - The Mixture of Shift Estimation Experts (MSEE) predicts the cross-modal offset between RGB and TIR search regions.
   - It uses a shared expert for common misalignment patterns and multiple scale experts for different offset ranges.
   - A router selects the most suitable expert, and the predicted offset is used to relocate the TIR search region.

2. **Feature-level alignment and fusion: CMAF**
   - After coarse image alignment, Cross-Modal Alignment and Fusion (CMAF) further handles feature misalignment caused by nonlinear deformation.
   - It applies deformable convolution blocks to align RGB/TIR tokens and uses a lightweight gated fusion module to integrate complementary features.

3. **Tracking head**
   - The aligned and fused features are fed into a center-based tracking head to predict the final RGB target bounding box.
   - Training uses offset regression loss for MSEE and OSTrack-style classification/GIoU/L1 losses for tracking.

## Code Structure

```text
SFCATrack/
├── experiments/sfcatrack/rgbt.yaml          # Main training/testing config
├── lib/models/sfcatrack/ostrack_adapter.py  # SFCATrack model entry
├── lib/models/sfcatrack/vit_ce_adapter.py   # Dual-stream ViT backbone with CMAF insertion
├── lib/models/layers/moe_block.py           # MSEE / MoE offset prediction module
├── lib/models/layers/dcn_layer.py           # CMAF, deformable alignment, gated fusion
├── lib/train/actors/sfcatrack.py            # Training actor and losses
├── lib/train/dataset/luart.py               # LUART dataset loader
├── tracking/train.py                        # Training launcher
└── RGBT_workspace/test_rgbt_mgpus_notalign.py # Unaligned RGBT testing script
```

## Environment

```bash
bash install_env.sh
```

Main dependencies include PyTorch, torchvision, timm, OpenCV, PyYAML, easydict, pandas, tqdm, scipy, lmdb, tensorboard, and pycocotools.

## Dataset Preparation

For LUART-style testing, each sequence is expected to contain unaligned visible/infrared frames and dual-modal annotations:

```text
sequence_name/
├── NotAlign/
│   ├── visible/*.jpg
│   └── infrared/*.jpg
├── visible.txt
└── infrared.txt
```

Before training or testing, update local dataset and output paths in:

```text
lib/train/admin/local.py
lib/test/evaluation/local.py
```

or generate default local files by:

```bash
python tracking/create_default_local_file.py \
  --workspace_dir . \
  --data_dir /path/to/datasets \
  --save_dir ./output
```

## Training
**A more detailed training process will be updated after we reorganize the code.**
Edit `experiments/sfcatrack/rgbt.yaml` to set the required training stage:

```yaml
TRAIN:
  PROMPT:
    TYPE: att_base
```

Common stages:

| Stage | Purpose |
|---|---|
| `att_moe_phase1` | Train shared expert and offset prediction head |
| `att_moe_phase2` | Train scale-specific expert; set `TRAIN.EXPERT_INDEX` |
| `att_moe_phase3` | Train router for adaptive expert selection |
| `att_base` | Train tracking backbone, CMAF, and tracking head |

Run training:

```bash
CUDA_VISIBLE_DEVICES=0 python tracking/train.py \
  --script sfcatrack \
  --config rgbt \
  --save_dir ./output \
  --mode single \
  --nproc_per_node 1
```

Checkpoints are saved under:

```text
output/checkpoints/train/sfcatrack/rgbt/
```

## Testing

Run SFCATrack on unaligned LUART data:

```bash
python ./RGBT_workspace/test_rgbt_mgpus_notalign.py \
  --script_name sfcatrack \
  --dataset_name LUART \
  --yaml_name rgbt \
  --epoch 60 \
  --threads 1 \
  --num_gpus 1 \
  --debug 0
```

Tracking results are saved to:

```text
RGBT_workspace/results/NotAlign/LUART/rgbt/
```

You can also directly download our pre-trained weights in LUART for inference.

[Weights Baidu Yun](https://pan.baidu.com/s/1nYFr9ywEaW34ZSMcrQHjbw?pwd=6d4j)
**Code**: 6d4j

## Notes

- `MSEE` corresponds to the image alignment branch and is implemented mainly in `moe_block.py`.
- `CMAF` corresponds to the feature alignment and fusion branch and is implemented mainly in `dcn_layer.py` and inserted in `vit_ce_adapter.py`.
- The default config uses ViT-B/16 with search size `256`, template size `128`, and 60 training epochs.
