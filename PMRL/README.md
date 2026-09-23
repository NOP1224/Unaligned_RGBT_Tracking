<h1 align="center">
PMRL: Progressive Multi-Cue Relation Learning

<p align="center">
  <img src="assets/firstv4.jpg" width="100%">
</p>
</h1>

<p align="center">
  <b>A journal extension of PMATrack for unaligned RGBT tracking</b>
</p>

<div align="center">

[![Base](https://img.shields.io/badge/Extension%20of-PMATrack-blue)](../PMATrack)
[![Status](https://img.shields.io/badge/Manuscript-Under%20Review-orange)](#publication-status)
[![Task](https://img.shields.io/badge/Task-Unaligned%20RGBT%20Tracking-red)](../README.md)

</div>

## Framework

<div align="center">

<p align="center">
  <img src="assets/frameworkv3.jpg" width="100%">
</p>

</div>

## Core Innovations

### 1. Unified relation modeling for alignment and fusion

PMRL represents cross-modal correspondences with a shared relation matrix.
The same relation is used to decode spatial alignment parameters and aggregate
complementary RGB–TIR features. This formulation reduces the direct dependence
of fusion on a potentially inaccurate geometric transformation and allows
tracking and alignment supervision to optimize the same correspondence
evidence.

### 2. Progressive relation learning

Instead of estimating all alignment parameters in a single step, PMRL learns
the relation progressively through **center localization**, **scale
transformation**, and **global refinement**. Stage-specific feasible
boundaries narrow the matching range as the alignment state improves, reducing
the difficulty of correspondence estimation under large and dynamically
changing misalignment.

### 3. Adaptive multi-cue relation estimation

PMRL combines **target-semantic**, **local-structure**, and **fine-grained
appearance** cues according to the current alignment state. A soft cue mixer
guides adaptive-capacity unbalanced optimal transport, while token reliability
controls how strongly uncertain observations participate in relation
estimation.

### 4. Relation-guided fusion and online alignment update

The learned relation directly transports cross-modal features and uses
retained mass to suppress unreliable content. During tracking, relation
confidence, entropy, support, valid-region ratio, tracking confidence, and
appearance gain jointly verify whether the estimated alignment should update
the online geometric state.

## Method Overview

The final implementation follows one paper-consistent path:

1. extract RGB and TIR template/search features with the shared tracking backbone;
2. construct target, structure, and detail evidence at the progressive stages;
3. solve the cross-modal relation within the current feasible boundary;
4. decode center, scale, and global residual alignment from the shared relation;
5. use the same relation for bidirectional feature transport and gated fusion;
6. predict the target state and verify the online alignment update.

The progressive relation modules are inserted after Transformer blocks 9, 10,
and 11 for center, scale, and global-refinement reasoning, respectively.

## Repository Structure

```text
PMRL/
├── experiments/ostrack/       # LasHeR-Unaligned and LUART configurations
├── lib/
│   ├── models/layers/         # Cue mixer, relation solver, and transport
│   ├── models/ostrack/        # PMRL tracking backbone and model builder
│   ├── train/                 # Dataset, actor, trainer, and objectives
│   └── test/                  # Parameters and PMRL online tracker
├── RGBT_workspace/
│   └── test_rgbt_mgpus_stepalign.py
├── train.sh
├── test.sh
└── requirements.txt
```

## Environment

A recent PyTorch environment is recommended.

```bash
conda create -n pmrl python=3.10 -y
conda activate pmrl
pip install -r requirements.txt
```

Set the dataset paths before training:

```bash
export LASHER_UNALIGNED_DIR=/path/to/LasHeR-Unaligned
export LUART_DIR=/path/to/LUART
```

Optional environment variables are defined in
`lib/train/admin/local.py`:

| Variable | Purpose |
| --- | --- |
| `PMRL_WORKSPACE` | Project workspace |
| `PMRL_TENSORBOARD_DIR` | TensorBoard output directory |
| `PMRL_PRETRAINED_DIR` | Pretrained model directory |
| `LASHER_UNALIGNED_DIR` | LasHeR-Unaligned root |
| `LUART_DIR` | LUART root |

## Pretrained Backbone

Place the DropTrack initialization checkpoint at:

```text
pretrained_models/DropTrack_k700_800E_alldata.pth.tar
```

## Training

Two final configurations are provided:

- `experiments/ostrack/pmrl_lasher.yaml`
- `experiments/ostrack/pmrl_luart.yaml`

Edit `CONFIG`, `LOG`, and `GPUS` at the top of `train.sh`, then run:

```bash
bash train.sh
```

The original PMATrack launch style is retained: training runs through
`nohup`, records the process ID and log file, and follows the log with
`tail -f`.

## Evaluation

Before evaluation:

1. edit the dataset root and split-file paths in
   `RGBT_workspace/test_rgbt_mgpus_stepalign.py`;
2. set `CONFIG`, `DATASET`, `EPOCH`, `MODAL`, and GPU options in
   `test.sh`;
3. run:

```bash
bash test.sh
```

The checkpoint is expected at:

```text
<MODAL>/checkpoints/train/ostrack/<CONFIG>/OSTrack_twobranch_ep<EPOCH>.pth.tar
```

The evaluator retains the original multi-process PMATrack testing workflow,
dataset branches, progress reporting, and result layout.

## Performance

### Tracking Accuracy

| Tracker | LasHeR-Unaligned PR/NPR/SR | LUART PR/NPR/SR | MUART244 PR/NPR/SR | FPS |
| --- | --- | --- | --- | ---: |
| PMATrack (CVPR 2026) | 64.4 / 58.7 / 50.6 | 60.5 / 55.8 / 47.0 | 62.7 / 55.9 / 45.8 | 28.0 |
| **PMRL (Under Review)** | **71.5 / 66.1 / 55.4** | **65.3 / 59.5 / 50.1** | **70.6 / 61.0 / 49.6** | **20.4** |

PR, NPR, and SR are reported as percentages. The MUART244 model is trained on
LasHeR-Unaligned and evaluated without fine-tuning, following the PMATrack
protocol.

### Model Complexity

| Tracker | Parameters (M) ↓ | FLOPs (G) ↓ | FPS ↑ |
| --- | ---: | ---: | ---: |
| PMATrack (CVPR 2026) | 237.5 | 72.6 | 28.0 |
| **PMRL (Under Review)** | **147.5** | **71.0** | **20.4** |

PMRL reduces the parameter count from 237.5M to 147.5M and FLOPs from 72.6G
to 71.0G while substantially improving tracking accuracy on all three
benchmarks.

## Supported Data

| Dataset | Training | Evaluation |
| --- | :---: | :---: |
| LasHeR-Unaligned | ✓ | ✓ |
| LUART | ✓ | ✓ |
| MUART244 | — | ✓ |

Dataset download links and the unified evaluation toolkit are available in the
[root project README](../README.md).

## Publication Status

The PMRL manuscript is currently **under review**. The paper link, pretrained
models, complete tracking results, and BibTeX entry will be added after the
review process permits public release.

## Acknowledgements

This implementation is developed from
[PMATrack](../PMATrack) and follows the OSTrack/DropTrack-style tracking
pipeline. We thank the authors and maintainers of the related open-source
projects and datasets.

