# Unified Evaluation Toolkit for Unaligned RGBT Tracking

This repository provides a unified evaluation toolkit for **unaligned RGBT tracking**. It is built upon the standard RGBT evaluation protocol and follows the usage style of [`RGBT_toolkit_python`](https://github.com/Alexadlu/RGBT_toolkit_python), while adding support for unaligned RGBT benchmarks such as **LasHeR-UA**, **LUART**, and **MUART244**.

The toolkit supports the common one-pass evaluation metrics used in RGBT tracking:

- **PR**: Precision Rate based on center location error.
- **NPR**: Normalized Precision Rate based on normalized center location error.
- **SR**: Success Rate based on overlap / IoU.
- **Attribute-based evaluation**: metric computation on specific challenge subsets.
- **Visualization**: precision/success plots and attribute radar charts.

---

## 1. Features

- Unified evaluation interface for multiple unaligned RGBT datasets.
- Compatible with standard RGBT result files, where each sequence has one `.txt` prediction file.
- Supports multiple bounding-box formats through automatic conversion.
- Provides dataset-specific evaluation behavior for:
  - `LasHeR_Unalign`
  - `LUART`
  - `MUART244`
- Supports challenge attribute evaluation through predefined attribute annotations.
- Includes MATLAB-compatible handling for LUART evaluation, including invalid-frame processing and robust sequence alignment.

---

## 2. Repository Structure

```text
eval_tracker-ua/
├── eval_lasher_unalign.py       # Example script for LasHeR-UA evaluation
├── eval_luart.py                # Example script for LUART evaluation
├── eval_muart244.py             # Example script for MUART244 evaluation
└── metrics/
    ├── __init__.py
    ├── utils.py                 # bbox conversion, IoU, CLE, text loading utilities
    ├── dataset/
    │   ├── basedataset.py        # base dataset and tracker-result loader
    │   ├── lasher_unalign_dataset.py
    │   ├── luart_dataset.py
    │   ├── muart244_dataset.py
    │   └── ...                  # original RGBT dataset definitions
    ├── metrics/
    │   ├── metrics.py            # PR / NPR / SR for standard and LasHeR-style evaluation
    │   ├── metrics_luart.py      # MATLAB-compatible LUART metrics
    │   └── utils_luart_eval.py   # robust LUART evaluation utilities
    ├── vis/
    │   ├── plot.py
    │   ├── radar.py
    │   └── draw_utils.py
    └── gt_file/
        ├── LasHeR_Unalign/
        ├── LUART/
        └── MUART244/
```

---

## 3. Install dependencies

```bash
pip install numpy matplotlib
pip install rgbt
```

The package `rgbt` is required because this toolkit inherits part of the namespace and visualization utilities from `RGBT_toolkit_python`.

Run all evaluation scripts from the project root so that the local `metrics` package can be correctly imported.

---

## 4. Supported Datasets

| Dataset | Class | Default evaluation metrics | Sequence list |
|---|---|---|---|
| LasHeR-UA | `LasHeR_Unalign` | PR / NPR / SR | `metrics/gt_file/LasHeR_Unalign/lashertest.txt` |
| LUART | `LUART` | PR / NPR / SR | `metrics/gt_file/LUART/testing_list.txt` |
| MUART244 | `MUART244` | PR / NPR / SR | `metrics/gt_file/MUART244/testinglist.txt` |

The toolkit already provides ground-truth annotation files and sequence lists under `metrics/gt_file/`.

---

## 5. Quick Start

### 5.1 Evaluate LasHeR-UA

```python
from metrics import LasHeR_Unalign

lasher_ua = LasHeR_Unalign(
    gt_path="metrics/gt_file/LasHeR_Unalign/annos/",
    seq_name_path="metrics/gt_file/LasHeR_Unalign/lashertest.txt"
)

lasher_ua(
    tracker_name="MyTracker",
    result_path="output/tracking_results/MyTracker/LasHeR-Unaligned",
    bbox_type="ltwh"
)

pr_dict = lasher_ua.PR()
npr_dict = lasher_ua.NPR()
sr_dict = lasher_ua.SR()

print("PR :", pr_dict["MyTracker"][0])
print("NPR:", npr_dict["MyTracker"][0])
print("SR :", sr_dict["MyTracker"][0])
```

You can also run the provided example script after editing `result_path`:

```bash
python eval_lasher_unalign.py
```

---

### 5.2 Evaluate LUART

```python
from metrics import LUART

luart = LUART(
    gt_path="metrics/gt_file/LUART/luart_gt/rgb",
    seq_name_path="metrics/gt_file/LUART/testing_list.txt"
)

luart(
    tracker_name="MyTracker",
    result_path="output/tracking_results/MyTracker/LUART",
    bbox_type="ltwh"
)

pr_dict = luart.PR()
npr_dict = luart.NPR()
sr_dict = luart.SR()

print("PR :", pr_dict["MyTracker"][0])
print("NPR:", npr_dict["MyTracker"][0])
print("SR :", sr_dict["MyTracker"][0])
```

Run the example script:

```bash
python eval_luart.py
```

---

### 5.3 Evaluate MUART244

```python
from metrics import MUART244

muart244 = MUART244(
    gt_path="metrics/gt_file/MUART244/annos/visible",
    seq_name_path="metrics/gt_file/MUART244/testinglist.txt"
)

muart244(
    tracker_name="MyTracker",
    result_path="output/tracking_results/MyTracker/MUART244",
    bbox_type="ltwh"
)

pr_dict = muart244.PR()
npr_dict = muart244.NPR()
sr_dict = muart244.SR()

print("PR :", pr_dict["MyTracker"][0])
print("NPR:", npr_dict["MyTracker"][0])
print("SR :", sr_dict["MyTracker"][0])
```

Run the example script:

```bash
python eval_muart244.py
```

---

## 6. Acknowledgement

This toolkit is built upon the standard RGBT evaluation protocol and follows the interface design of [`RGBT_toolkit_python`](https://github.com/Alexadlu/RGBT_toolkit_python). We thank the authors of the original toolkit for providing a unified Python evaluation framework for GTOT, RGBT210, RGBT234, and LasHeR.
