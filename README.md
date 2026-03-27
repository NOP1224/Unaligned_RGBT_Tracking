<h1 align="center">
<span>Unaligned RGBT Tracking Project
</span>
</h1>

<div align="center">
  
<p align="center">
  <img src="logo.png" width="100%">
</p>

[![PDF](https://img.shields.io/badge/AAAI26Paper-SFCATrack-orange?logoSize=auto)](https://ojs.aaai.org/index.php/AAAI/article/view/38079/42041)
[![Code](https://img.shields.io/badge/Code-SFCATrack-yellow?logoSize=auto)](https://github.com/Yhw-lol127/SFCATrack)
[![Dataset](https://img.shields.io/badge/Dataset-LUART-red?logoSize=auto)](https://github.com/NOP1224/Unaligned_RGBT_Tracking/blob/main/README.md#-download-links)

</div>

This repository contains multiple research modules related to **multi-modal tracking**, **RGB–TIR fusion**, and **unaligned cross-modal UAV tracking**.  
Among them, our recent work:

> **“Progressive Multi-cue Alignment for Unaligned RGBT Tracking”**  
> has been **accepted by CVPR 2026** 🎉.
> 
> **“Unaligned UAV RGBT Tracking: A Largescale Benchmark and A Novel Approach”**  
> has been **accepted by AAAI 2026** 🎉.

This repository includes more than this single paper, but LUART and SFCATrack are important components released here.

---

## 🔔 News
- **2026.02** – Our **Progressive Multi-cue Alignment for Unaligned RGBT Tracking** is accepted by **CVPR 2026**.  
- **2025.11** – Our **Unaligned UAV RGBT Tracking: A Largescale Benchmark and A Novel Approach** is accepted by **AAAI 2026**.  
- **2025.11** – **LUART** (1.02M dual-modality frames) dataset is available for download.  
- Additional modules and trackers will be released soon.

---

## 📢 Public Release
- **2026.02**, we will release the **PMATrack(CVPR 2026)**
- **2025.12**, we publicly released:
- [**AAAI 2026 paper**](https://github.com/NOP1224/Unaligned_RGBT_Tracking/blob/main/Unaligned_UAV_RGBT_Tracking__A_Largescale_Benchmark_and_A_Novel_Approach_AAAI_CRC.pdf),
- [**SFCATrack**](https://github.com/Yhw-lol127/SFCATrack),
- [**LUART dataset**](https://github.com/NOP1224/Unaligned_RGBT_Tracking/blob/main/README.md#-download-links)
- [**LUART evaluation toolkit**](https://github.com/NOP1224/Unaligned_RGBT_Tracking/blob/main/README.md#-download-links)

to support reproducibility and future research on unaligned RGBT tracking.

---

## 📦 MUART244 Dataset (Multi-platform Unaligned RGBT Tracking)
To be coming soon....

## 📦 LUART Dataset (Unaligned UAV RGBT Tracking)

**LUART** is the first large-scale benchmark focusing on *unaligned* UAV visible–thermal tracking.  
It includes:

- **1,453** RGB–TIR sequence pairs  
- **1.02M** dual-modality frames  
- **42** object categories  
- **22** challenge attributes  
- Original UAV resolutions:  
  - RGB: **1920×1080**  
  - TIR: **640×512**

### 📥 Download Links

**LUART Dataset**  
- Baidu Cloud:  
  https://pan.baidu.com/s/168vWYtxPqoagds8WcPuJUA  
- Access Code: `er4r`

**Evaluation Toolkit**  
- Baidu Cloud:  
  https://pan.baidu.com/s/1lv0IBj6UtxZhj1S1UNMPsQ  
- Access Code: `t1vv`

## 📦 LasHeR-Unaligned

We also provide **LasHeR-Unaligned**, a derived benchmark based on  
[LasHeR](https://github.com/BUGPLEASEOUT/LasHeR), where spatial alignment assumptions are explicitly removed to support fair evaluation of unaligned RGBT trackers.

---
## 📊 Benchmark Results

### ⭐ LUART (Test Set)

| Tracker | PR ↑ | NPR ↑ | SR ↑ |
|--------|------|-------|------|
| Best previous method | 54.7 | 49.6 | 42.6 |
| **SFCATrack (Ours)** | **57.3** | **51.9** | **44.6** |

---

### ⭐ LasHeR-Unaligned

| Tracker       | Publication   | PR ↑     | NPR ↑    | SR ↑     |
| ------------- | ------------- | -------- | -------- | -------- |
| MANet         | ICCVW 2019    | 32.9     | 26.6     | 24.1     |
| MaCNet        | Sensors 2020  | 38.4     | 30.7     | 27.0     |
| CAT           | ECCV 2020     | 36.3     | 29.9     | 25.3     |
| FANet         | TIV 2021      | 32.8     | 26.6     | 22.7     |
| ADRNet        | IJCV 2021     | 34.5     | 29.2     | 23.8     |
| MANet++       | TIP 2021      | 30.1     | 23.9     | 20.3     |
| APFNet        | AAAI 2022     | 40.3     | 32.4     | 29.1     |
| DMCNet        | TNNLS 2022    | 35.1     | 27.7     | 25.7     |
| ToMP          | CVPR 2022     | 46.3     | 41.4     | 36.0     |
| OSTrack       | ECCV 2022     | 59.2     | 53.8     | 46.7     |
| TBSI          | CVPR 2023     | 60.3     | 55.2     | 47.7     |
| ViPT          | CVPR 2023     | 55.2     | 51.1     | 44.2     |
| SDSTrack      | CVPR 2024     | 57.6     | 52.5     | 45.3     |
| UnTrack       | CVPR 2024     | 56.5     | 51.5     | 44.7     |
| BAT           | AAAI 2024     | 60.5     | 55.1     | 47.7     |
| AFter         | TIP 2025      | 57.5     | 52.3     | 44.8     |
| SUTrack       | AAAI 2025     | 57.4     | 52.5     | 45.0     |
| CAFormer      | AAAI 2025     | 59.0     | 53.8     | 46.7     |
| AINet         | AAAI 2025     | 61.4     | 55.7     | 48.3     |
| NAT           | CISE 2024     | 58.1     | 52.3     | 44.8     ||
| **SFCATrack** | **AAAI 2026** | **60.7** | **55.1** | **47.9** |
| **PMATrack**      | **CVPR 2026** | **64.4** | **58.7** | **50.6** |


---

## 💻 Open-source Tracker

### SFCATrack
Official implementation of our AAAI 2026 method:

🔗 https://github.com/Yhw-lol127/SFCATrack

---

## 📚 Citation

If you find this repository or the LUART dataset useful for your research,  
please consider citing our AAAI 2026 paper:

```bibtex
