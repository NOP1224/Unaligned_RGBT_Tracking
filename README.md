# Unaligned RGBT Tracking Project
<p align="center">
  <img src="unaligned_rgbt_tracking.png" width="100%">
</p>

This repository contains multiple research modules related to **multi-modal tracking**, **RGB–TIR fusion**, and **unaligned cross-modal UAV tracking**.  
Among them, our recent work:

> **“Unaligned UAV RGBT Tracking: A Largescale Benchmark and A Novel Approach”**  
> has been **accepted by AAAI 2026** 🎉.

This repository includes more than this single paper, but LUART and SFCATrack are important components released here.

---

## 🔔 News
- **2025.11** – Our unaligned UAV RGBT tracking paper is accepted by **AAAI 2026**.  
- **2025.11** – LUART (1.02M dual-modality frames) dataset is available for download.  
- Additional modules and trackers will be released soon.

---

## 📢 Public Release

In **December 2025**, we publicly released:
- the **AAAI 2026 paper**,
- the **official implementation**,
- the **LUART dataset**, and
- the **evaluation toolkit**

to support reproducibility and future research on unaligned RGBT tracking.

---

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

| Tracker | PR ↑ | NPR ↑ | SR ↑ |
|--------|------|-------|------|
| Best previous method | 58.7 | 54.0 | 46.9 |
| **SFCATrack (Ours)** | **60.7** | **55.1** | **47.9** |

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
