# Unaligned RGBT Tracking Project

This repository contains multiple research modules related to **multi-modal tracking**, **RGB–TIR fusion**, and **unaligned cross-modal UAV tracking**.  
Among them, our recent work:

> **“Unaligned UAV RGBT Tracking: A Largescale Benchmark and A Novel Approach”**  
> has been **accepted by AAAI 2026** 🎉.

This repository includes more than this single paper, but LUART and SFCATrack are important components released here.

---

## 🌟 News
- **2026.01** – Our unaligned UAV RGBT tracking paper is accepted by **AAAI 2026**.  
- **2026.01** – LUART (1.02M dual-modality frames) dataset is available for download.  
- Additional modules and trackers will be released soon.

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

### 📥 Download

**Baidu Cloud**  
🔗 https://pan.baidu.com/s/168vWYtxPqoagds8WcPuJUA  
🔑 Code: `er4r`

(Additional mirrors will be added later.)

---

## 📈 Benchmark Results

### ⭐ LUART (Test Set, retrained all methods)
| Tracker | PR ↑ | NPR ↑ | SR ↑ |
|--------|------|-------|------|
| Best previous | 54.7 | 49.6 | 42.6 |
| **SFCATrack (Ours)** | **57.3** | **51.9** | **44.6** |

### ⭐ LasHeR-Unaligned
| Tracker | PR ↑ | NPR ↑ | SR ↑ |
|--------|------|-------|------|
| Best previous | 58.7 | 54.0 | 46.9 |
| **SFCATrack (Ours)** | **60.7** | **55.1** | **47.9** |

SFCATrack achieves new state-of-the-art results on two major unaligned RGBT benchmarks.
