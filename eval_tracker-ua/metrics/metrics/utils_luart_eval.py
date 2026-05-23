import numpy as np

EPS = 1e-12

def _is_invalid_box(b: np.ndarray) -> bool:
    """Matlab 对齐：NaN/Inf/复数 或 w/h<=0 都视为无效"""
    if np.iscomplexobj(b):
        return True
    if not np.isfinite(b).all():
        return True
    if b[2] <= 0 or b[3] <= 0:
        return True
    return False

def _valid_gt_mask_like_matlab(gt: np.ndarray) -> np.ndarray:
    """
    Matlab idx = sum(rect_anno > 0, 2) == 4
    即 x,y,w,h 都必须 > 0 才是有效 GT
    """
    gt = np.asarray(gt, dtype=np.float64)
    return np.all(gt > 0, axis=1)

def sanitize_and_align_like_matlab(serial, gt):
    """
    复现 Matlab calc_seq_err_robust 的关键行为：
    - 长度对齐：取 min_len
    - 第一帧：serial[0] = gt[0]
    - 后续帧：结果无效(NaN/Inf/复数/w/h<=0) -> 用上一帧结果替换（仅按 Matlab 原意实现）
    """
    serial = np.asarray(serial, dtype=np.float64).copy()
    gt = np.asarray(gt, dtype=np.float64)

    L = min(len(serial), len(gt))
    serial = serial[:L].copy()
    gt = gt[:L].copy()

    if L == 0:
        return serial, gt

    # ignore first frame (强制用 GT)
    serial[0] = gt[0]

    for i in range(1, L):
        if _is_invalid_box(serial[i]):
            serial[i] = serial[i - 1].copy()

    return serial, gt

def CLE(pred_box, gt_box):
    """Center Location Error (像素距离), box=[x,y,w,h]"""
    p = np.asarray(pred_box, dtype=np.float64)
    g = np.asarray(gt_box, dtype=np.float64)
    cx_p = p[0] + (p[2] - 1.0) / 2.0
    cy_p = p[1] + (p[3] - 1.0) / 2.0
    cx_g = g[0] + (g[2] - 1.0) / 2.0
    cy_g = g[1] + (g[3] - 1.0) / 2.0
    return float(np.sqrt((cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2))

def normalize_CLE(pred_box, gt_box):
    """
    Matlab norm_dst==1 的逻辑：中心坐标分别除以 GT 的 w/h
    注意：这里按“先算中心，再除以 gt w/h”的等价实现。
    """
    p = np.asarray(pred_box, dtype=np.float64)
    g = np.asarray(gt_box, dtype=np.float64)

    cx_p = p[0] + (p[2] - 1.0) / 2.0
    cy_p = p[1] + (p[3] - 1.0) / 2.0
    cx_g = g[0] + (g[2] - 1.0) / 2.0
    cy_g = g[1] + (g[3] - 1.0) / 2.0

    # 防止除零（与 Matlab 语义一致：无效 GT 会在外层置 -1）
    w = g[2] if g[2] != 0 else 1.0
    h = g[3] if g[3] != 0 else 1.0

    dx = (cx_p / w) - (cx_g / w)
    dy = (cy_p / h) - (cy_g / h)
    return float(np.sqrt(dx * dx + dy * dy))

def IoU(pred_box, gt_box):
    """
    Matlab calc_rect_int 对齐版本 IoU（axis-aligned），并包含：
    if overlap < 0.025 => 0
    """
    p = np.asarray(pred_box, dtype=np.float64)
    g = np.asarray(gt_box, dtype=np.float64)

    # [x1,y1,x2,y2] where x2 = x + w - 1
    p_x1, p_y1 = p[0], p[1]
    p_x2, p_y2 = p[0] + p[2] - 1.0, p[1] + p[3] - 1.0
    g_x1, g_y1 = g[0], g[1]
    g_x2, g_y2 = g[0] + g[2] - 1.0, g[1] + g[3] - 1.0

    inter_x1 = max(p_x1, g_x1)
    inter_y1 = max(p_y1, g_y1)
    inter_x2 = min(p_x2, g_x2)
    inter_y2 = min(p_y2, g_y2)

    iw = inter_x2 - inter_x1 + 1.0
    ih = inter_y2 - inter_y1 + 1.0
    if iw <= 0 or ih <= 0:
        return 0.0

    inter = iw * ih
    area_p = max(p[2], 0.0) * max(p[3], 0.0)
    area_g = max(g[2], 0.0) * max(g[3], 0.0)
    union = area_p + area_g - inter
    if union <= 0:
        return 0.0

    iou = float(inter / union)
    if iou < 0.025:  # Matlab 特殊规则
        iou = 0.0
    return iou

def serial_process(fn, serial, gt):
    """逐帧计算 fn(serial[i], gt[i])"""
    L = min(len(serial), len(gt))
    return [fn(serial[i], gt[i]) for i in range(L)]
