
from .base import Metric
from rgbt.dataset.basedataset import BaseRGBTDataet, TrackerResult
import numpy as np

from metrics.metrics.utils_luart_eval import (
    EPS,
    sanitize_and_align_like_matlab,
    _valid_gt_mask_like_matlab,
    serial_process,
    CLE,
    normalize_CLE,
    IoU
)


def _get_gt_and_serial(dataset, result, seq_name):
    """
    兼容你现在的 try/except 访问方式。
    注意：这里不在原地修改 result[seq_name]，避免污染后续指标。
    """
    try:
        gt = dataset[seq_name]
    except Exception:
        gt = dataset[seq_name]["visible"]

    serial = result[seq_name]
    return gt, serial


class NPR_LUART(Metric):
    """Normalized Precision Rate（对齐 Matlab：0.2 点；并复现 Matlab 对无效 GT 帧的处理）"""
    def __init__(self, thr=np.linspace(0, 0.5, 51)) -> None:
        super().__init__()
        self.thr = np.asarray(thr, dtype=np.float64)

    def __call__(self, dataset, result, seqs: list):
        pr_allseq = []

        for seq_name in seqs:
            gt, serial = _get_gt_and_serial(dataset, result, seq_name)

            serial, gt = sanitize_and_align_like_matlab(serial, gt)
            if len(gt) == 0:
                continue

            res = np.asarray(serial_process(normalize_CLE, serial, gt), dtype=np.float64)

            # Matlab 对齐：GT 无效帧 => err_center = -1（会被计入成功）
            valid_gt = _valid_gt_mask_like_matlab(gt)
            res[~valid_gt] = -1.0

            den = len(res) + EPS  # Matlab 分母：所有帧
            pr_curve = [(np.sum(res <= t) / den) for t in self.thr]
            pr_allseq.append(pr_curve)

        if len(pr_allseq) == 0:
            return 0.0, np.zeros_like(self.thr, dtype=np.float64)

        pr_allseq = np.asarray(pr_allseq, dtype=np.float64)     # [num_seq, 51]
        pr_curve = pr_allseq.mean(axis=0)                       # [51]

        # 0..0.5, 51 点，步长 0.01；0.2 对应 index=20（0-based）
        pr_val = float(pr_curve[20])
        return pr_val, pr_curve


class PR_LUART(Metric):
    """Precision Rate（对齐 Matlab：5px 点；并复现 Matlab 对无效 GT 帧的处理）"""
    def __init__(self, thr=np.linspace(0, 50, 51)) -> None:
        super().__init__()
        self.thr = np.asarray(thr, dtype=np.float64)

    def __call__(self, dataset, result, seqs: list):
        pr_allseq = []

        for seq_name in seqs:
            gt, serial = _get_gt_and_serial(dataset, result, seq_name)

            serial, gt = sanitize_and_align_like_matlab(serial, gt)
            if len(gt) == 0:
                continue

            res = np.asarray(serial_process(CLE, serial, gt), dtype=np.float64)

            # Matlab 对齐：GT 无效帧 => err_center = -1（会被计入成功）
            valid_gt = _valid_gt_mask_like_matlab(gt)
            res[~valid_gt] = -1.0

            den = len(res) + EPS  # Matlab 分母：所有帧
            pr_curve = [(np.sum(res <= t) / den) for t in self.thr]
            pr_allseq.append(pr_curve)

        if len(pr_allseq) == 0:
            return 0.0, np.zeros_like(self.thr, dtype=np.float64)

        pr_allseq = np.asarray(pr_allseq, dtype=np.float64)
        pr_curve = pr_allseq.mean(axis=0)

        # 0..50, 51 点；5px 对应 index=5（0-based）
        pr_val = float(pr_curve[5])
        return pr_val, pr_curve
    
MATLAB_EPS = np.finfo(np.float64).eps  # 对齐 Matlab eps

class SR_LUART(Metric):
    """
    Success Rate 计算，严格对齐 Matlab:
    - eval_tracker.m: success_num_overlap(t) = sum(err_coverage > thr(t)); den = len_all=size(anno,1)
    - calc_seq_err_robust.m: 无效 GT(idx=false) => errCoverage=-1
    - calc_rect_int.m: IoU < 0.025 => 0
    - plot_draw_save.m: aa = aa(sum(aa,2)>eps,:) 过滤全 0 序列; bb = mean(aa)

    参数：
    - ranking_type: 'AUC' 或 'threshold'
    - rank_idx: Matlab 1-based 索引（threshold 模式下），默认 11 对应 IoU=0.5（0:0.05:1）
    - drop_zero_seq: 是否复现 plot_draw_save 的全零序列过滤（默认 True）
    """
    def __init__(
        self,
        thr=np.linspace(0, 1, 21),
        ranking_type="AUC",
        rank_idx=11,
        drop_zero_seq=True,
    ) -> None:
        super().__init__()
        self.thr = np.asarray(thr, dtype=np.float64)
        self.ranking_type = ranking_type
        self.rank_idx = int(rank_idx)  # Matlab: 1-based
        self.drop_zero_seq = bool(drop_zero_seq)

    def __call__(self, dataset, result, seqs: list):
        seq_curves = []

        for seq_name in seqs:
            gt, serial = _get_gt_and_serial(dataset, result, seq_name)

            # 对齐 calc_seq_err_robust：长度对齐 + 第一帧用GT + 无效结果回填
            serial, gt = sanitize_and_align_like_matlab(serial, gt)
            if len(gt) == 0:
                continue

            # 每帧 IoU（IoU 内部已对齐 calc_rect_int：含 <0.025 -> 0）
            res = np.asarray(serial_process(IoU, serial, gt), dtype=np.float64)

            # 对齐 calc_seq_err_robust：GT 无效帧 => errCoverage=-1
            valid_gt = _valid_gt_mask_like_matlab(gt)
            res[~valid_gt] = -1.0

            # 对齐 eval_tracker：分母是全帧 len_all=size(anno,1)
            den = len(res) + MATLAB_EPS
            curve = np.asarray([(np.sum(res > th) / den) for th in self.thr], dtype=np.float64)

            seq_curves.append(curve)

        if len(seq_curves) == 0:
            return 0.0, np.zeros_like(self.thr, dtype=np.float64)

        aa = np.stack(seq_curves, axis=0)  # [num_seq, T]

        # 对齐 plot_draw_save.m：过滤“整条曲线全 0”的序列
        if self.drop_zero_seq:
            keep = aa.sum(axis=1) > MATLAB_EPS
            aa = aa[keep]
            if aa.shape[0] == 0:
                return 0.0, np.zeros_like(self.thr, dtype=np.float64)

        bb = aa.mean(axis=0)  # Matlab: bb = mean(aa)

        if self.ranking_type.lower() == "auc":
            # Matlab: perf(i)=mean(bb)
            sr_val = float(bb.mean())
        elif self.ranking_type.lower() == "threshold":
            # Matlab: perf(i)=bb(rank_idx) (rank_idx 是 1-based)
            idx0 = self.rank_idx - 1
            sr_val = float(bb[idx0])
        else:
            raise ValueError(f"Unknown ranking_type: {self.ranking_type}")

        return sr_val, bb