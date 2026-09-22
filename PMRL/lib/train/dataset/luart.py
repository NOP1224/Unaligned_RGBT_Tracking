import os
import os.path
import numpy as np
import torch
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame, get_x_framev2
import cv2

def sample_decay_gap(low, high, scale):
    """
    Sample integer gap from [low, high], where smaller gaps are more likely.
    Larger scale => flatter decay.
    """
    low = int(low)
    high = int(high)
    if high <= low:
        return low

    width = high - low
    u = np.random.rand()

    # truncated exponential inverse CDF
    x = -scale * np.log(1.0 - u * (1.0 - np.exp(-width / scale)))
    gap = low + int(round(x))

    return int(np.clip(gap, low, high))

class LUART_Dataset(BaseVideoDataset):

    SUPPORTED_MODES = ("aligned", "random_offset", "no_sample")

    def __init__(
        self,
        root=None,
        split="train",
        dtype="rgbrgb",
        seq_ids=None,
        data_fraction=None,
        # unified switch
        mode: str = "aligned",
        # random offset augmentation controls (only meaningful for mode="random_offset")
        use_data_aug: bool = True,
        max_trans: float = None,
        max_scale: float = 1.2,
        min_scale: float = 0.8,
        align_aug_cfg=None,
        random_offset_prob: float = None,
        history_offset_prob: float = None,
        history_min_gap: int = None,
        history_max_gap: int = None,
        history_gap_mu: float = None,
        history_gap_sigma: float = None,
    ):
        root = env_settings().luart_dir if root is None else root
        assert split in ["train", "val", "all", "test"], \
            f"Only support all/train/val/test split in LUART, got {split}"

        super().__init__("Mydataset", root)
        if isinstance(mode, list):
            mode = mode[0]
        mode = str(mode).lower().strip()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode={mode}. Supported: {self.SUPPORTED_MODES}")

        self.dtype = dtype
        self.split = split
        self.mode =  "random_offset"

        self.sequence_list = self._get_sequence_list(split)

        if seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))
        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(
                self.sequence_list, int(len(self.sequence_list) * data_fraction)
            )

        def cfg_value(name, default):
            return getattr(align_aug_cfg, name, default) if align_aug_cfg is not None else default

        # random/history offset params
        # MAX_TRANS is interpreted as an object-size scale, not absolute pixels.
        # Per sample: max_trans_px = MAX_TRANS * sqrt(bbox_w * bbox_h).
        # Example: MAX_TRANS=1.0 allows translation up to one geometric target size.
        self.max_trans = 2.0
        self.max_scale = float(max_scale if max_scale is not None else cfg_value("MAX_SCALE", 1.2))
        self.min_scale = float(min_scale if min_scale is not None else cfg_value("MIN_SCALE", 0.8))
        self.random_offset_prob = float(
            random_offset_prob if random_offset_prob is not None else cfg_value("RANDOM_OFFSET_PROB", 0.1)
        )
        self.history_offset_prob = float(
            history_offset_prob if history_offset_prob is not None else cfg_value("HISTORY_OFFSET_PROB", 0.9)
        )
        self.history_min_gap = int(
            history_min_gap if history_min_gap is not None else cfg_value("HISTORY_MIN_GAP", 1)
        )
        self.history_max_gap = int(
            history_max_gap if history_max_gap is not None else cfg_value("HISTORY_MAX_GAP", 75)
        )
        self.history_gap_mu = float(
            history_gap_mu if history_gap_mu is not None else cfg_value("HISTORY_GAP_MU", 20.0)
        )
        self.history_gap_sigma = float(
            history_gap_sigma if history_gap_sigma is not None else cfg_value("HISTORY_GAP_SIGMA", 12.0)
        )

        # mode-4-like offset sampling distribution.  Keep the original two-branch
        # random_offset pipeline unchanged:
        #   1) history_offset: simulate step-align / pre-align online state
        #   2) random_offset: simulate alignment failure / residual outliers
        self.history_recent_prob = float(cfg_value("HISTORY_RECENT_PROB", 0.70))
        self.history_delay_prob = float(cfg_value("HISTORY_DELAY_PROB", 0.25))
        self.history_recent_scale = float(cfg_value("HISTORY_RECENT_SCALE", 8.0))
        self.history_stale_min_gap = int(cfg_value("HISTORY_STALE_MIN_GAP", 40))
        self.history_min_visible_ratio = float(cfg_value("HISTORY_MIN_VISIBLE_RATIO", 0.25))
        self.history_min_bbox_size = float(cfg_value("HISTORY_MIN_BBOX_SIZE", 2.0))

        self.failure_small_prob = float(cfg_value("FAILURE_SMALL_PROB", 0.60))
        self.failure_mid_prob = float(cfg_value("FAILURE_MID_PROB", 0.30))
        self.failure_small_gamma_scale = float(cfg_value("FAILURE_SMALL_GAMMA_SCALE", 6.0))
        self.failure_mid_gamma_scale = float(cfg_value("FAILURE_MID_GAMMA_SCALE", 16.0))
        self.failure_severe_min_ratio = float(cfg_value("FAILURE_SEVERE_MIN_RATIO", 0.45))
        self.failure_log_scale_small = float(cfg_value("FAILURE_LOG_SCALE_SMALL", 0.025))
        self.failure_log_scale_mid = float(cfg_value("FAILURE_LOG_SCALE_MID", 0.055))

        # LUART-specific online-state simulation.
        # All constraints are enforced inside the dataset augmentation itself so
        # downstream Processing never needs to invalidate a sample.
        self.search_factor = float(cfg_value("SEARCH_FACTOR", 4.0))
        self.search_center_jitter = float(cfg_value("SEARCH_CENTER_JITTER", 3.0))
        self.search_scale_jitter = float(cfg_value("SEARCH_SCALE_JITTER", 0.25))
        self.search_crop_retry = int(cfg_value("SEARCH_CROP_RETRY", 80))
        # Datav3: keep LUART's natural causal-history residual distribution.
        # Do not reshape it into hand-crafted easy/medium/hard buckets.
        # Only add a small previous-state estimation error, then accept the
        # candidate iff an RGB-only sampled search crop can fully observe TIR.
        self.history_prior_error_probs = np.asarray(
            cfg_value("HISTORY_PRIOR_ERROR_PROBS", (0.90, 0.10)), dtype=np.float64
        )
        self.history_prior_error_probs /= self.history_prior_error_probs.sum()
        self.history_candidate_retry = int(cfg_value("HISTORY_CANDIDATE_RETRY", 80))
        self.using_random_align_sample = True

        # At validation/test time, random_offset mode should mimic online step-align:
        # mostly history-derived pre-align states, not independent synthetic failures.
        # aligned/no_sample should keep their own semantics.
        if self.split != "train" and self.mode == "random_offset":
            self.random_offset_prob = 0.0
            self.history_offset_prob = 1.0
            self.using_random_align_sample = True
        # logging
        if self.mode == "aligned":
            print(f"----Split [{self.split}] Mode=aligned (GT align & fuse) ----")
        elif self.mode == "no_sample":  
            print(f"----Split [{self.split}] Mode=no_sample ----")
        else:
            if self.using_random_align_sample:
                print(f"----Split [{self.split}] Mode=random_offset with aug "
                      f"--RANDOM: [{self.random_offset_prob}] HISTORY: [{self.history_offset_prob}] "
                      f"GAP: [{self.history_min_gap}, {self.history_max_gap}] "
                      f"HIST_DIST: [recent/delay/stale] FAIL_DIST: [long_tail_logscale] "
                      f"MAX_TRANS_SCALE: [{self.max_trans}] SCALE: [{self.max_scale}, {self.min_scale}]-- ----")
            else:
                print(f"----Split [{self.split}] Mode=random_offset without aug ----")

    def get_name(self):
        return "luart"

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True  # w=h=0 in visible.txt and infrared.txt is occlusion/oov

    # ---------------------------
    # Geometry helpers
    # ---------------------------
    def _bbox_to_center_size(self, bbox_np):
        """bbox: numpy [x,y,w,h] -> center(x,y), size(w,h)"""
        x, y, w, h = bbox_np
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        return np.array([cx, cy], dtype=np.float32), np.array([w, h], dtype=np.float32)

    def _build_4param_homography(self, dx, dy, sx, sy):
        """RGB scale homography:
        x' = sx * x + dx
        y' = sy * y + dy
        """
        H = np.array([[sx, 0.0, dx],
                      [0.0, sy, dy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        return H

    def _warp_bbox_only(self, bbox, H, img_w, img_h):
        """Warp bbox corners by homography and return the enclosing bbox (xywh)."""
        x, y, w, h = bbox.astype(np.float32)
        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32)

        corners_h = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1).T  # (3,4)
        warped = H @ corners_h
        warped = (warped[:2, :] / warped[2:, :]).T  # (4,2)

        warped[:, 0] = np.clip(warped[:, 0], 0, img_w - 1)
        warped[:, 1] = np.clip(warped[:, 1], 0, img_h - 1)

        x_min, y_min = warped.min(axis=0)
        x_max, y_max = warped.max(axis=0)
        new_bbox = np.array([x_min,
                             y_min,
                             max(1.0, x_max - x_min),
                             max(1.0, y_max - y_min)], dtype=np.float32)
        return new_bbox

    def _get_scaled_bbox_ir(self, bbox_ir, w_rgb, h_rgb, w_tir, h_tir):
        """Scale TIR bbox (TIR coords) into RGB coordinate system."""
        scale = torch.tensor(
            [w_rgb / w_tir,
             h_rgb / h_tir,
             w_rgb / w_tir,
             h_rgb / h_tir],
            dtype=bbox_ir.dtype,
            device=bbox_ir.device
        )
        return bbox_ir * scale

    def _calc_align_gt_4param(self, bbox_rgb, bbox_tir_scaled):
        """Return align_gt = [dx,dy,sx,sy] where dx,dy are center offsets (TIR - RGB) and sx,sy are size ratios (TIR/RGB)."""
        if isinstance(bbox_rgb, torch.Tensor):
            b_rgb = bbox_rgb.detach().cpu().numpy().astype(np.float32)
        else:
            b_rgb = np.asarray(bbox_rgb, dtype=np.float32)

        if isinstance(bbox_tir_scaled, torch.Tensor):
            b_tir = bbox_tir_scaled.detach().cpu().numpy().astype(np.float32)
        else:
            b_tir = np.asarray(bbox_tir_scaled, dtype=np.float32)

        # invalid bbox -> default
        if b_rgb.shape[0] != 4 or b_tir.shape[0] != 4 or b_rgb[2] <= 0 or b_rgb[3] <= 0 or b_tir[2] <= 0 or b_tir[3] <= 0:
            return torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)

        c_rgb, s_rgb = self._bbox_to_center_size(b_rgb)
        c_tir, s_tir = self._bbox_to_center_size(b_tir)

        dx = float(c_tir[0] - c_rgb[0])
        dy = float(c_tir[1] - c_rgb[1])
        sx = float(s_tir[0] / (s_rgb[0] + 1e-6))
        sy = float(s_tir[1] / (s_rgb[1] + 1e-6))

        return torch.tensor([dx, dy, sx, sy], dtype=torch.float32)

    def _bbox_to_numpy(self, bbox):
        if isinstance(bbox, torch.Tensor):
            return bbox.detach().cpu().numpy().astype(np.float32)
        return np.asarray(bbox, dtype=np.float32)

    def _offset_to_numpy(self, offset):
        if offset is None:
            return None
        if isinstance(offset, torch.Tensor):
            offset = offset.detach().cpu().numpy()
        offset = np.asarray(offset, dtype=np.float32).reshape(-1)
        if offset.shape[0] != 4 or not np.all(np.isfinite(offset)):
            return None
        return offset

    def _compute_bbox_offset_4param(self, bbox_rgb, bbox_tir_scaled):
        bbox_rgb_np = self._bbox_to_numpy(bbox_rgb)
        bbox_tir_np = self._bbox_to_numpy(bbox_tir_scaled)
        if (
            bbox_rgb_np.shape[0] != 4
            or bbox_tir_np.shape[0] != 4
            or bbox_rgb_np[2] <= 0
            or bbox_rgb_np[3] <= 0
            or bbox_tir_np[2] <= 0
            or bbox_tir_np[3] <= 0
        ):
            return None
        center_rgb, size_rgb = self._bbox_to_center_size(bbox_rgb_np)
        center_tir, size_tir = self._bbox_to_center_size(bbox_tir_np)
        return np.array(
            [
                center_tir[0] - center_rgb[0],
                center_tir[1] - center_rgb[1],
                size_tir[0] / (size_rgb[0] + 1e-6),
                size_tir[1] / (size_rgb[1] + 1e-6),
            ],
            dtype=np.float32,
        )

    def _bbox_geometric_size(self, bbox, default=1.0):
        """Return sqrt(w*h) for an xywh bbox."""
        bbox_np = self._bbox_to_numpy(bbox)
        if bbox_np.shape[0] != 4 or not np.all(np.isfinite(bbox_np)):
            return float(default)
        w = float(bbox_np[2])
        h = float(bbox_np[3])
        if w <= 0.0 or h <= 0.0:
            return float(default)
        return float(np.sqrt(max(w * h, 1e-6)))

    def _resolve_max_trans_px(self, max_trans_scale=None, bbox_ref=None, default_px=1.0):
        """Resolve dynamic translation threshold.

        max_trans_scale is a scale factor, not an absolute pixel value:
            max_trans_px = max_trans_scale * sqrt(bbox_w * bbox_h)

        bbox_ref should be the current RGB target bbox for current-frame
        augmentation and validation.
        """
        scale = self.max_trans if max_trans_scale is None else float(max_trans_scale)
        if bbox_ref is None:
            return max(float(default_px), float(scale))
        geom = self._bbox_geometric_size(bbox_ref, default=float(default_px))
        return max(float(default_px), float(scale) * float(geom))

    def _is_valid_offset_4param(self, offset, max_trans=None, min_scale=None, max_scale=None, bbox_ref=None):
        offset_np = self._offset_to_numpy(offset)
        if offset_np is None:
            return False
        dx, dy, sx, sy = offset_np.tolist()
        if max_trans is not None:
            max_trans_px = self._resolve_max_trans_px(max_trans, bbox_ref=bbox_ref) if bbox_ref is not None else float(max_trans)
            # Use Euclidean center displacement as the primary constraint.
            # This avoids allowing sqrt(2)*max_trans when both dx and dy are large.
            if np.sqrt(dx * dx + dy * dy) > max_trans_px:
                return False
        if min_scale is not None and (sx < min_scale or sy < min_scale):
            return False
        if max_scale is not None and (sx > max_scale or sy > max_scale):
            return False
        return True

    def _sample_gaussian_history_frame_id(
        self,
        cur_frame_id,
        frame_length,
        min_gap=1,
        max_gap=75,
        mu=20.0,
        sigma=12.0,
        max_retry=50,
    ):
        """Sample a causal history frame using only frames before ``cur_frame_id``.

        The online step-alignment state can only originate from a past frame.
        The recent/delayed/stale mixture controls the gap distribution, while
        ``[min_gap, max_gap]`` remains a hard frame-index interval.
        """
        del mu, sigma, max_retry  # retained for backward-compatible configuration
        cur_frame_id = int(cur_frame_id)
        frame_length = int(frame_length)
        min_gap = max(1, int(min_gap))
        max_gap = max(min_gap, int(max_gap))

        # Only past frames are legal. This also prevents future-frame leakage.
        max_available_gap = min(max_gap, cur_frame_id, frame_length - 1)
        if frame_length <= 1 or max_available_gap < min_gap:
            return None

        recent_prob = float(np.clip(getattr(self, "history_recent_prob", 0.70), 0.0, 1.0))
        delay_prob = float(np.clip(getattr(self, "history_delay_prob", 0.25), 0.0, 1.0 - recent_prob))
        stale_prob = max(0.0, 1.0 - recent_prob - delay_prob)

        ranges = []
        weights = []
        candidates = (
            (min_gap, min(max_available_gap, 50), recent_prob, 35.0),
            (max(min_gap, 50), min(max_available_gap, 100), delay_prob, 35.0),
            (max(min_gap, 100), min(max_available_gap, 200), stale_prob, 70.0),
        )
        for low, high, weight, scale in candidates:
            if low <= high and weight > 0.0:
                ranges.append((low, high, scale))
                weights.append(weight)

        if not ranges:
            return cur_frame_id - min_gap

        weights_np = np.asarray(weights, dtype=np.float64)
        weights_np /= weights_np.sum()
        branch = int(np.random.choice(len(ranges), p=weights_np))
        low, high, scale = ranges[branch]
        gap = sample_decay_gap(low=low, high=high, scale=scale)
        gap = int(np.clip(gap, min_gap, max_available_gap))
        hist_id = cur_frame_id - gap
        return hist_id if 0 <= hist_id < cur_frame_id else None

    def _resolve_history_frame_length(self, anno, frame_length):
        """Return the common safe length of images and all accessed annotations."""
        lengths = []
        try:
            candidate = int(frame_length)
            if candidate > 0:
                lengths.append(candidate)
        except (TypeError, ValueError):
            pass

        if anno is not None:
            for key in ("bbox_rgb", "bbox_ir", "valid"):
                value = anno.get(key) if hasattr(anno, "get") else None
                if value is None:
                    continue
                try:
                    candidate = int(len(value))
                except TypeError:
                    continue
                if candidate > 0:
                    lengths.append(candidate)
        return min(lengths) if lengths else 0

    def _bbox_pair_to_rgb_to_tir_homography(self, bbox_rgb, bbox_tir_scaled):
        """Build the diagonal affine transform mapping RGB coordinates to TIR."""
        bbox_rgb_np = self._bbox_to_numpy(bbox_rgb)
        bbox_tir_np = self._bbox_to_numpy(bbox_tir_scaled)
        if (
            bbox_rgb_np.shape[0] != 4 or bbox_tir_np.shape[0] != 4
            or not np.all(np.isfinite(bbox_rgb_np)) or not np.all(np.isfinite(bbox_tir_np))
            or bbox_rgb_np[2] <= 0 or bbox_rgb_np[3] <= 0
            or bbox_tir_np[2] <= 0 or bbox_tir_np[3] <= 0
        ):
            return None
        center_rgb, size_rgb = self._bbox_to_center_size(bbox_rgb_np)
        center_tir, size_tir = self._bbox_to_center_size(bbox_tir_np)
        sx = float(size_tir[0] / max(size_rgb[0], 1e-6))
        sy = float(size_tir[1] / max(size_rgb[1], 1e-6))
        tx = float(center_tir[0] - sx * center_rgb[0])
        ty = float(center_tir[1] - sy * center_rgb[1])
        H = self._build_4param_homography(tx, ty, sx, sy)
        return H if np.all(np.isfinite(H)) else None

    def _warp_bbox_with_visibility(self, bbox, H, img_w, img_h):
        """Warp an xywh box and return its clipped box plus retained area ratio."""
        bbox_np = self._bbox_to_numpy(bbox)
        H_np = np.asarray(H, dtype=np.float32)
        if bbox_np.shape[0] != 4 or H_np.shape != (3, 3):
            return None, 0.0
        x, y, bw, bh = bbox_np.tolist()
        corners = np.array(
            [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]],
            dtype=np.float32,
        )
        corners_h = np.concatenate(
            [corners, np.ones((4, 1), dtype=np.float32)], axis=1
        ).T
        warped_h = H_np @ corners_h
        denom = warped_h[2:3, :]
        if np.any(np.abs(denom) < 1e-8):
            return None, 0.0
        warped = (warped_h[:2, :] / denom).T
        if not np.all(np.isfinite(warped)):
            return None, 0.0

        x_min, y_min = warped.min(axis=0)
        x_max, y_max = warped.max(axis=0)
        full_w = max(float(x_max - x_min), 0.0)
        full_h = max(float(y_max - y_min), 0.0)
        full_area = full_w * full_h
        if full_area <= 1e-6:
            return None, 0.0

        clip_x0 = float(np.clip(x_min, 0.0, max(float(img_w - 1), 0.0)))
        clip_y0 = float(np.clip(y_min, 0.0, max(float(img_h - 1), 0.0)))
        clip_x1 = float(np.clip(x_max, 0.0, max(float(img_w - 1), 0.0)))
        clip_y1 = float(np.clip(y_max, 0.0, max(float(img_h - 1), 0.0)))
        clip_w = max(clip_x1 - clip_x0, 0.0)
        clip_h = max(clip_y1 - clip_y0, 0.0)
        visible_ratio = float((clip_w * clip_h) / max(full_area, 1e-6))
        clipped_bbox = np.array([clip_x0, clip_y0, clip_w, clip_h], dtype=np.float32)
        return clipped_bbox, visible_ratio

    def _bbox_inside_search_crop_from_anchor(self, search_anchor, bbox_tir, eps=1e-5):
        """Check whether the full TIR target lies in the actual 4x search crop.

        ``search_anchor`` is generated only from the RGB target using the same
        jitter distribution as Processing.  TIR is used strictly as an
        accept/reject condition and never changes crop center or crop size.
        """
        anchor = self._bbox_to_numpy(search_anchor)
        b_tir = self._bbox_to_numpy(bbox_tir)
        if (
            anchor.shape[0] != 4 or b_tir.shape[0] != 4
            or not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(b_tir))
            or anchor[2] <= 0 or anchor[3] <= 0 or b_tir[2] <= 0 or b_tir[3] <= 0
        ):
            return False

        anchor_center, anchor_size = self._bbox_to_center_size(anchor)
        crop_side = float(
            np.ceil(np.sqrt(max(anchor_size[0] * anchor_size[1], 1e-6)) * self.search_factor)
        )
        half = 0.5 * crop_side
        crop_x0, crop_y0 = anchor_center - half
        crop_x1, crop_y1 = anchor_center + half

        tx0, ty0, tw, th = b_tir.tolist()
        tx1, ty1 = tx0 + tw, ty0 + th
        return bool(
            tx0 >= crop_x0 - eps and ty0 >= crop_y0 - eps
            and tx1 <= crop_x1 + eps and ty1 <= crop_y1 + eps
        )

    def _sample_rgb_search_anchor_containing_tir(self, bbox_rgb, bbox_tir, max_retry=None):
        """Sample a standard RGB-only jitter anchor and reject if TIR is outside.

        This exactly preserves the causal direction used at test time:
        RGB tracking state -> search crop.  TIR GT never participates in crop
        generation; it is only checked after a crop has already been sampled.
        """
        b_rgb = self._bbox_to_numpy(bbox_rgb)
        b_tir = self._bbox_to_numpy(bbox_tir)
        if (
            b_rgb.shape[0] != 4 or b_tir.shape[0] != 4
            or not np.all(np.isfinite(b_rgb)) or not np.all(np.isfinite(b_tir))
            or b_rgb[2] <= 0 or b_rgb[3] <= 0 or b_tir[2] <= 0 or b_tir[3] <= 0
        ):
            return None

        retries = self.search_crop_retry if max_retry is None else int(max_retry)
        center_rgb, size_rgb = self._bbox_to_center_size(b_rgb)
        for _ in range(max(retries, 1)):
            # Same distribution as SFCAProcessing._get_jittered_box_notalign2.
            jittered_size = size_rgb * np.exp(
                np.random.randn(2).astype(np.float32) * self.search_scale_jitter
            )
            max_offset = float(
                np.sqrt(max(jittered_size[0] * jittered_size[1], 1e-6))
                * self.search_center_jitter
            )
            jittered_center = center_rgb + max_offset * (
                np.random.rand(2).astype(np.float32) - 0.5
            )
            anchor = np.concatenate(
                [jittered_center - 0.5 * jittered_size, jittered_size]
            ).astype(np.float32)
            if self._bbox_inside_search_crop_from_anchor(anchor, b_tir):
                return torch.from_numpy(anchor)
        return None

    def _sample_history_prior_error(self, bbox_rgb):
        """Small target-centric error that mimics an imperfect accumulated state.

        Large displacement is provided by LUART's real temporal geometry.  This
        perturbation stays deliberately small: 90% mild residuals and 10%
        medium residuals, with no synthetic hard-history branch.
        """
        b_rgb = self._bbox_to_numpy(bbox_rgb)
        center_rgb, _ = self._bbox_to_center_size(b_rgb)
        target_size = self._bbox_geometric_size(b_rgb, default=1.0)

        branch = int(np.random.choice(2, p=self.history_prior_error_probs))
        if branch == 0:
            # Mild previous-state error, capped at 0.15 target sizes.
            radius = min(
                float(np.random.gamma(shape=2.0, scale=0.035 * target_size)),
                0.15 * target_size,
            )
            log_scale_std = 0.010
        else:
            # Medium previous-state error.  Real history supplies hard motion.
            radius = float(np.random.uniform(0.15 * target_size, 0.35 * target_size))
            log_scale_std = 0.025

        theta = np.random.uniform(0.0, 2.0 * np.pi)
        dx = float(radius * np.cos(theta))
        dy = float(radius * np.sin(theta))
        sx = float(np.clip(np.exp(np.random.laplace(0.0, log_scale_std)), 0.90, 1.10))
        sy = float(np.clip(np.exp(np.random.laplace(0.0, log_scale_std)), 0.90, 1.10))

        cx, cy = center_rgb.tolist()
        tx = float(cx + dx - sx * cx)
        ty = float(cy + dy - sy * cy)
        return self._build_4param_homography(tx, ty, sx, sy)

    def _compute_history_prealign_homography(
        self,
        cur_frame_id,
        anno,
        frame_length,
        w_rgb,
        h_rgb,
        w_tir,
        h_tir,
        current_bbox_rgb,
        current_bbox_tir_scaled,
        max_retry=30,
    ):
        """Sample a historical pre-alignment state and its RGB-only search crop.

        Training keeps LUART's natural causal-history motion distribution.  A
        small accumulated-state perturbation is composed after the GT history
        transform.  A candidate is accepted only if a *standard RGB-only*
        search jitter crop fully contains the transformed TIR target.  TIR GT
        never changes the sampled crop.

        Validation/test retain the original exact-history behavior and return
        no pre-sampled search anchor.
        """
        safe_frame_length = self._resolve_history_frame_length(anno, frame_length)
        cur_frame_id = int(cur_frame_id)
        if safe_frame_length <= 1 or not (0 <= cur_frame_id < safe_frame_length):
            return None, None

        min_size = float(getattr(self, "history_min_bbox_size", 2.0))
        train_state_sampling = self.split == "train"
        retry_count = (
            max(int(max_retry), int(self.history_candidate_retry))
            if train_state_sampling else int(max_retry)
        )

        for _ in range(retry_count):
            hist_id = self._sample_gaussian_history_frame_id(
                cur_frame_id,
                safe_frame_length,
                min_gap=self.history_min_gap,
                max_gap=self.history_max_gap,
                mu=self.history_gap_mu,
                sigma=self.history_gap_sigma,
            )
            if hist_id is None:
                return None, None
            if not (0 <= hist_id < cur_frame_id):
                continue
            if "valid" in anno and not bool(anno["valid"][hist_id]):
                continue

            bbox_rgb_hist = anno["bbox_rgb"][hist_id].clone()
            bbox_tir_hist = anno["bbox_ir"][hist_id].clone()
            bbox_tir_hist_scaled = self._get_scaled_bbox_ir(
                bbox_tir_hist, w_rgb, h_rgb, w_tir, h_tir
            )
            G_hist = self._bbox_pair_to_rgb_to_tir_homography(
                bbox_rgb_hist, bbox_tir_hist_scaled
            )
            if G_hist is None:
                continue
            try:
                H_hist = np.linalg.inv(G_hist).astype(np.float32)
            except np.linalg.LinAlgError:
                continue

            H_candidate = H_hist
            if train_state_sampling:
                H_error = self._sample_history_prior_error(current_bbox_rgb)
                H_candidate = (H_error @ H_hist).astype(np.float32)

            warped_bbox, visible_ratio = self._warp_bbox_with_visibility(
                current_bbox_tir_scaled, H_candidate, w_rgb, h_rgb
            )
            if warped_bbox is None:
                continue
            if warped_bbox[2] < min_size or warped_bbox[3] < min_size:
                continue

            if train_state_sampling:
                # Full observability is required.  If part of the transformed
                # target leaves the source image, no search crop can recover it.
                if visible_ratio < 0.999:
                    continue
                search_anchor = self._sample_rgb_search_anchor_containing_tir(
                    current_bbox_rgb, warped_bbox
                )
                if search_anchor is None:
                    # Reject this history/noise state inside the History branch.
                    continue
                return H_candidate, search_anchor

            # Validation/test: preserve the original exact-history distribution.
            min_visible = float(getattr(self, "history_min_visible_ratio", 0.25))
            if visible_ratio >= min_visible:
                return H_candidate, None

        return None, None

    def _build_homography_from_target_offset(self, bbox_rgb, bbox_tir_scaled, target_offset):
        bbox_rgb_np = self._bbox_to_numpy(bbox_rgb)
        bbox_tir_np = self._bbox_to_numpy(bbox_tir_scaled)
        target_offset_np = self._offset_to_numpy(target_offset)
        center_rgb, size_rgb = self._bbox_to_center_size(bbox_rgb_np)
        center_tir, size_tir = self._bbox_to_center_size(bbox_tir_np)
        dx, dy, sx, sy = target_offset_np.tolist()
        target_size = np.maximum(size_rgb * np.array([sx, sy], dtype=np.float32), 1.0)
        target_center = center_rgb + np.array([dx, dy], dtype=np.float32)
        warp_sx = target_size[0] / max(size_tir[0], 1e-6)
        warp_sy = target_size[1] / max(size_tir[1], 1e-6)
        warp_dx = target_center[0] - warp_sx * center_tir[0]
        warp_dy = target_center[1] - warp_sy * center_tir[1]
        return self._build_4param_homography(warp_dx, warp_dy, warp_sx, warp_sy)

    # ---------------------------
    # Mode=random_offset augmentation
    # ---------------------------
    def _sample_failure_offset_4param(
        self,
        max_trans,
        min_scale,
        max_scale,
    ):
        """Sample offset for the original random_offset failure branch.

        Alignment failure is not uniform over the whole search range.  We use a
        long-tailed radial translation distribution and log-space scale noise:
          - most failures are mild/moderate residuals
          - a small portion are severe outliers
          - scale perturbation is multiplicative, so it is sampled in log-space
        """
        max_trans = float(max_trans)
        min_scale = float(min_scale)
        max_scale = float(max_scale)

        small_prob = float(np.clip(getattr(self, "failure_small_prob", 0.60), 0.0, 1.0))
        mid_prob = float(np.clip(getattr(self, "failure_mid_prob", 0.30), 0.0, 1.0 - small_prob))
        small_scale = max(float(getattr(self, "failure_small_gamma_scale", 6.0)), 1e-6)
        mid_scale = max(float(getattr(self, "failure_mid_gamma_scale", 16.0)), 1e-6)
        severe_min_ratio = float(np.clip(getattr(self, "failure_severe_min_ratio", 0.45), 0.0, 1.0))

        u = np.random.rand()
        if u < small_prob:
            r = np.random.gamma(shape=2.0, scale=small_scale)
        elif u < small_prob + mid_prob:
            r = np.random.gamma(shape=2.0, scale=mid_scale)
        else:
            r = np.random.uniform(severe_min_ratio * max_trans, max_trans)
        r = float(np.clip(r, 0.0, max_trans))

        theta = np.random.uniform(0.0, 2.0 * np.pi)
        dx_raw = float(r * np.cos(theta))
        dy_raw = float(r * np.sin(theta))

        max_log_scale = max(abs(np.log(max(max_scale, 1e-6))), abs(np.log(max(min_scale, 1e-6))))
        if u < small_prob:
            log_sx = np.random.laplace(0.0, float(getattr(self, "failure_log_scale_small", 0.025)))
            log_sy = np.random.laplace(0.0, float(getattr(self, "failure_log_scale_small", 0.025)))
        elif u < small_prob + mid_prob:
            log_sx = np.random.laplace(0.0, float(getattr(self, "failure_log_scale_mid", 0.055)))
            log_sy = np.random.laplace(0.0, float(getattr(self, "failure_log_scale_mid", 0.055)))
        else:
            log_sx = np.random.uniform(-max_log_scale, max_log_scale)
            log_sy = np.random.uniform(-max_log_scale, max_log_scale)

        sx_raw = float(np.clip(np.exp(log_sx), min_scale, max_scale))
        sy_raw = float(np.clip(np.exp(log_sy), min_scale, max_scale))
        return dx_raw, dy_raw, sx_raw, sy_raw

    def _apply_random_rgb_tir_offset_4param(
        self,
        img,
        bbox_rgb,
        bbox_tir_scaled,
        max_trans=None,
        min_scale=0.8,
        max_scale=1.2,
        max_retry=1000,
        history_transform=None,
        enforce_search_crop=False,
        search_anchor=None,
    ):
        """Warp TIR and return geometry plus the accepted RGB-only search anchor.

        For LUART training, a candidate is accepted only when the standard
        RGB-only search jitter crop fully contains the transformed TIR target.
        Random-failure retries stay inside the 10% Random branch; History uses
        the pre-sampled anchor returned by the history-state sampler.
        """
        h, w = img.shape[:2]
        bbox_rgb_np = self._bbox_to_numpy(bbox_rgb)
        bbox_tir_np = self._bbox_to_numpy(bbox_tir_scaled)
        if (
            bbox_rgb_np.shape[0] != 4 or bbox_tir_np.shape[0] != 4
            or bbox_rgb_np[2] <= 0 or bbox_rgb_np[3] <= 0
            or bbox_tir_np[2] <= 0 or bbox_tir_np[3] <= 0
        ):
            align_gt = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
            fallback_anchor = torch.as_tensor(bbox_rgb).clone().to(torch.float32) if enforce_search_crop else None
            return img, bbox_rgb, bbox_tir_scaled, align_gt, fallback_anchor

        center_rgb, size_rgb = self._bbox_to_center_size(bbox_rgb_np)
        max_trans_px = self._resolve_max_trans_px(max_trans, bbox_ref=bbox_rgb_np)
        min_visible = float(getattr(self, "history_min_visible_ratio", 0.25))
        min_size = float(getattr(self, "history_min_bbox_size", 2.0))
        accepted = False
        H_final = np.eye(3, dtype=np.float32)
        bbox_tir_new_np = bbox_tir_np.copy()
        accepted_anchor = search_anchor

        history_requested = history_transform is not None
        if history_requested:
            H = np.asarray(history_transform, dtype=np.float32)
            bbox_tir_tmp, visible_ratio = self._warp_bbox_with_visibility(bbox_tir_np, H, w, h)
            anchor_ok = True
            if enforce_search_crop:
                if visible_ratio < 0.999:
                    anchor_ok = False
                elif accepted_anchor is None:
                    accepted_anchor = self._sample_rgb_search_anchor_containing_tir(
                        bbox_rgb_np, bbox_tir_tmp
                    )
                    anchor_ok = accepted_anchor is not None
                else:
                    anchor_ok = self._bbox_inside_search_crop_from_anchor(
                        accepted_anchor, bbox_tir_tmp
                    )
            if (
                H.shape == (3, 3) and np.all(np.isfinite(H))
                and bbox_tir_tmp is not None
                and visible_ratio >= (0.999 if enforce_search_crop else min_visible)
                and bbox_tir_tmp[2] >= min_size and bbox_tir_tmp[3] >= min_size
                and anchor_ok
            ):
                accepted = True
                H_final = H
                bbox_tir_new_np = bbox_tir_tmp
        else:
            # The Random branch remains exactly a branch-local long-tail failure
            # sampler.  Crop rejection triggers another Random candidate, never
            # a History candidate and never a Processing-level sample drop.
            for _ in range(max_retry):
                dx_raw, dy_raw, sx_raw, sy_raw = self._sample_failure_offset_4param(
                    max_trans=max_trans_px,
                    min_scale=min_scale,
                    max_scale=max_scale,
                )
                H = self._build_4param_homography(dx_raw, dy_raw, sx_raw, sy_raw)
                bbox_tir_tmp, visible_ratio = self._warp_bbox_with_visibility(bbox_tir_np, H, w, h)
                if bbox_tir_tmp is None:
                    continue
                if visible_ratio < (0.999 if enforce_search_crop else min_visible):
                    continue

                center_tir1, size_tir1 = self._bbox_to_center_size(bbox_tir_tmp)
                dx_tmp = center_tir1[0] - center_rgb[0]
                dy_tmp = center_tir1[1] - center_rgb[1]
                sx_tmp = size_tir1[0] / (size_rgb[0] + 1e-6)
                sy_tmp = size_tir1[1] / (size_rgb[1] + 1e-6)
                if not (
                    bbox_tir_tmp[2] >= min_size and bbox_tir_tmp[3] >= min_size
                    and self._is_valid_offset_4param(
                        [dx_tmp, dy_tmp, sx_tmp, sy_tmp],
                        max_trans=max_trans_px,
                        min_scale=min_scale,
                        max_scale=max_scale,
                    )
                ):
                    continue

                candidate_anchor = None
                if enforce_search_crop:
                    candidate_anchor = self._sample_rgb_search_anchor_containing_tir(
                        bbox_rgb_np, bbox_tir_tmp
                    )
                    if candidate_anchor is None:
                        continue

                accepted = True
                H_final = H
                bbox_tir_new_np = bbox_tir_tmp
                accepted_anchor = candidate_anchor
                break

        if not accepted and enforce_search_crop:
            # Rare bounded-retry fallback.  Preserve branch identity but use a
            # neutral, observable state.  The crop still comes only from RGB;
            # TIR never determines its center or size.
            H_safe = self._build_homography_from_target_offset(
                bbox_rgb_np, bbox_tir_np, [0.0, 0.0, 1.0, 1.0]
            )
            bbox_safe, visible_ratio = self._warp_bbox_with_visibility(
                bbox_tir_np, H_safe, w, h
            )
            if bbox_safe is not None and visible_ratio >= 0.999:
                safe_anchor = self._sample_rgb_search_anchor_containing_tir(
                    bbox_rgb_np, bbox_safe
                )
                if safe_anchor is None:
                    # Deterministic RGB-centered crop; no TIR-guided fallback.
                    safe_anchor = torch.from_numpy(bbox_rgb_np.copy()).to(torch.float32)
                if self._bbox_inside_search_crop_from_anchor(safe_anchor, bbox_safe):
                    accepted = True
                    H_final = H_safe
                    bbox_tir_new_np = bbox_safe
                    accepted_anchor = safe_anchor

        if not accepted:
            natural = self._compute_bbox_offset_4param(bbox_rgb_np, bbox_tir_np)
            natural_gt = (
                torch.tensor(natural, dtype=torch.float32)
                if natural is not None
                else torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
            )
            return img, bbox_rgb, bbox_tir_scaled, natural_gt, None

        if img.ndim == 3 and img.shape[2] > 3:
            rgb_part = img[:, :, :3]
            tir_part = img[:, :, 3:].copy()
            tir_part_warp = cv2.warpAffine(
                tir_part, H_final[:2, :], (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            img_aug = np.concatenate([rgb_part, tir_part_warp], axis=2)
        else:
            img_aug = cv2.warpAffine(
                img, H_final[:2, :], (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )

        center_tir1, size_tir1 = self._bbox_to_center_size(bbox_tir_new_np)
        dx_eff = float(center_tir1[0] - center_rgb[0])
        dy_eff = float(center_tir1[1] - center_rgb[1])
        sx_eff = float(size_tir1[0] / (size_rgb[0] + 1e-6))
        sy_eff = float(size_tir1[1] / (size_rgb[1] + 1e-6))
        if isinstance(bbox_tir_scaled, torch.Tensor):
            bbox_tir_new = torch.from_numpy(bbox_tir_new_np).to(bbox_tir_scaled.device).type_as(bbox_tir_scaled)
        else:
            bbox_tir_new = bbox_tir_new_np
        align_gt = torch.tensor([dx_eff, dy_eff, sx_eff, sy_eff], dtype=torch.float32)
        return img_aug, bbox_rgb, bbox_tir_new, align_gt, accepted_anchor

    def _ensure_3ch(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return img
        if img.ndim == 2:
            img = img[:, :, None]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        return img

    def _fuse_rgb_tir_by_gt(
        self,
        rgb: np.ndarray,
        tir: np.ndarray,
        bbox_rgb,
        bbox_tir,
        border_value=0,
    ):
        """
        Align TIR to RGB using GT boxes, then return fused=[rgb, tir_aligned].

        bbox_rgb: RGB coords [x,y,w,h]
        bbox_tir: TIR coords [x,y,w,h]
        """
        rgb = self._ensure_3ch(rgb)
        tir = self._ensure_3ch(tir)

        H_rgb, W_rgb = rgb.shape[:2]
        H_tir, W_tir = tir.shape[:2]

        if isinstance(bbox_rgb, torch.Tensor):
            b_rgb = bbox_rgb.detach().cpu().numpy().astype(np.float32)
        else:
            b_rgb = np.asarray(bbox_rgb, dtype=np.float32)

        if isinstance(bbox_tir, torch.Tensor):
            b_tir = bbox_tir.detach().cpu().numpy().astype(np.float32)
        else:
            b_tir = np.asarray(bbox_tir, dtype=np.float32)

        # invalid -> simple resize+concat
        if b_rgb.shape[0] != 4 or b_tir.shape[0] != 4 or b_rgb[2] <= 0 or b_rgb[3] <= 0 or b_tir[2] <= 0 or b_tir[3] <= 0:
            tir_rs = cv2.resize(tir, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
            fused = np.concatenate([rgb, tir_rs], axis=2)
            align_gt = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
            b_tir_scaled = np.array([0, 0, 0, 0], dtype=np.float32)
            return fused, tir_rs, b_tir_scaled, align_gt

        # 1) resize TIR to RGB scale
        tir_rs = cv2.resize(tir, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)

        # 2) map TIR bbox to RGB scale
        sx0 = W_rgb / (W_tir + 1e-6)
        sy0 = H_rgb / (H_tir + 1e-6)
        b_tir_scaled = b_tir * np.array([sx0, sy0, sx0, sy0], dtype=np.float32)

        # 3) compute affine to align TIR -> RGB using box center/size
        c_rgb, s_rgb = self._bbox_to_center_size(b_rgb)
        c_ir, s_ir = self._bbox_to_center_size(b_tir_scaled)

        sx = float(s_rgb[0] / (s_ir[0] + 1e-6))
        sy = float(s_rgb[1] / (s_ir[1] + 1e-6))
        tx = float(c_rgb[0] - sx * c_ir[0])
        ty = float(c_rgb[1] - sy * c_ir[1])

        A = np.array([[sx, 0.0, tx],
                      [0.0, sy, ty]], dtype=np.float32)

        tir_aligned = cv2.warpAffine(
            tir_rs, A, (W_rgb, H_rgb),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value
        )

        fused = np.concatenate([rgb, tir_aligned], axis=2)

        # align_gt describes *pre-align* misalignment (TIR(mapped to RGB) relative to RGB)
        dx = float(c_ir[0] - c_rgb[0])
        dy = float(c_ir[1] - c_rgb[1])
        sx_rel = float(s_ir[0] / (s_rgb[0] + 1e-6))
        sy_rel = float(s_ir[1] / (s_rgb[1] + 1e-6))
        align_gt = torch.tensor([dx, dy, sx_rel, sy_rel], dtype=torch.float32)

        return fused, tir_aligned, b_tir_scaled, align_gt

    # ---------------------------
    # Dataset indexing
    # ---------------------------
    def _get_sequence_list(self, split):
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        if split == "train":
            file_path = os.path.join(ltr_path, "data_specs", "luart_training_list.txt")
        elif split == "test":
            file_path = os.path.join(ltr_path, "data_specs", "luart_testing_list.txt")
        elif split == "all":
            file_path = os.path.join(ltr_path, "data_specs", "luart_all_list.txt")
        else:
            # val: reuse train list unless you maintain a separate file
            file_path = os.path.join(ltr_path, "data_specs", "luart_training_list.txt")

        with open(file_path, "r") as f:
            dir_list = f.read().splitlines()
        return dir_list

    def _read_bb_anno(self, seq_path):
        rgb_bb_anno_file = os.path.join(seq_path, "visible.txt")
        ir_bb_anno_file = os.path.join(seq_path, "infrared.txt")
        try:
            rgb_gt = pandas.read_csv(
                rgb_bb_anno_file, delimiter=",", header=None,
                dtype=np.float32, na_filter=False, low_memory=False
            ).values
            ir_gt = pandas.read_csv(
                ir_bb_anno_file, delimiter=",", header=None,
                dtype=np.float32, na_filter=False, low_memory=False
            ).values
        except Exception:
            print(seq_path)
            raise
        return torch.tensor(rgb_gt), torch.tensor(ir_gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox_rgb, bbox_ir = self._read_bb_anno(seq_path)
        valid = (bbox_rgb[:, 2] > 0) & (bbox_rgb[:, 3] > 0) & (bbox_ir[:, 2] > 0) & (bbox_ir[:, 3] > 0)
        visible = valid.clone().byte()
        return {"bbox_rgb": bbox_rgb, "bbox_ir": bbox_ir, "valid": valid, "visible": visible}

    def _get_frame_path(self, seq_path, frame_id):
        rgb_frame_path = sorted(os.listdir(seq_path + "/NotAlign/visible"))
        ir_frame_path = sorted(os.listdir(seq_path + "/NotAlign/infrared"))
        rgb_pre = seq_path + "/NotAlign/visible"
        ir_pre = seq_path + "/NotAlign/infrared"
        return os.path.join(rgb_pre, rgb_frame_path[frame_id]), os.path.join(ir_pre, ir_frame_path[frame_id])

    def _get_frame(self, seq_path, frame_id):
        self._last_ir_aligned_in_frame = False
        rgb_frame_path, ir_frame_path = self._get_frame_path(seq_path, frame_id)
        img_rgb_size, img_tir_size = get_x_frame(rgb_frame_path, ir_frame_path, dtype=self.dtype)
        return img_rgb_size, img_tir_size
    
    # ---------------------------
    # Main API
    # ---------------------------
    def get_frames(self, seq_id, frame_ids, anno=None):
        """
        Returns:
          frame_list_rgb_size: [N] fused frames on RGB scale.
          frame_list_tir_size: [N] fused frames on TIR scale (or raw depending on get_x_frame* impl).
          anno_frames: dict of lists aligned with frame_ids.

        Keys added/normalized:
          - bbox_ir_scaled: TIR bbox mapped to RGB coordinate system (per frame).
          - align_gt_4param: [dx,dy,sx,sy] (TIR(mapped to RGB) relative to RGB).
            * mode="aligned": this is the *pre-align* misalignment (computed from GT).
            * mode="random_offset" and aug enabled: this is the *post-aug* misalignment.
            * mode="no_sample": no alignment or random-offset sampling is applied.
          - bbox_ir_aligned: (mode="aligned" only) set to bbox_rgb (since we output aligned fused).
        """
        seq_path = self._get_sequence_path(seq_id)

        frame_list_rgb_size = []
        frame_list_tir_size = []
        ir_aligned_flags = []

        for f_id in frame_ids:
            frame_rgb, frame_tir = self._get_frame(seq_path, f_id)
            frame_list_rgb_size.append(frame_rgb)
            frame_list_tir_size.append(frame_tir)
            ir_aligned_flags.append(bool(self._last_ir_aligned_in_frame))

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {k: [v[f_id, ...].clone() for f_id in frame_ids] for k, v in anno.items()}

        # infer sizes
        RGB_H, RGB_W = frame_list_rgb_size[0].shape[0], frame_list_rgb_size[0].shape[1]
        TIR_H, TIR_W = frame_list_tir_size[0].shape[0], frame_list_tir_size[0].shape[1]

        # bbox_ir_scaled (always)
        anno_frames["bbox_ir_scaled"] = [
            self._get_scaled_bbox_ir(bbox_ir, RGB_W, RGB_H, TIR_W, TIR_H)
            for bbox_ir in anno_frames["bbox_ir"]
        ]

        # natural misalignment supervision (always computable)
        align_gt_list = [
            self._calc_align_gt_4param(anno_frames["bbox_rgb"][i], anno_frames["bbox_ir_scaled"][i])
            for i in range(len(frame_ids))
        ]

        # --- mode=aligned: produce aligned fused frames ---
        if self.mode == "aligned":
            fused_list = []
            bbox_ir_scaled_list = []
            align_gt_out = []

            for i in range(len(frame_ids)):
                rgb_rgbscale = frame_list_rgb_size[i]
                tir_tirscale = frame_list_tir_size[i]
                fused, tir_aligned, b_tir_scaled_np, align_gt = self._fuse_rgb_tir_by_gt(
                    rgb=rgb_rgbscale[:, :, :3],
                    tir=tir_tirscale[:, :, 3:] if (tir_tirscale.ndim == 3 and tir_tirscale.shape[2] > 3) else tir_tirscale,
                    bbox_rgb=anno_frames["bbox_rgb"][i],
                    bbox_tir=anno_frames["bbox_ir"][i],
                    border_value=0
                )
                fused_list.append(fused)
                bbox_ir_scaled_list.append(torch.from_numpy(b_tir_scaled_np).to(torch.float32))
                align_gt_out.append(align_gt)

            frame_list_rgb_size = fused_list
            anno_frames["bbox_ir_scaled"] = bbox_ir_scaled_list
            anno_frames["bbox_ir_aligned"] = [b.clone() for b in anno_frames["bbox_rgb"]]
            anno_frames["align_gt_4param"] = align_gt_out
        # --- mode=random_offset: dataset-level LUART state sampling ---
        elif self.mode == "random_offset":
            if self.using_random_align_sample:
                align_gt_aug = []
                search_anchor_aug = []

                frame_length = int(anno["bbox_rgb"].shape[0]) if anno is not None and "bbox_rgb" in anno else 0
                for i, f_id in enumerate(frame_ids):
                    img_fused = frame_list_rgb_size[i]
                    bbox_rgb = anno_frames["bbox_rgb"][i]
                    bbox_tir_scaled = anno_frames["bbox_ir_scaled"][i]

                    policy = np.random.rand()
                    history_transform = None
                    history_anchor = None
                    use_random_offset = False

                    if policy < self.history_offset_prob:
                        history_transform, history_anchor = self._compute_history_prealign_homography(
                            f_id, anno, frame_length, RGB_W, RGB_H, TIR_W, TIR_H,
                            current_bbox_rgb=bbox_rgb,
                            current_bbox_tir_scaled=bbox_tir_scaled,
                        )
                        if history_transform is None and self.split == "train":
                            # Rare first/edge-frame fallback.  Keep this sample in
                            # the History branch and use a neutral observable state;
                            # never convert it into the 10% Random branch.
                            history_transform = self._build_homography_from_target_offset(
                                bbox_rgb, bbox_tir_scaled, [0.0, 0.0, 1.0, 1.0]
                            )
                    elif policy < self.history_offset_prob + self.random_offset_prob:
                        use_random_offset = True

                    if history_transform is not None or use_random_offset:
                        img_aug, bbox_rgb_aug, bbox_tir_scaled_aug, align_gt, search_anchor = (
                            self._apply_random_rgb_tir_offset_4param(
                                img=img_fused,
                                bbox_rgb=bbox_rgb,
                                bbox_tir_scaled=bbox_tir_scaled,
                                max_trans=self.max_trans,
                                max_scale=self.max_scale,
                                min_scale=self.min_scale,
                                history_transform=history_transform,
                                enforce_search_crop=(self.split == "train"),
                                search_anchor=history_anchor,
                            )
                        )
                        frame_list_rgb_size[i] = img_aug
                        anno_frames["bbox_rgb"][i] = bbox_rgb_aug
                        anno_frames["bbox_ir_scaled"][i] = bbox_tir_scaled_aug
                    else:
                        # Only reachable when configured branch probabilities sum
                        # to <1.  Preserve natural geometry and normal RGB jitter.
                        align_gt = align_gt_list[i]
                        search_anchor = None
                        if self.split == "train":
                            search_anchor = self._sample_rgb_search_anchor_containing_tir(
                                bbox_rgb, bbox_tir_scaled
                            )
                            if search_anchor is None:
                                # Keep observability without a Processing-level
                                # rejection.  Crop generation remains RGB-only.
                                H_safe = self._build_homography_from_target_offset(
                                    bbox_rgb, bbox_tir_scaled, [0.0, 0.0, 1.0, 1.0]
                                )
                                img_aug, bbox_rgb_aug, bbox_tir_scaled_aug, align_gt, search_anchor = (
                                    self._apply_random_rgb_tir_offset_4param(
                                        img=img_fused,
                                        bbox_rgb=bbox_rgb,
                                        bbox_tir_scaled=bbox_tir_scaled,
                                        max_trans=self.max_trans,
                                        max_scale=self.max_scale,
                                        min_scale=self.min_scale,
                                        history_transform=H_safe,
                                        enforce_search_crop=True,
                                    )
                                )
                                frame_list_rgb_size[i] = img_aug
                                anno_frames["bbox_rgb"][i] = bbox_rgb_aug
                                anno_frames["bbox_ir_scaled"][i] = bbox_tir_scaled_aug

                    align_gt_aug.append(align_gt)
                    if self.split == "train":
                        if search_anchor is None:
                            # Defensive fallback only.  RGB bbox is itself the
                            # deterministic no-jitter RGB anchor.
                            search_anchor = torch.as_tensor(anno_frames["bbox_rgb"][i]).clone().to(torch.float32)
                        search_anchor_aug.append(search_anchor)

                anno_frames["align_gt_4param"] = align_gt_aug
                if self.split == "train":
                    # Processing consumes these anchors directly.  It performs
                    # no TIR-visibility rejection and does not resample them.
                    anno_frames["search_jit_anno_rgb"] = search_anchor_aug
            else:
                anno_frames["align_gt_4param"] = align_gt_list

        # --- mode=no_sample: no alignment and no random-offset sampling ---
        else:
            anno_frames["align_gt_4param"] = align_gt_list

        object_meta = OrderedDict({
            "object_class_name": None,
            "motion_class": None,
            "major_class": None,
            "root_class": None,
            "motion_adverb": None
        })

        seq_name = seq_path.split("/")[-1]
        return frame_list_rgb_size, frame_list_tir_size, anno_frames, object_meta, seq_name
