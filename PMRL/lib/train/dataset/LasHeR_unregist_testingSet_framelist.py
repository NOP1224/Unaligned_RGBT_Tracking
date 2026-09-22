import torch
import os
import os.path
import numpy as np
import pandas
import random
from collections import OrderedDict

from lib.train.data.image_loader import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from lib.train.admin.environment import env_settings
from lib.train.dataset.depth_utils import get_x_frame, get_x_framev2

import random
import cv2


def sample_decay_gap(low, high, scale):
    """Sample a smaller-gap-biased integer from the closed interval."""
    low = int(low)
    high = int(high)
    if high <= low:
        return low
    width = high - low
    u = np.random.rand()
    x = -float(scale) * np.log(1.0 - u * (1.0 - np.exp(-width / float(scale))))
    return int(np.clip(low + int(round(x)), low, high))

class LasHeR_unregist_testingSet_framelist(BaseVideoDataset):
    def __init__(self, root=None, dtype='rgbrgb', image_loader=jpeg4py_loader, data_fraction=None, framelist=1, pre_align=False):
        
        self.root = env_settings().LasHeR_unregist_dir if root is None else root
        super().__init__('LasHeR_unregist_testingSet_framelist', root, image_loader)

        # video_name for each sequence
        self.sequence_list = ['blkgirlumbrella', 'blkmoto2north', 'catbrownback2bush', 'right5thflag', 'girlinrain', 'foamatgirl`srighthand', 'whitegirlinlight', 'boywalkinginsnow2', 'midgreyboyrunningcoming', 'carlightcome2', 'redroadlatboy', 'darkcarturn', 'boy2treesfindbike', 'right2ndflagformath', 'blkboytakesumbrella', 'girlafterglassdoor', 'blkboydown', "leftgirl'swhitebag", 'whiteboyrightcoccergoal', 'waitresscoming', 'bottlebetweenboy`sfeet', 'bikeboyintodark', 'blkboylefttheNo_21', 'turning1strowleft2ndboy', 'bike', 'blktribikecome', 'midboyNo_9', 'ab_pingpongball2', 'firstexercisebook', 'bikeboywithumbrella', 'guardunderthecolumn', 'whitesuvturn', 'redetricycle', 'boyinplatform', '11leftboy', '1stcol4thboy', 'bikeinrain', 'blkboyback', 'girldownstairfromlight', 'motogoesaloongS', 'whiteskirtgirlcomingfromgoal', 'blkstandboy', 'rightboy504', 'carbehindtrees', 'carturn117', 'girl`sblkbag', '7rightorangegirl', '3rdfatboy', 'girlfromlight_quezhen', '4thboywithwhite', 'advancedredcup', 'carcomingfromlight', 'mototurneast', 'boydownplatform', 'whitebikebelow', 'lastleftgirl', 'bike2left', 'shinybikeboy2left', 'leftmirrorlikesky', 'midof3girls', 'whiteridingbike', 'bikeboy', 'pinkwithblktopcup', 'umbreboyoncall', 'leftboyoutofthetroop', 'rightcomingstrongboy', 'blkboybetweenredandwhite', 'blueboy', 'bluegirlbiketurn', 'blackboy', 'blkboy`shead', 'boyride2path', 'boyunder2baskets', 'girlrightthewautress', 'leftpingpongball', 'boyshead9684', 'drillmasterfollowingatright', 'broom', 'manfromtoilet', 'littelbabycryingforahug', 'umbrellawillopen', 'redboygoright', 'ab_bolstershaking', 'basketball849', 'small-gai', 'bawgirl', 'boyaroundtrees', 'lefthyalinepaperfrontpants', 'foldedfolderatlefthand', 'darktreesboy', 'boy`headwithouthat', 'leftmirror', 'rightblkfatboyleftwhite', 'boyfromdark', 'carcominginlight', 'whiteofboys', 'hyalinepaperfrontface', 'boyatdoorturnright', 'raincarturn', 'girllongskirt', 'lefterbike', '11runtwo', 'whitecarturnleft', 'midblkgirl', 'whitegirltakingchopsticks', '1strowleftboyturning', 'lowerfoamboard', 'boytakingbasketballfollowing', 'boylefttheNo_9boy', 'rightgirltakingcup', 'rightbottlecomes', 'leftchair', 'minibusgoes2left', 'blkhairgirltakingblkbag', 'swan_0109', 'boyscomeleft', 'boyoncall', 'girlof2leaders', '3rdgrouplastboy', 'basketballathand', 'redmidboy', 'rightdarksingleman', 'leftexcersicebookyellow', 'bike2trees', 'carcomeonlight', 'ab_girlchoosesbike', 'leftrushingboy', 'middrillmaster', 'redcarcominginlight', 'carwillturn', 'pingpingpad3', 'mangetsoff', 'rightbluewhite', 'girlunderthestreetlamp', '3pinkleft', 'leftblkTboy', 'rightbike', 'bikegoindark', 'leftuphand', '1strowrightgirl3540', 'boyleftblkrunning2crowd', 'leftboy2jointhe4', 'ab_blkskirtgirl', 'shinycarcoming2', 'rightbike-gai', 'rightblkboystand', 'biketurnright', 'leftfarboycomingpicktheball', 'boytakingplate2left', 'boyinsnowfield3', 'leftopenexersicebook', 'midrunboywithwhite', 'standblkboy', 'rightcameraman', 'motowithbluetop', '10runone', 'lefthyalinepaper2rgb', '1strowrightdrillmaster', 'whiterunningboy', 'catbrown2', 'leftunderbasket', 'motocomeonlight', 'mototaking2boys306', 'ab_rightlowerredcup_quezhen', 'umbrellawillbefold', 'bikefromlight', 'moto', 'drillmaster1117', 'mandownstair', 'redtricycle', 'darkouterwhiteboy', 'shinycarcoming', 'boyruninsnow', 'leftmirrorside', 'whitecarturn683', 'blkboystand', '2runseven', 'blkboyhead', 'boy2buildings', 'leftbottle2hang', 'ballshootatthebasket3times', 'besom3', 'rainycarcome_ab', 'bluebuscoming', '1boycoming', 'midredboy', 'AQgirlwalkinrain', 'blackboyoncall', 'womanback2car', 'boy`sheadingreycol', 'blueboy421', 'carlight2', 'boy2basketballground', 'rightcar-chongT', 'boyinlight', 'runningcameragirl', '1blackteacher', 'belowdarkgirl', 'bikeboyturntimes', 'ab_whiteboywithbluebag', 'boy2trees', 'blkcaratfrontbluebus', 'rightwaiter1_quezhen', 'whitecarcomeinrain', 'large']
        # self.sequence_list=os.listdir(self.root)[700:]
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))
        self.framelist = framelist
        self.margin = 150
        self.dtype = dtype
        self.pre_align = pre_align
        # Match LUART validation/test: history-derived pre-alignment only.
        # History offsets are real annotation states and are not constrained by
        # synthetic translation/scale bounds.
        self.max_trans = 2.0
        self.max_scale = 1.2
        self.min_scale = 0.8
        self.random_offset_prob = 0.0
        self.history_offset_prob = 1.0
        self.history_min_gap = 1
        self.history_max_gap = 75
        self.history_gap_mu = 20.0
        self.history_gap_sigma = 12.0
        self.history_recent_prob = 0.70
        self.history_delay_prob = 0.25
        self.history_recent_scale = 8.0
        self.history_stale_min_gap = 40
        self.history_min_visible_ratio = 0.25
        self.history_min_bbox_size = 2.0
        self.failure_small_prob = 0.60
        self.failure_mid_prob = 0.30
        self.failure_small_gamma_scale = 6.0
        self.failure_mid_gamma_scale = 16.0
        self.failure_severe_min_ratio = 0.45
        self.failure_log_scale_small = 0.025
        self.failure_log_scale_mid = 0.055
        self.using_history_align_sample = True
    def get_name(self):
        return 'LasHeR_unregist_testingSet_framelist'

    def _read_bb_anno_v(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'visible.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)
    
    def _read_bb_anno_i(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'infrared.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, seq_name)
        bbox_v = self._read_bb_anno_v(seq_path)
        bbox_i = self._read_bb_anno_i(seq_path)
        valid = (bbox_v[:, 2] < 1000) & (bbox_v[:, 2] > 10) & \
                (bbox_v[:, 3] < 1000) & (bbox_v[:, 3] > 10) & \
                (bbox_v[:, 0] < 1000) & (bbox_v[:, 0] > 0)  & \
                (bbox_v[:, 1] < 1000) & (bbox_v[:, 1] > 0)  & \
                (bbox_i[:, 2] < 1000) & (bbox_i[:, 2] > 10) & \
                (bbox_i[:, 3] < 1000) & (bbox_i[:, 3] > 10) & \
                (bbox_i[:, 0] < 1000) & (bbox_i[:, 0] > 0)  & \
                (bbox_i[:, 1] < 1000) & (bbox_i[:, 1] > 0)
        visible = valid.clone().byte()
        return {'bbox_rgb': bbox_v, 'bbox_ir': bbox_i, 'valid': valid, 'visible': visible}
        # return {'bbox_v': bbox_v, 'bbox_i': bbox_i, 'valid': valid, 'visible': visible}
    def get_num_sequences(self):
        return len(self.sequence_list)
    
    def _get_frame_v(self, seq_path, frame_id):
        frame_path_v = os.path.join(seq_path, 'visible', sorted([p for p in os.listdir(os.path.join(seq_path, 'visible')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return frame_path_v
        # return self.image_loader(frame_path_v)
        
    def _get_frame_i(self, seq_path, frame_id):
        frame_path_i = os.path.join(seq_path, 'infrared', sorted([p for p in os.listdir(os.path.join(seq_path, 'infrared')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return frame_path_i
        # return self.image_loader(frame_path_i)

    def _get_frame(self, seq_path, frame_id):
        rgb_frame_path = self._get_frame_v(seq_path, frame_id)
        ir_frame_path = self._get_frame_i(seq_path, frame_id)
        
        img_rgb_size,img_tir_size = get_x_frame(rgb_frame_path, ir_frame_path, dtype=self.dtype)
        
        return img_rgb_size, img_tir_size
    

    def _bbox_to_numpy(self, bbox):
        if isinstance(bbox, torch.Tensor):
            return bbox.detach().cpu().numpy().astype(np.float32)
        return np.asarray(bbox, dtype=np.float32)

    def _bbox_to_center_size(self, bbox_np):
        """bbox: numpy [x,y,w,h] -> center(x,y), size(w,h)."""
        x, y, w, h = bbox_np
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        return np.array([cx, cy], dtype=np.float32), np.array([w, h], dtype=np.float32)

    def _build_4param_homography(self, dx, dy, sx, sy):
        """x' = sx*x + dx, y' = sy*y + dy, all in RGB-scale pixels."""
        return np.array([[sx, 0.0, dx],
                         [0.0, sy, dy],
                         [0.0, 0.0, 1.0]], dtype=np.float32)

    def _build_homography_from_target_offset(self, bbox_rgb, bbox_tir, target_offset):
        """
        Build an affine transform that moves the current TIR bbox to the target
        offset relative to the current RGB bbox.

        target_offset = [dx, dy, sx, sy], where dx/dy are center offsets and
        sx/sy are target TIR/RGB size ratios in RGB-scale coordinates.
        """
        bbox_rgb_np = self._bbox_to_numpy(bbox_rgb)
        bbox_tir_np = self._bbox_to_numpy(bbox_tir)
        target_offset = self._offset_to_numpy(target_offset)

        center_rgb, size_rgb = self._bbox_to_center_size(bbox_rgb_np)
        center_tir, size_tir = self._bbox_to_center_size(bbox_tir_np)
        dx, dy, sx, sy = target_offset.tolist()

        target_size = np.maximum(size_rgb * np.array([sx, sy], dtype=np.float32), 1.0)
        target_center = center_rgb + np.array([dx, dy], dtype=np.float32)

        warp_sx = target_size[0] / max(size_tir[0], 1e-6)
        warp_sy = target_size[1] / max(size_tir[1], 1e-6)
        warp_dx = target_center[0] - warp_sx * center_tir[0]
        warp_dy = target_center[1] - warp_sy * center_tir[1]
        return self._build_4param_homography(warp_dx, warp_dy, warp_sx, warp_sy)

    def _warp_bbox_only(self, bbox, H, img_w, img_h):
        """Warp only bbox with homography H and clip it to image size."""
        bbox = np.asarray(bbox, dtype=np.float32)
        x, y, w, h = bbox
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        corners_h = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1).T
        warped = H @ corners_h
        warped = (warped[:2, :] / (warped[2:, :] + 1e-8)).T
        warped[:, 0] = np.clip(warped[:, 0], 0, img_w - 1)
        warped[:, 1] = np.clip(warped[:, 1], 0, img_h - 1)
        x_min, y_min = warped.min(axis=0)
        x_max, y_max = warped.max(axis=0)
        return np.array([x_min, y_min, max(1.0, x_max - x_min), max(1.0, y_max - y_min)], dtype=np.float32)

    def _scale_tir_bbox_to_rgb(self, bbox_tir, w_rgb, h_rgb, w_tir, h_tir):
        """Scale TIR bbox from original TIR image coordinates to RGB image coordinates."""
        if isinstance(bbox_tir, torch.Tensor):
            scale = torch.tensor([w_rgb / w_tir, h_rgb / h_tir, w_rgb / w_tir, h_rgb / h_tir],
                                 dtype=bbox_tir.dtype, device=bbox_tir.device)
            return bbox_tir * scale
        bbox_tir = np.asarray(bbox_tir, dtype=np.float32)
        return bbox_tir * np.array([w_rgb / w_tir, h_rgb / h_tir, w_rgb / w_tir, h_rgb / h_tir], dtype=np.float32)

    def _offset_to_numpy(self, offset):
        if offset is None:
            return None
        if isinstance(offset, torch.Tensor):
            offset = offset.detach().cpu().numpy()
        offset = np.asarray(offset, dtype=np.float32).reshape(-1)
        if offset.shape[0] != 4 or not np.all(np.isfinite(offset)):
            return None
        return offset

    def _identity_offset(self):
        return torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)

    def _compute_bbox_offset_4param(self, bbox_rgb, bbox_tir_rgb):
        """Compute [dx, dy, sx, sy] from RGB bbox and RGB-scaled TIR bbox."""
        bbox_rgb_np = self._bbox_to_numpy(bbox_rgb)
        bbox_tir_np = self._bbox_to_numpy(bbox_tir_rgb)
        center_rgb, size_rgb = self._bbox_to_center_size(bbox_rgb_np)
        center_tir, size_tir = self._bbox_to_center_size(bbox_tir_np)
        if np.any(size_rgb <= 1e-6) or np.any(size_tir <= 1e-6):
            return None
        offset = np.array([
            center_tir[0] - center_rgb[0],
            center_tir[1] - center_rgb[1],
            size_tir[0] / (size_rgb[0] + 1e-6),
            size_tir[1] / (size_rgb[1] + 1e-6),
        ], dtype=np.float32)
        return offset if np.all(np.isfinite(offset)) else None

    def _bbox_geometric_size(self, bbox, default=1.0):
        bbox_np = self._bbox_to_numpy(bbox)
        if bbox_np.shape[0] != 4 or not np.all(np.isfinite(bbox_np)):
            return float(default)
        w, h = float(bbox_np[2]), float(bbox_np[3])
        if w <= 0.0 or h <= 0.0:
            return float(default)
        return float(np.sqrt(max(w * h, 1e-6)))

    def _resolve_max_trans_px(self, max_trans_scale=None, bbox_ref=None, default_px=1.0):
        scale = self.max_trans if max_trans_scale is None else float(max_trans_scale)
        if bbox_ref is None:
            return max(float(default_px), scale)
        return max(float(default_px), scale * self._bbox_geometric_size(bbox_ref, default_px))

    def _is_valid_offset_4param(self, offset, max_trans=None, min_scale=None, max_scale=None, bbox_ref=None):
        offset_np = self._offset_to_numpy(offset)
        if offset_np is None:
            return False
        dx, dy, sx, sy = offset_np.tolist()
        if max_trans is not None:
            max_trans_px = (
                self._resolve_max_trans_px(max_trans, bbox_ref=bbox_ref)
                if bbox_ref is not None else float(max_trans)
            )
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

    def _compute_history_prealign_homography(
        self,
        cur_frame_id,
        anno,
        frame_length,
        w_rgb,
        h_rgb,
        w_tir,
        h_tir,
        current_bbox_tir_scaled,
        max_retry=30,
    ):
        """Return the historical TIR->RGB transform used by online step alignment.

        Let ``G_t`` map current RGB coordinates to current TIR coordinates and
        ``G_h`` be the equivalent transform at a past frame. Online inference
        first applies ``G_h^{-1}`` to the current TIR image, so the network sees
        the residual geometry ``G_h^{-1} G_t``. This function returns exactly
        ``G_h^{-1}``; it never filters the historical offset by synthetic
        translation or scale bounds.
        """
        safe_frame_length = self._resolve_history_frame_length(anno, frame_length)
        cur_frame_id = int(cur_frame_id)
        if safe_frame_length <= 1 or not (0 <= cur_frame_id < safe_frame_length):
            return None

        min_visible = float(getattr(self, "history_min_visible_ratio", 0.25))
        min_size = float(getattr(self, "history_min_bbox_size", 2.0))
        for _ in range(max_retry):
            hist_id = self._sample_gaussian_history_frame_id(
                cur_frame_id,
                safe_frame_length,
                min_gap=self.history_min_gap,
                max_gap=self.history_max_gap,
                mu=self.history_gap_mu,
                sigma=self.history_gap_sigma,
            )
            if hist_id is None:
                return None
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

            warped_bbox, visible_ratio = self._warp_bbox_with_visibility(
                current_bbox_tir_scaled, H_hist, w_rgb, h_rgb
            )
            if (
                warped_bbox is not None
                and visible_ratio >= min_visible
                and warped_bbox[2] >= min_size
                and warped_bbox[3] >= min_size
            ):
                return H_hist
        return None

    def _sample_failure_offset_4param(self, max_trans, min_scale, max_scale):
        max_trans = float(max_trans)
        min_scale = float(min_scale)
        max_scale = float(max_scale)

        small_prob = float(np.clip(self.failure_small_prob, 0.0, 1.0))
        mid_prob = float(np.clip(self.failure_mid_prob, 0.0, 1.0 - small_prob))
        u = np.random.rand()
        if u < small_prob:
            radius = np.random.gamma(2.0, max(self.failure_small_gamma_scale, 1e-6))
        elif u < small_prob + mid_prob:
            radius = np.random.gamma(2.0, max(self.failure_mid_gamma_scale, 1e-6))
        else:
            radius = np.random.uniform(self.failure_severe_min_ratio * max_trans, max_trans)
        radius = float(np.clip(radius, 0.0, max_trans))
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        dx_raw = float(radius * np.cos(theta))
        dy_raw = float(radius * np.sin(theta))

        max_log_scale = max(
            abs(np.log(max(max_scale, 1e-6))),
            abs(np.log(max(min_scale, 1e-6))),
        )
        if u < small_prob:
            log_sx = np.random.laplace(0.0, self.failure_log_scale_small)
            log_sy = np.random.laplace(0.0, self.failure_log_scale_small)
        elif u < small_prob + mid_prob:
            log_sx = np.random.laplace(0.0, self.failure_log_scale_mid)
            log_sy = np.random.laplace(0.0, self.failure_log_scale_mid)
        else:
            log_sx = np.random.uniform(-max_log_scale, max_log_scale)
            log_sy = np.random.uniform(-max_log_scale, max_log_scale)

        sx_raw = float(np.clip(np.exp(log_sx), min_scale, max_scale))
        sy_raw = float(np.clip(np.exp(log_sy), min_scale, max_scale))
        return dx_raw, dy_raw, sx_raw, sy_raw

    def _warp_tir_channels(self, img, H):
        """Apply affine transform to TIR channels only when RGB+TIR are concatenated."""
        h, w = img.shape[:2]
        A = H[:2, :]
        if img.ndim == 3 and img.shape[2] > 3:
            rgb_part = img[:, :, :3]
            tir_part = img[:, :, 3:].copy()
            tir_part_warp = cv2.warpAffine(
                tir_part, A, (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            return np.concatenate([rgb_part, tir_part_warp], axis=2)
        return cv2.warpAffine(
            img, A, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

    def _apply_random_rgb_tir_offset_4param(
        self,
        img,
        bbox_rgb,
        bbox_tir,
        max_trans=None,
        min_scale=0.8,
        max_scale=1.2,
        max_retry=1000,
        history_transform=None,
    ):
        """Warp TIR and return the effective RGB->prealigned-TIR box geometry."""
        h, w = img.shape[:2]
        bbox_rgb_np = self._bbox_to_numpy(bbox_rgb)
        bbox_tir_np = self._bbox_to_numpy(bbox_tir)
        if (
            bbox_rgb_np.shape[0] != 4 or bbox_tir_np.shape[0] != 4
            or bbox_rgb_np[2] <= 0 or bbox_rgb_np[3] <= 0
            or bbox_tir_np[2] <= 0 or bbox_tir_np[3] <= 0
        ):
            return img, bbox_rgb, bbox_tir, self._identity_offset()

        center_rgb, size_rgb = self._bbox_to_center_size(bbox_rgb_np)
        max_trans_px = self._resolve_max_trans_px(max_trans, bbox_ref=bbox_rgb_np)
        min_visible = float(getattr(self, "history_min_visible_ratio", 0.25))
        min_size = float(getattr(self, "history_min_bbox_size", 2.0))
        accepted = False
        H_final = np.eye(3, dtype=np.float32)
        bbox_tir_new_np = bbox_tir_np.copy()

        history_requested = history_transform is not None
        if history_requested:
            H = np.asarray(history_transform, dtype=np.float32)
            bbox_tir_tmp, visible_ratio = self._warp_bbox_with_visibility(bbox_tir_np, H, w, h)
            if (
                H.shape == (3, 3) and np.all(np.isfinite(H))
                and bbox_tir_tmp is not None and visible_ratio >= min_visible
                and bbox_tir_tmp[2] >= min_size and bbox_tir_tmp[3] >= min_size
            ):
                accepted = True
                H_final = H
                bbox_tir_new_np = bbox_tir_tmp
        else:
            for _ in range(max_retry):
                dx_raw, dy_raw, sx_raw, sy_raw = self._sample_failure_offset_4param(
                    max_trans_px, min_scale, max_scale
                )
                H = self._build_4param_homography(dx_raw, dy_raw, sx_raw, sy_raw)
                bbox_tir_tmp, visible_ratio = self._warp_bbox_with_visibility(bbox_tir_np, H, w, h)
                if bbox_tir_tmp is None or visible_ratio < min_visible:
                    continue
                center_tir1, size_tir1 = self._bbox_to_center_size(bbox_tir_tmp)
                dx_tmp = center_tir1[0] - center_rgb[0]
                dy_tmp = center_tir1[1] - center_rgb[1]
                sx_tmp = size_tir1[0] / (size_rgb[0] + 1e-6)
                sy_tmp = size_tir1[1] / (size_rgb[1] + 1e-6)
                if (
                    bbox_tir_tmp[2] >= min_size and bbox_tir_tmp[3] >= min_size
                    and self._is_valid_offset_4param(
                        [dx_tmp, dy_tmp, sx_tmp, sy_tmp],
                        max_trans=max_trans_px,
                        min_scale=min_scale,
                        max_scale=max_scale,
                    )
                ):
                    accepted = True
                    H_final = H
                    bbox_tir_new_np = bbox_tir_tmp
                    break

        # A rejected history sample must not silently become a synthetic failure.
        # Keep the natural current geometry instead.
        if not accepted:
            natural = self._compute_bbox_offset_4param(bbox_rgb_np, bbox_tir_np)
            natural_gt = (
                torch.tensor(natural, dtype=torch.float32)
                if natural is not None else self._identity_offset()
            )
            return img, bbox_rgb, bbox_tir, natural_gt

        img_aug = self._warp_tir_channels(img, H_final)
        center_tir1, size_tir1 = self._bbox_to_center_size(bbox_tir_new_np)
        dx_eff = float(center_tir1[0] - center_rgb[0])
        dy_eff = float(center_tir1[1] - center_rgb[1])
        sx_eff = float(size_tir1[0] / (size_rgb[0] + 1e-6))
        sy_eff = float(size_tir1[1] / (size_rgb[1] + 1e-6))

        if isinstance(bbox_tir, torch.Tensor):
            bbox_tir_new = torch.from_numpy(bbox_tir_new_np).to(bbox_tir.device).type_as(bbox_tir)
        else:
            bbox_tir_new = bbox_tir_new_np
        align_gt = torch.tensor([dx_eff, dy_eff, sx_eff, sy_eff], dtype=torch.float32)
        return img_aug, bbox_rgb, bbox_tir_new, align_gt

    def _get_scaled_bbox_ir(self, bbox_ir, w_rgb, h_rgb, w_tir, h_tir):
        # bbox_ir: tensor([x, y, w, h]) in TIR coords
        scale = torch.tensor(
            [w_rgb / w_tir,  # 对 x 缩放
            h_rgb / h_tir,  # 对 y 缩放
            w_rgb / w_tir,  # 对 w 缩放
            h_rgb / h_tir], # 对 h 缩放
            dtype=bbox_ir.dtype,
            device=bbox_ir.device
        )
        bbox_ir_bias = bbox_ir * scale   # 逐元素相乘，结果仍然是 tensor([x', y', w', h'])
        return bbox_ir_bias
    
    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, seq_name)
        frame_length = len(os.listdir(os.path.join(seq_path, "visible")))
        frame_list_rgb_size = []
        frame_list_tir_size = []

        for f_id in frame_ids:
            frame_rgb, frame_tir = self._get_frame(seq_path, f_id)
            frame_list_rgb_size.append(frame_rgb)
            frame_list_tir_size.append(frame_tir)

        # 2. 读标注
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        # 尺度：RGB 尺度用来计算 offset & bbox_ir_scaled
        RGB_H, RGB_W = frame_list_rgb_size[0].shape[0], frame_list_rgb_size[0].shape[1]
        TIR_H, TIR_W = frame_list_tir_size[0].shape[0], frame_list_tir_size[0].shape[1]

        # 3. 把 TIR bbox 映射到 RGB 尺度（逐帧）
        anno_frames['bbox_ir_scaled'] = [
            self._get_scaled_bbox_ir(bbox_ir, RGB_W, RGB_H, TIR_W, TIR_H)
            for bbox_ir in anno_frames['bbox_ir']
        ]

        # Same source policy as LUART. History offsets are annotation-derived and
        # bypass all synthetic translation/scale bounds.
        align_gt_list = []
        for i, f_id in enumerate(frame_ids):
            img_fused = frame_list_rgb_size[i]
            bbox_rgb = anno_frames['bbox_rgb'][i]
            bbox_tir_rgb = anno_frames['bbox_ir_scaled'][i]
            natural_offset = self._compute_bbox_offset_4param(bbox_rgb, bbox_tir_rgb)
            natural_align_gt = (
                torch.tensor(natural_offset, dtype=torch.float32)
                if natural_offset is not None else self._identity_offset()
            )

            policy = np.random.rand()
            history_transform = None
            use_random_offset = False
            if policy < self.history_offset_prob:
                history_transform = self._compute_history_prealign_homography(
                    f_id, anno, frame_length, RGB_W, RGB_H, TIR_W, TIR_H,
                    current_bbox_tir_scaled=bbox_tir_rgb,
                )
            elif policy < self.history_offset_prob + self.random_offset_prob:
                use_random_offset = True

            if history_transform is not None or use_random_offset:
                img_aug, bbox_rgb_aug, bbox_tir_rgb_aug, align_gt = (
                    self._apply_random_rgb_tir_offset_4param(
                        img=img_fused,
                        bbox_rgb=bbox_rgb,
                        bbox_tir=bbox_tir_rgb,
                        max_trans=self.max_trans,
                        min_scale=self.min_scale,
                        max_scale=self.max_scale,
                        history_transform=history_transform,
                    )
                )
                frame_list_rgb_size[i] = img_aug
                anno_frames['bbox_rgb'][i] = bbox_rgb_aug
                anno_frames['bbox_ir_scaled'][i] = bbox_tir_rgb_aug
            else:
                align_gt = natural_align_gt
            align_gt_list.append(align_gt)

        anno_frames['align_gt_4param'] = align_gt_list

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        return frame_list_rgb_size, frame_list_tir_size, anno_frames, object_meta, seq_name
