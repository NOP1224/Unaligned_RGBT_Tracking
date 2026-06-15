import os
import os.path
from pickle import NONE
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame, get_x_framev2, get_x_framev3
import cv2
import time
import lmdb
import pyarrow as pa  # 推荐用来序列化 numpy
import pandas as pd
import io

import random
import math

# ===== Sharded LMDB Reader =====
import json

class ShardedLMDBReader:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.index_path = os.path.join(root_dir, "index.json")
        if not os.path.isfile(self.index_path):
            raise FileNotFoundError(f"index.json not found in {root_dir}")
        with open(self.index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        self.seq_to_shard = self.index.get("seq_to_shard", {})
        self._env_cache = {}
        self._txn_cache = {}

    def _open_env(self, shard_rel_path: str):
        if shard_rel_path in self._env_cache:
            return self._env_cache[shard_rel_path], self._txn_cache[shard_rel_path]
        shard_dir = os.path.join(self.root_dir, shard_rel_path)
        env = lmdb.open(shard_dir, readonly=True, lock=False, readahead=False, meminit=False, subdir=True, max_dbs=1)
        txn = env.begin(write=False)
        self._env_cache[shard_rel_path] = env
        self._txn_cache[shard_rel_path] = txn
        return env, txn

    def _get_txn_by_seq(self, seq_name: str):
        shard_rel = self.seq_to_shard.get(seq_name, None)
        if shard_rel is None:
            shards = self.index.get("shards", [])
            if not shards:
                raise KeyError(f"Sequence {seq_name} not found in index.json")
            shard_rel = shards[-1]["path"]
        _, txn = self._open_env(shard_rel)
        return txn

    def get_bytes(self, seq_name: str, key: str):
        txn = self._get_txn_by_seq(seq_name)
        return txn.get(key.encode())

    def get_num_frames(self, seq_name: str):
        val = self.get_bytes(seq_name, f"{seq_name}_num_frames")
        if val is None:
            return None
        try:
            return int(val.decode("ascii"))
        except Exception:
            return None


def _std_normal():
    # Box–Muller：O(1) 生成 N(0,1)
    u1 = 1.0 - random.random()  # 避免 log(0)
    u2 = random.random()
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return r * math.cos(theta)

def sample_bimodal_edge_gaussian(N, M, K, sigma=3.0, p_left=0.5, exclude_self=True):
    """
    在 [N-K, N+K] 内采样，让左右两端 (N-K) 与 (N+K) 各自呈高斯峰（往内侧衰减）。
    - O(1) 计算：先选边，再从该边的“半正态”距离采样。
    - sigma: 控制离端点的“远近”偏好（越小越贴近端点，越大越向内扩散）
    - p_left: 选择左端作为基点的概率（默认左右对称 0.5）
    - exclude_self: 是否避免采到 N（通常不会采到 N，但极端参数下可能）
    """
    assert 0 <= N < M and M > 0
    if M == 1:
        return 0

    L = max(0, N - K)
    R = min(M - 1, N + K)
    if L == R:
        return L

    # 1) 选边：左端或右端
    from_left = (random.random() < p_left)
    edge = L if from_left else R

    # 2) 从该边往内“半正态”距离 d（离散化）
    #    距离 d ~ |N(0, sigma)|，再四舍五入到整数
    d = int(round(abs(_std_normal()) * sigma))

    # 3) 计算候选索引：从边界往内推进 d
    j = edge + d if from_left else edge - d

    # 4) 限制在 [L, R] 之内；若超出就裁剪（不会影响 O(1)）
    if j < L:
        j = L
    elif j > R:
        j = R

    # 5) 可选：避免采到自身 N（极少发生）
    if exclude_self and j == N:
        # 往更靠近边的方向挪一步
        if from_left and j < R:
            j = min(R, j + 1)
        elif (not from_left) and j > L:
            j = max(L, j - 1)
        # 如果还是 N（例如 K 很小），就换到另一端点
        if j == N:
            j = L if not from_left else R

    return j


class LUART_Dataset(BaseVideoDataset):
    

    def __init__(self, root=None, split='train', dtype='rgbrgb', seq_ids=None, 
                 data_fraction=None, min_bias = 0, max_bias = 0,
                 use_lmdb=False, lmdb_path = "/data1/Datasets/Tracking/LUART/luart_train_set_withalign/"):
        
        root = env_settings().lasher_dir if root is None else root
        assert split in ['train', 'val','all','test'], 'Only support all, train or val split in LasHeR, got {}'.format(split)
        super().__init__('Mydataset', root)
        self.dtype = dtype
        self.split = split
        # all folders inside the root
        self.sequence_list = self._get_sequence_list(split)

        # seq_id is the index of the folder inside the got10k root path
        if seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))
        self.min_bias = min_bias
        self.max_bias = max_bias
        # 自动检测 LMDB
        self.use_lmdb = use_lmdb
        self.lmdb_env = None
        self.txn = None
        self.sharded = False
        self.sharded_reader = None
        self.using_align = True
        lmdb_path = lmdb_path
        self.sharded = True
        self.sharded_reader = ShardedLMDBReader(lmdb_path)
        print(f"[LUART_Dataset] Using SHARDED LMDB at {lmdb_path} (index.json found)")
        self.margin = 150
    def get_name(self):
        return 'luart'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True # w=h=0 in visible.txt and infrared.txt is occlusion/oov

    def _lmdb_get(self, key):
        with self.lmdb_env.begin(write=False) as txn:
            byteflow = txn.get(key.encode('ascii'))
        if byteflow is None:
            raise KeyError(f"Key {key} not found in LMDB")
        return pa.deserialize(byteflow)
    
    def _get_sequence_list(self, split):
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        # file_path = os.path.join(ltr_path, 'data_specs', 'mydataset_training_list.txt')
        if split == 'train':
            file_path = os.path.join(ltr_path, 'data_specs', 'luart_training_list.txt')
        if split == 'test':
            file_path = os.path.join(ltr_path, 'data_specs', 'luart_testing_list.txt')
        if split == 'all':
            file_path = os.path.join(ltr_path, 'data_specs', 'luart_all_list.txt')
        with open(file_path, 'r') as f:
            dir_list = f.read().splitlines()
        return dir_list

    def _read_bb_anno(self, seq_path):
        return self._read_bb_anno_lmdb(seq_path)
        
    
    def _lmdb_get_for_seq(self, seq_name: str, key_suffix: str):
        key = f"{seq_name}{key_suffix}"
        if getattr(self, "sharded", False) and self.sharded_reader is not None:
            return self.sharded_reader.get_bytes(seq_name, key)
        elif self.txn is not None:
            return self.txn.get(key.encode())
        else:
            return None

    def _decode_image_from_bytes(self, byteflow):
        if byteflow is None or byteflow == b"NULL":
            return None
        arr = np.frombuffer(byteflow, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    def _read_bb_anno_lmdb(self, seq_path):
        seq_name = os.path.basename(seq_path)

        rgb_txt = self._lmdb_get_for_seq(seq_name, "_anno_rgb")
        ir_txt  = self._lmdb_get_for_seq(seq_name, "_anno_ir")

        if not rgb_txt or rgb_txt == b"NULL":
            rgb_gt = np.zeros((0, 4), dtype=np.float32)
        else:
            rgb_gt = pd.read_csv(io.StringIO(rgb_txt.decode()), header=None).values.astype(np.float32)

        if not ir_txt or ir_txt == b"NULL":
            ir_gt = np.zeros((0, 4), dtype=np.float32)
        else:
            ir_gt = pd.read_csv(io.StringIO(ir_txt.decode()), header=None).values.astype(np.float32)

        return torch.tensor(rgb_gt), torch.tensor(ir_gt)


    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox_rgb, bbox_ir= self._read_bb_anno(seq_path)
        valid = (bbox_rgb[:, 2] > 0) & (bbox_rgb[:, 3] > 0) & (bbox_ir[:, 2] > 0) & (bbox_ir[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox_rgb': bbox_rgb, 'bbox_ir': bbox_ir, 'valid': valid, 'visible': visible}
    
    def _get_frame(self, seq_path, frame_id):
         return self._get_frame_lmdb(seq_path, frame_id)

    def _get_frame_lmdb(self, seq_path, frame_id):
        seq_name = os.path.basename(seq_path)

        rgb_enc = self._lmdb_get_for_seq(seq_name, f"_frame_{frame_id}_rgb")
        ir_enc  = self._lmdb_get_for_seq(seq_name, f"_frame_{frame_id}_ir")
        

        img_rgb = self._decode_image_from_bytes(rgb_enc)
        img_ir  = self._decode_image_from_bytes(ir_enc)
        
        img_rgb_size, img_tir_size = get_x_framev3(img_rgb, img_ir)

        return img_rgb_size, img_tir_size
    
    def _get_bias_bbox_ir(self, bbox_rgb, dx, dy):
        bbox_ir_bias = bbox_rgb + torch.tensor([dx, dy, 0, 0])
        return bbox_ir_bias
    
    def _recheck_valid(self, bbox_ir_bias):
        valid = (bbox_ir_bias[0] > 0) & (bbox_ir_bias[1] > 0) & (bbox_ir_bias[0] < 640) & (bbox_ir_bias [1] < 512)
        return valid

    def _align_numpy(self, tir_image, rgb_bbox, tir_bbox, warped_tir_bbox=None,
                    h_rgb=1080, w_rgb=1920, fast_mode=True, use_cuda=False):
        """
        对齐 TIR->RGB，默认优先使用快速仿射；必要时退回 homography。

        返回：aligned(HxWxC)、warped_bbox([x,y,w,h])、meta字典
        meta 里包含：'mode'('affine'/'homo')、'M'(2x3 或 3x3)
        """
        rgb_bbox = np.asarray(rgb_bbox, dtype=np.float32)
        tir_bbox = np.asarray(tir_bbox, dtype=np.float32)
        h_rgb, w_rgb = 1080, 1920
        h_tir, w_tir = 512, 640
        
        # 按同样比例缩放 TIR bbox
        scale_x = w_rgb / w_tir
        scale_y = h_rgb / h_tir
        tir_bbox_scaled = np.array([
            tir_bbox[0]*scale_x, tir_bbox[1]*scale_y,
            tir_bbox[2]*scale_x, tir_bbox[3]*scale_y
        ], dtype=np.float32)
        
        warped_tir_bbox_scaled = np.array([
            warped_tir_bbox[0]*scale_x, warped_tir_bbox[1]*scale_y,
            warped_tir_bbox[2]*scale_x, warped_tir_bbox[3]*scale_y
        ], dtype=np.float32)
        # -------- 快路径：各向异性仿射 --------
        if fast_mode:
            x_r, y_r, w_r, h_r = rgb_bbox
            x_t, y_t, w_t, h_t = tir_bbox_scaled
            if w_t > 1e-3 and h_t > 1e-3:
                sx = (w_r / w_t); sy = (h_r / h_t)
                tx = x_r - sx * x_t; ty = y_r - sy * y_t
                M = np.array([[sx, 0.0, tx],
                            [0.0, sy, ty]], dtype=np.float32)

                aligned = cv2.warpAffine(
                    tir_image, M, (w_rgb, h_rgb),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

                warped_bbox = None
                xt, yt, wt, ht = map(float, warped_tir_bbox_scaled)
                warped_bbox = np.array([sx*xt + tx, sy*yt + ty, sx*wt, sy*ht], dtype=np.float32)
                return aligned, warped_bbox

        # -------- 兜底：透视变换（避免 RANSAC 可用 0/LMEDS）--------
        def bbox_to_corners(b):
            x, y, w, h = b
            return np.array([[x, y],[x+w, y],[x+w, y+h],[x, y+h]], dtype=np.float32)

        pts_rgb = bbox_to_corners(rgb_bbox)
        pts_tir = bbox_to_corners(tir_bbox)

        # 不用 RANSAC（稳且快），或改用 LMEDS
        H, _ = cv2.findHomography(pts_tir, pts_rgb, method=0)  # 0=0RANSAC=0, LMEDS=4

        aligned = cv2.warpPerspective(
            tir_image, H, (w_rgb, h_rgb),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        warped_bbox = None
        if warped_tir_bbox is not None:
            pts_warp = bbox_to_corners(np.asarray(warped_tir_bbox, dtype=np.float32)).reshape(-1,1,2)
            pts_warped = cv2.perspectiveTransform(pts_warp, H).reshape(-1,2)
            x_min, y_min = pts_warped[:,0].min(), pts_warped[:,1].min()
            x_max, y_max = pts_warped[:,0].max(), pts_warped[:,1].max()
            warped_bbox = np.array([x_min, y_min, x_max-x_min, y_max-y_min], dtype=np.float32)

        return aligned, warped_bbox

    def _align_images(self, tir_image, rgb_bbox, tir_bbox, warped_tir_bbox):
        return self._align_numpy(tir_image, rgb_bbox, tir_bbox, warped_tir_bbox)
    
    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        frame_length = len(os.listdir(os.path.join(seq_path, "NotAlign/visible")))
        
        frame_list_rgb_size = []
        frame_list_tir_size = []
        frame_list_rgb_prior = []
        frame_list_tir_prior = []
        sample_frameids = []
        for f_id in frame_ids:
            frame_rgb, frame_tir = self._get_frame(seq_path, f_id)
            sampled_fid = sample_bimodal_edge_gaussian(f_id, frame_length, self.margin)
            
            frame_tir_weakly_align, weakly_align_bbox = self._align_images(frame_rgb[:,:,3:], anno['bbox_rgb'][sampled_fid],
                                                            anno['bbox_ir'][sampled_fid] , anno['bbox_ir'][f_id])
            # weakly_align_bbox = self._get_align_ir(seq_path, sampled_fid, anno['bbox_rgb'][f_id],
            #                                             anno['bbox_ir'][f_id], anno['bbox_ir'][f_id])
            
            frame_rgb = np.concatenate([frame_rgb[:,:,:3], frame_tir_weakly_align], axis=2)
            frame_list_rgb_size.append(frame_rgb)
            frame_list_tir_size.append(frame_tir)
            sample_frameids.append(sampled_fid)
            anno_ir = torch.from_numpy(weakly_align_bbox)
            

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        anno_frames_prior = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
            anno_frames_prior[key] = [value[max(0, f_id-1), ...].clone() for f_id in frame_ids]
            
        anno_frames['bbox_ir_bias'] = [self._get_bias_bbox_ir(anno_ir, 0, 0)]
        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        seq_name = seq_path.split('/')[-1]
        return frame_list_rgb_size, frame_list_tir_size, frame_list_rgb_prior, frame_list_tir_prior, anno_frames, anno_frames_prior, object_meta, seq_name, sampled_fid
