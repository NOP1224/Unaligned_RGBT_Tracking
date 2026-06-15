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
from lib.train.dataset.depth_utils import get_x_frame, get_x_framev2
import cv2
import time
import lmdb
import pyarrow as pa  # 推荐用来序列化 numpy
import pandas as pd
import io

import random
import math

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


class MUART244(BaseVideoDataset):
    def __init__(self, root=None, split='train', dtype='rgbrgb', seq_ids=None, 
                 data_fraction=None):
        
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

        self.margin = 150
    def get_name(self):
        return 'muart244'

    def _align_images(self, tir_image, rgb_bbox, tir_bbox, warped_tir_bbox):
        return self._align_numpy(tir_image, rgb_bbox, tir_bbox, warped_tir_bbox)

    def _align_numpy(self, tir_image, rgb_bbox, tir_bbox, warped_tir_bbox):
        
        # 确保整数
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
        
        # 定义四个角点 (x, y)
        def bbox_to_corners(bbox):
            x, y, w, h = bbox
            return np.array([
                [x, y],           # 左上
                [x + w, y],       # 右上
                [x + w, y + h],   # 右下
                [x, y + h]        # 左下
            ], dtype=np.float32)
        
        pts_rgb = bbox_to_corners(rgb_bbox)
        pts_tir = bbox_to_corners(tir_bbox_scaled)
        
        # 计算单应性矩阵
        H, _ = cv2.findHomography(pts_tir, pts_rgb, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        
        aligned = cv2.warpPerspective(
            tir_image, H, (w_rgb, h_rgb),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        warped_bbox = None
        if warped_tir_bbox_scaled is not None:
            pts_warp = bbox_to_corners(warped_tir_bbox_scaled).reshape(-1, 1, 2)
            pts_warped = cv2.perspectiveTransform(pts_warp, H).reshape(-1, 2)
            # 重新取最小外接矩形
            x_min, y_min = pts_warped[:,0].min(), pts_warped[:,1].min()
            x_max, y_max = pts_warped[:,0].max(), pts_warped[:,1].max()
            warped_bbox = np.array([x_min, y_min, x_max - x_min, y_max - y_min], dtype=np.float32)
            
        return aligned, warped_bbox
    
    def _random_image_offset(self, image, bbox, offset_min, offset_max, aspect_ratio_threshold=0.25, length_ratio_threshold=0.5, max_retries=100):
        height, width = image.shape[:2]
        x, y, w, h = bbox
        original_aspect_ratio = w / h

        for _ in range(max_retries):
            dy_top_left = np.random.randint(offset_min, offset_max + 1)
            dx_top_left = np.random.randint(offset_min, offset_max + 1)

            if np.random.randint(0, 2) == 0:
                dy_top_left = -dy_top_left
            if np.random.randint(0, 2) == 0:
                dx_top_left = -dx_top_left


            dy_bottom_right = np.random.randint(offset_min, offset_max + 1)
            dx_bottom_right = np.random.randint(offset_min, offset_max + 1)

            if np.random.randint(0, 2) == 0:
                dy_bottom_right = -dy_bottom_right
            if np.random.randint(0, 2) == 0:
                dx_bottom_right = -dx_bottom_right

            src_points = np.float32([
                [x, y],
                [x + w, y],
                [x, y + h],
                [x + w, y + h]
            ])

            dst_points = np.float32([
                [x + dx_top_left, y + dy_top_left],
                [x + w + dx_bottom_right, y + dy_top_left],
                [x + dx_top_left, y + h + dy_bottom_right],
                [x + w + dx_bottom_right, y + h + dy_bottom_right]
            ])

            top_left = dst_points[0]
            bottom_right = dst_points[3]
            if top_left[0] >= bottom_right[0] or top_left[1] >= bottom_right[1]:
                continue

            M = cv2.getAffineTransform(src_points[:3], dst_points[:3])

            new_corners = cv2.transform(np.array([src_points]), M)[0]

            new_x = np.min(new_corners[:, 0])
            new_y = np.min(new_corners[:, 1])
            new_w = np.max(new_corners[:, 0]) - new_x
            new_h = np.max(new_corners[:, 1]) - new_y
            new_aspect_ratio = new_w / new_h

            aspect_ratio_diff = abs(new_aspect_ratio - original_aspect_ratio) / original_aspect_ratio
            if aspect_ratio_diff > aspect_ratio_threshold:
                continue

            width_ratio = new_w / w
            height_ratio = new_h / h
            if abs(width_ratio - 1) > length_ratio_threshold or abs(height_ratio - 1) > length_ratio_threshold:
                continue
            offset_image = cv2.warpAffine(image, M, (width, height))
            new_bbox = torch.Tensor([new_x, new_y, new_w, new_h])
            return offset_image, new_bbox

        return image, torch.Tensor(bbox)
    
    def _get_sequence_list(self, split):
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        # file_path = os.path.join(ltr_path, 'data_specs', 'mydataset_training_list.txt')
        file_path = os.path.join(ltr_path, 'data_specs', 'muart244_testinglist.txt')
        with open(file_path, 'r') as f:
            dir_list = f.read().splitlines()
        return dir_list

    def _read_bb_anno(self, seq_path):
        if self.use_lmdb:
            return self._read_bb_anno_lmdb(seq_path)
        else:
            return self._read_bb_anno_ori(seq_path)
        
    def _read_bb_anno_ori(self, seq_path):
        rgb_bb_anno_file = os.path.join(seq_path, "visible.txt")
        ir_bb_anno_file = os.path.join(seq_path, "infrared.txt")
        rgb_gt = pandas.read_csv(rgb_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        ir_gt = pandas.read_csv(ir_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(rgb_gt),torch.tensor(ir_gt)

    def _read_bb_anno_lmdb(self, seq_path):
        seq_name = os.path.basename(seq_path)

        rgb_txt = self.txn.get(f"{seq_name}_anno_rgb".encode())
        ir_txt  = self.txn.get(f"{seq_name}_anno_ir".encode())

        if rgb_txt is None or rgb_txt == b"NULL":
            rgb_gt = np.zeros((0, 4), dtype=np.float32)
        else:
            rgb_gt = pd.read_csv(io.StringIO(rgb_txt.decode()), header=None).values.astype(np.float32)

        if ir_txt is None or ir_txt == b"NULL":
            ir_gt = np.zeros((0, 4), dtype=np.float32)
        else:
            ir_gt  = pd.read_csv(io.StringIO(ir_txt.decode()), header=None).values.astype(np.float32)

        return torch.tensor(rgb_gt), torch.tensor(ir_gt)


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
        valid = (bbox_v[:, 2] < 10000) & (bbox_v[:, 2] > 10) & \
                (bbox_v[:, 3] < 10000) & (bbox_v[:, 3] > 10) & \
                (bbox_v[:, 0] < 10000) & (bbox_v[:, 0] > 0)  & \
                (bbox_v[:, 1] < 10000) & (bbox_v[:, 1] > 0)  & \
                (bbox_i[:, 2] < 10000) & (bbox_i[:, 2] > 10) & \
                (bbox_i[:, 3] < 10000) & (bbox_i[:, 3] > 10) & \
                (bbox_i[:, 0] < 10000) & (bbox_i[:, 0] > 0)  & \
                (bbox_i[:, 1] < 10000) & (bbox_i[:, 1] > 0)
            
        visible = valid.clone().byte()
        return {'bbox_rgb': bbox_v, 'bbox_ir': bbox_i, 'valid': valid, 'visible': visible}
    
        
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

 
    def _align_images(self, tir_image, rgb_bbox, tir_bbox, warped_tir_bbox, h_rgb, w_rgb, h_tir, w_tir, pre_aligned=True):
        return self._align_numpy(tir_image, rgb_bbox, tir_bbox, warped_tir_bbox, h_rgb, w_rgb, h_tir, w_tir, pre_aligned)

    def _align_numpy(self, tir_image, rgb_bbox, tir_bbox, warped_tir_bbox, h_rgb, w_rgb, h_tir, w_tir, pre_aligned):
        """
        使用单应性矩阵对齐 TIR 图像到 RGB 图像

        rgb_image: HxWxC, numpy 或 tensor
        tir_image: HxWxC, numpy 或 tensor
        rgb_bbox: [x, y, w, h] in RGB
        tir_bbox: [x, y, w, h] in TIR
        """
        
        # 确保整数
        rgb_bbox = np.asarray(rgb_bbox, dtype=np.float32)
        tir_bbox = np.asarray(tir_bbox, dtype=np.float32)

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
        
        if not pre_aligned: 
            return tir_image, warped_tir_bbox_scaled
        
        
        # 定义四个角点 (x, y)
        def bbox_to_corners(bbox):
            x, y, w, h = bbox
            return np.array([
                [x, y],           # 左上
                [x + w, y],       # 右上
                [x + w, y + h],   # 右下
                [x, y + h]        # 左下
            ], dtype=np.float32)
        
        pts_rgb = bbox_to_corners(rgb_bbox)
        pts_tir = bbox_to_corners(tir_bbox_scaled)
        
        # 计算单应性矩阵
        H, _ = cv2.findHomography(pts_tir, pts_rgb, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        
        aligned = cv2.warpPerspective(
            tir_image, H, (w_rgb, h_rgb),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        warped_bbox = None
        if warped_tir_bbox_scaled is not None:
            pts_warp = bbox_to_corners(warped_tir_bbox_scaled).reshape(-1, 1, 2)
            pts_warped = cv2.perspectiveTransform(pts_warp, H).reshape(-1, 2)
            # 重新取最小外接矩形
            x_min, y_min = pts_warped[:,0].min(), pts_warped[:,1].min()
            x_max, y_max = pts_warped[:,0].max(), pts_warped[:,1].max()
            warped_bbox = np.array([x_min, y_min, x_max - x_min, y_max - y_min], dtype=np.float32)
            
        return aligned, warped_bbox

    def _get_bias_bbox_ir(self, bbox_rgb, dx, dy):
        bbox_ir_bias = bbox_rgb + torch.tensor([dx, dy, 0, 0])
        return bbox_ir_bias
    
    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, seq_name)
        frame_length = len(os.listdir(os.path.join(seq_path, "visible")))
            
        frame_list_rgb_size = []
        frame_list_tir_size = []
        frame_list_rgb_prior = []
        frame_list_tir_prior = []
        sample_frameids = []
        for f_id in frame_ids:
            frame_rgb, frame_tir = self._get_frame(seq_path, f_id)
            h_rgb, w_rgb = frame_rgb.shape[:2]
            h_tir, w_tir = frame_tir.shape[:2]
            sampled_fid = sample_bimodal_edge_gaussian(f_id, frame_length, self.margin)
            frame_tir_weakly_align, weakly_align_bbox = self._align_images(frame_rgb[:,:,3:], anno['bbox_rgb'][sampled_fid],
                                                            anno['bbox_ir'][sampled_fid] , anno['bbox_ir'][f_id], h_rgb, w_rgb, h_tir, w_tir, pre_aligned=True)
            
            frame_rgb = np.concatenate([frame_rgb[:,:,:3], frame_tir_weakly_align], axis=2)
            frame_list_rgb_size.append(frame_rgb)
            frame_list_tir_size.append(frame_tir)
            sample_frameids.append(sampled_fid)
            anno_ir = torch.from_numpy(weakly_align_bbox)
            
        
        if anno is None:
            anno = self.get_sequence_info(seq_path)

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
        
        return frame_list_rgb_size, frame_list_tir_size, frame_list_rgb_prior, frame_list_tir_prior, anno_frames, anno_frames_prior, object_meta, seq_name, sampled_fid, h_rgb, w_rgb
