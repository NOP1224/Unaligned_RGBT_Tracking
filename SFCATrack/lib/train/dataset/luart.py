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
from lib.train.dataset.depth_utils import get_x_frame
import cv2
import time

class LUART_Dataset(BaseVideoDataset):
    

    def __init__(self, root=None, split='train', dtype='rgbrgb', seq_ids=None, data_fraction=None, min_bias = 0, max_bias = 0, train_phase = NONE):
        """
        args:
            root - path to the LasHeR trainingset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().lasher_dir if root is None else root
        assert split in ['train', 'val','all','test'], 'Only support all, train or val split in LasHeR, got {}'.format(split)
        super().__init__('LUART', root)
        self.dtype = dtype

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
        self.train_phase = train_phase
        


    def get_name(self):
        return 'luart'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True # w=h=0 in visible.txt and infrared.txt is occlusion/oov

    def _align_images(self, rgb_image, tir_image, rgb_bbox, tir_bbox):
        try:
            if isinstance(rgb_image, torch.Tensor) and isinstance(tir_image, torch.Tensor):
                return self._align_tensors(rgb_image, tir_image, rgb_bbox, tir_bbox)
            else:
                return self._align_numpy(rgb_image, tir_image, rgb_bbox, tir_bbox)
        except Exception as e:
            print(f"发生错误: {e}")
            return None

    def _align_tensors(self, rgb_image, tir_image, rgb_bbox, tir_bbox):
        if rgb_image.dim() == 4:
            rgb_image = rgb_image.squeeze(0)
        if tir_image.dim() == 4:
            tir_image = tir_image.squeeze(0)

        rgb_bbox = torch.as_tensor(rgb_bbox, device=rgb_image.device)
        tir_bbox = torch.as_tensor(tir_bbox, device=tir_image.device)

        _, rgb_h, rgb_w = rgb_image.shape
        _, tir_h, tir_w = tir_image.shape

        scale_x = rgb_w / tir_w
        scale_y = rgb_h / tir_h

        scaled_tir = F.interpolate(
            tir_image.unsqueeze(0),
            size=(rgb_h, rgb_w),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        scale_factors = torch.tensor([scale_x, scale_y, scale_x, scale_y], device=rgb_image.device)
        scaled_tir_bbox = (tir_bbox.float() * scale_factors).round().int()

        def get_center(bbox):
            return torch.stack([bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2])
        
        offset = get_center(rgb_bbox) - get_center(scaled_tir_bbox)

        aligned = torch.zeros_like(rgb_image)

        src_x = torch.clamp(-offset[0], min=0)
        src_y = torch.clamp(-offset[1], min=0)
        src_w = torch.clamp(rgb_w - offset[0], max=tir_w)
        src_h = torch.clamp(rgb_h - offset[1], max=tir_h)
        
        dst_x = torch.clamp(offset[0], min=0)
        dst_y = torch.clamp(offset[1], min=0)
        dst_w = torch.clamp(offset[0] + tir_w, max=rgb_w)
        dst_h = torch.clamp(offset[1] + tir_h, max=rgb_h)

        aligned[:, dst_y:dst_h, dst_x:dst_w] = scaled_tir[:, src_y:src_h, src_x:src_w]
        return aligned

    def _align_numpy(self, rgb_image, tir_image, rgb_bbox, tir_bbox):

        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.cpu().permute(1,2,0).numpy()
        if isinstance(tir_image, torch.Tensor):
            tir_image = tir_image.cpu().permute(1,2,0).numpy()

        rgb_bbox = np.asarray(rgb_bbox, dtype=np.int32)
        tir_bbox = np.asarray(tir_bbox, dtype=np.int32)
        
        h, w = rgb_image.shape[:2]
        scaled_tir = cv2.resize(tir_image, (w, h), interpolation=cv2.INTER_LINEAR)
        
        scale = np.array([w/tir_image.shape[1], h/tir_image.shape[0]]).repeat(2)
        scaled_tir_bbox = (tir_bbox * scale).astype(np.int32)

        def get_center(bbox):
            return np.array([bbox[0]+bbox[2]//2, bbox[1]+bbox[3]//2])
        
        offset = get_center(rgb_bbox) - get_center(scaled_tir_bbox)

        aligned = np.zeros_like(rgb_image)
        
        src_x = max(-offset[0], 0)
        src_y = max(-offset[1], 0)
        src_w = min(w - offset[0], scaled_tir.shape[1])
        src_h = min(h - offset[1], scaled_tir.shape[0])
        
        dst_x = max(offset[0], 0)
        dst_y = max(offset[1], 0)
        dst_w = min(offset[0] + scaled_tir.shape[1], w)
        dst_h = min(offset[1] + scaled_tir.shape[0], h)

        aligned[dst_y:dst_h, dst_x:dst_w] = scaled_tir[src_y:src_h, src_x:src_w]
        return aligned
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
        rgb_bb_anno_file = os.path.join(seq_path, "visible.txt")
        ir_bb_anno_file = os.path.join(seq_path, "infrared.txt")
        rgb_gt = pandas.read_csv(rgb_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        ir_gt = pandas.read_csv(ir_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(rgb_gt),torch.tensor(ir_gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox_rgb, bbox_ir= self._read_bb_anno(seq_path)
        valid = (bbox_rgb[:, 2] > 0) & (bbox_rgb[:, 3] > 0) & (bbox_ir[:, 2] > 0) & (bbox_ir[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox_rgb': bbox_rgb, 'bbox_ir': bbox_ir, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        rgb_frame_path = sorted(os.listdir(seq_path+'/NotAlign/visible/'))
        ir_frame_path = sorted(os.listdir(seq_path+'/NotAlign/infrared/'))
        rgb_pre=seq_path+'/NotAlign/visible/'
        ir_pre=seq_path+'/NotAlign/infrared/'
        return os.path.join(rgb_pre, rgb_frame_path[frame_id]), os.path.join(ir_pre, ir_frame_path[frame_id])

    def _get_frame(self, seq_path, frame_id):
        rgb_frame_path, ir_frame_path = self._get_frame_path(seq_path, frame_id)
        img_rgb_size,img_tir_size = get_x_frame(rgb_frame_path, ir_frame_path, dtype=self.dtype)
        return img_rgb_size,img_tir_size
    def _get_bias_bbox_ir(self, bbox_rgb, dx, dy):
        bbox_ir_bias = bbox_rgb + torch.tensor([dx, dy, 0, 0])
        return bbox_ir_bias
    
    def _recheck_valid(self, bbox_ir_bias):
        valid = (bbox_ir_bias[0] > 0) & (bbox_ir_bias[1] > 0) & (bbox_ir_bias[0] < 640) & (bbox_ir_bias [1] < 512)
        return valid

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        frame_list_rgb_size = []
        frame_list_tir_size = []
        frame_list_rgb_prior = []
        frame_list_tir_prior = []
        for f_id in frame_ids:
            frame_rgb, frame_tir = self._get_frame(seq_path, f_id)
            frame_list_rgb_size.append(frame_rgb)
            frame_list_tir_size.append(frame_tir)
            if f_id == 0:
                if self.train_phase == 'att_moe_phase2':
                    frame_tir_align = self._align_images(frame_rgb[:,:,:3], frame_tir[:,:,3:], anno['bbox_rgb'][0], anno['bbox_ir'][0])
                    frame_tir_align, new_bbox = self._random_image_offset(frame_tir_align, self.min_bias, self.max_bias)
                    frame_rgb = np.concatenate([frame_rgb[:,:,:3], frame_tir_align], axis=2)
                else:
                    anno_ir = anno['bbox_ir'][0]
                frame_list_rgb_prior.append(frame_rgb)
                frame_list_tir_prior.append(frame_tir)
                
                
            else:
                frame_rgb_prior, frame_tir_prior = self._get_frame(seq_path, f_id-1)
                if self.train_phase == 'att_moe_phase2':
                    frame_tir_prior_align = self._align_images(frame_rgb_prior[:,:,:3], frame_tir_prior[:,:,3:], anno['bbox_rgb'][f_id-1], anno['bbox_ir'][f_id-1])
                    frame_tir_prior_align, new_bbox= self._random_image_offset(frame_tir_prior_align, anno['bbox_rgb'][f_id-1], self.min_bias, self.max_bias)
                    frame_rgb_prior = np.concatenate([frame_rgb_prior[:,:,:3], frame_tir_prior_align], axis=2)
                else:
                    anno_ir = anno['bbox_ir'][f_id-1]
                frame_list_rgb_prior.append(frame_rgb_prior)
                frame_list_tir_prior.append(frame_tir_prior)

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        anno_frames_prior = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
            anno_frames_prior[key] = [value[max(0, f_id-1), ...].clone() for f_id in frame_ids]
        if self.train_phase == 'att_moe_phase1' or self.train_phase == 'att_moe_phase3' or self.train_phase == 'att_base':
            anno_frames['bbox_ir_bias'] = [self._get_bias_bbox_ir(anno_ir, 0, 0)]
        else:
            new_bbox = new_bbox / torch.tensor([1920, 1080, 1920, 1080], device = new_bbox.device)####
            new_bbox = new_bbox * torch.tensor([640, 512, 640, 512], device= new_bbox.device) ####
            anno_frames['bbox_ir_bias'] = [new_bbox]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list_rgb_size, frame_list_tir_size, frame_list_rgb_prior, frame_list_tir_prior, anno_frames, anno_frames_prior, object_meta
