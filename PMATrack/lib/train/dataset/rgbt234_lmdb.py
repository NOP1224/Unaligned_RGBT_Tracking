import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings

from lib.utils.lmdb_utils import *

class RGBT234_lmdb(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None,data_fraction=None):
        root = env_settings().lmdb_dir if root is None else root
        super().__init__('RGBT234_lmdb', root, image_loader)

        # video_name for each sequence
        self.sequence_list = os.listdir(env_settings().rgbt234_dir)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        self.key_root = 'rgbt234.'
        
    def get_name(self):
        return 'rgbt234'

    def _read_bb_anno(self, seq_path):
        bb_anno_file = seq_path+'.init_lbl'
        gt_str_list = decode_str(self.root, bb_anno_file)  # the last line is empty
        gt_str_list = gt_str_list.split('\r\n')  if '\r\n' in gt_str_list else gt_str_list.split('\n')## the last line is empty
        while gt_str_list[-1]=='':
            del gt_str_list[-1]
        gt_list = [list(map(float, line.split(','))) for line in gt_str_list]
        # gt_list = [np.fromstring(line, sep=',') for line in gt_str_list]
        gt_arr = np.array(gt_list).astype(np.float32)
        return torch.tensor(gt_arr)

    def get_sequence_info(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        seq_path = self.key_root + seq_name
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_v(self, seq_path, frame_id):
        frame_path_v = seq_path + '.visible.' + str(frame_id)
        # frame_path_v = os.path.join(seq_path, 'visible', sorted([p for p in os.listdir(os.path.join(seq_path, 'visible')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        # return self.image_loader(frame_path_v)
        return decode_img(self.root, frame_path_v)
        
    def _get_frame_i(self, seq_path, frame_id):
        frame_path_i = seq_path + '.infrared.' + str(frame_id)
        # frame_path_i = os.path.join(seq_path, 'infrared', sorted([p for p in os.listdir(os.path.join(seq_path, 'infrared')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        # return self.image_loader(frame_path_i)
        return decode_img(self.root, frame_path_i)

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_name = self.sequence_list[seq_id]
        seq_path = self.key_root+seq_name
        frame_list_v = [self._get_frame_v(seq_path, f) for f in frame_ids]
        frame_list_i = [self._get_frame_i(seq_path, f) for f in frame_ids]
        frame_list  = frame_list_v + frame_list_i # 6
        #print('get_frames frame_list', len(frame_list))
        if anno is None:
            anno = self.get_sequence_info(seq_path)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        #return frame_list_v, frame_list_i, anno_frames, object_meta
        return frame_list, anno_frames, object_meta
