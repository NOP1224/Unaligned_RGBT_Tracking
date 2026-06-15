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

class GTOT(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader,split=None, data_fraction=None):
        self.root = env_settings().gtot_dir if root is None else root
        super().__init__('GTOT', root, image_loader)

        # video_name for each sequence
        self.sequence_list = os.listdir(self.root)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))
        
    def get_name(self):
        return 'gtot'

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'init.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter='\t', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_name):
        seq_path = os.path.join(self.root, seq_name)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_v(self, seq_path, frame_id):
        frame_path_v = os.path.join(seq_path, 'v', sorted([p for p in os.listdir(os.path.join(seq_path, 'v')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return self.image_loader(frame_path_v)
        
    def _get_frame_i(self, seq_path, frame_id):
        frame_path_i = os.path.join(seq_path, 'i', sorted([p for p in os.listdir(os.path.join(seq_path, 'i')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return self.image_loader(frame_path_i)

    def get_frames(self, seq_name, frame_ids, anno=None):
        seq_path = os.path.join(self.root, seq_name)
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
