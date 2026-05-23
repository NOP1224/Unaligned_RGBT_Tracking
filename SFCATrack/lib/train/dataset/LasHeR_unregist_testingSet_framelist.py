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
import math
import cv2


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

        frame_list_rgb_size = []
        frame_list_tir_size = []
        frame_list_rgb_prior = []
        frame_list_tir_prior = []
        sample_frameids = []
        for f_id in frame_ids:
            frame_rgb, frame_tir = self._get_frame(seq_path, f_id)
            frame_list_rgb_size.append(frame_rgb)
            frame_list_tir_size.append(frame_tir)
            anno_ir = torch.from_numpy(anno['bbox_ir'][f_id])
            
        
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
        
        return frame_list_rgb_size, frame_list_tir_size, frame_list_rgb_prior, frame_list_tir_prior, anno_frames, anno_frames_prior, object_meta, seq_name
