import math

from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor, PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import numpy as np
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_iou
from lib.train.actors.homography_compute import SimpleHomoNormalizer
from lib.models.layers.refiner_light import NoParam_Feature_Shifter, WarpGridRefinerLight
def offset_to_homography(offset, device):
    B = offset.shape[0]

    dx, dy, sx, sy = offset[:,0], offset[:,1], offset[:,2], offset[:,3]
    H = torch.eye(3, device=device).unsqueeze(0).repeat(B,1,1)
    H[:,0,0] = sx
    H[:,1,1] = sy
    H[:,0,2] = dx
    H[:,1,2] = dy
    
    return H


def get_align(rgb, tir, offset):
    tir_resized_aligned = cv2.warpPerspective(
        tir, offset, (rgb.shape[1], rgb.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    img_rgb_size = cv2.merge((rgb, tir_resized_aligned))
    return img_rgb_size

class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorMM()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        if getattr(params, 'debug', None) is None:
            setattr(params, 'debug', 0)
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.dynamic_z_tensor = None
        self.updated_freq = 20
        self.alpha = 0.5
        self.beta = 0.7
        self.homo_norm = SimpleHomoNormalizer(self.cfg.DATA.SEARCH.SIZE, self.cfg.DATA.SEARCH.SIZE, margin_ratio=0.0)
        self.warp_refiner = NoParam_Feature_Shifter(16,16)
    def initialize_notalign(self, image_rgb, image_tir, info: dict):
        # forward the template once
        self.rgb_bbox_0 = info['init_bbox_rgb']
        self.ir_bbox_0 = info['init_bbox_tir']
        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image_rgb, info['init_bbox_rgb'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        z_patch_arr_tir, resize_factor_tir, z_amask_arr_tir  = sample_target(image_tir, info['init_bbox_tir'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image_rgb, info['init_bbox_rgb'], self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        self.z_patch_arr_rgb = z_patch_arr
        self.z_patch_arr_tir = z_patch_arr_tir
        template_rgb = self.preprocessor.process(z_patch_arr)
        teamplate_tir = self.preprocessor.process(z_patch_arr_tir)
        search = self.preprocessor.process(x_patch_arr)
        
        self.search_prior = search
        # rgb_img = z_patch_arr[:, :, :3]
        # tir_img = z_patch_arr_tir[:, :, 3:6]
        # overlay_savedir = f'./template/'
        # os.makedirs(overlay_savedir, exist_ok=True)
        # save_path_rgb = os.path.join(overlay_savedir, f"template_rgb_{self.frame_id:04d}_prealign.jpg")
        # save_path_tir = os.path.join(overlay_savedir, f"template_tir_{self.frame_id:04d}_prealign.jpg")
        # cv2.imwrite(save_path_rgb, rgb_img)
        # cv2.imwrite(save_path_tir, tir_img)
        
        template = torch.cat((template_rgb[:,:3,:,:], teamplate_tir[:,3:,:,:]), dim=1)
        with torch.no_grad():
            self.z_tensor = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox_rgb'], resize_factor,
                                                        template.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)

        # save states
        self.state = info['init_bbox_rgb']
        self.state_bias = info['init_bbox_rgb']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox_rgb'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}
    
        self.first_search = search
        self.fisrt_resize_factor_rgb = resize_factor
        self.init_bbox = info['init_bbox_rgb']
        self.H, self.W, _ = image_rgb.shape
        self.H_T, self.W_T, _ = image_tir.shape

    def track_notalign(self, image, seq_name, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        x_patch_arr = np.concatenate((x_patch_arr[:,:,:3], x_patch_arr[:, :, 3:6]), axis=2)
        search = self.preprocessor.process(x_patch_arr)
        
        templates = [self.z_tensor]
        with torch.no_grad():
            x_tensor = search
            out_dict = self.network.forward(
                template=templates, search=x_tensor, ce_template_mask=self.box_mask_z)
        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # clip the box
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
        # scores = pred_score_map.reshape(256).max()
        # with torch.no_grad():
        #     if scores > self.alpha:
                
        #         x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.template_factor,
        #                                                                 output_sz=self.params.template_size)  # (x1, y1, w, h)
                
        #         dynamic_z_tensor = self.preprocessor.process(x_patch_arr)
                
        #         reversed_out_dict = self.network.forward(
        #             template=[dynamic_z_tensor], 
        #             search=self.first_search,
        #             ce_template_mask=self.box_mask_z)    
        #         pred_score_map = reversed_out_dict['score_map']
        #         response = self.output_window * pred_score_map
        #         pred_boxes = self.network.box_head.cal_bbox(response, reversed_out_dict['size_map'], reversed_out_dict['offset_map'])
        #         pred_boxes = pred_boxes.view(-1, 4)
        #         # Baseline: Take the mean of all pred boxes as the final result
        #         pred_box = (pred_boxes.mean(
        #             dim=0) * self.params.search_size / self.fisrt_resize_factor_rgb).tolist()  # (cx, cy, w, h) [0,1]
        #         # get the final box result
        #         reversed_state = clip_box(self.map_box_back(pred_box, self.fisrt_resize_factor_rgb), H, W, margin=10)
        #         init_bbox = box_xywh_to_xyxy(torch.tensor(self.init_bbox).unsqueeze(0))
        #         reversed_state = box_xywh_to_xyxy(torch.tensor(reversed_state).unsqueeze(0))
        #         iou_scores,_ = box_iou(reversed_state, init_bbox)
        #         if iou_scores[0] > self.beta:
        #             self.dynamic_z_tensor = dynamic_z_tensor
        
        # for debug
        if self.debug == 1:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                cv2.putText(image_BGR, 'max_score:' + str(round(max_score, 3)), (40, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 2)
                cv2.imshow('debug_vis', image_BGR)
                cv2.waitKey(1)
            else:
                self.visdom.register((image[:,:,:3], info['gt_bbox_rgb'], self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr[:,:,:3]).permute(2,0,1), 'image', 1, 'search_region_rgb')
                self.visdom.register(torch.from_numpy(x_patch_arr[:,:,3:]).permute(2,0,1), 'image', 1, 'search_region_tir')
                self.visdom.register(torch.from_numpy(self.z_patch_arr_rgb[:,:,:3]).permute(2,0,1), 'image', 1, 'template_rgb')
                self.visdom.register(torch.from_numpy(self.z_patch_arr_tir[:,:,3:]).permute(2,0,1), 'image', 1, 'template_tir')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s'] and (out_dict['removed_indexes_s'][0] is not None):
                        removed_indexes_s = out_dict['removed_indexes_s']
                        removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                        masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                        self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break
                    
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score}

    def track_stepalign(self, image, image_unalign, seq_name, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        merge_flag = False
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        x_patch_arr = np.concatenate((x_patch_arr[:,:,:3], x_patch_arr[:, :, 3:6]), axis=2)
        search = self.preprocessor.process(x_patch_arr)
        
        templates = [self.z_tensor] if self.dynamic_z_tensor is None else \
            [self.z_tensor, self.dynamic_z_tensor]
        with torch.no_grad():
            x_tensor = search
            out_dict = self.network.forward(
                template=templates, search=x_tensor, ce_template_mask=self.box_mask_z)
        
        local_offset_norm = out_dict["offsets"][-1]
        local_offset_norm = offset_to_homography(local_offset_norm, local_offset_norm.device)
        local_offset = self.homo_norm.denormalize(local_offset_norm)
        
        # === (2) 直接实现 local_to_global ===
        # 提取 state (bbox中心点、大小)
        x1, y1, w, h = self.state  # 假设 state 是 [x, y, w, h]
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        patch_H, patch_W = self.params.search_size, self.params.search_size
        scale_x = w * self.params.search_factor / patch_W
        scale_y = h * self.params.search_factor / patch_H

        # 构造将 local patch 映射回全图的仿射矩阵
        # 从 crop 到原图的尺度 + 平移
        T_crop_to_global = np.array([
            [scale_x, 0, cx - (patch_W / 2) * scale_x],
            [0, scale_y, cy - (patch_H / 2) * scale_y],
            [0, 0, 1]
        ], dtype=np.float32)

        # 将 local_offset 应用于原图尺度
        local_offset_np = local_offset.cpu().numpy()
        offset = T_crop_to_global @ local_offset_np @ np.linalg.inv(T_crop_to_global)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # clip the box
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
        scores = pred_score_map.reshape(256).max()
        with torch.no_grad():
            if max_score > self.alpha:
                image_aligned = get_align(image[:,:,:3], image[:, :, 3:6], np.squeeze(offset, axis=0)) # 使用现在预测出来的跟踪第一帧分数
                x_patch_arr_aligned, resize_factor_aligned, x_amask_arr_aligned = sample_target(image_aligned, self.state, self.params.template_factor,
                                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
                x_patch_arr, resize_factor, x_amask_arr_aligned = sample_target(image, self.state, self.params.template_factor,
                                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
                
                dynamic_z_tensor_aligned = self.preprocessor.process(x_patch_arr_aligned) 
                reversed_out_dict_aligned  = self.network.forward(
                    template=[dynamic_z_tensor_aligned], 
                    search=self.first_search,
                    ce_template_mask=self.box_mask_z)  
                
                dynamic_z_tensor = self.preprocessor.process(x_patch_arr) 
                reversed_out_dict  = self.network.forward(
                    template=[dynamic_z_tensor], 
                    search=self.first_search,
                    ce_template_mask=self.box_mask_z)  
                 
                pred_score_map_aligned = reversed_out_dict_aligned['score_map']
                response_aligned = self.output_window * pred_score_map_aligned
                pred_boxes_aligned = self.network.box_head.cal_bbox(response_aligned, reversed_out_dict_aligned['size_map'], reversed_out_dict_aligned['offset_map'])
                pred_boxes_aligned = pred_boxes_aligned.view(-1, 4)
                pred_box_aligned = (pred_boxes_aligned.mean(
                    dim=0) * self.params.search_size / self.fisrt_resize_factor_rgb).tolist()  # (cx, cy, w, h) [0,1]
                reversed_state_aligned = clip_box(self.map_box_back(pred_box_aligned, self.fisrt_resize_factor_rgb), H, W, margin=10)
                reversed_state_aligned = box_xywh_to_xyxy(torch.tensor(reversed_state_aligned).unsqueeze(0))
                
                pred_score_map = reversed_out_dict['score_map']
                response = self.output_window * pred_score_map
                pred_boxes = self.network.box_head.cal_bbox(response, reversed_out_dict['size_map'], reversed_out_dict['offset_map'])
                pred_boxes = pred_boxes.view(-1, 4)
                pred_box = (pred_boxes.mean(
                    dim=0) * self.params.search_size / self.fisrt_resize_factor_rgb).tolist()  # (cx, cy, w, h) [0,1]
                reversed_state = clip_box(self.map_box_back(pred_box, self.fisrt_resize_factor_rgb), H, W, margin=10)
                reversed_state = box_xywh_to_xyxy(torch.tensor(reversed_state).unsqueeze(0))
                
                init_bbox = box_xywh_to_xyxy(torch.tensor(self.init_bbox).unsqueeze(0))
                
                iou_scores_aligned,_ = box_iou(reversed_state_aligned, init_bbox)
                iou_scores,_ = box_iou(reversed_state, init_bbox)
                
                
                if iou_scores_aligned[0] >= iou_scores[0] and iou_scores_aligned[0] > self.beta:
                    merge_flag = True
                    self.dynamic_z_tensor = dynamic_z_tensor_aligned
                elif iou_scores[0] > self.beta:
                    self.dynamic_z_tensor = dynamic_z_tensor
        
        # for debug
        if self.debug == 1:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                cv2.putText(image_BGR, 'max_score:' + str(round(max_score, 3)), (40, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 2)
                cv2.imshow('debug_vis', image_BGR)
                cv2.waitKey(1)
            else:
                self.visdom.register((image[:,:,:3], info['gt_bbox_rgb'], self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr[:,:,:3]).permute(2,0,1), 'image', 1, 'search_region_rgb')
                self.visdom.register(torch.from_numpy(x_patch_arr[:,:,3:]).permute(2,0,1), 'image', 1, 'search_region_tir')
                self.visdom.register(torch.from_numpy(self.z_patch_arr_rgb[:,:,:3]).permute(2,0,1), 'image', 1, 'template_rgb')
                self.visdom.register(torch.from_numpy(self.z_patch_arr_tir[:,:,3:]).permute(2,0,1), 'image', 1, 'template_tir')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s'] and (out_dict['removed_indexes_s'][0] is not None):
                        removed_indexes_s = out_dict['removed_indexes_s']
                        removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                        masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                        self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break
                    
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score,
                    "offset":offset,
                    "merge_flag":merge_flag}


    def track_stepalign_vis(self, image, image_unalign, seq_name, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        merge_flag = False
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image_unalign, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        x_patch_arr = np.concatenate((x_patch_arr[:,:,:3], x_patch_arr[:, :, 3:6]), axis=2)
        search = self.preprocessor.process(x_patch_arr)
        
        templates = [self.z_tensor] if self.dynamic_z_tensor is None else \
            [self.z_tensor, self.dynamic_z_tensor]
        with torch.no_grad():
            x_tensor = search
            out_dict = self.network.forward(
                template=templates, search=x_tensor, ce_template_mask=self.box_mask_z)
        local_offset_norm = out_dict["offsets"][-1]
        
        local_offset_norm = offset_to_homography(local_offset_norm, local_offset_norm.device)
        local_offset = self.homo_norm.denormalize(local_offset_norm)

        # 提取 state (bbox中心点、大小)
        x1, y1, w, h = self.state  # 假设 state 是 [x, y, w, h]
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        patch_H, patch_W = self.params.search_size, self.params.search_size
        scale_x = w * self.params.search_factor / patch_W
        scale_y = h * self.params.search_factor / patch_H

        # 构造将 local patch 映射回全图的仿射矩阵
        # 从 crop 到原图的尺度 + 平移
        T_crop_to_global = np.array([
            [scale_x, 0, cx - (patch_W / 2) * scale_x],
            [0, scale_y, cy - (patch_H / 2) * scale_y],
            [0, 0, 1]
        ], dtype=np.float32)

        # 将 local_offset 应用于原图尺度
        local_offset_np = local_offset.cpu().numpy()
        offset = T_crop_to_global @ local_offset_np @ np.linalg.inv(T_crop_to_global)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # clip the box
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
        scores = pred_score_map.reshape(256).max()
        with torch.no_grad():
            if scores > self.alpha :
                image_aligned = get_align(image[:,:,:3], image[:, :, 3:6], np.squeeze(offset, axis=0)) # 使用现在预测出来的跟踪第一帧分数
                x_patch_arr_aligned, resize_factor_aligned, x_amask_arr_aligned = sample_target(image_aligned, self.state, self.params.template_factor,
                                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
                x_patch_arr, resize_factor, x_amask_arr_aligned = sample_target(image, self.state, self.params.template_factor,
                                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
                
                dynamic_z_tensor_aligned = self.preprocessor.process(x_patch_arr_aligned) 
                reversed_out_dict_aligned  = self.network.forward(
                    template=[dynamic_z_tensor_aligned], 
                    search=self.first_search,
                    ce_template_mask=self.box_mask_z)  
                
                dynamic_z_tensor = self.preprocessor.process(x_patch_arr) 
                reversed_out_dict  = self.network.forward(
                    template=[dynamic_z_tensor], 
                    search=self.first_search,
                    ce_template_mask=self.box_mask_z)  
                 
                pred_score_map_aligned = reversed_out_dict_aligned['score_map']
                response_aligned = self.output_window * pred_score_map_aligned
                pred_boxes_aligned = self.network.box_head.cal_bbox(response_aligned, reversed_out_dict_aligned['size_map'], reversed_out_dict_aligned['offset_map'])
                pred_boxes_aligned = pred_boxes_aligned.view(-1, 4)
                pred_box_aligned = (pred_boxes_aligned.mean(
                    dim=0) * self.params.search_size / self.fisrt_resize_factor_rgb).tolist()  # (cx, cy, w, h) [0,1]
                reversed_state_aligned = clip_box(self.map_box_back(pred_box_aligned, self.fisrt_resize_factor_rgb), H, W, margin=10)
                reversed_state_aligned = box_xywh_to_xyxy(torch.tensor(reversed_state_aligned).unsqueeze(0))
                
                pred_score_map = reversed_out_dict['score_map']
                response = self.output_window * pred_score_map
                pred_boxes = self.network.box_head.cal_bbox(response, reversed_out_dict['size_map'], reversed_out_dict['offset_map'])
                pred_boxes = pred_boxes.view(-1, 4)
                pred_box = (pred_boxes.mean(
                    dim=0) * self.params.search_size / self.fisrt_resize_factor_rgb).tolist()  # (cx, cy, w, h) [0,1]
                reversed_state = clip_box(self.map_box_back(pred_box, self.fisrt_resize_factor_rgb), H, W, margin=10)
                reversed_state = box_xywh_to_xyxy(torch.tensor(reversed_state).unsqueeze(0))
                
                init_bbox = box_xywh_to_xyxy(torch.tensor(self.init_bbox).unsqueeze(0))
                
                iou_scores_aligned,_ = box_iou(reversed_state_aligned, init_bbox)
                iou_scores,_ = box_iou(reversed_state, init_bbox)
                
                
                if iou_scores_aligned[0] >= iou_scores[0] and self.frame_id % self.updated_freq==0:
                    if iou_scores[0] > self.beta :
                        self.dynamic_z_tensor = dynamic_z_tensor_aligned
                    merge_flag = True
                elif iou_scores[0] > self.beta:
                    self.dynamic_z_tensor = dynamic_z_tensor
                    
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score,
                    "merge_flag":merge_flag}

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights

def get_tracker_class():
    return OSTrack

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize

def features_to_heatmap_mean(features, mode='mean'):
    # features: (B, H, W, D) 或 (B, D, H, W)
    if features.ndim == 4 and features.shape[-1] != features.shape[1]:
        # assume (B, H, W, D)
        feat = features
    elif features.ndim == 4 and features.shape[1] != features.shape[-1]:
        # (B, D, H, W) -> (B, H, W, D)
        feat = features.permute(0,2,3,1)
    else:
        raise ValueError("expected 4-d tensor")

    if mode == 'mean':
        heat = feat.mean(dim=-1)        # (B, H, W)
    elif mode == 'sum':
        heat = feat.sum(dim=-1)
    elif mode == 'l2':
        heat = torch.norm(feat, dim=-1) # L2 norm
    else:
        raise ValueError

    # normalize per-sample
    heat = heat - heat.view(heat.shape[0], -1).min(dim=1)[0].view(-1,1,1)
    heat = heat / (heat.view(heat.shape[0], -1).max(dim=1)[0].view(-1,1,1) + 1e-8)
    return heat  # (B, H, W)

# overlay helper
def overlay_heatmap_on_image(image, heatmap, alpha=0.5):
    # image: np.uint8 HxW x3, heatmap: float HxW in [0,1]
    import matplotlib.cm as cm
    cmap = cm.get_cmap('jet')
    heat_rgb = cmap(heatmap)[...,:3]  # H W 3
    overlay = (1-alpha)*image/255.0 + alpha*heat_rgb
    overlay = np.clip(overlay, 0,1)
    return (overlay*255).astype(np.uint8)

def save_visualization(ref_grid, samp_grid, samp_grid_flat, attn_w, Hf, Wf, save_path, prefix):
    """
    可视化参考网格、变形后采样区域、细化后采样区域和注意力权重，并保存为 PNG 文件
    
    参数:
    - ref_grid: 初始参考网格，形状 (B, L, 2)
    - samp_grid: 变形后采样区域，形状 (B, heads, L, K, 2)
    - samp_grid_flat: 细化后采样区域，形状 (B * heads, L * K, 1, 2)
    - attn_w: 注意力权重，形状 (B, L, heads, K)
    - Hf: 特征图高度
    - Wf: 特征图宽度
    - save_path: 保存图像的路径
    - prefix: 文件名前缀
    """
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    B, L, _ = ref_grid.shape
    K = samp_grid.shape[-2]

    # 1. 可视化参考网格 (ref_grid)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(ref_grid[0, :, 0].cpu().numpy(), ref_grid[0, :, 1].cpu().numpy(), c='r', label="Ref grid")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("Initial Sampling Grid (Ref Grid)")
    plt.legend()
    ref_grid_path = os.path.join(save_path, f"{prefix}_ref_grid.png")
    plt.savefig(ref_grid_path)
    plt.close()

    # 2. 可视化变形后采样区域 (samp_grid)
    plt.figure(figsize=(12, 6))
    for i in range(K):
        plt.scatter(samp_grid[0, 0, :, i, 0].cpu().numpy(), samp_grid[0, 0, :, i, 1].cpu().numpy(), label=f"Deformed Grid {i+1}")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("Deformed Sampling Grid")
    plt.legend()
    deformed_grid_path = os.path.join(save_path, f"{prefix}_deformed_grid.png")
    plt.savefig(deformed_grid_path)
    plt.close()

    # 3. 注意力权重可视化
    plt.figure(figsize=(12, 6))
    attn_w_reshaped = attn_w[0, :, 0, :].cpu().numpy()
    plt.imshow(attn_w_reshaped, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Attention Weights')
    plt.title("Attention Weights")
    plt.xlabel("Sampling Points")
    plt.ylabel("Query Tokens")
    attn_w_path = os.path.join(save_path, f"{prefix}_attn_weights.png")
    plt.savefig(attn_w_path)
    plt.close()

    # 4. 可视化细化后采样区域 (samp_grid_flat)
    plt.figure(figsize=(6, 6))
    for i in range(K):
        plt.scatter(samp_grid_flat[0, i, 0].cpu().numpy(), samp_grid_flat[0, i, 1].cpu().numpy(), label=f"Refined Grid {i+1}")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("Refined Sampling Grid (samp_grid_flat)")
    plt.legend()
    refined_grid_path = os.path.join(save_path, f"{prefix}_refined_grid.png")
    plt.savefig(refined_grid_path)
    plt.close()

def FeatureMapVisible(outputs):
    outputs = (outputs ** 2).sum(1)
    b, h, w = outputs.size()
    outputs = outputs.view(b, h * w)
    outputs = F.normalize(outputs, p=2, dim=1)
    outputs = outputs.view(b, h, w)
    for j in range(outputs.size(0)):
        am = outputs[j, ...].cpu().numpy()
        am = cv2.resize(am, (256, 256))
        am = 255 * (am - np.min(am)) / (
                np.max(am) - np.min(am) + 1e-12
        )
        am = np.uint8(np.floor(am))
        
        am=np.stack((am,am,am),axis=0)
        am=np.transpose(am,(1,2,0))  # 这里转换通道为RGB

        am=cv2.applyColorMap(am,cv2.COLORMAP_JET)

        return am