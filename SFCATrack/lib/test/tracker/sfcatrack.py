import math
from lib.models.sfcatrack import build_sfcatrack
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
from lib.test.utils.bbox_align import align_targets
from lib.test.tracker.data_utils import PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import visdom
import numpy as np
import os

def inverse_transform(pred_bias, rgb_bbox):
    
    if not isinstance(pred_bias, torch.Tensor):
        raise TypeError(f"pred_bias must be torch.Tensor, got {type(pred_bias)}")
    if pred_bias.ndim != 2 or pred_bias.shape[1] != 4:
        raise ValueError("pred_bias must have shape (B, 4)")
    
    if not isinstance(rgb_bbox, torch.Tensor):
        rgb_bbox = torch.tensor(rgb_bbox, dtype=torch.float32, device=pred_bias.device)
    
    if (rgb_bbox[..., 2:] <= 0).any():  
        invalid_idx = torch.where(rgb_bbox[..., 2:] <= 0)
        raise ValueError(f"Invalid RGB bbox at index {invalid_idx}: width/height <=0")
    # print('pred_bias:', pred_bias)
    # print('rgb_bbox:', rgb_bbox)

    
    rgb_bbox = torch.tensor(rgb_bbox, dtype=torch.float32).to(pred_bias.device)
    # print('rgb_bbox_tensor:', rgb_bbox)
    
    rgb_norm = rgb_bbox / torch.tensor([1920, 1080, 1920, 1080], device=pred_bias.device)
    # print('rgb_norm:', rgb_norm)
    x_rgb_norm, y_rgb_norm, w_rgb_norm, h_rgb_norm = rgb_norm.unbind(-1)
    
    rgb_bbox_norm = torch.stack([x_rgb_norm, y_rgb_norm, w_rgb_norm, h_rgb_norm], dim=-1)
    rgb_bbox = rgb_bbox_norm * torch.tensor([1920, 1080, 1920, 1080], device=pred_bias.device)
    
    delta_x1, delta_y1, delta_x2, delta_y2 = pred_bias.unbind(-1)
    
    x1_ir_norm = x_rgb_norm - delta_x1
    y1_ir_norm = y_rgb_norm - delta_y1
    x2_ir_norm = x_rgb_norm + w_rgb_norm  - delta_x2
    y2_ir_norm = y_rgb_norm + h_rgb_norm  - delta_y2
    # w_ir_norm = w_rgb_norm.expand_as(x_ir_norm)
    # h_ir_norm = h_rgb_norm.expand_as(x_ir_norm)
    
    ir_bbox_norm = torch.stack([x1_ir_norm, y1_ir_norm, x2_ir_norm - x1_ir_norm, y2_ir_norm - y1_ir_norm], dim=-1)
    ir_bbox = ir_bbox_norm * torch.tensor([1920, 1080, 1920, 1080], device=pred_bias.device)
    if ir_bbox[..., 2:].min() < 1:
        # print('ir_bbox:', ir_bbox)
        # print('rgb_bbox:', rgb_bbox)
        return rgb_bbox
    
    # print('ir_bbox:', ir_bbox)
    return ir_bbox[0]

def transform_ir_image(rgb_img, ir_img, src, dst, frame_id, seq_name, out_size=(1920, 1080)):
    
    def generate_corners(src, dst):
        
        src_x, src_y, src_w, src_h = src
        dst_x, dst_y, dst_w, dst_h = dst
        src_corners = np.array([
            [src_x, src_y],  # 左上 (x1, y1)
            [src_x, src_y + src_h],  # 左下 (x1, y2)
            [src_x + src_w, src_y + src_h],  # 右下 (x2, y2)
            [src_x + src_w, src_y]  # 右上 (x2, y1)
        ], dtype=np.float32)

        
        dst_corners = np.array([
            [dst_x, dst_y],  # 左上 (x1, y1)
            [dst_x, dst_y + dst_h],  # 左下 (x1, y2)
            [dst_x + dst_w, dst_y + dst_h],  # 右下 (x2, y2)
            [dst_x + dst_w, dst_y]  # 右上 (x2, y1)
        ])

        return src_corners, dst_corners

    def compute_scaling_translation_matrix(src_pts, dst_pts):
        
        s_x = (dst_pts[2][0] - dst_pts[0][0]) / (src_pts[2][0] - src_pts[0][0])
        s_y = (dst_pts[2][1] - dst_pts[0][1]) / (src_pts[2][1] - src_pts[0][1])

        
        t_x = dst_pts[0][0] - src_pts[0][0] * s_x
        t_y = dst_pts[0][1] - src_pts[0][1] * s_y

        
        M = np.array([
            [s_x, 0, t_x],
            [0, s_y, t_y]
        ], dtype=np.float32)

        return M

    
    src_pts, dst_pts = generate_corners(src, dst)
    M = compute_scaling_translation_matrix(src_pts, dst_pts)
    transformed_ir = cv2.warpAffine(ir_img, M, out_size)

    return transformed_ir

class SFCATrack(BaseTracker):
    def __init__(self, params, test_mode):
        super(SFCATrack, self).__init__(params)
        network = build_sfcatrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)  
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorMM()
        self.state = None
        self.state_bias = None
        self.seq_name = params.seq_name
        self.bias = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        if getattr(params, 'debug', None) is None:
            setattr(params, 'debug', 0)
        self.use_visdom = True #params.debug   
        #self._init_visdom(None, 1)
        self.debug = params.debug
        if self.debug:
            self._init_visdom(None, 1)
        self.frame_id = 0
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        
        self.search_prior = None
        
        self.test_mode = test_mode
        
        self.rgb_bbox_0 = None
        self.ir_bbox_0 = None

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        with torch.no_grad():
            self.z_tensor = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)

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
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            x_tensor = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z)
            

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

        #self.debug = 1
        
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
                self.visdom.register((image[:,:,:3], info['gt_bbox'], self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr[:,:,:3]).permute(2,0,1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr[:,:,:3]).permute(2,0,1), 'image', 1, 'template_rgb')
                self.visdom.register(torch.from_numpy(self.z_patch_arr[:,:,3:]).permute(2,0,1), 'image', 1, 'template_tir')
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
    def track_notalign(self, image, seq_name, info: dict = None):
        H, W, _ = image.shape
        # image = align_targets(image, self.rgb_bbox_0, self.ir_bbox_0)
        self.frame_id += 1
        # transform_ir_image(image[:,:,:3], image[:,:,3:], self.state_bias, self.state, self.frame_id, seq_name)
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        x_patch_arr_bias, resize_factor_bias, x_amask_arr_bias = sample_target(image, self.state_bias, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        
        x_patch_arr = np.concatenate((x_patch_arr[:,:,:3], x_patch_arr_bias[:, :, 3:]), axis=2)
        search = self.preprocessor.process(x_patch_arr)
        

        with torch.no_grad():
            x_tensor = search
            # merge the template and the search
            # run the transformer
            # out_dict = self.network.forward(
            #     template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z)
            bias = self.network.forward(template=self.z_tensor, search=x_tensor, search_prior=self.search_prior, ce_template_mask=self.box_mask_z, stage='phase1')
            # print(bias)
            out_dict, aux_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor, search_prior=None, ce_template_mask=self.box_mask_z)
        self.search_prior = search
        


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
        # get the final box result
        self.bias = bias
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        self.state_bias = inverse_transform(bias, self.state).cpu().numpy().tolist()
        
        
        
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


def get_tracker_class():
    return BATTrack
