import pdb, cv2
import torch.nn as nn
import torch
from torch.functional import F
from . import BaseActor
from .homography_compute import HomographyBatch, HomographyNormalizer, SimpleHomoNormalizer
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from ...utils.slove import DLT_solve, transform
from lib.train.admin import multigpu
from PIL import Image
import os
import numpy as np
import math
from ..data.processing_utils import transform_image_to_crop
from .offset_loss import OffsetLoss
import random
# 常量（确认）
W_rgb, H_rgb = 1920, 1080
W_tir, H_tir = 640, 512
PATCH_W, PATCH_H = 256, 256   # 你的 search patch 尺寸
import json
def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return x

def save_small_csv(aux, outdir="debug_dump", prefix="step0001"):
    os.makedirs(outdir, exist_ok=True)
    if "gate_logits" in aux:
        np.savetxt(os.path.join(outdir, f"{prefix}_gate_logits.csv"),
                   _to_numpy(aux["gate_logits"]),
                   delimiter=",", fmt="%.6f",
                   header="p_shift,p_scale,p_warp", comments="")
    if "experts" in aux:  # [B,3,4] -> 展平为 [B,12] 方便看
        e = _to_numpy(aux["experts"]).reshape(aux["experts"].shape[0], -1)
        header = ",".join([f"E{ei}_d{dj}" for ei in range(3) for dj in range(4)])
        np.savetxt(os.path.join(outdir, f"{prefix}_experts.csv"),
                   e, delimiter=",", fmt="%.6f", header=header, comments="")


def offset_to_homography(offset, device):
    B = offset.shape[0]

    dx, dy, sx, sy = offset[:,0], offset[:,1], offset[:,2], offset[:,3]
    H = torch.eye(3, device=device).unsqueeze(0).repeat(B,1,1)
    H[:,0,0] = sx
    H[:,1,1] = sy
    H[:,0,2] = dx
    H[:,1,2] = dy
    
    return H

def homography_to_offset(H):
    delta = H[:, :2, 2]  # Δx, Δy
    scale_x = H[:,0,0]
    scale_y = H[:,1,1]
    scale = torch.stack([scale_x, scale_y], dim=1)
    offset = torch.cat([delta, scale], dim=1)  # [B,4]
    return offset

def denorm_tensor_to_uint8(img_tensor, mean, std, mode='RGB'):
    """
    img_tensor: torch.Tensor CxHxW, dtype float, normalized
    mean, std: list or torch.Tensor, length 3 (RGB) or 1 (TIR)
    mode: 'RGB' or 'TIR'
    returns: HxWx3 uint8 image (BGR for cv2)
    """
    img = img_tensor.clone().cpu()

    if mode == 'RGB':
        # RGB 去归一化
        if isinstance(mean, torch.Tensor):
            mean_t = mean.view(-1,1,1)
            std_t = std.view(-1,1,1)
        else:
            mean_t = torch.tensor(mean).view(-1,1,1)
            std_t = torch.tensor(std).view(-1,1,1)
        img = img * std_t + mean_t
        img = img.clamp(0.0, 1.0)
        img = (img * 255.0).permute(1,2,0).numpy().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    elif mode == 'TIR':
        # TIR 单通道去归一化
        if isinstance(mean, torch.Tensor):
            mean_t = mean.view(-1,1,1)
            std_t = std.view(-1,1,1)
        else:
            mean_t = torch.tensor(mean).view(-1,1,1)
            std_t = torch.tensor(std).view(-1,1,1)
        img = img * std_t + mean_t
        img = img.clamp(0.0, 1.0)
        img = (img * 255.0).squeeze().numpy().astype(np.uint8)  # HxW
        img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)  # 也可以用 COLORMAP_HO

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return img


def save_tensor_image(tensor_img, mean, std, save_path, mode='RGB'):
    """
    tensor_img: torch.Tensor (C,H,W), normalized
    mean, std: list or torch.Tensor
    mode: 'RGB' or 'TIR'
    """
    img = tensor_img.clone()

    if mode == 'RGB':
        # 反标准化
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        # 裁剪到 [0,1]
        img = torch.clamp(img, 0, 1)
        # 转 numpy HWC
        img_np = img.permute(1,2,0).cpu().numpy()
        # 转 0-255 uint8
        img_np = (img_np * 255).astype(np.uint8)
        # 保存
        Image.fromarray(img_np).save(save_path)

    elif mode == 'TIR':
        # 如果多通道, 取第一个通道或均值
        if img.shape[0] > 1:
            img = img.mean(dim=0, keepdim=True)
        # 反标准化
        t = img[0]
        if isinstance(mean, torch.Tensor):
            mean_t = mean.view(-1,1,1)
            std_t = std.view(-1,1,1)
        else:
            mean_t = torch.tensor(mean).view(-1,1,1)
            std_t = torch.tensor(std).view(-1,1,1)
        img = img * std_t + mean_t
        img = img.clamp(0.0, 1.0)
        img = (img * 255.0).squeeze().numpy().astype(np.uint8)  # HxW
        # 生成红-白热力图
        img_color = cv2.applyColorMap(img, cv2.COLORMAP_HOT)  # BGR
        # 保存
        cv2.imwrite(save_path, img_color)

    else:
        raise ValueError("mode must be 'RGB' or 'TIR'")
    
def warp_bbox(bbox, homo_mat, img_w, img_h):
    """
    bbox: [x1, y1, x2, y2] 归一化坐标 (0~1)
    homo_mat: 3x3 单应性矩阵 (torch.Tensor 或 np.ndarray)
    img_w, img_h: 图像的宽高
    
    return: 变换后的 bbox [x1', y1', x2', y2']，像素坐标
    """
    # 如果是 torch.Tensor 转 numpy
    if isinstance(homo_mat, torch.Tensor):
        homo_mat = homo_mat.detach().cpu().numpy().astype(np.float32)
    homo_mat = np.linalg.inv(homo_mat)
    # bbox 转成像素坐标
    x, y, w, h = bbox.detach().cpu().numpy()
    x1, y1 = x * img_w, y * img_h
    x2, y2 = (x + w) * img_w, (y + h) * img_h
    
    # 四个角点
    pts = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.float32).reshape(-1, 1, 2)  # (4,1,2)
    
    # 透视变换
    warped_pts = cv2.perspectiveTransform(pts, homo_mat)  # (4,1,2)
    warped_pts = warped_pts.reshape(-1, 2)  # (4,2)
    
    # 得到新的 bbox
    x_min, y_min = warped_pts[:,0].min(), warped_pts[:,1].min()
    x_max, y_max = warped_pts[:,0].max(), warped_pts[:,1].max()
    
    return [x_min, y_min, x_max, y_max]

class OSTrack_Actor(BaseActor):
    """ Actor for training BAT models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.bias_loss = OffsetLoss(loss_type="SmoothL1", beta=1)
        self.search_scale_factor = self.cfg.DATA.SEARCH.FACTOR
        self.homo_compute = HomographyBatch()
        self.homo_norm = SimpleHomoNormalizer(self.cfg.DATA.SEARCH.SIZE, self.cfg.DATA.SEARCH.SIZE, margin_ratio=0.0)
        
    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.box_head.apply(self.fix_bn)

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data, None)

        return loss, status

    def visible_homo_search(self, data, search_img, homo_gt_search, endfix="", epoch=0, out_dict=None):
        mean = torch.tensor(self.cfg.DATA.MEAN)
        std = torch.tensor(self.cfg.DATA.STD)
        B = search_img.size(0)
        seq_names = data["seq_name"]  # list/tuple 长度为 B

        # ---- 仅当有 out_dict 时记录 JSONL ----
        if out_dict is not None and "offset_experts" in out_dict and "offsets" in out_dict:
            
            save_dir = f'./debug_search_img_warped_lasher/Epoch_{str(epoch).zfill(2)}'
            os.makedirs(save_dir, exist_ok=True)
            log_path = os.path.join(save_dir, f"offsets.log")  # 支持 endfix
            
            # 一次性打开文件，按 JSONL 写入
            with open(log_path, "a", encoding="utf-8") as f:
                num_layers = len(out_dict["offset_experts"])
                for lidx in range(num_layers):
                    experts_layer = out_dict["offset_experts"][lidx]  # [B, 3, 4]
                    offsets_layer = out_dict["offsets"][lidx]          # [B, 4]
                    gate_logits = out_dict["gate_logits"][lidx]
                    gt = homography_to_offset(data["homo_gt_search"])
                    # 如果还没转成 list（例如是 tensor），这里兜底转一下
                    if hasattr(experts_layer, "detach"):
                        experts_layer = experts_layer.detach().cpu().tolist()
                    if hasattr(offsets_layer, "detach"):
                        offsets_layer = offsets_layer.detach().cpu().tolist()
                    if hasattr(gt, "detach"):
                        gt = gt.detach().cpu().tolist()
                    if hasattr(gate_logits, "detach"):
                        gate_logits = gate_logits.detach().cpu().tolist()
                    for bidx in range(B):
                        row = {
                            "epoch": int(epoch),
                            "seq_name": str(seq_names[bidx]),
                            "layer": int(lidx),
                            "gate_logits":gate_logits[bidx],
                            "ground_truth": gt[bidx],
                            "offset": offsets_layer[bidx],          # [4]
                            "offset_experts": experts_layer[bidx],  # [3, 4]
                        }
                        json.dump(row, f, ensure_ascii=False)
                        f.write("\n")
                f.flush()
                
        for idx in range(search_img.size(0)):
            seq_name = data["seq_name"][idx]
            save_dir = f'./debug_search_img_warped_SynMU/Epoch_{str(epoch).zfill(2)}/{seq_name}'
            os.makedirs(save_dir, exist_ok=True)
            search_frame_ids = data["search_frame_ids"][idx]
            tir_tensor = search_img[idx].clone()[3:6, :, :].cpu()  # C x H x W
            # tir_align_tensor = search_img[idx].clone()[6:9, :, :].cpu()  # C x H x W
            tir_np = denorm_tensor_to_uint8(tir_tensor, mean, std, mode='RGB')  # H x W x 3 (BGR)
            rgb_tensor = search_img[idx].clone()[:3,:,:].cpu()
            rgb_np = denorm_tensor_to_uint8(rgb_tensor, mean, std)
            # tir_align_np = denorm_tensor_to_uint8(tir_align_tensor, mean, std)
            # 转 numpy (确保是 float32，OpenCV 推荐 float32)
            homo_mat = homo_gt_search[idx]
            if isinstance(homo_mat, torch.Tensor):
                homo_mat = homo_mat.detach().cpu().numpy().astype(np.float32)
            rgb_bbox = data['search_anno'][-1][idx]
            tir_bbox = warp_bbox(rgb_bbox, homo_mat, PATCH_W, PATCH_H)
            
            # (1) 原始 RGB BBox (要从归一化坐标还原到像素)
            x1, y1, w, h = rgb_bbox
            x1, x2 = int(x1 * PATCH_W), int((x1+w) * PATCH_W)
            y1, y2 = int(y1 * PATCH_H), int((y1+h) * PATCH_H)
            cv2.rectangle(rgb_np, (x1, y1), (x2, y2), (0,255,0), 2)  # 绿色框: RGB bbox
            # (2) Warp 后的 TIR BBox
            tx1, ty1, tx2, ty2 = map(int, tir_bbox)
            cv2.rectangle(tir_np, (tx1, ty1), (tx2, ty2), (0,0,255), 2)  # 红色框: TIR bbox

            warped_tir = cv2.warpPerspective(tir_np, homo_mat, (PATCH_W, PATCH_H),
                                            flags=cv2.INTER_NEAREST, 
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=(0,0,0))
            overlay = cv2.addWeighted(rgb_np, 0.4, warped_tir, 0.6, 0)
            overlay_path = os.path.join(save_dir, f'{seq_name}_{search_frame_ids}_overlay{endfix}.png')
            cv2.imwrite(overlay_path, overlay)
            
            if endfix == '_groundtruth':
                overlay = cv2.addWeighted(rgb_np, 0.4, tir_np, 0.9, 0)
                overlay_path = os.path.join(save_dir, f'{seq_name}_{search_frame_ids}_overlay_withoutalign.png')
                cv2.imwrite(overlay_path, overlay)
                cv2.imwrite(os.path.join(save_dir, f'{seq_name}_{search_frame_ids}_rgb.png'), rgb_np)
                cv2.imwrite(os.path.join(save_dir, f'{seq_name}_{search_frame_ids}_tir.png'), tir_np)

    def get_bboxes_masks(self, bboxes, B, H, W, patch_size=16):
        # 计算grid大小
        grid_size = H // patch_size

        # 初始化mask，大小为[B, grid_size, grid_size]
        bboxes_masks = torch.zeros(B, grid_size, grid_size, dtype=torch.bool, device=bboxes.device)

        # 获取所有bbox的归一化坐标
        x1, y1, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        # 将归一化坐标转换为像素坐标
        x1_pixel = x1
        y1_pixel = y1
        w_pixel = w
        h_pixel = h
        x2_pixel = x1_pixel + w_pixel
        y2_pixel = y1_pixel + h_pixel

        # 计算patch的索引
        patch_x1 = (x1_pixel // patch_size).long()
        patch_y1 = (y1_pixel // patch_size).long()
        patch_x2 = (x2_pixel // patch_size).long()
        patch_y2 = (y2_pixel // patch_size).long()

        # 手动限制索引在有效范围内 (0 - grid_size - 1)
        patch_x1 = torch.clamp(patch_x1, 0, grid_size - 1)
        patch_y1 = torch.clamp(patch_y1, 0, grid_size - 1)
        patch_x2 = torch.clamp(patch_x2, 0, grid_size - 1)
        patch_y2 = torch.clamp(patch_y2, 0, grid_size - 1)

        # 使用广播将bbox位置标记在mask中
        for b in range(B):
            bboxes_masks[b, patch_y1[b]:patch_y2[b] + 1, patch_x1[b]:patch_x2[b] + 1] = True

        return bboxes_masks     
      
    def forward_pass(self, data):
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 6, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 6, 320, 320)
        # search_img_ir = data['search_images_ir'][0].view(-1, *data['search_images_ir'].shape[2:]) 

        rgb_boxes = data["oringin_rgb_anno"][0] # torch.Size([32, 4]) X1,Y1,W,H 绝对坐标
        tir_boxes = data["oringin_ir_anno_bias"][0] # torch.Size([32, 4]) X1,Y1,W,H 绝对坐标
        is_flip_batch = data["search_is_flip_rgb"]
        is_flip_first_batch = data["search_is_flip_rgb_first"]
        jit_rgbs = data["search_jit_anno_rgb"][-1].clone().float()
        crop_szs = data["search_crop_sz_rgb"][-1]
        # Runner_4 548
        homo_gt_full, homo_gt_search, rgb_in_patch, tir_in_patch = self.homo_compute.compute_batch(rgb_boxes, tir_boxes, jit_rgbs, crop_szs,
                                                         is_flip_batch, is_flip_first_batch, data['W_RGB'], data['H_RGB'])
        if isinstance(homo_gt_full, list):
            homo_gt_full = torch.stack([torch.from_numpy(h).float() for h in homo_gt_full], dim=0).to(search_img.device)
        if isinstance(homo_gt_search, list):
            homo_gt_search = torch.stack([torch.from_numpy(h).float() for h in homo_gt_search], dim=0).to(search_img.device)

        data["homo_gt_full"] = self.homo_norm.normalize(homo_gt_full) 
        data["homo_gt_search"] = self.homo_norm.normalize(homo_gt_search)
        data["rgb_search_masks"] = self.get_bboxes_masks(rgb_in_patch, rgb_in_patch.shape[0], 256, 256, 1)
        data["tir_search_masks"] = self.get_bboxes_masks(tir_in_patch, tir_in_patch.shape[0], 256, 256, 1)
        
        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])
            
            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]
        
        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False,
                            offset_gt=homography_to_offset(data["homo_gt_search"])
                            )
        return out_dict


    def compute_losses(self, pred_dict, gt_dict, bias, return_status=True):
        if pred_dict is not None:
            # gt gaussian map
            gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            #gt_bbox_ir = gt_dict['search_anno_ir'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            
            gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
            gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # (B,1,H,W)

            # Get boxes
            pred_boxes = pred_dict['pred_boxes']
            
            
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                            max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
            # compute giou and iou
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            # compute location loss
            if 'score_map' in pred_dict:
                location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)
            
            
            delta_gt = gt_dict["homo_gt_search"][:, :2, 2]  # Δx, Δy
            scale_x = gt_dict["homo_gt_search"][:,0,0]
            scale_y = gt_dict["homo_gt_search"][:,1,1]
            scale_gt = torch.stack([scale_x, scale_y], dim=1)
            # target = torch.cat([delta_gt, scale_gt], dim=1)  # [B,4]
            layer_loss_w = [0.25, 0.5, 1.0]
            layer_loss_scale_w = [0., 0.5, 1.0]
            offset_loss = torch.tensor(0.0, device=l1_loss.device)
            scale_loss = torch.tensor(0.0, device=l1_loss.device)
            offsets = pred_dict["offsets"]
            for lidx, shift_offset in enumerate(offsets):
                offset_loss = offset_loss + layer_loss_w[lidx] * self.bias_loss(shift_offset[:,:2], delta_gt)
                scale_loss = scale_loss + layer_loss_scale_w[lidx] * self.bias_loss(shift_offset[:,2:], scale_gt)
            offset_loss = offset_loss/len(offsets)
            scale_loss = scale_loss/len(offsets)
            
            offset_experts = pred_dict["offset_experts"]
            B,N,_ = pred_dict["offset_experts"][0].shape
            delta_gt = delta_gt.unsqueeze(1).expand(-1, N, -1)
            scale_gt = scale_gt.unsqueeze(1).expand(-1, N, -1)
            offset_expert_loss = torch.tensor(0.0, device=l1_loss.device)
            scale_expert_loss = torch.tensor(0.0, device=l1_loss.device)
            for lidx, offset in enumerate(offset_experts):
                offset_expert_loss = offset_expert_loss + layer_loss_w[lidx] * self.bias_loss(offset[:, :, :2], delta_gt)
                scale_expert_loss = scale_expert_loss + layer_loss_scale_w[lidx] * self.bias_loss(offset[:, :, 2:], scale_gt)
                
            offset_expert_loss = offset_expert_loss/(N*len(offset_experts))
            scale_expert_loss = scale_expert_loss/(N*len(offset_experts))
            
            router_loss = sum(pred_dict['router_loss'])
            localizaion_loss_v =  torch.zeros(1, device=l1_loss.device, requires_grad=True)
            localizaion_loss_i =  torch.zeros(1, device=l1_loss.device, requires_grad=True)
            B = gt_dict["rgb_search_masks"].shape[0]
            for attn_weights_v, attn_weights_i in zip(pred_dict["target_respones_v"], pred_dict["target_respones_i"]):
                attn_weights_v = attn_weights_v.mean(dim=1).reshape(B, 16, 16)
                attn_weights_i = attn_weights_i.mean(dim=1).reshape(B, 16, 16)
                # attn_weights_v: (B, 16, 16)
                attn_weights_v_up = F.interpolate(
                    attn_weights_v.unsqueeze(1),  # -> (B,1,16,16)，加一个通道维度
                    size=(256, 256),              # 目标大小
                    mode='bilinear',              # 插值方式，可选 'nearest' / 'bicubic' 等
                    align_corners=False
                ).squeeze(1)                      # -> (B,256,256)
                # 同理对 attn_weights_i
                attn_weights_i_up = F.interpolate(
                    attn_weights_i.unsqueeze(1),
                    size=(256, 256),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                search_loc_labels_v = gt_dict["rgb_search_masks"].float()
                search_loc_labels_i = gt_dict["tir_search_masks"].float()
                localizaion_loss_v = localizaion_loss_v + F.binary_cross_entropy_with_logits(attn_weights_v_up.view(B,-1), search_loc_labels_v.view(B,-1))
                localizaion_loss_i = localizaion_loss_i + F.binary_cross_entropy_with_logits(attn_weights_i_up.view(B,-1), search_loc_labels_i.view(B,-1))
                localizaion_loss_v = localizaion_loss_v/len(pred_dict["target_respones_v"])
                localizaion_loss_i = localizaion_loss_i/len(pred_dict["target_respones_v"])
            
            if gt_dict['epoch'] == 1:
                localizaion_w = 0.005
            elif gt_dict['epoch'] <=5:
                localizaion_w = 0.1
            else:
                localizaion_w = 3.0
            
            off_w = 20.0
            scale_w = 40.0
            exp_off_w = 40.0
            exp_scale_w = 80.0
            router_w = 2.0
            loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + \
                offset_loss * off_w + scale_loss * scale_w + offset_expert_loss * exp_off_w + scale_expert_loss * exp_scale_w + router_loss * router_w + \
                localizaion_w * localizaion_loss_v + localizaion_w * localizaion_loss_i
            if return_status:
                # status for log
                mean_iou = iou.detach().mean()
                status = {"Loss/total": loss.item(),
                          "Loss/giou": giou_loss.item(),
                          "Loss/l1": l1_loss.item(),
                          "Loss/location": location_loss.item(),
                          "Loss/off": offset_loss.item() * off_w,
                          "Loss/scale": scale_loss.item() * scale_w,
                          "Loss/off_exp": offset_expert_loss.item() * exp_off_w,
                          "Loss/scale_exp": scale_expert_loss.item()* exp_scale_w,
                          "Loss/router": router_loss.item() * router_w,
                          "Loss/loc": ((localizaion_loss_v + localizaion_loss_i)/2).item(),
                          "IoU": mean_iou.item(),
                                    }
                return loss, status
            else:
                return loss