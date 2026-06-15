import pdb, cv2
import torch.nn as nn
import torch
from torch.functional import F

from . import BaseActor
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
# 常量（确认）
W_rgb, H_rgb = 1920, 1080
W_tir, H_tir = 640, 512
PATCH_W, PATCH_H = 256, 256   # 你的 search patch 尺寸

def crop_to_patch_matrix(center, win_size, out_size=256):
    Wc, Hc = win_size
    sx, sy = out_size / Wc, out_size / Hc
    x0, y0 = center[0] - Wc/2.0, center[1] - Hc/2.0
    return np.array([[sx, 0,  -sx * x0],
                     [0,  sy, -sy * y0],
                     [0,  0,   1      ]], dtype=np.float64)

def rect_corners(x, y, w, h):
    return np.array([[x,     y    ],
                     [x+w,   y    ],
                     [x+w,   y+h  ],
                     [x,     y+h  ]], dtype=np.float64)
    
def clamp_int(v, low, high):
    return int(max(low, min(high, int(round(v)))))

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
            m_t = mean[0] if mean.numel()>1 else mean
            s_t = std[0] if std.numel()>1 else std
        else:
            m_t = mean[0] if len(mean)>1 else mean[0]
            s_t = std[0] if len(std)>1 else std[0]
        t.mul_(s_t).add_(m_t)
        # 裁剪到 [0,1]
        t = torch.clamp(t, 0, 1)
        # 转 numpy
        img_gray = (t.cpu().numpy() * 255).astype(np.uint8)
        # 生成红-白热力图
        img_color = cv2.applyColorMap(img_gray, cv2.COLORMAP_HOT)  # BGR
        # 保存
        cv2.imwrite(save_path, img_color)

    else:
        raise ValueError("mode must be 'RGB' or 'TIR'")

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
        # TIR 一般单通道, 取第一通道即可
        if img.shape[0] > 1:
            img = img[0:1,:,:]  # 保证单通道
        # 去归一化
        if isinstance(mean, torch.Tensor):
            mean_t = mean.view(-1,1,1)
            std_t = std.view(-1,1,1)
        else:
            mean_t = torch.tensor(mean).view(-1,1,1)
            std_t = torch.tensor(std).view(-1,1,1)
        img = img * std_t + mean_t
        img = img.clamp(0.0, 1.0)
        img_gray = (img[0].numpy() * 255.0).astype(np.uint8)  # HxW
        # apply "hot" colormap: 红色->黄色->白色
        img = cv2.applyColorMap(img_gray, cv2.COLORMAP_HOT)  # BGR
    else:
        raise ValueError("mode must be 'RGB' or 'TIR'")
    
    return img


def calculate_offset(gt_bbox_rgb, gt_bbox_ir):

    assert gt_bbox_rgb.shape[0] == 1 and gt_bbox_ir.shape[0] == 1, "Batch维度应为1"
    B = gt_bbox_rgb.shape[1]

    rgb_norm = gt_bbox_rgb.squeeze(0) / torch.tensor([1920, 1080, 1920, 1080], device=gt_bbox_rgb.device)
    ir_norm = gt_bbox_ir.squeeze(0) / torch.tensor([640, 512, 640, 512], device=gt_bbox_ir.device)
    
    x1_rgb = rgb_norm[:, 0]
    y1_rgb = rgb_norm[:, 1]
    x2_rgb = x1_rgb + rgb_norm[:, 2]
    y2_rgb = y1_rgb + rgb_norm[:, 3]

    x1_ir = ir_norm[:, 0]
    y1_ir = ir_norm[:, 1]
    x2_ir = x1_ir + ir_norm[:, 2]
    y2_ir = y1_ir + ir_norm[:, 3]

    delta_x1 = x1_rgb - x1_ir
    delta_y1 = y1_rgb - y1_ir
    delta_x2 = x2_rgb - x2_ir
    delta_y2 = y2_rgb - y2_ir
    
    # target = torch.stack([delta_x, delta_y], dim=1)
    target = torch.stack([delta_x1, delta_y1, delta_x2, delta_y2], dim=1)
    return target

class OffsetLoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.criterion = nn.L1Loss() #nn.SmoothL1Loss(beta=beta)

    def forward(self, pred_bias, gt_bbox_rgb, gt_bbox_ir):
        target = calculate_offset(gt_bbox_rgb, gt_bbox_ir)
        # print("pred_bias,target:", pred_bias, target)
        bias_loss = self.criterion(pred_bias, target)
        return bias_loss

    
class Baseline_Actor(BaseActor):
    """ Actor for training BAT models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.bias_loss = OffsetLoss()
        self.search_scale_factor = self.cfg.DATA.SEARCH.FACTOR
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
    
    def forward_pass(self, data):
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 6, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 6, 320, 320)
        search_img_prior = None
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
            # ce_keep_rate = 0.7

        if len(template_list) == 1:
            template_list = template_list[0]
        
        out_dict = self.net(template=template_list,
                            search=search_img,
                            search_prior=search_img_prior,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False
                            )
        return out_dict


    def compute_losses(self, pred_dict, gt_dict, bias, return_status=True):
        if pred_dict is not None:
            # gt gaussian map
            gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            # gt_bbox_ir = gt_dict['search_anno_ir'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            
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
            loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
            # bias_loss= self.bias_loss(bias, gt_dict['oringin_rgb_anno'], gt_dict['oringin_ir_anno_bias'])
            if return_status:
                # status for log
                mean_iou = iou.detach().mean()
                status = {"Loss/total": loss.item(),
                                    "Loss/giou": giou_loss.item(),
                                    "Loss/l1": l1_loss.item(),
                                    "Loss/location": location_loss.item(),
                                    "IoU": mean_iou.item(),
                                    }
                return loss, status
            else:
                return loss