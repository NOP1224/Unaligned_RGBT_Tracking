import pdb, cv2
import torch.nn as nn
import torch
from torch.functional import F

from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from lib.train.admin import multigpu


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

    
class SFCAActor(BaseActor):
    """ Actor for training BAT models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.bias_loss = OffsetLoss()

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
        # forward pass
        if self.cfg.TRAIN.PROMPT.TYPE == "att_moe_phase1" or self.cfg.TRAIN.PROMPT.TYPE == "att_moe_phase2" or self.cfg.TRAIN.PROMPT.TYPE == "att_moe_phase3":
            bias = self.forward_pass(data, self.cfg.TRAIN.PROMPT.TYPE, self.cfg.TRAIN.EXPERT_INDEX)
            loss, status = self.compute_losses(None, data, bias)
        else:
            out_dict = self.forward_pass(data, self.cfg.TRAIN.PROMPT.TYPE, self.cfg.TRAIN.EXPERT_INDEX)
            loss, status = self.compute_losses(out_dict, data, None)
        

        # compute losses
        # loss, status = self.compute_losses(out_dict, data, bias)

        return loss, status

    def forward_pass(self, data, phase, expert_index):
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 6, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 6, 320, 320)
        search_img_prior = data['search_images_prior'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 6, 320, 320)

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
        
        if phase == "att_moe_phase1" or phase == "att_moe_phase2" or phase == "att_moe_phase3":
            bias = self.net(template=template_list,
                                search=search_img,
                                search_prior=search_img_prior,
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False,
                                stage=phase,
                                expert_index=expert_index
                                )
            return bias
        else:
            out_dict = self.net(template=template_list,
                                search=search_img,
                                search_prior=search_img_prior,
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False,
                                stage=phase,
                                expert_index=expert_index
                                )
            return out_dict

        # return out_dict, bias

    def compute_losses(self, pred_dict, gt_dict, bias, return_status=True):
        if pred_dict is not None:
            # gt gaussian map
            gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            gt_bbox_ir = gt_dict['search_anno_ir'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            
            gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
            gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # (B,1,H,W)

            # Get boxes
            pred_boxes = pred_dict['pred_boxes']
            # print("pred_boxes:",pred_boxes)
            
            
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                            max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
            # print("pred_boxes_vec:",pred_boxes_vec)
            # print("gt_boxes_vec:",gt_boxes_vec)
            # print("/n")
            
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
        else:
            bias_loss= self.bias_loss(bias, gt_dict['oringin_rgb_anno'], gt_dict['oringin_ir_anno_bias'])
            loss = 10 * bias_loss
            if return_status:
                # status for log
                status = {"Loss/total": loss.item(),
                          "Loss/bias": bias_loss.item(),
                        }
                return loss, status
            else:
                return loss
            