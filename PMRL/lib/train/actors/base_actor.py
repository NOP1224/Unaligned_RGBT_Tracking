from lib.utils import TensorDict
import torch
import torch.nn.functional as F   # ★ 新增

def xywh_to_xyxy(boxes_xywh: torch.Tensor) -> torch.Tensor:
    """
    boxes_xywh: (..., 4)  [x,y,w,h]
    return: (..., 4)  [x1,y1,x2,y2]
    """
    x, y, w, h = boxes_xywh.unbind(-1)
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    return boxes_xyxy

def calculate_offset_between_boxes(boxes1_xyxy: torch.Tensor, boxes2_xyxy: torch.Tensor) -> torch.Tensor:
    """
    boxes1_xyxy, boxes2_xyxy: (B,4)  [x1,y1,x2,y2]
    return: (B,4)  offset = [delta_x1, delta_y1, delta_x2, delta_y2]
    """
    offset = boxes2_xyxy - boxes1_xyxy
    return offset


def xywh01_to_xyxy_grid(box_xywh01: torch.Tensor) -> torch.Tensor:
    """
    box_xywh01: (B,4) with (x,y,w,h), x,y = top-left in [0,1]
    return: (B,4) (x1,y1,x2,y2) in grid coords [-1,1]
    """
    x, y, w, h = box_xywh01.unbind(dim=-1)
    x1_01, y1_01 = x, y
    x2_01, y2_01 = x + w, y + h
    # to grid [-1,1]
    x1 = x1_01 * 2.0 - 1.0
    y1 = y1_01 * 2.0 - 1.0
    x2 = x2_01 * 2.0 - 1.0
    y2 = y2_01 * 2.0 - 1.0
    return torch.stack([x1, y1, x2, y2], dim=-1)

def grid_xyxy_to_01(xyxy_grid: torch.Tensor) -> torch.Tensor:
    # (...,4) [-1,1] -> (...,4) [0,1]
    return (xyxy_grid + 1.0) * 0.5

def affine_params_to_mapped_xyxy01(
    params: torch.Tensor,
    src_ir_xywh01: torch.Tensor,
    log_scale: bool = True,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Apply the normalized centered diagonal affine used by RCCA/TCMDA.

    params: (...,4) with [dx,dy,sx,sy] or [dx,dy,log_sx,log_sy].
            dx,dy are in [0,1] search-region normalized coordinates.
    src_ir_xywh01: (B,4), source bbox in [0,1] xywh.

    Forward convention:
        p_tgt = scale * (p_src - 0.5) + 0.5 + shift

    return: mapped_xyxy01: (...,4) in [0,1] xyxy.
    """
    x, y, w, h = src_ir_xywh01.unbind(dim=-1)
    x1_i, y1_i = x, y
    x2_i, y2_i = x + w, y + h

    # Broadcast source coords to match params dims, e.g. (B,E,4).
    while params.dim() > x1_i.dim():
        x1_i = x1_i.unsqueeze(1)
        y1_i = y1_i.unsqueeze(1)
        x2_i = x2_i.unsqueeze(1)
        y2_i = y2_i.unsqueeze(1)

    dx, dy, sx, sy = params.unbind(dim=-1)
    if log_scale:
        sx = torch.exp(sx).clamp_min(eps)
        sy = torch.exp(sy).clamp_min(eps)

    x1 = sx * (x1_i - 0.5) + 0.5 + dx
    x2 = sx * (x2_i - 0.5) + 0.5 + dx
    y1 = sy * (y1_i - 0.5) + 0.5 + dy
    y2 = sy * (y2_i - 0.5) + 0.5 + dy

    mapped_xyxy_01 = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1)
    return mapped_xyxy_01

def gt_diag_affine_from_boxes_xywh01(
    src_ir_xywh01: torch.Tensor,
    tgt_rgb_xywh01: torch.Tensor,
    eps: float = 1e-6,
    use_log_scale: bool = False,
    scale_clamp=(0.25, 4.0),
) -> torch.Tensor:
    """
    Solve the same normalized centered diagonal affine used by RCCA/TCMDA:

        p_tgt = diag(sx, sy) * (p_src - 0.5) + 0.5 + [dx, dy]

    Inputs are xywh boxes in [0,1] search-region coordinates.
    Return is (B,4) = [dx, dy, sx, sy], or [dx,dy,log_sx,log_sy]
    when use_log_scale=True.

    Important: dx/dy are NOT in [-1,1] grid coordinates. They are normalized
    by the whole search region, so pixel displacement = dx * SEARCH.SIZE.
    """
    xs, ys, ws, hs = src_ir_xywh01.unbind(dim=-1)
    xt, yt, wt, ht = tgt_rgb_xywh01.unbind(dim=-1)

    sx = (wt / ws.clamp_min(eps)).clamp(scale_clamp[0], scale_clamp[1])
    sy = (ht / hs.clamp_min(eps)).clamp(scale_clamp[0], scale_clamp[1])

    src_cx = xs + 0.5 * ws
    src_cy = ys + 0.5 * hs
    tgt_cx = xt + 0.5 * wt
    tgt_cy = yt + 0.5 * ht

    # Centered affine shift in [0,1] coordinates.
    dx = tgt_cx - ((src_cx - 0.5) * sx + 0.5)
    dy = tgt_cy - ((src_cy - 0.5) * sy + 0.5)

    if use_log_scale:
        sx = torch.log(sx.clamp_min(eps))
        sy = torch.log(sy.clamp_min(eps))

    return torch.stack([dx, dy, sx, sy], dim=-1)

class BaseActor:
    """ Base class for actor. The actor class handles the passing of the data through the network
    and calculation the loss"""
    def __init__(self, net, objective):
        """
        args:
            net - The network to train
            objective - The loss function
        """
        self.net = net
        self.objective = objective

    def __call__(self, data: TensorDict):
        """ Called in each training iteration. Should pass in input data through the network, calculate the loss, and
        return the training stats for the input data
        args:
            data - A TensorDict containing all the necessary data blocks.

        returns:
            loss    - loss for the input data
            stats   - a dict containing detailed losses
        """
        raise NotImplementedError

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

    def eval(self):
        """ Set network to eval mode"""
        self.train(False)

    def fix_bns(self):
        pass

    def fix_bn(self, m):
        pass

    def __call__(self, data):
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data, None)
        return loss, status
