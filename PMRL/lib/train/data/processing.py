import pdb

import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F

import cv2
import numpy
import time
def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

import torch
import numpy as np

def horiz_flip_bbox_xywh(
    bboxes,
    img_width: int = None,
    use_norm: bool = False,
    do_flip: bool = True,
):
    """
    水平翻转 bbox，支持归一化和像素坐标。

    Args:
        bboxes: [..., 4]，最后一维为 [x, y, w, h]。
                可以是 torch.Tensor 或 np.ndarray。
        img_width: 图像宽度（像素）。仅在 use_norm=False 时需要。
        use_norm: True 表示 bboxes 是归一化坐标 (0~1)，False 表示像素坐标。
        do_flip:  是否真的翻转；False 时直接原样返回。

    Returns:
        与 bboxes 同类型、同形状的翻转后结果。
    """
    if not do_flip:
        return bboxes

    # 转成 tensor 统一操作
    is_numpy = isinstance(bboxes, np.ndarray)
    if is_numpy:
        bboxes_t = torch.from_numpy(bboxes)
    else:
        bboxes_t = bboxes

    bboxes_t = bboxes_t.clone()

    if use_norm:
        # 归一化坐标：x' = 1 - x - w
        bboxes_t[..., 0] = 1.0 - bboxes_t[..., 0] - bboxes_t[..., 2]
    else:
        assert img_width is not None, "use_norm=False 时必须提供 img_width"
        # 像素坐标：x' = W - x - w
        bboxes_t[..., 0] = img_width - bboxes_t[..., 0] - bboxes_t[..., 2]

    if is_numpy:
        return bboxes_t.numpy()
    return bboxes_t


def to_numpy(x):
    """支持 torch.Tensor / np.ndarray，统一成 [H, W, C] 的 numpy"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)

    # 如果是 [C, H, W]，转成 [H, W, C]
    if x.ndim == 3 and x.shape[0] in (3, 6):
        x = np.transpose(x, (1, 2, 0))

    return x


def denorm_img(img):
    """简单做个归一化，确保能显示出来"""
    img = img.astype(np.float32)
    if img.max() > 1.5:  # 认为是 0~255
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)
    return img


def convert_norm_boxes_to_xyxy(boxes, W, H, mode="xywh"):
    """
    boxes: [B, 4] 归一化坐标 (0~1)
    mode="xywh": [x, y, w, h]，x,y 为左上角  ← 默认
    mode="cxcywh": [cx, cy, w, h]，cx,cy 为中心点
    返回: [B, 4]，绝对像素坐标 [x1, y1, x2, y2]
    """
    boxes = torch.as_tensor(boxes, dtype=torch.float32)

    if mode == "xywh":
        x = boxes[:, 0] * W
        y = boxes[:, 1] * H
        w = boxes[:, 2] * W
        h = boxes[:, 3] * H

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

    elif mode == "cxcywh":
        # 如果你的 bbox 是中心点形式，就把 mode 改成 "cxcywh"
        cx = boxes[:, 0] * W
        cy = boxes[:, 1] * H
        w  = boxes[:, 2] * W
        h  = boxes[:, 3] * H

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

    else:
        raise ValueError(f"Unknown mode: {mode}")

    xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    return xyxy.numpy()


def debug_show_rgbt_and_boxes(
    rgbt_array,
    rgb_boxes=None,
    tir_boxes=None,
    box_mode="xywh",
    save_path="debug_rgb_tir.png"
):
    """
    rgbt_array: [H, W, 6]，RGB-TIR 通道拼接
    rgb_boxes:  [B, 4]，归一化坐标，针对 RGB 图像
    tir_boxes:  [B, 4]，归一化坐标，针对 TIR 图像
    box_mode:   "xywh" 或 "cxcywh"
    save_path:  保存的文件名
    """
    rgbt = to_numpy(rgbt_array)

    if rgbt.shape[-1] != 6:
        raise ValueError(f"期望通道为 6，但得到 {rgbt.shape}")

    H, W, C = rgbt.shape
    rgb = rgbt[..., :3]
    tir = rgbt[..., 3:]

    rgb = denorm_img(rgb)
    tir = denorm_img(tir)

    # TIR 转成灰度或 3 通道显示
    if tir.shape[-1] == 1:
        tir_show = tir[..., 0]
    elif tir.shape[-1] == 3:
        tir_show = tir
    else:
        # 其他情况就取第一个通道
        tir_show = tir[..., 0]

    # 计算 bbox 像素坐标
    rgb_xyxy = None
    tir_xyxy = None
    if rgb_boxes is not None:
        rgb_xyxy = convert_norm_boxes_to_xyxy(rgb_boxes, W, H, mode=box_mode)
    if tir_boxes is not None:
        tir_xyxy = convert_norm_boxes_to_xyxy(tir_boxes, W, H, mode=box_mode)

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # --- RGB ---
    ax = axes[0]
    ax.imshow(rgb)
    ax.set_title("RGB")
    ax.axis("off")

    if rgb_xyxy is not None:
        for (x1, y1, x2, y2) in rgb_xyxy:
            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=2
            )
            ax.add_patch(rect)

    # --- TIR ---
    ax = axes[1]
    if tir_show.ndim == 2:
        ax.imshow(tir_show, cmap="gray")
    else:
        ax.imshow(tir_show)
    ax.set_title("TIR")
    ax.axis("off")

    if tir_xyxy is not None:
        for (x1, y1, x2, y2) in tir_xyxy:
            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=2
            )
            ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"保存到: {save_path}")

class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, transform_all=None, search_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'joint': joint_transform,
                          'all': transform if transform_all is None else transform_all }

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class SFCAProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', loader_mode='train', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings
        self.loader_mode = loader_mode

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)
    
    def _get_jittered_box_notalign(self, box_rgb, box_ir, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """
        if self.loader_mode == 'val':
            return box_rgb.clone(), box_ir.clone()

        jittered_size_random_num = torch.randn(2)
        jittered_size_rgb = box_rgb[2:4] * torch.exp(jittered_size_random_num * self.scale_jitter_factor[mode])
        jittered_size_ir = box_ir[2:4] * torch.exp(jittered_size_random_num * self.scale_jitter_factor[mode])
        
        max_offset_rgb = (jittered_size_rgb.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        max_offset_ir = (jittered_size_ir.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        
        jittered_center_random_num = torch.rand(2)
        jittered_center_rgb = box_rgb[0:2] + 0.5 * box_rgb[2:4] + max_offset_rgb * (jittered_center_random_num - 0.5)
        jittered_center_ir = box_ir[0:2] + 0.5 * box_ir[2:4] + max_offset_ir * (jittered_center_random_num - 0.5)

        return torch.cat((jittered_center_rgb - 0.5 * jittered_size_rgb, jittered_size_rgb), dim=0), torch.cat((jittered_center_ir - 0.5 * jittered_size_ir, jittered_size_ir), dim=0)

    def _get_jittered_box_notalign2(self, box_rgb, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """
        if self.loader_mode == 'val':
            return box_rgb.clone()

        jittered_size_random_num = torch.randn(2)
        jittered_size_rgb = box_rgb[2:4] * torch.exp(jittered_size_random_num * self.scale_jitter_factor[mode])
        # jittered_size_ir = box_ir[2:4] * torch.exp(jittered_size_random_num * self.scale_jitter_factor[mode])
        
        max_offset_rgb = (jittered_size_rgb.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        # max_offset_ir = (jittered_size_ir.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        
        jittered_center_random_num = torch.rand(2)
        jittered_center_rgb = box_rgb[0:2] + 0.5 * box_rgb[2:4] + max_offset_rgb * (jittered_center_random_num - 0.5)
        # jittered_center_ir = box_ir[0:2] + 0.5 * box_ir[2:4] + max_offset_ir * (jittered_center_random_num - 0.5)

        return torch.cat((jittered_center_rgb - 0.5 * jittered_size_rgb, jittered_size_rgb), dim=0)


    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
                images: list of np.ndarray [(H,W,6)]
                anno: list of torch.Tensor [(4,)]
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno'
        """
        if self.transform['joint'] is not None:
            data['template_images_rgb'], data['template_anno_rgb'], data['template_is_flip_rgb_first'] = self.transform['joint'](
                image=data['template_images_rgb'], bbox=data['template_anno_rgb'])
            data['search_images_rgb'], data['search_anno_rgb'], data['search_is_flip_rgb_first'] = self.transform['joint'](
                image=data['search_images_rgb'], bbox=data['search_anno_rgb'], new_roll=False)
            data['template_images_ir'], data['template_anno_ir'], data['template_is_flip_ir_first']  = self.transform['joint'](
                image=data['template_images_ir'], bbox=data['template_anno_ir'], new_roll=False)
            for i in range(len(data['search_anno_ir'])):
                data['search_anno_ir'][i] = horiz_flip_bbox_xywh(data['search_anno_ir'][i], data['rgb_w'], use_norm=False, do_flip=data['search_is_flip_rgb_first'])
            if data.get('search_jit_anno_rgb', None) is not None:
                for i in range(len(data['search_jit_anno_rgb'])):
                    data['search_jit_anno_rgb'][i] = horiz_flip_bbox_xywh(
                        data['search_jit_anno_rgb'][i], data['rgb_w'],
                        use_norm=False, do_flip=data['search_is_flip_rgb_first']
                    )
        jittered_anno_dict = TensorDict({
                'rgb': None,
                'ir': None})
        jittered_anno_rgb, jittered_anno_ir = zip(*[self._get_jittered_box_notalign(a0,a1,'template') for a0,a1 in zip(data['template_anno_rgb'],data['template_anno_ir'])])
        jittered_anno_dict['rgb'] = jittered_anno_rgb
        jittered_anno_dict['ir'] = jittered_anno_ir
        modalities = ['rgb','ir']
        for modality in modalities:
            stack_jittered_anno_dict = torch.stack(jittered_anno_dict[modality], dim=0)
            w, h = stack_jittered_anno_dict[:, 2], stack_jittered_anno_dict[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor['template'])
            if (crop_sz < 1).any():
                data['valid'] = False
                return data
            crops, boxes, _, _ = prutils.jittered_center_crop(data['template_images_'+modality], jittered_anno_dict[modality],
                                                                data['template_anno_'+modality], self.search_area_factor['template'],
                                                                self.output_sz['template'])
            # Apply transforms
            data['template_images_' + modality], data['template_anno_' + modality ], is_hor_flip = self.transform['template'](
                image=crops,
                bbox=boxes,
                joint=False,
                new_roll=(modality == 'rgb'),
            )
            data['template_crop_sz_' + modality] = crop_sz
            data['template_jit_anno_' + modality] = jittered_anno_dict[modality]
            # 取当前 transform 的翻转状态
            curr_flip = int(is_hor_flip[0]) if isinstance(is_hor_flip, (list, torch.Tensor)) else int(is_hor_flip)
            data['template_is_flip_' + modality] = curr_flip
        # Search
        jittered_anno_dict = TensorDict({
                    'rgb': None,
                    'ir': None})
        if data.get('search_jit_anno_rgb', None) is not None:
            jittered_anno_rgb = data['search_jit_anno_rgb']
        else:
            jittered_anno_rgb = [self._get_jittered_box_notalign2(a0,'search') for a0 in data['search_anno_rgb']]
        jittered_anno_dict['rgb'] = jittered_anno_rgb        
        stack_jittered_anno_dict = torch.stack(jittered_anno_dict['rgb'], dim=0)
        w, h = stack_jittered_anno_dict[:, 2], stack_jittered_anno_dict[:, 3]
        crop_sz_search = torch.ceil(torch.sqrt(w * h) * self.search_area_factor['search'])
        
        if (crop_sz_search < 1).any():
            data['valid'] = False
            return data
        crop_search, boxes_rgb_search, boxes_ir_search, _, _ = prutils.jittered_center_crop_not_align(data['search_images_rgb'], jittered_anno_dict['rgb'],
                                                                            data['search_anno_rgb'], data['search_anno_ir'], self.search_area_factor['search'],
                                                                            self.output_sz['search'])


        data['search_images'], data['search_anno_rgb'], is_hor_flip = self.transform['search'](image=crop_search, bbox=boxes_rgb_search, joint=False)
        for i in range(len(boxes_ir_search)):
            boxes_ir_search[i] = horiz_flip_bbox_xywh(boxes_ir_search[i], data['rgb_w'], use_norm=True, do_flip=is_hor_flip[i])
        data['search_anno_ir'] = boxes_ir_search
        data['search_crop_sz'] = crop_sz_search
        data['search_jit_anno_rgb'] = jittered_anno_dict['rgb']
        # 取当前 transform 的翻转状态
        curr_flip = int(is_hor_flip[0]) if isinstance(is_hor_flip, (list, torch.Tensor)) else int(is_hor_flip)
        data['search_is_flip_rgb'] = curr_flip                
        data['search_images'] = data['search_images']           
        
        data['template_images'] = torch.cat([data['template_images_rgb'][0][:3,:,:], data['template_images_ir'][0][3:,:,:]], dim=0)
        data['template_anno'] = data['template_anno_rgb']
        data['valid'] = True

        del data["search_images_rgb"]
        del data["template_images_rgb"]
        del data["template_images_ir"]
        del data["template_anno_rgb"]
        del data["template_anno_ir"]

        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data