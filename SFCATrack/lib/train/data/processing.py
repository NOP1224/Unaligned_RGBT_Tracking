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


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
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
                          'joint': joint_transform}

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
        jittered_size_random_num = torch.randn(2)
        jittered_size_rgb = box_rgb[2:4] * torch.exp(jittered_size_random_num * self.scale_jitter_factor[mode])
        jittered_size_ir = box_ir[2:4] * torch.exp(jittered_size_random_num * self.scale_jitter_factor[mode])
        
        max_offset_rgb = (jittered_size_rgb.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        max_offset_ir = (jittered_size_ir.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        
        jittered_center_random_num = torch.rand(2)
        jittered_center_rgb = box_rgb[0:2] + 0.5 * box_rgb[2:4] + max_offset_rgb * (jittered_center_random_num - 0.5)
        jittered_center_ir = box_ir[0:2] + 0.5 * box_ir[2:4] + max_offset_ir * (jittered_center_random_num - 0.5)

        return torch.cat((jittered_center_rgb - 0.5 * jittered_size_rgb, jittered_size_rgb), dim=0), torch.cat((jittered_center_ir - 0.5 * jittered_size_ir, jittered_size_ir), dim=0)

    def __call__(self, data: TensorDict, Align_mode = True):
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
        if Align_mode == True:
            # Apply joint transforms
            if self.transform['joint'] is not None:
                data['template_images'], data['template_anno'] = self.transform['joint'](
                    image=data['template_images'], bbox=data['template_anno'])
                data['search_images'], data['search_anno'] = self.transform['joint'](
                    image=data['search_images'], bbox=data['search_anno'], new_roll=False)

            for s in ['template', 'search']:
                assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                    "In pair mode, num train/test frames must be 1"


                # Add a uniform noise to the center pos. RGB and X modalities are aligned.
                jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

                # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes stack (Ns, 4)
                w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

                crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
                if (crop_sz < 1).any():
                    data['valid'] = False
                    # print("Too small box is found. Replace it with new data.")
                    return data

                # Crop image region centered at jittered_anno box
                # Here, we normalize anno to 0-1
                crops, boxes, _, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                data[s + '_anno'], self.search_area_factor[s],
                                                                self.output_sz[s])

                # Apply transforms
                data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

            data['valid'] = True
            # Prepare output
            if self.mode == 'sequence':
                data = data.apply(stack_tensors)
            else:
                data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

            return data
        else:
            if self.transform['joint'] is not None:
                data['template_images_rgb'], data['template_anno_rgb'] = self.transform['joint'](
                    image=data['template_images_rgb'], bbox=data['template_anno_rgb'])
                data['search_images_rgb'], data['search_anno_rgb'] = self.transform['joint'](
                    image=data['search_images_rgb'], bbox=data['search_anno_rgb'], new_roll=False)
                data['template_images_ir'], data['template_anno_ir'] = self.transform['joint'](
                    image=data['template_images_ir'], bbox=data['template_anno_ir'])
                data['search_images_ir'], data['search_anno_ir'] = self.transform['joint'](
                    image=data['search_images_ir'], bbox=data['search_anno_ir'], new_roll=False)
                data['template_images_rgb_prior'], data['template_anno_rgb_prior'] = self.transform['joint'](
                    image=data['template_images_rgb_prior'], bbox=data['template_anno_rgb_prior'])
                data['search_images_rgb_prior'], data['search_anno_rgb_prior'] = self.transform['joint'](
                    image=data['search_images_rgb_prior'], bbox=data['search_anno_rgb_prior'], new_roll=False)
                data['template_images_ir_prior'], data['template_anno_ir_prior'] = self.transform['joint'](
                    image=data['template_images_ir_prior'], bbox=data['template_anno_ir_prior'])
                data['search_images_ir_prior'], data['search_anno_ir_prior'] = self.transform['joint'](
                    image=data['search_images_ir_prior'], bbox=data['search_anno_ir_prior'], new_roll=False)
            
            for tail in ['_prior','']:
                for s in ['template', 'search']:
                    assert self.mode == 'sequence' or len(data[s + '_images_rgb'+tail]) == 1 or len(data[s + '_images_ir'+tail]) == 1, \
                        "In pair mode, num train/test frames must be 1"


                    # Add a uniform noise to the center pos. RGB and X modalities are not aligned.
                    jittered_anno_dict = TensorDict({
                        'rgb': None,
                        'ir': None})
                    jittered_anno_rgb, jittered_anno_ir = zip(*[self._get_jittered_box_notalign(a0,a1,s) for a0,a1 in zip(data[s+'_anno_rgb'+tail],data[s+'_anno_ir'+tail])])
                    jittered_anno_dict['rgb'] = jittered_anno_rgb
                    jittered_anno_dict['ir'] = jittered_anno_ir
                    
                    for modality in ['rgb','ir']:
                        # w, h = torch.stack(jittered_anno_dict[modality], dim=0)[:, 2], torch.stack(jittered_anno_dict[modality], dim=0)[:, 3]
                        stack_jittered_anno_dict = torch.stack(jittered_anno_dict[modality], dim=0)
                        w, h = stack_jittered_anno_dict[:, 2], stack_jittered_anno_dict[:, 3]

                        crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
                        if (crop_sz < 1).any():
                            data['valid'] = False
                            # print("Too small box is found. Replace it with new data.")
                            return data

                        # Crop image region centered at jittered_anno box
                        # Here, we normalize anno to 0-1
                        crops, boxes, _, _ = prutils.jittered_center_crop(data[s + '_images_'+modality], jittered_anno_dict[modality],
                                                                        data[s + '_anno_'+modality+tail],self.search_area_factor[s],
                                                                        self.output_sz[s])

                        # Apply transforms
                        data[s + '_images_'+modality+tail], data[s + '_anno_'+modality+tail] = self.transform[s](image=crops, bbox=boxes, joint=False)
            if self.loader_mode == 'train':
                data['search_images'] = [torch.cat([data['search_images_rgb'][0][:3,:,:], data['search_images_rgb'][0][3:,:,:]], dim=0)]
                data['search_images_prior'] = [torch.cat([data['search_images_rgb_prior'][0][:3,:,:], data['search_images_rgb_prior'][0][3:,:,:]], dim=0)]
            else:
                data['search_images'] = [torch.cat([data['search_images_rgb'][0][:3,:,:], data['search_images_rgb'][0][3:,:,:]], dim=0)]
                data['search_images_prior'] = [torch.cat([data['search_images_rgb_prior'][0][:3,:,:], data['search_images_rgb_prior'][0][3:,:,:]], dim=0)]
            data['template_images'] = [torch.cat([data['template_images_rgb'][0][:3,:,:], data['template_images_ir'][0][3:,:,:]], dim=0)]
            # data['search_images'] = data['search_images_ir'][0][3:,:,:]
            data['search_anno'] = data['search_anno_rgb']
            data['template_anno'] = data['template_anno_rgb']
            data['search_anno_ir'] = data['search_anno_ir']
            data['valid'] = True
            
        
            
            # Prepare output
            if self.mode == 'sequence':
                data = data.apply(stack_tensors)
            else:
                data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

            return data