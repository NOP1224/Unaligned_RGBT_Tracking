import math
import logging
import pdb
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens, token2feature, feature2token
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock, candidate_elimination_adapter, CABlock, CABlock_Warp
from ..layers.alignnet import NeuroNet
from ..layers.deformattn import DeformHomographyAttn
from ...train.actors.homography_compute import SimpleHomoNormalizer

_logger = logging.getLogger(__name__)


def offset_to_homography(offset, device):
    B = offset.shape[0]

    dx, dy, sx, sy = offset[:,0], offset[:,1], offset[:,2], offset[:,3]
    H = torch.eye(3, device=device).unsqueeze(0).repeat(B,1,1)
    H[:,0,0] = sx
    H[:,1,1] = sy
    H[:,0,2] = dx
    H[:,1,2] = dy
    
    return H

class VisionTransformerCE(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None,
                 insert_layer=[3,6,9]):
        
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        #self.patch_embed_adapter = embed_layer(
        #    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # it's redundant
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.depth = depth

        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search=new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template=new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """     #
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))
        
        self.attn_list_rgb = []
        self.attn_list_ir = []

        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        ce_index = 0
        self.ce_loc = ce_loc
        
        
        self.norm = norm_layer(embed_dim)
        self.init_weights(weight_init)
        
        blocks = []
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1
            if i < 20:
                blocks.append(
                    CEBlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        keep_ratio_search=ce_keep_ratio_i)
                )
            else:
                blocks.append(
                    CEBlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        keep_ratio_search=ce_keep_ratio_i)
                )
        self.blocks = nn.Sequential(*blocks)
        
        H = W = 16
        self.H, self.W = H, W
        self.insert_layer = insert_layer
        
        self.interaction_layers_v2i = nn.ModuleList()
        self.interaction_layers_i2v = nn.ModuleList()
        self.neuro_net = nn.ModuleList()
        self.homo_norm = SimpleHomoNormalizer(256, 256)
        if self.insert_layer is not None and type(self.insert_layer) == list:
            for i in range(len(self.insert_layer)):
                self.neuro_net.append(NeuroNet(H, W, embed_dim))
                self.interaction_layers_v2i.append(DeformHomographyAttn(C=embed_dim, n_heads=8, n_points=4))
                self.interaction_layers_i2v.append(DeformHomographyAttn(C=embed_dim, n_heads=8, n_points=4))
    
    
    def forward_features(self, z, x, offset_gt=None, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False):

        z1=None
        if isinstance(z, list):
            if len(z)==1:
                z = z[0]
            else:
                z, z1 = z[0], z[1]
                
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        # rgb_img
        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]
        # depth thermal event images
        x_dte = x[:, 3:6, :, :]
        z_dte = z[:, 3:6, :, :]
        # overwrite x & z
        x, z = x_rgb, z_rgb
        xi, zi = x_dte, z_dte

        #print("input x",x.size())

        z = self.patch_embed(z)
        x = self.patch_embed(x)

        #print("after patch_embed x",x.size())

        xi = self.patch_embed(xi)
        zi = self.patch_embed(zi)
        

        ###################################################################===========
        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        zi += self.pos_embed_z
        xi += self.pos_embed_x
        
            
        if z1 is not None:
            z1_rgb = z1[:, :3, :, :]
            z1_dte = z1[:, 3:6, :, :]
            z1 = self.patch_embed(z1_rgb)
            z1_i = self.patch_embed(z1_dte)
            z1 += self.pos_embed_z
            z1_i += self.pos_embed_z
            z = torch.cat([z, z1],1)
            zi = torch.cat([zi, z1_i],1)
            
        x = combine_tokens(z, x, mode=self.cat_mode)  

        xi = combine_tokens(zi, xi, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)
            xi = torch.cat([cls_tokens, xi], dim=1)

        x = self.pos_drop(x)
        xi = self.pos_drop(xi)

        lens_z = self.pos_embed_z.shape[1] if z1 is None else self.pos_embed_z.shape[1]*2
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        global_index_ti = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_ti = global_index_ti.repeat(B, 1)

        global_index_si = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_si = global_index_si.repeat(B, 1)

        removed_indexes_s = []
        removed_indexes_si = []
        offsets = []
        offset_experts = []
        gate_logits = []
        router_losses =[]
        warped_feats = []
        target_respones_v = []
        target_respones_i = []
        chosens = []
        insert_idx = 0
        for i, blk in enumerate(self.blocks):
            B, L, D = x.shape
            x, global_index_t, global_index_s, removed_index_s, attn = blk(x, global_index_t, global_index_s, mask_x, ce_template_mask,
                    ce_keep_rate)
            xi, global_index_ti, global_index_si, removed_index_si, attn_i= \
                blk(xi, global_index_ti, global_index_si, mask_x, ce_template_mask,
                    ce_keep_rate)
            
            lens_x_new = global_index_s.shape[1]

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)
                removed_indexes_si.append(removed_index_si)
                
            if i in self.insert_layer:
                lens_x_new = global_index_s.shape[1]
                lens_z_new = global_index_t.shape[1]
                lens_xi_new = global_index_si.shape[1]
                lens_zi_new = global_index_ti.shape[1]

                z_rgb = x[:, :lens_z_new]
                x_rgb = x[:, lens_z_new:]
                z_tir = xi[:, :lens_zi_new]
                x_tir  = xi[:, lens_zi_new:]
                x_tir_resize = x_tir.view(B, self.H, self.W, D).permute(0,3,1,2)
                x_rgb_resize = x_rgb.view(B, self.H, self.W, D).permute(0,3,1,2)
                
                pred, losses, aux = self.neuro_net[insert_idx](z_rgb, x_rgb, z_tir, x_tir, offset_gt, return_all=True, layer_index=insert_idx, previous_preds=offsets)
                # pred, aux = self.neuro_net[insert_idx].infer(z_rgb, x_rgb, z_tir, x_tir,
                #                                                 return_aux=True, layer_index=insert_idx,
                #                                                 previous_preds=offsets, selected_id=0)
                offsets.append(pred)
                router_losses.append(losses)
                offset_experts.append(aux['experts'])
                gate_logits.append(aux["gate_logits"])
                # chosens.append(aux['chosen_expert'])
                target_respones_v.append(aux["rgb_target"])
                target_respones_i.append(aux["tir_target"])
                H = offset_to_homography(pred, x.device)           # [B,3,3], source->target
                H_img_v2i = self.homo_norm.denormalize(H, inverse=False)  # 仍以 256×256 为基准
                H_img_i2v = self.homo_norm.denormalize(H, inverse=True)
                x_rgb = x_rgb + self.interaction_layers_v2i[insert_idx](x_rgb, x_tir_resize, H_img_v2i)
                x_tir = x_tir + self.interaction_layers_i2v[insert_idx](x_tir, x_rgb_resize, H_img_i2v)
                x = torch.cat([z_rgb, x_rgb],1)
                xi = torch.cat([z_tir, x_tir],1)
                insert_idx += 1
                
        
        x = self.norm(x)
        xi = self.norm(xi)

        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]
        lens_xi_new = global_index_si.shape[1]
        lens_zi_new = global_index_ti.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]
        zi = xi[:, :lens_zi_new]
        xi = xi[:, lens_zi_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)
        
        if removed_indexes_si and removed_indexes_si[0] is not None:
            removed_indexes_cat_i = torch.cat(removed_indexes_si, dim=1)

            pruned_lens_xi = lens_x - lens_xi_new                                ########################
            pad_xi = torch.zeros([B, pruned_lens_xi, xi.shape[2]], device=xi.device)
            xi = torch.cat([xi, pad_xi], dim=1)
            index_all = torch.cat([global_index_si, removed_indexes_cat_i], dim=1)
            # recover original token order
            C = xi.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            xi = torch.zeros_like(xi).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=xi)
        
        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
        xi = recover_tokens(xi, lens_zi_new, lens_x, mode=self.cat_mode)
        
        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)
        xi = torch.cat([zi, xi], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
            "attn_i": attn_i,
            "offsets":offsets,
            "offset_experts": offset_experts,
            "gate_logits":gate_logits,
            "warped_feats":warped_feats,
            "target_respones_v":target_respones_v,
            "target_respones_i":target_respones_i,
            "align_aux": aux,
            "router_loss": router_losses,
            "chosens":chosens
        }


        return x, aux_dict

    def forward(self, z, x, offset_gt=None, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, aux_dict = self.forward_features(z, x, offset_gt=offset_gt, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate, return_last_attn = return_last_attn)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)

    return model


def vit_base_patch16_224_ce_adapter(pretrained=False, **kwargs):
    
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce_adapter(pretrained=False, **kwargs):
    
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
