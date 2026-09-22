import math
import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock
from ..layers.rcca import ProgressiveAlignmentController, geometry_to_dirmag
from ..layers.tcmda import TCMDAFusion
_logger = logging.getLogger(__name__)

class VisionTransformerCE(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None, joint_align_fusion_cfg=None):
        super().__init__()
        self.img_size = img_size if isinstance(img_size, tuple) else to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.depth = depth

        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search = new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template = new_P_H * new_P_W

        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
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
            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )
        self.blocks = nn.Sequential(*blocks)

        search_feat_sz = int(math.sqrt(self.num_patches_search))
        self.joint_align_fusion_cfg = dict(joint_align_fusion_cfg or {})
        self.progressive_alignment = ProgressiveAlignmentController(
            dim=embed_dim,
            feat_sz=search_feat_sz,
            num_heads=num_heads,
            joint_config=self.joint_align_fusion_cfg,
        )
        
        self.tcmda = TCMDAFusion(
            dim=embed_dim,
            feat_sz=search_feat_sz,
            num_heads=num_heads,
            n_points=4,
            joint_config=self.joint_align_fusion_cfg,
        ) 
        # RCCA relation prediction and relation-guided TCMDA are both placed
        # after layers 9/10/11. The UOT relation preserves one-to-many token
        # matches and directly guides local alignment/fusion at the same depth.
        self.tcmda_layers = {9, 10, 11}

    @staticmethod
    def _make_index(batch_size, length, device):
        return torch.arange(length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)

    @staticmethod
    def _recover_search_order(search_tokens, global_index, removed_indexes, original_len):
        if not removed_indexes:
            return search_tokens
        valid_removed = [idx for idx in removed_indexes if idx is not None]
        if not valid_removed:
            return search_tokens

        removed = torch.cat(valid_removed, dim=1)
        B, _, C = search_tokens.shape
        pruned_len = original_len - global_index.shape[1]
        if pruned_len <= 0:
            return search_tokens

        pad = torch.zeros(B, pruned_len, C, device=search_tokens.device, dtype=search_tokens.dtype)
        src = torch.cat([search_tokens, pad], dim=1)
        index_all = torch.cat([global_index, removed], dim=1)
        return torch.zeros_like(src).scatter_(
            dim=1,
            index=index_all.unsqueeze(-1).expand(B, -1, C).long(),
            src=src,
        )

    def _split_modal_input(self, z, x):
        if isinstance(z, list):
            if len(z)==1:
                return z[0][:, :3], None, x[:, :3], z[0][:, 3:6], None, x[:, 3:6]
            else:
                return z[0][:, :3], z[1][:, :3], x[:, :3], z[0][:, 3:6], z[1][:, 3:6], x[:, 3:6]
        else:
            return z[:, :3], None, x[:, :3], z[:, 3:6], None, x[:, 3:6]
        
    def _densify_search_for_tcmda(self, xs, xis, global_index_s, global_index_si,
                                  removed_indexes_s, removed_indexes_si, lens_z_cur, lens_zi_cur, lens_x):
        xs = self._recover_search_order(xs, global_index_s, removed_indexes_s, lens_x)
        xis = self._recover_search_order(xis, global_index_si, removed_indexes_si, lens_x)
        xs = recover_tokens(xs, lens_z_cur, lens_x, mode=self.cat_mode)
        xis = recover_tokens(xis, lens_zi_cur, lens_x, mode=self.cat_mode)
        return xs, xis

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False):
        B = x.shape[0]
        z_rgb, z1_rgb, x_rgb, z_tir, z1_tir, x_tir = self._split_modal_input(z, x)

        z = self.patch_embed(z_rgb)
        x = self.patch_embed(x_rgb)
        zi = self.patch_embed(z_tir)
        xi = self.patch_embed(x_tir)




        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)
            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)
            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode).squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z = z + self.pos_embed_z
        x = x + self.pos_embed_x
        zi = zi + self.pos_embed_z
        xi = xi + self.pos_embed_x

        if self.add_sep_seg:
            x = x + self.search_segment_pos_embed
            z = z + self.template_segment_pos_embed
            xi = xi + self.search_segment_pos_embed
            zi = zi + self.template_segment_pos_embed

        if z1_rgb is not None and z1_tir is not None :
            z1 = self.patch_embed(z1_rgb)
            z1_i = self.patch_embed(z1_tir)
            z1 += self.pos_embed_z
            z1_i += self.pos_embed_z
            if self.add_sep_seg:
                z1 += self.template_segment_pos_embed
                z1_i += self.template_segment_pos_embed
            z = torch.cat([z, z1],1)
            zi = torch.cat([zi, z1_i],1)

        x = combine_tokens(z, x, mode=self.cat_mode)
        xi = combine_tokens(zi, xi, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)
            xi = torch.cat([cls_tokens, xi], dim=1)

        x = self.pos_drop(x)
        xi = self.pos_drop(xi)

        lens_z = self.pos_embed_z.shape[1] if z1_rgb is None else self.pos_embed_z.shape[1]*2
        lens_x = self.pos_embed_x.shape[1]
        global_index_t = self._make_index(B, lens_z, x.device)
        global_index_s = self._make_index(B, lens_x, x.device)
        global_index_ti = self._make_index(B, lens_z, xi.device)
        global_index_si = self._make_index(B, lens_x, xi.device)
        count=0
        removed_indexes_s = []
        removed_indexes_si = []
        aux_alignment = {}
        detail_div_terms = []
        detail_peak_terms = []
        attn = None
        attn_i = None
        align_state = self.progressive_alignment.initial_state(B, x.device, x.dtype)
        for i, blk in enumerate(self.blocks):
            layer_id = i + 1
            final_rgb_only_layer = (
                i == self.depth - 1
                and not self.progressive_alignment.has_stage(layer_id)
                and layer_id not in self.tcmda_layers
            )
            x, global_index_t, global_index_s, removed_index_s, attn = blk(
                x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)
            if final_rgb_only_layer:
                # All cross-modal alignment/fusion stages are completed after layer 11.
                # The tracker head consumes only the RGB branch, so the layer-12 TIR
                # block and all following TIR-only bookkeeping are dead computation.
                removed_index_si = None
            else:
                xi, global_index_ti, global_index_si, removed_index_si, attn_i = blk(
                    xi, global_index_ti, global_index_si, mask_x, ce_template_mask, ce_keep_rate)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)
                if removed_index_si is not None:
                    removed_indexes_si.append(removed_index_si)

            if final_rgb_only_layer:
                # ``x`` is already in the combined template/search order returned by
                # the final CEBlock. No RCCA/TCMDA stage follows, so skip touching xi.
                continue

            lens_z_cur = global_index_t.shape[1]
            lens_zi_cur = global_index_ti.shape[1]
            z_cur, xs_cur = x[:, :lens_z_cur], x[:, lens_z_cur:]
            zi_cur, xis_cur = xi[:, :lens_zi_cur], xi[:, lens_zi_cur:]

            # 9/10/11: solve one RCCA relation stage and immediately use the
            # relation for TCMDA alignment/fusion at the same semantic depth.
            current_stage_name = None
            if self.progressive_alignment.has_stage(layer_id):
                current_stage_name = self.progressive_alignment.get_stage_name(layer_id)

                # Keep the alignment predictor isolated from previous-stage
                # feature residuals. Geometry state is still propagated inside
                # ProgressiveAlignment, but no unvalidated feature prior is injected.
                xs_align = xs_cur
                xis_align = xis_cur

                align_state, stage_aux = self.progressive_alignment.forward_stage(
                    layer=layer_id,
                    rgb_search=xs_align,
                    tir_search=xis_align,
                    rgb_template=z_cur,
                    tir_template=zi_cur,
                    rgb_index=global_index_s,
                    tir_index=global_index_si,
                    state=align_state,
                )
                aux_alignment.update(stage_aux)

                for k, v in stage_aux.items():
                    if k.endswith("loss_detail_div"):
                        detail_div_terms.append(v)
                    elif k.endswith("loss_detail_peak"):
                        detail_peak_terms.append(v)

            if layer_id in self.tcmda_layers and current_stage_name is not None:
                xs_dense, xis_dense = self._densify_search_for_tcmda(
                    xs_cur, xis_cur,
                    global_index_s, global_index_si,
                    removed_indexes_s, removed_indexes_si,
                    lens_z_cur, lens_zi_cur, lens_x,
                )
                relation = stage_aux[f"{current_stage_name}_relation"]
                relation_coords = stage_aux[f"{current_stage_name}_coords"]
                src_confidence = stage_aux.get(f"{current_stage_name}_src_conf", None)
                tgt_confidence = stage_aux.get(f"{current_stage_name}_tgt_conf", None)
                src_quality = stage_aux.get(f"{current_stage_name}_src_fusion_quality", None)
                tgt_quality = stage_aux.get(f"{current_stage_name}_tgt_fusion_quality", None)
                alignment_geometry = stage_aux.get(f"{current_stage_name}_geometry", None)
                alignment_confidence = stage_aux.get(f"{current_stage_name}_confidence", None)
                xs_fused, xis_fused = self.tcmda(
                    rgb_search=xs_dense,
                    tir_search=xis_dense,
                    relation=relation,
                    coords=relation_coords,
                    stage_name=current_stage_name,
                    src_confidence=src_confidence,
                    tgt_confidence=tgt_confidence,
                    src_quality=src_quality,
                    tgt_quality=tgt_quality,
                    alignment_geometry=alignment_geometry,
                    alignment_confidence=alignment_confidence,
                )

                # TCMDA is the only cross-modal fusion path. It receives backbone
                # tokens and the RCCA relation; no extra dense T-based fusion or
                # Agent-feature injection is used.
                aux_alignment[f"AdapterNorm/{current_stage_name}"] = torch.zeros(
                    (), device=x.device, dtype=x.dtype
                )
                x = torch.cat([z_cur, xs_fused], dim=1)
                xi = torch.cat([zi_cur, xis_fused], dim=1)
                global_index_s = self._make_index(B, lens_x, x.device)
                global_index_si = self._make_index(B, lens_x, xi.device)
                removed_indexes_s = []
                removed_indexes_si = []
                aux_alignment[f"tcmda_layer_{layer_id}"] = torch.ones((), device=x.device, dtype=x.dtype)
                count +=1
            else:
                x = torch.cat([z_cur, xs_cur], dim=1)
                xi = torch.cat([zi_cur, xis_cur], dim=1)

        x = self.norm(x)

        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]
        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        x = self._recover_search_order(x, global_index_s, removed_indexes_s, lens_x)
        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,
            "attn_i": attn_i,
            "pred_offset": align_state.geometry,
            "pred_offset_dirmag": geometry_to_dirmag(align_state.geometry) if align_state.geometry is not None else None,
            "pred_offset_uncertainty": align_state.uncertainty,
        }
        aux_dict.update(aux_alignment)
        if detail_div_terms:
            aux_dict["loss_detail_div"] = torch.stack(detail_div_terms).mean()
        if detail_peak_terms:
            aux_dict["loss_detail_peak"] = torch.stack(detail_peak_terms).mean()
        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):
        x, aux_dict = self.forward_features(
            z, x, ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn)
        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)
    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["net"], strict=False)
    return model


def vit_base_patch16_224_ce_adapter(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    return _create_vision_transformer(pretrained=pretrained, **model_kwargs)


def vit_large_patch16_224_ce_adapter(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    return _create_vision_transformer(pretrained=pretrained, **model_kwargs)
