import pdb, cv2
import torch.nn as nn
import torch
import torch.nn.functional as F
from . import BaseActor, gt_diag_affine_from_boxes_xywh01
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from ...utils.heapmap_utils import generate_heatmap
from lib.train.admin import multigpu


class OSTrack_Actor(BaseActor):
    """Actor for OSTrack with progressive alignment supervision."""

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg
        self.search_scale_factor = self.cfg.DATA.SEARCH.FACTOR
        self.num_classes = 0

    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.box_head.apply(self.fix_bn)

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def __call__(self, data):
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data, None)
        return loss, status

    def forward_pass(self, data):
        template_img = data['template_images'].permute(1, 0, 2, 3).contiguous()
        rgb_boxes = data["search_anno_rgb"][0]
        tir_boxes = data["search_anno_ir"][0]
        search_img = data['search_images'][0]

        # Convention: model geometry maps RGB search coordinates to TIR search coordinates.
        # dx/dy are normalized by the whole search region, not [-1,1] grid units.
        gt_affine = gt_diag_affine_from_boxes_xywh01(
            src_ir_xywh01=rgb_boxes,
            tgt_rgb_xywh01=tir_boxes,
            use_log_scale=True,
            scale_clamp=(0.25, 4.0),
        )

        outputs = self.net(template=template_img, search=search_img)
        out_dict = {
            'gt_affine': gt_affine,
            'gt_bbox': rgb_boxes,
            'tir_gt': tir_boxes,
        }
        out_dict.update(outputs)
        return out_dict

    @staticmethod
    def _apply_diag_geometry(coords, geom):
        center = coords.new_tensor([0.5, 0.5]).view(1, 1, 2)
        shift = geom[:, None, 0:2]
        scale = geom[:, None, 2:4].exp().clamp(0.25, 4.0)
        return (coords - center) * scale + center + shift

    def _shift_pixel_scale(self, device, dtype):
        search_size = getattr(self.cfg.DATA.SEARCH, 'SIZE', 256)
        if isinstance(search_size, (tuple, list)):
            sx = float(search_size[0])
            sy = float(search_size[1])
        else:
            sx = sy = float(search_size)
        return torch.tensor([sx, sy], device=device, dtype=dtype).view(1, 2)

    def _decomposed_offset_loss(self, pred, target, include_scale=True, prefix='OffsetLoss', pred_dirmag=None):
        """Supervise offset as direction + magnitude.

        Compatible geometry remains [dx, dy, log_sx, log_sy], but supervision is
        applied on explicit direction/magnitude factors.  Direction receives the
        larger default weight because a wrong sign/heading makes the alignment
        qualitatively wrong even if magnitude is close.
        """
        if pred is None or target is None:
            z = target.sum() * 0.0 if target is not None else torch.tensor(0.0)
            return z, {}

        eps = 1e-6
        scale_xy = self._shift_pixel_scale(pred.device, pred.dtype)
        pred_shift_px = pred[:, :2] * scale_xy
        tgt_shift_px = target[:, :2] * scale_xy
        tgt_mag_px = tgt_shift_px.norm(dim=-1)

        # dx/dy: 2-D direction + scalar magnitude.  pred_dirmag layout is
        # [xy_dir(2), xy_mag, scale_dir(2), scale_mag(2)].  xy_mag is stored in
        # normalized search coordinates, then converted to pixels robustly.
        if pred_dirmag is not None:
            pred_xy_dir = F.normalize(pred_dirmag[:, 0:2], dim=-1, eps=eps)
            pred_xy_vec_norm = pred_xy_dir * pred_dirmag[:, 2:3].clamp_min(0.0)
            pred_mag_px = (pred_xy_vec_norm * scale_xy).norm(dim=-1)
        else:
            pred_xy_dir = F.normalize(pred_shift_px, dim=-1, eps=eps)
            pred_mag_px = pred_shift_px.norm(dim=-1)

        loss_xy_mag = F.smooth_l1_loss(torch.log1p(pred_mag_px), torch.log1p(tgt_mag_px), reduction='mean')
        valid_dir = tgt_mag_px > getattr(self.cfg.TRAIN, 'OFFSET_DIR_MIN_PX', 5.0)
        if valid_dir.any():
            tgt_xy_dir = F.normalize(tgt_shift_px, dim=-1, eps=eps)
            cos = F.cosine_similarity(pred_xy_dir[valid_dir], tgt_xy_dir[valid_dir], dim=-1, eps=eps)
            loss_xy_dir = (1.0 - cos).mean()
        else:
            loss_xy_dir = pred.sum() * 0.0
        loss_xy_raw = F.smooth_l1_loss(pred[:, :2], target[:, :2], reduction='mean')

        # log_sx/log_sy: per-axis expansion/shrink direction + per-axis magnitude.
        if include_scale:
            pred_scale = pred[:, 2:4]
            tgt_scale = target[:, 2:4]
            scale_dir_tau = getattr(self.cfg.TRAIN, 'OFFSET_SCALE_DIR_TAU', 0.03)
            if pred_dirmag is not None:
                pred_scale_dir = pred_dirmag[:, 3:5].clamp(-1.0, 1.0)
                pred_scale_mag = pred_dirmag[:, 5:7].clamp_min(0.0)
            else:
                pred_scale_dir = torch.tanh(pred_scale / max(float(scale_dir_tau), 1e-6))
                pred_scale_mag = pred_scale.abs()

            loss_scale_mag = F.smooth_l1_loss(pred_scale_mag, tgt_scale.abs(), reduction='mean')
            tgt_scale_dir = torch.tanh(tgt_scale / max(float(scale_dir_tau), 1e-6))
            scale_dir_min = getattr(self.cfg.TRAIN, 'OFFSET_SCALE_DIR_MIN_LOG', 0.015)
            valid_scale_dir = tgt_scale.abs() > float(scale_dir_min)
            if valid_scale_dir.any():
                loss_scale_dir = F.smooth_l1_loss(
                    pred_scale_dir[valid_scale_dir],
                    tgt_scale_dir[valid_scale_dir].detach(),
                    reduction='mean'
                )
            else:
                loss_scale_dir = pred_scale.sum() * 0.0
            loss_scale_raw = F.smooth_l1_loss(pred_scale, tgt_scale, reduction='mean')
        else:
            loss_scale_dir = pred[:, 2:4].sum() * 0.0
            loss_scale_mag = pred[:, 2:4].sum() * 0.0
            loss_scale_raw = pred[:, 2:4].sum() * 0.0

        w_xy_dir = getattr(self.cfg.TRAIN, 'OFFSET_XY_DIR_WEIGHT', 0.35)
        w_xy_mag = getattr(self.cfg.TRAIN, 'OFFSET_XY_MAG_WEIGHT', 0.15)
        w_xy_raw = getattr(self.cfg.TRAIN, 'OFFSET_XY_RAW_WEIGHT', 0.05)
        w_scale_dir = getattr(self.cfg.TRAIN, 'OFFSET_SCALE_DIR_WEIGHT', 0.30) if include_scale else 0.0
        w_scale_mag = getattr(self.cfg.TRAIN, 'OFFSET_SCALE_MAG_WEIGHT', 0.10) if include_scale else 0.0
        w_scale_raw = getattr(self.cfg.TRAIN, 'OFFSET_SCALE_RAW_WEIGHT', 0.05) if include_scale else 0.0

        total = (
            w_xy_dir * loss_xy_dir
            + w_xy_mag * loss_xy_mag
            + w_xy_raw * loss_xy_raw
            + w_scale_dir * loss_scale_dir
            + w_scale_mag * loss_scale_mag
            + w_scale_raw * loss_scale_raw
        )
        logs = {
            f'{prefix}/xy_dir': loss_xy_dir.detach(),
            f'{prefix}/xy_mag': loss_xy_mag.detach(),
            f'{prefix}/xy_raw': loss_xy_raw.detach(),
            f'{prefix}/scale_dir': loss_scale_dir.detach(),
            f'{prefix}/scale_mag': loss_scale_mag.detach(),
            f'{prefix}/scale_raw': loss_scale_raw.detach(),
            # Compatibility aliases for old log parsers.
            f'{prefix}/xy': loss_xy_raw.detach(),
            f'{prefix}/mag': loss_xy_mag.detach(),
            f'{prefix}/dir': loss_xy_dir.detach(),
            f'{prefix}/scale': loss_scale_raw.detach(),
        }
        return total, logs

    @staticmethod
    def _weighted_smooth_l1(pred, target, weight, beta=1.0):
        raw = F.smooth_l1_loss(
            pred,
            target.detach(),
            reduction='none',
            beta=max(float(beta), 1e-6),
        ).mean(dim=-1)
        w = weight.detach().clamp_min(0.0)
        return (raw * w).sum() / w.sum().clamp_min(1.0)

    def _stage_geometry_loss(self, pred_dict, gt_offsets):
        terms, logs = [], {}
        stage_weights = {
            'center': getattr(self.cfg.TRAIN, 'CENTER_STAGE_FACTOR', 0.20),
            'scale': getattr(self.cfg.TRAIN, 'SCALE_STAGE_FACTOR', 0.40),
            'refinement': getattr(self.cfg.TRAIN, 'REFINE_STAGE_FACTOR', 0.20),
        }
        for name, w in stage_weights.items():
            if w <= 0:
                continue

            # Scale is supervised at the relation-statistical solution so the
            # learned residual branch cannot hide an inaccurate OT scale.
            if name == 'scale' and pred_dict.get('scale_geometry_ot', None) is not None:
                key = 'scale_geometry_ot'
                pred_dirmag = None
            else:
                key = f'{name}_geometry'
                pred_dirmag = pred_dict.get(f'{name}_geometry_dirmag', None)
            if key not in pred_dict or pred_dict[key] is None:
                continue

            pred = pred_dict[key]
            if name == 'center':
                target = torch.cat([gt_offsets[:, :2], torch.zeros_like(gt_offsets[:, 2:4])], dim=-1)
                val, sub_logs = self._decomposed_offset_loss(
                    pred, target, include_scale=False,
                    prefix=f'StageLoss/{name}', pred_dirmag=pred_dirmag,
                )
            else:
                target = gt_offsets
                val, sub_logs = self._decomposed_offset_loss(
                    pred, target, include_scale=True,
                    prefix=f'StageLoss/{name}', pred_dirmag=pred_dirmag,
                )
            terms.append(float(w) * val)
            logs[f'StageGeo/{name}'] = val.detach()
            logs.update(sub_logs)

        if not terms:
            z = gt_offsets.sum() * 0.0
            return z, logs
        return torch.stack(terms).sum(), logs

    def _transition_supervision_loss(self, pred_dict, gt_offsets, gt_bbox):
        """Supervise relation probabilities and their first/second moments.

        The first moment constrains expected correspondence position. The second
        central moment directly constrains relation width, which is the relation-
        space signal for one-to-many support and cross-modal scale change.
        """
        relation_terms, conf_terms, mean_terms, var_terms, logs = [], [], [], [], {}
        stage_width = {
            'center': 3.0,
            'scale': 2.0,
            'refinement': 1.25,
        }
        mean_factor = {
            'center': 1.0,
            'scale': 1.0,
            'refinement': 0.5,
        }
        var_factor = {
            'center': 0.0,
            'scale': 1.0,
            'refinement': 0.5,
        }
        mean_beta = getattr(self.cfg.TRAIN, 'RELATION_MEAN_BETA', 0.5)
        var_beta = getattr(self.cfg.TRAIN, 'RELATION_VAR_BETA', 0.5)

        for name, width_multiplier in stage_width.items():
            t_key = f'{name}_transition'
            c_key = f'{name}_coords'
            if t_key not in pred_dict or c_key not in pred_dict:
                continue
            pred_t = pred_dict[t_key]
            coords = pred_dict[c_key]
            if pred_t is None or coords is None:
                continue

            target_xy = self._apply_diag_geometry(coords, gt_offsets)
            inside = (
                (target_xy[..., 0] >= 0.0) & (target_xy[..., 0] <= 1.0) &
                (target_xy[..., 1] >= 0.0) & (target_xy[..., 1] <= 1.0)
            ).float()

            token_side = max(int(round(coords.shape[1] ** 0.5)), 1)
            token_size = 1.0 / float(token_side)
            target_scale = gt_offsets[:, 2:4].exp().clamp(0.25, 4.0)
            footprint = 0.5 * token_size * torch.sqrt(target_scale.pow(2) + 1.0)
            sigma_xy = (float(width_multiplier) * footprint).clamp_min(0.5 * token_size)
            residual = coords.unsqueeze(1) - target_xy.unsqueeze(2)
            mahalanobis = (residual / sigma_xy[:, None, None, :]).pow(2).sum(dim=-1)
            target = torch.exp(-0.5 * mahalanobis)
            target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            pred = pred_t.clamp_min(1e-8)
            ce_row = -(target.detach() * pred.log()).sum(dim=-1)

            # Preserve the existing target-aware weighting for relation CE. The
            # new second-moment term adds scale observability without introducing
            # another feature branch or response head.
            box_center = gt_bbox[:, 0:2] + 0.5 * gt_bbox[:, 2:4]
            box_sigma = (0.5 * gt_bbox[:, 2:4]).clamp_min(token_size)
            row_gaussian = torch.exp(
                -0.5 * ((coords - box_center[:, None, :]) / box_sigma[:, None, :]).pow(2).sum(dim=-1)
            )
            weight = inside * (0.25 + 0.75 * row_gaussian.detach())

            row_key = f'{name}_row_mass'
            if row_key in pred_dict and pred_dict[row_key] is not None:
                row_mass = pred_dict[row_key]
                logs[f'Mass/{name}'] = row_mass.detach().sum(dim=-1).mean()
                logs[f'MassRow/{name}'] = row_mass.detach().mean()
            src_conf_key = f'{name}_src_conf'
            if src_conf_key in pred_dict and pred_dict[src_conf_key] is not None:
                src_conf = pred_dict[src_conf_key]
                logs[f'OTConf/src_{name}'] = src_conf.detach().mean()
                weight = weight * (0.25 + 0.75 * src_conf.detach().clamp(0.0, 1.0))

            relation_loss = (ce_row * weight).sum() / weight.sum().clamp_min(1.0)
            relation_terms.append(relation_loss)
            logs[f'Relation/{name}'] = relation_loss.detach()
            logs[f'Trans/{name}'] = relation_loss.detach()  # compatibility alias

            # Relation moments in token units. Multiplying by P and P^2 avoids
            # vanishing losses caused by normalized-coordinate magnitudes.
            target_detached = target.detach()
            pred_mean = torch.matmul(pred_t, coords)
            target_mean = torch.matmul(target_detached, coords)
            target_grid = coords.unsqueeze(1)
            pred_var = (
                pred_t.unsqueeze(-1)
                * (target_grid - pred_mean.unsqueeze(2)).pow(2)
            ).sum(dim=2)
            target_var = (
                target_detached.unsqueeze(-1)
                * (target_grid - target_mean.unsqueeze(2)).pow(2)
            ).sum(dim=2)

            mean_loss = self._weighted_smooth_l1(
                pred_mean * float(token_side),
                target_mean * float(token_side),
                weight,
                beta=mean_beta,
            )
            var_loss = self._weighted_smooth_l1(
                pred_var * float(token_side ** 2),
                target_var * float(token_side ** 2),
                weight,
                beta=var_beta,
            )
            if mean_factor[name] > 0:
                mean_terms.append((float(mean_factor[name]), mean_loss))
            if var_factor[name] > 0:
                var_terms.append((float(var_factor[name]), var_loss))

            logs[f'MomentMean/{name}'] = mean_loss.detach()
            logs[f'MomentVar/{name}'] = var_loss.detach()
            pred_var_mean = pred_var.detach().mean()
            target_var_mean = target_var.detach().mean()
            logs[f'PredVar/{name}'] = pred_var_mean
            logs[f'GTVar/{name}'] = target_var_mean
            logs[f'VarRatio/{name}'] = pred_var_mean / target_var_mean.clamp_min(1e-8)
            # Compatibility alias: old DenseRef was a first-moment loss.
            logs[f'DenseRef/{name}'] = mean_loss.detach()

            tgt_conf_key = f'{name}_tgt_conf'
            if tgt_conf_key in pred_dict and pred_dict[tgt_conf_key] is not None:
                logs[f'OTConf/tgt_{name}'] = pred_dict[tgt_conf_key].detach().mean()
            col_key = f'{name}_col_mass'
            if col_key in pred_dict and pred_dict[col_key] is not None:
                logs[f'MassCol/{name}'] = pred_dict[col_key].detach().mean()
            entropy_key = f'{name}_relation_entropy'
            if entropy_key in pred_dict and pred_dict[entropy_key] is not None:
                logs[f'RelationEntropy/{name}'] = pred_dict[entropy_key].detach().mean()

            conf_key = f'{name}_confidence'
            if conf_key in pred_dict and pred_dict[conf_key] is not None:
                conf_target = (
                    weight.sum(dim=-1) / inside.sum(dim=-1).clamp_min(1.0)
                ).detach().clamp(0.0, 1.0)
                conf = pred_dict[conf_key].clamp(0.0, 1.0)
                conf_loss = F.smooth_l1_loss(conf, conf_target, reduction='mean')
                conf_terms.append(conf_loss)
                logs[f'ConfLoss/{name}'] = conf_loss.detach()
                logs[f'ConfVal/{name}'] = conf.detach().mean()

        zero = gt_offsets.sum() * 0.0
        relation_loss = torch.stack(relation_terms).mean() if relation_terms else zero
        conf_loss = torch.stack(conf_terms).mean() if conf_terms else zero

        def weighted_stage_mean(items):
            if not items:
                return zero
            total_weight = sum(w for w, _ in items)
            return torch.stack([w * value for w, value in items]).sum() / max(total_weight, 1e-8)

        mean_loss = weighted_stage_mean(mean_terms)
        var_loss = weighted_stage_mean(var_terms)
        return relation_loss, conf_loss, mean_loss, var_loss, logs


    def _offset_diagnostics(self, pred_offsets, gt_offsets, prefix='Off', compact=True):
        """
        Diagnostics only, not used in loss.
        Offset convention: [dx, dy, log_sx, log_sy] in normalized search coordinates.
        Pixel metrics convert dx/dy by DATA.SEARCH.SIZE.
        Ratio metrics:
          - center_ratio = ||pred_shift|| / ||gt_shift||; <1 means under-prediction, >1 over-prediction.
          - scale_ratio  = exp(pred_log_scale) / exp(gt_log_scale); <1 smaller, >1 larger.
          - *_sym        = max(r, 1/r), always >=1 and closer to 1 is better.
        """
        if pred_offsets is None or gt_offsets is None:
            return {}
        eps = 1e-6
        pred = pred_offsets.detach()
        gt = gt_offsets.detach()
        if pred.shape != gt.shape:
            return {}

        search_size = getattr(self.cfg.DATA.SEARCH, 'SIZE', 256)
        if isinstance(search_size, (tuple, list)):
            sx = float(search_size[0])
            sy = float(search_size[1])
        else:
            sx = sy = float(search_size)
        scale_xy = pred.new_tensor([sx, sy]).view(1, 2)

        pred_shift_px = pred[:, 0:2] * scale_xy
        gt_shift_px = gt[:, 0:2] * scale_xy
        err_shift_px = pred_shift_px - gt_shift_px

        pred_mag = pred_shift_px.norm(dim=-1)
        gt_mag = gt_shift_px.norm(dim=-1)
        err_mag = err_shift_px.norm(dim=-1)

        # Ratio is meaningful only when GT displacement is non-trivial.
        valid = gt_mag > 1.0
        ratio = pred_mag / gt_mag.clamp_min(eps)
        ratio_valid = ratio[valid] if valid.any() else ratio
        ratio_sym = torch.maximum(ratio_valid, 1.0 / ratio_valid.clamp_min(eps))

        cos = F.cosine_similarity(pred_shift_px, gt_shift_px, dim=-1, eps=eps)
        sign_x = (torch.sign(pred[:, 0]) == torch.sign(gt[:, 0])).float()
        sign_y = (torch.sign(pred[:, 1]) == torch.sign(gt[:, 1])).float()

        pred_scale = pred[:, 2:4].exp().clamp(0.05, 20.0)
        gt_scale = gt[:, 2:4].exp().clamp(0.05, 20.0)
        scale_ratio = pred_scale / gt_scale.clamp_min(eps)
        scale_ratio_sym = torch.maximum(scale_ratio, 1.0 / scale_ratio.clamp_min(eps))
        scale_pct_err = (scale_ratio - 1.0).abs() * 100.0
        scale_dir_valid = gt[:, 2:4].abs() > getattr(self.cfg.TRAIN, 'OFFSET_SCALE_DIR_MIN_LOG', 0.015)
        scale_sign_match = (torch.sign(pred[:, 2:4]) == torch.sign(gt[:, 2:4])).float()
        scale_dir_acc = scale_sign_match[scale_dir_valid].mean() if scale_dir_valid.any() else scale_sign_match.mean()

        gt_dir = gt_shift_px / gt_mag.clamp_min(eps).unsqueeze(-1)
        parallel = (err_shift_px * gt_dir).sum(dim=-1).abs()
        perp_vec = err_shift_px - ((err_shift_px * gt_dir).sum(dim=-1, keepdim=True) * gt_dir)
        perp = perp_vec.norm(dim=-1)
        angle = torch.rad2deg(torch.acos(cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)))

        logs = {
            f'{prefix}/err_px': err_mag.mean().item(),
            f'{prefix}/ratio': ratio_valid.mean().item(),
            f'{prefix}/cos': cos.mean().item(),
            f'{prefix}/scale_pct': scale_pct_err.mean().item(),
            f'{prefix}/scale_dir_acc': scale_dir_acc.item(),
        }
        if compact:
            return logs

        logs.update({
            f'{prefix}/dx_err_px': err_shift_px[:, 0].abs().mean().item(),
            f'{prefix}/dy_err_px': err_shift_px[:, 1].abs().mean().item(),
            f'{prefix}/parallel_err_px': parallel.mean().item(),
            f'{prefix}/perp_err_px': perp.mean().item(),
            f'{prefix}/angle_deg': angle[valid].mean().item() if valid.any() else angle.mean().item(),
            f'{prefix}/pred_mag_px': pred_mag.mean().item(),
            f'{prefix}/gt_mag_px': gt_mag.mean().item(),
            f'{prefix}/scale_ratio': scale_ratio.mean().item(),
        })
        return logs

    def _stage_offset_diagnostics(self, pred_dict, gt_offsets):
        """Compact stage diagnostics focused on Scale and Refinement behavior."""
        logs = {}
        final_scale_pct = {}
        name_map = {
            'center': 'S1',
            'scale': 'S2',
            'refinement': 'S3',
        }
        for name, short_name in name_map.items():
            key = f'{name}_geometry'
            if key not in pred_dict or pred_dict[key] is None:
                continue
            stage_pred = pred_dict[key]
            if name == 'center':
                stage_gt = torch.cat([gt_offsets[:, :2], torch.zeros_like(gt_offsets[:, 2:4])], dim=-1)
            else:
                stage_gt = gt_offsets
            stage_logs = self._offset_diagnostics(stage_pred, stage_gt, prefix=short_name)
            logs[f'{short_name}/err_px'] = stage_logs[f'{short_name}/err_px']
            if name != 'center':
                logs[f'{short_name}/scale_pct'] = stage_logs[f'{short_name}/scale_pct']
                final_scale_pct[name] = stage_logs[f'{short_name}/scale_pct']

            ot_key = f'{name}_geometry_ot'
            if name != 'center' and pred_dict.get(ot_key, None) is not None:
                ot_prefix = f'{short_name}_OT'
                ot_logs = self._offset_diagnostics(pred_dict[ot_key], gt_offsets, prefix=ot_prefix)
                logs[f'{ot_prefix}/scale_pct'] = ot_logs[f'{ot_prefix}/scale_pct']

        if 'scale' in final_scale_pct and 'refinement' in final_scale_pct:
            logs['S2_to_S3/scale_gain'] = final_scale_pct['scale'] - final_scale_pct['refinement']

        refine_delta = pred_dict.get('refinement_delta', None)
        if refine_delta is not None:
            scale_xy = self._shift_pixel_scale(refine_delta.device, refine_delta.dtype)
            logs['Refine/delta_xy_px'] = (refine_delta[:, :2] * scale_xy).norm(dim=-1).mean().item()
            logs['Refine/delta_scale_pct'] = (
                (refine_delta[:, 2:4].exp() - 1.0).abs() * 100.0
            ).mean().item()
            near_xy = refine_delta[:, :2].abs() >= (0.95 * 0.02)
            near_scale = refine_delta[:, 2:4].abs() >= (0.95 * 0.03)
            logs['Refine/clamp_ratio'] = torch.cat([near_xy, near_scale], dim=-1).float().mean().item()
        return logs

    def compute_losses(self, pred_dict, gt_dict, bias, return_status=True):
        if pred_dict is None:
            return None

        gt_bbox = pred_dict['gt_bbox']
        gt_gaussian_maps = generate_heatmap(
            pred_dict['gt_bbox'].unsqueeze(0),
            self.cfg.DATA.SEARCH.SIZE,
            self.cfg.MODEL.BACKBONE.STRIDE,
        )[-1].unsqueeze(1)

        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError('Network outputs is NAN! Stop Training')
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(0.0, 1.0)
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except Exception:
            giou_loss = torch.tensor(0.0, device=pred_boxes.device)
            iou = torch.tensor(0.0, device=pred_boxes.device)
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
        location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps) if 'score_map' in pred_dict else torch.tensor(0.0, device=l1_loss.device)

        gt_offsets = pred_dict['gt_affine']
        if 'pred_offset' in pred_dict and pred_dict['pred_offset'] is not None:
            loss_offset, offset_loss_logs = self._decomposed_offset_loss(
                pred_dict['pred_offset'], gt_offsets, include_scale=True, prefix='OffsetLoss', pred_dirmag=pred_dict.get('pred_offset_dirmag', None)
            )
        else:
            loss_offset = gt_offsets.sum() * 0.0
            offset_loss_logs = {}

        loss_stage_geo, stage_logs = self._stage_geometry_loss(pred_dict, gt_offsets)
        loss_trans, _, loss_moment_mean, loss_moment_var, trans_logs = self._transition_supervision_loss(
            pred_dict, gt_offsets, gt_bbox
        )

        loss_track = (
            self.loss_weight['giou'] * giou_loss
            + self.loss_weight['l1'] * l1_loss
            + self.loss_weight['focal'] * location_loss
        )
        loss = loss_track
        if 'pred_offset' in pred_dict:
            # Decomposed offset loss has a larger numeric scale than the previous
            # SmoothL1-only loss. Use a separate lower default and do not inherit
            # the legacy OFFSET_WEIGHT=50 setting.
            loss = loss + getattr(self.cfg.TRAIN, 'DECOMP_OFFSET_WEIGHT', 10.0) * loss_offset
        loss = loss + getattr(self.cfg.TRAIN, 'STAGE_ALIGN_WEIGHT_REBALANCED', 0.25) * loss_stage_geo
        loss = loss + getattr(self.cfg.TRAIN, 'RELATION_WEIGHT', 0.02) * loss_trans
        loss = loss + getattr(self.cfg.TRAIN, 'RELATION_MEAN_WEIGHT', 0.03) * loss_moment_mean
        loss = loss + getattr(self.cfg.TRAIN, 'RELATION_VAR_WEIGHT', 0.0) * loss_moment_var
        if 'loss_detail_div' in pred_dict:
            loss = loss + getattr(self.cfg.TRAIN, 'DETAIL_DIV_WEIGHT', 0.001) * pred_dict['loss_detail_div']
        if 'loss_detail_peak' in pred_dict:
            loss = loss + getattr(self.cfg.TRAIN, 'DETAIL_PEAK_WEIGHT', 0.0001) * pred_dict['loss_detail_peak']

        if not return_status:
            return loss

        status = {
            'Loss/total': loss.item(),
            'Loss/track': loss_track.item(),
            'IoU': iou.detach().mean().item(),
            'Loss/offset': loss_offset.item(),
            'Loss/stage_geo': loss_stage_geo.item(),
            'Loss/trans': loss_trans.item(),
            'Loss/moment_mean': loss_moment_mean.item(),
            'Loss/moment_var': loss_moment_var.item(),
        }
        status.update(self._offset_diagnostics(pred_dict.get('pred_offset', None), gt_offsets, prefix='Off', compact=True))
        status.update(self._stage_offset_diagnostics(pred_dict, gt_offsets))

        # Minimal OT health signals. Full breakdown is available through
        # TRAIN.ALIGN_DEBUG_LOG=True.
        for k in (
            'Mass/center', 'Mass/scale', 'Mass/refinement',
            'MomentVar/scale', 'VarRatio/scale',
        ):
            if k in trans_logs:
                v = trans_logs[k]
                status[k] = v.item() if torch.is_tensor(v) else v

        if getattr(self.cfg.TRAIN, 'ALIGN_DEBUG_LOG', False):
            for stage in ('center', 'scale', 'refinement'):
                cue_weights = pred_dict.get(f'{stage}_cue_weights', None)
                if cue_weights is not None:
                    mean_family = cue_weights.detach().mean(dim=0)
                    for idx, family in enumerate(('R', 'S', 'D')):
                        status[f'Cue/{stage}_{family}'] = mean_family[idx].item()
            status.update({k: v.item() if torch.is_tensor(v) else v for k, v in offset_loss_logs.items()})
            status.update({k: v.item() if torch.is_tensor(v) else v for k, v in stage_logs.items()})
            status.update({k: v.item() if torch.is_tensor(v) else v for k, v in trans_logs.items()})
            status.update(self._offset_diagnostics(pred_dict.get('pred_offset', None), gt_offsets, prefix='OffDebug', compact=False))
        return loss, status
