import math

from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor, PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import numpy as np
def diag_affine_offset_to_local_homography(offset, search_size, log_scale=True):
    """
    Convert predicted local offset [dx, dy, log_sx, log_sy] to a pixel-space
    homography on the search crop.

    Model convention:
        p_tir_norm = scale * (p_rgb_norm - 0.5) + 0.5 + shift

    Therefore in crop pixel coordinates:
        p_tir_pix = scale * p_rgb_pix + search_size * (shift + 0.5 * (1 - scale))

    Return: (B,3,3), RGB local crop -> TIR local crop.
    """
    if offset.dim() == 1:
        offset = offset.unsqueeze(0)
    offset = offset.detach().float()
    b = offset.shape[0]
    dx, dy, sx, sy = offset[:, 0], offset[:, 1], offset[:, 2], offset[:, 3]
    if log_scale:
        sx = sx.exp()
        sy = sy.exp()

    search_size = float(search_size)
    tx = search_size * (dx + 0.5 * (1.0 - sx))
    ty = search_size * (dy + 0.5 * (1.0 - sy))

    h = torch.eye(3, device=offset.device, dtype=offset.dtype).unsqueeze(0).repeat(b, 1, 1)
    h[:, 0, 0] = sx
    h[:, 1, 1] = sy
    h[:, 0, 2] = tx
    h[:, 1, 2] = ty
    return h


def crop_to_global_homography(state, resize_factor, search_size, device, dtype):
    """
    Build crop-local -> full-image homography for sample_target().

    sample_target crops a square centered at current RGB state and resizes it to
    search_size. The crop side length in the full image is search_size / resize_factor.
    """
    x1, y1, w, h = [float(v) for v in state]
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    crop_to_global_scale = 1.0 / float(resize_factor)
    half_side = 0.5 * float(search_size) * crop_to_global_scale

    return torch.tensor([
        [crop_to_global_scale, 0.0, cx - half_side],
        [0.0, crop_to_global_scale, cy - half_side],
        [0.0, 0.0, 1.0],
    ], device=device, dtype=dtype).unsqueeze(0)


def local_offset_to_global_homography(local_offset, state, resize_factor, search_size):
    """
    Convert local normalized offset to full-image homographies.

    Return:
        global_rgb_to_tir: (B,3,3), RGB full-image coordinates -> TIR full-image coordinates.
        global_tir_to_rgb: (B,3,3), inverse transform for cv2.warpPerspective(tir, M, ...).
        local_rgb_to_tir:  (B,3,3), RGB local crop -> TIR local crop.
    """
    local_rgb_to_tir = diag_affine_offset_to_local_homography(
        local_offset, search_size=search_size, log_scale=True
    )
    t_crop_to_global = crop_to_global_homography(
        state=state,
        resize_factor=resize_factor,
        search_size=search_size,
        device=local_rgb_to_tir.device,
        dtype=local_rgb_to_tir.dtype,
    ).expand(local_rgb_to_tir.shape[0], -1, -1)
    t_global_to_crop = torch.linalg.inv(t_crop_to_global)

    global_rgb_to_tir = t_crop_to_global @ local_rgb_to_tir @ t_global_to_crop
    global_tir_to_rgb = torch.linalg.inv(global_rgb_to_tir)
    return global_rgb_to_tir, global_tir_to_rgb, local_rgb_to_tir


def get_align(rgb, tir, offset):
    tir_resized_aligned = cv2.warpPerspective(
        tir, offset, (rgb.shape[1], rgb.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    img_rgb_size = cv2.merge((rgb, tir_resized_aligned))
    return img_rgb_size

class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        checkpoint = torch.load(self.params.checkpoint, map_location='cpu')['net']
        missing_keys, unexpected_keys = network.load_state_dict(checkpoint, strict=False)
        if missing_keys:
            raise RuntimeError(
                "Tracker checkpoint is incompatible with the current model code. "
                f"Missing required parameters: {missing_keys}"
            )
        if unexpected_keys:
            print(
                "Ignored obsolete ablation-only checkpoint parameters: "
                f"{len(unexpected_keys)} keys"
            )
        
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorMM()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        if getattr(params, 'debug', None) is None:
            setattr(params, 'debug', 0)
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.dynamic_z_tensor = None

        # Relation-verified online alignment update (Sec. 3.6).
        self.tocu_track_score_thr = 0.5
        self.tocu_gain_thr = 0.01
        self.tocu_relation_conf_thr = 0.15
        self.tocu_relation_entropy_thr = 0.92
        self.tocu_valid_ratio_thr = 0.65
        self.tocu_relation_mahal_thr = 3.0
        self.tocu_commit_streak = 2
        self.tocu_kernel_base_scale = 0.75
        self.tocu_kernel_relation_var_scale = 1.0
        self._tocu_good_streak = 0

        # ------------------------------------------------------------------
        # Dynamic template FIFO
        # ------------------------------------------------------------------
        # Manual constants:
        #   FIFO length = 10
        #   insert every 20 frames
        #   insert only when tracking score > 0.6
        # Tracking always uses the highest-score entry currently in the FIFO.
        self.template_fifo_len = 10
        self.template_update_interval = 20
        self.template_score_thr = 0.6
        self.template_fifo = []

    @staticmethod
    def _apply_normalized_diag_geometry(coords, geometry):
        """Apply [dx,dy,log_sx,log_sy] to normalized RCCA token coords."""
        center = coords.new_tensor([0.5, 0.5]).view(1, 1, 2)
        scale = geometry[:, None, 2:4].exp().clamp(0.25, 4.0)
        shift = geometry[:, None, 0:2]
        return (coords - center) * scale + center + shift

    def _relation_counterfactual_score(self, appearance_cost, coords, mapped_coords,
                                       relation_var, token_weight):
        """Query geometry-independent appearance cost under one geometry hypothesis.

        The spatial kernel is widened by the relation-derived covariance rather
        than by a hand-crafted scale coefficient.  This preserves one-to-many
        support while keeping candidate verification independent from the final
        relation weights themselves.
        """
        b, n, _ = coords.shape
        grid_step = 1.0 / max(float(n) ** 0.5, 1.0)
        base_var = (self.tocu_kernel_base_scale * grid_step) ** 2
        kernel_var = (base_var + self.tocu_kernel_relation_var_scale * relation_var).clamp_min(1e-6)

        # [B, N_src, N_tgt, 2]
        diff = coords[:, None, :, :] - mapped_coords[:, :, None, :]
        mahal = (diff.square() / kernel_var[:, :, None, :]).sum(dim=-1)
        kernel = torch.exp(-0.5 * mahal)
        kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        token_cost = (kernel * appearance_cost).sum(dim=-1)
        score = (token_cost * token_weight).sum(dim=-1) / token_weight.sum(dim=-1).clamp_min(1e-8)
        return score

    def _find_rcca_stage(self, out_dict):
        """Return the deepest RCCA stage that exposes cached verifier evidence."""
        for stage in ("refinement", "scale", "center"):
            required = (
                f"{stage}_appearance_cost",
                f"{stage}_coords",
            )
            if all(k in out_dict and out_dict[k] is not None for k in required):
                if (f"{stage}_transition" in out_dict and out_dict[f"{stage}_transition"] is not None) or \
                        (f"{stage}_transport_plan" in out_dict and out_dict[f"{stage}_transport_plan"] is not None) or \
                        (f"{stage}_relation" in out_dict and out_dict[f"{stage}_relation"] is not None):
                    return stage
        return None

    def _tocu_relation_update(self, out_dict, local_offset, max_score):
        """Single-forward History-vs-Candidate verifier using shared RCCA evidence.

        Candidate generation uses the final RCCA relation. Candidate-vs-history
        comparison deliberately uses ``appearance_cost`` computed before geometry
        injection, while the final relation is used only for reliability/support.
        This avoids using R to directly verify geometry decoded from the same R.
        """
        info = {
            "method": "relation",
            "accepted": False,
            "reason": "unknown",
        }
        stage = self._find_rcca_stage(out_dict)
        if stage is None:
            self._tocu_good_streak = 0
            info["reason"] = "missing_rcca_evidence"
            return False, info

        appearance_cost = out_dict[f"{stage}_appearance_cost"].detach().float()
        coords = out_dict[f"{stage}_coords"].detach().float()
        transition = out_dict.get(f"{stage}_transition", None)
        transport = out_dict.get(f"{stage}_transport_plan", out_dict.get(f"{stage}_relation", None))
        if transition is None:
            if transport is None:
                self._tocu_good_streak = 0
                info["reason"] = "missing_relation"
                return False, info
            transport = transport.detach().float()
            transition = transport / transport.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            transition = transition.detach().float()
        if transport is not None:
            transport = transport.detach().float()

        # q_i is a reliability weight only. It is identical for history and
        # candidate, so R cannot directly favor the proposal it produced.
        src_conf = out_dict.get(f"{stage}_src_conf", None)
        if src_conf is not None:
            token_weight = src_conf.detach().float().clamp(0.0, 1.0)
        elif transport is not None:
            row_mass = transport.sum(dim=-1)
            token_weight = row_mass / row_mass.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            token_weight = torch.ones(
                transition.shape[:2], device=transition.device, dtype=transition.dtype
            )

        # Relation-supported soft correspondence and covariance.
        relation_center = torch.matmul(transition, coords)
        relation_diff = coords[:, None, :, :] - relation_center[:, :, None, :]
        relation_var = (transition[:, :, :, None] * relation_diff.square()).sum(dim=2).clamp_min(1e-8)

        geometry = local_offset.detach().float()
        hist_coords = coords
        cand_coords = self._apply_normalized_diag_geometry(coords, geometry)

        e_hist = self._relation_counterfactual_score(
            appearance_cost, coords, hist_coords, relation_var, token_weight
        )
        e_cand = self._relation_counterfactual_score(
            appearance_cost, coords, cand_coords, relation_var, token_weight
        )
        gain = (e_hist - e_cand) / e_hist.abs().clamp_min(1e-8)

        relation_conf = token_weight.mean(dim=-1)
        relation_entropy = out_dict.get(f"{stage}_relation_entropy", None)
        if relation_entropy is None:
            entropy = -(transition.clamp_min(1e-8) * transition.clamp_min(1e-8).log()).sum(dim=-1)
            entropy = entropy / math.log(max(int(transition.shape[-1]), 2))
        else:
            entropy = relation_entropy.detach().float()
        entropy_global = (entropy * token_weight).sum(dim=-1) / token_weight.sum(dim=-1).clamp_min(1e-8)

        # Candidate must stay within the spatial support of the shared relation.
        # This is a support/safety check, not the history-vs-candidate score.
        support_mahal = ((relation_center - cand_coords).square() / relation_var.clamp_min(1e-6)).sum(dim=-1).sqrt()
        support_mahal = (support_mahal * token_weight).sum(dim=-1) / token_weight.sum(dim=-1).clamp_min(1e-8)

        valid_token = ((cand_coords >= 0.0) & (cand_coords <= 1.0)).all(dim=-1).float()
        valid_ratio = (valid_token * token_weight).sum(dim=-1) / token_weight.sum(dim=-1).clamp_min(1e-8)

        # Test tracker is batch-size one; keep all returned values scalar.
        gain_v = float(gain[0].item())
        e_hist_v = float(e_hist[0].item())
        e_cand_v = float(e_cand[0].item())
        conf_v = float(relation_conf[0].item())
        entropy_v = float(entropy_global[0].item())
        mahal_v = float(support_mahal[0].item())
        valid_v = float(valid_ratio[0].item())

        gates = {
            "tracking": float(max_score) > self.tocu_track_score_thr,
            "appearance_gain": gain_v > self.tocu_gain_thr,
            "relation_conf": conf_v > self.tocu_relation_conf_thr,
            "relation_entropy": entropy_v < self.tocu_relation_entropy_thr,
            "relation_support": mahal_v < self.tocu_relation_mahal_thr,
            "valid_region": valid_v > self.tocu_valid_ratio_thr,
        }
        current_pass = all(gates.values())
        self._tocu_good_streak = self._tocu_good_streak + 1 if current_pass else 0
        accepted = current_pass and self._tocu_good_streak >= max(self.tocu_commit_streak, 1)

        info.update({
            "stage": stage,
            "accepted": bool(accepted),
            "current_pass": bool(current_pass),
            "streak": int(self._tocu_good_streak),
            "appearance_hist": e_hist_v,
            "appearance_cand": e_cand_v,
            "appearance_gain": gain_v,
            "relation_conf": conf_v,
            "relation_entropy": entropy_v,
            "relation_support_mahal": mahal_v,
            "valid_ratio": valid_v,
            "tracking_score": float(max_score),
            "gates": gates,
            "reason": "accepted" if accepted else ("waiting_streak" if current_pass else "gate_reject"),
        })
        return bool(accepted), info

    def _select_fifo_dynamic_template(self):
        """Use the highest-score entry in the FIFO as the dynamic template."""
        if not self.template_fifo:
            self.dynamic_z_tensor = None
            return None
        best = max(self.template_fifo, key=lambda x: (x["score"], x["frame_id"]))
        self.dynamic_z_tensor = best["template"]
        return self.dynamic_z_tensor

    def _maybe_push_template_fifo(self, image, offset, merge_flag, max_score):
        """Insert one current template every N frames when tracking is reliable."""
        if self.template_fifo_len <= 0 or self.template_update_interval <= 0:
            return
        if self.frame_id % self.template_update_interval != 0:
            return
        if float(max_score) <= self.template_score_thr:
            return

        # If a candidate homography is committed, crop the template from the
        # candidate-aligned current frame; otherwise keep the current historical
        # alignment. No extra tracker forward is required.
        template_image = image
        if merge_flag and image.ndim == 3 and image.shape[2] >= 6:
            template_image = get_align(
                image[:, :, :3], image[:, :, 3:6], np.squeeze(offset, axis=0)
            )
        patch, _, _ = sample_target(
            template_image, self.state, self.params.template_factor,
            output_sz=self.params.template_size
        )
        with torch.no_grad():
            template_tensor = self.preprocessor.process(patch).detach()
        self.template_fifo.append({
            "score": float(max_score),
            "frame_id": int(self.frame_id),
            "template": template_tensor,
        })
        if len(self.template_fifo) > self.template_fifo_len:
            self.template_fifo.pop(0)
        self._select_fifo_dynamic_template()

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
        self.template_fifo = []
        self.dynamic_z_tensor = None
        self._tocu_good_streak = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox_rgb'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}
    
        self.first_search = search
        self.fisrt_resize_factor_rgb = resize_factor
        self.init_bbox = info['init_bbox_rgb']
        self.H, self.W, _ = image_rgb.shape
        self.H_T, self.W_T, _ = image_tir.shape

    def track_notalign(self, image, seq_name, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        x_patch_arr = np.concatenate((x_patch_arr[:,:,:3], x_patch_arr[:, :, 3:6]), axis=2)
        search = self.preprocessor.process(x_patch_arr)
        
        templates = [self.z_tensor]
        with torch.no_grad():
            x_tensor = search
            out_dict = self.network.forward(
                template=templates, search=x_tensor, ce_template_mask=self.box_mask_z)
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
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
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

    def track_stepalign(self, image, image_unalign, seq_name, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        merge_flag = False
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        x_patch_arr = np.concatenate((x_patch_arr[:,:,:3], x_patch_arr[:, :, 3:6]), axis=2)
        search = self.preprocessor.process(x_patch_arr)
        
        self._select_fifo_dynamic_template()
        templates = [self.z_tensor] if self.dynamic_z_tensor is None else \
            [self.z_tensor, self.dynamic_z_tensor]
        with torch.no_grad():
            x_tensor = search
            out_dict = self.network.forward(
                template=templates, search=x_tensor)

        local_offset = out_dict["pred_offset"]
        # local_offset: [dx, dy, log_sx, log_sy], normalized by the current search crop.
        # 1) 反归一化到 search crop 像素坐标；
        # 2) 用 sample_target 的 crop->global 变换提升到整图坐标。
        # 模型几何约定是 RGB -> TIR；cv2.warpPerspective(tir, M, ...) 需要 TIR -> RGB，
        # 因此 get_align 使用 inverse 后的 offset。
        offset_rgb_to_tir_t, offset_tir_to_rgb_t, local_offset_px_t = local_offset_to_global_homography(
            local_offset=local_offset,
            state=self.state,
            resize_factor=resize_factor,
            search_size=self.params.search_size,
        )
        offset_rgb_to_tir = offset_rgb_to_tir_t.detach().cpu().numpy()
        local_offset_px = local_offset_px_t.detach().cpu().numpy()
        offset = offset_tir_to_rgb_t.detach().cpu().numpy()

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
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
        merge_flag, tocu_info = self._tocu_relation_update(
            out_dict, local_offset, max_score
        )
        self._maybe_push_template_fifo(
            image=image, offset=offset, merge_flag=merge_flag, max_score=max_score
        )

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
                    "best_score": max_score,
                    "offset":offset,
                    "offset_rgb_to_tir": offset_rgb_to_tir,
                    "local_offset_px": local_offset_px,
                    "merge_flag":merge_flag,
                    "tocu_info": tocu_info}


    def track_stepalign_vis(self, image, image_unalign, seq_name, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        merge_flag = False
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image_unalign, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        x_patch_arr = np.concatenate((x_patch_arr[:,:,:3], x_patch_arr[:, :, 3:6]), axis=2)
        search = self.preprocessor.process(x_patch_arr)
        
        self._select_fifo_dynamic_template()
        templates = [self.z_tensor] if self.dynamic_z_tensor is None else \
            [self.z_tensor, self.dynamic_z_tensor]
        with torch.no_grad():
            x_tensor = search
            out_dict = self.network.forward(
                template=templates, search=x_tensor, ce_template_mask=self.box_mask_z)
        chosens = {
            stage: out_dict.get(f'{stage}_branch_weight', None)
            for stage in ('center', 'scale', 'refinement')
        }
        # print(out_dict['gate_logits'])
        # ori_feats = out_dict['ori_feats']  
        # warped_feats = out_dict['warped_feats'] 
        # deformed_feats = out_dict['deformed_feats'] 
        # deformed_mid = out_dict['deformed_mid'] 
        # ref_grid = deformed_mid['ref_grid'] 
        # samp_grid = deformed_mid['samp_grid'] 
        # samp_grid_flat = deformed_mid['samp_grid_flat'] 
        # attn_w = deformed_mid['attn_w'] 
        # x_rgb_resize = ori_feats[0][0].view(1,16,16,768)
        # x_tir_resize= ori_feats[0][1].view(1,16,16,768)
        # warp_x_tir = self.warp_refiner(x_tir_resize, out_dict["offsets"][-1])
        # warp_x_tir = warp_x_tir.view(B, self.H*self.W, D)
                
        # warp_x_rgb = self.warp_refiner(x_rgb_resize, out_dict["offsets"][-1], inverse=True)
        # warp_x_rgb = warp_x_rgb.view(B, self.H*self.W, D)
        # ori_feats.append([x_rgb, x_tir])
        # warped_feats.append([warp_x_rgb, warp_x_tir])
        
        # ori_heat = FeatureMapVisible(ori_feats[0][0].view(1,16,16,768).permute(0,3,1,2))
        # warped_heat = FeatureMapVisible((ori_feats[0][0]+warped_feats[0][0]).view(1,16,16,768).permute(0,3,1,2))
        # deformed_heat = FeatureMapVisible((ori_feats[0][0] + deformed_feats[0][0]).view(1,16,16,768).permute(0,3,1,2))
        
        # warped_heat = torch.tensor(warped_heat)[:,:,[2,1,0]]    # BGR->RGB
        # deformed_heat = torch.tensor(deformed_heat)[:,:,[2,1,0]]
        # ori_heat = torch.tensor(ori_heat)[:,:,[2,1,0]]    # BGR->RGB

        # margin = torch.ones((256,20,3))*255       # 图像间距
        # imgs_feat = torch.cat([ori_heat, margin, warped_heat, margin, deformed_heat],1)
        # vis_path = './vis_feat'
        # # 画搜索区及对应特征图
        # plt.imshow(imgs_feat/255)
        # plt.axis(False)
        # plt.savefig(os.path.join(vis_path,'{}_feat.jpg'.format(seq_name)),bbox_inches='tight', dpi=200, pad_inches=0.0)
        # plt.close()
        
        # save_visualization(ref_grid[-1][0], samp_grid[-1][0], samp_grid_flat[-1][0], attn_w[-1][0], 256, 256, './vis_deform', 'visible_')
        # save_visualization(ref_grid[-1][1], samp_grid[-1][1], samp_grid_flat[-1][1], attn_w[-1][1], 256, 256, './vis_deform', 'infrared_')
        # breakpoint()
        local_offset_norm = out_dict["pred_offset"]
        local_offset_norm_center = out_dict.get("center_geometry", local_offset_norm)
        local_offset_norm_scale = out_dict.get("scale_geometry", local_offset_norm)

        _, offset_tir_to_rgb_t, _ = local_offset_to_global_homography(
            local_offset_norm, self.state, resize_factor, self.params.search_size
        )
        _, offset_center_tir_to_rgb_t, _ = local_offset_to_global_homography(
            local_offset_norm_center, self.state, resize_factor, self.params.search_size
        )
        _, offset_scale_tir_to_rgb_t, _ = local_offset_to_global_homography(
            local_offset_norm_scale, self.state, resize_factor, self.params.search_size
        )
        offset = offset_tir_to_rgb_t.detach().cpu().numpy()
        offset_center = offset_center_tir_to_rgb_t.detach().cpu().numpy()
        offset_scale = offset_scale_tir_to_rgb_t.detach().cpu().numpy()
        
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
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
        merge_flag, tocu_info = self._tocu_relation_update(
            out_dict, local_offset_norm, max_score
        )
        self._maybe_push_template_fifo(
            image=image_unalign, offset=offset, merge_flag=merge_flag, max_score=max_score
        )

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score,
                    "offset":offset,
                    "offset_center":offset_center,
                    "offset_scale":offset_scale,
                    "merge_flag":merge_flag,
                    "tocu_info": tocu_info,
                    "chosens":chosens}

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        self.template_fifo = []
        self.dynamic_z_tensor = None
        self._tocu_good_streak = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

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

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
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
    return OSTrack
