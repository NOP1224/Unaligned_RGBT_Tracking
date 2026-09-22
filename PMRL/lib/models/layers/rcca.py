"""
RCCA + dense optimal-transport transition + staged geometry readout.

Agent construction is in agent_builder.py. TCMDA fusion is in tcmda.py.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent_builder import AgentConstructor, detail_regularization
from .agent_router import AdaptiveCueMixer


def _cfg_value(config: Optional[dict], key: str, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


@dataclass(frozen=True)
class AlignmentStageSpec:
    name: str
    layer: int
    stage_id: int
    experts: Tuple[str, ...]
    down_factor: int
    sigma: float


@dataclass
class AlignmentState:
    geometry: Optional[torch.Tensor] = None  # (B,4): dx,dy,log_sx,log_sy
    confidence: Optional[torch.Tensor] = None
    uncertainty: Optional[torch.Tensor] = None


def _grid_coords(h: int, w: int, device: torch.device, dtype: torch.dtype, batch: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(0.5 / h, 1.0 - 0.5 / h, h, device=device, dtype=dtype),
        torch.linspace(0.5 / w, 1.0 - 0.5 / w, w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=-1).reshape(1, h * w, 2).expand(batch, -1, -1).contiguous()


def apply_diag_geometry(coords: torch.Tensor, geometry: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply normalized diagonal affine geometry to coordinates in [0,1]."""
    if geometry is None:
        return coords
    center = coords.new_tensor([0.5, 0.5]).view(1, 1, 2)
    shift = geometry[:, None, 0:2]
    scale = geometry[:, None, 2:4].exp().clamp(0.25, 4.0)
    return (coords - center) * scale + center + shift


def compose_diag_geometry(prev: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Compose two centered diagonal affine transforms.

    Each transform is represented as:
        y = s * (x - 0.5) + 0.5 + t

    Applying prev first and residual second gives:
        s = s_res * s_prev
        t = s_res * t_prev + t_res
    """
    prev_shift, prev_log_s = prev[:, 0:2], prev[:, 2:4]
    res_shift, res_log_s = residual[:, 0:2], residual[:, 2:4]
    res_scale = res_log_s.exp().clamp(0.25, 4.0)
    out_shift = res_scale * prev_shift + res_shift
    out_log_s = (prev_log_s + res_log_s).clamp(min=-2.0, max=2.0)
    return torch.cat([out_shift, out_log_s], dim=-1)


def bound_diag_residual(
    residual: torch.Tensor,
    max_shift: float = 0.02,
    max_log_scale: float = 0.03,
) -> torch.Tensor:
    """Smoothly limit a residual diagonal affine transform.

    Refinement is intended to correct the Scale estimate rather than replace it.
    A smooth tanh bound preserves gradients around zero while preventing the last
    stage from producing another unrestricted absolute fit.
    """
    shift_limit = residual.new_tensor(float(max_shift)).clamp_min(1e-6)
    scale_limit = residual.new_tensor(float(max_log_scale)).clamp_min(1e-6)
    shift = shift_limit * torch.tanh(residual[:, 0:2] / shift_limit)
    log_scale = scale_limit * torch.tanh(residual[:, 2:4] / scale_limit)
    return torch.cat([shift, log_scale], dim=-1)


def geometry_to_dirmag(geom: torch.Tensor, scale_dir_tau: float = 0.03) -> torch.Tensor:
    """Encode [dx, dy, log_sx, log_sy] as direction/magnitude parts.

    Layout: [xy_dir_x, xy_dir_y, xy_mag, scale_dir_x, scale_dir_y,
             scale_mag_x, scale_mag_y].  Scale direction is a smooth
    expansion/shrink indicator rather than a hard sign so gradients remain
    usable when this representation is used by losses.
    """
    xy = geom[:, 0:2]
    xy_mag = xy.norm(dim=-1, keepdim=True)
    xy_dir = xy / xy_mag.clamp_min(1e-6)
    log_s = geom[:, 2:4]
    scale_dir = torch.tanh(log_s / max(float(scale_dir_tau), 1e-6))
    scale_mag = log_s.abs()
    return torch.cat([xy_dir, xy_mag, scale_dir, scale_mag], dim=-1)


def downsample_search_tokens(
    tokens: torch.Tensor,
    index: torch.Tensor,
    feat_sz: int,
    factor: int,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Scatter sparse search tokens to the original grid, then masked-average downsample."""
    b, l, c = tokens.shape
    h = w = feat_sz
    feat = tokens.new_zeros(b, h * w, c)
    mask = tokens.new_zeros(b, h * w, 1)
    feat.scatter_(1, index.long().unsqueeze(-1).expand(-1, -1, c), tokens)
    mask.scatter_(1, index.long().unsqueeze(-1), 1.0)

    feat = feat.transpose(1, 2).reshape(b, c, h, w)
    mask = mask.transpose(1, 2).reshape(b, 1, h, w)

    out_h = max(h // factor, 1)
    out_w = max(w // factor, 1)
    feat_sum = F.adaptive_avg_pool2d(feat * mask, (out_h, out_w))
    mask_sum = F.adaptive_avg_pool2d(mask, (out_h, out_w)).clamp_min(1e-6)
    feat_down = feat_sum / mask_sum
    tokens_down = feat_down.flatten(2).transpose(1, 2).contiguous()
    coords_down = _grid_coords(out_h, out_w, tokens.device, tokens.dtype, b)
    return tokens_down, coords_down, (out_h, out_w)


class OptimalTransportTransition(nn.Module):
    """Progressive-boundary unbalanced OT whose output is the token relation ``R``.

    The OT problem separates three quantities with non-overlapping semantics:
      * ``boundary``: stage-wise geometric feasibility (where a relation may exist);
      * ``semantic_cost``: cross-modal clue/appearance evidence (which feasible edge is preferred);
      * ``source/target_capacity``: token-wise reliable transport mass (how much mass participates).

    The progressive boundary is induced by the geometric degrees of freedom that
    remain unresolved at each stage rather than by shrinking a Gaussian around a
    single transform estimate:
      Center     -> translation-feasible family;
      Scale      -> center-conditioned scale-feasible family;
      Refinement -> bounded residual family around the Scale estimate.
    """

    STAGE_EPS_SCALE = {
        "center": 1.35,
        "scale": 1.00,
        "refinement": 0.75,
    }
    STAGE_ROW_TAU_SCALE = {
        "center": 0.80,
        "scale": 1.00,
        "refinement": 1.20,
    }
    STAGE_COL_TAU_SCALE = {
        "center": 0.45,
        "scale": 0.55,
        "refinement": 0.70,
    }

    # These ranges follow the geometry parameterization used by the readout.
    SCALE_MIN = 0.25
    SCALE_MAX = 4.0
    REFINE_MAX_SHIFT = 0.02
    REFINE_MAX_LOG_SCALE = 0.03

    def __init__(self, dim: int, align_dim: int = 128, epsilon: float = 0.08, iters: int = 18):
        super().__init__()
        self.iters = int(iters)
        self.rgb_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, align_dim))
        self.tir_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, align_dim))

        # Semantic evidence mixing. Softmax keeps the two evidences on a relative
        # scale and avoids the unconstrained simultaneous growth of all cost terms.
        self.clue_logit = nn.Parameter(torch.tensor(0.0))
        self.appearance_logit = nn.Parameter(torch.tensor(0.0))
        self.log_clue_temperature = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.log_appearance_temperature = nn.Parameter(torch.log(torch.tensor(0.50)))

        # Keep the historical parameter name for checkpoint compatibility, but
        # reinterpret it as the strength of the geometric reference measure.
        self.geometry_logit = nn.Parameter(torch.tensor(0.0))

        self.log_epsilon = nn.Parameter(torch.log(torch.tensor(float(epsilon))))
        self.log_row_tau = nn.Parameter(torch.log(torch.tensor(0.80)))
        self.log_col_tau = nn.Parameter(torch.log(torch.tensor(0.30)))

        # Boundary softness grows with previous-stage uncertainty. Detach the
        # uncertainty before use so the model cannot enlarge the feasible set by
        # deliberately increasing its own uncertainty.
        self.boundary_uncertainty_gain = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _normalize_vector(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp_min(1e-8)
        return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _distance_to_interval(value: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """Element-wise Euclidean residual to a closed interval [lower, upper]."""
        return F.relu(lower - value) + F.relu(value - upper)

    def _appearance_cost(self, rgb_tokens: torch.Tensor, tir_tokens: torch.Tensor) -> torch.Tensor:
        rgb = F.normalize(self.rgb_proj(rgb_tokens), dim=-1)
        tir = F.normalize(self.tir_proj(tir_tokens), dim=-1)
        # Cosine distance is non-negative and bounded in [0, 2].
        return (1.0 - torch.matmul(rgb, tir.transpose(-2, -1))).clamp_min(0.0)

    def _semantic_cost(self, clue_cost: torch.Tensor, appearance_cost: torch.Tensor) -> torch.Tensor:
        """Cue--appearance evidence only; geometry is intentionally excluded."""
        clue_tau = self.log_clue_temperature.exp().clamp(0.25, 4.0)
        app_tau = self.log_appearance_temperature.exp().clamp(0.05, 2.0)
        weights = F.softmax(torch.stack([self.clue_logit, self.appearance_logit]), dim=0)
        clue = clue_cost.clamp_min(0.0) / clue_tau
        appearance = appearance_cost / app_tau
        return weights[0] * clue + weights[1] * appearance

    def _progressive_boundary(
        self,
        stage_name: str,
        coords: torch.Tensor,
        prev_geometry: torch.Tensor,
        prev_uncertainty: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the Progressive Feasible-Set Relation Boundary.

        ``boundary[i,j]`` measures only geometric feasibility. It never consumes
        appearance/clue evidence, preventing the content -> geometry -> content
        self-confirming loop of the previous implementation.
        """
        b, l, _ = coords.shape
        side = max(int(round(l ** 0.5)), 1)
        token_size = 1.0 / float(side)
        src = coords.unsqueeze(2)  # B,L,1,2
        tgt = coords.unsqueeze(1)  # B,1,L,2
        center = coords.new_tensor([0.5, 0.5]).view(1, 1, 1, 2)

        unc = prev_uncertainty.detach().clamp(0.0, 1.0).view(b, 1, 1, 1)
        unc_gain = F.softplus(self.boundary_uncertainty_gain).clamp(0.0, 3.0)
        # Uncertainty changes only the softness outside the admissible set; it
        # does not change the physical parameter family itself.
        edge_sigma = token_size * (0.50 + 0.50 * unc_gain * unc).clamp_min(0.25)

        if stage_name == "center":
            # Center stage should discover the translation instead of receiving
            # one from the same semantic cost. Permit a broad translation family.
            # Initial uncertainty=1 gives almost global support with a soft falloff.
            max_shift = coords.new_tensor(0.50)
            displacement = tgt - src
            residual = F.relu(displacement.abs() - max_shift)
            reference = coords.new_zeros((b, 2))

        elif stage_name == "scale":
            # Translation is known from Center, while scale is deliberately left
            # unresolved. A target token is feasible if some scale in
            # [SCALE_MIN, SCALE_MAX] can explain it (plus one-token center slack).
            x = src - center
            shift = prev_geometry[:, None, None, 0:2]
            y = tgt - center - shift
            s_min = coords.new_tensor(self.SCALE_MIN)
            s_max = coords.new_tensor(self.SCALE_MAX)
            cand_min = torch.minimum(s_min * x, s_max * x)
            cand_max = torch.maximum(s_min * x, s_max * x)
            # Center was estimated at the preceding stage; allow one token of
            # translation slack so semantic/tracking gradients can correct it.
            shift_slack = coords.new_tensor(token_size)
            lower = cand_min - shift_slack
            upper = cand_max + shift_slack
            residual = self._distance_to_interval(y, lower, upper)
            reference = prev_geometry[:, 0:2].detach()

        elif stage_name == "refinement":
            # Scale has been estimated. Only the same bounded residual family used
            # by the Refinement decoder is geometrically admissible here.
            mapped = apply_diag_geometry(coords, prev_geometry).unsqueeze(2)
            q = mapped - center
            s_lo = coords.new_tensor(math.exp(-self.REFINE_MAX_LOG_SCALE))
            s_hi = coords.new_tensor(math.exp(self.REFINE_MAX_LOG_SCALE))
            cand_min = center + torch.minimum(s_lo * q, s_hi * q)
            cand_max = center + torch.maximum(s_lo * q, s_hi * q)
            lower = cand_min - self.REFINE_MAX_SHIFT
            upper = cand_max + self.REFINE_MAX_SHIFT
            residual = self._distance_to_interval(tgt, lower, upper)
            reference = prev_geometry[:, 0:2].detach()

        else:
            raise ValueError(f"Unknown stage name: {stage_name}")

        distance2 = (residual / edge_sigma).pow(2).sum(dim=-1)
        boundary = torch.exp(-0.5 * distance2).clamp_min(1e-8)
        return boundary, reference

    def _build_capacities(
        self,
        src_mass: torch.Tensor,
        tgt_mass: torch.Tensor,
        src_transport_mass: torch.Tensor,
        tgt_transport_mass: torch.Tensor,
        src_match: torch.Tensor,
        tgt_match: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build token-wise capacities without reusing pairwise semantic/geometry cost.

        Agent coverage chooses which tokens should participate; the learned
        ``transport_mass`` head gives each token's available OT mass. Matchability
        is used only as confidence, not multiplied into the marginal a second time.
        """
        l_src = max(src_mass.shape[-1], 1)
        l_tgt = max(tgt_mass.shape[-1], 1)
        src_tm = src_transport_mass.clamp(0.02, 1.0)
        tgt_tm = tgt_transport_mass.clamp(0.02, 1.0)

        source_capacity = self._normalize_vector(src_mass * src_tm + 0.05 / l_src)
        target_capacity = self._normalize_vector(tgt_mass * tgt_tm + 0.05 / l_tgt)

        # Confidence may use both notions, while the OT capacity itself has only
        # one token-mass variable. This keeps capacity and pairwise evidence clean.
        src_quality = torch.sqrt(src_tm * src_match.clamp(0.02, 1.0))
        tgt_quality = torch.sqrt(tgt_tm * tgt_match.clamp(0.02, 1.0))
        return source_capacity, target_capacity, src_quality, tgt_quality

    @staticmethod
    def _unbalanced_sinkhorn(
        log_kernel: torch.Tensor,
        log_a: torch.Tensor,
        log_b: torch.Tensor,
        rho_row: torch.Tensor,
        rho_col: torch.Tensor,
        iters: int,
    ) -> torch.Tensor:
        u = torch.zeros_like(log_a)
        v = torch.zeros_like(log_b)
        for _ in range(iters):
            u = rho_row * (log_a - torch.logsumexp(log_kernel + v.unsqueeze(1), dim=2))
            v = rho_col * (log_b - torch.logsumexp(log_kernel + u.unsqueeze(2), dim=1))
        log_relation = (log_kernel + u.unsqueeze(2) + v.unsqueeze(1)).clamp(-30.0, 12.0)
        return torch.exp(log_relation)

    def forward(
        self,
        stage_name: str,
        rgb_tokens: torch.Tensor,
        tir_tokens: torch.Tensor,
        coords: torch.Tensor,
        prev_geometry: torch.Tensor,
        prev_uncertainty: torch.Tensor,
        src_mass: torch.Tensor,
        tgt_mass: torch.Tensor,
        src_transport_mass: torch.Tensor,
        tgt_transport_mass: torch.Tensor,
        src_match: torch.Tensor,
        tgt_match: torch.Tensor,
        bandwidth_multiplier: float,
        clue_cost: torch.Tensor,
        appearance_cost: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        del bandwidth_multiplier  # kept in the public interface for checkpoint/config compatibility

        if appearance_cost is None:
            appearance_cost = self._appearance_cost(rgb_tokens, tir_tokens)
        semantic_cost = self._semantic_cost(clue_cost, appearance_cost)
        boundary, boundary_reference = self._progressive_boundary(
            stage_name, coords, prev_geometry, prev_uncertainty
        )

        source_capacity, target_capacity, src_quality, tgt_quality = self._build_capacities(
            src_mass,
            tgt_mass,
            src_transport_mass,
            tgt_transport_mass,
            src_match,
            tgt_match,
        )

        eps = self.log_epsilon.exp().clamp(0.03, 0.25) * self.STAGE_EPS_SCALE.get(stage_name, 1.0)
        row_tau = self.log_row_tau.exp().clamp(0.10, 3.0) * self.STAGE_ROW_TAU_SCALE.get(stage_name, 1.0)
        col_tau = self.log_col_tau.exp().clamp(0.05, 2.0) * self.STAGE_COL_TAU_SCALE.get(stage_name, 1.0)
        rho_row = row_tau / (row_tau + eps)
        rho_col = col_tau / (col_tau + eps)

        # Generalized-KL reference measure:
        #   K = B^beta * exp(-C_sem / eps)
        # Geometry therefore defines the feasible prior and is never normalized
        # into the same numerical space as semantic evidence.
        boundary_strength = F.softplus(self.geometry_logit).clamp(0.10, 4.0)
        log_kernel = (
            boundary_strength * boundary.clamp_min(1e-8).log()
            - semantic_cost / eps
        )
        relation = self._unbalanced_sinkhorn(
            log_kernel,
            source_capacity.clamp_min(1e-8).log(),
            target_capacity.clamp_min(1e-8).log(),
            rho_row,
            rho_col,
            self.iters,
        )
        row_mass = relation.sum(dim=-1)
        col_mass = relation.sum(dim=-2)
        conditional = relation / row_mass.unsqueeze(-1).clamp_min(1e-8)
        reverse_conditional = relation.transpose(-2, -1) / col_mass.unsqueeze(-1).clamp_min(1e-8)

        src_mass_ratio = (row_mass / source_capacity.clamp_min(1e-8)).clamp(0.0, 1.0)
        tgt_mass_ratio = (col_mass / target_capacity.clamp_min(1e-8)).clamp(0.0, 1.0)
        entropy = -(conditional.clamp_min(1e-8) * conditional.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / max(float(torch.log(torch.tensor(float(max(conditional.shape[-1], 2))))), 1e-6)
        reverse_entropy = -(
            reverse_conditional.clamp_min(1e-8) * reverse_conditional.clamp_min(1e-8).log()
        ).sum(dim=-1)
        reverse_entropy = reverse_entropy / max(
            float(torch.log(torch.tensor(float(max(reverse_conditional.shape[-1], 2))))), 1e-6
        )
        src_conf = (
            src_quality * src_mass_ratio.sqrt() * (0.25 + 0.75 * (1.0 - entropy).clamp(0.0, 1.0))
        ).clamp(0.0, 1.0)
        tgt_conf = (
            tgt_quality * tgt_mass_ratio.sqrt() * (0.25 + 0.75 * (1.0 - reverse_entropy).clamp(0.0, 1.0))
        ).clamp(0.0, 1.0)

        # Keep ``geometry_cost`` and ``coarse_shift`` as compatibility/diagnostic
        # aliases. Geometry is no longer an additive term in ``cost_matrix``.
        geometry_cost = -boundary.clamp_min(1e-8).log()
        aux = {
            "appearance_cost": appearance_cost,
            "semantic_cost": semantic_cost,
            "geometry_cost": geometry_cost,
            "relation_boundary": boundary,
            "boundary_reference": boundary_reference,
            "boundary_strength": boundary_strength.expand(rgb_tokens.shape[0]),
            "clue_cost": clue_cost,
            "row_mass": row_mass,
            "col_mass": col_mass,
            "src_conf": src_conf,
            "tgt_conf": tgt_conf,
            "source_capacity": source_capacity,
            "target_capacity": target_capacity,
            "src_quality": src_quality,
            "tgt_quality": tgt_quality,
            "src_transport_mass": src_transport_mass,
            "tgt_transport_mass": tgt_transport_mass,
            # Explicit fusion quality is the retained UOT mass ratio. It remains
            # separate from entropy-based alignment confidence.
            "src_fusion_quality": src_mass_ratio,
            "tgt_fusion_quality": tgt_mass_ratio,
            "relation_v2t": conditional,
            "relation_t2v": reverse_conditional,
            "reverse_transition": reverse_conditional,
            "relation_entropy": entropy,
            "coarse_shift": boundary_reference,
            "epsilon": eps.expand(rgb_tokens.shape[0]),
            "rho_row": rho_row.expand(rgb_tokens.shape[0]),
            "rho_col": rho_col.expand(rgb_tokens.shape[0]),
        }
        return conditional, relation, semantic_cost, row_mass, aux


class TransitionGeometryReadout(nn.Module):
    """Decode global diagonal affine geometry directly from relation mass."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def absolute_to_residual(prev: torch.Tensor, target_abs: torch.Tensor) -> torch.Tensor:
        prev_shift, prev_log_s = prev[:, 0:2], prev[:, 2:4]
        tgt_shift, tgt_log_s = target_abs[:, 0:2], target_abs[:, 2:4]
        res_log_s = (tgt_log_s - prev_log_s).clamp(min=-2.0, max=2.0)
        res_scale = res_log_s.exp().clamp(0.25, 4.0)
        res_shift = tgt_shift - res_scale * prev_shift
        return torch.cat([res_shift, res_log_s], dim=-1)

    @staticmethod
    def _weighted_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (w.unsqueeze(-1) * x).sum(dim=1)

    def _fit_diagonal_affine(self, coords: torch.Tensor, exp_tgt: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        x = coords - 0.5
        y = exp_tgt - 0.5
        mean_x = self._weighted_mean(x, weights)
        mean_y = self._weighted_mean(y, weights)
        xc = x - mean_x[:, None, :]
        yc = y - mean_y[:, None, :]
        var_x = (weights.unsqueeze(-1) * xc.pow(2)).sum(dim=1).clamp_min(1e-6)
        cov_xy = (weights.unsqueeze(-1) * xc * yc).sum(dim=1)
        scale = (cov_xy / var_x).clamp(0.25, 4.0)
        shift = mean_y - scale * mean_x
        return torch.cat([shift, scale.log()], dim=-1)

    def forward(
        self,
        stage_name: str,
        relation: torch.Tensor,
        coords: torch.Tensor,
        source_confidence: torch.Tensor,
        prev_geometry: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row_mass = relation.sum(dim=-1)
        transition = relation / row_mass.unsqueeze(-1).clamp_min(1e-8)
        exp_tgt = torch.matmul(transition, coords)
        raw_weight = row_mass * source_confidence.clamp(0.0, 1.0)
        weights = raw_weight / raw_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        if stage_name == "center":
            shift = self._weighted_mean(exp_tgt - coords, weights)
            geom_update = torch.cat([shift, torch.zeros_like(shift)], dim=-1)
            geometry_abs = geom_update
        elif stage_name == "scale":
            # Scale is the last stage allowed to recover an absolute diagonal
            # affine transform directly from the relation statistics.
            geometry_abs = self._fit_diagonal_affine(coords, exp_tgt, weights)
            geom_update = self.absolute_to_residual(prev_geometry, geometry_abs)
        elif stage_name == "refinement":
            # True residual refinement: first pre-align source coordinates with
            # the Scale geometry, then fit only the remaining local transform.
            prealigned = apply_diag_geometry(coords, prev_geometry)
            geom_update = self._fit_diagonal_affine(prealigned, exp_tgt, weights)
            geom_update = bound_diag_residual(geom_update, max_shift=0.02, max_log_scale=0.03)
            geometry_abs = compose_diag_geometry(prev_geometry, geom_update)
        else:
            raise ValueError(f"Unknown stage name: {stage_name}")

        entropy = -(transition.clamp_min(1e-8) * transition.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / torch.log(transition.new_tensor(float(max(transition.shape[-1], 2))))
        confidence = (weights * source_confidence.clamp(0.0, 1.0)).sum(dim=-1).clamp(0.0, 1.0)
        uncertainty = (weights * entropy.clamp(0.0, 1.0)).sum(dim=-1).clamp(0.0, 1.0)
        return geom_update, geometry_abs, confidence, uncertainty


class RelationOffsetResidualHead(nn.Module):
    """Small relation-only residual decoder.

    Geometry statistics provide the main solution. This branch sees only fields
    derived from ``R`` and predicts a bounded direction/magnitude residual for
    token-grid quantization, asymmetric truncation and local multi-modal noise.
    """

    def __init__(self, stage_name: str):
        super().__init__()
        self.stage_name = stage_name
        hidden = 48
        self.spatial = nn.Sequential(
            nn.Conv2d(7, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 1),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(nn.LayerNorm(hidden + 6), nn.Linear(hidden + 6, hidden), nn.GELU())
        self.xy_dir = nn.Linear(hidden, 2)
        self.xy_mag = nn.Linear(hidden, 1)
        self.scale_dir = nn.Linear(hidden, 2)
        self.scale_mag = nn.Linear(hidden, 2)

        if stage_name == "center":
            residual_max = [0.04, 0.04, 0.0, 0.0]
        elif stage_name == "scale":
            residual_max = [0.03, 0.03, 0.08, 0.08]
        else:
            # The OT readout already predicts a bounded residual around Scale.
            # The learned relation residual only corrects remaining grid/local error.
            residual_max = [0.010, 0.010, 0.015, 0.015]
        self.register_buffer("residual_max", torch.tensor(residual_max))

        for module in (self.xy_dir, self.scale_dir):
            nn.init.normal_(module.weight, std=1e-4)
            nn.init.zeros_(module.bias)
        for module in (self.xy_mag, self.scale_mag):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, -6.0)

    def forward(
        self,
        relation: torch.Tensor,
        coords: torch.Tensor,
        geometry_abs: torch.Tensor,
        source_confidence: torch.Tensor,
        return_parts: bool = False,
    ):
        b, l, _ = relation.shape
        side = int(round(l ** 0.5))
        if side * side != l:
            raise ValueError(f"Relation residual decoder expects a square token grid, got L={l}")

        row_mass = relation.sum(dim=-1)
        transition = relation / row_mass.unsqueeze(-1).clamp_min(1e-8)
        matched_coord = torch.matmul(transition, coords)
        predicted_coord = apply_diag_geometry(coords, geometry_abs)
        delta = matched_coord - predicted_coord
        diff = coords.unsqueeze(1) - matched_coord.unsqueeze(2)
        variance = (transition.unsqueeze(-1) * diff.pow(2)).sum(dim=2)
        entropy = -(transition.clamp_min(1e-8) * transition.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / torch.log(transition.new_tensor(float(max(l, 2))))
        mass_norm = row_mass / row_mass.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        fields = torch.cat([
            delta,
            variance,
            source_confidence.unsqueeze(-1),
            entropy.unsqueeze(-1),
            mass_norm.clamp(0.0, 2.0).unsqueeze(-1),
        ], dim=-1)
        feature = self.spatial(fields.transpose(1, 2).reshape(b, 7, side, side)).mean(dim=(2, 3))
        global_stats = torch.cat([
            geometry_abs,
            source_confidence.mean(dim=-1, keepdim=True),
            entropy.mean(dim=-1, keepdim=True),
        ], dim=-1)
        h = self.trunk(torch.cat([feature, global_stats], dim=-1))

        residual_max = self.residual_max.to(h.device, h.dtype)
        xy_dir = F.normalize(self.xy_dir(h), dim=-1, eps=1e-6)
        xy_mag = torch.sigmoid(self.xy_mag(h)) * residual_max[0:2].norm().clamp_min(1e-8)
        residual_xy = xy_dir * xy_mag
        scale_dir = torch.tanh(self.scale_dir(h))
        scale_mag = torch.sigmoid(self.scale_mag(h)) * residual_max[2:4].abs().view(1, 2)
        residual_scale = scale_dir * scale_mag
        if self.stage_name == "center":
            residual_scale = torch.zeros_like(residual_scale)
        residual = torch.cat([residual_xy, residual_scale], dim=-1)

        if return_parts:
            return residual, {
                "xy_dir": xy_dir,
                "xy_mag": xy_mag.squeeze(-1),
                "scale_dir": scale_dir,
                "scale_mag": scale_mag,
            }
        return residual


class RCCAAgentInteraction(nn.Module):
    """Agent-to-token attention used to construct the three cue costs."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim
        self.scale = self.head_dim ** -0.5
        self.norm_token = nn.LayerNorm(dim)
        self.norm_agent = nn.LayerNorm(dim)
        self.k_token = nn.Linear(dim, dim)
        self.q_agent = nn.Linear(dim, dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def project_tokens(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize tokens once and cache their multi-head K projection."""
        token_n = self.norm_token(tokens)
        token_key = self._split_heads(self.k_token(token_n))
        return token_n, token_key

    def project_agent_query(self, agents: torch.Tensor) -> torch.Tensor:
        """Project an Agent set to the shared multi-head Q space."""
        return self._split_heads(self.q_agent(self.norm_agent(agents)))

    def agent_to_token_attn_from_key(
        self,
        token_key: torch.Tensor,
        agents: Optional[torch.Tensor] = None,
        agent_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Agent->Token attention from a cached token K projection."""
        if agent_query is None:
            if agents is None:
                raise ValueError("Either agents or agent_query must be provided")
            agent_query = self.project_agent_query(agents)
        attn = F.softmax(
            (agent_query * self.scale) @ token_key.transpose(-2, -1), dim=-1
        )
        return attn


class RCCAStage(nn.Module):
    """One RCCA stage: Agent evidence selection -> one UOT relation -> readout."""

    def __init__(
        self,
        dim: int,
        feat_sz: int,
        spec: AlignmentStageSpec,
        joint_config: Optional[dict] = None,
    ):
        super().__init__()
        self.dim = dim
        self.feat_sz = feat_sz
        self.spec = spec
        self.joint_config = dict(joint_config or {})
        self.cue_mixer_temperature = max(
            float(_cfg_value(self.joint_config, "CUE_MIXER_TEMPERATURE", 0.20)), 1e-4
        )
        self.combination_router = AdaptiveCueMixer(
            dim=dim,
            temperature=self.cue_mixer_temperature,
        )
        self.agent_constructor = AgentConstructor(dim=dim)
        self.search_pos_mlp = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.agent_interaction = RCCAAgentInteraction(dim=dim, num_heads=8)

        # Six directional branches form the target, structure, and detail cues.
        self.branch_names = ("R_same", "R_fused", "S_rgb", "S_tir", "D_rgb", "D_tir")
        self.family_names = ("R", "S", "D")
        self.family_branches = {
            "R": ("R_same", "R_fused"),
            "S": ("S_rgb", "S_tir"),
            "D": ("D_rgb", "D_tir"),
        }
        self.ot = OptimalTransportTransition(dim=dim)
        self.readout = TransitionGeometryReadout()
        self.relation_residual_head = RelationOffsetResidualHead(stage_name=spec.name)

    def _concat_agents(
        self, agent_set, expert_names: Optional[Tuple[str, ...]] = None
    ) -> Tuple[torch.Tensor, Dict[str, slice]]:
        chunks = []
        slices: Dict[str, slice] = {}
        start = 0
        names = self.spec.experts if expert_names is None else expert_names
        for name in names:
            agents = agent_set.agents[name]
            end = start + agents.shape[1]
            chunks.append(agents)
            slices[name] = slice(start, end)
            start = end
        return torch.cat(chunks, dim=1), slices

    @staticmethod
    def _normalize_vector(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp_min(1e-8)
        return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _flatten_agent_attn(attn: torch.Tensor) -> torch.Tensor:
        b, h, k, l = attn.shape
        return attn.reshape(b, h * k, l)

    def _branch_marginals(
        self,
        a_src: torch.Tensor,
        a_tgt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Agent coverage only; token reliability is applied once in OT capacity."""
        a_src = self._flatten_agent_attn(a_src)
        a_tgt = self._flatten_agent_attn(a_tgt)
        m_src = self._normalize_vector(a_src.sum(dim=1))
        m_tgt = self._normalize_vector(a_tgt.sum(dim=1))
        if self.spec.name == "center":
            floor_ratio = 0.30
        elif self.spec.name == "scale":
            floor_ratio = 0.20
        else:
            floor_ratio = 0.05
        src_uniform = torch.full_like(m_src, 1.0 / max(m_src.shape[-1], 1))
        tgt_uniform = torch.full_like(m_tgt, 1.0 / max(m_tgt.shape[-1], 1))
        src_mass = (1.0 - floor_ratio) * m_src + floor_ratio * src_uniform
        tgt_mass = (1.0 - floor_ratio) * m_tgt + floor_ratio * tgt_uniform
        return self._normalize_vector(src_mass), self._normalize_vector(tgt_mass)

    def _branch_clue_cost(self, a_src: torch.Tensor, a_tgt: torch.Tensor) -> torch.Tensor:
        """Bhattacharyya distance between token distributions over clue slots."""
        src = self._flatten_agent_attn(a_src).clamp_min(1e-8)
        tgt = self._flatten_agent_attn(a_tgt).clamp_min(1e-8)
        src = (src / src.sum(dim=1, keepdim=True).clamp_min(1e-8)).transpose(1, 2)
        tgt = (tgt / tgt.sum(dim=1, keepdim=True).clamp_min(1e-8)).transpose(1, 2)
        coefficient = torch.matmul(torch.sqrt(src), torch.sqrt(tgt).transpose(-2, -1)).clamp_min(1e-8)
        return -torch.log(coefficient)

    def _agent_appearance_cost(
        self,
        rgb_agent_attn: torch.Tensor,
        tir_agent_attn: torch.Tensor,
    ) -> torch.Tensor:
        """Derive C_app directly from the shared proxy-attention coordinate system.

        Each token is represented by its normalized response over the same
        head-agent slots. Cross-modal token similarity is then the
        Bhattacharyya coefficient between those proxy-response signatures.
        No enhanced-token projection or additional token-to-token attention is
        evaluated in this path.
        """
        rgb_signature = self._flatten_agent_attn(rgb_agent_attn).transpose(1, 2).clamp_min(0.0)
        tir_signature = self._flatten_agent_attn(tir_agent_attn).transpose(1, 2).clamp_min(0.0)
        rgb_signature = rgb_signature / rgb_signature.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        tir_signature = tir_signature / tir_signature.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        coefficient = torch.matmul(
            torch.sqrt(rgb_signature.clamp_min(1e-8)),
            torch.sqrt(tir_signature.clamp_min(1e-8)).transpose(-2, -1),
        ).clamp(1e-8, 1.0)
        return -torch.log(coefficient)

    def _family_evidence(
        self,
        branch_attn: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """Reduce six directional branches to R/S/D family evidence."""
        branch_cost: Dict[str, torch.Tensor] = {}
        branch_src: Dict[str, torch.Tensor] = {}
        branch_tgt: Dict[str, torch.Tensor] = {}
        aux: Dict[str, torch.Tensor] = {}
        for branch_name, (a_src, a_tgt) in branch_attn.items():
            src_mass, tgt_mass = self._branch_marginals(a_src, a_tgt)
            cost = self._branch_clue_cost(a_src, a_tgt)
            branch_cost[branch_name] = cost
            branch_src[branch_name] = src_mass
            branch_tgt[branch_name] = tgt_mass
            aux[f"Branch/{branch_name}_clue_cost"] = cost.detach().mean(dim=(1, 2))
            aux[f"Branch/{branch_name}_src_support"] = src_mass.detach().mean(dim=-1)
            aux[f"Branch/{branch_name}_tgt_support"] = tgt_mass.detach().mean(dim=-1)

        family_costs, family_src, family_tgt = [], [], []
        if not branch_cost:
            raise RuntimeError("At least one cue family must be active")
        proto_cost = next(iter(branch_cost.values()))
        proto_src = next(iter(branch_src.values()))
        proto_tgt = next(iter(branch_tgt.values()))
        for family in self.family_names:
            names = tuple(name for name in self.family_branches[family] if name in branch_cost)
            if names:
                cost = torch.stack([branch_cost[name] for name in names], dim=1).mean(dim=1)
                src = self._normalize_vector(torch.stack([branch_src[name] for name in names], dim=1).mean(dim=1))
                tgt = self._normalize_vector(torch.stack([branch_tgt[name] for name in names], dim=1).mean(dim=1))
            else:
                cost = torch.zeros_like(proto_cost)
                src = torch.full_like(proto_src, 1.0 / max(proto_src.shape[-1], 1))
                tgt = torch.full_like(proto_tgt, 1.0 / max(proto_tgt.shape[-1], 1))
            family_costs.append(cost)
            family_src.append(src)
            family_tgt.append(tgt)
            aux[f"Family/{family}_clue_cost"] = cost.detach().mean(dim=(1, 2))
        return tuple(family_costs), tuple(family_src), tuple(family_tgt), aux

    @staticmethod
    def _select_family_evidence(
        family_costs: Tuple[torch.Tensor, ...],
        family_src: Tuple[torch.Tensor, ...],
        family_tgt: Tuple[torch.Tensor, ...],
        family_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_cost = (
            torch.stack(family_costs, dim=1) * family_weight[:, :, None, None]
        ).sum(dim=1)
        selected_src = (
            torch.stack(family_src, dim=1) * family_weight[:, :, None]
        ).sum(dim=1)
        selected_tgt = (
            torch.stack(family_tgt, dim=1) * family_weight[:, :, None]
        ).sum(dim=1)
        return selected_cost, selected_src, selected_tgt

    def forward(
        self,
        rgb_search: torch.Tensor,
        tir_search: torch.Tensor,
        rgb_template: torch.Tensor,
        tir_template: torch.Tensor,
        rgb_index: torch.Tensor,
        tir_index: torch.Tensor,
        state: AlignmentState,
    ) -> Tuple[AlignmentState, Dict[str, torch.Tensor]]:
        rgb_down, coords, grid_hw = downsample_search_tokens(
            rgb_search, rgb_index, self.feat_sz, self.spec.down_factor
        )
        tir_down, _, _ = downsample_search_tokens(
            tir_search, tir_index, self.feat_sz, self.spec.down_factor
        )
        expected_side = self.feat_sz // self.spec.down_factor
        expected_tokens = expected_side * expected_side
        if grid_hw != (expected_side, expected_side):
            raise RuntimeError(
                f"{self.spec.name} RCCA grid mismatch: expected "
                f"{expected_side}x{expected_side}, got {grid_hw}"
            )
        if rgb_down.shape[1] != expected_tokens or tir_down.shape[1] != expected_tokens:
            raise RuntimeError(
                f"{self.spec.name} RCCA token mismatch: expected {expected_tokens}, "
                f"got RGB={rgb_down.shape[1]}, TIR={tir_down.shape[1]}"
            )

        b = rgb_search.shape[0]
        if state.geometry is None:
            prev_geometry = torch.zeros(b, 4, device=rgb_search.device, dtype=rgb_search.dtype)
            prev_conf = torch.zeros(b, device=rgb_search.device, dtype=rgb_search.dtype)
            prev_unc = torch.ones(b, device=rgb_search.device, dtype=rgb_search.dtype)
        else:
            prev_geometry, prev_conf, prev_unc = state.geometry, state.confidence, state.uncertainty
        state_vec = torch.cat([prev_geometry, prev_conf[:, None], prev_unc[:, None]], dim=-1)

        route = self.combination_router(
            rgb_search=rgb_down,
            tir_search=tir_down,
            rgb_template=rgb_template,
            tir_template=tir_template,
            state_vec=state_vec,
        )
        active_experts = tuple(self.spec.experts)

        agent_pack = self.agent_constructor(
            rgb_search=rgb_down,
            tir_search=tir_down,
            rgb_template=rgb_template,
            tir_template=tir_template,
            search_coords=coords,
            template_coords=None,
            experts=active_experts,
            stage_id=self.spec.stage_id,
            state_vec=state_vec,
            grid_hw=grid_hw,
            sparse_build=False,
        )
        rgb_agents, rgb_slices = self._concat_agents(agent_pack["rgb"], active_experts)
        tir_agents, tir_slices = self._concat_agents(agent_pack["tir"], active_experts)
        search_pos = self.search_pos_mlp(coords)
        rgb_agent_input = rgb_down + search_pos
        tir_agent_input = tir_down + search_pos
        rgb_token_projection = self.agent_interaction.project_tokens(rgb_agent_input)
        tir_token_projection = self.agent_interaction.project_tokens(tir_agent_input)
        rgb_token_key = rgb_token_projection[1]
        tir_token_key = tir_token_projection[1]

        rgb_agent_attn = self.agent_interaction.agent_to_token_attn_from_key(
            rgb_token_key, agents=rgb_agents
        )
        tir_agent_attn = self.agent_interaction.agent_to_token_attn_from_key(
            tir_token_key, agents=tir_agents
        )
        appearance_cost = self._agent_appearance_cost(rgb_agent_attn, tir_agent_attn)
        rgb_relation_tokens = rgb_down
        tir_relation_tokens = tir_down

        src_match = agent_pack["rgb"].aux.get(
            "matchability", torch.ones(rgb_down.shape[:2], device=rgb_down.device, dtype=rgb_down.dtype)
        ).clamp(0.02, 1.0)
        tgt_match = agent_pack["tir"].aux.get(
            "matchability", torch.ones(tir_down.shape[:2], device=tir_down.device, dtype=tir_down.dtype)
        ).clamp(0.02, 1.0)
        src_transport_mass = agent_pack["rgb"].aux.get(
            "transport_mass", torch.ones(rgb_down.shape[:2], device=rgb_down.device, dtype=rgb_down.dtype)
        ).clamp(0.02, 1.0)
        tgt_transport_mass = agent_pack["tir"].aux.get(
            "transport_mass", torch.ones(tir_down.shape[:2], device=tir_down.device, dtype=tir_down.dtype)
        ).clamp(0.02, 1.0)

        branch_attn: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        if "R" in active_experts:
            r_same_src = rgb_agent_attn[:, :, rgb_slices["R"], :]
            r_same_tgt = tir_agent_attn[:, :, tir_slices["R"], :]
            fused_r = agent_pack["fused"].agents["R"]
            fused_r_query = self.agent_interaction.project_agent_query(fused_r)
            r_fused_src = self.agent_interaction.agent_to_token_attn_from_key(
                rgb_token_key, agent_query=fused_r_query
            )
            r_fused_tgt = self.agent_interaction.agent_to_token_attn_from_key(
                tir_token_key, agent_query=fused_r_query
            )
            branch_attn["R_same"] = (r_same_src, r_same_tgt)
            branch_attn["R_fused"] = (r_fused_src, r_fused_tgt)
        if "S" in active_experts:
            s_rgb_src = rgb_agent_attn[:, :, rgb_slices["S"], :]
            s_rgb_tgt = self.agent_interaction.agent_to_token_attn_from_key(
                tir_token_key, agents=agent_pack["rgb"].agents["S"]
            )
            s_tir_src = self.agent_interaction.agent_to_token_attn_from_key(
                rgb_token_key, agents=agent_pack["tir"].agents["S"]
            )
            s_tir_tgt = tir_agent_attn[:, :, tir_slices["S"], :]
            branch_attn["S_rgb"] = (s_rgb_src, s_rgb_tgt)
            branch_attn["S_tir"] = (s_tir_src, s_tir_tgt)
        if "D" in active_experts:
            d_rgb_src = rgb_agent_attn[:, :, rgb_slices["D"], :]
            d_rgb_tgt = self.agent_interaction.agent_to_token_attn_from_key(
                tir_token_key, agents=agent_pack["rgb"].agents["D"]
            )
            d_tir_src = self.agent_interaction.agent_to_token_attn_from_key(
                rgb_token_key, agents=agent_pack["tir"].agents["D"]
            )
            d_tir_tgt = tir_agent_attn[:, :, tir_slices["D"], :]
            branch_attn["D_rgb"] = (d_rgb_src, d_rgb_tgt)
            branch_attn["D_tir"] = (d_tir_src, d_tir_tgt)


        family_costs, family_src, family_tgt, branch_aux = self._family_evidence(branch_attn)
        selected_clue_cost, selected_src_mass, selected_tgt_mass = self._select_family_evidence(
            family_costs,
            family_src,
            family_tgt,
            route["family_weight"],
        )

        transition, relation, cost, row_mass, ot_aux = self.ot(
            stage_name=self.spec.name,
            rgb_tokens=rgb_relation_tokens,
            tir_tokens=tir_relation_tokens,
            coords=coords,
            prev_geometry=prev_geometry,
            prev_uncertainty=prev_unc,
            src_mass=selected_src_mass,
            tgt_mass=selected_tgt_mass,
            src_transport_mass=src_transport_mass,
            tgt_transport_mass=tgt_transport_mass,
            src_match=src_match,
            tgt_match=tgt_match,
            bandwidth_multiplier=self.spec.sigma,
            clue_cost=selected_clue_cost,
            appearance_cost=appearance_cost,
        )

        geom_update_ot, geometry_ot, confidence, uncertainty = self.readout(
            self.spec.name,
            relation,
            coords,
            ot_aux["src_conf"],
            prev_geometry,
        )
        relation_residual, residual_parts = self.relation_residual_head(
            relation,
            coords,
            geometry_ot,
            ot_aux["src_conf"],
            return_parts=True,
        )
        if self.spec.name == "refinement":
            # Compose both residual corrections in the residual coordinate frame,
            # then bound the complete Refinement update around the Scale result.
            geom_update = compose_diag_geometry(geom_update_ot, relation_residual)
            geom_update = bound_diag_residual(geom_update, max_shift=0.02, max_log_scale=0.03)
            geometry = compose_diag_geometry(prev_geometry, geom_update)
        else:
            geometry = compose_diag_geometry(geometry_ot, relation_residual)
            if self.spec.name == "center":
                geom_update = geometry
            else:
                geom_update = self.readout.absolute_to_residual(prev_geometry, geometry)
        new_state = AlignmentState(geometry=geometry, confidence=confidence, uncertainty=uncertainty)

        if agent_pack["rgb"].aux["detail_select"].shape[1] > 0:
            div_rgb, peak_rgb = detail_regularization(agent_pack["rgb"].aux["detail_select"])
            div_tir, peak_tir = detail_regularization(agent_pack["tir"].aux["detail_select"])
        else:
            div_rgb = peak_rgb = rgb_down.sum() * 0.0
            div_tir = peak_tir = tir_down.sum() * 0.0

        aux: Dict[str, torch.Tensor] = {}
        aux.update(branch_aux)
        aux.update(ot_aux)
        aux.update({
            "cue_weights": route["family_weight"],
            "cue_mixer_entropy": route["entropy"],
            "cue_mixer_logits": route["logits"],
            "rcca_grid_hw": rgb_down.new_tensor(grid_hw, dtype=torch.long),
            "rcca_token_count": rgb_down.new_tensor(float(expected_tokens)),
            # ``relation`` is the UOT solution. ``transition`` is only its row
            # conditional form for supervision and geometric expectation.
            "relation": relation,
            "transition": transition,
            "transport_plan": relation,
            "cost_matrix": cost,
            "row_mass": row_mass,
            "col_mass": ot_aux["col_mass"],
            "coords": coords,
            "prealigned_coords": apply_diag_geometry(coords, prev_geometry),
            "delta_ot": geom_update_ot,
            "delta_ot_dirmag": geometry_to_dirmag(geom_update_ot),
            "relation_offset_residual": relation_residual,
            # Compatibility aliases for existing parsers/checkpoints are kept in
            # output names only; the residual no longer reads raw feature pools.
            "feature_offset_residual": relation_residual,
            "feature_offset_residual_dirmag": geometry_to_dirmag(relation_residual),
            "feature_residual_xy_dir": residual_parts["xy_dir"],
            "feature_residual_xy_mag": residual_parts["xy_mag"],
            "feature_residual_scale_dir": residual_parts["scale_dir"],
            "feature_residual_scale_mag": residual_parts["scale_mag"],
            "delta": geom_update,
            "delta_dirmag": geometry_to_dirmag(geom_update),
            "geometry_ot": geometry_ot,
            "geometry": geometry,
            "geometry_dirmag": geometry_to_dirmag(geometry),
            "confidence": confidence,
            "uncertainty": uncertainty,
            "loss_detail_div": 0.5 * (div_rgb + div_tir),
            "loss_detail_peak": 0.5 * (peak_rgb + peak_tir),
            "rgb_detail_score": agent_pack["rgb"].aux["detail_score"],
            "tir_detail_score": agent_pack["tir"].aux["detail_score"],
            "rgb_same_template_response": agent_pack["rgb"].aux["same_template_response"],
            "tir_same_template_response": agent_pack["tir"].aux["same_template_response"],
            "rgb_fused_template_response": agent_pack["rgb"].aux["fused_template_response"],
            "tir_fused_template_response": agent_pack["tir"].aux["fused_template_response"],
            "rgb_d_positions": agent_pack["rgb"].aux["d_positions"],
            "tir_d_positions": agent_pack["tir"].aux["d_positions"],
        })
        return new_state, aux


class ProgressiveAlignmentController(nn.Module):
    """Center/Scale/Refinement prediction at layers 9/10/11."""

    def __init__(
        self,
        dim: int,
        feat_sz: int,
        num_heads: int = 8,
        joint_config: Optional[dict] = None,
    ):
        super().__init__()
        self.joint_config = dict(joint_config or {})
        # The paper evaluates all three stages on the complete token grid.
        down_factors = (1, 1, 1)
        self.stage_specs = (
            AlignmentStageSpec("center", 9, 0, ("R", "S", "D"), down_factors[0], 4.0),
            AlignmentStageSpec("scale", 10, 1, ("R", "S", "D"), down_factors[1], 2.0),
            AlignmentStageSpec("refinement", 11, 2, ("R", "S", "D"), down_factors[2], 1.0),
        )
        self.stage_by_layer = {s.layer: s for s in self.stage_specs}
        self.stages = nn.ModuleDict({
            s.name: RCCAStage(dim, feat_sz, s, joint_config=self.joint_config)
            for s in self.stage_specs
        })

    def has_stage(self, layer: int) -> bool:
        return layer in self.stage_by_layer

    def get_stage_name(self, layer: int) -> str:
        return self.stage_by_layer[layer].name

    def is_complete(self, state: AlignmentState) -> bool:
        return state.geometry is not None and state.confidence is not None

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> AlignmentState:
        return AlignmentState(
            geometry=torch.zeros(batch_size, 4, device=device, dtype=dtype),
            confidence=torch.zeros(batch_size, device=device, dtype=dtype),
            uncertainty=torch.ones(batch_size, device=device, dtype=dtype),
        )

    def forward_stage(
        self,
        layer: int,
        rgb_search: torch.Tensor,
        tir_search: torch.Tensor,
        rgb_template: torch.Tensor,
        tir_template: torch.Tensor,
        rgb_index: torch.Tensor,
        tir_index: torch.Tensor,
        state: AlignmentState,
    ) -> Tuple[AlignmentState, Dict[str, torch.Tensor]]:
        spec = self.stage_by_layer[layer]
        state, aux = self.stages[spec.name](
            rgb_search=rgb_search,
            tir_search=tir_search,
            rgb_template=rgb_template,
            tir_template=tir_template,
            rgb_index=rgb_index,
            tir_index=tir_index,
            state=state,
        )
        prefixed = {f"{spec.name}_{k}": v for k, v in aux.items() if isinstance(v, torch.Tensor)}
        prefixed[f"{spec.name}_down_factor"] = torch.tensor(float(spec.down_factor), device=rgb_search.device)
        return state, prefixed
