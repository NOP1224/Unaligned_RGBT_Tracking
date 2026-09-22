"""
Agent construction only.

No cross-modal operation, no RCCA attention, no transition matrix, and no TCMDA
fusion is implemented here. This file only builds stage-conditioned, modality-
conditioned R/S/D and derived same-side agent tokens.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AgentSet:
    agents: Dict[str, torch.Tensor]
    positions: Dict[str, torch.Tensor]
    aux: Dict[str, torch.Tensor]


class DetailUtilityHead(nn.Module):
    """Utility scorer for scene-detail D-Agent anchors."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act = nn.GELU()
        self.score = nn.Conv2d(dim, 1, kernel_size=1)
        self.fallback = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1))

    def forward(self, x: torch.Tensor, grid_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        b, l, c = x.shape
        if grid_hw is not None and grid_hw[0] * grid_hw[1] == l:
            h, w = grid_hw
            feat = self.norm(x).transpose(1, 2).reshape(b, c, h, w)
            score = self.score(self.act(self.dwconv(feat))).flatten(1)
            return torch.sigmoid(score)
        return torch.sigmoid(self.fallback(x).squeeze(-1))


class SpatialSoftTopK(nn.Module):
    """
    Differentiable D-Agent selector without doing attention.

    Each slot owns a learnable spatial anchor. The token selection score is the
    same-side detail utility plus a learnable spatial prior around that anchor.
    """

    def __init__(self, num_slots: int, init_temperature: float = 0.7, hard_inference: bool = False):
        super().__init__()
        self.num_slots = num_slots
        self.hard_inference = hard_inference
        # Initialize anchors on a regular 2-D grid. ``slot_xy`` is stored in
        # logit space because forward() applies sigmoid. Random values followed
        # by sigmoid cluster all anchors around the image centre and leave large
        # regions uncovered at initialization.
        grid_h = max(int(round(num_slots ** 0.5)), 1)
        while grid_h > 1 and num_slots % grid_h != 0:
            grid_h -= 1
        grid_w = max(num_slots // grid_h, 1)
        yy, xx = torch.meshgrid(
            torch.linspace(0.5 / grid_h, 1.0 - 0.5 / grid_h, grid_h),
            torch.linspace(0.5 / grid_w, 1.0 - 0.5 / grid_w, grid_w),
            indexing="ij",
        )
        init_xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2)[:num_slots]
        init_xy = init_xy.clamp(1e-4, 1.0 - 1e-4)
        self.slot_xy = nn.Parameter(torch.logit(init_xy))
        self.log_sigma = nn.Parameter(torch.full((num_slots, 2), -1.2))
        self.slot_bias = nn.Parameter(torch.zeros(num_slots))
        self.temperature = init_temperature

    def forward(self, x: torch.Tensor, coords: torch.Tensor, utility: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l, c = x.shape
        slot_xy = self.slot_xy.sigmoid().view(1, self.num_slots, 1, 2)
        sigma = self.log_sigma.exp().clamp_min(0.03).view(1, self.num_slots, 1, 2)
        dist = ((coords.unsqueeze(1) - slot_xy) / sigma).pow(2).sum(dim=-1)
        logits = utility.unsqueeze(1) - 0.5 * dist + self.slot_bias.view(1, self.num_slots, 1)

        # Keep train/val behavior consistent. Hard TopK is only for explicit deployment
        # export, otherwise validation collapses to one-hot anchors and diverges from
        # the trained soft selection distribution.
        if (not self.training) and self.hard_inference:
            idx = logits.argmax(dim=-1)
            select = x.new_zeros(b, self.num_slots, l)
            select.scatter_(2, idx.unsqueeze(-1), 1.0)
        else:
            select = F.softmax(logits / max(self.temperature, 1e-4), dim=-1)

        agents = torch.matmul(select, x)
        positions = torch.matmul(select, coords)
        return agents, positions, select


class RegularGridPool(nn.Module):
    """Token pooling for R/S agents. This is pooling, not attention."""

    def __init__(self, num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens

    @staticmethod
    def _factor_grid(length: int) -> Optional[Tuple[int, int]]:
        h = max(int(length ** 0.5), 1)
        while h > 1 and length % h != 0:
            h -= 1
        w = max(length // h, 1)
        return h, w

    @staticmethod
    def _output_grid(num_tokens: int) -> Tuple[int, int]:
        h = max(int(num_tokens ** 0.5), 1)
        while h > 1 and num_tokens % h != 0:
            h -= 1
        return h, max(num_tokens // h, 1)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        grid_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, l, c = x.shape
        if l == self.num_tokens:
            return x, coords

        # Preserve the 2-D topology. For multiple templates, tokens are expected
        # to be concatenated template-by-template; corresponding spatial cells
        # are averaged before pooling to the requested agent grid.
        if grid_hw is None:
            # Prefer a square per-template grid. E.g. L=128 is interpreted as
            # two 8x8 templates rather than a single 8x16 strip.
            base_hw = None
            for num_segments in range(1, 9):
                if l % num_segments != 0:
                    continue
                seg_len = l // num_segments
                side = int(round(seg_len ** 0.5))
                if side * side == seg_len:
                    base_hw = (side, side)
                    break
            grid_hw = base_hw if base_hw is not None else self._factor_grid(l)

        h, w = grid_hw
        per_grid = h * w
        if per_grid > 0 and l % per_grid == 0:
            num_segments = l // per_grid
            feat = x.reshape(b, num_segments, h, w, c).mean(dim=1)
            feat = feat.permute(0, 3, 1, 2).contiguous()
            out_h, out_w = self._output_grid(self.num_tokens)
            pooled = F.adaptive_avg_pool2d(feat, (out_h, out_w)).flatten(2).transpose(1, 2).contiguous()

            yy, xx = torch.meshgrid(
                torch.linspace(0.5 / out_h, 1.0 - 0.5 / out_h, out_h, device=x.device, dtype=x.dtype),
                torch.linspace(0.5 / out_w, 1.0 - 0.5 / out_w, out_w, device=x.device, dtype=x.dtype),
                indexing="ij",
            )
            pooled_coords = torch.stack([xx, yy], dim=-1).reshape(1, -1, 2).expand(b, -1, -1).contiguous()
            return pooled, pooled_coords

        # Defensive fallback for malformed token layouts. Current PMATrack paths
        # all enter the 2-D branch above.
        xt = x.transpose(1, 2)
        pooled = F.adaptive_avg_pool1d(xt, self.num_tokens).transpose(1, 2)
        ct = coords.transpose(1, 2)
        pooled_coords = F.adaptive_avg_pool1d(ct, self.num_tokens).transpose(1, 2)
        return pooled, pooled_coords


class AgentConstructor(nn.Module):
    """
    Construct same-side agents with state, position, stage and modality encodings.

    R-Agent: pooled template identity tokens.
    S-Agent: pooled search layout tokens.
    D-Agent: learnable detail anchors from utility + spatial slots.
    R2S/S2D/RS2D: same-side state-conditioned derived agents.
    """

    def __init__(self, dim: int, r_agents: int = 64, s_agents: int = 64, d_agents: int = 64, num_stages: int = 3):
        super().__init__()
        self.dim = dim
        self.r_pool = RegularGridPool(r_agents)
        self.s_pool = RegularGridPool(s_agents)
        self.d_selector = SpatialSoftTopK(d_agents)
        self.detail_head = DetailUtilityHead(dim)

        self.pos_mlp = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.search_pos_mlp = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.template_pos_mlp = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.template_id_embed = nn.Embedding(4, dim)
        nn.init.zeros_(self.template_id_embed.weight)
        self.state_mlp = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.stage_embed = nn.Embedding(num_stages, dim)
        self.modality_embed = nn.Embedding(2, dim)  # 0 RGB, 1 TIR
        self.type_embed = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, 1, dim))
            for name in ("R", "S", "D", "R2S", "S2D", "RS2D")
        })
        for p in self.type_embed.values():
            nn.init.trunc_normal_(p, std=0.02)

        self.r2s = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Sigmoid())
        self.s2d = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Sigmoid())
        self.rs2d = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim), nn.Sigmoid())

        # R-Agent enhancement only. These same-side / fused template-search
        # transitions are not injected into the OT pairwise cost. They only
        # improve the target-agent state used by RCCA to estimate token mass and
        # matchability.
        self.template_q = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.search_k = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.r_same_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.r_fuse_proj = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.fuse_gate = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.mass_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1))
        self.match_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1))

    @staticmethod
    def _default_template_layout(
        batch: int,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Return repeated 2-D template coordinates and template IDs.

        A single template is usually 8x8. Dynamic-template inference concatenates
        two such grids, so L=128 must remain two 8x8 layouts rather than one line.
        """
        num_templates, side = 1, int(round(length ** 0.5))
        if side * side != length:
            for n in range(2, 5):
                if length % n != 0:
                    continue
                seg_len = length // n
                seg_side = int(round(seg_len ** 0.5))
                if seg_side * seg_side == seg_len:
                    num_templates, side = n, seg_side
                    break
        if side * side * num_templates != length:
            h = max(int(length ** 0.5), 1)
            while h > 1 and length % h != 0:
                h -= 1
            w = max(length // h, 1)
            num_templates, grid_hw = 1, (h, w)
        else:
            grid_hw = (side, side)

        h, w = grid_hw
        yy, xx = torch.meshgrid(
            torch.linspace(0.5 / h, 1.0 - 0.5 / h, h, device=device, dtype=dtype),
            torch.linspace(0.5 / w, 1.0 - 0.5 / w, w, device=device, dtype=dtype),
            indexing="ij",
        )
        base = torch.stack([xx, yy], dim=-1).reshape(1, h * w, 2)
        coords = base.repeat(batch, num_templates, 1)
        ids = torch.arange(num_templates, device=device).clamp_max(3)
        ids = ids[:, None].expand(num_templates, h * w).reshape(1, -1).repeat(batch, 1)
        return coords, ids, grid_hw

    def _template_position(self, template: torch.Tensor, template_coords: Optional[torch.Tensor]):
        b, l, _ = template.shape
        if template_coords is None:
            template_coords, template_ids, grid_hw = self._default_template_layout(
                b, l, template.device, template.dtype
            )
        else:
            template_ids = torch.zeros((b, l), device=template.device, dtype=torch.long)
            side = int(round(l ** 0.5))
            grid_hw = (side, side) if side * side == l else None
        pos = self.template_pos_mlp(template_coords)
        pos = pos + self.template_id_embed(template_ids)
        return template_coords, pos, grid_hw

    def _encode(
        self,
        q: torch.Tensor,
        pos: torch.Tensor,
        expert: str,
        stage_id: int,
        modality_id: int,
        state_vec: torch.Tensor,
    ) -> torch.Tensor:
        b = q.shape[0]
        stage = self.stage_embed.weight[stage_id].view(1, 1, -1)
        modality = self.modality_embed.weight[modality_id].view(1, 1, -1)
        state = self.state_mlp(state_vec).view(b, 1, -1)
        return q + self.pos_mlp(pos) + self.type_embed[expert] + stage + modality + state

    def _encode_fused(
        self,
        q: torch.Tensor,
        pos: torch.Tensor,
        expert: str,
        stage_id: int,
        state_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Encode modality-neutral fused agents for cross-modal matching."""
        b = q.shape[0]
        stage = self.stage_embed.weight[stage_id].view(1, 1, -1)
        state = self.state_mlp(state_vec).view(b, 1, -1)
        return q + self.pos_mlp(pos) + self.type_embed[expert] + stage + state

    def _template_search_context(
        self,
        template_query: torch.Tensor,
        search_key: torch.Tensor,
    ) -> torch.Tensor:
        """Return the token-level template-search response map.

        Template pooling/Q projection and search positional/K projection are
        prepared once by :meth:`forward` and reused across same/fused responses.
        The historical ``attn @ search`` context was never consumed and is
        intentionally removed.
        """
        attn = F.softmax(
            torch.matmul(template_query, search_key.transpose(-2, -1)) * (self.dim ** -0.5),
            dim=-1,
        )
        return attn.mean(dim=1)

    def forward_one_side(
        self,
        search: torch.Tensor,
        template: torch.Tensor,
        search_coords: torch.Tensor,
        template_coords: Optional[torch.Tensor],
        experts: Iterable[str],
        stage_id: int,
        modality_id: int,
        state_vec: torch.Tensor,
        grid_hw: Optional[Tuple[int, int]],
        sparse_build: bool = False,
        search_with_pos: Optional[torch.Tensor] = None,
        pooled_template: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> AgentSet:
        b, l, c = search.shape
        experts = tuple(experts)
        expert_set = set(experts)
        need_r = (not sparse_build) or bool(expert_set.intersection({"R", "R2S", "RS2D"}))
        need_s = (not sparse_build) or bool(expert_set.intersection({"S", "R2S", "S2D", "RS2D"}))
        need_d = (not sparse_build) or bool(expert_set.intersection({"D", "S2D", "RS2D"}))

        search_pos = (
            search + self.search_pos_mlp(search_coords)
            if search_with_pos is None else search_with_pos
        )
        detail_score = self.detail_head(search_pos, grid_hw=grid_hw)
        token_prior = detail_score
        transport_mass = torch.sigmoid(self.mass_head(search_pos).squeeze(-1) + token_prior)
        matchability = torch.sigmoid(self.match_head(search_pos).squeeze(-1) + 0.5 * token_prior)

        r = r_pos = None
        if need_r:
            if pooled_template is None:
                template_coords_resolved, template_pos_embed, template_grid_hw = self._template_position(
                    template, template_coords
                )
                template_pos = template + template_pos_embed
                r, r_pos = self.r_pool(
                    template_pos, template_coords_resolved, grid_hw=template_grid_hw
                )
            else:
                r, r_pos = pooled_template

        s = s_pos = None
        if need_s:
            s, s_pos = self.s_pool(search_pos, search_coords, grid_hw=grid_hw)

        if need_d:
            d, d_pos, d_select = self.d_selector(search_pos, search_coords, detail_score)
        else:
            d = search.new_empty((b, 0, c))
            d_pos = search_coords.new_empty((b, 0, 2))
            d_select = search.new_empty((b, 0, l))

        r_state = r.mean(dim=1) if r is not None else search.new_zeros((b, c))
        s_state = s.mean(dim=1) if s is not None else search.new_zeros((b, c))
        agents: Dict[str, torch.Tensor] = {}
        positions: Dict[str, torch.Tensor] = {}

        for name in experts:
            if name == "R":
                q, pos = r, r_pos
            elif name == "S":
                q, pos = s, s_pos
            elif name == "D":
                q, pos = d, d_pos
            elif name == "R2S":
                q, pos = s * self.r2s(r_state).unsqueeze(1), s_pos
            elif name == "S2D":
                q, pos = d * self.s2d(s_state).unsqueeze(1), d_pos
            elif name == "RS2D":
                q, pos = d * self.rs2d(torch.cat([r_state, s_state], dim=-1)).unsqueeze(1), d_pos
            else:
                raise ValueError(f"Unknown expert: {name}")
            if q is None or pos is None:
                raise RuntimeError(f"Expert {name} was requested but its prerequisite was not built")
            agents[name] = self._encode(q, pos, name, stage_id, modality_id, state_vec)
            positions[name] = pos

        aux = {
            "detail_score": detail_score,
            "detail_select": d_select,
            "d_positions": d_pos,
            "transport_mass": transport_mass,
            "matchability": matchability,
        }
        return AgentSet(agents=agents, positions=positions, aux=aux)

    def forward(
        self,
        rgb_search: torch.Tensor,
        tir_search: torch.Tensor,
        rgb_template: torch.Tensor,
        tir_template: torch.Tensor,
        search_coords: torch.Tensor,
        template_coords: Optional[torch.Tensor],
        experts: Iterable[str],
        stage_id: int,
        state_vec: torch.Tensor,
        grid_hw: Optional[Tuple[int, int]],
        sparse_build: bool = False,
    ) -> Dict[str, AgentSet]:
        experts = tuple(experts)
        expert_set = set(experts)
        b = rgb_search.shape[0]
        input_template_coords = template_coords

        # Search coordinates are shared by RGB/TIR. Compute their positional
        # projection once, then reuse the positioned tokens in agent construction
        # and in all template-response branches.
        search_pos_embed = self.search_pos_mlp(search_coords)
        rgb_search_pos = rgb_search + search_pos_embed
        tir_search_pos = tir_search + search_pos_embed

        need_r = (not sparse_build) or bool(
            expert_set.intersection({"R", "R2S", "RS2D"})
        )
        template_layout = None
        rgb_template_pool = None
        tir_template_pool = None
        if need_r:
            template_layout = self._template_position(
                rgb_template, input_template_coords
            )
            template_coords_resolved, template_pos_embed, template_grid_hw = template_layout
            rgb_template_pos = rgb_template + template_pos_embed
            rgb_template_pool = self.r_pool(
                rgb_template_pos, template_coords_resolved, grid_hw=template_grid_hw
            )

            if tir_template.shape[1] == rgb_template.shape[1]:
                tir_template_pos = tir_template + template_pos_embed
                tir_template_pool = self.r_pool(
                    tir_template_pos, template_coords_resolved, grid_hw=template_grid_hw
                )
            else:
                # Defensive fallback for non-mirrored template layouts.
                tir_coords, tir_pos_embed, tir_grid_hw = self._template_position(
                    tir_template, input_template_coords
                )
                tir_template_pool = self.r_pool(
                    tir_template + tir_pos_embed, tir_coords, grid_hw=tir_grid_hw
                )

        fused_template = None
        if (not sparse_build) or "R" in expert_set:
            if template_layout is None:
                template_layout = self._template_position(
                    rgb_template, input_template_coords
                )
            fused_coords, fused_template_pos_embed, template_grid_hw = template_layout
            fused_template = self.r_fuse_proj(torch.cat([rgb_template, tir_template], dim=-1))
            fused_template_pos = fused_template + fused_template_pos_embed
            fused_r, fused_r_pos = self.r_pool(
                fused_template_pos, fused_coords, grid_hw=template_grid_hw
            )
            fused_agent = self._encode_fused(fused_r, fused_r_pos, "R", stage_id, state_vec)
            fused_set = AgentSet(agents={"R": fused_agent}, positions={"R": fused_r_pos}, aux={})
        else:
            fused_set = AgentSet(agents={}, positions={}, aux={})

        rgb_set = self.forward_one_side(
            rgb_search, rgb_template, search_coords, input_template_coords,
            experts, stage_id, 0, state_vec, grid_hw, sparse_build=sparse_build,
            search_with_pos=rgb_search_pos, pooled_template=rgb_template_pool,
        )
        tir_set = self.forward_one_side(
            tir_search, tir_template, search_coords, input_template_coords,
            experts, stage_id, 1, state_vec, grid_hw, sparse_build=sparse_build,
            search_with_pos=tir_search_pos, pooled_template=tir_template_pool,
        )

        if sparse_build:
            # These maps are diagnostics only; sparse routing must not trigger
            # four additional template-search attention passes.
            rgb_set.aux["same_template_response"] = rgb_search.new_zeros((b, rgb_search.shape[1]))
            tir_set.aux["same_template_response"] = tir_search.new_zeros((b, tir_search.shape[1]))
            rgb_set.aux["fused_template_response"] = rgb_search.new_zeros((b, rgb_search.shape[1]))
            tir_set.aux["fused_template_response"] = tir_search.new_zeros((b, tir_search.shape[1]))
        else:
            assert rgb_template_pool is not None and tir_template_pool is not None
            assert fused_template is not None
            rgb_search_key = self.search_k(rgb_search_pos)
            tir_search_key = self.search_k(tir_search_pos)
            rgb_template_query = self.template_q(rgb_template_pool[0])
            tir_template_query = self.template_q(tir_template_pool[0])
            fused_template_query = self.template_q(fused_r)

            rgb_same_resp = self._template_search_context(
                rgb_template_query, rgb_search_key
            )
            tir_same_resp = self._template_search_context(
                tir_template_query, tir_search_key
            )
            rgb_fused_resp = self._template_search_context(
                fused_template_query, rgb_search_key
            )
            tir_fused_resp = self._template_search_context(
                fused_template_query, tir_search_key
            )
            rgb_set.aux["same_template_response"] = rgb_same_resp
            tir_set.aux["same_template_response"] = tir_same_resp
            rgb_set.aux["fused_template_response"] = rgb_fused_resp
            tir_set.aux["fused_template_response"] = tir_fused_resp

        return {
            "rgb": rgb_set,
            "tir": tir_set,
            "fused": fused_set,
        }



def detail_regularization(select: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    b, k, _ = select.shape
    gram = torch.matmul(select, select.transpose(-2, -1))
    eye = torch.eye(k, device=select.device, dtype=select.dtype).unsqueeze(0)
    loss_div = (gram - eye).pow(2).mean()
    p = select.clamp_min(1e-8)
    loss_peak = -(p * p.log()).sum(dim=-1).mean()
    return loss_div, loss_peak
