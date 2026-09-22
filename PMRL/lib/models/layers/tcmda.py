"""Shared-relation feature transport for PMRL."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCMDAFusion(nn.Module):
    """Aggregate opposite-modality values with the AC-UOT relation.

    The previous file also implemented deformable, warped, corrected-relation,
    top-1, and FFN alternatives used only by ablation studies.  The paper's
    final model uses the same conditional relation for coordinate and feature
    transport, with retained transport mass gating feature injection.
    """

    STAGE_STRENGTH = {"center": 0.35, "scale": 0.65, "refinement": 1.0}

    def __init__(
        self,
        dim: int,
        feat_sz: int,
        num_heads: int = 8,
        n_points: int = 4,
        joint_config: Optional[dict] = None,
    ):
        super().__init__()
        del num_heads, n_points
        self.feat_sz = int(feat_sz)
        config = dict(joint_config or {})
        self.quality_floor = min(max(float(config.get("QUALITY_FLOOR", 0.10)), 0.0), 1.0)
        self.shared_rgb_value = nn.Linear(dim, dim)
        self.shared_tir_value = nn.Linear(dim, dim)
        self.rgb_out = nn.Linear(dim, dim)
        self.tir_out = nn.Linear(dim, dim)
        self.fusion_scale_logit = nn.Parameter(torch.tensor(-3.0))

    @staticmethod
    def _conditional_relations(relation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        visible_to_thermal = relation / relation.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        reverse = relation.transpose(-2, -1).contiguous()
        thermal_to_visible = reverse / reverse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return visible_to_thermal, thermal_to_visible

    @staticmethod
    def _square_side(length: int, name: str) -> int:
        side = int(math.isqrt(int(length)))
        if side * side != int(length):
            raise ValueError(f"{name} must form a square token grid, got {length}")
        return side

    @staticmethod
    def _resize_token_field(
        field: Optional[torch.Tensor], source_side: int, target_side: int
    ) -> Optional[torch.Tensor]:
        if field is None or source_side == target_side:
            return field
        if field.ndim != 2 or field.shape[1] != source_side * source_side:
            raise ValueError(f"Invalid token quality shape: {tuple(field.shape)}")
        fmap = field.reshape(field.shape[0], 1, source_side, source_side)
        return F.interpolate(
            fmap, size=(target_side, target_side), mode="bilinear", align_corners=False
        ).flatten(1)

    @staticmethod
    def _lift_relation(relation: torch.Tensor, source_side: int, target_side: int) -> torch.Tensor:
        if source_side == target_side:
            return relation
        batch = relation.shape[0]
        original_mass = relation.sum(dim=(1, 2), keepdim=True)

        lifted = relation.reshape(batch * source_side * source_side, 1, source_side, source_side)
        lifted = F.interpolate(
            lifted, size=(target_side, target_side), mode="bilinear", align_corners=False
        ).reshape(batch, source_side, source_side, target_side, target_side)

        lifted = lifted.permute(0, 3, 4, 1, 2).contiguous().reshape(
            batch * target_side * target_side, 1, source_side, source_side
        )
        lifted = F.interpolate(
            lifted, size=(target_side, target_side), mode="bilinear", align_corners=False
        )
        lifted = lifted.reshape(
            batch, target_side, target_side, target_side, target_side
        ).permute(0, 3, 4, 1, 2).contiguous()
        lifted = lifted.reshape(batch, target_side * target_side, target_side * target_side)
        lifted = lifted.clamp_min(0.0)
        return lifted * (original_mass / lifted.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8))

    def _quality(self, quality: Optional[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
        if quality is None:
            return reference.new_ones(reference.shape[0], reference.shape[1], 1)
        quality = quality.clamp(0.0, 1.0)
        return (self.quality_floor + (1.0 - self.quality_floor) * quality).unsqueeze(-1)

    def forward(
        self,
        rgb_search: torch.Tensor,
        tir_search: torch.Tensor,
        relation: torch.Tensor,
        coords: torch.Tensor,
        stage_name: str = "refinement",
        src_confidence: Optional[torch.Tensor] = None,
        tgt_confidence: Optional[torch.Tensor] = None,
        src_quality: Optional[torch.Tensor] = None,
        tgt_quality: Optional[torch.Tensor] = None,
        alignment_geometry: Optional[torch.Tensor] = None,
        alignment_confidence: Optional[torch.Tensor] = None,
        rgb_index: Optional[torch.Tensor] = None,
        tir_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del coords, src_confidence, tgt_confidence, alignment_geometry
        del alignment_confidence, rgb_index, tir_index

        batch, rgb_length, _ = rgb_search.shape
        tir_length = tir_search.shape[1]
        if relation.ndim != 3 or relation.shape[0] != batch:
            raise ValueError(f"Expected relation BxLxL, got {tuple(relation.shape)}")

        dense_side = self._square_side(rgb_length, "RGB search")
        if self._square_side(tir_length, "TIR search") != dense_side:
            raise ValueError("RGB and TIR search grids must have the same size")
        relation_side = self._square_side(relation.shape[1], "relation source")
        if self._square_side(relation.shape[2], "relation target") != relation_side:
            raise ValueError("Relation source and target grids must have the same size")
        if relation_side != dense_side:
            relation = self._lift_relation(relation, relation_side, dense_side)
            src_quality = self._resize_token_field(src_quality, relation_side, dense_side)
            tgt_quality = self._resize_token_field(tgt_quality, relation_side, dense_side)

        visible_to_thermal, thermal_to_visible = self._conditional_relations(relation)
        rgb_cross = torch.bmm(visible_to_thermal, self.shared_tir_value(tir_search))
        tir_cross = torch.bmm(thermal_to_visible, self.shared_rgb_value(rgb_search))
        rgb_cross = rgb_cross * self._quality(src_quality, rgb_cross)
        tir_cross = tir_cross * self._quality(tgt_quality, tir_cross)

        scale = torch.sigmoid(self.fusion_scale_logit)
        strength = float(self.STAGE_STRENGTH.get(stage_name, 1.0))
        return (
            rgb_search + scale * strength * self.rgb_out(rgb_cross),
            tir_search + scale * strength * self.tir_out(tir_cross),
        )
