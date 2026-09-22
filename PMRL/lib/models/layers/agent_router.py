"""Adaptive mixing of target, structure, and detail correspondence cues."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCueMixer(nn.Module):
    """Predict the three non-negative cue weights used by Eq. (4).

    The historical implementation also contained categorical combinations,
    hard routing, sparse expert execution, and compute-cost regularization for
    ablation studies. PMRL uses a single soft mixer, so only that path remains.
    """

    FAMILY_NAMES: Tuple[str, ...] = ("R", "S", "D")

    def __init__(self, dim: int, temperature: float = 0.2, hidden_dim: int = 48):
        super().__init__()
        self.temperature = max(float(temperature), 1e-4)
        self.token_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 32))
        # Preserve the learned mixer parameter name used by current checkpoints.
        self.family_router = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.FAMILY_NAMES)),
        )

    def _token_summary(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        projected = F.normalize(self.token_proj(tokens), dim=-1)
        pooled = F.normalize(projected.mean(dim=1), dim=-1)
        dispersion = projected.std(dim=1, unbiased=False).mean(dim=-1)
        return pooled, dispersion

    def _diagnostics(
        self,
        rgb_search: torch.Tensor,
        tir_search: torch.Tensor,
        rgb_template: torch.Tensor,
        tir_template: torch.Tensor,
        state_vec: torch.Tensor,
    ) -> torch.Tensor:
        rgb_s, rgb_disp = self._token_summary(rgb_search)
        tir_s, tir_disp = self._token_summary(tir_search)
        rgb_t, _ = self._token_summary(rgb_template)
        tir_t, _ = self._token_summary(tir_template)
        return torch.stack(
            [
                (rgb_s * tir_s).sum(dim=-1),
                (rgb_t * tir_t).sum(dim=-1),
                (rgb_s - tir_s).abs().mean(dim=-1),
                rgb_disp,
                tir_disp,
                (rgb_disp - tir_disp).abs(),
                state_vec[:, 0:2].norm(dim=-1),
                state_vec[:, 2:4].abs().mean(dim=-1),
                state_vec[:, 4],
                state_vec[:, 5],
            ],
            dim=-1,
        )

    def forward(
        self,
        rgb_search: torch.Tensor,
        tir_search: torch.Tensor,
        rgb_template: torch.Tensor,
        tir_template: torch.Tensor,
        state_vec: torch.Tensor,
        mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        del mode
        diagnostics = self._diagnostics(
            rgb_search, tir_search, rgb_template, tir_template, state_vec
        )
        logits = self.family_router(diagnostics)
        weights = F.softmax(logits / self.temperature, dim=-1)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        return {
            "diagnostics": diagnostics,
            "logits": logits,
            "family_weight": weights,
            "entropy": entropy,
        }
