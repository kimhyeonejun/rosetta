"""Verbatim mirror of the Rate loss component from ic4r/loss/scr_loss.py.

Only the ``Rate`` module is mirrored (the SCR4R training step uses
``Rate.components`` exclusively). ``Distortion`` and ``SCRActionLoss`` are
not needed because the SCR4RBase implementation computes the
``Σ recon · grad`` distortion surrogate inline.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class Rate(nn.Module):
    def __init__(
        self,
        weight: float = 1.0,
        eps: float = 1e-9,
        mask_rate_alpha: float = 1.0,
    ):
        """Rate loss used by SCR-style objectives.

        Args:
            mask_rate_alpha: Exponent on per-cell bit cost in the latent-
                mask rate term. Controls how strongly the selector's
                rate-driven drop signal is biased toward expensive cells.
                Must be ≥ 0. Continuous knob, fixed (not learned):
                  - ``1.0`` (default): standard rate
                    ``Σ(-log2(y) · mask) / (B·num_pixels)``; each mask
                    cell's gradient ∝ its own bit cost, so the selector
                    preferentially drops expensive cells.
                  - ``0.0``: uniform mask gradient; rate-side has no
                    preference for *which* cells to drop, only how many.
                    VLA gradient alone decides which to drop.
                  - ``0 < alpha < 1``: soft blend (weaker bias toward
                    expensive cells).
                  - ``alpha > 1``: amplified bias toward expensive cells.

                Forward formula: ``loss = raw_y_bpp · Σ(w · mask) / Σ w``
                with ``w[i] = (-log2 y[i])^alpha``. Reduces exactly to
                ``masked_y_bpp`` at α=1 and ``raw_y_bpp · mask.mean()``
                at α=0. Forward value stays in bpp units across α, so
                ``rate_distortion_lambda`` does not need re-tuning when
                α is swept.
        """
        super().__init__()
        alpha = float(mask_rate_alpha)
        if alpha < 0:
            raise ValueError(
                f"mask_rate_alpha must be ≥ 0, got {mask_rate_alpha!r}."
            )
        self.weight = float(weight)
        self.eps = float(eps)
        self.mask_rate_alpha = alpha

    def components(
        self,
        y_likelihoods: torch.Tensor,
        z_likelihoods: torch.Tensor,
        *,
        y_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        num_pixels: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return weighted rate plus masked/raw BPP components."""
        if num_pixels is None:
            if images is None:
                raise ValueError("Either images or num_pixels must be provided to compute rate.")
            num_pixels = images.shape[-2] * images.shape[-1]

        batch_size = y_likelihoods.shape[0]
        # Entropy models can emit zeros (or tiny negatives from numeric noise).
        # Clamp before log to avoid -inf/NaN in rate terms.
        y_safe = torch.clamp(y_likelihoods, min=self.eps)
        z_safe = torch.clamp(z_likelihoods, min=self.eps)
        raw_y_bpp = -torch.log2(y_safe).sum() / (batch_size * num_pixels)
        z_bpp = -torch.log2(z_safe).sum() / (batch_size * num_pixels)

        if y_mask is not None:
            if y_mask.shape != y_safe.shape:
                raise ValueError(
                    "y_mask must match y_likelihoods shape: "
                    f"got {y_mask.shape} vs {y_safe.shape}"
                )
            mask_f = y_mask.to(device=y_safe.device, dtype=y_safe.dtype)
            masked_y_bpp = (-torch.log2(y_safe) * mask_f).sum() / (batch_size * num_pixels)
            # Generalized rate term controlled by ``mask_rate_alpha``:
            #   w[i] = (-log2 y[i])^alpha
            #   loss_y_bpp = raw_y_bpp · Σ(w · mask) / Σ w
            # Reduces to ``masked_y_bpp`` at alpha=1 and to
            # ``raw_y_bpp · mask.mean()`` at alpha=0. The mask gradient at
            # cell i is ∝ w[i] / Σ w, so larger alpha concentrates the
            # selector's drop signal on higher-bit-cost cells. Forward value
            # stays in bpp units across alpha (raw_y_bpp ·
            # weighted-mean-of-mask), so ``rate_distortion_lambda`` doesn't
            # need re-tuning when sweeping alpha. ``masked_y_bpp`` above
            # remains the true mask-weighted bpp metric.
            cell_costs = (-torch.log2(y_safe)).clamp(min=self.eps)
            weights = cell_costs ** self.mask_rate_alpha
            loss_y_bpp = raw_y_bpp * (weights * mask_f).sum() / weights.sum()
        else:
            masked_y_bpp = raw_y_bpp
            loss_y_bpp = raw_y_bpp

        weighted_rate = loss_y_bpp + self.weight * z_bpp
        masked_total_bpp = masked_y_bpp + z_bpp
        raw_total_bpp = raw_y_bpp + z_bpp
        return weighted_rate, masked_total_bpp, masked_y_bpp, raw_total_bpp, raw_y_bpp

    def forward(
        self,
        images: Optional[torch.Tensor],
        y_likelihoods: torch.Tensor,
        z_likelihoods: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the weighted SCR rate term."""
        weighted_rate, _, _, _, _ = self.components(
            y_likelihoods,
            z_likelihoods,
            y_mask=kwargs.get("y_mask"),
            images=images,
            num_pixels=kwargs.get("num_pixels"),
        )
        return weighted_rate


__all__ = ["Rate"]
