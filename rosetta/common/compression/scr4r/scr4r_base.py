"""Verbatim mirror of ic4r/experiments/scr4r_base.py.

Kept inside rosetta_ic4r so this package is self-contained — must stay
behaviourally identical to the upstream class so checkpoints + the
selector training contract round-trip.

Model-agnostic SCR4R training skeleton.

SCR4R (Scalable Compression with Residuals for Robotics) selectively
compresses visual observations for a VLA policy by learning a mask network
on top of a learned image codec.

This module is intentionally free of any specific:
  - Compression model  (MS-ILLM, Cheng2020, VQ-VAE, ...)
  - VLA policy          (OpenPI, Octo, RT-2, ...)
  - Benchmark / data    (VLABench, LIBERO, RLBench, ...)
  - Training framework  (PyTorch Lightning, plain loop, ...)

Subclass ``SCR4RBase`` and implement the abstract methods to plug in your
own models.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Data contracts ───────────────────────────────────────────────────────────

@dataclass
class CompressionEncodeOutput:
    """Intermediate result from the compression encoder."""

    quantized_latent: torch.Tensor
    """[B, C, H, W] quantized latent to be masked and decoded."""

    selector_features: torch.Tensor
    """[B, F, h, w] features fed into the mask network."""

    original_hw: tuple[int, int]
    """(H, W) of the input image before any codec-internal padding/resizing."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Anything else ``_decode`` or ``_compute_rate_loss`` needs
    (likelihoods, scales, intermediate tensors, ...)."""


@dataclass
class VLALossOutput:
    """Result from VLA action-loss computation."""

    loss_value: float
    """Scalar loss for logging (not part of the compression graph)."""

    image_grads: dict[str, torch.Tensor]
    """{key: [B, C, H, W]} **detached** gradients of the VLA loss
    with respect to the reconstructed images."""


@dataclass
class RateLossOutput:
    """Result from compression rate-loss computation."""

    loss: torch.Tensor
    """Differentiable scalar rate loss (flows through the codec)."""

    metrics: dict[str, float] = field(default_factory=dict)
    """Logging-only numbers (bpp, latent_bpp, hyper_bpp, ...)."""


# ── Base class ───────────────────────────────────────────────────────────────

class SCR4RBase(nn.Module, abc.ABC):
    """Model-agnostic SCR4R training algorithm.

    Subclass checklist:
      1. ``super().__init__(...)``
      2. ``self.mask_net = ...``   (any ``nn.Module`` following the mask-net contract)
      3. Implement all ``@abstractmethod`` hooks
    """

    def __init__(
        self,
        *,
        temporal_history: int = 1,
        smoothness_weight: float = 1e-4,
        smoothness_keys: list[str] | None = None,
        grad_scale: float = 1.0,
        rate_distortion_lambda: float = 1.0,
        selector_mask_mode: str = "noisy_ste",
        selector_gumbel_temperature: float = 1.0,
        selector_gumbel_hard: bool = True,
    ):
        super().__init__()
        selector_mask_mode = str(selector_mask_mode).strip().lower()
        if selector_mask_mode not in ("noisy_ste", "gumbel_softmax"):
            raise ValueError(
                "selector_mask_mode must be one of 'noisy_ste' or "
                f"'gumbel_softmax', got {selector_mask_mode!r}."
            )
        if float(selector_gumbel_temperature) <= 0.0:
            raise ValueError(
                "selector_gumbel_temperature must be > 0, got "
                f"{selector_gumbel_temperature}."
            )
        self.temporal_history = temporal_history
        self.smoothness_weight = smoothness_weight
        self._smoothness_keys = smoothness_keys
        self.grad_scale = grad_scale
        self.rate_distortion_lambda = rate_distortion_lambda
        self.selector_mask_mode = selector_mask_mode
        self.selector_gumbel_temperature = float(selector_gumbel_temperature)
        self.selector_gumbel_hard = bool(selector_gumbel_hard)
        self._prev_cache: dict[str, torch.Tensor] = {}
        self._last_bit_alloc_maps: dict[str, torch.Tensor] = {}
        self._last_cell_drop_ratio_maps: dict[str, torch.Tensor] = {}
        self._last_image_hw: dict[str, tuple[int, int]] = {}

    # ── Abstract: compression ────────────────────────────────────────────

    @abc.abstractmethod
    def _compress(self, image: torch.Tensor) -> CompressionEncodeOutput:
        """Encode ``image [B, C, H, W]`` through the codec."""

    @abc.abstractmethod
    def _decode(
        self,
        masked_latent: torch.Tensor,
        encode_output: CompressionEncodeOutput,
    ) -> torch.Tensor:
        """Decode ``masked_latent [B, C, H, W]`` → reconstructed ``[B, C, H, W]``."""

    @abc.abstractmethod
    def _compute_rate_loss(
        self,
        encode_outputs: dict[str, CompressionEncodeOutput],
        masks: dict[str, torch.Tensor],
    ) -> RateLossOutput:
        """Compute the rate term for all compressed streams."""

    # ── Abstract: VLA ────────────────────────────────────────────────────

    @abc.abstractmethod
    def _compute_vla_loss(
        self,
        recon_images: dict[str, torch.Tensor],
        batch: Any,
    ) -> VLALossOutput:
        """Compute VLA action loss and return **detached** image gradients."""

    # ── Abstract: data / streams ─────────────────────────────────────────

    @abc.abstractmethod
    def _get_images_to_compress(self, batch: Any) -> dict[str, torch.Tensor]:
        """Return ``{image_key: [B, C, H, W] float}`` tensors to compress."""

    @abc.abstractmethod
    def _stream_key(self, image_key: str) -> str:
        """Map a raw image key to a canonical selector stream key."""

    @abc.abstractmethod
    def _get_history_valid(self, batch: Any) -> torch.Tensor | None:
        """Return a ``[B, temporal_history-1]`` bool validity mask, or ``None``."""

    @abc.abstractmethod
    def _mask_net_kind(self) -> str:
        """Return a short string tag for the mask-net type (used in checkpoints)."""

    # ── Overridable: selector mask relaxation ───────────────────────────────

    def _build_selector_mask(
        self,
        mask_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert logits → ``(mask_probs, mask_relaxed)``.

        Default: noisy STE.  Set ``selector_mask_mode="gumbel_softmax"`` to
        sample binary keep/drop masks via a two-class Gumbel-Softmax.
        """
        mask_probs = torch.sigmoid(mask_logits)
        if self.selector_mask_mode == "gumbel_softmax":
            binary_logits = torch.stack(
                [torch.zeros_like(mask_logits), mask_logits],
                dim=-1,
            )
            mask_relaxed = F.gumbel_softmax(
                binary_logits,
                tau=self.selector_gumbel_temperature,
                hard=self.selector_gumbel_hard,
                dim=-1,
            )[..., 1]
            return mask_probs, mask_relaxed

        samples = mask_probs + (torch.rand_like(mask_probs) - 0.5)
        mask_relaxed = samples + samples.round().detach() - samples.detach()
        return mask_probs, mask_relaxed

    # ── Rolling cache ────────────────────────────────────────────────────

    def reset_cache(self) -> None:
        """Clear the rolling feature cache (call on epoch boundary or reshuffle)."""
        self._prev_cache.clear()

    @torch.no_grad()
    def _build_history_from_cache(
        self,
        stream_key: str,
        current_features: torch.Tensor,
        previous_len: int,
    ) -> torch.Tensor:
        """Assemble ``[B, previous_len, F, h, w]`` from the rolling cache.

        Assumes sequential (non-shuffled) data: the previous batch's features
        are temporally adjacent to the current batch's.
        """
        B = current_features.shape[0]
        prev = self._prev_cache.get(stream_key)
        if prev is not None and prev.shape[0] == B:
            combined = torch.cat([prev, current_features], dim=0)
            B_prev = prev.shape[0]
        else:
            combined = current_features
            B_prev = 0

        frames = []
        for offset in range(previous_len, 0, -1):
            indices = torch.arange(B, device=current_features.device) + B_prev - offset
            indices = indices.clamp(0, combined.shape[0] - 1)
            frames.append(combined[indices])
        return torch.stack(frames, dim=1)

    # ── Temporal smoothness regularization ───────────────────────────────

    def _selector_regularization(
        self,
        mask_probs_seq: dict[str, torch.Tensor],
        history_valid_by_key: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``(smoothness_raw, smoothness_weighted)``."""
        if self._smoothness_keys is not None:
            target_keys = [k for k in self._smoothness_keys if k in mask_probs_seq]
            if not target_keys:
                target_keys = sorted(mask_probs_seq.keys())
        else:
            target_keys = sorted(mask_probs_seq.keys())

        terms: list[torch.Tensor] = []
        for key in target_keys:
            probs = mask_probs_seq[key]
            if probs.ndim < 5 or probs.shape[1] < 2:
                continue
            diff = torch.diff(probs, dim=1).abs().mean(
                dim=tuple(range(2, probs.ndim)),
            )
            if history_valid_by_key and key in history_valid_by_key:
                valid = history_valid_by_key[key].to(
                    device=diff.device, dtype=torch.bool,
                )
                if valid.ndim == 1:
                    valid = valid.unsqueeze(0).expand(probs.shape[0], -1)
                pair_valid = valid[:, 1:] & valid[:, :-1]
                if pair_valid.any():
                    terms.append(diff[pair_valid].mean())
            else:
                terms.append(diff.mean())

        if not terms:
            zero = torch.tensor(
                0.0,
                device=next(iter(mask_probs_seq.values())).device,
            )
            return zero, zero
        smoothness = torch.stack(terms).mean()
        return smoothness, smoothness * self.smoothness_weight

    # ── Core training step ───────────────────────────────────────────────

    def compute_scr4r_step(
        self,
        batch: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run one SCR4R forward pass and return ``(loss, metrics)``."""
        if not hasattr(self, "mask_net"):
            raise RuntimeError(
                "Subclass must set self.mask_net in __init__."
            )

        images = self._get_images_to_compress(batch)
        history_valid = self._get_history_valid(batch)

        # ── 1. Encode ────────────────────────────────────────────────────
        encode_outputs: dict[str, CompressionEncodeOutput] = {}
        stream_to_image: dict[str, str] = {}

        feature_inputs: dict[str, torch.Tensor] = {}
        valid_inputs: dict[str, torch.Tensor] = {}

        for key, image in images.items():
            image.requires_grad_(True)
            enc = self._compress(image)
            stream = self._stream_key(key)

            encode_outputs[key] = enc
            stream_to_image[stream] = key

            cur_feat = enc.selector_features.unsqueeze(1)         # [B,1,F,h,w]
            cur_valid = torch.ones(
                cur_feat.shape[0], 1,
                device=cur_feat.device, dtype=torch.bool,
            )

            if self.temporal_history > 1:
                prev_len = self.temporal_history - 1
                if history_valid is None:
                    raise ValueError(
                        "temporal_history > 1 but _get_history_valid returned None."
                    )
                prev_valid = history_valid.to(device=cur_valid.device)
                if prev_valid.ndim == 1:
                    prev_valid = prev_valid.unsqueeze(0).expand(
                        cur_feat.shape[0], -1,
                    )
                prev_feat = self._build_history_from_cache(
                    stream, enc.selector_features, prev_len,
                )
                feat_seq = torch.cat([prev_feat, cur_feat], dim=1)
                feat_valid = torch.cat([prev_valid, cur_valid], dim=1)
                feat_seq = feat_seq * feat_valid.to(
                    dtype=feat_seq.dtype, device=feat_seq.device,
                ).view(*feat_valid.shape, 1, 1, 1)
            else:
                feat_seq = cur_feat
                feat_valid = cur_valid

            feature_inputs[stream] = feat_seq
            valid_inputs[stream] = feat_valid

        # Update rolling cache
        for stream, feat_seq in feature_inputs.items():
            self._prev_cache[stream] = feat_seq[:, -1].detach()

        # ── 2. Mask network ──────────────────────────────────────────────
        ordered = sorted(feature_inputs.keys())
        logits_last, logits_all = self.mask_net(
            feature_inputs,
            ordered,
            history_valid=valid_inputs,
            return_all_steps=True,
        )

        # ── 3. Apply masks + decode ──────────────────────────────────────
        recons: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        probs_seq: dict[str, torch.Tensor] = {}

        for stream in ordered:
            key = stream_to_image[stream]
            enc = encode_outputs[key]

            _, mask_relaxed = self._build_selector_mask(
                logits_last[stream],
            )
            masks[key] = mask_relaxed
            probs_seq[stream] = torch.sigmoid(logits_all[stream])

            recons[key] = self._decode(
                enc.quantized_latent * mask_relaxed, enc,
            )

        # ── 4. VLA loss ──────────────────────────────────────────────────
        vla = self._compute_vla_loss(recons, batch)

        # ── 5. Rate loss ─────────────────────────────────────────────────
        rate = self._compute_rate_loss(encode_outputs, masks)

        # ── 6. Selector regularization ───────────────────────────────────
        smooth_raw, smooth_weighted = self._selector_regularization(
            probs_seq, history_valid_by_key=valid_inputs,
        )

        # ── 7. Combined loss ─────────────────────────────────────────────
        # Distortion = sum(recon * vla_grad)  — gradient-injection trick.
        # Backprop of this term yields d(VLA_loss)/d(codec_params).
        device = rate.loss.device
        distortion = torch.tensor(0.0, device=device)
        for key, recon in recons.items():
            grad = vla.image_grads.get(key)
            if grad is not None:
                distortion = distortion + (
                    recon * grad.detach().to(device=recon.device, dtype=recon.dtype)
                ).sum()

        total_loss = (
            self.rate_distortion_lambda * distortion * self.grad_scale
            + rate.loss
            + smooth_weighted
        )

        mask_mean = (
            torch.stack([m.mean() for m in masks.values()]).mean()
            if masks else torch.tensor(0.0, device=device)
        )

        # Per-stream bit allocation maps for visualization, plus channel-/
        # spatial-level drop ratios. The 2D bit map sums across channels
        # (``sum(dim=1)``), so a spatial cell reads as 0 *only* if every
        # latent channel at that location is masked out. ``mask_mean``
        # alone can rise while the 2D map looks "all dark" purely because
        # the viridis colormap renormalizes min/max — disambiguate via:
        #   channel_drop_ratio       — per-element drop frequency (1 - mean)
        #   cell_drop_ratio          — per-(b,h,w) cell drop fraction
        #                              (avg over channels; continuous)
        #   cell_complete_drop_ratio — fraction of (b,h,w) cells where
        #                              all channels are 0 (= true zero
        #                              cells in the rendered 2D map)
        bit_alloc_maps: dict[str, torch.Tensor] = {}
        cell_drop_ratio_maps: dict[str, torch.Tensor] = {}
        image_hw: dict[str, tuple[int, int]] = {}
        channel_drops: list[torch.Tensor] = []
        cell_drops: list[torch.Tensor] = []
        with torch.no_grad():
            for key, m in masks.items():
                hard = (m > 0.5).to(dtype=torch.float32)
                channel_drops.append((hard == 0).float().mean())
                cell_drop_ratio = (hard == 0).float().mean(dim=1)
                cell_drop_ratio_maps[key] = cell_drop_ratio.detach()
                cell_drops.append((hard.sum(dim=1) == 0).float().mean())
        for key, enc in encode_outputs.items():
            if key in masks:
                image_hw[key] = enc.original_hw
            y_lik = enc.extra.get("latent_likelihoods")
            if y_lik is None or key not in masks:
                continue
            y_safe = torch.clamp(y_lik, min=1e-9)
            m = masks[key].to(dtype=y_safe.dtype, device=y_safe.device)
            bits_per_loc = (-torch.log2(y_safe) * m).sum(dim=1)  # [B, h, w]
            bit_alloc_maps[key] = bits_per_loc.detach()
        self._last_bit_alloc_maps = bit_alloc_maps
        self._last_cell_drop_ratio_maps = cell_drop_ratio_maps
        self._last_image_hw = image_hw

        channel_drop_ratio = (
            torch.stack(channel_drops).mean().item() if channel_drops else 0.0
        )
        cell_drop_ratio = (
            torch.stack(
                [drop_map.mean() for drop_map in cell_drop_ratio_maps.values()]
            ).mean().item()
            if cell_drop_ratio_maps else 0.0
        )
        cell_complete_drop_ratio = (
            torch.stack(cell_drops).mean().item() if cell_drops else 0.0
        )

        metrics: dict[str, float] = {
            "vla_loss": vla.loss_value,
            "distortion": distortion.item(),
            "rate_loss": rate.loss.item(),
            "mask_mean": mask_mean.item(),
            "channel_drop_ratio": channel_drop_ratio,
            "cell_drop_ratio": cell_drop_ratio,
            "cell_complete_drop_ratio": cell_complete_drop_ratio,
            "total_loss": total_loss.item(),
            **rate.metrics,
        }
        return total_loss, metrics

    # ── Checkpoint helpers ───────────────────────────────────────────────

    def checkpoint_state(self) -> dict[str, Any]:
        """Serialisable state for ``torch.save``."""
        return {
            "mask_net_state_dict": self.mask_net.state_dict(),
            "mask_net_kind": self._mask_net_kind(),
            "temporal_history": self.temporal_history,
            "selector_mask_mode": self.selector_mask_mode,
            "selector_gumbel_temperature": self.selector_gumbel_temperature,
            "selector_gumbel_hard": self.selector_gumbel_hard,
            # Persist selector flags whose forward-time behavior cannot be
            # inferred from state_dict shapes alone. disable_stream_embedding
            # zeros out stream_embeddings inside TemporalLatentSelector.forward
            # — eval must reconstruct the selector with the same flag, else
            # the untrained stream_embeddings.weight gets injected at rollout.
            "disable_stream_embedding": bool(
                getattr(self.mask_net, "disable_stream_embedding", False)
            ),
            "cross_camera_attention": bool(
                getattr(self.mask_net, "cross_camera_attention", False)
            ),
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Restore mask-net weights from a saved checkpoint."""
        self.mask_net.load_state_dict(state["mask_net_state_dict"])
