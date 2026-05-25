"""SCR4R concrete implementation: MS-ILLM (hific_scr) + lerobot pi0.5.

Mirrors the structure of ``ic4r/experiments/openpi_scr4r.OpenPISCR4R`` but
swaps the OpenPI/JAX VLA for the upstream lerobot pi0.5 PyTorch port.
Because lerobot pi0.5 is pure PyTorch, the JAX↔PyTorch dlpack bridge that
the upstream class needs (`_compute_vla_loss` doing
``jax_dlpack.from_dlpack``) collapses into a single ``torch.autograd.grad``
call here. Every other piece (selector wiring, ``_compress``/``_decode``,
rate loss, stream alias map, warmup step) follows the upstream verbatim.

Batch contract (post pi05 preprocessor):
    batch[<image_key>]              : (B, C, H, W) float in [0, 1]
    batch[OBS_STATE]                : (B, state_dim) — optional FiLM cond
    batch[OBS_LANGUAGE_TOKENS], ... : language tokens for pi05
    batch[ACTION]                   : action targets
    batch["selector_history"]       : {"image": {key: [B,T-1,C,H,W]},
                                       "valid": [B, T-1] bool} (optional)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import torch
import torch.nn.functional as F

from .codec_models import HiFiCAutoencoder_SCR
from .rate_loss import Rate
from .scr4r_base import (
    CompressionEncodeOutput,
    RateLossOutput,
    SCR4RBase,
    VLALossOutput,
)
from .temporal_selector import TemporalLatentSelector


# ── Per-benchmark profile ───────────────────────────────────────────────────


@dataclass
class LeRobotBenchmarkProfile:
    """Per-dataset knobs for the lerobot pi0.5 SCR4R runtime.

    ``compress_streams`` lists which raw image keys the codec should run on.
    ``canonical_alias_map`` maps raw image keys → canonical selector stream
    keys so multiple cameras can share one selector head if desired.
    """

    name: str
    compress_streams: tuple[str, ...]
    canonical_alias_map: dict[str, str] = field(default_factory=dict)
    default_selector_context_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.canonical_alias_map:
            # Identity alias by default.
            self.canonical_alias_map = {k: k for k in self.compress_streams}
        if not self.default_selector_context_order:
            self.default_selector_context_order = tuple(
                self.canonical_alias_map.get(k, k) for k in self.compress_streams
            )


# ── Core module ─────────────────────────────────────────────────────────────


class Pi05SCR4R(SCR4RBase):
    """SCR4RBase implementation wiring MS-ILLM (codec) and lerobot pi0.5 (VLA).

    The frozen VLA forward and image-gradient extraction are pure PyTorch
    (``torch.autograd.grad``) — no JAX dlpack bridge. Everything else
    (selector head, rate loss, mask STE, smoothness regularization,
    checkpoint payload) is identical to the upstream OpenPI variant.
    """

    def __init__(
        self,
        *,
        msillm_model: HiFiCAutoencoder_SCR,
        pi05_policy,
        msillm_config: dict[str, Any],
        selector_context_order: list[str],
        selector_smoothness_keys: list[str],
        profile: LeRobotBenchmarkProfile,
        state_key: str = "observation.state",
    ):
        if msillm_model is None:
            raise RuntimeError(
                "SCR4R training requires an active MS-ILLM codec, "
                "but `msillm_model` is None."
            )
        if pi05_policy is None:
            raise RuntimeError("SCR4R training requires an active pi05 policy.")

        temporal_history = int(msillm_config.get("selector_temporal_history", 1))
        super().__init__(
            temporal_history=temporal_history,
            smoothness_weight=float(msillm_config.get("selector_smoothness_weight", 1e-4)),
            smoothness_keys=selector_smoothness_keys,
            grad_scale=float(msillm_config.get("grad_scale", 10000.0)),
            rate_distortion_lambda=float(msillm_config.get("rate_distortion_lambda", 1.0)),
            selector_mask_mode=str(msillm_config.get("selector_mask_mode", "noisy_ste")),
            selector_gumbel_temperature=float(msillm_config.get("selector_gumbel_temperature", 1.0)),
            selector_gumbel_hard=bool(msillm_config.get("selector_gumbel_hard", True)),
        )

        # Codec ownership: register as a side attribute (matches upstream
        # OpenPISCR4R which uses object.__setattr__ to avoid double-tracking
        # under Lightning). Here we don't use Lightning but keep the same
        # pattern so the parameter ownership is unambiguous.
        object.__setattr__(self, "_codec", msillm_model)
        # Same for the frozen VLA — params are frozen so nn.Module tracking
        # would just bloat the state_dict.
        object.__setattr__(self, "_pi05", pi05_policy)

        self._config = msillm_config
        self._selector_context_order = list(selector_context_order)
        self._profile = profile
        self._state_key = state_key

        latent_features = int(msillm_config.get("latent_features", 220))
        hyper_features = int(msillm_config.get("hyper_features", 320))

        self.mask_net = TemporalLatentSelector(
            in_channels=hyper_features,
            out_channels=latent_features,
            hidden_dim=int(msillm_config.get("selector_hidden_dim", hyper_features)),
            history_len=temporal_history,
            num_layers=int(msillm_config.get("selector_num_layers", 2)),
            num_heads=int(msillm_config.get("selector_num_heads", 8)),
            mlp_ratio=float(msillm_config.get("selector_mlp_ratio", 4.0)),
            dropout=float(msillm_config.get("selector_dropout", 0.0)),
            stream_names=self._selector_context_order,
            robot_state_dim=int(msillm_config.get("robot_state_dim", 0)),
            disable_stream_embedding=bool(msillm_config.get("disable_stream_embedding", False)),
            cross_camera_attention=bool(msillm_config.get("cross_camera_attention", False)),
        )

        self._rate_fn = Rate(
            weight=float(msillm_config.get("hyper_weight", 1.0)),
            mask_rate_alpha=float(msillm_config.get("mask_rate_alpha", 1.0)),
        )

        if temporal_history <= 1:
            logging.warning(
                "[selector] selector_temporal_history <= 1, so the selector does not use temporal context."
            )

    # ── Stream key map ──────────────────────────────────────────────────

    def _canonical_selector_key(self, key: str) -> str:
        return self._profile.canonical_alias_map.get(key, key)

    # ── Diagnostic: pi05 loss on UNCOMPRESSED images ────────────────────

    @torch.no_grad()
    def compute_vla_loss_uncompressed(self, batch) -> float | None:
        """Run pi05 on raw (uncompressed) images for an apples-to-apples
        comparison against the codec-reconstructed loss.

        Same image normalization as the codec path (`_image_to_01_chw`) so
        the only difference vs `_compute_vla_loss(recons, batch)` is that
        no compress/decode is in the loop. Use to disentangle codec damage
        from pi05's intrinsic flow-matching variance.
        """
        images = self._get_images_to_compress(batch)
        if not images:
            return None
        pi05_batch = dict(batch)
        for k, img in images.items():
            pi05_batch[k] = img  # _get_images_to_compress already maps to [0,1] CHW

        _pi_param = next(self._pi05.parameters())
        _pi_dtype = _pi_param.dtype
        _pi_device_type = _pi_param.device.type
        if _pi_dtype != torch.float32:
            with torch.autocast(device_type=_pi_device_type, dtype=_pi_dtype):
                loss, _ = self._pi05.forward(pi05_batch)
        else:
            loss, _ = self._pi05.forward(pi05_batch)
        if loss.ndim != 0:
            loss = loss.mean()
        return float(loss.detach().item())

    # ── Warmup step (decoder-only, no selector / no rate) ───────────────

    def compute_warmup_step(self, batch):
        """MS-ILLM decoder-only fine-tuning step (no selector, no rate loss).

        Pure first-order surrogate ``Σₖ Σ_pixel recon_k · grad_k`` — the
        decoder is pulled in the ``-grad`` direction (locally reduces VLA
        loss). Image-fidelity drift is intentional and accepted.

        Verbatim port of ``OpenPISCR4R.compute_warmup_step`` modulo the
        VLA loss source (pi05 PyTorch instead of OpenPI JAX).
        """
        images = self._get_images_to_compress(batch)

        recons: dict[str, torch.Tensor] = {}
        for key, image in images.items():
            image.requires_grad_(True)
            enc = self._compress(image)
            recons[key] = self._decode(enc.quantized_latent, enc)

        vla = self._compute_vla_loss(recons, batch)

        if recons:
            device = next(iter(recons.values())).device
        else:
            device = torch.device("cpu")
        distortion = torch.tensor(0.0, device=device)
        for key, recon in recons.items():
            grad = vla.image_grads.get(key)
            if grad is not None:
                distortion = distortion + (
                    recon * grad.detach().to(
                        device=recon.device, dtype=recon.dtype,
                    )
                ).sum()

        total_loss = distortion * self.grad_scale

        metrics = {
            "vla_loss": vla.loss_value,
            "distortion": distortion.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, metrics

    # ── FiLM conditioning + delegation to base ──────────────────────────

    def compute_scr4r_step(self, batch):
        if self.mask_net.use_film:
            robot_state = self._extract_robot_state(batch)
            rsd = int(self.mask_net.film_generators[0][0].weight.shape[1])
            if robot_state.shape[-1] != rsd:
                raise ValueError(
                    f"SCR4R FiLM state dim mismatch: extracted state has "
                    f"{robot_state.shape[-1]} dims but mask_net expects {rsd}. "
                    "Set msillm.robot_state_dim to the state dim used at "
                    "training time."
                )
            self.mask_net.set_conditioning(robot_state=robot_state)
        return super().compute_scr4r_step(batch)

    def _extract_robot_state(self, batch) -> torch.Tensor:
        if self._state_key not in batch:
            raise KeyError(
                f"SCR4R FiLM selector expected `{self._state_key}` in the batch "
                f"(set msillm.robot_state_dim=0 to disable FiLM); "
                f"got keys: {sorted(batch.keys())}"
            )
        state = batch[self._state_key]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state)
        return state.to(dtype=torch.float32)

    # ── Compression: MS-ILLM (hific_scr) ────────────────────────────────

    def _compress(self, image: torch.Tensor) -> CompressionEncodeOutput:
        """Encode ``image [B, 3, H, W]`` in [0, 1] through the SCR-aware codec.

        Verbatim port of ``OpenPISCR4R._compress`` (the OpenPI bit-allocation
        helpers consume the same intermediate tensors).
        """
        _, _, h, w = image.shape
        factor = 64
        if h % factor != 0 or w % factor != 0:
            new_h = ((h + factor - 1) // factor) * factor
            new_w = ((w + factor - 1) // factor) * factor
            x = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
            resize_needed = True
        else:
            x = image
            resize_needed = False

        latent = self._codec.encoder(x)
        hyper_latent = self._codec.hyper_analysis(latent)
        hyper_latent_q, hyper_likelihoods = self._codec.hyper_bottleneck(hyper_latent)
        hyper_likelihoods = torch.clamp(hyper_likelihoods, min=1e-9)

        means = self._codec.hyper_synthesis_mean(hyper_latent_q)
        scales, penultimate = self._codec.hyper_synthesis_scale(
            hyper_latent_q, return_penultimate=True,
        )
        scales = torch.clamp(scales, min=0.11)

        quantized, latent_likelihoods = self._codec.latent_bottleneck(
            latent, scales, means=means,
        )
        latent_likelihoods = torch.clamp(latent_likelihoods, min=1e-9)

        return CompressionEncodeOutput(
            quantized_latent=quantized,
            selector_features=penultimate,
            original_hw=(h, w),
            extra={
                "latent_likelihoods": latent_likelihoods,
                "hyper_likelihoods": hyper_likelihoods,
                "resize_needed": resize_needed,
            },
        )

    def _decode(
        self,
        masked_latent: torch.Tensor,
        encode_output: CompressionEncodeOutput,
    ) -> torch.Tensor:
        """Decode ``[B, C_latent, h, w]`` → reconstructed image in [0, 1].

        Pi0.5's image preprocessor expects ``[0, 1]`` and internally maps to
        ``[-1, 1]``, so unlike the upstream OpenPI variant (which outputs
        ``[-1, 1]``) we keep the reconstruction in ``[0, 1]``.
        """
        recon = self._codec.decoder(masked_latent).clamp(0.0, 1.0)
        h, w = encode_output.original_hw
        if encode_output.extra["resize_needed"]:
            recon = F.interpolate(recon, size=(h, w), mode="bilinear", align_corners=False)
        return recon

    def _compute_rate_loss(
        self,
        encode_outputs: dict[str, CompressionEncodeOutput],
        masks: dict[str, torch.Tensor],
    ) -> RateLossOutput:
        if not encode_outputs:
            return RateLossOutput(loss=torch.zeros((), requires_grad=True), metrics={})

        device = next(iter(encode_outputs.values())).quantized_latent.device
        total_rate = torch.zeros((), device=device)
        bpp_sum = latent_bpp_sum = hyper_bpp_sum = 0.0
        raw_bpp_sum = raw_latent_bpp_sum = 0.0
        count = 0

        for key, enc in encode_outputs.items():
            rate, bpp, latent_bpp, raw_bpp, raw_latent_bpp = self._rate_fn.components(
                enc.extra["latent_likelihoods"],
                enc.extra["hyper_likelihoods"],
                y_mask=masks.get(key),
                num_pixels=enc.original_hw[0] * enc.original_hw[1],
            )
            total_rate = total_rate + rate
            bpp_sum += bpp.item()
            latent_bpp_sum += latent_bpp.item()
            hyper_bpp_sum += (bpp - latent_bpp).item()
            raw_bpp_sum += raw_bpp.item()
            raw_latent_bpp_sum += raw_latent_bpp.item()
            count += 1

        return RateLossOutput(
            loss=total_rate / count,
            metrics={
                "bpp": bpp_sum / count,
                "latent_bpp": latent_bpp_sum / count,
                "hyper_bpp": hyper_bpp_sum / count,
                "raw_bpp": raw_bpp_sum / count,
                "raw_latent_bpp": raw_latent_bpp_sum / count,
            },
        )

    # ── VLA: lerobot pi0.5 (pure PyTorch) ───────────────────────────────

    def _compute_vla_loss(
        self,
        recon_images: dict[str, torch.Tensor],
        batch: Any,
    ) -> VLALossOutput:
        """Compute pi05 action loss + detached image grads.

        The JAX→PyTorch dlpack bridge the upstream OpenPI version uses is
        replaced by a single ``torch.autograd.grad`` call — pi05 is pure
        PyTorch so the action loss is already a torch scalar with a
        complete autograd graph back to the image inputs.
        """
        # Detached, grad-tracking copies of the reconstructions so the VLA
        # forward never back-props into codec parameters (the SCR4R
        # surrogate handles that via Σ recon · g.detach() later).
        detached: dict[str, torch.Tensor] = {}
        for k, v in recon_images.items():
            d = v.detach().clone().requires_grad_(True)
            detached[k] = d

        pi05_batch = dict(batch)
        for k, d in detached.items():
            pi05_batch[k] = d

        # If pi05 is bf16/fp16 (frozen-VLA dtype cast in train.py), pi05's own
        # nn.Linear layers (action_in_proj, time_mlp_*, action_out_proj) reject
        # fp32 inputs at F.linear. We can't just cast `action` to bf16 either:
        # inside model.forward, x_t = time*noise + (1-time)*actions mixes the
        # hardcoded-fp32 `noise`/`time` with `actions`, and PyTorch promotes
        # back to fp32 before action_in_proj sees it. Wrap the whole forward
        # in torch.autocast so every amp-eligible op casts its input to the
        # frozen-VLA dtype automatically; fp32 input images stay fp32 in the
        # autograd graph back to the codec (autocast handles the cross-dtype
        # boundary).
        _pi_param = next(self._pi05.parameters())
        _pi_dtype = _pi_param.dtype
        _pi_device_type = _pi_param.device.type

        if not getattr(self, "_dbg_shape_dumped", False):
            import logging as _logging
            _dbg = _logging.getLogger("scr4r.dbg")
            _dbg.setLevel(_logging.INFO)
            _dbg.warning("[DBG-once] pi05 dtype=%s device=%s", _pi_dtype, _pi_device_type)
            _dbg.warning("[DBG-once] pi05_batch keys before forward: %s", sorted(pi05_batch.keys()))
            for _k, _v in pi05_batch.items():
                if hasattr(_v, "shape"):
                    _dbg.warning("[DBG-once]   %s: shape=%s dtype=%s", _k, tuple(_v.shape), getattr(_v, "dtype", "?"))
                elif isinstance(_v, (list, tuple)):
                    _dbg.warning("[DBG-once]   %s: %s len=%d sample=%r", _k, type(_v).__name__, len(_v), _v[0] if _v else None)
                else:
                    _dbg.warning("[DBG-once]   %s: %s", _k, type(_v).__name__)
            self._dbg_shape_dumped = True

        if _pi_dtype != torch.float32:
            with torch.autocast(device_type=_pi_device_type, dtype=_pi_dtype):
                loss, _loss_dict = self._pi05.forward(pi05_batch)
        else:
            loss, _loss_dict = self._pi05.forward(pi05_batch)
        if loss.ndim != 0:
            loss = loss.mean()

        grads = torch.autograd.grad(
            loss,
            list(detached.values()),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        image_grads: dict[str, torch.Tensor] = {}
        for key, g in zip(detached.keys(), grads, strict=True):
            if g is not None:
                image_grads[key] = g.detach()

        return VLALossOutput(
            loss_value=float(loss.detach().item()),
            image_grads=image_grads,
        )

    # ── Data / streams ──────────────────────────────────────────────────

    def _get_images_to_compress(self, batch: Any) -> dict[str, torch.Tensor]:
        images: dict[str, torch.Tensor] = {}
        for key in self._profile.compress_streams:
            if not self._should_compress_image(key):
                continue
            if key not in batch:
                continue
            v = batch[key]
            if not isinstance(v, torch.Tensor):
                continue
            images[key] = self._image_to_01_chw(v)
        return images

    def _stream_key(self, image_key: str) -> str:
        return self._canonical_selector_key(image_key)

    def _get_history_valid(self, batch: Any) -> torch.Tensor | None:
        selector_history = batch.get("selector_history") if isinstance(batch, dict) else None
        if selector_history is None:
            return None
        valid = selector_history.get("valid")
        if valid is None:
            return None
        if not isinstance(valid, torch.Tensor):
            valid = torch.as_tensor(valid)
        return valid.to(dtype=torch.bool)

    def _mask_net_kind(self) -> str:
        return "temporal_transformer"

    # ── Checkpoint ──────────────────────────────────────────────────────

    def checkpoint_state(self) -> dict[str, Any]:
        state = super().checkpoint_state()
        state["selector_context_order"] = list(self._selector_context_order)
        state["selector_temporal_history"] = self.temporal_history
        state["benchmark"] = self._profile.name
        return state

    # ── Helpers ─────────────────────────────────────────────────────────

    def _should_compress_image(self, image_key: str) -> bool:
        if image_key not in self._profile.compress_streams:
            return False
        selector_key = self._canonical_selector_key(image_key)
        # Per-stream toggles mirror the OpenPI config: compress_image /
        # compress_second_image / compress_wrist_image. If the selector
        # alias map points to one of those canonical names we honor the
        # flag; otherwise default to True.
        if selector_key == "base_0_rgb":
            return bool(self._config.get("compress_image", True))
        if selector_key == "base_1_rgb":
            return bool(self._config.get("compress_second_image", True))
        if selector_key == "wrist_0_rgb":
            return bool(self._config.get("compress_wrist_image", True))
        return True

    @staticmethod
    def _image_to_01_chw(img: torch.Tensor) -> torch.Tensor:
        if img.ndim != 4:
            raise ValueError(f"expected 4D image batch, got shape {tuple(img.shape)}.")
        if img.dtype != torch.float32:
            img = img.to(torch.float32)
        # Accept (B, H, W, C) layout too.
        if img.shape[1] != 3 and img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2).contiguous()
        if img.max().item() > 1.5:
            img = img / 255.0
        return img.clamp(0.0, 1.0)


__all__ = ["LeRobotBenchmarkProfile", "Pi05SCR4R"]
