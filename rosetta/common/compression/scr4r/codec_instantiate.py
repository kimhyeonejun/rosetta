"""MS-ILLM (hific_scr) instantiation + checkpoint loading.

Mirrors the hific_scr-specific paths from ``ic4r/models/compression_models/instantiate.py``
so checkpoints saved by either runtime load into the other. The
checkpoint-key fixups (wrapper-key remap, legacy entropy parameter remap,
serialized-entropy-buffer skip) are critical: SCR4R checkpoints produced
on different compressai versions name the entropy bottleneck buffers
differently (``._matrix0`` vs ``.matrices.0``), and the SCR variant wraps
``hyper_synthesis_{mean,scale}`` in a ``layers`` submodule that older
HiFiC ckpts don't have. Loading without these remaps silently drops 14+
keys → random-init hyper_bottleneck → z_bpp ≈ 0.55 flat (see the upstream
feedback note on env mismatches).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch

from .codec_models import HiFiCAutoencoder_SCR


_ENTROPY_BUFFER_SUFFIXES = (
    "._quantized_cdf",
    "._offset",
    "._cdf_length",
    ".scale_table",
)

_LEGACY_ENTROPY_PARAMETER_STEMS = (
    ("matrix", "matrices"),
    ("bias", "biases"),
    ("factor", "factors"),
)


def _extract_msillm_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in (
            "msillm_model_state_dict",
            "compressor",
            "state_dict",
            "model_state_dict",
        ):
            maybe_state_dict = checkpoint.get(key)
            if isinstance(maybe_state_dict, dict):
                return maybe_state_dict
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported MS-ILLM checkpoint type: {type(checkpoint)!r}")
    return checkpoint


def _normalize_msillm_state_key(key: str) -> str:
    prefixes = ("msillm_model.", "module.", "model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _remap_msillm_wrapper_key(
    key: str,
    model_state_dict: dict[str, Any],
) -> str:
    wrapper_modules = (
        "hyper_synthesis_mean",
        "hyper_synthesis_scale",
    )

    for module_name in wrapper_modules:
        base_prefix = f"{module_name}."
        wrapped_prefix = f"{module_name}.layers."

        # SCR wraps these transforms in a nn.Module with `layers = nn.Sequential(...)`,
        # while older HiFiC checkpoints store them directly as nn.Sequential children.
        if key.startswith(base_prefix) and not key.startswith(wrapped_prefix):
            wrapped_key = key.replace(base_prefix, wrapped_prefix, 1)
            if wrapped_key in model_state_dict:
                return wrapped_key

        if key.startswith(wrapped_prefix):
            base_key = key.replace(wrapped_prefix, base_prefix, 1)
            if base_key in model_state_dict:
                return base_key

    return key


def _remap_msillm_legacy_entropy_key(
    key: str,
    model_state_dict: dict[str, Any],
) -> str:
    """Bridge CompressAI entropy parameter naming differences across versions."""
    for singular, plural in _LEGACY_ENTROPY_PARAMETER_STEMS:
        # OLD singular (``._matrix0``) → NEW plural (``.matrices.0``)
        old_marker = f"._{singular}"
        if old_marker in key:
            prefix, index = key.rsplit(old_marker, 1)
            if prefix and index.isdigit():
                remapped_key = f"{prefix}.{plural}.{index}"
                if remapped_key in model_state_dict:
                    return remapped_key

        # OLD plural-list (``._matrices.0``) → NEW plural (``.matrices.0``)
        old_list_marker = f"._{plural}."
        if old_list_marker in key:
            prefix, index = key.rsplit(old_list_marker, 1)
            if prefix and index.isdigit():
                remapped_key = f"{prefix}.{plural}.{index}"
                if remapped_key in model_state_dict:
                    return remapped_key

        # NEW plural (``.matrices.0``) → OLD singular (``._matrix0``).
        new_marker = f".{plural}."
        if new_marker in key:
            prefix, index = key.rsplit(new_marker, 1)
            if prefix and index.isdigit():
                remapped_key = f"{prefix}._{singular}{index}"
                if remapped_key in model_state_dict:
                    return remapped_key

    return key


def _update_msillm_entropy_tables(
    model: torch.nn.Module,
    *,
    source: str,
    reason: str,
) -> None:
    if not hasattr(model, "update"):
        return

    try:
        model.update(force=True)
        update_call = "model.update(force=True)"
    except TypeError:
        model.update()
        update_call = "model.update()"

    logging.info(
        "[MS-ILLM] Rebuilt entropy coding tables from %s with %s (%s).",
        source,
        update_call,
        reason,
    )


def _load_msillm_weights_into_model(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
    *,
    source: str,
) -> None:
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    skipped_entropy_buffers = []
    skipped_shape_mismatches = []
    skipped_missing_keys = []
    remapped_legacy_entropy_keys = []

    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        new_key = _normalize_msillm_state_key(key)
        new_key = _remap_msillm_wrapper_key(new_key, model_state_dict)
        remapped_key = _remap_msillm_legacy_entropy_key(new_key, model_state_dict)
        was_legacy_entropy_key = remapped_key != new_key
        new_key = remapped_key
        if new_key.endswith(_ENTROPY_BUFFER_SUFFIXES):
            skipped_entropy_buffers.append(new_key)
            continue
        if new_key not in model_state_dict:
            skipped_missing_keys.append(new_key)
            continue

        current_value = model_state_dict[new_key]
        if hasattr(current_value, "shape") and hasattr(value, "shape"):
            if tuple(current_value.shape) != tuple(value.shape):
                skipped_shape_mismatches.append(
                    (new_key, tuple(value.shape), tuple(current_value.shape))
                )
                continue
        filtered_state_dict[new_key] = value
        if was_legacy_entropy_key:
            remapped_legacy_entropy_keys.append((key, new_key))

    incompatible = torch.nn.Module.load_state_dict(
        model, filtered_state_dict, strict=False
    )
    logging.info(
        "[MS-ILLM] Loaded %d tensors from %s into %s",
        len(filtered_state_dict),
        source,
        model.__class__.__name__,
    )

    if skipped_entropy_buffers:
        logging.info(
            "[MS-ILLM] Skipped %d entropy buffer keys from %s; rebuilding with model.update(force=True).",
            len(skipped_entropy_buffers),
            source,
        )
    if remapped_legacy_entropy_keys:
        preview = ", ".join(
            f"{old} -> {new}" for old, new in remapped_legacy_entropy_keys[:5]
        )
        logging.info(
            "[MS-ILLM] Remapped %d legacy entropy parameter keys from %s: %s",
            len(remapped_legacy_entropy_keys),
            source,
            preview,
        )
    if skipped_shape_mismatches:
        preview = ", ".join(
            f"{name}: ckpt={ckpt_shape} model={model_shape}"
            for name, ckpt_shape, model_shape in skipped_shape_mismatches[:5]
        )
        logging.warning(
            "[MS-ILLM] Skipped %d shape-mismatched keys from %s: %s",
            len(skipped_shape_mismatches),
            source,
            preview,
        )
    if skipped_missing_keys:
        preview = ", ".join(skipped_missing_keys[:5])
        logging.warning(
            "[MS-ILLM] Skipped %d unknown keys from %s: %s",
            len(skipped_missing_keys),
            source,
            preview,
        )
    unexpected_missing_keys = [
        key for key in incompatible.missing_keys if key not in skipped_entropy_buffers
    ]
    if unexpected_missing_keys:
        preview = ", ".join(unexpected_missing_keys[:5])
        logging.info(
            "[MS-ILLM] Missing after load from %s (%d): %s",
            source,
            len(unexpected_missing_keys),
            preview,
        )
    if incompatible.unexpected_keys:
        preview = ", ".join(incompatible.unexpected_keys[:5])
        logging.warning(
            "[MS-ILLM] Unexpected after load from %s (%d): %s",
            source,
            len(incompatible.unexpected_keys),
            preview,
        )

    if skipped_entropy_buffers or remapped_legacy_entropy_keys:
        reasons = []
        if skipped_entropy_buffers:
            reasons.append("skipped serialized entropy buffers")
        if remapped_legacy_entropy_keys:
            reasons.append("loaded legacy entropy parameters")
        _update_msillm_entropy_tables(
            model,
            source=source,
            reason=", ".join(reasons),
        )


def _load_msillm_torchhub_state_dict(entrypoint: str) -> dict[str, Any]:
    hub_repo = "facebookresearch/NeuralCompression:main"
    try:
        pretrained_model = torch.hub.load(
            hub_repo, entrypoint, pretrained=True, verbose=False
        )
    except TypeError:
        pretrained_model = torch.hub.load(hub_repo, entrypoint, pretrained=True)
    return pretrained_model.state_dict()


_QUALITY_MAP = {
    "1": "msillm_quality_1",
    "2": "msillm_quality_2",
    "3": "msillm_quality_3",
    "4": "msillm_quality_4",
    "5": "msillm_quality_5",
    "6": "msillm_quality_6",
    "vlo1": "msillm_quality_vlo1",
    "vlo2": "msillm_quality_vlo2",
}


def instantiate_msillm_scr_model(
    *,
    quality: Optional[str] = None,
    pretrained_checkpoint: Optional[Union[str, Path]] = None,
    device: Union[str, int, torch.device] = "cuda",
    in_channels: int = 3,
    latent_features: int = 220,
    hyper_features: int = 320,
    num_residual_blocks: int = 9,
    freeze_encoder: bool = True,
    freeze_bottleneck: bool = True,
    freeze_decoder: bool = False,
) -> HiFiCAutoencoder_SCR:
    """Construct ``HiFiCAutoencoder_SCR`` and optionally load weights.

    Loading priority:
      1. ``pretrained_checkpoint`` if set (local file path).
      2. ``quality`` if set (torch.hub MS-ILLM weights; entrypoint via
         :data:`_QUALITY_MAP`).
      3. Otherwise, returns a fresh model.

    Stays in ``device_setting='forward'`` (all parameters on the same
    device) — SCR4R training never uses the rANS compress/decompress path.
    """
    if isinstance(device, str):
        device_obj = torch.device(device)
    elif isinstance(device, int):
        device_obj = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = device

    model = HiFiCAutoencoder_SCR(
        in_channels=in_channels,
        latent_features=latent_features,
        hyper_features=hyper_features,
        num_residual_blocks=num_residual_blocks,
        freeze_encoder=freeze_encoder,
        freeze_bottleneck=freeze_bottleneck,
        freeze_decoder=freeze_decoder,
    )

    if pretrained_checkpoint is not None:
        ckpt_path = Path(pretrained_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"MS-ILLM checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = _extract_msillm_state_dict(checkpoint)
        _load_msillm_weights_into_model(model, state_dict, source=str(ckpt_path))
    elif quality is not None:
        q = str(quality).strip().lower()
        if q not in _QUALITY_MAP:
            raise ValueError(
                f"Unknown MS-ILLM quality {quality!r}; valid options: {sorted(_QUALITY_MAP)}"
            )
        entrypoint = _QUALITY_MAP[q]
        state_dict = _load_msillm_torchhub_state_dict(entrypoint)
        _load_msillm_weights_into_model(model, state_dict, source=f"torch.hub:{entrypoint}")

    model = model.to(device_obj)
    # SCR4R never calls rANS — keep everything on the model device so
    # autograd can flow through every codec submodule.
    model.update_tensor_devices("forward")
    return model


__all__ = ["instantiate_msillm_scr_model"]
