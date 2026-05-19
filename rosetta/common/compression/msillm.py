"""MS-ILLM codec — entropy-coded neural image compression via torch.hub.

Wires Meta's MS-ILLM (Muckley et al., ICML 2023 / Careil et al., ICLR 2024) from
``facebookresearch/NeuralCompression``. Weights are auto-downloaded on first
encode/decode call into torch.hub's cache directory; override the cache path
with ``LEROBOT_MSILLM_HUB_DIR`` so the same checkpoint can be pre-staged on
offline servers.

Key constraints inherited from the upstream model:
  - Inputs are float tensors in [0, 1]; (H, W) is padded to a multiple of the
    model's ``_factor`` (= 64). The upstream ``decompress`` crops back to the
    original (H, W) before returning, so the caller never sees padded pixels.
  - Entropy coding runs on CPU. When ``device != 'cpu'`` we keep the entropy
    bottlenecks on CPU via ``update_tensor_devices('compress')`` and pass
    ``force_cpu=False`` — the analysis/synthesis transforms still run on GPU.
  - Pretrained weights are released under the CC-BY-NC 4.0 license
    (https://github.com/facebookresearch/NeuralCompression/blob/main/WEIGHTS_LICENSE) —
    research/demo use only, not commercial deployment.

Bytestream layout (self-describing — the decoder loads the matching quality
without out-of-band coordination):

    [u8  version = 1]
    [u8  quality]                       MS-ILLM quality (1..6)
    [u16 image_h] [u16 image_w]         original pre-pad size
    [u16 padded_h] [u16 padded_w]       post-pad size (multiples of 64)
    [u32 latent_len]   [latent_bytes...]
    [u32 hyper_len]    [hyper_latent_bytes...]

Configuration (environment variables, snapshotted on first model load):
    LEROBOT_MSILLM_QUALITY   1..6 (default 4 ~ 0.3 bpp). Encoder-side only;
                             the decoder reads quality from the payload.
    LEROBOT_MSILLM_DEVICE    'cuda' | 'cpu' | 'cuda:N' (default cuda if available)
    LEROBOT_MSILLM_HUB_DIR   override torch.hub cache (offline-server friendly)

Runtime overrides are also available via ``MsIllmCompressor.configure()`` for
tests; the registered singleton can be reconfigured before first use.
"""
from __future__ import annotations

import os
import struct
import threading

import numpy as np
import torch

from .base import register

_VERSION = 1
_PAD_FACTOR = 64
# u8 version, u8 quality, u16 image_h, u16 image_w, u16 padded_h, u16 padded_w
_HEADER = struct.Struct("<BBHHHH")
_VALID_QUALITIES = (1, 2, 3, 4, 5, 6)


def _resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _flatten_strings(x) -> bytes:
    """Coerce a compressai strings entry (batch=1) to a single bytes object.

    Bottlenecks return ``List[bytes]`` of length=batch_size; some variants nest
    one extra level. We unwrap defensively up to two levels.
    """
    for _ in range(2):
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        else:
            break
    if not isinstance(x, (bytes, bytearray)):
        raise TypeError(
            f"expected bytes after unwrapping strings list; got {type(x).__name__}"
        )
    return bytes(x)


@register("msillm")
class MsIllmCompressor:
    name = "msillm"

    def __init__(self) -> None:
        self._configured_quality: int | None = None
        self._configured_device: str | None = None
        self._model_cache: dict[tuple[int, str], torch.nn.Module] = {}
        self._cache_lock = threading.Lock()
        # Entropy coding is not thread-safe; serialize all compress/decompress
        # calls per process.
        self._inference_lock = threading.Lock()

    # --- configuration --------------------------------------------------

    @property
    def quality(self) -> int:
        if self._configured_quality is not None:
            return self._configured_quality
        env = os.environ.get("LEROBOT_MSILLM_QUALITY")
        return int(env) if env else 4

    @property
    def device(self) -> str:
        if self._configured_device is not None:
            return self._configured_device
        return _resolve_device(os.environ.get("LEROBOT_MSILLM_DEVICE"))

    def configure(
        self, *, quality: int | None = None, device: str | None = None
    ) -> None:
        """Override env-var defaults at runtime (e.g. inside tests).

        Does not invalidate already-loaded models; call ``reset()`` first if
        switching device after a model has been instantiated.
        """
        if quality is not None:
            if quality not in _VALID_QUALITIES:
                raise ValueError(
                    f"quality must be one of {_VALID_QUALITIES}; got {quality}"
                )
            self._configured_quality = int(quality)
        if device is not None:
            self._configured_device = device

    def reset(self) -> None:
        """Drop cached models so subsequent calls reload on the current device."""
        with self._cache_lock:
            self._model_cache.clear()

    # --- model loading --------------------------------------------------

    def _get_model(self, quality: int) -> torch.nn.Module:
        device = self.device
        key = (quality, device)
        with self._cache_lock:
            cached = self._model_cache.get(key)
            if cached is not None:
                return cached

            hub_dir = os.environ.get("LEROBOT_MSILLM_HUB_DIR")
            if hub_dir:
                torch.hub.set_dir(hub_dir)

            entrypoint = f"msillm_quality_{quality}"
            model = torch.hub.load(
                "facebookresearch/NeuralCompression",
                entrypoint,
                trust_repo=True,
            )
            model.eval()
            model.update()
            if device != "cpu":
                model.to(device)
                # Keep entropy bottlenecks on CPU so compress/decompress can run
                # with force_cpu=False (analysis/synthesis on GPU, coding on CPU).
                model.update_tensor_devices("compress")

            self._model_cache[key] = model
            return model

    # --- encode/decode --------------------------------------------------

    @torch.inference_mode()
    def encode(self, image: np.ndarray) -> bytes:
        if image.ndim != 3:
            raise ValueError(f"expected (H, W, C); got shape {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"expected uint8 image; got dtype {image.dtype}")
        h, w, c = image.shape
        if c != 3:
            raise ValueError(f"expected 3 RGB channels; got C={c}")

        quality = self.quality
        if quality not in _VALID_QUALITIES:
            raise ValueError(
                f"LEROBOT_MSILLM_QUALITY must be in {_VALID_QUALITIES}; got {quality}"
            )

        model = self._get_model(quality)
        device = self.device

        x = (
            torch.from_numpy(image)
            .to(device, non_blocking=True)
            .permute(2, 0, 1)
            .to(torch.float32)
            .div_(255.0)
            .unsqueeze(0)  # (1, 3, H, W)
        )

        with self._inference_lock:
            compressed = model.compress(x, force_cpu=(device == "cpu"))

        latent_bytes = _flatten_strings(compressed.latent_strings)
        hyper_bytes = _flatten_strings(compressed.hyper_latent_strings)
        image_h, image_w = compressed.image_size
        padded_h, padded_w = compressed.padded_size

        header = _HEADER.pack(
            _VERSION, quality, int(image_h), int(image_w), int(padded_h), int(padded_w)
        )
        return b"".join(
            (
                header,
                struct.pack("<I", len(latent_bytes)),
                latent_bytes,
                struct.pack("<I", len(hyper_bytes)),
                hyper_bytes,
            )
        )

    @torch.inference_mode()
    def decode(self, payload: bytes) -> np.ndarray:
        head_size = _HEADER.size
        if len(payload) < head_size:
            raise ValueError(
                f"MS-ILLM payload too short ({len(payload)} < {head_size} bytes)"
            )
        version, quality, image_h, image_w, padded_h, padded_w = _HEADER.unpack(
            payload[:head_size]
        )
        if version != _VERSION:
            raise ValueError(f"unsupported MS-ILLM payload version {version}")
        if quality not in _VALID_QUALITIES:
            raise ValueError(
                f"MS-ILLM payload references unknown quality {quality}; "
                f"expected one of {_VALID_QUALITIES}"
            )

        off = head_size
        if off + 4 > len(payload):
            raise ValueError("truncated MS-ILLM payload (latent length)")
        (latent_len,) = struct.unpack_from("<I", payload, off)
        off += 4
        if off + latent_len > len(payload):
            raise ValueError("truncated MS-ILLM payload (latent body)")
        latent_bytes = bytes(payload[off : off + latent_len])
        off += latent_len

        if off + 4 > len(payload):
            raise ValueError("truncated MS-ILLM payload (hyper length)")
        (hyper_len,) = struct.unpack_from("<I", payload, off)
        off += 4
        if off + hyper_len > len(payload):
            raise ValueError("truncated MS-ILLM payload (hyper body)")
        hyper_bytes = bytes(payload[off : off + hyper_len])
        off += hyper_len
        if off != len(payload):
            raise ValueError(
                f"MS-ILLM payload has {len(payload) - off} trailing bytes"
            )

        # Local import: the type isn't needed unless we actually decode.
        from neuralcompression import HyperpriorCompressedOutput

        compressed = HyperpriorCompressedOutput(
            latent_strings=[latent_bytes],
            hyper_latent_strings=[hyper_bytes],
            image_size=(int(image_h), int(image_w)),
            padded_size=(int(padded_h), int(padded_w)),
        )

        model = self._get_model(quality)
        device = self.device
        with self._inference_lock:
            rec = model.decompress(compressed, force_cpu=(device == "cpu"))

        # rec: (1, 3, image_h, image_w) float in [0, 1] (upstream already cropped + clamped)
        rec = rec.mul(255.0).round_().clamp_(0.0, 255.0).to(torch.uint8)
        return rec.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
