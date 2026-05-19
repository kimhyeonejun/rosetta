"""No-op codec — verifies the plumbing without depending on a real model.

Encodes shape and dtype as a small header so decode is self-contained.
Useful for unit tests and as a baseline to measure compression ratios against.
"""
from __future__ import annotations

import struct

import numpy as np

from .base import register

_HEADER = struct.Struct("<HHB16s")  # H, W, C, dtype-name (16-byte padded)


@register("identity")
class IdentityCompressor:
    name = "identity"

    def encode(self, image: np.ndarray) -> bytes:
        if image.ndim != 3:
            raise ValueError(f"expected (H, W, C); got shape {image.shape}")
        h, w, c = image.shape
        dt = str(image.dtype).encode().ljust(16, b"\x00")
        return _HEADER.pack(h, w, c, dt) + image.tobytes()

    def decode(self, payload: bytes) -> np.ndarray:
        head_size = _HEADER.size
        h, w, c, dt = _HEADER.unpack(payload[:head_size])
        dtype = dt.rstrip(b"\x00").decode()
        body = payload[head_size:]
        return np.frombuffer(body, dtype=dtype).reshape(h, w, c)
