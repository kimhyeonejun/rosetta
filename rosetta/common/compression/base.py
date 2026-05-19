"""Codec interface + registry.

A codec turns one numpy image (H, W, C) uint8 into self-describing bytes and
back. The interface is intentionally minimal: each codec encodes whatever
metadata it needs (shape, dtype) into its own bitstream. CompressedImagePayload
also carries shape/dtype so the receiving side can dispatch and sanity-check
without invoking the codec.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ImageCompressor(Protocol):
    name: str

    def encode(self, image: np.ndarray) -> bytes: ...
    def decode(self, payload: bytes) -> np.ndarray: ...


_REGISTRY: dict[str, ImageCompressor] = {}


def register(name: str):
    def deco(cls):
        instance = cls() if isinstance(cls, type) else cls
        if not isinstance(instance, ImageCompressor):
            raise TypeError(f"{cls!r} does not satisfy ImageCompressor protocol")
        _REGISTRY[name] = instance
        return cls
    return deco


def get(name: str) -> ImageCompressor:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown codec '{name}'. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
