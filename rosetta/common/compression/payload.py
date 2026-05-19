"""Sentinel object that replaces image numpy arrays in the observation dict.

Picklable; the receiving side detects it by isinstance and dispatches to the
right codec by `codec` name.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompressedImagePayload:
    codec: str
    shape: tuple[int, ...]
    dtype: str
    data: bytes

    def __repr__(self) -> str:
        return (
            f"CompressedImagePayload(codec={self.codec!r}, shape={self.shape}, "
            f"dtype={self.dtype!r}, nbytes={len(self.data)})"
        )
