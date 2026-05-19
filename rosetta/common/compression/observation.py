"""Walk an observation dict and (de)compress every image-like value in place.

Two robot-wrapper conventions appear in this codebase:

  - LeRobot canonical form (after server-side renaming): keys look like
    ``observation.images.top``, ``observation.images.side``, ...
  - Rosetta wrapper raw form (what arrives at ``pickle.dumps(obs)`` on the
    client): images are keyed by their short camera name — ``top``, ``side``,
    ``wrist`` — because ``RosettaConfig.observation_features`` strips the
    ``observation.images.`` prefix to match LeRobot's robot-features schema.

Both shapes need to compress, so detection falls back to a shape/dtype check
when the key naming does not match. State, action, language, and other
non-image values are passed through untouched.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import ImageCompressor, get
from .payload import CompressedImagePayload

IMAGE_KEY_PREFIX = "observation.images."


def is_image_array(key: str, value: Any) -> bool:
    if not isinstance(value, np.ndarray):
        return False
    if value.ndim != 3:
        return False
    # Fast path: canonical LeRobot key naming.
    if key.startswith(IMAGE_KEY_PREFIX):
        return True
    # Fallback for robot wrappers (e.g. RosettaRobot) that return images keyed
    # by short camera name. Restrict to uint8 (H, W, C) with C in {1, 3, 4} so
    # we don't accidentally swallow a 3D state tensor.
    return value.dtype == np.uint8 and value.shape[-1] in (1, 3, 4)


def compress_observation(
    obs: dict[str, Any], compressor: ImageCompressor
) -> dict[str, Any]:
    """Replace every image array with a CompressedImagePayload."""
    out: dict[str, Any] = {}
    for k, v in obs.items():
        if is_image_array(k, v):
            out[k] = CompressedImagePayload(
                codec=compressor.name,
                shape=tuple(v.shape),
                dtype=str(v.dtype),
                data=compressor.encode(v),
            )
        else:
            out[k] = v
    return out


def decompress_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Restore CompressedImagePayload entries to numpy arrays.

    Each payload carries its own codec name so different keys can use different
    codecs in principle. The receiving side does not need to know which one was
    used in advance.
    """
    out: dict[str, Any] = {}
    for k, v in obs.items():
        if isinstance(v, CompressedImagePayload):
            codec = get(v.codec)
            arr = codec.decode(v.data)
            if arr.shape != v.shape or str(arr.dtype) != v.dtype:
                raise ValueError(
                    f"codec {v.codec!r} round-trip mismatch for {k}: "
                    f"got shape={arr.shape}, dtype={arr.dtype}; "
                    f"expected shape={v.shape}, dtype={v.dtype}"
                )
            out[k] = arr
        else:
            out[k] = v
    return out
