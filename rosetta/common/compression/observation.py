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

import logging
import os
import time
from typing import Any

import numpy as np

from .base import ImageCompressor, get
from .parallel_decode import _decode_one, get_decode_pool
from .payload import CompressedImagePayload

IMAGE_KEY_PREFIX = "observation.images."

_TIMING_LOG = logging.getLogger("rosetta.common.compression.timing")


def _timing_on() -> bool:
    return bool(
        os.environ.get("LEROBOT_CODEC_TIMING") or os.environ.get("LEROBOT_SCR4R_TIMING")
    )


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
    """Replace every image array with a CompressedImagePayload.

    A codec may opt into *observation-level* encoding by implementing
    ``encode_observation(obs) -> dict``. Stateless per-image codecs (msillm,
    identity) don't, so they take the default per-image path below. Codecs
    that need cross-stream context at encode time — e.g. scr4r, whose
    temporal selector is conditioned on the robot state, the per-camera
    stream identity, and a rolling frame history — implement the hook and
    receive the whole observation dict at once.
    """
    t0 = time.perf_counter()
    encode_observation = getattr(compressor, "encode_observation", None)
    if callable(encode_observation):
        out = encode_observation(obs)
    else:
        out = {}
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
    if _timing_on():
        n = sum(1 for v in out.values() if isinstance(v, CompressedImagePayload))
        _TIMING_LOG.warning(
            "[codec-timing] compress codec=%s %d img: %.0f ms",
            getattr(compressor, "name", "?"), n, (time.perf_counter() - t0) * 1000,
        )
    return out


def decompress_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Restore CompressedImagePayload entries to numpy arrays.

    Each payload carries its own codec name so different keys can use different
    codecs in principle. The receiving side does not need to know which one was
    used in advance.

    Symmetric to ``compress_observation``: if the codec that produced the
    payloads provides an observation-level ``decode_observation(obs)`` hook
    (e.g. scr4r, which recomputes its selection mask from the decoded hyper +
    the robot state + per-stream history rather than transmitting it), the
    whole dict is handed to it. Per-image codecs fall through to the loop.
    """
    t0 = time.perf_counter()
    codec_name = "?"
    for v in obs.values():
        if isinstance(v, CompressedImagePayload):
            codec_name = v.codec
            decode_observation = getattr(get(v.codec), "decode_observation", None)
            if callable(decode_observation):
                out = decode_observation(obs)
                if _timing_on():
                    n = sum(1 for x in obs.values() if isinstance(x, CompressedImagePayload))
                    _TIMING_LOG.warning(
                        "[codec-timing] decompress codec=%s %d img: %.0f ms",
                        codec_name, n, (time.perf_counter() - t0) * 1000,
                    )
                return out
            break

    # Decode every image payload. With >1 image and a configured worker pool,
    # decode them in parallel processes — the entropy decoding is CPU-bound and
    # single-threaded, so separate processes are the only way to overlap it.
    image_items = [(k, v) for k, v in obs.items() if isinstance(v, CompressedImagePayload)]
    pool = get_decode_pool()
    if pool is not None and len(image_items) > 1:
        arrs = list(pool.map(_decode_one, [(v.codec, v.data) for _, v in image_items]))
    else:
        arrs = [get(v.codec).decode(v.data) for _, v in image_items]

    decoded_by_key: dict[str, Any] = {}
    for (k, v), arr in zip(image_items, arrs):
        if arr.shape != v.shape or str(arr.dtype) != v.dtype:
            raise ValueError(
                f"codec {v.codec!r} round-trip mismatch for {k}: "
                f"got shape={arr.shape}, dtype={arr.dtype}; "
                f"expected shape={v.shape}, dtype={v.dtype}"
            )
        decoded_by_key[k] = arr

    # Preserve original key order; substitute decoded arrays for payloads.
    out: dict[str, Any] = {
        k: (decoded_by_key[k] if k in decoded_by_key else v) for k, v in obs.items()
    }
    n = len(image_items)
    if _timing_on():
        _TIMING_LOG.warning(
            "[codec-timing] decompress codec=%s %d img: %.0f ms",
            codec_name, n, (time.perf_counter() - t0) * 1000,
        )
    return out
