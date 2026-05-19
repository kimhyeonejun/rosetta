"""Image compression layer for observation streams.

Pipeline insertion point:

    [decoders.py: ROS CompressedImage → uint8 numpy]
                          │
                          ▼
    obs dict {observation.images.top: np.uint8 (H,W,3), ...}
                          │
                          ▼          ← compress_observation(obs, codec)
    obs dict with CompressedImagePayload in image slots
                          │
                          ▼
    pickle.dumps  →  gRPC stream  →  pickle.loads
                          │
                          ▼          ← decompress_observation(obs)
    obs dict with reconstructed numpy
                          │
                          ▼
    policy.predict_action_chunk(obs)

Public API:
    register, get, available — codec registry
    compress_observation, decompress_observation — dict walkers
    CompressedImagePayload — picklable sentinel
    ImageCompressor — Protocol for new codecs
"""
from .base import ImageCompressor, available, get, register
from .observation import (
    IMAGE_KEY_PREFIX,
    compress_observation,
    decompress_observation,
    is_image_array,
)
from .payload import CompressedImagePayload

# Side-effect imports register codecs in the global registry.
from . import identity  # noqa: F401  -> registers 'identity'
from . import msillm    # noqa: F401  -> registers 'msillm'

__all__ = [
    "ImageCompressor",
    "CompressedImagePayload",
    "IMAGE_KEY_PREFIX",
    "available",
    "compress_observation",
    "decompress_observation",
    "get",
    "is_image_array",
    "register",
]
