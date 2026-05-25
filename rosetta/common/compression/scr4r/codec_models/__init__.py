"""SCR-aware MS-ILLM codec architecture (verbatim mirror).

Mirrored from src/compression_models/ms-illm/ inside the IC4R repo so the
rosetta_ic4r SCR4R training is self-contained. Architecture / state_dict
layouts MUST stay identical to the upstream files — SCR4R checkpoints
trained under either copy round-trip through ``msillm_model_state_dict``.
"""
from .hific_with_scr import (
    HiFiCAutoencoder_SCR,
    HyperpriorCompressedOutput,
    HyperpriorOutput,
    pad_image_to_factor,
)

__all__ = [
    "HiFiCAutoencoder_SCR",
    "HyperpriorCompressedOutput",
    "HyperpriorOutput",
    "pad_image_to_factor",
]
