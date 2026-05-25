"""SCR4R training pipeline for the lerobot pi0.5 PyTorch port + MS-ILLM codec.

Self-contained inside rosetta_ic4r — no edits to upstream lerobot or
neuralcompression. Mirrors the canonical SCR4R training pipeline from
``ic4r/experiments/openpi_scr4r.py``, swapping the OpenPI/JAX VLA for
upstream ``lerobot.policies.pi05``.

Public surface:
  * :class:`TemporalLatentSelector` — verbatim mirror of the canonical
    selector head (state_dict round-trips with upstream ckpts).
  * :class:`SCR4RBase`              — model-agnostic SCR4R skeleton.
  * :class:`Pi05SCR4R`              — concrete subclass wiring MS-ILLM
                                     (hific_scr) ↔ lerobot pi0.5.
  * :class:`LeRobotBenchmarkProfile`— per-dataset stream / alias config.
  * :func:`instantiate_msillm_scr_model` — load hific_scr from
                                     torch.hub MS-ILLM or local ckpt.
"""

from .codec_instantiate import instantiate_msillm_scr_model
from .codec_models import HiFiCAutoencoder_SCR
from .pi05_scr4r import LeRobotBenchmarkProfile, Pi05SCR4R
from .rate_loss import Rate
from .scr4r_base import (
    CompressionEncodeOutput,
    RateLossOutput,
    SCR4RBase,
    VLALossOutput,
)
from .stage import (
    SCR4RStageConfig,
    apply_scr4r_stage_requires_grad,
    resolve_scr4r_stage_config,
    scr4r_in_warmup,
)
from .temporal_selector import TemporalLatentSelector

__all__ = [
    "CompressionEncodeOutput",
    "HiFiCAutoencoder_SCR",
    "LeRobotBenchmarkProfile",
    "Pi05SCR4R",
    "Rate",
    "RateLossOutput",
    "SCR4RBase",
    "SCR4RStageConfig",
    "TemporalLatentSelector",
    "VLALossOutput",
    "apply_scr4r_stage_requires_grad",
    "instantiate_msillm_scr_model",
    "resolve_scr4r_stage_config",
    "scr4r_in_warmup",
]
