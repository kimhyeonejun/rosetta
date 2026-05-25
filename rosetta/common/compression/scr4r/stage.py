"""SCR4R 2-stage scheduler (decoder-warmup ↔ SCR4R).

Mirrored from ic4r/experiments/common/scr4r_training.py — only the
functions actually used by SCR4R training (stage resolution + per-stage
trainability policy + optimizer param collection + checkpoint payload +
gumbel temperature scheduler) are kept here; the Lightning-coupled
checkpoint Callback and the wandb visualisation helpers are not mirrored
(this package doesn't use Lightning).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


VALID_TRAINING_STAGES = ("combined", "decoder", "scr4r")
ENCODER_SIDE_MODULES = (
    "encoder",
    "hyper_analysis",
    "hyper_synthesis_mean",
    "hyper_synthesis_scale",
)
BOTTLENECK_MODULES = ("hyper_bottleneck", "latent_bottleneck")


@dataclass(frozen=True)
class SCR4RStageConfig:
    stage: str
    decoder_warmup_steps: int


def resolve_scr4r_stage_config(
    msillm_config: Dict[str, Any],
    *,
    logger: Optional[logging.Logger] = None,
) -> SCR4RStageConfig:
    """Parse and normalize the shared SCR4R stage settings."""

    stage = str(msillm_config.get("training_stage", "combined")).lower()
    if stage not in VALID_TRAINING_STAGES:
        raise ValueError(
            f"msillm.training_stage={stage!r}: must be one of "
            "'combined', 'decoder', 'scr4r'."
        )

    warmup_steps = int(msillm_config.get("decoder_warmup_steps", 0))
    if stage == "scr4r" and warmup_steps > 0:
        log = logger or logging.getLogger(__name__)
        log.warning(
            "[stage] training_stage=scr4r forces decoder_warmup_steps=0 "
            "(was %d) because warmup is disabled in this stage.",
            warmup_steps,
        )
        warmup_steps = 0

    return SCR4RStageConfig(stage=stage, decoder_warmup_steps=warmup_steps)


def scr4r_in_warmup(stage_config: SCR4RStageConfig, step: int) -> bool:
    if stage_config.stage == "decoder":
        return True
    if stage_config.stage == "scr4r":
        return False
    return int(step) < stage_config.decoder_warmup_steps


def should_skip_combined_warmup_checkpoint(stage_config: SCR4RStageConfig, step: int) -> bool:
    return (
        stage_config.stage == "combined"
        and int(step) < stage_config.decoder_warmup_steps
    )


def _clear_msillm_freeze_latches(msillm_model: Any) -> None:
    """Let the SCR4R stage policy own MS-ILLM trainability.

    The MS-ILLM ``freeze_*`` helpers latch internal ``_frozen_*`` flags that
    are re-applied from the model's custom ``train()`` override. SCR4R changes
    trainability by stage, so those latches can otherwise undo the policy set
    below when ``module.train()`` is called.
    """

    for attr in ("_frozen_encoder", "_frozen_bottleneck", "_frozen_decoder"):
        if hasattr(msillm_model, attr):
            setattr(msillm_model, attr, False)


def _set_module_trainability(module: Any, attr: str, trainable: bool) -> None:
    submodule = getattr(module, attr, None)
    if submodule is None:
        return
    if hasattr(submodule, "train"):
        submodule.train(trainable)
    for param in submodule.parameters():
        param.requires_grad_(trainable)


def apply_scr4r_stage_requires_grad(
    *,
    msillm_model: Any,
    mask_net: torch.nn.Module,
    msillm_config: Dict[str, Any],
    warmup: bool,
) -> None:
    """Apply the shared decoder-warmup vs SCR4R trainability policy."""

    freeze_decoder_post_warmup = bool(msillm_config.get("freeze_decoder", False))
    decoder_trainable = warmup or not freeze_decoder_post_warmup

    cfg_freeze_encoder = bool(msillm_config.get("freeze_encoder", True))
    cfg_freeze_bottleneck = bool(msillm_config.get("freeze_bottleneck", True))
    encoder_trainable = (not warmup) and (not cfg_freeze_encoder)
    bottleneck_trainable = (not warmup) and (not cfg_freeze_bottleneck)

    if msillm_model is not None:
        _clear_msillm_freeze_latches(msillm_model)
        _set_module_trainability(msillm_model, "decoder", decoder_trainable)
        for name in ENCODER_SIDE_MODULES:
            _set_module_trainability(msillm_model, name, encoder_trainable)
        for name in BOTTLENECK_MODULES:
            _set_module_trainability(msillm_model, name, bottleneck_trainable)

    mask_net.train(not warmup)
    for param in mask_net.parameters():
        param.requires_grad_(not warmup)


def _extend_module_params(params: List[torch.nn.Parameter], module: Any, attr: str) -> None:
    submodule = getattr(module, attr, None)
    if submodule is None:
        return
    params.extend(list(submodule.parameters()))


def _dedupe_params(params: List[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    seen: set[int] = set()
    out: List[torch.nn.Parameter] = []
    for param in params:
        ident = id(param)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(param)
    return out


def collect_msillm_optimizer_params(
    *,
    msillm_model: Any,
    msillm_config: Dict[str, Any],
    stage_config: SCR4RStageConfig,
) -> List[torch.nn.Parameter]:
    """Collect codec params that may be trainable in any configured stage."""

    params: List[torch.nn.Parameter] = []
    if msillm_model is None:
        return params

    decoder_needed = (
        stage_config.stage == "decoder"
        or stage_config.decoder_warmup_steps > 0
        or not bool(msillm_config.get("freeze_decoder", False))
    )
    if decoder_needed:
        _extend_module_params(params, msillm_model, "decoder")

    if not bool(msillm_config.get("freeze_encoder", True)):
        for name in ENCODER_SIDE_MODULES:
            _extend_module_params(params, msillm_model, name)
        for name in BOTTLENECK_MODULES:
            _extend_module_params(params, msillm_model, name)

    return _dedupe_params(params)


def selector_lr_configured(msillm_config: Dict[str, Any]) -> bool:
    value = msillm_config.get("selector_lr")
    if value is None:
        return False
    return str(value).strip().lower() not in ("", "none", "null")


def build_selector_optimizer_param_group(
    *,
    mask_net: torch.nn.Module,
    msillm_config: Dict[str, Any],
    base_lr: float,
    name: str = "selector",
) -> Optional[Dict[str, Any]]:
    params = list(mask_net.parameters())
    if not params:
        return None
    selector_lr = (
        msillm_config.get("selector_lr")
        if selector_lr_configured(msillm_config)
        else base_lr
    )
    return {
        "params": params,
        "lr": float(selector_lr),
        "name": name,
    }


def build_msillm_selector_optimizer_param_groups(
    *,
    msillm_model: Any,
    mask_net: torch.nn.Module,
    msillm_config: Dict[str, Any],
    stage_config: SCR4RStageConfig,
    base_lr: float,
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []

    codec_params = collect_msillm_optimizer_params(
        msillm_model=msillm_model,
        msillm_config=msillm_config,
        stage_config=stage_config,
    )
    if codec_params:
        groups.append({"params": codec_params, "lr": float(base_lr), "name": "msillm"})

    selector_group = build_selector_optimizer_param_group(
        mask_net=mask_net,
        msillm_config=msillm_config,
        base_lr=base_lr,
    )
    if selector_group is not None:
        groups.append(selector_group)

    return groups


def build_scr4r_checkpoint_payload(
    *,
    msillm_model: torch.nn.Module,
    scr4r,
    step: int,
    training_stage: str,
) -> Dict[str, Any]:
    """Common MS-ILLM + selector checkpoint schema for SCR4R eval adapters."""
    return {
        "step": int(step),
        "msillm_model_state_dict": msillm_model.state_dict(),
        "training_stage": str(training_stage).lower(),
        **scr4r.checkpoint_state(),
    }


class GumbelTemperatureScheduler:
    """Linear anneal of ``scr4r.selector_gumbel_temperature`` from ``tau_start``
    → ``tau_end`` over ``anneal_steps`` training steps.

    This is the plain-Python equivalent of the upstream Lightning Callback —
    just call ``apply(scr4r, step)`` once per training step.
    """

    def __init__(self, tau_start: float, tau_end: float, anneal_steps: int):
        if anneal_steps <= 0:
            raise ValueError(f"anneal_steps must be positive, got {anneal_steps}")
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.anneal_steps = int(anneal_steps)

    def _tau_at(self, step: int) -> float:
        progress = min(1.0, max(0.0, step / self.anneal_steps))
        return self.tau_start + (self.tau_end - self.tau_start) * progress

    def apply(self, scr4r_module, step: int) -> None:
        scr4r_module.selector_gumbel_temperature = self._tau_at(int(step))


def maybe_build_gumbel_temperature_scheduler(
    msillm_config: Dict[str, Any],
    num_train_steps: int,
) -> Optional[GumbelTemperatureScheduler]:
    """Build the Gumbel-tau annealing helper when configured, else None."""
    tau_start_raw = msillm_config.get("selector_gumbel_temperature_start", None)
    if tau_start_raw is None:
        return None
    tau_end = float(msillm_config.get("selector_gumbel_temperature", 1.0))
    anneal_steps_raw = msillm_config.get("selector_gumbel_anneal_steps", None)
    if anneal_steps_raw is None:
        anneal_steps = int(num_train_steps)
    else:
        anneal_steps = int(anneal_steps_raw)
    return GumbelTemperatureScheduler(
        tau_start=float(tau_start_raw),
        tau_end=tau_end,
        anneal_steps=anneal_steps,
    )
