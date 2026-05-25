"""SCR4R training entrypoint for lerobot pi0.5 + MS-ILLM (hific_scr).

Runs the full SCR4R pipeline as mirrored from
``ic4r/experiments/openpi_scr4r.py`` — temporal selector head, decoder
warmup, gradient-injection distortion, rate-distortion loss, rolling
feature cache, chunk-aligned strided history, checkpoint payload
compatible with the upstream eval adapters.

Run:
    cd rosetta_ic4r
    python -m scr4r.train --config scr4r/configs/train_scr4r_pi05.yaml \\
        codec.quality=vlo2 stage.training=combined ...
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


# Self-contained package: prefer relative imports so PYTHONPATH only needs
# to point at the parent ``rosetta_ic4r`` directory.
try:
    from .codec_instantiate import instantiate_msillm_scr_model
    from .data import (
        build_dataloader,
        build_lerobot_dataset,
        build_pi05_processors,
    )
    from .pi05_scr4r import LeRobotBenchmarkProfile, Pi05SCR4R
    from .stage import (
        apply_scr4r_stage_requires_grad,
        build_msillm_selector_optimizer_param_groups,
        build_scr4r_checkpoint_payload,
        maybe_build_gumbel_temperature_scheduler,
        resolve_scr4r_stage_config,
        scr4r_in_warmup,
        should_skip_combined_warmup_checkpoint,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scr4r.codec_instantiate import instantiate_msillm_scr_model
    from scr4r.data import (
        build_dataloader,
        build_lerobot_dataset,
        build_pi05_processors,
    )
    from scr4r.pi05_scr4r import LeRobotBenchmarkProfile, Pi05SCR4R
    from scr4r.stage import (
        apply_scr4r_stage_requires_grad,
        build_msillm_selector_optimizer_param_groups,
        build_scr4r_checkpoint_payload,
        maybe_build_gumbel_temperature_scheduler,
        resolve_scr4r_stage_config,
        scr4r_in_warmup,
        should_skip_combined_warmup_checkpoint,
    )


_LOG = logging.getLogger("scr4r.train")


# ── Config loading ──────────────────────────────────────────────────────────


def _load_config(argv: list[str] | None = None):
    from omegaconf import OmegaConf

    default_cfg = str(Path(__file__).with_name("configs") / "train_scr4r_pi05.yaml")
    parser = argparse.ArgumentParser(add_help=False)
    # action="append" lets callers stack overlays on top of the base config:
    #   --config base.yaml --config task0_overlay.yaml
    # If --config is omitted, the base file is used.
    parser.add_argument("--config", action="append", default=None)
    parser.add_argument("overrides", nargs="*")
    args, _ = parser.parse_known_args(argv)
    config_paths = args.config if args.config else [default_cfg]
    cfg = OmegaConf.load(config_paths[0])
    for p in config_paths[1:]:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(p))
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.overrides))
    return cfg


# ── Helpers ─────────────────────────────────────────────────────────────────


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            # selector_history.image[...] tensors + valid
            sub: dict[str, Any] = {}
            for sk, sv in v.items():
                if isinstance(sv, torch.Tensor):
                    sub[sk] = sv.to(device, non_blocking=True)
                elif isinstance(sv, dict):
                    sub[sk] = {
                        ssk: (ssv.to(device, non_blocking=True) if isinstance(ssv, torch.Tensor) else ssv)
                        for ssk, ssv in sv.items()
                    }
                else:
                    sub[sk] = sv
            out[k] = sub
        else:
            out[k] = v
    return out


def _resolve_profile(cfg) -> LeRobotBenchmarkProfile:
    streams = tuple(cfg.profile.compress_streams)
    alias = dict(cfg.profile.canonical_alias_map) if cfg.profile.get("canonical_alias_map") else {}
    ctx_order = tuple(cfg.profile.selector_context_order) if cfg.profile.get("selector_context_order") else ()
    return LeRobotBenchmarkProfile(
        name=str(cfg.profile.name),
        compress_streams=streams,
        canonical_alias_map=alias,
        default_selector_context_order=ctx_order,
    )


def _resolve_history_image_keys(profile: LeRobotBenchmarkProfile, msillm_cfg: dict) -> tuple[str, ...]:
    """Return raw image keys whose history should be assembled.

    Honours the upstream per-stream toggles (compress_image /
    compress_second_image / compress_wrist_image) when the canonical
    alias matches; otherwise includes the stream.
    """
    keys: list[str] = []
    for k in profile.compress_streams:
        canon = profile.canonical_alias_map.get(k, k)
        flag_map = {
            "base_0_rgb": "compress_image",
            "base_1_rgb": "compress_second_image",
            "wrist_0_rgb": "compress_wrist_image",
        }
        flag = flag_map.get(canon)
        if flag is not None and not bool(msillm_cfg.get(flag, True)):
            continue
        keys.append(k)
    return tuple(keys)


# ── Logger helpers ──────────────────────────────────────────────────────────


def _resolve_cuda_visible_devices_or_warn():
    """Echo CUDA_VISIBLE_DEVICES so any misrouting shows up in logs early."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    _LOG.info("CUDA_VISIBLE_DEVICES=%s (physical GPU pinning is the caller's responsibility)", cvd)


def _log_codec_param_health(msillm_model: torch.nn.Module) -> None:
    """Detect 'weights never loaded' by checking encoder/decoder weight stats.

    Random-init Conv2d kaiming weights have ‖W‖ near the init scale; a
    pretrained MS-ILLM codec has spent training drifting them. If the
    state_dict load silently dropped the codec weights, every conv here
    will sit at init scale and PSNR below will collapse — but this log
    surfaces the issue before the first forward.
    """
    import torch.nn as nn

    def _block_norm(prefix: str) -> tuple[int, float]:
        total = 0
        sq = 0.0
        for name, m in msillm_model.named_modules():
            if not name.startswith(prefix):
                continue
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                w = m.weight.detach()
                sq += float((w * w).sum().item())
                total += int(w.numel())
        rms = (sq / max(total, 1)) ** 0.5 if total else float("nan")
        return total, rms

    for prefix in ("encoder", "decoder", "hyper_analysis", "hyper_synthesis_mean", "hyper_synthesis_scale"):
        n, rms = _block_norm(prefix)
        _LOG.info("[codec-health] %-25s n_weights=%-9d rms=%.4f", prefix, n, rms)


@torch.no_grad()
def _log_codec_reconstruction_sanity(
    scr4r,
    batch: dict[str, Any],
    device: torch.device,
) -> None:
    """One-shot PSNR check on a real batch through the codec.

    Verdict:
      PSNR ≥ 30 dB  → codec weights loaded fine; high `vla_loss` is
                       pi05's normal flow-matching variance, NOT a load bug.
      PSNR ≤ 15 dB  → codec is producing garbage; suspect dropped keys
                       (grep startup log for 'Skipped … shape-mismatched'
                       or 'unknown keys'), wrong quality, or encoder/decoder
                       frozen-at-init.
    """
    images = scr4r._get_images_to_compress(batch)
    if not images:
        _LOG.warning("[codec-sanity] no images to compress — check profile.compress_streams")
        return
    for key, img in images.items():
        img_dev = img.to(device)
        enc = scr4r._compress(img_dev)
        recon = scr4r._decode(enc.quantized_latent, enc).clamp(0.0, 1.0)
        mse = float(torch.mean((recon - img_dev) ** 2).item())
        psnr = float("inf") if mse == 0 else 10.0 * float(torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item())
        _LOG.info(
            "[codec-sanity] stream=%s shape=%s mse=%.5f psnr=%.2f dB",
            key, tuple(img_dev.shape), mse, psnr,
        )


# ── Train loop ──────────────────────────────────────────────────────────────


def train(cfg) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
    )
    _resolve_cuda_visible_devices_or_warn()
    torch.manual_seed(int(cfg.training.seed))
    device = torch.device(cfg.training.device)

    # ── Dataset + pi05 + processor ──────────────────────────────────────
    _LOG.info("Building dataset %s (root=%s)", cfg.dataset.repo_id, cfg.dataset.root)
    dataset = build_lerobot_dataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=list(cfg.dataset.episodes) if cfg.dataset.episodes else None,
        video_backend=cfg.dataset.get("video_backend"),
    )

    _LOG.info("Loading frozen pi05 VLA from %s", cfg.vla.pretrained)
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    pi05 = PI05Policy.from_pretrained(cfg.vla.pretrained)
    for p in pi05.parameters():
        p.requires_grad_(False)
    pi05.eval()
    # Frozen-VLA dtype cast — default bf16 (see cfg.vla.dtype). pi05's
    # forward auto-casts the embedded prefix/suffix to bf16 when the first
    # layer's q_proj is bf16, so no further plumbing is needed.
    vla_dtype_name = str(cfg.vla.get("dtype", "float32")).lower()
    _DTYPE_MAP = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    if vla_dtype_name not in _DTYPE_MAP:
        raise ValueError(
            f"cfg.vla.dtype={cfg.vla.get('dtype')!r} not in {sorted(_DTYPE_MAP)}"
        )
    vla_dtype = _DTYPE_MAP[vla_dtype_name]
    _LOG.info("Casting frozen pi05 to %s", vla_dtype)
    pi05.to(device=device, dtype=vla_dtype)

    # pi05 expects an action *chunk* of length config.chunk_size, not a single
    # timestep. Without delta_timestamps the dataset returns shape (B, D), and
    # model.forward then broadcasts (B,1,1)*(B,D) → (B,B,D), shrinking the
    # suffix sequence to B and producing pad/att-mask length mismatches. Use
    # lerobot's helper to pull `action_delta_indices` off the policy config.
    from lerobot.datasets.factory import resolve_delta_timestamps as _resolve_delta_timestamps
    delta_timestamps = _resolve_delta_timestamps(pi05.config, dataset.meta)
    if delta_timestamps:
        _LOG.info(
            "Rebuilding dataset with delta_timestamps (chunk_size=%d, fps=%s, keys=%s)",
            int(pi05.config.chunk_size), dataset.meta.fps, sorted(delta_timestamps.keys()),
        )
        dataset = build_lerobot_dataset(
            repo_id=cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=list(cfg.dataset.episodes) if cfg.dataset.episodes else None,
            video_backend=cfg.dataset.get("video_backend"),
            delta_timestamps=delta_timestamps,
        )

    _LOG.info("Building pi05 pre/post processors (prefer saved pipelines from ckpt)")
    preprocessor, _postprocessor = build_pi05_processors(
        pi05, dataset, checkpoint_path=cfg.vla.pretrained,
    )

    # ── MS-ILLM (hific_scr) ─────────────────────────────────────────────
    msillm_cfg = dict(cfg.msillm)
    _LOG.info(
        "Instantiating hific_scr codec quality=%s pretrained_checkpoint=%s",
        msillm_cfg.get("quality"), msillm_cfg.get("pretrained_checkpoint"),
    )
    msillm_model = instantiate_msillm_scr_model(
        quality=msillm_cfg.get("quality"),
        pretrained_checkpoint=msillm_cfg.get("pretrained_checkpoint"),
        device=device,
        latent_features=int(msillm_cfg.get("latent_features", 220)),
        hyper_features=int(msillm_cfg.get("hyper_features", 320)),
        freeze_encoder=bool(msillm_cfg.get("freeze_encoder", True)),
        freeze_bottleneck=bool(msillm_cfg.get("freeze_bottleneck", True)),
        freeze_decoder=bool(msillm_cfg.get("freeze_decoder", False)),
    )

    # ── Profile + Pi05SCR4R ─────────────────────────────────────────────
    profile = _resolve_profile(cfg)
    ctx_order = [str(k) for k in (
        msillm_cfg.get("selector_context_order")
        or list(profile.default_selector_context_order)
    )]
    smooth_keys = [str(k) for k in (
        msillm_cfg.get("selector_smoothness_keys")
        or list(profile.default_selector_context_order)
    )]

    scr4r = Pi05SCR4R(
        msillm_model=msillm_model,
        pi05_policy=pi05,
        msillm_config=msillm_cfg,
        selector_context_order=ctx_order,
        selector_smoothness_keys=smooth_keys,
        profile=profile,
        state_key=str(cfg.vla.state_key),
    ).to(device)

    # Warm-start selector from a SCR4R-format ckpt if requested.
    pretrained_scr4r_ckpt = msillm_cfg.get("pretrained_checkpoint")
    if pretrained_scr4r_ckpt:
        try:
            scr_state = torch.load(pretrained_scr4r_ckpt, map_location="cpu", weights_only=False)
            if isinstance(scr_state, dict) and "mask_net_state_dict" in scr_state:
                scr4r.load_checkpoint_state(scr_state)
                _LOG.info(
                    "[scr4r] warm-started mask_net from %s (step=%s)",
                    pretrained_scr4r_ckpt, scr_state.get("step", "?"),
                )
        except (FileNotFoundError, KeyError, RuntimeError) as exc:
            _LOG.warning("[scr4r] mask_net warm-start skipped (%s)", exc)

    # ── Stage config + initial param-grad ───────────────────────────────
    stage_config = resolve_scr4r_stage_config(msillm_cfg, logger=_LOG)
    training_stage = stage_config.stage
    _LOG.info(
        "SCR4R stage=%s decoder_warmup_steps=%d",
        training_stage, stage_config.decoder_warmup_steps,
    )
    apply_scr4r_stage_requires_grad(
        msillm_model=msillm_model,
        mask_net=scr4r.mask_net,
        msillm_config=msillm_cfg,
        warmup=scr4r_in_warmup(stage_config, 0),
    )

    # ── DataLoader (with chunk-aligned history when configured) ─────────
    history_keys = _resolve_history_image_keys(profile, msillm_cfg)
    _LOG.info(
        "Building DataLoader: temporal_history=%d replan_steps=%d chunk_aligned=%s",
        int(msillm_cfg.get("selector_temporal_history", 1)),
        int(msillm_cfg.get("replan_steps", 1)),
        bool(msillm_cfg.get("chunk_aligned_history", False)),
    )

    def _on_epoch_start(epoch: int) -> None:
        # Rolling feature cache reflects positions from the previous batch;
        # at epoch boundaries (and especially under chunk-aligned offset
        # rotation) those positions don't align with the new pass — clear it.
        scr4r.reset_cache()

    dataloader = build_dataloader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        selector_temporal_history=int(msillm_cfg.get("selector_temporal_history", 1)),
        selector_history_image_keys=history_keys,
        replan_steps=int(msillm_cfg.get("replan_steps", 1)),
        chunk_aligned_history=bool(msillm_cfg.get("chunk_aligned_history", False)),
        shuffle=bool(cfg.training.shuffle),
        num_workers=int(cfg.training.num_workers),
        seed=int(cfg.training.seed),
        pin_memory=device.type == "cuda",
        on_epoch_start=_on_epoch_start,
    )

    # ── Optimizer ───────────────────────────────────────────────────────
    base_lr = float(msillm_cfg.get("lr", 1e-4))
    param_groups = build_msillm_selector_optimizer_param_groups(
        msillm_model=msillm_model,
        mask_net=scr4r.mask_net,
        msillm_config=msillm_cfg,
        stage_config=stage_config,
        base_lr=base_lr,
    )
    if not param_groups:
        raise RuntimeError(
            "No trainable parameters resolved: check msillm.freeze_decoder, "
            "msillm.freeze_encoder, and msillm.freeze_bottleneck."
        )
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        weight_decay=float(cfg.training.weight_decay),
    )
    _LOG.info(
        "Optimizer param groups: %s",
        [{"name": g.get("name"), "lr": g.get("lr"), "n_params": sum(p.numel() for p in g["params"])} for g in param_groups],
    )

    # Gumbel-tau annealing
    gumbel_sched = maybe_build_gumbel_temperature_scheduler(
        msillm_cfg, num_train_steps=int(cfg.training.steps),
    )

    # ── Checkpoint dir + wandb ─────────────────────────────────────────
    ckpt_dir = Path(str(msillm_cfg.get("checkpoint_dir", cfg.training.output_dir)))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = bool(cfg.logging.wandb_enable)
    wandb = None
    if use_wandb:
        import wandb as _wandb  # type: ignore
        from omegaconf import OmegaConf

        wandb = _wandb
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.get("wandb_entity"),
            name=cfg.logging.wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        _LOG.info(
            "[wandb] init project=%s entity=%s run=%s",
            cfg.logging.wandb_project,
            cfg.logging.get("wandb_entity"),
            wandb.run.name if wandb.run is not None else None,
        )

    # ── One-shot codec health diagnostics ───────────────────────────────
    # Answers "did the MS-ILLM weights actually load?" before the first
    # train step — quality=6 should give PSNR ≥ 30 dB on Pyper RGB; <15 dB
    # means weights silently dropped (check the [MS-ILLM] startup lines).
    _log_codec_param_health(msillm_model)
    try:
        _sanity_iter = iter(dataloader)
        _sanity_batch = preprocessor(next(_sanity_iter))
        _sanity_batch = _move_batch_to_device(_sanity_batch, device)
        _log_codec_reconstruction_sanity(scr4r, _sanity_batch, device)
        del _sanity_iter, _sanity_batch
    except (StopIteration, RuntimeError) as _exc:
        _LOG.warning("[codec-sanity] skipped (%s)", _exc)

    # ── Loop ───────────────────────────────────────────────────────────
    step = 0
    t0 = time.time()
    grad_clip = float(msillm_cfg.get("grad_clip_norm", cfg.training.get("grad_clip", 1.0)))
    log_every = int(cfg.logging.log_every)
    save_interval = int(msillm_cfg.get("save_interval", 5000))
    save_top_k = int(msillm_cfg.get("save_top_k", 3))
    saved_checkpoints: list[Path] = []

    def _save_checkpoint(step_now: int) -> None:
        if should_skip_combined_warmup_checkpoint(stage_config, step_now):
            return
        path = ckpt_dir / f"msillm_scr_q{msillm_cfg.get('quality', 'scr')}_step_{step_now}.pt"
        payload = build_scr4r_checkpoint_payload(
            msillm_model=msillm_model,
            scr4r=scr4r,
            step=step_now,
            training_stage=training_stage,
        )
        payload["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(payload, path)
        _LOG.info("[ckpt] saved %s", path)
        saved_checkpoints.append(path)
        while len(saved_checkpoints) > save_top_k:
            oldest = saved_checkpoints.pop(0)
            if oldest.exists():
                oldest.unlink()
                _LOG.info("[ckpt] pruned %s", oldest)

    max_steps = int(cfg.training.steps)
    data_iter = iter(dataloader)

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # The lerobot saved preprocessor only knows policy features
        # (observation.state/images, action, task) and silently drops any
        # extra keys. SCR4R's `selector_history` (added by build_dataloader)
        # is one of those — losing it makes `_get_history_valid` return None
        # and crashes compute_scr4r_step at the warmup→SCR4R transition.
        # Detach + reattach so the field survives the pipeline.
        _selector_history = batch.pop("selector_history", None) if isinstance(batch, dict) else None
        batch = preprocessor(batch)
        if _selector_history is not None and isinstance(batch, dict):
            batch["selector_history"] = _selector_history
        batch = _move_batch_to_device(batch, device)

        in_warmup = scr4r_in_warmup(stage_config, step)
        apply_scr4r_stage_requires_grad(
            msillm_model=msillm_model,
            mask_net=scr4r.mask_net,
            msillm_config=msillm_cfg,
            warmup=in_warmup,
        )
        if gumbel_sched is not None:
            gumbel_sched.apply(scr4r, step)

        if in_warmup:
            loss, metrics = scr4r.compute_warmup_step(batch)
        else:
            loss, metrics = scr4r.compute_scr4r_step(batch)

        if loss.requires_grad:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g["params"]],
                    grad_clip,
                )
            optimizer.step()

        if step % log_every == 0:
            dt = time.time() - t0
            # Apples-to-apples baseline: pi05 loss on UNCOMPRESSED images
            # for the same batch. Δ = vla(recon) - vla(uncompressed) is the
            # codec's direct impact on the policy (subject to per-batch t/ε
            # variance, which both calls share approximately).
            try:
                vla_uncompressed = scr4r.compute_vla_loss_uncompressed(batch)
            except (RuntimeError, AttributeError) as _exc:
                _LOG.warning("[vla-uncomp] skipped (%s)", _exc)
                vla_uncompressed = None
            if vla_uncompressed is not None:
                metrics["vla_loss_uncompressed"] = vla_uncompressed
                metrics["vla_loss_delta"] = (
                    float(metrics.get("vla_loss", 0.0)) - vla_uncompressed
                )

            _LOG.info(
                "step %d/%d | stage=%s | total=%.4f | vla=%.4f | vla_uncomp=%.4f | bpp=%.4f | latent_bpp=%.4f | mask_mean=%.3f | dt=%.1fs",
                step, max_steps,
                "warmup" if in_warmup else "scr4r",
                metrics.get("total_loss", 0.0),
                metrics.get("vla_loss", 0.0),
                metrics.get("vla_loss_uncompressed", float("nan")),
                metrics.get("bpp", 0.0),
                metrics.get("latent_bpp", 0.0),
                metrics.get("mask_mean", 0.0),
                dt,
            )
            if wandb is not None:
                wandb_payload = {f"train/{k}": v for k, v in metrics.items()}
                wandb_payload["train/step"] = step
                wandb_payload["train/warmup"] = 1.0 if in_warmup else 0.0
                wandb_payload["train/gumbel_tau"] = float(scr4r.selector_gumbel_temperature)
                wandb.log(wandb_payload, step=step)

        if save_interval > 0 and step > 0 and step % save_interval == 0:
            _save_checkpoint(step)

        step += 1

    _save_checkpoint(step)
    if wandb is not None:
        wandb.finish()


def main() -> None:
    cfg = _load_config()
    train(cfg)


if __name__ == "__main__":
    main()
