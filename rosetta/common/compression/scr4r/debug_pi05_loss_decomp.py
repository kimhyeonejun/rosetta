"""Decompose pi05 loss to find why vla_uncompressed sits at ~0.5 instead
of ~0.05 (pi05's expected converged loss).

Tests:
  A) action_is_pad rate — if many steps are padded, padded-step MSE may
     dominate the loss (pi05.forward doesn't mask).
  B) bf16 vs fp32 — bf16 numerical drift in flow-matching loss.
  C) per-chunk-step loss distribution — see if any specific positions
     in the action chunk dominate.

Run inside the srun shell:
  cd /home1/khj20343/IC4R
  python src/rosetta_ic4r/scr4r/debug_pi05_loss_decomp.py
"""
from __future__ import annotations

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home1/khj20343/IC4R/src/rosetta_ic4r")

from scr4r.train import _load_config, _move_batch_to_device, _resolve_profile
from scr4r.codec_instantiate import instantiate_msillm_scr_model
from scr4r.data import build_lerobot_dataset, build_dataloader, build_pi05_processors
from scr4r.pi05_scr4r import Pi05SCR4R
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.datasets.factory import resolve_delta_timestamps


def _setup(dtype: torch.dtype):
    cfg = _load_config(
        [
            "--config", "src/rosetta_ic4r/scr4r/configs/train_scr4r_pi05.yaml",
            "--config", "src/rosetta_ic4r/scr4r/configs/train_scr4r_pi05_task0.yaml",
            "msillm.quality=6", "training.batch_size=8",
        ]
    )
    device = torch.device("cuda")
    eps = list(cfg.dataset.episodes) if cfg.dataset.episodes else None

    ds = build_lerobot_dataset(cfg.dataset.repo_id, cfg.dataset.root, eps, video_backend="pyav")
    pi05 = PI05Policy.from_pretrained(cfg.vla.pretrained)
    for p in pi05.parameters():
        p.requires_grad_(False)
    pi05.eval().to(device=device, dtype=dtype)

    dt = resolve_delta_timestamps(pi05.config, ds.meta)
    if dt:
        ds = build_lerobot_dataset(cfg.dataset.repo_id, cfg.dataset.root, eps, video_backend="pyav", delta_timestamps=dt)
    pre, _ = build_pi05_processors(pi05, ds)
    dl = build_dataloader(
        ds, batch_size=8, selector_temporal_history=2,
        selector_history_image_keys=tuple(cfg.profile.compress_streams),
        replan_steps=5, chunk_aligned_history=True,
        shuffle=False, num_workers=0, seed=42, pin_memory=True,
    )
    batch = pre(next(iter(dl)))
    batch = _move_batch_to_device(batch, device)
    return pi05, batch, cfg, device


def _per_chunkstep_loss(pi05, batch, dtype):
    """Get per-(B, chunk_step, action_dim) loss BEFORE reduction.

    Uses pi05's own preprocessors (action padding from 7→32, image
    masks, etc.) so we match the top-level pi05.forward path exactly.
    """
    images, img_masks = pi05._preprocess_images(batch)
    tokens = batch["observation.language.tokens"]
    masks = batch["observation.language.attention_mask"]
    actions = pi05.prepare_action(batch)  # pads 7 → 32

    if dtype != torch.float32:
        with torch.autocast(device_type="cuda", dtype=dtype):
            per = pi05.model.forward(images, img_masks, tokens, masks, actions)
    else:
        per = pi05.model.forward(images, img_masks, tokens, masks, actions)

    orig_dim = pi05.config.output_features["action"].shape[0]
    return per[:, :, :orig_dim]  # (B, chunk, 7)


def main():
    print("\n=== A) action_is_pad rate on first batch ===")
    pi05_bf, batch_bf, _, device = _setup(torch.bfloat16)
    pad = batch_bf.get("action_is_pad")
    if pad is not None:
        total = pad.numel()
        n_pad = int(pad.sum().item())
        print(f"  shape={tuple(pad.shape)} padded={n_pad}/{total} = {100*n_pad/total:.1f}%")
        if pad.ndim == 2:
            per_step = pad.float().mean(dim=0)
            print(f"  pad rate per chunk step (B-avg):\n  {[float(v) for v in per_step.tolist()]}")
    else:
        print("  no action_is_pad field")

    print("\n=== B) per-chunkstep loss (bf16) — first batch ===")
    per_bf = _per_chunkstep_loss(pi05_bf, batch_bf, torch.bfloat16)
    print(f"  shape={tuple(per_bf.shape)}")
    print(f"  mean overall:              {per_bf.float().mean().item():.4f}")
    if pad is not None:
        not_pad = (~pad).unsqueeze(-1).float()
        masked_mean = (per_bf.float() * not_pad).sum() / not_pad.sum().clamp_min(1).item() / per_bf.shape[-1]
        print(f"  mean over NON-padded steps: {masked_mean.item():.4f}  ← if much lower, padding inflates loss")
    chunk_means = per_bf.float().mean(dim=(0, 2))
    print(f"  per-chunk-step mean (50 vals): {[round(v,3) for v in chunk_means.tolist()]}")

    print("\n=== C) bf16 vs fp32 — same (cached) batch, 5 trials each ===")
    print(f"  {'trial':>5} {'bf16':>8} {'fp32':>8}")
    # bf16 trials — pi05_bf is in bf16, forward needs autocast wrapper to
    # cast inputs from fp32 down. Same pattern as pi05_scr4r._compute_vla_loss.
    bf_losses = []
    for i in range(5):
        torch.manual_seed(100 + i)  # same t/ε per trial across dtypes → apples vs apples
        torch.cuda.manual_seed_all(100 + i)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss_bf, _ = pi05_bf.forward({k: v for k, v in batch_bf.items()})
        bf_losses.append(float(loss_bf.item()))
    # Free bf16, load fp32 (~28 GB) — reuse SAME batch (no torchcodec).
    del pi05_bf
    torch.cuda.empty_cache()
    pi05_fp = PI05Policy.from_pretrained(
        "/home1/khj20343/IC4R/pretrained_checkpoints/real_world_checkpoints/task_0_checkpoint/pretrained_model"
    )
    for p in pi05_fp.parameters():
        p.requires_grad_(False)
    pi05_fp.eval().to(device=device, dtype=torch.float32)
    fp_losses = []
    for i in range(5):
        torch.manual_seed(100 + i)
        torch.cuda.manual_seed_all(100 + i)
        loss_fp, _ = pi05_fp.forward({k: v for k, v in batch_bf.items()})
        fp_losses.append(float(loss_fp.item()))
    for i in range(5):
        delta = bf_losses[i] - fp_losses[i]
        print(f"  {i:>5d} {bf_losses[i]:>8.3f} {fp_losses[i]:>8.3f}  Δ={delta:+.3f}")
    print(f"  mean  {sum(bf_losses)/5:>8.3f} {sum(fp_losses)/5:>8.3f}")


if __name__ == "__main__":
    main()
