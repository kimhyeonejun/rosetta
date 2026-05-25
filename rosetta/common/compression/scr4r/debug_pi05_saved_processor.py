"""Use the SAVED preprocessor pipeline (the one shipped with the
checkpoint) instead of make_pi05_pre_post_processors (fresh build).

If predictions go from MSE 2.55 → ~0 with the saved processor,
then our build_pi05_processors path is producing subtly different
inputs from what pi05 was trained/deployed with.

Run:
  python src/rosetta_ic4r/scr4r/debug_pi05_saved_processor.py
"""
from __future__ import annotations

import sys
import torch

sys.path.insert(0, "/home1/khj20343/IC4R/src/rosetta_ic4r")

from scr4r.train import _load_config, _move_batch_to_device
from scr4r.data import build_lerobot_dataset, build_dataloader, build_pi05_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.datasets.factory import resolve_delta_timestamps


CKPT = "/home1/khj20343/IC4R/pretrained_checkpoints/real_world_checkpoints/task_0_checkpoint/pretrained_model"


def _build_dataset(cfg):
    eps = list(cfg.dataset.episodes) if cfg.dataset.episodes else None
    ds = build_lerobot_dataset(cfg.dataset.repo_id, cfg.dataset.root, eps, video_backend="pyav")
    pi05_tmp = PI05Policy.from_pretrained(CKPT)
    dt = resolve_delta_timestamps(pi05_tmp.config, ds.meta)
    del pi05_tmp
    torch.cuda.empty_cache()
    if dt:
        ds = build_lerobot_dataset(cfg.dataset.repo_id, cfg.dataset.root, eps, video_backend="pyav", delta_timestamps=dt)
    return ds


def _make_dl(ds, cfg):
    return build_dataloader(
        ds, batch_size=4, selector_temporal_history=2,
        selector_history_image_keys=tuple(cfg.profile.compress_streams),
        replan_steps=5, chunk_aligned_history=True,
        shuffle=False, num_workers=0, seed=42, pin_memory=True,
    )


def _try_load_saved_pipeline():
    """Try a few API entry points to load the saved preprocessor."""
    errors = []
    # 1) PolicyProcessorPipeline.from_pretrained with config_filename
    try:
        from lerobot.processor.pipeline import PolicyProcessorPipeline
        return PolicyProcessorPipeline.from_pretrained(
            CKPT, config_filename="policy_preprocessor.json"
        ), "PolicyProcessorPipeline.from_pretrained(config_filename=...)"
    except Exception as e:
        errors.append(("PolicyProcessorPipeline.from_pretrained", repr(e)))
    try:
        from lerobot.processor import PolicyProcessorPipeline as PP
        return PP.from_pretrained(
            CKPT, config_filename="policy_preprocessor.json"
        ), "lerobot.processor.PolicyProcessorPipeline"
    except Exception as e:
        errors.append(("lerobot.processor.PolicyProcessorPipeline", repr(e)))
    raise RuntimeError("Could not load saved preprocessor:\n" + "\n".join(f"  {n}: {e}" for n, e in errors))


def _run_pred(pi05, batch, device, label):
    gt = batch["action"].clone()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = pi05.predict_action_chunk(batch)
    pred = pred[..., :gt.shape[-1]].to(torch.float32)
    L = min(pred.shape[1], gt.shape[1])
    mse = ((pred[:, :L].cpu() - gt[:, :L].cpu()) ** 2).mean().item()
    print(f"  {label:35s}  MSE={mse:.4f}  pred(mean={pred.mean().item():+.3f},std={pred.std().item():.3f})  gt(mean={gt.mean().item():+.3f},std={gt.std().item():.3f})")
    return mse


def main():
    cfg = _load_config([
        "--config", "src/rosetta_ic4r/scr4r/configs/train_scr4r_pi05.yaml",
        "--config", "src/rosetta_ic4r/scr4r/configs/train_scr4r_pi05_task0.yaml",
        "training.batch_size=4",
    ])
    device = torch.device("cuda")

    ds = _build_dataset(cfg)
    pi05 = PI05Policy.from_pretrained(CKPT)
    for p in pi05.parameters():
        p.requires_grad_(False)
    pi05.eval().to(device=device, dtype=torch.bfloat16)

    raw_batch = next(iter(_make_dl(ds, cfg)))

    print("\n=== prediction MSE ===")

    # Path A: our build_pi05_processors (fresh make_pi05_pre_post_processors)
    pre_a, _ = build_pi05_processors(pi05, ds)
    batch_a = pre_a(raw_batch)
    batch_a = _move_batch_to_device(batch_a, device)
    mse_a = _run_pred(pi05, batch_a, device, "A: fresh make_pi05_pre_post")

    # Path B: SAVED preprocessor from checkpoint
    pre_b, src = _try_load_saved_pipeline()
    print(f"  (loaded saved pre via: {src})")
    batch_b = pre_b(raw_batch)
    batch_b = _move_batch_to_device(batch_b, device)
    mse_b = _run_pred(pi05, batch_b, device, "B: SAVED preprocessor")

    print("\n=== diff ===")
    delta = mse_a - mse_b
    if delta > 1.0:
        print(f"  Path A MSE > Path B MSE by {delta:.2f} → our build_pi05_processors is producing wrong inputs.")
        print(f"  Fix: swap data.build_pi05_processors to load from the checkpoint dir instead.")
    elif abs(delta) < 0.2:
        print(f"  Both paths similar (Δ={delta:+.3f}). Preprocessor isn't the issue.")
    else:
        print(f"  Mixed signal: Δ={delta:+.3f}")


if __name__ == "__main__":
    main()
