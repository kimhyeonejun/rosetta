"""Test whether loading the LoRA adapter on top of model.safetensors changes
the loss. If loss drops from ~0.7 to ~0.05, then model.safetensors is the
BASE pi05 (not merged) and we've been doing inference without the
fine-tuning weights.

Run:
  python src/rosetta_ic4r/scr4r/debug_pi05_with_lora.py
"""
from __future__ import annotations

import sys
import torch

sys.path.insert(0, "/home1/khj20343/IC4R/src/rosetta_ic4r")

from scr4r.train import _load_config, _move_batch_to_device
from scr4r.data import build_lerobot_dataset, build_dataloader, build_pi05_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.datasets.factory import resolve_delta_timestamps


CKPT_DIR = "/home1/khj20343/IC4R/pretrained_checkpoints/real_world_checkpoints/task_0_checkpoint/pretrained_model"


def _setup_batch():
    cfg = _load_config([
        "--config", "src/rosetta_ic4r/scr4r/configs/train_scr4r_pi05.yaml",
        "--config", "src/rosetta_ic4r/scr4r/configs/train_scr4r_pi05_task0.yaml",
        "training.batch_size=8",
    ])
    device = torch.device("cuda")
    eps = list(cfg.dataset.episodes) if cfg.dataset.episodes else None
    ds = build_lerobot_dataset(cfg.dataset.repo_id, cfg.dataset.root, eps, video_backend="pyav")
    pi05_for_dt = PI05Policy.from_pretrained(CKPT_DIR)
    dt = resolve_delta_timestamps(pi05_for_dt.config, ds.meta)
    del pi05_for_dt
    torch.cuda.empty_cache()
    if dt:
        ds = build_lerobot_dataset(cfg.dataset.repo_id, cfg.dataset.root, eps, video_backend="pyav", delta_timestamps=dt)
    return ds, cfg, device


def _build_pi05(ds, with_lora: bool, device):
    pi05 = PI05Policy.from_pretrained(CKPT_DIR)
    if with_lora:
        from peft import PeftModel
        pi05 = PeftModel.from_pretrained(pi05, CKPT_DIR, is_trainable=False)
    for p in pi05.parameters():
        p.requires_grad_(False)
    pi05.eval().to(device=device, dtype=torch.bfloat16)
    pre, _ = build_pi05_processors(pi05.base_model if with_lora else pi05, ds)
    return pi05, pre


def _run_trials(pi05, pre, ds, device, label, n=5):
    cfg = _load_config([
        "--config", "src/rosetta_ic4r/scr4r/configs/train_scr4r_pi05.yaml",
        "--config", "src/rosetta_ic4r/scr4r/configs/train_scr4r_pi05_task0.yaml",
        "training.batch_size=8",
    ])
    dl = build_dataloader(
        ds, batch_size=8, selector_temporal_history=2,
        selector_history_image_keys=tuple(cfg.profile.compress_streams),
        replan_steps=5, chunk_aligned_history=True,
        shuffle=False, num_workers=0, seed=42, pin_memory=True,
    )
    batch = pre(next(iter(dl)))
    batch = _move_batch_to_device(batch, device)

    losses = []
    for i in range(n):
        torch.manual_seed(100 + i)
        torch.cuda.manual_seed_all(100 + i)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = pi05.forward({k: v for k, v in batch.items()})
        losses.append(float(loss.item()))
    print(f"  {label:30s}  trials={losses}  mean={sum(losses)/n:.3f}")
    return losses


def main():
    print("\n=== loading dataset (with delta timestamps) ===")
    ds, cfg, device = _setup_batch()

    print("\n=== TEST 1: pi05 from_pretrained (model.safetensors only) ===")
    pi05_a, pre_a = _build_pi05(ds, with_lora=False, device=device)
    bare = _run_trials(pi05_a, pre_a, ds, device, "no_adapter")

    del pi05_a, pre_a
    torch.cuda.empty_cache()

    print("\n=== TEST 2: pi05 + PEFT LoRA adapter on top ===")
    try:
        pi05_b, pre_b = _build_pi05(ds, with_lora=True, device=device)
        adapted = _run_trials(pi05_b, pre_b, ds, device, "with_adapter")
    except Exception as exc:
        print(f"  PEFT load failed: {exc}")
        adapted = None

    print("\n=== Verdict ===")
    if adapted is not None:
        delta = (sum(bare) / len(bare)) - (sum(adapted) / len(adapted))
        if delta > 0.3:
            print(f"  Adapter REDUCES loss by {delta:.3f} → model.safetensors is BASE pi05; we must load the adapter at train time.")
        elif abs(delta) < 0.1:
            print(f"  Adapter has no effect (Δ={delta:.3f}) → model.safetensors already merged; LoRA is not the issue.")
        else:
            print(f"  Mixed signal: Δ={delta:.3f}.")


if __name__ == "__main__":
    main()
