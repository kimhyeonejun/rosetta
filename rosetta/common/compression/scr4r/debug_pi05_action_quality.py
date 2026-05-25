"""End-to-end sanity: run pi05 prediction on a real batch and compare
predicted action chunk vs ground-truth action chunk.

- If predicted ≈ ground truth (MSE small) → model weights are loaded
  correctly; the elevated `pi05.forward()` loss must come from t/ε
  averaging in flow-matching loss (which is NOT the same as prediction
  error).
- If predicted is far from ground truth (MSE large) → model isn't doing
  what it was trained to do → load is broken or there's a setup mismatch.

Run:
  python src/rosetta_ic4r/scr4r/debug_pi05_action_quality.py
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


def _layer_norms(pi05):
    """Print L2 norm + std of key action/state projections. Trained
    weights are typically much larger than fresh init (~0.02 std)."""
    print("\n=== Key layer weight statistics ===")
    candidates = [
        "model.paligemma_with_expert.gemma_expert.model.state_proj.weight",
        "model.paligemma_with_expert.gemma_expert.model.action_in_proj.weight",
        "model.paligemma_with_expert.gemma_expert.model.action_out_proj.weight",
        "model.paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight",
        "model.paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight",
    ]
    name_to_param = dict(pi05.named_parameters())
    for name in candidates:
        # Try a few common name variants
        for cand in [name, name.replace("gemma_expert.model.", "gemma_expert."), name.replace(".model.", ".")]:
            if cand in name_to_param:
                w = name_to_param[cand]
                print(f"  {cand}: shape={tuple(w.shape)} mean={w.mean().item():+.4f} std={w.std().item():.4f} ‖w‖={w.norm().item():.2f}")
                break
        else:
            # search by substring
            keyword = name.split(".")[-2]  # e.g. "action_in_proj"
            matches = [(n, p) for n, p in name_to_param.items() if keyword in n][:1]
            if matches:
                n, w = matches[0]
                print(f"  [found via '{keyword}'] {n}: shape={tuple(w.shape)} mean={w.mean().item():+.4f} std={w.std().item():.4f} ‖w‖={w.norm().item():.2f}")
            else:
                print(f"  [missing] {name}")


def main():
    cfg = _load_config([
        "--config", "src/rosetta_ic4r/scr4r/configs/train_scr4r_pi05.yaml",
        "--config", "src/rosetta_ic4r/scr4r/configs/train_scr4r_pi05_task0.yaml",
        "training.batch_size=4",
    ])
    device = torch.device("cuda")
    eps = list(cfg.dataset.episodes) if cfg.dataset.episodes else None

    ds = build_lerobot_dataset(cfg.dataset.repo_id, cfg.dataset.root, eps, video_backend="pyav")
    pi05 = PI05Policy.from_pretrained(CKPT)
    dt = resolve_delta_timestamps(pi05.config, ds.meta)
    if dt:
        ds = build_lerobot_dataset(cfg.dataset.repo_id, cfg.dataset.root, eps, video_backend="pyav", delta_timestamps=dt)
    for p in pi05.parameters():
        p.requires_grad_(False)
    pi05.eval().to(device=device, dtype=torch.bfloat16)
    pre, post = build_pi05_processors(pi05, ds)

    _layer_norms(pi05)

    dl = build_dataloader(
        ds, batch_size=4, selector_temporal_history=2,
        selector_history_image_keys=tuple(cfg.profile.compress_streams),
        replan_steps=5, chunk_aligned_history=True,
        shuffle=False, num_workers=0, seed=42, pin_memory=True,
    )
    batch = pre(next(iter(dl)))
    batch = _move_batch_to_device(batch, device)
    gt_actions = batch["action"].clone()  # (B, 50, 7) normalized
    print(f"\n=== ground-truth action chunk stats ===")
    print(f"  shape={tuple(gt_actions.shape)}  mean={gt_actions.mean().item():+.4f}  std={gt_actions.std().item():.4f}")

    print("\n=== pi05 prediction via predict_action_chunk ===")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = pi05.predict_action_chunk(batch)  # (B, n_action_steps, action_dim)
    pred = pred[..., :gt_actions.shape[-1]].to(torch.float32)
    print(f"  pred shape={tuple(pred.shape)}  mean={pred.mean().item():+.4f}  std={pred.std().item():.4f}")

    # Crop to common chunk length
    L = min(pred.shape[1], gt_actions.shape[1])
    pred_c = pred[:, :L, :].cpu()
    gt_c = gt_actions[:, :L, :].cpu()
    mse = ((pred_c - gt_c) ** 2).mean().item()
    mae = (pred_c - gt_c).abs().mean().item()
    print(f"\n=== prediction vs ground truth ===")
    print(f"  MSE  = {mse:.4f}")
    print(f"  MAE  = {mae:.4f}")
    print(f"  per-chunk-step MSE: {[round(v,3) for v in ((pred_c - gt_c)**2).mean(dim=(0,2)).tolist()]}")

    print("\n=== Verdict ===")
    if mse < 0.05:
        print("  PASS — model predicts actions close to ground truth. Weights load correctly.")
        print("         Elevated pi05.forward() loss is from flow-matching's t/ε averaging,")
        print("         which is NOT the same as prediction error.")
    elif mse < 0.3:
        print("  MID — model has some signal but predictions are noisy.")
    else:
        print("  FAIL — predictions far from ground truth. Weight load IS likely broken,")
        print("         or there's a state/image/normalization mismatch we haven't found.")


if __name__ == "__main__":
    main()
