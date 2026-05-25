"""Check what fraction of state dims fall outside [-1, 1] after the
checkpoint's normalizer. If most dims saturate at -1/255 bins,
the language conditioning loses discriminative power → uniform loss
inflation across the chunk (especially at the early/observation-anchored
steps).
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "/home1/khj20343/IC4R/src/rosetta_ic4r")

from scr4r.train import _load_config, _move_batch_to_device
from scr4r.data import build_lerobot_dataset, build_dataloader, build_pi05_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.datasets.factory import resolve_delta_timestamps


def main():
    cfg = _load_config([
        "--config", "src/rosetta_ic4r/scr4r/configs/train_scr4r_pi05.yaml",
        "--config", "src/rosetta_ic4r/scr4r/configs/train_scr4r_pi05_task0.yaml",
        "training.batch_size=64",
    ])
    eps = list(cfg.dataset.episodes) if cfg.dataset.episodes else None

    ds = build_lerobot_dataset(cfg.dataset.repo_id, cfg.dataset.root, eps, video_backend="pyav")
    pi05 = PI05Policy.from_pretrained(cfg.vla.pretrained)
    dt = resolve_delta_timestamps(pi05.config, ds.meta)
    if dt:
        ds = build_lerobot_dataset(cfg.dataset.repo_id, cfg.dataset.root, eps, video_backend="pyav", delta_timestamps=dt)
    pre, _ = build_pi05_processors(pi05, ds)
    dl = build_dataloader(
        ds, batch_size=64, selector_temporal_history=2,
        selector_history_image_keys=tuple(cfg.profile.compress_streams),
        replan_steps=5, chunk_aligned_history=True,
        shuffle=False, num_workers=0, seed=42, pin_memory=False,
    )

    print("\n=== state distribution after preprocessor normalize (10 batches × 64) ===")
    all_states = []
    it = iter(dl)
    for _ in range(10):
        batch = pre(next(it))
        s = batch["observation.state"].cpu().numpy()  # (B, 7)
        all_states.append(s)
    S = np.concatenate(all_states, axis=0)  # (640, 7)
    print(f"  total samples: {S.shape}")
    print(f"  per-dim min:  {S.min(axis=0)}")
    print(f"  per-dim max:  {S.max(axis=0)}")
    print(f"  per-dim mean: {S.mean(axis=0)}")
    print(f"  per-dim std:  {S.std(axis=0)}")
    in_range = np.abs(S) <= 1.0
    print(f"  in-[-1,1] rate per dim: {in_range.mean(axis=0)}")
    print(f"  OOR rate per dim (|x|>1): {(~in_range).mean(axis=0)}")
    print(f"  OVERALL in-range cells: {in_range.mean():.3f}  → {(~in_range).mean():.3f} OOR")


if __name__ == "__main__":
    main()
