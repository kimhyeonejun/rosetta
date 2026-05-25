"""Compare the normalizer stats saved with the pi05 checkpoint vs the
stats that `build_pi05_processors` would compute fresh from the dataset.

If they disagree → that's almost certainly why vla(uncompressed) is sitting
at 0.3-1.5 instead of 0.05-0.15 (pi05's actual converged loss).

Run with:
  cd /home1/khj20343/IC4R
  conda activate robocasa
  ulimit -v unlimited
  export HF_TOKEN="$(cat /home1/khj20343/.cache/huggingface/token)"
  export PYTHONPATH=/home1/khj20343/IC4R/src/rosetta_ic4r
  python src/rosetta_ic4r/scr4r/debug_normalizer_stats_diff.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, "/home1/khj20343/IC4R/src/rosetta_ic4r")

from scr4r.data import build_lerobot_dataset


CKPT = Path(
    "/home1/khj20343/IC4R/pretrained_checkpoints/real_world_checkpoints/task_0_checkpoint/pretrained_model"
)
NORM_FILE = CKPT / "policy_preprocessor_step_2_normalizer_processor.safetensors"
EPISODES = list(range(62))


def _dataset_stats():
    ds = build_lerobot_dataset(
        repo_id="pyper/effl_dataset",
        root="/home/khj20343/IC4R/data/lerobot/pyper/effl_dataset",
        episodes=EPISODES,
    )
    return ds.meta.stats


def _ckpt_stats():
    return load_file(str(NORM_FILE))


def _summarize(t):
    if not isinstance(t, torch.Tensor):
        return f"<{type(t).__name__}>"
    return f"shape={tuple(t.shape)} mean={t.mean().item():.4f} std={t.std().item():.4f} min={t.min().item():.4f} max={t.max().item():.4f}"


def main():
    print(f"\n=== ckpt normalizer: {NORM_FILE} ===")
    ckpt = _ckpt_stats()
    for k, v in sorted(ckpt.items()):
        print(f"  {k}: {_summarize(v)}")

    print("\n=== dataset stats (rebuilt fresh from /home/khj20343/IC4R/data/lerobot/pyper/effl_dataset) ===")
    ds = _dataset_stats()
    for k, stats in sorted(ds.items()):
        if not isinstance(stats, dict):
            continue
        for sk, sv in stats.items():
            print(f"  {k}.{sk}: {_summarize(sv)}")

    print("\n=== diff: state/action mean+std (ckpt vs dataset) ===")
    for feat in ("observation.state", "action"):
        ds_stats = ds.get(feat, {})
        ds_mean = ds_stats.get("mean") if isinstance(ds_stats, dict) else None
        ds_std = ds_stats.get("std") if isinstance(ds_stats, dict) else None
        # ckpt keys look like "buffer." prefixed or flat — try both
        ck_mean = None
        ck_std = None
        for k, v in ckpt.items():
            if feat in k and "mean" in k:
                ck_mean = v
            if feat in k and "std" in k:
                ck_std = v
        if ds_mean is None or ck_mean is None:
            print(f"  {feat}: ckpt={'?' if ck_mean is None else 'OK'}  ds={'?' if ds_mean is None else 'OK'} (skip)")
            continue
        print(f"  {feat}.mean  ckpt={ck_mean.flatten().tolist()}")
        print(f"  {feat}.mean    ds={ds_mean.flatten().tolist()}")
        print(f"  {feat}.std   ckpt={ck_std.flatten().tolist()}")
        print(f"  {feat}.std     ds={ds_std.flatten().tolist()}")
        try:
            delta_mean = (ck_mean.flatten().float() - ds_mean.flatten().float()).abs().max().item()
            delta_std = (ck_std.flatten().float() - ds_std.flatten().float()).abs().max().item()
            print(f"  {feat}.|Δmean|_max = {delta_mean:.5f}")
            print(f"  {feat}.|Δstd|_max  = {delta_std:.5f}")
        except Exception as e:
            print(f"  diff failed: {e}")


if __name__ == "__main__":
    main()
