"""Load pi05 via iblnkn fork's lerobot (the version the model was actually
trained with) and run the SAME forward pass we use in our SCR4R loop.
Compare loss to our current (custom pi_gemma) baseline.

If loss drops from ~0.7 to ~0.05 → fork swap fixes everything.
If loss stays near 0.7 → version diff isn't the gap; something else.

Run:
  cd /home1/khj20343/IC4R
  conda activate robocasa
  ulimit -v unlimited
  export HF_TOKEN="$(cat /home1/khj20343/.cache/huggingface/token)"
  python src/rosetta_ic4r/scr4r/debug_iblnkn_forward.py
"""
from __future__ import annotations

import sys

# Strip the IC4R editable-install meta_path finder for `lerobot` so the
# fork at IBLNKN_PATH wins. pip's PEP-660 editable installs register a
# MetaPathFinder that bypasses sys.path; we must remove it first.
IBLNKN_PATH = "/home1/khj20343/iblnkn_lerobot_full/src"

# Minimal: just put iblnkn at front of sys.path and clear lerobot from
# sys.modules. Standalone-tested: this resolves to iblnkn correctly.
sys.path.insert(0, IBLNKN_PATH)
for _k in list(sys.modules):
    if _k == "lerobot" or _k.startswith("lerobot."):
        del sys.modules[_k]

# Also drop IC4R's lerobot path so PathFinder can't fall back to it.
_IC4R_LEROBOT = "/home1/khj20343/IC4R/src/lerobot/src"
sys.path = [p for p in sys.path if p != _IC4R_LEROBOT]

print(f"[dbg] sys.path[0:3] = {sys.path[0:3]}")
print(f"[dbg] full sys.path:")
for i, p in enumerate(sys.path):
    print(f"  {i}: {p!r}")
import os
_iblnkn_init = os.path.join(IBLNKN_PATH, "lerobot/__init__.py")
print(f"[dbg] iblnkn __init__ exists: {os.path.exists(_iblnkn_init)}")
print(f"[dbg] readable: {os.access(_iblnkn_init, os.R_OK)}")
print(f"[dbg] iblnkn lerobot dir: {os.listdir(os.path.join(IBLNKN_PATH, 'lerobot'))[:5]}")

import lerobot
print(f"[dbg] lerobot loaded from: {lerobot.__file__}")

import lerobot.policies.pi05.modeling_pi05 as mp
if IBLNKN_PATH not in mp.__file__:
    print(f"[WARN] meta_path strip didn't take — still loading from {mp.__file__}")
    print(f"[WARN] current sys.meta_path: {[c.__class__.__name__ for c in sys.meta_path]}")
    raise SystemExit(1)
print(f"[OK] using iblnkn lerobot at {mp.__file__}")

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05 import make_pi05_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.factory import resolve_delta_timestamps


CKPT = "/home1/khj20343/IC4R/pretrained_checkpoints/real_world_checkpoints/task_0_checkpoint/pretrained_model"
REPO = "pyper/effl_dataset"
ROOT = "/home/khj20343/IC4R/data/lerobot/pyper/effl_dataset"
EPS = list(range(62))


def main():
    print("\n=== building dataset ===")
    ds = LeRobotDataset(repo_id=REPO, root=ROOT, episodes=EPS, video_backend="pyav")
    print("\n=== loading pi05 (iblnkn version) ===")
    pi05 = PI05Policy.from_pretrained(CKPT)
    dt = resolve_delta_timestamps(pi05.config, ds.meta)
    if dt:
        ds = LeRobotDataset(repo_id=REPO, root=ROOT, episodes=EPS, video_backend="pyav", delta_timestamps=dt)
    for p in pi05.parameters():
        p.requires_grad_(False)
    pi05.eval().to(device="cuda", dtype=torch.bfloat16)

    # Use SAVED preprocessor (matches deployment exactly)
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    pre = PolicyProcessorPipeline.from_pretrained(CKPT, config_filename="policy_preprocessor.json")

    print("\n=== get one batch ===")
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    batch = pre(next(iter(dl)))
    # Move to cuda
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to("cuda")

    print("\n=== forward loss (10 trials, seeded) ===")
    losses = []
    for i in range(10):
        torch.manual_seed(100 + i)
        torch.cuda.manual_seed_all(100 + i)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = pi05.forward({k: v for k, v in batch.items()})
        losses.append(float(loss.item()))
        print(f"  trial {i}: loss={loss.item():.4f}")
    print(f"\n  mean = {sum(losses)/len(losses):.4f}")
    print(f"\n  (compare to our LOCAL lerobot baseline: mean ~ 0.72)")


if __name__ == "__main__":
    main()
