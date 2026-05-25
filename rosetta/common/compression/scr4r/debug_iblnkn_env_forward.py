"""Run pi05 forward in the iblnkn_pi05 conda env (which has iblnkn lerobot
+ custom transformers @ fix/lerobot_openpi installed cleanly).

Compare loss to our robocasa-env baseline (~0.72).

Run:
  conda activate iblnkn_pi05
  ulimit -v unlimited
  export HF_TOKEN="$(cat /home1/khj20343/.cache/huggingface/token)"
  python src/rosetta_ic4r/scr4r/debug_iblnkn_env_forward.py
"""
from __future__ import annotations

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.processor.pipeline import PolicyProcessorPipeline

CKPT = "/home1/khj20343/IC4R/pretrained_checkpoints/real_world_checkpoints/task_0_checkpoint/pretrained_model"
REPO = "pyper/effl_dataset"
ROOT = "/home/khj20343/IC4R/data/lerobot/pyper/effl_dataset"
EPS = list(range(62))


def main():
    print(f"[env] lerobot at: {__import__('lerobot').__file__}")
    print(f"[env] transformers at: {__import__('transformers').__file__}")

    print("\n=== building dataset ===")
    ds = LeRobotDataset(repo_id=REPO, root=ROOT, episodes=EPS, video_backend="pyav")
    print("\n=== loading pi05 (iblnkn version, native env) ===")
    pi05 = PI05Policy.from_pretrained(CKPT)
    dt = resolve_delta_timestamps(pi05.config, ds.meta)
    if dt:
        ds = LeRobotDataset(repo_id=REPO, root=ROOT, episodes=EPS, video_backend="pyav", delta_timestamps=dt)
    for p in pi05.parameters():
        p.requires_grad_(False)
    pi05.eval().to(device="cuda", dtype=torch.bfloat16)

    pre = PolicyProcessorPipeline.from_pretrained(CKPT, config_filename="policy_preprocessor.json")

    print("\n=== get one batch ===")
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    batch = pre(next(iter(dl)))
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
    print(f"  (robocasa env baseline: mean ~ 0.72)")


if __name__ == "__main__":
    main()
