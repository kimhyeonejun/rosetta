"""Smoke test: are vla(comp) and vla(uncomp) identical on the SAME batch
when called repeatedly? If yes for steps ≥2 → train-run spike at step 0/1
is a CUDA/cudnn warmup artifact, not a codec damage signal.

Run with:
  cd /home1/khj20343/IC4R
  conda activate robocasa
  ulimit -v unlimited
  export HF_TOKEN="$(cat /home1/khj20343/.cache/huggingface/token)"
  export PYTHONPATH=/home1/khj20343/IC4R/src/rosetta_ic4r
  python src/rosetta_ic4r/scr4r/debug_vla_comp_vs_uncomp.py
"""
from __future__ import annotations

import sys
import torch

sys.path.insert(0, "/home1/khj20343/IC4R/src/rosetta_ic4r")

from scr4r.train import _load_config, _move_batch_to_device, _resolve_profile
from scr4r.codec_instantiate import instantiate_msillm_scr_model
from scr4r.data import (
    build_lerobot_dataset,
    build_dataloader,
    build_pi05_processors,
)
from scr4r.pi05_scr4r import Pi05SCR4R
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.datasets.factory import resolve_delta_timestamps


def main() -> None:
    cfg = _load_config(
        [
            "--config",
            "src/rosetta_ic4r/scr4r/configs/train_scr4r_pi05.yaml",
            "--config",
            "src/rosetta_ic4r/scr4r/configs/train_scr4r_pi05_task0.yaml",
            "msillm.quality=6",
            "training.batch_size=8",
        ]
    )

    device = torch.device("cuda")
    eps = list(cfg.dataset.episodes) if cfg.dataset.episodes else None

    ds = build_lerobot_dataset(cfg.dataset.repo_id, cfg.dataset.root, eps)
    pi05 = PI05Policy.from_pretrained(cfg.vla.pretrained)
    for p in pi05.parameters():
        p.requires_grad_(False)
    pi05.eval().to(device=device, dtype=torch.bfloat16)

    dt = resolve_delta_timestamps(pi05.config, ds.meta)
    if dt:
        ds = build_lerobot_dataset(
            cfg.dataset.repo_id, cfg.dataset.root, eps, delta_timestamps=dt
        )
    pre, _ = build_pi05_processors(pi05, ds)

    codec = instantiate_msillm_scr_model(
        quality="6",
        device=device,
        freeze_encoder=True,
        freeze_bottleneck=True,
        freeze_decoder=False,
    )
    profile = _resolve_profile(cfg)
    scr = Pi05SCR4R(
        msillm_model=codec,
        pi05_policy=pi05,
        msillm_config=dict(cfg.msillm),
        selector_context_order=list(cfg.msillm.selector_context_order),
        selector_smoothness_keys=list(cfg.msillm.selector_smoothness_keys),
        profile=profile,
        state_key=str(cfg.vla.state_key),
    ).to(device)

    dl = build_dataloader(
        ds,
        batch_size=8,
        selector_temporal_history=2,
        selector_history_image_keys=tuple(cfg.profile.compress_streams),
        replan_steps=5,
        chunk_aligned_history=True,
        shuffle=False,
        num_workers=0,
        seed=42,
        pin_memory=True,
    )
    batch = pre(next(iter(dl)))
    batch = _move_batch_to_device(batch, device)

    print("\n=== same batch, 10 consecutive forwards (no train step) ===")
    print(f"{'trial':>5} {'comp':>8} {'uncomp':>8} {'delta':>8}")
    for i in range(10):
        # comp via the warmup step (does encode→decode→pi05). Internally
        # uses torch.autograd.grad, so we cannot wrap in torch.no_grad();
        # we just don't take an optimizer step, so the decoder stays put
        # and any cross-iteration variance is purely pi05 t/ε sampling.
        _loss, mets = scr.compute_warmup_step(batch)
        comp = mets["vla_loss"]
        uncomp = scr.compute_vla_loss_uncompressed(batch)
        delta = comp - uncomp
        print(f"{i:>5d} {comp:>8.3f} {uncomp:>8.3f} {delta:>+8.3f}")


if __name__ == "__main__":
    main()
