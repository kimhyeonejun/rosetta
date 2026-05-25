"""LeRobotDataset + pi05 processor + SCR4R history machinery.

Returns a DataLoader yielding batches that pi05's ``forward(batch)`` consumes,
extended with two SCR4R-specific fields:

  ``selector_history.image[<key>]`` : ``[B, T-1, C, H, W]`` history frames
  ``selector_history.valid``        : ``[B, T-1]`` bool (same-episode mask)

The history is assembled by :class:`EpisodeTemporalHistoryDataset`, mirroring
the OpenPI variant under ``ic4r/experiments/openpi_scr4r.py``. The strided
sampler matches that file's :class:`StridedEpochSampler` so train-time
``replan_steps``-apart history aligns with eval's mask_net cadence.

Important: pi05's normalisation_mapping sets ``VISUAL: IDENTITY``
(``configuration_pi05.py:74``), so the processor pipeline leaves images in
``[0, 1]`` at ``(B, C, H, W)`` — exactly the input contract MS-ILLM
encodes from. The SCR4R training loop calls ``codec(batch[image_key])``
then ``replace_images(batch, recon)`` before invoking pi05.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


# ── LeRobotDataset / pi05 processor builders ────────────────────────────────


def build_lerobot_dataset(
    repo_id: str,
    root: str | Path | None = None,
    episodes: list[int] | None = None,
    video_backend: str | None = None,
    delta_timestamps: dict[str, list[float]] | None = None,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        video_backend=video_backend,
        delta_timestamps=delta_timestamps,
    )


def build_pi05_processors(policy, dataset, checkpoint_path=None):
    """Construct the pi05 pre/post processor pipelines.

    Prefer the EXACT pipelines saved with the checkpoint
    (`policy_preprocessor.json` + `policy_postprocessor.json`) so we
    apply the same normalizer / state-tokenizer / tokenizer as at
    train+deploy time. Empirically the fresh-build path silently
    differs (prediction MSE drops by ~1.0 when the saved pipeline is
    used). Falls back to fresh-build if the checkpoint path is missing.
    """
    from pathlib import Path
    import logging as _logging

    if checkpoint_path:
        ckpt_dir = Path(checkpoint_path)
        pre_json = ckpt_dir / "policy_preprocessor.json"
        post_json = ckpt_dir / "policy_postprocessor.json"
        if pre_json.is_file() and post_json.is_file():
            try:
                from lerobot.processor.pipeline import PolicyProcessorPipeline
                preprocessor = PolicyProcessorPipeline.from_pretrained(
                    str(ckpt_dir), config_filename="policy_preprocessor.json"
                )
                postprocessor = PolicyProcessorPipeline.from_pretrained(
                    str(ckpt_dir), config_filename="policy_postprocessor.json"
                )
                _logging.getLogger(__name__).info(
                    "[pi05-proc] loaded SAVED pipelines from %s", ckpt_dir
                )
                return preprocessor, postprocessor
            except Exception as exc:
                _logging.getLogger(__name__).warning(
                    "[pi05-proc] saved pipeline load failed (%s); fresh-build fallback",
                    exc,
                )

    from lerobot.policies.pi05 import make_pi05_pre_post_processors
    stats = dataset.meta.stats if hasattr(dataset, "meta") else None
    return make_pi05_pre_post_processors(policy.config, dataset_stats=stats)


def discover_image_keys(batch: dict[str, Any], image_features) -> list[str]:
    """Filter image_features down to keys actually present in the batch."""
    keys = []
    for k in image_features:
        v = batch.get(k)
        if isinstance(v, torch.Tensor) and v.ndim == 4:
            keys.append(k)
    return keys


# ── History dataset + sampler ───────────────────────────────────────────────


def _extract_episode_index(item: dict[str, Any]) -> int:
    """LeRobotDataset puts `episode_index` directly on each item."""
    if "episode_index" not in item:
        raise KeyError(
            "Temporal history requires `episode_index` on each item; "
            "got keys: " + ", ".join(sorted(item.keys()))
        )
    v = item["episode_index"]
    if isinstance(v, torch.Tensor):
        return int(v.reshape(-1)[0].item())
    return int(np.asarray(v).reshape(()).item())


def _zero_like_image(image: Any):
    if isinstance(image, torch.Tensor):
        return torch.zeros_like(image)
    return np.zeros_like(np.asarray(image))


def _stack_images(frames):
    if not frames:
        return None
    if isinstance(frames[0], torch.Tensor):
        return torch.stack(frames, dim=0)
    return np.stack(frames, axis=0)


class EpisodeTemporalHistoryDataset(torch.utils.data.Dataset):
    """Wrap a LeRobotDataset with explicit previous-frame history per episode.

    All frames are used for training (no subsampling). ``replan_steps`` only
    controls the spacing between history frames — e.g. ``replan_steps=5,
    history_len=4`` fetches frames at offsets -15, -10, -5, 0 relative to
    the current frame. Same contract as the upstream OpenPI
    :class:`EpisodeTemporalHistoryDataset`.
    """

    def __init__(
        self,
        dataset,
        *,
        history_len: int,
        image_keys: tuple[str, ...],
        replan_steps: int = 1,
    ):
        if history_len < 1:
            raise ValueError(f"history_len must be >= 1, got {history_len}.")
        self._dataset = dataset
        self._previous_history_len = max(history_len - 1, 0)
        self._image_keys = tuple(image_keys)
        self._replan_steps = max(replan_steps, 1)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        current_item = self._dataset[index]
        if self._previous_history_len == 0:
            return current_item

        current_episode = _extract_episode_index(current_item)

        history_images: dict[str, list[Any]] = {key: [] for key in self._image_keys}
        history_valid: list[bool] = []

        for offset in range(self._previous_history_len, 0, -1):
            history_real_idx = index - offset * self._replan_steps
            history_item = None
            history_matches = history_real_idx >= 0
            if history_matches:
                history_item = self._dataset[history_real_idx]
                history_matches = _extract_episode_index(history_item) == current_episode

            history_valid.append(history_matches)
            for key in self._image_keys:
                current_image = current_item.get(key)
                if history_matches and history_item is not None and key in history_item:
                    history_images[key].append(history_item[key])
                else:
                    if current_image is None:
                        raise KeyError(
                            f"image key {key!r} missing in current item; "
                            f"got: {sorted(current_item.keys())}"
                        )
                    history_images[key].append(_zero_like_image(current_image))

        selector_history = {
            "image": {key: _stack_images(frames) for key, frames in history_images.items()},
            "valid": np.asarray(history_valid, dtype=np.bool_),
        }

        out = dict(current_item)
        out["selector_history"] = selector_history
        return out


class StridedEpochSampler(Sampler):
    """Sequential stride sampler with per-epoch random offset.

    Verbatim port of the upstream :class:`StridedEpochSampler`. On each
    ``__iter__`` call, picks ``offset ∈ [0, stride)`` from an epoch-seeded
    RNG, then yields ``offset, offset+stride, ...`` in order. Over
    ``stride`` epochs every dataset index is the leading sample exactly
    once, so full frame coverage is preserved.

    Required by SCR4R chunk-aligned history: a single ``__iter__`` pass
    must remain sequential (no inner shuffle) so that the SCR4R rolling
    feature cache sees two consecutive batches whose samples are exactly
    ``stride`` real frames apart — matching the eval-side replan cadence.
    """

    def __init__(self, data_source, stride: int, seed: int = 0):
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}.")
        self._n = len(data_source)
        self._stride = int(stride)
        self._seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        rng = torch.Generator()
        rng.manual_seed(self._seed + self._epoch)
        offset = int(torch.randint(0, self._stride, (1,), generator=rng).item())
        logging.info(
            "[StridedEpochSampler] epoch=%d offset=%d stride=%d n=%d",
            self._epoch, offset, self._stride, self._n,
        )
        return iter(range(offset, self._n, self._stride))

    def __len__(self) -> int:
        # Lower bound — true length varies by ±1 across offsets, but DataLoader
        # is constructed with drop_last=True so the trailing partial batch is
        # discarded anyway.
        return self._n // self._stride


# ── DataLoader assembly ────────────────────────────────────────────────────


def _scr4r_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Default collate that preserves ``selector_history`` structure.

    Falls back to torch.utils.data.default_collate for everything else.
    The history field has the structure:
      selector_history = {"image": {<key>: ndarray/tensor[T-1, C, H, W]},
                          "valid": ndarray[T-1] bool}
    Collate by stacking along the batch dim, keeping the dict/list shape.
    """
    from torch.utils.data._utils.collate import default_collate

    if not samples:
        return {}

    histories = [s.pop("selector_history", None) for s in samples]
    base = default_collate(samples)

    if any(h is None for h in histories):
        return base

    # All samples must share image_keys
    image_keys = list(histories[0]["image"].keys())
    stacked_images: dict[str, torch.Tensor] = {}
    for k in image_keys:
        per_sample = [h["image"][k] for h in histories]
        if isinstance(per_sample[0], torch.Tensor):
            stacked_images[k] = torch.stack(per_sample, dim=0)
        else:
            stacked_images[k] = torch.as_tensor(np.stack(per_sample, axis=0))

    valid = torch.as_tensor(
        np.stack([h["valid"] for h in histories], axis=0),
        dtype=torch.bool,
    )

    base["selector_history"] = {"image": stacked_images, "valid": valid}
    return base


class SCR4RDataLoader:
    """Wraps a torch DataLoader with chunk-aligned strided sampling + epoch hooks.

    Mirrors the on_epoch_start callback path from the upstream
    ``_SCR4RJAXLoader`` so SCR4R can flush its rolling feature cache between
    passes when the strided offset rotates.
    """

    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        stride: int,
        num_workers: int = 0,
        seed: int = 0,
        pin_memory: bool = True,
        on_epoch_start: Optional[Callable[[int], None]] = None,
    ):
        self._sampler = StridedEpochSampler(dataset, stride=stride, seed=seed)
        self._on_epoch_start = on_epoch_start
        self._dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self._sampler,
            num_workers=num_workers,
            collate_fn=_scr4r_collate,
            drop_last=True,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        epoch = 0
        while True:
            self._sampler.set_epoch(epoch)
            if self._on_epoch_start is not None:
                self._on_epoch_start(epoch)
            for batch in self._dataloader:
                yield batch
            epoch += 1


def build_dataloader(
    dataset,
    *,
    batch_size: int,
    selector_temporal_history: int = 1,
    selector_history_image_keys: tuple[str, ...] = (),
    replan_steps: int = 1,
    chunk_aligned_history: bool = False,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 0,
    pin_memory: bool = True,
    on_epoch_start: Optional[Callable[[int], None]] = None,
):
    """Top-level helper: wraps dataset → optionally history → optionally strided loader."""
    if selector_temporal_history > 1 and selector_history_image_keys:
        dataset = EpisodeTemporalHistoryDataset(
            dataset,
            history_len=selector_temporal_history,
            image_keys=selector_history_image_keys,
            replan_steps=replan_steps,
        )

    use_strided = (
        chunk_aligned_history
        and replan_steps > 1
        and not shuffle
    )

    if use_strided:
        logging.info(
            "[scr4r.data] Using chunk-aligned strided loader (stride=replan_steps=%d).",
            replan_steps,
        )
        return SCR4RDataLoader(
            dataset,
            batch_size=batch_size,
            stride=replan_steps,
            num_workers=num_workers,
            seed=seed,
            pin_memory=pin_memory,
            on_epoch_start=on_epoch_start,
        )

    if chunk_aligned_history and shuffle:
        logging.warning(
            "[scr4r.data] chunk_aligned_history=True but shuffle=True; "
            "falling back to default loader (rolling cache requires sequential data)."
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_scr4r_collate,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


__all__ = [
    "EpisodeTemporalHistoryDataset",
    "SCR4RDataLoader",
    "StridedEpochSampler",
    "build_dataloader",
    "build_lerobot_dataset",
    "build_pi05_processors",
    "discover_image_keys",
]
