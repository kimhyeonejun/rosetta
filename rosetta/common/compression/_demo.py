"""Round-trip sanity check.

Run from the workspace root:
    python -m rosetta.common.compression._demo            # identity codec (lossless)
    python -m rosetta.common.compression._demo msillm     # MS-ILLM (lossy)
    python -m rosetta.common.compression._demo msillm 25  # fail if PSNR < 25 dB

Confirms:
  1. compress_observation replaces image arrays with CompressedImagePayload
  2. The whole obs dict pickles cleanly
  3. decompress_observation reconstructs arrays — bit-exact for lossless codecs,
     above a configurable PSNR threshold for lossy codecs.

The PSNR threshold defaults to inf (i.e. require bit-exact) for the identity
codec and to 20 dB for any lossy codec — which is loose enough to pass on
random noise (where neural codecs do badly) but still catches plumbing
errors that corrupt large regions.
"""
from __future__ import annotations

import pickle
import sys
import time

import numpy as np

from . import (
    available,
    compress_observation,
    decompress_observation,
    get,
)
from .payload import CompressedImagePayload

_LOSSLESS = {"identity"}


def _synth_image(h: int = 224, w: int = 224, seed: int = 0) -> np.ndarray:
    """Build a smooth-but-structured uint8 RGB image.

    Neural codecs collapse random noise to near-mean grey (no exploitable
    structure → PSNR ~10 dB), which makes the round-trip check meaningless.
    A gradient + low-frequency sinusoid is closer to what robot cameras
    actually see and lets the codec hit realistic PSNR (~25–35 dB at q4).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (
        0.4 * (xx / w)
        + 0.3 * (yy / h)
        + 0.15 * np.sin(xx / 12) * np.cos(yy / 16)
    )
    img = np.stack(
        [
            base + 0.05 * rng.standard_normal((h, w)),
            np.roll(base, 8, axis=0) + 0.05 * rng.standard_normal((h, w)),
            np.roll(base, 8, axis=1) + 0.05 * rng.standard_normal((h, w)),
        ],
        axis=-1,
    )
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def make_fake_obs() -> dict:
    rng = np.random.default_rng(0)
    return {
        "observation.state": rng.standard_normal(7).astype(np.float64),
        "observation.images.top": _synth_image(seed=1),
        "observation.images.side": _synth_image(seed=2),
        "observation.images.wrist": _synth_image(seed=3),
        "task": "pick up the block",
    }


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10((255.0**2) / mse)


def main(codec_name: str = "identity", min_psnr: float | None = None) -> int:
    print(f"available codecs: {available()}")
    print(f"using codec: {codec_name}")
    codec = get(codec_name)

    if min_psnr is None:
        min_psnr = float("inf") if codec_name in _LOSSLESS else 20.0
    print(f"min PSNR threshold: {min_psnr}")

    obs = make_fake_obs()
    raw_pickle = pickle.dumps(obs)
    print(f"raw pickle size:        {len(raw_pickle):>10,} bytes")

    # Warm-up pass: triggers torch.hub clone, weight download, entropy-table
    # init, JIT etc. None of this happens again on subsequent calls, so
    # including it would massively overstate the per-frame cost.
    t0 = time.perf_counter()
    packed = compress_observation(obs, codec)
    warm_ms = (time.perf_counter() - t0) * 1e3
    for k, v in packed.items():
        if isinstance(v, CompressedImagePayload):
            print(f"  packed[{k!r}] -> {v}")

    blob = pickle.dumps(packed)
    print(f"compressed pickle size: {len(blob):>10,} bytes")
    print(f"ratio:                  {len(blob) / len(raw_pickle):.4f}")
    print(f"warmup encode:          {warm_ms:.1f} ms (one-shot model load)")

    # Steady-state encode / decode timing (median of N runs).
    n_runs = 5
    enc_runs = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        packed = compress_observation(obs, codec)
        enc_runs.append((time.perf_counter() - t0) * 1e3)
    enc_runs.sort()

    blob = pickle.dumps(packed)
    dec_runs = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        restored = decompress_observation(pickle.loads(blob))
        dec_runs.append((time.perf_counter() - t0) * 1e3)
    dec_runs.sort()

    n_img = sum(1 for v in obs.values() if isinstance(v, np.ndarray) and v.ndim == 3)
    print(
        f"steady encode (n={n_runs}):   "
        f"min={enc_runs[0]:.1f} ms  median={enc_runs[n_runs // 2]:.1f} ms"
        f"  ({enc_runs[n_runs // 2] / max(n_img, 1):.1f} ms / image)"
    )
    print(
        f"steady decode (n={n_runs}):   "
        f"min={dec_runs[0]:.1f} ms  median={dec_runs[n_runs // 2]:.1f} ms"
        f"  ({dec_runs[n_runs // 2] / max(n_img, 1):.1f} ms / image)"
    )

    failed = False
    for k, v in obs.items():
        rv = restored[k]
        if isinstance(v, np.ndarray) and v.ndim == 3:
            psnr = _psnr(v, rv)
            print(f"  {k}: PSNR={psnr:.2f} dB shape={rv.shape} dtype={rv.dtype}")
            if psnr < min_psnr:
                print(
                    f"  FAIL: {k} PSNR {psnr:.2f} below threshold {min_psnr}",
                    file=sys.stderr,
                )
                failed = True
        elif isinstance(v, np.ndarray):
            if not np.array_equal(v, rv):
                print(f"  FAIL: array mismatch on {k}", file=sys.stderr)
                failed = True
        else:
            if v != rv:
                print(f"  FAIL: value mismatch on {k}", file=sys.stderr)
                failed = True

    if failed:
        return 1
    print("round-trip OK")
    return 0


if __name__ == "__main__":
    codec = sys.argv[1] if len(sys.argv) > 1 else "identity"
    psnr = float(sys.argv[2]) if len(sys.argv) > 2 else None
    raise SystemExit(main(codec, psnr))
