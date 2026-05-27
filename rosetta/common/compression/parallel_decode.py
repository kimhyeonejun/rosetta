"""Multiprocess decode pool for per-image codecs (e.g. MS-ILLM).

Why a process pool and not threads: the heavy cost of decoding an MS-ILLM
payload is the *entropy (range) decoding*, which runs single-threaded on the
CPU (the GPU only handles the synthesis convolutions, a few ms). compressai's
range coder is also not thread-safe — ``MsIllmCompressor`` serialises every
call behind a process-wide lock — so threads cannot overlap the work. Separate
processes can. With three cameras per observation this turns ~3x136ms serial
into roughly one decode of wall time.

Lifecycle / spawn safety:
  - Workers use the ``spawn`` start method (required for CUDA — a forked
    process inherits a broken CUDA context).
  - The pool MUST be created and its workers forced to start *before*
    ``run_policy_server_compressed.main`` calls ``runpy.run_module(...,
    run_name='__main__', alter_sys=True)``, because that call rebinds
    ``sys.modules['__main__']`` to lerobot's policy_server module. spawn
    re-imports the parent's main module in each worker; if that happened after
    the rebind, workers could re-execute the server. ``get_decode_pool()``
    submits warm-up tasks that force every worker to spawn immediately, and
    ``wrap_policy_server()`` calls it while ``__main__`` is still the guarded
    entry point. See run_policy_server_compressed.py / integration.py.

Configuration (environment variables):
    LEROBOT_DECODE_WORKERS   number of decode worker processes (default 3;
                             match the camera count). 0 disables the pool and
                             falls back to in-process serial decode.
    LEROBOT_DECODE_GPUS      comma-separated GPU ids to pin workers to,
                             round-robin (e.g. '0,1,2'). Empty -> every worker
                             uses the default device (cuda:0). The codec's GPU
                             synthesis is cheap, so pinning is optional; the win
                             comes from parallel CPU entropy decoding.
"""
from __future__ import annotations

import atexit
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

# Per-worker singleton; populated by _init_worker in each child process.
_codec = None
_pool: ProcessPoolExecutor | None = None


def decode_workers() -> int:
    return int(os.environ.get("LEROBOT_DECODE_WORKERS", "3"))


def _init_worker(gpu_ids: list[int]) -> None:
    """Runs once per worker at startup: pin a GPU, load + warm the codec.

    Setting CUDA_VISIBLE_DEVICES here is effective because spawn gives each
    worker a fresh interpreter where CUDA has not been initialised yet.
    """
    ident = mp.current_process()._identity
    wid = (ident[0] - 1) if ident else 0
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[wid % len(gpu_ids)])
    # The pinned GPU is cuda:0 from inside this process.
    os.environ["LEROBOT_MSILLM_DEVICE"] = "cuda"

    import numpy as np

    from rosetta.common.compression.msillm import MsIllmCompressor

    global _codec
    _codec = MsIllmCompressor()
    # Warm both code paths so the first real observation isn't penalised by
    # weight download, model.to(cuda) and cuDNN autotuning.
    dummy = _codec.encode(np.zeros((64, 64, 3), np.uint8))
    _codec.decode(dummy)


def _decode_one(item: tuple[str, bytes]):
    """Decode a single payload in the worker, dispatching by codec name.

    Mirrors the in-process serial path (``get(codec).decode(data)``) so the
    pool supports any registered per-image codec, not just MS-ILLM.
    """
    codec_name, data = item
    from rosetta.common.compression.base import get

    return get(codec_name).decode(data)


def _noop(_=None) -> bool:
    """Warm-up task whose only purpose is to force a worker to spawn + init."""
    return True


def get_decode_pool() -> ProcessPoolExecutor | None:
    """Return the shared decode pool, creating + warming it on first use.

    Returns None when ``LEROBOT_DECODE_WORKERS <= 0`` so callers fall back to
    serial decode. The first call eagerly starts every worker (submitting
    ``_noop`` tasks) so all child processes exist before the caller proceeds —
    this is what keeps spawn safe under the server's runpy launch.
    """
    global _pool
    n = decode_workers()
    if n <= 0:
        return None
    if _pool is None:
        gpus = os.environ.get("LEROBOT_DECODE_GPUS", "")
        gpu_ids = [int(x) for x in gpus.split(",") if x.strip()]
        _pool = ProcessPoolExecutor(
            max_workers=n,
            mp_context=mp.get_context("spawn"),  # CUDA requires spawn, not fork
            initializer=_init_worker,
            initargs=(gpu_ids,),
        )
        atexit.register(_pool.shutdown, wait=False)
        # Force every worker to spawn now. The slow initializer keeps each
        # worker busy, so submitting n tasks reliably scales the pool to its
        # full width before any task completes.
        futures = [_pool.submit(_noop) for _ in range(n)]
        for f in futures:
            f.result(timeout=300)
    return _pool
