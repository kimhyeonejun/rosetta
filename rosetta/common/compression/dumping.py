"""Persist decompressed observation images to disk for visual inspection,
plus optional per-payload bits-per-pixel metrics.

Two independent hooks fire from ``decompress_inplace`` on the server:

  1. PNG sink (LEROBOT_MSILLM_DUMP_DIR) — saves every decoded image to
     {dump_dir}/{key}/{timestep:08d}.png. AFTER decode, so what lands on
     disk is exactly what the policy consumes (lossy round-trip).
  2. BPP sink (LEROBOT_MSILLM_BPP_CSV / LEROBOT_MSILLM_BPP_LOG) — measures
     the compressed payload size BEFORE decode so the bitrate is computed
     against the raw on-wire bytes.

Convert a directory of PNGs to mp4 with ffmpeg::

    ffmpeg -framerate 30 -pattern_type glob -i 'top/*.png' \\
           -c:v libx264 -pix_fmt yuv420p top.mp4

Environment knobs:
    LEROBOT_MSILLM_DUMP_DIR    if set, enables PNG dump (default: off)
    LEROBOT_MSILLM_DUMP_KEYS   comma-separated obs keys to include in PNG
                               dump (default: every detected image array)
    LEROBOT_MSILLM_BPP_CSV     if set, append (timestep, key, codec, quality,
                               payload_bytes, height, width, bpp, cum_avg_bpp,
                               n) rows to this CSV. ``cum_avg_bpp`` is the
                               running mean bpp per (key, quality) so the last
                               row of any run is the converged compression
                               rate. Header written on first create.
    LEROBOT_MSILLM_BPP_LOG     truthy → logger.info one line per payload
"""
from __future__ import annotations

import atexit
import csv
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .observation import is_image_array
from .payload import CompressedImagePayload

_LOGGER = logging.getLogger("rosetta.common.compression.dumping")
_ENV_DIR = "LEROBOT_MSILLM_DUMP_DIR"
_ENV_KEYS = "LEROBOT_MSILLM_DUMP_KEYS"
_ENV_BPP_CSV = "LEROBOT_MSILLM_BPP_CSV"
_ENV_BPP_LOG = "LEROBOT_MSILLM_BPP_LOG"


def timestep_of(obs_obj: Any, fallback_counter: list[int] | None = None) -> int:
    """Best-effort integer timestep from a TimedObservation-like wrapper.

    Falls back to a monotonic counter (mutated in place) when the wrapper has
    no ``get_timestep`` — useful for bare-dict callers and tests.
    """
    getter = getattr(obs_obj, "get_timestep", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:  # noqa: BLE001
            pass
    if fallback_counter is not None:
        fallback_counter[0] += 1
        return fallback_counter[0]
    return 0


class _NullSink:
    """Cheap no-op sink used when LEROBOT_MSILLM_DUMP_DIR is unset."""

    enabled = False

    def dump(self, obs_obj: Any) -> None:  # noqa: D401
        return


class _PngSink:
    enabled = True

    def __init__(self, root: Path, keys: Optional[set[str]]) -> None:
        self._root = root
        self._keys = keys  # None = dump every detected image
        self._lock = threading.Lock()
        self._made_dirs: set[Path] = set()
        self._cv2 = None  # lazy import
        self._fallback_step = 0
        root.mkdir(parents=True, exist_ok=True)
        _LOGGER.info(
            "MS-ILLM decompressed-image dump enabled -> %s (keys=%s)",
            root,
            "ALL" if keys is None else sorted(keys),
        )

    def _ensure_cv2(self):
        if self._cv2 is None:
            import cv2  # noqa: PLC0415

            self._cv2 = cv2
        return self._cv2

    def dump(self, obs_obj: Any) -> None:
        # Two shapes appear in the wild: TimedObservation wrapping a dict, or a
        # bare dict (older codepaths / tests).
        inner = getattr(obs_obj, "observation", None)
        if not isinstance(inner, dict):
            if isinstance(obs_obj, dict):
                inner = obs_obj
            else:
                return
        step = self._timestep(obs_obj)
        for k, v in inner.items():
            if not is_image_array(k, v):
                continue
            if self._keys is not None and k not in self._keys:
                continue
            self._write(k, step, v)

    def _timestep(self, obs_obj: Any) -> int:
        # Bare-dict callers have no monotonic timestep; fall back to a
        # process-wide counter so successive frames still get distinct names.
        with self._lock:
            counter = [self._fallback_step]
            step = timestep_of(obs_obj, counter)
            self._fallback_step = counter[0]
        return step

    def _write(self, key: str, step: int, img: np.ndarray) -> None:
        safe = key.replace("/", "_").replace("\\", "_")
        sub = self._root / safe
        if sub not in self._made_dirs:
            with self._lock:
                sub.mkdir(parents=True, exist_ok=True)
                self._made_dirs.add(sub)
        path = sub / f"{step:08d}.png"
        cv2 = self._ensure_cv2()
        # Observation images are uint8 RGB (H, W, C). cv2.imwrite expects BGR.
        if img.ndim == 3 and img.shape[-1] == 3:
            arr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            arr = img
        cv2.imwrite(str(path), arr)


_SinkT = _NullSink  # for type-checking only
_SINK: Optional[_SinkT] = None
_INIT_LOCK = threading.Lock()


def sink() -> _NullSink:
    """Process-wide singleton, instantiated on first access.

    The env vars are read once. Restart the process to pick up changes.
    """
    global _SINK
    if _SINK is not None:
        return _SINK
    with _INIT_LOCK:
        if _SINK is not None:
            return _SINK
        dump_dir = os.environ.get(_ENV_DIR)
        if not dump_dir:
            _SINK = _NullSink()
        else:
            keys_env = os.environ.get(_ENV_KEYS)
            keys = (
                {k.strip() for k in keys_env.split(",") if k.strip()}
                if keys_env
                else None
            )
            _SINK = _PngSink(Path(dump_dir).expanduser(), keys)
        return _SINK


# --- BPP measurement -------------------------------------------------------


class _NullBppSink:
    enabled = False

    def record(
        self, key: str, step: int, payload: CompressedImagePayload
    ) -> None:
        return


def _quality_of(payload: CompressedImagePayload) -> Optional[int]:
    """Extract MS-ILLM quality from the payload header.

    MS-ILLM bytestream header is ``<BBHHHH>`` (msillm._HEADER); byte[1] is the
    quality code (1..6, 10, 11). Other codecs have no quality concept — return
    None so the CSV column stays empty.
    """
    if payload.codec != "msillm" or len(payload.data) < 2:
        return None
    return int(payload.data[1])


class _BppSink:
    enabled = True
    _HEADER = (
        "timestep",
        "key",
        "codec",
        "quality",
        "payload_bytes",
        "height",
        "width",
        "bpp",
        "cum_avg_bpp",
        "n",
    )

    def __init__(self, csv_path: Optional[Path], log_each: bool) -> None:
        self._lock = threading.Lock()
        self._writer = None
        self._fh = None
        # (key, quality) -> (count, running mean bpp). Welford-style update so
        # the running average stays numerically stable across long runs.
        self._running: dict[tuple[str, Optional[int]], tuple[int, float]] = {}
        if csv_path is not None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            new = (not csv_path.exists()) or csv_path.stat().st_size == 0
            self._fh = csv_path.open("a", newline="")
            self._writer = csv.writer(self._fh)
            if new:
                self._writer.writerow(self._HEADER)
                self._fh.flush()
            atexit.register(self._close)
        self._log_each = log_each
        _LOGGER.info(
            "MS-ILLM bpp metrics enabled (csv=%s, log_each=%s)",
            csv_path,
            log_each,
        )

    def _close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._fh = None

    def record(
        self, key: str, step: int, payload: CompressedImagePayload
    ) -> None:
        shape = payload.shape
        if len(shape) < 2:
            return
        h = int(shape[0])
        w = int(shape[1])
        if h <= 0 or w <= 0:
            return
        size = len(payload.data)
        bpp = size * 8.0 / (h * w)
        quality = _quality_of(payload)
        with self._lock:
            n, mean = self._running.get((key, quality), (0, 0.0))
            n += 1
            mean += (bpp - mean) / n
            self._running[(key, quality)] = (n, mean)
            cum_avg = mean
            count = n
            if self._log_each:
                _LOGGER.info(
                    "bpp key=%s step=%d codec=%s q=%s bytes=%d bpp=%.4f avg=%.4f n=%d",
                    key,
                    step,
                    payload.codec,
                    quality,
                    size,
                    bpp,
                    cum_avg,
                    count,
                )
            if self._writer is not None:
                self._writer.writerow(
                    (
                        step,
                        key,
                        payload.codec,
                        "" if quality is None else quality,
                        size,
                        h,
                        w,
                        f"{bpp:.6f}",
                        f"{cum_avg:.6f}",
                        count,
                    )
                )
                self._fh.flush()


_BPP_SINK: Optional[_NullBppSink] = None
_BPP_INIT_LOCK = threading.Lock()


def _truthy(v: Optional[str]) -> bool:
    return bool(v) and v.strip().lower() not in ("", "0", "false", "no", "off")


def bpp_sink() -> _NullBppSink:
    """Process-wide BPP sink singleton. Env vars read once at first access."""
    global _BPP_SINK
    if _BPP_SINK is not None:
        return _BPP_SINK
    with _BPP_INIT_LOCK:
        if _BPP_SINK is not None:
            return _BPP_SINK
        csv_env = os.environ.get(_ENV_BPP_CSV)
        log_each = _truthy(os.environ.get(_ENV_BPP_LOG))
        csv_path = Path(csv_env).expanduser() if csv_env else None
        if csv_path is None and not log_each:
            _BPP_SINK = _NullBppSink()
        else:
            _BPP_SINK = _BppSink(csv_path, log_each)
        return _BPP_SINK
