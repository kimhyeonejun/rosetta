"""SCR4R codec — temporal-selector sparse neural image compression.

Runtime wrapper around the SCR4R model (``compression/scr4r``): the HiFiC/MS-ILLM
hyperprior autoencoder plus the trained :class:`TemporalLatentSelector` that
predicts, per latent position, whether it is worth entropy-coding. Only the
selected positions are range-coded, so the on-wire bitstream is a sparse subset
of the full latent grid.

**The selection mask is NOT transmitted.** A dense per-channel mask is
``latent_features × h × w`` bits (≈7 KB for a 224² frame) — often larger than
the sparse latent it gates, which would defeat the compression. Instead the
decoder *recomputes* the mask: it is a deterministic function of side
information the receiver already has.

  * ``penultimate`` (the selector's feature input) comes from
    ``hyper_synthesis_scale(hyper_latent_decoded)``, and ``hyper_latent_decoded``
    is bit-identical on both sides — both range-decode the same hyper bytes,
    exactly like the hyperprior means/scales the latent coder already relies on.
  * the selector runs deterministically at inference (``mask = logit > 0``).
  * the selector runs on **CPU on both sides** (penultimate already lives on
    CPU in the compress layout) so there is no GPU↔CPU float drift that could
    flip a mask bit and desync the sparse range decode.

The selector also needs context a single ``encode(image)`` call can't supply,
so this codec works at the *observation* level on both ends:

  * ``encode_observation(obs) -> dict`` (client): the whole obs dict — every
    camera image plus the robot state — so each image is encoded with its
    canonical stream key, the shared FiLM state, and that stream's rolling
    1-frame history.
  * ``decode_observation(obs) -> dict`` (server): symmetric — recomputes each
    mask from the decoded hyper + the state (already in the obs dict) + the
    stream key + the server's own per-stream history.

**Cross-frame dependency caveat.** Because the selector has a 1-frame history
(``selector_temporal_history > 1``), the encoder and decoder must process the
*same* observation sequence in the *same* order for their histories — and thus
the recomputed masks — to match. lerobot's gRPC stream is reliable + in-order
and the decode hook fires on every deserialized observation, so this holds in
the normal case. If observations can be silently dropped before deserialization,
set ``LEROBOT_SCR4R_SEND_MASK=1`` to transmit the (bit-packed) mask instead —
self-contained per frame like MS-ILLM, at the cost of the extra bytes.

Checkpoint: a SCR4R training payload (see
``scr4r/stage.build_scr4r_checkpoint_payload``) — ``msillm_model_state_dict``
(codec, incl. the fine-tuned decoder) + the selector ``checkpoint_state()``.

Configuration (environment variables, snapshotted on first model load):
    LEROBOT_SCR4R_CHECKPOINT  path to the SCR4R .pt checkpoint (REQUIRED).
    LEROBOT_SCR4R_DEVICE      'cuda' | 'cpu' | 'cuda:N' (default cuda if avail).
    LEROBOT_SCR4R_STATE_KEY   obs key for the FiLM robot state
                              (default 'observation.state').
    LEROBOT_SCR4R_STATE_STATS path to QUANTILE stats for the FiLM state. Either
                              a flat JSON {"q01": [...], "q99": [...]} (/ .pt),
                              or a LeRobot dataset meta/stats.json keyed by
                              feature — the codec reads [state_key].q01/q99
                              (e.g. observation.state) straight from it. If
                              unset, the codec falls back to the project-
                              standard dataset stats (_DEFAULT_STATE_STATS)
                              when that file exists, so no per-machine setup is
                              needed. Pi0.5 normalizes
                              observation.state with NormalizationMode.QUANTILES
                              — 2*(x - q01)/(q99 - q01) - 1 → [-1, 1] — and the
                              selector trained on that normalized state. The
                              codec sees RAW state on both ends (normalization
                              runs later, in the policy preprocessor), which is
                              out-of-distribution for the selector. Without
                              stats the round-trip stays exact (both ends agree)
                              but mask quality degrades. Supply the policy
                              preprocessor / dataset q01,q99 here (or bake
                              "state_q01"/"state_q99" into the checkpoint) to
                              match training. Applied on both ends.
    LEROBOT_SCR4R_NUM_HEADS   selector attention heads — NOT recoverable from
                              the state_dict, must match training (default 8).
    LEROBOT_SCR4R_SEND_MASK   '1' to transmit the mask instead of recomputing
                              it (drop-robust, larger payload). Default '0'.

Bytestream layout (self-describing; the decoder needs no out-of-band config):

    [u8  version = 1]
    [u8  flags]                          bit0 = mask is included
    [u16 image_h] [u16 image_w]          original pre-pad size
    [u16 padded_h] [u16 padded_w]        post-pad size (multiples of 64)
    [u32 hyper_len]   [hyper_latent_bytes...]
    [u32 latent_len]  [latent_bytes...]            sparse (masked) latent
    # only when flags bit0 set:
    [u16 latent_c] [u16 latent_h] [u16 latent_w]
    [mask_bytes...]   ceil(latent_c*latent_h*latent_w / 8) bit-packed bool
"""
from __future__ import annotations

import logging
import os
import re
import struct
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

_LOGGER = logging.getLogger("rosetta.common.compression.scr4r")

from .base import get, register
from .observation import IMAGE_KEY_PREFIX, is_image_array
from .payload import CompressedImagePayload

_VERSION = 1
_FLAG_MASK_INCLUDED = 0x01
# version, flags, orig_h, orig_w, image_h, image_w, padded_h, padded_w
# orig_* is the pre-resize input size; image_* is the codec input after
# resizing to a multiple of 64 (matching training, which interpolates rather
# than pads). decode resizes the reconstruction back to orig_*.
_HEADER = struct.Struct("<BBHHHHHH")
_CODEC_FACTOR = 64
# latent_c, latent_h, latent_w (only present when the mask is transmitted)
_MASK_HEADER = struct.Struct("<HHH")

# Encoder downsampling factors (HiFiC): latent = padded / 16, hyper = latent / 4.
_LATENT_DOWNSAMPLE = 16
_HYPER_DOWNSAMPLE = 4

# Raw camera name → canonical selector stream key. The selector's learned
# stream embeddings are keyed by these canonical names, so the mapping must
# match the training profile (configs/train_scr4r_pi05.yaml).
_DEFAULT_ALIAS_MAP: dict[str, str] = {
    "top": "base_0_rgb",
    "side": "base_1_rgb",
    "wrist": "wrist_0_rgb",
}

# Project-standard dataset stats location (same absolute path on the dev box,
# the GPU server, and the robot client). When the FiLM checkpoint carries no
# baked stats and no env override is set, the codec falls back to reading
# observation.state q01/q99 straight from here — so no per-machine setup is
# needed. Override with LEROBOT_SCR4R_STATE_STATS if your dataset lives
# elsewhere.
_DEFAULT_STATE_STATS = (
    "/workspaces/rosetta_ws/datasets/lerobot/pyper/effl_dataset/meta/stats.json"
)


def _resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_state_stats(ckpt: dict, state_key: str):
    """Return (q01, q99) float32 CPU tensors for FiLM-state QUANTILE
    normalization, or (None, None). Checkpoint-baked stats win over the env
    path.

    ``LEROBOT_SCR4R_STATE_STATS`` may point at either:
      * a flat ``{"q01": [...], "q99": [...]}`` JSON / torch .pt, or
      * a LeRobot dataset ``meta/stats.json`` keyed by feature — we dig into
        ``[state_key]["q01"/"q99"]`` (e.g. observation.state). This lets the
        codec read stats straight from the dataset without a separate export
        or baking them into the checkpoint.
    """
    q01 = ckpt.get("state_q01")
    q99 = ckpt.get("state_q99")
    if q01 is None or q99 is None:
        path = os.environ.get("LEROBOT_SCR4R_STATE_STATS")
        if not path and os.path.exists(_DEFAULT_STATE_STATS):
            path = _DEFAULT_STATE_STATS  # project-standard dataset location
        if not path:
            return None, None
        if path.endswith((".pt", ".pth")):
            blob = torch.load(path, map_location="cpu", weights_only=False)
        else:
            import json
            with open(path) as fh:
                blob = json.load(fh)
        if "q01" in blob and "q99" in blob:
            q01, q99 = blob["q01"], blob["q99"]
        elif state_key in blob and "q01" in blob[state_key] and "q99" in blob[state_key]:
            q01, q99 = blob[state_key]["q01"], blob[state_key]["q99"]
        else:
            raise KeyError(
                f"LEROBOT_SCR4R_STATE_STATS={path} has neither top-level q01/q99 "
                f"nor {state_key!r} with q01/q99 (LeRobot meta/stats.json format)."
            )
    q01 = torch.as_tensor(np.asarray(q01), dtype=torch.float32).reshape(-1)
    q99 = torch.as_tensor(np.asarray(q99), dtype=torch.float32).reshape(-1)
    return q01, q99


def _flatten_strings(x) -> bytes:
    """Coerce a compressai strings entry (batch=1) to a single bytes object."""
    for _ in range(2):
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        else:
            break
    if not isinstance(x, (bytes, bytearray)):
        raise TypeError(
            f"expected bytes after unwrapping strings list; got {type(x).__name__}"
        )
    return bytes(x)


@register("scr4r")
class Scr4rCompressor:
    name = "scr4r"

    def __init__(self) -> None:
        self._configured_device: str | None = None
        self._configured_checkpoint: str | None = None
        self._lock = threading.Lock()
        # Entropy coding is not thread-safe; serialize compress/decompress.
        self._inference_lock = threading.Lock()
        # Lazily-built, device-keyed bundle: (codec, selector, meta).
        self._bundle: dict[str, tuple] = {}
        # Per-stream rolling history of the previous frame's selector feature.
        # Separate caches so a single process exercising both ends (e.g. a
        # round-trip test) doesn't cross-contaminate; in deployment encode and
        # decode run in different processes and only touch their own cache.
        self._enc_prev: dict[str, torch.Tensor] = {}
        self._dec_prev: dict[str, torch.Tensor] = {}

    # --- configuration --------------------------------------------------

    @property
    def device(self) -> str:
        if self._configured_device is not None:
            return self._configured_device
        return _resolve_device(os.environ.get("LEROBOT_SCR4R_DEVICE"))

    @property
    def checkpoint_path(self) -> str:
        path = self._configured_checkpoint or os.environ.get("LEROBOT_SCR4R_CHECKPOINT")
        if not path:
            raise RuntimeError(
                "scr4r codec needs a checkpoint: set LEROBOT_SCR4R_CHECKPOINT "
                "(or call configure(checkpoint=...))."
            )
        return path

    @property
    def state_key(self) -> str:
        return os.environ.get("LEROBOT_SCR4R_STATE_KEY", "observation.state")

    @property
    def send_mask(self) -> bool:
        return os.environ.get("LEROBOT_SCR4R_SEND_MASK", "0").strip().lower() in (
            "1", "true", "yes", "on",
        )

    def configure(
        self,
        *,
        device: str | None = None,
        checkpoint: str | None = None,
    ) -> None:
        if device is not None:
            self._configured_device = device
        if checkpoint is not None:
            self._configured_checkpoint = checkpoint

    def reset(self) -> None:
        """Drop cached models and per-stream history."""
        with self._lock:
            self._bundle.clear()
            self._enc_prev.clear()
            self._dec_prev.clear()

    # --- model loading --------------------------------------------------

    def _get_bundle(self):
        device = self.device
        with self._lock:
            cached = self._bundle.get(device)
            if cached is not None:
                return cached

            from .scr4r.codec_models import HiFiCAutoencoder_SCR
            from .scr4r.temporal_selector import TemporalLatentSelector

            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            codec_sd = ckpt["msillm_model_state_dict"]
            sel_sd = ckpt["mask_net_state_dict"]

            # Codec dims from "hyper_analysis.0.weight": [hyper_features, latent_features, 3, 3].
            ha = codec_sd["hyper_analysis.0.weight"]
            hyper_features, latent_features = int(ha.shape[0]), int(ha.shape[1])

            codec = HiFiCAutoencoder_SCR(
                in_channels=3,
                latent_features=latent_features,
                hyper_features=hyper_features,
            )
            codec.load_state_dict(codec_sd)
            codec.eval()

            # Selector architecture from the state_dict (the checkpoint doesn't
            # store every arch knob), num_heads from env (not recoverable).
            sel_in = int(sel_sd["input_proj.weight"].shape[1])
            sel_hidden = int(sel_sd["input_proj.weight"].shape[0])
            sel_out = int(sel_sd["output_proj.weight"].shape[0])
            history_len = int(sel_sd["time_embeddings.weight"].shape[0])
            num_streams = int(sel_sd["stream_embeddings.weight"].shape[0])
            num_layers = 1 + max(
                int(k.split(".")[1])
                for k in sel_sd
                if k.startswith("transformer_layers.")
            )
            ff = sel_sd["transformer_layers.0.linear1.weight"].shape[0]
            mlp_ratio = float(ff) / float(sel_hidden)
            use_film = any(k.startswith("film_generators.") for k in sel_sd)
            robot_state_dim = (
                int(sel_sd["film_generators.0.0.weight"].shape[1]) if use_film else 0
            )
            num_heads = int(os.environ.get("LEROBOT_SCR4R_NUM_HEADS", "8"))

            stream_order = list(
                ckpt.get(
                    "selector_context_order",
                    list(_DEFAULT_ALIAS_MAP.values())[:num_streams],
                )
            )
            selector = TemporalLatentSelector(
                in_channels=sel_in,
                out_channels=sel_out,
                hidden_dim=sel_hidden,
                history_len=history_len,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=0.0,
                stream_names=stream_order,
                robot_state_dim=robot_state_dim,
                disable_stream_embedding=bool(ckpt.get("disable_stream_embedding", False)),
                cross_camera_attention=bool(ckpt.get("cross_camera_attention", False)),
            )
            selector.load_state_dict(sel_sd)
            selector.eval()
            # Selector stays on CPU on BOTH ends: penultimate already lives on
            # CPU in the compress layout, and matching devices guarantees the
            # mask (a hard logit>0 threshold) is bit-identical encoder↔decoder.
            selector.to("cpu")

            if device != "cpu":
                codec.to(device)
                # Entropy bottlenecks + hyper transforms on CPU; encoder/decoder
                # convolutions on GPU.
                codec.update_tensor_devices("compress")
            else:
                codec.update()

            state_q01, state_q99 = (None, None)
            if use_film:
                state_q01, state_q99 = _resolve_state_stats(ckpt, self.state_key)
                if state_q01 is None:
                    import logging
                    logging.getLogger(__name__).warning(
                        "[scr4r] FiLM selector but no state stats "
                        "(LEROBOT_SCR4R_STATE_STATS / checkpoint state_q01+q99). "
                        "Using RAW state — out-of-distribution vs training; mask "
                        "quality will degrade. Round-trip stays exact."
                    )

            meta = {
                "use_film": use_film,
                "robot_state_dim": robot_state_dim,
                "history_len": history_len,
                "latent_features": latent_features,
                "state_q01": state_q01,
                "state_q99": state_q99,
            }
            bundle = (codec, selector, meta)
            self._bundle[device] = bundle
            return bundle

    # --- selector mask (shared by encode + decode) ----------------------

    def _selector_mask(
        self, selector, meta, penultimate: torch.Tensor, stream_key: str, prev_cache: dict
    ) -> torch.Tensor:
        """Run the selector on CPU and return a bool mask [1, out_C, h, w].

        ``prev_cache`` is the caller's per-stream history dict (encoder vs
        decoder keep their own). FiLM conditioning, if any, must already be set
        on ``selector`` for this frame.
        """
        cur = penultimate.detach().to("cpu").unsqueeze(1)  # [1, 1, F, h, w]
        if meta["history_len"] > 1:
            prev = prev_cache.get(stream_key)
            if prev is not None and prev.shape == cur[:, 0].shape:
                feat = torch.cat([prev.unsqueeze(1), cur], dim=1)
                valid = torch.tensor([[True, True]], dtype=torch.bool)
            else:
                # Cold start: pad with the current frame, mark it invalid.
                feat = torch.cat([cur.clone(), cur], dim=1)
                valid = torch.tensor([[False, True]], dtype=torch.bool)
        else:
            feat = cur
            valid = torch.ones(1, 1, dtype=torch.bool)

        logits = selector(
            {stream_key: feat}, [stream_key], history_valid={stream_key: valid}
        )[stream_key]  # [1, out_C, h, w]
        prev_cache[stream_key] = cur[:, 0].detach()
        return logits > 0

    def _set_film(self, selector, meta, obs: dict) -> None:
        if not meta["use_film"]:
            return
        state = self._gather_state(obs)                       # np.ndarray [D]
        t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)  # [1, D], CPU
        q01, q99 = meta["state_q01"], meta["state_q99"]
        if q01 is not None:
            # QUANTILES, matching Pi0.5's NormalizationMode.QUANTILES at train
            # time: 2*(x - q01)/(q99 - q01) - 1 → [-1, 1].
            t = 2.0 * (t - q01) / (q99 - q01).clamp_min(1e-8) - 1.0
        selector.set_conditioning(robot_state=t)

    def _gather_state(self, obs: dict) -> np.ndarray:
        """Assemble the FiLM robot-state vector from the observation.

        The Rosetta raw client flattens a JointState into per-name scalar keys
        (e.g. ``position.joint1`` … ``position.gripper``) rather than a single
        ``observation.state`` array, so the codec reassembles the vector here.
        Order must match training (the contract selector order = the q01/q99
        order). Resolution:
          1. a single array under ``self.state_key`` (LEROBOT_SCR4R_STATE_KEY);
          2. explicit ordered component keys from ``LEROBOT_SCR4R_STATE_KEYS``;
          3. auto: piper-style ``position.joint{N}`` (numeric order) + trailing
             ``position.gripper``.
        Same obs reaches encode (client) and decode (server), so any of these
        yields an identical vector on both ends.
        """
        v = obs.get(self.state_key)
        if v is not None:
            return np.asarray(v, dtype=np.float32).reshape(-1)

        keys_env = os.environ.get("LEROBOT_SCR4R_STATE_KEYS")
        if keys_env:
            keys = [k.strip() for k in keys_env.split(",") if k.strip()]
        else:
            joints = sorted(
                (k for k in obs if re.fullmatch(r"position\.joint\d+", str(k))),
                key=lambda k: int(str(k).rsplit("joint", 1)[1]),
            )
            keys = joints + (["position.gripper"] if "position.gripper" in obs else [])

        if not keys or any(k not in obs for k in keys):
            raise KeyError(
                f"scr4r FiLM (robot_state_dim) needs the robot state, but "
                f"'{self.state_key}' is absent and component keys "
                f"{keys or '(none auto-detected)'} are not all present. Set "
                f"LEROBOT_SCR4R_STATE_KEY or LEROBOT_SCR4R_STATE_KEYS. "
                f"Available obs keys: {sorted(map(str, obs))}."
            )
        return np.asarray(
            [float(np.asarray(obs[k]).reshape(-1)[0]) for k in keys], dtype=np.float32
        )

    # --- stream-key mapping ---------------------------------------------

    def _stream_key(self, image_key: str) -> str:
        short = image_key
        if short.startswith(IMAGE_KEY_PREFIX):
            short = short[len(IMAGE_KEY_PREFIX):]
        return _DEFAULT_ALIAS_MAP.get(short, short)

    # --- observation-level encode (client) ------------------------------

    def encode_observation(self, obs: dict) -> dict:
        codec, selector, meta = self._get_bundle()
        out: dict = {}
        t0 = time.perf_counter()
        n = 0
        with self._inference_lock:
            self._set_film(selector, meta, obs)
            for k, v in obs.items():
                if is_image_array(k, v):
                    out[k] = CompressedImagePayload(
                        codec=self.name,
                        shape=tuple(v.shape),
                        dtype=str(v.dtype),
                        data=self._encode_image(codec, selector, meta, v, self._stream_key(k)),
                    )
                    n += 1
                else:
                    out[k] = v
        if os.environ.get("LEROBOT_SCR4R_TIMING"):
            dt = (time.perf_counter() - t0) * 1000
            _LOGGER.warning("[scr4r] encode %d cam(s) on %s: %.0f ms", n, self.device, dt)
        return out

    def _encode_image(self, codec, selector, meta, image: np.ndarray, stream_key: str) -> bytes:
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"scr4r expects (H, W, 3) uint8; got shape {image.shape} dtype {image.dtype}"
            )
        device = self.device
        x = (
            torch.from_numpy(image)
            .to(device)
            .permute(2, 0, 1)
            .to(torch.float32)
            .div_(255.0)
            .unsqueeze(0)
        )

        # Resize to a multiple of 64 by interpolation — matching training
        # (Pi05SCR4R._compress), NOT reflect-padding. Feeding the model the same
        # input distribution it was trained on is critical for reconstruction
        # quality (and thus the policy's actions).
        orig_h, orig_w = int(x.shape[-2]), int(x.shape[-1])
        if orig_h % _CODEC_FACTOR != 0 or orig_w % _CODEC_FACTOR != 0:
            new_h = ((orig_h + _CODEC_FACTOR - 1) // _CODEC_FACTOR) * _CODEC_FACTOR
            new_w = ((orig_w + _CODEC_FACTOR - 1) // _CODEC_FACTOR) * _CODEC_FACTOR
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        def mask_provider(penultimate: torch.Tensor) -> torch.Tensor:
            return self._selector_mask(selector, meta, penultimate, stream_key, self._enc_prev)

        with torch.inference_mode():
            compressed = codec.compress(
                x, force_cpu=(device == "cpu"), mask_provider=mask_provider
            )

        latent_bytes = _flatten_strings(compressed.latent_strings)
        hyper_bytes = _flatten_strings(compressed.hyper_latent_strings)
        image_h, image_w = compressed.image_size
        padded_h, padded_w = compressed.padded_size

        send = self.send_mask
        flags = _FLAG_MASK_INCLUDED if send else 0
        parts = [
            _HEADER.pack(_VERSION, flags, orig_h, orig_w, int(image_h), int(image_w), int(padded_h), int(padded_w)),
            struct.pack("<I", len(hyper_bytes)), hyper_bytes,
            struct.pack("<I", len(latent_bytes)), latent_bytes,
        ]
        if send:
            mask = compressed.mask
            if mask is None:
                raise RuntimeError("scr4r compress() returned no mask")
            _, latent_c, latent_h, latent_w = mask.shape
            mask_bits = np.packbits(
                mask.detach().cpu().numpy().astype(bool).reshape(-1)
            ).tobytes()
            parts.append(_MASK_HEADER.pack(int(latent_c), int(latent_h), int(latent_w)))
            parts.append(mask_bits)
        return b"".join(parts)

    # --- per-image encode (protocol compliance / no-FiLM fallback) ------

    def encode(self, image: np.ndarray) -> bytes:
        codec, selector, meta = self._get_bundle()
        if meta["use_film"]:
            raise RuntimeError(
                "scr4r was trained with FiLM conditioning, so it needs the robot "
                "state and must be driven through encode_observation(obs) / "
                "compress_observation, not per-image encode()."
            )
        with self._inference_lock:
            return self._encode_image(codec, selector, meta, image, self._stream_key("top"))

    # --- payload parsing -------------------------------------------------

    @staticmethod
    def _parse(payload: bytes):
        head = _HEADER.size
        if len(payload) < head:
            raise ValueError(f"scr4r payload too short ({len(payload)} < {head})")
        version, flags, orig_h, orig_w, image_h, image_w, padded_h, padded_w = _HEADER.unpack(payload[:head])
        if version != _VERSION:
            raise ValueError(f"unsupported scr4r payload version {version}")
        off = head
        (hyper_len,) = struct.unpack_from("<I", payload, off)
        off += 4
        hyper_bytes = bytes(payload[off:off + hyper_len])
        off += hyper_len
        (latent_len,) = struct.unpack_from("<I", payload, off)
        off += 4
        latent_bytes = bytes(payload[off:off + latent_len])
        off += latent_len

        mask = None
        if flags & _FLAG_MASK_INCLUDED:
            latent_c, latent_h, latent_w = _MASK_HEADER.unpack_from(payload, off)
            off += _MASK_HEADER.size
            n = latent_c * latent_h * latent_w
            bits = np.unpackbits(
                np.frombuffer(payload[off:], dtype=np.uint8), count=n
            ).astype(bool)
            mask = torch.from_numpy(bits.reshape(1, latent_c, latent_h, latent_w).copy())
        return {
            "orig_size": (int(orig_h), int(orig_w)),
            "image_size": (int(image_h), int(image_w)),
            "padded_size": (int(padded_h), int(padded_w)),
            "hyper_bytes": hyper_bytes,
            "latent_bytes": latent_bytes,
            "mask": mask,
        }

    # --- observation-level decode (server) ------------------------------

    def decode_observation(self, obs: dict) -> dict:
        codec, selector, meta = self._get_bundle()
        out: dict = {}
        t0 = time.perf_counter()
        n = 0
        with self._inference_lock:
            self._set_film(selector, meta, obs)
            for k, v in obs.items():
                if isinstance(v, CompressedImagePayload) and v.codec == self.name:
                    arr = self._decode_image(codec, selector, meta, v.data, self._stream_key(k))
                    if arr.shape != v.shape or str(arr.dtype) != v.dtype:
                        raise ValueError(
                            f"codec {v.codec!r} round-trip mismatch for {k}: "
                            f"got shape={arr.shape}, dtype={arr.dtype}; "
                            f"expected shape={v.shape}, dtype={v.dtype}"
                        )
                    out[k] = arr
                    n += 1
                elif isinstance(v, CompressedImagePayload):
                    out[k] = get(v.codec).decode(v.data)
                else:
                    out[k] = v
        if os.environ.get("LEROBOT_SCR4R_TIMING"):
            dt = (time.perf_counter() - t0) * 1000
            _LOGGER.warning("[scr4r] decode %d cam(s) on %s: %.0f ms", n, self.device, dt)
        return out

    def _decode_image(self, codec, selector, meta, payload: bytes, stream_key: str) -> np.ndarray:
        fields = self._parse(payload)
        padded_h, padded_w = fields["padded_size"]
        image_h, image_w = fields["image_size"]
        orig_h, orig_w = fields["orig_size"]
        device = self.device

        with torch.inference_mode():
            # Recompute the hyper latent (bit-identical to the encoder side).
            hyper_h = padded_h // _LATENT_DOWNSAMPLE // _HYPER_DOWNSAMPLE
            hyper_w = padded_w // _LATENT_DOWNSAMPLE // _HYPER_DOWNSAMPLE
            hyper_latent_decoded = codec.hyper_bottleneck.decompress(
                [fields["hyper_bytes"]], (hyper_h, hyper_w)
            )
            means = codec.hyper_synthesis_mean(hyper_latent_decoded)
            scales, penultimate = codec.hyper_synthesis_scale(
                hyper_latent_decoded, return_penultimate=True
            )

            mask = fields["mask"]
            if mask is None:
                # Recompute the selection mask from the decoded hyper + state +
                # this stream's decoder-side history.
                mask = self._selector_mask(
                    selector, meta, penultimate, stream_key, self._dec_prev
                )
            else:
                # Transmitted-mask path: still advance the history cache so a
                # later switch back to recompute stays in sync.
                self._dec_prev[stream_key] = penultimate.detach().to("cpu")
            mask = mask.to(device=means.device, dtype=torch.bool)

            indexes_full = codec.latent_bottleneck.build_indexes(scales)
            mask_flat = mask.reshape(-1)
            indexes_sparse = indexes_full.reshape(-1)[mask_flat].unsqueeze(0)
            means_sparse = means.reshape(-1)[mask_flat].unsqueeze(0)
            latent_sparse = codec.latent_bottleneck.decompress(
                [fields["latent_bytes"]], indexes_sparse, means=means_sparse
            )
            latent_decoded = torch.zeros_like(means)
            flat = latent_decoded.reshape(-1)
            flat[mask_flat] = latent_sparse.reshape(-1)
            latent_decoded = flat.view_as(means)

            dec_device = codec.decoder.blocks[-1].weight.device
            recon = codec.decoder(latent_decoded.to(dec_device)).clamp(0.0, 1.0)
            recon = recon[:, :, :image_h, :image_w]
            # Resize back to the original input size (inverse of the encode-side
            # interpolation), matching Pi05SCR4R._decode.
            if (image_h, image_w) != (orig_h, orig_w):
                recon = F.interpolate(
                    recon, size=(orig_h, orig_w), mode="bilinear", align_corners=False
                ).clamp(0.0, 1.0)

        recon = recon.mul(255.0).round_().clamp_(0.0, 255.0).to(torch.uint8)
        return recon.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()

    # --- per-image decode (protocol compliance) -------------------------

    def decode(self, payload: bytes) -> np.ndarray:
        """Per-image decode. Only valid for transmitted-mask payloads, since a
        recompute-mode payload needs the stream key + robot state + history that
        only ``decode_observation`` has."""
        codec, selector, meta = self._get_bundle()
        fields = self._parse(payload)
        if fields["mask"] is None:
            raise RuntimeError(
                "scr4r payload omits the mask (recompute mode); decode it via "
                "decode_observation(obs) / decompress_observation, which has the "
                "stream key, robot state and history needed to regenerate it. "
                "Or set LEROBOT_SCR4R_SEND_MASK=1 on the encoder for self-"
                "contained per-image payloads."
            )
        with self._inference_lock:
            return self._decode_image(codec, selector, meta, payload, self._stream_key("top"))
