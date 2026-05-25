"""Verbatim mirror of ic4r/models/temporal_selector.py.

Kept inside rosetta_ic4r so this package is self-contained — the upstream
file at /home1/khj20343/IC4R/ic4r/models/temporal_selector.py is the source
of truth for research use; this copy MUST stay byte-for-byte equivalent in
the ``state_dict`` layout so SCR4R checkpoints roundtrip in either direction.
Update both files together if the architecture ever changes.

Canonical TemporalLatentSelector — SCR4R latent-rate mask predictor.

Single source of truth for the SCR4R selector head. Pure-PyTorch (no JAX,
no openpi, no benchmark deps) so any of the SCR4R training/eval code paths
(``openpi_scr4r``, ``vlabench/openpi/eval_adapter``,
``vlabench/gr00t/codec_eval_adapter``, ``common/scr4r_msillm_eval``) can
``from <pkg>.temporal_selector import TemporalLatentSelector`` without
pulling heavy framework imports.

Shape contract::

    forward(
        feature_maps  : dict[str, [B, T, F, h, w]],   # T <= history_len
        ordered_keys  : list[str],
        history_valid : dict[str, [B, T] bool] | None,
        *,
        return_all_steps : bool = False,
    ) -> dict[str, [B, out_C, h, w]]                  # last step only
       | tuple[dict, dict[str, [B, T, out_C, h, w]]]  # last + all steps

Checkpoint compatibility — every existing SCR4R/MS-ILLM ckpt was produced
by a class with this exact ``state_dict`` layout, so ckpts roundtrip
through any of the import sites without conversion.
"""

from __future__ import annotations

import torch


class TemporalLatentSelector(torch.nn.Module):
    """Predict latent-rate masks with temporal attention + FiLM from robot state."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        history_len: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        stream_names: list[str],
        robot_state_dim: int = 0,
        disable_stream_embedding: bool = False,
        cross_camera_attention: bool = False,
    ):
        super().__init__()
        self.disable_stream_embedding = bool(disable_stream_embedding)
        # When True, all cameras' (B·H·W, T_c, D) blocks are concatenated along
        # the token dim into one (B·H·W, ΣT_c, D) sequence per pixel before the
        # transformer runs. This gives the stream_embeddings a real job —
        # disambiguating tokens that are now attending across cameras — instead
        # of acting as a per-camera constant bias that LayerNorm+residual washes
        # out. State_dict layout is unchanged so existing ckpts still load.
        self.cross_camera_attention = bool(cross_camera_attention)
        if history_len < 1:
            raise ValueError(f"Temporal selector history_len must be >= 1, got {history_len}.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"Temporal selector hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.history_len = history_len
        self.use_film = robot_state_dim > 0
        self.stream_to_idx = {name: idx for idx, name in enumerate(stream_names)}
        self.input_proj = torch.nn.Linear(in_channels, hidden_dim)
        self.stream_embeddings = torch.nn.Embedding(max(len(stream_names), 1), hidden_dim)
        self.time_embeddings = torch.nn.Embedding(history_len, hidden_dim)

        self.transformer_layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=int(hidden_dim * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        if self.use_film:
            self.film_generators = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(robot_state_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, hidden_dim * 2),
                )
                for _ in range(num_layers)
            ])

        self.output_norm = torch.nn.LayerNorm(hidden_dim)
        self.output_proj = torch.nn.Linear(hidden_dim, out_channels)
        # Init mask logits to zero: weight=0, bias=0 → sigmoid=2.5 → noisy STE
        # flips ~50% of latents at step 0 (max gradient sigmoid'(0)=0.25).
        # Higher bias values (e.g. 1.0, 3.0, 10.0) saturate sigmoid and pin
        # the mask near 1 with vanishing gradient, which prevents the
        # selector from actually learning to introduce zeros.
        torch.nn.init.zeros_(self.output_proj.weight)
        torch.nn.init.constant_(self.output_proj.bias, 2.5)

        self._robot_state: torch.Tensor | None = None

    def set_conditioning(self, *, robot_state: torch.Tensor) -> None:
        self._robot_state = robot_state

    def _stream_index(self, key: str) -> int:
        return self.stream_to_idx.get(key, 0)

    def forward(
        self,
        feature_maps: dict[str, torch.Tensor],
        ordered_keys: list[str],
        history_valid: dict[str, torch.Tensor] | None = None,
        *,
        return_all_steps: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not ordered_keys:
            return ({}, {}) if return_all_steps else {}

        film_params: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if self.use_film:
            if self._robot_state is None:
                raise RuntimeError("Call set_conditioning(robot_state=...) before forward.")
            film_params = []
            for film_gen in self.film_generators:
                gb = film_gen(self._robot_state)
                gamma, beta = gb.chunk(2, dim=-1)
                film_params.append((gamma, beta))

        outputs: dict[str, torch.Tensor] = {}
        all_outputs: dict[str, torch.Tensor] = {}
        history_valid = history_valid or {}

        if self.cross_camera_attention and len(ordered_keys) > 1:
            return self._forward_cross_camera(
                feature_maps,
                ordered_keys,
                history_valid,
                film_params,
                return_all_steps=return_all_steps,
            )

        for key in ordered_keys:
            feature_map = feature_maps[key]
            if feature_map.ndim == 4:
                feature_sequence = feature_map.unsqueeze(1)
            elif feature_map.ndim == 5:
                feature_sequence = feature_map
            else:
                raise ValueError(
                    f"Temporal selector expects 4D or 5D feature maps, got {feature_map.shape} for {key}."
                )

            bsz, temporal_len, channels, height, width = feature_sequence.shape
            if temporal_len > self.history_len:
                raise ValueError(
                    f"Temporal selector received temporal_len={temporal_len} greater than configured "
                    f"history_len={self.history_len} for {key}."
                )

            tokens = feature_sequence.permute(0, 3, 4, 1, 2).reshape(
                bsz * height * width,
                temporal_len,
                channels,
            )
            tokens = self.input_proj(tokens)
            time_embed = self.time_embeddings.weight[:temporal_len].view(1, temporal_len, self.hidden_dim)
            if self.disable_stream_embedding:
                stream_embed = tokens.new_zeros((1, 1, self.hidden_dim))
            else:
                stream_embed = self.stream_embeddings.weight[self._stream_index(key)].view(1, 1, -1)

            key_valid = history_valid.get(key)
            src_key_padding_mask = None
            if key_valid is not None:
                key_valid = key_valid.to(device=tokens.device, dtype=torch.bool)
                if key_valid.ndim == 1:
                    key_valid = key_valid.unsqueeze(0).expand(bsz, -1)
                if key_valid.shape != (bsz, temporal_len):
                    raise ValueError(
                        f"Temporal selector history_valid for {key} must have shape {(bsz, temporal_len)}, "
                        f"got {tuple(key_valid.shape)}."
                    )
                src_key_padding_mask = (~key_valid).view(bsz, 1, 1, temporal_len)
                src_key_padding_mask = src_key_padding_mask.expand(bsz, height, width, temporal_len)
                src_key_padding_mask = src_key_padding_mask.reshape(bsz * height * width, temporal_len)

            tokens = tokens + time_embed + stream_embed

            for layer_idx, layer in enumerate(self.transformer_layers):
                tokens = layer(tokens, src_key_padding_mask=src_key_padding_mask)
                if film_params is not None:
                    gamma, beta = film_params[layer_idx]
                    g = gamma[:, None, None, :].expand(bsz, height, width, -1)
                    g = g.reshape(bsz * height * width, 1, self.hidden_dim)
                    b = beta[:, None, None, :].expand(bsz, height, width, -1)
                    b = b.reshape(bsz * height * width, 1, self.hidden_dim)
                    tokens = tokens * (1.0 + g) + b

            all_tokens = self.output_norm(tokens)
            key_logits = self.output_proj(all_tokens).reshape(
                bsz,
                height,
                width,
                temporal_len,
                self.out_channels,
            ).permute(0, 3, 4, 1, 2).contiguous()
            outputs[key] = key_logits[:, -1]
            if return_all_steps:
                all_outputs[key] = key_logits

        if return_all_steps:
            return outputs, all_outputs
        return outputs

    def _forward_cross_camera(
        self,
        feature_maps: dict[str, torch.Tensor],
        ordered_keys: list[str],
        history_valid: dict[str, torch.Tensor],
        film_params: list[tuple[torch.Tensor, torch.Tensor]] | None,
        *,
        return_all_steps: bool,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Joint per-pixel attention over (T × num_cameras) tokens. Requires all
        # cameras to share (B, H, W) at the selector's feature-map resolution;
        # MS-ILLM's hyper encoder produces uniform h × w for fixed-size inputs
        # so this holds for the standard SCR4R setup. Mismatches surface as a
        # ValueError below rather than silent broadcasting.
        cam_token_blocks: list[torch.Tensor] = []
        cam_pad_blocks: list[torch.Tensor] = []
        cam_lengths: list[int] = []
        shape_check: tuple[int, int, int] | None = None

        for key in ordered_keys:
            feature_map = feature_maps[key]
            if feature_map.ndim == 4:
                feature_sequence = feature_map.unsqueeze(1)
            elif feature_map.ndim == 5:
                feature_sequence = feature_map
            else:
                raise ValueError(
                    f"Temporal selector expects 4D or 5D feature maps, got {feature_map.shape} for {key}."
                )

            bsz, temporal_len, channels, height, width = feature_sequence.shape
            if shape_check is None:
                shape_check = (bsz, height, width)
            elif shape_check != (bsz, height, width):
                raise ValueError(
                    f"cross_camera_attention=True requires all cameras to share (B, H, W); "
                    f"got {(bsz, height, width)} for {key}, expected {shape_check}."
                )
            if temporal_len > self.history_len:
                raise ValueError(
                    f"Temporal selector received temporal_len={temporal_len} greater than configured "
                    f"history_len={self.history_len} for {key}."
                )

            tokens = feature_sequence.permute(0, 3, 4, 1, 2).reshape(
                bsz * height * width,
                temporal_len,
                channels,
            )
            tokens = self.input_proj(tokens)
            time_embed = self.time_embeddings.weight[:temporal_len].view(1, temporal_len, self.hidden_dim)
            if self.disable_stream_embedding:
                stream_embed = tokens.new_zeros((1, 1, self.hidden_dim))
            else:
                stream_embed = self.stream_embeddings.weight[self._stream_index(key)].view(1, 1, -1)
            tokens = tokens + time_embed + stream_embed

            key_valid = history_valid.get(key)
            if key_valid is not None:
                key_valid = key_valid.to(device=tokens.device, dtype=torch.bool)
                if key_valid.ndim == 1:
                    key_valid = key_valid.unsqueeze(0).expand(bsz, -1)
                if key_valid.shape != (bsz, temporal_len):
                    raise ValueError(
                        f"Temporal selector history_valid for {key} must have shape {(bsz, temporal_len)}, "
                        f"got {tuple(key_valid.shape)}."
                    )
                pad = (~key_valid).view(bsz, 1, 1, temporal_len)
                pad = pad.expand(bsz, height, width, temporal_len)
                pad = pad.reshape(bsz * height * width, temporal_len)
            else:
                pad = torch.zeros(
                    bsz * height * width, temporal_len, dtype=torch.bool, device=tokens.device,
                )

            cam_token_blocks.append(tokens)
            cam_pad_blocks.append(pad)
            cam_lengths.append(temporal_len)

        joint_tokens = torch.cat(cam_token_blocks, dim=1)
        joint_pad = torch.cat(cam_pad_blocks, dim=1)

        bsz, height, width = shape_check  # type: ignore[misc]
        for layer_idx, layer in enumerate(self.transformer_layers):
            joint_tokens = layer(joint_tokens, src_key_padding_mask=joint_pad)
            if film_params is not None:
                gamma, beta = film_params[layer_idx]
                g = gamma[:, None, None, :].expand(bsz, height, width, -1)
                g = g.reshape(bsz * height * width, 1, self.hidden_dim)
                b = beta[:, None, None, :].expand(bsz, height, width, -1)
                b = b.reshape(bsz * height * width, 1, self.hidden_dim)
                joint_tokens = joint_tokens * (1.0 + g) + b

        joint_tokens = self.output_norm(joint_tokens)
        joint_logits = self.output_proj(joint_tokens)

        outputs: dict[str, torch.Tensor] = {}
        all_outputs: dict[str, torch.Tensor] = {}
        offset = 0
        for key, T_c in zip(ordered_keys, cam_lengths):
            cam_logits = joint_logits[:, offset:offset + T_c, :].reshape(
                bsz, height, width, T_c, self.out_channels,
            ).permute(0, 3, 4, 1, 2).contiguous()
            outputs[key] = cam_logits[:, -1]
            if return_all_steps:
                all_outputs[key] = cam_logits
            offset += T_c

        if return_all_steps:
            return outputs, all_outputs
        return outputs


__all__ = ["TemporalLatentSelector"]
