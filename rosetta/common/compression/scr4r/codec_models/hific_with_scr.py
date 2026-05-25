# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Verbatim mirror of src/compression_models/ms-illm/hific_with_scr.py.

from typing import List, NamedTuple, Optional, Tuple, Union
import logging

import torch
from torch import Tensor

import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from ._hyperprior_autoencoder import _HyperpriorAutoencoderBase

from typing import List, Tuple

import torch.nn.functional as F


def pad_image_to_factor(
    image: Tensor, factor: int, mode: str = "reflect"
) -> Tuple[Tensor, Tuple[int, int]]:

    # pad image if it's not divisible by downsamples
    _, _, height, width = image.shape
    pad_height = (factor - (height % factor)) % factor
    pad_width = (factor - (width % factor)) % factor
    if pad_height != 0 or pad_width != 0:
        image = F.pad(image, (0, pad_width, 0, pad_height), mode=mode)

    return image, (height, width)

class _ResidualBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.sequence = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding=padding),
            _channel_norm_2d(channels, affine=True),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, stride, padding=padding),
            _channel_norm_2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.sequence(x)

class ChannelNorm2D(nn.Module):
    def __init__(self, input_channels: int, epsilon: float = 1e-3, affine: bool = True):
        super().__init__()

        if input_channels <= 1:
            raise ValueError(
                "ChannelNorm only valid for channel counts greater than 1."
            )

        self.epsilon = epsilon
        self.affine = affine

        if affine is True:
            self.gamma = nn.Parameter(torch.ones(1, input_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, input_channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.var(x, dim=1, keepdim=True)

        x_normed = (x - mean) * torch.rsqrt(variance + self.epsilon)

        if self.affine is True:
            x_normed = self.gamma * x_normed + self.beta

        return x_normed

def _channel_norm_2d(input_channels, affine=True):
    return ChannelNorm2D(
        input_channels,
        affine=affine,
    )

class HyperpriorOutput(NamedTuple):
    image: Tensor
    latent: Tensor
    latent_likelihoods: Tensor
    quantized_latent_likelihoods: Tensor
    hyper_latent: Tensor
    hyper_latent_likelihoods: Tensor
    quantized_hyper_latent_likelihoods: Tensor
    quantized_latent: Optional[Tensor] = None
    quantized_hyper_latent: Optional[Tensor] = None
    hyper_decoder_penultimate: Optional[Tensor] = None


class HyperpriorCompressedOutput(NamedTuple):
    latent_strings: Union[List[str], List[List[str]]]
    hyper_latent_strings: List[str]
    image_size: Tuple[int, int]
    padded_size: Tuple[int, int]
    # Optional SCR latent mask. When set, ``latent_strings`` was produced
    # from a sparse subset of the latent tensor (only positions where
    # ``mask == True``) and ``decompress()`` must scatter the recovered
    # values back into a zero tensor of the original shape before running
    # the decoder. Shape matches the full latent tensor: [B, C_latent, h, w]
    # with dtype ``torch.bool``. ``None`` means the stock dense path was
    # used.
    mask: Optional[Tensor] = None


def _channel_norm_2d(input_channels, affine=True):
    return ChannelNorm2D(
        input_channels,
        affine=affine,
    )
class HiFiCEncoder(torch.nn.Module):
    """
    High-Fidelity Generative Image Compression (HiFiC) encoder.

    Args:
        input_dimensions: shape of the input tensor
        latent_features: number of bottleneck features
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_features: int = 220,
    ):
        super().__init__()

        blocks: List[nn.Module] = []
        for index, out_channels in enumerate((60, 120, 240, 480, 960)):
            if index == 0:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
                    _channel_norm_2d(out_channels, affine=True),
                    nn.ReLU(),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    _channel_norm_2d(out_channels, affine=True),
                    nn.ReLU(),
                )

            in_channels = out_channels
            blocks += [block]

        blocks += [nn.Conv2d(out_channels, latent_features, kernel_size=3, padding=1)]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)

class HiFiCGenerator(torch.nn.Module):
    """
    High-Fidelity Generative Image Compression (HiFiC) generator.

    Args:
        input_dimensions: shape of the input tensor
        batch_size: number of images per batch
        latent_features: number of bottleneck features
        n_residual_blocks: number of residual blocks
    """

    def __init__(
        self,
        image_channels: int = 3,
        latent_features: int = 220,
        n_residual_blocks: int = 9,
    ):
        super(HiFiCGenerator, self).__init__()

        self.n_residual_blocks = n_residual_blocks

        filters = [960, 480, 240, 120, 60]

        self.block_0 = nn.Sequential(
            _channel_norm_2d(latent_features, affine=True),
            nn.Conv2d(latent_features, filters[0], kernel_size=3, padding=1),
            _channel_norm_2d(filters[0], affine=True),
        )

        resid_blocks = []
        for _ in range(self.n_residual_blocks):
            resid_blocks.append(_ResidualBlock((filters[0])))

        self.resid_blocks = nn.Sequential(*resid_blocks)

        blocks: List[nn.Module] = []
        in_channels = filters[0]
        for out_channels in filters[1:]:
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        output_padding=1,
                        stride=2,
                        padding=1,
                    ),
                    _channel_norm_2d(out_channels, affine=True),
                    nn.ReLU(),
                )
            )

            in_channels = out_channels

        blocks.append(
            nn.Conv2d(
                filters[-1], out_channels=image_channels, kernel_size=7, padding=3
            )
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block_0(x)
        x = x + self.resid_blocks(x)
        for block in self.blocks:
            x = block(x)
        return x


def _conv(
    cin: int,
    cout: int,
    kernel_size: int = 5,
    stride: int = 2,
) -> nn.Conv2d:
    return nn.Conv2d(
        cin, cout, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
    )


def _deconv(
    cin: int,
    cout: int,
    kernel_size: int = 5,
    stride: int = 2,
) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        cin,
        cout,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=1,
        padding=kernel_size // 2,
    )

class _HyperSynthesisTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            _deconv(in_channels, in_channels),
            nn.ReLU(inplace=True),
            _deconv(in_channels, in_channels),
            nn.ReLU(inplace=True),
            _conv(
                in_channels,
                out_channels,
                stride=1,
                kernel_size=3,
            ),
        )

    def forward(
        self, x: Tensor, return_penultimate: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        penultimate = None
        for idx, layer in enumerate(self.layers):
            if idx == len(self.layers) - 1:
                penultimate = x
            x = layer(x)

        if return_penultimate:
            return x, penultimate
        return x

class HiFiCAutoencoder_SCR(_HyperpriorAutoencoderBase):

    def __init__(
        self,
        in_channels: int = 3,
        latent_features: int = 220,
        hyper_features: int = 320,
        num_residual_blocks: int = 9,
        freeze_encoder: bool = False,
        freeze_bottleneck: bool = False,
        freeze_decoder: bool = False,
    ):
        super().__init__()
        self._factor = 64  # this is the full downsampling factor
        self._frozen_encoder = False
        self._frozen_bottleneck = False
        self._frozen_decoder = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.encoder = HiFiCEncoder(
            in_channels=in_channels, latent_features=latent_features
        )
        self.decoder = HiFiCGenerator(
            image_channels=in_channels,
            latent_features=latent_features,
            n_residual_blocks=num_residual_blocks,
        )
        self.hyper_analysis = nn.Sequential(
            _conv(latent_features, hyper_features, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            _conv(hyper_features, hyper_features),
            nn.ReLU(inplace=True),
            _conv(hyper_features, hyper_features),
        )
        self.hyper_synthesis_mean = _HyperSynthesisTransform(
            hyper_features, latent_features
        )
        self.hyper_synthesis_scale = _HyperSynthesisTransform(
            hyper_features, latent_features
        )
        self.hyper_bottleneck = EntropyBottleneck(hyper_features)
        self.latent_bottleneck = GaussianConditional(scale_table=None)

        if freeze_encoder:
            self.freeze_encoder()
        if freeze_bottleneck:
            self.freeze_bottleneck()
        if freeze_decoder:
            self.freeze_decoder()

        self.set_compress_cpu_layers(
            [
                self.hyper_bottleneck,
                self.latent_bottleneck,
                self.hyper_synthesis_scale,
                self.hyper_synthesis_mean,
                self.hyper_analysis,
            ]
        )

    @property
    def factor(self) -> int:
        return self._factor

    @property
    def frozen_encoder(self) -> bool:
        return self._frozen_encoder

    @property
    def frozen_bottleneck(self) -> bool:
        return self._frozen_bottleneck

    @property
    def frozen_decoder(self) -> bool:
        return self._frozen_decoder

    def update_tensor_devices(self, target_operation: str):
        """
        Updates location of model weights based on target_operation.

        Args:
            target_operation: Either ''forward'' or ''compress''. For
                ''forward'', all tensors will be located on the model device.
                For ''compress'', weights that are used for likelihood
                calculation will be held on the CPU.
        """
        if target_operation not in ("forward", "compress"):
            raise ValueError("Target operation must be 'forward' or 'compress'.")

        if target_operation == "forward":
            target_device = self.encoder.blocks[0][0].weight.device
            self.to(target_device)
        else:
            self.update()
            self._set_devices_for_compress()

        self._device_setting = target_operation

    def _set_devices_for_compress(self):
        cpu = torch.device("cpu")
        self.hyper_bottleneck.to(cpu)
        self.latent_bottleneck.to(cpu)
        self.hyper_synthesis_scale.to(cpu)
        self.hyper_synthesis_mean.to(cpu)
        self.hyper_analysis.to(cpu)

    def _check_compress_devices(self) -> bool:
        result = True
        cpu = torch.device("cpu")
        for module in [
            self.hyper_bottleneck,
            self.latent_bottleneck,
            self.hyper_analysis,
            self.hyper_synthesis_mean,
            self.hyper_synthesis_scale,
        ]:
            for param in module.parameters():
                if not param.device == cpu:
                    result = False

        return result

    def freeze_encoder(self):
        """Freeze and disable training for the encoder."""
        self._frozen_encoder = True
        self.logger.info("Freezing encoder!")
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    def freeze_bottleneck(self):
        """Freeze and disable training for the bottleneck."""
        module: nn.Module
        self._frozen_bottleneck = True
        self.logger.info("Freezing bottleneck!")
        for module in [
            self.hyper_bottleneck,
            self.hyper_synthesis_mean,
            self.hyper_synthesis_scale,
            self.hyper_analysis,
            self.latent_bottleneck,
        ]:
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def freeze_decoder(self):
        """Freeze and disable training for the decoder."""
        self._frozen_decoder = True
        self.logger.info("Freezing decoder!")
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True) -> "HiFiCAutoencoder_SCR":
        model = super().train(mode)
        if model._frozen_encoder:
            model.freeze_encoder()
        if model._frozen_bottleneck:
            model.freeze_bottleneck()
        if model._frozen_decoder:
            model.freeze_decoder()

        return model

    def forward(self, image: Tensor) -> HyperpriorOutput:
        if not self.device_setting == "forward":
            raise RuntimeError(
                "Must call update_tensor_devices('forward') to use model forward."
            )

        # encode image to latent
        latent = self.encoder(image)

        # bottleneck processing
        hyper_latent = self.hyper_analysis(latent)
        hyper_latent, hyper_likelihoods = self.hyper_bottleneck(hyper_latent)
        with torch.no_grad():
            _, quantized_hyper_latent_likelihoods = self.hyper_bottleneck(
                hyper_latent, training=False
            )
        means = self.hyper_synthesis_mean(hyper_latent)
        scales, hyper_decoder_penultimate = self.hyper_synthesis_scale(
            hyper_latent, return_penultimate=True
        )
        quantized_latents, latent_likelihoods = self.latent_bottleneck(
            latent, scales, means=means
        )
        with torch.no_grad():
            _, quantized_latent_likelihoods = self.latent_bottleneck(
                latent, scales, means=means, training=False
            )
        if self.training:
            # we use straight-through to train the generator
            latent = self._ste_quantize(latent, means)
        else:
            # means we're in eval mode and the latents have been rounded
            latent = quantized_latents

        # reconstruct the image
        reconstruction = self.decoder(latent)

        return HyperpriorOutput(
            image=reconstruction,
            latent=latent,
            latent_likelihoods=latent_likelihoods,
            quantized_latent_likelihoods=quantized_latent_likelihoods.detach(),
            hyper_latent=hyper_latent,
            hyper_latent_likelihoods=hyper_likelihoods,
            quantized_hyper_latent_likelihoods=quantized_hyper_latent_likelihoods.detach(),
            hyper_decoder_penultimate=hyper_decoder_penultimate,
        )

    def compress(
        self,
        image: Tensor,
        force_cpu: bool = True,
        mask_provider=None,
    ) -> HyperpriorCompressedOutput:
        """
        Compress a batch of images into strings.

        Args:
            image: Tensor to compress (in [0, 1] floating point range).
            force_cpu: Whether to throw an error if any operations are not on
                CPU.
            mask_provider: Optional callable
                ``(hyper_decoder_penultimate: Tensor) -> Tensor`` returning a
                bool / {0,1} tensor the same shape as the latent
                (``[B, C_latent, h, w]``). When supplied, the gaussian
                conditional bottleneck only entropy-codes the latent
                positions where the mask is ``True``, so the rANS loop is
                proportional to the number of active positions rather than
                the full latent size — this is what makes SCR sparse
                compression actually faster at inference time. The selected
                mask is stored on the returned ``HyperpriorCompressedOutput``
                and consumed by :meth:`decompress`.
        """
        is_compress_layout = False
        if not self._on_cpu():
            if force_cpu:
                raise ValueError("All params must be on CPU if force_cpu=True.")
            if self._check_compress_devices():
                is_compress_layout = True
            else:
                # Must be full-GPU single-device layout.
                all_devices = {p.device for p in self.parameters()}
                if len(all_devices) != 1:
                    raise ValueError(
                        "Parameters span multiple devices without being in the "
                        "compress layout. Call update_tensor_devices('compress') "
                        "or move every parameter to a single device."
                    )

        image, (height, width) = pad_image_to_factor(image, self._factor)

        # image analysis
        encoder_out = self.encoder(image)
        # Under the compress layout the hyper modules live on CPU, so we
        # must move the encoder output there too. In the full-CPU or
        # full-GPU layouts we keep the latent on whatever device the
        # encoder produced it on.
        latent = encoder_out.cpu() if is_compress_layout else encoder_out

        # hyper analysis
        hyper_latent = self.hyper_analysis(latent)

        # hyper bottleneck — real range-coder encode + decode.
        hyper_latent_strings = self.hyper_bottleneck.compress(hyper_latent)
        hyper_latent_decoded = self.hyper_bottleneck.decompress(
            hyper_latent_strings, hyper_latent.shape[-2:]
        )

        # hyper synthesis. Ask for the penultimate feature when a mask
        # provider is supplied — that's the feature map the SCR selector
        # head consumes to predict the importance map.
        means = self.hyper_synthesis_mean(hyper_latent_decoded)
        if mask_provider is not None:
            scales, hyper_decoder_penultimate = self.hyper_synthesis_scale(
                hyper_latent_decoded, return_penultimate=True
            )
        else:
            scales = self.hyper_synthesis_scale(hyper_latent_decoded)

        mask: Optional[Tensor] = None
        if mask_provider is not None:
            mask = mask_provider(hyper_decoder_penultimate)
            if mask is None:
                raise ValueError("mask_provider returned None")
            mask = mask.to(device=latent.device, dtype=torch.bool)
            if mask.shape != latent.shape:
                raise ValueError(
                    f"SCR mask shape {tuple(mask.shape)} does not match "
                    f"latent shape {tuple(latent.shape)}."
                )

        # latent compression
        indexes_full = self.latent_bottleneck.build_indexes(scales)
        if mask is not None:
            mask_flat = mask.reshape(-1)
            latent_sparse = latent.reshape(-1)[mask_flat].unsqueeze(0)
            indexes_sparse = indexes_full.reshape(-1)[mask_flat].unsqueeze(0)
            means_sparse = means.reshape(-1)[mask_flat].unsqueeze(0)
            latent_strings = self.latent_bottleneck.compress(
                latent_sparse, indexes_sparse, means=means_sparse,
            )
        else:
            latent_strings = self.latent_bottleneck.compress(
                latent, indexes_full, means=means,
            )

        return HyperpriorCompressedOutput(
            latent_strings=latent_strings,
            hyper_latent_strings=hyper_latent_strings,
            image_size=(height, width),
            padded_size=(image.shape[-2], image.shape[-1]),
            mask=mask,
        )

    def decompress(
        self, compressed_data: HyperpriorCompressedOutput, force_cpu: bool = True
    ) -> Tensor:
        """Decompress a batch of images from strings."""
        if not self._on_cpu():
            if force_cpu:
                raise ValueError("All params must be on CPU if force_cpu=True.")
            if not self._check_compress_devices():
                all_devices = {p.device for p in self.parameters()}
                if len(all_devices) != 1:
                    raise ValueError(
                        "Parameters span multiple devices without being in the "
                        "compress layout. Call update_tensor_devices('compress') "
                        "or move every parameter to a single device."
                    )
            device = self.decoder.blocks[-1].weight.device
        else:
            device = torch.device("cpu")

        latent_size = (
            compressed_data.padded_size[0] // 2**4,
            compressed_data.padded_size[1] // 2**4,
        )
        hyper_latent_size = (latent_size[0] // 2**2, latent_size[1] // 2**2)

        # hyper synthesis
        hyper_latent_decoded = self.hyper_bottleneck.decompress(
            compressed_data.hyper_latent_strings, hyper_latent_size
        )
        means = self.hyper_synthesis_mean(hyper_latent_decoded)
        scales = self.hyper_synthesis_scale(hyper_latent_decoded)

        indexes_full = self.latent_bottleneck.build_indexes(scales)

        mask = compressed_data.mask
        if mask is not None:
            mask_cpu = mask.to(device=means.device, dtype=torch.bool)
            mask_flat = mask_cpu.reshape(-1)
            indexes_sparse = indexes_full.reshape(-1)[mask_flat].unsqueeze(0)
            means_sparse = means.reshape(-1)[mask_flat].unsqueeze(0)

            latent_sparse = self.latent_bottleneck.decompress(
                compressed_data.latent_strings, indexes_sparse, means=means_sparse,
            )

            latent_decoded = torch.zeros_like(means)
            latent_decoded_flat = latent_decoded.reshape(-1)
            latent_decoded_flat[mask_flat] = latent_sparse.reshape(-1)
            latent_decoded = latent_decoded_flat.view_as(means)
        else:
            latent_decoded = self.latent_bottleneck.decompress(
                compressed_data.latent_strings, indexes_full, means=means,
            )

        reconstruction: Tensor = self.decoder(latent_decoded.to(device))

        return reconstruction[
            :, :, : compressed_data.image_size[0], : compressed_data.image_size[1]
        ].clamp(0.0, 1.0)
