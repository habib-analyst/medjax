"""Flax NNX models: TinyCNN (classifier), UNet and SegResNet (segmentation).

Channel-last (NHWC / NDHWC) throughout, matching the transforms module.
GroupNorm instead of BatchNorm so models stay stateless and jit-friendly —
no ``batch_stats`` plumbing, no train/eval mode bugs.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx


class ConvBlock(nnx.Module):
    """Conv -> GroupNorm -> ReLU -> Conv -> GroupNorm -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int, spatial_dims: int, rngs: nnx.Rngs):
        k = (3,) * spatial_dims
        self.conv1 = nnx.Conv(in_ch, out_ch, k, padding="SAME", rngs=rngs)
        self.norm1 = nnx.GroupNorm(out_ch, num_groups=min(8, out_ch), rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, k, padding="SAME", rngs=rngs)
        self.norm2 = nnx.GroupNorm(out_ch, num_groups=min(8, out_ch), rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.relu(self.norm1(self.conv1(x)))
        return nnx.relu(self.norm2(self.conv2(x)))


class Down(nnx.Module):
    """Strided-conv downsampling (no max-pool: keeps gradients dense)."""

    def __init__(self, in_ch: int, out_ch: int, spatial_dims: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_ch, out_ch, (3,) * spatial_dims,
            strides=(2,) * spatial_dims, padding="SAME", rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.conv(x)


def _upsample(x: jax.Array, spatial_dims: int) -> jax.Array:
    new_spatial = tuple(s * 2 for s in x.shape[1 : 1 + spatial_dims])
    return jax.image.resize(
        x, x.shape[:1] + new_spatial + x.shape[-1:], method="nearest"
    )


class Up(nnx.Module):
    """Nearest upsample -> ConvBlock on (skip || upsampled)."""

    def __init__(self, in_ch: int, out_ch: int, spatial_dims: int, rngs: nnx.Rngs):
        self.spatial_dims = spatial_dims
        self.block = ConvBlock(in_ch, out_ch, spatial_dims, rngs)

    def __call__(self, x: jax.Array, skip: jax.Array) -> jax.Array:
        x = _upsample(x, self.spatial_dims)
        return self.block(jnp.concatenate([skip, x], axis=-1))


class UNet(nnx.Module):
    """2D/3D UNet. ``x``: ``(B, *spatial, in_ch)`` -> ``(B, *spatial, out_ch)``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: Sequence[int] = (32, 64, 128, 256),
        spatial_dims: int = 2,
        seed: int = 0,
    ):
        rngs = nnx.Rngs(seed)
        self.spatial_dims = spatial_dims
        chs = [in_channels, *features]
        self.down_blocks = nnx.List([
            ConvBlock(chs[i], chs[i + 1], spatial_dims, rngs)
            for i in range(len(features))
        ])
        self.downs = nnx.List([
            Down(chs[i + 1], chs[i + 1], spatial_dims, rngs)
            for i in range(len(features) - 1)
        ])
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2, spatial_dims, rngs)
        rev = list(reversed(features))
        # decoder stage j: x has (2*rev[0] if j==0 else rev[j]) ch, skip has rev[j+1] ch
        self.ups = nnx.List([
            Up(
                (rev[j] if j > 0 else 2 * rev[0]) + rev[j + 1],
                rev[j + 1],
                spatial_dims,
                rngs,
            )
            for j in range(len(features) - 1)
        ])
        self.head = nnx.Conv(rev[-1], out_channels, (1,) * spatial_dims, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        skips = []
        for i, block in enumerate(self.down_blocks[:-1]):
            x = block(x)
            skips.append(x)
            x = self.downs[i](x)
        x = self.down_blocks[-1](x)
        x = self.bottleneck(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        return self.head(x)


class ResBlock(nnx.Module):
    """Pre-activation residual block (SegResNet style)."""

    def __init__(self, ch: int, spatial_dims: int, rngs: nnx.Rngs):
        k = (3,) * spatial_dims
        self.norm1 = nnx.GroupNorm(ch, num_groups=min(8, ch), rngs=rngs)
        self.conv1 = nnx.Conv(ch, ch, k, padding="SAME", rngs=rngs)
        self.norm2 = nnx.GroupNorm(ch, num_groups=min(8, ch), rngs=rngs)
        self.conv2 = nnx.Conv(ch, ch, k, padding="SAME", rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.relu(self.norm1(x))
        h = self.conv1(h)
        h = nnx.relu(self.norm2(h))
        return x + self.conv2(h)


class SegResNet(nnx.Module):
    """SegResNet-lite: residual encoder, transposed-conv decoder with skip adds."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: Sequence[int] = (32, 64, 128),
        spatial_dims: int = 2,
        seed: int = 0,
    ):
        rngs = nnx.Rngs(seed)
        chs = [in_channels, *features]
        self.stem = nnx.Conv(
            in_channels, features[0], (3,) * spatial_dims, padding="SAME", rngs=rngs
        )
        self.enc_blocks = nnx.List([])
        self.downs = nnx.List([])
        for i in range(len(features)):
            self.enc_blocks.append(ResBlock(chs[i + 1], spatial_dims, rngs))
            if i < len(features) - 1:
                self.downs.append(
                    nnx.Conv(
                        chs[i + 1], chs[i + 2], (3,) * spatial_dims,
                        strides=(2,) * spatial_dims, padding="SAME", rngs=rngs,
                    )
                )
        self.ups = nnx.List([])
        self.dec_blocks = nnx.List([])
        for i in reversed(range(len(features) - 1)):
            self.ups.append(
                nnx.ConvTranspose(
                    chs[i + 2], chs[i + 1], (2,) * spatial_dims,
                    strides=(2,) * spatial_dims, padding="SAME", rngs=rngs,
                )
            )
            self.dec_blocks.append(ResBlock(chs[i + 1], spatial_dims, rngs))
        self.head = nnx.Conv(features[0], out_channels, (1,) * spatial_dims, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.stem(x)
        skips = []
        for i, block in enumerate(self.enc_blocks):
            x = block(x)
            skips.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        for up, dec, skip in zip(self.ups, self.dec_blocks, reversed(skips[:-1])):
            x = up(x) + skip
            x = dec(x)
        return self.head(x)


class TinyCNN(nnx.Module):
    """Small classifier for the MedMNIST-JAX demo.

    ``(B, H, W, C) -> (B, num_classes)``. Global average pooling keeps the
    dense layer shape independent of input resolution.
    """

    def __init__(
        self, in_channels: int, num_classes: int, width: int = 32, seed: int = 0
    ):
        rngs = nnx.Rngs(seed)
        self.conv1 = nnx.Conv(in_channels, width, (3, 3), padding="SAME", rngs=rngs)
        self.norm1 = nnx.GroupNorm(width, num_groups=min(8, width), rngs=rngs)
        self.conv2 = nnx.Conv(width, width * 2, (3, 3), padding="SAME", rngs=rngs)
        self.norm2 = nnx.GroupNorm(width * 2, num_groups=min(8, width * 2), rngs=rngs)
        self.conv3 = nnx.Conv(width * 2, width * 4, (3, 3), padding="SAME", rngs=rngs)
        self.norm3 = nnx.GroupNorm(width * 4, num_groups=min(8, width * 4), rngs=rngs)
        self.fc1 = nnx.Linear(width * 4, 128, rngs=rngs)
        self.fc2 = nnx.Linear(128, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.max_pool(nnx.relu(self.norm1(self.conv1(x))), (2, 2), (2, 2))
        x = nnx.max_pool(nnx.relu(self.norm2(self.conv2(x))), (2, 2), (2, 2))
        x = nnx.relu(self.norm3(self.conv3(x)))
        x = jnp.mean(x, axis=(1, 2))  # global average pool
        return self.fc2(nnx.relu(self.fc1(x)))
