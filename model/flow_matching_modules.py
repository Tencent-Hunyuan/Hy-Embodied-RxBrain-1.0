"""Flow Matching helper modules for UnifiedMoT.

Provides:
  - TimestepEmbedder, PositionEmbedding (re-exported from model_utils)
  - patchify_latent / unpatchify_latent — reshape between (C, H_lat, W_lat)
    and (h*w, p*p*C) packed-patch token form
  - sample_timesteps — sigmoid(N(0,1)) → [0,1] with timestep_shift applied
  - build_noisy_latent — linear interp x_t = (1-t)*x_0 + t*noise
  - patch_boundary_loss — L1 across patch boundaries on predicted x_0
"""
from __future__ import annotations

from typing import List, Tuple

import torch

# Re-exported from model_utils for a stable import path (see __all__ below).
from .model_utils import (  # noqa: F401  # pylint: disable=unused-import
    TimestepEmbedder,
    PositionEmbedding,
)


# ---------------------------------------------------------------------------
# Patchify / Unpatchify
# ---------------------------------------------------------------------------

def patchify_latent(latent: torch.Tensor, h: int, w: int, patch_size: int, channels: int) -> torch.Tensor:
    """(C, H_lat, W_lat) → (h*w, p*p*C) where H_lat=h*p, W_lat=w*p."""
    p = patch_size
    latent = latent[:, : h * p, : w * p].reshape(channels, h, p, w, p)
    latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * channels)
    return latent


def unpatchify_latent(packed: torch.Tensor, h: int, w: int, patch_size: int, channels: int) -> torch.Tensor:
    """(h*w, p*p*C) → (C, H_lat, W_lat). Inverse of patchify_latent."""
    p = patch_size
    x = packed.reshape(h, w, p, p, channels)
    x = torch.einsum("hwpqc->chpwq", x).reshape(channels, h * p, w * p)
    return x


def pack_latents(
    padded_latents: List[torch.Tensor],
    shapes: List[Tuple[int, int]],
    patch_size: int,
    channels: int,
) -> torch.Tensor:
    """Concatenate per-image patchified latents into one (sum_h*w, p*p*C) tensor."""
    parts = []
    for latent, (h, w) in zip(padded_latents, shapes):
        parts.append(patchify_latent(latent, h, w, patch_size, channels))
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Timestep sampling
# ---------------------------------------------------------------------------

def sample_timesteps(
    shapes: List[Tuple[int, int]],
    timestep_shift: float,
    device,
    dtype,
) -> torch.Tensor:
    """Per-image t ~ sigmoid(N(0,1)) broadcast over each image's tokens, then shifted.

    Returns a flat tensor of length sum(h*w), one t per token.
    """
    raw_t = torch.randn(len(shapes), device=device, dtype=dtype)
    pieces = []
    for (h, w), t in zip(shapes, raw_t):
        pieces.append(torch.sigmoid(t).expand(h * w).clone())
    timesteps = torch.cat(pieces, dim=0)
    timesteps = timestep_shift * timesteps / (1.0 + (timestep_shift - 1.0) * timesteps)
    return timesteps


def build_noisy_latent(
    clean: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """x_t = (1 - t) * x_0 + t * noise, t broadcast over feature dim."""
    return (1.0 - timesteps[:, None]) * clean + timesteps[:, None] * noise


# ---------------------------------------------------------------------------
# Boundary loss
# ---------------------------------------------------------------------------

def patch_boundary_loss(
    x0_pred: torch.Tensor,
    shapes: List[Tuple[int, int]],
    patch_size: int,
    channels: int,
) -> torch.Tensor:
    """L1 across horizontal/vertical patch boundaries on predicted x_0.

    x0_pred: (sum_h*w, p*p*C). For each image's hxw grid, compares the right
    column of patch (i,j) to the left column of patch (i,j+1) and similarly
    for vertical neighbours. Encourages spatial smoothness in the latent
    that the VAE decoder will turn into pixels.
    """
    p = patch_size
    losses = []
    offset = 0
    for (h, w) in shapes:
        n = h * w
        x = x0_pred[offset : offset + n].reshape(h, w, p, p, channels)
        if w > 1:
            right_edge = x[:, :-1, :, p - 1, :]   # (h, w-1, p, C)
            left_edge = x[:, 1:, :, 0, :]          # (h, w-1, p, C)
            losses.append((right_edge - left_edge).abs().mean())
        if h > 1:
            bottom_edge = x[:-1, :, p - 1, :, :]   # (h-1, w, p, C)
            top_edge = x[1:, :, 0, :, :]           # (h-1, w, p, C)
            losses.append((bottom_edge - top_edge).abs().mean())
        offset += n
    if not losses:
        return x0_pred.sum() * 0
    return torch.stack(losses).mean()


__all__ = [
    "TimestepEmbedder",
    "PositionEmbedding",
    "patchify_latent",
    "unpatchify_latent",
    "pack_latents",
    "sample_timesteps",
    "build_noisy_latent",
    "patch_boundary_loss",
]
