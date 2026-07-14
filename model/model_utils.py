# SPDX-License-Identifier: Apache-2.0
#
# Independent reimplementation of standard sinusoidal position- and
# timestep-embedding utilities, written from their public mathematical
# definitions (see the per-symbol references below). This file is NOT derived
# from the DiT source tree and carries no third-party (CC BY-NC) copyright; it
# is released under Apache-2.0 like the rest of this package.
#
# References for the underlying formulas (math only, no code reuse):
#   * Sinusoidal position encoding — Vaswani et al., "Attention Is All You
#     Need" (2017), Section 3.5.
#   * Sinusoidal timestep embedding — Ho et al., "Denoising Diffusion
#     Probabilistic Models" (2020); the same closed form is reused across
#     diffusion and flow-matching models.
#
# The module layout, parameter names, and numerical outputs are intentionally
# identical to the previous version so that existing checkpoints load unchanged
# and inference results are bit-for-bit reproducible.

import math

import numpy as np
import torch
from torch import nn


# --------------------------------------------------------------------------- #
# Sinusoidal (sin-cos) position embeddings
#
# For a position p and channel index i the embedding interleaves
#   sin(p / 10000^(2i/d))  and  cos(p / 10000^(2i/d)),
# i.e. the classic Transformer positional encoding. The 2D variant encodes the
# height and width axes with half the channels each and concatenates them.
# --------------------------------------------------------------------------- #
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """1D sin-cos embedding for a flat array of positions.

    Args:
        embed_dim: even output dimension per position.
        pos: array of positions, any shape; flattened to (M,).
    Returns:
        (M, embed_dim) array, [sin | cos] halves concatenated.
    """
    assert embed_dim % 2 == 0
    # Inverse frequencies 1 / 10000^(k/(embed_dim/2)) for k = 0 .. embed_dim/2-1.
    inv_freq = np.arange(embed_dim // 2, dtype=np.float64)
    inv_freq /= embed_dim / 2.0
    inv_freq = 1.0 / 10000**inv_freq  # (embed_dim/2,)

    angles = np.einsum("m,d->md", pos.reshape(-1), inv_freq)  # outer product (M, D/2)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)  # (M, D)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """2D sin-cos embedding: half the channels encode each spatial axis."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """Sin-cos table for a ``grid_size x grid_size`` patch grid.

    Returns an ``(grid_size**2 [+ extra_tokens], embed_dim)`` array. When
    ``cls_token`` is set and ``extra_tokens > 0``, that many zero rows are
    prepended.
    """
    axis_h = np.arange(grid_size, dtype=np.float32)
    axis_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(axis_w, axis_h)  # width varies fastest
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


# --------------------------------------------------------------------------- #
# Timestep embedding
# --------------------------------------------------------------------------- #
class TimestepEmbedder(nn.Module):
    """Embed scalar (possibly fractional) timesteps into vectors.

    Sinusoidal frequency features are fed through a two-layer MLP with a SiLU
    non-linearity. This is the standard timestep conditioning used by diffusion
    and flow-matching models.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Sinusoidal features for a 1D tensor of (possibly fractional) timesteps.

        Args:
            t: (N,) tensor of timestep values.
            dim: output feature dimension.
            max_period: lowest angular frequency (longest period).
        Returns:
            (N, dim) tensor of [cos | sin] features (zero-padded if ``dim`` is odd).
        """
        half = dim // 2
        # Geometrically spaced frequencies over [1, 1/max_period].
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        freq_features = self.timestep_embedding(t, self.frequency_embedding_size)
        freq_features = freq_features.to(next(self.mlp.parameters()).dtype)
        return self.mlp(freq_features)


class PositionEmbedding(nn.Module):
    """Fixed 2D sin-cos position table exposed as a lookup by position id."""

    def __init__(self, max_num_patch_per_side, hidden_size):
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        # Build the (non-trainable) table eagerly. If it were created lazily it
        # could stay on the meta device through transformers'
        # from_pretrained(dtype=...) path and later materialize as uninitialized
        # memory when moved to the GPU.
        table = get_2d_sincos_pos_embed(hidden_size, max_num_patch_per_side)
        self.pos_embed = nn.Parameter(
            torch.from_numpy(table).float(),
            requires_grad=False,
        )

    def _reset_parameters(self):
        """Recompute the table after a meta-init path (call post-from_pretrained)."""
        table = get_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side)
        on_meta = self.pos_embed.is_meta or self.pos_embed.device.type == "meta"
        materialized = torch.from_numpy(table).to(
            device="cpu" if on_meta else self.pos_embed.device,
            dtype=torch.float32 if on_meta else self.pos_embed.dtype,
        )
        if on_meta:
            self.pos_embed = nn.Parameter(materialized.float(), requires_grad=False)
        else:
            self.pos_embed.data.copy_(materialized.to(self.pos_embed.dtype))

    def forward(self, position_ids):
        return self.pos_embed[position_ids]
