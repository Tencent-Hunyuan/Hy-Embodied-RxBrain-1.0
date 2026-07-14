"""Configuration for UnifiedMoT.

Subclasses transformers.models.hunyuan_vl_mot config to add HY-Unified
project-level fields used by the Flow Matching path:

  - mlp_g_intermediate_size / mlp_g_init_noise_std  (Net2Wider on generation MLP)
  - boundary_loss_weight                            (patch-boundary smoothness L1)
  - latent_patch_size / max_latent_size             (FM latent grid)
  - vae_image_downsample / vae_z_channels           (VAE encoder spec)
  - timestep_shift                                  (t-distribution shift for FM)
  - use_moe                                         (toggle moe_layers wrapper)
  - flow_loss_weight / text_loss_weight             (loss aggregation knobs)

The text-level fields (mlp_g_*) live on text_config because the proxy in
HunYuanVLMoTConfig surfaces them at the top level too.
"""
from __future__ import annotations

from typing import Optional

from transformers.models.hunyuan_vl_mot.configuration_hunyuan_vl_mot import (
    HunYuanVLMoTConfig,
    HunYuanVLMoTTextConfig,
    HunYuanVLMoTVisionConfig,
)


class UnifiedMoTTextConfig(HunYuanVLMoTTextConfig):
    """Extends upstream text config with the three-path MoT (mlp_g) knobs."""

    model_type = "unified_mot"

    def __init__(
        self,
        mlp_g_intermediate_size: Optional[int] = None,
        mlp_g_init_noise_std: float = 1e-4,
        use_moe: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mlp_g_intermediate_size = mlp_g_intermediate_size
        self.mlp_g_init_noise_std = float(mlp_g_init_noise_std)
        self.use_moe = bool(use_moe)


class UnifiedMoTConfig(HunYuanVLMoTConfig):
    """Top-level config for the unified flow-matching model.

    Inherits HunYuanVLMoTConfig (text_config + vision_config + token ids) and
    layers on Flow-Matching specific fields.
    """

    model_type = "unified_mot"
    sub_configs = {
        "vision_config": HunYuanVLMoTVisionConfig,
        "text_config": UnifiedMoTTextConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        # Flow Matching geometry
        latent_patch_size: int = 2,
        max_latent_size: int = 32,
        vae_image_downsample: int = 16,
        vae_z_channels: int = 16,
        timestep_shift: float = 1.0,
        # Loss aggregation
        boundary_loss_weight: float = 0.0,
        flow_loss_weight: float = 1.0,
        text_loss_weight: float = 1.0,
        # Special token used as the FM latent placeholder in the input sequence.
        # Defaults to upstream's `latent_token_id` (120690) so the same checkpoint
        # works without changes.
        flow_latent_placeholder_id: Optional[int] = None,
        **kwargs,
    ):
        # If text_config wasn't pre-built, route the **kwargs through so caller-level
        # fields like mlp_g_intermediate_size still land on text_config.
        if text_config is None:
            text_config = UnifiedMoTTextConfig(**kwargs)
        elif isinstance(text_config, dict):
            text_config = UnifiedMoTTextConfig(**text_config)

        super().__init__(text_config=text_config, vision_config=vision_config, **kwargs)

        self.latent_patch_size = int(latent_patch_size)
        self.max_latent_size = int(max_latent_size)
        self.vae_image_downsample = int(vae_image_downsample)
        self.vae_z_channels = int(vae_z_channels)
        self.timestep_shift = float(timestep_shift)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.flow_loss_weight = float(flow_loss_weight)
        self.text_loss_weight = float(text_loss_weight)
        self.flow_latent_placeholder_id = (
            int(flow_latent_placeholder_id)
            if flow_latent_placeholder_id is not None
            else int(self.latent_token_id)
        )


__all__ = ["UnifiedMoTConfig", "UnifiedMoTTextConfig"]
