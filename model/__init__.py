"""HY-Unified model package.

New entry points — built on transformers 4.57's upstream
hunyuan_vl_mot base model, extended with the third (`_g`) generation MoT path
and Flow Matching modules.

Importing this package triggers the module-level replacement of upstream's
`_flash_attention_forward_mot` with our packed/padded/decode three-stage
version (see `attention_mot_packed`).

Public surface:
    UnifiedMoTConfig, UnifiedMoTTextConfig
    UnifiedMoTForConditionalGeneration, UnifiedMoTModel, UnifiedMoTOutput
    MoTDecoderLayer
    mask_apply_3way
    maybe_init_generation_path
    TimestepEmbedder, PositionEmbedding
    pack_latents, sample_timesteps, build_noisy_latent, patch_boundary_loss
"""
# Order matters: attention rebind must happen before any HunYuanVLMoT* instantiation.
from . import attention_mot_packed  # noqa: F401  (side effect)

from .config_unified_mot import UnifiedMoTConfig, UnifiedMoTTextConfig
from .modeling_decoder_mot import MoTDecoderLayer, mask_apply_3way
from .modeling_text_model_mot import MoTTextModel, MoTTextForCausalLM
from .modeling_unified_mot import (
    UnifiedMoTForConditionalGeneration,
    UnifiedMoTModel,
    UnifiedMoTOutput,
    HY_VL_MOT_IMAGE_TOKEN_ID,
    HY_VL_MOT_VIDEO_TOKEN_ID,
)
from .mot_init_utils import (
    maybe_init_generation_path,
    init_mlp_g_from_mlp_v_net2wider,
    verify_mlp_g_equals_mlp_v,
    checkpoint_has_g_keys,
)
from .flow_matching_modules import (
    TimestepEmbedder,
    PositionEmbedding,
    pack_latents,
    sample_timesteps,
    build_noisy_latent,
    patch_boundary_loss,
)


__all__ = [
    # Config
    "UnifiedMoTConfig", "UnifiedMoTTextConfig",
    # Models
    "UnifiedMoTForConditionalGeneration", "UnifiedMoTModel", "UnifiedMoTOutput",
    "MoTDecoderLayer", "MoTTextModel", "MoTTextForCausalLM",
    "mask_apply_3way",
    # Token IDs
    "HY_VL_MOT_IMAGE_TOKEN_ID", "HY_VL_MOT_VIDEO_TOKEN_ID",
    # Initialization
    "maybe_init_generation_path", "init_mlp_g_from_mlp_v_net2wider",
    "verify_mlp_g_equals_mlp_v", "checkpoint_has_g_keys",
    # Flow Matching utils
    "TimestepEmbedder", "PositionEmbedding",
    "pack_latents", "sample_timesteps", "build_noisy_latent", "patch_boundary_loss",
]
