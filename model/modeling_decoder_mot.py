"""MoT decoder layer with three-path routing (text / vision-input / vision-gen).

Subclasses `HunYuanVLMoTDecoderLayer` from upstream to add the third (`_g`)
path used by Flow Matching. Upstream provides `mlp / mlp_v` and matching
LayerNorms; we add `mlp_g / input_layernorm_g / post_attention_layernorm_g`.

`modality_mask` semantics:
  * 0 = text token            → mlp / input_layernorm / post_attention_layernorm
  * 1 = input vision token    → mlp_v / *_v
  * 2 = generated vision token (FM latent) → mlp_g / *_g

Attention QKVO is **not** extended to a third path — generation tokens reuse
the `_v` projection (matches the existing HY-Unified design where
`q_proj_v / k_proj_v / v_proj_v / o_proj_v` cover both input and generated
vision tokens). We just need to convert the int modality mask to a bool
"vision" mask before calling `self.self_attn`.

mlp_g may optionally have a wider intermediate (Net2Wider scale-up). When
`config.mlp_g_intermediate_size` is set and larger than `intermediate_size`,
we build mlp_g with a cloned config that overrides `intermediate_size`.
"""
from __future__ import annotations

import copy
from typing import Optional

import torch

from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.deprecation import deprecate_kwarg

from transformers.models.hunyuan_vl_mot.modeling_hunyuan_vl_mot import (
    HunYuanVLMoTDecoderLayer,
    HunYuanVLMoTMLP,
    HunYuanVLMoTRMSNorm,
)

# Side-effect import: replaces upstream's _flash_attention_forward_mot
from . import attention_mot_packed  # noqa: F401  # pylint: disable=unused-import


# ---------------------------------------------------------------------------
# Three-way mask_apply
# ---------------------------------------------------------------------------

def mask_apply_3way(
    hidden_states: torch.Tensor,
    modality_mask: Optional[torch.Tensor],
    text_funcs,
    vision_funcs,
    gen_funcs,
    out_dims=None,
    padding_mask: Optional[torch.Tensor] = None,
):
    """Routes tokens to text/vision/generation function lists by modality_mask.

    hidden_states: (B, S, D)
    modality_mask: (B, S) int tensor with values in {0, 1, 2}, or None
        0 = text, 1 = vision input, 2 = vision generation
    padding_mask: (B, S) int tensor, 1 = valid token, 0 = padding

    Returns a list of stacked (B, S, out_d) tensors — one per `text_funcs[i]`.
    """
    if modality_mask is None:
        # All-text: skip routing entirely
        return [text_funcs[i](hidden_states) for i in range(len(text_funcs))]

    bsz, seq_len, hidden_dim = hidden_states.size()
    flat = hidden_states.reshape(bsz * seq_len, hidden_dim)
    mask_flat = modality_mask.reshape(bsz * seq_len)
    if padding_mask is not None:
        valid_flat = padding_mask.reshape(bsz * seq_len).bool()
    else:
        valid_flat = None

    placeholder = hidden_states[0:1, 0:1, :]  # (1, 1, D)
    zero_feature = 0

    num_outputs = len(text_funcs)
    if out_dims is None:
        out_dims_resolved = [hidden_dim] * num_outputs
    else:
        out_dims_resolved = list(out_dims)

    # Pre-allocate output buffers (empty, not zeros — we overwrite all valid
    # positions and the rest are masked out by the caller / padding).
    out_flat = [
        torch.empty(bsz * seq_len, od, device=flat.device, dtype=flat.dtype)
        for od in out_dims_resolved
    ]
    # Padding positions need to be zeroed — we won't touch them in the
    # gather/scatter below. Cheaper than zeroing the whole tensor: only
    # zero the rows that won't be hit by any of the three modalities.
    if valid_flat is not None:
        invalid_flat = ~valid_flat
        if invalid_flat.any():
            for buf in out_flat:
                buf[invalid_flat] = 0
    # else: all rows will be hit by exactly one of {text, vision, gen}.

    def _dispatch(idx_mask, funcs):
        """Run `funcs` on rows selected by `idx_mask`, scatter back. If no
        rows are selected, multiply through a placeholder so the params still
        receive grad (avoids "unused parameter" DDP errors)."""
        nonlocal zero_feature
        if idx_mask.any():
            hs_sel = flat[idx_mask]
            for i, fn in enumerate(funcs):
                out_flat[i][idx_mask] = fn(hs_sel)
        else:
            for fn in funcs:
                zero_feature = zero_feature + fn(placeholder).mean() * 0

    # Text: mask == 0
    text_idx = (mask_flat == 0)
    if valid_flat is not None:
        text_idx = text_idx & valid_flat
    _dispatch(text_idx, text_funcs)

    # Vision input: mask == 1
    vis_idx = (mask_flat == 1)
    if valid_flat is not None:
        vis_idx = vis_idx & valid_flat
    _dispatch(vis_idx, vision_funcs)

    # Generation: mask == 2
    gen_idx = (mask_flat == 2)
    if valid_flat is not None:
        gen_idx = gen_idx & valid_flat
    _dispatch(gen_idx, gen_funcs)

    result = [out.view(bsz, seq_len, -1) for out in out_flat]
    result[0] = result[0] + zero_feature
    return result


# ---------------------------------------------------------------------------
# Decoder layer subclass
# ---------------------------------------------------------------------------

def _make_g_config(config, mlp_g_intermediate_size: int):
    """Clone config with a wider intermediate_size for mlp_g (Net2Wider).

    MUST be `copy.deepcopy`, NOT `copy.copy`. Reason:
      HunYuanVLMoTConfig.__setattr__ is a proxy that re-routes any attribute
      writes (when the key is in text_config.__dict__) into self.text_config.
      Shallow-copying the outer config keeps `g_cfg.text_config is
      config.text_config` — same object — so the subsequent
      `g_cfg.intermediate_size = ...` setattr ends up mutating the SHARED
      text_config used by every other decoder layer's mlp/mlp_v construction.
      Layer 0 escapes (its mlp/mlp_v are built before _make_g_config runs),
      but layer 1+ then sees text_config.intermediate_size = 12288 and builds
      mlp/mlp_v at the wider size — at that point ckpt weights (which are 6144
      for mlp/mlp_v) no longer match and `from_pretrained` fails with size
      mismatch. Deepcopy the whole config tree to break this aliasing.
    """
    g_cfg = copy.deepcopy(config)
    g_cfg.intermediate_size = int(mlp_g_intermediate_size)
    return g_cfg


class MoTDecoderLayer(HunYuanVLMoTDecoderLayer):
    """Three-path decoder layer: text / vision-input / vision-gen.

    Adds `mlp_g`, `input_layernorm_g`, `post_attention_layernorm_g` on top of
    upstream's two-path layer. `modality_mask` is interpreted as int{0,1,2}
    instead of bool.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        # Text-config-aware lookup (HunYuanVLMoTConfig proxies these to text_config)
        intermediate_size = getattr(config, "intermediate_size", None)
        mlp_g_inter = getattr(config, "mlp_g_intermediate_size", None)

        if mlp_g_inter is None or intermediate_size is None or mlp_g_inter == intermediate_size:
            g_cfg = config
        else:
            assert mlp_g_inter >= intermediate_size, (
                f"mlp_g_intermediate_size ({mlp_g_inter}) must be >= "
                f"intermediate_size ({intermediate_size}); shrinking is not supported."
            )
            g_cfg = _make_g_config(config, mlp_g_inter)

        self.mlp_g = HunYuanVLMoTMLP(g_cfg)
        self.input_layernorm_g = HunYuanVLMoTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_g = HunYuanVLMoTRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        modality_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Pull padding_mask from attention_mask dict for GEMM exclusion
        padding_mask = None
        if isinstance(attention_mask, dict):
            pm = attention_mask.get("padding_mask", None)
            if pm is not None and hidden_states.shape[1] == pm.shape[1]:
                padding_mask = pm

        # Convert int modality_mask to bool for attention QKVO routing
        # (attention has only 2 paths: text vs vision; gen tokens go through _v)
        attn_modality_mask = None
        if modality_mask is not None:
            attn_modality_mask = (modality_mask > 0)

        residual = hidden_states

        # Pre-attention LayerNorm — three paths
        hidden_states = mask_apply_3way(
            hidden_states, modality_mask,
            [self.input_layernorm],
            [self.input_layernorm_v],
            [self.input_layernorm_g],
            padding_mask=padding_mask,
        )[0]

        # Self-attention (upstream's two-path attention with bool mask)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            modality_mask=attn_modality_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP — three paths (LayerNorm + MLP fused per path to match upstream layout)
        residual = hidden_states
        hidden_states = mask_apply_3way(
            hidden_states, modality_mask,
            [lambda x: self.mlp(self.post_attention_layernorm(x))],
            [lambda x: self.mlp_v(self.post_attention_layernorm_v(x))],
            [lambda x: self.mlp_g(self.post_attention_layernorm_g(x))],
            padding_mask=padding_mask,
        )[0]
        hidden_states = residual + hidden_states

        # Zero out padding positions to keep residuals clean (matches upstream behavior)
        if padding_mask is not None and hidden_states.shape[1] == padding_mask.shape[1]:
            hidden_states = hidden_states * padding_mask.unsqueeze(-1)

        return hidden_states


__all__ = ["MoTDecoderLayer", "mask_apply_3way"]
