"""UnifiedMoT main model: HunYuanVLMoT base + Flow Matching for image generation.

Architecture:
  * Inherits `transformers.models.hunyuan_vl_mot.HunYuanVLMoTModel` and uses
    its HYViT2 vision tower via `self.visual`. The inner language model is
    swapped to `MoTTextForCausalLM` so all decoder layers are 3-path
    (mlp_t / mlp_v / mlp_g).
  * Adds Flow Matching modules on the `ForConditionalGeneration` class:
    `time_embedder`, `vae2llm`, `llm2vae`, `latent_pos_embed`.
  * Forward supports both packed (P-Pack training) and padded modes.
  * Computes split text-CE + flow-MSE + optional patch-boundary L1 loss
    and exposes loss aggregates on the output (image_loss_sum / text_loss_sum /
    image_loss_count / text_loss_count) for trainer-side weighting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput

from transformers.models.hunyuan_vl_mot.modeling_hunyuan_vl_mot import (
    HY_VL_MOT_IMAGE_TOKEN_ID,
    HY_VL_MOT_VIDEO_TOKEN_ID,
    HunYuanVLMoTForConditionalGeneration,
    HunYuanVLMoTModel,
    HunYuanVLMoTPreTrainedModel,
    HunYuanVLMoTModelOutputWithPast,
)

# Side-effect imports: replace upstream attention + extend decoder/text-model
from . import attention_mot_packed  # noqa: F401  # pylint: disable=unused-import

from .config_unified_mot import UnifiedMoTConfig
from .modeling_text_model_mot import MoTTextForCausalLM
from .flow_matching_modules import (
    TimestepEmbedder,
    PositionEmbedding,
    pack_latents,
    sample_timesteps,
    build_noisy_latent,
    patch_boundary_loss,
)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class UnifiedMoTOutput(ModelOutput):
    """Output of UnifiedMoTForConditionalGeneration.

    Includes the standard CausalLMOutputWithPast fields plus split text/image
    loss aggregates so the trainer can apply per-modality weights.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # Split aggregates (sum-reduced; count of contributing tokens)
    image_loss_sum: Optional[torch.FloatTensor] = None
    image_loss_count: Optional[torch.LongTensor] = None
    text_loss_sum: Optional[torch.FloatTensor] = None
    text_loss_count: Optional[torch.LongTensor] = None
    boundary_loss: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# UnifiedMoTModel — wraps HunYuanVLMoTModel, swaps language_model to MoT version
# ---------------------------------------------------------------------------

class UnifiedMoTModel(HunYuanVLMoTModel):
    """HunYuanVLMoT wrapper with 3-path MoT decoder layers.

    `__init__` first calls upstream's super-__init__ to set up `self.visual`,
    `self.language_model`, and the misc plumbing, then **rebuilds**
    `self.language_model` as `MoTTextForCausalLM` so its decoder layers are
    `MoTDecoderLayer` (with `mlp_g`).

    Forward injects optional `flow_embeds` at `flow_positions` (absolute packed
    coords) before delegating to language_model. All packed kwargs pass
    straight through to the inner text model.
    """
    config_class = UnifiedMoTConfig

    def __init__(self, config: UnifiedMoTConfig):
        super().__init__(config)
        # Replace upstream's language_model with our MoT (3-path) version.
        # Done via `_from_config` to honor PretrainedConfig conventions.
        self.language_model = MoTTextForCausalLM._from_config(config)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        labels: Optional[torch.LongTensor] = None,
        # Packed-mode signals
        cu_seqlens: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        input_image_mask: Optional[torch.Tensor] = None,
        # Flow Matching
        flow_embeds: Optional[torch.FloatTensor] = None,    # (total_flow_tokens, D)
        flow_positions: Optional[torch.Tensor] = None,       # (M, 2) absolute packed coords
        g_seqlens: Optional[torch.Tensor] = None,            # (M, 2) absolute, used by attention
        **kwargs,
    ) -> HunYuanVLMoTModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # ----- Vision: encode input images/videos and scatter into embeds -----
        union_mask = None
        has_input_media = (pixel_values is not None) or (pixel_values_videos is not None)
        if self.training or has_input_media:
            if has_input_media:
                image_embeds, video_embeds, zero_feature = self.get_image_video_features(
                    pixel_values, pixel_values_videos,
                    image_grid_thw, video_grid_thw,
                    inputs_embeds.device, inputs_embeds.dtype,
                )
                if len(image_embeds) > 0:
                    image_embeds_cat = torch.cat(image_embeds, dim=0).to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    image_mask, _, union_mask = self.get_placeholder_mask(
                        input_ids, inputs_embeds, image_features=image_embeds_cat
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)
                if len(video_embeds) > 0:
                    video_embeds_cat = torch.cat(video_embeds, dim=0).to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    _, video_mask, union_mask = self.get_placeholder_mask(
                        input_ids, inputs_embeds, video_features=video_embeds_cat
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds_cat)
                inputs_embeds = inputs_embeds + zero_feature

        # ----- Flow Matching: inject pre-built flow embeds at flow_positions ---
        if flow_embeds is not None and flow_positions is not None and flow_positions.numel() > 0:
            seq_len = inputs_embeds.shape[1]
            replacement = inputs_embeds[0].clone()
            position_mask = torch.zeros(seq_len, device=inputs_embeds.device, dtype=torch.bool)
            # PERF: bulk D2H of flow_positions (small tensor) avoids 2 .item()
            # calls per range × ~25 ranges per forward.
            flow_positions_cpu = flow_positions.tolist()
            idx = 0
            for row in flow_positions_cpu:
                s = int(row[0])
                e = int(row[1])
                n = e - s
                replacement[s:e] = flow_embeds[idx : idx + n].to(inputs_embeds.dtype)
                position_mask[s:e] = True
                idx += n
            mask_f = position_mask.to(inputs_embeds.dtype).unsqueeze(-1)
            inputs_embeds = (
                inputs_embeds[0] * (1.0 - mask_f) + replacement * mask_f
            ).unsqueeze(0)

        # ----- modality_mask sanity ------------------------------------------
        if modality_mask is None:
            if union_mask is not None:
                # union_mask is bool (input image/video tokens). Promote to int.
                modality_mask = union_mask.long()
            else:
                modality_mask = torch.zeros(
                    inputs_embeds.shape[:-1], dtype=torch.long, device=inputs_embeds.device
                )

        # ----- Forward through inner language model -------------------------
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            labels=labels,
            modality_mask=modality_mask,
            cu_seqlens=cu_seqlens,
            sample_ids=sample_ids,
            g_seqlens=g_seqlens,
            input_image_mask=input_image_mask,
            **kwargs,
        )

        return HunYuanVLMoTModelOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


# ---------------------------------------------------------------------------
# UnifiedMoTForConditionalGeneration — main entry point
# ---------------------------------------------------------------------------

class UnifiedMoTForConditionalGeneration(HunYuanVLMoTForConditionalGeneration):
    """Full unified model: MoT VL backbone + Flow Matching head."""
    config_class = UnifiedMoTConfig

    # mlp_g and FM modules don't exist in upstream HunYuanVLMoT base checkpoints;
    # let `from_pretrained` skip them.
    _keys_to_ignore_on_load_missing = [
        r"time_embedder\..*",
        r"vae2llm\..*",
        r"llm2vae\..*",
        r"latent_pos_embed\..*",
        r".*\.mlp_g\..*",
        r".*\.input_layernorm_g\..*",
        r".*\.post_attention_layernorm_g\..*",
    ]

    def __init__(self, config: UnifiedMoTConfig):
        # Bypass upstream's __init__ (which would build a regular HunYuanVLMoTModel
        # and miss our 3-path layers). Init the PreTrainedModel base directly.
        HunYuanVLMoTPreTrainedModel.__init__(self, config)
        self.model = UnifiedMoTModel(config)
        self.config = config

        hidden_size = config.text_config.hidden_size

        # Flow Matching geometry (cached for hot path)
        self.latent_patch_size = config.latent_patch_size
        self.timestep_shift = config.timestep_shift
        self.max_latent_size = config.max_latent_size
        self.latent_channel = config.vae_z_channels
        # Effective downsample from pixel to packed-patch latent token grid
        self.latent_downsample = config.vae_image_downsample  # already includes patch_size
        self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel

        # FM modules — initialized from scratch (zero-loaded by from_pretrained)
        self.time_embedder = TimestepEmbedder(hidden_size)
        self.vae2llm = nn.Linear(self.patch_latent_dim, hidden_size)
        self.llm2vae = nn.Linear(hidden_size, self.patch_latent_dim)
        self.latent_pos_embed = PositionEmbedding(self.max_latent_size, hidden_size)

        # Loss weights
        self.boundary_loss_weight = config.boundary_loss_weight

        self.post_init()

    # -----------------------------------------------------------------
    # Helpers — delegate vision encoding/placeholder to upstream methods
    # -----------------------------------------------------------------

    def get_image_features(self, *args, **kwargs):
        return self.model.get_image_features(*args, **kwargs)

    def get_video_features(self, *args, **kwargs):
        return self.model.get_video_features(*args, **kwargs)

    def get_image_video_features(self, *args, **kwargs):
        return self.model.get_image_video_features(*args, **kwargs)

    def get_placeholder_mask(self, *args, **kwargs):
        return self.model.get_placeholder_mask(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.language_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.model.language_model.set_output_embeddings(value)

    # -----------------------------------------------------------------
    # FM input builder — turn padded VAE latents into flow_embeds
    # -----------------------------------------------------------------

    def _build_flow_embeds(
        self,
        padded_latents: List[torch.Tensor],
        latent_shapes: List[Tuple[int, int]],
        latent_positions: Optional[torch.Tensor],
        device,
        dtype,
    ):
        """Pack VAE latents → noisy x_t → vae2llm + time_embed + pos_embed.

        Returns (flow_embeds, vae_latent_clean, noise, timesteps).
        """
        clean = pack_latents(padded_latents, latent_shapes, self.latent_patch_size, self.latent_channel)
        clean = clean.to(device=device, dtype=dtype)
        noise = torch.randn_like(clean)
        timesteps = sample_timesteps(latent_shapes, self.timestep_shift, device, dtype)
        noisy = build_noisy_latent(clean, noise, timesteps)

        flow_embeds = self.vae2llm(noisy.to(self.vae2llm.weight.dtype)) + self.time_embedder(timesteps)
        if latent_positions is not None:
            flow_embeds = flow_embeds + self.latent_pos_embed(latent_positions)
        return flow_embeds, clean, noise, timesteps

    def _build_input_latent_embeds(
        self,
        padded_latents: List[torch.Tensor],
        latent_shapes: List[Tuple[int, int]],
        latent_positions: Optional[torch.Tensor],
        device,
        dtype,
    ):
        """For TI2I: clean (t=0) input image VAE latents projected into LLM space."""
        clean = pack_latents(padded_latents, latent_shapes, self.latent_patch_size, self.latent_channel)
        clean = clean.to(device=device, dtype=dtype)
        t_zero = torch.zeros(clean.shape[0], device=device, dtype=dtype)
        embeds = self.vae2llm(clean.to(self.vae2llm.weight.dtype)) + self.time_embedder(t_zero)
        if latent_positions is not None:
            embeds = embeds + self.latent_pos_embed(latent_positions)
        return embeds

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        labels: Optional[torch.LongTensor] = None,
        shift_labels: Optional[torch.LongTensor] = None,
        # Packed signals
        cu_seqlens: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        input_image_mask: Optional[torch.Tensor] = None,
        # FM training inputs
        use_flow_matching: bool = False,
        padded_latent: Optional[List[torch.Tensor]] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        vae_latent_indexes: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        vae_latent_positions: Optional[torch.Tensor] = None,
        flow_positions: Optional[torch.Tensor] = None,
        g_seqlens: Optional[torch.Tensor] = None,
        # TI2I (input image VAE latent conditioning)
        input_padded_latent: Optional[List[torch.Tensor]] = None,
        input_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        input_vae_latent_indexes: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        input_vae_latent_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> UnifiedMoTOutput:
        device = (input_ids if input_ids is not None else inputs_embeds).device
        flow_embeds = None
        vae_latent_clean = None
        noise = None
        timesteps = None

        # ---- TI2I clean input latents (t=0, gets injected at input_*_indexes) -
        # When provided, prepend them into inputs_embeds via flow_embeds path
        # using the same flow_positions mechanism. Many call sites only use one
        # of (input, generation) latents, so we handle them additively.
        input_flow_embeds = None
        if input_padded_latent is not None and input_vae_latent_shapes is not None:
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
                input_ids = None  # already materialized
            input_flow_embeds = self._build_input_latent_embeds(
                input_padded_latent, input_vae_latent_shapes,
                input_vae_latent_positions, device, inputs_embeds.dtype,
            )
            # Splice into inputs_embeds at the input_vae_latent_indexes positions
            if input_vae_latent_indexes is not None:
                b_idx, s_idx = input_vae_latent_indexes
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[b_idx, s_idx] = input_flow_embeds.to(inputs_embeds.dtype)

        # ---- Generation flow embeds (noisy x_t, used during FM training) ----
        if use_flow_matching and padded_latent is not None and patchified_vae_latent_shapes:
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
                input_ids = None
            flow_embeds, vae_latent_clean, noise, timesteps = self._build_flow_embeds(
                padded_latent, patchified_vae_latent_shapes,
                vae_latent_positions, device, inputs_embeds.dtype,
            )

        # ---- Forward through model -----------------------------------------
        # Prefer pre-built inputs_embeds when we touched them above.
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            labels=None,  # we compute loss ourselves
            cu_seqlens=cu_seqlens,
            sample_ids=sample_ids,
            modality_mask=modality_mask,
            input_image_mask=input_image_mask,
            flow_embeds=flow_embeds,
            flow_positions=flow_positions,
            g_seqlens=g_seqlens,
            **kwargs,
        )

        last_hidden_state = outputs.hidden_states  # actually carries last_hidden_state per upstream output type
        logits = outputs.logits

        # ---- Compute losses ------------------------------------------------
        text_loss_sum = None
        text_loss_count = None
        image_loss_sum = None
        image_loss_count = None
        boundary_loss = None
        loss = None

        # ---- Text CE loss (on shift_labels mask)
        train_labels = shift_labels if shift_labels is not None else labels
        if train_labels is not None and last_hidden_state is not None:
            flat_hs = last_hidden_state.reshape(-1, last_hidden_state.size(-1))
            flat_labels = train_labels.reshape(-1)
            valid = flat_labels >= 0
            n_valid = int(valid.sum().item())
            if n_valid > 0:
                hs_v = flat_hs[valid]
                lbl_v = flat_labels[valid]
                text_logits = self.get_output_embeddings()(hs_v).float()
                ce_per_tok = F.cross_entropy(text_logits, lbl_v, reduction="none")
                text_loss_sum = ce_per_tok.sum()
                text_loss_count = torch.tensor(n_valid, device=device, dtype=torch.long)

        # ---- Flow MSE loss + optional boundary loss
        if vae_latent_clean is not None and vae_latent_indexes is not None and last_hidden_state is not None:
            b_idx, s_idx = vae_latent_indexes
            mse_hidden = last_hidden_state[b_idx, s_idx]
            mse_preds = self.llm2vae(mse_hidden)
            target = noise - vae_latent_clean  # velocity target
            has_mse = timesteps > 0
            if has_mse.any():
                mse_pred_f32 = mse_preds[has_mse].float()
                target_f32 = target[has_mse].float()
                # Per-token MSE summed for split aggregation; trainer divides by count
                mse_per_tok = ((mse_pred_f32 - target_f32) ** 2).mean(dim=-1)
                image_loss_sum = mse_per_tok.sum()
                # PERF: avoid D2H .item() round-trip — has_mse.sum() is already
                # a GPU tensor, just cast it to long.
                image_loss_count = has_mse.sum().long()

            if self.boundary_loss_weight > 0.0:
                # Recover predicted x_0 = x_t - t * v_pred
                noisy = build_noisy_latent(vae_latent_clean.detach(), noise.detach(), timesteps.detach())
                x0_pred = noisy - timesteps[:, None].detach() * mse_preds
                boundary_loss = patch_boundary_loss(
                    x0_pred, patchified_vae_latent_shapes,
                    self.latent_patch_size, self.latent_channel,
                )

        # ---- Aggregate scalar loss -----------------------------------------
        # By default: mean text-CE + mean flow-MSE + boundary_weight * boundary
        parts = []
        if text_loss_sum is not None and text_loss_count is not None and text_loss_count.item() > 0:
            parts.append(self.config.text_loss_weight * (text_loss_sum / text_loss_count.float()))
        if image_loss_sum is not None and image_loss_count is not None and image_loss_count.item() > 0:
            parts.append(self.config.flow_loss_weight * (image_loss_sum / image_loss_count.float()))
        if boundary_loss is not None:
            parts.append(self.boundary_loss_weight * boundary_loss)
        if parts:
            loss = parts[0]
            for extra in parts[1:]:
                loss = loss + extra

        return UnifiedMoTOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=last_hidden_state,
            image_loss_sum=image_loss_sum,
            image_loss_count=image_loss_count,
            text_loss_sum=text_loss_sum,
            text_loss_count=text_loss_count,
            boundary_loss=boundary_loss,
        )


__all__ = [
    "UnifiedMoTModel",
    "UnifiedMoTForConditionalGeneration",
    "UnifiedMoTOutput",
    "HY_VL_MOT_IMAGE_TOKEN_ID",
    "HY_VL_MOT_VIDEO_TOKEN_ID",
]
