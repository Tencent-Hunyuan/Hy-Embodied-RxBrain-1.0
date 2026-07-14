"""Inner text-decoder subclass that threads packed kwargs into the causal_mask dict.

The upstream `_HunYuanVLMoTTextModel.forward` builds:

    causal_mask = {"v_seqlens": visual_segs, "padding_mask": attention_mask}

and passes it to each decoder layer. For P-Pack training we additionally need
`cu_seqlens / sample_ids / g_seqlens / input_image_mask` in that dict so our
patched flash-attention function can dispatch to the packed code path.

Rather than monkey-patching upstream's forward, we subclass `_HunYuanVLMoTTextModel`
and `_HunYuanVLMoTTextForCausalLM` cleanly and replace the decoder layers with
`MoTDecoderLayer` (three-path mlp_t / mlp_v / mlp_g) inside __init__.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from transformers.models.hunyuan_vl_mot.modeling_hunyuan_vl_mot import (
    _HunYuanVLMoTTextModel,
    _HunYuanVLMoTTextForCausalLM,
    _modality_mask_to_segments,
)

from .modeling_decoder_mot import MoTDecoderLayer


class MoTTextModel(_HunYuanVLMoTTextModel):
    """Pure text decoder using MoTDecoderLayer (3-path) and packed kwargs.

    Drop-in replacement for `_HunYuanVLMoTTextModel`:
      * Replaces all decoder layers with `MoTDecoderLayer`
      * Forward accepts `cu_seqlens / sample_ids / g_seqlens / input_image_mask`
        and threads them into the `attention_mask` dict that decoder layers see
      * `modality_mask` may be int{0,1,2} (text/vision-input/vision-gen);
        v_seqlens are derived from `input_image_mask` if provided, else from
        `modality_mask > 0`
    """

    def __init__(self, config):
        super().__init__(config)
        # Replace decoder layers with MoT (3-path). Init weights match upstream
        # for the shared keys; mlp_g/_g get default init (overwritten later by
        # Net2Wider in mot_init_utils.maybe_init_generation_path).
        self.layers = nn.ModuleList(
            [MoTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        modality_mask: Optional[torch.Tensor] = None,
        # Packed-mode signals
        cu_seqlens: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
        g_seqlens: Optional[torch.Tensor] = None,
        input_image_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Position IDs:
        #   * Packed mode (cu_seqlens given): caller must pass position_ids (per-sample arange)
        #   * Padded B>1: derive from attention_mask cumsum
        #   * B==1 / decode: cache_position
        if position_ids is None:
            if cu_seqlens is not None:
                raise ValueError("Packed mode requires position_ids from the collator.")
            if attention_mask is not None and attention_mask.shape[0] > 1:
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids = position_ids.clamp(min=0)
                seq_len = inputs_embeds.shape[1]
                if position_ids.shape[1] > seq_len:
                    position_ids = position_ids[:, -seq_len:]
            else:
                position_ids = cache_position.unsqueeze(0)
        text_position_ids = position_ids

        if modality_mask is None:
            modality_mask = torch.zeros(
                inputs_embeds.shape[:-1], dtype=torch.long, device=inputs_embeds.device
            )

        # v_seqlens source: input_image_mask (input images only) when training
        # flow generation; otherwise modality_mask > 0.
        if input_image_mask is not None:
            vis_mask = input_image_mask.bool()
        else:
            vis_mask = modality_mask > 0
        visual_segs = _modality_mask_to_segments(vis_mask)

        # Truncate modality_mask if shape mismatch (decode KV cache)
        seq_len = inputs_embeds.shape[1]
        if modality_mask.shape[1] > seq_len:
            modality_mask = modality_mask[:, -seq_len:]

        # Build extended causal_mask dict with packed signals.
        # When cu_seqlens is set, padding_mask MUST be None (we treat as packed).
        # PERF: pre-compute max_seqlen ONCE here so the per-layer attention
        # forward doesn't redo `.max().item()` (a D2H sync) 32 times.
        max_seqlen_packed = None
        if cu_seqlens is not None:
            with torch.no_grad():
                max_seqlen_packed = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        causal_mask = {
            "v_seqlens": visual_segs,
            "g_seqlens": g_seqlens,
            "cu_seqlens": cu_seqlens,
            "sample_ids": sample_ids,
            "padding_mask": attention_mask if cu_seqlens is None else None,
            "max_seqlen_packed": max_seqlen_packed,
        }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, text_position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                modality_mask=modality_mask,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class MoTTextForCausalLM(_HunYuanVLMoTTextForCausalLM):
    """Inner text + lm_head wrapper using MoTTextModel.

    Threads packed kwargs and `shift_labels` through. When `shift_labels` is
    provided, computes split text/image loss aggregates so the outer wrapper
    can report them separately.
    """

    def __init__(self, config):
        super().__init__(config)
        # Replace inner text model with MoT (3-path) version
        self.model = MoTTextModel(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        modality_mask: Optional[torch.Tensor] = None,
        # Packed kwargs threaded to inner MoTTextModel
        cu_seqlens: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
        g_seqlens: Optional[torch.Tensor] = None,
        input_image_mask: Optional[torch.Tensor] = None,
        # Optional pre-shifted labels (used when image regions need -100 masking)
        shift_labels: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            modality_mask=modality_mask,
            cu_seqlens=cu_seqlens,
            sample_ids=sample_ids,
            g_seqlens=g_seqlens,
            input_image_mask=input_image_mask,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # Choose label source: explicit shift_labels (collator-prepared, with
        # image-region -100 masking) takes precedence over `labels`.
        train_labels = shift_labels if shift_labels is not None else labels

        if train_labels is not None:
            flat_hs = hidden_states.reshape(-1, hidden_states.size(-1))
            flat_labels = train_labels.reshape(-1)
            valid = flat_labels >= 0
            if valid.sum() == 0:
                flat_hs_v = flat_hs[:1]
                flat_labels_v = flat_labels[:1]
            else:
                flat_hs_v = flat_hs[valid]
                flat_labels_v = flat_labels[valid]
            logits = self.lm_head(flat_hs_v)
            loss = self.loss_function(
                logits=logits, labels=flat_labels_v, vocab_size=self.config.vocab_size, **kwargs
            )
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            loss = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.last_hidden_state,
        )


__all__ = ["MoTTextModel", "MoTTextForCausalLM"]
