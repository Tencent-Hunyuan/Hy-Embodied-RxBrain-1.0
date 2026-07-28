"""Packed (P-Pack) data collator.

Consumes per-sample dicts emitted by `train.dataset.UnifiedDataset.__getitem__`
(or `ParquetT2IDataset`) and concatenates them into a single (1, T) packed batch
with the signals `UnifiedMoTForConditionalGeneration` expects.

Why packed (not padded):
  * No wasted FFN / LayerNorm / MoT routing on padding tokens
  * No cross-sample attention leakage — `cu_seqlens` enforces sample boundaries
    in the patched flash-attention function (`model/attention_mot_packed.py`)
  * Each sample's `position_ids` resets to 0 → RoPE rotates correctly per sample

Per-sample input dict (from `train.dataset.UnifiedDataset`):
  * input_ids: 1D long tensor (variable length)
  * attention_mask: 1D long tensor (all 1s)
  * labels: 1D long tensor (-100 in image regions, token ids elsewhere)
  * use_flow_matching: bool
  * vae_image_infos: list of dicts with
      - image_tensor: (C_vae_input, H, W) float tensor
      - latent_shape: (h, w) tuple
      - latent_start_idx: int — position of first latent token in this sample
      - num_latent_tokens: int — h * w
      - latent_position_ids: 1D long tensor of pos ids into latent grid
  * input_vae_image_infos: same shape as vae_image_infos but for TI2I input images
  * pixel_values: optional (N_img, ...) tensor (input images for understanding)
  * image_grid_thw: optional (N_img, 3) tensor
  * pixel_values_videos / video_grid_thw: optional (videos)

Packed batch output:
  * input_ids: (1, T) long
  * attention_mask: None (packed-mode signal)
  * labels: (1, T) long
  * shift_labels: (1, T) long — per-sample shift (next-token target); -100 in
    image regions and at the last position of each sample
  * position_ids: (1, T) long — per-sample arange(0, L_i)
  * cu_seqlens: (N+1,) int32 — cumulative sequence lengths
  * sample_ids: (1, T) int32 — which sample each token belongs to
  * modality_mask: (1, T) long — 0=text, 1=input vision, 2=generated latent
  * input_image_mask: (1, T) bool — True only at input image/video placeholders
  * flow_positions: (M, 2) int32 — absolute packed coords of each gen latent block
  * g_seqlens: (M, 2) int32 — same as flow_positions (used by attention)
  * vae_latent_indexes: tuple (batch_idx, seq_idx) both 1D long, addressing
    flat (1, T) positions for loss extraction (batch_idx is all 0)
  * vae_latent_positions: 1D long tensor — 2D pos ids concatenated across images
  * padded_images: list of (C, H, W) float tensors — RAW pixel tensors after
    VAEImageTransform (normalized to [-1, 1]). The training loop is expected
    to encode these via the VAE before passing to the model as `padded_latent`.
  * patchified_vae_latent_shapes: list of (h, w) tuples
  * use_flow_matching: bool
  * pixel_values / image_grid_thw / pixel_values_videos / video_grid_thw:
    concatenated across samples (or None if none)
  * (TI2I) input_padded_latent / input_vae_latent_shapes /
    input_vae_latent_indexes / input_vae_latent_positions
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_1d_long(x) -> torch.Tensor:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.reshape(-1).to(torch.long)
    return torch.tensor(x, dtype=torch.long)


def _cat_or_none(items):
    items = [x for x in items if x is not None]
    return torch.cat(items, dim=0) if items else None


# ---------------------------------------------------------------------------
# Packed collator
# ---------------------------------------------------------------------------

@dataclass
class UnifiedDataCollator:
    """Collator that packs `batch_size` per-sample dicts into one (1, T) row.

    Args:
        max_total_tokens: hard cap on packed length. If the incoming samples
            sum to more than this, drop trailing samples (or truncate the only
            sample to fit). Set to None to disable the cap.
        latent_placeholder_id: input_ids value used to mark FM latent positions
            (modality_mask=2). Default matches `train.dataset.LATENT_PLACEHOLDER_ID`.
        input_latent_placeholder_id: same but for TI2I input image VAE latent.
        input_image_placeholder_ids: ids used by the upstream HunYuanVL processor
            for input image/video placeholder tokens (modality_mask=1).
    """

    max_total_tokens: Optional[int] = None
    # Upstream HunYuanVLMoTConfig.latent_token_id; matches train.dataset.LATENT_PLACEHOLDER_ID
    latent_placeholder_id: int = 120690
    input_latent_placeholder_id: int = 120691
    input_image_placeholder_ids: tuple = (120687, 120688)  # IMAGE / VIDEO placeholders
    # Mark positions where input ViT features get scattered too:
    # the upstream HunYuanVLMoT processor uses 120687 (image) / 120688 (video)
    # as the IDs that `masked_scatter` overwrites with vision features.

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            return {}

        # ---- Step 1: collect raw 1D sequences and lengths ------------------
        per_sample: List[Dict[str, Any]] = []
        for f in features:
            ids = _to_1d_long(f["input_ids"])
            labels = _to_1d_long(f.get("labels"))
            if labels is None:
                labels = ids.clone()
            per_sample.append({
                "ids": ids,
                "labels": labels,
                "vae_image_infos": f.get("vae_image_infos", []) or [],
                "input_vae_image_infos": f.get("input_vae_image_infos", []) or [],
                "pixel_values": f.get("pixel_values"),
                "image_grid_thw": f.get("image_grid_thw"),
                "pixel_values_videos": f.get("pixel_values_videos"),
                "video_grid_thw": f.get("video_grid_thw"),
                "use_flow_matching": bool(f.get("use_flow_matching", False)),
            })

        # ---- Step 2: enforce max_total_tokens cap --------------------------
        if self.max_total_tokens is not None:
            kept: List[Dict[str, Any]] = []
            running = 0
            for s in per_sample:
                L = s["ids"].shape[0]
                if running + L <= self.max_total_tokens:
                    kept.append(s)
                    running += L
                else:
                    break
            if not kept:
                # The FIRST sample alone exceeds the cap. Truncating its input_ids is
                # safe ONLY for samples WITHOUT input-ViT features: pixel_values/
                # image_grid_thw are NOT trimmable here, so cutting their placeholder
                # tokens breaks placeholder↔feature parity -> model get_placeholder_mask
                # HARD-RAISES "Image feature/token count mismatch" and aborts the whole
                # 312-GPU job (the step-~50 crash loop). (vae_image_infos / generated
                # latents ARE trimmable.) So:
                #   1) prefer any sample in the batch that fits uncut;
                #   2) else prefer any sample WITHOUT input vision (safe to truncate);
                #   3) else (all remaining carry input vision and none fit) drop them
                #      and emit a 1-token no-op pack — DDP-safe (every rank still does
                #      one forward/backward) and never corrupt. These over-cap vision
                #      samples are meant to be pre-filtered by UnifiedDataset._est_seq_len.
                def _has_input_vision(s):
                    pv = s.get("pixel_values"); pvv = s.get("pixel_values_videos")
                    return (pv is not None and getattr(pv, "shape", [0])[0] > 0) or \
                           (pvv is not None and getattr(pvv, "shape", [0])[0] > 0)
                fit = next((s for s in per_sample
                            if s["ids"].shape[0] <= self.max_total_tokens), None)
                if fit is not None:
                    kept = [fit]
                else:
                    s = next((s for s in per_sample if not _has_input_vision(s)), None)
                    if s is not None:
                        cap = self.max_total_tokens
                        s["ids"] = s["ids"][:cap]
                        s["labels"] = s["labels"][:cap]
                        s["vae_image_infos"] = [
                            info for info in s["vae_image_infos"]
                            if info["latent_start_idx"] + info["num_latent_tokens"] <= cap
                        ]
                        s["input_vae_image_infos"] = [
                            info for info in s["input_vae_image_infos"]
                            if info["latent_start_idx"] + info["num_latent_tokens"] <= cap
                        ]
                        kept = [s]
                    else:
                        # All-oversized + all carry input vision: emit a no-op pack.
                        eos = 0
                        kept = [{
                            "ids": torch.zeros(1, dtype=torch.long),
                            "labels": torch.full((1,), -100, dtype=torch.long),
                            "vae_image_infos": [], "input_vae_image_infos": [],
                            "pixel_values": None, "image_grid_thw": None,
                            "pixel_values_videos": None, "video_grid_thw": None,
                            "use_flow_matching": False,
                        }]
            per_sample = kept

        N = len(per_sample)
        lengths = [s["ids"].shape[0] for s in per_sample]

        # ---- Step 3: cu_seqlens / position_ids / sample_ids ----------------
        cu_list = [0]
        for L in lengths:
            cu_list.append(cu_list[-1] + L)
        T = cu_list[-1]

        cu_seqlens = torch.tensor(cu_list, dtype=torch.int32)
        position_ids = torch.cat([torch.arange(L, dtype=torch.long) for L in lengths]).unsqueeze(0)
        sample_ids = torch.cat(
            [torch.full((L,), i, dtype=torch.int32) for i, L in enumerate(lengths)]
        ).unsqueeze(0)

        # ---- Step 4: input_ids / labels concatenated -----------------------
        input_ids = torch.cat([s["ids"] for s in per_sample]).unsqueeze(0)
        labels = torch.cat([s["labels"] for s in per_sample]).unsqueeze(0)

        # ---- Step 5: shift_labels (per-sample shift; last pos in each = -100)
        # shift_labels[i] = labels[i+1] when i+1 is in the same sample; -100 otherwise
        shift_labels = torch.full_like(labels, -100)
        for i, off in enumerate(cu_list[:-1]):
            L = lengths[i]
            if L > 1:
                shift_labels[0, off : off + L - 1] = labels[0, off + 1 : off + L]
            # last token of each sample stays -100
        # Additionally, force -100 at image-region positions (latent placeholders)
        # — even if the input_ids next token is non-image, we don't want to predict
        # latent placeholders as text tokens (they aren't real vocab).
        is_latent = (input_ids == self.latent_placeholder_id) | (
            input_ids == self.input_latent_placeholder_id
        )
        shift_labels = torch.where(is_latent, torch.full_like(shift_labels, -100), shift_labels)

        # ---- Step 6: modality_mask / input_image_mask ----------------------
        # 0 = text, 1 = input vision (placeholder + input VAE latent), 2 = gen latent
        modality_mask = torch.zeros(1, T, dtype=torch.long)
        input_image_mask = torch.zeros(1, T, dtype=torch.bool)

        for placeholder_id in self.input_image_placeholder_ids:
            mask = input_ids == placeholder_id
            modality_mask[mask] = 1
            input_image_mask[mask] = True

        mask_input_latent = input_ids == self.input_latent_placeholder_id
        modality_mask[mask_input_latent] = 1
        input_image_mask[mask_input_latent] = True

        mask_gen_latent = input_ids == self.latent_placeholder_id
        modality_mask[mask_gen_latent] = 2

        # ---- Step 7: flow_positions / vae_latent_indexes / latent_positions
        # All positions in absolute packed coords. vae_latent_indexes is the
        # (batch_idx, seq_idx) tuple the model uses to slice out generation
        # hidden states for the FM MSE loss; batch_idx is always 0 in packed mode.
        flow_positions_list: List[List[int]] = []
        vae_batch_idx: List[torch.Tensor] = []
        vae_seq_idx: List[torch.Tensor] = []
        vae_latent_positions: List[torch.Tensor] = []
        patchified_vae_latent_shapes: List = []
        padded_images: List[torch.Tensor] = []  # RAW pixel tensors; trainer VAE-encodes

        # Same containers for TI2I input image VAE latents
        input_flow_positions_list: List[List[int]] = []
        input_vae_batch_idx: List[torch.Tensor] = []
        input_vae_seq_idx: List[torch.Tensor] = []
        input_vae_latent_positions: List[torch.Tensor] = []
        input_vae_latent_shapes: List = []
        input_padded_images: List[torch.Tensor] = []  # RAW pixel tensors (TI2I input)

        has_flow = False
        for i, s in enumerate(per_sample):
            off = cu_list[i]
            for info in s["vae_image_infos"]:
                start_abs = off + int(info["latent_start_idx"])
                n_tok = int(info["num_latent_tokens"])
                end_abs = start_abs + n_tok
                flow_positions_list.append([start_abs, end_abs])
                vae_batch_idx.append(torch.zeros(n_tok, dtype=torch.long))
                vae_seq_idx.append(torch.arange(start_abs, end_abs, dtype=torch.long))
                vae_latent_positions.append(_to_1d_long(info["latent_position_ids"]))
                patchified_vae_latent_shapes.append(tuple(info["latent_shape"]))
                padded_images.append(info["image_tensor"])
                has_flow = True
            for info in s["input_vae_image_infos"]:
                start_abs = off + int(info["latent_start_idx"])
                n_tok = int(info["num_latent_tokens"])
                end_abs = start_abs + n_tok
                input_flow_positions_list.append([start_abs, end_abs])
                input_vae_batch_idx.append(torch.zeros(n_tok, dtype=torch.long))
                input_vae_seq_idx.append(torch.arange(start_abs, end_abs, dtype=torch.long))
                input_vae_latent_positions.append(_to_1d_long(info["latent_position_ids"]))
                input_vae_latent_shapes.append(tuple(info["latent_shape"]))
                input_padded_images.append(info["image_tensor"])

        flow_positions = (
            torch.tensor(flow_positions_list, dtype=torch.int32)
            if flow_positions_list else torch.zeros((0, 2), dtype=torch.int32)
        )
        g_seqlens = flow_positions.clone()

        if vae_batch_idx:
            vae_latent_indexes = (torch.cat(vae_batch_idx), torch.cat(vae_seq_idx))
            vae_latent_positions_tensor = torch.cat(vae_latent_positions, dim=0)
        else:
            vae_latent_indexes = None
            vae_latent_positions_tensor = None

        if input_vae_batch_idx:
            input_vae_latent_indexes = (
                torch.cat(input_vae_batch_idx), torch.cat(input_vae_seq_idx)
            )
            input_vae_latent_positions_tensor = torch.cat(input_vae_latent_positions, dim=0)
        else:
            input_vae_latent_indexes = None
            input_vae_latent_positions_tensor = None

        # ---- Step 8: concatenate input images / videos across samples ------
        pixel_values = _cat_or_none([s["pixel_values"] for s in per_sample])
        image_grid_thw = _cat_or_none([s["image_grid_thw"] for s in per_sample])
        pixel_values_videos = _cat_or_none([s["pixel_values_videos"] for s in per_sample])
        video_grid_thw = _cat_or_none([s["video_grid_thw"] for s in per_sample])

        # ---- Step 9: assemble output dict ----------------------------------
        batch = {
            "input_ids": input_ids,
            "attention_mask": None,  # packed signal — model treats as P-Pack
            "labels": labels,
            "shift_labels": shift_labels,
            "position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
            "sample_ids": sample_ids,
            "modality_mask": modality_mask,
            "input_image_mask": input_image_mask,
            "flow_positions": flow_positions,
            "g_seqlens": g_seqlens,
            "use_flow_matching": has_flow,
        }

        # Flow Matching generation targets (raw pixels — trainer encodes via VAE)
        if has_flow:
            batch.update({
                "padded_images": padded_images,
                "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
                "vae_latent_indexes": vae_latent_indexes,
                "vae_latent_positions": vae_latent_positions_tensor,
            })
        # TI2I input image VAE latents (raw pixels — trainer encodes via VAE)
        if input_padded_images:
            batch.update({
                "input_padded_images": input_padded_images,
                "input_vae_latent_shapes": input_vae_latent_shapes,
                "input_vae_latent_indexes": input_vae_latent_indexes,
                "input_vae_latent_positions": input_vae_latent_positions_tensor,
            })

        # Input ViT pixel_values
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            batch["image_grid_thw"] = image_grid_thw
        if pixel_values_videos is not None:
            batch["pixel_values_videos"] = pixel_values_videos
        if video_grid_thw is not None:
            batch["video_grid_thw"] = video_grid_thw

        return batch


__all__ = ["UnifiedDataCollator"]
