"""Three-stage packed/padded MoT flash attention for UnifiedMoT.

This module replaces the upstream `_flash_attention_forward_mot` (a module-level
function in `transformers.models.hunyuan_vl_mot.modeling_hunyuan_vl_mot`) with a
version that supports three modes:

  A) Packed prefill (P-Pack training)
     attention_mask is a dict carrying `cu_seqlens` (N+1,), `sample_ids` (1, T),
     and `g_seqlens` (M, 2). Inputs are (1, T, H, D) — a single packed row of
     concatenated samples. Three-pass attention:
       1) Causal varlen — `cu_seqlens` enforces sample boundaries
       2) Visual bidirectional override on each `v_seqlens` segment (absolute coords)
       3) Generation-block override on each `g_seqlens` segment, with KV range
          `[sample_start, g_e]` so KV NEVER crosses sample boundaries

  B) Padded prefill (legacy / inference fallback)
     attention_mask dict has `padding_mask` (B, S). Same as upstream:
     unpad → varlen → repad. Visual override layered on top. Generation override
     not used in this mode.

  C) Decode (KV-cache active, S_q != S_k)
     Simple (B, 1, H, D) path with no varlen.

The replacement is **module-scope rebinding** of upstream's
`_flash_attention_forward_mot` — surgical and reversible. Apply by importing
this module before instantiating any HunYuanVLMoT* class:

    from model import attention_mot_packed  # noqa: F401
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from flash_attn import flash_attn_varlen_func
from transformers.models.hunyuan_vl_mot import modeling_hunyuan_vl_mot as _upstream

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_ranges(obj) -> bool:
    if obj is None:
        return False
    if torch.is_tensor(obj):
        return obj.numel() > 0 and obj.shape[-1] in (2, 3)
    if isinstance(obj, list):
        if not obj:
            return False
        if torch.is_tensor(obj[0]):
            return any(t.numel() > 0 for t in obj)
        return True
    return False


def _iter_g_ranges(obj):
    """Yield (g_start, g_end, batch_idx) triples from `g_seqlens`.

    Accepts:
      - Tensor of shape (M, 2) — absolute packed coords (b_idx implicit = 0)
      - Tensor of shape (M, 3) — (s, e, b_idx)
      - List of tuples — same as above
      - List of (N_i, 2) tensors (per-sample, padded mode)
    """
    if torch.is_tensor(obj):
        for row in obj.tolist():
            if len(row) == 2:
                yield int(row[0]), int(row[1]), 0
            else:
                yield int(row[0]), int(row[1]), int(row[2])
    elif isinstance(obj, list):
        if obj and torch.is_tensor(obj[0]):
            for b_idx, t in enumerate(obj):
                if t.numel() == 0:
                    continue
                for row in t.tolist():
                    yield int(row[0]), int(row[1]), b_idx
        else:
            for item in obj:
                if torch.is_tensor(item):
                    item = item.tolist()
                if len(item) == 2:
                    yield int(item[0]), int(item[1]), 0
                else:
                    yield int(item[0]), int(item[1]), int(item[2])


def _find_sample_start(cu_seqlens: torch.Tensor, g_s: int) -> int:
    """Largest cu_seqlens[i] <= g_s. Used when sample_ids is unavailable."""
    cu = cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else cu_seqlens
    lo, hi = 0, len(cu) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cu[mid] <= g_s:
            lo = mid
        else:
            hi = mid - 1
    return int(cu[lo])


# ---------------------------------------------------------------------------
# Step 2 helper: visual bidirectional override on (B, S, H, D) tensors
# ---------------------------------------------------------------------------

def _apply_visual_bidirectional(
    attn_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    v_seqlens,
) -> torch.Tensor:
    """Override input-image segments with non-causal flash attention.

    Operates on (B, S, H, D). v_seqlens may be a (M, 2) tensor (B==1) or a list
    of (N_i, 2) tensors (per-sample for B>1 padded mode).

    PERF: convert each segs tensor to CPU once (`.tolist()`) instead of
    issuing two `.item()` syncs per row inside the inner loop. Each layer
    calls this once during forward and once during backward; without the
    fix, 32 layers × ~25 segments × 2 .item()/seg × 2 (fwd+bwd) ≈ 3200
    GPU→CPU syncs per step, each blocking on the previous attention kernel.
    """
    device = query.device
    if isinstance(v_seqlens, list):
        segs_list = v_seqlens
    else:
        segs_list = [v_seqlens]  # B==1 path

    visual_q = []
    visual_k = []
    visual_v = []
    cu_v = [0]
    max_v = 0
    write_back = []  # (b_idx, s, e)
    has_any = False

    for b_idx, segs in enumerate(segs_list):
        if segs is None or not torch.is_tensor(segs) or segs.numel() == 0:
            continue
        # PERF: one bulk D2H copy instead of per-element .item()
        segs_cpu = segs.tolist()
        for row in segs_cpu:
            s = int(row[0])
            e = int(row[1])
            if e <= s:
                continue
            has_any = True
            visual_q.append(query[b_idx, s:e])
            visual_k.append(key[b_idx, s:e])
            visual_v.append(value[b_idx, s:e])
            ln = e - s
            cu_v.append(cu_v[-1] + ln)
            max_v = max(max_v, ln)
            write_back.append((b_idx, s, e))

    if not has_any:
        # Preserve autograd graph topology across ranks even when this rank has
        # no visual segments (matches upstream's `fake_visual` trick).
        dummy = query[:1, :1].sum() * 0
        return attn_output + dummy

    vq = torch.cat(visual_q, dim=0)
    vk = torch.cat(visual_k, dim=0)
    vv = torch.cat(visual_v, dim=0)
    cu_v_t = torch.tensor(cu_v, device=device, dtype=torch.int32)

    vis_out = flash_attn_varlen_func(
        vq, vk, vv,
        cu_seqlens_q=cu_v_t, cu_seqlens_k=cu_v_t,
        max_seqlen_q=max_v, max_seqlen_k=max_v,
        causal=False,
    )

    attn_output = attn_output.clone()
    off = 0
    for b_idx, s, e in write_back:
        ln = e - s
        attn_output[b_idx, s:e] = vis_out[off:off + ln]
        off += ln
    return attn_output


# ---------------------------------------------------------------------------
# Step 3 helper: generation-block override (P-Pack only)
# ---------------------------------------------------------------------------

def _apply_generation_block(
    attn_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g_seqlens,
    cu_seqlens: torch.Tensor,
    sample_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    """Q over each generation block, K/V from sample-start..g_end (no cross-sample).

    PERF: previously each iteration of the per-block loop did:
      - `sample_ids[0, g_s].item()` — D2H sync
      - `cu_seqlens[sid].item()` — D2H sync
      - `torch.tensor([0, q_len], device=cuda, ...)` — small tensor alloc + H2D
      - same for cu_k_g
    For 25 gen blocks × 32 layers × 2 (fwd+bwd) ≈ 3200 ops/step, each blocking
    on the prior attention kernel. Now we do **one** D2H copy of cu_seqlens
    and sample_ids[0] up front, build all `cu_q_g` / `cu_k_g` pairs as a
    single (M, 2) tensor with one H2D, and view rows in the loop.
    """
    device = query.device

    # ---- One-shot metadata D2H ------------------------------------------
    if sample_ids is not None and torch.is_tensor(sample_ids):
        sample_ids_cpu = (
            sample_ids[0].tolist() if sample_ids.dim() == 2 else sample_ids.tolist()
        )
    else:
        sample_ids_cpu = None
    if torch.is_tensor(cu_seqlens):
        cu_seqlens_cpu = cu_seqlens.tolist()
    else:
        cu_seqlens_cpu = list(cu_seqlens)

    # ---- Pre-compute per-block specs entirely on CPU --------------------
    specs = []  # (g_s, g_e, b_idx, sample_start, q_len, kv_len)
    for g_s, g_e, b_idx in _iter_g_ranges(g_seqlens):
        if g_e <= g_s:
            continue
        if sample_ids_cpu is not None:
            sid = sample_ids_cpu[g_s]
            sample_start = cu_seqlens_cpu[sid]
        else:
            sample_start = _find_sample_start(cu_seqlens_cpu, g_s)
        q_len = g_e - g_s
        kv_len = g_e - sample_start
        specs.append((g_s, g_e, b_idx, sample_start, q_len, kv_len))

    if not specs:
        # No generation blocks on this rank — preserve autograd topology.
        dummy = query[:1, :1].sum() * 0
        return attn_output + dummy

    # ---- Single H2D for all cu pairs ------------------------------------
    cu_q_all = torch.tensor(
        [[0, s[4]] for s in specs], device=device, dtype=torch.int32
    )
    cu_k_all = torch.tensor(
        [[0, s[5]] for s in specs], device=device, dtype=torch.int32
    )

    attn_output = attn_output.clone()
    for i, (g_s, g_e, b_idx, sample_start, q_len, kv_len) in enumerate(specs):
        gen_out = flash_attn_varlen_func(
            query[b_idx, g_s:g_e].contiguous(),
            key[b_idx, sample_start:g_e].contiguous(),
            value[b_idx, sample_start:g_e].contiguous(),
            cu_seqlens_q=cu_q_all[i], cu_seqlens_k=cu_k_all[i],
            max_seqlen_q=q_len, max_seqlen_k=kv_len,
            causal=False,
        )
        attn_output[b_idx, g_s:g_e] = gen_out
    return attn_output


# ---------------------------------------------------------------------------
# Main entry point — drop-in replacement for upstream's _flash_attention_forward_mot
# ---------------------------------------------------------------------------

def flash_attention_forward_mot_packed(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
):
    """Packed/padded/decode dispatcher. See module docstring."""
    _upstream._check_flash_attn()

    if kwargs.get("output_attentions", False):
        logger.warning_once("`flash_attention_2` does not support `output_attentions=True`.")

    # (B, heads, S, D) -> (B, S, heads, D)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(m for m in module.modules() if isinstance(m, nn.Linear)).weight.dtype
        query, key, value = query.to(target_dtype), key.to(target_dtype), value.to(target_dtype)

    bsz, s_q, n_heads, head_dim = query.shape
    s_k = key.shape[1]

    if isinstance(attention_mask, dict):
        v_seqlens = attention_mask.get("v_seqlens", None)
        g_seqlens = attention_mask.get("g_seqlens", None)
        cu_seqlens_packed = attention_mask.get("cu_seqlens", None)
        padding_mask = attention_mask.get("padding_mask", None)
        sample_ids = attention_mask.get("sample_ids", None)
        # PERF: pre-computed by the language_model wrapper so we avoid a D2H
        # sync on every layer. None means we still have to compute it here.
        max_seqlen_packed = attention_mask.get("max_seqlen_packed", None)
    else:
        # Dummy: allow odd callers (e.g., HF generation builders) that pass tensor masks.
        v_seqlens = None
        g_seqlens = None
        cu_seqlens_packed = None
        padding_mask = None
        sample_ids = None
        max_seqlen_packed = None

    packed_mode = (cu_seqlens_packed is not None) and (s_q == s_k) and (bsz == 1)

    # ---------------- Mode A: Packed prefill ---------------------------------
    if packed_mode:
        h_kv = key.shape[2]
        q_flat = query.contiguous().view(s_q, n_heads, head_dim)
        k_flat = key.contiguous().view(s_k, h_kv, head_dim)
        v_flat = value.contiguous().view(s_k, h_kv, head_dim)

        cu = cu_seqlens_packed.to(device=query.device, dtype=torch.int32)
        # Use pre-computed max_seqlen if the wrapper supplied it (saves a
        # D2H sync per layer × 32 layers).
        if max_seqlen_packed is not None:
            max_seqlen = max_seqlen_packed
        else:
            with torch.no_grad():
                max_seqlen = int((cu[1:] - cu[:-1]).max().item())

        attn_flat = flash_attn_varlen_func(
            q_flat, k_flat, v_flat,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=True,
        )
        attn_output = attn_flat.reshape(1, s_q, n_heads, head_dim)

        if v_seqlens is not None:
            attn_output = _apply_visual_bidirectional(attn_output, query, key, value, v_seqlens)

        if _has_ranges(g_seqlens):
            attn_output = _apply_generation_block(
                attn_output, query, key, value, g_seqlens, cu, sample_ids
            )

        return attn_output, None

    # ---------------- Mode B: Padded prefill --------------------------------
    if padding_mask is not None and bsz > 1 and s_q == s_k:
        pad_bool = padding_mask.bool()
        seqlens = padding_mask.sum(dim=-1).to(torch.int32)
        cu_seqlens = torch.zeros(bsz + 1, device=query.device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
        max_seqlen = int(seqlens.max().item())

        q_unpad = query[pad_bool]
        k_unpad = key[pad_bool]
        v_unpad = value[pad_bool]

        out_unpad = flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=True,
        )
        attn_output = query.new_zeros(bsz, s_q, n_heads, head_dim)
        attn_output[pad_bool] = out_unpad

        if v_seqlens is not None:
            attn_output = _apply_visual_bidirectional(attn_output, query, key, value, v_seqlens)

        # Padded path supports generation-block override too (B>1 with per-sample triples)
        if _has_ranges(g_seqlens):
            # Build a synthetic cu_seqlens from padding_mask for sample_start lookup
            cu_pad = torch.zeros(bsz + 1, device=query.device, dtype=torch.int32)
            cu_pad[1:] = torch.cumsum(seqlens, dim=0)
            attn_output = _apply_generation_block(
                attn_output, query, key, value, g_seqlens, cu_pad, sample_ids
            )

        return attn_output, None

    # ---------------- Mode C: Single-sample / decode ------------------------
    h_kv = key.shape[2]
    q_flat = query.contiguous().view(bsz * s_q, n_heads, head_dim)
    k_flat = key.contiguous().view(bsz * s_k, h_kv, head_dim)
    v_flat = value.contiguous().view(bsz * s_k, h_kv, head_dim)
    cu_q_t = torch.arange(0, bsz + 1, dtype=torch.int32, device=query.device) * s_q
    cu_k_t = torch.arange(0, bsz + 1, dtype=torch.int32, device=query.device) * s_k

    attn_flat = flash_attn_varlen_func(
        q_flat, k_flat, v_flat,
        cu_seqlens_q=cu_q_t, cu_seqlens_k=cu_k_t,
        max_seqlen_q=s_q, max_seqlen_k=s_k,
        causal=(s_q == s_k),
    )
    attn_output = attn_flat.reshape(bsz, s_q, n_heads, head_dim)

    # Visual override is valid during prefill (S_q == S_k) only
    if v_seqlens is not None and s_q == s_k:
        attn_output = _apply_visual_bidirectional(attn_output, query, key, value, v_seqlens)

    if _has_ranges(g_seqlens) and s_q == s_k and bsz == 1:
        # B==1 inference: synthetic cu_seqlens covers the whole row
        cu_one = torch.tensor([0, s_q], device=query.device, dtype=torch.int32)
        attn_output = _apply_generation_block(
            attn_output, query, key, value, g_seqlens, cu_one, sample_ids
        )

    return attn_output, None


# ---------------------------------------------------------------------------
# Activate: rebind upstream module-level reference
# ---------------------------------------------------------------------------

_upstream._flash_attention_forward_mot = flash_attention_forward_mot_packed
logger.info("attention_mot_packed: replaced upstream _flash_attention_forward_mot")


__all__ = ["flash_attention_forward_mot_packed"]
