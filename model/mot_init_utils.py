"""Initialization utilities for the MoT generation path (mlp_g, layernorm_g).

Two regimes are handled by a single entry point `maybe_init_generation_path`:

    (a) same-size copy     : mlp_g.intermediate_size == mlp_v.intermediate_size
                             -> plain state_dict copy (legacy behavior).
    (b) Net2Wider expansion: mlp_g.intermediate_size >  mlp_v.intermediate_size
                             -> function-preserving neuron duplication (Chen et al.,
                             "Net2Net") with small Gaussian noise to break symmetry.

Only integer-multiple widening is exact; non-integer multiples use round-robin
replicate-and-trim with reciprocal-count scaling (still function-preserving, but
logged as a warning — integer multiples are strongly preferred).

All layernorm_g state is always plain-copied from layernorm_v because layernorms
are on hidden_size and unaffected by MLP widening.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@torch.no_grad()
def _net2wider_expand_mlp(
    mlp_g: nn.Module,
    mlp_v: nn.Module,
    noise_std: float = 1e-4,
    generator: Optional[torch.Generator] = None,
) -> None:
    """In-place initialize `mlp_g` to be function-equivalent to `mlp_v` via neuron
    duplication. After this call, for any input x, `mlp_g(x) ~= mlp_v(x)` within
    O(noise_std * ||x||).

    Expects a SwiGLU-style MLP with linear modules `.gate_proj`, `.up_proj`,
    `.down_proj`, all bias-free, with shapes:
        gate_proj / up_proj : (D, H)
        down_proj           : (H, D)
    """
    d_v = mlp_v.gate_proj.out_features
    d_g = mlp_g.gate_proj.out_features
    assert mlp_v.gate_proj.in_features == mlp_g.gate_proj.in_features, (
        f"hidden_size mismatch: mlp_v={mlp_v.gate_proj.in_features}, "
        f"mlp_g={mlp_g.gate_proj.in_features}"
    )
    assert mlp_v.down_proj.out_features == mlp_g.down_proj.out_features, (
        f"hidden_size mismatch on down_proj: mlp_v={mlp_v.down_proj.out_features}, "
        f"mlp_g={mlp_g.down_proj.out_features}"
    )
    assert d_g >= d_v, (
        f"mlp_g width {d_g} < mlp_v width {d_v} (shrinking unsupported)"
    )

    # Same-size fast path: plain copy, ignore noise.
    if d_g == d_v:
        mlp_g.load_state_dict(mlp_v.state_dict())
        return

    if d_g % d_v != 0:
        logger.warning(
            f"Net2Wider: d_g={d_g} not an integer multiple of d_v={d_v}; "
            f"using replicate-and-trim. Integer multiples (e.g. 2x) are strongly "
            f"preferred for cleanest function preservation."
        )

    device = mlp_v.gate_proj.weight.device
    # indices[j] = source neuron index for destination neuron j
    # Round-robin: for D_g = k*D_v this produces [0,1,...,D_v-1, 0,1,...,D_v-1, ...]
    # which makes twins "adjacent in source" rather than "adjacent in destination".
    # Either layout works mathematically; this choice keeps each source's twins
    # spread out along the destination axis, which is marginally friendlier to
    # downstream gather/scatter operations if any later code happens to assume
    # contiguity within a source group.
    indices = torch.arange(d_g, device=device) % d_v
    counts = torch.bincount(indices, minlength=d_v)  # (D_v,)

    # --- gate_proj and up_proj: row-wise replication ---------------------------
    for name in ("gate_proj", "up_proj"):
        src = getattr(mlp_v, name).weight.data          # (D_v, H)
        dst_param = getattr(mlp_g, name).weight
        dst = src[indices].clone().to(dtype=dst_param.dtype)  # (D_g, H)
        if noise_std > 0:
            noise = torch.empty(
                dst.shape, dtype=dst.dtype, device=dst.device,
            ).normal_(mean=0.0, std=noise_std, generator=generator)
            dst.add_(noise)
        dst_param.data.copy_(dst)

    # --- down_proj: column-wise replication with reciprocal-count scaling -----
    # down_v: (H, D_v), down_g: (H, D_g).
    # For each destination column j with source s = indices[j]:
    #     down_g[:, j] = down_v[:, s] / counts[s]
    # so that   Sum_{j: indices[j]=s}  down_g[:, j] * h_s
    #         = down_v[:, s] * h_s          (function preserved exactly).
    down_v = mlp_v.down_proj.weight.data                # (H, D_v)
    dst_param = mlp_g.down_proj.weight
    scale = (1.0 / counts.to(down_v.dtype))             # (D_v,)
    dst = (down_v[:, indices] * scale[indices]).clone().to(dtype=dst_param.dtype)  # (H, D_g)
    if noise_std > 0:
        noise = torch.empty(
            dst.shape, dtype=dst.dtype, device=dst.device,
        ).normal_(mean=0.0, std=noise_std, generator=generator)
        dst.add_(noise)
    dst_param.data.copy_(dst)


@torch.no_grad()
def init_mlp_g_from_mlp_v_net2wider(
    layer,
    noise_std: float = 1e-4,
    generator: Optional[torch.Generator] = None,
) -> dict:
    """Initialize a single decoder layer's generation path (mlp_g + layernorm_g)
    from its vision path (mlp_v + layernorm_v).

    Auto-detects same-size vs expansion case by comparing MLP widths.
    Layernorms are always plain-copied (shape is hidden_size, unchanged).

    Defensive against meta-device parameters: when transformers'
    `from_pretrained(dtype=...)` leaves `_g` params on the meta device (because
    they aren't in the upstream state_dict), a plain `load_state_dict` on
    those params doesn't materialize them — they stay meta and the `.to(...)`
    that follows produces garbage GPU memory. To avoid that we rebuild the
    `_g` parameters as fresh tensors on the same device/dtype as their `_v`
    counterparts.
    """
    d_v = layer.mlp_v.gate_proj.out_features
    d_g = layer.mlp_g.gate_proj.out_features

    # Materialize meta-device params before in-place copy: rebuild _g linear
    # weights on the same device/dtype as _v.
    def _materialize_linear(dst_lin: nn.Linear, src_lin: nn.Linear):
        if dst_lin.weight.is_meta or dst_lin.weight.device.type == "meta":
            new_w = torch.empty_like(src_lin.weight)
            dst_lin.weight = nn.Parameter(new_w, requires_grad=dst_lin.weight.requires_grad)
            if dst_lin.bias is not None:
                new_b = torch.empty_like(src_lin.bias) if src_lin.bias is not None else torch.zeros(
                    dst_lin.out_features, dtype=src_lin.weight.dtype, device=src_lin.weight.device,
                )
                dst_lin.bias = nn.Parameter(new_b, requires_grad=dst_lin.bias.requires_grad)

    def _materialize_norm(dst_norm, src_norm):
        if dst_norm.weight.is_meta or dst_norm.weight.device.type == "meta":
            new_w = torch.empty_like(src_norm.weight)
            dst_norm.weight = nn.Parameter(new_w, requires_grad=dst_norm.weight.requires_grad)

    for name in ("gate_proj", "up_proj", "down_proj"):
        if d_g == d_v:
            _materialize_linear(getattr(layer.mlp_g, name), getattr(layer.mlp_v, name))
        # else: widening path will rebuild mlp_g entirely in maybe_init_generation_path
    _materialize_norm(layer.input_layernorm_g, layer.input_layernorm_v)
    _materialize_norm(layer.post_attention_layernorm_g, layer.post_attention_layernorm_v)

    _net2wider_expand_mlp(
        layer.mlp_g, layer.mlp_v,
        noise_std=noise_std if d_g > d_v else 0.0,
        generator=generator,
    )
    layer.input_layernorm_g.load_state_dict(layer.input_layernorm_v.state_dict())
    layer.post_attention_layernorm_g.load_state_dict(
        layer.post_attention_layernorm_v.state_dict()
    )
    return {"D_v": d_v, "D_g": d_g, "expanded": d_g > d_v, "noise_std": noise_std}


@torch.no_grad()
def verify_mlp_g_equals_mlp_v(
    layer,
    num_samples: int = 4,
    seq_len: int = 8,
) -> float:
    """Diagnostic: forward random activations through both paths (under their
    respective post-attention layernorms) and return max absolute difference.

    Called from unit tests and logged right after init on layer 0 as a sanity
    check. Uses fp32 computation when possible (casts output back if needed).
    """
    hidden = layer.hidden_size
    dev = layer.mlp_v.gate_proj.weight.device
    dt = layer.mlp_v.gate_proj.weight.dtype
    x = torch.randn(num_samples, seq_len, hidden, device=dev, dtype=dt)
    y_v = layer.mlp_v(layer.post_attention_layernorm_v(x))
    y_g = layer.mlp_g(layer.post_attention_layernorm_g(x))
    return (y_v.float() - y_g.float()).abs().max().item()


def checkpoint_has_g_keys(model_load_path: Optional[str]) -> bool:
    """True iff the checkpoint on disk already contains `_g.` weights.

    This is the robust way to tell "generation path was previously saved; load
    it as-is" from "generation path is absent; we need to initialize it from
    mlp_v".

    Returns False if the path does not exist or no safetensors file is found
    (caller may fall back to a tensor-level check if needed).
    """
    if model_load_path is None:
        return False
    import os
    import json as _json
    index_path = os.path.join(model_load_path, "model.safetensors.index.json")
    single_path = os.path.join(model_load_path, "model.safetensors")
    if os.path.exists(index_path):
        with open(index_path) as f:
            keys = set(_json.load(f)["weight_map"].keys())
        return any("_g." in k for k in keys)
    if os.path.exists(single_path):
        try:
            from safetensors import safe_open
        except ImportError:
            return False
        with safe_open(single_path, framework="pt") as f:
            return any("_g." in k for k in f.keys())
    return False


def _resolve_decoder_layers(model):
    """Walk common wrapper paths to find the list of transformer decoder layers.

    Supports:
        UnifiedMoTForConditionalGeneration -> .model.language_model.model.layers
        bare HunYuan MoT model            -> .language_model.model.layers
        even-barer HF model               -> .model.layers
    """
    for path in (
        "model.language_model.model.layers",
        "language_model.model.layers",
        "model.layers",
    ):
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj
    raise AttributeError(
        "Could not locate decoder layers on model; tried "
        "model.language_model.model.layers / language_model.model.layers / model.layers"
    )


def maybe_init_generation_path(
    model,
    model_load_path: Optional[str] = None,
    noise_std: float = 1e-4,
    logger_: Optional[logging.Logger] = None,
    seed: int = 0xA17C0D1F,
    target_mlp_g_intermediate_size: Optional[int] = None,
) -> bool:
    """Top-level entry point used by train / inference / eval scripts.

    Behavior:
      * If `model_load_path` is provided and the checkpoint contains any `_g.`
        key, do nothing and return False (generation path was already loaded
        from the checkpoint — this is the T2I-resume / TI2I-continuation path).
      * Otherwise iterate every decoder layer and call
        `init_mlp_g_from_mlp_v_net2wider`:
            - Same-size case  -> plain state_dict copy (noise ignored).
            - Expansion case  -> Net2Wider neuron duplication + Gaussian noise
              (std = `noise_std`).

    `target_mlp_g_intermediate_size`: if provided and larger than the current
    `mlp_g.intermediate_size`, REBUILD each layer's `mlp_g` to the wider size
    before running Net2Wider init. Use this when the base checkpoint was
    loaded with mlp_g at mlp_v's width (so from_pretrained accepted the
    shapes) and you want to widen post-load.

    Returns True if initialization was performed.
    """
    log = logger_ or logger

    # ---- Re-materialize the latent_pos_embed (sin-cos table) ---------------
    # PositionEmbedding's parameter is computed at __init__ time but
    # `from_pretrained(dtype=...)` uses meta-init that bypasses the
    # constructor's data assignment, leaving the param uninitialized.
    pos_embed_module = None
    for path in ("latent_pos_embed", "model.latent_pos_embed"):
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            pos_embed_module = obj
            break
    if pos_embed_module is not None and hasattr(pos_embed_module, "_reset_parameters"):
        pos_embed_module._reset_parameters()
        log.info(
            f"latent_pos_embed re-initialized from sin-cos table "
            f"(shape={tuple(pos_embed_module.pos_embed.shape)})"
        )

    if checkpoint_has_g_keys(model_load_path):
        log.info(
            "Generation path present in checkpoint (found `_g.` keys); "
            "skipping mlp_g initialization."
        )
        return False

    layers = _resolve_decoder_layers(model)
    d_v = layers[0].mlp_v.gate_proj.out_features
    d_g_current = layers[0].mlp_g.gate_proj.out_features

    # ---- Optional: rebuild mlp_g at a wider size before init ----
    if target_mlp_g_intermediate_size is not None and target_mlp_g_intermediate_size > d_g_current:
        from transformers.models.hunyuan_vl_mot.modeling_hunyuan_vl_mot import HunYuanVLMoTMLP
        for layer in layers:
            base_cfg = layer.mlp_g.gate_proj.weight  # to inherit device/dtype
            device = base_cfg.device
            dtype = base_cfg.dtype
            # Use a duck-typed mini config matching upstream MLP's expected fields
            class _MiniCfg:
                pass
            mini = _MiniCfg()
            mini.hidden_size = layer.mlp_v.gate_proj.in_features
            mini.intermediate_size = target_mlp_g_intermediate_size
            mini.hidden_act = "silu"
            mini.mlp_bias = (layer.mlp_v.gate_proj.bias is not None)
            new_mlp_g = HunYuanVLMoTMLP(mini).to(device=device, dtype=dtype)
            layer.mlp_g = new_mlp_g
        d_g = target_mlp_g_intermediate_size
        log.info(f"Rebuilt mlp_g for all layers at intermediate_size={d_g}.")
    else:
        d_g = d_g_current

    expansion = d_g > d_v
    mode = "copy" if not expansion else f"net2wider (x{d_g / d_v:.3f})"
    log.info(
        f"Initializing mlp_g/layernorm_g from mlp_v/layernorm_v: "
        f"mode={mode} (d_v={d_v}, d_g={d_g}, "
        f"noise_std={noise_std if expansion else 0.0:g})"
    )

    # Single deterministic generator for reproducibility across layers.
    gen = torch.Generator(device=layers[0].mlp_v.gate_proj.weight.device)
    gen.manual_seed(seed)

    for layer in layers:
        init_mlp_g_from_mlp_v_net2wider(
            layer,
            noise_std=noise_std if expansion else 0.0,
            generator=gen,
        )

    # Spot-check layer 0: forward a random input through both paths. The diff
    # should be ~O(noise_std * sqrt(D_v) * ||x||); much larger indicates a bug.
    try:
        diff = verify_mlp_g_equals_mlp_v(layers[0])
        log.info(
            f"  post-init forward diff (layer 0, random input): "
            f"max |mlp_v - mlp_g| = {diff:.3e}"
        )
    except Exception as e:  # pylint: disable=broad-except  # pragma: no cover - diagnostic only
        log.warning(f"verify_mlp_g_equals_mlp_v diagnostic failed: {e}")

    return True
