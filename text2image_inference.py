"""Minimal text-to-image inference for the packed UnifiedMoT model.

Performs Euler-ODE sampling in the LLM hidden space:
  x_T ~ N(0, 1)
  for t in linspace(1, 0, num_steps + 1)[:-1]:
      flow_embed = vae2llm(x_t) + time_embedder(t) + latent_pos_embed
      hidden    = model(... flow_embed injected at latent positions ...)
      v_pred    = llm2vae(hidden_at_latent_positions)
      x_{t-dt} = x_t - dt * v_pred

Then VAE-decodes the final x_0 to pixels.

This script is single-sample / single-image. Classifier-free guidance (CFG)
is optional via ``--cfg_scale`` (>1 enables it): each ODE step runs a second
forward with an empty prompt, matching the 10% ``text_cond_dropout`` used in
training. ``--cfg_scale 1`` (default) disables CFG for the fastest path.

Run:
    # no CFG (fastest)
    python text2image_inference.py --ckpt /path/to/checkpoint \\
                               --prompt "a watercolor cat" \\
                               --vae /path/to/vae.safetensors \\
                               --out out.png \\
                               --height 256 --width 256 --num_steps 25

    # with CFG
    python text2image_inference.py --ckpt /path/to/checkpoint \\
                               --prompt "a watercolor cat" \\
                               --vae /path/to/vae.safetensors \\
                               --cfg_scale 5.0 --num_steps 50
"""
from __future__ import annotations

import argparse

import torch
from PIL import Image
from transformers.models.hunyuan_vl_mot import HunYuanVLMoTProcessor

from model import (
    UnifiedMoTConfig,
    UnifiedMoTForConditionalGeneration,
    maybe_init_generation_path,
)
from model.flow_matching_modules import (
    unpatchify_latent,
)
from vae_model.autoencoder import load_ae


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Model checkpoint directory")
    p.add_argument("--vae", required=True, help="VAE safetensors path")
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", default="out.png")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--num_steps", type=int, default=25)
    p.add_argument("--cfg_scale", type=float, default=1.0,
                   help="Classifier-free guidance scale. >1 enables CFG "
                        "(runs an extra empty-prompt forward per step; 2-5 typical). "
                        "1.0 disables CFG.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def get_2d_position_ids(h: int, w: int, max_per_side: int) -> torch.Tensor:
    """2D-flattened position ids matching model.PositionEmbedding lookup table."""
    rows = torch.arange(h)[:, None] * max_per_side
    cols = torch.arange(w)[None, :]
    return (rows + cols).reshape(-1).long()


@torch.no_grad()
def _build_seq_meta(processor, prompt, cfg, n_latent_tokens, device):
    """Build the packed (1, T) input sequence + routing tensors for one prompt.

    Sequence:
        chat_template(user/assistant) + <Image> + LATENT*N + </Image> + EOS
    Returns a dict of everything the per-step forward needs (constant across
    ODE steps except the latent region, which the caller patches each step).
    """
    eos = cfg.eos_token_id
    prompt_messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
        {"role": "assistant", "content": ""},
    ]
    prompt_inputs = processor.apply_chat_template(
        prompt_messages, return_dict=True, tokenize=True, add_generation_prompt=False,
    )
    prompt_ids = list(prompt_inputs["input_ids"][0])
    # Strip trailing EOS so the assistant turn flows into <Image>
    while prompt_ids and prompt_ids[-1] == eos:
        prompt_ids.pop()

    latent_ph = cfg.flow_latent_placeholder_id  # upstream latent_token_id
    image_start = cfg.image_start_token_id
    image_end = cfg.image_end_token_id

    seq = (
        prompt_ids
        + [image_start]
        + [latent_ph] * n_latent_tokens
        + [image_end]
        + [eos]
    )
    input_ids = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    seq_len = input_ids.shape[1]
    latent_start = len(prompt_ids) + 1  # right after <Image>
    latent_end = latent_start + n_latent_tokens

    modality_mask = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    modality_mask[0, latent_start:latent_end] = 2  # gen-latent route
    flow_positions = torch.tensor([[latent_start, latent_end]], dtype=torch.int32, device=device)
    g_seqlens = flow_positions.clone()
    # Single sample → packed degenerate: still feed cu_seqlens/sample_ids so the
    # same attention path runs as during training.
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    sample_ids = torch.zeros(1, seq_len, dtype=torch.int32, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    return {
        "input_ids": input_ids, "T": seq_len,
        "latent_start": latent_start, "latent_end": latent_end,
        "modality_mask": modality_mask, "g_seqlens": g_seqlens,
        "cu_seqlens": cu_seqlens, "sample_ids": sample_ids,
        "position_ids": position_ids,
    }


@torch.no_grad()
def _forward_v(model, inner, base_embeds, flow_embed, meta, dtype):
    """Run one forward with `flow_embed` injected at the latent span; return velocity."""
    inputs_embeds = base_embeds.clone()
    inputs_embeds[0, meta["latent_start"]:meta["latent_end"]] = flow_embed
    out = inner(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        position_ids=meta["position_ids"],
        cu_seqlens=meta["cu_seqlens"],
        sample_ids=meta["sample_ids"],
        modality_mask=meta["modality_mask"],
        input_image_mask=torch.zeros(1, meta["T"], dtype=torch.bool, device=inputs_embeds.device),
        flow_embeds=None,  # we already pre-built inputs_embeds
        flow_positions=None,
        g_seqlens=meta["g_seqlens"],
    )
    # FM velocity target during training is v = noise - x_0
    return model.llm2vae(out.hidden_states[0, meta["latent_start"]:meta["latent_end"]]).to(dtype)


@torch.no_grad()
def generate_image(
    model: UnifiedMoTForConditionalGeneration,
    vae,
    processor: HunYuanVLMoTProcessor,
    prompt: str,
    height: int,
    width: int,
    num_steps: int,
    device,
    dtype,
    cfg_scale: float = 1.0,
):
    """T2I single-sample sampling (optional CFG). Returns a (3, H, W) tensor in [-1, 1].

    With ``cfg_scale > 1`` each ODE step runs cond + uncond forwards and combines
      v = v_uncond + cfg_scale * (v_cond - v_uncond)
    The uncond branch uses an empty prompt, matching text_cond_dropout=0.1 in
    training where 10% of samples have the caption replaced with "".
    """
    cfg: UnifiedMoTConfig = model.config
    p = model.latent_patch_size
    downsample = cfg.vae_image_downsample  # pixel → patch-token (e.g. 16 = VAE(8) * patch(2))
    h_lat = height // downsample
    w_lat = width // downsample
    n_latent_tokens = h_lat * w_lat

    do_cfg = (cfg_scale != 1.0)
    embed_layer = model.get_input_embeddings()

    cond_meta = _build_seq_meta(processor, prompt, cfg, n_latent_tokens, device)
    cond_base = embed_layer(cond_meta["input_ids"])
    if do_cfg:
        uncond_meta = _build_seq_meta(processor, "", cfg, n_latent_tokens, device)
        uncond_base = embed_layer(uncond_meta["input_ids"])

    # 2D position ids for latent_pos_embed lookup
    latent_pos_ids = get_2d_position_ids(h_lat, w_lat, cfg.max_latent_size).to(device)

    # Initial noise x_T
    patch_latent_dim = p * p * cfg.vae_z_channels
    x = torch.randn(n_latent_tokens, patch_latent_dim, device=device, dtype=dtype)

    # Euler ODE: t from 1.0 → 0.0 in `num_steps`
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
    inner = model.model  # UnifiedMoTModel

    for i in range(num_steps):
        t = ts[i]
        dt = ts[i] - ts[i + 1]  # positive

        # Build flow_embed for the current x_t (shared by cond & uncond branches)
        time_emb = model.time_embedder(t.expand(n_latent_tokens)).to(dtype)
        x_proj = model.vae2llm(x.to(model.vae2llm.weight.dtype)).to(dtype)
        pos_emb = model.latent_pos_embed(latent_pos_ids).to(dtype)
        flow_embed = x_proj + time_emb + pos_emb  # (n_latent_tokens, D)

        v_cond = _forward_v(model, inner, cond_base, flow_embed, cond_meta, dtype)
        if do_cfg:
            v_uncond = _forward_v(model, inner, uncond_base, flow_embed, uncond_meta, dtype)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        # Euler step: x_{t-dt} = x_t - dt * v
        x = x - dt * v

    # Unpatchify + VAE decode
    x_lat = unpatchify_latent(x.float(), h_lat, w_lat, p, cfg.vae_z_channels)  # (C, H_lat, W_lat)
    vae_dtype = next(vae.parameters()).dtype
    x_lat = x_lat.unsqueeze(0).to(device=device, dtype=vae_dtype)
    img = vae.decode(x_lat)
    if hasattr(img, "sample"):
        img = img.sample
    return img.squeeze(0).float().clamp(-1, 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)

    print(f"Loading processor from {args.ckpt}...")
    processor = HunYuanVLMoTProcessor.from_pretrained(args.ckpt, trust_remote_code=True)

    print(f"Loading model from {args.ckpt}...")
    model = UnifiedMoTForConditionalGeneration.from_pretrained(args.ckpt, dtype=dtype)
    
    maybe_init_generation_path(model, model_load_path=args.ckpt)
    model.to(device)
    model.eval()

    print(f"Loading VAE from {args.vae}...")
    vae, _ = load_ae(args.vae)
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device, dtype=dtype)

    cfg_note = f", CFG {args.cfg_scale}" if args.cfg_scale != 1.0 else " (no CFG)"
    print(f"Generating: '{args.prompt}' @ {args.width}x{args.height}, {args.num_steps} ODE steps{cfg_note}")
    img = generate_image(
        model, vae, processor, args.prompt,
        height=args.height, width=args.width, num_steps=args.num_steps,
        device=device, dtype=dtype, cfg_scale=args.cfg_scale,
    )

    # Save: (C, H, W) in [-1, 1] → uint8 PNG
    arr = ((img.cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(args.out)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
