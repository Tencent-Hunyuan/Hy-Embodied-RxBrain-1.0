"""Interleaved multi-image inference with a SigLIP-context handoff.

Generates a sequence of frames autoregressively:

    for k in 1..Y:
        seq_k = chat_template(user[obs + decoded f_1..f_{k-1}] + text) + <Image>+LAT*N+</Image>+EOS
        x_0   = Euler-ODE(seq_k)            # prior frames condition via SigLIP
        f_k   = VAE.decode(x_0)             # decode to pixels
        # f_k is fed (as a SigLIP image) into seq_{k+1}

The crucial property: a previously generated frame enters the next step's context
as a **SigLIP-encoded image in a user turn** — NOT as its VAE latent — so the
context representation is identical at train and test (clean pixels either way),
avoiding the noised-VAE-latent train/test mismatch of single-sequence interleave.

`build_conditioned_sequence` is the single sequence constructor; training builds
the prompt the same way (apply_chat_template([user(images)+text, assistant:""])
-> strip EOS -> <Image>+LAT*N+</Image>+EOS), so train and infer token layouts
match exactly.

Run:
    python interleave_inference.py --ckpt <ckpt> --vae <vae> \
        --frames obs.jpg --task "put the cup on the shelf" \
        --max_frames 3 --num_steps 50 --out_dir out_interleave
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import torch
from PIL import Image
from transformers.models.hunyuan_vl_mot import HunYuanVLMoTProcessor

# NOTE: heavy imports (model package -> flash_attn, vae_model) are done lazily
# inside the functions that need them, so `build_conditioned_sequence` can be
# imported with only a processor.

# Input-image placeholder token ids the upstream HunYuanVL processor scatters
# ViT features into.
INPUT_IMAGE_PLACEHOLDER_IDS = (120687, 120688)


def build_conditioned_sequence(processor, input_image_paths: List[str], prompt_text: str,
                               prior_steps=None, task_instruction: Optional[str] = None):
    """Build the prompt token ids + processor vision tensors for a
    (multi-image + text) -> 1-generated-image sample.

    apply_chat_template([user(...), assistant:""]) then strip trailing EOS.

    prior_steps: optional list of (text, frame_path) for already-produced steps.
        When given, the user turn is the interleaved context the JOINT converter
        emits — [obs frames] + task + [text_1, frame_1, ..., text_m, frame_m] —
        so prior plan texts stay in context and prior frames enter as SigLIP inputs.
        input_image_paths are then the OBSERVATION frames only.
        When None, the legacy [all images] + text layout is used.
    task_instruction: optional mode instruction appended as the LAST user-text item,
        e.g. "Generate interleave goal planning".

    Returns (prompt_ids: list[int], proc_inputs: dict).
    """
    user_content = [{"type": "image", "image": p} for p in input_image_paths]
    user_content.append({"type": "text", "text": prompt_text})
    for t, f in (prior_steps or []):
        if t:
            user_content.append({"type": "text", "text": t})
        user_content.append({"type": "image", "image": f})
    if task_instruction:
        user_content.append({"type": "text", "text": task_instruction})
    prompt_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": ""},
    ]
    proc_inputs = processor.apply_chat_template(
        prompt_messages, return_dict=True, tokenize=True, add_generation_prompt=False,
    )
    eos = processor.tokenizer.eos_token_id
    prompt_ids = list(proc_inputs["input_ids"][0])
    while prompt_ids and prompt_ids[-1] == eos:
        prompt_ids.pop()
    return prompt_ids, proc_inputs


@torch.no_grad()
def generate_step_joint(
    model, vae, processor,
    input_image_paths: List[str],
    task_text: str,
    height: int, width: int, num_steps: int,
    device, dtype,
    max_text_tokens: int = 64,
    prior_steps=None,
):
    """v2 joint step: autoregressively decode the plan TEXT, then (on <Image>)
    flow-match the FRAME. Returns (text_str, image_tensor).

    prior_steps: list of (text, frame_path) already produced — threaded into the
    context so the rollout matches the JOINT training layout (prior texts kept,
    prior frames as SigLIP). When given, input_image_paths = observation frames only.

    The prefix is built by the SAME `build_conditioned_sequence` used at train
    time, so the text the model emits after `</answer>` and the <Image> trigger
    match what training taught."""
    from model.flow_matching_modules import unpatchify_latent
    from text2image_inference import get_2d_position_ids

    cfg = model.config
    eos = cfg.eos_token_id
    image_start = cfg.image_start_token_id
    latent_ph = cfg.flow_latent_placeholder_id
    inner = model.model

    # Append the interleave mode instruction (train==infer parity).
    from inference_utils import TASK_INSTRUCTION_INTERLEAVE
    prompt_ids, proc = build_conditioned_sequence(processor, input_image_paths, task_text,
                                                   prior_steps=prior_steps,
                                                   task_instruction=TASK_INSTRUCTION_INTERLEAVE)
    pixel_values = proc.get("pixel_values")
    image_grid_thw = proc.get("image_grid_thw")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device=device, dtype=dtype)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device=device)

    # ---- 1. autoregressive text decode until <Image> (KV-cached: exact, ~2.8x) ----
    # prefill the prompt once (use_cache), then feed one token/step against the growing
    # KV cache — the model's native past_key_values plumbing (Mode C handles the decode
    # shape). Validated bit-identical to the non-cached per-token full-forward decode.
    from transformers.cache_utils import DynamicCache

    def _dmask(ids):
        seq_len = len(ids)
        inp = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        mod = torch.zeros(1, seq_len, dtype=torch.long, device=device)
        iim = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        for pid in INPUT_IMAGE_PLACEHOLDER_IDS:
            m = inp[0] == pid
            mod[0, m] = 1
            iim[0, m] = True
        return inp, mod, iim

    _empty_g = torch.zeros((0, 2), dtype=torch.int32, device=device)
    pkv = DynamicCache()
    prompt_len = len(prompt_ids)
    inp, mod, iim = _dmask(list(prompt_ids))
    out = inner(input_ids=inp, position_ids=torch.arange(prompt_len, device=device).unsqueeze(0),
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                cu_seqlens=torch.tensor([0, prompt_len], dtype=torch.int32, device=device),
                sample_ids=torch.zeros(1, prompt_len, dtype=torch.int32, device=device),
                modality_mask=mod, input_image_mask=iim,
                flow_embeds=None, flow_positions=None, g_seqlens=_empty_g,
                use_cache=True, past_key_values=pkv, cache_position=torch.arange(prompt_len, device=device))
    pkv = out.past_key_values
    nxt = int(out.logits[0, -1].argmax().item())
    cur = prompt_len
    text_ids = []
    for _ in range(max_text_tokens):
        if nxt == image_start or nxt == eos:
            break
        text_ids.append(nxt)
        out = inner(input_ids=torch.tensor([[nxt]], dtype=torch.long, device=device),
                    position_ids=torch.tensor([[cur]], device=device),
                    pixel_values=None, image_grid_thw=None,
                    cu_seqlens=torch.tensor([0, cur + 1], dtype=torch.int32, device=device),
                    sample_ids=torch.zeros(1, 1, dtype=torch.int32, device=device),
                    modality_mask=torch.zeros(1, 1, dtype=torch.long, device=device),
                    input_image_mask=torch.zeros(1, 1, dtype=torch.bool, device=device),
                    flow_embeds=None, flow_positions=None, g_seqlens=_empty_g,
                    use_cache=True, past_key_values=pkv, cache_position=torch.tensor([cur], device=device))
        pkv = out.past_key_values
        nxt = int(out.logits[0, -1].argmax().item())
        cur += 1
    seq = list(prompt_ids) + text_ids
    text_str = processor.tokenizer.decode(text_ids, skip_special_tokens=True)

    # ---- 2. flow-match the frame — diffusion PREFIX-CACHE: prefill (prompt+text+IMAGE_START)
    # once, then each denoise step forwards ONLY the n latent tokens against the cached prefix
    # (Mode C causal=False -> latents attend bidirectionally to prefix+latents, == the gen-block).
    # Validated equivalent to the full-forward loop within ROCm non-determinism. ----
    p = model.latent_patch_size
    ds = cfg.vae_image_downsample
    h_lat, w_lat = height // ds, width // ds
    n_latent = h_lat * w_lat
    prefix = seq + [image_start]
    prefix_len = len(prefix)                      # = latent_start
    inp_p, mod_p, iim_p = _dmask(prefix)
    pkv_f = DynamicCache()
    inner(input_ids=inp_p, position_ids=torch.arange(prefix_len, device=device).unsqueeze(0),
          pixel_values=pixel_values, image_grid_thw=image_grid_thw,
          cu_seqlens=torch.tensor([0, prefix_len], dtype=torch.int32, device=device),
          sample_ids=torch.zeros(1, prefix_len, dtype=torch.int32, device=device),
          modality_mask=mod_p, input_image_mask=iim_p,
          flow_embeds=None, flow_positions=None, g_seqlens=_empty_g,
          use_cache=True, past_key_values=pkv_f, cache_position=torch.arange(prefix_len, device=device))

    latent_pos_ids = get_2d_position_ids(h_lat, w_lat, cfg.max_latent_size).to(device)
    latent_pos_emb = model.latent_pos_embed(latent_pos_ids).to(dtype)
    lat_ids = torch.tensor([[latent_ph] * n_latent], dtype=torch.long, device=device)
    fp_rel = torch.tensor([[0, n_latent]], dtype=torch.int32, device=device)   # rel to the n-token input
    mod2 = torch.full((1, n_latent), 2, dtype=torch.long, device=device)
    iim2 = torch.zeros(1, n_latent, dtype=torch.bool, device=device)
    pos_l = torch.arange(prefix_len, prefix_len + n_latent, device=device).unsqueeze(0)
    cu_l = torch.tensor([0, prefix_len + n_latent], dtype=torch.int32, device=device)
    sid_l = torch.zeros(1, n_latent, dtype=torch.int32, device=device)

    x = torch.randn(n_latent, p * p * cfg.vae_z_channels, device=device, dtype=dtype)
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
    for i in range(num_steps):
        t, dt = ts[i], ts[i] - ts[i + 1]
        fe = (model.vae2llm(x.to(model.vae2llm.weight.dtype)).to(dtype)
              + model.time_embedder(t.expand(n_latent)).to(dtype) + latent_pos_emb)
        out = inner(input_ids=lat_ids, inputs_embeds=None, attention_mask=None,
                    position_ids=pos_l, pixel_values=None, image_grid_thw=None,
                    cu_seqlens=cu_l, sample_ids=sid_l, modality_mask=mod2, input_image_mask=iim2,
                    flow_embeds=fe, flow_positions=fp_rel, g_seqlens=_empty_g,
                    use_cache=True, past_key_values=pkv_f, cache_position=pos_l[0])
        v = model.llm2vae(out.hidden_states[0, 0:n_latent]).to(dtype)
        x = x - dt * v
        pkv_f.crop(prefix_len)                    # drop the latents; keep the fixed prefix for next step
    x_lat = unpatchify_latent(x.float(), h_lat, w_lat, p, cfg.vae_z_channels)
    x_lat = x_lat.unsqueeze(0).to(device=device, dtype=next(vae.parameters()).dtype)
    img = vae.decode(x_lat)
    if hasattr(img, "sample"):
        img = img.sample
    return text_str, img.squeeze(0).float().clamp(-1, 1)


@torch.no_grad()
def interleave_generate_joint(
    model, vae, processor,
    obs_frames: List[str], task_text: str, num_frames: int,
    height: int, width: int, num_steps: int, device, dtype, out_dir: str,
    max_text_tokens: int = 64,
):
    """v2 rollout: at each step the model emits plan text + a frame; the decoded
    frame AND its plan text are fed back as context for the next step — prior
    frames via SigLIP, prior texts kept — matching the JOINT training layout."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    gen_paths, texts = [], []
    prior_steps = []  # (text, frame_path) accumulated across steps
    for k in range(num_frames):
        text, img = generate_step_joint(
            model, vae, processor, list(obs_frames), task_text,
            height, width, num_steps, device, dtype, max_text_tokens,
            prior_steps=list(prior_steps),
        )
        arr = ((img.cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype("uint8")
        path = os.path.join(out_dir, f"joint_step{k+1}.png")
        Image.fromarray(arr).save(path)
        gen_paths.append(path)
        texts.append(text)
        prior_steps.append((text, path))
        print(f"  [joint step {k+1}/{num_frames}] prior={len(prior_steps)-1} | TEXT: {text[:80]!r} -> {path}")
    return gen_paths, texts


def _row(paths, w, h):
    """Horizontal strip of images (resized to w x h, 4px gaps)."""
    imgs = [Image.open(p).convert("RGB").resize((w, h)) for p in paths]
    n = len(imgs)
    strip = Image.new("RGB", (w * n + 4 * (n - 1), h), (20, 20, 20))
    for i, im in enumerate(imgs):
        strip.paste(im, (i * (w + 4), 0))
    return strip


def main():
    ap = argparse.ArgumentParser(description="Joint interleaved rollout: decode ALL plan text + ALL frames.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--frames", nargs="+", required=True, help="observation frame path(s)")
    ap.add_argument("--task", required=True, help="overall task text")
    ap.add_argument("--max_frames", type=int, default=None, help="cap #frames to generate")
    ap.add_argument("--out_dir", default="out_interleave")
    ap.add_argument("--height", type=int, default=144)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--num_steps", type=int, default=50, help="ODE steps per frame")
    ap.add_argument("--max_text_tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--understanding_max_pixels", type=int, default=524288,
                    help="Cap obs/prior-frame ViT input pixels to MATCH training. "
                         "Default 524288 (~494 tok/frame); the model default 4194304 (~4050 tok/frame) "
                         "is an 8x train/infer resolution mismatch that degrades eval. Set 0 to disable.")
    args = ap.parse_args()

    from model import UnifiedMoTForConditionalGeneration, maybe_init_generation_path
    from vae_model.autoencoder import load_ae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)

    # ---- resolve obs frames + task ----
    obs, task, n_frames = args.frames, args.task, (args.max_frames or 1)

    processor = HunYuanVLMoTProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    # train==infer parity: cap obs/prior-frame ViT pixels to the SAME value training
    # used. Prior frames re-enter via SigLIP at rollout, so this must match training too.
    if args.understanding_max_pixels and args.understanding_max_pixels > 0:
        ip = processor.image_processor
        ip.max_pixels = args.understanding_max_pixels
        if isinstance(getattr(ip, "size", None), dict) and "longest_edge" in ip.size:
            ip.size["longest_edge"] = args.understanding_max_pixels
    model = UnifiedMoTForConditionalGeneration.from_pretrained(args.ckpt, dtype=dtype)
    maybe_init_generation_path(model, model_load_path=args.ckpt)
    model.to(device).eval()
    vae, _ = load_ae(args.vae)
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device, dtype=dtype)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"JOINT interleaved rollout | obs={len(obs)} frame(s) | steps={n_frames}")
    print(f"TASK: {task}\n")

    gen_paths, gen_texts = interleave_generate_joint(
        model, vae, processor, obs, task, n_frames,
        args.height, args.width, args.num_steps, device, dtype, args.out_dir,
        max_text_tokens=args.max_text_tokens,
    )

    # ---- write the FULL decoded output (text) + montages (image) ----
    lines = [f"TASK: {task}", ""]
    for k in range(len(gen_texts)):
        lines.append(f"--- step {k+1} ---")
        lines.append(f"GEN text: {gen_texts[k]}")
        lines.append("")
    txt_path = os.path.join(args.out_dir, "result.txt")
    open(txt_path, "w").write("\n".join(lines))
    print("\n".join(lines))

    # image montage: GEN row
    width, height = args.width, args.height
    gen_row = _row(gen_paths, width, height)
    gen_row.save(os.path.join(args.out_dir, "rollout_GEN.png"))
    print(f"\nDecoded {len(gen_paths)} frames + {len(gen_texts)} texts -> {args.out_dir}")
    print(f"  text  : {txt_path}")
    print(f"  images: {args.out_dir}/joint_step*.png + rollout montage")


if __name__ == "__main__":
    main()
