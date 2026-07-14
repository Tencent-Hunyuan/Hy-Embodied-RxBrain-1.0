"""Joint fixed-N multi-frame generation inference (world-model / x_to_N).

Unlike `interleave_inference.py` (SigLIP-handoff autoregressive rollout — a prior
frame re-enters the next step as a *clean SigLIP image*), multi-frame generation
lays out **all N `<Image>` blocks in ONE sequence** and integrates them **jointly**
with a single Euler ODE. A prior frame stays in context as its **VAE latent**
(at the shared ODE timestep), exactly matching training, where the assistant turn
held N latent blocks denoised together (per-frame independent timesteps + hybrid
block attention: causal across frames, bidirectional within). So inference is:

    seq = prompt(+ViT obs) + "open text" + N×(<Image>[LAT]*n</Image>) + EOS
    x_k ~ N(0,1) for k in 1..N
    for t in linspace(1,0,steps):                       # ONE shared schedule
        flow_embed_k = vae2llm(x_k) + time(t) + pos     # k = 1..N, concatenated
        hidden       = model(... flow_embeds @ N flow_positions ...)   # ONE forward
        x_k         -= dt * llm2vae(hidden @ block_k)   # k = 1..N
    frame_k = VAE.decode(x_k)

The N frames cohere because frame k attends to frames 1..k (cross-frame causal),
all in the single forward pass — no per-frame autoregressive loop.

Alignment with training (this is the load-bearing correspondence)
-----------------------------------------------------------------
TRAINING (modeling_unified_mot._build_flow_embeds + flow_matching_modules.sample_timesteps):
  the N target frames are packed into one sequence; `sample_timesteps` draws ONE
  INDEPENDENT timestep per frame (`randn(len(shapes))`), each broadcast over that
  frame's tokens; `x_t=(1-t)·clean+t·noise` per frame; a SINGLE forward predicts
  the velocity for ALL N frames and the FM-MSE is summed over all N. So: every
  frame independently noised, one forward → N frames.

INFERENCE (here): one forward PER Euler step injects all N frames' current x_t
  (each with its own `time_embedder(t_k)`) at the N flow_positions and reads the
  velocity for ALL N frames — structurally identical to the training forward
  (one forward, N frames, per-frame time embedding). We therefore do NOT loop
  per frame and do NOT re-encode prior frames via SigLIP.

The only inference-time choice is the per-frame timestep SCHEDULE `t_k(step)`:
  - lockstep (default, validated ~28 dB): all frames share the same t each step,
    swept 1→0 together. This is the equal-t diagonal of the joint t-space that the
    independent-per-frame training already covers, so it is in-distribution.
  - Because training randomized t per frame (incl. cases where earlier frames are
    much cleaner than later ones), a staggered "diffusion-forcing" schedule
    (frame k offset so it lags frame k-1) is also in-distribution and tends to
    help long autoregressive rollouts — left as a future `--schedule` option.
  Either way each step is ONE forward over all N frames; we never collapse the
  model's per-frame timestep capability into N separate passes.

Run:
    python multiframe_inference.py --ckpt <ckpt> --vae <ae.safetensors> \
        --frames obs0.jpg obs1.jpg --task "imagine the next frames" \
        --num_frames 4 --num_steps 50 --out_dir multiframe_out
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import torch
from PIL import Image
from transformers.models.hunyuan_vl_mot import HunYuanVLMoTProcessor

# Reuse the (obs frames + task) prompt constructor, ViT placeholder ids, and the
# montage helper from the interleave inference module.
from interleave_inference import (
    build_conditioned_sequence,
    INPUT_IMAGE_PLACEHOLDER_IDS,
    _row,
)


@torch.no_grad()
def _ar_decode_opening_text(
    inner, processor, prompt_ids, pixel_values, image_grid_thw,
    image_start_id, eos_id, device, max_text_tokens,
):
    """Greedily decode the assistant opening text until the model emits <Image>.

    Mirrors the text loop in interleave_inference.generate_step_joint. Returns the
    decoded token ids (without the terminating <Image>/EOS)."""
    seq = list(prompt_ids)
    text_ids: List[int] = []

    def logits_last(ids):
        seq_len = len(ids)
        inp = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        mod = torch.zeros(1, seq_len, dtype=torch.long, device=device)
        iim = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        for pid in INPUT_IMAGE_PLACEHOLDER_IDS:
            m = inp[0] == pid
            mod[0, m] = 1
            iim[0, m] = True
        out = inner(
            input_ids=inp, inputs_embeds=None, attention_mask=None,
            position_ids=torch.arange(seq_len, device=device).unsqueeze(0),
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            cu_seqlens=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            sample_ids=torch.zeros(1, seq_len, dtype=torch.int32, device=device),
            modality_mask=mod, input_image_mask=iim,
            flow_embeds=None, flow_positions=None,
            g_seqlens=torch.zeros((0, 2), dtype=torch.int32, device=device),
        )
        return out.logits[0, -1]

    for _ in range(max_text_tokens):
        nxt = int(logits_last(seq).argmax().item())
        if nxt == image_start_id or nxt == eos_id:
            break
        text_ids.append(nxt)
        seq.append(nxt)
    return text_ids

@torch.no_grad()
def generate_multiframe_joint(
    model, vae, processor,
    obs_frames: List[str], task_text: str, num_frames: int,
    height: int, width: int, num_steps: int, device, dtype,
    max_text_tokens: int = 96, opening_text: Optional[str] = None,
):
    """Jointly denoise N frames in one sequence. Returns (opening_text, [imgs])."""
    from model.flow_matching_modules import unpatchify_latent
    from text2image_inference import get_2d_position_ids

    cfg = model.config
    eos = cfg.eos_token_id
    # Multi-frame uses a single <Video> ... </Video> wrapper instead of N×<Image>.
    video_start = cfg.video_start_token_id  # 120122
    video_end = cfg.video_end_token_id      # 120123
    latent_ph = cfg.flow_latent_placeholder_id
    inner = model.model

    p = model.latent_patch_size
    ds = cfg.vae_image_downsample
    h_lat, w_lat = height // ds, width // ds
    n_latent = h_lat * w_lat
    patch_dim = p * p * cfg.vae_z_channels

    # Append the multi-frame task instruction to the user turn (train==infer parity).
    # build_conditioned_sequence folds task_text into the prompt.
    from inference_utils import TASK_INSTRUCTION_MULTI_FRAME
    instr_text = (task_text + "\n" + TASK_INSTRUCTION_MULTI_FRAME) if task_text else TASK_INSTRUCTION_MULTI_FRAME
    prompt_ids, proc = build_conditioned_sequence(processor, obs_frames, instr_text)
    pixel_values = proc.get("pixel_values")
    image_grid_thw = proc.get("image_grid_thw")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device=device, dtype=dtype)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device=device)

    # ---- opening text: teacher-force from GT or AR-decode until <Video> ----
    if opening_text is not None:
        text_ids = processor.tokenizer.encode(opening_text, add_special_tokens=False)
        text_str = opening_text
    else:
        text_ids = _ar_decode_opening_text(
            inner, processor, prompt_ids, pixel_values, image_grid_thw,
            video_start, eos, device, max_text_tokens,
        )
        text_str = processor.tokenizer.decode(text_ids, skip_special_tokens=True)

    # ---- lay out the full sequence: <Video> + N×[LAT] + </Video> (bare latents,
    # matches the dataset's multi-frame layout; one flow_positions row per frame) ----
    seq = list(prompt_ids) + list(text_ids)
    seq.append(video_start)
    frame_spans = []  # (latent_start, latent_end) per frame
    for _ in range(num_frames):
        ls = len(seq)
        seq.extend([latent_ph] * n_latent)
        le = len(seq)
        frame_spans.append((ls, le))
    seq.append(video_end)
    seq.append(eos)
    seq_len = len(seq)
    input_ids = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

    mod = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    iim = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
    for pid in INPUT_IMAGE_PLACEHOLDER_IDS:
        m = input_ids[0] == pid
        mod[0, m] = 1
        iim[0, m] = True
    for ls, le in frame_spans:
        mod[0, ls:le] = 2  # generation-latent route

    # flow_positions / g_seqlens in frame order — must match flow_embeds concat order
    # (model injects flow_embeds slices into flow_positions rows in order; see
    #  modeling_unified_mot.py:177-183).
    flow_positions = torch.tensor(
        [[ls, le] for ls, le in frame_spans], dtype=torch.int32, device=device
    )
    g_seqlens = flow_positions.clone()
    latent_pos_ids = get_2d_position_ids(h_lat, w_lat, cfg.max_latent_size).to(device)
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    sid = torch.zeros(1, seq_len, dtype=torch.int32, device=device)
    pos = torch.arange(seq_len, device=device).unsqueeze(0)

    # ---- N latent buffers, joint Euler ODE (ONE forward/step over all N frames) ----
    # Per-frame timestep schedule t_k(step): training noised each frame at its OWN t,
    # so we keep a per-frame t here too. Default = lockstep (all frames share the
    # step's t) — the validated in-distribution schedule. `frame_t_offset[k]` is the
    # hook for a staggered "diffusion-forcing" schedule (kept 0 = lockstep).
    xs = [torch.randn(n_latent, patch_dim, device=device, dtype=dtype) for _ in range(num_frames)]
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
    frame_t_offset = [0.0] * num_frames  # all 0 -> lockstep; set per-frame for diffusion-forcing
    pos_emb = model.latent_pos_embed(latent_pos_ids).to(dtype)  # same per-frame (ids restart at 0)
    for i in range(num_steps):
        dt = ts[i] - ts[i + 1]
        fe_parts = []
        for k in range(num_frames):
            t_k = (ts[i] + frame_t_offset[k]).clamp(0.0, 1.0)        # this frame's timestep
            time_emb_k = model.time_embedder(t_k.expand(n_latent)).to(dtype)
            x_proj = model.vae2llm(xs[k].to(model.vae2llm.weight.dtype)).to(dtype)
            fe_parts.append(x_proj + time_emb_k + pos_emb)
        flow_embeds = torch.cat(fe_parts, dim=0)  # (N*n_latent, D), frame order == flow_positions
        out = inner(
            input_ids=input_ids, inputs_embeds=None, attention_mask=None,
            position_ids=pos, pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            cu_seqlens=cu, sample_ids=sid, modality_mask=mod, input_image_mask=iim,
            flow_embeds=flow_embeds, flow_positions=flow_positions, g_seqlens=g_seqlens,
        )
        hidden = out.hidden_states
        for k, (ls, le) in enumerate(frame_spans):
            v = model.llm2vae(hidden[0, ls:le]).to(dtype)
            xs[k] = xs[k] - dt * v

    # ---- decode each frame ----
    imgs = []
    vae_dtype = next(vae.parameters()).dtype
    for k in range(num_frames):
        x_lat = unpatchify_latent(xs[k].float(), h_lat, w_lat, p, cfg.vae_z_channels)
        x_lat = x_lat.unsqueeze(0).to(device=device, dtype=vae_dtype)
        img = vae.decode(x_lat)
        if hasattr(img, "sample"):
            img = img.sample
        imgs.append(img.squeeze(0).float().clamp(-1, 1))
    return text_str, imgs


def vae_target_hw(img_path: str, max_size: int = 256, min_size: int = 96, stride: int = 16):
    """(h_px, w_px) that training's VAEImageTransform produces for this image — so the
    generated frame's latent grid matches what the model was trained to output. Runs the
    REAL VAEImageTransform for exact parity; returns None on failure.

    Why this matters: training resized each target frame aspect-preserving to <= max_size,
    stride-divisible — jaka/umi 848x480 -> 256x144 (144 latent tok), xtrainer 640x480 ->
    256x192 (192 latent tok). Generating every robot at a fixed 256x144 is wrong for
    xtrainer (4:3): wrong token count + wrong aspect, which corrupts both its output and
    its PSNR-vs-GT comparison (GT got squished 4:3 -> 16:9).
    """
    try:
        from inference_utils import VAEImageTransform
        t = VAEImageTransform(max_size=max_size, min_size=min_size, stride=stride)
        tensor = t(Image.open(img_path).convert("RGB"))   # (3, H, W)
        return int(tensor.shape[1]), int(tensor.shape[2])
    except Exception:  # pylint: disable=broad-except  # any failure → caller falls back
        return None


def main():
    ap = argparse.ArgumentParser(description="Joint fixed-N multi-frame generation.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--frames", nargs="+", required=True, help="observation frame path(s)")
    ap.add_argument("--task", required=True, help="overall task text")
    ap.add_argument("--num_frames", type=int, default=4,
                    help="number of future frames to generate")
    ap.add_argument("--out_dir", default="multiframe_out")
    ap.add_argument("--height", type=int, default=None,
                    help="override auto-derived generation height (px). Default: derive "
                         "per-record from the GT/obs frame via the training VAEImageTransform "
                         "(jaka/umi 848x480 -> 144, xtrainer 640x480 -> 192).")
    ap.add_argument("--width", type=int, default=None,
                    help="override auto-derived generation width (px). Default: derived (256).")
    ap.add_argument("--max_image_size", type=int, default=256,
                    help="VAEImageTransform max side — MUST match training (256).")
    ap.add_argument("--min_image_size", type=int, default=96,
                    help="VAEImageTransform min side — MUST match training (96).")
    ap.add_argument("--image_stride", type=int, default=16,
                    help="VAEImageTransform stride — MUST match training (16).")
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--max_text_tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--understanding_max_pixels", type=int, default=524288,
                    help="Cap obs-frame ViT input pixels to MATCH training. "
                         "Default 524288 (~494 tok/frame); the model default 4194304 (~4050 tok/frame) "
                         "is an 8x train/infer resolution mismatch that degrades eval. Set 0 to disable.")
    args = ap.parse_args()

    from model import UnifiedMoTForConditionalGeneration, maybe_init_generation_path
    from vae_model.autoencoder import load_ae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)

    # ---- resolve obs frames + task ----
    obs, task = args.frames, args.task
    n_frames = args.num_frames or 1
    if not n_frames:
        raise ValueError("num_frames resolved to 0 — pass --num_frames >= 1")

    processor = HunYuanVLMoTProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    # train==infer parity: cap obs-frame ViT pixels to the SAME value training used.
    # Without this, inference obs frames are ~8x higher-res than what the model saw at train.
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
    print(f"JOINT multi-frame | obs={len(obs)} frame(s) | num_frames={n_frames}")
    print(f"TASK: {task}\n")

    # ---- generation resolution: match training's VAEImageTransform per record ----
    # Training resized each TARGET frame aspect-preserving to <=max_image_size,
    # stride-divisible (jaka/umi -> 256x144, xtrainer -> 256x192). Generating every
    # robot at a fixed 256x144 was wrong for xtrainer (4:3). Derive from the data;
    # explicit --height/--width override.
    if args.height and args.width:
        height, width = args.height, args.width
    else:
        ref = obs[0] if obs else None
        hw = (vae_target_hw(ref, args.max_image_size, args.min_image_size, args.image_stride)
              if ref else None)
        if hw is None:
            height, width = 144, 256
            print("[warn] could not derive resolution from data; falling back to 144x256")
        else:
            height, width = hw
    ds = model.config.vae_image_downsample
    print(f"  gen resolution: {width}x{height} px  (latent {width // ds}x{height // ds} = {(width // ds) * (height // ds)} tok/frame)")

    text_str, imgs = generate_multiframe_joint(
        model, vae, processor, obs, task, n_frames,
        height, width, args.num_steps, device, dtype,
        max_text_tokens=args.max_text_tokens, opening_text=None,
    )

    # ---- save frames + montage + result.txt ----
    gen_paths = []
    for k, img in enumerate(imgs):
        arr = ((img.cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype("uint8")
        path = os.path.join(args.out_dir, f"frame{k + 1}.png")
        Image.fromarray(arr).save(path)
        gen_paths.append(path)

    lines = [f"TASK: {task}", "",
             f"OPEN text (decoded): {text_str}"]
    lines.append("")
    for k in range(len(imgs)):
        lines.append(f"frame {k + 1}")
    txt_path = os.path.join(args.out_dir, "result.txt")
    open(txt_path, "w").write("\n".join(lines))
    print("\n".join(lines))

    from PIL import ImageDraw
    # montage: INPUT (obs frames) / GEN (ours),
    # each a horizontal strip of W×H cells (W,H = the derived generation resolution),
    # with a left label column.
    label_w = 64
    gap = 6
    rows = [("INPUT", list(obs)),
            ("GEN", gen_paths)]
    rows = [(lbl, paths) for lbl, paths in rows if paths]   # drop empty
    strips = [(lbl, _row(paths, width, height)) for lbl, paths in rows]
    montage_w = label_w + max(s.width for _, s in strips)
    montage_h = sum(s.height for _, s in strips) + gap * (len(strips) - 1)
    montage = Image.new("RGB", (montage_w, montage_h), (0, 0, 0))
    draw = ImageDraw.Draw(montage)
    y = 0
    for lbl, strip in strips:
        montage.paste(strip, (label_w, y))
        draw.text((6, y + strip.height // 2 - 4), lbl, fill=(255, 255, 255))
        y += strip.height + gap
    out_png = os.path.join(args.out_dir, "multiframe_input_GEN.png")
    montage.save(out_png)
    print(f"\nGenerated {len(imgs)} frames -> {args.out_dir}")
    print(f"  montage (INPUT/GEN): {out_png}")


if __name__ == "__main__":
    main()
