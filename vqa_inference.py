"""Minimal VQA / text-understanding inference for RxBrain (UnifiedMoT).

Pure autoregressive text decoding: (image(s) + question) -> answer text.
No VAE / flow-matching needed. Reuses the SAME prompt construction
(`build_conditioned_sequence`) and KV-cached decode plumbing the interleaved
planner uses for its text step, but decodes all the way to EOS instead of
stopping at the <Image> modality-transition token.
"""
from __future__ import annotations

import argparse
from typing import List

import torch
from transformers.models.hunyuan_vl_mot import HunYuanVLMoTProcessor
from transformers.cache_utils import DynamicCache

from model import UnifiedMoTForConditionalGeneration, maybe_init_generation_path
from interleave_inference import build_conditioned_sequence, INPUT_IMAGE_PLACEHOLDER_IDS


@torch.no_grad()
def answer(model, processor, image_paths: List[str], question: str,
           device, dtype, max_new_tokens: int = 256) -> str:
    cfg = model.config
    eos = cfg.eos_token_id
    inner = model.model

    # No generation task-instruction: this is understanding, not image gen.
    prompt_ids, proc = build_conditioned_sequence(processor, image_paths, question)

    pixel_values = proc.get("pixel_values")
    image_grid_thw = proc.get("image_grid_thw")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device=device, dtype=dtype)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device=device)

    def _dmask(ids):
        n = len(ids)
        inp = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        mod = torch.zeros(1, n, dtype=torch.long, device=device)
        iim = torch.zeros(1, n, dtype=torch.bool, device=device)
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
                use_cache=True, past_key_values=pkv,
                cache_position=torch.arange(prompt_len, device=device))
    pkv = out.past_key_values
    nxt = int(out.logits[0, -1].argmax().item())
    cur = prompt_len
    text_ids = []
    for _ in range(max_new_tokens):
        if nxt == eos:
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
                    use_cache=True, past_key_values=pkv,
                    cache_position=torch.tensor([cur], device=device))
        pkv = out.past_key_values
        nxt = int(out.logits[0, -1].argmax().item())
        cur += 1
    return processor.tokenizer.decode(text_ids, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    print(f"Loading processor from {args.ckpt}...")
    processor = HunYuanVLMoTProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    print(f"Loading model from {args.ckpt}...")
    model = UnifiedMoTForConditionalGeneration.from_pretrained(args.ckpt, dtype=dtype)
    maybe_init_generation_path(model, model_load_path=args.ckpt)
    model.to(device)
    model.eval()

    print(f"\nIMAGES: {args.images}")
    print(f"Q: {args.question}")
    ans = answer(model, processor, args.images, args.question, device, dtype, args.max_new_tokens)
    print(f"A: {ans}")


if __name__ == "__main__":
    main()
