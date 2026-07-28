"""Convert mm_cond_gen `continuous` (x_to_N) records into **multi-frame generation**
training samples — the joint world-model task, NOT the interleave per-step split.

Why this exists (vs. convert_interleave.py)
------------------------------------------------------
`convert_interleave.py` *expands* a trajectory into N independent
single-frame samples (one generation block each), with prior frames recast as
clean SigLIP inputs — the autoregressive-rollout design.

This converter does the opposite: it keeps the **whole assistant turn as one
sample** — an opening text plus all N `generated_image` blocks in order:

    user      : [obs frames] + task
    assistant : "Imagined frame sequence ..."   (text, CE-supervised)
              + generated_image_1 ... generated_image_N   (N FM targets)

Run through `train.dataset.UnifiedDataset._process_generation_sample`, that
becomes the multi-frame sequence

    prompt(+ViT for obs) "open text" <Image>[LATENT]*n</Image> ... (×N) EOS

with N independent per-frame timesteps and the hybrid block attention
(causal across frames, bidirectional within) — i.e. autoregression along time,
diffusion across space, denoised jointly in one forward. The model side already
supports this natively; this converter just feeds it the un-expanded record.

Two load-bearing transforms on the raw mm_cond_gen record:
  1. **Rename assistant `generated_image.image` -> `raw_image`.** `_process_
     generation_sample` reads ONLY `raw_image`/`image_tokens` (train/dataset.py
     :813); the raw key is `image`, so without the rename every target resolves to "".
  2. **Path remap** (foreign cluster -> local), reusing `DEFAULT_REMAP` from the
     interleave converter.

Output is the `messages` schema consumed by `UnifiedDataset`, written as minimal
JSONL (paths + short text only) — small enough for the map-style loader.

Usage
-----
    python train/data_prep/convert_multiframe.py \
        --in  ${DATA_ROOT}/mm_cond_gen/<robot>/jsonl/continuous/2_to_4.jsonl \
        --out ${OUT}/multiframe_train.jsonl \
        [--max_records N] [--verify_exists] \
        [--max_frames 4] [--min_frames 2] [--strip_text]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

# Runnable from any working directory: put the repository root on sys.path so
# the sibling converter imports as a package module.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the path-rewrite and backend-filter helpers from the interleave converter.
from train.data_prep.convert_interleave import DEFAULT_REMAP, remap_path, is_rule_backend


def _extract(record: Dict, rules):
    """Pull (obs_images, task_text, blocks) out of one mm_cond_gen record.

    Returns:
        obs_images: list[str]               — user-side input frame paths (remapped)
        task_text:  str                     — first user/system text (the instruction)
        blocks:     list[("text"|"image", value)]  — assistant content IN ORDER;
                    text values are strings, image values are remapped paths.
    """
    obs_images: List[str] = []
    task_text = ""
    blocks: List = []
    for msg in record.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        if role in ("user", "system"):
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "image" and c.get("image"):
                    obs_images.append(remap_path(c["image"], rules))
                elif c.get("type") == "text" and not task_text:
                    task_text = c.get("text", "")
        elif role == "assistant":
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "text":
                    t = c.get("text", "")
                    if t.strip():
                        blocks.append(("text", t))
                elif c.get("type") == "generated_image":
                    # raw key is `image`; also tolerate raw_image/image_tokens.
                    raw = c.get("image") or c.get("raw_image") or c.get("image_tokens")
                    if raw:
                        blocks.append(("image", remap_path(raw, rules)))
    return obs_images, task_text, blocks


def build_record(record: Dict, strip_text: bool) -> Dict:
    """Build ONE multi-frame `messages` sample from a raw mm_cond_gen record.

    The full assistant turn (opening text + every generated frame, in order) is
    kept as a single sample. Returns {} if the record has no generated frames.
    """
    rules = DEFAULT_REMAP
    obs_images, task_text, blocks = _extract(record, rules)
    n_gen = sum(1 for kind, _ in blocks if kind == "image")
    if n_gen == 0:
        return {}

    user_content = [{"type": "image", "image": p} for p in obs_images]
    if task_text:
        user_content.append({"type": "text", "text": task_text})

    asst_content = []
    for kind, val in blocks:
        if kind == "text":
            if not strip_text:
                asst_content.append({"type": "text", "text": val})
        else:  # image -> rename to raw_image (required by _process_generation_sample)
            asst_content.append({"type": "generated_image", "raw_image": val})

    meta = record.get("metadata", {})
    rid = record.get("id", "rec")
    return {
        "id": f"{rid}_mf{n_gen}",
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": asst_content},
        ],
        "metadata": {
            "source_id": rid,
            "num_input_images": len(obs_images),
            "num_generated_images": n_gen,
            "dataset": meta.get("dataset"),
            "data_class": meta.get("data_class"),
            "construction_mode": meta.get("construction_mode"),
        },
    }


def _sample_paths(sample: Dict) -> List[str]:
    paths = [c["image"] for c in sample["messages"][0]["content"] if "image" in c]
    paths += [c["raw_image"] for c in sample["messages"][1]["content"] if "raw_image" in c]
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="mm_cond_gen continuous jsonl")
    ap.add_argument("--out", required=True, help="output multi-frame training jsonl")
    ap.add_argument("--max_records", type=int, default=None,
                    help="only read this many source records (files are 40-72 GB)")
    ap.add_argument("--max_frames", type=int, default=None,
                    help="drop records with more than N generated frames")
    ap.add_argument("--min_frames", type=int, default=1,
                    help="drop records with fewer than N generated frames (default 1)")
    ap.add_argument("--strip_text", action="store_true",
                    help="drop the assistant opening text -> pure multi-image generation "
                         "(default: keep it as a CE target, task-type interleaved)")
    ap.add_argument("--verify_exists", action="store_true",
                    help="drop samples whose images don't all exist locally")
    ap.add_argument("--no_backend_filter", action="store_true",
                    help="KEEP rule/rule_fallback-backend records (default: skip them — "
                         "goal_backend == rule|rule_fallback; continuous is 'none' so "
                         "this is normally a no-op for multiframe, kept for symmetry)")
    args = ap.parse_args()

    n_in = n_out = n_missing = n_filtered = n_dirty = 0
    filter_backend = not args.no_backend_filter
    with open(args.inp) as fin, open(args.out, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            if args.max_records is not None and n_in >= args.max_records:
                break
            n_in += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if filter_backend and is_rule_backend(rec.get("metadata", {})):
                n_dirty += 1
                continue
            sample = build_record(rec, strip_text=args.strip_text)
            if not sample:
                continue
            n_gen = sample["metadata"]["num_generated_images"]
            if n_gen < args.min_frames or (args.max_frames is not None and n_gen > args.max_frames):
                n_filtered += 1
                continue
            if args.verify_exists and not all(os.path.exists(p) for p in _sample_paths(sample)):
                n_missing += 1
                continue
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[convert_multiframe] read {n_in} records -> wrote {n_out} "
          f"multi-frame samples (skipped {n_dirty} rule-backend, filtered {n_filtered} "
          f"by frame count, dropped {n_missing} for missing images)")


if __name__ == "__main__":
    main()
