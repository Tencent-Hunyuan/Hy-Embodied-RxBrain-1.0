"""Convert mm_cond_gen chat-format records into per-step generation samples
with a **SigLIP-context handoff** for interleaved (multi-image) generation.

Why this exists
---------------
mm_cond_gen records are interleaved:  user[obs frames + task]  ->
assistant[ (text_1, gen_img_1), (text_2, gen_img_2), ... ].

Training them as ONE packed sequence is WRONG for the current FM infra:
each generated image is noised at an independent timestep
(`flow_matching_modules.sample_timesteps`), and the generation-block attention
override lets a later image attend to an *earlier image's noised VAE latent*
(`attention_mot_packed._apply_generation_block`, KV range = [sample_start, g_end]).
At inference the earlier image is already fully denoised, so train != test.

The fix (per the design): the context that image k sees for image <k must be the
**SigLIP-encoded frame** (clean, identical at train and test), NOT the noised VAE
latent. We realise this by **segmenting on images** — a Y-frame record expands into
Y independent generation samples, where in each sample every prior output image is
recast as a SigLIP *input* and only the last image is the (noised) FM target:

    joint sample k:
        user      : [obs frames] + task
                    + [text_1, frame_1, ..., text_{k-1}, frame_{k-1}]
                    (prior frames -> SigLIP input; prior plan texts -> masked context)
        assistant : text_k (CE target)  +  { generated_image : frame_k }  (FM target)

Each sample ends in exactly ONE generation block, so it is a leaf (nothing attends
to its noised latent). Every prior frame enters as a SigLIP input image — the same
representation used at autoregressive inference, where the prior frames are the
decoded outputs of earlier steps — and the prior plan texts stay in context so the
model has a textual memory of the steps it has already produced (progress is no
longer inferred from an imperfect generated frame). Only the final (text_k, frame_k)
pair carries loss; everything before it is masked context. This reuses the
already-validated single-image generation path verbatim, so the chat-template
train/test parity established in inference_packed.py carries over unchanged.

(v1, joint=False: the plan text is folded into the user prompt and the assistant
emits only the image — pure frame prediction, prior texts dropped.)

`n_to_1` / `continuous` records (Y == 1) collapse to a single sample — i.e. plain
TI2I frame prediction — so this converter handles both interleave types.

Output records are in the `messages` schema consumed by
`train.dataset.UnifiedDataset`. The generated image uses the `raw_image`
key (what `_process_generation_sample` reads); input frames use `image`.

Path remap (optional): if your `mm_cond_gen` image paths carry a prefix that
differs from where the frames actually live, add `(old_prefix, new_prefix)` pairs
to DEFAULT_REMAP below (empty by default = no rewrite).

Usage
-----
    python train/data_prep/convert_interleave.py \
        --in  ${DATA_ROOT}/mm_cond_gen/<robot>/jsonl/planning/1_to_3.jsonl \
        --out ${OUT}/interleave_train.jsonl \
        [--max_records N] [--keep_step_text] [--verify_exists]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

# Optional prefix rewrites for image paths, tried in order (first match wins).
# Empty by default. Add your own pairs if the paths recorded in the jsonl differ
# from where the frames live on disk, e.g.:
#   DEFAULT_REMAP = [("/some/recorded/prefix/", "/your/local/prefix/")]
DEFAULT_REMAP = []


def remap_path(p: str, rules) -> str:
    for old, new in rules:
        if p.startswith(old):
            p = new + p[len(old):]
            break
    # (Add any dataset-specific path fixups here, e.g. repairing a double-nested
    # extracted-frames directory. None applied by default.)
    return p


# --------------------------------------------------------------------------- #
#  Dirty-data filter: skip records whose goal/plan/imagination text came from a
#  rule-based fallback backend instead of the LLM. Per the user's instruction
#  these are low-quality and must NOT enter training. The check is CLASS-AGNOSTIC
#  (it inspects every field that can carry a "rule"/"rule_fallback" verdict), so
#  one helper covers continuous / planning / high-level-planning / imagination:
#    * goal_backend                 — all classes (continuous is 'none' -> kept)
#    * fsi_main_task_backend        — imagination only
#    * planner_components_backend   — planning only (dict; ANY subfield triggers)
# --------------------------------------------------------------------------- #
RULE_BACKENDS = {"rule", "rule_fallback"}


def is_rule_backend(metadata: Dict) -> bool:
    """Return True if this record should be SKIPPED (rule/rule_fallback backend)."""
    if not isinstance(metadata, dict):
        return False
    if metadata.get("goal_backend") in RULE_BACKENDS:
        return True
    if metadata.get("fsi_main_task_backend") in RULE_BACKENDS:
        return True
    pc = metadata.get("planner_components_backend")
    if isinstance(pc, dict) and any(v in RULE_BACKENDS for v in pc.values()):
        return True
    return False


def _extract(record: Dict, rules):
    """Pull (obs_images, steps) out of one mm_cond_gen chat record.

    Returns:
        obs_images: list[str]  — user-side input frame paths (remapped)
        steps:      list[(text, gen_image_path)]  — assistant text/image pairs,
                    in generation order. text may be "" .
    """
    obs_images: List[str] = []
    steps: List = []
    for msg in record.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        if role in ("user", "system"):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "image" and c.get("image"):
                    obs_images.append(remap_path(c["image"], rules))
        elif role == "assistant":
            pending_text = ""
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "text":
                    pending_text = (pending_text + " " + c.get("text", "")).strip()
                elif c.get("type") == "generated_image":
                    # mm_cond_gen uses the `image` key on generated_image
                    raw = c.get("image") or c.get("raw_image") or c.get("image_tokens")
                    if raw:
                        steps.append((pending_text, remap_path(raw, rules)))
                    pending_text = ""
    return obs_images, steps


def expand_record(record: Dict, keep_step_text: bool, joint: bool = False) -> List[Dict]:
    """Expand one interleave record into per-step SigLIP-handoff samples.

    joint=False (v1): plan text is folded into the user prompt (conditioning);
        the assistant emits only the image. Pure frame prediction.
    joint=True  (v2): the per-step plan text stays an ASSISTANT target (text CE)
        followed by the generated image (FM); the user holds only obs/prior frames
        + the overall task. Trains true text+image interleaved generation.
    """
    rules = DEFAULT_REMAP
    obs_images, steps = _extract(record, rules)
    if not steps:
        return []

    base_task = ""
    # Recover the task text from the user turn (first text content).
    for msg in record.get("messages", []):
        if msg.get("role") in ("user", "system") and isinstance(msg.get("content"), list):
            for c in msg["content"]:
                if isinstance(c, dict) and c.get("type") == "text":
                    base_task = c.get("text", "")
                    break
            if base_task:
                break

    out = []
    meta = record.get("metadata", {})
    rid = record.get("id", "rec")
    for k in range(len(steps)):
        # context frames: obs + previously generated frames 1..k-1 (SigLIP)
        ctx_frames = list(obs_images) + [steps[j][1] for j in range(k)]
        target = steps[k][1]
        step_text = steps[k][0]

        if joint:
            # Interleaved-context handoff (per the design). Segment by image: in
            # sample k every prior output image becomes a SigLIP *input*, the prior
            # output PLAN TEXTS stay in context, and only the last (text, image)
            # pair is a training target:
            #   user      : [obs frames] + task
            #               + [text_1, frame_1, ..., text_{k-1}, frame_{k-1}]
            #               (prior frames -> SigLIP input; prior texts -> masked context)
            #   assistant : text_k (CE)  +  generated frame_k (FM target)
            # Prior frames enter via SigLIP exactly as at autoregressive inference,
            # and the prior plan texts give the model a textual memory of the steps
            # it has already produced (so progress no longer has to be inferred from
            # an imperfect generated frame).
            user_content = [{"type": "image", "image": fp} for fp in obs_images]
            if base_task:
                user_content.append({"type": "text", "text": base_task})
            for j in range(k):
                if steps[j][0]:
                    user_content.append({"type": "text", "text": steps[j][0]})
                user_content.append({"type": "image", "image": steps[j][1]})
            asst_content = []
            if step_text:
                asst_content.append({"type": "text", "text": step_text})
            asst_content.append({"type": "generated_image", "raw_image": target})
        else:
            # v1: plan text folded into the prompt; assistant emits only the image.
            parts = [base_task] if base_task else []
            if keep_step_text:
                parts += [steps[j][0] for j in range(k + 1) if steps[j][0]]
            elif step_text:
                parts.append(step_text)
            prompt_text = "\n".join(p for p in parts if p).strip()
            user_content = [{"type": "image", "image": fp} for fp in ctx_frames]
            user_content.append({"type": "text", "text": prompt_text})
            asst_content = [{"type": "generated_image", "raw_image": target}]

        out.append({
            "id": f"{rid}_step{k+1}of{len(steps)}",
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": asst_content},
            ],
            "metadata": {
                "source_id": rid,
                "step_index": k + 1,
                "num_steps": len(steps),
                "num_context_frames": len(ctx_frames),
                "data_class": meta.get("data_class"),
                "construction_mode": meta.get("construction_mode"),
            },
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="mm_cond_gen jsonl")
    ap.add_argument("--out", required=True, help="output training jsonl")
    ap.add_argument("--max_records", type=int, default=None,
                    help="only read this many source records")
    ap.add_argument("--keep_step_text", action="store_true",
                    help="(v1 only) include all plan-step texts up to step k in the prompt "
                         "(default: only the current step's instruction)")
    ap.add_argument("--joint", action="store_true",
                    help="v2: keep per-step plan text as an assistant generation target "
                         "(text CE) followed by the frame (FM) — true text+image interleave")
    ap.add_argument("--verify_exists", action="store_true",
                    help="drop samples whose images don't all exist locally")
    ap.add_argument("--no_backend_filter", action="store_true",
                    help="KEEP rule/rule_fallback-backend records (default: skip them — "
                         "see is_rule_backend; goal_backend / fsi_main_task_backend / "
                         "planner_components_backend == rule|rule_fallback are dropped)")
    ap.add_argument("--max_context_frames", type=int, default=None,
                    help="keep only per-step samples whose context (obs + prior "
                         "frames) is <= N; e.g. 3 => total images <= 4 (1 generated). "
                         "Drops deep samples BEFORE the verify_exists stat. "
                         "Default: no cap (all depths kept).")
    args = ap.parse_args()

    n_in = n_out = n_missing = n_dirty = 0
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
            for sample in expand_record(rec, args.keep_step_text, joint=args.joint):
                if args.max_context_frames is not None and \
                        sample["metadata"]["num_context_frames"] > args.max_context_frames:
                    continue
                if args.verify_exists:
                    paths = [c["image"] for c in sample["messages"][0]["content"]
                             if "image" in c]
                    paths += [c["raw_image"] for c in sample["messages"][1]["content"]
                              if "raw_image" in c]
                    if not all(os.path.exists(p) for p in paths):
                        n_missing += 1
                        continue
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n_out += 1

    print(f"[convert_interleave] read {n_in} records -> wrote {n_out} "
          f"per-step samples (skipped {n_dirty} rule-backend, dropped {n_missing} "
          f"for missing images)")


if __name__ == "__main__":
    main()
