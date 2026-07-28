"""Offline VAE round-trip mirror builder for the prior-frame exposure-bias fix.

WHY
---
At autoregressive interleave inference, every prior-step frame fed back into
SigLIP is the model's own `vae.decode(latent)` output (lossy), NOT the sharp GT
JPEG used at training. That train/test domain gap is the dominant, content-
independent half of the image-side exposure bias. This script pre-computes, for
every prior/step frame, its VAE-round-tripped version  decode(encode(GT))  so
that — during a fine-tune phase — `UnifiedDataset` can swap a prior frame's path
to this mirror with some probability, making the SigLIP input distribution at
train match inference.

It reproduces the inference fed-back frame faithfully:
  * resize GT to the inference generation resolution (e.g. 144x256, latent 9x16),
  * Normalize([0.5]*3,[0.5]*3) -> [-1,1],
  * vae.encode with the POSTERIOR MEAN (reg.sample = False — inference is a
    deterministic ODE output, not a sampled posterior),
  * VAE in bf16 (same dtype as inference),
  * vae.decode -> clamp(-1,1) -> (x+1)*127.5 -> uint8 -> save PNG (lossless).

Mirror path = src_root -> dst_root prefix swap, extension forced to .png.
Idempotent: existing mirrors are skipped, so the job is resumable.

USAGE
-----
  # (the patched transformers must be importable; e.g. export PYTHONPATH=/path/to/transformers/src)
  # dry-run: just build + print the unique work-list size
  python train/data_prep/build_vae_roundtrip.py \
    --data_glob '${DATA_ROOT}/mm_cond_gen_interleave/*/*.jsonl' \
    --src_root ${DATA_ROOT}/mm_cond_gen \
    --dst_root ${DATA_ROOT}/mm_cond_gen_vae_rt \
    --vae ${MODEL_ROOT}/ae.safetensors \
    --dry_run
  # real run across 8 GPUs
  python train/data_prep/build_vae_roundtrip.py \
    --data_glob '${DATA_ROOT}/mm_cond_gen_interleave/*/*.jsonl' \
    --src_root ${DATA_ROOT}/mm_cond_gen \
    --dst_root ${DATA_ROOT}/mm_cond_gen_vae_rt \
    --vae ${MODEL_ROOT}/ae.safetensors \
    --gpus 0,1,2,3,4,5,6,7 --height 144 --width 256
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time


# --------------------------------------------------------------------------- #
#  Work-list construction (orchestrator side, no torch)
# --------------------------------------------------------------------------- #
def _collect_prior_frames(data_glob: str) -> list:
    """Unique set of frames that can ever be a PRIOR-STEP context frame.

    A frame is a prior frame in some sample iff it is a step TARGET frame
    (assistant `raw_image`) — step j's frame is the target of `*_to_j` and a prior
    of every deeper sample. We also union in any user-side image that appears
    AFTER the first user text (defensive: covers subsets that drop `*_to_1`).
    Obs frames (before the first user text) are NEVER prior frames and are
    excluded — they stay real at inference too.
    """
    files = sorted(glob.glob(data_glob))
    if not files:
        raise SystemExit(f"[build_vae_rt] no jsonl matched: {data_glob}")
    uniq = set()
    t0 = time.time()
    for fi, f in enumerate(files):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                for m in rec.get("messages", []):
                    role = m.get("role")
                    content = m.get("content")
                    if not isinstance(content, list):
                        continue
                    if role == "assistant":
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "generated_image":
                                p = c.get("raw_image") or c.get("image_tokens")
                                if p:
                                    uniq.add(p)
                    elif role in ("user", "system"):
                        seen_text = False
                        for c in content:
                            if not isinstance(c, dict):
                                continue
                            if c.get("type") == "text":
                                seen_text = True
                            elif c.get("type") == "image" and seen_text:
                                p = c.get("image")
                                if p:
                                    uniq.add(p)
        if (fi + 1) % 10 == 0 or fi + 1 == len(files):
            print(f"[build_vae_rt] scanned {fi+1}/{len(files)} shards, "
                  f"{len(uniq)} unique frames, {time.time()-t0:.0f}s", flush=True)
    return sorted(uniq)


def _mirror_path(path: str, src_root: str, dst_root: str) -> str:
    rel = path[len(src_root):] if path.startswith(src_root) else path
    return os.path.splitext(dst_root + rel)[0] + ".png"


# --------------------------------------------------------------------------- #
#  Worker (per-GPU): load VAE once, round-trip its stride of the work-list
# --------------------------------------------------------------------------- #
def _run_worker(args):
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.transforms.functional import InterpolationMode
    from PIL import Image

    # Repository root is three levels up (train/data_prep/<file>).
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from vae_model.autoencoder import load_ae

    with open(args.worklist) as f:
        worklist = json.load(f)
    mine = worklist[args.shard::args.nshard]
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    vae, vae_params = load_ae(args.vae)
    vae.requires_grad_(False)
    vae.eval()
    vae.reg.sample = False                       # posterior MEAN, not a sample
    vae.to(device, dtype=torch.bfloat16)

    H, W = args.height, args.width
    done = skipped = failed = 0
    t0 = time.time()
    for i, src in enumerate(mine):
        dst = _mirror_path(src, args.src_root, args.dst_root)
        if os.path.exists(dst):
            skipped += 1
            continue
        try:
            img = Image.open(src).convert("RGB")
            img = TF.resize(img, (H, W), InterpolationMode.BICUBIC, antialias=True)
            x = TF.to_tensor(img)                          # [0,1] (3,H,W)
            x = (x - 0.5) / 0.5                            # [-1,1]
            x = x.unsqueeze(0).to(device, dtype=torch.bfloat16)
            with torch.no_grad():
                z = vae.encode(x)
                y = vae.decode(z)
            y = y.squeeze(0).float().clamp(-1, 1)
            arr = ((y.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype("uint8")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            Image.fromarray(arr).save(dst)                 # PNG (dst ends .png)
            done += 1
        except Exception as e:
            failed += 1
            if failed <= 20:
                print(f"[worker gpu{args.gpu}] FAIL {src}: {e}", flush=True)
        if (i + 1) % 2000 == 0:
            rate = (done + skipped) / max(1e-6, time.time() - t0)
            print(f"[worker gpu{args.gpu}] {i+1}/{len(mine)} | done={done} "
                  f"skip={skipped} fail={failed} | {rate:.0f}/s", flush=True)
    print(f"[worker gpu{args.gpu}] DONE done={done} skip={skipped} fail={failed} "
          f"of {len(mine)} in {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- #
#  Orchestrator
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", help="glob for the interleave jsonl shards (orchestrator only)")
    ap.add_argument("--src_root", required=True, help="original frame root prefix")
    ap.add_argument("--dst_root", required=True, help="mirror (round-trip) root prefix")
    ap.add_argument("--vae", required=True, help="ae.safetensors path")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--height", type=int, default=144)   # inference gen-res H
    ap.add_argument("--width", type=int, default=256)    # inference gen-res W
    ap.add_argument("--dry_run", action="store_true", help="build work-list, print size, exit")
    ap.add_argument("--rescan", action="store_true", help="ignore cached _worklist.json")
    # worker-mode (internal)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worklist")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()

    if args.worker:
        _run_worker(args)
        return

    if not args.data_glob:
        raise SystemExit("[build_vae_rt] --data_glob is required in orchestrator mode")
    os.makedirs(args.dst_root, exist_ok=True)
    worklist_file = os.path.join(args.dst_root, "_worklist.json")
    if os.path.exists(worklist_file) and not args.rescan:
        with open(worklist_file) as f:
            worklist = json.load(f)
        print(f"[build_vae_rt] reusing cached work-list {worklist_file} "
              f"({len(worklist)} frames; --rescan to rebuild)")
    else:
        worklist = _collect_prior_frames(args.data_glob)
        with open(worklist_file, "w") as f:
            json.dump(worklist, f)
        print(f"[build_vae_rt] wrote work-list -> {worklist_file}")

    if args.dry_run:
        # The remaining-count is a serial stat over the whole worklist (12M+ on
        # netfs = very slow). Only pay it for the dry-run report; in a real run
        # the per-worker os.path.exists skip handles resume without this precount.
        remaining = sum(
            0 if os.path.exists(_mirror_path(p, args.src_root, args.dst_root)) else 1
            for p in worklist
        )
        print(f"[build_vae_rt] {len(worklist)} unique prior frames | "
              f"{remaining} still to round-trip | dst_root={args.dst_root}")
        print("[build_vae_rt] --dry_run set, exiting before round-trip.")
        return
    print(f"[build_vae_rt] {len(worklist)} unique prior frames | dst_root={args.dst_root} "
          f"(workers skip existing mirrors; resume-safe, no precount)", flush=True)

    gpus = [g for g in args.gpus.split(",") if g.strip() != ""]
    procs = []
    for i, g in enumerate(gpus):
        cmd = [
            sys.executable, os.path.abspath(__file__), "--worker",
            "--worklist", worklist_file, "--src_root", args.src_root,
            "--dst_root", args.dst_root, "--vae", args.vae,
            "--gpu", g, "--shard", str(i), "--nshard", str(len(gpus)),
            "--height", str(args.height), "--width", str(args.width),
        ]
        # workers select their device explicitly via cuda:{gpu}; keep all GPUs visible
        env = os.environ.copy()
        log = open(os.path.join(args.dst_root, f"_worker_gpu{g}.log"), "w")
        procs.append((g, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)))
        print(f"[build_vae_rt] launched worker gpu{g} (shard {i}/{len(gpus)})")
    rc = 0
    for g, p in procs:
        c = p.wait()
        rc |= c
        print(f"[build_vae_rt] worker gpu{g} exit {c}")
    print(f"[build_vae_rt] all workers done (rc={rc}). Mirrors under {args.dst_root}")


if __name__ == "__main__":
    main()
