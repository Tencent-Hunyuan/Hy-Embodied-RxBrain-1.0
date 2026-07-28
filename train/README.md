# Training / Fine-tuning RxBrain

Fine-tunes **Hy-Embodied-RxBrain-1.0** on understanding (cross-entropy) and image
generation (flow matching over VAE latents) in one packed sequence — together or
separately, mixed by integer ratios in a single run.

| Task | Data | Notes |
|---|---|---|
| ① VQA / understanding | jsonl (or parquet) | cross-entropy only |
| ② Text-to-image | parquet | flow matching only |
| ③ Multi-frame generation | jsonl | 4 frames, one joint forward |
| ④ Interleaved planning | jsonl **+ VAE mirrors** | per-step text + frame; needs [Stage 2](#stage-2--build-the-vae-mirror-frames-essential) |

The model lives in the repo-root `model/` and `vae_model/` packages, shared with
`inference/` — a checkpoint you fine-tune runs with the inference scripts unchanged.

## Quick start

```bash
pip install -r requirements.txt          # from the repository root

# interleave task only: trajectories -> training jsonl -> VAE mirror frames
python train/data_prep/convert_interleave.py --in <traj>.jsonl --out interleave.jsonl
python train/data_prep/build_vae_roundtrip.py --data_glob 'interleave.jsonl' \
    --src_root <frames_root> --dst_root <mirror_root> --vae ae.safetensors --gpus 0,1,2,3

cp train/configs/example.yaml train/configs/my_run.yaml   # edit paths + ratios
bash train/launch.sh                                       # single or multi node
```

## Data format

One entry in `data.train_data_path` = one **bucket** = one weight in
`data.mix_ratios`. A bucket is a parquet directory, a `.jsonl` glob, or a list of
either. A bucket's share per step is `ratio / Σ(ratios)`; the bucket defining epoch
length is `argmax(mix_ratios)`.

Each jsonl line is `{"messages": [...], "metadata": {...}}`, roles `user` /
`assistant` — there is **no system prompt**, so prompt through the user turn at
inference too. Content is a list of typed items:

| `type` | where | meaning |
|---|---|---|
| `text` | any | plain text |
| `image` | user | conditioning / understanding image → vision tower (`_v`) |
| `generated_image` | assistant | flow-matching **target**; path in `raw_image` (`_g`) |

The task is **inferred from the sample's shape** — you never declare it:

| shape | task | appended instruction |
|---|---|---|
| no `generated_image` | understanding / VQA | *(none)* |
| 1 `generated_image` | single-frame generation | `Generate future image of the goal` |
| ≥2 `generated_image` | multi-frame generation | `Generate future video of the goal` |
| 1 `generated_image` + `metadata.step_index` | interleaved planning | `Generate interleave goal planning` |

These instruction strings are duplicated in `inference/utils.py`; that parity is what
makes train and inference token layouts identical — **keep the two in sync**. T2I
parquet deliberately gets no instruction (foundation captioning).

### ① VQA / understanding (jsonl)

```jsonc
{"messages": [
  {"role": "user",      "content": [{"type": "image", "image": "/data/img.jpg"},
                                    {"type": "text",  "text": "Which drawer is open?"}]},
  {"role": "assistant", "content": [{"type": "text",  "text": "<think>…</think><answer>the top one</answer>"}]}
]}
```

Cross-entropy only — expect `image=0.0000` in the loss readout for a pure
understanding bucket. Understanding can also come from parquet (embedded image
bytes + `caption` / `conversations` / `question`+`answer`); the schema is sniffed
per file, so no extra config.

### ② Text-to-image (parquet)

T2I trains from **parquet with embedded JPEG bytes**, not jsonl — random-access
jsonl would pull a whole multi-hundred-MB shard per row off network storage. Point a
bucket at a *directory*; `.parquet` files are found recursively, `.partial` skipped,
a corrupt file skipped with a warning, and the file index cached beside the data.

```
t2i_bucket/
├── shard_00000.parquet     jpg: binary   (raw JPEG bytes)
├── shard_00001.parquet     text: string  (caption; blip2_caption also accepted)
└── …
```

The caption column is sniffed once and cached, so **all files in one bucket must
agree**. Each row becomes `user:[caption] → assistant:[generated_image]` and trains
with flow matching only — expect `text=0.0000` for a pure T2I bucket.
`text_cond_dropout_prob` drops the caption on that fraction of rows, which is what
makes classifier-free guidance work at inference.

This is the layout the [BLIP3-o](https://github.com/JiuhaiChen/BLIP3o) recaptioned
T2I corpora ship in, so those can be pointed at directly; convert your own captions
to the same two columns.

### ③ Multi-frame generation (jsonl)

The target frames sit in **one** assistant turn and are denoised **jointly in a
single forward** (autoregression along time, diffusion across space), wrapped in one
`<Video> … </Video>` block automatically. The released model uses **4 frames** — the
same `--num_frames 4` that `inference/multiframe.py` defaults to — so keep training
and inference at the same count; the code itself accepts any N ≥ 2.

```jsonc
{"messages": [
  {"role": "user",      "content": [{"type": "image", "image": "/data/obs_1.jpg"},
                                    {"type": "text",  "text": "fold the garment"}]},
  {"role": "assistant", "content": [{"type": "text", "text": "Imagined frame sequence…"},
                                    {"type": "generated_image", "raw_image": "/data/f1.jpg"},
                                    {"type": "generated_image", "raw_image": "/data/f2.jpg"},
                                    {"type": "generated_image", "raw_image": "/data/f3.jpg"},
                                    {"type": "generated_image", "raw_image": "/data/f4.jpg"}]}
]}
```

Use `--max_frames 4 --min_frames 2` on the converter to hold the count down.

### ④ Interleaved planning (jsonl)

**A Y-step trajectory becomes Y separate samples, not one.** In sample *k* the user
turn holds the observations, the task, and the already-produced steps as alternating
(plan text, frame) pairs — prior frames entering through the **vision tower**,
exactly as at autoregressive inference. The assistant turn holds **only step k**.

```
sample k:
  user      : [obs_1 … obs_m] + task
              + (text_1, frame_1) … (text_{k-1}, frame_{k-1})   ← prior steps → vision tower
  assistant : text_k (CE)  +  generated_image frame_k (FM target)
```

```jsonc
// step 3 of a 5-step trajectory, 1 observation frame
{"messages": [
  {"role": "user", "content": [
      {"type": "image", "image": "/data/ep7/obs_1.jpg"},           // observation
      {"type": "text",  "text":  "put the cup on the shelf"},      // task
      {"type": "text",  "text":  "reach for the cup"},             // step 1 plan
      {"type": "image", "image": "/data/ep7/f1.jpg"},              // step 1 frame
      {"type": "text",  "text":  "lift the cup"},                  // step 2 plan
      {"type": "image", "image": "/data/ep7/f2.jpg"}               // step 2 frame
  ]},
  {"role": "assistant", "content": [
      {"type": "text", "text": "move the cup above the shelf"},    // -> CE
      {"type": "generated_image", "raw_image": "/data/ep7/f3.jpg"} // -> FM target
  ]}
],
 "metadata": {"step_index": 3, "num_steps": 5, "num_context_frames": 3, "source_id": "ep7"}}
```

`metadata` is **required**, not decorative:

| field | why |
|---|---|
| `step_index` | selects the interleave instruction; marks the obs/prior boundary |
| `num_context_frames` | total user-side frames = observations + prior-step frames |
| `num_steps` | bookkeeping / filtering only |

The split is `num_obs = num_context_frames - (step_index - 1)`: the first `num_obs`
user-side images are observations, the rest are prior-step frames. Stage 2 relies on
this arithmetic (not on text positions), so it stays correct when texts are empty or
dropped by CFG. **Miss either field and the sample silently degrades to single-frame
generation**, with the Stage-2 fix becoming a no-op on it. `convert_interleave.py`
emits all of this for you.

## Preparing ③ / ④ data from robot trajectories

Getting raw robot episodes into the intermediate trajectory record below is
dataset-specific and **not shipped here**:

```
user      : [obs frame(s)] + task text
assistant : (text_1, generated_image_1), (text_2, generated_image_2), …   # per step
metadata  : { step_index, num_context_frames, …, *_backend }
```

From there: **convert → mirrors → train, in that order.** Stage 2 applies to the
interleave task only.

### Stage 1 — convert

```bash
# Interleaved planning: expand a Y-step trajectory into Y samples, prior output
# frames recast as clean vision-tower inputs (what inference/interleave.py expects).
python train/data_prep/convert_interleave.py \
    --in <traj>.jsonl --out interleave_train.jsonl [--keep_step_text] [--verify_exists]

# Multi-frame: keep the turn whole — opening text + all N frames, one forward.
python train/data_prep/convert_multiframe.py \
    --in <traj>.jsonl --out multiframe_train.jsonl [--max_frames 4] [--min_frames 2]
```

- `DEFAULT_REMAP` in `convert_interleave.py` is empty; add `(old_prefix, new_prefix)`
  pairs only if recorded image paths differ from disk.
- Records whose text came from a rule-based fallback backend
  (`metadata.*_backend ∈ {rule, rule_fallback}`) are skipped; `--no_backend_filter` keeps them.

### Stage 2 — build the VAE mirror frames (essential)

**Skip this and you bake in a train/test mismatch that compounds with every rollout
step.** At inference, every prior frame in the context is the model's own lossy
`vae.decode` output; in raw training data it is a sharp GT JPEG. So the model learns
to read crisp frames and is asked to read its own reconstructions — the conditioning
distribution is wrong from step 2 on, and since each output becomes the next input,
the error compounds. This image-side **exposure bias** is the main failure mode of
long interleaved rollouts.

Fix: for every GT frame that can appear as a prior-step frame, pre-compute its VAE
`encode → decode` mirror, then swap prior frames to it with probability
`prior_frame_vae_roundtrip_prob`.

```bash
python train/data_prep/build_vae_roundtrip.py \
    --data_glob '<interleave_jsonl_glob>' \   # the jsonl from Stage 1
    --src_root <frames_root>               \   # prefix to strip
    --dst_root <mirror_root>               \   # prefix to write under
    --vae /path/to/ae.safetensors --gpus 0,1,2,3,4,5,6,7 --height 144 --width 256
```

```yaml
task:
  prior_frame_vae_roundtrip_prob: 0.5                 # 0 disables the fix
  prior_frame_vae_roundtrip_src_root: <frames_root>   # must match --src_root
  prior_frame_vae_roundtrip_dst_root: <mirror_root>   # must match --dst_root
```

- **Order matters.** The work-list comes from scanning the *converted* jsonl for every
  assistant `raw_image` — run it before Stage 1 and it finds nothing; change the
  converter output later and re-run so new frames get mirrors.
- **One lossless PNG per GT frame** at `src_root → dst_root`; budget disk accordingly.
- **Resumable and idempotent** — existing mirrors are skipped. `--dry_run` reports the
  work-list size; it is cached at `<dst_root>/_worklist.json` (`--rescan` rebuilds).
- **Faithful to inference** by design: generation-resolution resize, `[-1,1]`
  normalize, **posterior mean** (not a sample — inference is a deterministic ODE
  output), bf16. Set `--height/--width` to the resolution you generate at.
- **Only prior-step frames are swapped.** Observations stay real (they are real at
  inference too) and the FM **target** is always the true GT frame — never its mirror.
- A missing mirror is a **silent no-op** for that frame, so incomplete coverage
  degrades quietly: check the builder's `done/skip/fail` tallies.

Multi-frame training needs none of this — all N frames come from one forward, so no
generated frame is ever fed back.

## Configure

Copy `configs/example.yaml`, replace every `${DATA_ROOT}` / `${MODEL_ROOT}`
placeholder with a real absolute path (**no env expansion**), and set your ratios.
Every knob is documented inline; the load-bearing ones:

| knob | section | effect |
|---|---|---|
| `train_data_path` / `mix_ratios` | data | the buckets and their sampling weights |
| `max_length` / `max_total_tokens` | data | per-sample and per-packed-row token caps |
| `model_path` / `vae_path` | model | checkpoint to fine-tune; `ae.safetensors` |
| `mlp_g_intermediate_size` | model | generation-FFN width (`null` = base width) |
| `ce_weight` / `mse_weight` | task | text vs image-generation loss balance |
| `text_cond_dropout_prob` | task | conditioning dropout that enables CFG |
| `max_image_size` / `min_image_size` / `image_stride` | task | generated-frame resolution bounds |
| `understanding_max_pixels` | task | ViT input cap — main driver of worst-case memory |
| `enable_input_vae_latent` | task | also feed input frames as clean VAE latents (generation samples only) |
| `prior_frame_vae_roundtrip_prob` | task | Stage-2 exposure-bias fix (`0` = off) |
| `freeze_*` / `lr_*` | model / training | per-stream freezing and learning rates |
| `resume_from_checkpoint` | training | resume from the newest `live-{step}/` in `output_dir` (model + optimizer + scheduler + RNG); empty dir starts fresh. `live_keep_count` bounds how many are kept |

Global batch = `per_device_train_batch_size × num_processes ×
gradient_accumulation_steps`. Keep `Σ(mix_ratios)` dividing the per-step draw so
bucket interleaving stays exact.

⚠️ **`mlp_g_intermediate_size` must match the base checkpoint.** If the checkpoint
already has `_g` weights, this must equal their width — the loader widens a *fresh*
generation MLP (Net2Wider) but never re-shapes an existing one. A mismatch either
fails to load or silently trains at the wrong width.

## Launch

```bash
bash train/launch.sh                     # reads train/configs/example.yaml

accelerate launch --config_file train/configs/deepspeed_zero2.yaml \
    train/finetune.py --config train/configs/example.yaml     # equivalent, explicit
```

Multi node: your scheduler exports `WORLD_SIZE`, `RANK`, `GPU_NUM`, `MASTER_ADDR`,
`MASTER_PORT`; `launch.sh` forwards them and those **override** the node counts in the
accelerate yaml, so the same script scales unchanged. Each node runs a GPU preflight
and aborts fast, naming the host, if it has no visible GPU.

`configs/deepspeed_zero2.yaml` defaults to 1 node × 8 GPUs. ZeRO-2 shards optimizer
state and gradients but replicates parameters — budget roughly
`(model in bf16) + (optimizer state / ranks)` per rank. A full checkpoint is ~12 GB,
so `output_dir` needs room for `live_keep_count` of them plus the final save.

