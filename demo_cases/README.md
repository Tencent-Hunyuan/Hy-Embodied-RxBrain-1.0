# RxBrain — Interleaved Planning Demo Cases

Three ready-to-run **interleaved embodied-planning** cases. Given the observation frame(s) in `input/`
plus the task instruction in `prompt.txt`, RxBrain **narrates a step-by-step subgoal plan and imagines the
frame after each step**.

Each case is self-contained:

```
<case>/
  input/        observation frame(s) you feed to the model  (obs_1.jpg, ...)
  prompt.txt    the task instruction to feed
```

(machine-readable list — robot · suggested steps · prompt — in `index.json`)

| case | robot | steps | task |
|---|---|--:|---|
| `umi_fold_sock` | umi | 5 | Fold the purple sock in half |
| `xtrainer_fold_garment` | xtrainer | 3 | Fold the lower garment in half vertically |
| `bridgev2_move_toy` | bridgev2 | 3 | Move the green toy onto the metal pot on the stove |

## Run one case

```bash
CASE=umi_fold_sock
STEPS=5                      # subgoal steps to imagine (see the table above)
python interleave_inference.py \
    --ckpt  <path-or-hf-id of the released checkpoint> \
    --vae   <path to ae.safetensors> \
    --frames  demo_cases/$CASE/input/*.jpg \
    --task    "$(cat demo_cases/$CASE/prompt.txt)" \
    --max_frames $STEPS \
    --num_steps 50 \
    --out_dir out_$CASE
```

The model generates autoregressively: for each step it conditions on the observation frame(s) **plus the
frames it already decoded**, emits the step's subgoal text, and imagines that step's frame. Output lands in
`out_$CASE/` — `joint_step*.png` (imagined frames), `rollout_GEN.png` (montage), and `result.txt` (narration).

### Notes
- `--frames` takes **all** the `input/` frames in order (some cases have 2–3 observation views).
- `--max_frames` sets how many subgoal steps to generate; the table lists a suggested value per case.
- Generation resolution is set by `--height/--width` (script defaults). Keep the aspect ratio of the
  observation frames for best results.
- These are held-out real-robot scenes across three embodiments (umi / xtrainer / bridgev2). Running them
  needs the released checkpoint + FLUX VAE and a CUDA GPU — see the top-level [`README.md`](../README.md).
