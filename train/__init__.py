"""Training pipeline for Hy-Embodied-RxBrain-1.0.

Modules:
    train.py    training entry point (accelerate + DeepSpeed ZeRO-2)
    dataset.py  understanding + generation sample processing (map & streaming)
    collator.py P-Pack collator: one packed row per step, cu_seqlens-bounded
    sampler.py  length-bucketed sampler + ratio-controlled bucket mixing
    data_prep/  converters that turn robot trajectories into training jsonl

The model definition lives in the top-level ``model`` / ``vae_model`` packages,
shared with ``inference/``. See ``train/README.md`` to get started.
"""
