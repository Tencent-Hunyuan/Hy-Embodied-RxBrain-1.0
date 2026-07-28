"""Inference entry points for Hy-Embodied-RxBrain-1.0.

Each module in this package is runnable as a script, e.g.

    python inference/text2image.py --model_path ... --prompt "..."

The scripts put the repository root on ``sys.path`` themselves, so they also
work as modules (``python -m inference.text2image``) and from any working
directory. The model definition itself lives in the top-level ``model`` /
``vae_model`` packages, which are shared with ``train/``.
"""
