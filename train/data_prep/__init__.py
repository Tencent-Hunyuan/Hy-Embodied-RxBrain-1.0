"""Data preparation for embodied (interleave / multi-frame) fine-tuning.

    convert_interleave.py     trajectory -> per-step interleaved planning samples
    convert_multiframe.py     trajectory -> one joint N-frame world-model sample
    build_vae_roundtrip.py    pre-compute VAE encode->decode mirror frames for
                              the interleave exposure-bias fix (optional)
"""
