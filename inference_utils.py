"""Inference-time utilities for HY-Unified.

Self-contained slice of the pieces the inference scripts need — the image
transform used to derive train/inference resolution parity, and the
task-instruction prompts that cue each generation mode.

The SAME task strings are used at train and inference (parity is load-bearing:
``interleave_inference`` / ``multiframe_inference`` build the prompt the same
way the model was trained). Single source of truth, imported by the scripts.
"""

import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


# Task-instruction prompts appended to the user turn so each generation MODE is
# explicitly cued.
TASK_INSTRUCTION_SINGLE_FRAME = "Generate future image of the goal"
TASK_INSTRUCTION_MULTI_FRAME = "Generate future video of the goal"
TASK_INSTRUCTION_INTERLEAVE = "Generate interleave goal planning"


class VAEImageTransform:
    """Aspect-preserving resize + normalize to [-1, 1] for VAE encoding.

    Sides are clamped to be ``stride``-divisible and within ``[min_size,
    max_size]``; extreme aspect ratios are re-scaled to stay inside ``max_size``
    (otherwise latent position IDs could go out of bounds).
    """

    def __init__(self, max_size=512, min_size=256, stride=16):
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride

    def _make_divisible(self, val):
        return max(self.stride, round(val / self.stride) * self.stride)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        w, h = img.size
        scale = min(self.max_size / max(w, h), 1.0)
        scale = max(scale, self.min_size / min(w, h))
        new_w = self._make_divisible(round(w * scale))
        new_h = self._make_divisible(round(h * scale))
        # Clamp to max_size to handle extreme aspect ratios where the min_size
        # constraint overrides the max_size constraint (e.g. w=1000,h=50 would
        # produce new_w=5120 which causes position ID out-of-bounds).
        max_div = self._make_divisible(self.max_size)
        if new_w > max_div or new_h > max_div:
            # Re-scale to fit within max_size while preserving aspect ratio.
            shrink = min(max_div / new_w, max_div / new_h)
            new_w = self._make_divisible(round(new_w * shrink))
            new_h = self._make_divisible(round(new_h * shrink))
        img = TF.resize(img, (new_h, new_w), InterpolationMode.BICUBIC, antialias=True)
        tensor = TF.to_tensor(img)  # [0, 1]
        tensor = T.Normalize([0.5] * 3, [0.5] * 3)(tensor)  # [-1, 1]
        return tensor
