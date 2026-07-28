"""Training entry point for the RxBrain UnifiedMoT model.

Fine-tunes `model.UnifiedMoTForConditionalGeneration` jointly on understanding
(cross-entropy) and image generation (flow matching over VAE latents):

  * HunYuanVLMoTProcessor (Qwen2-VL image processor backed) for vision inputs
  * UnifiedDataCollator — P-Pack: one packed row per step, cu_seqlens-bounded
  * LengthBucketedSampler — length-homogeneous mini-batches to cut padding
  * Per-group learning rates (6 groups: vision_tower / text / vision /
    generation / shared / flow_matching), split by parameter-name regex
  * Optional Net2Wider `mlp_g` initialization at startup (model.mot_init_utils)
  * FLUX VAE encoding on the fly (raw pixels -> continuous latents -> model)

Run:
    accelerate launch train/finetune.py --config train/configs/example.yaml

or via the launcher, which also handles multi-node env plumbing:
    bash train/launch.sh
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

# Runnable from any working directory: put the repository root on sys.path so
# the shared `model` / `vae_model` packages and the `train` package resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.distributed as dist
import yaml
from datetime import timedelta
from accelerate import Accelerator, DistributedDataParallelKwargs, DataLoaderConfiguration
from accelerate.utils import set_seed, InitProcessGroupKwargs
from torch.utils.data import ChainDataset, ConcatDataset, DataLoader
from tqdm import tqdm
from transformers import get_scheduler
from transformers.models.hunyuan_vl_mot import HunYuanVLMoTProcessor

from model import (
    UnifiedMoTConfig,
    UnifiedMoTForConditionalGeneration,
    maybe_init_generation_path,
)
from train.dataset import TaskConfig, UnifiedDataset, StreamingUnifiedDataset, ParquetT2IDataset, ParquetMMUDataset
from train.collator import UnifiedDataCollator
from train.sampler import (
    LengthBucketedSampler, LengthBalancedShuffleBuffer, RatioMixIterableDataset,
    ChainIterableDataset, estimate_sample_length, estimate_record_length,
)
from vae_model.autoencoder import load_ae


# =============================================================================
# Config loading
# =============================================================================

def load_config(yaml_path: str) -> Dict[str, Any]:
    """Load and flatten training YAML."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Common keys (only what this pipeline reads)
    return {
        # Model
        "model_path": cfg["model"]["model_path"],
        "vae_path": cfg["model"].get("vae_path"),
        "latent_patch_size": cfg["model"].get("latent_patch_size", 2),
        "max_latent_size": cfg["model"].get("max_latent_size", 32),
        "timestep_shift": cfg["model"].get("timestep_shift", 1.0),
        "mlp_g_intermediate_size": cfg["model"].get("mlp_g_intermediate_size"),
        "mlp_g_init_noise_std": float(cfg["model"].get("mlp_g_init_noise_std", 1e-4)),
        "freeze_vision_tower": cfg["model"].get("freeze_vision_tower", False),
        "freeze_text_transformer": cfg["model"].get("freeze_text_transformer", False),
        "freeze_vision_transformer": cfg["model"].get("freeze_vision_transformer", False),
        "freeze_generation_transformer": cfg["model"].get("freeze_generation_transformer", False),
        "freeze_shared": cfg["model"].get("freeze_shared", False),
        # Data
        "train_data_path": cfg["data"]["train_data_path"],
        # Optional: integer mix ratios parallel to train_data_path. When set,
        # ALL sources converge to iterable (jsonl -> StreamingUnifiedDataset) and
        # are interleaved by RatioMixIterableDataset so each optimizer step's
        # global batch carries the exact ratio. None = single-paradigm
        # behavior (no change to existing configs).
        "mix_ratios": cfg["data"].get("mix_ratios", None),
        # StreamingUnifiedDataset shuffle-buffer size (records). The buffer is
        # filled BEFORE the first sample is yielded; under self-rank-sharding
        # (1/(W*num_workers) sparse selection) filling N records scans
        # N*(W*num_workers) lines, so a large buffer makes a long COLD START.
        # Lower it (e.g. 2000) for fast startup; raise for stronger local shuffle.
        "shuffle_buffer_size": int(cfg["data"].get("shuffle_buffer_size", 20000)),
        "max_length": cfg["data"].get("max_length", 4096),
        "max_total_tokens": cfg["data"].get("max_total_tokens", 8192),
        "image_path_remap": cfg["data"].get("image_path_remap", {}),
        "shuffle": cfg["data"].get("shuffle", True),     # set False to skip per-epoch file+row shuffle
        # Task
        "task_config": cfg["task"].get("task_config", "all"),
        "vae_image_downsample": cfg["task"].get("vae_image_downsample", 16),
        "max_image_size": cfg["task"].get("max_image_size", 512),
        "min_image_size": cfg["task"].get("min_image_size", 256),
        "image_stride": cfg["task"].get("image_stride", 16),
        # Understanding ViT input-image pixel cap. 0 = use the processor's
        # preprocessor_config.json default (max_pixels=4194304=2048^2 -> up to 4096
        # ViT tokens/image). Set to bound the worst-case high-res doc/infographic
        # image (the bs-OOM driver): tokens/image ≈ max_pixels / (patch*merge)^2
        # = max_pixels/1024, so 1048576 (1024^2) -> ~1024 tokens/image.
        "understanding_max_pixels": int(cfg["task"].get("understanding_max_pixels", 0)),
        "text_cond_dropout_prob": cfg["task"].get("text_cond_dropout_prob", 0.1),
        "boundary_loss_weight": cfg["task"].get("boundary_loss_weight", 0.0),
        "mse_weight": cfg["task"].get("mse_weight", 1.0),
        "ce_weight": cfg["task"].get("ce_weight", 1.0),
        "enable_input_vae_latent": cfg["task"].get("enable_input_vae_latent", False),
        "train_image_start_token": cfg["task"].get("train_image_start_token", False),
        "train_video_start_token": cfg["task"].get("train_video_start_token", True),
        # Prior-frame VAE round-trip augmentation (image-side exposure-bias fix).
        # 0.0 = no-op (exact current behavior). See TaskConfig + build_vae_roundtrip.py.
        "prior_frame_vae_roundtrip_prob": float(cfg["task"].get("prior_frame_vae_roundtrip_prob", 0.0)),
        "prior_frame_vae_roundtrip_src_root": cfg["task"].get("prior_frame_vae_roundtrip_src_root", ""),
        "prior_frame_vae_roundtrip_dst_root": cfg["task"].get("prior_frame_vae_roundtrip_dst_root", ""),
        # Training
        "output_dir": cfg["training"]["output_dir"],
        "num_train_epochs": cfg["training"].get("num_train_epochs", 3),
        # Optional hard cap on optimizer steps (0 = no cap). Lets a short fine-tune
        # phase run a fraction of an epoch WITHOUT a fractional num_train_epochs
        # (which would crash range(float)). num_train_epochs stays integer; the
        # existing `global_step >= max_train_steps` break ends the run early.
        "max_train_steps": int(cfg["training"].get("max_train_steps", 0)),
        "per_device_train_batch_size": cfg["training"].get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": cfg["training"].get("gradient_accumulation_steps", 1),
        "learning_rate": float(cfg["training"].get("learning_rate", 1e-4)),
        "lr_vision_tower": cfg["training"].get("lr_vision_tower"),
        "lr_text_transformer": cfg["training"].get("lr_text_transformer"),
        "lr_vision_transformer": cfg["training"].get("lr_vision_transformer"),
        "lr_generation_transformer": cfg["training"].get("lr_generation_transformer"),
        "lr_shared": cfg["training"].get("lr_shared"),
        "lr_flow_matching": cfg["training"].get("lr_flow_matching"),
        "weight_decay": float(cfg["training"].get("weight_decay", 0.0)),
        "warmup_steps": cfg["training"].get("warmup_steps", 0),
        "warmup_ratio": cfg["training"].get("warmup_ratio", None),  # alternative to warmup_steps
        "lr_scheduler_type": cfg["training"].get("lr_scheduler_type", "cosine"),
        "max_grad_norm": float(cfg["training"].get("max_grad_norm", 1.0)),
        "mixed_precision": cfg["training"].get("mixed_precision", "bf16"),
        # NCCL collective watchdog timeout (minutes). Default 30 = unchanged. Lower
        # it (e.g. 10) so a HANG aborts with a rank-naming traceback BEFORE the
        # cluster scheduler silently SIGKILLs the job — turns a black-box hang into
        # "Rank X stuck on collective Y, ranks [...] never arrived".
        "nccl_timeout_min": int(cfg["training"].get("nccl_timeout_min", 30)),
        "save_steps": cfg["training"].get("save_steps", 5000),
        "logging_steps": cfg["training"].get("logging_steps", 10),
        "num_workers": cfg["training"].get("num_workers", 4),
        "pin_memory": cfg["training"].get("pin_memory", True),
        "dataloader_prefetch_factor": cfg["training"].get("dataloader_prefetch_factor", 2),
        "seed": cfg["training"].get("seed", 42),
        # Checkpoint / auto-resume (two-tier, matches hymot-unified-vl):
        # - live-{step}/: full state (model + optim + scheduler + RNG +
        #   training_state.json), high-frequency, rolling. Used for resume.
        #   ONLY written when resume_from_checkpoint=True.
        # - checkpoint-{step}/: weights only (model + processor +
        #   training_state.json), low-frequency, permanent. Used for eval.
        "resume_from_checkpoint": bool(cfg["training"].get("resume_from_checkpoint", False)),
        "resume_from_checkpoint_step": int(cfg["training"].get("resume_from_checkpoint_step", 500)),
        "live_keep_count": int(cfg["training"].get("live_keep_count", 2)),
        # Logging
        "use_tensorboard": cfg.get("logging", {}).get("use_tensorboard", True),
        # Packing
        "bucket_size_multiplier": cfg["training"].get("bucket_size_multiplier", 50),
        # Diagnostics
        "enable_phase_timing": cfg.get("logging", {}).get("enable_phase_timing", False),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to training YAML")
    return p.parse_args()


# =============================================================================
# Parameter-group classification by parameter-name regex
# =============================================================================

SHARED_PARAM_PATTERNS = ["embed_tokens", "lm_head", ".norm.", "rotary_emb", "query_layernorm", "key_layernorm"]
FLOW_MATCHING_PREFIXES = ("time_embedder.", "vae2llm.", "llm2vae.", "latent_pos_embed.")


def is_shared_param(name: str) -> bool:
    if ".norm." in f".{name}.":
        return True
    return any(p in name for p in SHARED_PARAM_PATTERNS if p != ".norm.")


def is_generation_path_param(name: str) -> bool:
    return "_g." in name and not is_shared_param(name)


def is_vision_path_param(name: str) -> bool:
    return "_v" in name and not is_shared_param(name) and not is_generation_path_param(name)


def is_text_path_param(name: str) -> bool:
    return (
        not is_shared_param(name)
        and not is_vision_path_param(name)
        and not is_generation_path_param(name)
    )


def apply_freezing(model, CONFIG, logger):
    """Set requires_grad on parameter groups according to freeze_* config flags."""
    # Vision tower (HYViT2 — `model.visual`)
    visual = model.model.visual
    rg = not CONFIG["freeze_vision_tower"]
    for p in visual.parameters():
        p.requires_grad = rg
    logger.info(f"vision_tower frozen={not rg}")

    # Language model (text / vision / generation / shared)
    lm = model.model.language_model
    flag_to_check = [
        ("freeze_text_transformer", is_text_path_param, "text_transformer"),
        ("freeze_vision_transformer", is_vision_path_param, "vision_transformer"),
        ("freeze_generation_transformer", is_generation_path_param, "generation_transformer"),
        ("freeze_shared", is_shared_param, "shared"),
    ]
    for flag, predicate, name in flag_to_check:
        rg = not CONFIG[flag]
        for n, p in lm.named_parameters():
            if predicate(n):
                p.requires_grad = rg
        logger.info(f"{name} frozen={not rg}")

    # FM modules (top-level): always trainable unless explicitly excluded later
    for fm_prefix in FLOW_MATCHING_PREFIXES:
        for n, p in model.named_parameters():
            if n.startswith(fm_prefix):
                p.requires_grad = True


def build_optimizer(model, CONFIG, logger):
    """AdamW with 6 parameter groups, each with its own LR."""
    base_lr = CONFIG["learning_rate"]
    lrs = {
        "vision_tower": CONFIG["lr_vision_tower"] or base_lr,
        "text_transformer": CONFIG["lr_text_transformer"] or base_lr,
        "vision_transformer": CONFIG["lr_vision_transformer"] or base_lr,
        "generation_transformer": CONFIG["lr_generation_transformer"] or base_lr,
        "shared": CONFIG["lr_shared"] or base_lr,
        "flow_matching": CONFIG["lr_flow_matching"] or base_lr,
    }
    lrs = {k: float(v) for k, v in lrs.items()}

    groups: Dict[str, List] = {k: [] for k in lrs}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(pref) for pref in FLOW_MATCHING_PREFIXES):
            groups["flow_matching"].append(p)
        elif name.startswith("model.visual."):
            groups["vision_tower"].append(p)
        elif name.startswith("model.language_model."):
            lm_name = name[len("model.language_model."):]
            if is_shared_param(lm_name):
                groups["shared"].append(p)
            elif is_generation_path_param(lm_name):
                groups["generation_transformer"].append(p)
            elif is_vision_path_param(lm_name):
                groups["vision_transformer"].append(p)
            else:
                groups["text_transformer"].append(p)
        else:
            groups["shared"].append(p)

    param_groups = []
    for gname, gparams in groups.items():
        if not gparams:
            continue
        param_groups.append({"params": gparams, "lr": lrs[gname], "name": gname})
        logger.info(f"  group '{gname}': {len(gparams)} tensors, lr={lrs[gname]:g}")

    return torch.optim.AdamW(
        param_groups, lr=base_lr, weight_decay=CONFIG["weight_decay"],
    )


# =============================================================================
# Task config builder
# =============================================================================

def build_task_config(CONFIG: Dict[str, Any]) -> TaskConfig:
    """Translate CONFIG dict into a TaskConfig dataclass for the dataset."""
    return TaskConfig(
        enable_text_to_text=CONFIG["task_config"] in ("all", "t2t", "understanding"),
        enable_multimodal_to_text=CONFIG["task_config"] in ("all", "m2t", "understanding"),
        enable_text_to_image=CONFIG["task_config"] in ("all", "t2i", "generation"),
        enable_multimodal_to_image=CONFIG["task_config"] in ("all", "m2i", "ti2i", "generation"),
        enable_interleaved_generation=CONFIG["task_config"] in ("all", "interleaved"),
        enable_input_vae_latent=CONFIG.get("enable_input_vae_latent", False),
        train_image_start_token=CONFIG.get("train_image_start_token", False),
        train_video_start_token=CONFIG.get("train_video_start_token", True),
        prior_frame_vae_roundtrip_prob=CONFIG.get("prior_frame_vae_roundtrip_prob", 0.0),
        prior_frame_vae_roundtrip_src_root=CONFIG.get("prior_frame_vae_roundtrip_src_root", ""),
        prior_frame_vae_roundtrip_dst_root=CONFIG.get("prior_frame_vae_roundtrip_dst_root", ""),
        max_length=CONFIG["max_length"],
        vae_image_downsample=CONFIG["vae_image_downsample"],
        max_image_size=CONFIG["max_image_size"],
        min_image_size=CONFIG["min_image_size"],
        image_stride=CONFIG["image_stride"],
        max_latent_size=CONFIG["max_latent_size"],
        text_cond_dropout_prob=CONFIG["text_cond_dropout_prob"],
        image_path_remap=CONFIG.get("image_path_remap", {}),
    )


# =============================================================================
# VAE encoding on-the-fly (raw pixels → continuous latents)
# =============================================================================

@torch.no_grad()
def encode_images_via_vae(vae, padded_images: List[torch.Tensor], device, dtype):
    """Encode a list of raw pixel tensors to a list of VAE-encoded latents.

    Per-image loop: each call processes one image through the VAE encoder.
    We tried batched encode (padding + masked GroupNorm) but the masked
    GroupNorm overhead ate up the kernel-launch savings; group-by-shape gave
    1.3x speedup but per-image is what's used in production for numerical
    consistency. Keep this implementation simple.
    """
    out = []
    vae_param_dtype = next(vae.parameters()).dtype
    for img in padded_images:
        img_in = img.unsqueeze(0).to(device=device, dtype=vae_param_dtype, non_blocking=True)
        z = vae.encode(img_in)
        if hasattr(z, "latent_dist"):
            z = z.latent_dist.sample()
        elif hasattr(z, "sample"):
            z = z.sample()
        out.append(z.squeeze(0).to(dtype=dtype))
    return out


# =============================================================================
# Live (rolling) checkpoint helpers — match hymot-unified-vl two-tier scheme:
# live-{step}/: full state, rolling, used for resume
# checkpoint-{step}/: weights only, permanent, used for eval
# =============================================================================

_LIVE_RE = re.compile(r"^live-(\d+)$")


def _list_live_ckpts(output_dir: str):
    """Return [(step, abs_path), ...] sorted DESCENDING by step.

    Only matches `live-{digits}` exactly — half-written `.tmp` directories from
    a crashed save are skipped, so resume always picks a known-good ckpt.
    """
    if not os.path.isdir(output_dir):
        return []
    out = []
    for name in os.listdir(output_dir):
        m = _LIVE_RE.match(name)
        if not m:
            continue
        full = os.path.join(output_dir, name)
        if os.path.isdir(full):
            out.append((int(m.group(1)), full))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def find_latest_live_ckpt(output_dir: str):
    """Newest `live-{step}/` directory under output_dir, or None."""
    cands = _list_live_ckpts(output_dir)
    return cands[0][1] if cands else None


def prune_live_ckpts(output_dir: str, keep: int, logger=None):
    """Remove all but the `keep` most recent live-{step}/ directories."""
    if keep <= 0:
        return
    cands = _list_live_ckpts(output_dir)
    for step, path in cands[keep:]:
        try:
            shutil.rmtree(path)
            if logger:
                logger.info(f"  pruned old live ckpt: live-{step}")
        except Exception as e:
            if logger:
                logger.warning(f"  failed to prune live-{step}: {e}")


# =============================================================================
# Main
# =============================================================================

def _move_to_device(x, device):
    """Recursively move tensors in a (possibly nested) batch to `device`.

    Needed on the map-style path because the dataloader is NOT passed to
    accelerator.prepare() (the data is already per-rank sharded), so the
    collated batch arrives on CPU and must be placed on the compute device
    before the model forward. No-op for tensors already on `device` (so it is
    harmless on the iterable path where the prepared loader already placed them).
    """
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_move_to_device(v, device) for v in x)
    return x


def main():
    args = parse_args()
    CONFIG = load_config(args.config)

    # Poison-sample probe target (see UnifiedDataset.__getitem__). Set before the
    # DataLoader is built so forked workers inherit it. Disable by unsetting.
    os.environ["IMG_PROBE_DIR"] = os.path.join(CONFIG["output_dir"], "img_probe")

    accelerator = Accelerator(
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        mixed_precision=CONFIG["mixed_precision"],
        # find_unused_parameters=False: relies on `mask_apply_3way` placeholder
        # fallback (model/modeling_decoder_mot.py:_dispatch) routing every
        # mlp_t/mlp_v/mlp_g param through `func(placeholder).mean() * 0` even
        # when its modality is absent in the current packed batch. Frozen
        # vision_tower is requires_grad=False so DDP/DeepSpeed won't expect
        # grads on it. If any "expected to mark variable ready" error fires
        # at startup, flip back to True; we kept it on for safety historically
        # but each backward then walks all params, which is wasted work.
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=False),
            # Shorten the NCCL collective watchdog so a HANG self-reports (rank +
            # collective) before the scheduler SIGKILLs silently. Default 30 min.
            InitProcessGroupKwargs(timeout=timedelta(minutes=CONFIG["nccl_timeout_min"])),
        ],
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
        log_with="tensorboard" if CONFIG.get("use_tensorboard", True) else None,
        project_dir=CONFIG["output_dir"],
    )
    set_seed(CONFIG["seed"])

    # Logger: stream to stdout (main rank) and append to <output_dir>/training.log
    is_main = accelerator.is_main_process
    logger = logging.getLogger("train.finetune")
    logger.setLevel(logging.INFO if is_main else logging.WARNING)
    logger.propagate = False
    logger.handlers.clear()
    if is_main:
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(sh)
        fh = logging.FileHandler(os.path.join(CONFIG["output_dir"], "training.log"), mode="a")
        fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
    if accelerator.is_main_process:
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        with open(os.path.join(CONFIG["output_dir"], "resolved_config.yaml"), "w") as f:
            yaml.safe_dump(CONFIG, f, sort_keys=False)

    # ---- Processor (tokenizer + image + video + chat template) -------------
    logger.info("Loading processor and tokenizer...")
    processor = HunYuanVLMoTProcessor.from_pretrained(
        CONFIG["model_path"], trust_remote_code=True,
    )
    tokenizer = processor.tokenizer

    # ---- Cap understanding-image resolution (memory/OOM control) ------------
    # The image_processor's max_pixels comes from preprocessor_config.json
    # (default 4194304=2048^2 -> up to 4096 ViT tokens/image). High-res doc/OCR/
    # infographic samples (ureader_*, infographic_vqa) routinely hit ~4000 tokens;
    # several in one pack OOM-stall a rank -> NCCL collective timeout (the bs=10
    # crash). Lowering max_pixels caps the worst image (e.g. 1048576=1024^2 ->
    # ~1024 tokens) WITHOUT touching median images (natural photos/captions are
    # already below the cap). Set BOTH max_pixels and size.longest_edge so every
    # transformers version's smart-resize honors it.
    _ump = CONFIG["understanding_max_pixels"]
    if _ump and _ump > 0:
        ip = processor.image_processor
        ip.max_pixels = _ump
        if isinstance(getattr(ip, "size", None), dict) and "longest_edge" in ip.size:
            ip.size["longest_edge"] = _ump
        logger.info(
            f"Understanding ViT image cap: max_pixels={_ump} "
            f"(~{_ump // (ip.patch_size * ip.merge_size) ** 2} tokens/image max)"
        )

    # ---- VAE (frozen encoder) ----------------------------------------------
    logger.info(f"Loading VAE from {CONFIG['vae_path']}...")
    vae, vae_params = load_ae(CONFIG["vae_path"])
    vae.requires_grad_(False)
    vae.eval()
    # Expose downsample on the module for encode_images_via_vae
    vae.downsample = int(vae_params.downsample)
    # bf16 to match the LLM compute dtype and let SDPA dispatch to faster
    # backends. (We previously kept fp32 to dodge NaN on a few images during
    # the early debug; bf16 should be fine on the COYO/LAION corpus — fall
    # back to fp32 if NaN reappears.)
    vae_dtype = torch.bfloat16 if CONFIG["mixed_precision"] == "bf16" else torch.float32
    vae.to(accelerator.device, dtype=vae_dtype)

    # ---- Model -------------------------------------------------------------
    logger.info(f"Loading UnifiedMoTForConditionalGeneration from {CONFIG['model_path']}...")
    model_cfg = UnifiedMoTConfig.from_pretrained(CONFIG["model_path"])
    # Inject FM hyperparameters from training config
    model_cfg.latent_patch_size = CONFIG["latent_patch_size"]
    model_cfg.max_latent_size = CONFIG["max_latent_size"]
    model_cfg.timestep_shift = CONFIG["timestep_shift"]
    model_cfg.vae_image_downsample = CONFIG["vae_image_downsample"]
    model_cfg.boundary_loss_weight = CONFIG["boundary_loss_weight"]
    model_cfg.flow_loss_weight = CONFIG["mse_weight"]
    model_cfg.text_loss_weight = CONFIG["ce_weight"]
    if CONFIG["mlp_g_intermediate_size"]:
        # IMPORTANT: do NOT set model_cfg.text_config.mlp_g_intermediate_size yet —
        # if we did, the MoTDecoderLayer would build mlp_g at the wider size and
        # from_pretrained would fail because base ckpt has no _g.* keys at that
        # shape. Defer the widening to maybe_init_generation_path (post-load).
        target_mlp_g_inter = int(CONFIG["mlp_g_intermediate_size"])
        model_cfg.text_config.mlp_g_init_noise_std = CONFIG["mlp_g_init_noise_std"]
    else:
        target_mlp_g_inter = None
    # vae_z_channels — pull from VAE params
    model_cfg.vae_z_channels = int(vae_params.z_channels)

    model = UnifiedMoTForConditionalGeneration.from_pretrained(
        CONFIG["model_path"], config=model_cfg, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,  # stream shards into the model instead of holding
                                 # a second full state-dict copy in host RAM — keeps
                                 # per-rank load peak ~halved so 8 ranks fit the
                                 # 137 GB cgroup limit on this pod.
    )

    # Init mlp_g from mlp_v (Net2Wider or same-size copy). No-op if ckpt
    # already contains _g keys (resume / TI2I continuation). When
    # target_mlp_g_inter is set, mlp_g is REBUILT wider post-load before init.
    maybe_init_generation_path(
        model, model_load_path=CONFIG["model_path"],
        noise_std=CONFIG["mlp_g_init_noise_std"], logger_=logger,
        target_mlp_g_intermediate_size=target_mlp_g_inter,
    )
    # Sync the config so saved checkpoints know about the widened size.
    if target_mlp_g_inter is not None:
        model.config.text_config.mlp_g_intermediate_size = target_mlp_g_inter

    apply_freezing(model, CONFIG, logger)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: trainable={n_trainable:,} / total={n_total:,}")

    # ---- Dataset, sampler, collator, dataloader -----------------------------
    logger.info("Loading training datasets...")
    task_config = build_task_config(CONFIG)
    train_paths = CONFIG["train_data_path"]
    if isinstance(train_paths, str):
        train_paths = [train_paths]
    mix_ratios = CONFIG.get("mix_ratios")
    if mix_ratios is not None:
        mix_ratios = [int(r) for r in mix_ratios]
        if len(mix_ratios) != len(train_paths):
            raise ValueError(
                f"data.mix_ratios {mix_ratios} must be parallel to train_data_path "
                f"({len(train_paths)} entries)"
            )
    datasets = []
    has_iterable = False
    has_map = False
    # Self-sharded mix: in mix mode every source self-rank-shards (no IDS) so each
    # rank processes only its 1/W of samples — avoids the W× processing
    # amplification of accelerate's IterableDatasetShard on heavy samples.
    self_sharded = mix_ratios is not None
    _rank = accelerator.process_index
    _world = accelerator.num_processes

    def _parquet_is_mmu(path: str) -> bool:
        """Sniff a parquet source by schema: T2I generation has a `jpg` bytes
        column; MMU understanding has an `image` struct/list column (+ caption /
        conversations / question-answer). Probes the first readable parquet under
        a directory, or the file itself. This is what routes a parquet source to
        ParquetMMUDataset vs ParquetT2IDataset with NO extra config plumbing."""
        import pyarrow.parquet as _pq
        if os.path.isdir(path):
            probes = []
            for root, _, files in os.walk(path):
                pq_here = [f for f in files if f.endswith(".parquet") and ".partial" not in f]
                if pq_here:
                    probes.append(os.path.join(root, sorted(pq_here)[0]))
                    break
        else:
            probes = [path]
        for pr in probes:
            try:
                names = _pq.read_schema(pr).names
            except Exception:
                continue
            if "jpg" in names:
                return False
            if "image" in names:
                return True
            raise ValueError(f"Parquet schema not recognized as T2I or MMU: {pr} cols={names}")
        raise ValueError(f"No readable parquet to sniff under {path}")

    def _make_parquet(path: str):
        """Build the right parquet dataset (MMU understanding vs T2I generation)
        for a dir or single .parquet file. Both self-rank-shard on the mix path."""
        if _parquet_is_mmu(path):
            return ParquetMMUDataset(
                data_path=path, processor=processor, config=task_config,
                shuffle=CONFIG["shuffle"], seed=CONFIG["seed"],
                rank=_rank if self_sharded else 0,
                world_size=_world if self_sharded else 1,
            )
        return ParquetT2IDataset(
            data_dir=path, processor=processor, config=task_config,
            shuffle=CONFIG["shuffle"],
            rank=_rank if self_sharded else 0,
            world_size=_world if self_sharded else 1,
        )

    def _build_source(p: str):
        """Build ONE dataset for a single string path. Returns (dataset, kind)
        where kind is 'iterable' or 'map'."""
        if p.endswith(".jsonl") or p.endswith(".json"):
            if mix_ratios is not None:
                # MIX path: streaming iterable, SELF rank-shards (rank/world_size),
                # dataloader is NOT prepared below -> no IDS, no W× discard.
                # RAM stays O(shuffle_buffer).
                return StreamingUnifiedDataset(
                    data_path=p, processor=processor, config=task_config,
                    shuffle=CONFIG["shuffle"], seed=CONFIG["seed"],
                    shuffle_buffer_size=CONFIG["shuffle_buffer_size"],
                    rank=_rank, world_size=_world,
                ), "iterable"
            # Map-style path: per-rank LOAD-TIME sharding (dataloader NOT prepared).
            return UnifiedDataset(
                data_path=p, processor=processor, config=task_config,
                rank=_rank, world_size=_world,
            ), "map"
        if os.path.isdir(p) or p.endswith(".parquet"):
            # Schema-sniffed: ParquetMMUDataset (understanding) vs ParquetT2IDataset
            # (generation). _make_parquet handles the self-shard rank/world wiring.
            return _make_parquet(p), "iterable"
        raise ValueError(f"Unsupported data path: {p}")

    def _build_group(paths):
        """Build ONE iterable source from a list of paths (a logical mix source,
        e.g. the 188-file interleave corpus or laion+coyo). jsonl files collapse
        into a SINGLE multi-file StreamingUnifiedDataset (even worker split across
        files + per-epoch file-order shuffle + shuffle buffer); each parquet dir
        or .parquet file becomes a ParquetMMUDataset/ParquetT2IDataset (by schema);
        multiple resulting datasets are chained. All members self-rank-shard."""
        jsonl_files = [p for p in paths if p.endswith(".jsonl") or p.endswith(".json")]
        parquet_srcs = [p for p in paths if p.endswith(".parquet") or os.path.isdir(p)]
        unknown = [p for p in paths if p not in jsonl_files and p not in parquet_srcs]
        if unknown:
            raise ValueError(f"Unsupported paths in group: {unknown}")
        members = []
        if jsonl_files:
            members.append(StreamingUnifiedDataset(
                data_path=jsonl_files, processor=processor, config=task_config,
                shuffle=CONFIG["shuffle"], seed=CONFIG["seed"],
                shuffle_buffer_size=CONFIG["shuffle_buffer_size"],
                rank=_rank, world_size=_world,
            ))
        for d in parquet_srcs:
            members.append(_make_parquet(d))
        if not members:
            raise ValueError(f"Empty group: {paths}")
        return members[0] if len(members) == 1 else ChainIterableDataset(members)

    # Glob expansion (mix mode only): a string entry with glob chars (* ? [)
    # expands to its matching files and becomes a GROUP (one ratio entry) — so a
    # 188-file source can be written as one ".../**/*.jsonl" line. Count is
    # preserved (1 entry -> 1 entry), so mix_ratios stays parallel.
    if mix_ratios is not None:
        def _expand(entry):
            if isinstance(entry, str) and any(ch in entry for ch in "*?["):
                matched = sorted(glob.glob(entry, recursive=True))
                if not matched:
                    raise ValueError(f"glob matched no files: {entry}")
                return matched
            return entry
        train_paths = [_expand(e) for e in train_paths]

    for entry in train_paths:
        if isinstance(entry, (list, tuple)):
            # GROUPED source: many paths -> ONE ratio entry. Requires mix mode.
            if mix_ratios is None:
                raise ValueError("Grouped/glob train_data_path entries require data.mix_ratios.")
            ds = _build_group(list(entry))
            has_iterable = True
            logger.info(f"  - group ({len(entry)} paths): {len(ds)} examples")
        else:
            ds, kind = _build_source(entry)
            if kind == "iterable":
                has_iterable = True
            else:
                has_map = True
            logger.info(f"  - {entry}: {len(ds)} examples")
        datasets.append(ds)

    if mix_ratios is not None:
        # Ratio-mixed multi-source run: every source is iterable now. Each B*W
        # window LBSB+IDS see must be exactly proportioned, which holds iff the
        # period p = sum(ratios) divides B*W (see RatioMixIterableDataset docs).
        # Self-sharded mix: each rank runs an INDEPENDENT per-rank stream (no IDS),
        # so the per-step composition is schedule[step*B : (step+1)*B], identical
        # across ranks. Every optimizer step's global batch is therefore exactly
        # proportioned iff p = sum(ratios) divides B (per_device_train_batch_size)
        # — NOT B*W (that was the IDS-window rule of the old prepared path).
        B = CONFIG["per_device_train_batch_size"]
        p = sum(mix_ratios)
        if B % p != 0:
            logger.warning(
                f"sum(mix_ratios)={p} does NOT divide per_device_train_batch_size={B}: "
                f"the per-step global ratio will WOBBLE around the target (it averages "
                f"exactly to {mix_ratios} over every {p} steps, since all ranks share the "
                f"schedule phase). Harmless for training; set B to a multiple of {p} "
                f"(e.g. {p} or {2*p}) if you want every single step exactly proportioned."
            )
        length_source = max(range(len(mix_ratios)), key=lambda i: mix_ratios[i])
        train_dataset = RatioMixIterableDataset(
            datasets, mix_ratios, length_source=length_source, seed=CONFIG["seed"],
            # infinite: dominant cycles too -> no data-driven epoch boundary ->
            # all ranks run exactly max_train_steps and break synchronously (no
            # cross-rank desync from per-rank shard sizes differing by ±1 record).
            infinite=True,
        )
        logger.info(
            f"RatioMixIterableDataset (self-sharded, infinite): ratios={mix_ratios} "
            f"(p={p}, B={B}), dominant/length_source=idx {length_source} "
            f"({train_paths[length_source]})"
        )
    elif has_iterable and has_map:
        raise ValueError(
            "Cannot mix JSONL (map-style) and parquet (iterable) datasets without "
            "`data.mix_ratios`. Set mix_ratios to enable ratio-controlled mixing."
        )
    elif len(datasets) == 1:
        train_dataset = datasets[0]
    elif has_iterable:
        # IterableDataset must use ChainDataset (yields from each in sequence).
        # Each underlying ParquetT2IDataset already shards files across DDP ranks,
        # so ChainDataset just stitches their outputs.
        train_dataset = ChainDataset(datasets)
    else:
        train_dataset = ConcatDataset(datasets)

    sampler = None
    is_iterable = isinstance(train_dataset, torch.utils.data.IterableDataset)
    if not is_iterable:
        # Map-style: use length-bucketing sampler. IterableDataset handles its
        # own sharding/shuffling internally; passing a sampler is illegal.
        #
        # length_fn must be CHEAP: LengthBucketedSampler computes lengths over
        # EVERY index up front (once, lazily on first __iter__, on every rank).
        # The old `estimate_sample_length(train_dataset[i])` materialized each
        # sample — i.e. re-ran the full image-load/VAE/SigLIP pipeline over the
        # entire corpus before step 0. Fine at the 100-sample overfit scale, but
        # at millions of map-style interleave samples it hangs the run on length
        # precompute. estimate_record_length reads only the raw messages schema
        # (text char + image counts), never touching an image. Resolve the
        # global index back to the underlying UnifiedDataset's raw record
        # (self.data) — through ConcatDataset's cumulative_sizes if multi-file.
        if isinstance(train_dataset, ConcatDataset):
            _subs = train_dataset.datasets
            _cum = train_dataset.cumulative_sizes

            def _raw_record(i):
                d = bisect.bisect_right(_cum, i)
                local = i - (_cum[d - 1] if d > 0 else 0)
                return _subs[d].data[local]
        else:
            def _raw_record(i):
                return train_dataset.data[i]

        sampler = LengthBucketedSampler(
            train_dataset,
            batch_size=CONFIG["per_device_train_batch_size"],
            length_fn=lambda i: estimate_record_length(_raw_record(i)),
            # world_size=1: the dataset is ALREADY per-rank sharded at load time
            # (UnifiedDataset rank/world_size above), so this sampler only shuffles
            # + length-buckets THIS rank's local records. It is NOT handed to
            # accelerate.prepare(), so there is no second (BatchSamplerShard) split.
            world_size=1,
            bucket_size_multiplier=CONFIG["bucket_size_multiplier"],
            seed=CONFIG["seed"],
            shuffle=True,
        )

        # Identical optimizer-step count on every rank or gradient all-reduce
        # desyncs → NCCL deadlock → heartbeat kill (the epoch-boundary bug, but
        # now from uneven SHARD sizes rather than uneven batch dealing). Each
        # rank's local shard differs by at most a few records, so clamp all ranks
        # to the GLOBAL MIN batch count. Deterministic across runs (shard size is
        # deterministic) → resume-safe.
        _bs = CONFIG["per_device_train_batch_size"]
        _local_batches = len(sampler) // _bs
        if accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized():
            _t = torch.tensor([_local_batches], device=accelerator.device, dtype=torch.long)
            dist.all_reduce(_t, op=dist.ReduceOp.MIN)
            _global_min_batches = int(_t.item())
        else:
            _global_min_batches = _local_batches
        sampler.set_max_samples(_global_min_batches * _bs)
        logger.info(
            f"Per-rank data shard: rank {accelerator.process_index} holds "
            f"{len(train_dataset)} records / {_local_batches} batches; "
            f"clamped to global-min {_global_min_batches} batches/epoch."
        )
    else:
        # Iterable path: balance per-batch token counts across ranks.
        #
        # accelerate's IterableDatasetShard gives each rank a contiguous slice
        # pool[r*B : (r+1)*B] of the IDS pool (size B*W). Without intervention
        # ranks see unsorted random samples — cross-rank packed_T max/min sits
        # at ~1.22 and drives ~10-15% of backward time as straggler wait.
        #
        # LengthBalancedShuffleBuffer sorts each IDS pool by length and then
        # permutes the entries so the rank's contiguous slice maps to the
        # stride-W subset of the sorted pool. Each rank ends up with B samples
        # covering the full length range → per-rank batch SUM is nearly
        # equal across ranks (smoke test: max/min ~ 1.047). Per-rank within-
        # batch length variance is high, but that's fine — packing only cares
        # about the total token count for forward/backward sizing.
        #
        # SKIP for the self-sharded mix path: LBSB's permutation assumes IDS
        # slices a B*W pool across ranks; with self-rank-sharding there is no IDS
        # and each rank already owns an independent 1/W stream, so LBSB would be
        # meaningless (and its B*W buffering would re-introduce a large cold-start
        # fill). The self-sharded path forgoes cross-rank length balancing (same as
        # the proven map path), trading a little straggler jitter for no W× waste.
        if not self_sharded and bool(CONFIG.get("balance_packed_len_across_ranks", True)):
            train_dataset = LengthBalancedShuffleBuffer(
                train_dataset,
                batch_size=CONFIG["per_device_train_batch_size"],
                num_processes=accelerator.num_processes,
            )
            logger.info(
                f"Wrapped iterable dataset with LengthBalancedShuffleBuffer "
                f"(batch_size={CONFIG['per_device_train_batch_size']}, "
                f"num_processes={accelerator.num_processes}, "
                f"pool_size={CONFIG['per_device_train_batch_size'] * accelerator.num_processes})"
            )

    collator = UnifiedDataCollator(max_total_tokens=CONFIG["max_total_tokens"])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG["per_device_train_batch_size"],
        sampler=sampler,
        collate_fn=collator,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        prefetch_factor=CONFIG["dataloader_prefetch_factor"] if CONFIG["num_workers"] > 0 else None,
        drop_last=True,
    )

    # ---- Optimizer + scheduler ---------------------------------------------
    optimizer = build_optimizer(model, CONFIG, logger)

    # num_update_steps_per_epoch — per-rank optimizer steps in one epoch.
    #   - map-style (per-rank sharded): the dataset is split across ranks at LOAD
    #     time and the sampler is clamped to the global-min batch count, so
    #     len(train_dataloader) is ALREADY the per-rank batch count. The dataloader
    #     is NOT prepared (see below), so this is read as-is — do NOT divide by
    #     num_processes (that would under-count by W and make the LR/epoch math
    #     think the run is W× shorter than it is).
    #   - IterableDataset: ParquetT2IDataset.__iter__ shards per-rank internally,
    #     but its __len__ returns the GLOBAL row count, and DataLoader.__len__
    #     for IterableDataset returns ceil(len/batch_size) without rank-awareness,
    #     so we still divide by num_processes here.
    #   - Mix path: RatioMixIterableDataset.__len__ is dominant-source-driven and
    #     APPROXIMATE (the minority cycles), so this is an estimate — set
    #     `training.max_train_steps` for a precise step budget.
    if is_iterable:
        raw_loader_len = len(train_dataloader)
        per_rank_loader_len = math.ceil(raw_loader_len / accelerator.num_processes)
    else:
        per_rank_loader_len = len(train_dataloader)
    num_update_steps_per_epoch = math.ceil(
        per_rank_loader_len / CONFIG["gradient_accumulation_steps"]
    )
    max_train_steps = CONFIG["num_train_epochs"] * num_update_steps_per_epoch

    # Optional hard cap for a short fine-tune phase. Applied BEFORE the scheduler
    # is built so warmup/total step counts match the real (capped) run; the inner
    # epoch loop still breaks early via `global_step >= max_train_steps`.
    _max_steps_override = int(CONFIG.get("max_train_steps", 0) or 0)
    if _max_steps_override > 0 and _max_steps_override < max_train_steps:
        logger.info(
            f"max_train_steps override: {max_train_steps} -> {_max_steps_override} "
            f"(short fine-tune phase)"
        )
        max_train_steps = _max_steps_override

    warmup_ratio = CONFIG.get("warmup_ratio")
    if warmup_ratio is not None:
        warmup_steps = int(max_train_steps * float(warmup_ratio))
    else:
        warmup_steps = int(CONFIG["warmup_steps"])

    lr_scheduler = get_scheduler(
        name=CONFIG["lr_scheduler_type"],
        optimizer=optimizer,
        # accelerator.prepare() wraps the scheduler as AcceleratedScheduler.
        # On every sync micro-batch, AcceleratedScheduler calls the inner
        # scheduler `num_processes` times (`accelerate/scheduler.py:75-82`);
        # on non-sync micro-batches it early-returns. So per outer optimizer
        # step the inner advances by exactly `num_processes` — independent of
        # `gradient_accumulation_steps`. Multiply step counts by num_processes
        # only; do NOT also multiply by grad_accum (would slow the schedule
        # by grad_accum×). Verified empirically on 8 GPU + ZeRO-2 with both
        # grad_accum=1 and grad_accum=4 — see probe_lr_scheduler.py.
        num_warmup_steps=warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    # DeepSpeed needs train_micro_batch_size_per_gpu. accelerate normally infers
    # it from the PREPARED dataloader — but on the map path we do NOT prepare the
    # dataloader (it is already per-rank sharded; preparing it would make
    # BatchSamplerShard re-shard it). So set it explicitly here, before prepare,
    # or DeepSpeed engine init raises "batch size ... auto ... please specify".
    _ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if _ds_plugin is not None:
        _ds_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            CONFIG["per_device_train_batch_size"]
        )
        _ds_plugin.deepspeed_config["gradient_accumulation_steps"] = (
            CONFIG["gradient_accumulation_steps"]
        )

    if is_iterable and not self_sharded:
        # Prepared iterable path (e.g. T2I-only parquet): accelerate inserts
        # IterableDatasetShard to do rank sharding + places batches on device.
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler,
        )
    else:
        # Map path AND self-sharded mix path: the dataset is ALREADY per-rank
        # sharded, so DON'T let accelerate re-shard the dataloader (that would
        # insert IterableDatasetShard, which reads B*W and discards (W-1)/W of the
        # PROCESSED samples — the W× amplification that starved the GPU). Prepare
        # everything else; iterate the raw dataloader and move each batch to device
        # in the loop.
        model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler,
        )

    # ---- Tensorboard tracker ----------------------------------------------
    if CONFIG.get("use_tensorboard", True) and is_main:
        accelerator.init_trackers("unified_mot_t2i_pretrain")

    # ---- Training loop -----------------------------------------------------
    logger.info("=" * 80)
    logger.info(f"Examples (global rows): {sum(len(ds) for ds in datasets)}")
    logger.info(f"Epochs: {CONFIG['num_train_epochs']}")
    logger.info(f"Per-device BS (samples/pack): {CONFIG['per_device_train_batch_size']}")
    logger.info(f"Grad accumulation: {CONFIG['gradient_accumulation_steps']}")
    logger.info(f"World size (num_processes): {accelerator.num_processes}")
    logger.info(f"Global batch size: {CONFIG['per_device_train_batch_size'] * accelerator.num_processes * CONFIG['gradient_accumulation_steps']}")
    logger.info(f"Steps per epoch (per-rank): {num_update_steps_per_epoch}")
    logger.info(f"Max train steps: {max_train_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Learning rate: {CONFIG['learning_rate']}")
    logger.info("=" * 80)
    progress = tqdm(range(max_train_steps), disable=not accelerator.is_main_process)
    global_step = 0
    start_epoch = 0
    resume_micro_step_offset = 0

    # ---- Live resume support (full state: model + optim + scheduler + RNG) ----
    # Two-tier ckpts (mirrors hymot-unified-vl/train_unified.py):
    #   live-{step}/        — full state, rolling `live_keep_count`, atomic
    #                          write, frequency `resume_from_checkpoint_step`.
    #                          Only emitted when `resume_from_checkpoint=True`.
    #                          Used here for auto-resume.
    #   checkpoint-{step}/  — weights only (model + processor +
    #                          training_state.json), frequency `save_steps`,
    #                          permanent retention. For eval / downstream use.
    #
    # The IterableDataset stream is deterministic given (epoch, seed), and
    # LengthBalancedShuffleBuffer + IDS are also deterministic — so after
    # `set_epoch(epoch)` we re-enter the same batch order as the original run,
    # and skipping the first `resume_micro_step_offset` batches lands us on
    # the very next batch the previous run would have seen.
    live_ckpt_path = None
    if CONFIG["resume_from_checkpoint"]:
        live_ckpt_path = find_latest_live_ckpt(CONFIG["output_dir"])
        if live_ckpt_path is not None:
            logger.info(f"[auto-resume] found live ckpt: {live_ckpt_path}")
        else:
            logger.info("[auto-resume] enabled, no live-* dir found; fresh start")

    if live_ckpt_path is not None:
        logger.info(f"Resuming optimizer/scheduler/RNG from {live_ckpt_path}")
        accelerator.load_state(live_ckpt_path)
        ts_path = os.path.join(live_ckpt_path, "training_state.json")
        if os.path.exists(ts_path):
            try:
                with open(ts_path) as f:
                    _state = json.load(f)
                global_step = int(_state.get("global_step", 0))
                saved_max = _state.get("max_train_steps")
                if saved_max is not None and saved_max != max_train_steps:
                    logger.warning(
                        f"max_train_steps changed across runs: saved={saved_max} "
                        f"vs current={max_train_steps}. LR trajectory will NOT "
                        f"match the original run — likely because train data, "
                        f"epoch count, or per_device_train_batch_size changed."
                    )
                saved_ratios = _state.get("mix_ratios")
                if saved_ratios is not None and saved_ratios != mix_ratios:
                    logger.warning(
                        f"mix_ratios changed across runs: saved={saved_ratios} vs "
                        f"current={mix_ratios}. The resumed data stream / ratio will "
                        f"differ from the original run; resume is best-effort."
                    )
            except Exception as e:
                logger.warning(f"Failed to read {ts_path}: {e}")
        else:
            logger.warning(
                f"  training_state.json missing in {live_ckpt_path}; "
                f"optimizer/scheduler restored but global_step reset to 0"
            )
        progress.update(global_step)
        # Recompute epoch / micro_step from global_step. Don't trust the
        # saved epoch — recomputing keeps the math right even if the
        # previous run crashed exactly on an epoch boundary.
        total_micro = global_step * CONFIG["gradient_accumulation_steps"]
        steps_per_epoch_micro = num_update_steps_per_epoch * CONFIG["gradient_accumulation_steps"]
        start_epoch = total_micro // steps_per_epoch_micro
        resume_micro_step_offset = total_micro % steps_per_epoch_micro
        try:
            _cur_lr = lr_scheduler.get_last_lr()[0]
        except Exception:
            _cur_lr = float("nan")
        logger.info(
            f"[auto-resume] global_step={global_step}, "
            f"starting_epoch={start_epoch}, "
            f"micro_skip_in_epoch={resume_micro_step_offset}/{steps_per_epoch_micro}, "
            f"lr={_cur_lr:.3e}"
        )
        if start_epoch >= CONFIG["num_train_epochs"]:
            logger.info("Resume point is at/after final epoch — exiting.")
            return

        # ---- Fast-skip via dataset.set_resume_skip_per_worker -------------
        # Naive resume calls `for ... in train_dataloader: if step < N: continue`
        # which still makes the dataloader read+decode+tokenize+collate every
        # skipped batch (just to throw it away). On 6000 step at bs=16 W=200
        # that's 19.2M sample reads (per rank) which takes 4-10 hours.
        #
        # ParquetT2IDataset has a fast-skip mode: rows that fall fully inside
        # already-skipped files are skipped WITHOUT opening the parquet file,
        # decoding, or running VAEImageTransform. This makes resume seconds-
        # to-minutes instead of hours.
        #
        # We compute per-worker skip rows from the resume offset, align it
        # to LengthBalancedShuffleBuffer chunk size (= bs × W) so chunk
        # boundaries match the original run's, and let the dataloader handle
        # any non-aligned remainder via the regular `continue` path.
        if has_iterable and resume_micro_step_offset > 0 and len(datasets) > 0:
            pool_size = CONFIG["per_device_train_batch_size"] * accelerator.num_processes
            nw = max(1, CONFIG["num_workers"])
            if mix_ratios is not None:
                # SELF-SHARDED mix: each rank consumes only B samples per step (no
                # IDS B*W window), so the rows THIS rank consumed = offset * B.
                rows_per_rank = resume_micro_step_offset * CONFIG["per_device_train_batch_size"]
                # Split the per-rank skip across sources by ratio and fast-skip each
                # child (parquet file-skip / streaming line-skip). BEST-EFFORT:
                # precise bit-for-bit resume across the ratio mixer + streaming
                # shuffle buffer is not attempted; we fast-skip ~the right number of
                # rows per source (so resume costs seconds, not hours) and set the
                # dataloader-level offset to 0. The optimizer/scheduler/RNG are
                # exactly restored by load_state above.
                p = sum(mix_ratios)
                handled = []
                for i, ds in enumerate(datasets):
                    rows_i_per_worker = (rows_per_rank * mix_ratios[i] // p) // nw
                    if rows_i_per_worker > 0 and hasattr(ds, "set_resume_skip_per_worker"):
                        ds.set_resume_skip_per_worker(rows_i_per_worker)
                        handled.append((train_paths[i], rows_i_per_worker))
                resume_micro_step_offset = 0  # absorbed by per-source fast-skip
                if handled:
                    logger.info(
                        f"[auto-resume] mix fast-skip by ratio {mix_ratios} "
                        f"(~{rows_per_rank:,} rows/rank): "
                        + ", ".join(f"{os.path.basename(str(pth))}={n:,}/worker" for pth, n in handled)
                        + " (best-effort; dataloader offset reset to 0)"
                    )
            else:
                # Prepared iterable path (IDS): each rank's dataset is iterated
                # B*W per batch, so rows consumed = offset * (B*W) = offset*pool_size.
                rows_per_rank = resume_micro_step_offset * pool_size
                rows_per_worker = rows_per_rank // nw
                # LBSB chunk_size == bs × W == pool_size, so aligning per-worker
                # skip to a multiple of pool_size keeps each worker's LBSB chunks
                # identical to the original run's.
                aligned = (rows_per_worker // pool_size) * pool_size
                if aligned > 0 and hasattr(datasets[0], "set_resume_skip_per_worker"):
                    datasets[0].set_resume_skip_per_worker(aligned)
                    # Reduce dataloader-level skip to whatever remainder is left
                    # after dataset-level fast-skip.
                    rows_handled = aligned * CONFIG["num_workers"]
                    rows_remaining = rows_per_rank - rows_handled
                    resume_micro_step_offset = rows_remaining // pool_size
                    logger.info(
                        f"[auto-resume] fast-skip {aligned:,} pre-shard rows/worker "
                        f"on {datasets[0].data_dir} "
                        f"(saved ~{rows_handled:,} row reads/rank); "
                        f"remaining {resume_micro_step_offset} batches via dataloader yield"
                    )

    # ----- Phase timing accumulators -----
    # Reset every `logging_steps` and printed as a single 'phases:' line.
    # Each phase brackets a torch.cuda.synchronize() on entry/exit so the
    # numbers reflect actual GPU-completed time, not just CPU-side dispatch.
    # OVERHEAD: 5 cuda.synchronize() per step costs ~50-100 ms when the
    # device is busy (each sync stalls the CPU until the prior kernel
    # completes, breaking PyTorch's host-device pipelining). For prod
    # training keep `logging.enable_phase_timing: false`; for perf tuning
    # / debugging, flip it on in the yaml.
    _phase_timing_on = bool(CONFIG.get("enable_phase_timing", False))
    phase_totals: Dict[str, float] = defaultdict(float)
    phase_counts: Dict[str, int] = defaultdict(int)

    def sync_clock() -> float:
        if _phase_timing_on and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def phase_record(name: str, dt: float) -> None:
        if not _phase_timing_on:
            return
        phase_totals[name] += dt
        phase_counts[name] += 1

    def phase_summary_line() -> str:
        # ordered phases (key = display name → internal key)
        order = [
            ("data_wait", "data_wait"),
            ("vae_encode", "vae_encode"),
            ("forward",   "forward"),
            ("backward",  "backward"),
            ("opt_step",  "opt_step"),
        ]
        parts = []
        total_ms = 0.0
        for label, key in order:
            n = phase_counts[key]
            if n == 0:
                continue
            avg_ms = phase_totals[key] / n * 1000.0
            total_ms += avg_ms
            parts.append(f"{label}={avg_ms:.0f}ms")
        parts.append(f"sum≈{total_ms:.0f}ms")
        return " | ".join(parts)

    def phase_reset() -> None:
        phase_totals.clear()
        phase_counts.clear()

    for epoch in range(start_epoch, CONFIG["num_train_epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        # Also propagate epoch to IterableDataset(s) (file+row shuffle seeds).
        # `datasets` holds the raw children (parquet / streaming jsonl) so they
        # are seeded directly — this also covers the ChainDataset path, which
        # does not forward set_epoch.
        for ds in (datasets if has_iterable else [train_dataset]):
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)
        # Mix path: also seed the top wrapper (LBSB -> RatioMix -> children). It
        # forwards to the children already seeded above (idempotent) but keeps the
        # wrapper/mixer epoch state correct.
        if has_iterable and train_dataset is not None and hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        model.train()

        data_wait_start = sync_clock()
        for micro_step, batch in enumerate(train_dataloader):
            # ---- Resume: precise micro-step skip within the resumed epoch ----
            # Iterating the dataloader still produces the same deterministic
            # batch order (set_epoch(epoch) above seeded ParquetT2IDataset's
            # shuffle, and LengthBalancedShuffleBuffer + IDS are deterministic
            # over that stream), so consuming + dropping the first N batches
            # advances the iterator past where the previous run stopped.
            # `continue` re-enters the loop and pulls the next batch.
            if epoch == start_epoch and micro_step < resume_micro_step_offset:
                continue

            # Map path: the dataloader is unprepared (data already per-rank
            # sharded), so the collated batch is on CPU — place it on the compute
            # device here. No-op on the iterable path (prepared loader already did).
            batch = _move_to_device(batch, accelerator.device)

            # ---- EPOCH-EDGE PROGRESS DIAGNOSTIC (v2: pin the start-of-epoch hang) --
            # Fix #1 (LengthBucketedSampler even-batch truncation) removed the
            # epoch-BOUNDARY batch-count desync, but a hang remains a few steps
            # INTO the new epoch. To pin it we log EVERY step in the epoch-edge
            # windows (first 20 + last 20 micro-steps) for EVERY rank, to a
            # PER-PROCESS file (pid in the name) so successive auto-restarts never
            # overwrite each other's snapshot. At a hang the frozen files reveal
            # which rank holds the LOWEST step (a straggler) vs all-stuck-same-step
            # (a true collective deadlock). Wrapped so it can NEVER take down training.
            try:
                _cu = batch.get("cu_seqlens")
                _n_samp = (int(_cu.numel()) - 1) if _cu is not None else -1
                _ii = batch.get("input_ids")
                _T = int(_ii.shape[1]) if _ii is not None else -1
                _steps_ep = num_update_steps_per_epoch * CONFIG["gradient_accumulation_steps"]
                _near_end = micro_step >= (_steps_ep - 20)
                _near_start = micro_step <= 20
                if _near_start or _near_end:
                    _diag_dir = os.path.join(CONFIG["output_dir"], "batch_diag2")
                    os.makedirs(_diag_dir, exist_ok=True)
                    _rank = accelerator.process_index
                    _tag = "epoch_start" if _near_start else "epoch_tail"
                    with open(os.path.join(_diag_dir, f"rank{_rank}_pid{os.getpid()}.log"), "a") as _f:
                        _f.write(
                            f"{time.strftime('%F %T')} step={global_step} epoch={epoch} "
                            f"micro={micro_step}/{_steps_ep} rank={_rank} n_samples={_n_samp} "
                            f"packed_tokens={_T} {_tag} FETCHED\n"
                        )
            except Exception:
                pass

            t_data_done = sync_clock()
            phase_record("data_wait", t_data_done - data_wait_start)

            # ---- Cross-rank packed length (straggler diagnostic) -------------
            # Each rank's packed (1, T) seq length T determines forward/backward
            # compute time. If T varies a lot across ranks, the slowest rank
            # dictates each collective sync, inflating the "backward" phase
            # measurement on every other rank with idle wait time.
            # Gated by the same flag as phase_timing so prod runs are quiet.
            gathered_lens = None
            if _phase_timing_on:
                packed_len = batch["input_ids"].shape[1]
                if accelerator.num_processes > 1:
                    _len_tensor = torch.tensor(
                        [packed_len], device=accelerator.device, dtype=torch.long
                    )
                    gathered_lens = accelerator.gather(_len_tensor)
                else:
                    gathered_lens = torch.tensor([packed_len], dtype=torch.long)

            with accelerator.accumulate(model):
                # ---- VAE encode ----
                t_a = sync_clock()
                padded_latent = None
                input_padded_latent = None
                if "padded_images" in batch:
                    padded_latent = encode_images_via_vae(
                        vae, batch["padded_images"], accelerator.device, torch.bfloat16,
                    )
                if "input_padded_images" in batch:
                    input_padded_latent = encode_images_via_vae(
                        vae, batch["input_padded_images"], accelerator.device, torch.bfloat16,
                    )
                t_b = sync_clock()
                phase_record("vae_encode", t_b - t_a)

                forward_kwargs = dict(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    position_ids=batch["position_ids"],
                    labels=batch.get("labels"),
                    shift_labels=batch.get("shift_labels"),
                    cu_seqlens=batch["cu_seqlens"],
                    sample_ids=batch["sample_ids"],
                    modality_mask=batch["modality_mask"],
                    input_image_mask=batch["input_image_mask"],
                    use_flow_matching=bool(batch.get("use_flow_matching", False)),
                    padded_latent=padded_latent,
                    patchified_vae_latent_shapes=batch.get("patchified_vae_latent_shapes"),
                    vae_latent_indexes=batch.get("vae_latent_indexes"),
                    vae_latent_positions=batch.get("vae_latent_positions"),
                    flow_positions=batch.get("flow_positions"),
                    g_seqlens=batch.get("g_seqlens"),
                    input_padded_latent=input_padded_latent,
                    input_vae_latent_shapes=batch.get("input_vae_latent_shapes"),
                    input_vae_latent_indexes=batch.get("input_vae_latent_indexes"),
                    input_vae_latent_positions=batch.get("input_vae_latent_positions"),
                    pixel_values=batch.get("pixel_values"),
                    image_grid_thw=batch.get("image_grid_thw"),
                    pixel_values_videos=batch.get("pixel_values_videos"),
                    video_grid_thw=batch.get("video_grid_thw"),
                )

                # ---- Forward ----
                t_a = sync_clock()
                outputs = model(**forward_kwargs)
                loss = outputs.loss
                t_b = sync_clock()
                phase_record("forward", t_b - t_a)

                # ---- Backward (includes ZeRO-2 reduce-scatter all-reduce) ----
                t_a = sync_clock()
                accelerator.backward(loss)
                t_b = sync_clock()
                phase_record("backward", t_b - t_a)

                # ---- Optimizer step (grad clip + Adam + LR + zero_grad) ----
                t_a = sync_clock()
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                t_b = sync_clock()
                phase_record("opt_step", t_b - t_a)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)

                if global_step % CONFIG["logging_steps"] == 0:
                    lr_now = optimizer.param_groups[0]["lr"]
                    # NOTE: text_loss / image_loss are RANK-LOCAL — they reflect
                    # only this rank's tokens. We tried doing an all_reduce here
                    # to get a global average, but the extra NCCL workspace
                    # alloc OOMs at ~90% memory pressure. Gradient updates are
                    # still correctly averaged by ZeRO/DDP — only the printed
                    # diagnostic loss is per-rank.
                    txt_loss = (
                        (outputs.text_loss_sum / outputs.text_loss_count.float().clamp(min=1)).item()
                        if outputs.text_loss_sum is not None else float("nan")
                    )
                    img_loss = (
                        (outputs.image_loss_sum / outputs.image_loss_count.float().clamp(min=1)).item()
                        if outputs.image_loss_sum is not None else float("nan")
                    )
                    progress.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "text": f"{txt_loss:.4f}",
                        "image": f"{img_loss:.4f}",
                        "lr": f"{lr_now:.2e}",
                    })
                    pct = 100.0 * global_step / max_train_steps if max_train_steps > 0 else 0.0
                    logger.info(
                        f"Epoch {epoch+1}/{CONFIG['num_train_epochs']} | "
                        f"Step {global_step}/{max_train_steps} ({pct:.2f}%) | "
                        f"loss={loss.item():.4f} text={txt_loss:.4f} image={img_loss:.4f} "
                        f"lr={lr_now:.2e}"
                    )
                    # Phase breakdown (averaged across the last `logging_steps` steps)
                    if _phase_timing_on:
                        logger.info(f"  phases (avg over {CONFIG['logging_steps']} steps): {phase_summary_line()}")
                        phase_reset()
                    # Packed-length cross-rank stats from THIS step (instantaneous,
                    # not averaged) — useful for telling if backward jitter is
                    # straggler-driven. Same gate as phase timing.
                    if _phase_timing_on and gathered_lens is not None and gathered_lens.numel() > 1:
                        _lens_f = gathered_lens.float()
                        _lmin = int(gathered_lens.min().item())
                        _lmax = int(gathered_lens.max().item())
                        _lmean = float(_lens_f.mean().item())
                        _lstd = float(_lens_f.std().item())
                        _ratio = _lmax / max(_lmin, 1)
                        logger.info(
                            f"  packed_len cross-rank (this step): "
                            f"min={_lmin} max={_lmax} mean={_lmean:.0f} "
                            f"std={_lstd:.0f} max/min={_ratio:.2f}"
                        )
                    # Tensorboard
                    if CONFIG.get("use_tensorboard", True) and is_main:
                        tb = {
                            "train/loss": loss.item(),
                            "train/text_loss": txt_loss,
                            "train/image_loss": img_loss,
                            "train/learning_rate": lr_now,
                            "train/epoch": epoch,
                        }
                        accelerator.log(tb, step=global_step)

                # ---- Live ckpt: full resume state, rolling, atomic ----
                # Only emitted when resume_from_checkpoint=True. Frequency
                # is `resume_from_checkpoint_step` (independent of save_steps).
                # `live-{step}.tmp/` -> os.replace -> `live-{step}/` makes the
                # commit atomic; a crash mid-save leaves only .tmp (skipped
                # by find_latest_live_ckpt) and the previous live-{step}/
                # stays intact.
                if CONFIG["resume_from_checkpoint"] and (
                    global_step % CONFIG["resume_from_checkpoint_step"] == 0
                ):
                    live_final = os.path.join(CONFIG["output_dir"], f"live-{global_step}")
                    live_tmp = live_final + ".tmp"
                    if accelerator.is_main_process:
                        if os.path.isdir(live_tmp):
                            shutil.rmtree(live_tmp, ignore_errors=True)
                        os.makedirs(live_tmp, exist_ok=True)
                    # Release the PyTorch allocator's CACHED VRAM before the
                    # save-time barrier. NCCL allocates its collective/comm
                    # buffers via raw cudaMalloc OUTSIDE the caching pool, so at
                    # the step-N memory peak the barrier's ~2.8 MB alloc fails
                    # ("Failed to CUDA calloc N bytes" -> ncclUnhandledCudaError),
                    # which is the prod live-save crash loop: the job dies at the
                    # very first live ckpt (step == resume_from_checkpoint_step),
                    # leaves an empty live-*.tmp, and restarts from 0 forever.
                    # Freeing cache on EVERY rank gives NCCL room to allocate.
                    import gc as _gc
                    _gc.collect()
                    torch.cuda.empty_cache()
                    accelerator.wait_for_everyone()
                    # save_state() is collective — ALL ranks must call it.
                    # Under DeepSpeed ZeRO-2 this writes sharded optimizer
                    # state + LR scheduler + RNG + model. Write directly into
                    # `live_tmp` (not into a nested `accel_state/` subdir) to
                    # match hymot-unified-vl/train_unified.py:667 — nested
                    # subdir requires every rank to mkdir the subdir on shared
                    # storage which can deadlock NFS metadata locks at 200
                    # ranks. Single shared parent dir avoids that.
                    accelerator.save_state(live_tmp)
                    if accelerator.is_main_process:
                        # Dump unwrapped HF-format weights inside the live dir
                        # too, so each live ckpt is also a runnable model
                        # checkpoint without needing a separate weights-only
                        # save at the same step.
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.save_pretrained(live_tmp, safe_serialization=True)
                        processor.save_pretrained(live_tmp)
                        with open(os.path.join(live_tmp, "training_state.json"), "w") as f:
                            json.dump({
                                "global_step": global_step,
                                "epoch": epoch,
                                "max_train_steps": max_train_steps,
                                "num_processes": accelerator.num_processes,
                                "per_device_train_batch_size": CONFIG["per_device_train_batch_size"],
                                "gradient_accumulation_steps": CONFIG["gradient_accumulation_steps"],
                                "mix_ratios": mix_ratios,
                            }, f, indent=2)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        if os.path.isdir(live_final):
                            shutil.rmtree(live_final, ignore_errors=True)
                        os.replace(live_tmp, live_final)
                        logger.info(f"Live ckpt saved: {live_final}")
                        prune_live_ckpts(CONFIG["output_dir"], CONFIG["live_keep_count"], logger)
                    accelerator.wait_for_everyone()

                # ---- Weights-only ckpt: for eval, permanent retention ----
                # Always emitted at save_steps cadence (independent of live).
                # Contains model + processor + training_state.json only — NO
                # optimizer / scheduler / RNG. Smaller, faster to load for
                # downstream inference.
                if global_step % CONFIG["save_steps"] == 0:
                    ck_dir = os.path.join(CONFIG["output_dir"], f"checkpoint-{global_step}")
                    if accelerator.is_main_process:
                        os.makedirs(ck_dir, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.save_pretrained(ck_dir, safe_serialization=True)
                        processor.save_pretrained(ck_dir)
                        with open(os.path.join(ck_dir, "training_state.json"), "w") as f:
                            json.dump({
                                "global_step": global_step,
                                "epoch": epoch,
                                "max_train_steps": max_train_steps,
                            }, f, indent=2)
                        logger.info(f"Weights-only ckpt saved: {ck_dir}")
                    accelerator.wait_for_everyone()

                if global_step >= max_train_steps:
                    break

            # Reset data_wait clock for the next iteration so the next
            # "data_wait" sample measures only the wait *between* steps,
            # not cumulative time since the very first batch arrived.
            data_wait_start = sync_clock()
        if global_step >= max_train_steps:
            break

    # ---- Final saves ---------------------------------------------------------
    # Always emit a weights-only `checkpoint-final/` for eval.
    # When `resume_from_checkpoint=True`, also emit a final live-{step}/ so
    # the next run (e.g. SFT off this base) can pick up cleanly.
    ck_final = os.path.join(CONFIG["output_dir"], "checkpoint-final")
    if accelerator.is_main_process:
        os.makedirs(ck_final, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(ck_final, safe_serialization=True)
        processor.save_pretrained(ck_final)
        with open(os.path.join(ck_final, "training_state.json"), "w") as f:
            json.dump({
                "global_step": global_step,
                "epoch": CONFIG["num_train_epochs"] - 1,
                "max_train_steps": max_train_steps,
            }, f, indent=2)
        logger.info(f"Final weights-only ckpt → {ck_final}")
    accelerator.wait_for_everyone()

    if CONFIG["resume_from_checkpoint"]:
        live_final = os.path.join(CONFIG["output_dir"], f"live-{global_step}")
        live_tmp = live_final + ".tmp"
        if accelerator.is_main_process:
            if os.path.isdir(live_tmp):
                shutil.rmtree(live_tmp, ignore_errors=True)
            os.makedirs(live_tmp, exist_ok=True)
        accelerator.wait_for_everyone()
        accelerator.save_state(live_tmp)
        if accelerator.is_main_process:
            accelerator.unwrap_model(model).save_pretrained(live_tmp, safe_serialization=True)
            processor.save_pretrained(live_tmp)
            with open(os.path.join(live_tmp, "training_state.json"), "w") as f:
                json.dump({
                    "global_step": global_step,
                    "epoch": CONFIG["num_train_epochs"] - 1,
                    "max_train_steps": max_train_steps,
                    "num_processes": accelerator.num_processes,
                    "per_device_train_batch_size": CONFIG["per_device_train_batch_size"],
                    "gradient_accumulation_steps": CONFIG["gradient_accumulation_steps"],
                    "mix_ratios": mix_ratios,
                }, f, indent=2)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if os.path.isdir(live_final):
                shutil.rmtree(live_final, ignore_errors=True)
            os.replace(live_tmp, live_final)
            logger.info(f"Final live ckpt → {live_final}")
            prune_live_ckpts(CONFIG["output_dir"], CONFIG["live_keep_count"], logger)
        accelerator.wait_for_everyone()

    if CONFIG.get("use_tensorboard", True):
        accelerator.end_training()


if __name__ == "__main__":
    main()
