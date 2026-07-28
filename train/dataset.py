"""
Unified Dataset for All Training Tasks.

Based on train_dataset.py and data_collator.py, extended for discrete diffusion.

Supports:
1. Pure text input -> Pure text output (text-only)
2. Multimodal input (image/video + text) -> Pure text output (understanding)
3. Pure text input -> Pure image output (T2I generation)
4. Multimodal input -> Pure image output (I2I generation)
5. Pure text input -> Interleaved multimodal output
6. Multimodal input -> Interleaved multimodal output

Data format (JSONL with messages, same as debug_data.jsonl):
{
    "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "..."},
            {"type": "image", "image": "/path/to/image.jpg"}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "..."},
            {"type": "generated_image", "image_tokens": "/path/to/tokens.pkl"}
        ]}
    ]
}
"""

import json
import logging
import io
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from torchvision import transforms as T
from torchvision.transforms import functional as TF, InterpolationMode
from PIL import Image


import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Individual random generator
_mask_rng = random.Random(time.time_ns())

# Special token IDs for discrete diffusion
# Based on original HunYuanVL vocab_size=120818
# New tokens are appended: 120818 ~ 129013
IMAGE_START_TOKEN_ID = 120119  # <Image> — upstream HunYuanVLMoTConfig.image_start_token_id
IMAGE_END_TOKEN_ID = 120120  # </Image> — upstream HunYuanVLMoTConfig.image_end_token_id
# Multi-frame (world-model "video") generation wrapper. We use the upstream
# HunYuanVLMoTConfig's EXISTING video token ids (video_start_token_id=120122,
# video_end_token_id=120123) — they are < vocab_size (120818), already in the
# embed_tokens table, and used nowhere else in the model/dataset (vestigial config
# defaults). So: NO resize_token_embeddings, NO tokenizer/config edits, and
# inference can read them straight off cfg.video_start_token_id. A multi-frame
# sample emits ONE <Video> ... [N frames of bare latent spans] ... </Video> instead
# of N×<Image> blocks, so the AR semantics of <Image> ("</Image> -> text -> <Image>")
# stay clean for interleave rollout (N×<Image> would teach "</Image> -> <Image>").
VIDEO_START_TOKEN_ID = 120122  # <Video> — multi-frame gen start (= cfg.video_start_token_id; CE target, like <Image>)
VIDEO_END_TOKEN_ID = 120123    # </Video> — multi-frame gen end   (= cfg.video_end_token_id;   label -100, like </Image>)
LATENT_PLACEHOLDER_ID = 120690  # upstream HunYuanVLMoTConfig.latent_token_id (was 0)
INPUT_LATENT_PLACEHOLDER_ID = 120691  # custom id for input-image VAE latent (TI2I); must be in vocab

# Assistant token (from original model)
ASSISTANT_TOKEN_ID = 120007
EOS_TOKEN_ID = 120020  # End of sequence token

# Image placeholder tokens (used by processor for ViT feature replacement)
IMAGE_PLACEHOLDER_TOKEN_ID = 120687  # <|placeholder|> → image_token
IMAGE_PLACEHOLDER_NEWLINE_ID = 120689  # <｜hy_place▁holder▁no▁671｜> row separator within ViT grid
IMAGE_PLACEHOLDER_IDS = {IMAGE_PLACEHOLDER_TOKEN_ID, IMAGE_PLACEHOLDER_NEWLINE_ID}


@dataclass
class TaskConfig:
    """Configuration for training tasks."""

    # Enable/disable specific task types
    enable_text_to_text: bool = True
    enable_multimodal_to_text: bool = True
    enable_text_to_image: bool = True
    enable_multimodal_to_image: bool = True
    enable_interleaved_generation: bool = True

    # Input image VQ token feature

    # Input image VAE latent injection (for TI2I spatial conditioning)
    enable_input_vae_latent: bool = False  # Inject input image VAE latent into sequence

    #VAE+flow matching
    vae_image_downsample: int = 16
    max_latent_size: int = 32
    max_image_size: int = 512
    min_image_size: int = 256
    image_stride: int = 16
    text_cond_dropout_prob: float = 0.1
    # Max sequence length
    max_length: int = 4096

    # Debug mode

    # Interleaved joint text+image generation (v2): when True, the <Image> start
    # token of each generated-image block is a CE target, so the model learns to
    # EMIT <Image> after the assistant text — the train/test-consistent trigger
    # that lets autoregressive inference switch from text decoding to flow-matching
    # image generation. Default False keeps the T2I/single-image path unchanged
    # (there <Image> is hard-appended at inference, never predicted).
    train_image_start_token: bool = False

    # Multi-frame "video" generation: when True, the <Video> start token wrapping a
    # multi-frame (>=2 generated frames) block is a CE target, so the model learns
    # to EMIT <Video> after the task instruction (the text->multiframe trigger),
    # analogous to train_image_start_token for the single-image <Image> switch.
    train_video_start_token: bool = True

    # Image path remap: {"old_prefix": "new_prefix", ...}
    # Applied to all image paths (image, raw_image, image_tokens) at load time
    image_path_remap: Optional[Dict[str, str]] = None

    # Prior-frame VAE round-trip augmentation (image-side exposure-bias fix).
    # When prob > 0, each PRIOR-STEP context frame (NOT obs frames, NOT the FM
    # target) is replaced — with this per-call probability — by a pre-computed
    # VAE-round-tripped (decode(encode(GT))) mirror image, so the SigLIP input
    # distribution at train matches the lossy decoded frames the model feeds back
    # at autoregressive inference. The mirror path is derived from the original by
    # the src_root->dst_root prefix swap (+ .png extension); the swap only happens
    # when the mirror file exists. 0.0 = exact original behavior (no-op). The
    # obs/prior boundary is taken from metadata: num_obs = num_context_frames -
    # (step_index - 1). See train/data_prep/build_vae_roundtrip.py.
    prior_frame_vae_roundtrip_prob: float = 0.0
    prior_frame_vae_roundtrip_src_root: str = ""
    prior_frame_vae_roundtrip_dst_root: str = ""


# Task-instruction prompts appended to the user turn so each generation MODE is
# explicitly cued, and the SAME text is used at train and inference (parity is
# load-bearing — interleave_inference/multiframe_inference build the prompt the
# same way). Single source of truth, imported by the inference scripts.
TASK_INSTRUCTION_SINGLE_FRAME = "Generate future image of the goal"
TASK_INSTRUCTION_MULTI_FRAME = "Generate future video of the goal"
TASK_INSTRUCTION_INTERLEAVE = "Generate interleave goal planning"


def build_task_instruction(metadata: Optional[Dict], n_gen: int) -> Optional[str]:
    """Return the user-turn instruction for a GENERATION sample, or None.

    Routing (structural + metadata; no data reconversion needed):
      * n_gen >= 2                         -> multi-frame  ("future video")
      * n_gen == 1 and metadata.step_index -> interleave   ("interleave goal planning")
      * n_gen == 1 otherwise               -> single-frame ("future image")
      * n_gen == 0                         -> None (understanding / not a gen sample)
    T2I parquet does NOT call this (foundation: no instruction); ParquetT2IDataset
    keeps its bare-caption path.
    """
    if n_gen <= 0:
        return None
    if n_gen >= 2:
        return TASK_INSTRUCTION_MULTI_FRAME
    meta = metadata or {}
    if meta.get("step_index") is not None:
        return TASK_INSTRUCTION_INTERLEAVE
    return TASK_INSTRUCTION_SINGLE_FRAME

# like Bagel
class VAEImageTransform:
    def __init__(self, max_size = 512,min_size = 256, stride = 16):
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride

    def _make_divisible(self, val):
        return max(self.stride, round(val / self.stride) * self.stride)
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        w,h = img.size
        scale= min(self.max_size / max(w,h),1.0)
        scale=max(scale, self.min_size / min(w,h))
        new_w = self._make_divisible(round(w * scale))
        new_h = self._make_divisible(round(h * scale))
        # Clamp to max_size to handle extreme aspect ratios where min_size
        # constraint overrides max_size constraint (e.g. w=1000,h=50 would
        # produce new_w=5120 which causes position ID out-of-bounds).
        max_div = self._make_divisible(self.max_size)
        if new_w > max_div or new_h > max_div:
            # Re-scale to fit within max_size while preserving aspect ratio
            shrink = min(max_div / new_w, max_div / new_h)
            new_w = self._make_divisible(round(new_w * shrink))
            new_h = self._make_divisible(round(new_h * shrink))
        img = TF.resize(img, (new_h, new_w), InterpolationMode.BICUBIC, antialias=True)
        tensor = TF.to_tensor(img) # [0, 1]
        tensor = T.Normalize([0.5]*3, [0.5]*3)(tensor) # [-1, 1]
        return tensor

def get_flattened_position_ids(img_h, img_w, patch_size, max_num_patches_per_side):
    """calculate 2D latent position IDs, for PositionEmbedding lookup."""
    num_h = img_h // patch_size
    num_w = img_w // patch_size
    coords_h = torch.arange(0, num_h)
    coords_w = torch.arange(0, num_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids



def build_shift_labels(input_ids: List[int], labels: List[int]) -> List[int]:
    """
    Build custom shift_labels that handles image blocks correctly.

    Logic:
    - Traverse from start, by default take the next position's label as current target
    - If next input_id is <Image>, switch to taking current position's label (no shift)
    - If current input_id is </Image>, resume taking next position's label
    - Last position gets -100

    Args:
        input_ids: The input token IDs
        labels: The labels (VQ token labels inside image block, text labels outside)

    Returns:
        shift_labels: Labels for loss computation
    """
    n = len(labels)
    shift_labels = []
    use_same_position = False  # Whether to take current position's label (inside image block)

    for i in range(n):
        if i == n - 1:
            # Last position, no next token
            shift_labels.append(-100)
        elif not use_same_position:
            # Normal mode: take next position's label
            if input_ids[i + 1] == IMAGE_START_TOKEN_ID:
                # Next is <Image>, switch to same-position mode
                shift_labels.append(IMAGE_START_TOKEN_ID)
                use_same_position = True
            else:
                shift_labels.append(labels[i + 1])
        else:
            # Inside image block: take current position's label
            if input_ids[i] == IMAGE_END_TOKEN_ID:
                # Current is </Image>, resume next-position mode
                shift_labels.append(labels[i + 1])
                use_same_position = False
            else:
                shift_labels.append(labels[i])

    return shift_labels


class UnifiedDataset(Dataset):
    """
    Unified dataset for all training tasks.
    Based on the working train_dataset.py implementation.

    Key improvements:
    - Direct VQ token concatenation (no decode-encode)
    - Multi-turn dialogue support
    - Proper text loss for interleaved generation
    - VQ token offset handling
    - CFG dropout preserves input images
    """

    def __init__(
        self,
        data_path: str,
        processor,
        config: Optional[TaskConfig] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Args:
            data_path: Path to JSONL data file
            processor: HunYuanVL processor for handling multimodal inputs
            config: Task configuration (uses defaults if None)
            rank, world_size: per-rank LOAD-TIME sharding. When world_size>1 each
                rank keeps only every world_size-th record (offset=rank) of this
                file, so `self.data` holds ~1/world_size of the corpus instead of
                the FULL corpus on every rank. This is the fix for the host-RAM
                OOM at scale: the map-style dataset is otherwise fully resident on
                all 8 ranks of a node (node RAM ∝ total_corpus × ranks_per_node).
                The split is per-file stride over the non-blank record index, so it
                is disjoint, deterministic, balanced (±1 record/file across ranks),
                and spans every robot/task group in the file. Only the kept lines
                are json.loads'd, so the parse-time transient is divided too.
                Default (0, 1) = no sharding (single-process / unchanged behavior).
        """
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.config = config or TaskConfig()
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))

        # Load data — supports both JSONL (one JSON object per line) and
        # JSON array files ([{...}, {...}, ...])
        self.data = []
        with open(data_path, "r") as f:
            # Peek at first non-whitespace character to detect format
            first_char = ''
            while True:
                ch = f.read(1)
                if not ch:
                    break
                if not ch.isspace():
                    first_char = ch
                    break
            f.seek(0)
            if first_char == '[':
                raw_items = json.load(f)
                if self.world_size > 1:
                    raw_items = raw_items[self.rank::self.world_size]
            else:
                # Per-rank stride over the non-blank record index. Parse ONLY the
                # lines this rank owns so the json.loads transient is ÷world_size.
                raw_items = []
                _rec_idx = 0
                for line in f:
                    if not line.strip():
                        continue
                    if _rec_idx % self.world_size == self.rank:
                        raw_items.append(json.loads(line))
                    _rec_idx += 1
        # Apply image path remap if configured
        if self.config.image_path_remap:
            remap_count = sum(self._remap_item(item) for item in raw_items)
            if remap_count > 0:
                logger.info(f"Remapped {remap_count} image paths using {len(self.config.image_path_remap)} rules")

        for item in raw_items:
            task_type = self._determine_task_type(item)
            if self._is_task_enabled(task_type, item):
                item["_task_type"] = task_type
                self.data.append(item)
        del raw_items

        self._filter_overlength_samples()
        self._print_statistics()

    def _remap_item(self, item: Dict) -> int:
        """Apply config.image_path_remap to ONE item IN PLACE; return the number
        of paths remapped (0 if no remap configured). Shared by the map-style
        __init__ and StreamingUnifiedDataset.__iter__ so both apply identical
        path rewriting."""
        remap = self.config.image_path_remap
        if not remap:
            return 0
        count = 0
        for msg in item.get("messages", []):
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                for key in ("image", "raw_image", "image_tokens"):
                    if key in c and c[key]:
                        for old_prefix, new_prefix in remap.items():
                            if c[key].startswith(old_prefix):
                                c[key] = new_prefix + c[key][len(old_prefix):]
                                count += 1
                                break
        return count

    def _est_seq_len(self, item: Dict) -> int:
        """Conservative OVER-estimate of one item's built sequence length (see
        _filter_overlength_samples for the rationale behind the constants).
        Shared by the map-style filter and StreamingUnifiedDataset.__iter__.

        Per-image cost is TASK-DEPENDENT and must over-estimate BOTH paths:
          * UNDERSTANDING images (no generated_image in the sample) go through the
            ViT capped by understanding_max_pixels — up to ~1024 tok/image at the
            1048576 cap. A >=8-image MMU sample (e.g. MAmmoTH-ov multi-image, up to
            12 imgs) then reaches >8k tok. If under-estimated, it slips past this
            pre-filter, the processor truncates its placeholders, and the model
            HARD-RAISES "Image feature/token count mismatch" -> whole-job restart.
          * GENERATION context frames (interleave/multiframe obs+prior) are native-
            res SigLIP ~397 tok; 420 covers them. VAE mirrors are far smaller.
        So: 1024/image for understanding samples, 420/frame for generation ones.
        """
        GEN_LATENT, MARGIN = 260, 80
        UNDERSTANDING_PER_IMG, GEN_PER_FRAME = 1024, 420
        # A sample is GENERATION iff it has any generated_image; else UNDERSTANDING.
        is_generation = any(
            isinstance(c, dict) and c.get("type") == "generated_image"
            for msg in item.get("messages", [])
            for c in (msg.get("content") or [])
            if isinstance(msg.get("content"), list)
        )
        per_img = GEN_PER_FRAME if is_generation else UNDERSTANDING_PER_IMG
        n_ctx = n_gen = text_chars = 0
        for msg in item.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")
            if isinstance(content, str):
                text_chars += len(content)
                continue
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                t = c.get("type")
                if t == "text":
                    text_chars += len(c.get("text", ""))
                elif t in ("image", "video"):
                    # Count input images REGARDLESS of role. Normally these are on
                    # user/system turns, but some converted MMU corpora (e.g.
                    # MAmmoTH-ov) contain ROLE-SWAPPED bad rows where the images sit
                    # on the assistant turn. The processor still emits pixel_values
                    # for them, so their token cost is real; counting only user/system
                    # images would estimate ~0 here, let a >=7-image row slip past the
                    # over-length pre-filter, get truncated by the collator (which
                    # can NOT trim pixel_values), and HARD-CRASH the model on a
                    # placeholder/feature mismatch. Counting all images closes that.
                    n_ctx += 1
                elif t == "generated_image":
                    n_gen += 1
        return n_ctx * per_img + n_gen * GEN_LATENT + text_chars // 3 + MARGIN

    def _filter_overlength_samples(self) -> None:
        """Drop samples whose built sequence would exceed config.max_length.

        WHY: context frames go through SigLIP at NATIVE resolution (the processor's
        Qwen2-VL image_processor, max_pixels=4M — it does NOT honor config
        max_image_size, which only governs the VAE/generation path). An 848×480
        frame is ~397 SigLIP tokens, so a deep interleave sample (many obs+prior
        frames) can reach ~6.5k tokens. If that exceeds max_length, the truncation
        in _process_generation_sample cuts image placeholder tokens from input_ids
        but does NOT trim the matching pixel_values/image_grid_thw → the model
        raises "Image feature/token count mismatch" and the whole (collective)
        job aborts. We CANNOT skip such samples at __getitem__ time (a per-rank
        skip desyncs DDP). Instead drop them here, at load, deterministically —
        every rank loads the same file with the same logic, so all ranks drop the
        exact same set and stay in lockstep.

        Estimate is a conservative OVER-estimate (per-frame 420 ≥ the observed 397
        max, text chars/3, gen-latent 260, +80 margin) so anything that would
        actually truncate is caught. With max_length=8192 this drops ~0 of the
        current corpus (deepest ~6.5k); it is a belt-and-braces guard for future
        data with more/larger frames.
        """
        max_len = getattr(self.config, "max_length", None)
        if not max_len:
            return
        kept, dropped = [], 0
        for item in self.data:
            if self._est_seq_len(item) > max_len:
                dropped += 1
            else:
                kept.append(item)
        if dropped:
            logger.warning(
                f"Dropped {dropped}/{len(self.data)} over-length samples "
                f"(est seq > max_length={max_len}); kept {len(kept)}."
            )
        self.data = kept

    def _determine_task_type(self, item: Dict) -> str:
        """Determine task type from data content."""
        messages = item.get("messages", [])
        has_generated_image = False
        has_text_output = False

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role in ["user", "system"]:
                pass

            elif role == "assistant":
                if isinstance(content, str) and content.strip():
                    has_text_output = True
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            if c.get("type") == "generated_image":
                                has_generated_image = True
                            elif c.get("type") == "text" and c.get("text", "").strip():
                                has_text_output = True

        if has_generated_image:
            return "interleaved" if has_text_output else "generation"
        return "understanding"

    def _is_task_enabled(self, task_type: str, item: Dict) -> bool:
        """Check if task type is enabled."""
        messages = item.get("messages", [])
        has_input_media = False

        for msg in messages:
            if msg.get("role") in ["user", "system"]:
                content = msg.get("content", "")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") in ["image", "video"]:
                            has_input_media = True
                            break

        if task_type == "understanding":
            return self.config.enable_multimodal_to_text if has_input_media else self.config.enable_text_to_text
        elif task_type == "generation":
            return self.config.enable_multimodal_to_image if has_input_media else self.config.enable_text_to_image
        elif task_type == "interleaved":
            return self.config.enable_interleaved_generation
        return True

    def _print_statistics(self):
        """Print dataset statistics."""
        stats = {}
        for item in self.data:
            task_type = item.get("_task_type", "unknown")
            stats[task_type] = stats.get(task_type, 0) + 1

        print(f"Loaded {len(self.data)} samples:")
        for task_type, count in stats.items():
            print(f"  {task_type}: {count}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # ---- Robust to corrupt / missing / unreadable images ----------------
        # A single bad image used to crash the WHOLE multi-rank job: the INPUT
        # (obs) frames are loaded by the processor inside _process_generation_sample
        # and were NOT individually guarded, so a PIL.UnidentifiedImageError /
        # FileNotFoundError there propagated out, exited the rank (exitcode 1) and
        # SIGTERM'd all ranks — the deterministic "stuck at step 1000 on one node"
        # restart loop (a corrupt seg004_pre_history obs frame on one rank's shard).
        # Fix: catch any per-sample failure and advance to the NEXT index, so one
        # bad file (or a briefly-flaky mount) can't bring the job down. Bounded so
        # a systemically broken mount still fails loud instead of spinning forever.
        _MAX_BAD_SAMPLE_RETRIES = 50
        n = len(self.data)
        last_err = None
        for _attempt in range(_MAX_BAD_SAMPLE_RETRIES):
            cur = (idx + _attempt) % n
            item = self.data[cur]
            task_type = item.get("_task_type", "understanding")

            # ---- POISON-SAMPLE PROBE ---------------------------------------
            # Writes the sample being processed to a per-WORKER file in TRUNCATE
            # mode: a worker frozen on a *blocking* sample (truncated file / slow
            # mount / decode bomb — which the try/except below CANNOT catch) leaves
            # its file pinned at "START idx=... <item>". Enabled when IMG_PROBE_DIR
            # is set; never throws.
            _pf = None
            _probe_dir = os.environ.get("IMG_PROBE_DIR")
            if _probe_dir:
                try:
                    os.makedirs(_probe_dir, exist_ok=True)
                    _rk = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
                    _pf = os.path.join(_probe_dir, f"rank{_rk}_pid{os.getpid()}.log")
                    with open(_pf, "w") as _f:
                        _f.write(f"{time.strftime('%F %T')} idx={cur} START {repr(item)[:1200]}\n")
                except Exception:
                    _pf = None

            try:
                if task_type == "understanding":
                    return self._process_understanding_sample(item)
                else:
                    return self._process_generation_sample(item)
            except Exception as e:
                last_err = e
                logger.warning(
                    f"[skip-bad-sample] idx={cur} id={item.get('id', '?')}: "
                    f"{type(e).__name__}: {str(e)[:200]} — advancing to next index"
                )
                continue
            finally:
                if _pf:
                    try:
                        with open(_pf, "w") as _f:
                            _f.write(f"{time.strftime('%F %T')} idx={cur} DONE\n")
                    except Exception:
                        pass

        raise RuntimeError(
            f"__getitem__ failed on {_MAX_BAD_SAMPLE_RETRIES} consecutive samples "
            f"starting at idx={idx} — likely a systemically broken data mount. "
            f"Last error: {type(last_err).__name__}: {last_err}"
        )

    def _create_multiturn_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create labels for multi-turn dialogue.

        For multi-turn conversations, we need to:
        - Mask all tokens before and including each assistant token
        - Only compute loss on assistant responses

        This handles patterns like:
        [user1][assistant1][user2][assistant2]...
        where we want to predict assistant1 and assistant2 but not user1 and user2.
        """
        labels = input_ids.clone()

        # Find all assistant token positions
        assistant_positions = (input_ids == ASSISTANT_TOKEN_ID).nonzero(as_tuple=True)[0].tolist()

        if len(assistant_positions) == 0:
            # No assistant tokens found, mask everything
            labels[:] = -100
            return labels

        # For multi-turn: mask regions between end of one assistant response
        # and the next assistant token
        #
        # Pattern: [sys/user tokens]<assistant>[response1][user tokens]<assistant>[response2]...
        # We want to mask: [sys/user tokens]<assistant> and [user tokens]<assistant>
        # We want to keep: [response1] and [response2]

        # First, mask everything up to and including the first assistant token
        first_assistant_pos = assistant_positions[0]
        labels[: first_assistant_pos + 1] = -100

        # For subsequent assistant tokens, we need to find where each assistant's
        # response ends (next assistant token or end of sequence) and mask the
        # user portion before each assistant token
        for i in range(1, len(assistant_positions)):
            curr_assistant_pos = assistant_positions[i]

            # Find where user content starts (we need to identify user tokens)
            # A simple heuristic: user content typically comes after the previous
            # assistant's response. We can look for patterns or specific tokens.
            #
            # For now, we use a conservative approach: if there's another assistant
            # token, mask from some point before it to include the assistant token
            #
            # The safest approach is to find user/system role markers, but since
            # we may not have them explicitly, we'll mask the assistant token itself
            labels[curr_assistant_pos] = -100

        return labels

    def _process_understanding_sample(self, item: Dict) -> Dict[str, Any]:
        """
        Process understanding sample with multi-turn support.
        """
        messages = item["messages"]

        # ---- Repair ROLE-SWAPPED rows (MMU-only; converter artifact) ----------
        # Some MAmmoTH-ov rows have user<->assistant labels swapped: the INPUT
        # images + question land on the `assistant` turn and the answer on `user`
        # (signature seen in data: [assistant{image,text}, user{text}]). A plain
        # `type=="image"` (INPUT image) on an assistant turn is illegal for the
        # understanding path — legitimate assistant images are always
        # `type=="generated_image"` (and those samples go through
        # _process_generation_sample, never here). Left alone, these rows either
        # train role-inverted garbage or (when many images) slip the over-length
        # filter and crash the model. Fix = swap the user/assistant role labels so
        # the images/question become the user turn and the reply the assistant
        # turn. This ONLY touches understanding samples (this method) and ONLY rows
        # that carry an input image on an assistant turn, so interleave/multiframe
        # (which legitimately put generated_image on assistant, via the other
        # method) are never affected.
        def _has_input_image(msg):
            c = msg.get("content")
            return isinstance(c, list) and any(
                isinstance(x, dict) and x.get("type") == "image" for x in c
            )
        if any(m.get("role") == "assistant" and _has_input_image(m) for m in messages):
            _swap = {"assistant": "user", "user": "assistant"}
            messages = [
                {**m, "role": _swap.get(m.get("role"), m.get("role"))}
                for m in messages
            ]
            item = {**item, "messages": messages}

        # ---- Fold any `system` turn into the first user turn -----------------
        # The HunYuanVLMoT chat template cannot handle a standalone system role:
        # with list content it does str+list ("can only concatenate str (not
        # 'list') to str"); with str content it breaks visual extraction. So a
        # system turn (e.g. ALL of open_orca / orca_agentinstruct, ~1M rows, and
        # system-prefixed ShareGPT/MAmmoTH samples) would otherwise raise and be
        # silently skipped. Prepend the system text to the first user turn's
        # content so the instruction is kept and no standalone system role remains.
        if any(m.get("role") == "system" for m in messages):
            sys_text = "\n".join(
                (m["content"] if isinstance(m.get("content"), str)
                 else " ".join(c.get("text", "") for c in (m.get("content") or [])
                               if isinstance(c, dict) and c.get("type") == "text"))
                for m in messages if m.get("role") == "system"
            ).strip()
            rest = [m for m in messages if m.get("role") != "system"]
            if sys_text:
                for idx, m in enumerate(rest):
                    if m.get("role") == "user":
                        c = m.get("content")
                        c = list(c) if isinstance(c, list) else [{"type": "text", "text": str(c)}]
                        rest[idx] = {**m, "content": [{"type": "text", "text": sys_text + "\n"}] + c}
                        break
            messages = rest if rest else messages

        # ---- Native /think rendering for reasoning (text_cot) samples ----------
        # The chat template wraps assistant CONTENT in
        #   <think>\n\n</think>\n<answer>\n{content}\n</answer>
        # so a sample whose data already carries <think>..</think><answer>..</answer>
        # gets DOUBLE-WRAPPED (nested, malformed) and the whole nest is supervised.
        # Instead, for a single-assistant-turn reasoning sample, render the PROMPT in
        # the model's native /think mode (template OPENS <think>) and supervise the
        # assistant text with its leading <think> stripped -> a clean single
        #   <think>{reasoning}</think>\n<answer>{answer}</answer>
        # that matches native inference (serve with enable_thinking=True). Non-reasoning
        # / multi-turn samples fall through to the default render below.
        _asst = [m for m in messages if m.get("role") == "assistant"]
        def _join_text(m):
            c = m.get("content")
            if isinstance(c, str):
                return c
            return "".join(x.get("text", "") for x in (c or [])
                           if isinstance(x, dict) and x.get("type") == "text")
        if len(_asst) == 1 and _join_text(_asst[0]).lstrip().startswith("<think>"):
            prompt_messages = [m for m in messages if m.get("role") != "assistant"]
            p_inputs = self.processor.apply_chat_template(
                prompt_messages, return_dict=True, tokenize=True,
                add_generation_prompt=True, enable_thinking=True,
                max_length=self.config.max_length, truncation=True,
            )
            prompt_ids = list(p_inputs["input_ids"][0])
            a_text = _join_text(_asst[0]).lstrip()[len("<think>"):]  # template already emitted <think>
            a_ids = self.tokenizer.encode(a_text, add_special_tokens=False)
            full_ids = prompt_ids + a_ids + [EOS_TOKEN_ID]
            lab = [-100] * len(prompt_ids) + a_ids + [EOS_TOKEN_ID]
            if len(full_ids) > self.config.max_length:
                full_ids = full_ids[: self.config.max_length]
                lab = lab[: self.config.max_length]
            input_ids = torch.tensor(full_ids, dtype=torch.long)
            labels = torch.tensor(lab, dtype=torch.long)
            out = {
                "input_ids": input_ids,
                "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                "labels": labels,
                "shift_labels": torch.tensor(build_shift_labels(full_ids, lab), dtype=torch.long),
                "use_discrete_diffusion": False,
                "image_block_ranges": [],
            }
            for _k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
                if _k in p_inputs:
                    out[_k] = p_inputs[_k]
            merge = getattr(getattr(self.processor, "image_processor", None), "merge_size", None) or 2
            for grid_key in ("image_grid_thw", "video_grid_thw"):
                grid = out.get(grid_key)
                if grid is None:
                    continue
                expected = int(sum(int(t) * int(h) * int(w) for t, h, w in grid)) // (merge * merge)
                actual = int((input_ids == IMAGE_PLACEHOLDER_TOKEN_ID).sum().item())
                if actual != expected:
                    raise ValueError(
                        f"[truncated-image] {grid_key}: placeholder={actual} != features={expected}; skipping")
            return out

        # ---- <mmthink> (multimodal-reasoning, NO generated image) understanding ----
        # These (SceneVerse / SenseNova, ~mmthink reasoning over INPUT frames with
        # <imagine:..> text, no generated_image) also collide with the template wrap.
        # <mmthink> is a CUSTOM block (not the native <think>), so render exactly like
        # the generation path renders mmthink text: empty-assistant scaffold as PROMPT,
        # then append the raw <mmthink>..</mmthink><answer>..</answer> as the supervised
        # response -> clean, no nesting, identical to the with-image mmthink samples.
        if len(_asst) == 1 and _join_text(_asst[0]).lstrip().startswith("<mmthink>"):
            prompt_messages = [m for m in messages if m.get("role") != "assistant"] + \
                              [{"role": "assistant", "content": ""}]
            p_inputs = self.processor.apply_chat_template(
                prompt_messages, return_dict=True, tokenize=True,
                add_generation_prompt=False, max_length=self.config.max_length, truncation=True,
            )
            prompt_ids = list(p_inputs["input_ids"][0])
            while prompt_ids and prompt_ids[-1] == EOS_TOKEN_ID:
                prompt_ids.pop()
            a_ids = self.tokenizer.encode(_join_text(_asst[0]).lstrip(), add_special_tokens=False)
            full_ids = prompt_ids + a_ids + [EOS_TOKEN_ID]
            lab = [-100] * len(prompt_ids) + a_ids + [EOS_TOKEN_ID]
            if len(full_ids) > self.config.max_length:
                full_ids = full_ids[: self.config.max_length]
                lab = lab[: self.config.max_length]
            input_ids = torch.tensor(full_ids, dtype=torch.long)
            labels = torch.tensor(lab, dtype=torch.long)
            out = {
                "input_ids": input_ids,
                "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                "labels": labels,
                "shift_labels": torch.tensor(build_shift_labels(full_ids, lab), dtype=torch.long),
                "use_discrete_diffusion": False,
                "image_block_ranges": [],
            }
            for _k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
                if _k in p_inputs:
                    out[_k] = p_inputs[_k]
            merge = getattr(getattr(self.processor, "image_processor", None), "merge_size", None) or 2
            for grid_key in ("image_grid_thw", "video_grid_thw"):
                grid = out.get(grid_key)
                if grid is None:
                    continue
                expected = int(sum(int(t) * int(h) * int(w) for t, h, w in grid)) // (merge * merge)
                actual = int((input_ids == IMAGE_PLACEHOLDER_TOKEN_ID).sum().item())
                if actual != expected:
                    raise ValueError(
                        f"[truncated-image] {grid_key}: placeholder={actual} != features={expected}; skipping")
            return out

        # Process with processor (handles multimodal data)
        inputs = self.processor.apply_chat_template(
            messages,
            return_dict=True,
            tokenize=True,
            max_length=self.config.max_length,
            truncation=True,
        )

        input_ids_list = list(inputs["input_ids"][0])

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.ones(len(input_ids_list), dtype=torch.long)

        # Create labels with multi-turn support
        labels = self._create_multiturn_labels(input_ids)

        # Build shift_labels for understanding samples
        # For understanding (no image generation blocks), shift_labels is simply
        # the standard shifted labels: predict next token at each position
        input_ids_list = input_ids.tolist()
        labels_list = labels.tolist()
        shift_labels = build_shift_labels(input_ids_list, labels_list)

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "shift_labels": torch.tensor(shift_labels, dtype=torch.long),
            "use_discrete_diffusion": False,
            "image_block_ranges": [],  # No generation blocks for understanding
        }

        # Add multimodal data if present
        if "pixel_values" in inputs:
            output["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            output["image_grid_thw"] = inputs["image_grid_thw"]
        if "pixel_values_videos" in inputs:
            output["pixel_values_videos"] = inputs["pixel_values_videos"]
        if "video_grid_thw" in inputs:     
            output["video_grid_thw"] = inputs["video_grid_thw"]

        # ---- GUARD: image/video placeholder tokens MUST match vision-feature count.
        # truncation=True can cut an image's placeholder run while pixel_values keeps
        # ALL its features -> the model's get_placeholder_mask then hard-raises
        # ("Image feature/token count mismatch") at FORWARD on some rank, restart-
        # looping the whole job. Validate here and RAISE so the bad sample is caught
        # by the dataset try/except and SKIPPED (a truncated image sample is never
        # trainable anyway). merge_size folds the ViT patch grid into tokens.
        merge = getattr(getattr(self.processor, "image_processor", None), "merge_size", None) or 2
        for grid_key, ph_id in (("image_grid_thw", IMAGE_PLACEHOLDER_TOKEN_ID),
                                ("video_grid_thw", IMAGE_PLACEHOLDER_TOKEN_ID)):
            grid = output.get(grid_key)
            if grid is None:
                continue
            expected = int(sum(int(t) * int(h) * int(w) for t, h, w in grid)) // (merge * merge)
            actual = int((input_ids == ph_id).sum().item())
            if actual != expected:
                raise ValueError(
                    f"[truncated-image] {grid_key}: placeholder tokens={actual} != "
                    f"expected vision features={expected} (max_length cut an image block); skipping sample"
                )

        return output

    def _mirror_path(self, path: str) -> Optional[str]:
        """Map an original frame path to its VAE-round-trip mirror (.png under
        dst_root). Returns the mirror path only if it exists on disk, else None.
        See train/data_prep/build_vae_roundtrip.py for how mirrors are made."""
        src_root = self.config.prior_frame_vae_roundtrip_src_root
        dst_root = self.config.prior_frame_vae_roundtrip_dst_root
        if not path or not src_root or not dst_root or not path.startswith(src_root):
            return None
        mirror = dst_root + path[len(src_root):]
        mirror = os.path.splitext(mirror)[0] + ".png"
        return mirror if os.path.exists(mirror) else None

    def _maybe_roundtrip_prior_frames(
        self, item: Dict, user_system_messages: List[Dict]
    ) -> List[Dict]:
        """Replace PRIOR-STEP context frames with their VAE-round-tripped mirror,
        each with per-call probability `prior_frame_vae_roundtrip_prob`.

        Obs frames (real at inference too) and the FM target (assistant side) are
        never touched. The obs/prior boundary comes from metadata — robust to
        empty texts / CFG dropout, unlike a "first text item" heuristic:
            num_obs = num_context_frames - (step_index - 1)
        The first `num_obs` user-side image items are obs; the rest are prior-step
        frames. Builds and returns a NEW message list — never mutates self.data."""
        p = self.config.prior_frame_vae_roundtrip_prob
        meta = item.get("metadata", {}) or {}
        ncf = meta.get("num_context_frames")
        sidx = meta.get("step_index")
        if ncf is None or sidx is None:
            return user_system_messages  # cannot locate boundary safely → no-op
        num_obs = int(ncf) - (int(sidx) - 1)
        if num_obs >= int(ncf):
            return user_system_messages  # step 1 (no prior frames) → no-op

        img_seen = 0
        swapped = False
        new_messages = []
        for msg in user_system_messages:
            content = msg.get("content")
            if msg.get("role") not in ("user", "system") or not isinstance(content, list):
                new_messages.append(msg)
                continue
            new_content = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "image":
                    is_prior = img_seen >= num_obs
                    img_seen += 1
                    if is_prior and random.random() < p:
                        mirror = self._mirror_path(c.get("image", ""))
                        if mirror is not None:
                            c = {**c, "image": mirror}
                            swapped = True
                new_content.append(c)
            new_messages.append({**msg, "content": new_content})
        return new_messages if swapped else user_system_messages

    def _process_generation_sample(self, item: Dict) -> Dict[str, Any]:
        """
        Process generation sample (T2I, I2I, or interleaved) with Flow Matching + VAE.
        Raw images are loaded, transformed for VAE, and placeholder tokens are inserted
        into the sequence. The actual VAE encoding happens in the training loop.
        """
        messages = item["messages"]
        vae_transform = VAEImageTransform(
            max_size=self.config.max_image_size,
            min_size=self.config.min_image_size,
            stride=self.config.image_stride,
        )
        # Collect input image paths from user messages (for input VAE latent injection)
        input_image_paths = []
        if self.config.enable_input_vae_latent:
            for msg in messages:
                if msg.get("role") in ["user", "system"]:
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "image":
                                input_image_paths.append(c.get("image", ""))
        # Collect generated image paths from assistant messages
        generated_image_paths = []
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "generated_image":
                            raw_path = c.get("raw_image", c.get("image_tokens", ""))
                            generated_image_paths.append(raw_path)

        user_system_messages = []
        assistant_contents = []
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role in ["user", "system"]:
                user_system_messages.append(msg)
            elif role == "assistant":
                is_last = all(m.get("role") != "assistant" for m in messages[i + 1:])
                assistant_contents.append((msg.get("content", ""), is_last))

        # n_gen = number of frames this assistant turn generates. Drives BOTH the
        # task instruction and the <Image> vs <Video> token layout below:
        #   n_gen >= 2 -> multi-frame, wrapped in ONE <Video> ... </Video>
        #   n_gen == 1 -> single image (<Image>...</Image>); interleave vs single-
        #                 frame distinguished by metadata.step_index in build_task_instruction
        n_gen = len(generated_image_paths)
        # Append the mode instruction to the user turn (part of the conditioning;
        # flows through the SAME CFG-dropout path as the rest of the user text, and
        # the inference scripts append the identical string -> train==infer parity).
        _instr = build_task_instruction(item.get("metadata"), n_gen)
        if _instr:
            _injected = False
            for _m in reversed(user_system_messages):
                if _m.get("role") == "user":
                    _c = _m.get("content")
                    if isinstance(_c, list):
                        _m["content"] = _c + [{"type": "text", "text": _instr}]
                    elif isinstance(_c, str):
                        _m["content"] = (_c + "\n" + _instr) if _c else _instr
                    else:
                        _m["content"] = [{"type": "text", "text": _instr}]
                    _injected = True
                    break
            if not _injected:
                user_system_messages.append(
                    {"role": "user", "content": [{"type": "text", "text": _instr}]}
                )
        # Prior-frame VAE round-trip augmentation (image-side exposure-bias fix):
        # swap PRIOR-STEP context frames to their pre-degraded mirror with prob p.
        # Returns a fresh list; never mutates self.data[idx]. No-op when prob == 0.
        if self.config.prior_frame_vae_roundtrip_prob > 0.0:
            user_system_messages = self._maybe_roundtrip_prior_frames(
                item, user_system_messages
            )
        # CFG dropout: drop text conditioning (and input VAE latent) with shared probability.
        # ViT features in the prompt are preserved (they're part of processor output),
        # but input VAE latent spatial conditioning is dropped together with text,
        # so the unconditional branch at inference has neither text nor spatial cues.
        cfg_dropped = random.random() < self.config.text_cond_dropout_prob
        if cfg_dropped:
            user_system_messages = self._apply_cfg_dropout(user_system_messages)
        # Tokenize user/system prompt
        prompt_messages = user_system_messages + [{"role": "assistant", "content": ""}]
        try:
            prompt_inputs = self.processor.apply_chat_template(
                prompt_messages,
                return_dict=True,
                tokenize=True,
                add_generation_prompt=False,
            )
            prompt_ids = list(prompt_inputs["input_ids"][0])
            while prompt_ids and prompt_ids[-1] == EOS_TOKEN_ID:
                prompt_ids.pop()
        except Exception as e:
            logger.warning(f"Failed to process prompt: {e}")
            prompt_inputs = self.processor.apply_chat_template(
                messages,
                return_dict=True,
                tokenize=True,
                max_length=self.config.max_length,
                truncation=True,
            )
            prompt_ids = list(prompt_inputs["input_ids"][0])
        # Build the full sequence: prompt + assistant responses
        final_input_ids = list(prompt_ids)
        final_labels = [-100] * len(prompt_ids)

        # Inject input image VAE latent placeholders after prompt (before assistant)
        # Skip when CFG dropout is active — the unconditional branch should have
        # neither text nor spatial conditioning from input images.
        input_vae_image_infos = []
        if self.config.enable_input_vae_latent and input_image_paths and not cfg_dropped:
            for img_path in input_image_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = vae_transform(img)
                except Exception as e:
                    logger.warning(f"Failed to load input image for VAE latent {img_path}: {e}")
                    continue
                H, W = img_tensor.shape[1:]
                h = H // self.config.vae_image_downsample
                w = W // self.config.vae_image_downsample
                num_latent_tokens = h * w
                latent_start = len(final_input_ids)
                final_input_ids.extend([INPUT_LATENT_PLACEHOLDER_ID] * num_latent_tokens)
                final_labels.extend([-100] * num_latent_tokens)
                pos_ids = get_flattened_position_ids(
                    H, W,
                    self.config.vae_image_downsample,
                    self.config.max_latent_size,
                )
                input_vae_image_infos.append({
                    "image_tensor": img_tensor,
                    "latent_shape": (h, w),
                    "latent_start_idx": latent_start,
                    "num_latent_tokens": num_latent_tokens,
                    "latent_position_ids": pos_ids,
                })

        vae_image_infos = []
        gen_img_idx = 0
        # Multi-frame (>=2 generated frames) is wrapped in ONE <Video> ... </Video>
        # with bare back-to-back latent spans (NO per-frame <Image>), so the AR
        # meaning of <Image> stays clean for interleave. n_gen==1 keeps the existing
        # <Image>...</Image> block. One vae_image_infos entry per frame either way,
        # so g_seqlens/flow_positions stay per-frame correct (model is unchanged).
        is_multiframe = n_gen >= 2
        video_open = False
        for turn_idx, (content, is_last_turn) in enumerate(assistant_contents):
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "generated_image":
                        if gen_img_idx < len(generated_image_paths):
                            img_path = generated_image_paths[gen_img_idx]
                            try:
                                img = Image.open(img_path).convert("RGB")
                                img_tensor = vae_transform(img)
                            except Exception as e:
                                logger.warning(f"Failed to load image {img_path}: {e}")
                                gen_img_idx += 1
                                continue
                            H, W = img_tensor.shape[1:]
                            h = H // self.config.vae_image_downsample
                            w = W // self.config.vae_image_downsample
                            num_latent_tokens = h * w
                            if is_multiframe:
                                # Open the <Video> wrapper once, before the first frame.
                                if not video_open:
                                    final_input_ids.append(VIDEO_START_TOKEN_ID)
                                    final_labels.append(
                                        VIDEO_START_TOKEN_ID if self.config.train_video_start_token else -100
                                    )
                                    video_open = True
                                # Bare latent span for this frame (no <Image>/</Image>).
                                latent_start = len(final_input_ids)
                                final_input_ids.extend([LATENT_PLACEHOLDER_ID] * num_latent_tokens)
                                final_labels.extend([-100] * num_latent_tokens)
                            else:
                                # <Image> token. In joint-interleave mode it is a CE
                                # target so the preceding text position learns to
                                # predict <Image> (the text->image switch trigger).
                                final_input_ids.append(IMAGE_START_TOKEN_ID)
                                final_labels.append(
                                    IMAGE_START_TOKEN_ID if self.config.train_image_start_token else -100
                                )
                                latent_start = len(final_input_ids)
                                final_input_ids.extend([LATENT_PLACEHOLDER_ID] * num_latent_tokens)
                                final_labels.extend([-100] * num_latent_tokens)
                                # </Image> token
                                final_input_ids.append(IMAGE_END_TOKEN_ID)
                                final_labels.append(-100)
                            # 2D position IDs
                            pos_ids = get_flattened_position_ids(
                                H, W,
                                self.config.vae_image_downsample,
                                self.config.max_latent_size
                            )

                            vae_image_infos.append({
                                "image_tensor": img_tensor,
                                "latent_shape": (h, w),
                                "latent_start_idx": latent_start,
                                "num_latent_tokens": num_latent_tokens,
                                "latent_position_ids": pos_ids,
                            })
                            gen_img_idx += 1
                    elif isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        if text.strip():
                            text_ids = self.tokenizer.encode(text, add_special_tokens=False)
                            final_input_ids.extend(text_ids)
                            final_labels.extend(text_ids)
        # Close the multi-frame <Video> wrapper after the last frame (label -100,
        # like </Image>). Only if it was opened (>=1 frame actually loaded).
        if video_open:
            final_input_ids.append(VIDEO_END_TOKEN_ID)
            final_labels.append(-100)
        # EOS
        if final_input_ids and final_input_ids[-1] != EOS_TOKEN_ID:
            final_input_ids.append(EOS_TOKEN_ID)
            final_labels.append(EOS_TOKEN_ID)
        max_len = self.config.max_length
        if len(final_input_ids) > max_len:
            orig_len = len(final_input_ids)
            final_input_ids = final_input_ids[:max_len]
            final_labels = final_labels[:max_len]
            # Filter out VAE infos whose latent tokens fall outside truncated sequence
            orig_vae_count = len(vae_image_infos)
            orig_input_vae_count = len(input_vae_image_infos)
            vae_image_infos = [
                info for info in vae_image_infos
                if info["latent_start_idx"] + info["num_latent_tokens"] <= max_len
            ]
            input_vae_image_infos = [
                info for info in input_vae_image_infos
                if info["latent_start_idx"] + info["num_latent_tokens"] <= max_len
            ]
            if len(vae_image_infos) < orig_vae_count or len(input_vae_image_infos) < orig_input_vae_count:
                logger.warning(
                    f"Sequence truncated from {orig_len} to {max_len}, "
                    f"dropped {orig_vae_count - len(vae_image_infos)} vae_image_infos, "
                    f"{orig_input_vae_count - len(input_vae_image_infos)} input_vae_image_infos"
                )
        output = {
            "input_ids": torch.tensor(final_input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(final_input_ids), dtype=torch.long),
            "labels": torch.tensor(final_labels, dtype=torch.long),
            "use_flow_matching": len(vae_image_infos) > 0,
            "vae_image_infos": vae_image_infos,
            "input_vae_image_infos": input_vae_image_infos,
        }
        # pixel_values und
        if "pixel_values" in prompt_inputs:
            output["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            output["image_grid_thw"] = prompt_inputs["image_grid_thw"]

        # ---- GUARD: input-ViT placeholder tokens MUST match pixel_values features.
        # The max_length truncation above cuts final_input_ids (and filters the
        # GENERATED-latent vae_image_infos) but CANNOT trim the input-ViT
        # pixel_values/image_grid_thw (obs/prior frames fed to SigLIP). So a deep
        # interleave/multiframe sample whose obs-frame ViT placeholders get cut by
        # truncation keeps all its pixel_values -> the model's get_placeholder_mask
        # hard-raises "Image feature/token count mismatch" on some rank and aborts
        # the whole collective job (the step-~50 crash loop). These deep samples are
        # meant to be dropped by _est_seq_len's load-time over-length pre-filter;
        # this is the last-line guard for any that still reach here. RAISE so the
        # dataset try/except skips the sample (a truncated-image sample is untrainable).
        _grid = output.get("image_grid_thw")
        if _grid is not None:
            _ids_t = output["input_ids"]
            merge = getattr(getattr(self.processor, "image_processor", None), "merge_size", None) or 2
            expected = int(sum(int(t) * int(h) * int(w) for t, h, w in _grid)) // (merge * merge)
            actual = int((_ids_t == IMAGE_PLACEHOLDER_TOKEN_ID).sum().item())
            if actual != expected:
                raise ValueError(
                    f"[truncated-input-image] generation-path placeholder tokens={actual} != "
                    f"expected ViT features={expected} (max_length cut obs/prior frame placeholders); "
                    f"skipping sample"
                )
        return output


    def _apply_cfg_dropout(self, messages: List[Dict]) -> List[Dict]:
        """
        Apply CFG dropout by clearing user text content but preserving images.

        For classifier-free guidance training, we want to sometimes drop the
        conditioning (text prompt) while keeping the structure. For image-to-image
        tasks, we should preserve input images.
        """
        modified = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Keep images, clear text
                    new_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") in ["image", "video"]:
                                new_content.append(item)  # Keep images/videos
                            elif item.get("type") == "text":
                                new_content.append({"type": "text", "text": ""})  # Clear text
                        elif isinstance(item, str):
                            new_content.append({"type": "text", "text": ""})
                    modified.append(
                        {"role": "user", "content": new_content if new_content else [{"type": "text", "text": ""}]}
                    )
                else:
                    # String content, just clear it
                    modified.append({"role": "user", "content": [{"type": "text", "text": ""}]})
            else:
                modified.append(msg)
        return modified


class ParquetT2IDataset(torch.utils.data.IterableDataset):
    """
    T2I dataset that streams from parquet files with embedded JPEG bytes.

    Uses IterableDataset to read parquet files sequentially — each file is read
    once and all rows are yielded before moving to the next file. This avoids
    the random-access problem where shuffle causes each sample to load an entire
    parquet file (~hundreds of MB) from CephFS just to read 1 row.

    Shuffling strategy:
      - File order is shuffled each epoch (via worker_init_fn seed)
      - Rows within each file are shuffled in memory after loading

    Multi-worker: files are sharded across DataLoader workers so each worker
    reads a disjoint subset. Multi-node (DDP): files are further sharded by rank.

    Output format matches UnifiedDataset._process_generation_sample() exactly,
    so the existing UnifiedDataCollator works unchanged.
    """

    def __init__(
        self,
        data_dir: str,
        processor,
        config: Optional[TaskConfig] = None,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.processor = processor
        self.config = config or TaskConfig()
        self.data_dir = data_dir
        self.shuffle = bool(shuffle)
        # Internal RANK sharding (files split over world_size*num_workers global
        # slots). Default (0,1) = no rank shard -> accelerate's IterableDatasetShard
        # does rank sharding (the original, prepared, T2I-only path — UNCHANGED).
        # Set (rank, world_size) ONLY on the self-sharded mix path, where the
        # dataloader is NOT prepared (so there is no IDS), and each rank must
        # produce just its own 1/world_size of files to avoid the W× processing
        # amplification.
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))

        # 1. Find all parquet files recursively (skip .partial files — incomplete writes)
        all_parquet_files = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(data_dir)
            for f in files if f.endswith('.parquet') and '.partial' not in f
        )
        if not all_parquet_files:
            raise ValueError(f"No .parquet files found under {data_dir}")

        # 2. Index caching.
        # If a cache exists but the on-disk file count drifted (files added/removed,
        # .partial files turning real, etc.), we used to do a full re-scan and crash
        # on the first corrupted parquet. Two fixes here:
        #   (a) when we have a cache, do an *incremental* update — keep cached
        #       entries whose path still exists and only metadata-scan new files;
        #   (b) wrap every read_metadata() in try/except so a single bad parquet
        #       no longer kills training. Bad files are skipped (and logged).
        cache_path = os.path.join(data_dir, ".parquet_index_cache.json")
        file_index = None
        cached = None
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                if cached.get("num_files") == len(all_parquet_files):
                    file_index = cached["files"]
                    logger.info(f"Loaded parquet index cache: {cache_path} ({len(file_index)} files)")
                else:
                    logger.info(
                        f"Cache stale (cached={cached.get('num_files')} vs on-disk={len(all_parquet_files)}); "
                        f"will incrementally update."
                    )
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")

        if file_index is None:
            import pyarrow.parquet as _pq

            cached_files = {p: n for p, n in cached["files"]} if cached and "files" in cached else {}
            on_disk_set = set(all_parquet_files)
            # Keep cached entries whose paths still exist on disk
            kept = [(p, cached_files[p]) for p in all_parquet_files if p in cached_files]
            new_paths = [p for p in all_parquet_files if p not in cached_files]

            if cached_files:
                logger.info(
                    f"Incremental scan: kept {len(kept)} cached entries, "
                    f"need to scan {len(new_paths)} new files."
                )
            else:
                logger.info(f"Scanning {len(all_parquet_files)} parquet files for metadata...")
                new_paths = list(all_parquet_files)

            scanned = []
            bad_files = []
            for i, path in enumerate(new_paths):
                try:
                    num_rows = _pq.read_metadata(path).num_rows
                    scanned.append((path, num_rows))
                except Exception as e:
                    bad_files.append((path, str(e)))
                    continue
                if (i + 1) % 1000 == 0:
                    logger.info(f"  scanned {i + 1}/{len(new_paths)} files...")

            if bad_files:
                logger.warning(
                    f"Skipped {len(bad_files)} corrupted/unreadable parquet files "
                    f"(showing up to 5):"
                )
                for p, err in bad_files[:5]:
                    logger.warning(f"  - {p}: {err}")

            file_index = kept + scanned

            # Determine caption_col from a healthy file (cached column wins if present)
            caption_col = cached.get("caption_col") if cached else None
            if caption_col is None:
                # Try files in order until one opens successfully
                for probe in [fi[0] for fi in file_index]:
                    try:
                        _schema = _pq.read_schema(probe)
                        _col_names = _schema.names
                        del _schema
                        caption_col = (
                            "text" if "text" in _col_names
                            else ("blip2_caption" if "blip2_caption" in _col_names else None)
                        )
                        if caption_col is not None:
                            break
                    except Exception:
                        continue
                if caption_col is None:
                    raise ValueError(
                        f"No caption column found in any readable parquet under {data_dir}."
                    )
            del _pq

            try:
                with open(cache_path, "w") as f:
                    json.dump(
                        {"num_files": len(file_index), "files": file_index, "caption_col": caption_col},
                        f,
                    )
                logger.info(f"Saved parquet index cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache {cache_path}: {e}")
        else:
            caption_col = cached.get("caption_col")
            if caption_col is None:
                import pyarrow.parquet as _pq
                # Probe in order, skip unreadable files
                caption_col = None
                for probe in [fi[0] for fi in file_index]:
                    try:
                        _schema = _pq.read_schema(probe)
                        _col_names = _schema.names
                        del _schema
                        caption_col = (
                            "text" if "text" in _col_names
                            else ("blip2_caption" if "blip2_caption" in _col_names else None)
                        )
                        if caption_col is not None:
                            break
                    except Exception:
                        continue
                del _pq
                if caption_col is None:
                    raise ValueError(f"No caption column found in any readable parquet under {data_dir}.")

        self.file_paths = [fi[0] for fi in file_index]
        self.file_rows = [fi[1] for fi in file_index]
        self.total_rows = sum(self.file_rows)
        self.caption_col = caption_col
        self._epoch = 0
        # Row-level precise resume: number of pre-shard rows each DataLoader
        # worker should skip on the next __iter__. Set via
        # set_resume_skip_per_worker(); auto-cleared after the iteration.
        self._resume_skip_per_worker = 0

        logger.info(
            f"ParquetT2IDataset (IterableDataset): {data_dir}\n"
            f"  Files: {len(self.file_paths)}, Total rows: {self.total_rows}, Caption column: {self.caption_col}"
        )

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic file-level shuffling across epochs."""
        self._epoch = epoch

    def set_resume_skip_per_worker(self, skip_rows_per_worker: int):
        """Make the next __iter__ on each worker yield rows starting AFTER
        `skip_rows_per_worker` pre-shard rows have been consumed by that worker.

        Row-level precise (not file-granular): used together with seeded
        in-file shuffling so every (epoch, worker, file_idx_in_shuffled_order,
        row_idx_in_file) is reproducible. Cleared automatically at end of
        __iter__ so subsequent epochs run from the start.
        """
        self._resume_skip_per_worker = max(0, int(skip_rows_per_worker))

    def __len__(self) -> int:
        """Approximate length for progress bar / scheduler estimation."""
        return self.total_rows

    def __iter__(self):
        import pyarrow.parquet as pq

        # Worker-level file sharding only.
        # Rank-level sharding is handled by accelerate's IterableDatasetShard
        # (wraps this dataset after accelerator.prepare(), keeps every world_size-th
        # sample for this rank). We must NOT do rank sharding here — otherwise
        # data gets double-sharded.
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        # Shuffle file order deterministically per epoch (optional)
        file_indices = list(range(len(self.file_paths)))
        if self.shuffle:
            rng = random.Random(self._epoch + 42)
            rng.shuffle(file_indices)

        # Shard files across GLOBAL slots = world_size * num_workers; this
        # (rank, worker) owns slot `rank*num_workers + worker_id`. With the
        # default (rank=0, world_size=1) this reduces to the original worker-only
        # round-robin (IDS then does rank sharding on the prepared loader). With a
        # real (rank, world_size) it self-rank-shards (no IDS, no W× discard).
        total_slots = self.world_size * num_workers
        my_slot = self.rank * num_workers + worker_id
        my_file_indices = [fi for i, fi in enumerate(file_indices) if i % total_slots == my_slot]

        # ---- Row-level precise resume ----
        # Pop the skip count once and clear, so the next epoch starts fresh.
        skip_remaining = int(getattr(self, "_resume_skip_per_worker", 0))
        self._resume_skip_per_worker = 0
        if skip_remaining > 0:
            logger.info(
                f"ParquetT2IDataset[{os.path.basename(self.data_dir)}] "
                f"worker {worker_id}/{num_workers}: row-level resume skip = {skip_remaining:,}"
            )
        # ---- end resume skip ----

        # Reuse these across all rows to avoid repeated allocation
        vae_transform = VAEImageTransform(
            max_size=self.config.max_image_size,
            min_size=self.config.min_image_size,
            stride=self.config.image_stride,
        )

        for file_pos, file_idx in enumerate(my_file_indices):
            # If we still need to skip whole files, fast-skip without opening them.
            file_n = self.file_rows[file_idx]
            if skip_remaining >= file_n:
                skip_remaining -= file_n
                continue

            try:
                table = pq.read_table(
                    self.file_paths[file_idx],
                    columns=["jpg", self.caption_col],
                )
            except Exception as e:
                logger.warning(f"ParquetT2IDataset: failed to read file {self.file_paths[file_idx]}: {e}")
                # We *would* have yielded file_n rows from this file; account for the
                # skip budget so we land on the same place we would have without errors.
                # (file_n already > skip_remaining here, so the file would have hosted
                # the boundary; clamp skip_remaining to 0.)
                skip_remaining = 0
                continue

            # Shuffle rows within file — DETERMINISTIC seed based on (epoch, file_idx).
            # This is what makes row-level resume precise: the same (epoch, file_idx)
            # always yields the same row order, so skip_remaining maps to the same row.
            num_rows = table.num_rows
            row_order = list(range(num_rows))
            if self.shuffle:
                row_rng = random.Random((self._epoch * 0x9E3779B1 + file_idx) & 0xFFFFFFFF)
                row_rng.shuffle(row_order)

            # Apply the boundary-file row skip (skip_remaining < num_rows here).
            start_row_pos = skip_remaining
            skip_remaining = 0
            row_order = row_order[start_row_pos:]

            for row_idx in row_order:
                try:
                    sample = self._process_row(table, row_idx, vae_transform)
                    if sample is not None:
                        yield sample
                except Exception as e:
                    logger.warning(f"ParquetT2IDataset: failed row {row_idx} in {self.file_paths[file_idx]}: {e}")
                    continue

            # Release memory for this file before loading next
            del table

    def _process_row(self, table, row_idx: int, vae_transform) -> Optional[Dict[str, Any]]:
        jpg_bytes = table.column("jpg")[row_idx].as_py()
        caption = table.column(self.caption_col)[row_idx].as_py()

        if not jpg_bytes:
            return None
        if not caption or not caption.strip():
            return None

        # Decode image
        img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")

        # VAE transform (reuse instance from __iter__ to avoid per-row allocation)
        img_tensor = vae_transform(img)

        # Build messages + tokenize
        user_system_messages = [{"role": "user", "content": [{"type": "text", "text": caption}]}]

        # CFG text dropout
        if random.random() < self.config.text_cond_dropout_prob:
            user_system_messages = [{"role": "user", "content": [{"type": "text", "text": ""}]}]

        prompt_messages = user_system_messages + [{"role": "assistant", "content": ""}]
        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            return_dict=True,
            tokenize=True,
            add_generation_prompt=False,
        )
        prompt_ids = list(prompt_inputs["input_ids"][0])
        while prompt_ids and prompt_ids[-1] == EOS_TOKEN_ID:
            prompt_ids.pop()

        # Build sequence: prompt + <Image> + [LATENT_PLACEHOLDER]*N + </Image> + EOS
        H, W = img_tensor.shape[1:]
        h = H // self.config.vae_image_downsample
        w = W // self.config.vae_image_downsample
        num_latent_tokens = h * w

        final_input_ids = list(prompt_ids)
        final_labels = [-100] * len(prompt_ids)

        final_input_ids.append(IMAGE_START_TOKEN_ID)
        final_labels.append(-100)

        latent_start = len(final_input_ids)
        final_input_ids.extend([LATENT_PLACEHOLDER_ID] * num_latent_tokens)
        final_labels.extend([-100] * num_latent_tokens)

        final_input_ids.append(IMAGE_END_TOKEN_ID)
        final_labels.append(-100)

        # EOS (match JSONL pipeline: conditional append)
        if final_input_ids[-1] != EOS_TOKEN_ID:
            final_input_ids.append(EOS_TOKEN_ID)
            final_labels.append(EOS_TOKEN_ID)

        pos_ids = get_flattened_position_ids(
            H, W,
            self.config.vae_image_downsample,
            self.config.max_latent_size,
        )

        vae_image_infos = [{
            "image_tensor": img_tensor,
            "latent_shape": (h, w),
            "latent_start_idx": latent_start,
            "num_latent_tokens": num_latent_tokens,
            "latent_position_ids": pos_ids,
        }]

        # Truncate
        max_len = self.config.max_length
        if len(final_input_ids) > max_len:
            final_input_ids = final_input_ids[:max_len]
            final_labels = final_labels[:max_len]
            vae_image_infos = [
                info for info in vae_image_infos
                if info["latent_start_idx"] + info["num_latent_tokens"] <= max_len
            ]

        return {
            "input_ids": torch.tensor(final_input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(final_input_ids), dtype=torch.long),
            "labels": torch.tensor(final_labels, dtype=torch.long),
            "use_flow_matching": len(vae_image_infos) > 0,
            "vae_image_infos": vae_image_infos,
            "input_vae_image_infos": [],
        }


class ParquetMMUDataset(UnifiedDataset):
    """Understanding (image/video -> text) dataset that streams from parquet files
    whose image column holds EMBEDDED bytes — the MMU counterpart of
    ParquetT2IDataset.

    It is the bridge for the HuggingFace MMU corpora (LLaVA-OV-1.5
    Mid-Training-85M captions, LLaVA-OV-1.5-Instruct conversations,
    VisualWebInstruct QA) whose on-disk schema is *not* the unified `messages`
    format: each row is decoded (image bytes -> PIL) and mapped, per-file, into a
    `messages` dict, then routed through the inherited
    UnifiedDataset._process_understanding_sample (pure-text CE loss, no VAE/flow
    matching). The image is passed as an in-memory PIL object straight into the
    message content — the HunYuanVLMoTProcessor fetches it (verified), so NO
    image is ever written to disk.

    Per-FILE schema sniffing (not global): a single data_dir may mix subsets with
    different layouts, so each file's columns are inspected at read time and the
    row->messages adapter is chosen accordingly:
      * has `caption`                 -> caption pretraining (user=image, asst=caption)
      * has `conversations`           -> ShareGPT multi-turn (human/gpt, <image> split)
      * has `question` and `answer`   -> single-turn VQA
    The image column ("image") may be a struct<bytes,path> (single image) or a
    list of such structs (multi-image); inline "<image>" placeholders in
    conversation/question text are split out and matched to images in order.

    Sharding / shuffle / resume mirror ParquetT2IDataset EXACTLY (per-epoch file
    shuffle, in-file deterministic row shuffle, global (rank,worker) slot file
    sharding, row-level precise resume) so the mix path behaves identically to the
    T2I source. RAM is O(one parquet file) per worker.
    """

    # role string (ShareGPT / chat) -> unified messages role
    _ROLE_MAP = {
        "human": "user", "user": "user",
        "gpt": "assistant", "assistant": "assistant", "model": "assistant", "bot": "assistant",
        "system": "system",
    }

    def __init__(
        self,
        data_path: str,
        processor,
        config: Optional[TaskConfig] = None,
        shuffle: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        caption_prompt: str = "",
        image_root: str = "",
    ):
        # Mirror StreamingUnifiedDataset: do NOT call super().__init__ (it would
        # json.load a jsonl corpus). Set only what the reused per-item methods need.
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.config = config or TaskConfig()
        self.data_path = data_path
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        # Optional instruction prepended to caption-pretraining user turns. Empty
        # (default) = image-only user turn -> caption as the assistant completion
        # (standard caption-pretraining; no fixed-prompt memorization).
        self.caption_prompt = caption_prompt
        # Root for resolving RELATIVE image paths when the parquet stores image
        # references as strings/paths (not embedded bytes). Empty = bytes-only
        # sources (Mid-Training-85M, LLaVA-OV-Instruct); rows whose images cannot
        # be decoded/found are skipped.
        self.image_root = image_root
        # Self rank-shard on the mix path (no IDS); see ParquetT2IDataset.
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.data = []  # keep empty: inherited paths never touch self.data
        self._epoch = 0
        self._resume_skip_per_worker = 0

        import pyarrow.parquet as _pq

        # 1. Discover parquet files: data_path is a directory (recurse) OR a single
        #    .parquet file. Skip .partial (incomplete writes).
        if os.path.isdir(data_path):
            all_parquet_files = sorted(
                os.path.join(root, f)
                for root, _, files in os.walk(data_path)
                for f in files if f.endswith(".parquet") and ".partial" not in f
            )
            cache_path = os.path.join(data_path, ".parquet_mmu_index_cache.json")
        elif data_path.endswith(".parquet"):
            all_parquet_files = [data_path]
            cache_path = None  # single file: scan is trivial, no cache
        else:
            raise ValueError(f"ParquetMMUDataset: not a dir or .parquet file: {data_path}")
        if not all_parquet_files:
            raise ValueError(f"No .parquet files found under {data_path}")

        # 2. Index (path, num_rows) for file sharding + row-level resume. Reuse a
        #    cache when the on-disk file count matches; otherwise metadata-scan,
        #    skipping unreadable files (one bad parquet must not kill the job).
        file_index = None
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                if cached.get("num_files") == len(all_parquet_files):
                    file_index = cached["files"]
                    logger.info(f"Loaded MMU parquet index cache: {cache_path} ({len(file_index)} files)")
            except Exception as e:
                logger.warning(f"Failed to load MMU cache {cache_path}: {e}")

        if file_index is None:
            logger.info(f"Scanning {len(all_parquet_files)} MMU parquet files for metadata...")
            scanned, bad = [], 0
            for i, path in enumerate(all_parquet_files):
                try:
                    scanned.append([path, _pq.read_metadata(path).num_rows])
                except Exception as e:
                    bad += 1
                    if bad <= 5:
                        logger.warning(f"  skip unreadable parquet {path}: {e}")
                if (i + 1) % 1000 == 0:
                    logger.info(f"  scanned {i + 1}/{len(all_parquet_files)} files...")
            if bad:
                logger.warning(f"ParquetMMUDataset: skipped {bad} unreadable parquet files under {data_path}")
            file_index = scanned
            if cache_path:
                try:
                    with open(cache_path, "w") as f:
                        json.dump({"num_files": len(all_parquet_files), "files": file_index}, f)
                    logger.info(f"Saved MMU parquet index cache: {cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to save MMU cache {cache_path}: {e}")

        self.file_paths = [fi[0] for fi in file_index]
        self.file_rows = [fi[1] for fi in file_index]
        self.total_rows = sum(self.file_rows)
        logger.info(
            f"ParquetMMUDataset (IterableDataset): {data_path}\n"
            f"  Files: {len(self.file_paths)}, Total rows: {self.total_rows:,}, "
            f"rank={self.rank}/{self.world_size}"
        )

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def set_resume_skip_per_worker(self, skip_rows_per_worker: int):
        self._resume_skip_per_worker = max(0, int(skip_rows_per_worker))

    def __len__(self) -> int:
        return self.total_rows

    # ---- schema -> (image_col, text_cols, kind) -----------------------------
    @staticmethod
    def _detect_kind(col_names: List[str]) -> Optional[Tuple[Optional[str], List[str], str]]:
        """Pick the row->messages adapter from a file's columns. `image_col` is
        None for text-only conversation subsets (no `image` column) — those train
        as text->text understanding. caption / qa REQUIRE an image column."""
        image_col = "image" if "image" in col_names else None
        if "caption" in col_names and image_col is not None:
            return image_col, ["caption"], "caption"
        if "conversations" in col_names:
            return image_col, ["conversations"], "conversations"
        if "question" in col_names and "answer" in col_names and image_col is not None:
            return image_col, ["question", "answer"], "qa"
        return None

    # ---- image bytes/path -> PIL --------------------------------------------
    def _decode_images(self, img_val) -> List[Image.Image]:
        """Accept a struct<bytes,path> (single), a list of such structs, a bare
        bytes value, or a string/list-of-string path; return a list of RGB PIL
        images. Embedded bytes win; otherwise a path is resolved (against
        self.image_root when relative) and opened from disk. Unreadable/missing
        entries are dropped (the row is skipped upstream if none survive)."""
        def _resolve(p: str) -> Optional[str]:
            if os.path.isabs(p):
                return p if os.path.exists(p) else None
            cand = os.path.join(self.image_root, p) if self.image_root else p
            return cand if os.path.exists(cand) else None

        def _one(v):
            try:
                if isinstance(v, dict):
                    b = v.get("bytes")
                    if b:
                        return Image.open(io.BytesIO(b)).convert("RGB")
                    p = v.get("path")
                    rp = _resolve(p) if p else None
                    return Image.open(rp).convert("RGB") if rp else None
                if isinstance(v, (bytes, bytearray)):
                    return Image.open(io.BytesIO(v)).convert("RGB")
                if isinstance(v, str):
                    rp = _resolve(v)
                    return Image.open(rp).convert("RGB") if rp else None
            except Exception:
                return None
            return None

        if isinstance(img_val, list):
            out = [_one(v) for v in img_val]
            return [im for im in out if im is not None]
        im = _one(img_val)
        return [im] if im is not None else []

    def _interleave(self, text: str, images: List[Image.Image], cursor: List[int]) -> List[Dict]:
        """Split `text` on '<image>' placeholders, inserting the next image (by
        `cursor`) at each placeholder; returns a content-part list."""
        content: List[Dict] = []
        if not text:
            text = ""
        parts = text.split("<image>")
        for i, part in enumerate(parts):
            if i > 0 and cursor[0] < len(images):
                content.append({"type": "image", "image": images[cursor[0]]})
                cursor[0] += 1
            part = part.strip()
            if part:
                content.append({"type": "text", "text": part})
        return content

    def _row_to_messages(self, kind: str, row: Dict, images: List[Image.Image]) -> Optional[List[Dict]]:
        if kind == "caption":
            if not images:
                return None
            cap = row.get("caption")
            if not cap or not str(cap).strip():
                return None
            user_content: List[Dict] = [{"type": "image", "image": im} for im in images]
            if self.caption_prompt:
                user_content.append({"type": "text", "text": self.caption_prompt})
            return [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": str(cap)}]},
            ]
        if kind == "qa":
            if not images:
                return None
            q, a = row.get("question"), row.get("answer")
            if not q or not a or not str(a).strip():
                return None
            cursor = [0]
            content = self._interleave(str(q), images, cursor)
            # No "<image>" in the question -> prepend the image(s).
            if cursor[0] == 0:
                content = [{"type": "image", "image": im} for im in images] + content
            return [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": str(a)}]},
            ]
        if kind == "conversations":
            convs = row.get("conversations")
            if not convs:
                return None
            # Per-ROW image decision (the `image` column may exist but be null on
            # text-only rows — common in these subsets). If the dialogue REFERENCES
            # an image ("<image>" placeholder) but none decoded, the row is corrupt
            # -> drop it (never train a VQA turn stripped of its image). If there is
            # no placeholder and no image, it is a legit text->text row -> keep.
            def _turn_text(t):
                v = t.get("value")
                return str(v if v is not None else (t.get("content") or ""))
            n_img_tokens = sum(_turn_text(t).count("<image>") for t in convs)
            if n_img_tokens > 0 and not images:
                return None
            cursor = [0]
            messages: List[Dict] = []
            for turn in convs:
                role = self._ROLE_MAP.get((turn.get("from") or turn.get("role") or "").lower())
                if role is None:
                    continue
                content = self._interleave(_turn_text(turn), images, cursor)
                if not content:
                    content = [{"type": "text", "text": ""}]
                messages.append({"role": role, "content": content})
            # Images not yet placed (no/too-few "<image>" tokens) -> attach to the
            # first user turn so the visual context is never silently dropped.
            if cursor[0] < len(images):
                leftover = [{"type": "image", "image": im} for im in images[cursor[0]:]]
                for m in messages:
                    if m["role"] == "user":
                        m["content"] = leftover + m["content"]
                        break
            # Need at least one user and one assistant turn for a trainable label.
            if not any(m["role"] == "user" for m in messages) or not any(
                m["role"] == "assistant" for m in messages
            ):
                return None
            return messages
        return None

    def __iter__(self):
        import pyarrow.parquet as pq

        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        file_indices = list(range(len(self.file_paths)))
        if self.shuffle:
            random.Random(self._epoch + 42).shuffle(file_indices)

        total_slots = self.world_size * num_workers
        my_slot = self.rank * num_workers + worker_id
        my_file_indices = [fi for i, fi in enumerate(file_indices) if i % total_slots == my_slot]

        skip_remaining = int(getattr(self, "_resume_skip_per_worker", 0))
        self._resume_skip_per_worker = 0
        if skip_remaining > 0:
            logger.info(
                f"ParquetMMUDataset[{os.path.basename(self.data_path)}] "
                f"worker {worker_id}/{num_workers}: row-level resume skip = {skip_remaining:,}"
            )

        for file_idx in my_file_indices:
            file_n = self.file_rows[file_idx]
            if skip_remaining >= file_n:
                skip_remaining -= file_n
                continue
            path = self.file_paths[file_idx]
            try:
                pf = pq.ParquetFile(path)
                col_names = pf.schema_arrow.names
                detected = self._detect_kind(col_names)
                if detected is None:
                    logger.warning(f"ParquetMMUDataset: no MMU schema in {path} (cols={col_names}); skipping file")
                    skip_remaining = 0
                    continue
                image_col, text_cols, kind = detected
                read_cols = ([image_col] if image_col else []) + text_cols
                n_rg = pf.metadata.num_row_groups
            except Exception as e:
                logger.warning(f"ParquetMMUDataset: failed to open {path}: {e}")
                skip_remaining = 0
                continue

            # Stream by ROW GROUP, never whole-file: these MMU parquets embed image
            # bytes and a single file can be 10-20 GB (e.g. ureader/VisualWebInstruct
            # subsets, ~100 rows/group × 188 groups). pq.read_table() of the whole
            # file ballooned one DataLoader worker to ~24 GB RSS and tripped the
            # container's memory cgroup OOM-killer. read_row_group caps resident
            # memory at ~one group (~0.1 GB). Shuffle = row-group order (per epoch+
            # file) + rows within each group; resume skip walks groups then rows.
            rg_order = list(range(n_rg))
            if self.shuffle:
                random.Random((self._epoch * 0x9E3779B1 + file_idx) & 0xFFFFFFFF).shuffle(rg_order)

            for rg in rg_order:
                rg_rows = pf.metadata.row_group(rg).num_rows
                if skip_remaining >= rg_rows:
                    skip_remaining -= rg_rows
                    continue
                try:
                    table = pf.read_row_group(rg, columns=read_cols)
                except Exception as e:
                    logger.warning(f"ParquetMMUDataset: failed row_group {rg} in {path}: {e}")
                    skip_remaining = 0
                    continue

                row_order = list(range(table.num_rows))
                if self.shuffle:
                    random.Random(
                        (self._epoch * 0x9E3779B1 + file_idx * 1000003 + rg) & 0xFFFFFFFF
                    ).shuffle(row_order)
                row_order = row_order[skip_remaining:]
                skip_remaining = 0

                cols = {c: table.column(c) for c in read_cols}
                _max_len = getattr(self.config, "max_length", None)
                for row_idx in row_order:
                    try:
                        row = {c: cols[c][row_idx].as_py() for c in cols}
                        images = self._decode_images(row.get(image_col)) if image_col else []
                        messages = self._row_to_messages(kind, row, images)
                        if messages is None:
                            continue
                        # Pre-filter over-length BEFORE the processor: a multi-image
                        # MMU row (e.g. >=8 imgs @ ~1024 tok each) exceeds max_length,
                        # the processor truncates its placeholders, and the model then
                        # HARD-RAISES a feature/token mismatch that aborts the whole
                        # collective job. _est_seq_len now costs understanding images
                        # at ~1024 tok, so such rows are skipped here deterministically.
                        if _max_len and self._est_seq_len({"messages": messages}) > _max_len:
                            continue
                        yield self._process_understanding_sample({"messages": messages})
                    except Exception as e:
                        logger.warning(f"ParquetMMUDataset: failed row {row_idx} rg {rg} in {path}: {e}")
                        continue
                del table, cols
                # Return pyarrow's cached freed blocks to the OS. Reading row-groups
                # from the 10-20GB embedded-image MMU parquets, the default arrow
                # memory pool retains freed buffers (does not munmap), so per-worker
                # RSS creeps file-after-file -> the pod cgroup (256GB) eventually
                # OOM-kills the whole container (hard kill, no traceback/exit code).
                # release_unused() bounds the creep at ~one row-group.
                import pyarrow as _pa
                _pa.default_memory_pool().release_unused()


class StreamingUnifiedDataset(UnifiedDataset):
    """Streaming (lazy) iterable view over the same jsonl UnifiedDataset reads —
    but it NEVER materializes the corpus.

    Why it exists
    -------------
    UnifiedDataset is map-style: __init__ json.loads the WHOLE file into
    self.data (per-rank sharded). At the multi-million-record interleave scale,
    holding even 1/world_size of the corpus per rank costs real host RAM. For the
    dataset-MIX path (T2I parquet + interleave jsonl in one run) everything is
    converged onto the IterableDataset paradigm and accelerate's
    IterableDatasetShard does rank sharding — so THIS side must emit a GLOBAL
    stream (no load-time rank shard). To keep RAM bounded we stream the file
    line-by-line and shuffle through a fixed-size buffer, so resident RAM is
    O(shuffle_buffer_size) per worker, independent of corpus size.

    Sharding model (mirrors ParquetT2IDataset)
    ------------------------------------------
    * WORKER sharding only (round-robin over the non-blank line index). RANK
      sharding is handled downstream by IterableDatasetShard — doing it here too
      would double-shard.
    * Deterministic per (seed, epoch, worker_id) so resume + cross-rank window
      alignment (under IterableDatasetShard) are reproducible.

    Reuses every per-item method of UnifiedDataset (_remap_item,
    _determine_task_type, _is_task_enabled, _est_seq_len, _process_*); only
    __init__ (no materialization) and __iter__ are new. Output dicts are
    byte-for-byte the same schema as UnifiedDataset.__getitem__, so the existing
    UnifiedDataCollator consumes them unchanged.
    """

    def __init__(
        self,
        data_path,
        processor,
        config: Optional[TaskConfig] = None,
        shuffle: bool = True,
        seed: int = 42,
        shuffle_buffer_size: int = 20000,
        rank: int = 0,
        world_size: int = 1,
    ):
        # Deliberately do NOT call super().__init__ — that would read the whole
        # file into RAM. Set only what the reused per-item methods need.
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.config = config or TaskConfig()
        # data_path may be a single jsonl path or a LIST of jsonl paths (one
        # logical source = many files, e.g. the 188-file interleave corpus).
        self.data_paths = [data_path] if isinstance(data_path, str) else list(data_path)
        if not self.data_paths:
            raise ValueError("StreamingUnifiedDataset needs at least one jsonl path")
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.shuffle_buffer_size = max(1, int(shuffle_buffer_size))
        # Internal RANK sharding. When world_size>1, this rank's workers process
        # ONLY their 1/world_size of records (sharded over world_size*num_workers
        # global slots) and the dataloader is NOT handed to accelerate.prepare()
        # — so there is NO IterableDatasetShard reading the full B*W window and
        # discarding (world_size-1)/world_size of the (expensive) processed
        # samples. That discard is the W× processing amplification that starved
        # the GPU on the mix path; self-rank-sharding here makes each rank pay for
        # only the samples it uses. RAM stays O(shuffle_buffer) (still streaming).
        # Default (0,1) = no rank shard (single-process / IDS-managed parquet path).
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        # Inherited per-item methods never touch self.data, but keep it empty so
        # any stray self.data / len() reference in shared code paths is safe.
        self.data = []
        self._epoch = 0
        self._resume_skip_per_worker = 0
        self._len_cache: Optional[int] = None
        logger.info(
            f"StreamingUnifiedDataset (IterableDataset): {len(self.data_paths)} file(s) "
            f"(shuffle_buffer={self.shuffle_buffer_size}, rank={self.rank}/{self.world_size}, "
            f"RAM=O(buffer))"
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def set_resume_skip_per_worker(self, skip_rows_per_worker: int) -> None:
        """Skip this many OWNED (post-worker-shard) records at the start of the
        next __iter__, then clear. Mirrors ParquetT2IDataset.set_resume_skip_per_worker;
        best-effort under the shuffle buffer (records are drawn deterministically
        given the same (seed, epoch, worker), so a fixed skip lands reproducibly)."""
        self._resume_skip_per_worker = max(0, int(skip_rows_per_worker))

    def __len__(self) -> int:
        """EXACT GLOBAL non-blank line count, with a per-file DISK CACHE.

        The count feeds RatioMixIterableDataset.__len__ -> num_update_steps_per_epoch
        -> the LR scheduler's num_training_steps (cosine length), so it must stay
        EXACT (a byte-estimate would drift the LR trajectory). But counting 167M
        interleave lines (302 GB) line-by-line took 4-15 min on EVERY rank at every
        (re)start. Fix: cache each file's exact count keyed by (size, mtime) in a
        sidecar JSON; the first run counts once and writes it, then all other ranks
        and every resume read the cache instantly. A file whose size/mtime changed
        is re-counted (stale entry ignored). In-process result also memoized."""
        if self._len_cache is not None:
            return self._len_cache

        # Sidecar cache next to the FIRST data file's directory (shared across ranks).
        cache_path = None
        try:
            d = os.path.dirname(os.path.abspath(self.data_paths[0]))
            cache_path = os.path.join(d, ".jsonl_linecount_cache.json")
        except Exception:
            cache_path = None

        cache = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    cache = json.load(f)
            except Exception:
                cache = {}

        total = 0
        dirty = False
        for path in self.data_paths:
            try:
                st = os.stat(path)
                key = os.path.abspath(path)
                ent = cache.get(key)
                # Valid cache hit iff size+mtime match.
                if ent and ent.get("size") == st.st_size and ent.get("mtime") == int(st.st_mtime):
                    total += int(ent["lines"])
                    continue
                cnt = 0
                with open(path, "r") as f:
                    for ln in f:
                        if ln.strip():
                            cnt += 1
                cache[key] = {"size": st.st_size, "mtime": int(st.st_mtime), "lines": cnt}
                total += cnt
                dirty = True
            except OSError:
                continue

        # Persist new/updated entries (best-effort; concurrent ranks may race —
        # last writer wins, entries are independent so a lost write just re-counts
        # that file next time). Atomic rename to avoid torn reads.
        if dirty and cache_path:
            try:
                tmp = f"{cache_path}.{os.getpid()}.tmp"
                with open(tmp, "w") as f:
                    json.dump(cache, f)
                os.replace(tmp, cache_path)
            except Exception:
                pass

        self._len_cache = max(1, total)
        return self._len_cache

    def _iter_owned_records(self, worker_id: int, num_workers: int):
        """Yield json-parsed records this (rank, worker) owns, across ALL files.
        Sharded over `world_size * num_workers` GLOBAL slots — this (rank, worker)
        owns slot `rank*num_workers + worker_id`. So each RANK processes exactly
        its 1/world_size of records (no IDS full-stream-then-discard). File order
        is shuffled per-epoch (identically on every worker — seeded by seed+epoch
        only) so the global record index is consistent; only owned lines are
        json.loads'd."""
        files = list(self.data_paths)
        if self.shuffle and len(files) > 1:
            random.Random(self.seed + self._epoch).shuffle(files)
        total_slots = self.world_size * num_workers
        my_slot = self.rank * num_workers + worker_id
        rec_idx = 0
        for path in files:
            with open(path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    if rec_idx % total_slots == my_slot:
                        try:
                            yield json.loads(line)
                        except Exception as e:
                            logger.warning(f"[stream] bad json line (idx={rec_idx}): {e}")
                    rec_idx += 1

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        rng = random.Random(self.seed + self._epoch * 100003 + worker_id)
        skip = int(self._resume_skip_per_worker)
        self._resume_skip_per_worker = 0  # one-shot, cleared like ParquetT2IDataset
        if skip > 0:
            logger.info(
                f"StreamingUnifiedDataset[{len(self.data_paths)} file(s)] "
                f"worker {worker_id}/{num_workers}: resume skip = {skip:,} owned records"
            )

        max_len = getattr(self.config, "max_length", None)
        buf: List[Dict] = []

        # ---- POISON-SAMPLE PROBE (mirrors UnifiedDataset.__getitem__) ----------
        # A worker stuck on a *blocking* image read (stale/slow CephFS handle,
        # truncated file, decode bomb) can NOT be caught by the try/except below —
        # it hangs forever, which at 376 ranks deadlocks the whole job at the next
        # all-reduce (a silent, traceback-less hang). This probe writes the sample
        # being processed to a per-WORKER file in TRUNCATE mode right BEFORE the
        # heavy _process call; on success it is overwritten with DONE. So a frozen
        # worker leaves its file pinned at "START <id + image paths>" — the poison
        # sample. Enabled when IMG_PROBE_DIR is set (train.finetune sets it); never throws.
        _probe_dir = os.environ.get("IMG_PROBE_DIR")
        _pf = None
        if _probe_dir:
            try:
                os.makedirs(_probe_dir, exist_ok=True)
                _rk = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
                _pf = os.path.join(_probe_dir, f"stream_rank{_rk}_w{worker_id}_pid{os.getpid()}.log")
            except Exception:
                _pf = None

        def _probe(state, rec=None):
            if not _pf:
                return
            try:
                with open(_pf, "w") as f:
                    if rec is None:
                        f.write(f"{time.strftime('%F %T')} {state}\n")
                    else:
                        f.write(f"{time.strftime('%F %T')} {state} id={rec.get('id', '?')} {repr(rec)[:1200]}\n")
            except Exception:
                pass

        def _emit(rec):
            _probe("START", rec)            # pinned here if _process blocks/hangs
            try:
                out = self._process(rec)
                _probe("DONE")
                return out
            except Exception as e:
                _probe("DONE")
                logger.warning(
                    f"[stream skip-bad-sample] id={rec.get('id', '?')}: "
                    f"{type(e).__name__}: {str(e)[:200]}"
                )
                return None

        for rec in self._iter_owned_records(worker_id, num_workers):
            if skip > 0:
                skip -= 1
                continue
            self._remap_item(rec)
            task_type = self._determine_task_type(rec)
            if not self._is_task_enabled(task_type, rec):
                continue
            if max_len and self._est_seq_len(rec) > max_len:
                continue
            rec["_task_type"] = task_type
            buf.append(rec)
            if len(buf) >= self.shuffle_buffer_size:
                pick = rng.randrange(len(buf)) if self.shuffle else 0
                out = _emit(buf.pop(pick))
                if out is not None:
                    yield out
        # Drain the buffer at end of stream.
        if self.shuffle:
            rng.shuffle(buf)
        for rec in buf:
            out = _emit(rec)
            if out is not None:
                yield out

    def _process(self, rec: Dict) -> Dict[str, Any]:
        """Materialize ONE record via the reused UnifiedDataset per-item path."""
        if rec.get("_task_type", "understanding") == "understanding":
            return self._process_understanding_sample(rec)
        return self._process_generation_sample(rec)


@dataclass
class UnifiedDataCollator:
    """
    Data collator for unified training.
    Based on data_collator.py with additions for discrete diffusion.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        if not features:
            return {}

        batch = {}

        # Pad input_ids, attention_mask, and labels (same logic as data_collator.py)
        input_ids = [f["input_ids"] for f in features]
        max_len = max(len(ids) for ids in input_ids)

        if self.pad_to_multiple_of:
            max_len = (max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of

        if self.max_length:
            max_len = min(max_len, self.max_length)

        padded_input_ids = []
        attention_masks = []
        labels_list = []
        shift_labels_list = []

        for f in features:
            ids = f["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)

            seq_len = min(len(ids), max_len)
            padding_length = max_len - seq_len

            # Pad input_ids
            padded_ids = torch.cat(
                [ids[:seq_len], torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=ids.dtype)]
            )

            # Attention mask
            mask = f.get("attention_mask")
            if mask is None:
                mask = torch.ones_like(ids)
            elif not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.long)
            mask = torch.cat([mask[:seq_len], torch.zeros(padding_length, dtype=mask.dtype)])

            # Labels
            label = f.get("labels")
            if label is None:
                label = ids.clone()
            elif not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.long)
            label = torch.cat([label[:seq_len], torch.full((padding_length,), -100, dtype=label.dtype)])

            # Shift labels (for custom shift behavior in image blocks)
            shift_label = f.get("shift_labels")
            if shift_label is not None:
                if not isinstance(shift_label, torch.Tensor):
                    shift_label = torch.tensor(shift_label, dtype=torch.long)
                shift_label = torch.cat(
                    [shift_label[:seq_len], torch.full((padding_length,), -100, dtype=shift_label.dtype)]
                )
            else:
                shift_label = None

            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)
            labels_list.append(label)
            if shift_label is not None:
                shift_labels_list.append(shift_label)

        batch["input_ids"] = torch.stack(padded_input_ids)
        batch["attention_mask"] = torch.stack(attention_masks)
        batch["labels"] = torch.stack(labels_list)
         # Add vae_image_infos if any sample has it
        all_vae_images = []
        all_vae_latent_shapes = []
        all_latent_position_ids = []
        batch_indexes = []  # (batch_idx, seq_idx) pairs
        has_flow_matching = False
        for batch_idx, f in enumerate(features):
            vae_infos = f.get("vae_image_infos", [])
            if vae_infos:
                has_flow_matching = True
            for info in vae_infos:
                all_vae_images.append(info["image_tensor"])
                all_vae_latent_shapes.append(info["latent_shape"])
                all_latent_position_ids.append(info["latent_position_ids"])
                start = info["latent_start_idx"]
                n = info["num_latent_tokens"]
                batch_indexes.append(
                    (torch.full((n,), batch_idx, dtype=torch.long),
                     torch.arange(start, start + n, dtype=torch.long))
                )
        batch["use_flow_matching"] = has_flow_matching
        if all_vae_images:
            sizes = [img.shape for img in all_vae_images]
            max_c = max(s[0] for s in sizes)
            max_h = max(s[1] for s in sizes)
            max_w = max(s[2] for s in sizes)
            padded_images = torch.zeros(len(all_vae_images), max_c, max_h, max_w)
            for i, img in enumerate(all_vae_images):
                padded_images[i, :, :img.shape[1], :img.shape[2]] = img

            batch["padded_images"] = padded_images
            batch["patchified_vae_latent_shapes"] = all_vae_latent_shapes
            batch["vae_latent_positions"] = torch.cat(all_latent_position_ids, dim=0)
            batch_idxs = torch.cat([bi[0] for bi in batch_indexes], dim=0)
            seq_idxs = torch.cat([bi[1] for bi in batch_indexes], dim=0)
            batch["vae_latent_indexes"] = (batch_idxs, seq_idxs)

        # Build g_seqlens for block-wise full attention on generated image blocks.
        # Format: (start, end, batch_idx) — matches _generation_block_attn convention.
        #   start = <Image> token position (inclusive)
        #   end   = position after last latent token (exclusive, where </Image> sits)
        g_seqlens = []
        for batch_idx, f in enumerate(features):
            for info in f.get("vae_image_infos", []):
                g_start = info["latent_start_idx"] - 1  # include <Image> token
                g_end = info["latent_start_idx"] + info["num_latent_tokens"]
                g_seqlens.append((g_start, g_end, batch_idx))
        if g_seqlens:
            batch["g_seqlens"] = g_seqlens

        # Collect input image VAE latent infos (for TI2I spatial conditioning)
        all_input_vae_images = []
        all_input_vae_latent_shapes = []
        all_input_latent_position_ids = []
        input_batch_indexes = []
        for batch_idx, f in enumerate(features):
            input_vae_infos = f.get("input_vae_image_infos", [])
            for info in input_vae_infos:
                all_input_vae_images.append(info["image_tensor"])
                all_input_vae_latent_shapes.append(info["latent_shape"])
                all_input_latent_position_ids.append(info["latent_position_ids"])
                start = info["latent_start_idx"]
                n = info["num_latent_tokens"]
                input_batch_indexes.append(
                    (torch.full((n,), batch_idx, dtype=torch.long),
                     torch.arange(start, start + n, dtype=torch.long))
                )
        if all_input_vae_images:
            sizes = [img.shape for img in all_input_vae_images]
            max_c = max(s[0] for s in sizes)
            max_h = max(s[1] for s in sizes)
            max_w = max(s[2] for s in sizes)
            padded_input_vae_images = torch.zeros(len(all_input_vae_images), max_c, max_h, max_w)
            for i, img in enumerate(all_input_vae_images):
                padded_input_vae_images[i, :, :img.shape[1], :img.shape[2]] = img
            batch["padded_input_vae_images"] = padded_input_vae_images
            batch["input_vae_latent_shapes"] = all_input_vae_latent_shapes
            batch["input_vae_latent_positions"] = torch.cat(all_input_latent_position_ids, dim=0)
            ib_idxs = torch.cat([bi[0] for bi in input_batch_indexes], dim=0)
            is_idxs = torch.cat([bi[1] for bi in input_batch_indexes], dim=0)
            batch["input_vae_latent_indexes"] = (ib_idxs, is_idxs)

        # Handle input images for understanding (same as data_collator.py)
        pixel_values_list = [f["pixel_values"] for f in features if "pixel_values" in f]
        if pixel_values_list:
            batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)

            image_grid_thw_list = [f["image_grid_thw"] for f in features if "image_grid_thw" in f]
            if image_grid_thw_list:
                batch["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)

        # Handle videos (same as data_collator.py)
        pixel_values_videos_list = [f["pixel_values_videos"] for f in features if "pixel_values_videos" in f]
        if pixel_values_videos_list:
            batch["pixel_values_videos"] = torch.cat(pixel_values_videos_list, dim=0)

            video_grid_thw_list = [f["video_grid_thw"] for f in features if "video_grid_thw" in f]
            if video_grid_thw_list:
                batch["video_grid_thw"] = torch.cat(video_grid_thw_list, dim=0)

        return batch


__all__ = [
    "UnifiedDataset",
    "StreamingUnifiedDataset",
    "ParquetT2IDataset",
    "ParquetMMUDataset",
    "UnifiedDataCollator",
    "TaskConfig",
    "build_shift_labels",
    "VAEImageTransform",
    "get_flattened_position_ids",
    "IMAGE_START_TOKEN_ID",
    "IMAGE_END_TOKEN_ID",
    "VIDEO_START_TOKEN_ID",
    "VIDEO_END_TOKEN_ID",
    "LATENT_PLACEHOLDER_ID",
    "INPUT_LATENT_PLACEHOLDER_ID",
    "ASSISTANT_TOKEN_ID",
    "IMAGE_PLACEHOLDER_TOKEN_ID",
    "build_task_instruction",
    "TASK_INSTRUCTION_SINGLE_FRAME",
    "TASK_INSTRUCTION_MULTI_FRAME",
    "TASK_INSTRUCTION_INTERLEAVE",
]
