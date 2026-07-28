"""Length-bucketing sampler for P-Pack training.

NOT a DistributedSampler — `accelerate.prepare()` shards the resulting DataLoader
across ranks exactly once via BatchSamplerShard. Making this a DistributedSampler
would cause double-sharding (each rank seeing 1/W^2 of the data).

Each epoch:
  1. Globally shuffle all indices (deterministic per `seed + epoch`).
  2. Drop any indices in `bad_indices` (kept identical across ranks so they all
     drop the exact same set → no desync).
  3. Split into chunks of `bucket_size = batch_size * world_size * multiplier`.
  4. Sort each chunk by length ascending.
  5. Emit the concatenated, bucketed index list (all `len(dataset)` indices).

`accelerate` then strides every batch_size-sized contiguous sub-range to a
different rank via `BatchSamplerShard`, so each rank gets length-homogeneous
batches drawn from within each bucket.
"""
from __future__ import annotations

from typing import Callable, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, Sampler


class LengthBucketedSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        length_fn: Callable[[int], int],
        world_size: int = 1,
        bucket_size_multiplier: int = 50,
        seed: int = 42,
        shuffle: bool = True,
        bad_indices: Optional["set[int]"] = None,
    ):
        # Deliberately NOT calling super().__init__(dataset) — torch's Sampler
        # signature has changed across versions; we store state ourselves.
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.world_size = max(1, int(world_size))
        self.bucket_size = max(
            self.batch_size * self.world_size * bucket_size_multiplier,
            self.batch_size * self.world_size,
        )
        self._length_fn = length_fn
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self._lengths_cache: Optional[List[int]] = None
        # Identical across ranks → no desync when filtered before bucketing.
        self.bad_indices: set = set(bad_indices or ())
        # Optional hard cap on the number of indices emitted per epoch. Used by
        # the per-rank-sharded map path: each rank owns a disjoint 1/world_size
        # slice of the corpus whose size differs by <= a few records, so every
        # rank is clamped to the GLOBAL MIN batch count (× batch_size) to keep
        # the optimizer-step count identical across ranks (else gradient
        # all-reduce desyncs → NCCL deadlock). None = no cap. Must be a multiple
        # of batch_size. Applied AFTER the _round truncation in __iter__.
        self._max_samples: Optional[int] = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_max_samples(self, n: Optional[int]) -> None:
        """Clamp the per-epoch index stream to exactly `n` indices (n must be a
        multiple of batch_size). See `_max_samples`."""
        self._max_samples = None if n is None else int(n)

    def _get_lengths(self) -> List[int]:
        if self._lengths_cache is None:
            self._lengths_cache = [int(self._length_fn(i)) for i in range(len(self.dataset))]
        return self._lengths_cache

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        if self.bad_indices:
            bad = self.bad_indices
            indices = [i for i in indices if i not in bad]

        if self.shuffle and n > 1:
            lengths = self._get_lengths()
            bs = self.bucket_size
            for i in range(0, len(indices), bs):
                chunk = indices[i : i + bs]
                chunk.sort(key=lambda idx: lengths[idx])
                indices[i : i + bs] = chunk

        # ---- EVEN per-rank batch counts (epoch-boundary desync fix) ----------
        # Truncate to a multiple of (batch_size * world_size) so the number of
        # global batches divides evenly across ranks. Otherwise the remainder
        # (e.g. 550102 batches / 360 ranks → 22 ranks get 1 extra batch) makes
        # those ranks cross the epoch boundary one optimizer-step LATER than the
        # rest; the two groups then issue mismatched gradient all-reduce calls
        # and NCCL deadlocks → heartbeat watchdog kill → restart loop. Dropping
        # the <= (batch_size*world_size - 1) tail samples per epoch (~0.07%) is
        # deterministic across ranks (same n/seed/world_size) so it introduces
        # no new desync. Must match __len__ below.
        _round = self.batch_size * self.world_size
        if _round > 1:
            keep = (len(indices) // _round) * _round
            indices = indices[:keep]

        # Hard cap to a globally-agreed batch count (per-rank-shard path).
        if self._max_samples is not None and len(indices) > self._max_samples:
            indices = indices[: self._max_samples]

        return iter(indices)

    def __len__(self) -> int:
        # Must match the truncation in __iter__ so max_train_steps / per-epoch
        # step counts are computed from the real (even) batch count.
        n = len(self.dataset) - len(self.bad_indices)
        _round = self.batch_size * self.world_size
        if _round > 1:
            n = (n // _round) * _round
        if self._max_samples is not None:
            n = min(n, self._max_samples)
        return n


def estimate_sample_length(item: dict, tokens_per_gen_image: int = 256) -> int:
    """Cheap length estimate for an already-processed dataset item.

    Pass this (or a custom faster estimator that doesn't materialize the sample)
    as `length_fn` to LengthBucketedSampler.
    """
    ids = item.get("input_ids")
    if ids is None:
        return 0
    if hasattr(ids, "shape"):
        return int(ids.shape[0])
    return len(ids)


def estimate_record_length(record: dict, tokens_per_gen_image: int = 256,
                           tokens_per_ctx_image: int = 144) -> int:
    """Cheap length estimate from a RAW (unprocessed) `messages` record.

    Unlike estimate_sample_length (which needs the materialized sample's
    input_ids), this reads only the messages schema — text char counts and
    image counts — so it NEVER loads or decodes an image. This is the length_fn
    to use at scale: LengthBucketedSampler computes lengths over every index up
    front, so `estimate_sample_length(dataset[i])` would re-run the full image
    pipeline over the ENTIRE corpus before step 0 (fine at the 100-sample
    overfit scale, prohibitive at millions of map-style samples — it hangs the
    run on length precompute).

    The estimate only needs good relative ordering for bucketing; the collator's
    max_total_tokens governs actual packing. ~4 chars/token is the usual rough
    BPE ratio; image token counts match the 256×144 frame budget (144 SigLIP
    context tokens / 256 generation-latent tokens).
    """
    n = 0
    for msg in record.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            n += len(content) // 4
            continue
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict):
                continue
            t = c.get("type")
            if t == "text":
                n += len(c.get("text", "")) // 4 + 2
            elif t == "image":
                n += tokens_per_ctx_image
            elif t in ("generated_image", "raw_image"):
                n += tokens_per_gen_image
    return max(1, n)


class LengthSortedChunkBuffer(torch.utils.data.IterableDataset):
    """Wrap an IterableDataset so consecutive samples have similar lengths.

    ============================================================
    WARNING — DO NOT USE WITH `accelerator.prepare(dataloader)`
    ============================================================
    Accelerate's `IterableDatasetShard` slices the wrapped stream into
    contiguous `batch_size × num_processes` chunks and gives each rank a
    fixed contiguous slice. Composed with chunk-sort this means every rank
    permanently lands on the SAME relative position in each sorted chunk —
    rank 0 always gets the shortest 1/W band, rank (W-1) always gets the
    longest. Empirically this BLOWS UP cross-rank max/min from ~1.22 to
    ~3.7× at chunk_multiplier=8 (see smoke_chunk_buffer_crossrank.py).

    This class is kept here only as a building block for a future correct
    cross-rank length balancer (which would also need either:
      (a) per-rank independent data streams (e.g. rank-shard happens
          BEFORE this wrapper), or
      (b) a coordinated permutation of sorted samples so each rank's
          contiguous slice contains a representative length distribution).

    Within-rank uses (single-rank training, or unit tests on independent
    streams) are still fine — the class itself is correct.
    """

    def __init__(
        self,
        base: "torch.utils.data.IterableDataset",
        chunk_size: int,
        length_fn=estimate_sample_length,
    ):
        super().__init__()
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        self.base = base
        self.chunk_size = int(chunk_size)
        self.length_fn = length_fn

    def __iter__(self):
        buf = []
        for item in self.base:
            buf.append(item)
            if len(buf) >= self.chunk_size:
                buf.sort(key=self.length_fn)
                for x in buf:
                    yield x
                buf = []
        if buf:
            buf.sort(key=self.length_fn)
            for x in buf:
                yield x

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.base)


__all__ = [
    "LengthBucketedSampler",
    "LengthSortedChunkBuffer",
    "LengthBalancedShuffleBuffer",
    "RatioMixIterableDataset",
    "ChainIterableDataset",
    "estimate_sample_length",
    "estimate_record_length",
]


class LengthBalancedShuffleBuffer(torch.utils.data.IterableDataset):
    """Cross-rank length-balanced re-shuffle for IterableDataset + IDS.

    Designed for the exact production path:
      ParquetT2IDataset (IterableDataset, global stream)
        → LengthBalancedShuffleBuffer  ← THIS CLASS
        → accelerate IterableDatasetShard (contiguous slice per rank)
        → DataLoader → collator

    What it does
    ------------
    Buffer exactly `batch_size * num_processes` samples (one IDS pool),
    sort by length, then yield them in a permuted order such that the
    contiguous slice `pool[r*B : (r+1)*B]` that IDS gives to rank r is the
    "stride-W, offset-r" sub-sequence of the sorted chunk:

        pool[r*B + b]  =  sort[b*W + r]

    Each rank thus receives B samples covering the full length range of
    the sorted chunk — so the per-rank batch's TOTAL token count (which
    is what determines forward/backward time) is nearly equalized across
    ranks. Within-rank length variance is high, but packing handles that
    naturally: the collator just concatenates the B samples regardless of
    individual lengths.

    Expected effect (validated by smoke_chunk_buffer_crossrank.py):
        baseline cross-rank max/min ~ 1.22
        with this wrapper          ~ 1.05–1.10

    Trailing residual (< chunk_size at end of stream) is yielded in plain
    sorted order — those final few batches won't be balanced, but that's
    a tiny tail edge case.

    Notes
    -----
    * `batch_size` here must equal `per_device_train_batch_size`. Do NOT
      pass a global batch size.
    * `num_processes` must equal `accelerator.num_processes`. Pass it
      explicitly from train/finetune.py.
    * Each DataLoader worker holds its own buffer (size = B * W samples).
      In production W=200, B=25 → buffer = 5000 samples ≈ tens of MB per
      worker. With num_workers=6 that's a few hundred MB extra RAM, fine
      on a node.
    * If chunk_size is not divisible by num_processes the math breaks; we
      hard-error on construction.
    """

    def __init__(
        self,
        base: "torch.utils.data.IterableDataset",
        batch_size: int,
        num_processes: int,
        length_fn=estimate_sample_length,
    ):
        super().__init__()
        if batch_size <= 0 or num_processes <= 0:
            raise ValueError(
                f"batch_size and num_processes must be > 0, "
                f"got batch_size={batch_size} num_processes={num_processes}"
            )
        self.base = base
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.length_fn = length_fn
        self.chunk_size = self.batch_size * self.num_processes

    def __iter__(self):
        B = self.batch_size
        W = self.num_processes
        chunk = self.chunk_size
        buf = []
        for item in self.base:
            buf.append(item)
            if len(buf) == chunk:
                buf.sort(key=self.length_fn)
                # Re-arrange: pool[r*B + b] = sort[b*W + r]
                # Yield in r-major, b-minor order.
                for r in range(W):
                    for b in range(B):
                        yield buf[b * W + r]
                buf = []
        # Flush — just yield remaining in sorted order. Bal. doesn't apply
        # for an incomplete final chunk.
        if buf:
            buf.sort(key=self.length_fn)
            for x in buf:
                yield x

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.base)


def _largest_remainder_schedule(ratios: List[int]) -> List[int]:
    """Build a length-`sum(ratios)` interleave pattern (a list of child indices)
    where child i appears exactly `ratios[i]` times, spaced as evenly as possible
    via the Hamilton / largest-remainder method.

    Every consecutive `sum(ratios)` window of the (cyclically repeated) pattern
    therefore carries the EXACT target ratio. Examples:
        [3, 1]    -> [0, 0, 1, 0]      (minority spaced mid-period, not clumped)
        [1, 1]    -> [0, 1]
        [5, 3, 2] -> 10-long pattern with counts 5/3/2
    Pure function of `ratios` (no rank/worker/epoch state) so every worker
    produces an identically-proportioned stream.
    """
    if not ratios or any(int(r) <= 0 for r in ratios):
        raise ValueError(f"ratios must be non-empty positive ints, got {ratios}")
    p = sum(ratios)
    emitted = [0] * len(ratios)
    out: List[int] = []
    for slot in range(p):
        # Pick the child furthest BEHIND its ideal cumulative quota after this slot.
        best_i, best_deficit = 0, None
        for i, r in enumerate(ratios):
            deficit = r * (slot + 1) / p - emitted[i]
            if best_deficit is None or deficit > best_deficit:
                best_i, best_deficit = i, deficit
        out.append(best_i)
        emitted[best_i] += 1
    return out


class RatioMixIterableDataset(torch.utils.data.IterableDataset):
    """Interleave N child IterableDatasets at a fixed integer ratio, emitting a
    single GLOBAL stream that is period-`p`-exact (p = sum(ratios)).

    Production placement:
        [ParquetT2IDataset, StreamingUnifiedDataset, ...]
          -> RatioMixIterableDataset   <- THIS CLASS  (period-p exact per worker)
          -> LengthBalancedShuffleBuffer (buffers B*W, permutes within window)
          -> accelerate IterableDatasetShard (rank r gets [r*B:(r+1)*B])
          -> DataLoader -> UnifiedDataCollator

    Because the schedule is a pure function of `ratios` (no rank/worker state)
    and each child shards by WORKER internally, every worker's mixed stream is
    period-p exact. Provided `p | (B*W)` (validated by the caller), each B*W
    window LBSB+IDS see is exactly proportioned, so every optimizer step's GLOBAL
    batch carries the exact ratio.

    Length semantics: the `length_source` child (default the dominant one)
    defines one epoch — when its iterator exhausts, iteration stops. All OTHER
    children CYCLE (re-iterate, re-shuffled) so the ratio holds regardless of
    size. With `infinite=True` even the length_source cycles, so __iter__ NEVER
    stops — used by the self-sharded (no-IDS) mix path, where the run is bounded
    solely by `max_train_steps` (a synchronized break on every rank), so there is
    NO data-driven epoch boundary and hence no cross-rank desync from per-rank
    shard sizes differing by ±1 record. `__len__` is still the dominant-driven
    estimate (used to compute max_train_steps).
    """

    def __init__(
        self,
        datasets: List["torch.utils.data.IterableDataset"],
        ratios: List[int],
        length_source: int = 0,
        seed: int = 42,
        infinite: bool = False,
    ):
        super().__init__()
        if not datasets or len(datasets) != len(ratios):
            raise ValueError(
                f"datasets and ratios must be non-empty and equal length, "
                f"got {len(datasets)} datasets / {len(ratios)} ratios"
            )
        if any(int(r) <= 0 for r in ratios):
            raise ValueError(f"ratios must be positive ints, got {ratios}")
        if not (0 <= int(length_source) < len(datasets)):
            raise ValueError(f"length_source {length_source} out of range")
        self.datasets = list(datasets)
        self.ratios = [int(r) for r in ratios]
        self.period = sum(self.ratios)
        self.length_source = int(length_source)
        self.seed = int(seed)
        self.infinite = bool(infinite)
        self._epoch = 0
        self._schedule = _largest_remainder_schedule(self.ratios)

    @property
    def schedule(self) -> List[int]:
        return self._schedule

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        for d in self.datasets:
            if hasattr(d, "set_epoch"):
                d.set_epoch(epoch)

    def __len__(self) -> int:
        # Effective GLOBAL length: the dominant source emits len(dom) samples,
        # i.e. ratios[ls] per period -> (len(dom)//ratios[ls]) periods, each
        # contributing `period` mixed samples.
        len_dom = len(self.datasets[self.length_source])
        r_dom = self.ratios[self.length_source]
        return (len_dom // r_dom) * self.period

    def __iter__(self):
        iters = [iter(d) for d in self.datasets]
        sched = self._schedule
        # Children whose per-worker shard is empty — stop retrying them so an
        # over-sharded minority (more workers than records) can't busy-loop.
        dead = [False] * len(self.datasets)
        # Per-child cycle counter. When a non-dominant source is much smaller than
        # the dominant it is consumed MANY times within one epoch (e.g. a 22M
        # interleave set vs a 200M T2I dominant cycles ~9x). Re-seed each cycle so
        # the repeats are RE-SHUFFLED rather than identical replays of the same
        # within-epoch order. Reset every epoch by set_epoch (called by the loop).
        cycles = [0] * len(self.datasets)
        pos = 0
        while True:
            child = sched[pos % self.period]
            pos += 1
            if dead[child]:
                continue
            try:
                yield next(iters[child])
            except StopIteration:
                if child == self.length_source and not self.infinite:
                    return  # dominant exhausted -> end of epoch (finite mode)
                # Cycle this child: re-shuffle (bump epoch) then rebuild its
                # (per-worker) iterator and retry once. In infinite mode the
                # length_source cycles here too, so __iter__ never stops.
                cycles[child] += 1
                d = self.datasets[child]
                if hasattr(d, "set_epoch"):
                    d.set_epoch(self._epoch * 1_000_003 + cycles[child])
                iters[child] = iter(d)
                try:
                    yield next(iters[child])
                except StopIteration:
                    dead[child] = True  # genuinely empty shard -> skip its slots
                    if all(dead):
                        return  # nothing left to yield (avoids an infinite busy-loop)


class ChainIterableDataset(torch.utils.data.IterableDataset):
    """Chain N iterable datasets into ONE logical source: yields all of dataset 0,
    then dataset 1, ... — while FORWARDING set_epoch / set_resume_skip_per_worker /
    __len__ to the children. (torch.utils.data.ChainDataset forwards NONE of these,
    so it can't carry per-epoch reshuffle or fast-resume.)

    Used to GROUP several T2I parquet dirs (e.g. laion + coyo) into a single ratio
    entry for RatioMixIterableDataset, so the mix ratio is expressed cleanly as
    `T2I : interleave` rather than per-parquet-dir. Each child still shards by
    WORKER internally; chaining just concatenates their (worker-sharded) streams,
    so the combined source is still a valid GLOBAL stream for IterableDatasetShard.
    """

    def __init__(self, datasets: List["torch.utils.data.IterableDataset"]):
        super().__init__()
        if not datasets:
            raise ValueError("ChainIterableDataset needs at least one child")
        self.datasets = list(datasets)

    def set_epoch(self, epoch: int) -> None:
        for d in self.datasets:
            if hasattr(d, "set_epoch"):
                d.set_epoch(epoch)

    def set_resume_skip_per_worker(self, skip_rows_per_worker: int) -> None:
        # Best-effort: distribute the per-worker skip across children by their
        # __len__ proportion and forward to each child's fast-skip. Precise
        # cross-child resume is not attempted (consistent with the mix path's
        # best-effort resume) — the total skipped row COUNT is ~correct.
        n = max(0, int(skip_rows_per_worker))
        total = sum(len(d) for d in self.datasets) or 1
        for d in self.datasets:
            if hasattr(d, "set_resume_skip_per_worker"):
                d.set_resume_skip_per_worker(n * len(d) // total)

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets)

    def __iter__(self):
        for d in self.datasets:
            yield from d
