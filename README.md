# torch-block-replay

A reusable disk → RAM → device replay buffer. Requires Python 3.10+ and PyTorch;
no Genesis, TensorDict, TorchRL, trainer, or simulator imports. It supports CPU,
CUDA and ROCm (through PyTorch's `cuda` API).

Install from GitHub (with access to this private repository):

```sh
pip install git+ssh://git@github.com/LuoShaoyinn/torch-block-replay.git
```

For local development:

```sh
pip install -e .
```

Install the appropriate CPU, CUDA, or ROCm PyTorch build for your machine first.
The installed import is `from block_replay import BlockReplayBuffer`.

```python
from block_replay import BlockReplayBuffer

with BlockReplayBuffer(
    "/data/replay/run-001",
    capacity=100_000_000,    # Disk transitions, independent of RAM size
    block_size=131_072,     # Rows per sequential disk block
    ram_blocks=32,         # Randomly selected RAM blocks
    batch_size=4096,       # Rows per device batch
    device="cuda:0",
    prefetch=8,            # Prepared device batches
    refresh_every=4,       # Refresh one random block every four prepared batches
    max_pending_writes=4,  # Bounded append queue
    gpu_cache_bytes=8 << 30, # Optional GPU chunk pool; 0 keeps CPU batch staging
    device_chunk_size=4096,
    device_refresh_every=8,
    device_refresh_chunks=8,
    seed=42,
) as replay:
    replay.append_async({"obs": obs, "action": action, "reward": reward})
    # Warmup must publish at least one block before sampling. flush() also
    # publishes a partial block; use at warmup/end, not every training step.
    replay.flush()
    batch = replay.sample_async().wait()
    # Request the next batch before computing on this one for overlap.
```

All fields have a common nonzero leading transition dimension and live on the
configured device. Shapes/dtypes must remain constant. Appends snapshot their
inputs; input storage can immediately be reused. One producer and one learner
are supported. Background errors propagate when waiting for queued work.

## Tiers and budgets

- Disk writes complete blocks sequentially, using buffered writes and atomic
  rename. Blocks are FIFO-pruned once the configured disk block count is reached.
- RAM caches a random subset of the retained blocks. New blocks reuse their
  immutable collection data through reservoir admission, avoiding a disk reread.
  An independent loader gradually replaces randomly chosen cache entries with
  disk blocks. Selecting an already cached block is a no-op. Linux may also cache files.
- With `gpu_cache_bytes > 0`, a separate uploader chooses contiguous chunks from
  random RAM blocks and transfers them into fixed GPU slots. A shared ordered
  stream protects chunk replacement and minibatch gathers. The sampler generates
  random indices and gathers complete minibatches on GPU, avoiding CPU per-block
  batch construction. Pinned sources remain alive until copies finish.
- GPU capacity rounds down to complete `device_chunk_size` slots. Slots are not
  sampleable until populated. Startup or complete retention invalidation waits
  for priming; normal sampling continues during refresh. The GPU budget excludes
  prefetched output batches, simulator and learner allocations.
- With `gpu_cache_bytes=0`, CPU minibatch staging remains the explicitly selected
  mode (and supports CPU-only projects). It is not an automatic failure fallback.

RAM cache budget is roughly `ram_blocks × block_size × row_bytes`. Allow extra
RAM for the block being built, bounded append staging, sampled pinned batches,
and serialization. Disk capacity rounds down to complete blocks; a partial flush
occupies one block slot. Keep blocks large enough for CIFS sequential throughput,
but small enough that replacing one does not take excessive time.

Only one disk refresh may be pending; a slow read leaves the existing RAM pool
usable. Loaded blocks are installed atomically once ready. Disk refresh does not
block batch preparation. Write queues still apply backpressure if storage cannot
keep up with collection.

The GPU uploader has at most one pending refresh. `device_refresh_every` counts
prepared minibatches; `device_refresh_chunks` caps each refresh. Start with disk
blocks of 32,768 rows and GPU chunks of 4,096, then compare 16,384-row chunks on
the actual machine. Sampling does not wait for a routine GPU refresh.

Lower `refresh_every` gives faster coverage of disk history and more I/O. Higher
values reuse RAM longer. Increasing `prefetch` hides latency at the cost of VRAM
and additional sample age. If disk writes or reads cannot keep up, bounded queues
apply backpressure rather than dropping new data or silently changing sampling.

## Sampling semantics

Transitions are uniform **within the current GPU pool** when enabled, otherwise
within the current RAM pool. GPU chunks are chosen randomly from RAM and replaced
gradually; this adds another cache correlation timescale, so keep many chunks
and refresh them regularly. RAM cache replacement does not invalidate a still
retained GPU chunk; disk pruning does.

For the RAM pool: Full, equal-size disk
blocks have equal inclusion probability at steady state, so sampling is uniform
across those blocks in expectation. Consecutive batches remain correlated while
the pool persists; new or recently rotated blocks need time to enter the pool.
Partial blocks are weighted by valid row count within RAM, but differing block
sizes mean exact global transition uniformity is not guaranteed. This is an
intentional block-cache sampler, not an exact full-disk random sampler. The same
partial-chunk caveat applies to GPU admission.

Only published blocks count in `len(replay)`; an incomplete append block is not
sampleable until full or flushed. Prefetched batches are immutable and may
contain rows from a block subsequently pruned from disk. Metrics returned by
`stats()` are CPU counters and require no device tensor readback.

Directories must be new and nonempty directories are rejected. Each instance
owns its directory; use separate directories per distributed rank. Existing
files are retained on close, but reopening/resuming replay is not implemented.
This component does not configure CIFS mounts or provide multi-process locking.

## Verification

From this repository after installation:

```sh
python -m unittest discover -s tests -v
```

Tests cover immutable appends, bounded disk history larger than RAM, rotation
coverage, partial blocks, concurrent collection/prefetch, continued sampling during an
artificially blocked disk refresh, zero rereads while all blocks fit RAM, GPU chunk
replacement/retention/partial chunks, and GPU transfers under
Genesis-style default CUDA device placement. CUDA/ROCm tests skip if unavailable.
CIFS throughput must be measured on the deployment machine separately.

## Origin

Extracted from `infra/block_replay` in
[LuoShaoyinn/srt2](https://github.com/LuoShaoyinn/srt2) at commit `8e7afbcc454e3d392ab6c4db480687c58798ff63`.
Core behavior is unchanged; one trailing blank line was removed. Tests and the benchmark only change
their imports to use the standalone package. No training code or simulator
dependencies are included. No open-source license has been assigned.
