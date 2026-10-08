"""Block-based replay. Only PyTorch and the Python standard library are required."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import random
import threading
import time

import torch


from .buffer_types import ReplayBatch
from .device_pool import DeviceBlockPool


class BlockReplayBuffer:
    """Single producer/learner replay with independent disk and RAM capacities.

    Append snapshots inputs. Completed blocks become sampleable after atomic disk
    publication. RAM holds a randomly refreshed subset of retained disk blocks;
    batches sample transitions uniformly within that subset. Disk retention is
    rounded down to complete blocks. Call flush to publish a partial final block.
    """

    def __init__(self, directory, *, capacity, block_size, ram_blocks, batch_size,
                 device, prefetch=4, refresh_every=8, max_pending_writes=4,
                 seed=0, gpu_cache_bytes=0, device_chunk_size=4096,
                 device_refresh_every=8, device_refresh_chunks=8):
        if min(capacity, block_size, ram_blocks, batch_size, prefetch,
               refresh_every, max_pending_writes) < 1 or capacity < block_size:
            raise ValueError("Positive budgets and capacity >= block_size required")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError(f"Replay directory is not empty: {self.directory}")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices are CPU, CUDA and ROCm")
        if gpu_cache_bytes < 0 or min(device_chunk_size, device_refresh_every, device_refresh_chunks) < 1:
            raise ValueError("Invalid GPU chunk pool budgets")
        if gpu_cache_bytes and self.device.type != "cuda":
            raise ValueError("GPU chunk pool requires CUDA or ROCm")
        self.gpu_cache_bytes = gpu_cache_bytes
        self.device_chunk_size = device_chunk_size
        self.device_refresh_every = device_refresh_every
        self.device_refresh_chunks = device_refresh_chunks
        self._device_pool = None
        self._seed = seed
        self.capacity, self.block_size = capacity, block_size
        self.disk_blocks = capacity // block_size
        self.ram_blocks = min(ram_blocks, self.disk_blocks)
        self.batch_size, self.prefetch = batch_size, prefetch
        self.refresh_every, self.max_pending_writes = refresh_every, max_pending_writes
        self._rng = random.Random(seed)
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self._lock = threading.RLock()
        self._writer = ThreadPoolExecutor(1, thread_name_prefix="block-replay-write")
        self._loader = ThreadPoolExecutor(1, thread_name_prefix="block-replay-load")
        self._sampler = ThreadPoolExecutor(1, thread_name_prefix="block-replay-sample")
        self._refresh_future = None
        self._stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._pending, self._samples, self._inflight = deque(), deque(), deque()
        self._blocks, self._cache = {}, {}
        self._building, self._used, self._next = None, 0, 0
        self._schema = None
        self._closed = False
        self._sample_count = 0
        self._reads = self._writes = 0
        self.sample_wait_seconds = self.write_wait_seconds = 0.0

    def __len__(self):
        with self._lock:
            return sum(count for _, count in self._blocks.values())

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Replay is closed")

    def append_async(self, batch):
        self._check_open()
        if not batch:
            raise ValueError("Empty transition mapping")
        schema = {k: (tuple(v.shape[1:]), v.dtype) for k, v in batch.items()}
        count = next(iter(batch.values())).shape[0]
        if not count or any(v.shape[0] != count or v.device != self.device for v in batch.values()):
            raise ValueError("Fields must have equal nonzero leading size on the configured device")
        if self._schema is None:
            self._schema = schema
            self.row_bytes = sum(math.prod(s) * torch.empty((), dtype=d, device="cpu").element_size()
                                 for s, d in schema.values())
            (self.directory / "schema.json").write_text(json.dumps({
                "format": "block-replay-v1", "block_size": self.block_size,
                "disk_blocks": self.disk_blocks, "ram_blocks": self.ram_blocks,
                "schema": {k: {"shape": s, "dtype": str(d)} for k, (s, d) in schema.items()}
            }, indent=2))
            if self.gpu_cache_bytes:
                self._device_pool = DeviceBlockPool(schema, device=self.device,
                    row_bytes=self.row_bytes, capacity=self.capacity, chunk_size=self.device_chunk_size,
                    cache_bytes=self.gpu_cache_bytes, refresh_chunks=self.device_refresh_chunks,
                    seed=self._seed)
        elif schema != self._schema:
            raise ValueError("Transition schema changed")
        while self._pending and self._pending[0].done():
            self._pending.popleft().result()
        if len(self._pending) >= self.max_pending_writes:
            started = time.perf_counter()
            self._pending.popleft().result()
            self.write_wait_seconds += time.perf_counter() - started
        snapshot = {k: v.detach().clone() for k, v in batch.items()}
        event = None
        if self._stream is not None:
            # Capture the producer before entering the transfer stream.
            producer = torch.cuda.current_stream(self.device)
            with torch.cuda.stream(self._stream):
                self._stream.wait_stream(producer)
                source = {k: torch.empty_like(v, device="cpu", pin_memory=True).copy_(v, non_blocking=True)
                          for k, v in snapshot.items()}
                for v in snapshot.values():
                    v.record_stream(self._stream)
                event = self._stream.record_event()
        else:
            source = snapshot

        def commit():
            if event is not None:
                from .cuda_gate import background_cuda_work
                with background_cuda_work():
                    event.synchronize()
            offset = 0
            while offset < count:
                if self._building is None:
                    self._building = {k: torch.empty((self.block_size, *s), dtype=d, device="cpu")
                                      for k, (s, d) in self._schema.items()}
                take = min(count - offset, self.block_size - self._used)
                for k, v in source.items():
                    self._building[k][self._used:self._used + take].copy_(v[offset:offset + take])
                self._used += take
                offset += take
                if self._used == self.block_size:
                    self._publish()
        self._pending.append(self._writer.submit(commit))

    def _publish(self):
        index, count = self._next, self._used
        data = {k: v[:count].clone() if count < self.block_size else v for k, v in self._building.items()}
        path = self.directory / f"block_{index:012d}.pt"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb", buffering=8 << 20) as stream:
            torch.save(data, stream)
        temporary.replace(path)
        with self._lock:
            self._blocks[index] = (path, count)
            self._writes += 1
            while len(self._blocks) > self.disk_blocks:
                oldest = next(iter(self._blocks))
                old_path, _ = self._blocks.pop(oldest)
                self._cache.pop(oldest, None)
                old_path.unlink()
            if self._device_pool is not None:
                self._device_pool.prune_before(next(iter(self._blocks)))
            # Reuse immutable collection data, avoiding a disk read after writing.
            # Reservoir admission prevents every new block displacing old history.
            if len(self._cache) < self.ram_blocks:
                self._cache[index] = data
            elif self._rng.random() < self.ram_blocks / len(self._blocks):
                del self._cache[self._rng.choice(list(self._cache))]
                self._cache[index] = data
        self._next += 1
        self._building, self._used = None, 0

    def _request_refresh(self):
        # At most one disk refresh is pending. Never queue unbounded I/O.
        if self._refresh_future is not None:
            if not self._refresh_future.done():
                return
            self._refresh_future.result()  # Propagate background I/O failures.
        self._refresh_future = self._loader.submit(self._refresh)

    def _refresh(self):
        with self._lock:
            if not self._blocks:
                return
            # One randomly selected block per request; sampling continues while
            # it loads. Cached selections need no network operation.
            index = self._rng.choice(list(self._blocks))
            if index in self._cache:
                return
        self._load(index)

    def _load(self, index):
        # Open under the metadata lock, then release it during network I/O.
        # The open file remains readable if retention unlinks its pathname.
        with self._lock:
            if index not in self._blocks:
                return
            stream = self._blocks[index][0].open("rb", buffering=8 << 20)
        with stream:
            data = torch.load(stream, map_location="cpu", weights_only=True)
        with self._lock:
            self._reads += 1
            if index not in self._blocks:
                return  # It aged out during the read.
            if len(self._cache) >= self.ram_blocks and index not in self._cache:
                del self._cache[self._rng.choice(list(self._cache))]
            self._cache[index] = data

    def _sample(self):
        if self._sample_count % self.refresh_every == 0:
            self._request_refresh()
        if self._device_pool is not None:
            if self._sample_count % self.device_refresh_every == 0:
                with self._lock:
                    entries = [(i, data, self._blocks[i][1]) for i, data in self._cache.items()]
                self._device_pool.request_refresh(entries)
            while self._device_pool.empty:
                with self._lock:
                    entries = [(i, data, self._blocks[i][1]) for i, data in self._cache.items()]
                if not entries:
                    raise RuntimeError("No published replay blocks; append enough rows or flush first")
                self._device_pool.request_refresh(entries)
                self._device_pool.wait_refresh()  # Initial/fully-pruned pool only.
            self._sample_count += 1
            return self._device_pool.sample(self.batch_size)
        self._sample_count += 1
        with self._lock:
            if not self._cache:
                raise RuntimeError("No published replay blocks; append enough rows or flush first")
            # Retain immutable references so pruning/replacement cannot invalidate
            # this batch. CPU gather and device copy do not hold the metadata lock.
            entries = [(data, self._blocks[i][1]) for i, data in self._cache.items()]
        sizes = torch.tensor([n for _, n in entries], device="cpu", dtype=torch.int64)
        ends = sizes.cumsum(0)
        positions = torch.randint(int(ends[-1]), (self.batch_size,),
                                  generator=self._generator, device="cpu")
        block_ids = torch.searchsorted(ends, positions, right=True)
        source = {k: torch.empty((self.batch_size, *shape), dtype=dtype, device="cpu",
                                pin_memory=self._stream is not None)
                  for k, (shape, dtype) in self._schema.items()}
        # Visit only blocks actually selected by this batch, not the entire pool.
        order = block_ids.argsort()
        unique, counts = torch.unique_consecutive(block_ids[order], return_counts=True)
        offset = 0
        for block_id, count in zip(unique.tolist(), counts.tolist()):
            selected = order[offset:offset + count]
            start = 0 if block_id == 0 else ends[block_id - 1]
            local = positions[selected] - start
            data = entries[block_id][0]
            for key in source:
                source[key][selected] = data[key][local]
            offset += count
        if self._stream is None:
            return ReplayBatch(source)
        with torch.cuda.stream(self._stream):
            data = {k: v.to(self.device, non_blocking=True) for k, v in source.items()}
            event = self._stream.record_event()
        return ReplayBatch(data, event, source)

    def sample_async(self):
        self._check_open()
        while self._inflight and self._inflight[0]._event.query():
            self._inflight.popleft()
        while len(self._samples) < self.prefetch:
            self._samples.append(self._sampler.submit(self._sample))
        started = time.perf_counter()
        result = self._samples.popleft().result()
        self.sample_wait_seconds += time.perf_counter() - started
        if result._event is not None:
            self._inflight.append(result)
        return result

    def stats(self):
        with self._lock:
            result = {"disk_transitions": len(self), "disk_blocks": len(self._blocks),
                    "ram_blocks": len(self._cache), "blocks_read": self._reads,
                    "blocks_written": self._writes,
                    "refresh_pending": int(self._refresh_future is not None and not self._refresh_future.done()),
                    "ram_cache_bytes": sum(self._blocks[i][1] for i in self._cache) * getattr(self, "row_bytes", 0),
                    "sample_wait_seconds": self.sample_wait_seconds,
                    "write_wait_seconds": self.write_wait_seconds}
            if self._device_pool is not None:
                result.update(self._device_pool.stats())
            return result

    def flush(self):
        self._check_open()
        while self._pending:
            self._pending.popleft().result()
        if self._used:
            self._writer.submit(self._publish).result()

    def close(self):
        if not self._closed:
            try:
                self._sampler.shutdown(wait=True)
                self._loader.shutdown(wait=True)
                if self._refresh_future is not None:
                    self._refresh_future.result()
                self.flush()
                if self._device_pool is not None:
                    self._device_pool.close()
                if self._stream is not None:
                    self._stream.synchronize()
            finally:
                self._closed = True
                if self._device_pool is not None:
                    self._device_pool.close()
                self._loader.shutdown(wait=True)
                self._writer.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
