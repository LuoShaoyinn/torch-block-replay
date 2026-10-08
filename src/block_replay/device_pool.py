"""Bounded GPU chunk cache; all sampling and mutation share an ordered stream."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import bisect
import random
import threading

import torch

from .buffer_types import ReplayBatch
from .cuda_gate import background_cuda_work, guarded_cuda_work


class DeviceBlockPool:
    def __init__(self, schema, *, device, row_bytes, capacity, chunk_size,
                 cache_bytes, refresh_chunks, seed):
        self.device, self.schema, self.chunk_size = device, schema, chunk_size
        self.slots = min(cache_bytes // (row_bytes * chunk_size),
                         (capacity + chunk_size - 1) // chunk_size)
        if self.slots < 1:
            raise ValueError("GPU cache budget must hold at least one complete transfer chunk")
        self.refresh_chunks = refresh_chunks
        self._rng = random.Random(seed)
        self._generator = torch.Generator(device=device).manual_seed(seed)
        self._lock = threading.RLock()
        self._stream = torch.cuda.Stream(device=device)
        self._worker = ThreadPoolExecutor(1, thread_name_prefix="block-replay-device")
        self._future = None
        self._tokens, self._counts, self._lookup = [None] * self.slots, [0] * self.slots, {}
        self._oldest = 0
        self._rows = 0
        self._closed = False
        self._metadata_sources = deque()
        self.chunks_copied = self.bytes_copied = 0
        with torch.cuda.stream(self._stream):
            self._data = {k: torch.empty((self.slots * chunk_size, *shape), dtype=dtype, device=device)
                          for k, (shape, dtype) in schema.items()}
            self._ends = torch.zeros(self.slots, dtype=torch.int64, device=device)
            self._starts = torch.arange(self.slots, device=device, dtype=torch.int64) * chunk_size
        self.row_bytes = row_bytes
        self.cache_bytes = self.slots * chunk_size * row_bytes

    def _metadata(self):
        counts = torch.tensor(self._counts, device="cpu", dtype=torch.int64)
        ends = counts.cumsum(0)
        self._rows = sum(self._counts)
        pinned_ends = ends.pin_memory()
        pinned_starts = (torch.arange(self.slots, device="cpu", dtype=torch.int64) * self.chunk_size
                         - (ends - counts)).pin_memory()
        self._ends.copy_(pinned_ends, non_blocking=True)
        self._starts.copy_(pinned_starts, non_blocking=True)
        while self._metadata_sources and self._metadata_sources[0][0].query():
            self._metadata_sources.popleft()
        self._metadata_sources.append((self._stream.record_event(), pinned_ends, pinned_starts))

    def request_refresh(self, entries):
        if self._future is not None:
            if not self._future.done():
                return
            self._future.result()
        self._future = self._worker.submit(self._refresh, entries)

    @guarded_cuda_work
    def _refresh(self, entries):
        if not entries:
            return
        ends, total = [], 0
        for _, _, count in entries:
            total += count
            ends.append(total)
        copied = []
        # Limit attempts too: a full cache may already contain all RAM chunks.
        for _ in range(self.refresh_chunks * 4):
            row = self._rng.randrange(total)
            index = bisect.bisect_right(ends, row)
            block_id, data, count = entries[index]
            local = row - (0 if index == 0 else ends[index - 1])
            offset = local // self.chunk_size * self.chunk_size
            token = (block_id, offset)
            length = min(self.chunk_size, count - offset)
            with self._lock:
                if block_id < self._oldest or token in self._lookup:
                    continue
            pinned = {k: v[offset:offset + length].pin_memory() for k, v in data.items()}
            with self._lock, torch.cuda.stream(self._stream):
                if block_id < self._oldest or token in self._lookup:
                    continue
                if 0 in self._counts:
                    slot = self._counts.index(0)
                else:
                    slot = self._rng.randrange(self.slots)
                    del self._lookup[self._tokens[slot]]
                for key in self._data:
                    self._data[key][slot * self.chunk_size:slot * self.chunk_size + length].copy_(
                        pinned[key], non_blocking=True)
                self._tokens[slot], self._counts[slot] = token, length
                self._lookup[token] = slot
                copied.append(pinned)
                self.chunks_copied += 1
                self.bytes_copied += length * self.row_bytes
                # Update metadata before unlocking: sampling can interleave with
                # upload, and must see data/counts from the same stream order.
                self._metadata()
            if len(copied) == self.refresh_chunks:
                break
        if copied:
            with self._lock, torch.cuda.stream(self._stream):
                event = self._stream.record_event()
            event.synchronize()  # Only uploader waits; retains pinned sources.

    @guarded_cuda_work
    def prune_before(self, oldest):
        with self._lock, torch.cuda.stream(self._stream):
            self._oldest = oldest
            changed = False
            for slot, token in enumerate(self._tokens):
                if token is not None and token[0] < oldest:
                    self._lookup.pop(token)
                    self._tokens[slot], self._counts[slot] = None, 0
                    changed = True
            if changed:
                self._metadata()

    @property
    def empty(self):
        with self._lock:
            return self._rows == 0

    def wait_refresh(self):
        if self._future is not None:
            self._future.result()

    def sample(self, batch_size):
        with self._lock:
            empty = self._rows == 0
        if empty:
            if self._future is None:
                raise RuntimeError("GPU replay pool has not been primed")
            self._future.result()  # Cold start only; normal sampling never awaits refresh.
        with background_cuda_work(), self._lock, torch.cuda.stream(self._stream):
            if not self._rows:
                raise RuntimeError("GPU replay pool is empty")
            positions = torch.randint(self._rows, (batch_size,), device=self.device,
                                      generator=self._generator)
            slots = torch.searchsorted(self._ends, positions, right=True)
            indices = positions + self._starts[slots]
            data = {k: v[indices] for k, v in self._data.items()}
            event = self._stream.record_event()
        return ReplayBatch(data, event)

    def stats(self):
        with self._lock:
            return {"gpu_cache_bytes": self.cache_bytes, "gpu_cached_transitions": self._rows,
                    "gpu_cached_chunks": len(self._lookup), "gpu_chunks_copied": self.chunks_copied,
                    "gpu_bytes_copied": self.bytes_copied,
                    "gpu_refresh_pending": int(self._future is not None and not self._future.done())}

    def close(self):
        if not self._closed:
            self._worker.shutdown(wait=True)
            try:
                if self._future is not None:
                    self._future.result()
            finally:
                self._stream.synchronize()
                self._closed = True
