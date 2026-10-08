import threading
from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest

import torch
from block_replay import BlockReplayBuffer


class BlockReplayTests(unittest.TestCase):
    def exercise(self, device):
        previous = torch.get_default_device()
        torch.set_default_device(device)
        try:
            with tempfile.TemporaryDirectory() as directory:
                with BlockReplayBuffer(directory, capacity=128, block_size=8,
                        ram_blocks=3, batch_size=64, device=device,
                        refresh_every=1, prefetch=3, seed=12) as buffer:
                    for start in range(0, 160, 10):
                        x = torch.arange(start, start + 10, dtype=torch.float32, device=device)
                        buffer.append_async({"x": x, "reward": -x, "done": x.remainder(2).bool()})
                        x.fill_(999)
                    buffer.flush()
                    self.assertEqual(len(buffer), 128)
                    seen = set()
                    for _ in range(256):
                        # Complete a refresh on the worker before checking broad
                        # coverage; asynchronous scheduling is tested separately.
                        buffer._sampler.submit(buffer._refresh).result(timeout=10)
                        batch = buffer.sample_async().wait()
                        # Coverage depends on completed refreshes, not on how
                        # fast this machine can finish 100 learner batches.
                        if buffer._refresh_future is not None:
                            buffer._refresh_future.result(timeout=10)
                        x = batch["x"]
                        self.assertTrue(torch.all((x >= 32) & (x < 160)))
                        self.assertTrue(torch.equal(batch["reward"], -x))
                        self.assertTrue(torch.equal(batch["done"], x.remainder(2).bool()))
                        seen.update(x.cpu().tolist())
                    self.assertEqual(seen, set(range(32, 160)))
                    self.assertEqual(buffer.stats()["ram_blocks"], 3)
                    self.assertGreater(buffer.stats()["blocks_read"], 16)
                    # Continue collecting while prefetch is active; pruned blocks
                    # already sampled may remain in queued immutable batches.
                    for start in range(160, 240, 10):
                        x = torch.arange(start, start + 10, dtype=torch.float32, device=device)
                        buffer.append_async({"x": x, "reward": -x, "done": x.remainder(2).bool()})
                        batch = buffer.sample_async().wait()
                        self.assertTrue(torch.equal(batch["reward"], -batch["x"]))
                    buffer.flush()
                    self.assertEqual(len(buffer), 128)
        finally:
            torch.set_default_device(previous)

    def test_cpu_retention_refresh_and_concurrency(self):
        self.exercise("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA or ROCm required")
    def test_gpu_staging_under_default_cuda_device(self):
        self.exercise("cuda:0")

    def test_slow_disk_refresh_does_not_stop_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            with BlockReplayBuffer(directory, capacity=64, block_size=8,
                    ram_blocks=2, batch_size=32, device="cpu", refresh_every=1) as buffer:
                buffer.append_async({"x": torch.arange(64, device="cpu")})
                buffer.flush()
                started, release = threading.Event(), threading.Event()
                original_load = buffer._load
                def slow_refresh():
                    with buffer._lock:
                        uncached = next(i for i in buffer._blocks if i not in buffer._cache)
                    started.set()
                    if not release.wait(5):
                        raise TimeoutError("Test did not release simulated disk read")
                    original_load(uncached)
                buffer._refresh = slow_refresh
                try:
                    with ThreadPoolExecutor(1) as caller:
                        first = caller.submit(buffer.sample_async)
                        self.assertTrue(started.wait(2))
                        # A blocked disk worker must not drain the sampler queue.
                        self.assertTrue(torch.all(first.result(timeout=1).wait()["x"] < 64))
                        for _ in range(8):
                            batch = caller.submit(buffer.sample_async).result(timeout=1).wait()
                            self.assertTrue(torch.all(batch["x"] < 64))
                        self.assertFalse(buffer._refresh_future.done())
                finally:
                    release.set()
                buffer._refresh_future.result(timeout=2)
                self.assertEqual(buffer.stats()["blocks_read"], 1)

    def test_new_blocks_reuse_ram_without_disk_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            with BlockReplayBuffer(directory, capacity=64, block_size=8,
                    ram_blocks=8, batch_size=32, device="cpu", refresh_every=1) as buffer:
                buffer.append_async({"x": torch.arange(64, device="cpu")})
                buffer.flush()
                for _ in range(16):
                    buffer.sample_async().wait()
                if buffer._refresh_future is not None:
                    buffer._refresh_future.result()
                self.assertEqual(buffer.stats()["ram_blocks"], 8)
                self.assertEqual(buffer.stats()["blocks_read"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA or ROCm required")
    def test_gpu_chunk_sampling_retention_partial_and_transfer_overlap(self):
        previous = torch.get_default_device()
        torch.set_default_device("cuda:0")
        try:
            with tempfile.TemporaryDirectory() as directory:
                with BlockReplayBuffer(directory, capacity=32, block_size=8,
                        ram_blocks=4, batch_size=32, device="cuda:0", prefetch=2,
                        gpu_cache_bytes=32 * 9, device_chunk_size=4,
                        device_refresh_every=1, device_refresh_chunks=8) as buffer:
                    def append(start, count):
                        x = torch.arange(start, start + count, dtype=torch.float32, device="cuda:0")
                        buffer.append_async({"x": x, "reward": -x, "done": x.remainder(2).bool()})
                        x.fill_(999)
                    append(0, 32)
                    buffer.flush()
                    seen = set()
                    for _ in range(64):
                        batch = buffer.sample_async().wait()
                        x = batch["x"]
                        self.assertTrue(torch.equal(batch["reward"], -x))
                        self.assertTrue(torch.equal(batch["done"], x.remainder(2).bool()))
                        self.assertTrue(torch.all((x >= 0) & (x < 32)))
                        seen.update(x.cpu().tolist())
                    self.assertEqual(seen, set(range(32)))
                    self.assertGreater(buffer.stats()["gpu_cached_chunks"], 1)
                    # Publish enough data to prune the entire old GPU population.
                    append(32, 40)
                    buffer.flush()
                    for _ in range(buffer.prefetch):
                        buffer.sample_async().wait()  # Discard previously prepared batches.
                    for _ in range(16):
                        batch = buffer.sample_async().wait()
                        self.assertTrue(torch.all((batch["x"] >= 40) & (batch["x"] < 72)))
                        self.assertTrue(torch.equal(batch["reward"], -batch["x"]))
                    append(72, 3)
                    buffer.flush()
                    for _ in range(16):
                        batch = buffer.sample_async().wait()
                        self.assertTrue(torch.equal(batch["reward"], -batch["x"]))
                        self.assertTrue(torch.all(batch["x"] < 75))
                    self.assertLessEqual(buffer.stats()["gpu_cached_transitions"], 32)
        finally:
            torch.set_default_device(previous)

    def test_partial_block_and_empty_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            with BlockReplayBuffer(directory, capacity=16, block_size=8,
                    ram_blocks=2, batch_size=32, device="cpu") as buffer:
                x = torch.arange(3, device="cpu")
                buffer.append_async({"x": x})
                buffer.flush()
                self.assertEqual(len(buffer), 3)
                self.assertTrue(torch.all(buffer.sample_async().wait()["x"] < 3))


if __name__ == "__main__":
    unittest.main()
