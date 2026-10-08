"""Minimal CPU example; use CUDA/ROCm and a VRAM budget in your trainer."""
import tempfile
import torch
from block_replay import BlockReplayBuffer


def main():
    with tempfile.TemporaryDirectory() as directory:
        with BlockReplayBuffer(directory, capacity=1024, block_size=128,
                               ram_blocks=4, batch_size=32, device="cpu") as replay:
            observation = torch.arange(512, dtype=torch.float32).unsqueeze(1)
            replay.append_async({"observation": observation, "reward": -observation[:, 0]})
            replay.flush()
            batch = replay.sample_async().wait()
            print({key: tuple(value.shape) for key, value in batch.items()})


if __name__ == "__main__":
    main()
