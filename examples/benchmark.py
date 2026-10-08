"""Compare GPU transfer chunk sizes with equal resident replay budgets."""
import argparse
import json
import tempfile
import time

import torch

from block_replay import BlockReplayBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--chunks', type=int, nargs='+', default=[4096, 16384])
    parser.add_argument('--rows', type=int, default=131072)
    parser.add_argument('--batches', type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for chunk in args.chunks:
        with tempfile.TemporaryDirectory() as directory:
            with BlockReplayBuffer(directory, capacity=args.rows, block_size=32768,
                    ram_blocks=max(1, args.rows // 32768), batch_size=2048,
                    device=args.device, gpu_cache_bytes=args.rows * 1454,
                    device_chunk_size=chunk, device_refresh_every=8,
                    device_refresh_chunks=8) as buffer:
                fields = {'observation': torch.randn(args.rows,169,device=args.device),
                          'next_observation': torch.randn(args.rows,169,device=args.device),
                          'action': torch.randn(args.rows,24,device=args.device),
                          'reward': torch.randn(args.rows,device=args.device),
                          'terminated': torch.zeros(args.rows,dtype=torch.bool,device=args.device),
                          'truncated': torch.zeros(args.rows,dtype=torch.bool,device=args.device)}
                buffer.append_async(fields)
                buffer.flush()
                pool = buffer._device_pool
                entries = [(i,d,buffer._blocks[i][1]) for i,d in buffer._cache.items()]
                start = time.perf_counter()
                for _ in range(100):
                    pool.request_refresh(entries)
                    pool.wait_refresh()
                    if pool.stats()['gpu_cached_transitions'] == args.rows:
                        break
                else:
                    raise RuntimeError('GPU pool did not fill')
                prime_seconds = time.perf_counter() - start
                for _ in range(16):
                    buffer.sample_async().wait()
                torch.cuda.synchronize(args.device)
                start = time.perf_counter()
                for _ in range(args.batches):
                    buffer.sample_async().wait()
                torch.cuda.synchronize(args.device)
                seconds = time.perf_counter() - start
                print(json.dumps({'chunk_rows':chunk, 'prime_seconds':prime_seconds,
                    'sample_us':seconds/args.batches*1e6, 'batches':args.batches,
                    **pool.stats()}),flush=True)


if __name__ == '__main__':
    main()
