# Portable DSA replay runner

Requires Python, PyTorch with CUDA, Triton, and the `temporal_dsa` package/source tree.

```bash
PYTHONPATH=src python artifacts/l40s_dsa_lock/cross_platform_runner/benchmark_dsa_replay.py --bundle artifacts/l40s_dsa_lock/replay_bundles/dsa_replay_c32768_layer17_code_heldout_3.pt --method full64 --top-k 2048 --warmup 200 --iters 2000
```
