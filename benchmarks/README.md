# Hot-path latency benchmarks

`bench_hotpath.py` measures the two compute kernels on the real-time critical
path, in isolation, without needing MPI or Trodes:

- `Encoder.get_joint_prob` - the per-spike clusterless KDE (encoder ranks)
- `ClusterlessDecoder.compute_posterior` - the per-time-bin posterior (decoder ranks)

It drives the *actual* classes from `realtime_decoder`, not a reimplementation,
so the numbers reflect the shipped code. It is the "measure" half of the
measure -> change one thing -> measure loop in `docs/realtime_tuning.md`.

## Usage

```bash
# latency distributions (p50/p99/p99.9/max) at SC66 sizes and stress sizes
python benchmarks/bench_hotpath.py

# prove the optimized encoder kernel still matches the original algorithm
python benchmarks/bench_hotpath.py --verify

# deterministic digest of kernel outputs (compare across code versions)
python benchmarks/bench_hotpath.py --checksum
```

Only `numpy` is required. The harness installs a tiny `mpi4py` stub so the
package imports without an MPI runtime; the kernels themselves do not use MPI.

## What to look at

Report **p99 and p99.9**, not the mean. Closed-loop latency is a tail problem:
a kernel that is fast on average but occasionally stalls for milliseconds will
force a larger spike jitter buffer (`decoder.time_bin.delay_samples`), which is
the single biggest controllable term in end-to-end latency. See
`docs/latency_analysis.md`.

## Sizes

Defaults mirror `config/SC66.yml`: `num_bins=41`, encoder mark buffer
`bufsize=50000`, `mark_dim=4`. `get_joint_prob` cost scales with the number of
stored marks, so the buffer-full case (50000) is the steady-state worst case.
