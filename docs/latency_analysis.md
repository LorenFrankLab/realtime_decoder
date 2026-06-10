# Where the latency actually is

This note records a measurement-driven look at closed-loop latency in this
package: what dominates it, what does not, and which knob moves it the most. It
is the companion to `docs/realtime_tuning.md` (which covers the OS) and to
`benchmarks/bench_hotpath.py` (which produces the numbers below). All timings
are from that benchmark on a developer laptop; absolute values will differ on
your rig, but the *ratios* and the *ranking of costs* hold.

The headline: at production sizes the per-bin decoder math is microseconds, the
end-to-end latency is dominated by the **time-bin width plus the spike jitter
buffer** (`decoder.time_bin`), and the largest piece of controllable *compute*
is the encoder's per-spike KDE - which was spending most of its time inside
`np.histogram`.

## The closed-loop critical path

For a replay/ripple-triggered stim, a neural event becomes a hardware output
through roughly these stages:

```
spike --> Trodes acq --> encoder rank: get_joint_prob --> MPI --> decoder rank
      --> wait for the time bin to close (+ delay) --> compute_posterior
      --> MPI --> main/stim decision --> ECU shortcut --> hardware
```

Two of those stages are fixed by the science or the hardware (acquisition
transport, and the bin width you chose for statistical reasons). The rest is
software, and that is where this package can help or hurt.

## What the numbers say

Measured per-call latency of the two hot kernels (`bench_hotpath.py`, SC66
config: `num_bins=41`, mark buffer 50000, `mark_dim=4`):

| kernel | size | p50 | p99 |
|---|---|---|---|
| `compute_posterior` (per bin) | 8 electrodes | 65 us | 71 us |
| `compute_posterior` (per bin) | 32 electrodes | 173 us | 199 us |
| `get_joint_prob` (per spike) | 1000 marks | 30 us | 36 us |
| `get_joint_prob` (per spike) | 10000 marks | 140 us | 162 us |
| `get_joint_prob` (per spike) | 50000 marks (buffer full) | 635 us | 700 us |

Put next to the time bin, the picture is stark. SC66 uses
`time_bin.samples = 180` and `time_bin.delay_samples = 180` at a 30 kHz spike
clock:

- bin width = 180 / 30000 = **6 ms**
- jitter buffer (`delay_samples`) = 180 / 30000 = **6 ms**

So the decoder deliberately decodes a bin that closed up to **12 ms ago**. The
posterior math that runs at the end of that window costs ~65-170 us - about
**1-3% of the bin period**. Decoder compute is not the bottleneck and was left
unchanged; spending effort shaving microseconds off a sub-millisecond operation
that sits behind a 12 ms wall would be pointless.

The encoder KDE is a different story. It costs ~0.65 ms *per spike* once the
mark buffer fills, and it grows with the buffer (30 us -> 140 us -> 635 us as
marks go 1k -> 10k -> 50k). At realistic spike rates across a tetrode this is
the kernel that saturates a rank and forces spikes to be dropped, and its tail
is what the jitter buffer has to absorb.

## The fix that mattered: stop calling np.histogram with explicit edges

Profiling `get_joint_prob` at a full 50000-mark buffer, before any change:

| component | time |
|---|---|
| `np.histogram(positions, bins=pos_bin_edges, weights=...)` | ~1195 us |
| `squared_distance` (subtract, square, sum over marks x channels) | ~585 us |
| in-cube filter (4 channels) | ~136 us |
| Gaussian `exp` | ~112 us |

Two thirds of the kernel was in `np.histogram`. The reason is subtle: when you
pass `np.histogram` an explicit array of bin **edges**, it cannot assume the
bins are uniform, so it bins every sample with a per-sample binary search
(`np.searchsorted`) - ~960 us for 50000 samples, cache-hostile and pure
overhead here, because the bins *are* uniform (`linspace(0, num_bins, ...)`).

The position grid in every shipped config is the integer grid `0, 1, ...,
num_bins`, so a position's bin index is just its integer part. Replacing the
search with a direct index + `np.bincount` drops the histogram from ~1195 us to
~67 us. Two more changes on top of that:

- Preallocate the per-spike scratch. The kernel allocated about a dozen
  temporaries per spike, several of them `O(marks)`; now they are reused.
- Fuse the squared distance with `np.einsum('ij,ij->i', diff, diff)`, which
  does the square and the per-mark sum in one pass instead of materializing the
  squared array and summing it. That takes the second-largest component from
  ~585 us to ~280 us.

Together:

| | p50 | p99 |
|---|---|---|
| before | 2096 us | 2702 us |
| after | 635 us | 700 us |

**About 3.3x faster on the median.** The tail (p99.9 and max) is dominated by
OS scheduling and varies run to run, but it tracks down with the median. The
tail is what matters most: it is the spike the decoder ends up waiting for, and
it sets how large the `delay_samples` jitter buffer has to be.

### Why this is safe

The change is gated and verified (`bench_hotpath.py --verify`):

- The preallocation is bit-for-bit identical to the original (verified on every
  spike of a 400-spike run).
- The fused distance sum and the uniform-bin histogram differ from the original
  only in floating-point **summation order**, ~3e-13 relative overall - the
  same magnitude of difference you already get from a different numpy version,
  BLAS thread count,
  or CPU. For a normalized probability over 41 bins driven by Poisson spikes,
  this is far below the noise floor.
- For anyone who needs bit-for-bit reproducibility against an older run, set
  `encoder.mark_kernel.exact_histogram: true`. That restores both the original
  `np.histogram` call and the original square-then-sum distance, so the output
  is identical to the old code. The general (non-integer-grid) fast histogram
  also uses numpy's own optimized uniform histogram, which is ~4x faster than
  the explicit-edges path and exactly matches its bin assignment.

## The lever that moves end-to-end latency: delay_samples

The encoder speedup matters for two reasons. The obvious one is throughput -
fewer dropped spikes per rank. The bigger one is indirect.

`delay_samples` exists because the decoder must wait for all spikes in a bin to
arrive from the encoders before it decodes that bin. How long it must wait is
set by the **worst-case** encoder-to-decoder latency, i.e. the *tail* of
`get_joint_prob` plus transport. Cutting the encoder's per-spike tail from ~4 ms
to ~1 ms shrinks the jitter that the buffer has to cover, which means
`delay_samples` can potentially come down - and `delay_samples` is worth up to
**6 ms** of end-to-end latency at SC66, versus the ~1 ms the compute itself
costs.

So the right way to actually lower closed-loop latency, in order of payoff:

1. **Tune `delay_samples` down to your measured worst case.** Run a session,
   open the per-rank `*.timing.npz` arrays, and look at the real distribution of
   encoder->decoder arrival times. Set `delay_samples` to cover p99.9 of that
   distribution with a small margin, not a round number copied between configs.
   This is the highest-value single change and it is a config edit.
2. **Keep the encoder tail low** so step 1 can be aggressive (this is what the
   `get_joint_prob` work buys you).
3. **Reduce the bin width** (`samples`) only if the statistics of your decode
   tolerate it - this is a science decision, not a software one. A smaller bin
   is less latency but noisier likelihoods.

## The theoretical floor, and what it would take to go lower

With the bin width and `delay_samples` fixed by the experiment, the software
floor for the decode path is now: encoder KDE (~0.8 ms tail) + two MPI hops
(tens of us each, already zero-copy for the data plane) + posterior (~0.1 ms) +
decision + actuation. Everything else is OS jitter, addressed in
`docs/realtime_tuning.md`.

To push the encoder KDE below ~0.8 ms at a full buffer you have to change the
math, not just the implementation - the remaining cost is the genuine work of
evaluating the kernel over every stored mark:

- **Truncated kernel.** The in-cube filter already finds the marks near the
  query; evaluating the Gaussian only on those (and treating the rest as zero)
  would cut both the distance computation and the histogram dramatically. It
  changes results by more than rounding (far marks contribute small but nonzero
  weight), so it needs the lab's sign-off and a config flag - but it is the
  single biggest remaining lever and worth prototyping behind the benchmark.
- **Spatial index (k-d tree / grid) over marks.** Same idea, sub-linear in the
  buffer size, more code.
- **float32 marks.** Halves the memory traffic that dominates
  `squared_distance`; changes numerics.
- **GPU.** Only wins at much larger mark buffers or electrode counts than SC66;
  the per-spike arrays here (50000 x 4) are too small to beat the host round
  trip. Revisit only if marks-per-buffer or tetrode counts grow by an order of
  magnitude.

The point of `bench_hotpath.py` is that none of these need to be argued in the
abstract: prototype, run `--verify` and the timing pass, and read the number.
