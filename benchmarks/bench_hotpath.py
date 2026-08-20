"""Standalone latency benchmark for the real-time hot-path compute kernels.

This drives the *actual* decoding kernels used in closed-loop experiments
(`ClusterlessDecoder.compute_posterior` and `Encoder.get_joint_prob`)
without needing MPI or Trodes, so latency can be measured and tuned on any
machine. It is the "measure" half of the measure -> change one thing ->
measure loop described in docs/realtime_tuning.md.

Two modes:

  python benchmarks/bench_hotpath.py            # timing: p50/p99/p99.9/max
  python benchmarks/bench_hotpath.py --checksum # deterministic output digest

The checksum mode runs a fixed, seeded sequence of inputs through the kernels
and prints a sha256 of every output array. Run it before and after a change to
a kernel: an identical digest proves the change did not alter the numerical
output (i.e. the optimization is semantics-preserving).

Sizes default to the SC66 production config (num_bins=41, mark buffer 50000,
mark_dim 4). Override with flags to explore the cost surface.
"""

import argparse
import hashlib
import os
import sys
import time
import types

import numpy as np

# allow `python benchmarks/bench_hotpath.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Import the package without MPI / Trodes present. The compute kernels do not
# use MPI at runtime; base.py only needs the names to exist at import time.
# ---------------------------------------------------------------------------
def _install_mpi_stub():
    if "mpi4py" in sys.modules:
        return

    class _AnyAttr:
        """Returns a throwaway type for any attribute access (MPI.Comm etc.)."""
        def __getattr__(self, name):
            return type(name, (), {})

    mpi4py = types.ModuleType("mpi4py")
    mpi4py.MPI = _AnyAttr()
    sys.modules["mpi4py"] = mpi4py


_install_mpi_stub()

from realtime_decoder.decoder_process import ClusterlessDecoder  # noqa: E402
from realtime_decoder.encoder_process import Encoder  # noqa: E402
from realtime_decoder.position import PositionBinStruct  # noqa: E402


# ---------------------------------------------------------------------------
# Config construction. Only the keys the kernels actually read are populated.
# Defaults mirror config/SC66.yml.
# ---------------------------------------------------------------------------
def make_config(*, num_bins, arm_coords, n_elec, bufsize, mark_dim, rank):
    return {
        "algorithm": "clusterless_decoder",
        "preloaded_model": False,
        "sampling_rate": {"spikes": 30000},
        "clusterless_decoder": {"state_labels": ["state"], "transmat_bias": 1},
        "decoder_assignment": {rank: list(range(1, n_elec + 1))},
        "decoder": {
            "time_bin": {"samples": 180, "delay_samples": 180},
        },
        "display": {
            "decoder": {"occupancy": 10_000_000},
            "encoder": {"occupancy": 10_000_000},
        },
        "encoder": {
            "mark_dim": mark_dim,
            "bufsize": bufsize,
            "use_channel_dist_from_max_amp": False,
            "mark_kernel": {
                "std": 20,
                "use_filter": True,
                "n_std": 1,
                "n_marks_min": 10,
            },
            "position": {
                "lower": 0,
                "upper": num_bins,
                "num_bins": num_bins,
                "arm_coords": arm_coords,
            },
        },
    }


def make_pos_bin_struct(config):
    p = config["encoder"]["position"]
    return PositionBinStruct(p["lower"], p["upper"], p["num_bins"])


# ---------------------------------------------------------------------------
# Percentile reporting
# ---------------------------------------------------------------------------
def summarize(name, samples_ns):
    a = np.sort(np.asarray(samples_ns, dtype=np.float64)) / 1000.0  # -> microseconds
    def pct(q):
        return a[min(len(a) - 1, int(q * len(a)))]
    print(
        f"  {name:<28} n={len(a):>6}  "
        f"p50={pct(0.50):8.2f}  p99={pct(0.99):8.2f}  "
        f"p99.9={pct(0.999):8.2f}  max={a[-1]:9.2f}   (us)"
    )
    return {"p50": pct(0.50), "p99": pct(0.99), "p999": pct(0.999), "max": a[-1]}


# ---------------------------------------------------------------------------
# Decoder benchmark
# ---------------------------------------------------------------------------
def build_decoder(config, rank, rng):
    dec = ClusterlessDecoder(rank, config, make_pos_bin_struct(config))
    # Populate occupancy realistically: accumulate position samples and apply
    # the no-animal boundary so gap bins become nan (exercises the isfinite
    # handling in compute_posterior).
    num_bins = config["encoder"]["position"]["num_bins"]
    for _ in range(2000):
        dec.update_position(int(rng.integers(0, num_bins)), True)
    return dec


def make_spike_arr(config, rank, n_spikes, num_bins, rng):
    """One time bin's worth of spikes in the layout compute_posterior expects:
    columns [timestamp, elec_grp_id, pos, cred_int, used_flag, hist(num_bins)].
    """
    elec_ids = config["decoder_assignment"][rank]
    arr = np.zeros((n_spikes, 5 + num_bins), dtype=np.float64)
    for i in range(n_spikes):
        arr[i, 0] = rng.integers(0, 10_000_000)
        arr[i, 1] = rng.choice(elec_ids)
        arr[i, 2] = rng.integers(0, num_bins)
        arr[i, 3] = rng.integers(0, 20)
        hist = rng.random(num_bins) + 1e-6
        hist /= hist.sum()
        arr[i, 5:] = hist
    return arr


def bench_decoder(*, num_bins, arm_coords, n_elec, n_iter, warmup, seed):
    rank = 1
    config = make_config(
        num_bins=num_bins, arm_coords=arm_coords, n_elec=n_elec,
        bufsize=2000, mark_dim=4, rank=rank,
    )
    rng = np.random.default_rng(seed)
    dec = build_decoder(config, rank, rng)

    # Realistic per-bin spike counts: mostly 0-4 spikes in a 6 ms bin.
    spike_counts = rng.integers(0, 5, size=n_iter + warmup)
    bins = [make_spike_arr(config, rank, int(c), num_bins, rng)
            for c in spike_counts]

    for i in range(warmup):
        dec.compute_posterior(bins[i])

    samples = np.empty(n_iter, dtype=np.int64)
    for i in range(n_iter):
        b = bins[warmup + i]
        t0 = time.perf_counter_ns()
        dec.compute_posterior(b)
        samples[i] = time.perf_counter_ns() - t0

    label = f"compute_posterior nb={num_bins} elec={n_elec}"
    return summarize(label, samples)


# ---------------------------------------------------------------------------
# Encoder benchmark
# ---------------------------------------------------------------------------
def build_encoder(config, trode, n_marks, rng):
    enc = Encoder(config, trode, make_pos_bin_struct(config))
    num_bins = config["encoder"]["position"]["num_bins"]
    mark_dim = config["encoder"]["mark_dim"]

    # Fill the mark buffer: a cluster of marks in mark space plus their
    # observed positions, and a realistic occupancy with no-animal gaps.
    n_marks = min(n_marks, enc._marks.shape[0])
    enc._marks[:n_marks] = rng.normal(100.0, 30.0, size=(n_marks, mark_dim))
    enc._positions[:n_marks] = rng.integers(0, num_bins, size=n_marks)
    enc._mark_idx = n_marks

    occ = rng.integers(1, 50, size=num_bins).astype(np.float64)
    from realtime_decoder import utils
    utils.apply_no_anim_boundary(
        enc._pos_bins, enc._arm_coords, occ, np.nan
    )
    enc._occupancy = occ
    return enc


def bench_encoder(*, num_bins, arm_coords, n_marks, n_iter, warmup, seed):
    config = make_config(
        num_bins=num_bins, arm_coords=arm_coords, n_elec=1,
        bufsize=max(n_marks, 50000), mark_dim=4, rank=1,
    )
    rng = np.random.default_rng(seed)
    enc = build_encoder(config, 1, n_marks, rng)

    # Query marks drawn near the cluster center so the n_marks_min gate passes.
    queries = rng.normal(100.0, 30.0, size=(n_iter + warmup, config["encoder"]["mark_dim"]))

    for i in range(warmup):
        enc.get_joint_prob(queries[i])

    samples = np.empty(n_iter, dtype=np.int64)
    for i in range(n_iter):
        q = queries[warmup + i]
        t0 = time.perf_counter_ns()
        enc.get_joint_prob(q)
        samples[i] = time.perf_counter_ns() - t0

    return summarize(f"get_joint_prob marks={n_marks}", samples)


# ---------------------------------------------------------------------------
# Checksum mode: deterministic digest of kernel outputs for equivalence checks
# ---------------------------------------------------------------------------
def _original_joint_prob(enc, mark):
    """Faithful re-implementation of the pre-optimization get_joint_prob,
    used only to prove the optimized version is numerically equivalent."""
    if enc._mark_idx == 0:
        return None
    mark_idx = min(enc._mark_idx, enc._marks.shape[0])

    in_range = np.ones(mark_idx, dtype=bool)
    if enc.p['use_filter']:
        std = enc.p['filter_std']
        n_std = enc.p['filter_n_std']
        for ii in range(enc._marks.shape[1]):
            in_range = np.logical_and(
                np.logical_and(
                    enc._marks[:mark_idx, ii] > mark[ii] - n_std * std,
                    enc._marks[:mark_idx, ii] < mark[ii] + n_std * std,
                ),
                in_range,
            )
        if np.sum(in_range) < enc.p['n_marks_min']:
            return None

    squared_distance = np.sum(np.square(enc._marks[:mark_idx] - mark), axis=1)
    weights = enc._k1 * np.exp(squared_distance * enc._k2)
    positions = enc._positions[:mark_idx]
    hist, _ = np.histogram(
        a=positions, bins=enc._pos_bin_struct.pos_bin_edges, weights=weights
    )
    hist += 0.0000001
    hist /= (enc._occupancy / np.nansum(enc._occupancy))
    hist[~np.isfinite(hist)] = 0.0
    hist /= (np.sum(hist) * enc._pos_bin_struct.pos_bin_delta)
    return hist


def verify(seed=20240601):
    """Prove the optimized encoder kernel preserves the result.

    exact_histogram=True must be bit-identical to the original algorithm;
    the fast histogram path must match to within floating-point rounding.
    """
    num_bins = 41
    arm_coords = [[0, 8], [13, 24], [29, 40]]
    config = make_config(
        num_bins=num_bins, arm_coords=arm_coords, n_elec=1,
        bufsize=50000, mark_dim=4, rank=1,
    )
    rng = np.random.default_rng(seed)
    enc = build_encoder(config, 1, 50000, rng)

    n_checks = 400
    exact_bit_identical = 0
    fast_max_rel = 0.0
    compared = 0
    for _ in range(n_checks):
        q = rng.normal(100.0, 30.0, size=4)
        ref = _original_joint_prob(enc, q)

        enc.p['exact_histogram'] = True
        e = enc.get_joint_prob(q)
        enc.p['exact_histogram'] = False
        f = enc.get_joint_prob(q)

        if ref is None:
            assert e is None and f is None
            continue
        compared += 1
        if np.array_equal(e.hist, ref):
            exact_bit_identical += 1
        nz = ref != 0
        if nz.any():
            fast_max_rel = max(
                fast_max_rel,
                float(np.max(np.abs(f.hist[nz] - ref[nz]) / np.abs(ref[nz]))),
            )

    print(f"compared {compared} non-trivial spikes")
    print(f"exact_histogram path bit-identical to original: "
          f"{exact_bit_identical}/{compared} "
          f"{'PASS' if exact_bit_identical == compared else 'FAIL'}")
    print(f"fast path max relative diff vs original: {fast_max_rel:.2e} "
          f"{'PASS (< 1e-6, rounding noise)' if fast_max_rel < 1e-6 else 'FAIL'}")


def checksum(seed=1234):
    h = hashlib.sha256()

    # Decoder
    num_bins = 41
    arm_coords = [[0, 8], [13, 24], [29, 40]]
    rank = 1
    config = make_config(
        num_bins=num_bins, arm_coords=arm_coords, n_elec=8,
        bufsize=2000, mark_dim=4, rank=rank,
    )
    rng = np.random.default_rng(seed)
    dec = build_decoder(config, rank, rng)
    for _ in range(500):
        n = int(rng.integers(0, 5))
        post, lk = dec.compute_posterior(make_spike_arr(config, rank, n, num_bins, rng))
        h.update(np.ascontiguousarray(post, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(lk, dtype=np.float64).tobytes())

    # Encoder
    enc_config = make_config(
        num_bins=num_bins, arm_coords=arm_coords, n_elec=1,
        bufsize=50000, mark_dim=4, rank=1,
    )
    rng2 = np.random.default_rng(seed + 1)
    enc = build_encoder(enc_config, 1, 50000, rng2)
    for _ in range(500):
        q = rng2.normal(100.0, 30.0, size=4)
        est = enc.get_joint_prob(q)
        if est is not None:
            h.update(np.ascontiguousarray(est.hist, dtype=np.float64).tobytes())

    print("output digest:", h.hexdigest())


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checksum", action="store_true",
                    help="print deterministic digest of kernel outputs and exit")
    ap.add_argument("--verify", action="store_true",
                    help="prove optimized encoder kernel matches the original and exit")
    ap.add_argument("--iter", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.verify:
        verify()
        return

    if args.checksum:
        checksum()
        return

    arm = [[0, 8], [13, 24], [29, 40]]
    print(f"numpy {np.__version__}")
    print("\nDecoder posterior (per time bin):")
    bench_decoder(num_bins=41, arm_coords=arm, n_elec=8,
                  n_iter=args.iter, warmup=args.warmup, seed=args.seed)
    bench_decoder(num_bins=41, arm_coords=arm, n_elec=32,
                  n_iter=args.iter, warmup=args.warmup, seed=args.seed)
    # Stress: larger position grid (e.g. 2D environments)
    big_arm = [[0, 127], [128, 255]]
    bench_decoder(num_bins=256, arm_coords=big_arm, n_elec=32,
                  n_iter=args.iter, warmup=args.warmup, seed=args.seed)

    print("\nEncoder joint prob (per spike, scales with stored marks):")
    for nm in (1000, 10000, 50000):
        bench_encoder(num_bins=41, arm_coords=arm, n_marks=nm,
                      n_iter=min(args.iter, 3000), warmup=200, seed=args.seed)


if __name__ == "__main__":
    main()
