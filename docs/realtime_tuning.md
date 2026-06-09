# Real-time tuning on Linux

This package is written in Python and runs on a stock Linux kernel by
default. That's usually fine for development and for offline analysis,
but for closed-loop experiments where the feedback has to land inside a
biologically meaningful window (tens of ms), the default OS configuration
is the wrong starting point. Mean latency is rarely the problem; the
problem is the long tail. A 100 ms hiccup once a minute is invisible in
benchmarks and visible in your experiment.

This doc collects the OS-level knobs that matter, the rough order to
apply them in, and how to verify each one is doing what you expect. None
of this changes the science; it changes how reliable the timing of the
science is.

A short note before starting: don't touch any of this in isolation. The
right loop is **measure -> change one thing -> measure again**. The package
already writes timing arrays per rank
(`*_decoder_*.timing.npz`, `*_encoder_*.timing.npz`); plot the
end-to-end latency histogram, look at p99 and p99.9 (not the mean), and
treat that as the metric you're tuning against.

---

## 1. Don't fight the GC for free latency

This is the only thing in this doc that's already wired up in code (see
`realtime_decoder/rt_tuning.py` and the `with
gc_paused_for_main_loop(...)` wrap around the hot-rank main loops).

CPython has a generational cyclic GC that wakes up on allocation
thresholds. In a long-running numeric loop that allocates a lot of
short-lived numpy objects, gen-2 collections can stall the interpreter
for tens of ms at a time. The pre-allocation work elsewhere in this
package eliminates most of the steady-state allocations; on top of that,
the hot-rank main loops now run with `gc.disable()` until they exit.

If you want to keep gc on for debugging (e.g. you're hunting a reference
cycle), opt out:

```yaml
performance:
  disable_gc_in_main_loop: false
```

## 2. Pin processes to dedicated cores

Default Linux scheduling will happily move your decoder rank between
cores in response to interrupts on neighboring cores. Every migration is
a cache-line warmup penalty.

`mpiexec` already supports binding. The minimum you should pass is
`-bind-to hwthread`, which the README already shows:

```
mpiexec -np 5 -bind-to hwthread python -u runscript.py config/my.yml
```

For tighter control, pin specific ranks to specific cores with
`--map-by rankfile`:

```
mpiexec -np 5 --map-by rankfile:file=rankfile.txt python -u runscript.py config/my.yml
```

`rankfile.txt`:
```
rank 0=localhost slot=0
rank 1=localhost slot=2
rank 2=localhost slot=4
rank 3=localhost slot=6
rank 4=localhost slot=8
```

(Use `lscpu --extended` to find the right physical core IDs for your
machine, especially on a hyperthreaded box where you usually want to
leave the SMT siblings free.)

## 3. Isolate those cores from the kernel scheduler

`isolcpus` removes a set of cores from the kernel's load balancer.
Combined with rankfile pinning, the rank gets a core that the kernel
won't dispatch other work onto.

Edit your bootloader (e.g. `/etc/default/grub`):

```
GRUB_CMDLINE_LINUX="... isolcpus=2,4,6,8 nohz_full=2,4,6,8 rcu_nocbs=2,4,6,8"
```

- `isolcpus`: don't schedule normal tasks here
- `nohz_full`: skip the periodic 1ms tick when only one task is running
- `rcu_nocbs`: offload RCU callbacks to other cores

Run `update-grub && reboot`. Verify after with:

```
cat /sys/devices/system/cpu/isolated
```

## 4. Move IRQs off the hot cores

Even with `isolcpus`, interrupts can still land on your isolated cores
unless you move them off explicitly. Default kernels send most IRQs to
CPU 0 anyway, which is convenient; double-check with:

```
cat /proc/interrupts
```

To force all IRQs onto a chosen set of cores (say 0 and 1):

```
echo 3 > /proc/irq/default_smp_affinity
for f in /proc/irq/*/smp_affinity; do echo 3 > "$f"; done
```

`3` is the bitmask for cores 0 and 1. Adjust to your topology.

## 5. SCHED_FIFO for the hot ranks

By default the Linux completely-fair scheduler (CFS) treats your decoder
as a normal time-shared process. `chrt` switches it to a real-time
scheduling class so it isn't preempted by background work.

Wrap your `mpiexec` call:

```
chrt -f 50 mpiexec -np 5 -bind-to hwthread python -u runscript.py config/my.yml
```

`-f 50` = SCHED_FIFO, priority 50 (anywhere in 1..99 is fine; higher
priorities preempt lower ones). The priority must be lower than your
network drivers' threaded IRQ handlers if you have them; 50 is a safe
default.

If you're on a stock kernel (not PREEMPT_RT), you also need:

```
sudo sysctl -w kernel.sched_rt_runtime_us=-1
```

(default `950000` caps RT tasks to 95% of CPU; for a dedicated
isolated core you want all 100%.)

## 6. Lock memory with mlockall

If any of your working set gets paged out, the page fault on the next
access is a multi-ms hit. `mlockall(MCL_CURRENT | MCL_FUTURE)` pins the
process's entire address space in RAM.

There isn't a great way to call this from Python without a small C
helper, but you can approximate it with `prlimit` to keep the process
from being swapped:

```
sudo sysctl -w vm.swappiness=0
```

For full mlockall, the simplest path is a tiny `LD_PRELOAD` shim or the
[`memlock` Python package](https://pypi.org/project/memlock/). Either is
ok; pick one and document it in your lab's runner script.

## 7. PREEMPT_RT kernel

Everything above helps, but the ceiling on tail latency with a stock
kernel is still in the few-ms range for unlucky interrupts (timer
ticks, network IRQs). PREEMPT_RT replaces most non-preemptible kernel
sections with priority-inheritance mutexes, pushing the worst-case
preempt latency into the microsecond range.

This is the heaviest change in this doc and is only worth it once you've
done everything else. Two practical paths:

1. **A stock RT-enabled distro**: Ubuntu offers a `linux-image-rt-generic`
   package since 22.04; SUSE Linux Enterprise Real Time has a long-standing
   RT kernel. Easiest if you can pick the distro.
2. **A self-built kernel**: download the latest PREEMPT_RT patch from
   <https://wiki.linuxfoundation.org/realtime/start>, apply against the
   matching mainline kernel, build with `CONFIG_PREEMPT_RT=y`. Time-consuming
   the first time and you'll do it once per kernel upgrade.

After booting an RT kernel, verify:

```
uname -a              # should mention PREEMPT_RT
cat /sys/kernel/realtime  # should print 1
```

## 8. Disable CPU frequency scaling

The on-demand governor will idle your isolated cores down to low
frequencies during the brief gaps between events. Coming back up is on
the order of microseconds but it's also non-deterministic. Pin to a
fixed performance state:

```
sudo cpupower frequency-set -g performance
```

For Intel chips, also disable turbo (variable boost adds jitter):

```
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

## 9. A runner script that ties it together

Once you've picked your knobs, wrap it all in a single launcher so
operators don't have to remember the recipe. A sketch:

```bash
#!/usr/bin/env bash
# launch.sh <config_yaml>
set -euo pipefail
CONFIG=${1:?usage: launch.sh <config_yaml>}

# pin scheduler / freq / swap
sudo sysctl -w kernel.sched_rt_runtime_us=-1 >/dev/null
sudo cpupower frequency-set -g performance >/dev/null

# put background interrupts on housekeeping cores
for f in /proc/irq/*/smp_affinity; do echo 3 > "$f" 2>/dev/null || true; done

# launch with RT priority + rankfile pinning
exec chrt -f 50 \
    mpiexec -np 5 --map-by rankfile:file=rankfile.txt \
    python -u runscript.py "$CONFIG"
```

Drop it next to `runscript.py`, mark it executable, and document it as
the entry point for closed-loop runs. The version-controlled file is the
thing that makes the tuning reproducible across grad student turnover.

## 10. Verification

A short post-flight checklist after each run:

- `cat /proc/$(pgrep -f runscript.py | head -1)/status` -> check `voluntary_ctxt_switches` is stable and `nonvoluntary_ctxt_switches` is small.
- Open the timing `.npz` and look at p99 / p99.9 of the per-tick decoder latency. A well-tuned system on a recent box gets to <2 ms p99.9 for the kind of work this package does.
- Run a quiescent baseline (no animal): if you still see >5 ms p99.9 spikes, something OS-level is intruding (housekeeping, swap, IRQ).

If any of this is wrong or out of date for your setup, please open a PR
with a correction.
