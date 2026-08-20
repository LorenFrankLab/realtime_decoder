"""Synthetic data source for the realtime_decoder.

Lets you install the package and run the full MPI pipeline end-to-end
without any acquisition hardware (Trodes, SpikeGLX, Open Ephys, etc.).

This is intended for:
  * smoke-testing a fresh install
  * developing/debugging the decoder loop on a laptop
  * regression testing in CI

It is NOT intended to be biologically realistic. The synthetic spikes are
Poisson with a fixed rate, marks are gaussian, and the synthetic position
walks back and forth on a simple linear track. The goal is to exercise the
data path and message plumbing, not to validate decoding accuracy.

Wiring is parallel to ``trodesnet.py``: a ``SyntheticDataReceiver`` that
mirrors ``TrodesDataReceiver`` and a ``SyntheticClient`` that mirrors
``TrodesClient``. The dispatch happens in ``runscript.py`` based on
``config['datasource']``.

Config block expected under the top-level ``synthetic`` key (all optional):

    synthetic:
      spike_rate_hz: 20         # per ntrode, Poisson
      mark_dim: 8               # must match encoder.mark_dim
      mark_amplitude_uv: 80     # mean spike amplitude
      track_length_cm: 200      # walk distance
      walk_speed_cm_s: 15       # synthetic animal speed
      startup_delay_s: 1.0      # delay before firing the startup callback
      run_duration_s: 60        # auto-terminate after this long
      voltage_scaling_factor: 0.195
"""

import time

import numpy as np

from realtime_decoder import utils
from realtime_decoder.base import DataSourceReceiver
from realtime_decoder.datatypes import (
    Datatypes,
    LFPPoint,
    SpikePoint,
    CameraModulePoint,
)


_DEFAULTS = {
    'spike_rate_hz': 20.0,
    'mark_dim': 4,
    'mark_amplitude_uv': 80.0,
    'track_length_cm': 200.0,
    'walk_speed_cm_s': 15.0,
    'startup_delay_s': 1.0,
    'run_duration_s': 60.0,
    'voltage_scaling_factor': 1.0,
}


def _params(config):
    """Read the ``synthetic`` config block, applying defaults."""
    p = dict(_DEFAULTS)
    p.update(config.get('synthetic') or {})
    return p


class SyntheticDataReceiver(DataSourceReceiver):
    """Drop-in synthetic replacement for ``trodesnet.TrodesDataReceiver``.

    Generates LFP / spike / position samples on demand at clock-driven
    rates. ``__next__`` returns None when no sample is due yet, matching
    the non-blocking semantics of the Trodes receiver — the polling main
    loops do not need to know they are reading synthetic data.
    """

    def __init__(self, comm, rank, config, datatype):
        if datatype not in (
            Datatypes.LFP,
            Datatypes.SPIKES,
            Datatypes.LINEAR_POSITION,
        ):
            raise TypeError(f"Invalid datatype {datatype}")
        super().__init__(comm, rank, config, datatype)

        self._p = _params(config)
        self._started = False
        self._stopped = False

        self.ntrode_ids = []

        # Per-stream pacing: we advance a deterministic virtual clock
        # (sample index) from t=0 at activate(), and emit samples as
        # wall-clock catches up. This gives roughly the same delivery
        # cadence as a live acquisition system at the configured rates.
        self._t0_wall = None
        self._next_sample_idx = 0
        if datatype == Datatypes.LFP:
            self._fs = config['sampling_rate']['lfp']
        elif datatype == Datatypes.SPIKES:
            self._fs = config['sampling_rate']['spikes']
        else:  # LINEAR_POSITION
            self._fs = config['sampling_rate']['position']

        self._spike_clock = config['sampling_rate']['spikes']

        # Spike-stream specific: independent Poisson process per ntrode.
        # ``_next_spike_sample[ntid]`` stores the spike-clock sample index
        # at which that ntrode's next spike will fire.
        self._rng = np.random.default_rng(seed=rank * 1009 + int(datatype))
        self._next_spike_sample = {}
        self._mark_dim = self._p['mark_dim']
        self._amp = self._p['mark_amplitude_uv']

    # ------------------------------------------------------------------
    # DataSourceReceiver contract
    # ------------------------------------------------------------------

    def register_datatype_channel(self, channel):
        ntrode_id = int(channel)
        if self.datatype in (Datatypes.LFP, Datatypes.SPIKES):
            if ntrode_id not in self.ntrode_ids:
                self.ntrode_ids.append(ntrode_id)
        # position has no channels

    def activate(self):
        self._t0_wall = time.time()
        self._next_sample_idx = 0
        if self.datatype == Datatypes.SPIKES:
            for ntid in self.ntrode_ids:
                self._schedule_next_spike(ntid, sample_now=0)
        self._started = True
        self.class_log.debug(
            f"Synthetic {self.datatype.name} datastream activated "
            f"({len(self.ntrode_ids)} ntrodes)"
        )

    def deactivate(self):
        self._started = False

    def stop_iterator(self):
        raise StopIteration()

    def __next__(self):
        if not self._started:
            return None

        elapsed = time.time() - self._t0_wall
        if (
            self._p['run_duration_s'] > 0
            and elapsed > self._p['run_duration_s']
            and not self._stopped
        ):
            # one-time log; the supervisor's termination is wired
            # through SyntheticClient.receive() below.
            self._stopped = True

        if self.datatype == Datatypes.LFP:
            return self._next_lfp(elapsed)
        elif self.datatype == Datatypes.SPIKES:
            return self._next_spike(elapsed)
        else:
            return self._next_position(elapsed)

    # ------------------------------------------------------------------
    # Per-datatype generators
    # ------------------------------------------------------------------

    def _next_lfp(self, elapsed):
        target_idx = int(elapsed * self._fs)
        if target_idx < self._next_sample_idx:
            return None
        idx = self._next_sample_idx
        self._next_sample_idx += 1
        # white-ish noise sized to (num_channels,), scaled the same way
        # TrodesDataReceiver does (raw * voltage_scaling_factor)
        n = max(1, len(self.ntrode_ids))
        raw = self._rng.standard_normal(n) * 200.0  # ~uV range pre-scale
        data = raw * self._p['voltage_scaling_factor']
        local_ts = idx  # LFP uses spike-clock timestamps in real Trodes;
        # at fs_lfp=1500, fs_spike=30000 the ratio is 20, but downstream
        # only cares about monotonicity within a stream, so use idx.
        system_ts = time.time_ns()
        return LFPPoint(
            local_ts,
            list(self.ntrode_ids),
            data,
            system_ts,
            time.time_ns(),
        )

    def _next_spike(self, elapsed):
        if not self.ntrode_ids:
            return None
        spike_sample_now = int(elapsed * self._spike_clock)
        # Find any ntrode whose next-spike sample has arrived.
        for ntid in self.ntrode_ids:
            if self._next_spike_sample[ntid] <= spike_sample_now:
                ts = self._next_spike_sample[ntid]
                self._schedule_next_spike(ntid, sample_now=spike_sample_now)
                # mark vector: gaussian around _amp, all channels positive
                samples = (
                    self._rng.standard_normal(self._mark_dim) * 8.0 + self._amp
                ) / self._p['voltage_scaling_factor']
                # SpikePoint.data is later multiplied by voltage_scaling_factor
                # in real Trodes; the encoder reads `max(mark_vec)` so we just
                # need the post-scaling magnitudes to clear `encoder.spk_amp`.
                return SpikePoint(
                    ts,
                    ntid,
                    samples * self._p['voltage_scaling_factor'],
                    time.time_ns(),
                    time.time_ns(),
                )
        return None

    def _next_position(self, elapsed):
        target_idx = int(elapsed * self._fs)
        if target_idx < self._next_sample_idx:
            return None
        idx = self._next_sample_idx
        self._next_sample_idx += 1

        # triangle-wave walk along a single linear segment between 0 and
        # track_length_cm
        L = self._p['track_length_cm']
        v = self._p['walk_speed_cm_s']
        t = elapsed
        period = 2.0 * L / max(v, 1e-6)
        phase = (t % period) / period  # 0..1
        pos_cm = L * (1.0 - abs(2.0 * phase - 1.0))
        # x/y/x2/y2 in "pixel" units — kinematics.scale_factor converts back
        sf = self.config['kinematics']['scale_factor']
        x = pos_cm / sf
        y = 100.0  # constant
        x2 = x + 5.0
        y2 = y
        return CameraModulePoint(
            idx,
            segment=0,
            position=pos_cm,
            x=x,
            y=y,
            x2=x2,
            y2=y2,
            t_recv_data=time.time_ns(),
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _schedule_next_spike(self, ntid, *, sample_now):
        rate = max(self._p['spike_rate_hz'], 1e-6)
        # exponential inter-arrival in seconds → samples
        gap_s = self._rng.exponential(1.0 / rate)
        gap_samples = max(1, int(gap_s * self._spike_clock))
        self._next_spike_sample[ntid] = sample_now + gap_samples


class SyntheticClient(object):
    """Drop-in synthetic replacement for ``trodesnet.TrodesClient``.

    Exposes the same surface used by the supervisor and stim decider:
      * ``set_startup_callback`` / ``set_termination_callback``
      * ``receive`` (called from the supervisor main loop)
      * ``send_statescript_shortcut_message`` (called from stim_decider)

    ``receive`` fires the startup callback once after ``startup_delay_s``
    of wall clock has elapsed, and fires termination once ``run_duration_s``
    has elapsed.
    """

    def __init__(self, config):
        self._startup_callback = utils.nop
        self._termination_callback = utils.nop
        self._p = _params(config)
        self._t0_wall = time.time()
        self._started = False
        self._terminated = False
        # log-only buffer of "shortcut messages" the stim decider would
        # have sent to ECU; useful for asserting in tests later.
        self.sent_shortcuts = []

    def receive(self):
        elapsed = time.time() - self._t0_wall
        if not self._started and elapsed >= self._p['startup_delay_s']:
            self._started = True
            self._startup_callback()
        if (
            self._started
            and not self._terminated
            and self._p['run_duration_s'] > 0
            and elapsed >= self._p['run_duration_s'] + self._p['startup_delay_s']
        ):
            self._terminated = True
            self._termination_callback()

    def send_statescript_shortcut_message(self, val):
        self.sent_shortcuts.append((time.time_ns(), int(val)))

    def set_startup_callback(self, callback):
        self._startup_callback = callback

    def set_termination_callback(self, callback):
        self._termination_callback = callback
