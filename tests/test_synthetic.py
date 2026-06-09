"""Unit tests for realtime_decoder.synthetic.

These tests don't spin up MPI; they construct receivers directly with a
fake comm and verify the per-datatype generators produce the right
shapes and types.
"""

import time

import numpy as np
import pytest

from realtime_decoder import synthetic, datatypes


@pytest.fixture
def base_config():
    return {
        'sampling_rate': {'spikes': 30000, 'lfp': 1500, 'position': 30},
        'kinematics': {'scale_factor': 0.25},
        'synthetic': {
            'spike_rate_hz': 1000.0,       # high so we see spikes fast
            'mark_dim': 4,
            'mark_amplitude_uv': 120.0,
            'track_length_cm': 40.0,
            'walk_speed_cm_s': 20.0,
            'startup_delay_s': 0.0,
            'run_duration_s': 5.0,
            'voltage_scaling_factor': 0.195,
        },
    }


class _FakeComm:
    """Minimal stub: receivers only need .Get_rank if anything; not used in __next__."""
    pass


def test_receiver_rejects_unknown_datatype(base_config):
    with pytest.raises(TypeError):
        synthetic.SyntheticDataReceiver(_FakeComm(), 0, base_config, datatype=999)


def test_lfp_receiver_emits_correct_shape(base_config):
    rx = synthetic.SyntheticDataReceiver(_FakeComm(), 0, base_config, datatypes.Datatypes.LFP)
    rx.register_datatype_channel(1)
    rx.register_datatype_channel(2)
    rx.activate()
    # poll until we get a sample (LFP at 1500hz, so first sample ~immediate)
    deadline = time.time() + 1.0
    sample = None
    while time.time() < deadline:
        sample = rx.__next__()
        if sample is not None:
            break
    assert sample is not None, "no LFP sample within 1s"
    assert isinstance(sample, datatypes.LFPPoint)
    assert sample.data.shape == (2,)


def test_lfp_receiver_returns_none_before_activate(base_config):
    rx = synthetic.SyntheticDataReceiver(_FakeComm(), 0, base_config, datatypes.Datatypes.LFP)
    rx.register_datatype_channel(1)
    assert rx.__next__() is None


def test_spike_receiver_emits_marks_above_amp_threshold(base_config):
    rx = synthetic.SyntheticDataReceiver(_FakeComm(), 0, base_config, datatypes.Datatypes.SPIKES)
    rx.register_datatype_channel(7)
    rx.activate()
    deadline = time.time() + 1.0
    spike = None
    while time.time() < deadline:
        spike = rx.__next__()
        if spike is not None:
            break
    assert spike is not None, "no spike within 1s at 1khz rate"
    assert isinstance(spike, datatypes.SpikePoint)
    assert spike.elec_grp_id == 7
    assert spike.data.shape == (base_config['synthetic']['mark_dim'],)
    # marks should clear a reasonable amplitude threshold post-scaling
    assert float(np.max(spike.data)) > 50.0


def test_position_receiver_walks_within_bounds(base_config):
    base_config['synthetic']['walk_speed_cm_s'] = 200  # fast walk so we cover range quickly
    rx = synthetic.SyntheticDataReceiver(_FakeComm(), 0, base_config, datatypes.Datatypes.LINEAR_POSITION)
    rx.activate()
    L = base_config['synthetic']['track_length_cm']
    deadline = time.time() + 2.0
    positions = []
    while time.time() < deadline and len(positions) < 30:
        p = rx.__next__()
        if p is not None:
            positions.append(p.position)
            assert isinstance(p, datatypes.CameraModulePoint)
    assert positions, "no position samples emitted"
    assert min(positions) >= 0.0
    assert max(positions) <= L + 1e-6


def test_synthetic_client_fires_startup_callback(base_config):
    base_config['synthetic']['startup_delay_s'] = 0.05
    base_config['synthetic']['run_duration_s'] = 0  # disable auto-term
    client = synthetic.SyntheticClient(base_config)

    calls = {'startup': 0, 'term': 0}
    client.set_startup_callback(lambda: calls.__setitem__('startup', calls['startup'] + 1))
    client.set_termination_callback(lambda: calls.__setitem__('term', calls['term'] + 1))

    # before delay elapses, no callback
    client.receive()
    assert calls['startup'] == 0

    time.sleep(0.1)
    client.receive()
    assert calls['startup'] == 1
    # subsequent calls should not refire startup
    client.receive()
    assert calls['startup'] == 1


def test_synthetic_client_records_shortcut_messages(base_config):
    client = synthetic.SyntheticClient(base_config)
    client.send_statescript_shortcut_message(22)
    client.send_statescript_shortcut_message(14)
    assert [v for _, v in client.sent_shortcuts] == [22, 14]
