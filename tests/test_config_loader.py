"""Unit tests for realtime_decoder.config_loader."""

import os
import textwrap

import pytest

from realtime_decoder import config_loader


# ---------------------------------------------------------------------------
# deep_merge
# ---------------------------------------------------------------------------


def test_deep_merge_nested_dicts():
    base = {'a': 1, 'b': {'c': 2, 'd': 3}}
    over = {'b': {'d': 99, 'e': 4}, 'f': 5}
    out = config_loader.deep_merge(base, over)
    assert out == {'a': 1, 'b': {'c': 2, 'd': 99, 'e': 4}, 'f': 5}
    # base must not be mutated
    assert base == {'a': 1, 'b': {'c': 2, 'd': 3}}


def test_deep_merge_lists_are_replaced_not_extended():
    base = {'xs': [1, 2, 3]}
    over = {'xs': [9]}
    assert config_loader.deep_merge(base, over) == {'xs': [9]}


def test_deep_merge_scalar_overrides_dict():
    base = {'k': {'nested': True}}
    over = {'k': 'flat'}
    assert config_loader.deep_merge(base, over) == {'k': 'flat'}


# ---------------------------------------------------------------------------
# load_config (extends, validation)
# ---------------------------------------------------------------------------


def _write(p, contents):
    p.write_text(textwrap.dedent(contents).lstrip())


@pytest.fixture
def minimal_valid_config():
    """Returns a function that writes a minimal valid config to a path."""
    def _write_to(path):
        _write(path, """
            algorithm: clusterless_decoder
            datasource: synthetic
            sampling_rate: {spikes: 30000, lfp: 1500, position: 30}
            files: {output_dir: /tmp/x, prefix: x}
            kinematics: {smoothing_filter: [1.0]}
            rank:
              supervisor: [0]
              decoders: [1]
              encoders: [2]
              ripples: [3]
              gui: [4]
            decoder_assignment: {1: [1]}
            ripples: {filter: {}, smoothing_filter: {}, threshold: {}}
            encoder:
              mark_dim: 4
              bufsize: 100
              spk_amp: 60
              position:
                lower: 0
                upper: 10
                num_bins: 10
                arm_ids: [0]
                arm_coords: [[0, 9]]
            decoder:
              bufsize: 100
              cred_int_bufsize: 10
              time_bin: {samples: 180, delay_samples: 180}
        """)
    return _write_to


def test_load_config_without_extends(tmp_path, minimal_valid_config):
    p = tmp_path / 'cfg.yml'
    minimal_valid_config(p)
    cfg = config_loader.load_config(str(p))
    assert cfg['algorithm'] == 'clusterless_decoder'


def test_load_config_with_extends_merges_parent(tmp_path, minimal_valid_config):
    parent = tmp_path / 'base.yml'
    child = tmp_path / 'child.yml'
    minimal_valid_config(parent)
    _write(child, """
        _extends: base.yml
        algorithm: clusterless_classifier
        files: {output_dir: /tmp/child, prefix: child}
    """)
    cfg = config_loader.load_config(str(child))
    # child overrides
    assert cfg['algorithm'] == 'clusterless_classifier'
    assert cfg['files']['output_dir'] == '/tmp/child'
    # inherited from parent
    assert cfg['sampling_rate']['spikes'] == 30000


def test_load_config_circular_extends_raises(tmp_path):
    a = tmp_path / 'a.yml'
    b = tmp_path / 'b.yml'
    _write(a, '_extends: b.yml\n')
    _write(b, '_extends: a.yml\n')
    with pytest.raises(config_loader.ConfigError, match='Circular'):
        config_loader.load_config(str(a))


def test_validate_missing_required_key_raises(tmp_path):
    p = tmp_path / 'cfg.yml'
    _write(p, 'algorithm: clusterless_decoder\n')
    with pytest.raises(config_loader.ConfigError) as exc:
        config_loader.load_config(str(p))
    msg = str(exc.value)
    assert 'rank' in msg
    assert 'sampling_rate' in msg


def test_validate_unknown_algorithm_raises(tmp_path, minimal_valid_config):
    p = tmp_path / 'cfg.yml'
    minimal_valid_config(p)
    # mutate the file to inject a bad algorithm
    src = p.read_text().replace('clusterless_decoder', 'bogus_algo')
    p.write_text(src)
    with pytest.raises(config_loader.ConfigError, match='bogus_algo'):
        config_loader.load_config(str(p))


def test_validate_decoder_rank_missing_assignment(tmp_path, minimal_valid_config):
    p = tmp_path / 'cfg.yml'
    minimal_valid_config(p)
    # remove the decoder_assignment line so rank 1 has no entry
    src = p.read_text().replace('decoder_assignment: {1: [1]}', 'decoder_assignment: {}')
    p.write_text(src)
    with pytest.raises(config_loader.ConfigError, match='decoder_assignment'):
        config_loader.load_config(str(p))


def test_validate_mark_dim_mismatch_between_encoder_and_synthetic(tmp_path, minimal_valid_config):
    p = tmp_path / 'cfg.yml'
    minimal_valid_config(p)
    src = p.read_text().rstrip() + '\nsynthetic: {mark_dim: 99}\n'
    p.write_text(src)
    with pytest.raises(config_loader.ConfigError, match='mark_dim'):
        config_loader.load_config(str(p))
