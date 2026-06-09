"""Config loader: YAML defaults inheritance + startup validation.

The historical pattern in this repo is one ~200-line YAML per animal,
near-duplicated across the colony. That breeds drift: a parameter
correctly tuned in `SC79_nTrode16.yml` quietly differs from the same
parameter in `SC80_nTrode16.yml`, and there is no single source of
truth for "what's the standard value of X."

This module provides two small affordances:

1. Optional ``_extends`` key that loads a parent YAML and deep-merges it
   under the current file. Per-animal files become *overrides* on top
   of a shared ``defaults.yml`` instead of full standalone configs.

2. Startup validation. Today, common operator mistakes (missing
   ``rank.supervisor``, unknown ``algorithm``, missing ``encoder.mark_dim``)
   surface as ``IndexError``/``KeyError``/``NotImplementedError`` deep
   inside a worker process — easy to lose in MPI log noise. ``validate``
   raises a single clear ``ConfigError`` *before* the MPI run starts.

The loader is backward compatible: existing configs without ``_extends``
load identically to ``yaml.safe_load`` (modulo validation).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

# PyYAML — stdlib-only dependency. (Python 3.7+ preserves insertion order
# in plain dicts, so no need for oyaml just to read configs.)
import yaml


class ConfigError(ValueError):
    """Raised when a config fails validation or cannot be loaded."""


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config, resolving ``_extends`` chains and validating.

    ``_extends`` may be a single path or a list of paths. Each parent is
    loaded recursively (parents may themselves ``_extends``) and merged
    in order, with the current file's keys taking precedence.

    Relative paths in ``_extends`` resolve relative to the file that
    declares them.
    """
    cfg = _load_with_inheritance(path, _seen=set())
    validate(cfg)
    return cfg


def _load_with_inheritance(path: str, *, _seen: set) -> Dict[str, Any]:
    abspath = os.path.abspath(path)
    if abspath in _seen:
        raise ConfigError(
            f"Circular `_extends` chain detected involving {abspath}"
        )
    _seen = _seen | {abspath}

    with open(abspath, 'r') as f:
        raw = yaml.safe_load(f) or {}

    extends = raw.pop('_extends', None)
    if extends is None:
        return raw

    if isinstance(extends, str):
        parents: List[str] = [extends]
    elif isinstance(extends, list):
        parents = list(extends)
    else:
        raise ConfigError(
            f"`_extends` in {abspath} must be a string or list, got {type(extends).__name__}"
        )

    here = os.path.dirname(abspath)
    merged: Dict[str, Any] = {}
    for parent in parents:
        parent_path = parent if os.path.isabs(parent) else os.path.join(here, parent)
        merged = deep_merge(merged, _load_with_inheritance(parent_path, _seen=_seen))
    return deep_merge(merged, raw)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, preferring ``override``.

    Nested dicts merge key-by-key. Lists and scalars are replaced, not
    appended — this matches operator intuition ("override X" means
    "replace X," not "extend X").
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


# Required top-level keys and the rough shape we expect. Kept as plain
# code rather than a third-party schema lib so this module has no new
# install-time dependencies; the checks below are cheap and the error
# messages are deliberately operator-friendly.
_REQUIRED_TOP: Tuple[str, ...] = (
    'rank',
    'algorithm',
    'sampling_rate',
    'files',
    'encoder',
    'decoder',
    'ripples',
    'kinematics',
)
_KNOWN_ALGORITHMS = ('clusterless_decoder', 'clusterless_classifier')
_KNOWN_DATASOURCES = ('trodes', 'synthetic')
_REQUIRED_RANK_ROLES = ('supervisor', 'decoders', 'encoders', 'ripples', 'gui')
_REQUIRED_SAMPLING = ('spikes', 'lfp', 'position')
_REQUIRED_FILES = ('output_dir', 'prefix')
_REQUIRED_ENCODER = ('mark_dim', 'bufsize', 'spk_amp', 'position')
_REQUIRED_ENCODER_POSITION = ('lower', 'upper', 'num_bins', 'arm_ids', 'arm_coords')
_REQUIRED_DECODER = ('bufsize', 'time_bin', 'cred_int_bufsize')


def validate(cfg: Dict[str, Any]) -> None:
    """Raise ``ConfigError`` with a clear message if ``cfg`` is malformed."""
    errors: List[str] = []

    for k in _REQUIRED_TOP:
        if k not in cfg:
            errors.append(f"missing required top-level key '{k}'")

    if cfg.get('algorithm') and cfg['algorithm'] not in _KNOWN_ALGORITHMS:
        errors.append(
            f"algorithm={cfg['algorithm']!r} is not one of {_KNOWN_ALGORITHMS}"
        )

    ds = cfg.get('datasource', 'trodes')
    if ds not in _KNOWN_DATASOURCES:
        errors.append(
            f"datasource={ds!r} is not one of {_KNOWN_DATASOURCES}"
        )

    rank = cfg.get('rank', {})
    if isinstance(rank, dict):
        for role in _REQUIRED_RANK_ROLES:
            v = rank.get(role)
            if v is None:
                errors.append(f"rank.{role} is missing")
            elif not isinstance(v, list) or not v:
                errors.append(f"rank.{role} must be a non-empty list of ints, got {v!r}")
        for role in ('supervisor', 'gui'):
            if isinstance(rank.get(role), list) and len(rank[role]) != 1:
                errors.append(
                    f"rank.{role} must contain exactly one rank, got {rank[role]!r}"
                )
    else:
        errors.append(f"rank must be a mapping, got {type(rank).__name__}")

    sr = cfg.get('sampling_rate', {})
    if isinstance(sr, dict):
        for k in _REQUIRED_SAMPLING:
            if k not in sr:
                errors.append(f"sampling_rate.{k} is missing")
            elif not isinstance(sr[k], (int, float)) or sr[k] <= 0:
                errors.append(f"sampling_rate.{k} must be a positive number, got {sr[k]!r}")

    files = cfg.get('files', {})
    if isinstance(files, dict):
        for k in _REQUIRED_FILES:
            if not files.get(k):
                errors.append(f"files.{k} is missing or empty")

    enc = cfg.get('encoder', {})
    if isinstance(enc, dict):
        for k in _REQUIRED_ENCODER:
            if k not in enc:
                errors.append(f"encoder.{k} is missing")
        pos = enc.get('position', {})
        if isinstance(pos, dict):
            for k in _REQUIRED_ENCODER_POSITION:
                if k not in pos:
                    errors.append(f"encoder.position.{k} is missing")

    dec = cfg.get('decoder', {})
    if isinstance(dec, dict):
        for k in _REQUIRED_DECODER:
            if k not in dec:
                errors.append(f"decoder.{k} is missing")
        tb = dec.get('time_bin', {})
        if isinstance(tb, dict):
            for k in ('samples', 'delay_samples'):
                if k not in tb:
                    errors.append(f"decoder.time_bin.{k} is missing")

    # Cross-field: each decoder rank must be a key in decoder_assignment.
    dec_ranks = (cfg.get('rank') or {}).get('decoders') or []
    assignment = cfg.get('decoder_assignment') or {}
    if isinstance(assignment, dict):
        for r in dec_ranks:
            if r not in assignment:
                errors.append(
                    f"decoder_assignment is missing an entry for rank {r}"
                )

    # Cross-field: encoder.mark_dim must match across encoder and any
    # synthetic-source override.
    syn = cfg.get('synthetic') or {}
    if isinstance(enc, dict) and isinstance(syn, dict):
        if 'mark_dim' in syn and 'mark_dim' in enc and syn['mark_dim'] != enc['mark_dim']:
            errors.append(
                f"synthetic.mark_dim ({syn['mark_dim']}) "
                f"!= encoder.mark_dim ({enc['mark_dim']})"
            )

    if errors:
        bullet = '\n  - '.join(errors)
        raise ConfigError(f"config validation failed:\n  - {bullet}")
