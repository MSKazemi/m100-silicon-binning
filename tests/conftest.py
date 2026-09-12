"""Shared fixtures for the test suite.

Every test module already inserts `analysis/` on `sys.path` for itself, so the only job here is to
do that once for all of them and to offer the two things more than one module needs: the repository
root, and a cached reader for the machine-readable result files.

Run the suite with:  .venv/bin/python -m pytest -q tests/
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / 'analysis'
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

_cache: dict[str, object] = {}


def result(name: str):
    """Load analysis/<name>.json once per session. Skips the test if it is not committed."""
    if name not in _cache:
        p = ANALYSIS / f'{name}.json'
        if not p.exists():
            pytest.skip(f'analysis/{name}.json is not present')
        _cache[name] = json.loads(p.read_text())
    return _cache[name]


@pytest.fixture(scope='session')
def root() -> Path:
    return ROOT


@pytest.fixture(scope='session')
def results():
    """`results('survival_cindex')['cindex']` -- the committed machine-readable results."""
    return result
