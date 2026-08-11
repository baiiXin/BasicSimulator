"""Shared pytest configuration for unit and Newton parity tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
UNIT_TESTS = REPO_ROOT / "tests" / "unit"
NEWTON_REPO = REPO_ROOT / "third_party" / "newton"

for import_root in (REPO_ROOT, UNIT_TESTS, NEWTON_REPO):
    if import_root.exists() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


@pytest.fixture(scope="session")
def contact_test_device():
    """Warp device used by collision tests.

    Override with:
        CONTACT_TEST_DEVICE=cuda:0 pytest -q tests/unit
    Default is CPU so correctness tests can run on any machine.
    """
    import warp as wp

    requested = os.environ.get("CONTACT_TEST_DEVICE", "cpu")
    try:
        device = wp.get_device(requested)
    except Exception as exc:
        pytest.skip(f"Warp device {requested!r} is unavailable: {exc}")
    return device
