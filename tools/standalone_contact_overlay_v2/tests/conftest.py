"""Test configuration for the expected local layout:

tools/
├── newton/                  # cloned/copied Newton repository
├── standalone_contact/      # standalone package
└── tests/                   # this directory
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


TOOLS_ROOT = Path(__file__).resolve().parents[1]
NEWTON_REPO = TOOLS_ROOT / "newton"

# Standalone package lives directly under tools/.
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

# The copied Newton repository contains its Python package at tools/newton/newton/.
if NEWTON_REPO.exists() and str(NEWTON_REPO) not in sys.path:
    sys.path.insert(0, str(NEWTON_REPO))


@pytest.fixture(scope="session")
def contact_test_device():
    """Warp device used by collision tests.

    Override with:
        CONTACT_TEST_DEVICE=cuda:0 pytest -q tests
    Default is CPU so correctness tests can run on any machine.
    """
    import warp as wp

    requested = os.environ.get("CONTACT_TEST_DEVICE", "cpu")
    try:
        device = wp.get_device(requested)
    except Exception as exc:
        pytest.skip(f"Warp device {requested!r} is unavailable: {exc}")
    return device
