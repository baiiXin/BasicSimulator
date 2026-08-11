# ContactDAT

ContactDAT is a standalone NVIDIA Warp implementation of triangle-mesh contact
detection and DAT displacement truncation, adapted from the Newton Physics VBD
contact path.

Runtime use depends on NumPy and Warp, not Newton. A local Newton checkout is
used only by the parity regressions.

ContactDAT intentionally preserves the validated Newton behavior. It is not a
strict CCD implementation, does not repair initial penetration, and does not
claim a universal penetration-free guarantee.

## Installation

ContactDAT requires Python 3.10 or newer.

```bash
python -m pip install -e .
```

Install the test and real-data tooling when needed:

```bash
python -m pip install -e ".[test,real]"
```

## Public API

The public package exposes two classes:

```python
from contactdat import PenetrationFreeContact, TriMeshCollisionDetector
```

A minimal contact epoch is:

```python
detector = TriMeshCollisionDetector(
    q_current,
    faces,
    device=device,
    vertex_collision_buffer_pre_alloc=64,
    vertex_collision_buffer_max_alloc=256,
    edge_collision_buffer_pre_alloc=64,
    edge_collision_buffer_max_alloc=256,
    topological_contact_filter_threshold=2,
    auto_resize=True,
)

contact = PenetrationFreeContact(
    detector,
    particle_count=len(q_current),
    edge_count=detector.edge_count,
    device=device,
    contact_margin=0.05,
    conservative_bound_relaxation=0.85,
)

contact.begin(q_current)
dx_safe, q_safe, truncation_t = contact.truncate(q_proposed - q_current)
```

The displacement passed to `truncate()` must be relative to the configuration
passed to the most recent `begin()` or `redetect()` call. See
[`docs/usage.md`](docs/usage.md) for data formats and complete examples.

## Tests

The normal test command runs the Newton-independent unit suite:

```bash
python -m pytest -q
```

Use `CONTACT_TEST_DEVICE=cuda:0` to select CUDA; the default is CPU.

Newton parity tests expect a compatible checkout at `third_party/newton/`:

```bash
python -m pytest -q tests/parity/test_collision_against_newton.py \
    tests/parity/test_dat_against_newton.py \
    tests/parity/test_topology_against_newton.py
```

The final real-case parity script compares candidate sets and DAT outputs for
the three validated garment cases:

```bash
python tests/parity/final_newton_consistency_test.py \
    RAW_SEQUENCE_ROOT PROCESSED_SEQUENCE_ROOT
```

## Real-data regressions

Real regressions require a user-supplied D-LAYERS sample and the optional
`real` dependencies.

```bash
python tests/real/frame0_residual_vs_truncation.py RAW_SEQUENCE_ROOT

python tests/real/frame63_known_failure.py \
    RAW_SEQUENCE_ROOT PROCESSED_SEQUENCE_ROOT
```

The frame-63 regression passes only when micro-steps 1 through 14 remain
non-intersecting and micro-step 15 reproduces the known intersection. This is
a known-behavior regression, not a failing test.

Do not redistribute D-LAYERS-derived arrays until the dataset license has been
checked. See [`tests/real/README.md`](tests/real/README.md).

## Repository layout

```text
contactdat/                     importable public package
contactdat/_src/                internal implementation
tests/unit/                     Newton-independent tests
tests/parity/                   Newton comparison tests
tests/real/                     selected real-data regressions
benchmarks/                     performance scripts
examples/sequence_postprocess/  sequence post-processing tools
experiments/archive/            retained diagnostics
third_party/newton/             local parity-only Newton checkout
```

## License and attribution

ContactDAT is distributed under Apache-2.0. Newton-adapted source files retain
their SPDX notices. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for
upstream attribution.
