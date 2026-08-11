# ContactDAT — Recommended Repository Layout

## Recommended name

**ContactDAT**

- repository directory: `contactdat/`
- Python package: `contactdat`
- internal implementation: `contactdat/_src/`
- public import surface: `contactdat/__init__.py`

Do not use `NewtonDAT` as the public name; the implementation is adapted from Newton Physics but is not an official Newton component.

## Recommended tree

```text
contactdat/                         # repository root; replaces current tools/
├── contactdat/                     # importable Python package
│   ├── __init__.py                 # public re-exports only
│   └── _src/                       # former standalone_contact/
│       ├── __init__.py
│       ├── collision_detector.py
│       ├── collision_kernels.py
│       ├── dat_kernels.py
│       ├── geometry.py
│       ├── mesh_topology.py
│       ├── penetration_free.py
│       └── penetration_free_contact.py
│
├── tests/
│   ├── unit/                       # Newton-independent tests
│   │   ├── conftest.py
│   │   ├── collision_test_utils.py
│   │   ├── group4_stress_utils.py
│   │   ├── test_block_dim_equivalence.py
│   │   ├── test_collision_auto_resize.py
│   │   ├── test_collision_smoke.py
│   │   ├── test_dat_geometry.py
│   │   ├── test_no_newton_dependency.py
│   │   ├── test_penetration_free.py
│   │   ├── test_penetration_free_contact.py
│   │   ├── test_penetration_free_stress.py
│   │   └── test_topology.py
│   │
│   ├── parity/                     # Newton-dependent regression tests
│   │   ├── test_collision_against_newton.py
│   │   ├── test_dat_against_newton.py
│   │   └── final_newton_consistency_test.py
│   │
│   └── real/                       # selected real D-LAYERS regressions
│       ├── README.md
│       ├── frame0_residual_vs_truncation.py
│       └── frame63_known_failure.py
│
├── benchmarks/
│   ├── benchmark_block_dim.py
│   └── ...                         # existing benchmark scripts
│
├── examples/
│   └── sequence_postprocess/
│       ├── process_garment_sequence.py
│       └── check_sequence_collisions.py
│
├── experiments/
│   └── archive/
│       └── frame63_precision/
│           ├── diagnose_frame63_microsteps.py
│           ├── replay_dat_failure.py
│           ├── replay_dat_failure_vs_newton.py
│           ├── replay_collision_detection_vs_newton.py
│           ├── scale_dat_failure_experiment.py
│           └── test_micro15_dat_float64.py
│
├── third_party/
│   └── newton/                     # preferably a pinned Git submodule
│
├── docs/
│   └── usage.md
│
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── README.md
├── pyproject.toml
└── .gitignore
```

## Public `__init__.py`

Keep the public API deliberately small:

```python
from ._src.collision_detector import TriMeshCollisionDetector
from ._src.penetration_free_contact import PenetrationFreeContact

__all__ = [
    "TriMeshCollisionDetector",
    "PenetrationFreeContact",
]
```

This is a packaging-only change and does not alter the validated algorithms.

## Mapping from the current repository

### Move the implementation

```text
standalone_contact/
    ->
contactdat/_src/
```

The root `THIRD_PARTY_NOTICES.md` should live at repository root. SPDX headers on Newton-adapted source files should remain untouched.

### Keep all current unit-test source code

Move the current Newton-independent tests to `tests/unit/`.

Move:

```text
test_collision_against_newton.py
test_dat_against_newton.py
```

to `tests/parity/`.

Move:

```text
final_newton_consistency_test.py
```

to `tests/parity/final_newton_consistency_test.py`.

### Keep exactly two selected real behavior regressions

1. `frame0_residual_vs_truncation.py`

   Derived from the current `tune_sequence_truncation.py`.

   Its purpose is to print/record the relationship:

   ```text
   truncation count K
        ->
   remaining displacement / residual ratio
   ```

   for the frame-0-to-frame-1 real garment transition.

2. `frame63_known_failure.py`

   Derived from the current frame-63 diagnostic/replay scripts.

   Its purpose is to record the validated Newton-equivalent behavior:

   ```text
   micro 1 ... 14 : non-intersecting
   micro 15       : intersecting
   ```

   This should be treated as a known-behavior regression, not as a claim that DAT guarantees CCD-level nonintersection.

### Preserve Newton consistency

Keep `final_newton_consistency_test.py` as the final parity regression.

The validated cases are:

```text
raw frame 0 -> raw frame 1
processed frame 62 -> raw frame 63
saved frame63 failing micro-step
```

The expected parity conditions are:

```text
VT candidate set: exact
EE directed set: exact
EE undirected set: exact

max |t_standalone - t_newton|  = 0
max |dx_standalone - dx_newton| = 0
max |q_standalone - q_newton|   = 0
```

## Files that should not remain in repository root

Generated diagnostics such as:

```text
*.npy
*.csv
*_report.json
collision_detection_parity.json
parity_*.npy
micro15_reference_*.npy
frame63_micro15_*.npy
fail_*.npy
```

should either:

- be moved into a dedicated local fixture/data directory when they are required by a real regression, or
- be excluded by `.gitignore` when they are generated artifacts.

Do not keep transient outputs mixed with source code.

Delete:

```text
__pycache__/
tmp_test.py
```

unless `tmp_test.py` contains something that still needs to be archived.

## Real-data fixtures

For local development, the two real regressions may point to an external D-LAYERS sample.

For public release, do not redistribute D-LAYERS-derived arrays unless the dataset license clearly permits redistribution.

A practical layout is:

```text
tests/real/data/
    # gitignored by default
```

and a `tests/real/README.md` explaining how to place the required sample there.

If redistribution is permitted, create a minimal fixture containing only the required topology and selected frames rather than the full sequence.

## Newton source

Runtime ContactDAT must remain Newton-independent.

For parity testing, prefer:

```text
third_party/newton/
```

as a Git submodule pinned to the exact Newton commit against which ContactDAT was validated.

This gives reproducible parity without making Newton a runtime dependency.

## Test commands

### Core unit tests

Canonical one-line command:

```bash
python -m pytest -q tests/unit
```

This is the command that should be advertised as the normal test command for ContactDAT.

### Newton parity tests

When the pinned Newton checkout is available:

```bash
python -m pytest -q tests/parity
```

### Real regressions

Run the two selected real cases explicitly because they depend on external fixture data:

```bash
python tests/real/frame0_residual_vs_truncation.py ...
python tests/real/frame63_known_failure.py ...
```

Then run the final Newton consistency regression:

```bash
python tests/parity/final_newton_consistency_test.py ...
```

## Suggested README positioning

A concise top-level description:

> ContactDAT is a standalone Warp implementation of triangle-mesh contact detection and DAT displacement truncation, adapted from the Newton Physics VBD contact path. The runtime has no Newton dependency. The repository includes Newton parity tests that reproduce candidate sets and DAT outputs exactly on validated real garment cases.

Avoid describing ContactDAT as a strict CCD or universal penetration-free guarantee. The implementation intentionally preserves Newton's validated behavior.
