# standalone_contact v2 overlay

This overlay is arranged for the following local layout:

```text
tools/
├── newton/                  # copied Newton repository
├── standalone_contact/      # copy/replace with this overlay's package
└── tests/                   # copy/merge this overlay's tests
```

The old `standalone_contact_v1/` directory can be kept temporarily for comparison,
but the tests import `tools/standalone_contact/`, not `standalone_contact_v1/`.

## What changed from v1

- Restored Newton-compatible `triangle -> colliding vertices` reverse VT index.
- Added `get_triangle_colliding_vertices*()` Warp accessors.
- Preserved the four overflow flags: VT / reverse-triangle / EE / tri-tri.
- Added optional standalone auto-resize + rerun for VT, reverse-triangle, and EE buffers.
- Added local Newton parity tests using the adjacent `tools/newton/` source tree.
- Added block-size equivalence test and a CUDA `block_dim` benchmark.

## Important current-Newton behavior

Current Newton `SolverVBD` uses fixed particle contact buffers. Its public solver
notes explicitly state that `particle_vertex_contact_buffer_size` and
`particle_edge_contact_buffer_size` are not dynamically resized at runtime.

Therefore this standalone package defaults to:

```python
auto_resize=False
```

for the closest hot-path behavior. Set `auto_resize=True` when you want robust
resize-and-retry behavior. That mode performs a GPU->host overflow check, so it
introduces a synchronization point and should not be enabled blindly in a
performance-critical inner loop.

## Install / apply

Copy the two directories into `tools/`:

```text
standalone_contact/  -> tools/standalone_contact/
tests/               -> tools/tests/
```

After verifying v2, you can archive/delete `tools/standalone_contact_v1/`.

## Run tests

From the `tools/` directory:

```bash
pytest -q tests
```

CPU explicitly:

```bash
CONTACT_TEST_DEVICE=cpu pytest -q tests
```

CUDA:

```bash
CONTACT_TEST_DEVICE=cuda:0 pytest -q tests
```

PowerShell:

```powershell
$env:CONTACT_TEST_DEVICE="cuda:0"
pytest -q tests
```

The important tests are:

```text
test_no_newton_dependency.py       production package has no Newton import
test_topology.py                   topology + exact local-Newton topology parity
test_collision_smoke.py            VT / EE / reverse VT / rest exclusion
test_collision_against_newton.py   candidate-set parity against tools/newton
test_collision_auto_resize.py       forced overflow -> grow -> rerun
test_block_dim_equivalence.py       16/32/64 scheduling gives same candidates
```

## block_dim benchmark

Run on CUDA:

```bash
CONTACT_TEST_DEVICE=cuda:0 python tests/benchmark_block_dim.py
```

This compares `16, 32, 64, 128, 256` on a two-layer cloth-like grid. Keep
`auto_resize=False` while benchmarking, otherwise the host overflow checks can
dominate small timings.

## Typical use

```python
from standalone_contact import TriMeshCollisionDetector

collision = TriMeshCollisionDetector(
    q_rest,
    faces,
    device="cuda:0",
    record_triangle_contacting_vertices=True,
    vertex_collision_buffer_pre_alloc=32,
    triangle_collision_buffer_pre_alloc=64,
    edge_collision_buffer_pre_alloc=64,
    topological_contact_filter_threshold=2,
    collision_detection_block_size=16,
    auto_resize=False,   # closest to current Newton SolverVBD behavior
)

info = collision.detect(
    q=q_current,
    margin=contact_margin,
    rest_q=q_rest,
    rest_exclusion_radius=rest_exclusion_radius,
)
```

For a robustness/debug run:

```python
info = collision.detect(
    q=q_current,
    margin=contact_margin,
    rest_q=q_rest,
    rest_exclusion_radius=rest_exclusion_radius,
    auto_resize=True,
)
```

If auto-resize replaces buffers, `collision.collision_info` is rebuilt to point
at the new arrays. Any *external device array containing an older copy* of the
struct must also be recreated; this matters later when DAT stores
`TriMeshCollisionInfo` inside another Warp array.
