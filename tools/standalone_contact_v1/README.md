# standalone_contact v1

Standalone NumPy + NVIDIA Warp extraction of the self-contact collision-detection
chain used by Newton VBD. Production modules do **not** import Newton.

## Included in v1

- `mesh_topology.py`
  - triangle -> Newton-compatible `[o0,o1,v0,v1]` edge topology
  - vertex adjacency CSR
  - n-ring VT and EE topology filters
- `geometry.py`
  - triangle closest point
  - integer upper-bound binary search
- `collision_kernels.py`
  - triangle/edge AABB kernels
  - VT BVH query + narrow phase + rest-pose exclusion + fixed buffer recording
  - EE BVH query + narrow phase + rest-pose exclusion + fixed buffer recording
- `collision_detector.py`
  - `TriMeshCollisionInfo`
  - `TriMeshCollisionDetector`
  - BVH build/refit
  - VT/EE fixed-capacity buffers
  - `detect(...)`

## Intentionally omitted in v1

- Newton `Model` / `State`
- multi-world (`particle_world`, grouped BVH roots, `world_count`)
- triangle->colliding-vertices reverse buffer
- automatic collision-buffer resizing
- DAT truncation (next migration group)

## Typical solver call

```python
detector = TriMeshCollisionDetector(
    q_rest,
    faces,
    device="cuda:0",
    topological_contact_filter_threshold=2,
)

collision_info = detector.detect(
    q=current_q,
    margin=contact_margin,
    rest_q=q_rest,
    rest_exclusion_radius=rest_exclusion_radius,
)
```

For performance, keep `current_q` and `q_rest` as Warp arrays on the target device.
Passing NumPy positions causes an upload.

## Tests

```bash
python tests/test_topology.py
python tests/test_collision_smoke.py
```

`test_collision_smoke.py` requires `warp-lang`.

## Attribution

This implementation was derived from the architecture and algorithms in
`newton-physics/newton`, especially:

- `newton/_src/utils/mesh.py`
- `newton/_src/geometry/kernels.py`
- `newton/_src/geometry/broad_phase_common.py`
- `newton/_src/solvers/vbd/tri_mesh_collision.py`

Newton is licensed under Apache-2.0. The source files in this package retain
the upstream SPDX copyright/license notice and identify the standalone
modifications. Source was inspected from Newton `main` on 2026-08-10.
Pin a specific upstream commit before final bit-for-bit parity benchmarking.
