# Third-party notices

## Newton Physics

Portions of this standalone implementation are adapted from the Newton Physics
project by The Newton Developers.

- Upstream: `newton-physics/newton`
- License: Apache License 2.0
- Relevant source areas:
  - `newton/_src/utils/mesh.py`
  - `newton/_src/geometry/kernels.py`
  - `newton/_src/geometry/broad_phase_common.py`
  - `newton/_src/solvers/vbd/tri_mesh_collision.py`

ContactDAT modifications remove runtime Newton framework dependencies and retain
NumPy + NVIDIA Warp as the implementation substrate. The checkout under
`third_party/newton/` is used only for parity testing and may advance beyond the
adaptation baseline; reproducible releases should pin its exact revision.
