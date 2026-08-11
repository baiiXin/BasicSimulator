# Third-party notices

## Newton Physics

Portions of this standalone implementation are adapted from the Newton Physics
project by The Newton Developers.

Upstream project: `newton-physics/newton`
License: Apache License 2.0

Relevant upstream source areas:
- `newton/_src/utils/mesh.py`
- `newton/_src/geometry/kernels.py`
- `newton/_src/geometry/broad_phase_common.py`
- `newton/_src/solvers/vbd/tri_mesh_collision.py`

The standalone modifications remove the Newton `Model`/`State` framework
dependency and keep NumPy + NVIDIA Warp as the implementation substrate.
