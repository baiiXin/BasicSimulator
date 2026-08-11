# ContactDAT — Usage Guide

ContactDAT provides a standalone Warp-based triangle-mesh contact detector and DAT truncation implementation adapted from the Newton Physics VBD contact path.

This guide intentionally documents only the outer post-processing interface used to truncate a solver-proposed displacement. Internal kernels and geometry helpers are not part of the public API.

## 1. Public imports

Public package API:

```python
from contactdat import (
    PenetrationFreeContact,
    TriMeshCollisionDetector,
)
```

The package-level `contactdat/__init__.py` only re-exports these public classes from `contactdat._src`. The algorithm implementation remains internal.

## 2. Required data

ContactDAT operates on a fixed-topology indexed triangle mesh.

### Vertex positions

```text
shape: (V, 3)
semantic: current vertex positions
runtime dtype: float32
```

Example:

```python
q_np = np.asarray(q_np, dtype=np.float32)
q = wp.array(q_np, dtype=wp.vec3, device="cuda:0")
```

For Newton-equivalent behavior, use float32 positions and displacements.

### Triangle indices

```text
shape: (T, 3)
dtype: int32
semantic: zero-based vertex indices
```

Example:

```python
faces = np.asarray(faces, dtype=np.int32)
```

The vertex ordering must remain fixed for the lifetime of the detector.

Multiple disconnected garments may be concatenated into one mesh. When concatenating meshes, offset the face indices of later meshes by the preceding vertex counts.

### Proposed displacement

`truncate()` receives a displacement relative to the most recent contact snapshot:

```text
shape: (V, 3)
runtime dtype: float32
semantic: proposed_position - snapshot_position
```

It is not an absolute target position.

## 3. Initialization

```python
import numpy as np
import warp as wp

from contactdat import (
    PenetrationFreeContact,
    TriMeshCollisionDetector,
)

device = wp.get_device("cuda:0")

q0_np = np.asarray(q0_np, dtype=np.float32)
faces = np.asarray(faces, dtype=np.int32)

q0 = wp.array(
    q0_np,
    dtype=wp.vec3,
    device=device,
)

detector = TriMeshCollisionDetector(
    q0,
    faces,
    device=device,

    # Validated production settings used in the Newton parity tests.
    vertex_collision_buffer_pre_alloc=64,
    vertex_collision_buffer_max_alloc=256,
    edge_collision_buffer_pre_alloc=64,
    edge_collision_buffer_max_alloc=256,

    topological_contact_filter_threshold=2,
    auto_resize=True,
)

contact = PenetrationFreeContact(
    detector,

    particle_count=q0_np.shape[0],
    edge_count=detector.edge_count,
    device=device,

    contact_margin=0.05,
    conservative_bound_relaxation=0.85,

    # Disable rest-distance exclusion if it is not needed.
    rest_q=None,
    rest_exclusion_radius=0.0,
)
```

The detector is topology-dependent and should normally be constructed once.

## 4. One solver iteration

Assume an external simulator or optimizer produces a proposed position `q_proposed`.

First begin a contact epoch from the currently accepted state:

```python
contact.begin(q_current)
```

`begin()` performs the contact detection associated with this snapshot and resets the accumulated displacement reference.

Compute the proposed displacement relative to that same snapshot:

```python
dx_np = q_proposed_np.astype(np.float32) - q_current.numpy()

dx = wp.array(
    dx_np,
    dtype=wp.vec3,
    device=device,
)
```

Then truncate it:

```python
dx_safe, q_safe, truncation_t = contact.truncate(dx)
```

Outputs:

```text
dx_safe
    shape (V, 3)
    accepted/truncated displacement

q_safe
    shape (V, 3)
    snapshot + dx_safe

truncation_t
    shape (V,)
    per-vertex DAT truncation coefficient
```

The accepted simulation state is then:

```python
q_current = q_safe
```

## 5. Re-detecting during iterative solving

If the solver wants to continue iterating from the newly accepted state, start a new collision epoch before applying a displacement relative to that new state:

```python
contact.redetect(q_current)

dx = proposed_next - q_current
dx_safe, q_safe, truncation_t = contact.truncate(dx)
```

`redetect(q)` is the explicit form used when the current accepted configuration has changed and new VT/EE candidates are required.

A common iterative structure is therefore:

```python
q_current = q_initial

for iteration in range(num_iterations):
    contact.begin(q_current)

    q_proposed = solver_iteration(q_current)

    dx = q_proposed - q_current

    dx_safe, q_safe, truncation_t = contact.truncate(dx)

    q_current = q_safe
```

If several displacement updates intentionally share the same collision snapshot and candidate set, call `begin()` only at the start of that collision epoch. Otherwise, re-detect from the newly accepted state.

## 6. Important semantics

### The snapshot and displacement must match

After:

```python
contact.begin(q_snapshot)
```

the displacement passed to:

```python
contact.truncate(dx)
```

must satisfy:

```text
dx = q_proposed - q_snapshot
```

Do not compute `dx` relative to some other configuration.

### Initial penetration is not repaired

ContactDAT reproduces Newton's DAT behavior. It is a displacement truncation mechanism, not a penetration-repair solver.

The state supplied to `begin()` should therefore be non-intersecting if the intended invariant is to remain non-intersecting.

### Repeated pursuit of an infeasible target

If the proposed target lies across a contact barrier, repeated DAT truncation can keep driving the state closer to the barrier. The implementation intentionally preserves Newton's behavior and does not add an additional CCD or repair layer.

### Fixed topology

The detector assumes the same vertex/triangle connectivity after initialization. Rebuild the detector if topology changes.

## 7. Minimal complete example

```python
import numpy as np
import warp as wp

from contactdat import PenetrationFreeContact, TriMeshCollisionDetector

device = wp.get_device("cuda:0")

faces = np.asarray(faces, dtype=np.int32)
q_np = np.asarray(initial_vertices, dtype=np.float32)

q = wp.array(q_np, dtype=wp.vec3, device=device)

detector = TriMeshCollisionDetector(
    q,
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
    particle_count=len(q_np),
    edge_count=detector.edge_count,
    device=device,
    contact_margin=0.05,
    conservative_bound_relaxation=0.85,
    rest_q=None,
    rest_exclusion_radius=0.0,
)

q_current = q

for _ in range(num_solver_iterations):
    contact.begin(q_current)

    proposed_np = run_external_solver(q_current.numpy())
    proposed_np = np.asarray(proposed_np, dtype=np.float32)

    dx_np = proposed_np - q_current.numpy()
    dx = wp.array(dx_np, dtype=wp.vec3, device=device)

    dx_safe, q_safe, truncation_t = contact.truncate(dx)

    q_current = q_safe
```

## 8. Runtime dependency

The standalone runtime does not import Newton.

Newton is required only by parity tests that explicitly compare ContactDAT with the validated Newton implementation.
