from __future__ import annotations

from types import SimpleNamespace
import numpy as np


def make_two_parallel_triangles(z_gap: float = 0.01):
    q = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_gap],
            [1.0, 0.0, z_gap],
            [0.0, 1.0, z_gap],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return q, faces


def make_stacked_triangles(num_layers: int = 6, dz: float = 0.005):
    vertices = []
    faces = []
    base = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    for layer in range(num_layers):
        start = 3 * layer
        p = base.copy()
        p[:, 2] = layer * dz
        vertices.append(p)
        faces.append([start, start + 1, start + 2])
    return np.concatenate(vertices, axis=0), np.asarray(faces, dtype=np.int32)


def paired_rows(buffer, offsets, counts, sizes):
    """Return each [owner, other] buffer row as a sorted tuple of `other` ids."""
    flat = buffer.numpy()
    offsets_np = offsets.numpy()
    counts_np = counts.numpy()
    sizes_np = sizes.numpy()
    rows = []
    for i in range(len(counts_np)):
        n = min(int(counts_np[i]), int(sizes_np[i]))
        start = int(offsets_np[i])
        if n == 0:
            rows.append(())
            continue
        pairs = flat[2 * start : 2 * (start + n)].reshape(-1, 2)
        assert np.all(pairs[:, 0] == i), (i, pairs)
        rows.append(tuple(sorted(int(x) for x in pairs[:, 1])))
    return rows


def scalar_rows(buffer, offsets, counts, sizes):
    """Return each flattened scalar-buffer row as a sorted tuple."""
    flat = buffer.numpy()
    offsets_np = offsets.numpy()
    counts_np = counts.numpy()
    sizes_np = sizes.numpy()
    rows = []
    for i in range(len(counts_np)):
        n = min(int(counts_np[i]), int(sizes_np[i]))
        start = int(offsets_np[i])
        rows.append(tuple(sorted(int(x) for x in flat[start : start + n])))
    return rows


def make_newton_dummy_model(q: np.ndarray, faces: np.ndarray, device):
    """Create just the fields current Newton TriMeshCollisionDetector consumes.

    This deliberately avoids ModelBuilder so parity tests stay small and isolate
    collision detection rather than model construction.
    """
    import warp as wp
    from newton._src.utils.mesh import MeshAdjacency

    q = np.asarray(q, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)

    adjacency = MeshAdjacency(tri_indices=faces).init_vertex_adjacency(len(q))
    edge_indices_np = np.asarray(adjacency.edge_indices, dtype=np.int32)

    model = SimpleNamespace()
    model.device = device
    model.particle_count = len(q)
    model.tri_count = len(faces)
    model.edge_count = len(edge_indices_np)
    model.particle_q = wp.array(q, dtype=wp.vec3, device=device)
    model.tri_indices = wp.array(faces, dtype=wp.int32, device=device)
    model.edge_indices = wp.array(edge_indices_np, dtype=wp.int32, device=device)
    model.particle_world = wp.zeros(shape=(len(q),), dtype=wp.int32, device=device)
    model.world_count = 1
    model.soft_mesh_adjacency = adjacency
    return model
