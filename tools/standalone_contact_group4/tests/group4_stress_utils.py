
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TwoLayerGrid:
    vertices: np.ndarray
    faces: np.ndarray
    edges: np.ndarray
    top_vertex_ids: np.ndarray
    bottom_vertex_ids: np.ndarray


def unique_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)

    e = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    return np.asfortranarray(e.astype(np.int32, copy=False))


def _single_grid(n: int, spacing: float, z: float):
    xs = np.arange(n, dtype=np.float64) * spacing
    ys = np.arange(n, dtype=np.float64) * spacing
    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    vertices = np.stack(
        [xx.reshape(-1), yy.reshape(-1), np.full(n * n, z)],
        axis=1,
    )

    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            v00 = j * n + i
            v10 = v00 + 1
            v01 = v00 + n
            v11 = v01 + 1

            # Consistent diagonal, two triangles per quad.
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    return vertices, np.asarray(faces, dtype=np.int32)


def make_two_layer_grid(
    *,
    n: int = 8,
    spacing: float = 0.05,
    gap: float = 0.02,
) -> TwoLayerGrid:
    """Build two disconnected, parallel triangle grids.

    Bottom layer: z=0.
    Top layer:    z=gap.

    Both layers are individually manifold open sheets and initially disjoint.
    """
    vb, fb = _single_grid(n, spacing, 0.0)
    vt, ft = _single_grid(n, spacing, gap)

    offset = len(vb)
    ft = ft + offset

    vertices = np.vstack([vb, vt]).astype(np.float32)
    faces = np.vstack([fb, ft]).astype(np.int32)
    edges = unique_edges_from_faces(faces)

    bottom_ids = np.arange(len(vb), dtype=np.int32)
    top_ids = np.arange(offset, offset + len(vt), dtype=np.int32)

    return TwoLayerGrid(
        vertices=vertices,
        faces=faces,
        edges=edges,
        top_vertex_ids=top_ids,
        bottom_vertex_ids=bottom_ids,
    )


def make_center_punch_displacement(
    mesh: TwoLayerGrid,
    *,
    n: int,
    downward_distance: float,
    center_half_width: int = 1,
) -> np.ndarray:
    """Move a square patch of the top layer through the bottom layer.

    The outer top layer remains fixed, so the candidate configuration contains
    triangles that cut through the bottom sheet instead of merely teleporting
    the whole top sheet to the other side.
    """
    dx = np.zeros_like(mesh.vertices, dtype=np.float32)

    c = (n - 1) / 2.0
    selected = []

    top_offset = int(mesh.top_vertex_ids[0])
    for j in range(n):
        for i in range(n):
            if (
                abs(i - c) <= center_half_width
                and abs(j - c) <= center_half_width
            ):
                selected.append(top_offset + j * n + i)

    selected = np.asarray(selected, dtype=np.int32)
    dx[selected, 2] = -float(downward_distance)
    return dx


def ipc_has_intersections(vertices: np.ndarray, edges: np.ndarray, faces: np.ndarray) -> bool:
    """Static self-intersection oracle using IPC Toolkit.

    Runtime solver code must not call this function. This is for tests only.
    """
    import ipctk

    # Current IPC Toolkit Python bindings request float64/int32 matrices;
    # Fortran layout is the safest common denominator across versions.
    V = np.asfortranarray(vertices, dtype=np.float64)
    E = np.asfortranarray(edges, dtype=np.int32)
    F = np.asfortranarray(faces, dtype=np.int32)

    collision_mesh = ipctk.CollisionMesh(V, E, F)
    return bool(ipctk.has_intersections(collision_mesh, V))
