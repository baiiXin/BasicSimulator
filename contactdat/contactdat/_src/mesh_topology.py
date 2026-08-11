# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Standalone adaptation for Warp/NumPy use without importing Newton.
# Modifications Copyright (c) 2026 Standalone Contact contributors.

from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np


@dataclass
class MeshTopology:
    triangles: np.ndarray
    edge_indices: np.ndarray
    edge_tri_indices: np.ndarray
    tri_edge_indices: np.ndarray
    v_adj_tris: np.ndarray
    v_adj_tris_offsets: np.ndarray
    v_adj_edges: np.ndarray
    v_adj_edges_offsets: np.ndarray


def build_edge_topology(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Newton-compatible edge topology.

    Returns
    -------
    edge_indices:
        int32 [E,4], rows [o0, o1, v0, v1].
    edge_tri_indices:
        int32 [E,2], adjacent triangle ids; -1 on a boundary side.
    tri_edge_indices:
        int32 [T,3], local triangle edge -> global edge id.

    Edge numbering follows first occurrence while traversing triangles in
    row-major order, matching Newton's current MeshAdjacency behavior.
    """
    tris = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    tri_count = tris.shape[0]
    if tri_count == 0:
        return (
            np.empty((0, 4), dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 3), dtype=np.int32),
        )

    # Local slots:
    # 0: edge (v0,v1), opposite v2
    # 1: edge (v1,v2), opposite v0
    # 2: edge (v2,v0), opposite v1
    entry_v0 = np.stack((tris[:, 0], tris[:, 1], tris[:, 2]), axis=1).reshape(-1)
    entry_v1 = np.stack((tris[:, 1], tris[:, 2], tris[:, 0]), axis=1).reshape(-1)
    entry_opp = np.stack((tris[:, 2], tris[:, 0], tris[:, 1]), axis=1).reshape(-1)
    entry_tri = np.repeat(np.arange(tri_count, dtype=np.int32), 3)
    entry_slot = np.tile(np.arange(3, dtype=np.int32), tri_count)

    keys = np.stack((np.minimum(entry_v0, entry_v1), np.maximum(entry_v0, entry_v1)), axis=1)
    _, first_entries, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)

    # np.unique sorts keys; remap edge ids to first-occurrence order.
    first_order = np.argsort(first_entries, kind="stable")
    edge_remap = np.empty(len(first_order), dtype=np.int32)
    edge_remap[first_order] = np.arange(len(first_order), dtype=np.int32)
    entry_edge = edge_remap[inverse]

    edge_count = len(first_order)
    edge_indices = np.full((edge_count, 4), -1, dtype=np.int32)
    edge_tri_indices = np.full((edge_count, 2), -1, dtype=np.int32)
    tri_edge_indices = np.full((tri_count, 3), -1, dtype=np.int32)
    fill_counts = np.zeros(edge_count, dtype=np.int32)

    for k, edge_id_ in enumerate(entry_edge):
        e = int(edge_id_)
        tri = int(entry_tri[k])
        slot = int(entry_slot[k])
        tri_edge_indices[tri, slot] = e

        side = int(fill_counts[e])
        if side == 0:
            edge_indices[e, 0] = entry_opp[k]
            edge_indices[e, 2] = entry_v0[k]
            edge_indices[e, 3] = entry_v1[k]
            edge_tri_indices[e, 0] = tri
        elif side == 1:
            edge_indices[e, 1] = entry_opp[k]
            edge_tri_indices[e, 1] = tri
        else:
            warnings.warn(f"Detected non-manifold edge id={e}", stacklevel=2)
        fill_counts[e] += 1

    return edge_indices, edge_tri_indices, tri_edge_indices


def _records_to_csr(records: list[list[tuple[int, int]]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(records) + 1, dtype=np.int32)
    sizes = np.fromiter((2 * len(row) for row in records), dtype=np.int32, count=len(records))
    np.cumsum(sizes, out=offsets[1:])

    values = np.empty(int(offsets[-1]), dtype=np.int32)
    cursor = 0
    for row in records:
        for element_id, local_slot in row:
            values[cursor] = element_id
            values[cursor + 1] = local_slot
            cursor += 2
    return values, offsets


def build_vertex_adjacency(
    particle_count: int,
    triangles: np.ndarray,
    edge_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build vertex -> triangles and vertex -> edge-records CSR.

    Records are stored as (element_id, local_slot) pairs.
    For edge records, local_slot 0/1 refers to opposite vertices o0/o1,
    and 2/3 refers to actual edge endpoints v0/v1.
    """
    tris = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    edges = np.asarray(edge_indices, dtype=np.int32).reshape(-1, 4)

    tri_records: list[list[tuple[int, int]]] = [[] for _ in range(particle_count)]
    edge_records: list[list[tuple[int, int]]] = [[] for _ in range(particle_count)]

    for tri_id, tri in enumerate(tris):
        for local_slot in range(3):
            v = int(tri[local_slot])
            if 0 <= v < particle_count:
                tri_records[v].append((tri_id, local_slot))

    for edge_id, edge in enumerate(edges):
        for local_slot in range(4):
            v = int(edge[local_slot])
            if 0 <= v < particle_count:
                edge_records[v].append((edge_id, local_slot))

    v_adj_tris, v_adj_tris_offsets = _records_to_csr(tri_records)
    v_adj_edges, v_adj_edges_offsets = _records_to_csr(edge_records)
    return v_adj_tris, v_adj_tris_offsets, v_adj_edges, v_adj_edges_offsets


def _csr_row(values: np.ndarray, offsets: np.ndarray, row: int) -> np.ndarray:
    return values[offsets[row] : offsets[row + 1]]


def one_ring_vertices(
    vertex: int,
    edge_indices: np.ndarray,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
) -> np.ndarray:
    """Vertices connected by a real edge endpoint to `vertex`."""
    rows = _csr_row(v_adj_edges, v_adj_edges_offsets, vertex)
    if rows.size == 0:
        return np.empty(0, dtype=np.int32)

    edge_ids = rows[::2]
    local_slots = rows[1::2]
    edge_ids = edge_ids[local_slots >= 2]
    if edge_ids.size == 0:
        return np.empty(0, dtype=np.int32)

    edges = np.asarray(edge_indices, dtype=np.int32)
    us = edges[edge_ids, 2]
    vs = edges[edge_ids, 3]
    neighbors = np.unique(np.concatenate((us, vs)))
    return neighbors[neighbors != vertex].astype(np.int32, copy=False)


def leq_n_ring_vertices(
    vertex: int,
    edge_indices: np.ndarray,
    n: int,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
) -> np.ndarray:
    visited = {int(vertex)}
    frontier = {int(vertex)}
    for _ in range(n):
        next_frontier: set[int] = set()
        for current in frontier:
            for neighbor in one_ring_vertices(current, edge_indices, v_adj_edges, v_adj_edges_offsets):
                neighbor = int(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        if not next_frontier:
            break
        frontier = next_frontier
    return np.fromiter(sorted(visited), dtype=np.int32)


def set_to_csr(list_of_sets: list[set[int]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(list_of_sets) + 1, dtype=np.int32)
    sizes = np.fromiter((len(s) for s in list_of_sets), dtype=np.int32, count=len(list_of_sets))
    np.cumsum(sizes, out=offsets[1:])

    flat = np.empty(int(offsets[-1]), dtype=np.int32)
    cursor = 0
    for row in list_of_sets:
        vals = np.fromiter(sorted(row), dtype=np.int32, count=len(row))
        flat[cursor : cursor + len(vals)] = vals
        cursor += len(vals)
    return flat, offsets


def build_vertex_n_ring_tris_collision_filter(
    n: int,
    particle_count: int,
    edge_indices: np.ndarray,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
    v_adj_tris: np.ndarray,
    v_adj_tris_offsets: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if n <= 1:
        return None, None

    filter_sets = [set() for _ in range(particle_count)]
    for v in range(particle_count):
        neighbors = (
            one_ring_vertices(v, edge_indices, v_adj_edges, v_adj_edges_offsets)
            if n == 2
            else leq_n_ring_vertices(v, edge_indices, n - 1, v_adj_edges, v_adj_edges_offsets)
        )

        incident_tris = set(_csr_row(v_adj_tris, v_adj_tris_offsets, v)[::2].tolist())
        out = filter_sets[v]
        for u in neighbors:
            u = int(u)
            if u != v:
                out.update(_csr_row(v_adj_tris, v_adj_tris_offsets, u)[::2].tolist())
        out.difference_update(incident_tris)

    return set_to_csr(filter_sets)


def build_edge_n_ring_edge_collision_filter(
    n: int,
    edge_indices: np.ndarray,
    v_adj_edges: np.ndarray,
    v_adj_edges_offsets: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if n <= 1:
        return None, None

    edges = np.asarray(edge_indices, dtype=np.int32)
    filter_sets = [set() for _ in range(edges.shape[0])]

    for e in range(edges.shape[0]):
        v0 = int(edges[e, 2])
        v1 = int(edges[e, 3])
        if n == 2:
            n0 = one_ring_vertices(v0, edges, v_adj_edges, v_adj_edges_offsets)
            n1 = one_ring_vertices(v1, edges, v_adj_edges, v_adj_edges_offsets)
        else:
            n0 = leq_n_ring_vertices(v0, edges, n - 1, v_adj_edges, v_adj_edges_offsets)
            n1 = leq_n_ring_vertices(v1, edges, n - 1, v_adj_edges, v_adj_edges_offsets)

        neighbor_vertices = set(n0.tolist()) | set(n1.tolist())
        incident0 = set(_csr_row(v_adj_edges, v_adj_edges_offsets, v0)[::2].tolist())
        incident1 = set(_csr_row(v_adj_edges, v_adj_edges_offsets, v1)[::2].tolist())

        out = filter_sets[e]
        for u in neighbor_vertices:
            if u == v0 or u == v1:
                continue
            rows = _csr_row(v_adj_edges, v_adj_edges_offsets, int(u))
            edge_ids = rows[::2]
            local_slots = rows[1::2]
            out.update(edge_ids[local_slots >= 2].tolist())

        out.difference_update(incident0)
        out.difference_update(incident1)

    return set_to_csr(filter_sets)


def build_mesh_topology(
    particle_count: int,
    triangles: np.ndarray,
) -> MeshTopology:
    triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    edge_indices, edge_tri_indices, tri_edge_indices = build_edge_topology(triangles)
    v_adj_tris, v_adj_tris_offsets, v_adj_edges, v_adj_edges_offsets = build_vertex_adjacency(
        particle_count, triangles, edge_indices
    )
    return MeshTopology(
        triangles=triangles,
        edge_indices=edge_indices,
        edge_tri_indices=edge_tri_indices,
        tri_edge_indices=tri_edge_indices,
        v_adj_tris=v_adj_tris,
        v_adj_tris_offsets=v_adj_tris_offsets,
        v_adj_edges=v_adj_edges,
        v_adj_edges_offsets=v_adj_edges_offsets,
    )
