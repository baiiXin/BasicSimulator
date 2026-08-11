from __future__ import annotations

import numpy as np

from contactdat._src.mesh_topology import build_mesh_topology


def test_topology_matches_local_newton():
    from newton._src.utils.mesh import MeshAdjacency

    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
            [1, 4, 3],
            [4, 5, 3],
        ],
        dtype=np.int32,
    )

    ours = build_mesh_topology(6, faces)
    reference = MeshAdjacency(tri_indices=faces).init_vertex_adjacency(6)

    np.testing.assert_array_equal(ours.edge_indices, reference.edge_indices)
    np.testing.assert_array_equal(ours.edge_tri_indices, reference.edge_tri_indices)
    np.testing.assert_array_equal(ours.tri_edge_indices, reference.tri_edge_indices)
    np.testing.assert_array_equal(ours.v_adj_edges, reference.v_adj_edges)
    np.testing.assert_array_equal(ours.v_adj_edges_offsets, reference.v_adj_edges_offsets)
    np.testing.assert_array_equal(ours.v_adj_tris, reference.v_adj_tris)
    np.testing.assert_array_equal(ours.v_adj_tris_offsets, reference.v_adj_tris_offsets)
