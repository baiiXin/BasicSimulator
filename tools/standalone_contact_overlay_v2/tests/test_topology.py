from __future__ import annotations

import numpy as np

from standalone_contact.mesh_topology import build_mesh_topology


def test_small_mesh_expected_topology():
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    topo = build_mesh_topology(4, faces)

    expected_edges = np.array(
        [
            [2, -1, 0, 1],
            [0, 3, 1, 2],
            [1, -1, 2, 0],
            [2, -1, 1, 3],
            [1, -1, 3, 2],
        ],
        dtype=np.int32,
    )
    expected_tri_edges = np.array([[0, 1, 2], [3, 4, 1]], dtype=np.int32)

    np.testing.assert_array_equal(topo.edge_indices, expected_edges)
    np.testing.assert_array_equal(topo.tri_edge_indices, expected_tri_edges)


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
