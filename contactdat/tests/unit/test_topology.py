from __future__ import annotations

import numpy as np

from contactdat._src.mesh_topology import build_mesh_topology


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
