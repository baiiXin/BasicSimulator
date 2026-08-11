
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from standalone_contact.mesh_topology import build_mesh_topology


def main():
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
        ],
        dtype=np.int32,
    )

    topo = build_mesh_topology(4, faces)

    expected_edges = np.array(
        [
            [2, -1, 0, 1],
            [0,  3, 1, 2],
            [1, -1, 2, 0],
            [2, -1, 1, 3],
            [1, -1, 3, 2],
        ],
        dtype=np.int32,
    )
    expected_tri_edges = np.array(
        [
            [0, 1, 2],
            [3, 4, 1],
        ],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(topo.edge_indices, expected_edges)
    np.testing.assert_array_equal(topo.tri_edge_indices, expected_tri_edges)

    # Vertex 1 belongs to both triangles.
    row = topo.v_adj_tris[
        topo.v_adj_tris_offsets[1] : topo.v_adj_tris_offsets[2]
    ].reshape(-1, 2)
    assert set(row[:, 0].tolist()) == {0, 1}

    print("PASS test_topology")


if __name__ == "__main__":
    main()
