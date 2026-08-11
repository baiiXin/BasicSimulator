
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import warp as wp
except ImportError:
    print("SKIP test_collision_smoke: warp is not installed")
    raise SystemExit(0)

from standalone_contact.collision_detector import TriMeshCollisionDetector


def main():
    # Two disconnected, almost-coincident triangles.
    q = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.01],
            [1.0, 0.0, 0.01],
            [0.0, 1.0, 0.01],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

    detector = TriMeshCollisionDetector(
        q,
        faces,
        device="cpu",
        vertex_collision_buffer_pre_alloc=8,
        edge_collision_buffer_pre_alloc=8,
    )

    detector.detect(q, margin=0.02)
    vt_counts = detector.vertex_colliding_triangles_count.numpy()
    ee_counts = detector.edge_colliding_edges_count.numpy()

    assert int(vt_counts.sum()) >= 6, vt_counts
    assert int(ee_counts.sum()) > 0, ee_counts
    assert detector.overflow_flags()[0] == 0
    assert detector.overflow_flags()[2] == 0

    # The same configuration used as rest pose, with a larger exclusion
    # radius, should remove those naturally-close pairs.
    detector.detect(
        q,
        margin=0.02,
        rest_q=q,
        rest_exclusion_radius=0.02,
    )
    vt_counts_filtered = detector.vertex_colliding_triangles_count.numpy()
    ee_counts_filtered = detector.edge_colliding_edges_count.numpy()

    assert int(vt_counts_filtered.sum()) == 0, vt_counts_filtered
    assert int(ee_counts_filtered.sum()) == 0, ee_counts_filtered

    print("PASS test_collision_smoke")


if __name__ == "__main__":
    main()
