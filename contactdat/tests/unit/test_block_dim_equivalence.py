from __future__ import annotations

from contactdat import TriMeshCollisionDetector
from collision_test_utils import make_two_parallel_triangles, paired_rows


def _candidate_rows(detector):
    return (
        paired_rows(
            detector.vertex_colliding_triangles,
            detector.vertex_colliding_triangles_offsets,
            detector.vertex_colliding_triangles_count,
            detector.vertex_colliding_triangles_buffer_sizes,
        ),
        paired_rows(
            detector.edge_colliding_edges,
            detector.edge_colliding_edges_offsets,
            detector.edge_colliding_edges_count,
            detector.edge_colliding_edges_buffer_sizes,
        ),
    )


def test_block_dim_changes_scheduling_not_results(contact_test_device):
    q, faces = make_two_parallel_triangles(z_gap=0.01)

    results = []
    for block_dim in (16, 32, 64):
        detector = TriMeshCollisionDetector(
            q,
            faces,
            device=contact_test_device,
            vertex_collision_buffer_pre_alloc=16,
            edge_collision_buffer_pre_alloc=32,
            collision_detection_block_size=block_dim,
        )
        detector.detect(q, margin=0.02)
        results.append(_candidate_rows(detector))

    assert results[0] == results[1] == results[2]
