from __future__ import annotations

import numpy as np

from standalone_contact import TriMeshCollisionDetector
from collision_test_utils import make_two_parallel_triangles, paired_rows, scalar_rows


def test_vt_ee_and_reverse_index(contact_test_device):
    q, faces = make_two_parallel_triangles(z_gap=0.01)

    detector = TriMeshCollisionDetector(
        q,
        faces,
        device=contact_test_device,
        record_triangle_contacting_vertices=True,
        vertex_collision_buffer_pre_alloc=8,
        triangle_collision_buffer_pre_alloc=8,
        edge_collision_buffer_pre_alloc=16,
        topological_contact_filter_threshold=0,
    )

    detector.detect(q, margin=0.02)

    vt_counts = detector.vertex_colliding_triangles_count.numpy()
    tri_counts = detector.triangle_colliding_vertices_count.numpy()
    ee_counts = detector.edge_colliding_edges_count.numpy()

    # Every vertex sees the other, disconnected triangle.
    np.testing.assert_array_equal(vt_counts, np.ones(6, dtype=np.int32))
    # Reverse VT index: each triangle sees all three vertices of the other triangle.
    np.testing.assert_array_equal(tri_counts, np.full(2, 3, dtype=np.int32))
    assert int(vt_counts.sum()) == int(tri_counts.sum())
    assert int(ee_counts.sum()) > 0

    vt_rows = paired_rows(
        detector.vertex_colliding_triangles,
        detector.vertex_colliding_triangles_offsets,
        detector.vertex_colliding_triangles_count,
        detector.vertex_colliding_triangles_buffer_sizes,
    )
    assert vt_rows[:3] == [(1,), (1,), (1,)]
    assert vt_rows[3:] == [(0,), (0,), (0,)]

    tri_rows = scalar_rows(
        detector.triangle_colliding_vertices,
        detector.triangle_colliding_vertices_offsets,
        detector.triangle_colliding_vertices_count,
        detector.triangle_colliding_vertices_buffer_sizes,
    )
    assert tri_rows[0] == (3, 4, 5)
    assert tri_rows[1] == (0, 1, 2)

    np.testing.assert_array_equal(detector.overflow_flags()[:3], np.zeros(3, dtype=np.int32))


def test_rest_shape_exclusion(contact_test_device):
    q, faces = make_two_parallel_triangles(z_gap=0.01)
    detector = TriMeshCollisionDetector(
        q,
        faces,
        device=contact_test_device,
        record_triangle_contacting_vertices=True,
        vertex_collision_buffer_pre_alloc=8,
        triangle_collision_buffer_pre_alloc=8,
        edge_collision_buffer_pre_alloc=16,
    )

    detector.detect(
        q,
        margin=0.02,
        rest_q=q,
        rest_exclusion_radius=0.02,
    )

    assert int(detector.vertex_colliding_triangles_count.numpy().sum()) == 0
    assert int(detector.triangle_colliding_vertices_count.numpy().sum()) == 0
    assert int(detector.edge_colliding_edges_count.numpy().sum()) == 0
