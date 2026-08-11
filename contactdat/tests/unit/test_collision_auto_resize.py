from __future__ import annotations

import numpy as np
import pytest

from contactdat import TriMeshCollisionDetector
from contactdat._src.collision_detector import CollisionBufferCapacityError
from collision_test_utils import make_stacked_triangles


def test_auto_resize_grows_all_three_buffer_families(contact_test_device):
    layers = 6
    q, faces = make_stacked_triangles(num_layers=layers, dz=0.005)

    detector = TriMeshCollisionDetector(
        q,
        faces,
        device=contact_test_device,
        record_triangle_contacting_vertices=True,
        vertex_collision_buffer_pre_alloc=1,
        vertex_collision_buffer_max_alloc=32,
        triangle_collision_buffer_pre_alloc=1,
        triangle_collision_buffer_max_alloc=32,
        edge_collision_buffer_pre_alloc=1,
        edge_collision_buffer_max_alloc=32,
        topological_contact_filter_threshold=0,
        auto_resize=True,
    )

    detector.detect(q, margin=0.04)

    # For identical disconnected triangles at different z:
    # each vertex sees one face in every other layer.
    expected_vt = layers - 1
    np.testing.assert_array_equal(
        detector.vertex_colliding_triangles_count.numpy(),
        np.full(3 * layers, expected_vt, dtype=np.int32),
    )

    # Each triangle sees all three vertices from every other layer.
    expected_reverse = 3 * (layers - 1)
    np.testing.assert_array_equal(
        detector.triangle_colliding_vertices_count.numpy(),
        np.full(layers, expected_reverse, dtype=np.int32),
    )

    # Each edge is close to all three edges in each other layer for this geometry.
    expected_ee = 3 * (layers - 1)
    np.testing.assert_array_equal(
        detector.edge_colliding_edges_count.numpy(),
        np.full(3 * layers, expected_ee, dtype=np.int32),
    )

    # Forward and reverse VT indices represent the same candidate pairs.
    assert int(detector.vertex_colliding_triangles_count.numpy().sum()) == int(
        detector.triangle_colliding_vertices_count.numpy().sum()
    )

    assert int(detector.vertex_colliding_triangles_buffer_sizes.numpy().max()) >= expected_vt
    assert int(detector.triangle_colliding_vertices_buffer_sizes.numpy().max()) >= expected_reverse
    assert int(detector.edge_colliding_edges_buffer_sizes.numpy().max()) >= expected_ee
    np.testing.assert_array_equal(detector.overflow_flags()[:3], np.zeros(3, dtype=np.int32))


def test_auto_resize_respects_max_alloc(contact_test_device):
    q, faces = make_stacked_triangles(num_layers=6, dz=0.005)
    detector = TriMeshCollisionDetector(
        q,
        faces,
        device=contact_test_device,
        vertex_collision_buffer_pre_alloc=1,
        vertex_collision_buffer_max_alloc=4,
        edge_collision_buffer_pre_alloc=1,
        edge_collision_buffer_max_alloc=32,
        auto_resize=True,
    )

    with pytest.raises(CollisionBufferCapacityError, match="max_alloc"):
        detector.detect(q, margin=0.04)
