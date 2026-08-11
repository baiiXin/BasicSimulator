from __future__ import annotations

import numpy as np

from contactdat import TriMeshCollisionDetector
from collision_test_utils import (
    make_newton_dummy_model,
    make_two_parallel_triangles,
    paired_rows,
    scalar_rows,
)


def test_collision_candidates_match_local_newton(contact_test_device):
    from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionDetector as NewtonDetector

    q, faces = make_two_parallel_triangles(z_gap=0.01)
    model = make_newton_dummy_model(q, faces, contact_test_device)

    reference = NewtonDetector(
        model,
        record_triangle_contacting_vertices=True,
        vertex_positions=model.particle_q,
        vertex_collision_buffer_pre_alloc=16,
        triangle_collision_buffer_pre_alloc=16,
        edge_collision_buffer_pre_alloc=32,
        topological_contact_filter_threshold=0,
        collision_detection_block_size=16,
    )
    reference.refit(model.particle_q)
    reference.vertex_triangle_collision_detection(0.02)
    reference.edge_edge_collision_detection(0.02)

    ours = TriMeshCollisionDetector(
        model.particle_q,
        faces,
        device=contact_test_device,
        record_triangle_contacting_vertices=True,
        vertex_collision_buffer_pre_alloc=16,
        triangle_collision_buffer_pre_alloc=16,
        edge_collision_buffer_pre_alloc=32,
        topological_contact_filter_threshold=0,
        collision_detection_block_size=16,
    )
    ours.detect(model.particle_q, margin=0.02)

    np.testing.assert_array_equal(
        ours.vertex_colliding_triangles_count.numpy(),
        reference.vertex_colliding_triangles_count.numpy(),
    )
    np.testing.assert_array_equal(
        ours.triangle_colliding_vertices_count.numpy(),
        reference.triangle_colliding_vertices_count.numpy(),
    )
    np.testing.assert_array_equal(
        ours.edge_colliding_edges_count.numpy(),
        reference.edge_colliding_edges_count.numpy(),
    )

    # BVH traversal order is not part of the API, so compare each row as a set/sorted tuple.
    assert paired_rows(
        ours.vertex_colliding_triangles,
        ours.vertex_colliding_triangles_offsets,
        ours.vertex_colliding_triangles_count,
        ours.vertex_colliding_triangles_buffer_sizes,
    ) == paired_rows(
        reference.vertex_colliding_triangles,
        reference.vertex_colliding_triangles_offsets,
        reference.vertex_colliding_triangles_count,
        reference.vertex_colliding_triangles_buffer_sizes,
    )

    assert scalar_rows(
        ours.triangle_colliding_vertices,
        ours.triangle_colliding_vertices_offsets,
        ours.triangle_colliding_vertices_count,
        ours.triangle_colliding_vertices_buffer_sizes,
    ) == scalar_rows(
        reference.triangle_colliding_vertices,
        reference.triangle_colliding_vertices_offsets,
        reference.triangle_colliding_vertices_count,
        reference.triangle_colliding_vertices_buffer_sizes,
    )

    assert paired_rows(
        ours.edge_colliding_edges,
        ours.edge_colliding_edges_offsets,
        ours.edge_colliding_edges_count,
        ours.edge_colliding_edges_buffer_sizes,
    ) == paired_rows(
        reference.edge_colliding_edges,
        reference.edge_colliding_edges_offsets,
        reference.edge_colliding_edges_count,
        reference.edge_colliding_edges_buffer_sizes,
    )

    np.testing.assert_allclose(
        ours.vertex_colliding_triangles_min_dist.numpy(),
        reference.vertex_colliding_triangles_min_dist.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        ours.edge_colliding_edges_min_dist.numpy(),
        reference.edge_colliding_edges_min_dist.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )
