
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import warp as wp


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from standalone_contact.collision_detector import TriMeshCollisionInfo
from standalone_contact.penetration_free import PenetrationFreeTruncator


def _device():
    return os.environ.get("CONTACT_TEST_DEVICE", "cpu")


def _empty_collision_info(particle_count, edge_count, tri_count, device):
    info = TriMeshCollisionInfo()

    info.vertex_colliding_triangles = wp.empty(0, dtype=wp.int32, device=device)
    info.vertex_colliding_triangles_offsets = wp.zeros(
        particle_count + 1, dtype=wp.int32, device=device
    )
    info.vertex_colliding_triangles_buffer_sizes = wp.zeros(
        particle_count, dtype=wp.int32, device=device
    )
    info.vertex_colliding_triangles_count = wp.zeros(
        particle_count, dtype=wp.int32, device=device
    )
    info.vertex_colliding_triangles_min_dist = wp.zeros(
        particle_count, dtype=float, device=device
    )

    info.triangle_colliding_vertices = wp.empty(0, dtype=wp.int32, device=device)
    info.triangle_colliding_vertices_offsets = wp.zeros(
        tri_count + 1, dtype=wp.int32, device=device
    )
    info.triangle_colliding_vertices_buffer_sizes = wp.zeros(
        tri_count, dtype=wp.int32, device=device
    )
    info.triangle_colliding_vertices_count = wp.zeros(
        tri_count, dtype=wp.int32, device=device
    )
    info.triangle_colliding_vertices_min_dist = wp.zeros(
        tri_count, dtype=float, device=device
    )

    info.edge_colliding_edges = wp.empty(0, dtype=wp.int32, device=device)
    info.edge_colliding_edges_offsets = wp.zeros(
        edge_count + 1, dtype=wp.int32, device=device
    )
    info.edge_colliding_edges_buffer_sizes = wp.zeros(
        edge_count, dtype=wp.int32, device=device
    )
    info.edge_colliding_edges_count = wp.zeros(
        edge_count, dtype=wp.int32, device=device
    )
    info.edge_colliding_edges_min_dist = wp.zeros(
        edge_count, dtype=float, device=device
    )

    return info


def test_wrapper_no_self_contact_is_identity():
    device = _device()

    q_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    dx_np = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, -0.3],
        ],
        dtype=np.float32,
    )

    tri_np = np.array([[0, 1, 2]], dtype=np.int32)
    edge_np = np.array(
        [
            [2, -1, 0, 1],
            [0, -1, 1, 2],
            [1, -1, 2, 0],
        ],
        dtype=np.int32,
    )

    q = wp.array(q_np, dtype=wp.vec3, device=device)
    dx = wp.array(dx_np, dtype=wp.vec3, device=device)
    tri = wp.array(tri_np, dtype=wp.int32, device=device)
    edges = wp.array(edge_np, dtype=wp.int32, device=device)

    info = _empty_collision_info(3, 3, 1, device)

    truncator = PenetrationFreeTruncator(
        particle_count=3,
        edge_count=3,
        collision_info=info,
        device=device,
        contact_margin=0.02,
    )

    dx_out, q_out, ts = truncator.truncate(
        snapshot_pos=q,
        displacement_in=dx,
        tri_indices=tri,
        edge_indices=edges,
        enable_self_contact=False,
        write_positions=True,
    )

    np.testing.assert_allclose(ts.numpy(), np.ones(3), atol=1.0e-7)
    np.testing.assert_allclose(dx_out.numpy(), dx_np, atol=1.0e-6)
    np.testing.assert_allclose(q_out.numpy(), q_np + dx_np, atol=1.0e-6)


def test_wrapper_self_contact_final_global_displacement_cap():
    device = _device()

    q_np = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    dx_np = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    q = wp.array(q_np, dtype=wp.vec3, device=device)
    dx = wp.array(dx_np, dtype=wp.vec3, device=device)

    tri = wp.empty((0, 3), dtype=wp.int32, device=device)
    edges = wp.empty((0, 4), dtype=wp.int32, device=device)

    info = _empty_collision_info(1, 0, 0, device)

    gamma = 0.85
    margin = 0.02

    truncator = PenetrationFreeTruncator(
        particle_count=1,
        edge_count=0,
        collision_info=info,
        device=device,
        conservative_bound_relaxation=gamma,
        contact_margin=margin,
    )

    dx_out, q_out, ts = truncator.truncate(
        snapshot_pos=q,
        displacement_in=dx,
        tri_indices=tri,
        edge_indices=edges,
        enable_self_contact=True,
        write_positions=True,
    )

    expected_max = margin * gamma * 0.5
    got = dx_out.numpy()[0]

    assert np.isclose(np.linalg.norm(got), expected_max, atol=1.0e-6)
    np.testing.assert_allclose(q_out.numpy()[0], got, atol=1.0e-6)
