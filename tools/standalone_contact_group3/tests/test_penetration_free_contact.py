
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import warp as wp

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from standalone_contact.collision_detector import TriMeshCollisionInfo
from standalone_contact.penetration_free_contact import PenetrationFreeContact


def _device():
    return os.environ.get("CONTACT_TEST_DEVICE", "cpu")


def _i32(values, device):
    return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)


def _f32(values, device):
    return wp.array(np.asarray(values, dtype=np.float32), dtype=float, device=device)


def _build_vt_collision_info(device):
    """One VT candidate: vertex 3 vs triangle 0."""
    particle_count = 4
    vt_sizes = np.array([0, 0, 0, 1], dtype=np.int32)
    vt_offsets = np.zeros(particle_count + 1, dtype=np.int32)
    vt_offsets[1:] = np.cumsum(vt_sizes)

    info = TriMeshCollisionInfo()

    info.vertex_colliding_triangles = _i32([3, 0], device)
    info.vertex_colliding_triangles_offsets = _i32(vt_offsets, device)
    info.vertex_colliding_triangles_buffer_sizes = _i32(vt_sizes, device)
    info.vertex_colliding_triangles_count = _i32([0, 0, 0, 1], device)
    info.vertex_colliding_triangles_min_dist = _f32(np.zeros(4), device)

    info.triangle_colliding_vertices = _i32([3], device)
    info.triangle_colliding_vertices_offsets = _i32([0, 1], device)
    info.triangle_colliding_vertices_buffer_sizes = _i32([1], device)
    info.triangle_colliding_vertices_count = _i32([1], device)
    info.triangle_colliding_vertices_min_dist = _f32([0.0], device)

    info.edge_colliding_edges = wp.empty(0, dtype=wp.int32, device=device)
    info.edge_colliding_edges_offsets = _i32([0], device)
    info.edge_colliding_edges_buffer_sizes = wp.empty(0, dtype=wp.int32, device=device)
    info.edge_colliding_edges_count = wp.empty(0, dtype=wp.int32, device=device)
    info.edge_colliding_edges_min_dist = wp.empty(0, dtype=float, device=device)

    return info


class FakeDetector:
    """Minimal Group-1-compatible detector for orchestration testing."""

    def __init__(self, device):
        self.edge_edge_parallel_epsilon = 1.0e-5

        self.tri_indices = wp.array(
            np.asarray([[0, 1, 2]], dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        self.edge_indices = wp.empty((0, 4), dtype=wp.int32, device=device)
        self.collision_info = _build_vt_collision_info(device)

        self.refit_calls = 0
        self.vt_calls = 0
        self.ee_calls = 0

    def refit(self, q):
        self.refit_calls += 1

    def vertex_triangle_collision_detection(
        self,
        query_radius,
        *,
        min_query_radius=0.0,
        min_distance_filtering_ref_pos=None,
    ):
        self.vt_calls += 1

    def edge_edge_collision_detection(
        self,
        query_radius,
        *,
        min_query_radius=0.0,
        min_distance_filtering_ref_pos=None,
    ):
        self.ee_calls += 1


def test_begin_displacement_dat_qsafe_redetect():
    device = _device()

    q0_np = np.array(
        [
            [-1.0, -1.0, 0.0],
            [ 1.0, -1.0, 0.0],
            [ 0.0,  1.0, 0.0],
            [ 0.0,  0.0, 0.2],
        ],
        dtype=np.float32,
    )
    q0 = wp.array(q0_np, dtype=wp.vec3, device=device)

    detector = FakeDetector(device)
    contact = PenetrationFreeContact(
        detector,
        particle_count=4,
        edge_count=0,
        device=device,
        contact_margin=1.0,
        conservative_bound_relaxation=0.85,
    )

    # Epoch 0: snapshot + collision detection.
    contact.begin(q0)

    assert contact.epoch == 0
    assert (detector.refit_calls, detector.vt_calls, detector.ee_calls) == (1, 1, 1)
    np.testing.assert_allclose(contact.q_snapshot.numpy(), q0_np, atol=1.0e-6)

    # External solver proposes motion relative to snapshot.
    dx_np = np.zeros_like(q0_np)
    dx_np[3] = [0.0, 0.0, -0.30]
    dx = wp.array(dx_np, dtype=wp.vec3, device=device)

    dx_safe, q_safe, ts = contact.truncate(dx)

    ts_np = ts.numpy()
    dx_safe_np = dx_safe.numpy()
    q_safe_np = q_safe.numpy()

    assert ts_np[3] < 1.0
    np.testing.assert_allclose(q_safe_np, q0_np + dx_safe_np, atol=1.0e-6)
    assert q_safe_np[3, 2] > 0.0

    # New collision epoch: old q_safe becomes new snapshot.
    contact.redetect(q_safe)

    assert contact.epoch == 1
    assert (detector.refit_calls, detector.vt_calls, detector.ee_calls) == (2, 2, 2)
    np.testing.assert_allclose(contact.q_snapshot.numpy(), q_safe_np, atol=1.0e-6)
    np.testing.assert_allclose(
        contact.displacement.numpy(),
        np.zeros_like(q0_np),
        atol=1.0e-7,
    )


def test_truncate_requires_begin():
    device = _device()

    detector = FakeDetector(device)
    contact = PenetrationFreeContact(
        detector,
        particle_count=4,
        edge_count=0,
        device=device,
        contact_margin=1.0,
    )

    dx = wp.zeros(4, dtype=wp.vec3, device=device)

    try:
        contact.truncate(dx)
    except RuntimeError:
        return

    raise AssertionError("truncate() should require begin() first")
