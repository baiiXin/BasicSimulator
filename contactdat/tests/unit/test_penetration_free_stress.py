
from __future__ import annotations

import os
import numpy as np
import pytest
import warp as wp

ipctk = pytest.importorskip("ipctk")

from contactdat import PenetrationFreeContact, TriMeshCollisionDetector
from group4_stress_utils import (
    ipc_has_intersections,
    make_center_punch_displacement,
    make_two_layer_grid,
)


def _device():
    return os.environ.get("CONTACT_TEST_DEVICE", "cpu")


def test_two_layer_grid_candidate_intersects_dat_output_does_not():
    """Full Group1 -> Group2 -> Group3 correctness test with IPC oracle."""
    device = _device()

    n = 8
    gap = 0.020
    spacing = 0.050

    mesh = make_two_layer_grid(
        n=n,
        spacing=spacing,
        gap=gap,
    )

    # Snapshot must start intersection-free.
    assert not ipc_has_intersections(mesh.vertices, mesh.edges, mesh.faces)

    q0 = wp.array(mesh.vertices, dtype=wp.vec3, device=device)
    q_rest = wp.array(mesh.vertices, dtype=wp.vec3, device=device)

    # Keep the query margin larger than the inter-layer gap so the initial
    # cross-layer candidates are present before the artificial solver move.
    contact_margin = 0.030

    detector = TriMeshCollisionDetector(
        q0,
        mesh.faces,
        device=device,
        vertex_collision_buffer_pre_alloc=32,
        edge_collision_buffer_pre_alloc=32,
        topological_contact_filter_threshold=2,
    )

    contact = PenetrationFreeContact(
        detector,
        particle_count=len(mesh.vertices),
        edge_count=detector.edge_count,
        device=device,
        contact_margin=contact_margin,
        conservative_bound_relaxation=0.85,
        rest_q=q_rest,
        # IMPORTANT for this synthetic test:
        # do not rest-exclude the close but disconnected second layer.
        rest_exclusion_radius=0.0,
    )

    contact.begin(q0)

    # Punch the center patch far enough below z=0 so the *static final
    # candidate* itself contains intersections.
    dx_np = make_center_punch_displacement(
        mesh,
        n=n,
        downward_distance=0.050,
        center_half_width=1,
    )

    q_candidate_np = mesh.vertices + dx_np
    assert ipc_has_intersections(
        q_candidate_np,
        mesh.edges,
        mesh.faces,
    ), "Synthetic candidate failed to create a static self-intersection."

    dx = wp.array(dx_np, dtype=wp.vec3, device=device)

    dx_safe, q_safe, truncation_ts = contact.truncate(dx)

    q_safe_np = q_safe.numpy()
    ts_np = truncation_ts.numpy()

    # Core Group-4 correctness criterion.
    assert not ipc_has_intersections(
        q_safe_np,
        mesh.edges,
        mesh.faces,
    )

    # Sanity checks: DAT actually activated and all output is finite.
    assert np.any(ts_np < 1.0 - 1.0e-6)
    assert np.isfinite(q_safe_np).all()
    assert np.isfinite(dx_safe.numpy()).all()

    # No fixed-capacity collision buffer should have silently overflowed.
    if hasattr(detector, "overflow_flags"):
        flags = np.asarray(detector.overflow_flags())
        assert not np.any(flags[:3]), f"collision buffer overflow flags={flags}"


def test_redetect_after_safe_configuration_stays_intersection_free():
    """Exercise a second collision epoch on the Group-4 synthetic mesh."""
    device = _device()

    n = 8
    mesh = make_two_layer_grid(n=n, spacing=0.05, gap=0.02)

    q0 = wp.array(mesh.vertices, dtype=wp.vec3, device=device)

    detector = TriMeshCollisionDetector(
        q0,
        mesh.faces,
        device=device,
        vertex_collision_buffer_pre_alloc=32,
        edge_collision_buffer_pre_alloc=32,
        topological_contact_filter_threshold=2,
    )

    contact = PenetrationFreeContact(
        detector,
        particle_count=len(mesh.vertices),
        edge_count=detector.edge_count,
        device=device,
        contact_margin=0.03,
        conservative_bound_relaxation=0.85,
        rest_q=q0,
        rest_exclusion_radius=0.0,
    )

    contact.begin(q0)

    dx1_np = make_center_punch_displacement(
        mesh,
        n=n,
        downward_distance=0.05,
        center_half_width=1,
    )
    dx1 = wp.array(dx1_np, dtype=wp.vec3, device=device)

    _, q1, _ = contact.truncate(dx1)
    assert not ipc_has_intersections(q1.numpy(), mesh.edges, mesh.faces)

    contact.redetect(q1)

    # A second smaller solver proposal relative to the NEW snapshot.
    dx2_np = np.zeros_like(mesh.vertices, dtype=np.float32)
    top_offset = int(mesh.top_vertex_ids[0])
    center = (n // 2) * n + (n // 2)
    dx2_np[top_offset + center, 2] = -0.02

    dx2 = wp.array(dx2_np, dtype=wp.vec3, device=device)
    _, q2, _ = contact.truncate(dx2)

    assert not ipc_has_intersections(q2.numpy(), mesh.edges, mesh.faces)
    assert np.isfinite(q2.numpy()).all()
