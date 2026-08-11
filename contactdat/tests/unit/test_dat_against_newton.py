
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp


TOOLS = Path(__file__).resolve().parents[1]
NEWTON_REPO = TOOLS / "newton"

sys.path.insert(0, str(TOOLS))
if NEWTON_REPO.exists():
    sys.path.insert(0, str(NEWTON_REPO))

from standalone_contact.collision_detector import TriMeshCollisionInfo as StandaloneCollisionInfo
from standalone_contact.dat_kernels import (
    NUM_THREADS_PER_COLLISION_PRIMITIVE,
    apply_planar_truncation_parallel_by_collision as standalone_dat,
    apply_truncation_ts as standalone_apply_t,
)

try:
    from newton._src.solvers.vbd.tri_mesh_collision import (
        TriMeshCollisionInfo as NewtonCollisionInfo,
    )
    from newton._src.solvers.vbd.particle_vbd_kernels import (
        apply_planar_truncation_parallel_by_collision as newton_dat,
        apply_truncation_ts as newton_apply_t,
    )
except Exception as exc:
    NewtonCollisionInfo = None
    NEWTON_IMPORT_ERROR = exc
else:
    NEWTON_IMPORT_ERROR = None


def _device():
    return os.environ.get("CONTACT_TEST_DEVICE", "cpu")


def _wp_i32(x, device):
    return wp.array(np.asarray(x, dtype=np.int32), dtype=wp.int32, device=device)


def _wp_f32(x, device):
    return wp.array(np.asarray(x, dtype=np.float32), dtype=float, device=device)


def _build_shared_collision_arrays(device):
    particle_count = 8
    tri_count = 1
    edge_count = 2

    vt_sizes_np = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int32)
    vt_offsets_np = np.zeros(particle_count + 1, dtype=np.int32)
    vt_offsets_np[1:] = np.cumsum(vt_sizes_np)

    arrays = {
        "vertex_colliding_triangles": _wp_i32([3, 0], device),
        "vertex_colliding_triangles_offsets": _wp_i32(vt_offsets_np, device),
        "vertex_colliding_triangles_buffer_sizes": _wp_i32(vt_sizes_np, device),
        "vertex_colliding_triangles_count": _wp_i32([0, 0, 0, 1, 0, 0, 0, 0], device),
        "vertex_colliding_triangles_min_dist": _wp_f32(np.zeros(particle_count), device),

        "triangle_colliding_vertices": _wp_i32([3], device),
        "triangle_colliding_vertices_offsets": _wp_i32([0, 1], device),
        "triangle_colliding_vertices_buffer_sizes": _wp_i32([1], device),
        "triangle_colliding_vertices_count": _wp_i32([1], device),
        "triangle_colliding_vertices_min_dist": _wp_f32(np.zeros(tri_count), device),

        "edge_colliding_edges": _wp_i32([0, 1], device),
        "edge_colliding_edges_offsets": _wp_i32([0, 1, 1], device),
        "edge_colliding_edges_buffer_sizes": _wp_i32([1, 0], device),
        "edge_colliding_edges_count": _wp_i32([1, 0], device),
        "edge_colliding_edges_min_dist": _wp_f32(np.zeros(edge_count), device),
    }
    return arrays


def _pack_info(StructType, arrays):
    info = StructType()
    for key, value in arrays.items():
        setattr(info, key, value)
    return info


@pytest.mark.skipif(
    NewtonCollisionInfo is None,
    reason=f"local Newton source import failed: {NEWTON_IMPORT_ERROR}",
)
def test_dat_kernel_matches_newton():
    device = _device()

    q_np = np.array(
        [
            [-1.0, -1.0, 0.0],
            [ 1.0, -1.0, 0.0],
            [ 0.0,  1.0, 0.0],
            [ 0.0,  0.0, 0.2],

            [-1.0,  0.0, 0.2],
            [ 1.0,  0.0, 0.2],
            [ 0.0, -1.0, 0.0],
            [ 0.0,  1.0, 0.0],
        ],
        dtype=np.float32,
    )

    dx_np = np.zeros_like(q_np)
    dx_np[3] = [0.0, 0.0, -0.3]
    dx_np[4] = [0.0, 0.0, -0.3]
    dx_np[5] = [0.0, 0.0, -0.3]

    tri_np = np.array([[0, 1, 2]], dtype=np.int32)
    edge_np = np.array(
        [
            [-1, -1, 4, 5],
            [-1, -1, 6, 7],
        ],
        dtype=np.int32,
    )

    q = wp.array(q_np, dtype=wp.vec3, device=device)
    dx = wp.array(dx_np, dtype=wp.vec3, device=device)
    tri = wp.array(tri_np, dtype=wp.int32, device=device)
    edges = wp.array(edge_np, dtype=wp.int32, device=device)

    arrays = _build_shared_collision_arrays(device)

    standalone_info = _pack_info(StandaloneCollisionInfo, arrays)
    newton_info = _pack_info(NewtonCollisionInfo, arrays)

    standalone_info_array = wp.array(
        [standalone_info],
        dtype=StandaloneCollisionInfo,
        device=device,
    )
    newton_info_array = wp.array(
        [newton_info],
        dtype=NewtonCollisionInfo,
        device=device,
    )

    ts_standalone = wp.ones(len(q_np), dtype=float, device=device)
    ts_newton = wp.ones(len(q_np), dtype=float, device=device)

    launch_size = max(
        len(q_np) * NUM_THREADS_PER_COLLISION_PRIMITIVE,
        len(edge_np) * NUM_THREADS_PER_COLLISION_PRIMITIVE,
    )

    parallel_eps = 1.0e-5
    gamma = 0.85

    wp.launch(
        standalone_dat,
        dim=launch_size,
        inputs=[
            q, dx, tri, edges,
            standalone_info_array,
            parallel_eps, gamma,
            ts_standalone,
        ],
        device=device,
    )

    wp.launch(
        newton_dat,
        dim=launch_size,
        inputs=[
            q, dx, tri, edges,
            newton_info_array,
            parallel_eps, gamma,
            ts_newton,
        ],
        device=device,
    )

    got = ts_standalone.numpy()
    ref = ts_newton.numpy()

    np.testing.assert_allclose(got, ref, atol=1.0e-6, rtol=1.0e-6)

    assert got[3] < 1.0
    assert got[4] < 1.0
    assert got[5] < 1.0


@pytest.mark.skipif(
    NewtonCollisionInfo is None,
    reason=f"local Newton source import failed: {NEWTON_IMPORT_ERROR}",
)
def test_apply_truncation_ts_matches_newton():
    device = _device()

    q_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    dx_np = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.2, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    ts_np = np.array([0.5, 0.25], dtype=np.float32)

    q = wp.array(q_np, dtype=wp.vec3, device=device)
    dx = wp.array(dx_np, dtype=wp.vec3, device=device)
    ts = wp.array(ts_np, dtype=float, device=device)

    dx_a = wp.empty_like(dx)
    q_a = wp.empty_like(q)
    dx_b = wp.empty_like(dx)
    q_b = wp.empty_like(q)

    max_displacement = 0.25

    wp.launch(
        standalone_apply_t,
        dim=len(q_np),
        inputs=[q, dx, ts, max_displacement, dx_a, q_a],
        device=device,
    )
    wp.launch(
        newton_apply_t,
        dim=len(q_np),
        inputs=[q, dx, ts, max_displacement, dx_b, q_b],
        device=device,
    )

    np.testing.assert_allclose(dx_a.numpy(), dx_b.numpy(), atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(q_a.numpy(), q_b.numpy(), atol=1.0e-6, rtol=1.0e-6)
