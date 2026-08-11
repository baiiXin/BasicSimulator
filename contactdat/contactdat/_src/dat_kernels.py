# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Standalone adaptation for Warp use without importing Newton.
# Modifications Copyright (c) 2026 Standalone Contact contributors.


from __future__ import annotations

import warp as wp

from .collision_detector import TriMeshCollisionInfo
from .geometry import triangle_closest_point


NUM_THREADS_PER_COLLISION_PRIMITIVE = 4


@wp.func
def segment_plane_intersects(
    v: wp.vec3,
    delta_v: wp.vec3,
    n: wp.vec3,
    d: wp.vec3,
    eps_parallel: float,
    eps_intersect_near: float,
    eps_intersect_far: float,
    coplanar_counts: bool,
) -> bool:
    nv = wp.dot(n, delta_v)
    num = -wp.dot(n, v - d)

    if wp.abs(nv) < eps_parallel:
        return coplanar_counts and (wp.abs(num) < eps_parallel)

    t = num / nv
    return (t >= eps_intersect_near) and (t <= 1.0 + eps_intersect_far)


@wp.func
def create_vertex_triangle_division_plane_closest_pt(
    v: wp.vec3,
    delta_v: wp.vec3,
    t1: wp.vec3,
    delta_t1: wp.vec3,
    t2: wp.vec3,
    delta_t2: wp.vec3,
    t3: wp.vec3,
    delta_t3: wp.vec3,
):
    closest_p, _bary, _feature_type = triangle_closest_point(t1, t2, t3, v)
    n_hat = v - closest_p

    if wp.length(n_hat) < 1.0e-12:
        return (
            wp.vector(False, False, False, False, length=4, dtype=wp.bool),
            wp.vec3(0.0),
            v,
        )

    n = wp.normalize(n_hat)

    delta_v_n = wp.max(-wp.dot(n, delta_v), 0.0)
    delta_t_n = wp.max(
        wp.vec4(
            wp.dot(n, delta_t1),
            wp.dot(n, delta_t2),
            wp.dot(n, delta_t3),
            0.0,
        )
    )

    if delta_t_n + delta_v_n == 0.0:
        d = closest_p + 0.5 * n_hat
    else:
        lmbd = delta_t_n / (delta_t_n + delta_v_n)
        lmbd = wp.clamp(lmbd, 0.05, 0.95)
        d = closest_p + lmbd * n_hat

    if delta_v_n == 0.0:
        is_dummy_for_v = True
    else:
        is_dummy_for_v = not segment_plane_intersects(
            v, delta_v, n, d, 1.0e-6, -1.0e-8, 1.0e-8, False
        )

    if delta_t_n == 0.0:
        is_dummy_for_t_1 = True
        is_dummy_for_t_2 = True
        is_dummy_for_t_3 = True
    else:
        is_dummy_for_t_1 = not segment_plane_intersects(
            t1, delta_t1, n, d, 1.0e-6, -1.0e-8, 1.0e-8, False
        )
        is_dummy_for_t_2 = not segment_plane_intersects(
            t2, delta_t2, n, d, 1.0e-6, -1.0e-8, 1.0e-8, False
        )
        is_dummy_for_t_3 = not segment_plane_intersects(
            t3, delta_t3, n, d, 1.0e-6, -1.0e-8, 1.0e-8, False
        )

    return (
        wp.vector(
            is_dummy_for_v,
            is_dummy_for_t_1,
            is_dummy_for_t_2,
            is_dummy_for_t_3,
            length=4,
            dtype=wp.bool,
        ),
        n,
        d,
    )


@wp.func
def robust_edge_pair_normal(
    e0_v0_pos: wp.vec3,
    e0_v1_pos: wp.vec3,
    e1_v0_pos: wp.vec3,
    e1_v1_pos: wp.vec3,
    eps: float = 1.0e-6,
) -> wp.vec3:
    dir0 = e0_v1_pos - e0_v0_pos
    dir1 = e1_v1_pos - e1_v0_pos

    len0 = wp.length(dir0)
    len1 = wp.length(dir1)

    if len0 > eps:
        dir0 = dir0 / len0
    else:
        dir0 = wp.vec3(0.0, 0.0, 0.0)

    if len1 > eps:
        dir1 = dir1 / len1
    else:
        dir1 = wp.vec3(0.0, 0.0, 0.0)

    n = wp.cross(dir0, dir1)
    len_n = wp.length(n)
    if len_n > eps:
        return n / len_n

    reference = dir0
    if wp.length(reference) <= eps:
        reference = dir1

    if wp.length(reference) <= eps:
        return wp.vec3(1.0, 0.0, 0.0)

    bridge = 0.5 * ((e1_v0_pos + e1_v1_pos) - (e0_v0_pos + e0_v1_pos))
    bridge_len = wp.length(bridge)
    if bridge_len > eps:
        n = wp.cross(reference, bridge / bridge_len)
        len_n = wp.length(n)
        if len_n > eps:
            return n / len_n

    fallback_axis = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(wp.dot(reference, fallback_axis)) > 0.9:
        fallback_axis = wp.vec3(0.0, 1.0, 0.0)

    n = wp.cross(reference, fallback_axis)
    len_n = wp.length(n)
    if len_n > eps:
        return n / len_n

    fallback_axis = wp.vec3(0.0, 0.0, 1.0)
    n = wp.cross(reference, fallback_axis)
    len_n = wp.length(n)
    if len_n > eps:
        return n / len_n

    return wp.vec3(1.0, 0.0, 0.0)


@wp.func
def create_edge_edge_division_plane_closest_pt(
    e0_v0_pos: wp.vec3,
    delta_e0_v0: wp.vec3,
    e0_v1_pos: wp.vec3,
    delta_e0_v1: wp.vec3,
    e1_v0_pos: wp.vec3,
    delta_e1_v0: wp.vec3,
    e1_v1_pos: wp.vec3,
    delta_e1_v1: wp.vec3,
):
    st = wp.closest_point_edge_edge(
        e0_v0_pos, e0_v1_pos, e1_v0_pos, e1_v1_pos, 1.0e-6
    )
    s = st[0]
    t = st[1]

    c1 = e0_v0_pos + (e0_v1_pos - e0_v0_pos) * s
    c2 = e1_v0_pos + (e1_v1_pos - e1_v0_pos) * t

    n_hat = c1 - c2

    if wp.length(n_hat) < 1.0e-12:
        return (
            wp.vector(False, False, False, False, length=4, dtype=wp.bool),
            robust_edge_pair_normal(
                e0_v0_pos, e0_v1_pos, e1_v0_pos, e1_v1_pos
            ),
            c1 * 0.5 + c2 * 0.5,
        )

    n = wp.normalize(n_hat)

    delta_e0 = wp.max(
        wp.vec3(
            -wp.dot(n, delta_e0_v0),
            -wp.dot(n, delta_e0_v1),
            0.0,
        )
    )
    delta_e1 = wp.max(
        wp.vec3(
            wp.dot(n, delta_e1_v0),
            wp.dot(n, delta_e1_v1),
            0.0,
        )
    )

    if delta_e0 + delta_e1 == 0.0:
        d = c2 + 0.5 * n_hat
    else:
        lmbd = delta_e1 / (delta_e1 + delta_e0)
        lmbd = wp.clamp(lmbd, 0.05, 0.95)
        d = c2 + lmbd * n_hat

    if delta_e0 == 0.0:
        is_dummy_for_e0_v0 = True
        is_dummy_for_e0_v1 = True
    else:
        is_dummy_for_e0_v0 = not segment_plane_intersects(
            e0_v0_pos, delta_e0_v0, n, d, 1.0e-6, -1.0e-8, 1.0e-6, False
        )
        is_dummy_for_e0_v1 = not segment_plane_intersects(
            e0_v1_pos, delta_e0_v1, n, d, 1.0e-6, -1.0e-8, 1.0e-6, False
        )

    if delta_e1 == 0.0:
        is_dummy_for_e1_v0 = True
        is_dummy_for_e1_v1 = True
    else:
        is_dummy_for_e1_v0 = not segment_plane_intersects(
            e1_v0_pos, delta_e1_v0, n, d, 1.0e-6, -1.0e-8, 1.0e-6, False
        )
        is_dummy_for_e1_v1 = not segment_plane_intersects(
            e1_v1_pos, delta_e1_v1, n, d, 1.0e-6, -1.0e-8, 1.0e-6, False
        )

    return (
        wp.vector(
            is_dummy_for_e0_v0,
            is_dummy_for_e0_v1,
            is_dummy_for_e1_v0,
            is_dummy_for_e1_v1,
            length=4,
            dtype=wp.bool,
        ),
        n,
        d,
    )


@wp.func
def planar_truncation(
    v: wp.vec3,
    delta_v: wp.vec3,
    n: wp.vec3,
    d: wp.vec3,
    eps: float,
    gamma_r: float,
    gamma_min: float = 1.0e-3,
):
    nv = wp.dot(n, delta_v)
    num = wp.dot(n, d - v)

    if wp.abs(nv) < eps:
        return delta_v

    t = num / nv
    t = wp.max(wp.min(t * gamma_r, t - gamma_min), 0.0)

    if t >= 1.0:
        return delta_v
    return t * delta_v


@wp.func
def planar_truncation_t(
    v: wp.vec3,
    delta_v: wp.vec3,
    n: wp.vec3,
    d: wp.vec3,
    eps: float,
    gamma_r: float,
    gamma_min: float = 1.0e-3,
):
    denom = wp.dot(n, delta_v)

    if wp.abs(denom) < eps:
        return 1.0

    t = wp.dot(n, d - v) / denom

    if t < 0.0:
        return 1.0

    t = wp.clamp(
        wp.min(t * gamma_r, t - gamma_min),
        0.0,
        1.0,
    )
    return t


@wp.kernel
def apply_planar_truncation_parallel_by_collision(
    pos: wp.array[wp.vec3],
    displacement_in: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    collision_info_array: wp.array[TriMeshCollisionInfo],
    parallel_eps: float,
    gamma: float,
    truncation_t_out: wp.array[float],
):
    t_id = wp.tid()
    collision_info = collision_info_array[0]

    primitive_id = t_id // NUM_THREADS_PER_COLLISION_PRIMITIVE
    local_lane = t_id % NUM_THREADS_PER_COLLISION_PRIMITIVE

    # Edge-edge candidates.
    if primitive_id < collision_info.edge_colliding_edges_buffer_sizes.shape[0]:
        e1_idx = primitive_id
        collision_buffer_counter = local_lane
        collision_buffer_offset = collision_info.edge_colliding_edges_offsets[primitive_id]

        while collision_buffer_counter < collision_info.edge_colliding_edges_buffer_sizes[primitive_id]:
            e2_idx = collision_info.edge_colliding_edges[
                2 * (collision_buffer_offset + collision_buffer_counter) + 1
            ]

            if e1_idx != -1 and e2_idx != -1:
                e1_v1 = edge_indices[e1_idx, 2]
                e1_v2 = edge_indices[e1_idx, 3]
                e2_v1 = edge_indices[e2_idx, 2]
                e2_v2 = edge_indices[e2_idx, 3]

                e1_v1_pos = pos[e1_v1]
                e1_v2_pos = pos[e1_v2]
                e2_v1_pos = pos[e2_v1]
                e2_v2_pos = pos[e2_v2]

                delta_e1_v1 = displacement_in[e1_v1]
                delta_e1_v2 = displacement_in[e1_v2]
                delta_e2_v1 = displacement_in[e2_v1]
                delta_e2_v2 = displacement_in[e2_v2]

                is_dummy, n, d = create_edge_edge_division_plane_closest_pt(
                    e1_v1_pos,
                    delta_e1_v1,
                    e1_v2_pos,
                    delta_e1_v2,
                    e2_v1_pos,
                    delta_e2_v1,
                    e2_v2_pos,
                    delta_e2_v2,
                )

                if not is_dummy[0]:
                    t = planar_truncation_t(
                        e1_v1_pos, delta_e1_v1, n, d, parallel_eps, gamma
                    )
                    wp.atomic_min(truncation_t_out, e1_v1, t)

                if not is_dummy[1]:
                    t = planar_truncation_t(
                        e1_v2_pos, delta_e1_v2, n, d, parallel_eps, gamma
                    )
                    wp.atomic_min(truncation_t_out, e1_v2, t)

                if not is_dummy[2]:
                    t = planar_truncation_t(
                        e2_v1_pos, delta_e2_v1, n, d, parallel_eps, gamma
                    )
                    wp.atomic_min(truncation_t_out, e2_v1, t)

                if not is_dummy[3]:
                    t = planar_truncation_t(
                        e2_v2_pos, delta_e2_v2, n, d, parallel_eps, gamma
                    )
                    wp.atomic_min(truncation_t_out, e2_v2, t)

            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    # Vertex-triangle candidates.
    if primitive_id < collision_info.vertex_colliding_triangles_buffer_sizes.shape[0]:
        particle_idx = primitive_id

        colliding_particle_pos = pos[particle_idx]
        colliding_particle_displacement = displacement_in[particle_idx]

        collision_buffer_counter = local_lane
        collision_buffer_offset = collision_info.vertex_colliding_triangles_offsets[primitive_id]

        while collision_buffer_counter < collision_info.vertex_colliding_triangles_buffer_sizes[primitive_id]:
            tri_idx = collision_info.vertex_colliding_triangles[
                2 * (collision_buffer_offset + collision_buffer_counter) + 1
            ]

            if particle_idx != -1 and tri_idx != -1:
                tri_a = tri_indices[tri_idx, 0]
                tri_b = tri_indices[tri_idx, 1]
                tri_c = tri_indices[tri_idx, 2]

                t1 = pos[tri_a]
                t2 = pos[tri_b]
                t3 = pos[tri_c]

                delta_t1 = displacement_in[tri_a]
                delta_t2 = displacement_in[tri_b]
                delta_t3 = displacement_in[tri_c]

                is_dummy, n, d = create_vertex_triangle_division_plane_closest_pt(
                    colliding_particle_pos,
                    colliding_particle_displacement,
                    t1,
                    delta_t1,
                    t2,
                    delta_t2,
                    t3,
                    delta_t3,
                )

                if not is_dummy[0]:
                    t = planar_truncation_t(
                        colliding_particle_pos,
                        colliding_particle_displacement,
                        n, d, parallel_eps, gamma
                    )
                    wp.atomic_min(truncation_t_out, particle_idx, t)

                if not is_dummy[1]:
                    t = planar_truncation_t(t1, delta_t1, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, tri_a, t)

                if not is_dummy[2]:
                    t = planar_truncation_t(t2, delta_t2, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, tri_b, t)

                if not is_dummy[3]:
                    t = planar_truncation_t(t3, delta_t3, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, tri_c, t)

            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE


@wp.kernel
def apply_truncation_ts(
    pos: wp.array[wp.vec3],
    displacement_in: wp.array[wp.vec3],
    truncation_ts: wp.array[float],
    max_displacement: float,
    displacement_out: wp.array[wp.vec3],
    pos_out: wp.array[wp.vec3],
):
    i = wp.tid()
    t = truncation_ts[i]
    particle_displacement = displacement_in[i] * t

    len_displacement = wp.length(particle_displacement)
    if len_displacement > max_displacement:
        particle_displacement = particle_displacement * max_displacement / len_displacement

    displacement_out[i] = particle_displacement
    if pos_out:
        pos_out[i] = pos[i] + particle_displacement
