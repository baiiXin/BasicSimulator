# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Standalone adaptation for Warp/NumPy use without importing Newton.
# Modifications Copyright (c) 2026 Standalone Contact contributors.

import warp as wp

from .geometry import binary_search_i32, triangle_closest_point


# Keep the same flag layout as Newton's geometry kernels.
VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX = wp.constant(0)
TRI_COLLISION_BUFFER_OVERFLOW_INDEX = wp.constant(1)
EDGE_COLLISION_BUFFER_OVERFLOW_INDEX = wp.constant(2)
TRI_TRI_COLLISION_BUFFER_OVERFLOW_INDEX = wp.constant(3)


@wp.kernel
def compute_tri_aabbs(
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    lower_bounds: wp.array[wp.vec3],
    upper_bounds: wp.array[wp.vec3],
):
    t = wp.tid()
    p0 = pos[tri_indices[t, 0]]
    p1 = pos[tri_indices[t, 1]]
    p2 = pos[tri_indices[t, 2]]
    lower_bounds[t] = wp.min(wp.min(p0, p1), p2)
    upper_bounds[t] = wp.max(wp.max(p0, p1), p2)


@wp.kernel
def compute_edge_aabbs(
    pos: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    lower_bounds: wp.array[wp.vec3],
    upper_bounds: wp.array[wp.vec3],
):
    e = wp.tid()
    p0 = pos[edge_indices[e, 2]]
    p1 = pos[edge_indices[e, 3]]
    lower_bounds[e] = wp.min(p0, p1)
    upper_bounds[e] = wp.max(p0, p1)


@wp.func
def vertex_adjacent_to_triangle(v: wp.int32, a: wp.int32, b: wp.int32, c: wp.int32):
    return v == a or v == b or v == c


@wp.kernel
def init_triangle_collision_data_kernel(
    query_radius: float,
    triangle_colliding_vertices_count: wp.array[wp.int32],
    triangle_colliding_vertices_min_dist: wp.array[float],
    resize_flags: wp.array[wp.int32],
):
    """Reset the triangle->vertex reverse index before a VT pass."""
    tri_index = wp.tid()
    triangle_colliding_vertices_count[tri_index] = 0
    triangle_colliding_vertices_min_dist[tri_index] = query_radius

    # Matches Newton's overflow-flag layout. VT is the first pass in detect().
    if tri_index == 0:
        for i in range(4):
            resize_flags[i] = 0


@wp.kernel
def vertex_triangle_collision_detection_kernel(
    max_query_radius: float,
    min_query_radius: float,
    bvh_id: wp.uint64,
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    vertex_colliding_triangles_offsets: wp.array[wp.int32],
    vertex_colliding_triangles_buffer_sizes: wp.array[wp.int32],
    triangle_colliding_vertices_offsets: wp.array[wp.int32],
    triangle_colliding_vertices_buffer_sizes: wp.array[wp.int32],
    vertex_triangle_filtering_list: wp.array[wp.int32],
    vertex_triangle_filtering_list_offsets: wp.array[wp.int32],
    min_distance_filtering_ref_pos: wp.array[wp.vec3],
    # outputs
    vertex_colliding_triangles: wp.array[wp.int32],
    vertex_colliding_triangles_count: wp.array[wp.int32],
    vertex_colliding_triangles_min_dist: wp.array[float],
    triangle_colliding_vertices: wp.array[wp.int32],
    triangle_colliding_vertices_count: wp.array[wp.int32],
    triangle_colliding_vertices_min_dist: wp.array[float],
    resize_flags: wp.array[wp.int32],
):
    """Discrete VT proximity detection with Newton-compatible forward/reverse buffers."""
    v_index = wp.tid()
    v = pos[v_index]

    vertex_buffer_offset = vertex_colliding_triangles_offsets[v_index]
    vertex_buffer_size = vertex_colliding_triangles_offsets[v_index + 1] - vertex_buffer_offset

    r = max_query_radius
    lower = wp.vec3(v[0] - r, v[1] - r, v[2] - r)
    upper = wp.vec3(v[0] + r, v[1] + r, v[2] + r)

    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    tri_index = wp.int32(0)
    vertex_num_collisions = wp.int32(0)
    min_dis_to_tris = max_query_radius

    while wp.bvh_query_next(query, tri_index):
        t0 = tri_indices[tri_index, 0]
        t1 = tri_indices[tri_index, 1]
        t2 = tri_indices[tri_index, 2]

        # Directly incident faces are not self-contact candidates.
        if vertex_adjacent_to_triangle(v_index, t0, t1, t2):
            continue

        # n-ring / external topology filter. The rows are sorted CSR slices.
        if vertex_triangle_filtering_list:
            fl_start = vertex_triangle_filtering_list_offsets[v_index]
            fl_end = vertex_triangle_filtering_list_offsets[v_index + 1]
            if fl_end > fl_start:
                first_val = vertex_triangle_filtering_list[fl_start]
                last_val = vertex_triangle_filtering_list[fl_end - 1]
                if tri_index >= first_val and tri_index <= last_val:
                    idx = binary_search_i32(vertex_triangle_filtering_list, tri_index, fl_start, fl_end)
                    if idx > fl_start and vertex_triangle_filtering_list[idx - 1] == tri_index:
                        continue

        p0 = pos[t0]
        p1 = pos[t1]
        p2 = pos[t2]
        closest, _bary, _feature_type = triangle_closest_point(p0, p1, p2, v)
        dist = wp.length(closest - v)

        # Optional rest/reference-shape exclusion.
        if min_distance_filtering_ref_pos and min_query_radius > 0.0:
            closest_ref, _bary_ref, _feature_type_ref = triangle_closest_point(
                min_distance_filtering_ref_pos[t0],
                min_distance_filtering_ref_pos[t1],
                min_distance_filtering_ref_pos[t2],
                min_distance_filtering_ref_pos[v_index],
            )
            dist_ref = wp.length(closest_ref - min_distance_filtering_ref_pos[v_index])
            if dist_ref < min_query_radius:
                continue

        if dist < max_query_radius:
            min_dis_to_tris = wp.min(min_dis_to_tris, dist)

            # Forward index: vertex -> colliding triangles, stored as [v, tri] pairs.
            if vertex_num_collisions < vertex_buffer_size:
                write = vertex_buffer_offset + vertex_num_collisions
                vertex_colliding_triangles[2 * write] = v_index
                vertex_colliding_triangles[2 * write + 1] = tri_index
            else:
                resize_flags[VERTEX_COLLISION_BUFFER_OVERFLOW_INDEX] = 1
            vertex_num_collisions += 1

            # Reverse index: triangle -> colliding vertices.
            if triangle_colliding_vertices:
                wp.atomic_min(triangle_colliding_vertices_min_dist, tri_index, dist)
                tri_buffer_size = triangle_colliding_vertices_buffer_sizes[tri_index]
                tri_num_collisions = wp.atomic_add(triangle_colliding_vertices_count, tri_index, 1)
                if tri_num_collisions < tri_buffer_size:
                    tri_buffer_offset = triangle_colliding_vertices_offsets[tri_index]
                    triangle_colliding_vertices[tri_buffer_offset + tri_num_collisions] = v_index
                else:
                    resize_flags[TRI_COLLISION_BUFFER_OVERFLOW_INDEX] = 1

    # Count records the true candidate count even if the preallocated slice overflowed.
    vertex_colliding_triangles_count[v_index] = vertex_num_collisions
    vertex_colliding_triangles_min_dist[v_index] = min_dis_to_tris


@wp.kernel
def edge_edge_collision_detection_kernel(
    max_query_radius: float,
    min_query_radius: float,
    bvh_id: wp.uint64,
    pos: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    edge_colliding_edges_offsets: wp.array[wp.int32],
    edge_colliding_edges_buffer_sizes: wp.array[wp.int32],
    edge_edge_parallel_epsilon: float,
    edge_filtering_list: wp.array[wp.int32],
    edge_filtering_list_offsets: wp.array[wp.int32],
    min_distance_filtering_ref_pos: wp.array[wp.vec3],
    # outputs
    edge_colliding_edges: wp.array[wp.int32],
    edge_colliding_edges_count: wp.array[wp.int32],
    edge_colliding_edges_min_dist: wp.array[float],
    resize_flags: wp.array[wp.int32],
):
    e0 = wp.tid()
    e0_v0 = edge_indices[e0, 2]
    e0_v1 = edge_indices[e0, 3]
    p0 = pos[e0_v0]
    p1 = pos[e0_v1]

    r = max_query_radius
    lower = wp.min(p0, p1)
    upper = wp.max(p0, p1)
    lower = wp.vec3(lower[0] - r, lower[1] - r, lower[2] - r)
    upper = wp.vec3(upper[0] + r, upper[1] + r, upper[2] + r)

    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    e1 = wp.int32(0)
    edge_num_collisions = wp.int32(0)
    min_dis_to_edges = max_query_radius

    while wp.bvh_query_next(query, e1):
        e1_v0 = edge_indices[e1, 2]
        e1_v1 = edge_indices[e1, 3]

        if e0_v0 == e1_v0 or e0_v0 == e1_v1 or e0_v1 == e1_v0 or e0_v1 == e1_v1:
            continue

        if edge_filtering_list:
            fl_start = edge_filtering_list_offsets[e0]
            fl_end = edge_filtering_list_offsets[e0 + 1]
            if fl_end > fl_start:
                first_val = edge_filtering_list[fl_start]
                last_val = edge_filtering_list[fl_end - 1]
                if e1 >= first_val and e1 <= last_val:
                    idx = binary_search_i32(edge_filtering_list, e1, fl_start, fl_end)
                    if idx > fl_start and edge_filtering_list[idx - 1] == e1:
                        continue

        q0 = pos[e1_v0]
        q1 = pos[e1_v1]
        std = wp.closest_point_edge_edge(p0, p1, q0, q1, edge_edge_parallel_epsilon)
        dist = std[2]

        if min_distance_filtering_ref_pos and min_query_radius > 0.0:
            std_ref = wp.closest_point_edge_edge(
                min_distance_filtering_ref_pos[e0_v0],
                min_distance_filtering_ref_pos[e0_v1],
                min_distance_filtering_ref_pos[e1_v0],
                min_distance_filtering_ref_pos[e1_v1],
                edge_edge_parallel_epsilon,
            )
            if std_ref[2] < min_query_radius:
                continue

        if dist < max_query_radius:
            buffer_offset = edge_colliding_edges_offsets[e0]
            buffer_size = edge_colliding_edges_offsets[e0 + 1] - buffer_offset
            min_dis_to_edges = wp.min(min_dis_to_edges, dist)

            if edge_num_collisions < buffer_size:
                write = buffer_offset + edge_num_collisions
                edge_colliding_edges[2 * write] = e0
                edge_colliding_edges[2 * write + 1] = e1
            else:
                resize_flags[EDGE_COLLISION_BUFFER_OVERFLOW_INDEX] = 1
            edge_num_collisions += 1

    edge_colliding_edges_count[e0] = edge_num_collisions
    edge_colliding_edges_min_dist[e0] = min_dis_to_edges
