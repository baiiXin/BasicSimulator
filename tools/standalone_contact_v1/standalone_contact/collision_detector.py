# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Standalone adaptation for Warp/NumPy use without importing Newton.
# Modifications Copyright (c) 2026 Standalone Contact contributors.

from __future__ import annotations

import numpy as np
import warp as wp

from .collision_kernels import (
    compute_edge_aabbs,
    compute_tri_aabbs,
    edge_edge_collision_detection_kernel,
    vertex_triangle_collision_detection_kernel,
)
from .mesh_topology import (
    MeshTopology,
    build_edge_n_ring_edge_collision_filter,
    build_mesh_topology,
    build_vertex_n_ring_tris_collision_filter,
)


@wp.struct
class TriMeshCollisionInfo:
    # VT: flattened [vertex, triangle] pairs.
    vertex_colliding_triangles: wp.array[wp.int32]
    vertex_colliding_triangles_offsets: wp.array[wp.int32]
    vertex_colliding_triangles_buffer_sizes: wp.array[wp.int32]
    vertex_colliding_triangles_count: wp.array[wp.int32]
    vertex_colliding_triangles_min_dist: wp.array[float]

    # Kept for Newton/DAT structural compatibility; not populated in v1.
    triangle_colliding_vertices: wp.array[wp.int32]
    triangle_colliding_vertices_offsets: wp.array[wp.int32]
    triangle_colliding_vertices_buffer_sizes: wp.array[wp.int32]
    triangle_colliding_vertices_count: wp.array[wp.int32]
    triangle_colliding_vertices_min_dist: wp.array[float]

    # EE: flattened [edge0, edge1] pairs.
    edge_colliding_edges: wp.array[wp.int32]
    edge_colliding_edges_offsets: wp.array[wp.int32]
    edge_colliding_edges_buffer_sizes: wp.array[wp.int32]
    edge_colliding_edges_count: wp.array[wp.int32]
    edge_colliding_edges_min_dist: wp.array[float]


@wp.func
def get_vertex_colliding_triangles_count(col_info: TriMeshCollisionInfo, v: int):
    return wp.min(
        col_info.vertex_colliding_triangles_count[v],
        col_info.vertex_colliding_triangles_buffer_sizes[v],
    )


@wp.func
def get_vertex_colliding_triangles(col_info: TriMeshCollisionInfo, v: int, i_collision: int):
    offset = col_info.vertex_colliding_triangles_offsets[v]
    return col_info.vertex_colliding_triangles[2 * (offset + i_collision) + 1]


@wp.func
def get_vertex_collision_buffer_vertex_index(col_info: TriMeshCollisionInfo, v: int, i_collision: int):
    offset = col_info.vertex_colliding_triangles_offsets[v]
    return col_info.vertex_colliding_triangles[2 * (offset + i_collision)]


@wp.func
def get_edge_colliding_edges_count(col_info: TriMeshCollisionInfo, e: int):
    return wp.min(
        col_info.edge_colliding_edges_count[e],
        col_info.edge_colliding_edges_buffer_sizes[e],
    )


@wp.func
def get_edge_colliding_edges(col_info: TriMeshCollisionInfo, e: int, i_collision: int):
    offset = col_info.edge_colliding_edges_offsets[e]
    return col_info.edge_colliding_edges[2 * (offset + i_collision) + 1]


@wp.func
def get_edge_collision_buffer_edge_index(col_info: TriMeshCollisionInfo, e: int, i_collision: int):
    offset = col_info.edge_colliding_edges_offsets[e]
    return col_info.edge_colliding_edges[2 * (offset + i_collision)]


def _fixed_buffer_layout(count: int, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.full(count, capacity, dtype=np.int32)
    offsets = np.zeros(count + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(sizes)
    return sizes, offsets


class TriMeshCollisionDetector:
    """Single-mesh standalone detector matching Newton VBD's VT/EE organization.

    Runtime dependency: NVIDIA Warp only.
    No Newton import is used.

    Notes
    -----
    * Multi-world/grouped BVH support is intentionally omitted in v1.
    * Fixed-capacity buffers match Newton's hot-path layout.
    * `detect()` clears overflow flags at the beginning of each full pass.
    """

    def __init__(
        self,
        vertices,
        triangles,
        *,
        device=None,
        vertex_collision_buffer_pre_alloc: int = 8,
        edge_collision_buffer_pre_alloc: int = 8,
        topological_contact_filter_threshold: int = 0,
        external_vertex_triangle_filtering_map: dict[int, set[int]] | None = None,
        external_edge_edge_filtering_map: dict[int, set[int]] | None = None,
        edge_edge_parallel_epsilon: float = 1.0e-5,
        collision_detection_block_size: int = 16,
    ):
        if isinstance(vertices, wp.array):
            self.vertex_positions = vertices
            self.device = vertices.device
            particle_count = len(vertices)
        else:
            vertices_np = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
            self.device = wp.get_device(device)
            self.vertex_positions = wp.array(vertices_np, dtype=wp.vec3, device=self.device)
            particle_count = vertices_np.shape[0]

        self.topology: MeshTopology = build_mesh_topology(particle_count, np.asarray(triangles, dtype=np.int32))
        self.particle_count = particle_count
        self.tri_count = self.topology.triangles.shape[0]
        self.edge_count = self.topology.edge_indices.shape[0]
        self.edge_edge_parallel_epsilon = float(edge_edge_parallel_epsilon)
        self.collision_detection_block_size = int(collision_detection_block_size)

        self.tri_indices = wp.array(self.topology.triangles, dtype=wp.int32, device=self.device)
        self.edge_indices = wp.array(self.topology.edge_indices, dtype=wp.int32, device=self.device)

        vt_list, vt_offsets = build_vertex_n_ring_tris_collision_filter(
            topological_contact_filter_threshold,
            particle_count,
            self.topology.edge_indices,
            self.topology.v_adj_edges,
            self.topology.v_adj_edges_offsets,
            self.topology.v_adj_tris,
            self.topology.v_adj_tris_offsets,
        )
        ee_list, ee_offsets = build_edge_n_ring_edge_collision_filter(
            topological_contact_filter_threshold,
            self.topology.edge_indices,
            self.topology.v_adj_edges,
            self.topology.v_adj_edges_offsets,
        )

        # Merge optional external filter maps.
        if external_vertex_triangle_filtering_map is not None:
            sets = [set() for _ in range(particle_count)]
            if vt_list is not None:
                for v in range(particle_count):
                    sets[v].update(vt_list[vt_offsets[v] : vt_offsets[v + 1]].tolist())
            for v, vals in external_vertex_triangle_filtering_map.items():
                sets[int(v)].update(int(x) for x in vals)
            from .mesh_topology import set_to_csr
            vt_list, vt_offsets = set_to_csr(sets)

        if external_edge_edge_filtering_map is not None:
            sets = [set() for _ in range(self.edge_count)]
            if ee_list is not None:
                for e in range(self.edge_count):
                    sets[e].update(ee_list[ee_offsets[e] : ee_offsets[e + 1]].tolist())
            for e, vals in external_edge_edge_filtering_map.items():
                sets[int(e)].update(int(x) for x in vals)
            from .mesh_topology import set_to_csr
            ee_list, ee_offsets = set_to_csr(sets)

        self.vertex_triangle_filtering_list = (
            None if vt_list is None else wp.array(vt_list, dtype=wp.int32, device=self.device)
        )
        self.vertex_triangle_filtering_list_offsets = (
            None if vt_offsets is None else wp.array(vt_offsets, dtype=wp.int32, device=self.device)
        )
        self.edge_filtering_list = (
            None if ee_list is None else wp.array(ee_list, dtype=wp.int32, device=self.device)
        )
        self.edge_filtering_list_offsets = (
            None if ee_offsets is None else wp.array(ee_offsets, dtype=wp.int32, device=self.device)
        )

        # Triangle BVH.
        self.lower_bounds_tris = wp.empty(self.tri_count, dtype=wp.vec3, device=self.device)
        self.upper_bounds_tris = wp.empty(self.tri_count, dtype=wp.vec3, device=self.device)
        wp.launch(
            compute_tri_aabbs,
            dim=self.tri_count,
            inputs=[self.vertex_positions, self.tri_indices, self.lower_bounds_tris, self.upper_bounds_tris],
            device=self.device,
        )
        self.bvh_tris = wp.Bvh(self.lower_bounds_tris, self.upper_bounds_tris)

        # Edge BVH.
        self.lower_bounds_edges = wp.empty(self.edge_count, dtype=wp.vec3, device=self.device)
        self.upper_bounds_edges = wp.empty(self.edge_count, dtype=wp.vec3, device=self.device)
        wp.launch(
            compute_edge_aabbs,
            dim=self.edge_count,
            inputs=[self.vertex_positions, self.edge_indices, self.lower_bounds_edges, self.upper_bounds_edges],
            device=self.device,
        )
        self.bvh_edges = wp.Bvh(self.lower_bounds_edges, self.upper_bounds_edges)

        # VT buffers.
        vt_sizes_np, vt_offsets_np = _fixed_buffer_layout(
            self.particle_count, vertex_collision_buffer_pre_alloc
        )
        self.vertex_colliding_triangles_buffer_sizes = wp.array(
            vt_sizes_np, dtype=wp.int32, device=self.device
        )
        self.vertex_colliding_triangles_offsets = wp.array(
            vt_offsets_np, dtype=wp.int32, device=self.device
        )
        self.vertex_colliding_triangles = wp.full(
            2 * int(vt_offsets_np[-1]), -1, dtype=wp.int32, device=self.device
        )
        self.vertex_colliding_triangles_count = wp.zeros(
            self.particle_count, dtype=wp.int32, device=self.device
        )
        self.vertex_colliding_triangles_min_dist = wp.empty(
            self.particle_count, dtype=float, device=self.device
        )

        # EE buffers.
        ee_sizes_np, ee_offsets_np = _fixed_buffer_layout(
            self.edge_count, edge_collision_buffer_pre_alloc
        )
        self.edge_colliding_edges_buffer_sizes = wp.array(
            ee_sizes_np, dtype=wp.int32, device=self.device
        )
        self.edge_colliding_edges_offsets = wp.array(
            ee_offsets_np, dtype=wp.int32, device=self.device
        )
        self.edge_colliding_edges = wp.full(
            2 * int(ee_offsets_np[-1]), -1, dtype=wp.int32, device=self.device
        )
        self.edge_colliding_edges_count = wp.zeros(
            self.edge_count, dtype=wp.int32, device=self.device
        )
        self.edge_colliding_edges_min_dist = wp.empty(
            self.edge_count, dtype=float, device=self.device
        )

        # Compatibility placeholders for triangle -> vertex buffers.
        self.triangle_colliding_vertices_min_dist = wp.empty(
            self.tri_count, dtype=float, device=self.device
        )
        self.resize_flags = wp.zeros(4, dtype=wp.int32, device=self.device)
        self.collision_info = self._pack_collision_info()

    def _pack_collision_info(self):
        info = TriMeshCollisionInfo()
        info.vertex_colliding_triangles = self.vertex_colliding_triangles
        info.vertex_colliding_triangles_offsets = self.vertex_colliding_triangles_offsets
        info.vertex_colliding_triangles_buffer_sizes = self.vertex_colliding_triangles_buffer_sizes
        info.vertex_colliding_triangles_count = self.vertex_colliding_triangles_count
        info.vertex_colliding_triangles_min_dist = self.vertex_colliding_triangles_min_dist
        info.triangle_colliding_vertices_min_dist = self.triangle_colliding_vertices_min_dist
        info.edge_colliding_edges = self.edge_colliding_edges
        info.edge_colliding_edges_offsets = self.edge_colliding_edges_offsets
        info.edge_colliding_edges_buffer_sizes = self.edge_colliding_edges_buffer_sizes
        info.edge_colliding_edges_count = self.edge_colliding_edges_count
        info.edge_colliding_edges_min_dist = self.edge_colliding_edges_min_dist
        return info

    def _coerce_positions(self, positions):
        if positions is None:
            return self.vertex_positions
        if isinstance(positions, wp.array):
            if positions.device != self.device:
                raise ValueError("positions must be on the detector device")
            if len(positions) != self.particle_count:
                raise ValueError("positions length mismatch")
            return positions
        arr = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        if arr.shape[0] != self.particle_count:
            raise ValueError("positions length mismatch")
        return wp.array(arr, dtype=wp.vec3, device=self.device)

    def refit_triangles(self):
        wp.launch(
            compute_tri_aabbs,
            dim=self.tri_count,
            inputs=[self.vertex_positions, self.tri_indices, self.lower_bounds_tris, self.upper_bounds_tris],
            device=self.device,
        )
        self.bvh_tris.refit()

    def refit_edges(self):
        wp.launch(
            compute_edge_aabbs,
            dim=self.edge_count,
            inputs=[self.vertex_positions, self.edge_indices, self.lower_bounds_edges, self.upper_bounds_edges],
            device=self.device,
        )
        self.bvh_edges.refit()

    def refit(self, new_pos=None):
        if new_pos is not None:
            self.vertex_positions = self._coerce_positions(new_pos)
        self.refit_triangles()
        self.refit_edges()

    def vertex_triangle_collision_detection(
        self,
        max_query_radius: float,
        min_query_radius: float = 0.0,
        min_distance_filtering_ref_pos=None,
    ):
        self.vertex_colliding_triangles.fill_(-1)
        ref_pos = self._coerce_positions(min_distance_filtering_ref_pos)

        wp.launch(
            vertex_triangle_collision_detection_kernel,
            dim=self.particle_count,
            inputs=[
                float(max_query_radius),
                float(min_query_radius),
                self.bvh_tris.id,
                self.vertex_positions,
                self.tri_indices,
                self.vertex_colliding_triangles_offsets,
                self.vertex_colliding_triangles_buffer_sizes,
                self.vertex_triangle_filtering_list,
                self.vertex_triangle_filtering_list_offsets,
                ref_pos,
                self.vertex_colliding_triangles,
                self.vertex_colliding_triangles_count,
                self.vertex_colliding_triangles_min_dist,
                self.resize_flags,
            ],
            device=self.device,
            block_dim=self.collision_detection_block_size,
        )

    def edge_edge_collision_detection(
        self,
        max_query_radius: float,
        min_query_radius: float = 0.0,
        min_distance_filtering_ref_pos=None,
    ):
        self.edge_colliding_edges.fill_(-1)
        ref_pos = self._coerce_positions(min_distance_filtering_ref_pos)

        wp.launch(
            edge_edge_collision_detection_kernel,
            dim=self.edge_count,
            inputs=[
                float(max_query_radius),
                float(min_query_radius),
                self.bvh_edges.id,
                self.vertex_positions,
                self.edge_indices,
                self.edge_colliding_edges_offsets,
                self.edge_colliding_edges_buffer_sizes,
                self.edge_edge_parallel_epsilon,
                self.edge_filtering_list,
                self.edge_filtering_list_offsets,
                ref_pos,
                self.edge_colliding_edges,
                self.edge_colliding_edges_count,
                self.edge_colliding_edges_min_dist,
                self.resize_flags,
            ],
            device=self.device,
            block_dim=self.collision_detection_block_size,
        )

    def detect(
        self,
        q=None,
        *,
        margin: float,
        rest_q=None,
        rest_exclusion_radius: float = 0.0,
    ):
        self.resize_flags.zero_()
        self.refit(q)
        ref = self.vertex_positions if rest_q is None else rest_q
        self.vertex_triangle_collision_detection(
            margin,
            min_query_radius=rest_exclusion_radius,
            min_distance_filtering_ref_pos=ref,
        )
        self.edge_edge_collision_detection(
            margin,
            min_query_radius=rest_exclusion_radius,
            min_distance_filtering_ref_pos=ref,
        )
        return self.collision_info

    def overflow_flags(self) -> np.ndarray:
        return self.resize_flags.numpy().copy()
