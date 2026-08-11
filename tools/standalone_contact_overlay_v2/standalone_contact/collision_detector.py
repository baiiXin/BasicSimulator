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
    init_triangle_collision_data_kernel,
    vertex_triangle_collision_detection_kernel,
)
from .mesh_topology import (
    MeshTopology,
    build_edge_n_ring_edge_collision_filter,
    build_mesh_topology,
    build_vertex_n_ring_tris_collision_filter,
    set_to_csr,
)


@wp.struct
class TriMeshCollisionInfo:
    # size = 2 * sum(vertex_colliding_triangles_buffer_sizes)
    # each record is [vertex_index, triangle_index]
    vertex_colliding_triangles: wp.array[wp.int32]
    vertex_colliding_triangles_offsets: wp.array[wp.int32]
    vertex_colliding_triangles_buffer_sizes: wp.array[wp.int32]
    vertex_colliding_triangles_count: wp.array[wp.int32]
    vertex_colliding_triangles_min_dist: wp.array[float]

    # reverse VT index: triangle -> colliding vertices
    triangle_colliding_vertices: wp.array[wp.int32]
    triangle_colliding_vertices_offsets: wp.array[wp.int32]
    triangle_colliding_vertices_buffer_sizes: wp.array[wp.int32]
    triangle_colliding_vertices_count: wp.array[wp.int32]
    triangle_colliding_vertices_min_dist: wp.array[float]

    # size = 2 * sum(edge_colliding_edges_buffer_sizes)
    # each record is [edge_index, colliding_edge_index]
    edge_colliding_edges: wp.array[wp.int32]
    edge_colliding_edges_offsets: wp.array[wp.int32]
    edge_colliding_edges_buffer_sizes: wp.array[wp.int32]
    edge_colliding_edges_count: wp.array[wp.int32]
    edge_colliding_edges_min_dist: wp.array[float]


@wp.func
def get_vertex_colliding_triangles_count(col_info: TriMeshCollisionInfo, v: int):
    return wp.min(col_info.vertex_colliding_triangles_count[v], col_info.vertex_colliding_triangles_buffer_sizes[v])


@wp.func
def get_vertex_colliding_triangles(col_info: TriMeshCollisionInfo, v: int, i_collision: int):
    offset = col_info.vertex_colliding_triangles_offsets[v]
    return col_info.vertex_colliding_triangles[2 * (offset + i_collision) + 1]


@wp.func
def get_vertex_collision_buffer_vertex_index(col_info: TriMeshCollisionInfo, v: int, i_collision: int):
    offset = col_info.vertex_colliding_triangles_offsets[v]
    return col_info.vertex_colliding_triangles[2 * (offset + i_collision)]


@wp.func
def get_triangle_colliding_vertices_count(col_info: TriMeshCollisionInfo, tri: int):
    return wp.min(col_info.triangle_colliding_vertices_count[tri], col_info.triangle_colliding_vertices_buffer_sizes[tri])


@wp.func
def get_triangle_colliding_vertices(col_info: TriMeshCollisionInfo, tri: int, i_collision: int):
    offset = col_info.triangle_colliding_vertices_offsets[tri]
    return col_info.triangle_colliding_vertices[offset + i_collision]


@wp.func
def get_edge_colliding_edges_count(col_info: TriMeshCollisionInfo, e: int):
    return wp.min(col_info.edge_colliding_edges_count[e], col_info.edge_colliding_edges_buffer_sizes[e])


@wp.func
def get_edge_colliding_edges(col_info: TriMeshCollisionInfo, e: int, i_collision: int):
    offset = col_info.edge_colliding_edges_offsets[e]
    return col_info.edge_colliding_edges[2 * (offset + i_collision) + 1]


@wp.func
def get_edge_collision_buffer_edge_index(col_info: TriMeshCollisionInfo, e: int, i_collision: int):
    offset = col_info.edge_colliding_edges_offsets[e]
    return col_info.edge_colliding_edges[2 * (offset + i_collision)]


class CollisionBufferCapacityError(RuntimeError):
    pass


def _validate_capacity(name: str, pre_alloc: int, max_alloc: int) -> None:
    if pre_alloc <= 0:
        raise ValueError(f"{name}_pre_alloc must be > 0, got {pre_alloc}")
    if max_alloc < pre_alloc:
        raise ValueError(f"{name}_max_alloc ({max_alloc}) must be >= pre_alloc ({pre_alloc})")


def _fixed_buffer_layout(count: int, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.full(count, capacity, dtype=np.int32)
    offsets = np.zeros(count + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(sizes, dtype=np.int64).astype(np.int32)
    return sizes, offsets


def _grow_layout_from_counts(
    counts: np.ndarray,
    current_sizes: np.ndarray,
    max_alloc: int,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Grow only rows that overflowed; old records are discarded and detection is rerun."""
    counts = np.asarray(counts, dtype=np.int64)
    sizes = np.asarray(current_sizes, dtype=np.int64).copy()

    too_large = np.flatnonzero(counts > max_alloc)
    if too_large.size:
        i = int(too_large[0])
        raise CollisionBufferCapacityError(
            f"{name} row {i} requires {int(counts[i])} records, exceeding max_alloc={max_alloc}. "
            "Increase the corresponding *_max_alloc setting."
        )

    grow = counts > sizes
    if not np.any(grow):
        return sizes.astype(np.int32), np.concatenate(([0], np.cumsum(sizes))).astype(np.int32)

    # Grow geometrically but never smaller than the observed required count.
    doubled = np.maximum(1, sizes[grow] * 2)
    sizes[grow] = np.minimum(max_alloc, np.maximum(counts[grow], doubled))

    offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)
    if offsets[-1] > np.iinfo(np.int32).max:
        raise CollisionBufferCapacityError(f"{name} flattened buffer exceeds int32 offset range")
    return sizes.astype(np.int32), offsets.astype(np.int32)


class TriMeshCollisionDetector:
    """Standalone single-mesh VT/EE detector using NumPy + NVIDIA Warp only.

    The GPU hot path mirrors Newton VBD's organization:
      * one Warp thread per vertex for VT,
      * one Warp thread per edge for EE,
      * triangle and edge ``wp.Bvh`` objects are built once and then refit,
      * collision rows use fixed-stride/CSR-like preallocated buffers.

    ``auto_resize`` is a standalone robustness extension. Current Newton SolverVBD
    uses fixed particle contact buffers. When auto-resize is enabled, this class
    reads overflow flags/counts back to the host, reallocates only on overflow,
    and reruns the affected detection pass. That host synchronization is why the
    default is ``False`` for Newton-like hot-path performance.
    """

    def __init__(
        self,
        vertices,
        triangles,
        *,
        device=None,
        record_triangle_contacting_vertices: bool = False,
        vertex_collision_buffer_pre_alloc: int = 8,
        vertex_collision_buffer_max_alloc: int = 256,
        triangle_collision_buffer_pre_alloc: int = 16,
        triangle_collision_buffer_max_alloc: int = 256,
        edge_collision_buffer_pre_alloc: int = 8,
        edge_collision_buffer_max_alloc: int = 256,
        topological_contact_filter_threshold: int = 0,
        external_vertex_triangle_filtering_map: dict[int, set[int] | list[int]] | None = None,
        external_edge_edge_filtering_map: dict[int, set[int] | list[int]] | None = None,
        edge_edge_parallel_epsilon: float = 1.0e-5,
        collision_detection_block_size: int = 16,
        auto_resize: bool = False,
    ):
        _validate_capacity("vertex_collision_buffer", vertex_collision_buffer_pre_alloc, vertex_collision_buffer_max_alloc)
        _validate_capacity("edge_collision_buffer", edge_collision_buffer_pre_alloc, edge_collision_buffer_max_alloc)
        if record_triangle_contacting_vertices:
            _validate_capacity(
                "triangle_collision_buffer",
                triangle_collision_buffer_pre_alloc,
                triangle_collision_buffer_max_alloc,
            )

        self.auto_resize = bool(auto_resize)
        self.record_triangle_contacting_vertices = bool(record_triangle_contacting_vertices)
        self.vertex_collision_buffer_max_alloc = int(vertex_collision_buffer_max_alloc)
        self.triangle_collision_buffer_max_alloc = int(triangle_collision_buffer_max_alloc)
        self.edge_collision_buffer_max_alloc = int(edge_collision_buffer_max_alloc)
        self.edge_edge_parallel_epsilon = float(edge_edge_parallel_epsilon)
        self.collision_detection_block_size = int(collision_detection_block_size)

        if self.collision_detection_block_size <= 0:
            raise ValueError("collision_detection_block_size must be > 0")

        if isinstance(vertices, wp.array):
            self.vertex_positions = vertices
            self.device = vertices.device
            self.particle_count = len(vertices)
        else:
            vertices_np = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
            self.device = wp.get_device(device)
            self.vertex_positions = wp.array(vertices_np, dtype=wp.vec3, device=self.device)
            self.particle_count = vertices_np.shape[0]

        self.topology: MeshTopology = build_mesh_topology(
            self.particle_count,
            np.asarray(triangles, dtype=np.int32),
        )
        self.tri_count = self.topology.triangles.shape[0]
        self.edge_count = self.topology.edge_indices.shape[0]

        self.tri_indices = wp.array(self.topology.triangles, dtype=wp.int32, device=self.device)
        self.edge_indices = wp.array(self.topology.edge_indices, dtype=wp.int32, device=self.device)

        self._build_filters(
            topological_contact_filter_threshold,
            external_vertex_triangle_filtering_map,
            external_edge_edge_filtering_map,
        )
        self._build_bvhs()
        self._allocate_collision_buffers(
            vertex_collision_buffer_pre_alloc,
            triangle_collision_buffer_pre_alloc,
            edge_collision_buffer_pre_alloc,
        )
        self.resize_flags = wp.zeros(shape=(4,), dtype=wp.int32, device=self.device)
        self.collision_info = self._pack_collision_info()

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def _build_filters(self, threshold, external_vt, external_ee):
        vt_list, vt_offsets = build_vertex_n_ring_tris_collision_filter(
            threshold,
            self.particle_count,
            self.topology.edge_indices,
            self.topology.v_adj_edges,
            self.topology.v_adj_edges_offsets,
            self.topology.v_adj_tris,
            self.topology.v_adj_tris_offsets,
        )
        ee_list, ee_offsets = build_edge_n_ring_edge_collision_filter(
            threshold,
            self.topology.edge_indices,
            self.topology.v_adj_edges,
            self.topology.v_adj_edges_offsets,
        )

        if external_vt is not None:
            sets = [set() for _ in range(self.particle_count)]
            if vt_list is not None:
                for v in range(self.particle_count):
                    sets[v].update(vt_list[vt_offsets[v] : vt_offsets[v + 1]].tolist())
            for v, vals in external_vt.items():
                sets[int(v)].update(int(x) for x in vals)
            vt_list, vt_offsets = set_to_csr(sets)

        if external_ee is not None:
            sets = [set() for _ in range(self.edge_count)]
            if ee_list is not None:
                for e in range(self.edge_count):
                    sets[e].update(ee_list[ee_offsets[e] : ee_offsets[e + 1]].tolist())
            for e, vals in external_ee.items():
                sets[int(e)].update(int(x) for x in vals)
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

    def _build_bvhs(self):
        self.lower_bounds_tris = wp.empty(shape=(self.tri_count,), dtype=wp.vec3, device=self.device)
        self.upper_bounds_tris = wp.empty(shape=(self.tri_count,), dtype=wp.vec3, device=self.device)
        if self.tri_count:
            wp.launch(
                kernel=compute_tri_aabbs,
                dim=self.tri_count,
                inputs=[self.vertex_positions, self.tri_indices, self.lower_bounds_tris, self.upper_bounds_tris],
                device=self.device,
            )
        self.bvh_tris = wp.Bvh(self.lower_bounds_tris, self.upper_bounds_tris)

        self.lower_bounds_edges = wp.empty(shape=(self.edge_count,), dtype=wp.vec3, device=self.device)
        self.upper_bounds_edges = wp.empty(shape=(self.edge_count,), dtype=wp.vec3, device=self.device)
        if self.edge_count:
            wp.launch(
                kernel=compute_edge_aabbs,
                dim=self.edge_count,
                inputs=[self.vertex_positions, self.edge_indices, self.lower_bounds_edges, self.upper_bounds_edges],
                device=self.device,
            )
        self.bvh_edges = wp.Bvh(self.lower_bounds_edges, self.upper_bounds_edges)

    def _allocate_collision_buffers(self, vertex_capacity, triangle_capacity, edge_capacity):
        vt_sizes, vt_offsets = _fixed_buffer_layout(self.particle_count, int(vertex_capacity))
        self._replace_vertex_buffer(vt_sizes, vt_offsets)

        if self.record_triangle_contacting_vertices:
            tri_sizes, tri_offsets = _fixed_buffer_layout(self.tri_count, int(triangle_capacity))
            self._replace_triangle_buffer(tri_sizes, tri_offsets)
        else:
            self.triangle_colliding_vertices = None
            self.triangle_colliding_vertices_offsets = None
            self.triangle_colliding_vertices_buffer_sizes = None
            self.triangle_colliding_vertices_count = None

        # This min-distance array exists even when the reverse list is disabled,
        # matching Newton's detector.
        self.triangle_colliding_vertices_min_dist = wp.empty(
            shape=(self.tri_count,), dtype=float, device=self.device
        )

        ee_sizes, ee_offsets = _fixed_buffer_layout(self.edge_count, int(edge_capacity))
        self._replace_edge_buffer(ee_sizes, ee_offsets)

    def _replace_vertex_buffer(self, sizes_np: np.ndarray, offsets_np: np.ndarray):
        self.vertex_colliding_triangles_buffer_sizes = wp.array(sizes_np, dtype=wp.int32, device=self.device)
        self.vertex_colliding_triangles_offsets = wp.array(offsets_np, dtype=wp.int32, device=self.device)
        self.vertex_colliding_triangles = wp.full(
            shape=(2 * int(offsets_np[-1]),), value=-1, dtype=wp.int32, device=self.device
        )
        if not hasattr(self, "vertex_colliding_triangles_count"):
            self.vertex_colliding_triangles_count = wp.zeros(
                shape=(self.particle_count,), dtype=wp.int32, device=self.device
            )
            self.vertex_colliding_triangles_min_dist = wp.empty(
                shape=(self.particle_count,), dtype=float, device=self.device
            )

    def _replace_triangle_buffer(self, sizes_np: np.ndarray, offsets_np: np.ndarray):
        self.triangle_colliding_vertices_buffer_sizes = wp.array(sizes_np, dtype=wp.int32, device=self.device)
        self.triangle_colliding_vertices_offsets = wp.array(offsets_np, dtype=wp.int32, device=self.device)
        self.triangle_colliding_vertices = wp.full(
            shape=(int(offsets_np[-1]),), value=-1, dtype=wp.int32, device=self.device
        )
        if not hasattr(self, "triangle_colliding_vertices_count") or self.triangle_colliding_vertices_count is None:
            self.triangle_colliding_vertices_count = wp.zeros(
                shape=(self.tri_count,), dtype=wp.int32, device=self.device
            )

    def _replace_edge_buffer(self, sizes_np: np.ndarray, offsets_np: np.ndarray):
        self.edge_colliding_edges_buffer_sizes = wp.array(sizes_np, dtype=wp.int32, device=self.device)
        self.edge_colliding_edges_offsets = wp.array(offsets_np, dtype=wp.int32, device=self.device)
        self.edge_colliding_edges = wp.full(
            shape=(2 * int(offsets_np[-1]),), value=-1, dtype=wp.int32, device=self.device
        )
        if not hasattr(self, "edge_colliding_edges_count"):
            self.edge_colliding_edges_count = wp.zeros(
                shape=(self.edge_count,), dtype=wp.int32, device=self.device
            )
            self.edge_colliding_edges_min_dist = wp.empty(
                shape=(self.edge_count,), dtype=float, device=self.device
            )

    def _pack_collision_info(self):
        info = TriMeshCollisionInfo()
        info.vertex_colliding_triangles = self.vertex_colliding_triangles
        info.vertex_colliding_triangles_offsets = self.vertex_colliding_triangles_offsets
        info.vertex_colliding_triangles_buffer_sizes = self.vertex_colliding_triangles_buffer_sizes
        info.vertex_colliding_triangles_count = self.vertex_colliding_triangles_count
        info.vertex_colliding_triangles_min_dist = self.vertex_colliding_triangles_min_dist

        if self.record_triangle_contacting_vertices:
            info.triangle_colliding_vertices = self.triangle_colliding_vertices
            info.triangle_colliding_vertices_offsets = self.triangle_colliding_vertices_offsets
            info.triangle_colliding_vertices_buffer_sizes = self.triangle_colliding_vertices_buffer_sizes
            info.triangle_colliding_vertices_count = self.triangle_colliding_vertices_count
        info.triangle_colliding_vertices_min_dist = self.triangle_colliding_vertices_min_dist

        info.edge_colliding_edges = self.edge_colliding_edges
        info.edge_colliding_edges_offsets = self.edge_colliding_edges_offsets
        info.edge_colliding_edges_buffer_sizes = self.edge_colliding_edges_buffer_sizes
        info.edge_colliding_edges_count = self.edge_colliding_edges_count
        info.edge_colliding_edges_min_dist = self.edge_colliding_edges_min_dist
        return info

    def _refresh_collision_info(self):
        # Buffer replacement changes array handles stored in the Warp struct.
        self.collision_info = self._pack_collision_info()

    # ------------------------------------------------------------------
    # positions / BVH
    # ------------------------------------------------------------------

    def _coerce_positions(self, positions):
        if positions is None:
            return self.vertex_positions
        if isinstance(positions, wp.array):
            if positions.device != self.device:
                raise ValueError(f"positions device {positions.device} != detector device {self.device}")
            if len(positions) != self.particle_count:
                raise ValueError("positions length mismatch")
            return positions
        arr = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        if arr.shape[0] != self.particle_count:
            raise ValueError("positions length mismatch")
        return wp.array(arr, dtype=wp.vec3, device=self.device)

    def refit_triangles(self):
        if self.tri_count:
            wp.launch(
                kernel=compute_tri_aabbs,
                dim=self.tri_count,
                inputs=[self.vertex_positions, self.tri_indices, self.lower_bounds_tris, self.upper_bounds_tris],
                device=self.device,
            )
            self.bvh_tris.refit()

    def refit_edges(self):
        if self.edge_count:
            wp.launch(
                kernel=compute_edge_aabbs,
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

    # ------------------------------------------------------------------
    # VT detection + optional host-side resize/retry
    # ------------------------------------------------------------------

    def _run_vertex_triangle_pass(self, max_query_radius, min_query_radius, ref_pos, *, clear_flags: bool):
        self.vertex_colliding_triangles.fill_(-1)
        if clear_flags:
            self.resize_flags.zero_()

        if self.record_triangle_contacting_vertices:
            self.triangle_colliding_vertices.fill_(-1)
            wp.launch(
                kernel=init_triangle_collision_data_kernel,
                dim=self.tri_count,
                inputs=[float(max_query_radius)],
                outputs=[
                    self.triangle_colliding_vertices_count,
                    self.triangle_colliding_vertices_min_dist,
                    self.resize_flags,
                ],
                device=self.device,
            )
        else:
            self.triangle_colliding_vertices_min_dist.fill_(float(max_query_radius))

        wp.launch(
            kernel=vertex_triangle_collision_detection_kernel,
            dim=self.particle_count,
            inputs=[
                float(max_query_radius),
                float(min_query_radius),
                self.bvh_tris.id,
                self.vertex_positions,
                self.tri_indices,
                self.vertex_colliding_triangles_offsets,
                self.vertex_colliding_triangles_buffer_sizes,
                self.triangle_colliding_vertices_offsets,
                self.triangle_colliding_vertices_buffer_sizes,
                self.vertex_triangle_filtering_list,
                self.vertex_triangle_filtering_list_offsets,
                ref_pos,
            ],
            outputs=[
                self.vertex_colliding_triangles,
                self.vertex_colliding_triangles_count,
                self.vertex_colliding_triangles_min_dist,
                self.triangle_colliding_vertices,
                self.triangle_colliding_vertices_count,
                self.triangle_colliding_vertices_min_dist,
                self.resize_flags,
            ],
            device=self.device,
            block_dim=self.collision_detection_block_size,
        )

    def _grow_vt_buffers_if_needed(self, flags: np.ndarray) -> bool:
        grew = False
        if int(flags[0]):
            sizes, offsets = _grow_layout_from_counts(
                self.vertex_colliding_triangles_count.numpy(),
                self.vertex_colliding_triangles_buffer_sizes.numpy(),
                self.vertex_collision_buffer_max_alloc,
                name="vertex->triangle collision buffer",
            )
            self._replace_vertex_buffer(sizes, offsets)
            grew = True

        if self.record_triangle_contacting_vertices and int(flags[1]):
            sizes, offsets = _grow_layout_from_counts(
                self.triangle_colliding_vertices_count.numpy(),
                self.triangle_colliding_vertices_buffer_sizes.numpy(),
                self.triangle_collision_buffer_max_alloc,
                name="triangle->vertex collision buffer",
            )
            self._replace_triangle_buffer(sizes, offsets)
            grew = True

        if grew:
            self._refresh_collision_info()
        return grew

    def vertex_triangle_collision_detection(
        self,
        max_query_radius: float,
        min_query_radius: float = 0.0,
        min_distance_filtering_ref_pos=None,
        *,
        auto_resize: bool | None = None,
        _clear_resize_flags: bool = True,
    ):
        ref_pos = self._coerce_positions(min_distance_filtering_ref_pos)
        do_resize = self.auto_resize if auto_resize is None else bool(auto_resize)

        # Usually one pass. More iterations only occur after an overflow allocation.
        for attempt in range(8):
            self._run_vertex_triangle_pass(
                max_query_radius,
                min_query_radius,
                ref_pos,
                clear_flags=_clear_resize_flags or attempt > 0,
            )
            if not do_resize:
                return

            # Host read synchronizes CUDA; this is intentionally opt-in.
            flags = self.resize_flags.numpy()
            if not int(flags[0]) and not int(flags[1]):
                return
            if not self._grow_vt_buffers_if_needed(flags):
                return

        raise RuntimeError("VT auto-resize did not converge after 8 retries")

    # ------------------------------------------------------------------
    # EE detection + optional host-side resize/retry
    # ------------------------------------------------------------------

    def _run_edge_edge_pass(self, max_query_radius, min_query_radius, ref_pos, *, clear_flags: bool):
        self.edge_colliding_edges.fill_(-1)
        if clear_flags:
            self.resize_flags.zero_()

        wp.launch(
            kernel=edge_edge_collision_detection_kernel,
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
            ],
            outputs=[
                self.edge_colliding_edges,
                self.edge_colliding_edges_count,
                self.edge_colliding_edges_min_dist,
                self.resize_flags,
            ],
            device=self.device,
            block_dim=self.collision_detection_block_size,
        )

    def _grow_edge_buffer_if_needed(self, flags: np.ndarray) -> bool:
        if not int(flags[2]):
            return False
        sizes, offsets = _grow_layout_from_counts(
            self.edge_colliding_edges_count.numpy(),
            self.edge_colliding_edges_buffer_sizes.numpy(),
            self.edge_collision_buffer_max_alloc,
            name="edge->edge collision buffer",
        )
        self._replace_edge_buffer(sizes, offsets)
        self._refresh_collision_info()
        return True

    def edge_edge_collision_detection(
        self,
        max_query_radius: float,
        min_query_radius: float = 0.0,
        min_distance_filtering_ref_pos=None,
        *,
        auto_resize: bool | None = None,
        _clear_resize_flags: bool = True,
    ):
        ref_pos = self._coerce_positions(min_distance_filtering_ref_pos)
        do_resize = self.auto_resize if auto_resize is None else bool(auto_resize)

        for attempt in range(8):
            self._run_edge_edge_pass(
                max_query_radius,
                min_query_radius,
                ref_pos,
                clear_flags=_clear_resize_flags or attempt > 0,
            )
            if not do_resize:
                return

            flags = self.resize_flags.numpy()
            if not int(flags[2]):
                return
            if not self._grow_edge_buffer_if_needed(flags):
                return

        raise RuntimeError("EE auto-resize did not converge after 8 retries")

    # ------------------------------------------------------------------
    # full pass
    # ------------------------------------------------------------------

    def detect(
        self,
        q=None,
        *,
        margin: float,
        rest_q=None,
        rest_exclusion_radius: float = 0.0,
        auto_resize: bool | None = None,
    ) -> TriMeshCollisionInfo:
        """Refit BVHs, run VT, then run EE, matching the SolverVBD call order."""
        self.refit(q)
        ref = self.vertex_positions if rest_q is None else self._coerce_positions(rest_q)
        do_resize = self.auto_resize if auto_resize is None else bool(auto_resize)

        # One reset for the whole collision pass. If record_triangle_contacting_vertices=True,
        # the triangle init kernel also resets the same flags before VT, as Newton does.
        self.resize_flags.zero_()
        self.vertex_triangle_collision_detection(
            margin,
            min_query_radius=rest_exclusion_radius,
            min_distance_filtering_ref_pos=ref,
            auto_resize=do_resize,
            _clear_resize_flags=False,
        )

        # In fixed-buffer mode we preserve any VT overflow bits while EE adds bit 2.
        # In auto-resize mode VT has already converged to zero overflow.
        self.edge_edge_collision_detection(
            margin,
            min_query_radius=rest_exclusion_radius,
            min_distance_filtering_ref_pos=ref,
            auto_resize=do_resize,
            _clear_resize_flags=False,
        )
        return self.collision_info

    def overflow_flags(self) -> np.ndarray:
        """Return [VT, triangle-reverse, EE, tri-tri] overflow flags (host sync on CUDA)."""
        return self.resize_flags.numpy().copy()
