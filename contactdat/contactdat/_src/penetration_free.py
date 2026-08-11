# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Standalone adaptation for Warp use without importing Newton.
# Modifications Copyright (c) 2026 Standalone Contact contributors.


from __future__ import annotations

import warp as wp

from .collision_detector import TriMeshCollisionInfo
from .dat_kernels import (
    NUM_THREADS_PER_COLLISION_PRIMITIVE,
    apply_planar_truncation_parallel_by_collision,
    apply_truncation_ts,
)


class PenetrationFreeTruncator:
    """Reusable standalone DAT launcher."""

    def __init__(
        self,
        *,
        particle_count: int,
        edge_count: int,
        collision_info: TriMeshCollisionInfo,
        device=None,
        parallel_eps: float = 1.0e-5,
        conservative_bound_relaxation: float = 0.85,
        contact_margin: float = 0.2,
    ):
        self.device = wp.get_device(device)
        self.particle_count = int(particle_count)
        self.edge_count = int(edge_count)

        self.parallel_eps = float(parallel_eps)
        self.gamma = float(conservative_bound_relaxation)
        self.contact_margin = float(contact_margin)

        self.truncation_ts = wp.ones(
            self.particle_count,
            dtype=float,
            device=self.device,
        )

        self.launch_size = max(
            self.particle_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
            self.edge_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
        )

        self.collision_info_array = None
        self.set_collision_info(collision_info)

    def set_collision_info(self, collision_info: TriMeshCollisionInfo) -> None:
        """Refresh after collision-buffer reallocation/resizing."""
        self.collision_info_array = wp.array(
            [collision_info],
            dtype=TriMeshCollisionInfo,
            device=self.device,
        )

    def compute_truncation_ts(
        self,
        *,
        snapshot_pos,
        displacement,
        tri_indices,
        edge_indices,
    ):
        self.truncation_ts.fill_(1.0)

        if self.launch_size > 0:
            wp.launch(
                kernel=apply_planar_truncation_parallel_by_collision,
                dim=self.launch_size,
                inputs=[
                    snapshot_pos,
                    displacement,
                    tri_indices,
                    edge_indices,
                    self.collision_info_array,
                    self.parallel_eps,
                    self.gamma,
                    self.truncation_ts,
                ],
                device=self.device,
            )

        return self.truncation_ts

    def truncate_into(
        self,
        *,
        snapshot_pos,
        displacement_in,
        tri_indices,
        edge_indices,
        displacement_out,
        pos_out=None,
        enable_self_contact: bool = True,
    ):
        self.truncation_ts.fill_(1.0)

        if enable_self_contact and self.launch_size > 0:
            wp.launch(
                kernel=apply_planar_truncation_parallel_by_collision,
                dim=self.launch_size,
                inputs=[
                    snapshot_pos,
                    displacement_in,
                    tri_indices,
                    edge_indices,
                    self.collision_info_array,
                    self.parallel_eps,
                    self.gamma,
                    self.truncation_ts,
                ],
                device=self.device,
            )
            max_displacement = self.contact_margin * self.gamma * 0.5
        else:
            max_displacement = wp.inf

        wp.launch(
            kernel=apply_truncation_ts,
            dim=self.particle_count,
            inputs=[
                snapshot_pos,
                displacement_in,
                self.truncation_ts,
                max_displacement,
                displacement_out,
                pos_out,
            ],
            device=self.device,
        )

        return displacement_out, pos_out, self.truncation_ts

    def truncate(
        self,
        *,
        snapshot_pos,
        displacement_in,
        tri_indices,
        edge_indices,
        enable_self_contact: bool = True,
        write_positions: bool = True,
    ):
        displacement_out = wp.empty_like(displacement_in)
        pos_out = wp.empty_like(snapshot_pos) if write_positions else None

        return self.truncate_into(
            snapshot_pos=snapshot_pos,
            displacement_in=displacement_in,
            tri_indices=tri_indices,
            edge_indices=edge_indices,
            displacement_out=displacement_out,
            pos_out=pos_out,
            enable_self_contact=enable_self_contact,
        )
