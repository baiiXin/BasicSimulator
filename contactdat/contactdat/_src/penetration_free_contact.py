# SPDX-FileCopyrightText: Copyright (c) 2026 Standalone Contact contributors.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import warp as wp

from .penetration_free import PenetrationFreeTruncator


class PenetrationFreeContact:
    """High-level collision snapshot + collision detection + DAT wrapper."""

    def __init__(
        self,
        detector,
        *,
        particle_count: int,
        edge_count: int,
        device=None,
        contact_margin: float = 0.2,
        conservative_bound_relaxation: float = 0.85,
        rest_q=None,
        rest_exclusion_radius: float = 0.0,
    ):
        self.detector = detector
        self.device = wp.get_device(device)

        self.particle_count = int(particle_count)
        self.edge_count = int(edge_count)

        self.contact_margin = float(contact_margin)
        self.gamma = float(conservative_bound_relaxation)
        self.rest_q = rest_q
        self.rest_exclusion_radius = float(rest_exclusion_radius)

        self.q_snapshot = wp.zeros(
            self.particle_count, dtype=wp.vec3, device=self.device
        )
        self.displacement = wp.zeros(
            self.particle_count, dtype=wp.vec3, device=self.device
        )
        self.displacement_safe = wp.zeros(
            self.particle_count, dtype=wp.vec3, device=self.device
        )
        self.q_safe = wp.zeros(
            self.particle_count, dtype=wp.vec3, device=self.device
        )

        parallel_eps = float(
            getattr(detector, "edge_edge_parallel_epsilon", 1.0e-5)
        )

        self.truncator = PenetrationFreeTruncator(
            particle_count=self.particle_count,
            edge_count=self.edge_count,
            collision_info=self.detector.collision_info,
            device=self.device,
            parallel_eps=parallel_eps,
            conservative_bound_relaxation=self.gamma,
            contact_margin=self.contact_margin,
        )

        self.epoch = -1
        self.has_snapshot = False

    def begin(self, q):
        """Start a new collision epoch."""
        self.q_snapshot.assign(q)
        self.displacement.zero_()

        self.detector.refit(self.q_snapshot)

        self.detector.vertex_triangle_collision_detection(
            self.contact_margin,
            min_query_radius=self.rest_exclusion_radius,
            min_distance_filtering_ref_pos=self.rest_q,
        )
        self.detector.edge_edge_collision_detection(
            self.contact_margin,
            min_query_radius=self.rest_exclusion_radius,
            min_distance_filtering_ref_pos=self.rest_q,
        )

        # Refresh after possible Group-1 collision-buffer reallocation.
        self.truncator.set_collision_info(self.detector.collision_info)

        self.epoch += 1
        self.has_snapshot = True
        return self.q_snapshot

    def truncate_into(
        self,
        displacement,
        *,
        q_out=None,
        displacement_out=None,
    ):
        """Apply DAT to displacement relative to the latest q_snapshot."""
        if not self.has_snapshot:
            raise RuntimeError(
                "PenetrationFreeContact.begin(q) must be called before truncate()."
            )

        self.displacement.assign(displacement)

        if q_out is None:
            q_out = self.q_safe
        if displacement_out is None:
            displacement_out = self.displacement_safe

        self.truncator.truncate_into(
            snapshot_pos=self.q_snapshot,
            displacement_in=self.displacement,
            tri_indices=self.detector.tri_indices,
            edge_indices=self.detector.edge_indices,
            displacement_out=displacement_out,
            pos_out=q_out,
            enable_self_contact=True,
        )

        return displacement_out, q_out, self.truncator.truncation_ts

    def truncate(self, displacement):
        return self.truncate_into(
            displacement,
            q_out=self.q_safe,
            displacement_out=self.displacement_safe,
        )

    def redetect(self, q):
        return self.begin(q)
