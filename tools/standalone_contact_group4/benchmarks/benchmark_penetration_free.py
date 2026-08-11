
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

TOOLS = Path(__file__).resolve().parents[1]
TESTS = TOOLS / "tests"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TESTS))

from group4_stress_utils import (
    make_center_punch_displacement,
    make_two_layer_grid,
)
from standalone_contact.collision_detector import TriMeshCollisionDetector
from standalone_contact.penetration_free_contact import PenetrationFreeContact


def sync(device):
    wp.synchronize_device(device)


def timed_ms(device, fn, repeats=20, warmup=5):
    for _ in range(warmup):
        fn()
    sync(device)

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)

    a = np.asarray(samples)
    return float(np.median(a)), float(np.percentile(a, 95))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=os.environ.get("CONTACT_TEST_DEVICE", "cuda:0"))
    parser.add_argument("--n", type=int, nargs="+", default=[16, 32, 48, 64])
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    device = wp.get_device(args.device)

    print(
        "n,V,T,E,VT_candidates,EE_candidates,truncated_vertices,min_t,"
        "refit_ms,vt_ms,ee_ms,dat_ms,total_ms"
    )

    for n in args.n:
        mesh = make_two_layer_grid(n=n, spacing=0.02, gap=0.008)
        q0 = wp.array(mesh.vertices, dtype=wp.vec3, device=device)
        q_rest = wp.array(mesh.vertices, dtype=wp.vec3, device=device)

        detector = TriMeshCollisionDetector(
            q0,
            mesh.faces,
            device=device,
            vertex_collision_buffer_pre_alloc=64,
            edge_collision_buffer_pre_alloc=64,
            topological_contact_filter_threshold=2,
        )

        contact = PenetrationFreeContact(
            detector,
            particle_count=len(mesh.vertices),
            edge_count=detector.edge_count,
            device=device,
            contact_margin=0.012,
            conservative_bound_relaxation=0.85,
            rest_q=q_rest,
            rest_exclusion_radius=0.0,
        )

        contact.begin(q0)

        dx_np = make_center_punch_displacement(
            mesh,
            n=n,
            downward_distance=0.02,
            center_half_width=max(1, n // 8),
        )
        dx = wp.array(dx_np, dtype=wp.vec3, device=device)

        # Warm one DAT call so Warp JIT compilation is outside timings.
        contact.truncate(dx)
        sync(device)

        def run_refit():
            detector.refit(q0)

        def run_vt():
            detector.vertex_triangle_collision_detection(
                contact.contact_margin,
                min_query_radius=0.0,
                min_distance_filtering_ref_pos=q_rest,
            )

        def run_ee():
            detector.edge_edge_collision_detection(
                contact.contact_margin,
                min_query_radius=0.0,
                min_distance_filtering_ref_pos=q_rest,
            )

        def run_dat():
            # Collision candidates remain the ones from the latest begin().
            contact.truncate(dx)

        def run_total():
            contact.begin(q0)
            contact.truncate(dx)

        refit_med, _ = timed_ms(device, run_refit, args.repeats)
        vt_med, _ = timed_ms(device, run_vt, args.repeats)
        ee_med, _ = timed_ms(device, run_ee, args.repeats)
        dat_med, _ = timed_ms(device, run_dat, args.repeats)
        total_med, _ = timed_ms(device, run_total, args.repeats)

        vt_counts = detector.vertex_colliding_triangles_count.numpy()
        ee_counts = detector.edge_colliding_edges_count.numpy()
        ts = contact.truncator.truncation_ts.numpy()

        print(
            f"{n},"
            f"{len(mesh.vertices)},"
            f"{len(mesh.faces)},"
            f"{detector.edge_count},"
            f"{int(vt_counts.sum())},"
            f"{int(ee_counts.sum())},"
            f"{int(np.count_nonzero(ts < 1.0 - 1e-6))},"
            f"{float(ts.min()):.7g},"
            f"{refit_med:.4f},"
            f"{vt_med:.4f},"
            f"{ee_med:.4f},"
            f"{dat_med:.4f},"
            f"{total_med:.4f}"
        )


if __name__ == "__main__":
    main()
