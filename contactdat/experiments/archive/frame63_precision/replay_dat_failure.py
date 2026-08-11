
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp
import ipctk

from contactdat import PenetrationFreeContact, TriMeshCollisionDetector

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FAILURE_DIR = REPO_ROOT / "tests" / "real" / "data"


def unique_edges_from_faces(faces):
    faces = np.asarray(faces, dtype=np.int32)
    e = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    e = np.sort(e, axis=1)
    return np.asfortranarray(np.unique(e, axis=0), dtype=np.int32)


def ipc_hit(V, E, F):
    V = np.asfortranarray(V, dtype=np.float64)
    E = np.asfortranarray(E, dtype=np.int32)
    F = np.asfortranarray(F, dtype=np.int32)
    mesh = ipctk.CollisionMesh(V, E, F)
    return bool(ipctk.has_intersections(mesh, V))


def load_faces(sample_root):
    fj = np.asarray(np.load(sample_root / "jacket" / "faces.npy"), dtype=np.int32)
    ft = np.asarray(np.load(sample_root / "tee_sleeveless" / "faces.npy"), dtype=np.int32)

    vj = np.load(sample_root / "jacket" / "vertices.npy", mmap_mode="r")
    nj = int(vj.shape[1])

    F = np.ascontiguousarray(np.vstack([fj, ft + nj]), dtype=np.int32)
    E = unique_edges_from_faces(F)
    return F, E


def run_one(q_before, q_target, F, E, ring, device, margin):
    q0 = wp.array(q_before.astype(np.float32), dtype=wp.vec3, device=device)

    detector = TriMeshCollisionDetector(
        q0,
        F,
        device=device,
        vertex_collision_buffer_pre_alloc=64,
        vertex_collision_buffer_max_alloc=256,
        edge_collision_buffer_pre_alloc=64,
        edge_collision_buffer_max_alloc=256,
        topological_contact_filter_threshold=ring,
        auto_resize=True,
    )

    contact = PenetrationFreeContact(
        detector,
        particle_count=len(q_before),
        edge_count=detector.edge_count,
        device=device,
        contact_margin=margin,
        conservative_bound_relaxation=0.85,
        rest_q=None,
        rest_exclusion_radius=0.0,
    )

    contact.begin(q0)

    vt_count = detector.vertex_colliding_triangles_count.numpy()
    ee_count = detector.edge_colliding_edges_count.numpy()
    vt_size = detector.vertex_colliding_triangles_buffer_sizes.numpy()
    ee_size = detector.edge_colliding_edges_buffer_sizes.numpy()

    dx = q_target.astype(np.float32) - q_before.astype(np.float32)
    dx_wp = wp.array(dx, dtype=wp.vec3, device=device)

    _, q_after, ts = contact.truncate(dx_wp)
    q_after_np = q_after.numpy().astype(np.float64)
    ts_np = ts.numpy()

    return {
        "ring": ring,
        "before_hit": ipc_hit(q_before, E, F),
        "after_hit": ipc_hit(q_after_np, E, F),
        "vt_max_count": int(vt_count.max(initial=0)),
        "ee_max_count": int(ee_count.max(initial=0)),
        "vt_max_capacity": int(vt_size.max(initial=0)),
        "ee_max_capacity": int(ee_size.max(initial=0)),
        "vt_total": int(vt_count.sum()),
        "ee_total": int(ee_count.sum()),
        "truncated_vertices": int(np.count_nonzero(ts_np < 1.0 - 1.0e-6)),
        "min_t": float(ts_np.min(initial=1.0)),
        "max_input_dx": float(np.linalg.norm(dx, axis=1).max(initial=0.0)),
        "max_output_dx": float(np.linalg.norm(q_after_np - q_before, axis=1).max(initial=0.0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_root", type=Path)
    ap.add_argument("--failure-dir", type=Path, default=DEFAULT_FAILURE_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--margin", type=float, default=0.05)
    args = ap.parse_args()

    d = args.failure_dir
    q_before = np.asarray(np.load(d / "fail_q_before.npy"), dtype=np.float64)
    q_after_saved = np.asarray(np.load(d / "fail_q_after.npy"), dtype=np.float64)
    q_target = np.asarray(np.load(d / "fail_q_target.npy"), dtype=np.float64)

    F, E = load_faces(args.sample_root)

    print("saved before intersects:", ipc_hit(q_before, E, F))
    print("saved after intersects: ", ipc_hit(q_after_saved, E, F))
    print("target intersects:      ", ipc_hit(q_target, E, F))
    print()

    for ring in (2, 1, 0):
        r = run_one(q_before, q_target, F, E, ring, args.device, args.margin)
        print(
            f"ring={ring} "
            f"before={r['before_hit']} after={r['after_hit']} "
            f"VTmax={r['vt_max_count']}/{r['vt_max_capacity']} "
            f"EEmax={r['ee_max_count']}/{r['ee_max_capacity']} "
            f"VTtotal={r['vt_total']} EEtotal={r['ee_total']} "
            f"trunc={r['truncated_vertices']} min_t={r['min_t']:.7g} "
            f"max_dx_in={r['max_input_dx']:.7g} "
            f"max_dx_out={r['max_output_dx']:.7g}"
        )


if __name__ == "__main__":
    main()
