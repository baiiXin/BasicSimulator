
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp
import ipctk

from standalone_contact.collision_detector import TriMeshCollisionDetector
from standalone_contact.penetration_free_contact import PenetrationFreeContact


def unique_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int32)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    return np.asfortranarray(np.unique(edges, axis=0), dtype=np.int32)


def ipc_has_intersections(vertices, edges, faces) -> bool:
    V = np.asfortranarray(vertices, dtype=np.float64)
    E = np.asfortranarray(edges, dtype=np.int32)
    F = np.asfortranarray(faces, dtype=np.int32)
    mesh = ipctk.CollisionMesh(V, E, F)
    return bool(ipctk.has_intersections(mesh, V))


def load_combined_frame(root: Path, frame: int):
    vj = np.load(root / "jacket" / "vertices.npy", mmap_mode="r")
    vt = np.load(root / "tee_sleeveless" / "vertices.npy", mmap_mode="r")

    q = np.ascontiguousarray(
        np.vstack(
            [
                np.asarray(vj[frame], dtype=np.float64),
                np.asarray(vt[frame], dtype=np.float64),
            ]
        ),
        dtype=np.float64,
    )
    return q


def load_combined_faces(raw_root: Path):
    vj = np.load(raw_root / "jacket" / "vertices.npy", mmap_mode="r")
    fj = np.asarray(
        np.load(raw_root / "jacket" / "faces.npy"),
        dtype=np.int32,
    )
    ft = np.asarray(
        np.load(raw_root / "tee_sleeveless" / "faces.npy"),
        dtype=np.int32,
    )

    nv_jacket = int(vj.shape[1])

    faces = np.ascontiguousarray(
        np.vstack(
            [
                fj,
                ft + nv_jacket,
            ]
        ),
        dtype=np.int32,
    )

    return faces


def min_active_distance(min_dist_array, counts):
    """Minimum min-distance among rows that actually contain candidates."""
    d = np.asarray(min_dist_array.numpy(), dtype=np.float64)
    c = np.asarray(counts.numpy(), dtype=np.int32)

    active = c > 0
    if not np.any(active):
        return float("nan")

    return float(np.min(d[active]))


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Diagnose frame 63 micro-iterations starting from processed frame 62."
        )
    )

    ap.add_argument(
        "raw_root",
        type=Path,
        help="Raw sequence root, e.g. intesection_test",
    )
    ap.add_argument(
        "processed_root",
        type=Path,
        help="Processed sequence root containing the accepted frame 62",
    )

    ap.add_argument("--source-frame", type=int, default=62)
    ap.add_argument("--target-frame", type=int, default=63)
    ap.add_argument("--micro-steps", type=int, default=15)

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.85)
    ap.add_argument("--topology-ring", type=int, default=2)

    ap.add_argument("--vt-buffer", type=int, default=64)
    ap.add_argument("--ee-buffer", type=int, default=64)

    args = ap.parse_args()

    raw_root = args.raw_root.resolve()
    processed_root = args.processed_root.resolve()

    faces = load_combined_faces(raw_root)
    edges = unique_edges_from_faces(faces)

    # IMPORTANT:
    # start from accepted processed frame 62, not raw frame 62.
    q_start64 = load_combined_frame(
        processed_root,
        args.source_frame,
    )

    # target is raw frame 63.
    q_target64 = load_combined_frame(
        raw_root,
        args.target_frame,
    )

    print("=== Initial state ===")
    print("source processed frame:", args.source_frame)
    print("target raw frame:      ", args.target_frame)
    print("source intersects:     ", ipc_has_intersections(q_start64, edges, faces))
    print("target intersects:     ", ipc_has_intersections(q_target64, edges, faces))
    print()

    q_start32 = q_start64.astype(np.float32)
    q_target32 = q_target64.astype(np.float32)

    initial_dx64 = q_target64 - q_start64
    initial_l2 = float(np.linalg.norm(initial_dx64))

    if initial_l2 == 0.0:
        raise RuntimeError("Frame 62 already equals raw frame 63.")

    device = wp.get_device(args.device)

    q_current = wp.array(
        q_start32,
        dtype=wp.vec3,
        device=device,
    )

    detector = TriMeshCollisionDetector(
        q_current,
        faces,
        device=device,
        vertex_collision_buffer_pre_alloc=args.vt_buffer,
        vertex_collision_buffer_max_alloc=256,
        edge_collision_buffer_pre_alloc=args.ee_buffer,
        edge_collision_buffer_max_alloc=256,
        topological_contact_filter_threshold=args.topology_ring,
        auto_resize=True,
    )

    contact = PenetrationFreeContact(
        detector,
        particle_count=len(q_start32),
        edge_count=detector.edge_count,
        device=device,
        contact_margin=args.margin,
        conservative_bound_relaxation=args.gamma,
        rest_q=None,
        rest_exclusion_radius=0.0,
    )

    print(
        "micro  residual_ratio     trunc       min_t"
        "       min_VT_dist       min_EE_dist"
        "       max_dx_in      max_dx_out  intersect"
    )

    for micro in range(1, args.micro_steps + 1):
        # ------------------------------------------------------------------
        # Collision detection at current safe snapshot.
        # ------------------------------------------------------------------
        contact.begin(q_current)
        wp.synchronize_device(device)

        vt_count = detector.vertex_colliding_triangles_count
        ee_count = detector.edge_colliding_edges_count

        min_vt = min_active_distance(
            detector.vertex_colliding_triangles_min_dist,
            vt_count,
        )
        min_ee = min_active_distance(
            detector.edge_colliding_edges_min_dist,
            ee_count,
        )

        # ------------------------------------------------------------------
        # Remaining displacement to the original raw frame 63 target.
        # ------------------------------------------------------------------
        q_cur32 = np.asarray(q_current.numpy(), dtype=np.float32)
        dx32 = q_target32 - q_cur32

        max_dx_in = float(
            np.linalg.norm(dx32, axis=1).max(initial=0.0)
        )

        dx_wp = wp.array(
            dx32,
            dtype=wp.vec3,
            device=device,
        )

        # ------------------------------------------------------------------
        # DAT.
        # ------------------------------------------------------------------
        dx_safe, q_next, ts = contact.truncate(dx_wp)
        wp.synchronize_device(device)

        q_next32 = np.asarray(q_next.numpy(), dtype=np.float32)
        q_next64 = q_next32.astype(np.float64)

        dx_safe_np = np.asarray(dx_safe.numpy(), dtype=np.float32)
        ts_np = np.asarray(ts.numpy(), dtype=np.float32)

        truncated_vertices = int(
            np.count_nonzero(ts_np < 1.0 - 1.0e-6)
        )
        min_t = float(ts_np.min(initial=1.0))

        max_dx_out = float(
            np.linalg.norm(dx_safe_np, axis=1).max(initial=0.0)
        )

        residual = q_target64 - q_next64
        residual_ratio = float(np.linalg.norm(residual) / initial_l2)

        intersects = ipc_has_intersections(
            q_next64,
            edges,
            faces,
        )

        print(
            f"{micro:5d}"
            f"  {residual_ratio:14.8e}"
            f"  {truncated_vertices:8d}"
            f"  {min_t:11.4e}"
            f"  {min_vt:16.8e}"
            f"  {min_ee:16.8e}"
            f"  {max_dx_in:13.6e}"
            f"  {max_dx_out:13.6e}"
            f"  {str(intersects):>9s}"
        )

        if intersects:
            print()
            print(
                f"FIRST INTERSECTION: micro={micro}"
            )

            np.save(
                f"frame63_micro{micro:02d}_q_before.npy",
                q_cur32,
            )
            np.save(
                f"frame63_micro{micro:02d}_q_after.npy",
                q_next64,
            )
            np.save(
                f"frame63_micro{micro:02d}_t.npy",
                ts_np,
            )

            break

        q_current = q_next


if __name__ == "__main__":
    main()
