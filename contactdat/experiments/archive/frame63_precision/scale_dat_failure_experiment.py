
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import warp as wp
import ipctk

from contactdat import PenetrationFreeContact, TriMeshCollisionDetector


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


def load_combined_frame(root: Path, frame: int) -> np.ndarray:
    vj = np.load(root / "jacket" / "vertices.npy", mmap_mode="r")
    vt = np.load(root / "tee_sleeveless" / "vertices.npy", mmap_mode="r")
    return np.ascontiguousarray(
        np.vstack(
            [
                np.asarray(vj[frame], dtype=np.float64),
                np.asarray(vt[frame], dtype=np.float64),
            ]
        ),
        dtype=np.float64,
    )


def load_combined_faces(raw_root: Path) -> np.ndarray:
    vj = np.load(raw_root / "jacket" / "vertices.npy", mmap_mode="r")
    fj = np.asarray(np.load(raw_root / "jacket" / "faces.npy"), dtype=np.int32)
    ft = np.asarray(
        np.load(raw_root / "tee_sleeveless" / "faces.npy"),
        dtype=np.int32,
    )
    nv_jacket = int(vj.shape[1])
    return np.ascontiguousarray(
        np.vstack([fj, ft + nv_jacket]),
        dtype=np.int32,
    )


def min_active_distance(min_dist_array, counts) -> float:
    d = np.asarray(min_dist_array.numpy(), dtype=np.float64)
    c = np.asarray(counts.numpy(), dtype=np.int32)
    active = c > 0
    if not np.any(active):
        return float("nan")
    return float(np.min(d[active]))


def run_scale(
    q_source_base64,
    q_target_base64,
    faces,
    edges,
    *,
    scale: float,
    base_margin: float,
    gamma: float,
    topology_ring: int,
    micro_steps: int,
    vt_buffer: int,
    ee_buffer: int,
    device,
):
    q_source64 = q_source_base64 * scale
    q_target64 = q_target_base64 * scale
    margin = base_margin * scale

    if ipc_has_intersections(q_source64, edges, faces):
        raise RuntimeError(f"scale={scale}: source already intersects")
    if ipc_has_intersections(q_target64, edges, faces):
        raise RuntimeError(f"scale={scale}: target already intersects")

    q_source32 = q_source64.astype(np.float32)
    q_target32 = q_target64.astype(np.float32)

    initial_l2 = float(np.linalg.norm(q_target64 - q_source64))

    q_current = wp.array(q_source32, dtype=wp.vec3, device=device)

    detector = TriMeshCollisionDetector(
        q_current,
        faces,
        device=device,
        vertex_collision_buffer_pre_alloc=vt_buffer,
        vertex_collision_buffer_max_alloc=256,
        edge_collision_buffer_pre_alloc=ee_buffer,
        edge_collision_buffer_max_alloc=256,
        topological_contact_filter_threshold=topology_ring,
        auto_resize=True,
    )

    contact = PenetrationFreeContact(
        detector,
        particle_count=len(q_source32),
        edge_count=detector.edge_count,
        device=device,
        contact_margin=margin,
        conservative_bound_relaxation=gamma,
        rest_q=None,
        rest_exclusion_radius=0.0,
    )

    rows = []
    first_intersection = None

    print()
    print(f"=== scale = {scale:g}  margin = {margin:g} ===")
    print(
        "micro  residual_ratio     trunc       min_t"
        "       min_VT_dist       min_EE_dist"
        "     VT/scale        EE/scale   intersect"
    )

    for micro in range(1, micro_steps + 1):
        contact.begin(q_current)
        wp.synchronize_device(device)

        min_vt = min_active_distance(
            detector.vertex_colliding_triangles_min_dist,
            detector.vertex_colliding_triangles_count,
        )
        min_ee = min_active_distance(
            detector.edge_colliding_edges_min_dist,
            detector.edge_colliding_edges_count,
        )

        q_cur32 = np.asarray(q_current.numpy(), dtype=np.float32)
        dx32 = q_target32 - q_cur32
        dx_wp = wp.array(dx32, dtype=wp.vec3, device=device)

        dx_safe, q_next, ts = contact.truncate(dx_wp)
        wp.synchronize_device(device)

        q_next32 = np.asarray(q_next.numpy(), dtype=np.float32)
        q_next64 = q_next32.astype(np.float64)
        ts_np = np.asarray(ts.numpy(), dtype=np.float32)

        residual_ratio = float(
            np.linalg.norm(q_target64 - q_next64) / initial_l2
        )

        truncated = int(np.count_nonzero(ts_np < 1.0 - 1.0e-6))
        min_t = float(ts_np.min(initial=1.0))

        intersects = ipc_has_intersections(q_next64, edges, faces)

        row = {
            "scale": scale,
            "micro": micro,
            "residual_ratio": residual_ratio,
            "truncated_vertices": truncated,
            "min_t": min_t,
            "min_vt_distance": min_vt,
            "min_ee_distance": min_ee,
            "normalized_min_vt_distance": min_vt / scale,
            "normalized_min_ee_distance": min_ee / scale,
            "intersects": intersects,
        }
        rows.append(row)

        print(
            f"{micro:5d}"
            f"  {residual_ratio:14.8e}"
            f"  {truncated:8d}"
            f"  {min_t:11.4e}"
            f"  {min_vt:16.8e}"
            f"  {min_ee:16.8e}"
            f"  {min_vt/scale:12.4e}"
            f"  {min_ee/scale:12.4e}"
            f"  {str(intersects):>9s}"
        )

        if intersects:
            first_intersection = micro
            break

        q_current = q_next

    return rows, first_intersection


def main():
    ap = argparse.ArgumentParser(
        description="Scale-consistency experiment for the frame-63 DAT failure."
    )
    ap.add_argument("raw_root", type=Path)
    ap.add_argument("processed_root", type=Path)

    ap.add_argument("--source-frame", type=int, default=62)
    ap.add_argument("--target-frame", type=int, default=63)
    ap.add_argument("--micro-steps", type=int, default=15)

    ap.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[1.0, 100.0],
        help="Default: 1 100",
    )

    ap.add_argument("--base-margin", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.85)
    ap.add_argument("--topology-ring", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vt-buffer", type=int, default=64)
    ap.add_argument("--ee-buffer", type=int, default=64)

    ap.add_argument(
        "--output",
        type=Path,
        default=Path("frame63_scale_experiment.csv"),
    )

    args = ap.parse_args()

    raw_root = args.raw_root.resolve()
    processed_root = args.processed_root.resolve()

    faces = load_combined_faces(raw_root)
    edges = unique_edges_from_faces(faces)

    q_source = load_combined_frame(processed_root, args.source_frame)
    q_target = load_combined_frame(raw_root, args.target_frame)

    print("=== Base geometry ===")
    print("processed source frame:", args.source_frame)
    print("raw target frame:      ", args.target_frame)
    print("base source intersects:", ipc_has_intersections(q_source, edges, faces))
    print("base target intersects:", ipc_has_intersections(q_target, edges, faces))

    device = wp.get_device(args.device)

    all_rows = []
    summary = []

    for scale in args.scales:
        rows, first_hit = run_scale(
            q_source,
            q_target,
            faces,
            edges,
            scale=scale,
            base_margin=args.base_margin,
            gamma=args.gamma,
            topology_ring=args.topology_ring,
            micro_steps=args.micro_steps,
            vt_buffer=args.vt_buffer,
            ee_buffer=args.ee_buffer,
            device=device,
        )
        all_rows.extend(rows)
        summary.append((scale, first_hit))

    out = args.output.resolve()
    fieldnames = [
        "scale",
        "micro",
        "residual_ratio",
        "truncated_vertices",
        "min_t",
        "min_vt_distance",
        "min_ee_distance",
        "normalized_min_vt_distance",
        "normalized_min_ee_distance",
        "intersects",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print()
    print("=== SCALE SUMMARY ===")
    for scale, first_hit in summary:
        if first_hit is None:
            print(
                f"scale={scale:g}: no intersection within "
                f"{args.micro_steps} micro-steps"
            )
        else:
            print(
                f"scale={scale:g}: first intersection at micro={first_hit}"
            )

    print()
    print("Interpretation:")
    print(
        "If scale=1 fails but a geometrically similar larger scale survives "
        "longer or does not fail, the method is not scale invariant in this "
        "regime, consistent with absolute epsilons / float32 numerical limits."
    )
    print()
    print("CSV:", out)


if __name__ == "__main__":
    main()
