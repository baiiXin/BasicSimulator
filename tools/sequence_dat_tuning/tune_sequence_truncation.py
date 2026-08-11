
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from standalone_contact.collision_detector import TriMeshCollisionDetector
from standalone_contact.penetration_free_contact import PenetrationFreeContact


def unique_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return np.asfortranarray(edges, dtype=np.int32)


def load_combined_clothes(sample_root: Path, garments: list[str], frame: int):
    if frame <= 0:
        raise ValueError("--frame must be >= 1 because tuning moves frame-1 -> frame.")

    vertices_prev = []
    vertices_target = []
    faces_all = []
    garment_meta = []

    vertex_offset = 0
    for garment_id, name in enumerate(garments, start=2):
        mesh_dir = sample_root / name
        vertices_path = mesh_dir / "vertices.npy"
        faces_path = mesh_dir / "faces.npy"

        if not vertices_path.exists():
            raise FileNotFoundError(vertices_path)
        if not faces_path.exists():
            raise FileNotFoundError(faces_path)

        seq = np.load(vertices_path, mmap_mode="r")
        faces = np.asarray(np.load(faces_path), dtype=np.int32)

        if seq.ndim != 3 or seq.shape[2] != 3:
            raise ValueError(f"{vertices_path}: expected [F,V,3], got {seq.shape}")
        if frame >= seq.shape[0]:
            raise IndexError(
                f"frame={frame} out of range for {name}: frame_count={seq.shape[0]}"
            )

        q_prev = np.asarray(seq[frame - 1], dtype=np.float64)
        q_target = np.asarray(seq[frame], dtype=np.float64)

        vertices_prev.append(q_prev)
        vertices_target.append(q_target)
        faces_all.append(faces + vertex_offset)

        garment_meta.append(
            {
                "name": name,
                "owner": garment_id,
                "vertex_offset": int(vertex_offset),
                "vertex_count": int(q_prev.shape[0]),
                "face_count": int(faces.shape[0]),
            }
        )
        vertex_offset += q_prev.shape[0]

    q_prev = np.ascontiguousarray(np.vstack(vertices_prev), dtype=np.float64)
    q_target = np.ascontiguousarray(np.vstack(vertices_target), dtype=np.float64)
    faces = np.ascontiguousarray(np.vstack(faces_all), dtype=np.int32)
    edges = unique_edges_from_faces(faces)

    return q_prev, q_target, faces, edges, garment_meta


def median_edge_length(vertices: np.ndarray, edges: np.ndarray) -> float:
    lengths = np.linalg.norm(
        vertices[edges[:, 1]] - vertices[edges[:, 0]],
        axis=1,
    )
    return float(np.median(lengths))


def ipc_has_intersections(vertices: np.ndarray, edges: np.ndarray, faces: np.ndarray) -> bool:
    import ipctk

    V = np.asfortranarray(vertices, dtype=np.float64)
    E = np.asfortranarray(edges, dtype=np.int32)
    F = np.asfortranarray(faces, dtype=np.int32)

    collision_mesh = ipctk.CollisionMesh(V, E, F)
    return bool(ipctk.has_intersections(collision_mesh, V))


def synchronize(device):
    wp.synchronize_device(device)


def timed_call_ms(device, fn):
    synchronize(device)
    t0 = time.perf_counter()
    result = fn()
    synchronize(device)
    return result, (time.perf_counter() - t0) * 1000.0


def get_detector_counts(detector):
    vt = int(np.asarray(detector.vertex_colliding_triangles_count.numpy()).sum())
    ee = int(np.asarray(detector.edge_colliding_edges_count.numpy()).sum())
    return vt, ee


def get_overflow_state(detector):
    if hasattr(detector, "overflow_flags"):
        try:
            flags = np.asarray(detector.overflow_flags()).astype(bool).reshape(-1)
            return bool(np.any(flags[:3])), flags.astype(int).tolist()
        except Exception:
            pass
    return None, None


def save_plot(rows, output_png: Path):
    k = np.array([r["k"] for r in rows], dtype=int)
    residual = np.array([r["residual_ratio_l2"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(k, residual, marker="o")
    ax.set_xlabel("DAT truncation / collision-detection count K")
    ax.set_ylabel("Remaining displacement ratio")
    ax.set_title("Residual displacement vs. DAT iterations")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Tune fixed DAT micro-iteration count K on one raw garment frame transition. "
            "Human is ignored; garments are merged as disconnected components."
        )
    )
    parser.add_argument("sample_root", type=Path)
    parser.add_argument(
        "--garments",
        nargs="+",
        default=["jacket", "tee_sleeveless"],
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=1,
        help="Target raw frame. The scan starts at raw frame-1.",
    )
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--contact-margin", type=float, default=None)
    parser.add_argument("--margin-edge-ratio", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.85)

    parser.add_argument("--vt-buffer", type=int, default=64)
    parser.add_argument("--ee-buffer", type=int, default=64)
    parser.add_argument("--topology-ring", type=int, default=2)

    parser.add_argument("--residual-tol", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-ipc-every-step", action="store_true")

    args = parser.parse_args()

    sample_root = args.sample_root.resolve()
    output_dir = (
        args.output.resolve()
        if args.output is not None
        else sample_root / "dat_tuning"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    q_start64, q_target64, faces, edges, garment_meta = load_combined_clothes(
        sample_root,
        args.garments,
        args.frame,
    )

    if ipc_has_intersections(q_start64, edges, faces):
        raise RuntimeError(
            f"Raw frame {args.frame - 1} is already intersecting for the merged clothes."
        )

    raw_target_intersects = ipc_has_intersections(q_target64, edges, faces)

    med_edge = median_edge_length(q_start64, edges)
    contact_margin = (
        float(args.contact_margin)
        if args.contact_margin is not None
        else med_edge * float(args.margin_edge_ratio)
    )
    if contact_margin <= 0:
        raise ValueError("contact margin must be positive")

    device = wp.get_device(args.device)

    q_start32 = np.asarray(q_start64, dtype=np.float32)
    q_target32 = np.asarray(q_target64, dtype=np.float32)

    q_current = wp.array(q_start32, dtype=wp.vec3, device=device)

    detector = TriMeshCollisionDetector(
        q_current,
        faces,
        device=device,
        vertex_collision_buffer_pre_alloc=args.vt_buffer,
        edge_collision_buffer_pre_alloc=args.ee_buffer,
        topological_contact_filter_threshold=args.topology_ring,
    )

    contact = PenetrationFreeContact(
        detector,
        particle_count=len(q_start32),
        edge_count=detector.edge_count,
        device=device,
        contact_margin=contact_margin,
        conservative_bound_relaxation=args.gamma,
        rest_q=None,
        rest_exclusion_radius=0.0,
    )

    initial_dx64 = q_target64 - q_start64
    initial_l2 = float(np.linalg.norm(initial_dx64))
    initial_max = float(np.linalg.norm(initial_dx64, axis=1).max(initial=0.0))
    if initial_l2 == 0.0:
        raise RuntimeError("Selected raw frame transition has zero displacement.")

    rows = [
        {
            "k": 0,
            "collision_detection_calls": 0,
            "dat_calls": 0,
            "residual_l2": initial_l2,
            "residual_ratio_l2": 1.0,
            "residual_max_vertex": initial_max,
            "truncated_vertices": 0,
            "min_t": 1.0,
            "vt_candidates": 0,
            "ee_candidates": 0,
            "detect_ms": 0.0,
            "dat_ms": 0.0,
            "iteration_ms": 0.0,
            "safe_intersection": False,
            "overflow": False,
            "overflow_flags": None,
        }
    ]

    accepted_positions = [q_start64.copy()]

    for k in range(1, args.k_max + 1):
        _, detect_ms = timed_call_ms(device, lambda: contact.begin(q_current))
        vt_count, ee_count = get_detector_counts(detector)
        overflow, overflow_flags = get_overflow_state(detector)

        q_current_np32 = np.asarray(q_current.numpy(), dtype=np.float32)
        dx_np32 = q_target32 - q_current_np32
        dx_wp = wp.array(dx_np32, dtype=wp.vec3, device=device)

        (dx_safe, q_next, ts), dat_ms = timed_call_ms(
            device,
            lambda: contact.truncate(dx_wp),
        )

        q_next32 = np.asarray(q_next.numpy(), dtype=np.float32)
        q_next64 = np.asarray(q_next32, dtype=np.float64)
        ts_np = np.asarray(ts.numpy())

        residual = q_target64 - q_next64
        residual_l2 = float(np.linalg.norm(residual))
        residual_ratio = residual_l2 / initial_l2
        residual_max = float(np.linalg.norm(residual, axis=1).max(initial=0.0))

        should_ipc = (not args.skip_ipc_every_step) or (k == args.k_max)
        safe_intersection = (
            ipc_has_intersections(q_next64, edges, faces)
            if should_ipc
            else None
        )

        rows.append(
            {
                "k": k,
                "collision_detection_calls": k,
                "dat_calls": k,
                "residual_l2": residual_l2,
                "residual_ratio_l2": residual_ratio,
                "residual_max_vertex": residual_max,
                "truncated_vertices": int(np.count_nonzero(ts_np < 1.0 - 1.0e-6)),
                "min_t": float(ts_np.min(initial=1.0)),
                "vt_candidates": vt_count,
                "ee_candidates": ee_count,
                "detect_ms": detect_ms,
                "dat_ms": dat_ms,
                "iteration_ms": detect_ms + dat_ms,
                "safe_intersection": safe_intersection,
                "overflow": overflow,
                "overflow_flags": overflow_flags,
            }
        )

        if safe_intersection is True:
            raise RuntimeError(f"Intersection detected after DAT at K={k}.")
        if overflow is True:
            raise RuntimeError(
                f"Collision buffer overflow detected at K={k}: {overflow_flags}"
            )

        accepted_positions.append(q_next64.copy())
        q_current = q_next

    recommended_k = None
    for r in rows[1:]:
        if (
            r["safe_intersection"] is not True
            and r["overflow"] is not True
            and r["residual_ratio_l2"] <= args.residual_tol
        ):
            recommended_k = int(r["k"])
            break

    stem = f"frame_{args.frame - 1:04d}_to_{args.frame:04d}"
    csv_path = output_dir / f"{stem}_k_scan.csv"
    json_path = output_dir / f"{stem}_k_scan.json"
    png_path = output_dir / f"{stem}_residual.png"
    accepted_path = output_dir / f"{stem}_accepted_positions.npy"

    fieldnames = [
        "k",
        "collision_detection_calls",
        "dat_calls",
        "residual_l2",
        "residual_ratio_l2",
        "residual_max_vertex",
        "truncated_vertices",
        "min_t",
        "vt_candidates",
        "ee_candidates",
        "detect_ms",
        "dat_ms",
        "iteration_ms",
        "safe_intersection",
        "overflow",
        "overflow_flags",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "sample_root": str(sample_root),
        "garments": garment_meta,
        "source_frame": args.frame - 1,
        "target_frame": args.frame,
        "vertex_count": int(len(q_start64)),
        "triangle_count": int(len(faces)),
        "edge_count": int(len(edges)),
        "raw_dtype": "float64",
        "standalone_runtime_dtype": "float32 (wp.vec3)",
        "median_edge_length": med_edge,
        "contact_margin": contact_margin,
        "margin_edge_ratio": contact_margin / med_edge,
        "gamma": args.gamma,
        "k_max": args.k_max,
        "raw_source_intersects": False,
        "raw_target_intersects": raw_target_intersects,
        "initial_displacement_l2": initial_l2,
        "initial_max_vertex_displacement": initial_max,
        "requested_residual_tol": args.residual_tol,
        "recommended_k": recommended_k,
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    np.save(
        accepted_path,
        np.stack(accepted_positions, axis=0).astype(np.float64, copy=False),
    )
    save_plot(rows, png_path)

    print("\n=== DAT single-frame K scan ===")
    print(f"sample:                 {sample_root.name}")
    print(f"transition:             {args.frame - 1} -> {args.frame}")
    print(f"garments:               {', '.join(args.garments)}")
    print(f"V/T/E:                  {len(q_start64)} / {len(faces)} / {len(edges)}")
    print(f"median edge:            {med_edge:.8g}")
    print(f"contact margin:         {contact_margin:.8g}")
    print(f"raw target intersects:  {raw_target_intersects}")
    print()

    print(
        "K  residual_ratio  max_residual  truncated  VTcand  EEcand  "
        "detect_ms  DAT_ms  intersect"
    )
    for r in rows:
        print(
            f"{r['k']:2d} "
            f"{r['residual_ratio_l2']:14.7g} "
            f"{r['residual_max_vertex']:12.7g} "
            f"{r['truncated_vertices']:9d} "
            f"{r['vt_candidates']:7d} "
            f"{r['ee_candidates']:7d} "
            f"{r['detect_ms']:9.4f} "
            f"{r['dat_ms']:7.4f} "
            f"{str(r['safe_intersection']):>9s}"
        )

    if recommended_k is None:
        print(
            f"\nNo K <= {args.k_max} reached residual_ratio <= "
            f"{args.residual_tol:g}."
        )
    else:
        print(
            f"\nFirst safe K with residual_ratio <= {args.residual_tol:g}: "
            f"K={recommended_k}"
        )

    print(f"\nCSV:       {csv_path}")
    print(f"JSON:      {json_path}")
    print(f"plot:      {png_path}")
    print(f"positions: {accepted_path}")


if __name__ == "__main__":
    main()
