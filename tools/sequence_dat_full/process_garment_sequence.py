
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import warp as wp

from standalone_contact.collision_detector import TriMeshCollisionDetector
from standalone_contact.penetration_free_contact import PenetrationFreeContact


def unique_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    return np.asfortranarray(np.unique(edges, axis=0), dtype=np.int32)


def load_garments(sample_root: Path, garments: list[str]):
    seqs = []
    faces_all = []
    offsets = []
    meta = []
    offset = 0
    frame_count = None

    for owner, name in enumerate(garments, start=2):
        gdir = sample_root / name
        seq = np.load(gdir / "vertices.npy", mmap_mode="r")
        faces = np.asarray(np.load(gdir / "faces.npy"), dtype=np.int32)

        if seq.ndim != 3 or seq.shape[2] != 3:
            raise ValueError(f"{name}: expected vertices [F,V,3], got {seq.shape}")

        if frame_count is None:
            frame_count = int(seq.shape[0])
        elif int(seq.shape[0]) != frame_count:
            raise ValueError("All garment sequences must have identical frame count.")

        seqs.append(seq)
        faces_all.append(faces + offset)
        offsets.append((offset, offset + seq.shape[1]))
        meta.append(
            dict(
                name=name,
                owner=owner,
                vertex_offset=int(offset),
                vertex_count=int(seq.shape[1]),
                triangle_count=int(faces.shape[0]),
            )
        )
        offset += seq.shape[1]

    faces_combined = np.ascontiguousarray(np.vstack(faces_all), dtype=np.int32)
    edges_combined = unique_edges_from_faces(faces_combined)
    return seqs, faces_combined, edges_combined, offsets, meta, frame_count


def combined_frame(seqs, frame: int) -> np.ndarray:
    return np.ascontiguousarray(
        np.vstack([np.asarray(s[frame], dtype=np.float64) for s in seqs]),
        dtype=np.float64,
    )


def ipc_has_intersections(vertices, edges, faces) -> bool:
    import ipctk

    V = np.asfortranarray(vertices, dtype=np.float64)
    E = np.asfortranarray(edges, dtype=np.int32)
    F = np.asfortranarray(faces, dtype=np.int32)
    mesh = ipctk.CollisionMesh(V, E, F)
    return bool(ipctk.has_intersections(mesh, V))


def synchronize(device):
    wp.synchronize_device(device)


def detector_counts(detector):
    vt = int(np.asarray(detector.vertex_colliding_triangles_count.numpy()).sum())
    ee = int(np.asarray(detector.edge_colliding_edges_count.numpy()).sum())
    return vt, ee


def overflow_state(detector):
    if hasattr(detector, "overflow_flags"):
        try:
            flags = np.asarray(detector.overflow_flags()).astype(bool).reshape(-1)
            return bool(np.any(flags[:3])), flags.astype(int).tolist()
        except Exception:
            pass
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_root", type=Path)
    ap.add_argument("output_root", type=Path)
    ap.add_argument("--garments", nargs="+", default=["jacket", "tee_sleeveless"])
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--contact-margin", type=float, required=True)
    ap.add_argument("--gamma", type=float, default=0.85)
    ap.add_argument("--vt-buffer", type=int, default=64)
    ap.add_argument("--ee-buffer", type=int, default=64)
    ap.add_argument("--topology-ring", type=int, default=2)
    ap.add_argument(
        "--check-every-frame",
        action="store_true",
        help="IPC static check on every accepted frame (recommended for first full run).",
    )
    ap.add_argument(
        "--save-float32",
        action="store_true",
        help="Save output sequence as float32 instead of float64.",
    )
    args = ap.parse_args()

    if args.k < 1:
        raise ValueError("--k must be >= 1")

    sample_root = args.sample_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    seqs, faces, edges, offsets, garment_meta, frame_count = load_garments(
        sample_root, args.garments
    )

    q0_64 = combined_frame(seqs, 0)
    if ipc_has_intersections(q0_64, edges, faces):
        raise RuntimeError("Combined garments are intersecting at frame 0.")

    device = wp.get_device(args.device)
    q_current = wp.array(q0_64.astype(np.float32), dtype=wp.vec3, device=device)

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
        particle_count=len(q0_64),
        edge_count=detector.edge_count,
        device=device,
        contact_margin=float(args.contact_margin),
        conservative_bound_relaxation=float(args.gamma),
        rest_q=None,
        rest_exclusion_radius=0.0,
    )

    out_dtype = np.float32 if args.save_float32 else np.float64

    # Use open_memmap to avoid holding 405 x 23k x 3 twice in RAM.
    out_arrays = []
    for (start, end), gmeta in zip(offsets, garment_meta):
        gdir = output_root / gmeta["name"]
        gdir.mkdir(parents=True, exist_ok=True)

        out_v = np.lib.format.open_memmap(
            gdir / "vertices.npy",
            mode="w+",
            dtype=out_dtype,
            shape=(frame_count, end - start, 3),
        )
        out_v[0] = q0_64[start:end].astype(out_dtype, copy=False)
        out_arrays.append(out_v)

        src_faces = sample_root / gmeta["name"] / "faces.npy"
        shutil.copy2(src_faces, gdir / "faces.npy")

    rows = []
    rows.append(
        dict(
            frame=0,
            k=args.k,
            residual_l2=0.0,
            residual_ratio_l2=0.0,
            residual_max_vertex=0.0,
            truncated_vertices_total=0,
            min_t=1.0,
            vt_candidates_last=0,
            ee_candidates_last=0,
            detect_ms_total=0.0,
            dat_ms_total=0.0,
            frame_ms=0.0,
            processed_intersects=False,
            overflow=False,
        )
    )

    for frame in range(1, frame_count):
        q_target64 = combined_frame(seqs, frame)
        q_target32 = q_target64.astype(np.float32)

        q_start64 = np.asarray(q_current.numpy(), dtype=np.float64)
        initial_residual = q_target64 - q_start64
        initial_l2 = float(np.linalg.norm(initial_residual))

        detect_ms_total = 0.0
        dat_ms_total = 0.0
        truncated_total = 0
        min_t_all = 1.0
        vt_last = 0
        ee_last = 0
        any_overflow = False

        synchronize(device)
        frame_t0 = time.perf_counter()

        for _ in range(args.k):
            synchronize(device)
            t0 = time.perf_counter()
            contact.begin(q_current)
            synchronize(device)
            detect_ms_total += (time.perf_counter() - t0) * 1000.0

            vt_last, ee_last = detector_counts(detector)
            overflow, flags = overflow_state(detector)
            if overflow is True:
                any_overflow = True
                raise RuntimeError(
                    f"Collision buffer overflow at frame={frame}, flags={flags}"
                )

            q_cur32 = np.asarray(q_current.numpy(), dtype=np.float32)
            dx32 = q_target32 - q_cur32
            dx_wp = wp.array(dx32, dtype=wp.vec3, device=device)

            synchronize(device)
            t0 = time.perf_counter()
            _, q_next, ts = contact.truncate(dx_wp)
            synchronize(device)
            dat_ms_total += (time.perf_counter() - t0) * 1000.0

            ts_np = np.asarray(ts.numpy())
            truncated_total += int(np.count_nonzero(ts_np < 1.0 - 1.0e-6))
            min_t_all = min(min_t_all, float(ts_np.min(initial=1.0)))

            q_current = q_next

        synchronize(device)
        frame_ms = (time.perf_counter() - frame_t0) * 1000.0

        q_accept64 = np.asarray(q_current.numpy(), dtype=np.float64)

        residual = q_target64 - q_accept64
        residual_l2 = float(np.linalg.norm(residual))
        residual_ratio = residual_l2 / initial_l2 if initial_l2 > 0 else 0.0
        residual_max = float(np.linalg.norm(residual, axis=1).max(initial=0.0))

        intersects = None
        if args.check_every_frame:
            intersects = ipc_has_intersections(q_accept64, edges, faces)
            if intersects:
                raise RuntimeError(
                    f"Accepted combined garment mesh intersects at frame {frame}."
                )

        for (start, end), out_v in zip(offsets, out_arrays):
            out_v[frame] = q_accept64[start:end].astype(out_dtype, copy=False)

        rows.append(
            dict(
                frame=frame,
                k=args.k,
                residual_l2=residual_l2,
                residual_ratio_l2=residual_ratio,
                residual_max_vertex=residual_max,
                truncated_vertices_total=truncated_total,
                min_t=min_t_all,
                vt_candidates_last=vt_last,
                ee_candidates_last=ee_last,
                detect_ms_total=detect_ms_total,
                dat_ms_total=dat_ms_total,
                frame_ms=frame_ms,
                processed_intersects=intersects,
                overflow=any_overflow,
            )
        )

        if frame % 10 == 0 or frame == frame_count - 1:
            print(
                f"[{frame:4d}/{frame_count-1}] "
                f"res={residual_ratio:.6g} "
                f"max={residual_max:.6g} "
                f"trunc={truncated_total} "
                f"VT={vt_last} EE={ee_last} "
                f"{frame_ms:.2f} ms "
                f"intersect={intersects}"
            )

    for out_v in out_arrays:
        out_v.flush()

    csv_path = output_root / "processing_report.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "source": str(sample_root),
        "output": str(output_root),
        "garments": garment_meta,
        "frame_count": frame_count,
        "k": args.k,
        "contact_margin": args.contact_margin,
        "gamma": args.gamma,
        "runtime_dtype": "float32 (wp.vec3)",
        "storage_dtype": str(np.dtype(out_dtype)),
        "check_every_frame": args.check_every_frame,
        "max_residual_ratio": float(max(r["residual_ratio_l2"] for r in rows)),
        "max_residual_vertex": float(max(r["residual_max_vertex"] for r in rows)),
        "total_truncated_vertex_events": int(
            sum(r["truncated_vertices_total"] for r in rows)
        ),
        "mean_frame_ms": float(np.mean([r["frame_ms"] for r in rows[1:]])),
    }
    (output_root / "processing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nDone.")
    print(f"processed sequence: {output_root}")
    print(f"report:             {csv_path}")


if __name__ == "__main__":
    main()
