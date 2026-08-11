
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FAILURE_DIR = REPO_ROOT / "tests" / "real" / "data"
DEFAULT_NEWTON_REPO = REPO_ROOT / "third_party" / "newton"


def load_combined_faces(sample_root: Path) -> np.ndarray:
    jacket_vertices = np.load(sample_root / "jacket" / "vertices.npy", mmap_mode="r")
    jacket_faces = np.asarray(
        np.load(sample_root / "jacket" / "faces.npy"),
        dtype=np.int32,
    )
    tee_faces = np.asarray(
        np.load(sample_root / "tee_sleeveless" / "faces.npy"),
        dtype=np.int32,
    )

    jacket_v_count = int(jacket_vertices.shape[1])

    return np.ascontiguousarray(
        np.vstack(
            [
                jacket_faces,
                tee_faces + jacket_v_count,
            ]
        ),
        dtype=np.int32,
    )


def canonical_triangle(tri_indices: np.ndarray, tri_id: int):
    return tuple(sorted(int(x) for x in tri_indices[tri_id]))


def canonical_edge(edge_indices: np.ndarray, edge_id: int):
    # Newton / standalone VBD collision edges store endpoints in columns 2,3.
    a = int(edge_indices[edge_id, 2])
    b = int(edge_indices[edge_id, 3])
    return (a, b) if a <= b else (b, a)


def extract_vt(detector, tri_indices: np.ndarray):
    collisions = np.asarray(detector.vertex_colliding_triangles.numpy())
    offsets = np.asarray(detector.vertex_colliding_triangles_offsets.numpy())
    sizes = np.asarray(detector.vertex_colliding_triangles_buffer_sizes.numpy())
    counts = np.asarray(detector.vertex_colliding_triangles_count.numpy())

    pairs = set()
    raw_valid = 0
    invalid = 0
    row_vertex_mismatch = 0
    overflow_rows = int(np.count_nonzero(counts > sizes))

    for row_v in range(len(counts)):
        n = min(int(counts[row_v]), int(sizes[row_v]))
        off = int(offsets[row_v])

        for j in range(n):
            k = 2 * (off + j)
            recorded_v = int(collisions[k])
            tri_id = int(collisions[k + 1])

            if recorded_v < 0 or tri_id < 0:
                invalid += 1
                continue

            raw_valid += 1
            if recorded_v != row_v:
                row_vertex_mismatch += 1

            pairs.add(
                (
                    recorded_v,
                    canonical_triangle(tri_indices, tri_id),
                )
            )

    return {
        "set": pairs,
        "raw_valid": raw_valid,
        "invalid": invalid,
        "overflow_rows": overflow_rows,
        "max_count": int(counts.max(initial=0)),
        "max_capacity": int(sizes.max(initial=0)),
        "row_vertex_mismatch": row_vertex_mismatch,
    }


def extract_ee(detector, edge_indices: np.ndarray):
    collisions = np.asarray(detector.edge_colliding_edges.numpy())
    offsets = np.asarray(detector.edge_colliding_edges_offsets.numpy())
    sizes = np.asarray(detector.edge_colliding_edges_buffer_sizes.numpy())
    counts = np.asarray(detector.edge_colliding_edges_count.numpy())

    directed = set()
    undirected = set()
    raw_valid = 0
    invalid = 0
    row_edge_mismatch = 0
    overflow_rows = int(np.count_nonzero(counts > sizes))

    for row_e in range(len(counts)):
        n = min(int(counts[row_e]), int(sizes[row_e]))
        off = int(offsets[row_e])

        for j in range(n):
            k = 2 * (off + j)
            recorded_e0 = int(collisions[k])
            e1 = int(collisions[k + 1])

            if recorded_e0 < 0 or e1 < 0:
                invalid += 1
                continue

            raw_valid += 1
            if recorded_e0 != row_e:
                row_edge_mismatch += 1

            e0_key = canonical_edge(edge_indices, recorded_e0)
            e1_key = canonical_edge(edge_indices, e1)

            directed.add((e0_key, e1_key))
            undirected.add(tuple(sorted((e0_key, e1_key))))

    return {
        "directed": directed,
        "undirected": undirected,
        "raw_valid": raw_valid,
        "invalid": invalid,
        "overflow_rows": overflow_rows,
        "max_count": int(counts.max(initial=0)),
        "max_capacity": int(sizes.max(initial=0)),
        "row_edge_mismatch": row_edge_mismatch,
    }


def serializable_examples(values, limit=20):
    out = []
    for value in sorted(values)[:limit]:
        out.append(value)
    return out


def print_set_comparison(name: str, standalone_set: set, newton_set: set):
    s_only = standalone_set - newton_set
    n_only = newton_set - standalone_set
    common = standalone_set & newton_set

    print(f"\n=== {name} set comparison ===")
    print(f"standalone unique: {len(standalone_set)}")
    print(f"Newton unique:     {len(newton_set)}")
    print(f"common:            {len(common)}")
    print(f"standalone only:   {len(s_only)}")
    print(f"Newton only:       {len(n_only)}")
    print(f"exact set parity:  {s_only == set() and n_only == set()}")

    if s_only:
        print("standalone-only examples:")
        for x in serializable_examples(s_only, 10):
            print(" ", x)

    if n_only:
        print("Newton-only examples:")
        for x in serializable_examples(n_only, 10):
            print(" ", x)

    return s_only, n_only


def build_newton_model(newton, q_before32, faces, device):
    # Match Newton's own collision-detector test construction.
    vertices = [wp.vec3(v) for v in q_before32]

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vertices=vertices,
        indices=faces.reshape(-1).tolist(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        density=0.02,
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
    )
    return builder.finalize(device=device)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compare standalone and Newton VBD collision detection on the saved "
            "real failure snapshot. Candidate sets are compared geometrically, "
            "not by internal edge IDs."
        )
    )
    ap.add_argument("sample_root", type=Path)
    ap.add_argument("--failure-dir", type=Path, default=DEFAULT_FAILURE_DIR)
    ap.add_argument("--newton-repo", type=Path, default=DEFAULT_NEWTON_REPO)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--topology-ring", type=int, default=0)
    ap.add_argument("--buffer", type=int, default=64)
    args = ap.parse_args()

    newton_repo = args.newton_repo.resolve()
    if not (newton_repo / "newton").is_dir():
        raise FileNotFoundError(
            f"Newton checkout not found at {newton_repo}; expected "
            f"{newton_repo / 'newton'}"
        )
    sys.path.insert(0, str(newton_repo))

    from contactdat import (
        TriMeshCollisionDetector as StandaloneDetector,
    )

    import newton
    from newton._src.solvers.vbd.tri_mesh_collision import (
        TriMeshCollisionDetector as NewtonDetector,
    )

    sample_root = args.sample_root.resolve()
    failure_dir = args.failure_dir.resolve()

    q_before64 = np.asarray(
        np.load(failure_dir / "fail_q_before.npy"),
        dtype=np.float64,
    )
    q_before32 = np.asarray(q_before64, dtype=np.float32)
    faces = load_combined_faces(sample_root)

    device = wp.get_device(args.device)
    q_wp = wp.array(q_before32, dtype=wp.vec3, device=device)

    # ------------------------------------------------------------------
    # Standalone detector
    # ------------------------------------------------------------------
    standalone = StandaloneDetector(
        q_wp,
        faces,
        device=device,
        vertex_collision_buffer_pre_alloc=args.buffer,
        vertex_collision_buffer_max_alloc=256,
        edge_collision_buffer_pre_alloc=args.buffer,
        edge_collision_buffer_max_alloc=256,
        topological_contact_filter_threshold=args.topology_ring,
        auto_resize=True,
    )

    standalone.refit(q_wp)
    standalone.vertex_triangle_collision_detection(
        args.margin,
        min_query_radius=0.0,
        min_distance_filtering_ref_pos=None,
    )
    standalone.edge_edge_collision_detection(
        args.margin,
        min_query_radius=0.0,
        min_distance_filtering_ref_pos=None,
    )
    wp.synchronize_device(device)

    s_tri = np.asarray(standalone.tri_indices.numpy(), dtype=np.int32)
    s_edge = np.asarray(standalone.edge_indices.numpy(), dtype=np.int32)

    # ------------------------------------------------------------------
    # Newton detector: independent ModelBuilder + independent BVHs/buffers.
    # ------------------------------------------------------------------
    model = build_newton_model(newton, q_before32, faces, device)

    newton_detector = NewtonDetector(
        model=model,
        record_triangle_contacting_vertices=False,
        vertex_collision_buffer_pre_alloc=args.buffer,
        vertex_collision_buffer_max_alloc=256,
        edge_collision_buffer_pre_alloc=args.buffer,
        edge_collision_buffer_max_alloc=256,
        topological_contact_filter_threshold=args.topology_ring,
    )

    # Explicitly put Newton on the exact same snapshot before detection.
    newton_detector.refit(model.particle_q)
    newton_detector.vertex_triangle_collision_detection(
        args.margin,
        min_query_radius=0.0,
        min_distance_filtering_ref_pos=None,
    )
    newton_detector.edge_edge_collision_detection(
        args.margin,
        min_query_radius=0.0,
        min_distance_filtering_ref_pos=None,
    )
    wp.synchronize_device(device)

    n_q = np.asarray(model.particle_q.numpy(), dtype=np.float32)
    n_tri = np.asarray(model.tri_indices.numpy(), dtype=np.int32)
    n_edge = np.asarray(model.edge_indices.numpy(), dtype=np.int32)

    print("=== topology / position sanity ===")
    print("particle count standalone/Newton:", len(q_before32), model.particle_count)
    print("triangle count standalone/Newton:", len(s_tri), len(n_tri))
    print("edge count standalone/Newton:    ", len(s_edge), len(n_edge))
    print("max |q_input - q_newton|:        ", float(np.max(np.abs(q_before32 - n_q))))
    print("triangle arrays exactly equal:   ", bool(np.array_equal(s_tri, n_tri)))

    s_edge_geom = {canonical_edge(s_edge, i) for i in range(len(s_edge))}
    n_edge_geom = {canonical_edge(n_edge, i) for i in range(len(n_edge))}
    print("edge geometry sets equal:        ", s_edge_geom == n_edge_geom)
    print("standalone-only topology edges:  ", len(s_edge_geom - n_edge_geom))
    print("Newton-only topology edges:      ", len(n_edge_geom - s_edge_geom))

    # ------------------------------------------------------------------
    # Extract candidate sets.
    # ------------------------------------------------------------------
    s_vt = extract_vt(standalone, s_tri)
    n_vt = extract_vt(newton_detector, n_tri)

    s_ee = extract_ee(standalone, s_edge)
    n_ee = extract_ee(newton_detector, n_edge)

    print("\n=== raw buffer statistics ===")
    print(
        "VT standalone: "
        f"raw={s_vt['raw_valid']} unique={len(s_vt['set'])} "
        f"max={s_vt['max_count']}/{s_vt['max_capacity']} "
        f"overflow_rows={s_vt['overflow_rows']} invalid={s_vt['invalid']}"
    )
    print(
        "VT Newton:     "
        f"raw={n_vt['raw_valid']} unique={len(n_vt['set'])} "
        f"max={n_vt['max_count']}/{n_vt['max_capacity']} "
        f"overflow_rows={n_vt['overflow_rows']} invalid={n_vt['invalid']}"
    )
    print(
        "EE standalone: "
        f"raw={s_ee['raw_valid']} "
        f"directed_unique={len(s_ee['directed'])} "
        f"undirected_unique={len(s_ee['undirected'])} "
        f"max={s_ee['max_count']}/{s_ee['max_capacity']} "
        f"overflow_rows={s_ee['overflow_rows']} invalid={s_ee['invalid']}"
    )
    print(
        "EE Newton:     "
        f"raw={n_ee['raw_valid']} "
        f"directed_unique={len(n_ee['directed'])} "
        f"undirected_unique={len(n_ee['undirected'])} "
        f"max={n_ee['max_count']}/{n_ee['max_capacity']} "
        f"overflow_rows={n_ee['overflow_rows']} invalid={n_ee['invalid']}"
    )

    vt_s_only, vt_n_only = print_set_comparison(
        "VT",
        s_vt["set"],
        n_vt["set"],
    )
    ee_d_s_only, ee_d_n_only = print_set_comparison(
        "EE directed geometry",
        s_ee["directed"],
        n_ee["directed"],
    )
    ee_u_s_only, ee_u_n_only = print_set_comparison(
        "EE undirected geometry",
        s_ee["undirected"],
        n_ee["undirected"],
    )

    exact_vt = not vt_s_only and not vt_n_only
    exact_ee_directed = not ee_d_s_only and not ee_d_n_only
    exact_ee_undirected = not ee_u_s_only and not ee_u_n_only

    print("\n=== FINAL ===")
    print("VT exact parity:            ", exact_vt)
    print("EE directed exact parity:   ", exact_ee_directed)
    print("EE undirected exact parity: ", exact_ee_undirected)
    print(
        "Collision detection parity:",
        exact_vt and exact_ee_directed,
    )

    report = {
        "margin": args.margin,
        "topology_ring": args.topology_ring,
        "topology": {
            "particle_count": int(len(q_before32)),
            "standalone_triangle_count": int(len(s_tri)),
            "newton_triangle_count": int(len(n_tri)),
            "standalone_edge_count": int(len(s_edge)),
            "newton_edge_count": int(len(n_edge)),
            "max_q_diff": float(np.max(np.abs(q_before32 - n_q))),
            "triangle_arrays_equal": bool(np.array_equal(s_tri, n_tri)),
            "edge_geometry_sets_equal": bool(s_edge_geom == n_edge_geom),
        },
        "vt": {
            "standalone_raw": s_vt["raw_valid"],
            "newton_raw": n_vt["raw_valid"],
            "standalone_unique": len(s_vt["set"]),
            "newton_unique": len(n_vt["set"]),
            "standalone_only": len(vt_s_only),
            "newton_only": len(vt_n_only),
            "exact_parity": exact_vt,
            "standalone_only_examples": serializable_examples(vt_s_only),
            "newton_only_examples": serializable_examples(vt_n_only),
        },
        "ee_directed": {
            "standalone_unique": len(s_ee["directed"]),
            "newton_unique": len(n_ee["directed"]),
            "standalone_only": len(ee_d_s_only),
            "newton_only": len(ee_d_n_only),
            "exact_parity": exact_ee_directed,
            "standalone_only_examples": serializable_examples(ee_d_s_only),
            "newton_only_examples": serializable_examples(ee_d_n_only),
        },
        "ee_undirected": {
            "standalone_unique": len(s_ee["undirected"]),
            "newton_unique": len(n_ee["undirected"]),
            "standalone_only": len(ee_u_s_only),
            "newton_only": len(ee_u_n_only),
            "exact_parity": exact_ee_undirected,
        },
        "collision_detection_parity": bool(exact_vt and exact_ee_directed),
    }

    report_path = failure_dir / "collision_detection_parity.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nreport:", report_path)


if __name__ == "__main__":
    main()
