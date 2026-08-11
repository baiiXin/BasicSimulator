
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAILURE_DIR = REPO_ROOT / "tests" / "real" / "data"
DEFAULT_NEWTON_REPO = REPO_ROOT / "third_party" / "newton"
DEFAULT_REPORT = REPO_ROOT / ".artifacts" / "final_newton_consistency_report.json"


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def load_combined_faces(root: Path) -> np.ndarray:
    vj = np.load(root / "jacket" / "vertices.npy", mmap_mode="r")
    fj = np.asarray(np.load(root / "jacket" / "faces.npy"), dtype=np.int32)
    ft = np.asarray(np.load(root / "tee_sleeveless" / "faces.npy"), dtype=np.int32)
    nj = int(vj.shape[1])
    return np.ascontiguousarray(np.vstack([fj, ft + nj]), dtype=np.int32)


def load_combined_frame(root: Path, frame: int) -> np.ndarray:
    vj = np.load(root / "jacket" / "vertices.npy", mmap_mode="r")
    vt = np.load(root / "tee_sleeveless" / "vertices.npy", mmap_mode="r")
    return np.ascontiguousarray(
        np.vstack([
            np.asarray(vj[frame], dtype=np.float64),
            np.asarray(vt[frame], dtype=np.float64),
        ]),
        dtype=np.float64,
    )


def canonical_triangle(tri_indices, tri_id):
    return tuple(sorted(int(x) for x in tri_indices[tri_id]))


def canonical_edge(edge_indices, edge_id):
    a = int(edge_indices[edge_id, 2])
    b = int(edge_indices[edge_id, 3])
    return (a, b) if a <= b else (b, a)


def extract_vt(detector, tri_indices):
    buf = np.asarray(detector.vertex_colliding_triangles.numpy(), dtype=np.int32)
    offs = np.asarray(detector.vertex_colliding_triangles_offsets.numpy(), dtype=np.int32)
    sizes = np.asarray(detector.vertex_colliding_triangles_buffer_sizes.numpy(), dtype=np.int32)
    counts = np.asarray(detector.vertex_colliding_triangles_count.numpy(), dtype=np.int32)

    out = set()
    for v in range(len(counts)):
        n = min(int(counts[v]), int(sizes[v]))
        off = int(offs[v])
        for j in range(n):
            k = 2 * (off + j)
            vv = int(buf[k])
            tt = int(buf[k + 1])
            if vv >= 0 and tt >= 0:
                out.add((vv, canonical_triangle(tri_indices, tt)))
    return out


def extract_ee(detector, edge_indices):
    buf = np.asarray(detector.edge_colliding_edges.numpy(), dtype=np.int32)
    offs = np.asarray(detector.edge_colliding_edges_offsets.numpy(), dtype=np.int32)
    sizes = np.asarray(detector.edge_colliding_edges_buffer_sizes.numpy(), dtype=np.int32)
    counts = np.asarray(detector.edge_colliding_edges_count.numpy(), dtype=np.int32)

    directed = set()
    undirected = set()

    for e0 in range(len(counts)):
        n = min(int(counts[e0]), int(sizes[e0]))
        off = int(offs[e0])

        for j in range(n):
            k = 2 * (off + j)
            ee0 = int(buf[k])
            ee1 = int(buf[k + 1])

            if ee0 < 0 or ee1 < 0:
                continue

            a = canonical_edge(edge_indices, ee0)
            b = canonical_edge(edge_indices, ee1)

            directed.add((a, b))
            undirected.add(tuple(sorted((a, b))))

    return directed, undirected


# -----------------------------------------------------------------------------
# Newton construction
# -----------------------------------------------------------------------------

def build_newton_model(newton, q32, faces, device):
    vertices = [wp.vec3(v) for v in q32]

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


def run_newton_dat(
    q,
    dx,
    detector,
    *,
    particle_count,
    margin,
    gamma,
    device,
    NewtonCI,
    newton_dat_kernel,
    newton_apply_kernel,
    threads_per_primitive,
):
    ci = detector.get_collision_data()
    ci_array = wp.array([ci], dtype=NewtonCI, device=device)

    t = wp.ones(particle_count, dtype=float, device=device)
    dx_safe = wp.zeros(particle_count, dtype=wp.vec3, device=device)
    q_safe = wp.zeros(particle_count, dtype=wp.vec3, device=device)

    dim = max(
        particle_count * threads_per_primitive,
        detector.model.edge_count * threads_per_primitive,
    )

    wp.launch(
        kernel=newton_dat_kernel,
        dim=dim,
        inputs=[
            q,
            dx,
            detector.model.tri_indices,
            detector.model.edge_indices,
            ci_array,
            float(detector.edge_edge_parallel_epsilon),
            float(gamma),
            t,
        ],
        device=device,
    )

    max_displacement = float(margin * gamma * 0.5)

    wp.launch(
        kernel=newton_apply_kernel,
        dim=particle_count,
        inputs=[
            q,
            dx,
            t,
            max_displacement,
            dx_safe,
            q_safe,
        ],
        device=device,
    )

    return dx_safe, q_safe, t


# -----------------------------------------------------------------------------
# One case
# -----------------------------------------------------------------------------

def run_case(
    name,
    q64,
    target64,
    faces,
    *,
    device,
    margin,
    gamma,
    topology_ring,
    StandaloneDetector,
    PenetrationFreeContact,
    newton,
    NewtonDetector,
    NewtonCI,
    newton_dat_kernel,
    newton_apply_kernel,
    threads_per_primitive,
    atol,
):
    q32 = np.asarray(q64, dtype=np.float32)
    target32 = np.asarray(target64, dtype=np.float32)
    dx32 = target32 - q32

    q_s = wp.array(q32, dtype=wp.vec3, device=device)
    dx_s = wp.array(dx32, dtype=wp.vec3, device=device)

    # ------------------------------------------------------------------
    # Standalone: independent detector + standalone DAT
    # ------------------------------------------------------------------
    sd = StandaloneDetector(
        q_s,
        faces,
        device=device,
        vertex_collision_buffer_pre_alloc=64,
        vertex_collision_buffer_max_alloc=256,
        edge_collision_buffer_pre_alloc=64,
        edge_collision_buffer_max_alloc=256,
        topological_contact_filter_threshold=topology_ring,
        auto_resize=True,
    )

    sc = PenetrationFreeContact(
        sd,
        particle_count=len(q32),
        edge_count=sd.edge_count,
        device=device,
        contact_margin=margin,
        conservative_bound_relaxation=gamma,
        rest_q=None,
        rest_exclusion_radius=0.0,
    )

    sc.begin(q_s)
    s_dx_safe, s_q_safe, s_t = sc.truncate(dx_s)
    wp.synchronize_device(device)

    s_tri = np.asarray(sd.tri_indices.numpy(), dtype=np.int32)
    s_edge = np.asarray(sd.edge_indices.numpy(), dtype=np.int32)
    s_vt = extract_vt(sd, s_tri)
    s_ee_d, s_ee_u = extract_ee(sd, s_edge)

    s_t_np = np.asarray(s_t.numpy(), dtype=np.float32)
    s_q_np = np.asarray(s_q_safe.numpy(), dtype=np.float32)
    s_dx_np = np.asarray(s_dx_safe.numpy(), dtype=np.float32)

    # ------------------------------------------------------------------
    # Newton: independent model + independent detector + Newton DAT
    # ------------------------------------------------------------------
    model = build_newton_model(newton, q32, faces, device)

    nd = NewtonDetector(
        model=model,
        record_triangle_contacting_vertices=False,
        vertex_collision_buffer_pre_alloc=64,
        vertex_collision_buffer_max_alloc=256,
        edge_collision_buffer_pre_alloc=64,
        edge_collision_buffer_max_alloc=256,
        topological_contact_filter_threshold=topology_ring,
    )

    nd.refit(model.particle_q)
    nd.vertex_triangle_collision_detection(
        margin,
        min_query_radius=0.0,
        min_distance_filtering_ref_pos=None,
    )
    nd.edge_edge_collision_detection(
        margin,
        min_query_radius=0.0,
        min_distance_filtering_ref_pos=None,
    )
    wp.synchronize_device(device)

    n_tri = np.asarray(model.tri_indices.numpy(), dtype=np.int32)
    n_edge = np.asarray(model.edge_indices.numpy(), dtype=np.int32)
    n_vt = extract_vt(nd, n_tri)
    n_ee_d, n_ee_u = extract_ee(nd, n_edge)

    dx_n = wp.array(dx32, dtype=wp.vec3, device=device)
    n_dx_safe, n_q_safe, n_t = run_newton_dat(
        model.particle_q,
        dx_n,
        nd,
        particle_count=len(q32),
        margin=margin,
        gamma=gamma,
        device=device,
        NewtonCI=NewtonCI,
        newton_dat_kernel=newton_dat_kernel,
        newton_apply_kernel=newton_apply_kernel,
        threads_per_primitive=threads_per_primitive,
    )
    wp.synchronize_device(device)

    n_t_np = np.asarray(n_t.numpy(), dtype=np.float32)
    n_q_np = np.asarray(n_q_safe.numpy(), dtype=np.float32)
    n_dx_np = np.asarray(n_dx_safe.numpy(), dtype=np.float32)

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    q_model_np = np.asarray(model.particle_q.numpy(), dtype=np.float32)

    topo_q_err = float(np.max(np.abs(q_model_np - q32)))
    triangles_equal = bool(np.array_equal(s_tri, n_tri))

    s_edge_geom = {canonical_edge(s_edge, i) for i in range(len(s_edge))}
    n_edge_geom = {canonical_edge(n_edge, i) for i in range(len(n_edge))}
    edge_topology_equal = bool(s_edge_geom == n_edge_geom)

    vt_equal = bool(s_vt == n_vt)
    ee_directed_equal = bool(s_ee_d == n_ee_d)
    ee_undirected_equal = bool(s_ee_u == n_ee_u)

    max_t = float(np.max(np.abs(s_t_np - n_t_np)))
    max_q = float(np.max(np.abs(s_q_np - n_q_np)))
    max_dx = float(np.max(np.abs(s_dx_np - n_dx_np)))

    detector_pass = (
        topo_q_err <= atol
        and triangles_equal
        and edge_topology_equal
        and vt_equal
        and ee_directed_equal
        and ee_undirected_equal
    )

    dat_pass = max_t <= atol and max_q <= atol and max_dx <= atol
    overall = detector_pass and dat_pass

    print()
    print(f"=== CASE: {name} ===")
    print("Detector:")
    print(f"  q topology max error:       {topo_q_err:.9g}")
    print(f"  triangles exact:            {triangles_equal}")
    print(f"  edge topology exact:        {edge_topology_equal}")
    print(f"  VT set exact:               {vt_equal} ({len(s_vt)} / {len(n_vt)})")
    print(f"  EE directed set exact:      {ee_directed_equal} ({len(s_ee_d)} / {len(n_ee_d)})")
    print(f"  EE undirected set exact:    {ee_undirected_equal} ({len(s_ee_u)} / {len(n_ee_u)})")
    print("DAT:")
    print(f"  max |t_s - t_n|:           {max_t:.9g}")
    print(f"  max |dx_s - dx_n|:         {max_dx:.9g}")
    print(f"  max |q_s - q_n|:           {max_q:.9g}")
    print(f"  detector PASS:              {detector_pass}")
    print(f"  DAT PASS:                   {dat_pass}")
    print(f"  CASE PASS:                  {overall}")

    return {
        "name": name,
        "topology_q_max_error": topo_q_err,
        "triangles_exact": triangles_equal,
        "edge_topology_exact": edge_topology_equal,
        "vt_standalone": len(s_vt),
        "vt_newton": len(n_vt),
        "vt_exact": vt_equal,
        "ee_directed_standalone": len(s_ee_d),
        "ee_directed_newton": len(n_ee_d),
        "ee_directed_exact": ee_directed_equal,
        "ee_undirected_standalone": len(s_ee_u),
        "ee_undirected_newton": len(n_ee_u),
        "ee_undirected_exact": ee_undirected_equal,
        "max_t_diff": max_t,
        "max_dx_diff": max_dx,
        "max_q_diff": max_q,
        "detector_pass": detector_pass,
        "dat_pass": dat_pass,
        "pass": overall,
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Final end-to-end consistency test: standalone collision detection "
            "+ DAT versus Newton collision detection + DAT."
        )
    )
    ap.add_argument("raw_root", type=Path)
    ap.add_argument("processed_root", type=Path)
    ap.add_argument("--failure-dir", type=Path, default=DEFAULT_FAILURE_DIR)
    ap.add_argument("--newton-repo", type=Path, default=DEFAULT_NEWTON_REPO)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.85)
    ap.add_argument("--topology-ring", type=int, default=2)
    ap.add_argument("--atol", type=float, default=1.0e-6)
    ap.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )
    args = ap.parse_args()

    newton_repo = args.newton_repo.resolve()
    if not (newton_repo / "newton").is_dir():
        raise FileNotFoundError(
            f"Newton checkout not found at {newton_repo}"
        )
    sys.path.insert(0, str(newton_repo))

    from contactdat import (
        PenetrationFreeContact,
        TriMeshCollisionDetector as StandaloneDetector,
    )

    import newton
    from newton._src.solvers.vbd.tri_mesh_collision import (
        TriMeshCollisionDetector as NewtonDetector,
        TriMeshCollisionInfo as NewtonCI,
    )
    from newton._src.solvers.vbd.particle_vbd_kernels import (
        NUM_THREADS_PER_COLLISION_PRIMITIVE,
        apply_planar_truncation_parallel_by_collision as newton_dat_kernel,
        apply_truncation_ts as newton_apply_kernel,
    )

    raw_root = args.raw_root.resolve()
    processed_root = args.processed_root.resolve()
    failure_dir = args.failure_dir.resolve()

    faces = load_combined_faces(raw_root)
    device = wp.get_device(args.device)

    # Three real cases:
    # 1) easy early motion
    # 2) real frame 62 -> 63 motion
    # 3) exact saved failing micro-step
    cases = [
        (
            "raw frame 0 -> raw frame 1",
            load_combined_frame(raw_root, 0),
            load_combined_frame(raw_root, 1),
        ),
        (
            "processed frame 62 -> raw frame 63",
            load_combined_frame(processed_root, 62),
            load_combined_frame(raw_root, 63),
        ),
        (
            "saved frame63 failing micro-step",
            np.asarray(
                np.load(failure_dir / "fail_q_before.npy"),
                dtype=np.float64,
            ),
            np.asarray(
                np.load(failure_dir / "fail_q_target.npy"),
                dtype=np.float64,
            ),
        ),
    ]

    print("=== FINAL NEWTON CONSISTENCY TEST ===")
    print("topology ring:", args.topology_ring)
    print("margin:       ", args.margin)
    print("gamma:        ", args.gamma)
    print("atol:         ", args.atol)

    results = []

    for name, q, target in cases:
        r = run_case(
            name,
            q,
            target,
            faces,
            device=device,
            margin=args.margin,
            gamma=args.gamma,
            topology_ring=args.topology_ring,
            StandaloneDetector=StandaloneDetector,
            PenetrationFreeContact=PenetrationFreeContact,
            newton=newton,
            NewtonDetector=NewtonDetector,
            NewtonCI=NewtonCI,
            newton_dat_kernel=newton_dat_kernel,
            newton_apply_kernel=newton_apply_kernel,
            threads_per_primitive=NUM_THREADS_PER_COLLISION_PRIMITIVE,
            atol=args.atol,
        )
        results.append(r)

    overall = all(r["pass"] for r in results)

    print()
    print("=== FINAL RESULT ===")
    for r in results:
        print(f"{r['name']}: {'PASS' if r['pass'] else 'FAIL'}")
    print()
    print(
        "Standalone collision + DAT vs Newton:",
        "PASS" if overall else "FAIL",
    )

    report = {
        "config": {
            "topology_ring": args.topology_ring,
            "margin": args.margin,
            "gamma": args.gamma,
            "atol": args.atol,
        },
        "cases": results,
        "overall_pass": overall,
    }

    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("report:", report_path)

    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
