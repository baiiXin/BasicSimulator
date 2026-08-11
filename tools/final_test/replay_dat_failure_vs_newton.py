
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import warp as wp
import ipctk


def unique_edges(faces):
    faces = np.asarray(faces, dtype=np.int32)
    e = np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]], axis=0)
    e = np.sort(e, axis=1)
    return np.asfortranarray(np.unique(e, axis=0), dtype=np.int32)


def ipc_hit(V, E, F):
    V = np.asfortranarray(V, dtype=np.float64)
    E = np.asfortranarray(E, dtype=np.int32)
    F = np.asfortranarray(F, dtype=np.int32)
    m = ipctk.CollisionMesh(V, E, F)
    return bool(ipctk.has_intersections(m, V))


def load_faces(sample_root):
    vj = np.load(sample_root/"jacket"/"vertices.npy", mmap_mode="r")
    fj = np.asarray(np.load(sample_root/"jacket"/"faces.npy"), dtype=np.int32)
    ft = np.asarray(np.load(sample_root/"tee_sleeveless"/"faces.npy"), dtype=np.int32)
    F = np.ascontiguousarray(np.vstack([fj, ft + int(vj.shape[1])]), dtype=np.int32)
    return F, unique_edges(F)


def make_newton_ci(NewtonCI, src, device):
    ci = NewtonCI()
    int_fields = [
        "vertex_colliding_triangles",
        "vertex_colliding_triangles_offsets",
        "vertex_colliding_triangles_buffer_sizes",
        "vertex_colliding_triangles_count",
        "triangle_colliding_vertices",
        "triangle_colliding_vertices_offsets",
        "triangle_colliding_vertices_buffer_sizes",
        "triangle_colliding_vertices_count",
        "edge_colliding_edges",
        "edge_colliding_edges_offsets",
        "edge_colliding_edges_buffer_sizes",
        "edge_colliding_edges_count",
    ]
    float_fields = [
        "vertex_colliding_triangles_min_dist",
        "triangle_colliding_vertices_min_dist",
        "edge_colliding_edges_min_dist",
    ]
    for name in int_fields:
        x = getattr(src, name, None)
        if x is None:
            x = wp.empty(0, dtype=wp.int32, device=device)
        setattr(ci, name, x)
    for name in float_fields:
        x = getattr(src, name, None)
        if x is None:
            x = wp.empty(0, dtype=float, device=device)
        setattr(ci, name, x)
    return wp.array([ci], dtype=NewtonCI, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_root", type=Path)
    ap.add_argument("--failure-dir", type=Path, default=Path("."))
    ap.add_argument("--newton-repo", type=Path, default=Path("newton"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.85)
    ap.add_argument("--topology-ring", type=int, default=0)
    args = ap.parse_args()

    tools = Path(__file__).resolve().parent
    sys.path.insert(0, str(tools))
    newton_repo = args.newton_repo.resolve()
    if not (newton_repo/"newton").is_dir():
        raise FileNotFoundError(f"Newton checkout not found: {newton_repo}")
    sys.path.insert(0, str(newton_repo))

    from standalone_contact.collision_detector import TriMeshCollisionDetector
    from standalone_contact.penetration_free_contact import PenetrationFreeContact

    from newton._src.solvers.vbd.particle_vbd_kernels import (
        NUM_THREADS_PER_COLLISION_PRIMITIVE,
        apply_planar_truncation_parallel_by_collision as newton_dat,
        apply_truncation_ts as newton_apply,
    )
    from newton._src.solvers.vbd.tri_mesh_collision import (
        TriMeshCollisionInfo as NewtonCI,
    )

    d = args.failure_dir.resolve()
    q_before64 = np.asarray(np.load(d/"fail_q_before.npy"), dtype=np.float64)
    q_saved64  = np.asarray(np.load(d/"fail_q_after.npy"), dtype=np.float64)
    q_target64 = np.asarray(np.load(d/"fail_q_target.npy"), dtype=np.float64)

    F, E = load_faces(args.sample_root.resolve())

    print("=== saved case ===")
    print("before:", ipc_hit(q_before64, E, F))
    print("saved after:", ipc_hit(q_saved64, E, F))
    print("target:", ipc_hit(q_target64, E, F))

    device = wp.get_device(args.device)
    q_before32 = q_before64.astype(np.float32)
    q_target32 = q_target64.astype(np.float32)
    dx32 = q_target32 - q_before32

    q_before = wp.array(q_before32, dtype=wp.vec3, device=device)
    dx = wp.array(dx32, dtype=wp.vec3, device=device)

    detector = TriMeshCollisionDetector(
        q_before,
        F,
        device=device,
        vertex_collision_buffer_pre_alloc=64,
        vertex_collision_buffer_max_alloc=256,
        edge_collision_buffer_pre_alloc=64,
        edge_collision_buffer_max_alloc=256,
        topological_contact_filter_threshold=args.topology_ring,
        auto_resize=True,
    )

    contact = PenetrationFreeContact(
        detector,
        particle_count=len(q_before32),
        edge_count=detector.edge_count,
        device=device,
        contact_margin=args.margin,
        conservative_bound_relaxation=args.gamma,
        rest_q=None,
        rest_exclusion_radius=0.0,
    )

    contact.begin(q_before)
    wp.synchronize_device(device)

    vt = detector.vertex_colliding_triangles_count.numpy()
    ee = detector.edge_colliding_edges_count.numpy()
    print("\n=== shared candidates ===")
    print("VT total/max:", int(vt.sum()), int(vt.max(initial=0)))
    print("EE total/max:", int(ee.sum()), int(ee.max(initial=0)))

    # Standalone DAT
    s_dx, s_q, s_t = contact.truncate(dx)
    wp.synchronize_device(device)
    s_t_np = np.asarray(s_t.numpy(), dtype=np.float32)
    s_q_np = np.asarray(s_q.numpy(), dtype=np.float32)
    s_dx_np = np.asarray(s_dx.numpy(), dtype=np.float32)

    # Newton DAT, exact same collision buffers
    n_ci = make_newton_ci(NewtonCI, detector.collision_info, device)
    n_t = wp.ones(len(q_before32), dtype=float, device=device)
    n_dx = wp.zeros(len(q_before32), dtype=wp.vec3, device=device)
    n_q = wp.zeros(len(q_before32), dtype=wp.vec3, device=device)

    dim = max(
        len(q_before32) * NUM_THREADS_PER_COLLISION_PRIMITIVE,
        detector.edge_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
    )

    wp.launch(
        kernel=newton_dat,
        dim=dim,
        inputs=[
            q_before, dx,
            detector.tri_indices, detector.edge_indices,
            n_ci,
            float(detector.edge_edge_parallel_epsilon),
            float(args.gamma),
            n_t,
        ],
        device=device,
    )

    max_disp = float(args.margin * args.gamma * 0.5)
    wp.launch(
        kernel=newton_apply,
        dim=len(q_before32),
        inputs=[q_before, dx, n_t, max_disp, n_dx, n_q],
        device=device,
    )
    wp.synchronize_device(device)

    n_t_np = np.asarray(n_t.numpy(), dtype=np.float32)
    n_q_np = np.asarray(n_q.numpy(), dtype=np.float32)
    n_dx_np = np.asarray(n_dx.numpy(), dtype=np.float32)

    td = np.abs(s_t_np - n_t_np)
    qd = np.abs(s_q_np - n_q_np)
    dxd = np.abs(s_dx_np - n_dx_np)

    print("\n=== parity ===")
    print("max |t_s-t_n|:", float(td.max(initial=0.0)))
    print("mean |t_s-t_n|:", float(td.mean()))
    print("count |dt|>1e-6:", int(np.count_nonzero(td > 1e-6)))
    print("max |dx_s-dx_n|:", float(dxd.max(initial=0.0)))
    print("max |q_s-q_n|:", float(qd.max(initial=0.0)))
    print("mean |q_s-q_n|:", float(qd.mean()))

    replay_err = float(np.max(np.abs(s_q_np.astype(np.float64) - q_saved64)))
    print("\n=== IPC ===")
    print("standalone intersects:", ipc_hit(s_q_np.astype(np.float64), E, F))
    print("Newton intersects:", ipc_hit(n_q_np.astype(np.float64), E, F))
    print("saved replay max q error:", replay_err)

    np.save(d/"parity_standalone_t.npy", s_t_np)
    np.save(d/"parity_newton_t.npy", n_t_np)
    np.save(d/"parity_standalone_q.npy", s_q_np)
    np.save(d/"parity_newton_q.npy", n_q_np)


if __name__ == "__main__":
    main()
