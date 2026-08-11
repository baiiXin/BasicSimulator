
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp
import ipctk

from standalone_contact.collision_detector import TriMeshCollisionDetector
from standalone_contact.penetration_free_contact import PenetrationFreeContact


# -----------------------------------------------------------------------------
# Mesh / IPC helpers
# -----------------------------------------------------------------------------

def unique_edges_from_faces(faces):
    faces = np.asarray(faces, dtype=np.int32)
    e = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    e = np.sort(e, axis=1)
    return np.asfortranarray(np.unique(e, axis=0), dtype=np.int32)


def load_faces(sample_root: Path):
    vj = np.load(sample_root / "jacket" / "vertices.npy", mmap_mode="r")
    fj = np.asarray(np.load(sample_root / "jacket" / "faces.npy"), dtype=np.int32)
    ft = np.asarray(np.load(sample_root / "tee_sleeveless" / "faces.npy"), dtype=np.int32)
    nj = int(vj.shape[1])
    F = np.ascontiguousarray(np.vstack([fj, ft + nj]), dtype=np.int32)
    E = unique_edges_from_faces(F)
    return F, E


def ipc_hit(V, E, F):
    V = np.asfortranarray(V, dtype=np.float64)
    E = np.asfortranarray(E, dtype=np.int32)
    F = np.asfortranarray(F, dtype=np.int32)
    mesh = ipctk.CollisionMesh(V, E, F)
    return bool(ipctk.has_intersections(mesh, V))


# -----------------------------------------------------------------------------
# Candidate extraction from the production detector
# -----------------------------------------------------------------------------

def extract_vt(detector):
    buf = np.asarray(detector.vertex_colliding_triangles.numpy(), dtype=np.int32)
    offs = np.asarray(detector.vertex_colliding_triangles_offsets.numpy(), dtype=np.int32)
    sizes = np.asarray(detector.vertex_colliding_triangles_buffer_sizes.numpy(), dtype=np.int32)
    counts = np.asarray(detector.vertex_colliding_triangles_count.numpy(), dtype=np.int32)

    pairs = []
    for v in range(len(counts)):
        n = min(int(counts[v]), int(sizes[v]))
        off = int(offs[v])
        for j in range(n):
            k = 2 * (off + j)
            vv = int(buf[k])
            tt = int(buf[k + 1])
            if vv >= 0 and tt >= 0:
                pairs.append((vv, tt))
    return pairs


def extract_ee(detector):
    buf = np.asarray(detector.edge_colliding_edges.numpy(), dtype=np.int32)
    offs = np.asarray(detector.edge_colliding_edges_offsets.numpy(), dtype=np.int32)
    sizes = np.asarray(detector.edge_colliding_edges_buffer_sizes.numpy(), dtype=np.int32)
    counts = np.asarray(detector.edge_colliding_edges_count.numpy(), dtype=np.int32)

    pairs = []
    for e0 in range(len(counts)):
        n = min(int(counts[e0]), int(sizes[e0]))
        off = int(offs[e0])
        for j in range(n):
            k = 2 * (off + j)
            ee0 = int(buf[k])
            ee1 = int(buf[k + 1])
            if ee0 >= 0 and ee1 >= 0:
                pairs.append((ee0, ee1))
    return pairs


# -----------------------------------------------------------------------------
# Scalar arithmetic helpers.
# The same Python implementation is run once with float32 and once with float64.
# -----------------------------------------------------------------------------

def dot(a, b, dtype):
    return dtype(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def norm(v, dtype):
    return dtype(np.sqrt(dot(v, v, dtype), dtype=dtype))


def normalize(v, dtype):
    l = norm(v, dtype)
    if l == dtype(0.0):
        return np.zeros(3, dtype=dtype)
    return np.asarray(v / l, dtype=dtype)


def cross(a, b, dtype):
    return np.asarray(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=dtype,
    )


def clamp(x, lo, hi, dtype):
    return dtype(min(max(x, dtype(lo)), dtype(hi)))


# Ericson / Warp-style triangle closest-point region tests.
def triangle_closest_point(a, b, c, p, dtype):
    a = np.asarray(a, dtype=dtype)
    b = np.asarray(b, dtype=dtype)
    c = np.asarray(c, dtype=dtype)
    p = np.asarray(p, dtype=dtype)

    ab = np.asarray(b - a, dtype=dtype)
    ac = np.asarray(c - a, dtype=dtype)
    ap = np.asarray(p - a, dtype=dtype)

    d1 = dot(ab, ap, dtype)
    d2 = dot(ac, ap, dtype)
    z = dtype(0.0)
    o = dtype(1.0)

    if d1 <= z and d2 <= z:
        return a.copy()

    bp = np.asarray(p - b, dtype=dtype)
    d3 = dot(ab, bp, dtype)
    d4 = dot(ac, bp, dtype)
    if d3 >= z and d4 <= d3:
        return b.copy()

    vc = dtype(d1 * d4 - d3 * d2)
    if vc <= z and d1 >= z and d3 <= z:
        v = dtype(d1 / (d1 - d3))
        return np.asarray(a + v * ab, dtype=dtype)

    cp = np.asarray(p - c, dtype=dtype)
    d5 = dot(ab, cp, dtype)
    d6 = dot(ac, cp, dtype)
    if d6 >= z and d5 <= d6:
        return c.copy()

    vb = dtype(d5 * d2 - d1 * d6)
    if vb <= z and d2 >= z and d6 <= z:
        w = dtype(d2 / (d2 - d6))
        return np.asarray(a + w * ac, dtype=dtype)

    va = dtype(d3 * d6 - d5 * d4)
    if va <= z and (d4 - d3) >= z and (d5 - d6) >= z:
        w = dtype((d4 - d3) / ((d4 - d3) + (d5 - d6)))
        return np.asarray(b + w * (c - b), dtype=dtype)

    denom = dtype(o / (va + vb + vc))
    v = dtype(vb * denom)
    w = dtype(vc * denom)
    return np.asarray(a + ab * v + ac * w, dtype=dtype)


# Segment-segment closest point.
# This is intentionally the SAME code for float32 and float64.
def closest_point_edge_edge(p1, q1, p2, q2, dtype, parallel_eps=1.0e-6):
    p1 = np.asarray(p1, dtype=dtype)
    q1 = np.asarray(q1, dtype=dtype)
    p2 = np.asarray(p2, dtype=dtype)
    q2 = np.asarray(q2, dtype=dtype)

    d1 = np.asarray(q1 - p1, dtype=dtype)
    d2 = np.asarray(q2 - p2, dtype=dtype)
    r = np.asarray(p1 - p2, dtype=dtype)

    a = dot(d1, d1, dtype)
    e = dot(d2, d2, dtype)
    f = dot(d2, r, dtype)

    eps = dtype(parallel_eps)
    z = dtype(0.0)
    o = dtype(1.0)

    if a <= eps and e <= eps:
        return z, z

    if a <= eps:
        s = z
        t = clamp(f / e, 0.0, 1.0, dtype) if e > eps else z
        return s, t

    c = dot(d1, r, dtype)

    if e <= eps:
        t = z
        s = clamp(-c / a, 0.0, 1.0, dtype)
        return s, t

    b = dot(d1, d2, dtype)
    denom = dtype(a * e - b * b)

    # Parallel / nearly parallel.
    if abs(float(denom)) > float(eps):
        s = clamp((b * f - c * e) / denom, 0.0, 1.0, dtype)
    else:
        s = z

    tnom = dtype(b * s + f)

    if tnom < z:
        t = z
        s = clamp(-c / a, 0.0, 1.0, dtype)
    elif tnom > e:
        t = o
        s = clamp((b - c) / a, 0.0, 1.0, dtype)
    else:
        t = dtype(tnom / e)

    return s, t


def robust_edge_pair_normal(p0, q0, p1, q1, dtype, eps=1.0e-6):
    eps = dtype(eps)
    dir0 = np.asarray(q0 - p0, dtype=dtype)
    dir1 = np.asarray(q1 - p1, dtype=dtype)

    len0 = norm(dir0, dtype)
    len1 = norm(dir1, dtype)

    if len0 > eps:
        dir0 = np.asarray(dir0 / len0, dtype=dtype)
    else:
        dir0 = np.zeros(3, dtype=dtype)

    if len1 > eps:
        dir1 = np.asarray(dir1 / len1, dtype=dtype)
    else:
        dir1 = np.zeros(3, dtype=dtype)

    n = cross(dir0, dir1, dtype)
    ln = norm(n, dtype)
    if ln > eps:
        return np.asarray(n / ln, dtype=dtype)

    reference = dir0
    if norm(reference, dtype) <= eps:
        reference = dir1

    if norm(reference, dtype) <= eps:
        return np.asarray([1.0, 0.0, 0.0], dtype=dtype)

    bridge = np.asarray(
        dtype(0.5) * ((p1 + q1) - (p0 + q0)),
        dtype=dtype,
    )
    bl = norm(bridge, dtype)
    if bl > eps:
        n = cross(reference, np.asarray(bridge / bl, dtype=dtype), dtype)
        ln = norm(n, dtype)
        if ln > eps:
            return np.asarray(n / ln, dtype=dtype)

    axis = np.asarray([1.0, 0.0, 0.0], dtype=dtype)
    if abs(float(dot(reference, axis, dtype))) > 0.9:
        axis = np.asarray([0.0, 1.0, 0.0], dtype=dtype)

    n = cross(reference, axis, dtype)
    ln = norm(n, dtype)
    if ln > eps:
        return np.asarray(n / ln, dtype=dtype)

    axis = np.asarray([0.0, 0.0, 1.0], dtype=dtype)
    n = cross(reference, axis, dtype)
    ln = norm(n, dtype)
    if ln > eps:
        return np.asarray(n / ln, dtype=dtype)

    return np.asarray([1.0, 0.0, 0.0], dtype=dtype)


def segment_plane_intersects(
    v,
    dv,
    n,
    d,
    dtype,
    eps_parallel,
    eps_near,
    eps_far,
):
    nv = dot(n, dv, dtype)
    num = dtype(-dot(n, np.asarray(v - d, dtype=dtype), dtype))

    if abs(float(nv)) < eps_parallel:
        return False

    t = dtype(num / nv)
    return bool(t >= dtype(eps_near) and t <= dtype(1.0 + eps_far))


def planar_t(v, dv, n, d, dtype, parallel_eps, gamma, gamma_min=1.0e-3):
    denom = dot(n, dv, dtype)

    if abs(float(denom)) < parallel_eps:
        return dtype(1.0)

    t = dtype(dot(n, np.asarray(d - v, dtype=dtype), dtype) / denom)

    if t < dtype(0.0):
        return dtype(1.0)

    t = min(dtype(t * dtype(gamma)), dtype(t - dtype(gamma_min)))
    return clamp(t, 0.0, 1.0, dtype)


def update_t(ts, vertex, t):
    if t < ts[vertex]:
        ts[vertex] = t


def dat_reference(
    q,
    dx,
    tri_indices,
    edge_indices,
    vt_pairs,
    ee_pairs,
    *,
    dtype,
    gamma,
    parallel_eps,
    margin,
):
    q = np.asarray(q, dtype=dtype)
    dx = np.asarray(dx, dtype=dtype)
    tri_indices = np.asarray(tri_indices, dtype=np.int32)
    edge_indices = np.asarray(edge_indices, dtype=np.int32)

    ts = np.ones(len(q), dtype=dtype)

    # ------------------------------------------------------------------
    # EE candidates
    # ------------------------------------------------------------------
    for e0, e1 in ee_pairs:
        a0 = int(edge_indices[e0, 2])
        a1 = int(edge_indices[e0, 3])
        b0 = int(edge_indices[e1, 2])
        b1 = int(edge_indices[e1, 3])

        p0, p1 = q[a0], q[a1]
        q0, q1 = q[b0], q[b1]

        dp0, dp1 = dx[a0], dx[a1]
        dq0, dq1 = dx[b0], dx[b1]

        s, t = closest_point_edge_edge(
            p0, p1, q0, q1, dtype, parallel_eps=1.0e-6
        )

        c1 = np.asarray(p0 + (p1 - p0) * s, dtype=dtype)
        c2 = np.asarray(q0 + (q1 - q0) * t, dtype=dtype)

        n_hat = np.asarray(c1 - c2, dtype=dtype)
        nh_len = norm(n_hat, dtype)

        if nh_len < dtype(1.0e-12):
            n = robust_edge_pair_normal(p0, p1, q0, q1, dtype)
            d = np.asarray(dtype(0.5) * c1 + dtype(0.5) * c2, dtype=dtype)
        else:
            n = np.asarray(n_hat / nh_len, dtype=dtype)

            de0 = max(
                dtype(-dot(n, dp0, dtype)),
                dtype(-dot(n, dp1, dtype)),
                dtype(0.0),
            )
            de1 = max(
                dtype(dot(n, dq0, dtype)),
                dtype(dot(n, dq1, dtype)),
                dtype(0.0),
            )

            if de0 + de1 == dtype(0.0):
                d = np.asarray(c2 + dtype(0.5) * n_hat, dtype=dtype)
            else:
                lmbd = dtype(de1 / (de1 + de0))
                lmbd = clamp(lmbd, 0.05, 0.95, dtype)
                d = np.asarray(c2 + lmbd * n_hat, dtype=dtype)

            if de0 != dtype(0.0):
                if segment_plane_intersects(
                    p0, dp0, n, d, dtype, 1e-6, -1e-8, 1e-6
                ):
                    update_t(ts, a0, planar_t(p0, dp0, n, d, dtype, parallel_eps, gamma))
                if segment_plane_intersects(
                    p1, dp1, n, d, dtype, 1e-6, -1e-8, 1e-6
                ):
                    update_t(ts, a1, planar_t(p1, dp1, n, d, dtype, parallel_eps, gamma))

            if de1 != dtype(0.0):
                if segment_plane_intersects(
                    q0, dq0, n, d, dtype, 1e-6, -1e-8, 1e-6
                ):
                    update_t(ts, b0, planar_t(q0, dq0, n, d, dtype, parallel_eps, gamma))
                if segment_plane_intersects(
                    q1, dq1, n, d, dtype, 1e-6, -1e-8, 1e-6
                ):
                    update_t(ts, b1, planar_t(q1, dq1, n, d, dtype, parallel_eps, gamma))

    # ------------------------------------------------------------------
    # VT candidates
    # ------------------------------------------------------------------
    for v_idx, tri_idx in vt_pairs:
        ia, ib, ic = [int(x) for x in tri_indices[tri_idx]]

        v = q[v_idx]
        dv = dx[v_idx]
        a, b, c = q[ia], q[ib], q[ic]
        da, db, dc = dx[ia], dx[ib], dx[ic]

        closest = triangle_closest_point(a, b, c, v, dtype)
        n_hat = np.asarray(v - closest, dtype=dtype)
        nh_len = norm(n_hat, dtype)

        if nh_len < dtype(1.0e-12):
            # Newton/Warp returns four non-dummy flags but n=0, d=v.
            n = np.zeros(3, dtype=dtype)
            d = v.copy()
            for vi, px, ddx in (
                (v_idx, v, dv),
                (ia, a, da),
                (ib, b, db),
                (ic, c, dc),
            ):
                update_t(
                    ts,
                    vi,
                    planar_t(px, ddx, n, d, dtype, parallel_eps, gamma),
                )
            continue

        n = np.asarray(n_hat / nh_len, dtype=dtype)

        dvn = max(dtype(-dot(n, dv, dtype)), dtype(0.0))
        dtn = max(
            dtype(dot(n, da, dtype)),
            dtype(dot(n, db, dtype)),
            dtype(dot(n, dc, dtype)),
            dtype(0.0),
        )

        if dtn + dvn == dtype(0.0):
            d = np.asarray(closest + dtype(0.5) * n_hat, dtype=dtype)
        else:
            lmbd = dtype(dtn / (dtn + dvn))
            lmbd = clamp(lmbd, 0.05, 0.95, dtype)
            d = np.asarray(closest + lmbd * n_hat, dtype=dtype)

        if dvn != dtype(0.0):
            if segment_plane_intersects(
                v, dv, n, d, dtype, 1e-6, -1e-8, 1e-8
            ):
                update_t(ts, v_idx, planar_t(v, dv, n, d, dtype, parallel_eps, gamma))

        if dtn != dtype(0.0):
            for vi, px, ddx in ((ia, a, da), (ib, b, db), (ic, c, dc)):
                if segment_plane_intersects(
                    px, ddx, n, d, dtype, 1e-6, -1e-8, 1e-8
                ):
                    update_t(
                        ts,
                        vi,
                        planar_t(px, ddx, n, d, dtype, parallel_eps, gamma),
                    )

    # Final Newton magnitude cap.
    disp = np.asarray(dx * ts[:, None], dtype=dtype)
    max_disp = dtype(margin * gamma * 0.5)

    lengths = np.sqrt(np.sum(disp * disp, axis=1, dtype=dtype), dtype=dtype)
    mask = lengths > max_disp
    if np.any(mask):
        disp[mask] = np.asarray(
            disp[mask] * (max_disp / lengths[mask])[:, None],
            dtype=dtype,
        )

    q_out = np.asarray(q + disp, dtype=dtype)
    return ts, disp, q_out


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Test-only float32 vs float64 DAT arithmetic on the saved "
            "frame-63 micro-15 failure. Production code is not modified."
        )
    )
    ap.add_argument("sample_root", type=Path)
    ap.add_argument("--failure-dir", type=Path, default=Path("."))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.85)
    ap.add_argument("--topology-ring", type=int, default=2)
    ap.add_argument("--vt-buffer", type=int, default=64)
    ap.add_argument("--ee-buffer", type=int, default=64)
    args = ap.parse_args()

    d = args.failure_dir.resolve()
    sample_root = args.sample_root.resolve()

    q_before64 = np.asarray(np.load(d / "fail_q_before.npy"), dtype=np.float64)
    q_saved_after64 = np.asarray(np.load(d / "fail_q_after.npy"), dtype=np.float64)
    q_target64 = np.asarray(np.load(d / "fail_q_target.npy"), dtype=np.float64)

    F, E_ipc = load_faces(sample_root)

    device = wp.get_device(args.device)

    q_before32 = q_before64.astype(np.float32)
    q_target32 = q_target64.astype(np.float32)

    q_wp = wp.array(q_before32, dtype=wp.vec3, device=device)

    detector = TriMeshCollisionDetector(
        q_wp,
        F,
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
        particle_count=len(q_before32),
        edge_count=detector.edge_count,
        device=device,
        contact_margin=args.margin,
        conservative_bound_relaxation=args.gamma,
        rest_q=None,
        rest_exclusion_radius=0.0,
    )

    contact.begin(q_wp)
    wp.synchronize_device(device)

    vt_pairs = extract_vt(detector)
    ee_pairs = extract_ee(detector)

    tri_indices = np.asarray(detector.tri_indices.numpy(), dtype=np.int32)
    edge_indices = np.asarray(detector.edge_indices.numpy(), dtype=np.int32)

    # Production Warp float32 baseline.
    dx_wp = wp.array(
        q_target32 - q_before32,
        dtype=wp.vec3,
        device=device,
    )
    warp_dx, warp_q, warp_t = contact.truncate(dx_wp)
    wp.synchronize_device(device)

    warp_q32 = np.asarray(warp_q.numpy(), dtype=np.float32)
    warp_t32 = np.asarray(warp_t.numpy(), dtype=np.float32)

    print("=== saved / production baseline ===")
    print("before intersects:       ", ipc_hit(q_before64, E_ipc, F))
    print("saved after intersects:  ", ipc_hit(q_saved_after64, E_ipc, F))
    print("Warp f32 intersects:     ", ipc_hit(warp_q32.astype(np.float64), E_ipc, F))
    print("saved replay max q error:", float(np.max(np.abs(warp_q32.astype(np.float64) - q_saved_after64))))
    print("VT candidates:", len(vt_pairs))
    print("EE candidates:", len(ee_pairs))

    # Same CPU reference implementation in f32 and f64.
    print("\nRunning CPU float32 reference...")
    ref32_t, ref32_dx, ref32_q = dat_reference(
        q_before32,
        q_target32 - q_before32,
        tri_indices,
        edge_indices,
        vt_pairs,
        ee_pairs,
        dtype=np.float32,
        gamma=args.gamma,
        parallel_eps=float(detector.edge_edge_parallel_epsilon),
        margin=args.margin,
    )

    print("Running CPU float64 reference...")
    ref64_t, ref64_dx, ref64_q = dat_reference(
        q_before64,
        q_target64 - q_before64,
        tri_indices,
        edge_indices,
        vt_pairs,
        ee_pairs,
        dtype=np.float64,
        gamma=args.gamma,
        parallel_eps=float(detector.edge_edge_parallel_epsilon),
        margin=args.margin,
    )

    ref32_hit = ipc_hit(ref32_q.astype(np.float64), E_ipc, F)
    ref64_hit = ipc_hit(ref64_q, E_ipc, F)

    q32_err = float(np.max(np.abs(ref32_q.astype(np.float64) - warp_q32.astype(np.float64))))
    t32_err = float(np.max(np.abs(ref32_t.astype(np.float64) - warp_t32.astype(np.float64))))

    t64_vs_32 = np.abs(ref64_t - ref32_t.astype(np.float64))
    q64_vs_32 = np.abs(ref64_q - ref32_q.astype(np.float64))

    print("\n=== reference validation ===")
    print("CPU ref f32 intersects:    ", ref32_hit)
    print("max |q_ref32 - q_Warp|:    ", q32_err)
    print("max |t_ref32 - t_Warp|:    ", t32_err)

    print("\n=== float64 experiment ===")
    print("CPU ref f64 intersects:    ", ref64_hit)
    print("max |t64 - t32|:           ", float(t64_vs_32.max(initial=0.0)))
    print("mean |t64 - t32|:          ", float(t64_vs_32.mean()))
    print("vertices |dt| > 1e-6:      ", int(np.count_nonzero(t64_vs_32 > 1e-6)))
    print("max |q64 - q32|:           ", float(q64_vs_32.max(initial=0.0)))
    print("min t ref32 / ref64:       ", float(ref32_t.min()), float(ref64_t.min()))

    # Save for later inspection.
    np.save(d / "micro15_reference_f32_t.npy", ref32_t)
    np.save(d / "micro15_reference_f64_t.npy", ref64_t)
    np.save(d / "micro15_reference_f32_q.npy", ref32_q)
    np.save(d / "micro15_reference_f64_q.npy", ref64_q)

    print("\n=== interpretation gate ===")
    if not ref32_hit:
        print(
            "INCONCLUSIVE: the test-only float32 reference did not reproduce "
            "the production failure, so do not attribute any f64 difference "
            "to precision alone."
        )
    elif ref64_hit:
        print(
            "RESULT: both the same-reference f32 and f64 paths intersect. "
            "This does NOT support float32 precision as the sole cause."
        )
    else:
        print(
            "RESULT: same-reference f32 intersects but f64 does not. "
            "This strongly supports finite-precision arithmetic in DAT "
            "geometry as the trigger for this micro-step failure."
        )


if __name__ == "__main__":
    main()
