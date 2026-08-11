"""Regression for the validated frame-63 micro-step behavior.

Passing means that ContactDAT preserves the observed Newton-equivalent result:
micro-steps 1 through 14 are non-intersecting and micro-step 15 intersects.
This is not a claim that DAT provides strict CCD guarantees.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ipctk
import numpy as np
import warp as wp

from contactdat import PenetrationFreeContact, TriMeshCollisionDetector


EXPECTED_FIRST_INTERSECTION = 15


def unique_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
        axis=0,
    )
    return np.asfortranarray(
        np.unique(np.sort(edges, axis=1), axis=0),
        dtype=np.int32,
    )


def load_combined_faces(raw_root: Path) -> np.ndarray:
    jacket_vertices = np.load(
        raw_root / "jacket" / "vertices.npy",
        mmap_mode="r",
    )
    jacket_faces = np.asarray(
        np.load(raw_root / "jacket" / "faces.npy"),
        dtype=np.int32,
    )
    tee_faces = np.asarray(
        np.load(raw_root / "tee_sleeveless" / "faces.npy"),
        dtype=np.int32,
    )
    jacket_vertex_count = int(jacket_vertices.shape[1])
    return np.ascontiguousarray(
        np.vstack((jacket_faces, tee_faces + jacket_vertex_count)),
        dtype=np.int32,
    )


def load_combined_frame(root: Path, frame: int) -> np.ndarray:
    jacket_vertices = np.load(
        root / "jacket" / "vertices.npy",
        mmap_mode="r",
    )
    tee_vertices = np.load(
        root / "tee_sleeveless" / "vertices.npy",
        mmap_mode="r",
    )
    return np.ascontiguousarray(
        np.vstack(
            (
                np.asarray(jacket_vertices[frame], dtype=np.float64),
                np.asarray(tee_vertices[frame], dtype=np.float64),
            )
        ),
        dtype=np.float64,
    )


def has_intersections(
    vertices: np.ndarray,
    edges: np.ndarray,
    faces: np.ndarray,
) -> bool:
    vertices_f64 = np.asfortranarray(vertices, dtype=np.float64)
    mesh = ipctk.CollisionMesh(
        vertices_f64,
        np.asfortranarray(edges, dtype=np.int32),
        np.asfortranarray(faces, dtype=np.int32),
    )
    return bool(ipctk.has_intersections(mesh, vertices_f64))


def run_regression(
    raw_root: Path,
    processed_root: Path,
    *,
    source_frame: int = 62,
    target_frame: int = 63,
    device_name: str = "cuda:0",
    margin: float = 0.05,
    gamma: float = 0.85,
    topology_ring: int = 2,
    vt_buffer: int = 64,
    ee_buffer: int = 64,
) -> int:
    faces = load_combined_faces(raw_root)
    edges = unique_edges_from_faces(faces)

    q_start64 = load_combined_frame(processed_root, source_frame)
    q_target64 = load_combined_frame(raw_root, target_frame)

    if has_intersections(q_start64, edges, faces):
        raise AssertionError(
            f"Processed source frame {source_frame} is already intersecting."
        )

    device = wp.get_device(device_name)
    q_target32 = np.asarray(q_target64, dtype=np.float32)
    q_current = wp.array(
        np.asarray(q_start64, dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )

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
        particle_count=len(q_start64),
        edge_count=detector.edge_count,
        device=device,
        contact_margin=margin,
        conservative_bound_relaxation=gamma,
        rest_q=None,
        rest_exclusion_radius=0.0,
    )

    first_intersection = None
    for micro_step in range(1, EXPECTED_FIRST_INTERSECTION + 1):
        contact.begin(q_current)
        q_current32 = np.asarray(q_current.numpy(), dtype=np.float32)
        displacement = wp.array(
            q_target32 - q_current32,
            dtype=wp.vec3,
            device=device,
        )
        _, q_next, truncation_t = contact.truncate(displacement)
        wp.synchronize_device(device)

        q_next64 = np.asarray(q_next.numpy(), dtype=np.float64)
        intersects = has_intersections(q_next64, edges, faces)
        min_t = float(np.asarray(truncation_t.numpy()).min(initial=1.0))
        print(
            f"micro={micro_step:02d} "
            f"intersects={intersects} "
            f"min_t={min_t:.9g}"
        )

        if intersects:
            first_intersection = micro_step
            break
        q_current = q_next

    if first_intersection != EXPECTED_FIRST_INTERSECTION:
        raise AssertionError(
            "Unexpected frame-63 behavior: expected the first intersection "
            f"at micro-step {EXPECTED_FIRST_INTERSECTION}, got "
            f"{first_intersection!r}."
        )

    print(
        "PASS: micro-steps 1-14 are non-intersecting and micro-step 15 "
        "intersects."
    )
    return first_intersection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("processed_root", type=Path)
    parser.add_argument("--source-frame", type=int, default=62)
    parser.add_argument("--target-frame", type=int, default=63)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--topology-ring", type=int, default=2)
    parser.add_argument("--vt-buffer", type=int, default=64)
    parser.add_argument("--ee-buffer", type=int, default=64)
    args = parser.parse_args()

    run_regression(
        args.raw_root.resolve(),
        args.processed_root.resolve(),
        source_frame=args.source_frame,
        target_frame=args.target_frame,
        device_name=args.device,
        margin=args.margin,
        gamma=args.gamma,
        topology_ring=args.topology_ring,
        vt_buffer=args.vt_buffer,
        ee_buffer=args.ee_buffer,
    )


if __name__ == "__main__":
    main()
