"""Small manual benchmark for collision_detection_block_size.

Run from the tools directory, preferably on CUDA:
    CONTACT_TEST_DEVICE=cuda:0 python tests/benchmark_block_dim.py

This is a benchmark, not a pytest correctness test.
"""

from __future__ import annotations

import os
import time
import numpy as np
import warp as wp

from standalone_contact import TriMeshCollisionDetector


def make_two_layer_grid(n=36, gap=0.004):
    xs = np.linspace(0.0, 1.0, n, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, n, dtype=np.float32)

    layer_vertices = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float32)
    layer_faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = a + 1
            c = a + n
            d = c + 1
            layer_faces.append([a, b, d])
            layer_faces.append([a, d, c])
    layer_faces = np.asarray(layer_faces, dtype=np.int32)

    q0 = layer_vertices.copy()
    q1 = layer_vertices.copy()
    q1[:, 2] = gap
    q = np.concatenate((q0, q1), axis=0)
    faces = np.concatenate((layer_faces, layer_faces + len(q0)), axis=0)
    return q, faces


def main():
    device_name = os.environ.get("CONTACT_TEST_DEVICE", "cuda:0")
    device = wp.get_device(device_name)
    q, faces = make_two_layer_grid()
    q_wp = wp.array(q, dtype=wp.vec3, device=device)

    print(f"device={device}, vertices={len(q)}, triangles={len(faces)}")
    if getattr(device, "is_cpu", False):
        print("NOTE: Warp documents block_dim as always 1 on CPU; use CUDA to tune this parameter.")

    for block_dim in (16, 32, 64, 128, 256):
        detector = TriMeshCollisionDetector(
            q_wp,
            faces,
            device=device,
            vertex_collision_buffer_pre_alloc=32,
            edge_collision_buffer_pre_alloc=64,
            collision_detection_block_size=block_dim,
            auto_resize=False,
        )

        # Warm up compilation/caches.
        for _ in range(3):
            detector.detect(q_wp, margin=0.008)
        wp.synchronize()

        repeats = 20
        t0 = time.perf_counter()
        for _ in range(repeats):
            detector.detect(q_wp, margin=0.008)
        wp.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / repeats
        flags = detector.overflow_flags()
        print(f"block_dim={block_dim:3d}: {ms:8.3f} ms/pass, overflow={flags.tolist()}")


if __name__ == "__main__":
    main()
