
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import warp as wp

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from standalone_contact.dat_kernels import (
    apply_truncation_ts,
    planar_truncation_t,
    robust_edge_pair_normal,
)


def _device():
    return os.environ.get("CONTACT_TEST_DEVICE", "cpu")


@wp.kernel
def _eval_planar_t(
    v: wp.array[wp.vec3],
    dv: wp.array[wp.vec3],
    n: wp.array[wp.vec3],
    d: wp.array[wp.vec3],
    eps: float,
    gamma: float,
    out: wp.array[float],
):
    i = wp.tid()
    out[i] = planar_truncation_t(v[i], dv[i], n[i], d[i], eps, gamma)


@wp.kernel
def _eval_robust_normal(
    p: wp.array[wp.vec3],
    out: wp.array[wp.vec3],
):
    out[0] = robust_edge_pair_normal(p[0], p[1], p[2], p[3])


def test_planar_truncation_t_expected():
    device = _device()

    v = wp.array([[0.0, 0.0, 0.2]], dtype=wp.vec3, device=device)
    dv = wp.array([[0.0, 0.0, -0.3]], dtype=wp.vec3, device=device)
    n = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3, device=device)
    d = wp.array([[0.0, 0.0, 0.01]], dtype=wp.vec3, device=device)
    out = wp.empty(1, dtype=float, device=device)

    wp.launch(
        _eval_planar_t,
        dim=1,
        inputs=[v, dv, n, d, 1.0e-5, 0.85, out],
        device=device,
    )

    got = float(out.numpy()[0])
    t_hit = 0.19 / 0.3
    expected = min(t_hit * 0.85, t_hit - 1.0e-3)

    assert np.isclose(got, expected, atol=1.0e-6), (got, expected)


def test_planar_truncation_parallel_means_no_cut():
    device = _device()

    v = wp.array([[0.0, 0.0, 0.2]], dtype=wp.vec3, device=device)
    dv = wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    n = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3, device=device)
    d = wp.array([[0.0, 0.0, 0.01]], dtype=wp.vec3, device=device)
    out = wp.empty(1, dtype=float, device=device)

    wp.launch(
        _eval_planar_t,
        dim=1,
        inputs=[v, dv, n, d, 1.0e-5, 0.85, out],
        device=device,
    )

    assert np.isclose(float(out.numpy()[0]), 1.0)


def test_robust_normal_parallel_edges_is_finite_unit_vector():
    device = _device()

    p = wp.array(
        [
            [-1.0, 0.0, 0.1],
            [ 1.0, 0.0, 0.1],
            [-1.0, 0.0, 0.0],
            [ 1.0, 0.0, 0.0],
        ],
        dtype=wp.vec3,
        device=device,
    )
    out = wp.empty(1, dtype=wp.vec3, device=device)

    wp.launch(_eval_robust_normal, dim=1, inputs=[p, out], device=device)
    normal = out.numpy()[0]

    assert np.isfinite(normal).all()
    assert np.isclose(np.linalg.norm(normal), 1.0, atol=1.0e-5)


def test_apply_truncation_ts_final_magnitude_clamp():
    device = _device()

    pos = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    dx = wp.array([[0.0, 0.0, 2.0]], dtype=wp.vec3, device=device)
    ts = wp.array([0.5], dtype=float, device=device)

    dx_out = wp.empty_like(dx)
    q_out = wp.empty_like(pos)

    wp.launch(
        apply_truncation_ts,
        dim=1,
        inputs=[pos, dx, ts, 0.25, dx_out, q_out],
        device=device,
    )

    got_dx = dx_out.numpy()[0]
    got_q = q_out.numpy()[0]

    assert np.isclose(np.linalg.norm(got_dx), 0.25, atol=1.0e-6)
    np.testing.assert_allclose(got_q, got_dx, atol=1.0e-6)
