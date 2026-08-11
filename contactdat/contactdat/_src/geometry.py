# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Standalone adaptation for Warp/NumPy use without importing Newton.
# Modifications Copyright (c) 2026 Standalone Contact contributors.

import warp as wp


TRI_CONTACT_FEATURE_VERTEX_A = wp.constant(0)
TRI_CONTACT_FEATURE_VERTEX_B = wp.constant(1)
TRI_CONTACT_FEATURE_VERTEX_C = wp.constant(2)
TRI_CONTACT_FEATURE_EDGE_AB = wp.constant(3)
TRI_CONTACT_FEATURE_EDGE_AC = wp.constant(4)
TRI_CONTACT_FEATURE_EDGE_BC = wp.constant(5)
TRI_CONTACT_FEATURE_FACE_INTERIOR = wp.constant(6)


@wp.func
def binary_search_i32(values: wp.array[wp.int32], value: int, lower: int, upper: int) -> int:
    """Upper-bound search over sorted values[lower:upper]."""
    while lower < upper:
        mid = (lower + upper) >> 1
        if values[mid] > value:
            upper = mid
        else:
            lower = mid + 1
    return upper


@wp.func
def triangle_closest_point(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    """Closest point on triangle, barycentric coordinates, and feature id."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)

    if d1 <= 0.0 and d2 <= 0.0:
        return a, wp.vec3(1.0, 0.0, 0.0), TRI_CONTACT_FEATURE_VERTEX_A

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, wp.vec3(0.0, 1.0, 0.0), TRI_CONTACT_FEATURE_VERTEX_B

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, wp.vec3(0.0, 0.0, 1.0), TRI_CONTACT_FEATURE_VERTEX_C

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        t = d1 / (d1 - d3)
        return a + t * ab, wp.vec3(1.0 - t, t, 0.0), TRI_CONTACT_FEATURE_EDGE_AB

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        t = d2 / (d2 - d6)
        return a + t * ac, wp.vec3(1.0 - t, 0.0, t), TRI_CONTACT_FEATURE_EDGE_AC

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + t * (c - b), wp.vec3(0.0, 1.0 - t, t), TRI_CONTACT_FEATURE_EDGE_BC

    inv = 1.0 / (va + vb + vc)
    v = vb * inv
    w = vc * inv
    bary = wp.vec3(1.0 - v - w, v, w)
    point = a + v * ab + w * ac
    return point, bary, TRI_CONTACT_FEATURE_FACE_INTERIOR
