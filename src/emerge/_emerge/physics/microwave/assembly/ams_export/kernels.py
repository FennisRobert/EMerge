# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version. See <https://www.gnu.org/licenses/>.
"""Numba kernels for the AMS operator assembly.

Both operators are built by an independent per-tet computation, so the loop is
embarrassingly parallel. A pure-Python loop over tets is far too slow at the
mesh sizes this is meant for -- these kernels run the whole loop under
@njit(parallel=True), with each tet's scratch arrays allocated inside the
prange body so threads do not share state.

Everything called from here is already njit-compiled in EMerge
(`_eval_f_3d`, `_nv`, `_ne`, `_nv_grad`, `_ne_grad`, `tet_coefficients`,
`local_tet_to_edgeid`, `local_tet_to_triid`), so no Python object mode is
involved.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from .....compiled.ccbf import _eval_f_3d
from .....compiled.legrange import _ne, _ne_grad, _nv, _nv_grad
from ..curlcurl import local_tet_to_edgeid, local_tet_to_triid, tet_coefficients

LOCAL_EDGES = np.array([[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]], dtype=np.int64)


@njit(cache=True, inline="always")
def _bary_coeff(verts):
    aas, bbs, ccs, dds, V = tet_coefficients(verts[0], verts[1], verts[2])
    coeff = np.empty((4, 4), dtype=np.float64)
    for m in range(4):
        coeff[0, m] = aas[m] / (6.0 * V)
        coeff[1, m] = bbs[m] / (6.0 * V)
        coeff[2, m] = ccs[m] / (6.0 * V)
        coeff[3, m] = dds[m] / (6.0 * V)
    return coeff, V


@njit(parallel=True, cache=True)
def gradient_kernel(nodes, tets, tris, edges, tet_to_edge, tet_to_tri,
                    tet_to_field, dofcodes, typearr, indexarr, bary,
                    out_vals, out_rows, out_cols, out_res):
    """Local discrete-gradient blocks for every tet.

    grad(P2) is exactly a subspace of Nedelec-2, so each local block is
    recovered by a least-squares fit whose residual must be ~0; that residual
    is returned per tet so the caller can assert it rather than trust it.
    """
    n_tets = tets.shape[1]
    ndof = dofcodes.shape[0]
    npts = bary.shape[1]
    for itet in prange(n_tets):
        verts = np.empty((3, 4), dtype=np.float64)
        for c in range(3):
            for m in range(4):
                verts[c, m] = nodes[c, tets[m, itet]]
        coeff, V = _bary_coeff(verts)
        coords = np.zeros((3, npts), dtype=np.float64)
        for c in range(3):
            for q in range(npts):
                acc = 0.0
                for m in range(4):
                    acc += verts[c, m] * bary[m, q]
                coords[c, q] = acc

        lem = local_tet_to_edgeid(tet_to_edge, tets, edges, itet)
        ltm = local_tet_to_triid(tet_to_tri, tets, tris, itet)

        A = np.empty((3 * npts, ndof), dtype=np.float64)
        buf = np.empty((3, npts), dtype=np.complex128)
        for j in range(ndof):
            idx = indexarr[j]
            if typearr[j] == 0:
                a = lem[0, idx]; b = lem[1, idx]; cc = 0
            else:
                a = ltm[0, idx]; b = ltm[1, idx]; cc = ltm[2, idx]
            _eval_f_3d(coeff, coords, a, b, cc, dofcodes[j], buf)
            for c in range(3):
                for q in range(npts):
                    A[c * npts + q, j] = buf[c, q].real

        R = np.empty((3 * npts, 10), dtype=np.float64)
        for v in range(4):
            g = _nv_grad(coeff, coords, v, v, 0)
            for c in range(3):
                for q in range(npts):
                    R[c * npts + q, v] = g[c, q].real
        for e in range(6):
            g = _ne_grad(coeff, coords, lem[0, e], lem[1, e], 0)
            for c in range(3):
                for q in range(npts):
                    R[c * npts + q, 4 + e] = g[c, q].real

        sol, res, rank, sv = np.linalg.lstsq(A, R)
        fit = A @ sol
        num = 0.0
        den = 0.0
        for p in range(3 * npts):
            for m in range(10):
                d = fit[p, m] - R[p, m]
                num += d * d
                den += R[p, m] * R[p, m]
        out_res[itet] = np.sqrt(num / den) if den > 0.0 else 0.0

        base = itet * ndof * 10
        for j in range(ndof):
            grow = tet_to_field[j, itet]
            for m in range(10):
                gcol = tets[m, itet] if m < 4 else (tet_to_edge[m - 4, itet] + nodes.shape[1])
                k = base + j * 10 + m
                out_rows[k] = grow
                out_cols[k] = gcol
                out_vals[k] = sol[j, m]


@njit(parallel=True, cache=True)
def interpolation_kernel(nodes, tets, tris, edges, tet_to_edge, tet_to_tri,
                         tet_to_field, dofcodes, typearr, indexarr,
                         gs, gws, tb, tw, n_scalar,
                         out_vals, out_rows, out_cols, out_cond):
    """Local Nedelec interpolation blocks Pi_loc = Q^-1 m(phi_a e_d).

    Moments are entity-local (edge integrals with weights 1 and 1-2s; face
    integrals against two in-plane directions), so the resulting DOF values are
    single-valued across tets. cond(Q) is returned per tet so the caller can
    assert unisolvence.
    """
    n_tets = tets.shape[1]
    ndof = dofcodes.shape[0]
    nq_e = gs.shape[0]
    nq_f = tb.shape[0]
    n_nodes = nodes.shape[1]
    for itet in prange(n_tets):
        verts = np.empty((3, 4), dtype=np.float64)
        for c in range(3):
            for m in range(4):
                verts[c, m] = nodes[c, tets[m, itet]]
        coeff, V = _bary_coeff(verts)
        lem = local_tet_to_edgeid(tet_to_edge, tets, edges, itet)
        ltm = local_tet_to_triid(tet_to_tri, tets, tris, itet)

        Q = np.zeros((ndof, ndof), dtype=np.float64)
        Fm = np.zeros((ndof, 3, 10), dtype=np.float64)

        for i in range(ndof):
            if i < 12:                                   # edge moment
                e = i % 6
                wid = i // 6
                npts = nq_e
                pts = np.empty((3, npts), dtype=np.float64)
                wts = np.empty(npts, dtype=np.float64)
                dirv = np.empty(3, dtype=np.float64)
                for c in range(3):
                    dirv[c] = verts[c, lem[1, e]] - verts[c, lem[0, e]]
                for q in range(npts):
                    for c in range(3):
                        pts[c, q] = verts[c, lem[0, e]] + gs[q] * dirv[c]
                    wts[q] = gws[q] * (1.0 if wid == 0 else (1.0 - 2.0 * gs[q]))
            else:                                        # face moment
                f = (i - 12) % 4
                wid = (i - 12) // 4
                npts = nq_f
                pts = np.empty((3, npts), dtype=np.float64)
                wts = np.empty(npts, dtype=np.float64)
                dirv = np.empty(3, dtype=np.float64)
                for c in range(3):
                    p0 = verts[c, ltm[0, f]]
                    dirv[c] = (verts[c, ltm[1, f]] - p0) if wid == 0 else (verts[c, ltm[2, f]] - p0)
                for q in range(npts):
                    for c in range(3):
                        pts[c, q] = (tb[q, 0] * verts[c, ltm[0, f]]
                                     + tb[q, 1] * verts[c, ltm[1, f]]
                                     + tb[q, 2] * verts[c, ltm[2, f]])
                    wts[q] = tw[q]

            buf = np.empty((3, npts), dtype=np.complex128)
            for j in range(ndof):
                idx = indexarr[j]
                if typearr[j] == 0:
                    a = lem[0, idx]; b = lem[1, idx]; cc = 0
                else:
                    a = ltm[0, idx]; b = ltm[1, idx]; cc = ltm[2, idx]
                _eval_f_3d(coeff, pts, a, b, cc, dofcodes[j], buf)
                acc = 0.0
                for q in range(npts):
                    s = 0.0
                    for c in range(3):
                        s += buf[c, q].real * dirv[c]
                    acc += wts[q] * s
                Q[i, j] = acc

            for v in range(4):
                sc = _nv(coeff, pts, v, v, 0)
                acc = 0.0
                for q in range(npts):
                    acc += wts[q] * sc[q].real
                for d in range(3):
                    Fm[i, d, v] = acc * dirv[d]
            for e2 in range(6):
                sc = _ne(coeff, pts, lem[0, e2], lem[1, e2], 0)
                acc = 0.0
                for q in range(npts):
                    acc += wts[q] * sc[q].real
                for d in range(3):
                    Fm[i, d, 4 + e2] = acc * dirv[d]

        out_cond[itet] = np.linalg.cond(Q)
        Pi_loc = np.linalg.solve(Q, Fm.reshape(ndof, 30))

        base = itet * ndof * 30
        for j in range(ndof):
            grow = tet_to_field[j, itet]
            for d in range(3):
                for m in range(10):
                    gcol = (tets[m, itet] if m < 4 else (tet_to_edge[m - 4, itet] + n_nodes)) + d * n_scalar
                    k = base + j * 30 + d * 10 + m
                    out_rows[k] = grow
                    out_cols[k] = gcol
                    out_vals[k] = Pi_loc[j, d * 10 + m]
