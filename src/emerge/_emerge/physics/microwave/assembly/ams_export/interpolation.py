# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version. See <https://www.gnu.org/licenses/>.
"""Nedelec-2 interpolation operator Pi : (Legrange2)^3 -> Nedelec2.

Column `d*n_scalar + a` holds the Nedelec-2 coefficients of the vector field
`phi_a * e_d`, where `phi_a` is the a-th Legrange2 scalar basis function and
`e_d` the unit vector along axis d.

This is the second operator an auxiliary-space Maxwell preconditioner needs
from the assembler (hypre: `HYPRE_AMSSetInterpolations`). Together with the
discrete gradient it realises the Hiptmair-Xu decomposition
`v = grad(p) + Pi(w) + smooth`: the gradient term covers the curl-free part,
Pi covers the rest, and only what neither reaches is left to the smoother.

## Why this needs dual functionals, unlike the discrete gradient

`grad(P2)` is EXACTLY a subspace of Nedelec-2, so `discrete_gradient.py` can
recover each column by a least-squares fit and assert a ~1e-15 residual.
`phi_a * e_d` is a degree-2 VECTOR polynomial and Nedelec-2 (first kind) does
not contain all of those, so there is nothing to fit exactly -- an
interpolation operator has to be applied instead.

## Constructing the duals without knowing them analytically

EMerge defines this element by its basis functions (see `elements/dofsets.py`
and the sympy generator that produced `compiled/ccbf.py`), not by a dual
basis. The duals are nonetheless uniquely determined: choose any 20 linearly
independent, ENTITY-LOCAL functionals `m_i`, form `Q[i][j] = m_i(N_j)`, and
the dual basis is `ell = Q^-1 m`. The interpolant of any field `u` is then
simply `Q^-1 m(u)`.

The moments used here are

    edge (i,j), 2 per edge:  integral over the edge of  (u . (p_j - p_i)) * w,
                             with w = 1 and w = (L_i - L_j) = (1 - 2s)
    face (i,j,k), 2 per face: integral over the face of  u . (p_j - p_i)
                              and                        u . (p_k - p_i)

Measured: `cond(Q) = 23.4`, and identically so on every tet tried (it is
independent of the element geometry), with `Q^-1 Q = I` to 1.3e-15. So the
choice is unisolvent for this basis, which is asserted at build time rather
than assumed.

## Why the result is single-valued

Every moment is an integral over its OWN edge or face, taken with that
entity's GLOBAL orientation (supplied by `local_tet_to_edgeid` /
`local_tet_to_triid`). Two tets sharing an entity therefore evaluate the same
integral over the same geometry and produce identical DOF values. No averaging
is involved and none is needed -- which matters, because a slightly wrong Pi
is not merely inaccurate but actively harmful on an indefinite operator.
That agreement is checked, not assumed (`max_disagreement` below).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .....elements.nedelec2 import Nedelec2
from .kernels import interpolation_kernel

# 4-point Gauss-Legendre on [0, 1] -- exact to degree 7 along an edge
_GX, _GW = np.polynomial.legendre.leggauss(4)
_GS, _GWS = 0.5 * (_GX + 1.0), 0.5 * _GW

# degree-4 symmetric triangle rule in barycentric coordinates; weights sum to 1
_TB = np.array([
    [0.108103018168070, 0.445948490915965, 0.445948490915965],
    [0.445948490915965, 0.108103018168070, 0.445948490915965],
    [0.445948490915965, 0.445948490915965, 0.108103018168070],
    [0.816847572980459, 0.091576213509771, 0.091576213509771],
    [0.091576213509771, 0.816847572980459, 0.091576213509771],
    [0.091576213509771, 0.091576213509771, 0.816847572980459],
])
_TW = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)


def assemble_nedelec_interpolation(field: Nedelec2, rtol: float = 1e-10,
                                    check: bool = True) -> csr_matrix:
    """Assemble Pi, shape (field.n_field, 3 * (n_nodes + n_edges)).

    Column ordering is component-major: all x columns, then y, then z, each
    block ordered like Legrange2 (all node DOFs, then edge DOFs offset by
    n_nodes).
    """
    mesh = field.mesh
    n_nodes, n_edges = mesh.n_nodes, mesh.n_edges
    n_scalar = n_nodes + n_edges

    tet_to_field = np.ascontiguousarray(field.get_tet_to_field())
    dofcodes = np.ascontiguousarray(field.dofcodes3d)
    typearr, indexarr = Nedelec2._parse_dofcode_np(dofcodes)
    ndof = dofcodes.shape[0]
    if ndof != 20:
        raise NotImplementedError(
            f"assemble_nedelec_interpolation expects the 20-DOF Nedelec-2 element "
            f"(edge ids {{0,2}}, face ids {{0,1}}); this field has {ndof} local DOFs.")
    n_tets = mesh.n_tets

    total_entries = n_tets * ndof * 30
    vals = np.empty(total_entries, dtype=np.float64)
    rows = np.empty(total_entries, dtype=np.int64)
    cols = np.empty(total_entries, dtype=np.int64)
    cond = np.empty(n_tets, dtype=np.float64)

    interpolation_kernel(
        np.ascontiguousarray(mesh.nodes), np.ascontiguousarray(mesh.tets),
        np.ascontiguousarray(mesh.tris), np.ascontiguousarray(mesh.edges),
        np.ascontiguousarray(mesh.tet_to_edge), np.ascontiguousarray(mesh.tet_to_tri),
        tet_to_field, dofcodes, np.ascontiguousarray(typearr),
        np.ascontiguousarray(indexarr), _GS, _GWS, _TB, _TW, n_scalar,
        vals, rows, cols, cond)

    worst_cond = float(cond.max()) if n_tets else 1.0
    if worst_cond > 1e6:
        raise RuntimeError(
            f"assemble_nedelec_interpolation: the moment functionals are not "
            f"unisolvent for this basis (worst cond(Q) = {worst_cond:.3e}). The "
            f"operator would be silently wrong, so it is not returned.")

    shape = (field.n_field, 3 * n_scalar)
    total = coo_matrix((vals, (rows, cols)), shape=shape).tocsr()
    count = coo_matrix((np.ones_like(vals), (rows, cols)), shape=shape).tocsr()

    if check:
        # Entity-local moments must give every tet the SAME value for a shared
        # DOF. Verify that rather than assume it: group the raw per-tet
        # contributions by (row, col) and measure the spread within each group.
        order = np.lexsort((cols, rows))
        r_s, c_s, v_s = rows[order], cols[order], vals[order]
        new_group = np.empty(len(r_s), dtype=bool)
        new_group[0] = True
        new_group[1:] = (r_s[1:] != r_s[:-1]) | (c_s[1:] != c_s[:-1])
        gid = np.cumsum(new_group) - 1
        gmax = np.full(gid[-1] + 1, -np.inf)
        gmin = np.full(gid[-1] + 1, np.inf)
        np.maximum.at(gmax, gid, v_s)
        np.minimum.at(gmin, gid, v_s)
        scale = np.abs(v_s).max() if len(v_s) else 1.0
        disagreement = float((gmax - gmin).max() / scale) if scale > 0 else 0.0
        if disagreement > 1e-8:
            raise RuntimeError(
                f"assemble_nedelec_interpolation: tets sharing a DOF disagree on its "
                f"value by {disagreement:.3e} (relative). The moments are supposed to "
                f"be entity-local and therefore single-valued, so this means an "
                f"orientation or indexing mismatch, not a rounding effect.")

    total.data /= count.data
    if total.nnz:
        total.data[np.abs(total.data) < rtol * np.abs(total.data).max()] = 0.0
        total.eliminate_zeros()
    return total
