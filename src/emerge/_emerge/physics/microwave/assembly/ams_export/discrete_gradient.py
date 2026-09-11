# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.
"""Discrete gradient operator G : Legrange2 -> Nedelec2.

G is the matrix of the gradient in the two finite element bases: column `a`
holds the coefficients of `grad(phi_a)` expanded in the Nedelec2 basis, where
`phi_a` is the a-th Legrange2 (order-2 Lagrange) scalar basis function.

This is the standard input that auxiliary-space Maxwell (AMS) preconditioners
require from the assembler -- hypre takes it via HYPRE_AMSSetDiscreteGradient,
and MFEM/Palace assemble it from the FE space. It cannot be reconstructed from
the assembled E/B matrices, because it depends on the basis definitions and on
tet connectivity. Its columns span exactly the null space of the curl-curl
operator, which is what the preconditioner needs in order to treat the
gradient modes separately from everything else.

Why it is exact rather than approximate: for the Nedelec first-kind order-2
space (20 DOF per tet: 2 per edge x 6 + 2 per face x 4), grad(P2) is a genuine
subspace, so expanding grad(phi_a) in the Nedelec basis has an exact solution.
The local expansion is recovered by evaluating both sides at the element
quadrature points and solving the resulting overdetermined system; the
least-squares residual is ~1e-15 in practice, and is asserted below rather
than assumed, since a non-zero residual would mean the basis definitions and
this routine had gone out of sync.

Cross-check worth knowing about: the block of G coupling the lowest-order
(Whitney, dofcode 64) edge DOFs to the vertex DOFs comes out as exactly the
signed vertex-edge incidence matrix (-1 at the first vertex, +1 at the
second), which is the classical lowest-order discrete gradient.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .....elements.nedelec2 import Nedelec2
from ..curlcurl import _DPTS
from .kernels import gradient_kernel


def assemble_discrete_gradient(field: Nedelec2, rtol: float = 1e-8) -> csr_matrix:
    """Assemble G, shape (field.n_field, mesh.n_nodes + mesh.n_edges).

    Column ordering matches Legrange2: all node DOFs first, then all edge DOFs
    offset by n_nodes (see elements/leg2.py).

    `rtol` drops entries smaller than rtol times the largest magnitude, so the
    result stays sparse rather than carrying quadrature round-off.
    """
    mesh = field.mesh
    n_nodes = mesh.n_nodes
    n_lagrange = n_nodes + mesh.n_edges

    tet_to_field = np.ascontiguousarray(field.get_tet_to_field())
    dofcodes = np.ascontiguousarray(field.dofcodes3d)
    typearr, indexarr = Nedelec2._parse_dofcode_np(dofcodes)
    ndof = dofcodes.shape[0]
    n_tets = mesh.n_tets

    total_entries = n_tets * ndof * 10
    vals = np.empty(total_entries, dtype=np.float64)
    rows = np.empty(total_entries, dtype=np.int64)
    cols = np.empty(total_entries, dtype=np.int64)
    res = np.empty(n_tets, dtype=np.float64)

    gradient_kernel(
        np.ascontiguousarray(mesh.nodes), np.ascontiguousarray(mesh.tets),
        np.ascontiguousarray(mesh.tris), np.ascontiguousarray(mesh.edges),
        np.ascontiguousarray(mesh.tet_to_edge), np.ascontiguousarray(mesh.tet_to_tri),
        tet_to_field, dofcodes, np.ascontiguousarray(typearr),
        np.ascontiguousarray(indexarr), np.ascontiguousarray(_DPTS[1:5, :]),
        vals, rows, cols, res)

    worst = float(res.max()) if n_tets else 0.0
    if worst > 1e-8:
        raise RuntimeError(
            f"assemble_discrete_gradient: grad(P2) did not lie in the Nedelec2 span "
            f"(worst relative least-squares residual {worst:.3e}). This means the "
            f"Legrange2/Nedelec2 basis definitions and this routine have gone out of "
            f"sync -- the operator would be silently wrong, so it is not returned.")

    # A DOF shared by several tets gets one contribution per tet, and they must
    # agree (both bases are globally single-valued). Sum, then divide by the
    # multiplicity, so the shared value is reproduced rather than multiplied.
    shape = (field.n_field, n_lagrange)
    total = coo_matrix((vals, (rows, cols)), shape=shape).tocsr()
    count = coo_matrix((np.ones_like(vals), (rows, cols)), shape=shape).tocsr()
    total.data /= count.data

    if total.nnz:
        total.data[np.abs(total.data) < rtol * np.abs(total.data).max()] = 0.0
        total.eliminate_zeros()
    return total
