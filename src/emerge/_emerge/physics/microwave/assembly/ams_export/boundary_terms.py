"""The second order absorbing boundary condition, exported as separate terms.

The export folds every boundary term into the mass matrix,
B = (E - K) / k0^2, which is all a solver needs but loses the one piece of
structure a preconditioner cannot recover afterwards: WHICH boundary term an
entry came from. That matters for the second order ABC, because

    K += c_curl * Curl + c_div * Div,      (c_curl, c_div) = bc.get_abccorr(k0)

with c_curl = i*b/k (b < 0 for every abctype) and c_div = -i*b/k. The surface
curl term therefore enters Im(K) NEGATIVE semidefinite, while the first order
ABC, the ports and the surface divergence term all enter it positive
semidefinite. A shifted preconditioner in the style of Palace
(palace/models/spaceoperator.cpp, `AssemblePreconditioner` with
`pc_mat_shifted`) needs Im(P) positive semidefinite: its coarse AMS solve runs
on the real matrix Re(P) + Im(P), and its Chebyshev smoothers assume the
spectrum of P lies in the right half plane on the positive side. Palace gets
this by construction (its own second order term is +i*(0.5/omega)*curl-curl,
farfieldboundaryoperator.cpp); downstream of the folded B it can only be
approximated by an eigendecomposition of the whole surface block, which also
mixes the first order and port terms in.

Exporting the two terms separately lets the preconditioner treat each at the
coefficient level, the way Palace treats material and boundary coefficients.
Both are returned as their contribution TO K (coefficient applied), so

    K = E - k0^2 B,    B_rest = B + (abc2_curl + abc2_div) / k0^2

recovers everything except the second order ABC.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix

from ..system_pattern import SystemPattern, bc_has_abc2_term


def assemble_abc2_terms(system: SystemPattern, robin_bcs: list, k0: float
                        ) -> tuple[csc_matrix | None, csc_matrix | None]:
    """The second order ABC's contribution to K, as (curl, div) matrices.

    Summed over every boundary condition that carries the term, each with its
    own coefficients from `bc.get_abccorr(k0)`, using the same cached unit
    matrices and the same `add_abc2_term` the system assembly uses, so the sum
    curl + div is bit-for-bit the term inside K. Returns (None, None) when no
    boundary condition has a second order term.
    """
    bcs = [bc for bc in robin_bcs if bc_has_abc2_term(bc)]
    if not bcs:
        return None, None
    curl = np.zeros(system.nnz, dtype=np.complex128)
    div = np.zeros(system.nnz, dtype=np.complex128)
    for bc in bcs:
        c_curl, c_div = bc.get_abccorr(k0)
        system.add_abc2_term(curl, bc, (c_curl, 0.0))
        system.add_abc2_term(div, bc, (0.0, c_div))
    return system.as_csc_matrix(curl), system.as_csc_matrix(div)
