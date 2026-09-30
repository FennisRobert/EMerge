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

"""Fixed sparsity pattern assembly of the microwave FEM system matrix.

The system matrix is a sum of terms that each factor into a frequency
independent (geometry only) matrix and a scalar coefficient:

    K(k0) = E - k0² B + Σ γ_bc(k0) R_bc + Σ c2_bc(k0) A_bc + Σ W_port(k0)

  - E, B        : volume curl-curl / mass matrices (material dependent)
  - R_bc        : Robin integral ∫ (n×N_i)·(n×N_j) dS, assembled at γ = 1
  - A_bc        : second order ABC integral, assembled at c2 = 1
  - W_port      : dense rank-1 wave port block (values recomputed per
                  frequency since the port mode changes, pattern fixed)

All terms are written into ONE CSC sparsity pattern: the union of the
patterns of every term, computed once. A matrix at a given frequency is a
data array aligned with that pattern (a "system data array"); terms are
only ever added into it in place. Scipy sparse addition and products are
never used on the system matrix because they drop entries whose value is
exactly zero. This keeps the pattern identical at every frequency,
including entries that are numerically zero (e.g. couplings between
gradient modes), so a solver can reuse its symbolic factorization.
"""

from __future__ import annotations
import hashlib
import numpy as np
from numba import njit, prange
from scipy.sparse import csc_matrix, spmatrix, sparray

from ....mth.csc_cast import CSCMapping
from ....elements.nedelec2 import Nedelec2
from ..bcs import RobinBC, ThinConductor, WavePortIH
from .robinbc import assemble_robin_bc
from .robin_abc_order2 import abc_order_2_matrix
from .wpbc import wpbc_dofs, wpbc_rowcol


############################################################
#                      NUMBA KERNELS                       #
############################################################

_LINEAR_SCAN_THRESHOLD = 16


@njit(cache=True, inline="always")
def _find_row_in_column(indices: np.ndarray, lo: int, hi: int, row: int) -> int:
    """Position of `row` in the sorted slice indices[lo:hi], or -1 if absent.
    Linear scan for short columns, binary search otherwise."""
    if hi - lo <= _LINEAR_SCAN_THRESHOLD:
        for k in range(lo, hi):
            if indices[k] == row:
                return k
        return -1
    while lo < hi:
        mid = (lo + hi) >> 1
        val = indices[mid]
        if val == row:
            return mid
        elif val < row:
            lo = mid + 1
        else:
            hi = mid
    return -1


@njit(cache=True, parallel=True)
def _locate_entries_kernel(
    indices: np.ndarray, indptr: np.ndarray, rows: np.ndarray, cols: np.ndarray, data_indices: np.ndarray
) -> None:
    for i in prange(rows.shape[0]):
        c = cols[i]
        data_indices[i] = _find_row_in_column(indices, indptr[c], indptr[c + 1], rows[i])


@njit(cache=True, nogil=True)
def _scatter_add_per_triangle(
    out: np.ndarray, data_indices: np.ndarray, unit_values: np.ndarray, tri_coeffs: np.ndarray, values_per_tri: int
) -> None:
    """out[data_indices[k]] += tri_coeffs[k // values_per_tri] * unit_values[k].
    Repeated data indices accumulate."""
    for i in range(tri_coeffs.shape[0]):
        c = tri_coeffs[i]
        for k in range(i * values_per_tri, (i + 1) * values_per_tri):
            out[data_indices[k]] += c * unit_values[k]


@njit(cache=True, nogil=True)
def _scatter_add(out: np.ndarray, data_indices: np.ndarray, values: np.ndarray, scale: complex) -> None:
    """out[data_indices[k]] += scale * values[k]. Repeated data indices accumulate."""
    for k in range(values.shape[0]):
        out[data_indices[k]] += scale * values[k]


@njit(cache=True, nogil=True, parallel=True)
def _write_a_plus_alpha_b(out: np.ndarray, a: np.ndarray, b: np.ndarray, alpha: complex) -> None:
    """out = a + alpha * b"""
    for i in prange(out.shape[0]):
        out[i] = a[i] + alpha * b[i]


def locate_entries_in_pattern(indptr: np.ndarray, indices: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Index into the data array of the CSC pattern (indptr, indices) of every
    (rows[i], cols[i]) entry.

    Row indices within each column must be sorted. Raises ValueError if any
    entry is not part of the pattern.
    """
    data_indices = np.empty(rows.shape[0], dtype=np.int64)
    _locate_entries_kernel(indices, indptr, rows.astype(indices.dtype, copy=False),
                           cols.astype(indices.dtype, copy=False), data_indices)
    if (data_indices < 0).any():
        raise ValueError("Some (row, col) entries are not present in the sparsity pattern.")
    return data_indices


############################################################
#                      BOUNDARY TERMS                      #
############################################################

def bc_has_matrix_term(bc: RobinBC) -> bool:
    """Whether bc adds a Robin (or wave port) term to the system matrix.
    False for BCs marked dont_assemble() and for PML-backed BCs."""
    return bool(bc._assemble_matrix) and not getattr(bc, "pml", False)


def bc_has_abc2_term(bc: RobinBC) -> bool:
    """Whether bc adds the second order absorbing boundary correction term."""
    return bool(bc._isabc) and getattr(bc, "order", 1) == 2


class _BoundaryTerm:
    """One boundary matrix term, located in the system pattern.

    Attributes:
        coo_indices: (rows, cols) of the term's COO entries, one pair per copy
            of the term (a ThinConductor adds the same values on both sides).
            Only used while building the pattern, then cleared.
        data_indices: per copy, the system data index of every COO entry.
        unit_values: COO values at coefficient 1 (per-triangle local matrices,
            values_per_tri values each), or None for a wave port whose values
            are recomputed every frequency.
        values_per_tri: number of COO values per triangle (n_tri_dofs²).
    """

    def __init__(
        self, coo_indices: list[tuple[np.ndarray, np.ndarray]], unit_values: np.ndarray | None, values_per_tri: int
    ) -> None:
        self.coo_indices: list[tuple[np.ndarray, np.ndarray]] = coo_indices
        self.unit_values: np.ndarray | None = unit_values
        self.values_per_tri: int = values_per_tri
        self.data_indices: list[np.ndarray] = []


############################################################
#                      SYSTEM PATTERN                      #
############################################################

class SystemPattern:
    """The fixed CSC pattern of the system matrix and the frequency
    independent data needed to fill it.

    Holds the COO→CSC mapping of the volume assembly, the unit-coefficient
    boundary terms of every Robin/ABC BC and the wave port blocks, all
    located in one union pattern. Optionally caches the volume matrices E
    and B when the materials are frequency independent.

    Valid for one combination of field, conductor tets and Robin BC
    structure; see compute_key.
    """

    def __init__(self, key: tuple, field: Nedelec2, volume_coo_to_csc: CSCMapping, robin_bcs: list[RobinBC]) -> None:
        """Builds the union pattern and assembles the unit-coefficient boundary terms.

        Args:
            key (tuple): The result of compute_key for these arguments.
            field (Nedelec2): The basis.
            volume_coo_to_csc (CSCMapping): COO→CSC mapping of the volume assembly
                (as returned by tet_mass_stiffness_matrices).
            robin_bcs (list[RobinBC]): All Robin BCs of the simulation.
        """
        self.key: tuple = key
        self.N: int = field.n_field
        self.volume_coo_to_csc: CSCMapping = volume_coo_to_csc

        # Volume matrices as system data arrays, cached by the assembler when materials are frequency independent
        self.cached_E: np.ndarray | None = None
        self.cached_B: np.ndarray | None = None

        mesh = field.mesh
        self._robin_terms: dict[int, _BoundaryTerm] = {}
        self._abc2_terms: dict[int, _BoundaryTerm] = {}
        self._abc2_div_values: dict[int, np.ndarray] = {}
        self._wave_port_terms: dict[int, _BoundaryTerm] = {}
        self._wave_port_dofs: dict[int, np.ndarray] = {}

        values_per_tri = field.n_tri_dofs ** 2

        # Assemble the boundary terms at unit coefficient
        for bc in robin_bcs:
            tri_ids = mesh.get_triangles(bc.tags)
            if bc_has_matrix_term(bc):
                if isinstance(bc, WavePortIH):
                    dofs = wpbc_dofs(field, tri_ids)
                    self._wave_port_dofs[id(bc)] = dofs
                    self._wave_port_terms[id(bc)] = _BoundaryTerm([wpbc_rowcol(dofs)], None, 0)
                else:
                    unit_values, rows, cols = assemble_robin_bc(field, tri_ids, np.ones(tri_ids.shape[0], dtype=np.complex128))
                    coo_indices = [(rows, cols)]
                    if isinstance(bc, ThinConductor):
                        coo_indices.append(field.tri_rowcol(tri_ids, other_side=True))
                    self._robin_terms[id(bc)] = _BoundaryTerm(coo_indices, unit_values, values_per_tri)
            if bc_has_abc2_term(bc):
                curl_values, div_values, rows, cols = abc_order_2_matrix(field, tri_ids)
                self._abc2_terms[id(bc)] = _BoundaryTerm([(rows, cols)], curl_values, values_per_tri)
                self._abc2_div_values[id(bc)] = div_values

        terms = [*self._robin_terms.values(), *self._abc2_terms.values(), *self._wave_port_terms.values()]

        # Union pattern: unique volume entries + all boundary entries
        volume_cols = np.repeat(np.arange(self.N, dtype=np.int64), np.diff(volume_coo_to_csc.indptr))
        rows_all = [volume_coo_to_csc.indices] + [r for t in terms for r, _ in t.coo_indices]
        cols_all = [volume_cols] + [c for t in terms for _, c in t.coo_indices]
        union = CSCMapping.from_rowcol(
            np.ascontiguousarray(np.concatenate(rows_all), dtype=np.int64),
            np.ascontiguousarray(np.concatenate(cols_all), dtype=np.int64),
            self.N,
        )

        # Let scipy pick its index dtype once, so building a matrix per frequency never converts.
        template = csc_matrix((np.zeros(union.nnz), union.indices, union.indptr), shape=(self.N, self.N))
        self.indptr: np.ndarray = template.indptr
        self.indices: np.ndarray = template.indices
        self.nnz: int = union.nnz
        del union, template, rows_all, cols_all

        # The volume pattern is a subset of the union; if equal in size it is identical.
        if volume_coo_to_csc.nnz == self.nnz:
            self._volume_to_system: np.ndarray | None = None
        else:
            self._volume_to_system = locate_entries_in_pattern(self.indptr, self.indices, volume_coo_to_csc.indices, volume_cols)

        for t in terms:
            t.data_indices = [locate_entries_in_pattern(self.indptr, self.indices, r, c) for r, c in t.coo_indices]
            t.coo_indices = []

    @staticmethod
    def compute_key(field: Nedelec2, conductor_tets: np.ndarray, robin_bcs: list[RobinBC]) -> tuple:
        """Identifies everything the pattern depends on. A SystemPattern can be
        reused as long as this key is unchanged.

        Args:
            field (Nedelec2): The basis (identity, number of DoFs and tets).
            conductor_tets (np.ndarray): Tets excluded from the volume assembly.
            robin_bcs (list[RobinBC]): The Robin BCs, with the tags and flags
                that decide which boundary terms they add.

        Returns:
            tuple: (field id, n_field, n_tets, conductor tet hash, per-BC signature)
        """
        cond_hash = hashlib.sha1(np.ascontiguousarray(conductor_tets, dtype=np.int64).tobytes()).hexdigest()
        bc_sig = tuple(
            (id(bc), bc_has_matrix_term(bc), bc_has_abc2_term(bc), isinstance(bc, WavePortIH),
             isinstance(bc, ThinConductor), tuple(bc.tags))
            for bc in robin_bcs
        )
        return (id(field), field.n_field, field.mesh.n_tets, cond_hash, bc_sig)

    ############################################################
    #                    SYSTEM DATA ARRAYS                    #
    ############################################################

    def volume_coo_to_system_data(self, coo_values: np.ndarray) -> np.ndarray:
        """Sums volume COO values (dataE or dataB from tet_mass_stiffness_matrices)
        into a new system data array."""
        volume_data = self.volume_coo_to_csc.scatter(coo_values)
        if self._volume_to_system is None:
            return volume_data
        out = np.zeros(self.nnz, dtype=np.complex128)
        out[self._volume_to_system] = volume_data
        return out

    def a_plus_alpha_b(self, a: np.ndarray, b: np.ndarray, alpha: complex) -> np.ndarray:
        """Returns a + alpha * b as a new system data array (e.g. E - k0² B)."""
        out = np.empty(self.nnz, dtype=np.complex128)
        _write_a_plus_alpha_b(out, a, b, complex(alpha))
        return out

    def add_robin_term(self, out: np.ndarray, bc: RobinBC, gamma: complex | np.ndarray, scale: complex = 1.0) -> None:
        """out += scale * Σ_tri gamma[tri] R_tri, in place.

        gamma is a scalar or one value per triangle of bc. For a ThinConductor
        the term is added on both sides.
        """
        term = self._robin_terms[id(bc)]
        n_tris = term.unit_values.shape[0] // term.values_per_tri
        tri_coeffs = np.ascontiguousarray(np.broadcast_to(gamma * scale, n_tris), dtype=np.complex128)
        for data_indices in term.data_indices:
            _scatter_add_per_triangle(out, data_indices, term.unit_values, tri_coeffs, term.values_per_tri)

    def add_abc2_term(
        self, out: np.ndarray, bc: RobinBC, c2: tuple[complex, complex], scale: complex = 1.0
    ) -> None:
        """out += scale * (c_curl * Curl_bc + c_div * Div_bc), in place (second order ABC correction).

        c2 is (c_curl, c_div) as returned by bc.get_abccorr(k0): the TE (surface curl)
        and TM (surface divergence) coefficients.
        """
        c_curl, c_div = c2
        term = self._abc2_terms[id(bc)]
        _scatter_add(out, term.data_indices[0], term.unit_values, complex(c_curl * scale))
        _scatter_add(out, term.data_indices[0], self._abc2_div_values[id(bc)], complex(c_div * scale))

    def wave_port_dofs(self, bc: RobinBC) -> np.ndarray:
        """The DoFs the dense wave port block of bc covers (all DoFs on its face)."""
        return self._wave_port_dofs[id(bc)]

    def add_wave_port_term(self, out: np.ndarray, bc: RobinBC, values: np.ndarray, scale: complex = 1.0) -> None:
        """out += scale * values, in place. values is the dense wave port block
        of bc, ordered as wpbc_rowcol(self.wave_port_dofs(bc))."""
        _scatter_add(out, self._wave_port_terms[id(bc)].data_indices[0],
                     np.ascontiguousarray(values, dtype=np.complex128), complex(scale))

    def as_csc_matrix(self, data: np.ndarray) -> csc_matrix:
        """Wraps a system data array as a csc_matrix without copying."""
        return csc_matrix((data, self.indices, self.indptr), shape=(self.N, self.N), copy=False)


############################################################
#                  PATTERN-SAFE PRODUCTS                   #
############################################################

def ones_on_pattern(M: spmatrix | sparray) -> csc_matrix:
    """A csc_matrix with M's pattern (explicit zeros included) and every value set to one."""
    M = M.tocsc()
    return csc_matrix((np.ones(M.nnz), M.indices, M.indptr), shape=M.shape)


def product_pattern(*mats: spmatrix | sparray) -> csc_matrix:
    """The sparsity pattern of mats[0] @ mats[1] @ ..., with sorted indices.

    Computed on all-ones copies, so no entry can cancel to zero and be dropped.
    """
    out = ones_on_pattern(mats[0])
    for M in mats[1:]:
        out = out @ ones_on_pattern(M)
    out = out.tocsc()
    out.sort_indices()
    return out


def copy_onto_pattern(M: spmatrix | sparray, pattern: csc_matrix) -> csc_matrix:
    """Returns M's values on the (sorted) pattern of `pattern`, with zeros
    elsewhere. Every entry of M must be part of that pattern."""
    coo = M.tocoo()
    data_indices = locate_entries_in_pattern(pattern.indptr, pattern.indices, coo.row, coo.col)
    data = np.zeros(pattern.nnz, dtype=np.complex128)
    _scatter_add(data, data_indices, np.ascontiguousarray(coo.data, dtype=np.complex128), 1.0 + 0.0j)
    return csc_matrix((data, pattern.indices, pattern.indptr), shape=pattern.shape, copy=False)
