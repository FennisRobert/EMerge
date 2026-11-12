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

# [Written by Claude] One solver for the port modes of ModalPort and WavePort (was SolverARPACK_BMA and
# SolveRoutine.eig_boundary_shifts).
"""The boundary mode solver: the modes A x = λ B x (λ = -β²) of a port face."""
from __future__ import annotations
import numpy as np
from scipy.linalg import eig  # type: ignore
from scipy.sparse import csc_matrix  # type: ignore
from scipy.sparse.linalg import eigs, splu, LinearOperator, ArpackNoConvergence  # type: ignore
from loguru import logger
from .solver_base import EigSolver, _pfx


class SolverARPACK_BMA(EigSolver):
    """Finds the port modes A x = λ B x (λ = -β²) closest to a list of shifts.

    The problem has a large cluster of spurious solutions at λ = 0 (β² = 0) and the physical modes elsewhere:
    propagating modes at λ < 0, evanescent modes at λ > 0. Modes are accepted on their eigenvalue and residual
    alone, never on the norms of the eigenvector, which depend on how a mode divides its norm over the edge and
    the E_z DOFs.

    Small problems (or direct=True) are solved densely. Larger ones get one shift-invert Arnoldi solve per cluster
    of shifts. If that returns fewer than nmodes valid modes, the subspace is enlarged and then, around a
    propagating shift, the search widens to β·(1.41, 0.71, 2, 0.5).
    """

    name = "ARPACK-BMA"
    dense_limit: int = 500
    widen_factors: tuple[float, ...] = (1.41, 0.71, 2.0, 0.5)

    def eig(  # type: ignore[override]
        self,
        A: csc_matrix,
        B: csc_matrix,
        nmodes: int,
        shifts: list[complex],
        kmax2: float,
        direct: bool = False,
        keep_evanescent: bool = True,
        descending: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the nmodes boundary modes closest to the shifts.

        Args:
            A (csc_matrix): The stiffness matrix on the free DOFs
            B (csc_matrix): The mass matrix on the free DOFs
            nmodes (int): The number of modes to return
            shifts (list[complex]): The values of λ = -β² to search around
            kmax2 (float): The largest physical β² (k0² εr μr max), which sets the scale of the spurious cluster
            direct (bool, optional): Always use the dense solver. Defaults to False.
            keep_evanescent (bool, optional): Keep modes with β² <= 0. Defaults to True.
            descending (bool, optional): Return the modes with the largest β² instead of those closest to the
                shifts. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: The eigenvalues λ and the eigenvectors (columns)
        """
        A, B = A.tocsc(), B.tocsc()
        n = A.shape[0]
        shifts_arr = np.array([complex(sh).real for sh in shifts])
        kmax2 = abs(kmax2)

        if direct or n < self.dense_limit:
            logger.debug(f"{_pfx(self.pre)} LAPACK: all {n} boundary modes")
            lam, vecs = eig(A.toarray(), B.toarray(), check_finite=False)
            lam, vecs = self._valid(A, B, lam, vecs, kmax2, keep_evanescent)
        else:
            lam, vecs = self._shift_invert(A, B, nmodes, shifts_arr, kmax2, keep_evanescent, extra=6)
            if lam.shape[0] < nmodes:
                more = self._shift_invert(A, B, nmodes, shifts_arr, kmax2, keep_evanescent, extra=max(2 * nmodes, 12))
                lam, vecs = self._unique(*_join((lam, vecs), more), kmax2)
            if lam.shape[0] < nmodes and not descending:
                for sh in shifts_arr[shifts_arr < 0]:
                    for q in self.widen_factors:
                        more = self._shift_invert(A, B, nmodes, np.array([q**2 * sh]), kmax2, keep_evanescent, extra=6)
                        lam, vecs = self._unique(*_join((lam, vecs), more), kmax2)
                        if lam.shape[0] >= nmodes:
                            break
                    if lam.shape[0] >= nmodes:
                        break

        if descending:
            order = np.argsort(lam.real, kind="stable")[:nmodes]
        else:
            distance = np.min(np.abs(lam.real[:, None] - shifts_arr[None, :]), axis=1)
            order = np.argsort(distance, kind="stable")[:nmodes]
        logger.debug(f"{_pfx(self.pre)} {order.shape[0]} boundary modes near β² = {[-s for s in shifts_arr]}")
        return lam[order], vecs[:, order]

    def _shift_invert(
        self,
        A: csc_matrix,
        B: csc_matrix,
        nmodes: int,
        shifts: np.ndarray,
        kmax2: float,
        keep_evanescent: bool,
        extra: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One shift-invert Arnoldi solve per cluster of shifts; returns the valid, unique modes."""
        n = A.shape[0]
        # Shifts closer than 5 % of kmax² share one factorization (at their mean)
        clusters: list[list[float]] = []
        for sh in np.sort(shifts):
            if clusters and abs(sh - clusters[-1][-1]) < 0.05 * kmax2:
                clusters[-1].append(sh)
            else:
                clusters.append([sh])
        lam_list, vec_list = [np.zeros((0,), dtype=np.complex128)], [np.zeros((n, 0), dtype=np.complex128)]
        for cluster in clusters:
            sigma = float(np.mean(cluster))
            k = min(max(len(cluster), nmodes // len(clusters)) + extra, n - 2)
            try:
                # [Written by Claude] A - sigma*B is symmetric: SuperLU's symmetric mode keeps the fill-reducing
                # ordering and pivots on the diagonal where it can. With the default (partial pivoting,
                # threshold 1) a row swap can destroy the ordering: the microstrip port of a 16k-DoF line
                # model filled to 132 M entries (199 s) instead of 1.2 M (0.17 s), same residual (1e-11).
                lu = splu((A - sigma * B).tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.01,
                          options=dict(SymmetricMode=True))
            except RuntimeError as err:  # singular: the shift is an eigenvalue
                logger.debug(f"{_pfx(self.pre)} SuperLU failed at β² = {-sigma:.1f}: {err}")
                continue
            op = LinearOperator(A.shape, matvec=lambda x, lu=lu: lu.solve(B @ x), dtype=np.complex128)
            try:
                theta, v = eigs(op, k=k, which="LM", ncv=min(n - 1, max(3 * k, 40)), tol=1e-9)
            except ArpackNoConvergence as err:
                logger.debug(f"{_pfx(self.pre)} ARPACK: {len(err.eigenvalues)}/{k} modes converged at β² = {-sigma:.1f}")
                theta, v = err.eigenvalues, err.eigenvectors
            lam_list.append(sigma + 1.0 / theta)
            vec_list.append(v)
        lam, vecs = np.concatenate(lam_list), np.hstack(vec_list)
        return self._unique(*self._valid(A, B, lam, vecs, kmax2, keep_evanescent), kmax2)

    @staticmethod
    def _valid(
        A: csc_matrix,
        B: csc_matrix,
        lam: np.ndarray,
        vecs: np.ndarray,
        kmax2: float,
        keep_evanescent: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Removes the spurious cluster (|λ| small), unphysical modes (β² > 2 kmax²) and unconverged vectors."""
        zero_tol = 1e-4 * kmax2
        keep = np.isfinite(lam) & (np.abs(lam) > zero_tol) & (-lam.real < 2 * kmax2)
        if not keep_evanescent:
            keep &= -lam.real > zero_tol
        lam, vecs = lam[keep], vecs[:, keep]
        Ax, Bx = A @ vecs, B @ vecs
        residual = np.linalg.norm(Ax - Bx * lam[None, :], axis=0) / (
            np.linalg.norm(Ax, axis=0) + np.abs(lam) * np.linalg.norm(Bx, axis=0) + 1e-300
        )
        keep = residual < 1e-6
        return lam[keep], vecs[:, keep]

    @staticmethod
    def _unique(lam: np.ndarray, vecs: np.ndarray, kmax2: float) -> tuple[np.ndarray, np.ndarray]:
        """Removes the copies of one mode found from several shifts (same λ and CMAC > 0.98)."""
        order = np.argsort(lam.real)
        lam, vecs = lam[order], vecs[:, order]
        unique: list[int] = []
        for i in range(lam.shape[0]):
            duplicate = False
            for j in unique:
                if abs(lam[i] - lam[j]) < 1e-6 * kmax2:
                    cmac = abs(np.vdot(vecs[:, j], vecs[:, i])) ** 2 / (
                        np.vdot(vecs[:, i], vecs[:, i]).real * np.vdot(vecs[:, j], vecs[:, j]).real
                    )
                    if cmac > 0.98:
                        duplicate = True
                        break
            if not duplicate:
                unique.append(i)
        return lam[unique], vecs[:, unique]


def _join(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([a[0], b[0]]), np.hstack([a[1], b[1]])
