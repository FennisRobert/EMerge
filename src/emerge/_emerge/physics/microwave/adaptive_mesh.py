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

# Last Cleanup: 2026-03-XX
from .microwave_data import MWField
import numpy as np
from loguru import logger
from emr.adaptive_mesh import compute_error_single


def print_sparam_matrix(pre: str, S: np.ndarray):
    """
    Print an N x N complex S-parameter matrix in dB∠deg format.
    Magnitude in dB rounded to 2 decimals, phase in degrees with 1 decimal.
    """
    S = np.asarray(S)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square (N x N) complex matrix")

    N = S.shape[0]
    logger.debug(pre + "S-parameter matrix (dB ∠ deg):")

    for i in range(N):
        row_str = []
        for j in range(N):
            mag_db = 20 * np.log10(np.abs(S[i, j]) + np.finfo(float).eps)
            phase_deg = np.degrees(np.angle(S[i, j]))
            row_str.append(f"{mag_db:6.2f} dB ∠ {phase_deg:6.1f}°")
        logger.debug(" | ".join(row_str))


def compute_convergence(Sold: np.ndarray, Snew: np.ndarray) -> float:
    """
    Return a single scalar: max |Snew - Sold|.
    Works for shapes (N,N) or (..., N, N); reduces over all axes.
    """

    Sold = np.asarray(Sold)
    Snew = np.asarray(Snew)
    print_sparam_matrix("Old:", Sold)
    print_sparam_matrix("New", Snew)
    if Sold.shape != Snew.shape:
        raise ValueError("Sold and Snew must have identical shapes")
    # amp_conv = float(np.abs(np.abs(Snew) - np.abs(Sold)).max())
    mag_conv = float(np.abs(np.abs(Snew) - np.abs(Sold)).max())
    amp_conv = float(np.abs(Snew - Sold).max())
    phase_conv = (
        float(np.abs(np.angle(Sold / Snew)).max()) * 180 / np.pi
    )
    return amp_conv, mag_conv, phase_conv


def compute_error_estimate(
    field: MWField, pec_tris: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Top level function to compute the EM error of a field solution.

    Args:
        field (MWField): The MWField object to analyse
        pec_tris (list[int]): A list of triangles that ought to be considered PEC (non-neighbouring.)

    Returns:
        np.ndarray, np.ndarray: The error estimate in an (Ntet,) float array and the tetrahedral size value.
    """
    mesh = field.mesh

    nodes = mesh.nodes
    tris = mesh.tris
    tets = mesh.tets
    edges = mesh.edges
    centers = mesh.centers

    As = mesh.areas
    tet_to_edge = mesh.tet_to_edge
    tet_to_tri = mesh.tet_to_tri
    tri_centers = mesh.tri_centers
    tri_to_tet = mesh.tri_to_tet
    tet_to_field = field.basis.tet_to_field
    dofcodes3d = field.basis.dofcodes3d  # NEW: needed for generic DOF dispatch
    er = field._der
    ur = field._dur

    Ls = mesh.edge_lengths

    pec_tris = np.sort(np.unique(np.array(pec_tris))).astype(np.int64)

    errors = []
    for key in field._fields.keys():
        excitation = field._fields[key]
        error, sizes = compute_error_single(
            nodes,
            tets,
            tris,
            edges,
            centers,
            excitation,
            Ls,
            As,
            tet_to_edge,
            tet_to_tri,
            tri_centers,
            tri_to_tet,
            tet_to_field,
            er,
            ur,
            pec_tris,
            field.k0,
            dofcodes3d,
        )
        errors.append(error)

    error = np.max(np.array(errors), axis=0)
    return error, sizes