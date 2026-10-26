# [Written by Claude] All code in this module was written by Claude (AI).
from __future__ import annotations
import numpy as np
from typing import Callable
from ....elements import Nedelec2
from ....const import MU0, C0
from .emr_microwave import assemble_robin_bc_bvec
from .wpbc import assemble_Bmatrix_entries


############################################################
#       DENSE WAVE PORT BOUNDARY CONDITION, ALL MODES      #
############################################################


def assemble_wpbc_multimode(
    field: Nedelec2,
    surf_triangle_indices: np.ndarray,
    active_dof_ids: np.ndarray,
    modes: list[tuple[int | float, Callable, complex]],
    k0: float,
) -> tuple[np.ndarray, dict[int | float, np.ndarray]]:
    """Assembles the dense Wave Port Boundary Condition of a port with several modes.

    The single mode WPBC (assemble_wpbc) adds -1/(j w0 mu0 kappa_m) G_m G_mᵀ for the first mode. Here the
    same rank-1 term is added for every mode m, each with its own mode overlap vector
    G_m = ∫ (γ_m e_tm + ∇e_zm) · N dS and kappa_m. The modes are orthogonal in ∫ e_tn · w_m dS, so the sum
    is the projection on all port modes, each absorbed with its own propagation constant. The excitation
    vector of mode m is -2 G_m.

    Args:
        field (Nedelec2): The Nedelec2 field object.
        surf_triangle_indices (np.ndarray): Indices of the port surface triangles.
        active_dof_ids (np.ndarray): The port DoFs to expand the block over (see wpbc_dofs).
        modes (list[tuple[int | float, Callable, complex]]): Per mode the S-matrix index, the effective mode
            profile γ_m e_tm + ∇e_zm and kappa_m (MultiModeWavePort.get_modepf_kappa_all).
        k0 (float): The free space wavenumber.

    Returns:
        tuple[np.ndarray, dict[int | float, np.ndarray]]: The dense B-matrix values (ordered as
        wpbc_rowcol(active_dof_ids)) and the excitation vector of every S-matrix index.
    """
    w0 = C0 * k0
    values = np.zeros((len(active_dof_ids) ** 2,), dtype=np.complex128)
    excitations: dict[int | float, np.ndarray] = {}
    for number, mprof, kappa_m in modes:
        G = assemble_robin_bc_bvec(field, surf_triangle_indices, mprof)
        values += assemble_Bmatrix_entries(G, -1.0 / (1j * w0 * MU0 * kappa_m), active_dof_ids)
        excitations[number] = -2 * G
    return values, excitations
