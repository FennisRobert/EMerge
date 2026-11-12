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
            profile γ_m e_tm + ∇e_zm and kappa_m (WavePort.get_modepf_kappa_all).
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


############################################################
#                  FLOQUET PORT PROJECTION                 #
############################################################


def assemble_floquet_port(
    field: Nedelec2,
    surf_triangle_indices: np.ndarray,
    active_dof_ids: np.ndarray,
    port,
    k0: float,
) -> tuple[np.ndarray, dict[int | float, np.ndarray]]:
    """Assembles the dense Floquet mode projection of a FloquetPort.

    The port operator is n×∇×E = Γ E_t with

        Γ E_t = γ_r E_t + Σ_i (γ_i - γ_r) e_i ∬ E_t·e_i* dS / ∬ |e_i|² dS

    for the basis modes e_i of the port (TE and TM Floquet harmonics, orthogonal in ∬ e_i·e_j* dS over the
    cell). γ_r E_t is the Robin term of the system pattern (FloquetPort.get_gamma); this function returns the
    dense remainder Σ_i (γ_i - γ_r)/N_i G_i G_i^H with G_i = ∫ N·e_i dS, and the excitation -2 Σ_i c_i γ_i G_i
    of every S-matrix mode.

    Args:
        field (Nedelec2): The Nedelec2 field object.
        surf_triangle_indices (np.ndarray): Indices of the port surface triangles.
        active_dof_ids (np.ndarray): The port DoFs to expand the block over (see wpbc_dofs).
        port (FloquetPort): The Floquet port.
        k0 (float): The free space wavenumber.

    Returns:
        tuple[np.ndarray, dict[int | float, np.ndarray]]: The dense B-matrix values (ordered as
        wpbc_rowcol(active_dof_ids)) and the excitation vector of every S-matrix index.
    """
    gamma_r = port.get_gamma(k0)
    n = len(active_dof_ids)
    block = np.zeros((n, n), dtype=np.complex128)
    G = []
    for i in range(port.n_basis):
        def e_i(x, y, z, i=i):
            return port.basis_field(k0, i, x, y, "Exy")

        Gi = assemble_robin_bc_bvec(field, surf_triangle_indices, e_i)
        gamma_i = port.basis_gamma(k0, i)
        Ga = Gi[active_dof_ids]
        block += (gamma_i - gamma_r) / port.basis_norm(k0, i) * np.outer(Ga, np.conj(Ga))
        G.append(gamma_i * Gi)
    excitations: dict[int | float, np.ndarray] = {}
    for number, mode_nr in port._iter_port_numbers():
        excitations[number] = -2 * sum(c * g for c, g in zip(port._mode_coeffs(mode_nr), G) if c != 0)
    return block.ravel(), excitations
