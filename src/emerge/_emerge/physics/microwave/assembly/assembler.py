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

# Last Cleanup: 2026-08-19 (deduplication pass)
import numpy as np
from ..bcs import (
    PEC,
    BoundaryCondition,
    ScatteredField,
    RobinBC,
    PortBC,
    MWBoundaryConditionSet,
    ThinConductor,
    SurfaceImpedance,
    WavePortIH,
)
from ...material_assignment import MaterialAssignment
from ....periodic import Periodic
from ....elements.nedelec2 import Nedelec2
from ....elements.nedleg2 import NedelecLegrange2
from ....elements.dofsets import DoFSet
from ....mth.csc_cast import CSCMapping
from emsutil import Material
from ....settings import Settings
from scipy.sparse import csc_matrix
from .matrix_add import add_coo_to_csc, csc_axpy_same_pattern
from loguru import logger
from ..simjob import SimJob
from ....const import EPS0, C0
import time

_PBC_DSMAX = 1e-15


############################################################
#                         FUNCTIONS                        #
############################################################

def _format_freq(freq: float) -> str:
    units = ["Hz", "kHz", "MHz", "GHz", "THz"]

    if freq == 0:
        return "0.00 Hz"

    i = int(np.floor(np.log10(abs(freq)) / 3))
    i = max(0, min(i, len(units) - 1))

    scaled_freq = freq / (1000.0 ** i)
    return f"{scaled_freq:.2f} {units[i]}"


def do_assemble_wpbc(bc: BoundaryCondition) -> bool:
    if isinstance(bc, WavePortIH):
        return True
    return False


def diagnose_matrix(mat: csc_matrix, basis: Nedelec2):
    """
    Performs high-fidelity diagnostics on the FEM system matrix.
    Crashes with a detailed report if the matrix is numerically or structurally unfit.
    """
    print("--- Starting FEM Matrix Diagnostics ---")

    n_dofs = mat.shape[0]
    report = []
    failed = False

    if mat.shape[0] != mat.shape[1]:
        report.append(f"CRITICAL: Non-square matrix detected ({mat.shape})")
        failed = True

    if n_dofs != basis.n_field:
        report.append(
            f"CRITICAL: DoF mismatch! Matrix size {n_dofs} != Basis nfield {basis.n_field}"
        )
        failed = True

    col_counts = np.diff(mat.indptr)
    empty_cols = np.where(col_counts == 0)[0]

    row_present = np.zeros(n_dofs, dtype=bool)
    row_present[mat.indices] = True
    empty_rows = np.where(~row_present)[0]

    if len(empty_cols) > 0 or len(empty_rows) > 0:
        failed = True
        report.append(
            f"CRITICAL: Found {len(empty_cols)} empty columns and {len(empty_rows)} empty rows."
        )

        nedges = basis.nedges
        ntris = basis.ntris

        def map_dof_to_entity(dofs):
            e_idx = dofs[
                (dofs < nedges)
                | ((dofs >= (nedges + ntris)) & (dofs < (2 * nedges + ntris)))
            ]
            t_idx = dofs[
                ((dofs >= nedges) & (dofs < (nedges + ntris)))
                | (dofs >= (2 * nedges + ntris))
            ]
            return e_idx, t_idx

        e_empty, t_empty = map_dof_to_entity(empty_cols)

        if len(e_empty) > 0:
            report.append(
                f" -> Problem involves {len(e_empty)} Edge DoFs. Check for unmeshed lines or duplicate STEP edges."
            )
        if len(t_empty) > 0:
            report.append(
                f" -> Problem involves {len(t_empty)} Triangle DoFs. Check for internal non-conformal faces."
            )

    diag = mat.diagonal()
    zero_diag = np.where(np.isclose(diag, 0, atol=1e-15))[0]
    if len(zero_diag) > 0:
        true_zero_diag = np.setdiff1d(zero_diag, empty_cols)
        if len(true_zero_diag) > 0:
            failed = True
            report.append(
                f"CRITICAL: {len(true_zero_diag)} non-empty columns have zero diagonal (Numerical Singularity)."
            )

    if (mat - mat.T).nnz > 0:
        max_asym = np.max(np.abs((mat - mat.T).data)) if mat.nnz > 0 else 0
        if max_asym > 1e-12:
            report.append(f"WARNING: Matrix is asymmetric. Max diff: {max_asym}")

    if failed:
        print("\n" + "!" * 50)
        print("MATRIX DIAGNOSTICS FAILED")
        print("!" * 50)
        for line in report:
            print(line)

        print("\nHINT: Your 'nfield' is based on mesh.n_edges and n_tris.")
        print("If parts aren't 'welded' with gmsh.model.mesh.removeDuplicateNodes(),")
        print("you will have extra DoFs on the interface that never get assembled.")
        print("!" * 50)
        identify_mesh_dead_zones(mat, basis)
        raise RuntimeError("FEM Matrix is singular or improperly assembled.")

    print("Diagnostics Passed: Matrix is structurally sound.")


def identify_mesh_dead_zones(mat, basis: "Nedelec2"):
    """
    Identifies exactly which physical parts of the STEP file
    correspond to the empty matrix columns.
    """
    col_counts = np.diff(mat.indptr)
    empty_indices = np.where(col_counts == 0)[0]

    if len(empty_indices) == 0:
        print("No empty columns found spatially.")
        return

    nedges = basis.nedges
    ntris = basis.ntris

    dead_elements = {"edges": [], "tris": []}
    dead_coords = []

    for idx in empty_indices:
        if idx < nedges:
            e_id = idx
            dead_elements["edges"].append(e_id)
            dead_coords.append(basis.mesh.edge_centers[:, e_id])
        elif nedges <= idx < (nedges + ntris):
            t_id = idx - nedges
            dead_elements["tris"].append(t_id)
            dead_coords.append(basis.mesh.tri_centers[:, t_id])
        elif (nedges + ntris) <= idx < (2 * nedges + ntris):
            e_id = idx - (nedges + ntris)
            dead_elements["edges"].append(e_id)
            dead_coords.append(basis.mesh.edge_centers[:, e_id])
        else:
            t_id = idx - (2 * nedges + ntris)
            dead_elements["tris"].append(t_id)
            dead_coords.append(basis.mesh.tri_centers[:, t_id])

    np.save("dead_coords.npy", np.array(dead_coords).T)
    dead_coords = np.array(dead_coords)
    avg_pos = np.mean(dead_coords, axis=0)
    spread = np.std(dead_coords, axis=0)

    print("\n--- Spatial Autopsy Report ---")
    print(f"Dead Zone Center of Mass: {avg_pos}")
    print(f"Dead Zone Bounding Box Spread (std): {spread}")

    troubled_groups = set()

    for e_id in set(dead_elements["edges"]):
        for tag, edges in basis.mesh.etag_to_edge.items():
            if e_id in edges:
                troubled_groups.add(f"Physical Curve (Tag {tag})")

    for t_id in set(dead_elements["tris"]):
        for tag, tris in basis.mesh.ftag_to_tri.items():
            if t_id in tris:
                troubled_groups.add(f"Physical Surface (Tag {tag})")

    if troubled_groups:
        print("The following Physical Groups contain dead DoFs:")
        for group in troubled_groups:
            print(f" - {group}")
    else:
        print("HINT: Dead DoFs do not belong to any Physical Group.")
        print(
            "This means they are INTERIOR elements that your assembly loop is missing."
        )


def plane_basis_from_points(points: np.ndarray) -> np.ndarray:
    """
    Compute an orthonormal basis from a cloud of 3D points dominantly
    lying on one plane.
    """
    if points.shape[0] != 3:
        raise ValueError("Input must have shape (3, N)")

    centroid = points.mean(axis=1, keepdims=True)
    points_centered = points - centroid
    C = (points_centered @ points_centered.T) / points.shape[1]

    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    return eigvecs


############################################################
#                    THE ASSEMBLER CLASS                   #
############################################################

class TimeLogger:

    def __init__(self):
        self.ctr: int = 1
        self.last_time = time.time()
        self.active = True

    def __call__(self, ref: str = ''):
        if not self.active:
            return
        logger.info(f'{ref}: {self.ctr}: \u0394T = {(time.time()-self.last_time)*1000:.2f}ms')
        self.last_time = time.time()
        self.ctr += 1


_TMR = TimeLogger()


class Assembler:
    """The assembler class is responsible for FEM EM problem assembly.

    It stores some cached properties to accellerate preformance.
    """

    def __init__(self, settings: Settings):

        self.cached_matrices = None
        self.cached_cscmap: CSCMapping | None = None
        self.settings: Settings = settings
        self.SELECT_INDEX: int = None
        self._partitioned: bool = False

        self._surf_imp_conductivity_limit: float = 1e4

    # ------------------------------------------------------------------
    # Shared helpers (used by assemble_freq_matrix / assemble_scattering_matrix
    # / assemble_eig_matrix). assemble_bma_matrices is structurally different
    # (boundary-only, mixed-order field) and does not use these.
    # ------------------------------------------------------------------

    def _assemble_materials(
        self, mat_assy: MaterialAssignment, field: Nedelec2, frequency: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Evaluates er/ur/tand/cond over all tets and folds loss into er.

        Returns (er, ur, cond, is_frequency_dependent).
        """
        W0 = 2 * np.pi * frequency
        n = field.mesh.n_tets
        er = np.zeros((3, 3, n), dtype=np.complex128)
        ur = np.zeros((3, 3, n), dtype=np.complex128)
        tand = np.zeros((3, 3, n), dtype=np.complex128)
        cond = np.zeros((3, 3, n), dtype=np.complex128)

        for mat, centers, ids in mat_assy.iter_materials():
            er = mat.er(frequency, er, centers, ids)
            ur = mat.ur(frequency, ur, centers, ids)
            tand = mat.tand(frequency, tand, centers, ids)
            cond = mat.cond(frequency, cond, centers, ids)

        er = er * (1 - 1j * tand) - 1j * cond / (W0 * EPS0)

        is_freq_dep = mat_assy.frequency_dependent() or np.any(
            (cond > 0) & (cond < self.settings.mw_3d_peclim)
        )
        return er, ur, cond, is_freq_dep

    def _find_conductor_tets(self, cond: np.ndarray, n_tets: int) -> np.ndarray:
        """Tets whose conductivity exceeds either the PEC or surf-impedance limit."""
        limit = min(self.settings.mw_3d_peclim, self.settings.mw_3d_surfimplim)
        return np.flatnonzero(cond[0, 0, :n_tets] > limit)

    def _collect_pec_dofs(
        self, field: Nedelec2, mesh, pec_bcs: list[PEC], conductor_tets: np.ndarray
    ) -> tuple[set[int], list[int]]:
        """Collects PEC degrees of freedom from volumetric conductors and
        explicit PEC boundary conditions. Returns (pec_dof_ids, pec_tri_ids).
        """
        pec_ids: list[int] = []
        pec_tris: list[int] = []

        for itet in conductor_tets:
            pec_ids.extend(field.tet_to_field[:, itet])
            pec_tris.extend(field.mesh.tet_to_tri[:, itet])
        if len(conductor_tets):
            logger.trace(
                f" - Extended PEC with {len(conductor_tets)} tets with a conductivity > {self.settings.mw_3d_peclim}."
            )

        for pec in pec_bcs:
            logger.trace(f" - Implementing: {pec}")
            if len(pec.tags) == 0:
                continue
            tri_ids = mesh.get_triangles(pec.tags)
            edge_ids = mesh.tri_to_edge[:, tri_ids].flatten()
            pec_ids.extend(field.edge_to_field[:, edge_ids].flatten())
            pec_ids.extend(field.tri_to_field[:, tri_ids].flatten())
            pec_tris.extend(tri_ids)

        return set(pec_ids), pec_tris

    def _assemble_robin_terms(
        self,
        field: Nedelec2,
        mesh,
        K0: float,
        robin_bcs: list[RobinBC],
        thin_conductor_bcs: list[ThinConductor],
        pec_ids: set[int],
        force_callback=None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, set[int]]:
        """Assembles the Robin-BC matrix contribution shared by all three
        frequency-based assembly paths.

        Handles: PEC-dof removal for SurfaceImpedance/ThinConductor, the
        opposite-side thin-conductor matrix term, PML skip (matrix term only
        -- force_callback still runs, so a ScatteredField behind a PML still
        gets its excitation), and the order-2 ABC correction (always via
        bc.get_abccorr(K0), applied per-BC inside the loop).

        force_callback(bc, tri_ids), if given, is invoked once per BC after
        the matrix term so each caller can assemble its own excitation vector
        (port_vectors vs. background_fields vs. none) without duplicating
        this loop.

        Returns (B_matrix_robin, B_matrix_robin_2, updated_pec_ids). Both
        matrices are None if there are no Robin BCs.
        """
        from .robinbc import assemble_robin_bc
        from .robin_abc_order2 import abc_order_2_matrix

        if len(robin_bcs) == 0:
            return None, None, pec_ids

        logger.debug(" - Assembling Robin Boundary Conditions.")
        B_matrix_robin = field.empty_tri_matrix()
        B_matrix_robin_2 = (
            B_matrix_robin.copy().astype(np.complex128) if thin_conductor_bcs else None
        )

        for bc in robin_bcs:
            logger.trace(f"   - Implementing {bc}")
            tri_ids = mesh.get_triangles(bc.tags)

            if isinstance(bc, (SurfaceImpedance, ThinConductor)):
                dofs = set(field.tri_to_field[:, tri_ids].flatten())
                pec_ids = pec_ids.difference(dofs)

            gamma = bc.get_gamma(K0)
            logger.trace(f"    - robin bc \u03b3={gamma:.3f}")

            is_pml = getattr(bc, "pml", False)
            if bc._assemble_matrix and not is_pml:
                B_matrix_robin = assemble_robin_bc(field, B_matrix_robin, tri_ids, gamma)
                if isinstance(bc, ThinConductor):
                    B_matrix_robin_2 = assemble_robin_bc(field, B_matrix_robin_2, tri_ids, gamma)

            if force_callback is not None:
                force_callback(bc, tri_ids)

            if bc._isabc and bc.order == 2:
                logger.debug("    - Implementing second order ABC correction.")
                c2 = bc.get_abccorr(K0)
                B_matrix_robin += abc_order_2_matrix(field, tri_ids, c2)

        return B_matrix_robin, B_matrix_robin_2, pec_ids

    def _assemble_periodic_terms(
        self, field: Nedelec2, mesh, K0: float, periodic_bcs: list[Periodic]
    ) -> tuple[csc_matrix | None, np.ndarray | None, bool]:
        """Builds the combined periodic reduction matrix P and the set of
        retained DOF indices. Returns (Pmat, keep_indices, has_periodic).
        """
        from ....mth.pairing import pair_coordinates
        from .periodicbc import gen_periodic_matrix

        if len(periodic_bcs) == 0:
            return None, None, False

        logger.debug(" - Implementing Periodic Boundary Conditions.")
        Pmats = []
        remove: set[int] = set()

        for pbc in periodic_bcs:
            logger.trace(f"    - Implementing {pbc}")
            tri_ids_1 = mesh.get_triangles(pbc.face1.tags)
            edge_ids_1 = mesh.get_edges(pbc.face1.tags)
            tri_ids_2 = mesh.get_triangles(pbc.face2.tags)
            edge_ids_2 = mesh.get_edges(pbc.face2.tags)
            dv = np.array(pbc.dv)
            logger.trace(f"    - displacement vector {dv}")
            linked_tris = pair_coordinates(mesh.tri_centers, tri_ids_1, tri_ids_2, dv, _PBC_DSMAX)
            linked_edges = pair_coordinates(mesh.edge_centers, edge_ids_1, edge_ids_2, dv, _PBC_DSMAX)
            phi = pbc.phi(K0)
            logger.trace(f"    - \u03d5={phi} rad/m")
            Pmat, rows = gen_periodic_matrix(
                tri_ids_1,
                edge_ids_1,
                field.tri_to_field,
                field.edge_to_field,
                linked_tris,
                linked_edges,
                field.dofcodes2d,
                field.n_field,
                phi,
            )
            remove.update(rows)
            Pmats.append(Pmat)

        logger.trace(f"  - periodic bc removes {len(remove)} boundary DoF")
        Pmat = Pmats[0]
        for P2 in Pmats[1:]:
            Pmat = Pmat @ P2
        keep_indices = np.setdiff1d(np.arange(field.n_field), np.sort(np.unique(list(remove))))
        Pmat = Pmat[:, keep_indices]
        return Pmat, keep_indices, True

    def _apply_periodic_reduction(
        self,
        K: csc_matrix,
        solve_ids: np.ndarray,
        Pmat: csc_matrix,
        keep_indices: np.ndarray,
        NF: int,
        vectors: dict | None = None,
    ) -> tuple[csc_matrix, np.ndarray]:
        """Projects K (and optionally a dict of excitation vectors) through
        the periodic reduction matrix P, and remaps solve_ids into the
        reduced DOF numbering.
        """
        mask = np.zeros((NF,))
        mask[solve_ids] = 1
        mask = mask[keep_indices]
        solve_ids = np.argwhere(mask == 1).flatten()

        Pd = Pmat.getH()
        K = (Pd @ K @ Pmat).tocsc()
        if vectors is not None:
            for key, b in list(vectors.items()):
                vectors[key] = Pd @ b

        return K, solve_ids

    # ------------------------------------------------------------------
    # Boundary mode analysis (unchanged -- different field type / shape,
    # does not share the Robin/periodic/PEC pattern above)
    # ------------------------------------------------------------------

    def assemble_bma_matrices(
        self,
        field: Nedelec2,
        er: np.ndarray,
        ur: np.ndarray,
        sig: np.ndarray,
        k0: float,
        port: PortBC,
        bc_set: MWBoundaryConditionSet,
        dofset: DoFSet

    ) -> tuple[csc_matrix, csc_matrix, np.ndarray, NedelecLegrange2]:
        """Computes the boundary mode analysis matrices

        Args:
            field (Nedelec2): The Nedelec2 field object
            er (np.ndarray): The relative permittivity tensor of shape (3,3,N)
            ur (np.ndarray): The relative permeability tensor of shape (3,3,N)
            sig (np.ndarray): The conductivity scalar of shape (N,)
            k0 (float): The simulation phase constant
            port (PortBC): The port boundary condition object
            bcs (MWBoundaryConditionSet): The other boundary conditions

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, NedelecLegrange2]: The E, B, solve ids and Mixed order field object.
        """
        from .generalized_eigen_hb import generelized_eigenvalue_matrix

        logger.debug("Assembling Boundary Mode Matrices")

        mesh = field.mesh
        tri_ids = mesh.get_triangles(port.tags)
        logger.trace(f".boundary face has {len(tri_ids)} triangles.")

        boundary_surface = mesh.boundary_surface(port.tags)
        nedlegfield = NedelecLegrange2(boundary_surface, port.cs, dofset)

        ermesh = er[:, :, tri_ids]
        urmesh = ur[:, :, tri_ids]
        sigmesh = sig[tri_ids]

        loss = -1j * sigmesh / (k0 * C0 * EPS0)
        ermesh[0, 0, :] = ermesh[0, 0, :] + loss
        ermesh[1, 1, :] = ermesh[1, 1, :] + loss
        ermesh[2, 2, :] = ermesh[2, 2, :] + loss

        logger.trace(f".assembling matrices for {nedlegfield} at k0={k0:.2f}")
        E, B = generelized_eigenvalue_matrix(
            nedlegfield, ermesh, urmesh, port.cs._basis, k0
        )

        # TODO: Simplified to all "conductors" loosely defined. Must change to implementing line robin boundary conditions.
        pecs: list[BoundaryCondition] = bc_set.get_conductors()

        if len(pecs) > 0:
            logger.debug(f".total of equiv. {len(pecs)} PEC BCs implemented for BMA")

        pec_ids = []

        for it in range(boundary_surface.n_tris):
            if (
                sigmesh[it] > self.settings.mw_3d_peclim
                or sigmesh[it] > self.settings.mw_3d_surfimplim
            ):
                pec_ids.extend(list(nedlegfield.tri_to_field[:, it]))

        for pec in pecs:
            logger.trace(f".implementing {pec}")
            if len(pec.tags) == 0:
                continue
            face_tags = pec.tags
            tri_ids = mesh.get_triangles(face_tags)
            edge_ids = list(mesh.tri_to_edge[:, tri_ids].flatten())
            for ii in edge_ids:
                i2 = nedlegfield.mesh.from_source_edge(ii)
                if i2 is None:
                    continue
                eids = nedlegfield.edge_to_field[:, i2]
                pec_ids.extend(list(eids))

        pec_ids_set: set[int] = set(pec_ids)

        logger.trace(f".total of {len(pec_ids_set)} pec DoF to remove.")
        solve_ids = [i for i in range(nedlegfield.n_field) if i not in pec_ids_set]

        return E, B, np.array(solve_ids), nedlegfield

    # ------------------------------------------------------------------
    # Frequency-domain (driven port) assembly
    # ------------------------------------------------------------------

    def assemble_freq_matrix(
        self,
        field: Nedelec2,
        mat_assy: MaterialAssignment,
        bcs: list[BoundaryCondition],
        frequency: float,
        cache_matrices: bool = False,
    ) -> SimJob:
        """Assembles the frequency domain FEM matrix

        Args:
            field (Nedelec2): The Nedelec2 object of the problems
            mat_assy (MaterialAssignment): Material assignment for the domain
            bcs (list[BoundaryCondition]): The boundary conditions
            frequency (float): The simulation frequency
            cache_matrices (bool, optional): Whether to use and cache matrices. Defaults to False.

        Returns:
            SimJob: The resultant SimJob object
        """
        from .curlcurl import tet_mass_stiffness_matrices
        from .robinbc import assemble_robin_bc_bvec

        logger.debug(f'Assembling frequency = {_format_freq(frequency)}')

        W0 = 2 * np.pi * frequency
        K0 = W0 / C0
        mesh = field.mesh
        NF = field.n_field

        er, ur, cond, is_frequency_dependent = self._assemble_materials(mat_assy, field, frequency)
        conductor_tets = self._find_conductor_tets(cond, field.n_tets)
        logger.debug(f' - Total of {len(conductor_tets)} PEC Tetrahedrons')

        full_caching = cache_matrices and not is_frequency_dependent
        if full_caching and self.cached_matrices is not None:
            logger.debug(" - Using cached matricies.")
            Evec, Bvec = self.cached_matrices
            K: csc_matrix = csc_axpy_same_pattern(Evec, Bvec, (-K0 ** 2))
        else:
            logger.debug(" - Calling matrix assembler...")
            t0 = time.time()
            Evec, Bvec, cscmap = tet_mass_stiffness_matrices(
                field, er, ur, conductor_tets, self.cached_cscmap
            )
            t1 = time.time()
            logger.debug(f' - Assembly speed: {(field.ntets - len(conductor_tets)) / (t1 - t0):.1f} tets/s')
            self.cached_cscmap = cscmap

            K: csc_matrix = self.cached_cscmap.to_csc(Evec - Bvec * (K0 ** 2))

            if full_caching:
                self.cached_matrices = (self.cached_cscmap.to_csc(Evec), self.cached_cscmap.to_csc(Bvec))

        thin_conductor_bcs: list[ThinConductor] = [bc for bc in bcs if isinstance(bc, ThinConductor)]
        pec_bcs: list[PEC] = [bc for bc in bcs if isinstance(bc, PEC)]
        robin_bcs: list[RobinBC] = [bc for bc in bcs if isinstance(bc, RobinBC)]
        port_bcs: list[PortBC] = [bc for bc in bcs if isinstance(bc, PortBC)]
        periodic_bcs: list[Periodic] = [bc for bc in bcs if isinstance(bc, Periodic)]

        port_vectors: dict[int | float, np.ndarray] = {}
        for port in sorted(port_bcs, key=lambda x: x.port_number):
            for mat_index, mode_nr in port._iter_port_numbers():
                port_vectors[mat_index] = np.zeros((NF,), dtype=np.complex128)

        logger.debug(" - Implementing PEC Boundary Conditions.")
        pec_ids, pec_tris = self._collect_pec_dofs(field, mesh, pec_bcs, conductor_tets)

        def force_callback(bc, tri_ids):
            if not (bc._include_force and bc.driven and not isinstance(bc, ScatteredField)):
                return
            for number, Ufunc in bc._iter_modes(K0):
                b_p = assemble_robin_bc_bvec(field, tri_ids, Ufunc)
                port_vectors[number] += b_p
                logger.trace(f"    - included force vector term with norm {np.linalg.norm(b_p):.3f}")

        B_matrix_robin, B_matrix_robin_2, pec_ids = self._assemble_robin_terms(
            field, mesh, K0, robin_bcs, thin_conductor_bcs, pec_ids, force_callback
        )

        if B_matrix_robin is not None:
            add_coo_to_csc(K, B_matrix_robin, field._rows, field._cols)

            if B_matrix_robin_2 is not None:
                logger.debug("    - Assembling opposite side matrix entries.")
                rows, cols = field.empty_tri_rowcol(other_side=True)
                K += field.generate_csc(B_matrix_robin_2, (rows, cols))

        if len(periodic_bcs) > 0:
            logger.debug("  - Implementing Periodic Boundary Conditions.")
        Pmat, keep_indices, has_periodic = self._assemble_periodic_terms(field, mesh, K0, periodic_bcs)

        mask = np.ones(NF, dtype=bool)
        mask[list(pec_ids)] = False
        solve_ids = np.flatnonzero(mask)

        if has_periodic:
            K, solve_ids = self._apply_periodic_reduction(K, solve_ids, Pmat, keep_indices, NF, port_vectors)

        logger.debug(f"  - Number of tets: {mesh.n_tets:,}")
        logger.debug(f"  - Number of DoF: {K.shape[0]:,}")
        logger.debug(f"  - Number of non-zero: {K.nnz:,}")

        K.eliminate_zeros()

        simjob = SimJob(
            K, port_vectors, K0 * 299792458 / (2 * np.pi), symmetric=not has_periodic
        )

        simjob.solve_ids = solve_ids
        simjob._pec_tris = pec_tris

        if has_periodic:
            simjob.P = Pmat
            simjob.has_periodic = has_periodic

        return simjob, (er, ur, cond)

    # ------------------------------------------------------------------
    # Scattered-field assembly
    # ------------------------------------------------------------------

    def assemble_scattering_matrix(
        self,
        field: Nedelec2,
        mat_assy: MaterialAssignment,
        bcs: list[BoundaryCondition],
        frequency: float,
        cache_matrices: bool = False,
    ) -> SimJob:
        """Assembles the scattered-field frequency domain FEM matrix

        Args:
            field (Nedelec2): The Nedelec2 object of the problems
            mat_assy (MaterialAssignment): Material assignment for the domain
            bcs (list[BoundaryCondition]): The boundary conditions
            frequency (float): The simulation frequency
            cache_matrices (bool, optional): Whether to use and cache matrices. Defaults to False.

        Returns:
            SimJob: The resultant SimJob object
        """
        from .curlcurl import tet_mass_stiffness_matrices
        from .robinbc import assemble_robin_bc_bvec_scat

        W0 = 2 * np.pi * frequency
        K0 = W0 / C0
        mesh = field.mesh
        NF = field.n_field

        er, ur, cond, is_frequency_dependent = self._assemble_materials(mat_assy, field, frequency)
        conductor_tets = self._find_conductor_tets(cond, field.n_tets)

        if cache_matrices and not is_frequency_dependent and self.cached_matrices is not None:
            logger.debug("Using cached matricies.")
            matrix_stiff_coo, matrix_mass_coo = self.cached_matrices
        else:
            logger.debug("Assembling matrices")
            matrix_stiff_coo, matrix_mass_coo, cscmap = tet_mass_stiffness_matrices(
                field, er, ur, conductor_tets, self.cached_cscmap
            )
            self.cached_cscmap = cscmap
            self.cached_matrices = (matrix_stiff_coo, matrix_mass_coo)

        matrix_fem: csc_matrix = self.cached_cscmap.to_csc(
            matrix_stiff_coo - matrix_mass_coo * (K0 ** 2)
        )

        thin_conductor_bcs: list[ThinConductor] = [bc for bc in bcs if isinstance(bc, ThinConductor)]
        pec_bcs: list[PEC] = [bc for bc in bcs if isinstance(bc, PEC)]
        robin_bcs: list[RobinBC] = [bc for bc in bcs if isinstance(bc, RobinBC)]
        periodic_bcs: list[Periodic] = [bc for bc in bcs if isinstance(bc, Periodic)]

        logger.debug("Implementing PEC Boundary Conditions.")
        pec_ids, pec_tris = self._collect_pec_dofs(field, mesh, pec_bcs, conductor_tets)

        background_fields: dict[tuple[float, float], np.ndarray] = {}

        def force_callback(bc, tri_ids):
            # ScatteredField is both the Robin absorbing term (handled by the
            # shared matrix path above, subject to the pml skip) and the
            # excitation for the incident field -- assembled here regardless
            # of whether this BC is flagged as being backed by a PML.
            if not isinstance(bc, ScatteredField):
                return
            normals = field.mesh.outward_normals(tri_ids)
            for bf in bc._iter_fields(K0):
                b_p = assemble_robin_bc_bvec_scat(field, tri_ids, bf.Uinc, bf.Uinc_curl, normals)
                if bf in background_fields:
                    background_fields[bf] += b_p
                else:
                    background_fields[bf] = b_p
                logger.debug(f".. Background field {bf} {np.linalg.norm(b_p):.3f}")

        B_matrix_robin, B_matrix_robin_2, pec_ids = self._assemble_robin_terms(
            field, mesh, K0, robin_bcs, thin_conductor_bcs, pec_ids, force_callback
        )

        if B_matrix_robin is not None:
            matrix_fem += field.generate_csc(B_matrix_robin)

            if B_matrix_robin_2 is not None:
                logger.debug("Assembling opposite side matrix entries.")
                rows, cols = field.empty_tri_rowcol(other_side=True)
                matrix_fem += field.generate_csc(B_matrix_robin_2, (rows, cols))

        if len(periodic_bcs) > 0:
            logger.debug("Implementing Periodic Boundary Conditions.")
        Pmat, keep_indices, has_periodic = self._assemble_periodic_terms(field, mesh, K0, periodic_bcs)

        mask = np.ones(NF, dtype=bool)
        mask[list(pec_ids)] = False
        solve_ids = np.flatnonzero(mask)

        if has_periodic:
            matrix_fem, solve_ids = self._apply_periodic_reduction(
                matrix_fem, solve_ids, Pmat, keep_indices, NF, background_fields
            )

        logger.debug(f"Number of tets: {mesh.n_tets:,}")
        logger.debug(f"Number of DoF: {matrix_fem.shape[0]:,}")
        logger.debug(f"Number of non-zero: {matrix_fem.nnz:,}")

        simjob = SimJob(
            matrix_fem, background_fields, K0 * 299792458 / (2 * np.pi), symmetric=True
        )

        simjob.solve_ids = solve_ids

        if has_periodic:
            simjob.P = Pmat
            simjob.has_periodic = has_periodic

        return simjob, (er, ur, cond)

    # ------------------------------------------------------------------
    # Eigenmode assembly
    # ------------------------------------------------------------------

    def assemble_eig_matrix(
        self,
        field: Nedelec2,
        mat_assy: MaterialAssignment,
        bcs: list[BoundaryCondition],
        frequency: float,
    ) -> SimJob:
        """Assembles the eigenmode analysis matrix

        The assembly process is frequency dependent because the frequency-dependent properties
        need a guess before solving. There is currently no adjustment after an eigenmode is found.
        The frequency-dependent properties are simply calculated once for the given frequency.

        Args:
            field (Nedelec2): The Nedelec2 field
            mat_assy (MaterialAssignment): Material assignment for the domain
            bcs (list[BoundaryCondition]): The list of boundary conditions
            frequency (float): The compilation frequency (for material properties only)

        Returns:
            SimJob: The resultant simulation job
        """
        from .curlcurl import tet_mass_stiffness_matrices

        mesh = field.mesh
        k0 = 2 * np.pi * frequency / C0

        er, ur, cond, _ = self._assemble_materials(mat_assy, field, frequency)
        conductor_tets = self._find_conductor_tets(cond, field.n_tets)

        logger.debug("Assembling matrices")
        stiff, mass, cscmap = tet_mass_stiffness_matrices(field, er, ur, conductor_tets)
        matrix_stiff = cscmap.to_csc(stiff)
        matrix_mass = cscmap.to_csc(mass)
        self.cached_matrices = (matrix_stiff, matrix_mass)

        NDoF = matrix_stiff.shape[0]

        thin_conductor_bcs: list[ThinConductor] = [bc for bc in bcs if isinstance(bc, ThinConductor)]
        pec_bcs: list[PEC] = [bc for bc in bcs if isinstance(bc, PEC)]
        robin_bcs: list[RobinBC] = [bc for bc in bcs if isinstance(bc, RobinBC)]
        periodic_bcs: list[Periodic] = [bc for bc in bcs if isinstance(bc, Periodic)]

        logger.debug("Implementing PEC Boundary Conditions.")
        pec_ids, _ = self._collect_pec_dofs(field, mesh, pec_bcs, conductor_tets)

        # No force_callback: eigenmode assembly has no excitation vectors.
        B_matrix_robin, B_matrix_robin_2, pec_ids = self._assemble_robin_terms(
            field, mesh, k0, robin_bcs, thin_conductor_bcs, pec_ids, force_callback=None
        )

        if B_matrix_robin is not None:
            matrix_mass -= field.generate_csc(B_matrix_robin) / (k0 ** 2)
            if B_matrix_robin_2 is not None:
                logger.debug("Assembling opposite side matrix entries.")
                rows, cols = field.empty_tri_rowcol(other_side=True)
                matrix_mass -= field.generate_csc(B_matrix_robin_2, (rows, cols)) / (k0 ** 2)

        if len(periodic_bcs) > 0:
            logger.debug("Implementing Periodic Boundary Conditions.")
        Pmat, keep_indices, has_periodic = self._assemble_periodic_terms(field, mesh, k0, periodic_bcs)

        solve_ids = np.array([i for i in range(NDoF) if i not in pec_ids])

        if has_periodic:
            mask = np.zeros((NDoF,))
            mask[solve_ids] = 1
            mask = mask[keep_indices]
            solve_ids = np.argwhere(mask == 1).flatten()
            Pd = Pmat.getH()
            matrix_stiff = Pd @ matrix_stiff @ Pmat
            matrix_mass = Pd @ matrix_mass @ Pmat

        logger.debug(f"Number of tets: {mesh.n_tets}")
        logger.debug(f"Number of DoF: {matrix_stiff.shape[0]}")

        simjob = SimJob(matrix_stiff, None, frequency, B=matrix_mass)
        simjob.solve_ids = solve_ids

        if has_periodic:
            simjob.P = Pmat
            simjob.has_periodic = has_periodic

        return simjob, (er, ur, cond)