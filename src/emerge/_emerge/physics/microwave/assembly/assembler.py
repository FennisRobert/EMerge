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

# Last Cleanup: 2025-01-01
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

    # Handle zero to avoid math domain errors with log
    if freq == 0:
        return "0.00 Hz"

    # Calculate index using log base 1000: floor(log10(abs(freq)) / 3)
    # This determines how many "groups of three zeros" are in the number
    i = int(np.floor(np.log10(abs(freq)) / 3))

    # Clamp index between 0 (Hz) and the last available unit (THz)
    i = max(0, min(i, len(units) - 1))

    scaled_freq = freq / (1000.0**i)
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

    # 1. Structural Check: Dimensions
    if mat.shape[0] != mat.shape[1]:
        report.append(f"CRITICAL: Non-square matrix detected ({mat.shape})")
        failed = True

    if n_dofs != basis.n_field:
        report.append(
            f"CRITICAL: DoF mismatch! Matrix size {n_dofs} != Basis nfield {basis.n_field}"
        )
        failed = True

    # 2. Empty Column/Row Detection (Efficiency: O(nnz))
    # For CSC, checking columns is fast via the 'indptr'
    col_counts = np.diff(mat.indptr)
    empty_cols = np.where(col_counts == 0)[0]

    # For Rows, we check the diagonal or convert to CSR (but sum is usually enough)
    # A faster way to find empty rows in CSC is to check which indices never appear in 'indices'
    row_present = np.zeros(n_dofs, dtype=bool)
    row_present[mat.indices] = True
    empty_rows = np.where(~row_present)[0]

    if len(empty_cols) > 0 or len(empty_rows) > 0:
        failed = True
        report.append(
            f"CRITICAL: Found {len(empty_cols)} empty columns and {len(empty_rows)} empty rows."
        )

        # --- PHYSICAL MAPPING ---
        # We identify if these DoFs belong to Edges or Triangles
        nedges = basis.nedges
        ntris = basis.ntris

        def map_dof_to_entity(dofs):
            edge_dofs = dofs[dofs < 2 * nedges]
            # Note: Nedelec2 mapping usually shifts by nedges or ntris based on basis.tet_to_field logic
            # Here we follow your nfield = 2*nedges + 2*ntris
            # Logic: [EdgeGroup1][TriGroup1][EdgeGroup2][TriGroup2]

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

    # 3. Numerical Check: Zero Diagonals
    # In Nedelec, even if a column isn't empty, a zero diagonal is a death sentence for most solvers.
    diag = mat.diagonal()
    zero_diag = np.where(np.isclose(diag, 0, atol=1e-15))[0]
    if len(zero_diag) > 0:
        # Filter out those already caught as empty
        true_zero_diag = np.setdiff1d(zero_diag, empty_cols)
        if len(true_zero_diag) > 0:
            failed = True
            report.append(
                f"CRITICAL: {len(true_zero_diag)} non-empty columns have zero diagonal (Numerical Singularity)."
            )

    # 4. Symmetry Check (Optional but recommended for RF)
    # RF matrices (S, M) should be symmetric unless using specific boundary conditions.
    if (mat - mat.T).nnz > 0:
        max_asym = np.max(np.abs((mat - mat.T).data)) if mat.nnz > 0 else 0
        if max_asym > 1e-12:
            report.append(f"WARNING: Matrix is asymmetric. Max diff: {max_asym}")

    # 5. Summary and Crash
    if failed:
        print("\n" + "!" * 50)
        print("MATRIX DIAGNOSTICS FAILED")
        print("!" * 50)
        for line in report:
            print(line)

        # Specific hint for GMSH users:
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

    # We will track which physical tags are "dead"
    dead_elements = {"edges": [], "tris": []}
    dead_coords = []

    for idx in empty_indices:
        # Determine if this index refers to an edge or a triangle
        # Based on your Nedelec2: [E_grp1][T_grp1][E_grp2][T_grp2]
        if idx < nedges:  # Edge Group 1
            e_id = idx
            dead_elements["edges"].append(e_id)
            dead_coords.append(basis.mesh.edge_centers[:, e_id])
        elif nedges <= idx < (nedges + ntris):  # Tri Group 1
            t_id = idx - nedges
            dead_elements["tris"].append(t_id)
            dead_coords.append(basis.mesh.tri_centers[:, t_id])
        elif (nedges + ntris) <= idx < (2 * nedges + ntris):  # Edge Group 2
            e_id = idx - (nedges + ntris)
            dead_elements["edges"].append(e_id)
            dead_coords.append(basis.mesh.edge_centers[:, e_id])
        else:  # Tri Group 2
            t_id = idx - (2 * nedges + ntris)
            dead_elements["tris"].append(t_id)
            dead_coords.append(basis.mesh.tri_centers[:, t_id])

    np.save("dead_coords.npy", np.array(dead_coords).T)
    # Convert to numpy for spatial analysis
    dead_coords = np.array(dead_coords)
    avg_pos = np.mean(dead_coords, axis=0)
    spread = np.std(dead_coords, axis=0)

    print("\n--- Spatial Autopsy Report ---")
    print(f"Dead Zone Center of Mass: {avg_pos}")
    print(f"Dead Zone Bounding Box Spread (std): {spread}")

    # Find which GMSH Physical Groups these belong to
    # We use the ftag_to_tri/etag_to_edge maps in your Mesh3D
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

    Parameters
    ----------
    points : ndarray, shape (3, N)
        3D coordinates of the point cloud.

    Returns
    -------
    basis : ndarray, shape (3, 3)
        Matrix whose columns are:
            - first principal direction (plane X axis)
            - second principal direction (plane Y axis)
            - plane normal vector (Z axis)
    """
    if points.shape[0] != 3:
        raise ValueError("Input must have shape (3, N)")

    # Compute centroid
    centroid = points.mean(axis=1, keepdims=True)

    # Center the data
    points_centered = points - centroid

    # Compute covariance matrix (3x3)
    C = (points_centered @ points_centered.T) / points.shape[1]

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(C)

    # Sort eigenvectors by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    # Columns of eigvecs = principal axes
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
        logger.info(f'{ref}: {self.ctr}: ΔT = {(time.time()-self.last_time)*1000:.2f}ms')
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

        # Process all concutors. Everything above the conductivity limit is considered pec.
        for it in range(boundary_surface.n_tris):
            if (
                sigmesh[it] > self.settings.mw_3d_peclim
                or sigmesh[it] > self.settings.mw_3d_surfimplim
            ):
                pec_ids.extend(list(nedlegfield.tri_to_field[:, it]))

        # Process all PEC Boundary Conditions
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

        # Process all port boundary Conditions
        pec_ids_set: set[int] = set(pec_ids)

        logger.trace(f".total of {len(pec_ids_set)} pec DoF to remove.")
        solve_ids = [i for i in range(nedlegfield.n_field) if i not in pec_ids_set]

        return E, B, np.array(solve_ids), nedlegfield

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
            er (np.ndarray): The relative dielectric permitivity tensor of shape (3,3,N)
            ur (np.ndarray): The relative magnetic permeability tensor of shape (3,3,N)
            sig (np.ndarray): The conductivity array of shape (N,)
            bcs (list[BoundaryCondition]): The boundary conditions
            frequency (float): The simulation frequency
            cache_matrices (bool, optional): Whether to use and cache matrices. Defaults to False.

        Returns:
            SimJob: The resultant SimJob object
        """
        
        logger.debug(f'Assembling frequency = {_format_freq(frequency)}')
        # We import these Numba compiled function here because they may not always be needed so compilation is postponed until they
        # are actually used.
        from .curlcurl import tet_mass_stiffness_matrices
        from .robinbc import assemble_robin_bc, assemble_robin_bc_bvec
        from ....mth.pairing import pair_coordinates
        from .periodicbc import gen_periodic_matrix
        from .robin_abc_order2 import abc_order_2_matrix

        #from .wpbc import assemble_wpbc
        # PREDEFINE CONSTANTS
        W0 = 2 * np.pi * frequency
        K0 = W0 / C0

        # Frequency dependence means that the material properties in the simulation are different for different frequencies.
        # If not, the matrix A which consists of to matrices A=E - k0^2 K can be cached and reused and simply scaled for each frequency
        # Conductors have frequency dependent material properties but so may any other user defined frequency dependent material paroperty.

        is_frequency_dependent = mat_assy.frequency_dependent()

        mesh = field.mesh
        
        # Prepare the 3x3 material property tensors.
        er = np.zeros((3, 3, field.mesh.n_tets), dtype=np.complex128)
        tand = np.zeros((3, 3, field.mesh.n_tets), dtype=np.complex128)
        cond = np.zeros((3, 3, field.mesh.n_tets), dtype=np.complex128)
        ur = np.zeros((3, 3, field.mesh.n_tets), dtype=np.complex128)
        
        # Take the material properties from the materials list.
        for mat, centers, ids in mat_assy.iter_materials():
            er = mat.er(frequency, er, centers, ids)
            ur = mat.ur(frequency, ur, centers, ids)
            tand = mat.tand(frequency, tand, centers, ids)
            cond = mat.cond(frequency, cond, centers, ids)
        
        # Define the complex dielectric constant:
        er = er * (1 - 1j * tand) - 1j * cond / (W0 * EPS0)

        is_frequency_dependent = is_frequency_dependent or np.any(
            (cond > 0) & (cond < self.settings.mw_3d_peclim)
        )  # type: ignore

        NF = field.n_field

        # Find Conductor domain tets:
        # 1. Simplify the mathematical check: (x > A or x > B) is identical to x > min(A, B)
        limit = min(self.settings.mw_3d_peclim, self.settings.mw_3d_surfimplim)
        conductor_tets = np.flatnonzero(cond[0, 0, :field.n_tets] > limit)

        logger.debug(f' - Total of {len(conductor_tets)} PEC Tetrahedrons')
        
        full_caching = (cache_matrices and not is_frequency_dependent)
        # Only used cahced matrices if they are there, it is asked and there are no frequency dependent material properties.
        if full_caching and self.cached_matrices is not None:
            # IF CACHED AND AVAILABLE PULL E AND B FROM CACHE
            logger.debug(" - Using cached matricies.")
            Evec, Bvec = self.cached_matrices
            K: csc_matrix = csc_axpy_same_pattern(Evec, Bvec, (-K0**2))
        else:
            # OTHERWISE, COMPUTE
            logger.debug(" - Calling matrix assembler...")
            t0 = time.time()
            # Store the E and B values in COO matrix format and the compressed-column object cscmap.
            Evec, Bvec, cscmap = tet_mass_stiffness_matrices(
                field, er, ur, conductor_tets, self.cached_cscmap
            )
            t1 = time.time()
            logger.debug(f' - Assembly speed: {(field.ntets -len(conductor_tets))/(t1-t0):.1f} tets/s')
            self.cached_cscmap = cscmap

            # COMBINE THE MASS AND STIFFNESS MATRIX
            K: csc_matrix = self.cached_cscmap.to_csc(Evec - Bvec * (K0**2))
            
            if full_caching:
                self.cached_matrices = (self.cached_cscmap.to_csc(Evec), self.cached_cscmap.to_csc(Bvec))
        
        # ISOLATE BOUNDARY CONDITIONS TO ASSEMBLE
        thin_conductor_bcs: list[ThinConductor] = [
            bc for bc in bcs if isinstance(bc, ThinConductor)
        ]
        pec_bcs: list[PEC] = [bc for bc in bcs if isinstance(bc, PEC)]
        robin_bcs: list[RobinBC] = [bc for bc in bcs if isinstance(bc, RobinBC)]
        port_bcs: list[PortBC] = [bc for bc in bcs if isinstance(bc, PortBC)]
        periodic_bcs: list[Periodic] = [bc for bc in bcs if isinstance(bc, Periodic)]

        # PREDEFINE THE FORCING VECTOR CONTAINER
        port_vectors: dict[int | float, np.ndarray] = {}
        for port in sorted(port_bcs, key=lambda x: x.port_number):
            for mat_index, mode_nr in port._iter_port_numbers():
                port_vectors[mat_index] = np.zeros((NF,), dtype=np.complex128)

        ############################################################
        #                      PEC BOUNDARY CONDITIONS             #
        ############################################################
        
        logger.debug(" - Implementing PEC Boundary Conditions.")

        # pec_ids is a list of degree of freedom indices that are 0 because
        # the E-field there is 0. For pec_ids these are references to the
        # degree of freedom, for the pec_tris these are references to the
        # triangle index. This is needed for Adaptive mesh refinement error estimates.

        pec_ids: list[int] = []
        pec_tris: list[int] = []
        # non_pec_ids: list[int] = []  # PEC DoF that aren't actually PEC

        # Conductivity above al imit, consider it all PEC
        ipec = 0

        # Volumetric PEC. Thus tets which are all PEC need to have all the
        # field indices of degrees of freedom of that tetrahedron be set to 0.
        # No E-field inside the TET

        for itet in conductor_tets:
            ipec += 1
            pec_ids.extend(field.tet_to_field[:, itet])
            for tri in field.mesh.tet_to_tri[:, itet]:
                pec_tris.append(tri)
                
        if ipec > 0:
            logger.trace(
                f" - Extended PEC with {ipec} tets with a conductivity > {self.settings.mw_3d_peclim}."
            )

        # Apply PEC boundary conditions
        for pec in pec_bcs:
            logger.trace(f" - Implementing: {pec}")
            if len(pec.tags) == 0:
                continue
            face_tags = pec.tags
            tri_ids = mesh.get_triangles(face_tags)
            edge_ids = list(mesh.tri_to_edge[:, tri_ids].flatten())

            # Set both edge and triangle PEC field degree of freedoms to zero by
            # adding it to the pec_ids list.
            for ii in edge_ids:
                eids = field.edge_to_field[:, ii]
                pec_ids.extend(list(eids))

            for ii in tri_ids:
                tids = field.tri_to_field[:, ii]
                pec_ids.extend(list(tids))

            pec_tris.extend(tri_ids)

        pec_ids: set[int] = set(pec_ids)

        ############################################################
        #                 ROBIN BOUNDARY CONDITIONS                #
        ############################################################
        # Robin boundary conditions are all ports, absorbing boundary dconditions and surface impedance etc.
        
        if len(robin_bcs) > 0:
            logger.debug(" - Assembling Robin Boundary Conditions.")
            
            # The contributions will be added to the mass+stiffness matrix A.
            # We assemble in B.

            B_matrix_robin = field.empty_tri_matrix()
            B_matrix_robin_2 = None
            
            if len(thin_conductor_bcs) > 0:
                B_matrix_robin_2 = B_matrix_robin.copy().astype(np.complex128)
            
            for bc in robin_bcs:
                logger.trace(f"   - Implementing {bc}")
                
                # Get all Robin BC face triangle and edge
                tri_ids = mesh.get_triangles(bc.tags)

                if isinstance(bc, (SurfaceImpedance, ThinConductor)):
                    dofs = set(field.tri_to_field[:, tri_ids].flatten())
                    pec_ids = pec_ids.difference(dofs)
                edge_ids = list(mesh.tri_to_edge[:, tri_ids].flatten())
                
                # Compute the γ parameter which is a generic scaling factor
                # used in the Robin boundary condition matrix etries.
                gamma = bc.get_gamma(K0)
                logger.trace(f"    - robin bc γ={gamma:.3f}")

                if bc._assemble_matrix:
                    # The assembler adds the contributions to the Bemptry matrix
                    B_matrix_robin = assemble_robin_bc(
                        field, B_matrix_robin, tri_ids, gamma
                    )  # type: ignore

                    if isinstance(bc, ThinConductor):
                        B_matrix_robin_2 = assemble_robin_bc(
                            field, B_matrix_robin_2, tri_ids, gamma
                        )
                
                # The the forcing vector b-entries for excited ports are added.
                # Don't include ScatteredField boundary conditions.
                if (
                    bc._include_force
                    and bc.driven
                    and not isinstance(bc, ScatteredField)
                ):
                    for number, Ufunc in bc._iter_modes(K0):
                        # Assemble and store in the port_vectors dictionary.
                        b_p = assemble_robin_bc_bvec(field, tri_ids, Ufunc)  # type: ignore
                        port_vectors[number] += b_p  # type: ignore
                        logger.trace(
                            f"    - included force vector term with norm {np.linalg.norm(b_p):.3f}"
                        )
                
                ## Second order absorbing boundary correction
                # Second order corrections are needed using gradient terms for improved absorption.
                # Only used in AbsorbingBoundary conditions of order 2.
                if bc._isabc:
                    if bc.order == 2:
                        c2 = bc.get_abccorr(K0)
                        logger.debug("    - Implementing second order ABC correction.")
                        mat = abc_order_2_matrix(field, tri_ids, c2)
                        B_matrix_robin += mat
                
            # Add the total contribution of B_matrix_robin to K
            add_coo_to_csc(K, B_matrix_robin, field._rows, field._cols)

            if B_matrix_robin_2 is not None:
                logger.debug("    - Assembling opposite side matrix entries.")
                rows, cols = field.empty_tri_rowcol(other_side=True)
                K += field.generate_csc(B_matrix_robin_2, (rows, cols))

            # Commented out because not yet implemented
            # for bc in [bc for bc in robin_bcs if isinstance(bc, WavePortIH)]:
            #     logger.info(f".Implementing WPBC {bc}")

            #     port_normal = mesh.inward_normal(bc.tags)

            #     mode_profile, mode_xy, kappa_m = bc.get_modepf_kappa(
            #         K0, mesh.nodes, mesh.tris
            #     )
            #     gamma_m = bc.get_gamma(K0)
            #     logger.info(f"..κm = {kappa_m} γm = {gamma_m}")
            #     # Matrix contribution (once per port)
            #     Bcoo, rows, cols, bvec = assemble_wpbc(
            #         field,
            #         tri_ids,
            #         mode_profile,
            #         mode_xy,
            #         kappa_m,
            #         gamma_m,
            #         K0,
            #         port_normal,
            #     )

            #     port_vectors[1] += bvec  # type: ignore
            #     logger.debug(
            #         f"..included force vector term with norm {np.linalg.norm(bvec):.3f}"
            #     )

            #     K += coo_matrix(
            #         (Bcoo, (rows, cols)), shape=(field.n_field, field.n_field)
            #     ).tocsc()

        ############################################################
        #                   PERIODIC BOUNDARY CONDITIONS          #
        ############################################################

        # Periodic boun
        if len(periodic_bcs) > 0:
            logger.debug("  - Implementing Periodic Boundary Conditions.")

        Pmats = []
        remove: set[int] = set()
        has_periodic = False

        # Implement periodic boundary conditions by assembling the matrix P which is non-square NxM
        # And reduces the number of degrees of freedom by linking the linked DOF on the two periodic boundaries
        # by a self term 1.0 and exp(jθ) of the linked boundary term.
        for pbc in periodic_bcs:
            logger.trace(f"    - Implementing {pbc}")
            has_periodic = True
            # Get the linked indices.
            tri_ids_1 = mesh.get_triangles(pbc.face1.tags)
            edge_ids_1 = mesh.get_edges(pbc.face1.tags)
            tri_ids_2 = mesh.get_triangles(pbc.face2.tags)
            edge_ids_2 = mesh.get_edges(pbc.face2.tags)
            dv = np.array(pbc.dv)
            logger.trace(f"    - displacement vector {dv}")
            # Pair these coordinates by computing which triangles ought to be linked to which other triangles.
            linked_tris = pair_coordinates(
                mesh.tri_centers, tri_ids_1, tri_ids_2, dv, _PBC_DSMAX
            )
            linked_edges = pair_coordinates(
                mesh.edge_centers, edge_ids_1, edge_ids_2, dv, _PBC_DSMAX
            )
            dv = np.array(pbc.dv)
            phi = pbc.phi(K0)
            logger.trace(f"    - ϕ={phi} rad/m")
            # Generate the matrix Pmat
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

        # Pmats contains a different matrix per periodic boundary. To assemble the total periodic
        # boundary, we simply multiply the matrices P1 @ P2 @ P3 etc.

        if Pmats:
            logger.trace(f"  - periodic bc removes {len(remove)} boundary DoF")
            Pmat = Pmats[0]
            for P2 in Pmats[1:]:
                Pmat = Pmat @ P2
            remove_array = np.sort(np.unique(list(remove)))
            all_indices = np.arange(NF)
            keep_indices = np.setdiff1d(all_indices, remove_array)
            Pmat = Pmat[:, keep_indices]
        else:
            Pmat = None

        ############################################################
        #                             FINALIZE                     #
        ############################################################
        
        mask = np.ones(NF, dtype=bool)
        mask[list(pec_ids)] = False
        solve_ids = np.flatnonzero(mask)

        # Because there are periodic boundaries which reduce the sets of degrees of freedom
        # We have to remap the indices that indicate which ones aren't PEC to the new counting system
        # If degree of freedom 10 is removed and 20 is PEC, then DOF number 20 becomes number 19 etc.
        if has_periodic:
            mask = np.zeros((NF,))
            mask[solve_ids] = 1
            mask = mask[keep_indices]
            solve_ids = np.argwhere(mask == 1).flatten()
            Pd = Pmat.getH()
            K = Pd @ K @ Pmat
            for key, b in port_vectors.items():
                port_vectors[key] = Pd @ b
            K = K.tocsc()

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

    def assemble_scattering_matrix(
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
            er (np.ndarray): The relative dielectric permitivity tensor of shape (3,3,N)
            ur (np.ndarray): The relative magnetic permeability tensor of shape (3,3,N)
            sig (np.ndarray): The conductivity array of shape (N,)
            bcs (list[BoundaryCondition]): The boundary conditions
            frequency (float): The simulation frequency
            cache_matrices (bool, optional): Whether to use and cache matrices. Defaults to False.

        Returns:
            SimJob: The resultant SimJob object
        """

        from .curlcurl import tet_mass_stiffness_matrices
        from .robinbc import assemble_robin_bc, assemble_robin_bc_bvec_scat
        from ....mth.pairing import pair_coordinates
        from .periodicbc import gen_periodic_matrix
        from .robin_abc_order2 import abc_order_2_matrix

        # For more detailed documentation on this function, refer to assemble_freq_matrix.
        # PREDEFINE CONSTANTS
        W0 = 2 * np.pi * frequency
        K0 = W0 / C0
        mesh = field.mesh

        is_frequency_dependent = mat_assy.frequency_dependent()

        er = np.zeros((3, 3, field.mesh.n_tets), dtype=np.complex128)
        tand = np.zeros((3, 3, field.mesh.n_tets), dtype=np.complex128)
        cond = np.zeros((3, 3, field.mesh.n_tets), dtype=np.complex128)
        ur = np.zeros((3, 3, field.mesh.n_tets), dtype=np.complex128)

        # Take the material properties from the materials list.
        for mat, centers, ids in mat_assy.iter_materials():
            er = mat.er(frequency, er, centers, ids)
            ur = mat.ur(frequency, ur, centers, ids)
            tand = mat.tand(frequency, tand, centers, ids)
            cond = mat.cond(frequency, cond, centers, ids)

        er = er * (1 - 1j * tand) - 1j * cond / (W0 * EPS0)

        is_frequency_dependent = is_frequency_dependent or np.any(
            (cond > 0) & (cond < self.settings.mw_3d_peclim)
        )  # type: ignore

        NF = field.n_field

        # Find Conductor domain tets:
        conductor_tets = []
        for itet in range(field.n_tets):
            if (
                cond[0, 0, itet] > self.settings.mw_3d_peclim
                or cond[0, 0, itet] > self.settings.mw_3d_surfimplim
            ):
                conductor_tets.append(itet)
        conductor_tets = np.array(conductor_tets)

        if (
            cache_matrices
            and not is_frequency_dependent
            and self.cached_matrices is not None
        ):
            # IF CACHED AND AVAILABLE PULL E AND B FROM CACHE
            logger.debug("Using cached matricies.")
            matrix_stiff_coo, matrix_mass_coo = self.cached_matrices
        else:
            # OTHERWISE, COMPUTE
            logger.debug("Assembling matrices")
            matrix_stiff_coo, matrix_mass_coo, cscmap = tet_mass_stiffness_matrices(
                field, er, ur, conductor_tets, self.cached_cscmap
            )
            self.cached_cscmap = cscmap
            self.cached_matrices = (matrix_stiff_coo, matrix_mass_coo)

        # COMBINE THE MASS AND STIFFNESS MATRIX
        matrix_fem: csc_matrix = self.cached_cscmap.to_csc(
            matrix_stiff_coo - matrix_mass_coo * (K0**2)
        )

        # ISOLATE BOUNDARY CONDITIONS TO ASSEMBLE
        thin_conductor_bcs: list[ThinConductor] = [
            bc for bc in bcs if isinstance(bc, ThinConductor)
        ]
        pec_bcs: list[PEC] = [bc for bc in bcs if isinstance(bc, PEC)]
        robin_bcs: list[RobinBC] = [bc for bc in bcs if isinstance(bc, RobinBC)]
        periodic_bcs: list[Periodic] = [bc for bc in bcs if isinstance(bc, Periodic)]

        background_fields: dict[tuple[float, float], np.ndarray] = {}
        b = np.zeros((NF,), dtype=np.complex128)

        ############################################################
        #                      PEC BOUNDARY CONDITIONS             #
        ############################################################

        logger.debug("Implementing PEC Boundary Conditions.")
        pec_ids: list[int] = []
        pec_tris: list[int] = []

        # Conductivity above al imit, consider it all PEC
        ipec = 0

        for itet in conductor_tets:
            ipec += 1
            pec_ids.extend(field.tet_to_field[:, itet])
            for tri in field.mesh.tet_to_tri[:, itet]:
                pec_tris.append(tri)

        if ipec > 0:
            logger.trace(
                f"Extended PEC with {ipec} tets with a conductivity > {self.settings.mw_3d_peclim}."
            )

        for pec in pec_bcs:
            logger.trace(f"Implementing: {pec}")
            if len(pec.tags) == 0:
                continue
            face_tags = pec.tags
            tri_ids = mesh.get_triangles(face_tags)
            edge_ids = list(mesh.tri_to_edge[:, tri_ids].flatten())

            for ii in edge_ids:
                eids = field.edge_to_field[:, ii]
                pec_ids.extend(list(eids))

            for ii in tri_ids:
                tids = field.tri_to_field[:, ii]
                pec_ids.extend(list(tids))

            pec_tris.extend(tri_ids)

        pec_ids: set[int] = set(pec_ids)

        ############################################################
        #                 ROBIN BOUNDARY CONDITIONS                #
        ############################################################

        if len(robin_bcs) > 0:
            logger.debug("Implementing Robin Boundary Conditions.")

            B_matrix_robin = field.empty_tri_matrix()
            B_matrix_robin_2 = None

            if len(thin_conductor_bcs) > 0:
                B_matrix_robin_2 = B_matrix_robin.copy().astype(np.complex128)

            for bc in robin_bcs:
                logger.trace(f".Implementing {bc}")

                tri_ids = mesh.get_triangles(bc.tags)

                if isinstance(bc, (SurfaceImpedance, ThinConductor)):
                    dofs = set(field.tri_to_field[:, tri_ids].flatten())
                    pec_ids = pec_ids.difference(dofs)
                edge_ids = list(mesh.tri_to_edge[:, tri_ids].flatten())

                gamma = bc.get_gamma(K0)
                logger.trace(f"..robin bc γ={gamma:.3f}")

                if not bc.pml:  # Flag that can be turned on by the user if the simulation domain termination is PML and thus no RBC is to be implemented
                    B_matrix_robin = assemble_robin_bc(
                        field, B_matrix_robin, tri_ids, gamma
                    )  # type: ignore

                    if isinstance(bc, ThinConductor):
                        B_matrix_robin_2 = assemble_robin_bc(
                            field, B_matrix_robin_2, tri_ids, gamma
                        )
                else:
                    B_matrix_robin *= 0.0

                if isinstance(bc, ScatteredField):
                    # Implement forcing vector terms
                    for bf in bc._iter_fields(K0):
                        normals = field.mesh.outward_normals(tri_ids)
                        b_p = assemble_robin_bc_bvec_scat(
                            field,
                            tri_ids,
                            bf.Uinc,
                            bf.Uinc_curl,
                            normals,
                        )  # type: ignore
                        if bf in background_fields.items():
                            background_fields[bf] += b_p
                        else:
                            background_fields[bf] = b_p  # type: ignore
                        logger.debug(
                            f".. Background field {bf} {np.linalg.norm(b_p):.3f}"
                        )

                ## Second order absorbing boundary correction
                if bc._isabc:
                    if bc.order == 2:
                        c2 = bc.o2coeffs[bc.abctype][1]
                        logger.debug("Implementing second order ABC correction.")
                        mat = abc_order_2_matrix(field, tri_ids, 1j * c2 / (K0))
                        B_matrix_robin += mat

            matrix_fem += field.generate_csc(B_matrix_robin)

            if B_matrix_robin_2 is not None:
                logger.debug("Assembling opposite side matrix entries.")
                rows, cols = field.empty_tri_rowcol(other_side=True)
                matrix_fem += field.generate_csc(B_matrix_robin_2, (rows, cols))

        if len(periodic_bcs) > 0:
            logger.debug("Implementing Periodic Boundary Conditions.")

        ############################################################
        #                   PERIODIC BOUNDARY CONDITIONS          #
        ############################################################

        Pmats = []
        remove: set[int] = set()
        has_periodic = False

        for pbc in periodic_bcs:
            logger.trace(f".Implementing {pbc}")
            has_periodic = True
            tri_ids_1 = mesh.get_triangles(pbc.face1.tags)
            edge_ids_1 = mesh.get_edges(pbc.face1.tags)
            tri_ids_2 = mesh.get_triangles(pbc.face2.tags)
            edge_ids_2 = mesh.get_edges(pbc.face2.tags)
            dv = np.array(pbc.dv)
            logger.trace(f"..displacement vector {dv}")
            linked_tris = pair_coordinates(
                mesh.tri_centers, tri_ids_1, tri_ids_2, dv, _PBC_DSMAX
            )
            linked_edges = pair_coordinates(
                mesh.edge_centers, edge_ids_1, edge_ids_2, dv, _PBC_DSMAX
            )
            dv = np.array(pbc.dv)
            phi = pbc.phi(K0)
            logger.trace(f"..ϕ={phi} rad/m")
            Pmat, rows = gen_periodic_matrix(
                tri_ids_1,
                edge_ids_1,
                field.tri_to_field,
                field.edge_to_field,
                linked_tris,
                linked_edges,
                field.n_field,
                phi,
            )
            remove.update(rows)
            Pmats.append(Pmat)

        if Pmats:
            logger.trace(f".periodic bc removes {len(remove)} boundary DoF")
            Pmat = Pmats[0]
            for P2 in Pmats[1:]:
                Pmat = Pmat @ P2
            remove_array = np.sort(np.unique(list(remove)))
            all_indices = np.arange(NF)
            keep_indices = np.setdiff1d(all_indices, remove_array)
            Pmat = Pmat[:, keep_indices]
        else:
            Pmat = None

        ############################################################
        #                             FINALIZE                     #
        ############################################################

        mask = np.ones(NF, dtype=bool)
        mask[list(pec_ids)] = False
        solve_ids = np.flatnonzero(mask)


        if has_periodic:
            mask = np.zeros((NF,))
            mask[solve_ids] = 1
            mask = mask[keep_indices]
            solve_ids = np.argwhere(mask == 1).flatten()
            Pd = Pmat.getH()
            matrix_fem = Pd @ matrix_fem @ Pmat
            for key, b in background_fields:
                background_fields[key] = Pd @ b

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
        The frequency-dependent properties are simply calculated once for the given frequency

        Args:
            field (Nedelec2): The Nedelec2 field
            er (np.ndarray): The relative permittivity tensor in shape (3,3,N)
            ur (np.ndarray): The relative permeability tensor in shape (3,3,N)
            sig (np.ndarray): The conductivity scalar in array (N,)
            bcs (list[BoundaryCondition]): The list of boundary conditions
            frequency (float): The compilation frequency (for material properties only)

        Returns:
            SimJob: The resultant simulation job
        """

        from .curlcurl import tet_mass_stiffness_matrices

        from .robinbc import assemble_robin_bc
        from ....mth.pairing import pair_coordinates
        from .periodicbc import gen_periodic_matrix
        from .robin_abc_order2 import abc_order_2_matrix

        # Precompute constants
        mesh = field.mesh
        w0 = 2 * np.pi * frequency
        k0 = w0 / C0

        # Reserve empty lists for the material properties
        er = np.zeros((3, 3, field.mesh.n_tets), dtype=np.complex128)
        tand = np.zeros((3, 3, field.mesh.n_tets), dtype=np.complex128)
        cond = np.zeros((3, 3, field.mesh.n_tets), dtype=np.complex128)
        ur = np.zeros((3, 3, field.mesh.n_tets), dtype=np.complex128)

        # Store material properties in the associated indices
        # Take the material properties from the materials list.
        for mat, centers, ids in mat_assy.iter_materials():
            er = mat.er(frequency, er, centers, ids)
            ur = mat.ur(frequency, ur, centers, ids)
            tand = mat.tand(frequency, tand, centers, ids)
            cond = mat.cond(frequency, cond, centers, ids)

        # Compute the complex dielectric constant
        er = er * (1 - 1j * tand) - 1j * cond / (w0 * EPS0)

        # Find Conductor domain tets:
        conductor_tets = []
        for itet in range(field.n_tets):
            if (
                cond[0, 0, itet] > self.settings.mw_3d_peclim
                or cond[0, 0, itet] > self.settings.mw_3d_surfimplim
            ):
                conductor_tets.append(itet)
        conductor_tets = np.array(conductor_tets)

        # Start the full assembly process
        logger.debug("Assembling matrices")

        # Assemble the E and B COO matrices plus cscmapping
        matrix_stiff, matrix_mass, cscmapping = tet_mass_stiffness_matrices(
            field, er, ur, conductor_tets
        )

        matrix_stiff = cscmapping.to_csc(matrix_stiff)
        matrix_mass = cscmapping.to_csc(matrix_mass)
        self.cached_matrices = (matrix_stiff, matrix_mass)

        # Number of degrees of freedom
        NDoF = matrix_stiff.shape[0]

        # Extract lists of relevant boundary condition categories
        thin_conductor_bcs: list[ThinConductor] = [
            bc for bc in bcs if isinstance(bc, ThinConductor)
        ]
        pecs: list[PEC] = [bc for bc in bcs if isinstance(bc, PEC)]
        robin_bcs: list[RobinBC] = [bc for bc in bcs if isinstance(bc, RobinBC)]  # type: ignore
        periodic: list[Periodic] = [bc for bc in bcs if isinstance(bc, Periodic)]

        # Process all PEC Boundary Conditions
        pec_ids: list = []

        logger.debug("Implementing PEC Boundary Conditions.")

        # Conductivity above a limit, consider it all PEC
        for itet in conductor_tets:
            pec_ids.extend(field.tet_to_field[:, itet])

        # PEC Boundary conditions
        for pec in pecs:
            if len(pec.tags) == 0:
                continue
            face_tags = pec.tags
            tri_ids = mesh.get_triangles(face_tags)
            edge_ids = list(mesh.tri_to_edge[:, tri_ids].flatten())

            for ii in edge_ids:
                eids = field.edge_to_field[:, ii]
                pec_ids.extend(list(eids))

            for ii in tri_ids:
                tids = field.tri_to_field[:, ii]
                pec_ids.extend(list(tids))

        pec_ids: set[int] = set(pec_ids)

        # Robin BCs
        if len(robin_bcs) > 0:
            logger.debug("Implementing Robin Boundary Conditions.")

            B_matrix_robin = field.empty_tri_matrix()
            B_matrix_robin_2 = None

            if len(thin_conductor_bcs) > 0:
                B_matrix_robin_2 = B_matrix_robin.copy().astype(np.complex128)

            for bc in robin_bcs:
                tri_ids = mesh.get_triangles(bc.tags)
                if isinstance(bc, (SurfaceImpedance, ThinConductor)):
                    dofs = set(field.tri_to_field[:, tri_ids].flatten())
                    pec_ids = pec_ids.difference(dofs)
                edge_ids = list(mesh.tri_to_edge[:, tri_ids].flatten())

                gamma = bc.get_gamma(k0)

                if bc._assemble_matrix:
                    # The assembler adds the contributions to the Bemptry matrix
                    B_matrix_robin = assemble_robin_bc(
                        field, B_matrix_robin, tri_ids, gamma
                    )  # type: ignore

                    if isinstance(bc, ThinConductor):
                        B_matrix_robin_2 = assemble_robin_bc(
                            field, B_matrix_robin_2, tri_ids, gamma
                        )
            ## Second order absorbing boundary correction
            if bc._isabc:
                if bc.order == 2:
                    c2 = bc.o2coeffs[bc.abctype][1]
                    logger.debug("Implementing second order ABC correction.")
                    mat = abc_order_2_matrix(field, tri_ids, 1j * c2 / k0)
                    B_matrix_robin += mat

            matrix_mass -= field.generate_csc(B_matrix_robin) / (k0**2)
            if B_matrix_robin_2 is not None:
                logger.debug("Assembling opposite side matrix entries.")
                rows, cols = field.empty_tri_rowcol(other_side=True)
                matrix_mass -= field.generate_csc(B_matrix_robin_2, (rows, cols)) / (
                    k0**2
                )

            del B_matrix_robin

        if len(periodic) > 0:
            logger.debug("Implementing Periodic Boundary Conditions.")

        # Periodic BCs
        Pmats = []
        remove = set()
        has_periodic = False

        for bcp in periodic:
            has_periodic = True
            tri_ids_1 = mesh.get_triangles(bcp.face1.tags)
            edge_ids_1 = mesh.get_edges(bcp.face1.tags)
            tri_ids_2 = mesh.get_triangles(bcp.face2.tags)
            edge_ids_2 = mesh.get_edges(bcp.face2.tags)
            dv = np.array(bcp.dv)
            linked_tris = pair_coordinates(
                mesh.tri_centers, tri_ids_1, tri_ids_2, dv, _PBC_DSMAX
            )
            linked_edges = pair_coordinates(
                mesh.edge_centers, edge_ids_1, edge_ids_2, dv, _PBC_DSMAX
            )
            dv = np.array(bcp.dv)
            phi = bcp.phi(k0)

            Pmat, rows = gen_periodic_matrix(
                tri_ids_1,
                edge_ids_1,
                field.tri_to_field,
                field.edge_to_field,
                linked_tris,
                linked_edges,
                field.n_field,
                phi,
            )
            remove.update(rows)
            Pmats.append(Pmat)

        if Pmats:
            Pmat = Pmats[0]
            for P2 in Pmats[1:]:
                Pmat = Pmat @ P2
            Pmat = Pmat.tocsr()
            remove_array = np.sort(np.array(list(remove)))
            all_indices = np.arange(NDoF)
            keep_indices = np.setdiff1d(all_indices, remove_array)
            Pmat = Pmat[:, keep_indices]
        else:
            Pmat = None

        # Final assembly operations
        solve_ids = np.array(
            [i for i in range(matrix_stiff.shape[0]) if i not in pec_ids]
        )

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
