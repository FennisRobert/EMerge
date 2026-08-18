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
from ....geometry import GeoVolume
from ....selection import DomainSelection

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
from .dd_tools import generate_reduction_matrix
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

class CNOSDDAssembler:
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


        self.assy_field: Nedelec2 = None
        self.assy_bcs: list[BoundaryCondition] = None
        self.assy_freq: float = None
        self.assy_pecdof: list[int] = []
        self.assy_port_vectors: dict[int | float, np.ndarray] = None

    def assemble_freq_matrix(
        self,
        field: Nedelec2,
        mat_assy: MaterialAssignment,
        bcs: list[BoundaryCondition],
        frequency: float,
        domains: list[GeoVolume | DomainSelection]
    ) -> SimJob:
        
        self.assy_field = field
        self.assy_bcs = bcs
        self.assy_freq = frequency
        
        logger.debug(f'Running DD Assembly for frequency = {_format_freq(frequency)}')

        # We import these Numba compiled function here because they may not always be needed so compilation is postponed until they
        # are actually used.
        from .curlcurl import tet_mass_stiffness_matrices_subdom

        W0 = 2 * np.pi * frequency
        K0 = W0 / C0
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

        limit = min(self.settings.mw_3d_peclim, self.settings.mw_3d_surfimplim)
        conductor_tets = np.flatnonzero(cond[0, 0, :field.n_tets] > limit)
        
        # Domain generation
        domains = {i: dom for i,dom in enumerate(domains)}
        domain_tets: dict[int, np.ndarray] = {i: mesh.get_tetrahedra(dom.tags) for i,dom in domains.items()}

        logger.debug('Domains:')
        for i,dom in domains.items():
            logger.debug(f' domain {i}: {dom}')

        n_domains = len(domains)

        # Domain interfaces
        intf_2_tags: dict[tuple[int,int], list[int]] = dict()
        domain_links: dict[int, list[int]] = {i: [] for i in range(n_domains)}

        for i1 in range(n_domains-1):
            for i2 in range(i1+1,n_domains):
                logger.debug(f'Testing interface between {i1}, {i2}')
                tris_dom1 = mesh.get_triangles(domains[i1].boundary().tags)
                tris_dom2 = mesh.get_triangles(domains[i2].boundary().tags)
                intersection = set(tris_dom1).intersection(tris_dom2)
                if len(intersection) > 0:
                    domain_links[i1].append(i2)
                    domain_links[i2].append(i1)
                    intf_2_tags[(i1,i2)] = [int(x) for x in sorted(list(intersection))]
                    logger.debug(f' - Domains {i1},{i2} share a boundary in tris: {intf_2_tags[(i1,i2)]}')

        # Sort domain links:
        domain_links = {i: sorted(ls) for i,ls in domain_links.items()}
        for i, adj in domain_links.items():
            logger.debug(f' - Domain {i} linked to: {adj}')

        if2id: dict[tuple[int,int], int] = {key: i for i,key in enumerate(intf_2_tags.keys())}
        id2if: dict[int, tuple[int,int]] = {i: key for key,i in if2id.items()}

        n_interfaces = len(if2id)
        for i in range(n_interfaces):
            logger.debug(f' - Interface {i} = {id2if[i]}')

        # Creating ordered submatrix lists
        idom_2_ifs: dict[int, list[int]] = dict()
        for idom in range(n_domains):
            ifs = []
            for idom_adj in domain_links[idom]:
                pair = (min(idom, idom_adj), max(idom, idom_adj))
                ifs.append(if2id.get(pair, None))
            idom_2_ifs[idom] = sorted([intf for intf in ifs if intf is not None])
            logger.debug(f' - Domain {idom} has interfaces: {idom_2_ifs[idom]}')

        # We now have the sharing faces
        # Time for subdomain assemblies

        # First we construct the reduction matrices Rs for the domains
        R_sub: dict[int, csc_matrix] = dict()
        A_sub: dict[int, csc_matrix] = dict()
        D_s_set: dict[int, np.ndarray] = dict()
        D_all = np.arange(field.n_field)

        for i,dom in domains.items():
            tets = domain_tets[i]
            Evec, Bvec, cscmap = tet_mass_stiffness_matrices_subdom(field, er, ur, tets, conductor_tets, None)

            D_s = np.sort(np.unique(field.tet_to_field[:,tets]))
            D_s_set[i] = D_s
            R_sub[i] = generate_reduction_matrix(D_all, D_s)
            logger.debug(f' - Rs={i}.shape = {R_sub[i].shape}')
            A_sub[i] = cscmap.to_csc(Evec - Bvec *(K0**2))[tets,:][:,tets]

        for i, mat in A_sub.items():
            logger.debug(f' - Domain {i} A.shape: {mat.shape}')

        # Now we generate the further reduction matrix from Ωs to Γis
        D_g: dict[int, np.ndarray] = dict()
        R_sg: dict[int, dict[int, csc_matrix]] = {i: dict() for i in range(n_domains)}

        for i_itf in range(n_interfaces):
            tri_ids = intf_2_tags[id2if[i_itf]]
            itf_dof = np.sort(np.unique(field.tri_to_field[:,tri_ids]))
            D_g[i_itf] = itf_dof

        for i_dom, adj_doms in domain_links.items():
            for i_dom_2 in adj_doms:
                itf_tuple = (min(i_dom, i_dom_2), max(i_dom, i_dom_2))
                itf_index = if2id[itf_tuple]
                R_sg[i_dom][itf_index] = generate_reduction_matrix(D_s_set[i_dom], D_g[itf_index])

                logger.debug(f' - RM {i_dom}({itf_index}) has shape {R_sg[i_dom][itf_index].shape}')

        # Next we create the information exchage vectors gij

        gijs: dict[tuple[int,int], np.ndarray] = dict()
        for i_intf in range(n_interfaces):
            i1, i2 = id2if[i_intf]
            gijs[(i1,i2)] = np.zeros_like(D_g[i_intf], dtype=np.complex128)
            gijs[(i2,i1)] = np.zeros_like(D_g[i_intf], dtype=np.complex128)

        for key,value in gijs.items():
            print(f'g_{key[0]},{key[1]} = {value.shape}')

        # now we start doing the full BC Assembly routine, PEC and Robin BC


        ############################################################
        #                      BOUNDARY CONDITION PREP             #
        ############################################################

        # ISOLATE BOUNDARY CONDITIONS TO ASSEMBLE
        thin_conductor_bcs: list[ThinConductor] = [
            bc for bc in bcs if isinstance(bc, ThinConductor)
        ]
        pec_bcs: list[PEC] = [bc for bc in bcs if isinstance(bc, PEC)]
        robin_bcs: list[RobinBC] = [bc for bc in bcs if isinstance(bc, RobinBC)]
        port_bcs: list[PortBC] = [bc for bc in bcs if isinstance(bc, PortBC)]
        
        self.assy_port_vectors = {}
        for port in sorted(port_bcs, key=lambda x: x.port_number):
            for mat_index, mode_nr in port._iter_port_numbers():
                self.assy_port_vectors[mat_index] = np.zeros((self.assy_field.n_field,), dtype=np.complex128)

        # PEC
        self._get_pec_ids(pec_bcs, conductor_tets)
        
        # ROBIN BC of the actual EM problem
        for bc in robin_bcs:
            K = self._get_robin_mat(bc)
            # K is in global coordinates, we now have to figure out in which domain it is
            # We can make a simple assumption in this case, We know that any domain only has
            # DD boundaries Γ plus internal boundaries plus external boundaries that are not adjascent
            # to other domains.
            # Any triangle on our robin boundary condition boundary should for now at most be inside one domain
            # So we may pick any degree of freedom and just test if its inside some domain
            sample_dof = 9
    def _get_robin_mat(self, bc: RobinBC) -> csc_matrix:
        
        from .robinbc import assemble_robin_bc, assemble_robin_bc_bvec
        from .robin_abc_order2 import abc_order_2_matrix
        
        
        ############################################################
        #                 ROBIN BOUNDARY CONDITIONS                #
        ############################################################
        
        # Robin boundary conditions are all ports, absorbing boundary dconditions and surface impedance etc.
        field = self.assy_field
        mesh = field.mesh
        K0 = 2*np.pi*self.assy_freq/C0
        
        NF = field.n_field
        B_matrix_robin = field.empty_tri_matrix()
        K = csc_matrix((NF,NF), dtype=np.complex128)
        logger.trace(f"   - Implementing {bc}")
        
        # Get all Robin BC face triangle and edge
        tri_ids = mesh.get_triangles(bc.tags)

        if isinstance(bc, (SurfaceImpedance, ThinConductor)):
            dofs = set(field.tri_to_field[:, tri_ids].flatten())
            self.assy_pecdof = self.assy_pecdof.difference(dofs)
        
        # Compute the γ parameter which is a generic scaling factor
        # used in the Robin boundary condition matrix etries.
        gamma = bc.get_gamma(K0)
        logger.trace(f"    - robin bc γ={gamma:.3f}")

        if bc._assemble_matrix:
            # The assembler adds the contributions to the Bemptry matrix
            B_matrix_robin = assemble_robin_bc(
                field, B_matrix_robin, tri_ids, gamma
            )  # type: ignore

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
                self.assy_port_vectors[number] += b_p  # type: ignore
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
        K = K + csc_matrix((B_matrix_robin, (NF, NF)), dtype=np.complex128)
        return K
        #add_coo_to_csc(K, B_matrix_robin, field._rows, field._cols)
        

    def _get_pec_ids(self, pec_bcs: list[PEC], conductor_tets: list[int]) -> list[int]:
        
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
            pec_ids.extend(self.assy_field.tet_to_field[:, itet])
            for tri in self.assy_field.mesh.tet_to_tri[:, itet]:
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
            tri_ids = self.assy_field.mesh.get_triangles(face_tags)
            edge_ids = list(self.assy_field.mesh.tri_to_edge[:, tri_ids].flatten())

            # Set both edge and triangle PEC field degree of freedoms to zero by
            # adding it to the pec_ids list.
            for ii in edge_ids:
                eids = self.assy_field.edge_to_field[:, ii]
                pec_ids.extend(list(eids))

            for ii in tri_ids:
                tids = self.assy_field.tri_to_field[:, ii]
                pec_ids.extend(list(tids))

            pec_tris.extend(tri_ids)

        self.assy_pecdof: set[int] = set(pec_ids)