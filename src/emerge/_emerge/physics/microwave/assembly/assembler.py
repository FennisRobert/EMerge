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

import numpy as np
from ..microwave_bc import PEC, BoundaryCondition, RectangularWaveguide, RobinBC, PortBC, Periodic, MWBoundaryConditionSet
from ....elements.nedelec2 import Nedelec2
from ....elements.nedleg2 import NedelecLegrange2
from ....mth.optimized import gaus_quad_tri
from ....mth.pairing import pair_coordinates

from scipy.sparse import csr_matrix
from loguru import logger
from ..simjob import SimJob
from .periodicbc import gen_periodic_matrix
from ....const import MU0, EPS0, C0


############################################################
#                         FUNCTIONS                        #
############################################################

def diagnose_matrix(mat: np.ndarray) -> None:

    if not isinstance(mat, np.ndarray):
        logger.debug('Converting sparse array to flattened array')
        mat = mat[mat.nonzero()].A1
        #mat = np.array(nonzero_mat)
    
    ''' Prints all indices of Nan's and infinities in a matrix '''
    ids = np.where(np.isnan(mat))
    if len(ids[0]) > 0:
        logger.error(f'Found NaN at {ids}')
    ids = np.where(np.isinf(mat))
    if len(ids[0]) > 0:
        logger.error(f'Found Inf at {ids}')
    ids = np.where(np.abs(mat) > 1e10)
    if len(ids[0]) > 0:
        logger.error(f'Found large values at {ids}')
    logger.info('Diagnostics finished')

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

class Assembler:
    """The assembler class is responsible for FEM EM problem assembly.

    It stores some cached properties to accellerate preformance.
    """
    def __init__(self):
        
        self.cached_matrices = None
        self.conductivity_limit = 1e7
        # Currently not used.
        #self._Pmat_cache: dict[tuple[int,int], csr_matrix] = dict()
        #self._remove_cache: list[int] = []
    
    def assemble_bma_matrices(self,
                              field: Nedelec2,
                        er: np.ndarray, 
                        ur: np.ndarray, 
                        sig: np.ndarray,
                        k0: float,
                        port: PortBC,
                        bc_set: MWBoundaryConditionSet) -> tuple[csr_matrix, csr_matrix, np.ndarray, NedelecLegrange2]:
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
        logger.debug('Assembling Boundary Mode Matrices')

        bcs = bc_set.boundary_conditions
        mesh = field.mesh
        tri_ids = mesh.get_triangles(port.tags)
        logger.trace(f'.boundary face has {len(tri_ids)} triangles.')

        origin = tuple([c-n for c,n in zip(port.cs.origin, port.cs.gzhat)])
        logger.trace(f'.boundary origin {origin}')

        boundary_surface = mesh.boundary_surface(port.tags, origin)
        nedlegfield = NedelecLegrange2(boundary_surface, port.cs)

        ermesh = er[:,:,tri_ids]
        urmesh = ur[:,:,tri_ids]
        sigmesh = sig[tri_ids]
        ermesh = ermesh - 1j * sigmesh/(k0*C0*EPS0)

        logger.trace(f'.assembling matrices for {nedlegfield} at k0={k0:.2f}')
        E, B = generelized_eigenvalue_matrix(nedlegfield, ermesh, urmesh, port.cs._basis, k0)

        # TODO: Simplified to all "conductors" loosely defined. Must change to implementing line robin boundary conditions.
        pecs: list[BoundaryCondition] = bc_set.get_conductors()#[bc for bc in bcs if isinstance(bc,PEC)]
        if len(pecs) > 0:
            logger.debug(f'.total of equiv. {len(pecs)} PEC BCs implemented')

        pec_ids = []

        # Process all concutors. Everything above the conductivity limit is considered pec.
        for it in range(boundary_surface.n_tris):
            if sigmesh[it] > self.conductivity_limit:
                pec_ids.extend(list(nedlegfield.tri_to_field[:,it]))

        # Process all PEC Boundary Conditions
        for pec in pecs:
            logger.trace(f'.implementing {pec}')
            face_tags = pec.tags
            tri_ids = mesh.get_triangles(face_tags)
            edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())
            for ii in edge_ids:
                i2 = nedlegfield.mesh.from_source_edge(ii)
                if i2 is None:
                    continue
                eids = nedlegfield.edge_to_field[:, i2]
                pec_ids.extend(list(eids))

        # Process all port boundary Conditions
        pec_ids_set: set[int] = set(pec_ids)

        logger.trace(f'.total of {len(pec_ids_set)} pec DoF to remove.')
        solve_ids = [i for i in range(nedlegfield.n_field) if i not in pec_ids_set]

        return E, B, np.array(solve_ids), nedlegfield

    def assemble_freq_matrix(self, field: Nedelec2, 
                        er: np.ndarray, 
                        ur: np.ndarray, 
                        sig: np.ndarray,
                        bcs: list[BoundaryCondition],
                        frequency: float,
                        cache_matrices: bool = False) -> SimJob:
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
        from .robinbc import assemble_robin_bc, assemble_robin_bc_excited

        # PREDEFINE CONSTANTS
        W0 = 2*np.pi*frequency
        K0 = W0/C0

        mesh = field.mesh
        er = er - 1j*sig/(W0*EPS0)*np.repeat(np.eye(3)[:, :, np.newaxis], er.shape[2], axis=2)
        is_frequency_dependent: bool = np.any((sig > 0) & (sig < self.conductivity_limit)) # type: ignore
        

        if cache_matrices and not is_frequency_dependent and self.cached_matrices is not None:
            # IF CACHED AND AVAILABLE PULL E AND B FROM CACHE
            logger.debug('Using cached matricies.')
            E, B = self.cached_matrices
        else:
            # OTHERWISE, COMPUTE
            logger.debug('Assembling matrices')
            E, B = tet_mass_stiffness_matrices(field, er, ur)
            self.cached_matrices = (E, B)

        # COMBINE THE MASS AND STIFFNESS MATRIX
        K: csr_matrix = (E - B*(K0**2)).tocsr()

        NF = E.shape[0]

        # ISOLATE BOUNDARY CONDITIONS TO ASSEMBLE
        pec_bcs: list[PEC] = [bc for bc in bcs if isinstance(bc,PEC)]
        robin_bcs: list[RobinBC] = [bc for bc in bcs if isinstance(bc,RobinBC)]
        port_bcs: list[PortBC] = [bc for bc in bcs if isinstance(bc, PortBC)]
        periodic_bcs: list[Periodic] = [bc for bc in bcs if isinstance(bc, Periodic)]

        # PREDEFINE THE FORCING VECTOR CONTAINER
        b = np.zeros((E.shape[0],)).astype(np.complex128)
        port_vectors = {port.port_number: np.zeros((E.shape[0],)).astype(np.complex128) for port in port_bcs}
        

        ############################################################
        #                      PEC BOUNDARY CONDITIONS             #
        ############################################################

        logger.debug('Implementing PEC Boundary Conditions.')
        pec_ids: list[int] = []
        # Conductivity above al imit, consider it all PEC
        ipec = 0
        for itet in range(field.n_tets):
            if sig[itet] > self.conductivity_limit:
                ipec+=1
                pec_ids.extend(field.tet_to_field[:,itet])
        if ipec>0:
            logger.trace(f'Extended PEC with {ipec} tets with a conductivity > {self.conductivity_limit}.')

        for pec in pec_bcs:
            logger.trace(f'Implementing: {pec}')
            if len(pec.tags)==0:
                continue
            face_tags = pec.tags
            tri_ids = mesh.get_triangles(face_tags)
            edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())

            for ii in edge_ids:
                eids = field.edge_to_field[:, ii]
                pec_ids.extend(list(eids))

            for ii in tri_ids:
                tids = field.tri_to_field[:, ii]
                pec_ids.extend(list(tids))


        ############################################################
        #                     ROBIN BOUNDARY CONDITIONS            #
        ############################################################

        if len(robin_bcs) > 0:
            logger.debug('Implementing Robin Boundary Conditions.')
        
            gauss_points = gaus_quad_tri(4)
            Bempty = field.empty_tri_matrix()
            for bc in robin_bcs:
                logger.trace(f'.Implementing {bc}')
                for tag in bc.tags:
                    face_tags = [tag,]

                    tri_ids = mesh.get_triangles(face_tags)
                    nodes = mesh.get_nodes(face_tags)
                    edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())

                    gamma = bc.get_gamma(K0)
                    logger.trace(f'..robin bc γ={gamma:.3f}')

                    def Ufunc(x,y,z): 
                        return bc.get_Uinc(x,y,z,K0)
                    
                    if bc._include_force:
                        Bempty, b_p = assemble_robin_bc_excited(field, Bempty, tri_ids, Ufunc, gamma, gauss_points) # type: ignore
                        port_vectors[bc.port_number] += b_p # type: ignore
                        logger.trace(f'..included force vector term with norm {np.linalg.norm(b_p):.3f}')
                    else:
                        Bempty = assemble_robin_bc(field, Bempty, tri_ids, gamma) # type: ignore
            B_p = field.generate_csr(Bempty)
            K = K + B_p
        
        if len(periodic_bcs) > 0:
            logger.debug('Implementing Periodic Boundary Conditions.')


        ############################################################
        #                   PERIODIC BOUNDARY CONDITIONS          #
        ############################################################

        Pmats = []
        remove: set[int] = set()
        has_periodic = False

        for pbc in periodic_bcs:
            logger.trace(f'.Implementing {pbc}')
            has_periodic = True
            tri_ids_1 = mesh.get_triangles(pbc.face1.tags)
            edge_ids_1 = mesh.get_edges(pbc.face1.tags)
            tri_ids_2 = mesh.get_triangles(pbc.face2.tags)
            edge_ids_2 = mesh.get_edges(pbc.face2.tags)
            dv = np.array(pbc.dv)
            logger.trace(f'..displacement vector {dv}')
            linked_tris = pair_coordinates(mesh.tri_centers, tri_ids_1, tri_ids_2, dv, 1e-9)
            linked_edges = pair_coordinates(mesh.edge_centers, edge_ids_1, edge_ids_2, dv, 1e-9)
            dv = np.array(pbc.dv)
            phi = pbc.phi(K0)
            logger.trace(f'..ϕ={phi} rad/m')
            Pmat, rows = gen_periodic_matrix(tri_ids_1,
                                       edge_ids_1,
                                       field.tri_to_field,
                                       field.edge_to_field,
                                       linked_tris,
                                       linked_edges,
                                       field.n_field,
                                       phi)
            remove.update(rows)
            Pmats.append(Pmat)

        if Pmats:
            logger.trace(f'.periodic bc removes {len(remove)} boundary DoF')
            Pmat = Pmats[0]
            for P2 in Pmats[1:]:
                Pmat = Pmat @ P2
            remove_array = np.sort(np.unique(list(remove)))
            all_indices = np.arange(NF)
            keep_indices = np.setdiff1d(all_indices, remove_array)
            Pmat = Pmat[:,keep_indices]
        else:
            Pmat = None
        

        ############################################################
        #                             FINALIZE                     #
        ############################################################

        pec_ids_set = set(pec_ids)
        solve_ids = np.array([i for i in range(E.shape[0]) if i not in pec_ids_set])
        
        if has_periodic:
            mask = np.zeros((NF,))
            mask[solve_ids] = 1
            mask = mask[keep_indices]
            solve_ids = np.argwhere(mask==1).flatten()

        logger.debug(f'Number of tets: {mesh.n_tets:,}')
        logger.debug(f'Number of DoF: {K.shape[0]:,}')
        logger.debug(f'Number of non-zero: {K.nnz:,}')
        simjob = SimJob(K, b, K0*299792458/(2*np.pi), True)
        
        simjob.port_vectors = port_vectors
        simjob.solve_ids = solve_ids

        if has_periodic:
            simjob.P = Pmat
            simjob.Pd = Pmat.getH()
            simjob.has_periodic = has_periodic

        return simjob
    
    def assemble_eig_matrix(self, field: Nedelec2, 
                        er: np.ndarray, 
                        ur: np.ndarray, 
                        sig: np.ndarray,
                        bcs: list[BoundaryCondition],
                        frequency: float) -> SimJob:
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

        mesh = field.mesh
        w0 = 2*np.pi*frequency
        k0 = w0/C0

        er = er - 1j*sig/(w0*EPS0)*np.repeat(np.eye(3)[:, :, np.newaxis], er.shape[2], axis=2)
        
        logger.debug('Assembling matrices')
        E, B = tet_mass_stiffness_matrices(field, er, ur)
        self.cached_matrices = (E, B)

        NF = E.shape[0]

        pecs: list[PEC] = [bc for bc in bcs if isinstance(bc,PEC)]
        robin_bcs: list[RectangularWaveguide] = [bc for bc in bcs if isinstance(bc,RobinBC)] # type: ignore
        periodic: list[Periodic] = [bc for bc in bcs if isinstance(bc, Periodic)]

        # Process all PEC Boundary Conditions
        pec_ids: list = []
        
        logger.debug('Implementing PEC Boundary Conditions.')
        
        # Conductivity above a limit, consider it all PEC
        for itet in range(field.n_tets):
            if sig[itet] > self.conductivity_limit:
                pec_ids.extend(field.tet_to_field[:,itet])
        
        # PEC Boundary conditions
        for pec in pecs:
            if len(pec.tags)==0:
                continue
            face_tags = pec.tags
            tri_ids = mesh.get_triangles(face_tags)
            edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())

            for ii in edge_ids:
                eids = field.edge_to_field[:, ii]
                pec_ids.extend(list(eids))

            for ii in tri_ids:
                tids = field.tri_to_field[:, ii]
                pec_ids.extend(list(tids))

        # Robin BCs
        if len(robin_bcs) > 0:
            logger.debug('Implementing Robin Boundary Conditions.')
        
        if len(robin_bcs) > 0:
            logger.debug('Implementing Robin Boundary Conditions.')
        
            gauss_points = gaus_quad_tri(4)
            Bempty = field.empty_tri_matrix()
            for bc in robin_bcs:

                for tag in bc.tags:
                    face_tags = [tag,]

                    tri_ids = mesh.get_triangles(face_tags)
                    nodes = mesh.get_nodes(face_tags)
                    edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())

                    gamma = bc.get_gamma(k0)
                    
                    ibasis = bc.get_inv_basis()
                    if ibasis is None:
                        basis = plane_basis_from_points(mesh.nodes[:,nodes]) + 1e-16
                        ibasis = np.linalg.pinv(basis)
                    
                    Bempty = assemble_robin_bc(field, Bempty, tri_ids, gamma) # type: ignore
            B_p = field.generate_csr(Bempty)
            B = B + B_p
        
        if len(periodic) > 0:
            logger.debug('Implementing Periodic Boundary Conditions.')

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
            linked_tris = pair_coordinates(mesh.tri_centers, tri_ids_1, tri_ids_2, dv, 1e-9)
            linked_edges = pair_coordinates(mesh.edge_centers, edge_ids_1, edge_ids_2, dv, 1e-9)
            dv = np.array(bcp.dv)
            phi = bcp.phi(k0)
            
            Pmat, rows = gen_periodic_matrix(tri_ids_1,
                                       edge_ids_1,
                                       field.tri_to_field,
                                       field.edge_to_field,
                                       linked_tris,
                                       linked_edges,
                                       field.n_field,
                                       phi)
            remove.update(rows)
            Pmats.append(Pmat)
        
        if Pmats:
            Pmat = Pmats[0]
            for P2 in Pmats[1:]:
                Pmat = Pmat @ P2
            Pmat = Pmat.tocsr()
            remove_array = np.sort(np.array(list(remove)))
            all_indices = np.arange(NF)
            keep_indices = np.setdiff1d(all_indices, remove_array)
            Pmat = Pmat[:,keep_indices]
        else:
            Pmat = None
        
        pec_ids_set = set(pec_ids)
        solve_ids = np.array([i for i in range(E.shape[0]) if i not in pec_ids_set])
        
        if has_periodic:
            mask = np.zeros((NF,))
            mask[solve_ids] = 1
            mask = mask[keep_indices]
            solve_ids = np.argwhere(mask==1).flatten()

        logger.debug(f'Number of tets: {mesh.n_tets}')
        logger.debug(f'Number of DoF: {E.shape[0]}')
        simjob = SimJob(E, None, frequency, False, B=B)
        
        simjob.solve_ids = solve_ids

        if has_periodic:
            simjob.P = Pmat
            simjob.Pd = Pmat.getH()
            simjob.has_periodic = has_periodic

        return simjob