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

from ...mesher import Mesher
from ...material import Material
from ...mesh3d import Mesh3D
from ...bc import BoundaryCondition, PEC, ModalPort, LumpedPort, PortBC
from .emdata import EMSimData
from ...elements.femdata import FEMBasis
from .assembler import Assembler
from ...solver import DEFAULT_ROUTINE, SolveRoutine
from ...selection import Selection, FaceSelection
from ...mth.sparam import sparam_field_power, sparam_mode_power
from ...coord import Line
from .port_functions import compute_avg_power_flux
from typing import Dict, List, Tuple, Callable
import numpy as np

from scipy.sparse import identity
from scipy import sparse
from loguru import logger

class SimulationError(Exception):
    pass

def _dimstring(data: list[float]):
    return '(' + ', '.join([f'{x*1000:.1f}mm' for x in data]) + ')'

def shortest_path(xyz1: np.ndarray, xyz2: np.ndarray, Npts: int) -> np.ndarray:
    """
    Finds the pair of points (one from xyz1, one from xyz2) that are closest in Euclidean distance,
    and returns a (3, Npts) array of points linearly interpolating between them.

    Parameters:
    xyz1 : np.ndarray of shape (3, N1)
    xyz2 : np.ndarray of shape (3, N2)
    Npts : int, number of points in the output path

    Returns:
    np.ndarray of shape (3, Npts)
    """
    # Compute pairwise distances (N1 x N2)
    diffs = xyz1[:, :, np.newaxis] - xyz2[:, np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=0)  # shape (N1, N2)

    # Find indices of closest pair
    i1, i2 = np.unravel_index(np.argmin(dists), dists.shape)
    p1 = xyz1[:, i1]
    p2 = xyz2[:, i2]

    # Interpolate linearly between p1 and p2
    t = np.linspace(0, 1, Npts)
    path = (1 - t) * p1[:, np.newaxis] + t * p2[:, np.newaxis]

    return path

class Electrodynamics3D:
    """The Electrodynamics time harmonic physics class.

    This class contains all physics dependent features to perform EM simuation in the time-harmonic
    formulation.

    """
    def __init__(self, mesher: Mesher, order: int = 2):
        self.frequencies: list[float] = []
        self.current_frequency = 0
        self.order: int = order
        self.resolution: float = 1

        self.mesher: Mesher = mesher
        self.mesh: Mesh3D = None

        self.assembler: Assembler = Assembler()
        self.boundary_conditions: list[BoundaryCondition] = []
        self.basis: FEMBasis = None
        self.data: EMSimData = None
        self.solveroutine: SolveRoutine = DEFAULT_ROUTINE
        self.set_order(order)

        ## States
        self._bc_initialized: bool = False

    def set_order(self, order: int) -> None:
        """Sets the order of the basis functions used. Currently only supports second order.

        Args:
            order (int): The order to use.

        Raises:
            ValueError: An error if a wrong order is used.
        """
        if order not in (2,):
            raise ValueError(f'Order {order} not supported. Only order-2 allowed.')
        
        self.order = order
        self.resolution = {1: 0.15, 2: 0.3}[order]

    @property
    def nports(self) -> int:
        """The number of ports in the physics.

        Returns:
            int: The number of ports
        """
        return len([bc for bc in self.boundary_conditions if isinstance(bc,PortBC)])
    
    def ports(self) -> list[PortBC]:
        """A list of all port boundary conditions.

        Returns:
            list[PortBC]: A list of all port boundary conditions
        """
        return sorted([bc for bc in self.boundary_conditions if isinstance(bc,PortBC)], key=lambda x: x.number)
    
    @logger.catch
    def _initialize_bcs(self) -> None:
        """Initializes the boundary conditions to set PEC as all exterior boundaries.
        """
        logger.debug('Initializing boundary conditions.')

        self.boundary_conditions = []

        tags = self.mesher.domain_boundary_face_tags
        pec = PEC(FaceSelection(tags))
        logger.info(f'Adding PEC boundary condition with tags {tags}.')
        self.boundary_conditions.append(pec)

    def set_frequency(self, frequency: float | list[float] | np.ndarray ) -> None:
        """Define the frequencies for the frequency sweep

        Args:
            frequency (float | list[float] | np.ndarray): The frequency points.
        """
        logger.info(f'Setting frequency as {frequency/1e6}MHz.')
        if isinstance(frequency, (tuple, list, np.ndarray)):
            self.frequencies = list(frequency)
        else:
            self.frequencies = [frequency]

    def get_discretizer(self) -> Callable:
        """Returns a discretizer function that defines the maximum mesh size.

        Returns:
            Callable: The discretizer function
        """
        def disc(material: Material):
            return 299792456/(max(self.frequencies) * np.abs(material.neff))
        return disc
    
    def _initialize_field(self):
        """Initializes the physics basis to the correct FEMBasis object.
        
        Currently it defaults to Nedelec2. Mixed basis are used for modal analysis. 
        This function does not have to be called by the user. Its automatically invoked.
        """
        if self.basis is not None:
            return
        if self.order == 1:
            raise NotImplementedError('Nedelec 1 is temporarily not supported')
            from ...elements.nedelec1 import Nedelec1
            self.basis = Nedelec1(self.mesh)
        elif self.order == 2:
            from ...elements.nedelec2 import Nedelec2
            self.basis = Nedelec2(self.mesh)

    def _initialize_bc_data(self):
        ''' Initializes auxilliary required boundary condition information before running simulations.
        '''
        logger.debug('Initializing boundary conditions')
        for bc in self.boundary_conditions:
            if isinstance(bc, LumpedPort):
                self.define_lumped_port_integration_points(bc)

    @logger.catch
    def modal_analysis(self, port: ModalPort, 
                       nmodes: int = 6, 
                       direct: bool = True,
                       TEM: bool = False,
                       target_kz=None,
                       diagonal_scaling=False,
                       static_condendsation=False,
                       kz_range: bool=False,
                       freq: float = None) -> EMSimData:
        ''' Execute a modal analysis on a given ModalPort boundary condition.
        
        Parameters:
        -----------
            port : ModalPort
                The port object to execute the analysis for.
            direct : bool
                Whether to use the direct solver (LAPACK) if True. Otherwise it uses the iterative
                ARPACK solver. The ARPACK solver required an estimate for the propagation constant and is faster
                for a large number of Degrees of Freedom.
            TEM : bool = True
                Whether to estimate the propagation constant assuming its a TEM transmisison line.
            target_k0 : float
                The expected propagation constant to find a mode for (direct = False).
            diagonal_scaling : bool = False
                Whether to use diagonal scaling for matrix conditioning. Might speed up solving with ARPACK.
            static_condensation : bool = False
                Uses a static condensation. Does not work. Do not turn on!
            kz_range : bool = True
                Wether to use ARPACK to look for modes in a given range.
        '''
        if self._bc_initialized is False:
            raise SimulationError('Cannot run a modal analysis because no boundary conditions have been assigned.')
        self._initialize_field()
        
        logger.debug('Retreiving material properties.')
        ertet = self.mesh.retreive(lambda mat,x,y,z: mat.fer3d_mat(x,y,z), self.mesher.volumes)
        urtet = self.mesh.retreive(lambda mat,x,y,z: mat.fur3d_mat(x,y,z), self.mesher.volumes)

        er = np.zeros((3,3,self.mesh.n_tris,), dtype=np.complex128)
        ur = np.zeros((3,3,self.mesh.n_tris,), dtype=np.complex128)

        for itri in range(self.mesh.n_tris):
            itet = self.mesh.tri_to_tet[0,itri]
            er[:,:,itri] = ertet[:,:,itet]
            ur[:,:,itri] = urtet[:,:,itet]

        ermean = np.mean(er[er>0].flatten())
        urmean = np.mean(ur[ur>0].flatten())
        ermax = np.max(er.flatten())
        urmax = np.max(ur.flatten())

        mode_data = EMSimData(self.basis)

        if freq is None:
            freq = self.frequencies[0]
        k0 = 2*np.pi*freq/299792458
        kmax = k0*np.sqrt(ermax*urmax)

        logger.info('Assembling boundary mode analysic Matrices')
        
        Amatrix, Bmatrix, solve_ids, nlf = self.assembler.assemble_bma_matrices(self.basis, er, ur, k0, port, self.boundary_conditions)
        
        logger.debug(f'Total of {Amatrix.shape[0]} Degrees of freedom.')
        logger.debug(f'Applied frequency: {freq/1e9:.2f}GHz')
        logger.debug(f'K0 = {k0} rad/m')

        F = -1

        if target_kz is None:
            if TEM:
                target_kz = ermean*urmean*1.1*k0
            else:
                target_kz = ermean*urmean*0.65*k0

        if diagonal_scaling:
            logger.debug('Applying diagonal scaling')
            diagBtt = Bmatrix.diagonal()[:nlf.n_xy]
            diagBzz = Bmatrix.diagonal()[nlf.n_xy:]

            logger.debug('Diagnoal means:')
            logger.debug(f'Btt = {np.sqrt(np.mean(diagBtt))}')
            logger.debug(f'Bzz = {np.sqrt(np.mean(diagBzz))}')

            s_Et = 1.0/(np.sqrt(np.mean(diagBtt)))
            s_Ez = 1.0/(np.sqrt(np.mean(diagBzz)))

            Sdiag = identity(nlf.n_field, dtype=np.complex128).tocsc()
            Sdiag[:nlf.n_xy,:nlf.n_xy] *= s_Et
            Sdiag[nlf.n_xy:, nlf.n_xy:] *= s_Ez

            Ascaled = Sdiag @ Amatrix @ Sdiag
            Bscaled = Sdiag @ Bmatrix @ Sdiag

            diagBtt = Bscaled.diagonal()[:nlf.n_xy]
            diagBzz = Bscaled.diagonal()[nlf.n_xy:]
            
            Amatrix = Ascaled
            Bmatrix = Bscaled
        
        if static_condendsation:
            logger.debug('Implementing static condensation')
            N = nlf.n_xy
            Dtt = Bmatrix[:N,:N]
            Dtz = Bmatrix[:N,N:]
            Dzt = Bmatrix[N:,:N]
            Dzz = Bmatrix[N:,N:]

            Dzzinv = sparse.linalg.inv(Dzz.tocsc())
            Amatrix = -Amatrix[:N,:N]
            Bmatrix =  (Dtz @ Dzzinv) @ Dzt - Dtt
            solve_ids = solve_ids[solve_ids<N]

        logger.debug(f'Solving for {solve_ids.shape[0]} degrees of freedom.')

        eigen_values, eigen_modes = self.solveroutine.eig(Amatrix, Bmatrix, solve_ids, nmodes, direct, target_kz)
        
        logger.debug(f'Eigenvalues: {np.sqrt(F*eigen_values)} rad/m')

        port._er = er
        port._ur = ur

        nmodes_found = eigen_values.shape[0]

        for i in range(nmodes_found):
            data = mode_data.new(freq=freq, ur=ur, er=er, k0=k0, mode=i+1)
            
            Emode = np.zeros((nlf.n_field,), dtype=np.complex128)
            eigenmode = eigen_modes[:,i]
            Emode[solve_ids] = np.squeeze(eigenmode)
            Emode = Emode * np.exp(-1j*np.angle(np.max(Emode)))

            beta = min(np.emath.sqrt(-eigen_values[i]).real,kmax.real)
            data._field = Emode
            residuals = -1

            portfE = nlf.interpolate_Ef(Emode)
            portfH = nlf.interpolate_Hf(Emode, k0, ur, beta)
            P = compute_avg_power_flux(nlf, Emode, k0, ur, beta)

            mode = port.add_mode(Emode, portfE, portfH, beta, k0, residuals, TEM=TEM, freq=freq)
            mode.set_power(P)

            Ez = np.max(np.abs(Emode[nlf.n_xy:]))
            Exy = np.max(np.abs(Emode[:nlf.n_xy]))

            if Ez/Exy < 1e-5 and not TEM:
                logger.debug('Low Ez/Et ratio detected, assuming TE mode')
                mode.modetype = 'TE'
            elif Ez/Exy > 1e-5 and not TEM:
                logger.debug('High Ez/Et ratio detected, assuming TM mode')
                mode.modetype = 'TM'
            elif TEM:
                G1, G2 = self._find_tem_conductors(port)
                cs, dls = self._compute_integration_line(G1,G2)
                
                Ex, Ey, Ez = portfE(cs[0,:], cs[1,:], cs[2,:])
                voltage = np.sum(Ex*dls[0,:] + Ey*dls[1,:] + Ez*dls[2,:])
                mode.Z0 = voltage**2/(2*P)
                logger.debug(f'Port Z0 = {mode.Z0}')

                
        logger.info(f'Total of {port.nmodes} found')

        return mode_data
    
    def define_lumped_port_integration_points(self, port: LumpedPort):
        logger.debug('Finding Lumped Port integration points')
        field_axis = port.direction.np

        points = self.mesh.get_nodes(port.tags)

        xs = self.mesh.nodes[0,points]
        ys = self.mesh.nodes[1,points]
        zs = self.mesh.nodes[2,points]

        dotprod = xs*field_axis[0] + ys*field_axis[1] + zs*field_axis[2]

        start_id = points[np.argwhere(dotprod == np.min(dotprod))]

        start = np.squeeze(np.mean(self.mesh.nodes[:,start_id],axis=1))
        logger.info(f'Starting node = {_dimstring(start)}')
        end = start + port.direction.np*port.height


        port.vintline = Line.from_points(start, end, 11)

        logger.info(f'Ending node = {_dimstring(end)}')
        port.voltage_integration_points = (start, end)
        port.v_integration = True

    def _compute_integration_line(self, group1: list[int], group2: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Computes an integration line for two node island groups by finding the closest two nodes.
        
        This method is used for the modal TEM analysis to find an appropriate voltage integration path
        by looking for the two closest points for the two conductor islands that where discovered.

        Currently it defaults to 11 integration line points.

        Args:
            group1 (list[int]): The first island node group
            group2 (list[int]): The second island node group

        Returns:
            centers (np.ndarray): The center points of the line segments
            dls (np.ndarray): The delta-path vectors for each line segment.
        """
        nodes1 = self.mesh.nodes[:,group1]
        nodes2 = self.mesh.nodes[:,group2]
        path = shortest_path(nodes1, nodes2, 11)
        centres = (path[:,1:] + path[:,:-1])/2
        dls = path[:,1:] - path[:,:-1]
        return centres, dls

    def _find_tem_conductors(self, port: ModalPort) -> tuple[list[int], list[int]]:
        ''' Returns two lists of global node indices corresponding to the TEM port conductors.
        
        This method is invoked during modal analysis with TEM modes. It looks at all edges
        exterior to the boundary face triangulation and finds two small subsets of nodes that
        lie on different exterior boundaries of the boundary face.

        Args:
            port (ModalPort): The modal port object.
            
        Returns:
            list[int]: A list of node integers of island 1.
            list[int]: A list of node integers of island 2.
        '''

        logger.debug('Finding PEC TEM conductors')
        pecs: list[PEC] = [bc for bc in self.boundary_conditions if isinstance(bc,PEC)]
        mesh = self.mesh

        # Process all PEC Boundary Conditions
        pec_edges = []
        for pec in pecs:
            face_tags = pec.tags
            tri_ids = mesh.get_triangles(face_tags)
            edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())
            pec_edges.extend(edge_ids)
        
        pec_edges = set(pec_edges)
        
        tri_ids = mesh.get_triangles(port.tags)
        edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())
        
        pec_port = np.array([i for i in list(pec_edges) if i in set(edge_ids)])
        
        pec_islands = mesh.find_edge_groups(pec_port)

        self.basis._pec_islands = pec_islands
        logger.debug(f'Found {len(pec_islands)} PEC islands.')

        if len(pec_islands) != 2:
            raise ValueError(f'Found {len(pec_islands)} PEC islands. Expected 2.')
        
        groups = []
        for island in pec_islands:
            group = set()
            for edge in island:
                group.add(mesh.edges[0,edge])
                group.add(mesh.edges[1,edge])
            groups.append(sorted(list(group)))
        
        group1 = groups[0]
        group2 = groups[1]

        return group1, group2

    @logger.catch
    def frequency_domain(self) -> EMSimData:
        ''' Executes the frequency domain study.'''
        mesh = self.mesh
        if self._bc_initialized is False:
            raise SimulationError('Cannot run a modal analysis because no boundary conditions have been assigned.')
        self._initialize_field()
        self._initialize_bc_data()
        
        er = self.mesh.retreive(lambda mat,x,y,z: mat.fer3d_mat(x,y,z), self.mesher.volumes)
        ur = self.mesh.retreive(lambda mat,x,y,z: mat.fur3d_mat(x,y,z), self.mesher.volumes)
        
        ertri = np.zeros((3,3,self.mesh.n_tris), dtype=np.complex128)
        urtri = np.zeros((3,3,self.mesh.n_tris), dtype=np.complex128)

        for itri in range(self.mesh.n_tris):
            itet = self.mesh.tri_to_tet[0,itri]
            ertri[:,:,itri] = er[:,:,itet]
            urtri[:,:,itri] = ur[:,:,itet]

        ### Does this move
        logger.debug('Initializing frequency domain sweep.')
        
        self.data = EMSimData(self.basis)
        
        #### Port settings

        active_ports = [bc for bc in self.boundary_conditions if isinstance(bc,PortBC) and bc.active]
        analyse_sparameters = False

        if len(active_ports) == 1:
            analyse_sparameters = True


        all_ports = [bc for bc in self.boundary_conditions if isinstance(bc,PortBC)]
        port_numbers = [port.port_number for port in all_ports]

        #####

        logger.debug(f'Starting the simulation of {len(self.frequencies)} frequency points.')
        for freq in self.frequencies:
            

            k0 = 2*np.pi*freq/299792458
            data = self.data.new(freq=freq,
                                 k0=k0)
            
            data.init_sp(port_numbers)

            data.er = np.squeeze(er[0,0,:])
            data.ur = np.squeeze(ur[0,0,:])

            logger.info(f'Frequency = {freq/1e9:.3f} GHz') 

            # Recording port information
            for port in all_ports:
                data.add_port_properties(port.port_number,
                                         mode_number=port.mode_number,
                                         k0 = k0,
                                         beta = port.get_beta(k0),
                                         Z0 = port.Z0,
                                         Pout= port.power)
            
            # Assembling matrix problem
            K, b, solve_ids = self.assembler.assemble_freq_matrix(self.basis, er, ur, self.boundary_conditions, freq, cache_matrices=True)
        
            logger.debug(f'Routine: {self.solveroutine}')

            solution = self.solveroutine.solve(K,b,solve_ids)

            data._field = solution
            
            if analyse_sparameters:

                fieldf = self.basis.interpolate_Ef(solution)

                Pout = 0
                for bc in active_ports:
                    tris = mesh.get_triangles(bc.tags)
                    tri_vertices = mesh.tris[:,tris]
                    erp = ertri[:,:,tris]
                    urp = urtri[:,:,tris]
                    pfield, pmode = self._compute_s_data(bc, fieldf, tri_vertices, k0, erp, urp)
                    logger.debug(f'Field Amplitude = {np.abs(pfield):.3f}, Excitation = {np.abs(pmode):.2f}')
                    Pout = pmode
                
                
                logger.info('Passive ports:')
                for bc in all_ports:
                    tris = mesh.get_triangles(bc.tags)
                    tri_vertices = mesh.tris[:,tris]
                    erp = ertri[:,:,tris]
                    urp = urtri[:,:,tris]
                    pfield, pmode = self._compute_s_data(bc, fieldf, tri_vertices, k0, erp, urp)
                    logger.debug(f'Field amplitude = {np.abs(pfield):.3f}, Excitation= {np.abs(pmode):.2f}')
                    
                    data.write_S(bc.port_number, active_ports[0].port_number, pfield/Pout)
            
            logger.info('Simulation Complete!')
        return self.data
    
    def _compute_s_data(self, bc: PortBC, 
                       fieldfunction: Callable, 
                       tri_vertices: np.ndarray, 
                       k0: float,
                       erp: np.ndarray,
                       urp: np.ndarray,) -> tuple[complex, complex]:
        """ Computes the S-parameter data for a given boundary condition and field function.

        Args:
            bc (PortBC): The port boundary condition
            fieldfunction (Callable): The field function that interpolates the solution field.
            tri_vertices (np.ndarray): The triangle vertex indices of the port face
            k₀ (float): The simulation phase constant
            erp (np.ndarray): The εᵣ of the port face triangles
            urp (np.ndarray): The μᵣ of the port face triangles.

        Returns:
            tuple[complex, complex]: _description_
        """
        if bc.v_integration:
           
            ln = bc.vintline
            Ex, Ey, Ez = fieldfunction(*ln.cmid)

            V = np.sum(Ex*ln.dxs + Ey*ln.dys + Ez*ln.dzs)
            if bc.active:
                a = bc.voltage
                b = (V-bc.voltage)
            else:
                a = 0
                b = V
            
            a = np.sqrt(a**2/(2*bc.Z0))
            b = np.sqrt(b**2/(2*bc.Z0))
            return b, a
        else:
            if bc.modetype == 'TEM':
                const = 1/(np.sqrt((urp[0,0,:] + urp[1,1,:] + urp[2,2,:])/(erp[0,0,:] + erp[1,1,:] + erp[2,2,:])))
            if bc.modetype == 'TE':
                const = 1/((urp[0,0,:] + urp[1,1,:] + urp[2,2,:])/3)
            elif bc.modetype == 'TM':
                const = 1/((erp[0,0,:] + erp[1,1,:] + erp[2,2,:])/3)
            const = np.squeeze(const)
            field_p = sparam_field_power(self.mesh.nodes, tri_vertices, bc, k0, fieldfunction, const)
            mode_p = sparam_mode_power(self.mesh.nodes, tri_vertices, bc, k0, const)
            return field_p, mode_p
        
    @logger.catch
    def assign(self, 
               *bcs: BoundaryCondition) -> None:
        """Assign a boundary-condition object to a domain or list of domains.
        This method must be called to submit any boundary condition object you made to the physics.

        Args:
            bcs *(BoundaryCondition): A list of boundary condition objects.
        """
        self._bc_initialized = True
        wordmap = {
            0: 'node',
            1: 'edge',
            2: 'face',
            3: 'domain'
        }
        for bc in bcs:
            bc.add_tags(bc.selection.dimtags)

            logger.info('Excluding other possible boundary conditions')

            for existing_bc in self.boundary_conditions:
                excluded = existing_bc.exclude_bc(bc)
                if excluded:
                    logger.warning(f'Removed the following {wordmap[bc.dim]}: {excluded} from {existing_bc}')
            
            self.boundary_conditions.append(bc)


## DEPRICATED



    # @logger.catch
    # def eigenmode(self, mesh: Mesh3D, solver = None, num_sols: int = 6):
    #     if solver is None:
    #         logger.info('Defaulting to BiCGStab.')
    #         solver = sparse.linalg.eigs

    #     if self.order == 1:
    #         logger.info('Detected 1st order elements.')
    #         from ...elements.nedelec1.assembly import assemble_eig_matrix
    #         ft = FieldType.VEC_LIN

    #     elif self.order == 2:
    #         logger.info('Detected 2nd order elements.')
    #         from ...elements.nedelec2.assembly import assemble_eig_matrix_E
    #         ft = FieldType.VEC_QUAD
        
    #     er = self.mesh.retreive(mesh.centers, lambda mat,x,y,z: mat.fer3d(x,y,z))
    #     ur = self.mesh.retreive(mesh.centers, lambda mat,x,y,z: mat.fur3d(x,y,z))
        
    #     dataset = Dataset3D(mesh, self.frequencies, 0, ft)
    #     dataset.er = er
    #     dataset.ur = ur
    #     logger.info('Solving eigenmodes.')
        
    #     f_target = self.frequencies[0]
    #     sigma = (2 * np.pi * f_target / 299792458)**2

    #     A, B, solvenodes = assemble_eig_matrix(mesh, er, ur, self.boundary_conditions)
        
    #     A = A[np.ix_(solvenodes, solvenodes)]
    #     B = B[np.ix_(solvenodes, solvenodes)]
    #     #A = sparse.csc_matrix(A)
    #     #B = sparse.csc_matrix(B)
        
    #     w, v = sparse.linalg.eigs(A, k=num_sols, M=B, sigma=sigma, which='LM')
        
    #     logger.info(f'Eigenvalues: {np.sqrt(w)*299792458/(2*np.pi) * 1e-9} GHz')

    #     Esol = np.zeros((num_sols, mesh.nfield), dtype=np.complex128)

    #     Esol[:, solvenodes] = v.T
        
    #     dataset.set_efield(Esol)

    #     self.basis = dataset
