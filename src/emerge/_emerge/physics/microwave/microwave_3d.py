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
from ...coord import Line
from ...elements.femdata import FEMBasis
from ...elements.nedelec2 import Nedelec2
from ...solver import DEFAULT_ROUTINE, SolveRoutine
from ...system import called_from_main_function
from ...selection import FaceSelection
from ...mth.optimized import compute_distances
from .microwave_bc import MWBoundaryConditionSet, PEC, ModalPort, LumpedPort, PortBC
from .microwave_data import MWData
from .assembly.assembler import Assembler
from .port_functions import compute_avg_power_flux
from .simjob import SimJob

from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from typing import Callable, Literal
import multiprocessing as mp
from cmath import sqrt as csqrt

import numpy as np
import threading
import time

class SimulationError(Exception):
    pass

def run_job_multi(job: SimJob) -> SimJob:
    """The job launcher for Multi-Processing environements

    Args:
        job (SimJob): The Simulation Job

    Returns:
        SimJob: The solved SimJob
    """
    routine = DEFAULT_ROUTINE.duplicate()._configure_routine('MP')
    for A, b, ids, reuse, aux in job.iter_Ab():
        solution, report = routine.solve(A, b, ids, reuse, id=job.id)
        report.add(**aux)
        job.submit_solution(solution, report)
    return job


def _dimstring(data: list[float] | np.ndarray) -> str:
    """A String formatter for dimensions in millimeters

    Args:
        data (list[float]): The list of floating point dimensions

    Returns:
        str: The formatted string
    """
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

def _pick_central(vertices: np.ndarray) -> np.ndarray:
    """Computes the coordinate in the vertex set that has the shortest square distance to all other points.
    

    Args:
        vertices (np.ndarray): The set of coordinates [3,:]

    Returns:
        np.ndarray: The most central point
    """
    Ds = compute_distances(vertices[0,:], vertices[1,:], vertices[2,:])
    sumDs = np.sum(Ds**2, axis=1)
    id_central = np.argwhere(sumDs==np.min(sumDs)).flatten()[0]
    return vertices[:, id_central].squeeze()
    
class Microwave3D:
    """The Electrodynamics time harmonic physics class.

    This class contains all physics dependent features to perform EM simuation in the time-harmonic
    formulation.

    """
    def __init__(self, mesher: Mesher, mwdata: MWData, order: int = 2):
        self.frequencies: list[float] = []
        self.current_frequency = 0
        self.order: int = order
        self.resolution: float = 1

        self.mesher: Mesher = mesher
        self.mesh: Mesh3D = Mesh3D(self.mesher)

        self.assembler: Assembler = Assembler()
        self.bc: MWBoundaryConditionSet = MWBoundaryConditionSet(None)
        self.basis: Nedelec2 | None = None
        self.solveroutine: SolveRoutine = DEFAULT_ROUTINE
        self.cache_matrices: bool = True

        ## States
        self._bc_initialized: bool = False
        self.data: MWData = mwdata

        ## Data
        self._params: dict[str, float] = dict()
        self._simstart: float = 0.0
        self._simend: float = 0.0

        self.set_order(order)

    def reset_data(self):
        self.data = MWData()

    def reset(self):
        self.bc.reset()
        self.basis: FEMBasis = None
        self.bc = MWBoundaryConditionSet(None)
        self.solveroutine.reset()

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
        return self.bc.count(PortBC)
    
    def ports(self) -> list[PortBC]:
        """A list of all port boundary conditions.

        Returns:
            list[PortBC]: A list of all port boundary conditions
        """
        return sorted(self.bc.oftype(PortBC), key=lambda x: x.number) # type: ignore
    
    
    def _initialize_bcs(self) -> None:
        """Initializes the boundary conditions to set PEC as all exterior boundaries.
        """
        logger.debug('Initializing boundary conditions.')

        tags = self.mesher.domain_boundary_face_tags
        tags = [tag for tag in tags if tag not in self.bc.assigned(2)]
        self.bc.PEC(FaceSelection(tags))
        logger.info(f'Adding PEC boundary condition with tags {tags}.')
        if self.mesher.periodic_cell is not None:
            self.mesher.periodic_cell.generate_bcs()
            for bc in self.mesher.periodic_cell.bcs:
                self.bc.assign(bc)

    def set_frequency(self, frequency: float | list[float] | np.ndarray ) -> None:
        """Define the frequencies for the frequency sweep

        Args:
            frequency (float | list[float] | np.ndarray): The frequency points.
        """
        logger.info(f'Setting frequency as {frequency}Hz.')
        if isinstance(frequency, (tuple, list, np.ndarray)):
            self.frequencies = list(frequency)
        else:
            self.frequencies = [frequency]

        self.mesher.max_size = self.resolution * 299792458 / max(self.frequencies)
        self.mesher.min_size = 0.1 * self.mesher.max_size

        logger.debug(f'Setting global mesh size range to: {self.mesher.min_size*1000:.3f}mm - {self.mesher.max_size*1000:.3f}mm')
    
    def set_frequency_range(self, fmin: float, fmax: float, Npoints: int) -> None:
        """Set the frequency range using the np.linspace syntax

        Args:
            fmin (float): The starting frequency
            fmax (float): The ending frequency
            Npoints (int): The number of points
        """
        self.set_frequency(np.linspace(fmin, fmax, Npoints))
    
    def fdense(self, Npoints: int) -> np.ndarray:
        if len(self.frequencies) == 1:
            raise ValueError('Only 1 frequency point known. At least two need to be defined.')
        fmin = min(self.frequencies)
        fmax = max(self.frequencies)
        return np.linspace(fmin, fmax, Npoints)
    
    def set_resolution(self, resolution: float) -> None:
        """Define the simulation resolution as the fraction of the wavelength.

        To define the wavelength as ¼λ, call .set_resolution(0.25)

        Args:
            resolution (float): The desired wavelength fraction.
            
        """
        self.resolution = resolution

    def set_conductivity_limit(self, condutivity: float) -> None:
        """Sets the limit of a material conductivity value beyond which
        the assembler considers it PEC. By default this value is
        set to 1·10⁷S/m which means copper conductivity is ignored.

        Args:
            condutivity (float): The conductivity level in S/m
        """
        if condutivity < 0:
            raise ValueError('Conductivity values must be above 0. Ignoring assignment')

        self.assembler.conductivity_limit = condutivity
    
    def get_discretizer(self) -> Callable:
        """Returns a discretizer function that defines the maximum mesh size.

        Returns:
            Callable: The discretizer function
        """
        def disc(material: Material):
            return 299792458/(max(self.frequencies) * np.real(material.neff))
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
            from ...elements import Nedelec1
            self.basis = Nedelec1(self.mesh)
        elif self.order == 2:
            from ...elements.nedelec2 import Nedelec2
            self.basis = Nedelec2(self.mesh)

    def _initialize_bc_data(self):
        ''' Initializes auxilliary required boundary condition information before running simulations.
        '''
        logger.debug('Initializing boundary conditions')
        for port in self.bc.oftype(LumpedPort):
            self.define_lumped_port_integration_points(port)
    
    def define_lumped_port_integration_points(self, port: LumpedPort) -> None:
        """Sets the integration points on Lumped Port objects for voltage integration

        Args:
            port (LumpedPort): The LumpedPort object

        Raises:
            SimulationError: An error if there are no nodes associated with the port.
        """
        logger.debug('Finding Lumped Port integration points')
        field_axis = port.Vdirection.np

        points = self.mesh.get_nodes(port.tags)

        if points.size==0:
            raise SimulationError(f'The lumped port {port} has no nodes associated with it')
        xs = self.mesh.nodes[0,points]
        ys = self.mesh.nodes[1,points]
        zs = self.mesh.nodes[2,points]

        dotprod = xs*field_axis[0] + ys*field_axis[1] + zs*field_axis[2]

        start_id = points[np.argwhere(dotprod == np.min(dotprod))]
        
        start = _pick_central(self.mesh.nodes[:,start_id.flatten()])
        logger.info(f'Starting node = {_dimstring(start)}')
        end = start + port.Vdirection.np*port.height


        port.vintline = Line.from_points(start, end, 21)

        logger.info(f'Ending node = {_dimstring(end)}')
        
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
        path = shortest_path(nodes1, nodes2, 21)
        centres = (path[:,1:] + path[:,:-1])/2
        dls = path[:,1:] - path[:,:-1]
        return centres, dls

    def _find_tem_conductors(self, port: ModalPort, sigtri: np.ndarray) -> tuple[list[int], list[int]]:
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
        if self.basis is None:
            raise ValueError('The field basis is not yet defined.')

        logger.debug('Finding PEC TEM conductors')
        pecs: list[PEC] = self.bc.get_conductors() # type: ignore
        mesh = self.mesh

        # Process all PEC Boundary Conditions
        pec_edges = []
        for pec in pecs:
            face_tags = pec.tags
            tri_ids = mesh.get_triangles(face_tags)
            edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())
            pec_edges.extend(edge_ids)
        
        # Process conductivity
        for itri in mesh.get_triangles(port.tags):
            if sigtri[itri] > 1e6:
                edge_ids = list(mesh.tri_to_edge[:,itri].flatten())
                pec_edges.extend(edge_ids)

        pec_edges = list(set(pec_edges))
        
        tri_ids = mesh.get_triangles(port.tags)
        edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())
        
        pec_port = np.array([i for i in pec_edges if i in set(edge_ids)])
        
        pec_islands = mesh.find_edge_groups(pec_port)

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
    
    def _compute_modes(self, freq: float):
        """Compute the modal port modes for a given frequency. Used internally by the frequency domain study.

        Args:
            freq (float): The simulation frequency
        """
        for bc in self.bc.oftype(ModalPort):
            
            # If there is a port mode (at least one) and the port does not have mixed materials. No new analysis is needed
            if not bc.mixed_materials and bc.initialized:
                continue
            
            self.modal_analysis(bc, 1, False, bc.TEM, freq=freq)

    def modal_analysis(self, 
                       port: ModalPort, 
                       nmodes: int = 6, 
                       direct: bool = True,
                       TEM: bool = False,
                       target_kz = None,
                       target_neff = None,
                       freq: float | None = None) -> None:
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
            target_neff : float
                The expected effective mode index defined as kz/k0 (1.0 = free space, <1 = TE/TM, >1=slow wavees)
            freq : float = None
                The desired frequency at which the mode is solved. If None then it uses the lowest frequency of the provided range.
        '''
        T0 = time.time()
        if self.bc._initialized is False:
            raise SimulationError('Cannot run a modal analysis because no boundary conditions have been assigned.')
        
        self._initialize_field()
        self._initialize_bc_data()

        if self.basis is None:
            raise SimulationError('Cannot proceed, the current basis class is undefined.')

        logger.debug('Retreiving material properties.')
        ertet = self.mesh.retreive(lambda mat,x,y,z: mat.fer3d_mat(x,y,z), self.mesher.volumes)
        urtet = self.mesh.retreive(lambda mat,x,y,z: mat.fur3d_mat(x,y,z), self.mesher.volumes)
        condtet = self.mesh.retreive(lambda mat,x,y,z: mat.cond, self.mesher.volumes)[0,0,:]

        er = np.zeros((3,3,self.mesh.n_tris,), dtype=np.complex128)
        ur = np.zeros((3,3,self.mesh.n_tris,), dtype=np.complex128)
        cond = np.zeros((self.mesh.n_tris,), dtype=np.complex128)

        for itri in range(self.mesh.n_tris):
            itet = self.mesh.tri_to_tet[0,itri]
            er[:,:,itri] = ertet[:,:,itet]
            ur[:,:,itri] = urtet[:,:,itet]
            cond[itri] = condtet[itet]

        itri_port = self.mesh.get_triangles(port.tags)

        ermean = np.mean(er[er>0].flatten()[itri_port])
        urmean = np.mean(ur[ur>0].flatten()[itri_port])
        ermax = np.max(er[:,:,itri_port].flatten())
        urmax = np.max(ur[:,:,itri_port].flatten())

        if freq is None:
            freq = self.frequencies[0]
        
        k0 = 2*np.pi*freq/299792458
        kmax = k0*np.sqrt(ermax.real*urmax.real)
        
        Amatrix, Bmatrix, solve_ids, nlf = self.assembler.assemble_bma_matrices(self.basis, er, ur, cond, k0, port, self.bc)
        
        logger.debug(f'Total of {Amatrix.shape[0]} Degrees of freedom.')
        logger.debug(f'Applied frequency: {freq/1e9:.2f}GHz')
        logger.debug(f'K0 = {k0} rad/m')

        F = -1

        if target_neff is not None:
            target_kz = k0*target_neff
        
        if target_kz is None:
            if TEM:
                target_kz = ermean*urmean*1.1*k0
            else:
                
                target_kz = ermean*urmean*0.7*k0

    
        logger.debug(f'Solving for {solve_ids.shape[0]} degrees of freedom.')

        eigen_values, eigen_modes, report = self.solveroutine.eig_boundary(Amatrix, Bmatrix, solve_ids, nmodes, direct, target_kz, sign=-1)
        
        logger.debug(f'Eigenvalues: {np.sqrt(F*eigen_values)} rad/m')

        port._er = er
        port._ur = ur

        nmodes_found = eigen_values.shape[0]

        for i in range(nmodes_found):
            
            Emode = np.zeros((nlf.n_field,), dtype=np.complex128)
            eigenmode = eigen_modes[:,i]
            Emode[solve_ids] = np.squeeze(eigenmode)
            Emode = Emode * np.exp(-1j*np.angle(np.max(Emode)))

            beta_base = np.emath.sqrt(-eigen_values[i])
            beta = min(k0*np.sqrt(ermax*urmax), beta_base)

            residuals = -1

            portfE = nlf.interpolate_Ef(Emode)
            portfH = nlf.interpolate_Hf(Emode, k0, ur, beta)

            P = compute_avg_power_flux(nlf, Emode, k0, ur, beta)

            mode = port.add_mode(Emode, portfE, portfH, beta, k0, residuals, TEM=TEM, freq=freq)
            if mode is None:
                continue
            
            Efxy = Emode[:nlf.n_xy]
            Efz = Emode[nlf.n_xy:]
            Ez = np.max(np.abs(Efz))
            Exy = np.max(np.abs(Efxy))
            
            # Exy = np.max(np.max(Emode))
            # Ez = 0
            if Ez/Exy < 1e-3 and not TEM:
                logger.debug('Low Ez/Et ratio detected, assuming TE mode')
                mode.modetype = 'TE'
            elif Ez/Exy > 1e-3 and not TEM:
                logger.debug('High Ez/Et ratio detected, assuming TM mode')
                mode.modetype = 'TM'
            elif TEM:
                G1, G2 = self._find_tem_conductors(port, sigtri=cond)
                cs, dls = self._compute_integration_line(G1,G2)
                mode.modetype='TEM'
                Ex, Ey, Ez = portfE(cs[0,:], cs[1,:], cs[2,:])
                voltage = np.sum(Ex*dls[0,:] + Ey*dls[1,:] + Ez*dls[2,:])
                mode.Z0 = voltage**2/(2*P)
                logger.debug(f'Port Z0 = {mode.Z0}')

            mode.set_power(P*port._qmode(k0)**2)
        
        port.sort_modes()

        logger.info(f'Total of {port.nmodes} found')

        T2 = time.time()    
        logger.info(f'Elapsed time = {(T2-T0):.2f} seconds.')
        return None
    
    def run_sweep(self, 
                parallel: bool = False,
                njobs: int = 2, 
                harddisc_threshold: int | None = None,
                harddisc_path: str = 'EMergeSparse',
                frequency_groups: int = -1,
                multi_processing: bool = False,
                automatic_modal_analysis: bool = True) -> MWData:
        """Executes a frequency domain study

        The study is distributed over "njobs" workers.
        As optional parameter you may set a harddisc_threshold as integer. This determines the maximum
        number of degrees of freedom before which the jobs will be cahced to the harddisk. The
        path that will be used to cache the sparse matrices can be specified.
        Additionally the term frequency_groups may be specified. This number will define in how
        many groups the matrices will be pre-computed before they are send to workers. This can minimize
        the total amound of RAM memory used. For example with 11 frequencies in gruops of 4, the following
        frequency indices will be precomputed and then solved: [[1,2,3,4],[5,6,7,8],[9,10,11]]

        Args:
            njobs (int, optional): The number of jobs. Defaults to 2.
            harddisc_threshold (int, optional): The number of DOF limit. Defaults to None.
            harddisc_path (str, optional): The cached matrix path name. Defaults to 'EMergeSparse'.
            frequency_groups (int, optional): The number of frequency points in a solve group. Defaults to -1.
            automatic_modal_analysis (bool, optional): Automatically compute port modes. Defaults to False.
            multi_processing (bool, optional): Whether to use multiprocessing instead of multi-threaded (slower on most machines).

        Raises:
            SimulationError: An error associated witha a problem during the simulation.

        Returns:
            MWSimData: The dataset.
        """
        
        self._simstart = time.time()
        if self.bc._initialized is False:
            raise SimulationError('Cannot run a modal analysis because no boundary conditions have been assigned.')
        
        self._initialize_field()
        self._initialize_bc_data()
        
        if self.basis is None:
            raise SimulationError('Cannot proceed, the simulation basis class is undefined.')

        er = self.mesh.retreive(lambda mat,x,y,z: mat.fer3d_mat(x,y,z), self.mesher.volumes)
        ur = self.mesh.retreive(lambda mat,x,y,z: mat.fur3d_mat(x,y,z), self.mesher.volumes)
        cond = self.mesh.retreive(lambda mat,x,y,z: mat.cond, self.mesher.volumes)[0,0,:]

        ### Does this move
        logger.debug('Initializing frequency domain sweep.')
        
        #### Port settings
        all_ports = self.bc.oftype(PortBC)

        ##### FOR PORT SWEEP SET ALL ACTIVE TO FALSE. THIS SHOULD BE FIXED LATER
        ### COMPUTE WHICH TETS ARE CONNECTED TO PORT INDICES

        for port in all_ports:
            port.active=False
            

        logger.info(f'Pre-assembling matrices of {len(self.frequencies)} frequency points.')

        # Thread-local storage for per-thread resources
        thread_local = threading.local()

        ## DEFINE SOLVE FUNCTIONS
        def get_routine():
            if not hasattr(thread_local, "routine"):
                thread_local.routine = self.solveroutine.duplicate()._configure_routine('MT')
            return thread_local.routine

        def run_job(job: SimJob):
            routine = get_routine()
            for A, b, ids, reuse, aux in job.iter_Ab():
                solution, report = routine.solve(A, b, ids, reuse, id=job.id)
                report.add(**aux)
                job.submit_solution(solution, report)
            return job
        
        def run_job_single(job: SimJob):
            for A, b, ids, reuse, aux in job.iter_Ab():
                solution, report = self.solveroutine.solve(A, b, ids, reuse, id=job.id)
                report.add(**aux)
                job.submit_solution(solution, report)
            return job
        
        ## GROUP FREQUENCIES
        # Each frequency group will be pre-assembled before submitting them to the parallel pool
        freq_groups = []
        if frequency_groups == -1:
            freq_groups=[self.frequencies,]
        else:
            n = frequency_groups
            freq_groups = [self.frequencies[i:i+n] for i in range(0, len(self.frequencies), n)]

        results: list[SimJob] = []

        ## Single threaded
        job_id = 1

        self._compute_modes(sum(self.frequencies)/len(self.frequencies))

        if not parallel:
            # ITERATE OVER FREQUENCIES
            freq_groups
            for i_group, fgroup in enumerate(freq_groups):
                logger.info(f'Precomputing group {i_group}.')
                jobs = []
                ## Assemble jobs
                for ifreq, freq in enumerate(fgroup):
                    logger.debug(f'Simulation frequency = {freq/1e9:.3f} GHz') 
                    if automatic_modal_analysis:
                        self._compute_modes(freq)
                    job = self.assembler.assemble_freq_matrix(self.basis, er, ur, cond, 
                                                            self.bc.boundary_conditions, 
                                                            freq, 
                                                            cache_matrices=self.cache_matrices)
                    job.store_limit = harddisc_threshold
                    job.relative_path = harddisc_path
                    job.id = job_id
                    job_id += 1
                    jobs.append(job)
                
                logger.info(f'Starting single threaded solve of {len(jobs)} jobs.')
                group_results = [run_job_single(job) for job in jobs]
                results.extend(group_results)
        elif not multi_processing:
             # MULTI THREADED
            with ThreadPoolExecutor(max_workers=njobs) as executor:
                # ITERATE OVER FREQUENCIES
                for i_group, fgroup in enumerate(freq_groups):
                    logger.info(f'Precomputing group {i_group}.')
                    jobs = []
                    ## Assemble jobs
                    for freq in fgroup:
                        logger.debug(f'Simulation frequency = {freq/1e9:.3f} GHz') 
                        if automatic_modal_analysis:
                            self._compute_modes(freq)
                        job = self.assembler.assemble_freq_matrix(self.basis, er, ur, cond, 
                                                                self.bc.boundary_conditions, 
                                                                freq, 
                                                                cache_matrices=self.cache_matrices)
                        job.store_limit = harddisc_threshold
                        job.relative_path = harddisc_path
                        job.id = job_id
                        job_id += 1
                        jobs.append(job)
                    
                    logger.info(f'Starting distributed solve of {len(jobs)} jobs with {njobs} threads.')
                    group_results = list(executor.map(run_job, jobs))
                    results.extend(group_results)
                executor.shutdown()
        else:
            ### MULTI PROCESSING
            # Check for if __name__=="__main__" Guard
            if not called_from_main_function():
                raise SimulationError(
                    "Multiprocess support must be launched from your "
                    "if __name__ == '__main__' guard in the top-level script."
                )
            # Start parallel pool
            with mp.Pool(processes=njobs) as pool:
                for i_group, fgroup in enumerate(freq_groups):
                    logger.debug(f'Precomputing group {i_group}.')
                    jobs = []
                    # Assemble jobs
                    for freq in fgroup:
                        logger.debug(f'Simulation frequency = {freq/1e9:.3f} GHz')
                        if automatic_modal_analysis:
                            self._compute_modes(freq)
                        
                        job = self.assembler.assemble_freq_matrix(
                            self.basis, er, ur, cond,
                            self.bc.boundary_conditions,
                            freq,
                            cache_matrices=self.cache_matrices
                        )

                        job.store_limit = harddisc_threshold
                        job.relative_path = harddisc_path
                        job.id = job_id
                        job_id += 1
                        jobs.append(job)

                    logger.info(
                        f'Starting distributed solve of {len(jobs)} jobs '
                        f'with {njobs} processes in parallel'
                    )
                    # Distribute taks
                    group_results = pool.map(run_job_multi, jobs)
                    results.extend(group_results)

        thread_local.__dict__.clear()
        logger.info('Solving complete')

        for freq, job in zip(self.frequencies, results):
            self.data.setreport(job.reports, freq=freq, **self._params)

        for variables, data in self.data.sim.iterate():
            logger.trace(f'Sim variable: {variables}')
            for item in data['report']:
                item.pretty_print(logger.trace)

        self.solveroutine.reset()
        ### Compute S-parameters and return
        self._post_process(results, er, ur, cond)
        return self.data
    
    def eigenmode(self, search_frequency: float,
                        nmodes: int = 6,
                        k0_limit: float = 1,
                        direct: bool = False,
                        deep_search: bool = False,
                        mode: Literal['LM','LR','SR','LI','SI']='LM') -> MWData:
        """Executes an eigenmode study

       

        Args:
            search_frequency (float): The frequency around which you would like to search
            nmodes (int, optional): The number of jobs. Defaults to 6.
            k0_limit (float): The lowest k0 value before which a mode is considered part of the null space. Defaults to 1e-3
        Raises:
            SimulationError: An error associated witha a problem during the simulation.

        Returns:
            MWSimData: The dataset.
        """
        
        self._simstart = time.time()
        if self.bc._initialized is False:
            raise SimulationError('Cannot run a modal analysis because no boundary conditions have been assigned.')
        
        self._initialize_field()
        self._initialize_bc_data()
        
        if self.basis is None:
            raise SimulationError('Cannot proceed. The simulation basis class is undefined.')

        er = self.mesh.retreive(lambda mat,x,y,z: mat.fer3d_mat(x,y,z), self.mesher.volumes)
        ur = self.mesh.retreive(lambda mat,x,y,z: mat.fur3d_mat(x,y,z), self.mesher.volumes)
        cond = self.mesh.retreive(lambda mat,x,y,z: mat.cond, self.mesher.volumes)[0,0,:]

        ### Does this move
        logger.debug('Initializing frequency domain sweep.')
            
        logger.info(f'Pre-assembling matrices of {len(self.frequencies)} frequency points.')
        
        job = self.assembler.assemble_eig_matrix(self.basis, er, ur, cond, 
                                                            self.bc.boundary_conditions, search_frequency)
        

        logger.info('Solving complete')

        A, C, solve_ids = job.yield_AC()

        target_k0 = 2*np.pi*search_frequency/299792458

        eigen_values, eigen_modes, report = self.solveroutine.eig(A, C, solve_ids, nmodes, direct, target_k0, which=mode)

        eigen_modes = job.fix_solutions(eigen_modes)

        logger.debug(f'Eigenvalues: {np.sqrt(eigen_values)} rad/m')

        nmodes_found = eigen_values.shape[0]

        for i in range(nmodes_found):
            

            eig_k0 = np.sqrt(eigen_values[i])
            if eig_k0 < k0_limit:
                logger.debug(f'Ignoring mode due to low k0: {eig_k0} < {k0_limit}')
                continue
            eig_freq = eig_k0*299792458/(2*np.pi)

            logger.debug(f'Found k0={eig_k0:.2f}, f0={eig_freq/1e9:.2f} GHz')
            Emode = eigen_modes[:,i]

            scalardata = self.data.scalar.new(**self._params)
            scalardata.k0 = eig_k0
            scalardata.freq = eig_freq

            fielddata = self.data.field.new(**self._params)
            fielddata.freq = eig_freq
            fielddata._der = np.squeeze(er[0,0,:])
            fielddata._dur = np.squeeze(ur[0,0,:])
            fielddata._mode_field = Emode
            fielddata.basis = self.basis
        ### Compute S-parameters and return
        
        return self.data

    def _post_process(self, results: list[SimJob], er: np.ndarray, ur: np.ndarray, cond: np.ndarray):
        """Compute the S-parameters after Frequency sweep

        Args:
            results (list[SimJob]): The set of simulation results
            er (np.ndarray): The domain εᵣ
            ur (np.ndarray): The domain μᵣ
            cond (np.ndarray): The domain conductivity
        """
        if self.basis is None:
            raise SimulationError('Cannot post-process. Simulation basis function is undefined.')
        mesh = self.mesh
        all_ports = self.bc.oftype(PortBC)
        port_numbers = [port.port_number for port in all_ports]
        all_port_tets = self.mesh.get_face_tets(*[port.tags for port in all_ports])

        logger.info('Computing S-parameters')
        
        ertri = np.zeros((3,3,self.mesh.n_tris), dtype=np.complex128)
        urtri = np.zeros((3,3,self.mesh.n_tris), dtype=np.complex128)
        condtri = np.zeros((self.mesh.n_tris,), dtype=np.complex128)

        for itri in range(self.mesh.n_tris):
            itet = self.mesh.tri_to_tet[0,itri]
            ertri[:,:,itri] = er[:,:,itet]
            urtri[:,:,itri] = ur[:,:,itet]
            condtri[itri] = cond[itet]

        for freq, job in zip(self.frequencies, results):

            k0 = 2*np.pi*freq/299792458

            scalardata = self.data.scalar.new(freq=freq, **self._params)
            scalardata.k0 = k0
            scalardata.freq = freq
            scalardata.init_sp(port_numbers) # type: ignore
            
            fielddata = self.data.field.new(freq=freq, **self._params)
            fielddata.freq = freq
            fielddata._der = np.squeeze(er[0,0,:])
            fielddata._dur = np.squeeze(ur[0,0,:])

            logger.info(f'Post Processing simulation frequency = {freq/1e9:.3f} GHz') 

            # Recording port information
            for active_port in all_ports:
                fielddata.add_port_properties(active_port.port_number,
                                         mode_number=active_port.mode_number,
                                         k0 = k0,
                                         beta = active_port.get_beta(k0),
                                         Z0 = active_port.portZ0(k0),
                                         Pout = active_port.power)
                scalardata.add_port_properties(active_port.port_number,
                                         mode_number=active_port.mode_number,
                                         k0 = k0,
                                         beta = active_port.get_beta(k0),
                                         Z0 = active_port.portZ0(k0),
                                         Pout= active_port.power)

                # Set port as active and add the port mode to the forcing vector
                active_port.active = True
                
                solution = job._fields[active_port.port_number]

                fielddata._fields = job._fields
                fielddata.basis = self.basis
                # Compute the S-parameters
                # Define the field interpolation function
                fieldf = self.basis.interpolate_Ef(solution, tetids=all_port_tets)
                Pout = 0.0 + 0j

                # Active port power
                tris = mesh.get_triangles(active_port.tags)
                tri_vertices = mesh.tris[:,tris]
                pfield, pmode = self._compute_s_data(active_port, fieldf, tri_vertices, k0, ertri[:,:,tris], urtri[:,:,tris])
                logger.debug(f'[{active_port.port_number}] Active port amplitude = {np.abs(pfield):.3f} (Excitation = {np.abs(pmode):.2f})')
                Pout = pmode
                
                #Passive ports
                for bc in all_ports:
                    tris = mesh.get_triangles(bc.tags)
                    tri_vertices = mesh.tris[:,tris]
                    pfield, pmode = self._compute_s_data(bc, fieldf,tri_vertices, k0, ertri[:,:,tris], urtri[:,:,tris])
                    logger.debug(f'[{bc.port_number}] Passive amplitude = {np.abs(pfield):.3f}')
                    scalardata.write_S(bc.port_number, active_port.port_number, pfield/Pout)
                active_port.active=False
            
            fielddata.set_field_vector()

        logger.info('Simulation Complete!')
        self._simend = time.time()    
        logger.info(f'Elapsed time = {(self._simend-self._simstart):.2f} seconds.')

    
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
        from .sparam import sparam_field_power, sparam_mode_power
        if bc.v_integration:
            if bc.vintline is None:
                raise SimulationError('Trying to compute characteristic impedance but no integration line is defined.')
            if bc.Z0 is None:
                raise SimulationError('Trying to compute the impedance of a boundary condition with no characteristic impedance.')
            
            V = bc.vintline.line_integral(fieldfunction)
            
            if bc.active:
                if bc.voltage is None:
                    raise ValueError('Cannot compute port S-paramer with a None port voltage.')
                a = bc.voltage
                b = (V-bc.voltage)
            else:
                a = 0
                b = V
            
            a_sig = a*csqrt(1/(2*bc.Z0))
            b_sig = b*csqrt(1/(2*bc.Z0))

            return b_sig, a_sig
        else:
            if bc.modetype(k0) == 'TEM':
                const = 1/(np.sqrt((urp[0,0,:] + urp[1,1,:] + urp[2,2,:])/(erp[0,0,:] + erp[1,1,:] + erp[2,2,:])))
            if bc.modetype(k0) == 'TE':
                const = 1/((urp[0,0,:] + urp[1,1,:] + urp[2,2,:])/3)
            elif bc.modetype(k0) == 'TM':
                const = 1/((erp[0,0,:] + erp[1,1,:] + erp[2,2,:])/3)
            const = np.squeeze(const)
            field_p = sparam_field_power(self.mesh.nodes, tri_vertices, bc, k0, fieldfunction, const, 5)
            mode_p = sparam_mode_power(self.mesh.nodes, tri_vertices, bc, k0, const, 5)
            return field_p, mode_p


    ############################################################
    #                     DEPRICATED FUNCTIONS                #
    ############################################################

    def frequency_domain(self, *args, **kwargs):
        """DEPRICATED VERSION: Use run_sweep() instead.
        """
        logger.warning('This function is depricated. Please use run_sweep() instead')
        return self.run_sweep(*args, **kwargs)