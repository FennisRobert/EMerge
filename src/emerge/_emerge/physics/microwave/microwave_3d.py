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

# Last Cleanup: 2026-03-04
from ..._global import _GlobalHandler
from ...mesher import Mesher
from ...mesh3d import Mesh3D
from ...coord import Line
from ...geometry import GeoSurface, GeoVolume
from ...elements.femdata import FEMBasis
from ...elements.nedelec2 import Nedelec2
from ...elements.dofsets import DoFSet, ElementSpace
from ...solver import SolveRoutine, AutomaticRoutine
from ...system import _called_from_main_function
from ...selection import FaceSelection
from ...settings import Settings
from ...simstate import SimState
from ...const import C0
from ...attributes import PhysicalAttribute, FiniteThickness, SurfaceRoughness, MetalCoating, WavePortAttribute, LumpedElementAttribute, LumpedPortAttribute
from ..material_assignment import MaterialAssignment

from .bcs.boundary_condition_set import MWBoundaryConditionSet
from .bcs.boundary_conditions import PEC, ThinConductor, ScatteredField
from .bcs.port_bcs import ModalPort, LumpedPort, PortBC, WavePortIH, UserDefinedPort

from .microwave_data import MWData
from .assembly.assembler import Assembler
from .port_functions import compute_avg_power_flux, compute_port_power_flux
from .simjob import SimJob

from emsutil import Material
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from typing import Callable, Literal, Any
import multiprocessing as mp
from cmath import sqrt as csqrt
from itertools import product
import numpy as np
import threading
import time
from collections import defaultdict
import psutil


class SimulationError(Exception):
    pass


############################################################
#                 MULTI PROCESSING FUNCTION                #
############################################################


def run_job_multi(job: SimJob) -> SimJob:
    """The job launcher for Multi-Processing environements

    Args:
        job (SimJob): The Simulation Job

    Returns:
        SimJob: The solved SimJob
    """
    nr = int(mp.current_process().name.split("-")[1])
    routine = AutomaticRoutine()._configure_routine("MP", proc_nr=nr)
    A, bmat, ids, aux = job.get_Ab()
    solution, report = routine.solve(A, bmat, ids, matrix_type=job.mtype, id=job.id)
    report.add(**aux)
    job.submit_solution(solution, report)
    return job


############################################################
#                     RAM MEMORY CHECK                    #
############################################################


def _check_ram(ntets: int, njobs: int, parallel: bool) -> None:
    """Checks if sufficient RAM is available."""
    available = psutil.virtual_memory().available / (1024**3)

    if not parallel:
        njobs = 1

    req_low = float(ntets * 0.17 * njobs / 1024)
    req_mid = float(ntets * 0.25 * njobs / 1024)
    req_high = float(ntets * 0.40 * njobs / 1024)

    logger.debug(f"Available RAM: {available:.2f}GB")
    logger.debug(
        f"Required RAM: (low: {req_low:.2f}GB, nom: {req_mid:.2f}GB, high: {req_high:.2f}GB)"
    )

    if available < req_low:
        raise RuntimeError(
            f"Not enough free RAM detected ({available:.1f}GB), at least {req_low:.1f}GB Required. Potential swap memory not included. You can reduce the number of parallel solvers or disable this check by changing model.settings.check_ram=False."
        )
    if available < req_mid:
        raise RuntimeError(
            f"Not enough free RAM detected ({available:.1f}GB),  around {req_mid:.1f}GB Required. Potential swap memory not included. You can reduce the number of parallel solvers or disable this check by changing model.settings.check_ram=False."
        )
    if available < req_high:
        logger.warning(
            f"Low free RAM detected ({available:.1f}GB), up to {req_high:.1f}GB could be Required. You can reduce the number of parallel solvers or disable this check by changing model.settings.check_ram=False."
        )


def _dimstring(data: list[float] | np.ndarray) -> str:
    """A String formatter for dimensions in millimeters

    Args:
        data (list[float]): The list of floating point dimensions

    Returns:
        str: The formatted string
    """
    return "(" + ", ".join([f"{x * 1000:.1f}mm" for x in data]) + ")"


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


def shortest_path(xyz1: np.ndarray, xyz2: np.ndarray, Npts: int) -> np.ndarray:
    """Finds the shortest path between two sets of points

    Args:
        xyz1 (np.ndarray): _description_
        xyz2 (np.ndarray): _description_
        Npts (int): _description_

    Returns:
        np.ndarray: _description_
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


def construct_pec_contour(
    mesh: "Mesh3D",
    surface_tri_ids: np.ndarray,
    pec_vertices: set[int],
    normal_vector: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Constructs a closed contour around a pec island

    Args:
        mesh (Mesh3D): The mesh data
        surface_tri_ids (np.ndarray): A list of surface triangle indices for the port boundary
        pec_vertices (set[int]): A list of vertices inside the PEC island
        normal_vector (np.ndarray): A vector normal to the boundary
        alpha (float, optional): The relative distance from PEC vertex to edge. Defaults to 0.2.

    Returns:
        np.ndarray: _description_
    """
    normal_vector = np.asarray(normal_vector, dtype=float)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    tris = mesh.tris
    nodes = mesh.nodes

    patch_tri_indices = []
    for j in surface_tri_ids:
        v0, v1, v2 = int(tris[0, j]), int(tris[1, j]), int(tris[2, j])
        if v0 in pec_vertices or v1 in pec_vertices or v2 in pec_vertices:
            patch_tri_indices.append(j)

    if not patch_tri_indices:
        raise ValueError(
            f"No patch triangles found. {len(pec_vertices)} PEC vertices, "
            f"{len(surface_tri_ids)} surface triangles."
        )

    edge_to_tris = defaultdict(list)
    for j in patch_tri_indices:
        v0, v1, v2 = int(tris[0, j]), int(tris[1, j]), int(tris[2, j])
        verts = (v0, v1, v2)
        for k in range(3):
            va, vb = verts[k], verts[(k + 1) % 3]
            edge_key = (min(va, vb), max(va, vb))
            edge_to_tris[edge_key].append(j)

    boundary_edges = []
    edge_to_patch_tri = {}
    for edge_key, patch_tris_list in edge_to_tris.items():
        if len(patch_tris_list) == 1:
            boundary_edges.append(edge_key)
            edge_to_patch_tri[edge_key] = patch_tris_list[0]

    if not boundary_edges:
        raise ValueError(
            f"No boundary edges. Patch has {len(patch_tri_indices)} triangles."
        )

    vert_to_edges = defaultdict(list)
    for edge_key in boundary_edges:
        vert_to_edges[edge_key[0]].append(edge_key)
        vert_to_edges[edge_key[1]].append(edge_key)

    used = set()
    start_edge = boundary_edges[0]
    used.add(start_edge)
    ordered_edges = [start_edge]

    current_vertex = start_edge[1]
    start_vertex = start_edge[0]

    while current_vertex != start_vertex:
        neighbors = vert_to_edges[current_vertex]
        next_edge = None
        for e in neighbors:
            if e not in used:
                next_edge = e
                break
        if next_edge is None:
            break
        used.add(next_edge)
        ordered_edges.append(next_edge)
        if next_edge[0] == current_vertex:
            current_vertex = next_edge[1]
        else:
            current_vertex = next_edge[0]

    if len(ordered_edges) < 3:
        raise ValueError(f"Contour has only {len(ordered_edges)} edges.")

    contour_points = np.empty((3, len(ordered_edges)))
    for i, edge_key in enumerate(ordered_edges):
        tri_idx = edge_to_patch_tri[edge_key]
        tri_verts = [
            int(tris[0, tri_idx]),
            int(tris[1, tri_idx]),
            int(tris[2, tri_idx]),
        ]

        pec_verts = [v for v in tri_verts if v in pec_vertices]
        non_pec_verts = [v for v in tri_verts if v not in pec_vertices]
        n_pec = len(pec_verts)

        if n_pec == 1:
            P = nodes[:, pec_verts[0]]
            A = nodes[:, non_pec_verts[0]]
            B = nodes[:, non_pec_verts[1]]
            pa = P + alpha * (A - P)
            pb = P + alpha * (B - P)
            contour_points[:, i] = 0.5 * (pa + pb)

        elif n_pec == 2:
            A = nodes[:, pec_verts[0]]
            B = nodes[:, pec_verts[1]]
            Q = nodes[:, non_pec_verts[0]]
            pec_mid = 0.5 * (A + B)
            contour_points[:, i] = pec_mid + alpha * (Q - pec_mid)

        else:
            contour_points[:, i] = (
                nodes[:, tri_verts[0]] + nodes[:, tri_verts[1]] + nodes[:, tri_verts[2]]
            ) / 3.0

    center = contour_points.mean(axis=1, keepdims=True)
    rel = contour_points - center
    n_pts = contour_points.shape[1]
    signed_area = 0.0
    for i in range(n_pts):
        j = (i + 1) % n_pts
        cross = np.cross(rel[:, i], rel[:, j])
        signed_area += np.dot(cross, normal_vector)

    if signed_area > 0:
        contour_points = contour_points[:, ::-1]

    contour_points = np.hstack([contour_points, contour_points[:, 0:1]])
    return contour_points


############################################################
#                      MICROWAVE CLASS                     #
############################################################
"""
The Microwave class is quite verbose and contains a lot of large business logic.

"""


class Microwave3D:
    """The Electrodynamics time harmonic physics class.

    This class contains all physics dependent features to perform EM simuation in the time-harmonic
    formulation.

    """

    def __init__(
        self, state: SimState, mesher: Mesher, settings: Settings, order: int = 2
    ):

        self._settings: Settings = settings

        self.frequencies: list[float] = []
        self.current_frequency = 0
        
        self.dofset: DoFSet = ElementSpace.SECOND_MIXED_VOLAKIS.get_set()
        self.dofset_em: DoFSet = ElementSpace.SECOND_MIXED_SAVAGE.get_set()
        
        self.resolution: float = (
            0.33  # Resolution of the mesh as the fraction of the wavelength
        )

        self.mesher: Mesher = mesher  # A reference to the Mesher object
        self._state: SimState = state  # A reference to the Simulation stat object

        self.assembler: Assembler = Assembler(self._settings)  # The assembler class
        self.bc: MWBoundaryConditionSet = MWBoundaryConditionSet(
            None
        )  # The boundary condition set class.
        self.basis: Nedelec2 | None = None  # The Basis function class

        self.solveroutine: SolveRoutine = AutomaticRoutine()

        self.cache_matrices: bool = True

        ## States
        self._default_bcs_initialized: bool = False
        self._simstart: float = 0.0
        self._simend: float = 0.0
        self._container: dict[str, Any] = dict()
        self._completed: bool = False

    ############################################################
    #                          PROPERTIES                     #
    ############################################################

    @property
    def _params(self) -> dict[str, float]:
        """Returns a dict of string->float values of the current parameter sweep
        variation set. Needed when storing the simulation solution to a specific point

        """
        return self._state.params

    @property
    def mesh(self) -> Mesh3D:
        """A shorthand alias for the Mesh. May be removed at some point"""
        return self._state.mesh

    @property
    def data(self) -> MWData:
        """A shorthand alias for the Microwave simulation data. May be removed at some point"""
        return self._state.data.mw

    def reset(self, _reset_bc: bool = True):
        """Resets the Boundary condition and Microwave physics state

        Resets the Finite element basis definition, solveroutine settings and assembler settings.
        Optionally also resets the boundary conditions.

        Args:
            _reset_bc (bool, optional): If the boundary conditions should be reset. Defaults to True.
        """
        if _reset_bc:
            logger.info('Clearning Microwave boundaryconditions.')
            self.bc = MWBoundaryConditionSet(None)
            self._default_bcs_initialized = False
        else:
            logger.info('Reset boundary condition states.')
            for bc in self.bc.oftype(ModalPort):
                bc.reset()
            
        self.basis: FEMBasis = None
        self.solveroutine.reset()
        self.assembler.cached_matrices = None
        self.assembler.cached_cscmap = None

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
        return sorted(self.bc.oftype(PortBC), key=lambda x: x.number)  # type: ignore

    ############################################################
    #                            SETTERS                       #
    ############################################################

    def set_basis_space(self, 
                    element_order: Literal[1,2] = 2, 
                    complete: bool = False, 
                    elementspace: ElementSpace | None = None,
                    dof_set: DoFSet | None = None) -> None:
        """Define the finite element basis order or element space.
        
        Choices for order that are supported are:
         1 - First order
         2 - Second order (Anders and Volakis by default)
        
        Complete:
         False - Use Mixed Order basis functions. 20 total. Best for most applications
         True - Use a Complete Order basis functions. 30 total. Slower but better for models with strong
                gradient fields usually associated with fringe fields of capacitors. Examples of models
                that benefit are Patch antennas and Capacitive waveguide irises.

        Args:
            element_order (Literal[1,2]): The order of basis functions
            complete (bool, optional): If complete over mixed set of basis function is to be used. Defaults to False.
            elementspace (ElementSpace | None, optional): Manually specify the ElementSpace. Defaults to None.
            dof_set (DoFSet | None, optional): A user defined DoF set. 
        """
        if dof_set is not None:
            self.dofset = dof_set
            return
        if elementspace is not None:
            self.dofset = elementspace.get_set()
            return
        
        if element_order == 1:
            if complete:
                self.dofset = ElementSpace.FIRST_ORDER_COMPLETE.get_set()
            else:
                self.dofset = ElementSpace.FIRST_ORDER_MIXED.get_set()
        elif element_order == 2:
            if complete:
                self.dofset = ElementSpace.SECOND_COMPLETE_VOLAKIS.get_set()
            else:
                self.dofset = ElementSpace.SECOND_MIXED_VOLAKIS.get_set()
        
    def set_frequency(self, frequency: float | list[float] | np.ndarray) -> None:
        """Define the frequencies for the frequency sweep

        Args:
            frequency (float | list[float] | np.ndarray): The frequency points.
        """
        logger.info(f"Setting frequency as {frequency}Hz.")
        if isinstance(frequency, (tuple, list, np.ndarray)):
            self.frequencies = list(frequency)
        else:
            self.frequencies = [frequency]

        self.mesher.max_size = self.resolution * 299792458 / max(self.frequencies)
        self.mesher.min_size = 0.1 * self.mesher.max_size

        logger.debug(
            f"Setting global mesh size range to: {self.mesher.min_size * 1000:.3f}mm - {self.mesher.max_size * 1000:.3f}mm"
        )

    set_frequencies = set_frequency

    def set_frequency_range(self, fmin: float, fmax: float, Npoints: int) -> None:
        """Set the frequency range using the np.linspace syntax

        Args:
            fmin (float): The starting frequency
            fmax (float): The ending frequency
            Npoints (int): The number of points
        """
        self.set_frequency(np.linspace(fmin, fmax, Npoints))

    def set_resolution(self, resolution: float) -> None:
        """Define the simulation resolution as the fraction of the wavelength.

        To define the wavelength as ¼λ, call .set_resolution(0.25)

        Args:
            resolution (float): The desired wavelength fraction.

        """
        if resolution > 0.5:
            logger.warning(
                "Resolutions greater than 0.5 cannot yield accurate results, capping resolution to 0.4"
            )
            resolution = 0.4
        elif resolution > 0.334:
            logger.warning("A resolution greater than 0.33 may cause accuracy issues.")

        self.resolution = resolution
        logger.trace(f"Resolution set to {self.resolution}")

    def get_discretizer(self) -> Callable:
        """Returns a discretizer function that defines the maximum mesh size.

        Returns:
            Callable: The discretizer function
        """

        def disc(material: Material):
            return 299792458 / (
                max(self.frequencies) * np.real(material.neff(max(self.frequencies)))
            )

        return disc

    def _initialize_field(self):
        """Initializes the physics basis to the correct FEMBasis object.

        Currently it defaults to Nedelec2. Mixed basis are used for modal analysis.
        This function does not have to be called by the user. Its automatically invoked.
        """
        from ...elements.nedelec2 import Nedelec2
        if self.basis is not None:
            return
        logger.info(f'Using {self.dofset} basis functions.')
        self.basis = Nedelec2(self.mesh, self.dofset)

    ############################################################
    #                        PRIVATE METHODS                   #
    ############################################################

    def _get_frequency_groups(self, n_groups: int) -> list[list[float]]:
        """Returns the simulation frequencies in groups of n_groups frequency points.
        Used by the solve functions to bundle frequencies to be pre-assembled before solving.

        Args:
            n_groups (int): The number of frequencies per group.

        Returns:
            list[list[float]]: A list of List of simulation frequencies
        """
        freq_groups: list[list[float]] = []

        if n_groups == -1:
            freq_groups = [self.frequencies]
        else:
            n = n_groups
            freq_groups = [
                self.frequencies[i : i + n] for i in range(0, len(self.frequencies), n)
            ]

        return freq_groups

    def _autogenerate_bcs(self) -> None:
        """Initializes the boundary conditions to set PEC as all exterior boundaries."""
        if self._default_bcs_initialized:
            return
        
        logger.info("Generating default boundary conditions.")
        
        # These tags are all faces that actually terminate the simulation domain.
        external_tags = self.mesher.domain_boundary_face_tags

        # Assign PEC to all unassigned external boundaries.
        external_tags = [tag for tag in external_tags if tag not in self.bc.assigned(2)]

        if len(external_tags) > 0:
            self.bc.no_overwrite().PEC(FaceSelection(external_tags))
        
        if self.mesher.periodic_cell is not None:
            self.mesher.periodic_cell.generate_bcs()
            for bc in self.mesher.periodic_cell.bcs:
                self.bc.no_overwrite().assign(bc)

        # Assign SurfaceImpedance to all conducting volume_boundaries
        material_map = defaultdict(set)
        
        for geometry in self._state.all2d:

            # Thin Condutor from PCB Traces
            if geometry.properties.get_attr(Material, 'name')=='PEC':
                self.bc.no_overwrite().PEC(geometry.selection)
                continue

            if geometry.properties.get_attr(Material, 'cond.value', default=0.0) > self.assembler.settings.mw_3d_surfimplim:
                surface_roughness = geometry.properties.get_value(SurfaceRoughness, 0.0)
                thickness = geometry.properties.get_value(FiniteThickness, 1.0)
                
                tags_ext = [tag for tag in geometry.tags if tag in external_tags]
                tags_int = [tag for tag in geometry.tags if tag not in external_tags]

                if len(tags_ext) > 0:
                    self.bc.no_overwrite().SurfaceImpedance(
                        FaceSelection(tags_ext), geometry.material, thickness=thickness, surface_roughness=surface_roughness
                    )
                if len(tags_int) > 0:
                    self.bc.no_overwrite().ThinConductor(
                        FaceSelection(tags_int), geometry.material, thickness=thickness, surface_roughness=surface_roughness
                    )
                for attr in geometry.properties:
                    logger.info(f'   - {attr}')

            if wpattr := geometry.properties.get(WavePortAttribute):
                self.bc.ModalPort(geometry, wpattr.port_number, modetype=wpattr.mode_type)
                for attr in geometry.properties:
                    logger.info(f'   - {attr}')
            
            if wpattr := geometry.properties.get(LumpedPortAttribute):
                self.bc.LumpedPort(geometry, wpattr.port_number, width=wpattr.width, height=wpattr.height, direction=wpattr.direction)
                for attr in geometry.properties:
                    logger.info(f'   - {attr}')
        
        for geometry in self._state.all3d:
            material = geometry.material
            if material.cond.value > self.assembler.settings.mw_3d_surfimplim and material.name != 'PEC':
                for attr in geometry.properties:
                    logger.info(f'   - {attr}')
                thickness = geometry.properties.get_value(FiniteThickness, 1.0)
                surface_roughness = geometry.properties.get_value(SurfaceRoughness, 0.0)

                self.bc.no_overwrite().SurfaceImpedance(geometry.boundary(), material=material)
                material_map[material].update(set(geometry.boundary().tags))

        for material, assignment in material_map.items():
            self.bc.no_overwrite().SurfaceImpedance(FaceSelection(list(assignment)), material=material)

        self._default_bcs_initialized = True

    def _initialize_bc_data(self):
        """Initializes auxilliary required boundary condition information before running simulations."""
        logger.debug("(Microwave) Initializing boundary condition data")
        self._autogenerate_bcs()
        # Removes non-assigned boundary conditions.
        # This happens for example if the initial boundary PEC gets overwritten.
        self.bc.cleanup()

        for port in self.bc.oftype(LumpedPort):
            self._define_lumped_port_integration_points(port)

        self.bc._selections_post_boolean_fragment()

        # Process thin conductor DOF split
        thin_conductor_bcs = self.bc.oftype(ThinConductor)
        if len(thin_conductor_bcs) > 0:
            logger.debug("Processing thin conductors")
            self.basis.partition_dof([x.tags for x in thin_conductor_bcs])
            self.basis._partitioned = True

    def _check_meshed(self) -> None:
        """Checks if a mesh is generated"""
        if not self.mesh.defined:
            raise SimulationError("Mesh is not defined. Call generate_mesh() first!")

    def _check_physics(self) -> None:
        """Executes a physics check before a simulation can be run.

        Raises:
           BoundaryConditionError: If any boundary condition is not setup correctly

        """
        self.bc._is_excited()
        self.bc._check_ports()

        # Check if lumped ports are inside the domain
        exterior_tags = set(self.mesher.domain_boundary_face_tags)
        for lumped_port in self.bc.oftype(LumpedPort):
            if not set(lumped_port.selection.tags).isdisjoint(exterior_tags):
                _GlobalHandler.active().debugcollector.add_report(
                    f"Lumped port {lumped_port} is assigned to face tags that are part of the exterior boundary. Lumped ports must be strictly inside the domain. Otherwise unexpected behavior may occur."
                )

        # Check if all ports are on the exterior
        for port in self.bc.oftype(PortBC):
            if isinstance(port, LumpedPort):
                continue
            if not set(port.selection.tags).issubset(exterior_tags):
                _GlobalHandler.active().debugcollector.add_report(
                    f"Port {port} is partially not on the exterior boundary of the simulation domain. This may cause unexpected behavior. Ports should be strictly part of the exterior boundary."
                )

    def _define_lumped_port_integration_points(self, port: LumpedPort) -> None:
        """Sets the integration points on Lumped Port objects for voltage integration

        Args:
            port (LumpedPort): The LumpedPort object

        Raises:
            SimulationError: An error if there are no nodes associated with the port.
        """
        if len(port.voltage_integration_line) > 0:
            return
        logger.debug(" - Finding Lumped Port integration points")
        field_axis = port.Vdirection.np

        points = self.mesh.get_nodes(port.tags)

        if points.size == 0:
            raise SimulationError(
                f"The lumped port {port} has no nodes associated with it"
            )

        xs = self.mesh.nodes[0, points]
        ys = self.mesh.nodes[1, points]
        zs = self.mesh.nodes[2, points]

        dotprod = xs * field_axis[0] + ys * field_axis[1] + zs * field_axis[2]

        start_id = np.argwhere(dotprod == np.min(dotprod)).flatten()

        xs = xs[start_id]
        ys = ys[start_id]
        zs = zs[start_id]

        # multiple integration lines are used and eventually averaged over when computing the
        # port voltage. A unique voltage is not guaranteed in the time-harmonic fomulation
        # A multi line average numerically more accurate than a single arbitrary integration line.
        # This also removes the potential problem of arbitrarily choosing a line.

        for x, y, z in zip(xs, ys, zs):
            start = np.array([x, y, z])
            end = start + port.Vdirection.np * port.height
            port.voltage_integration_line.append(Line.from_points(start, end, 21))
            logger.trace(
                f" - Port[{port.port_number}] integration line {start} -> {end}."
            )

        port.v_integration = True

    def _find_two_pec_islands(
        self, port: ModalPort, sigtri: np.ndarray
    ) -> tuple[list[int], list[int]]:
        """Returns two lists of global node indices corresponding to the TEM port conductors.

        This method is invoked during modal analysis with TEM modes. It looks at all edges
        exterior to the boundary face triangulation and finds two small subsets of nodes that
        lie on different exterior boundaries of the boundary face.

        Args:
            port (ModalPort): The modal port object.

        Returns:
            list[int]: A list of node integers of island 1.
            list[int]: A list of node integers of island 2.
        """
        if self.basis is None:
            raise ValueError("The field basis is not yet defined.")

        logger.debug(" - Finding PEC TEM conductors")
        mesh = self.mesh

        # Find all BC conductors
        pecs: list[PEC] = self.bc.get_conductors()  # type: ignore

        # Process all PEC Boundary Conditions
        pec_edges = []
        for pec in pecs:
            face_tags = pec.tags
            tri_ids = mesh.get_triangles(face_tags)
            edge_ids = list(mesh.tri_to_edge[:, tri_ids].flatten())
            pec_edges.extend(edge_ids)

        # Process conductivity
        for itri in mesh.get_triangles(port.tags):
            if sigtri[itri] > 1e6:
                edge_ids = list(mesh.tri_to_edge[:, itri].flatten())
                pec_edges.extend(edge_ids)

        # All PEC edges
        pec_edges = list(set(pec_edges))

        # Port mesh data
        tri_ids = mesh.get_triangles(port.tags)
        edge_ids = set(list(mesh.tri_to_edge[:, tri_ids].flatten()))

        port_pec_edges = np.array([i for i in pec_edges if i in edge_ids])

        pec_islands = mesh.find_edge_groups(port_pec_edges)

        logger.debug(f" - Found {len(pec_islands)} PEC islands.")

        if len(pec_islands) != 2:
            pec_island_tags = {
                i: self.mesh._get_dimtags(edges=pec_edge_group)
                for i, pec_edge_group in enumerate(pec_islands)
            }
            plus_term = None
            min_term = None

            for i, dimtags in pec_island_tags.items():
                if not set(dimtags).isdisjoint(port.plus_terminal):
                    plus_term = i

                if not set(dimtags).isdisjoint(port.minus_terminal):
                    min_term = i

            if plus_term is None or min_term is None:
                logger.error(
                    f" - Found {len(pec_islands)} PEC islands without a terminal definition. Please use .set_terminals() to define which conductors are which polarity, or define the integration line manually."
                )
                return None, None
            logger.debug(f"Positive island = {pec_island_tags[plus_term]}")
            logger.debug(f"Negative island = {pec_island_tags[min_term]}")
            pec_islands = [pec_islands[plus_term], pec_islands[min_term]]

        groups = []
        for island in pec_islands:
            group = set()
            for edge in island:
                group.add(mesh.edges[0, edge])
                group.add(mesh.edges[1, edge])
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

            if bc.forced_modetype == "TEM":
                TEM = True
            else:
                TEM = False
            self.modal_analysis(
                bc, bc._desired_number_of_modes, direct=False, freq=freq, TEM=TEM
            )

            bc._check_mode_betas()

    def _get_material_assignment(self) -> MaterialAssignment:
        """Retrieve the material properties of the geometry"""

        # In order to make EMerge projects saveable, the Materials are told which
        # geometries they have been assigned to. These material lists are stored in the final solution
        # The reason is that per simulation and frequency, the material propery value may be different.

        ma = MaterialAssignment(self._state.current_geo_state)
        ma.set_tet_assignment(self.mesh._get_tet_to_tag())
        
        return ma
    ############################################################
    #                   MAIN SIMULATION FUNCTIONS              #
    ############################################################

    def modal_analysis(
        self,
        port: ModalPort,
        nmodes: int = 6,
        direct: bool = True,
        TEM: bool = False,
        target_kz: float | None = None,
        target_neff: float | None = None,
        freq: float | None = None,
    ) -> None:
        """Execute a modal analysis on a given ModalPort boundary condition.

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
        """
        T0 = time.time()
        logger.info(f"Port {port.port_number}: Starting Mode Analysis for port {port}.")

        # First check if the boundary conditions are initialized.

        if self.bc._initialized_with_defaults is False:
            raise SimulationError(
                "Cannot run a modal analysis because no boundary conditions have been assigned."
            )

        # Initialize the FEM field and boundary condition data
        self._check_meshed()
        self._initialize_field()
        self._initialize_bc_data()

        if self.basis is None:
            raise SimulationError(
                "Cannot proceed, the current basis class is undefined."
            )

        logger.debug(" - retreiving material properties.")

        # If no simulation frequency is provided, use the lowest frequency
        if freq is None:
            freq = self.frequencies[0]

        # Align port normal vector
        inward_normal = self.mesh.inward_normal(port.tags)
        if sum(port.cs.zax.np * inward_normal) < 0:
            port.cs.flip_z()
        # Materials is now a list of materials and in the materials themselves
        # They know what values are assigned to which index. This is not uniform
        # because coordinate dependent material properties/materials are possible.
        materials = self._get_material_assignment(self.mesher.volumes)

        # Er, Tand, ur, and conductivity parameters are always 3 by 3 full
        # material property tensors per tetrahedron.
        # They are assumed constant within each tetrahedron.

        ertet = np.zeros((3, 3, self.mesh.n_tets), dtype=np.complex128)
        tandtet = np.zeros((3, 3, self.mesh.n_tets), dtype=np.complex128)
        urtet = np.zeros((3, 3, self.mesh.n_tets), dtype=np.complex128)
        condtet = np.zeros((3, 3, self.mesh.n_tets), dtype=np.complex128)

        # Evaluating the relavant function er, tand etc on these functions automatically
        # only assigned these material values to those arrays where they are valid.
        # The materials know (high coupling, I know) which tet-indices they are assigned to.
        for mat in materials:
            ertet = mat.er(freq, ertet)
            tandtet = mat.tand(freq, tandtet)
            urtet = mat.ur(freq, urtet)
            condtet = mat.cond(freq, condtet)

        # Compute the complex dielectric constant with the loss tangent.
        ertet = ertet * (1 - 1j * tandtet)

        # For boundary modes we need to know the material properties on the surfaces instead of in the tets.
        er = np.zeros((3, 3, self.mesh.n_tris), dtype=np.complex128)
        ur = np.zeros((3, 3, self.mesh.n_tris), dtype=np.complex128)
        cond = np.zeros((self.mesh.n_tris,), dtype=np.complex128)

        # We can use tri_to_tet to "ask" which tetrahedra are connected to the boundary triangle
        # and then use that value for our material assignment
        # Modal ports may only be connected to one tetrahedron so we can safely assume that the only
        # Assigned tet index is in array index 0.
        for itri in range(self.mesh.n_tris):
            itet = self.mesh.tri_to_tet[0, itri]
            er[:, :, itri] = ertet[:, :, itet]
            ur[:, :, itri] = urtet[:, :, itet]
            cond[itri] = condtet[0, 0, itet]

        # A list of all the triangles that are part of the Port.
        itri_port = self.mesh.get_triangles(port.tags)

        # Compute mean values for each material property.
        # These values are used to coarsely estimate what the
        # out of plane propagation constant may be.
        ermean = np.mean(er[er > 0].flatten()[itri_port])
        urmean = np.mean(ur[ur > 0].flatten()[itri_port])
        ermax = np.max(er[:, :, itri_port].flatten())
        urmax = np.max(ur[:, :, itri_port].flatten())
        ermin = np.min(er[:, :, itri_port].flatten())
        urmin = np.min(ur[:, :, itri_port].flatten())

        # Compute k0
        k0 = 2 * np.pi * freq / 299792458

        logger.debug(
            f" - mean(max): εr = {ermean:.2f}({ermax:.2f}), μr = {urmean:.2f}({urmax:.2f})"
        )

        # Assemble matrix A and B for the generalized eigenvalue problem: Ax = λBx
        Amatrix, Bmatrix, solve_ids, nlf = self.assembler.assemble_bma_matrices(
            self.basis, er, ur, cond, k0, port, self.bc, self.dofset_em
        )

        logger.debug(f"Total of {Amatrix.shape[0]} Degrees of freedom.")
        logger.debug(f"Applied frequency: {freq / 1e9:.2f}GHz")
        logger.debug(f"K0 = {k0} rad/m")

        # neff is the effective index for the port mode defined as kz = k0*neff
        # If specified, the search kz is k0*neff
        if target_neff is not None:
            target_kz = k0 * target_neff

        # If no target kz is provided, we first ask the port. In case of prior
        # Solves, the port can better estimate the port mote for this new frequency.
        # If there is no estimate we use a different value for TEM modes than TE modes
        # Which are fast modes as is the case with Waveguides for example.
        if target_kz is None:
            neff_estimate = port.neff_estimate(k0)
            if neff_estimate is not None:
                target_kz = neff_estimate * k0
            target_kz = ((ermax * urmax + ermin * urmin) / 2) ** 0.5 * 1.0 * k0

        logger.debug(f"Solving for {solve_ids.shape[0]} degrees of freedom.")

        # Only look at the real value
        target_kz = target_kz.real

        # Execute a boundary mode eigenvalue calculation.
        eigen_values, eigen_modes, report = self.solveroutine.eig_boundary(
            Amatrix, Bmatrix, solve_ids, nmodes, direct, target_kz, sign=-1
        )

        logger.debug(f"Eigenvalues: {np.sqrt(-eigen_values)} rad/m")

        # Store the material properties in the port.
        # Can't remember where this was used again.
        # I think nowhere.

        # port._er = er
        # port._ur = ur

        nmodes_found = eigen_values.shape[0]

        # For each found mode
        for i in range(nmodes_found):
            # The E-field must
            Emode = np.zeros((nlf.n_field,), dtype=np.complex128)
            eigenmode = eigen_modes[:, i]
            Emode[solve_ids] = np.squeeze(eigenmode)

            # Compute the out of plane propagation constant
            beta = np.emath.sqrt(-eigen_values[i])
            # beta = min(k0 * np.sqrt(ermax * urmax), beta_base)

            # Correct for et = kz Et, ez = -j Ez
            Emode[: nlf.n_xy] = Emode[: nlf.n_xy] / beta
            Emode[nlf.n_xy :] = Emode[nlf.n_xy :] / (-1j)

            # Phase correct for a phase of 0 degree at the port maximum.
            Emode = Emode * np.exp(
                -1j * np.angle(Emode[np.argmax(np.abs(Emode[: nlf.n_xy]))])
            )

            # If alignement vectors are defined (ask the port to be aligned positively in some direction.)
            if port._get_alignment_vector(i) is not None:
                # Compute the E-field, check in which direction it points on average and make sure the dot-product is positive.
                vec = port._get_alignment_vector(i)
                xyz_centers = self.mesh.tri_centers[
                    :, self.mesh.get_triangles(port.tags)
                ]
                E_centers = np.mean(
                    nlf.interpolate_Ef(Emode)(
                        xyz_centers[0, :], xyz_centers[1, :], xyz_centers[2, :]
                    ),
                    axis=1,
                )
                EdotVec = (
                    vec[0] * E_centers[0]
                    + vec[1] * E_centers[1]
                    + vec[2] * E_centers[2]
                )
                if EdotVec.real < 0:
                    logger.debug(
                        f"Mode polarization along alignment axis {vec} = {EdotVec.real:.3f}, inverting."
                    )
                    Emode = -Emode

            # These return field interpolation class objects.
            portfE = nlf.interpolate_Ef(Emode)
            portfH = nlf.interpolate_Hf(Emode, k0, ur, beta)

            logger.debug("Inferring port mode type...")

            tri_centers = self.mesh.tri_centers[
                :, self.mesh.get_triangles(port.selection.tags)
            ]

            Efxy = portfE.calcExy(
                tri_centers[0, :], tri_centers[1, :], tri_centers[2, :]
            )
            Efz = portfE.calcEz(tri_centers[0, :], tri_centers[1, :], tri_centers[2, :])
            Hfxy = portfH.calcHxy(
                tri_centers[0, :], tri_centers[1, :], tri_centers[2, :]
            )
            Hfz = portfH.calcHz(tri_centers[0, :], tri_centers[1, :], tri_centers[2, :])

            Exy = np.max(np.abs(Efxy))
            Ez = np.max(np.abs(Efz))
            Hxy = np.max(np.abs(Hfxy))
            Hz = np.max(np.abs(Hfz))

            logger.debug(f".. Ez/Exy = {Ez / Exy:.3f}")
            logger.debug(f".. Hz/Hxy = {Hz / Hxy:.3f}")

            TE = False
            TM = False
            if Ez / Exy < self._settings.qtem_limit:
                TE = True
            if Hz / Hxy < self._settings.qtem_limit:
                TM = True

            if TE and TM:
                mode_type = "TEM"
            elif TE:
                mode_type = "TE"
            elif TM:
                mode_type = "TM"
            else:
                mode_type = "TEM"
            logger.debug(f".. Mode type = {mode_type}")

            # Figure out if TE, TM, or TEM mode
            if port.forced_modetype is not None and port.forced_modetype.lower().strip() != 'any':
                if port.forced_modetype != mode_type:
                    logger.debug(
                        f".. Ignoring port mode({mode_type}) because its not a {port.forced_modetype} mode."
                    )
                    continue

            # Finally add the mode to the port as a valid port mode.
            mode = port.try_add_mode(Emode, portfE, portfH, beta, k0, freq=freq)

            # Mode is None if the mode energy is considered too.
            if mode is None:
                continue

            mode.modetype = mode_type

            # port power normalization
            inward_normal = self.mesh.inward_normal(port.tags)
            power = compute_avg_power_flux(nlf, portfE, portfH, inward_normal)
            logger.debug(f".. Port power flux = {power:.3e} [W]")
            mode.normalize_power(np.abs(power))
            logger.debug(f".. Inward normal = {inward_normal}")
            # flip port mode direction
            if power < 0.0:
                logger.warning(
                    "Negative port power, unhandled case. Its not clear if this even matters."
                )
            power = 1.0

            # Final port post processing. If the port is to be considered a TEM mode port:
            if mode_type == "TEM" or TEM:
                mode.modetype = "TEM"

                impedance_type = port.impedance_definition

                voltage = 0.0
                current = 0.0

                # Find so called conductor islands. For TEM modes these are groups of vertices that make up the
                # different port conductors.
                island_group_1 = None
                island_group_2 = None

                if "V" in impedance_type:
                    # Voltage needs to be calculated. Either find an integration line or take the predefined one.
                    if len(port.voltage_integration_line) > 0:
                        line = port.voltage_integration_line[0]
                    else:
                        island_group_1, island_group_2 = self._find_two_pec_islands(
                            port, sigtri=cond
                        )
                        if island_group_1 is None or island_group_2 is None:
                            logger.warning(
                                "Skipping characteristic impedance calculation."
                            )
                            continue

                        nodes1 = self.mesh.nodes[:, island_group_1]
                        nodes2 = self.mesh.nodes[:, island_group_2]
                        path = shortest_path(nodes1, nodes2, 2)
                        line = Line.from_points(path[:, 0], path[:, 1], 101)
                        port.voltage_integration_line.append(line)

                    line_centers = np.array(line.cmid)

                    logger.debug(
                        f"Integrating portmode from {line_centers[:, 0]} to {line_centers[:, -1]}"
                    )
                    voltage = line.line_integral(mode.E_function)

                # If the Current needs to be computed, either find the integration line or use the predefined one.
                if "I" in impedance_type:
                    if len(port.current_integration_line) > 0:
                        intline = port.current_integration_line[0]
                    else:
                        if island_group_1 is None or island_group_2 is None:
                            island_group_1, island_group_2 = self._find_two_pec_islands(
                                port, sigtri=cond
                            )
                        if island_group_1 is None or island_group_2 is None:
                            logger.warning(
                                "Skipping characteristic impedance calculation."
                            )
                            continue
                        path = construct_pec_contour(
                            self.mesh,
                            itri_port,
                            set(island_group_1),
                            inward_normal,
                            alpha=0.2,
                        )
                        intline = (
                            Line(path[0, :], path[1, :], path[2, :])
                            .subsample(20)
                            .smooth(5)
                        )
                        port.current_integration_line.append(intline)
                    current = intline.line_integral(mode.H_function.calcHxy)

                # Align mode polarity to positive voltage
                if voltage < 0:
                    mode.flip_polarity()

                logger.debug(
                    f".. U={voltage:.3f}[V], I={1000 * current:.3f}[mA], Power={power:.3F}[W]"
                )
                voltage = np.abs(voltage)
                current = np.abs(current)
                power = np.abs(power)

                if impedance_type == "PV":
                    mode.Z0 = abs(voltage**2 / (2 * power))
                elif impedance_type == "PI":
                    mode.Z0 = 2 * np.abs(power) / (np.abs(current) ** 2)
                elif impedance_type == "VI":
                    mode.Z0 = np.abs(voltage) / np.abs(current)

                logger.info(f"Port Z0 = {mode.Z0} Ω ({impedance_type})")

        # Sort the port modes on propagation constant.
        port.sort_modes()

        logger.info(f"Total of {port.nmodes} found")

        T2 = time.time()
        logger.info(f"Elapsed time = {(T2 - T0):.2f} seconds.")
        return None

    def run_sweep(
        self,
        parallel: bool = False,
        n_workers: int = 2,
        harddisc_path: str = "EMergeSparse",
        frequency_groups: int = -1,
        cache_harddisk: bool = False,
        multi_processing: bool = False,
        automatic_modal_analysis: bool = True,
        _reset_solvers: bool = True,
    ) -> MWData:
        """Executes a frequency domain study

        The study is distributed over "n_workers" workers.
        As optional parameter you may set cache_harddisk which will write the matrices to the harddsik. The
        path that will be used to cache the sparse matrices can be specified.
        Additionally the term frequency_groups may be specified. This number will define in how
        many groups the matrices will be pre-computed before they are send to workers. This can minimize
        the total amound of RAM memory used. For example with 11 frequencies in gruops of 4, the following
        frequency indices will be precomputed and then solved: [[1,2,3,4],[5,6,7,8],[9,10,11]]

        Args:
            n_workers (int, optional): The number of workers. Defaults to 2.
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

        # --------------------------------------------------------------------
        # States
        # --------------------------------------------------------------------

        self._completed = False
        self._simstart = time.time()
        self.solveroutine.symmetry_limit = self._settings.sim_symmetry_limit

        # --------------------------------------------------------------------
        # Local Variables
        # --------------------------------------------------------------------

        simulation_jobs: list[SimJob] = []
        material_set: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        job_counter: int = 1
        harddisc_path = str(self._state.modelpath / harddisc_path)

        ############################################################
        #                          FREQUENCY CHECK                 #
        ############################################################

        # Safety tests
        if len(self.frequencies) > 200:
            _GlobalHandler.active().debugcollector.add_report(
                f"More than 200 frequency points are detected ({len(self.frequencies)}). This may cause slow simulations. Consider using Vector Fitting to subsample S-parameters."
            )
        if min(self.frequencies) < 1e6:
            _GlobalHandler.active().debugcollector.add_report(
                f"A frequency smaller than 1MHz has been detected ({min(self.frequencies)} Hz). Perhaps you forgot to include usints like 1e6 for MHz etc."
            )
        if max(self.frequencies) > 1e12:
            _GlobalHandler.active().debugcollector.add_report(
                f"A frequency greater than THz has been detected ({min(self.frequencies)} Hz). Perhaps you double counted frequency units like twice 1e6 for MHz etc."
            )

        # --------------------------------------------------------------------
        # Checks
        # --------------------------------------------------------------------

        self._check_meshed()
        self._initialize_field()
        self._initialize_bc_data()
        self._check_physics()
        
        if self.basis is None:
            raise SimulationError(
                "Cannot proceed, the simulation basis class is undefined."
            )

        if self._settings.check_ram:
            _check_ram(self.mesh.n_tets, n_workers, parallel)

        logger.info(
            f"Starting frequency domain simulation (#tets = {self.mesh.n_tets:,})"
        )

        # --------------------------------------------------------------------
        # Material Assignments
        # --------------------------------------------------------------------

        logger.debug("Resolving material assingments.")
        mat_assy = self._get_material_assignment()

        # --------------------------------------------------------------------
        # Port BC prepratation
        # --------------------------------------------------------------------

        all_ports = self.bc.oftype(PortBC)
        for port in all_ports:
            port.active = False

        # --------------------------------------------------------------------
        # Initializing solve functions
        # --------------------------------------------------------------------

        thread_local = None

        if parallel:
            # Thread-local storage for per-thread resources
            thread_local = threading.local()

        def get_routine() -> SolveRoutine:
            if not hasattr(thread_local, "routine"):
                worker_nr = int(threading.current_thread().name.split("_")[1]) + 1
                thread_local.routine = self.solveroutine.duplicate()._configure_routine(
                    "MT", thread_nr=worker_nr
                )
            return thread_local.routine

        # Multi processing solves first need a SolveRoutine object to execute the solve.
        def run_job(job: SimJob) -> SimJob:
            routine = get_routine()
            A, bmat, ids, aux = job.get_Ab()
            solution, report = routine.solve(
                A, bmat, ids, matrix_type=job.mtype, id=job.id
            )
            report.add(**aux)
            job.submit_solution(solution, report)
            return job

        # Single threaded does not need this routine as the class's own routine can be used.
        def run_job_single(job: SimJob) -> SimJob:
            A, bmat, ids, aux = job.get_Ab()
            solution, report = self.solveroutine.solve(
                A, bmat, ids, matrix_type=job.mtype, id=job.id
            )
            report.add(**aux)
            job.submit_solution(solution, report)
            return job

        # --------------------------------------------------------------------
        # Grouping solve frequencies
        # --------------------------------------------------------------------

        # Frequency groups are the groups of frequencie that are assembled and then solved
        # Less assembled problems reduces RAM usage.
        freq_groups = self._get_frequency_groups(frequency_groups)

        for i, group in enumerate(freq_groups):
            group_GHz = [f"{f / 1e9:.3f}GHz" for f in group]
            logger.trace(f"Frequency group ({i}): {group_GHz}")

        # I am not sure if this is supposed to be there
        self._compute_modes(sum(self.frequencies) / len(self.frequencies))

        # --------------------------------------------------------------------
        # Single Threaded Solve
        # --------------------------------------------------------------------

        if not parallel:
            for i_group, fgroup in enumerate(freq_groups):
                logger.info(f"Precomputing group {i_group}.")
                jobs = []

                ## Assemble jobs
                # for each frequency in the frequency group
                for ifreq, freq in enumerate(fgroup):
                    logger.debug(f"Simulation frequency = {_format_freq(freq)}")

                    # Execute a new modal analysis if needed (only in cases where the propagation constant can't be predicted.)
                    if automatic_modal_analysis:
                        self._compute_modes(freq)

                    # Assemble the FEM problem
                    job, mats = self.assembler.assemble_freq_matrix(
                        self.basis,
                        mat_assy,
                        self.bc.boundary_conditions,
                        freq,
                        cache_matrices=self.cache_matrices,
                    )
                    # Cache to the harddrive if needed.
                    if cache_harddisk:
                        job.store_if_needed(harddisc_path)

                    job.id = job_counter
                    job_counter += 1
                    jobs.append(job)
                    material_set.append(mats)
                # Finally solve the problems for each frequency individually.
                logger.info(f"Starting single threaded solve of {len(jobs)} jobs.")
                group_results = [run_job_single(job) for job in jobs]
                simulation_jobs.extend(group_results)

        # --------------------------------------------------------------------
        # Multi-Threaded Solve
        # --------------------------------------------------------------------

        elif not multi_processing:
            with ThreadPoolExecutor(
                max_workers=n_workers, thread_name_prefix="WKR"
            ) as executor:
                # ITERATE OVER FREQUENCIES
                for i_group, fgroup in enumerate(freq_groups):
                    logger.info(f"Precomputing group {i_group}.")
                    jobs = []
                    ## Assemble jobs
                    for freq in fgroup:
                        logger.debug(f"Simulation frequency = {_format_freq(freq)}")
                        # Compute new port modes if needed.
                        if automatic_modal_analysis:
                            self._compute_modes(freq)

                        # Assemble the problem Ax=b
                        job, mats = self.assembler.assemble_freq_matrix(
                            self.basis,
                            materials,
                            self.bc.boundary_conditions,
                            freq,
                            cache_matrices=self.cache_matrices,
                        )
                        # Cache to the harddrive if needed
                        if cache_harddisk:
                            job.store_if_needed(harddisc_path)

                        job.id = job_counter
                        job_counter += 1
                        jobs.append(job)
                        material_set.append(mats)

                    # Solve all problems using the executor.
                    logger.info(
                        f"Starting distributed solve of {len(jobs)} jobs with {n_workers} threads."
                    )
                    group_results = list(executor.map(run_job, jobs))
                    simulation_jobs.extend(group_results)
                executor.shutdown()

        # --------------------------------------------------------------------
        # Multi-Processing Solve
        # --------------------------------------------------------------------

        else:
            # Check for entry point protection
            if not _called_from_main_function():
                raise SimulationError(
                    "Multiprocess support must be launched from your "
                    "if __name__ == '__main__' guard in the top-level script."
                )
            # Start parallel pool
            with mp.Pool(processes=n_workers) as pool:
                for i_group, fgroup in enumerate(freq_groups):
                    logger.debug(f"Precomputing group {i_group}.")
                    jobs = []
                    # Assemble jobs
                    for freq in fgroup:
                        logger.debug(f"Simulation frequency = {_format_freq(freq)}")
                        # Execute modal analysis if needed
                        if automatic_modal_analysis:
                            self._compute_modes(freq)

                        # Assemble the problem Ax=b
                        job, mats = self.assembler.assemble_freq_matrix(
                            self.basis,
                            materials,
                            self.bc.boundary_conditions,
                            freq,
                            cache_matrices=self.cache_matrices,
                        )

                        # Cache to the harddrive if needed
                        if cache_harddisk:
                            job.store_if_needed(harddisc_path)

                        job.id = job_counter
                        job_counter += 1
                        jobs.append(job)
                        material_set.append(mats)

                    logger.info(
                        f"Starting distributed solve of {len(jobs)} jobs "
                        f"with {n_workers} processes in parallel"
                    )
                    # Distribute taks
                    group_results = pool.map(run_job_multi, jobs)
                    simulation_jobs.extend(group_results)

        # --------------------------------------------------------------------
        # Cleanup
        # --------------------------------------------------------------------

        if parallel:
            thread_local.__dict__.clear()

        if _reset_solvers:
            self.solveroutine.reset()

        logger.info("Solving complete")

        # --------------------------------------------------------------------
        # Writing solve reports
        # --------------------------------------------------------------------

        for freq, job in zip(self.frequencies, simulation_jobs):
            self.data.setreport(job.reports, freq=freq, **self._params)

        for variables, data in self.data.sim.iterate():
            logger.trace(f"Sim variable: {variables}")
            for item in data["report"]:
                item.logprint(logger.trace)

        # --------------------------------------------------------------------
        # Post Processing
        # --------------------------------------------------------------------

        self._post_process(simulation_jobs, material_set)
        self._completed = True
        return self.data

    def run_adaptive_sweep(
        self,
        parallel: bool = False,
        n_workers: int = 2,
        n_initial: int = 5,
        n_max_new: int = 4,
        harddisc_path: str = "EMergeSparse",
        frequency_groups: int = -1,
        cache_harddisk: bool = False,
        multi_processing: bool = False,
        automatic_modal_analysis: bool = True,
        _reset_solvers: bool = True,
    ) -> MWData:
        """Executes a frequency domain study

        The study is distributed over "n_workers" workers.
        As optional parameter you may set a harddisc_threshold as integer. This determines the maximum
        number of degrees of freedom before which the jobs will be cahced to the harddisk. The
        path that will be used to cache the sparse matrices can be specified.
        Additionally the term frequency_groups may be specified. This number will define in how
        many groups the matrices will be pre-computed before they are send to workers. This can minimize
        the total amound of RAM memory used. For example with 11 frequencies in gruops of 4, the following
        frequency indices will be precomputed and then solved: [[1,2,3,4],[5,6,7,8],[9,10,11]]

        Args:
            n_workers (int, optional): The number of workers. Defaults to 2.
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

        f_target = np.array(self.frequencies)
        fmin = min(f_target)
        fmax = max(f_target)

        def getf(f1: float, f2: float) -> np.ndarray:
            sub = f_target[(f_target >= f1) & (f_target <= f2)]
            return np.linspace(f1, f2, max(len(sub), 21))

        def _smat_ok(Smat, errs) -> bool:
            return np.all(np.abs(Smat.flatten()) <= 1.0) and np.all(
                np.abs(errs.flatten()) < 0.01
            )

        frequency_set = list(np.linspace(fmin, fmax, max(n_initial, n_workers)))
        self.set_frequencies(frequency_set)
        istart = max(0, self.data.scalar.n - 1)

        converged = False
        last_converged = False
        N = len(frequency_set)
        while not converged:
            logger.info(f"Adaptive sweep at frequencies: {self.frequencies} GHz")
            dataset = self.run_sweep(
                parallel,
                n_workers=n_workers,
                harddisc_path=harddisc_path,
                frequency_groups=frequency_groups,
                cache_harddisk=cache_harddisk,
                multi_processing=multi_processing,
                automatic_modal_analysis=automatic_modal_analysis,
                _reset_solvers=False,
            )
            sgrid = dataset.scalar.slice_set(istart, None, sort_by="freq").grid

            newF = []

            if _smat_ok(
                sgrid.model_Smat(f_target), sgrid.model_Smat(sgrid.freq) - sgrid.Smat
            ):
                if not last_converged:
                    last_converged = True

                    dFi = sorted(
                        [
                            (i, f2 - f1)
                            for i, (f1, f2) in enumerate(
                                zip(frequency_set[:-1], frequency_set[1:])
                            )
                        ],
                        key=lambda df: df[1],
                        reverse=True,
                    )
                    for i, df in dFi[:4]:
                        newF.append(frequency_set[i] + df / 2)

                else:
                    logger.info(f"Adaptive sweep converged! with {N} total simulations")
                    break

            for i1, (f1, f2) in enumerate(zip(frequency_set[:-1], frequency_set[1:])):
                i2 = i1 + 1
                Smat = sgrid.model_Smat(getf(f1, f2), _warn=False)
                er1 = Smat[0, :, :] - sgrid.Smat[i1, :, :]
                er2 = Smat[-1, :, :] - sgrid.Smat[i2, :, :]
                if (
                    np.any(np.abs(er1.flatten() > 0.01))
                    or np.any(np.abs(er2.flatten() > 0.01))
                    or np.any(np.abs(Smat.flatten()) > 1.0)
                ):
                    newF.append((f1 * f2) ** 0.5)
                    logger.debug(f"  Adding {newF[-1] / 1e9} GHz as new sample point.")

            # if len(newF) > n_max_new:
            #     newF = newF[0:n_max_new]
            newF = sorted(newF)
            N += len(newF)
            frequency_set = sorted(frequency_set + newF)
            logger.debug(f"Resimulating at {newF}")
            self.set_frequencies(newF)

        self.solveroutine.reset()
        return self.data

    def run_scattered(
        self,
        parallel: bool = False,
        n_workers: int = 2,
        cache_harddisk: bool = False,
        harddisc_path: str = "EMergeSparse",
        frequency_groups: int = -1,
        multi_processing: bool = False,
        automatic_modal_analysis: bool = True,
    ) -> MWData:
        """Executes a Scattered field simulation

        The study is distributed over "n_workers" workers.
        As optional parameter you may set cache_harddisk which will write the matrices to the harddsik. The
        path that will be used to cache the sparse matrices can be specified.
        Additionally the term frequency_groups may be specified. This number will define in how
        many groups the matrices will be pre-computed before they are send to workers. This can minimize
        the total amound of RAM memory used. For example with 11 frequencies in gruops of 4, the following
        frequency indices will be precomputed and then solved: [[1,2,3,4],[5,6,7,8],[9,10,11]]

        Args:
            n_workers (int, optional): The number of workers. Defaults to 2.
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

        # --------------------------------------------------------------------
        # States
        # --------------------------------------------------------------------

        self._completed = False
        self._simstart = time.time()
        self.solveroutine.symmetry_limit = self._settings.sim_symmetry_limit

        # --------------------------------------------------------------------
        # Local Variables
        # --------------------------------------------------------------------

        results: list[SimJob] = []
        material_set: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        job_counter: int = 1

        # --------------------------------------------------------------------
        # Checks
        # --------------------------------------------------------------------
        
        self._check_meshed()
        self._initialize_field()
        self._initialize_bc_data()
        self._check_physics()

        if self.basis is None:
            raise SimulationError(
                "Cannot proceed, the simulation basis class is undefined."
            )

        if self._settings.check_ram:
            _check_ram(self.mesh.n_tets, n_workers, parallel)

        logger.info(
            f"Starting frequency domain simulation (#tets = {self.mesh.n_tets:,})"
        )

        # --------------------------------------------------------------------
        # Material Assignments
        # --------------------------------------------------------------------

        logger.debug("Resolving material assingments.")
        materials = self._get_material_assignment(self.mesher.volumes)

        # --------------------------------------------------------------------
        # Initializing solve functions
        # --------------------------------------------------------------------

        thread_local = None

        if parallel:
            # Thread-local storage for per-thread resources
            thread_local = threading.local()

        def get_routine() -> SolveRoutine:
            if not hasattr(thread_local, "routine"):
                worker_nr = int(threading.current_thread().name.split("_")[1]) + 1
                thread_local.routine = self.solveroutine.duplicate()._configure_routine(
                    "MT", thread_nr=worker_nr
                )
            return thread_local.routine

        def run_job(job: SimJob) -> SimJob:
            routine = get_routine()
            A, bmat, ids, aux = job.get_Ab()
            solution, report = routine.solve(
                A, bmat, ids, matrix_type=job.mtype, id=job.id
            )
            report.add(**aux)
            job.submit_solution(solution, report)
            return job

        def run_job_single(job: SimJob) -> SimJob:
            A, bmat, ids, aux = job.get_Ab()
            solution, report = self.solveroutine.solve(
                A, bmat, ids, matrix_type=job.mtype, id=job.id
            )
            report.add(**aux)
            job.submit_solution(solution, report)
            return job

        # --------------------------------------------------------------------
        # Grouping solve frequencies
        # --------------------------------------------------------------------

        freq_groups = self._get_frequency_groups(frequency_groups)
        for i, group in enumerate(freq_groups):
            group_GHz = [f"{f / 1e9:.3f}GHz" for f in group]
            logger.trace(f"Frequency group ({i}): {group_GHz}")

        # I am not sure if this is supposed to be there
        self._compute_modes(sum(self.frequencies) / len(self.frequencies))

        # --------------------------------------------------------------------
        # Single Threaded Solve
        # --------------------------------------------------------------------

        if not parallel:
            for i_group, fgroup in enumerate(freq_groups):
                logger.info(f"Precomputing group {i_group}.")
                jobs = []
                ## Assemble jobs
                for ifreq, freq in enumerate(fgroup):
                    logger.debug(f"Simulation frequency = {_format_freq(freq)}")
                    if automatic_modal_analysis:
                        self._compute_modes(freq)
                    job, mats = self.assembler.assemble_scattering_matrix(
                        self.basis,
                        materials,
                        self.bc.boundary_conditions,
                        freq,
                        cache_matrices=self.cache_matrices,
                    )
                    if cache_harddisk:
                        job.store_if_needed(harddisc_path)
                    job.id = job_counter
                    job_counter += 1
                    jobs.append(job)
                    material_set.append(mats)

                logger.info(f"Starting single threaded solve of {len(jobs)} jobs.")
                group_results = [run_job_single(job) for job in jobs]
                results.extend(group_results)

        # --------------------------------------------------------------------
        # Multi-Threaded Solve
        # --------------------------------------------------------------------

        elif not multi_processing:
            with ThreadPoolExecutor(
                max_workers=n_workers, thread_name_prefix="WKR"
            ) as executor:
                # ITERATE OVER FREQUENCIES
                for i_group, fgroup in enumerate(freq_groups):
                    logger.info(f"Precomputing group {i_group}.")
                    jobs = []
                    ## Assemble jobs
                    for freq in fgroup:
                        logger.debug(f"Simulation frequency = {_format_freq(freq)}")
                        if automatic_modal_analysis:
                            self._compute_modes(freq)
                        job, mats = self.assembler.assemble_scattering_matrix(
                            self.basis,
                            materials,
                            self.bc.boundary_conditions,
                            freq,
                            cache_matrices=self.cache_matrices,
                        )
                        if cache_harddisk:
                            job.store_if_needed(harddisc_path)
                        job.id = job_counter
                        job_counter += 1
                        jobs.append(job)
                        material_set.append(mats)

                    logger.info(
                        f"Starting distributed solve of {len(jobs)} jobs with {n_workers} threads."
                    )
                    group_results = list(executor.map(run_job, jobs))
                    results.extend(group_results)
                executor.shutdown()

        # --------------------------------------------------------------------
        # Multi-Processing Solve
        # --------------------------------------------------------------------

        else:
            # Check for entry point protection
            if not _called_from_main_function():
                raise SimulationError(
                    "Multiprocess support must be launched from your "
                    "if __name__ == '__main__' guard in the top-level script."
                )
            # Start parallel pool
            with mp.Pool(processes=n_workers) as pool:
                for i_group, fgroup in enumerate(freq_groups):
                    logger.debug(f"Precomputing group {i_group}.")
                    jobs = []
                    # Assemble jobs
                    for freq in fgroup:
                        logger.debug(f"Simulation frequency = {_format_freq(freq)}")
                        if automatic_modal_analysis:
                            self._compute_modes(freq)

                        job, mats = self.assembler.assemble_scattering_matrix(
                            self.basis,
                            materials,
                            self.bc.boundary_conditions,
                            freq,
                            cache_matrices=self.cache_matrices,
                        )

                        if cache_harddisk:
                            job.store_if_needed(harddisc_path)
                        job.id = job_counter
                        job_counter += 1
                        jobs.append(job)
                        material_set.append(mats)

                    logger.info(
                        f"Starting distributed solve of {len(jobs)} jobs "
                        f"with {n_workers} processes in parallel"
                    )
                    # Distribute taks
                    group_results = pool.map(run_job_multi, jobs)
                    results.extend(group_results)

        # --------------------------------------------------------------------
        # Cleanup
        # --------------------------------------------------------------------

        if parallel:
            thread_local.__dict__.clear()
        self.solveroutine.reset()

        logger.info("Solving complete")

        # --------------------------------------------------------------------
        # Writing solve reports
        # --------------------------------------------------------------------

        for freq, job in zip(self.frequencies, results):
            self.data.setreport(job.reports, freq=freq, **self._params)

        for variables, data in self.data.sim.iterate():
            logger.trace(f"Sim variable: {variables}")
            for item in data["report"]:
                item.logprint(logger.trace)

        # --------------------------------------------------------------------
        # Post Processing
        # --------------------------------------------------------------------

        self._post_process_scatter(results, material_set)
        self._completed = True
        return self.data

    def _run_adaptive_mesh(
        self, iteration: int, frequency: float, automatic_modal_analysis: bool = True
    ) -> tuple[MWData, list[int]]:
        """Executes a frequency domain study

        The study is distributed over "n_workers" workers.
        As optional parameter you may set a harddisc_threshold as integer. This determines the maximum
        number of degrees of freedom before which the jobs will be cahced to the harddisk. The
        path that will be used to cache the sparse matrices can be specified.
        Additionally the term frequency_groups may be specified. This number will define in how
        many groups the matrices will be pre-computed before they are send to workers. This can minimize
        the total amound of RAM memory used. For example with 11 frequencies in gruops of 4, the following
        frequency indices will be precomputed and then solved: [[1,2,3,4],[5,6,7,8],[9,10,11]]

        Args:
            iteration (int): The iteration number
            frequency (float): The simulation frequency

        Raises:
            SimulationError: An error associated witha a problem during the simulation.

        Returns:
            MWSimData: The dataset.
        """

        self._simstart = time.time()

        if self.bc._initialized_with_defaults is False:
            raise SimulationError(
                "Cannot run a modal analysis because no default boundary conditions have been assigned."
            )

        self._check_meshed()
        self._initialize_field()
        self._initialize_bc_data()
        self._check_physics()

        if self.basis is None:
            raise SimulationError(
                "Cannot proceed, the simulation basis class is undefined."
            )

        if self._settings.check_ram:
            _check_ram(self.mesh.n_tets, 1, False)

        materials = self._get_material_assignment(self.mesher.volumes)
        logger.debug("Initializing single frequency settings.")

        #### Port settings
        all_ports = self.bc.oftype(PortBC)

        ##### FOR PORT SWEEP SET ALL ACTIVE TO FALSE. THIS SHOULD BE FIXED LATER
        ### COMPUTE WHICH TETS ARE CONNECTED TO PORT INDICES

        for port in all_ports:
            port.active = False

        def run_job_single(job: SimJob):
            A, bmat, ids, aux = job.get_Ab()
            solution, report = self.solveroutine.solve(
                A, bmat, ids, matrix_type=job.mtype, id=job.id
            )
            report.add(**aux)
            job.submit_solution(solution, report)
            return job

        # --------------------------------------------------------------------
        # Compute the Modal port Modes
        # --------------------------------------------------------------------

        self._compute_modes(frequency)

        logger.debug(f"Simulation frequency = {frequency / 1e9:.3f} GHz")

        job, mats = self.assembler.assemble_freq_matrix(
            self.basis,
            materials,
            self.bc.boundary_conditions,
            frequency,
            cache_matrices=self.cache_matrices,
        )

        job.id = 0

        logger.info("Starting solve")
        job = run_job_single(job)

        logger.info("Solving complete")

        self.data.setreport(job.reports, freq=frequency, **self._params)

        for variables, data in self.data.sim.iterate():
            logger.trace(f"Sim variable: {variables}")
            for item in data["report"]:
                item.logprint(logger.trace)

        self.solveroutine.reset()
        ### Compute S-parameters and return
        self._post_process([job], [mats])
        return self.data, job._pec_tris

    def eigenmode(
        self,
        search_frequency: float,
        nmodes: int = 6,
        k0_limit: float = 1,
        direct: bool = False,
        deep_search: bool = False,
        mode: Literal["LM", "LR", "SR", "LI", "SI"] = "LM",
    ) -> MWData:
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
        

        self._check_meshed()
        self._initialize_field()
        self._initialize_bc_data()

        if self.basis is None:
            raise SimulationError(
                "Cannot proceed. The simulation basis class is undefined."
            )

        materials = self._get_material_assignment(self.mesher.volumes)

        ### Does this move
        logger.debug("Initializing frequency domain sweep.")

        logger.info(
            f"Pre-assembling matrices of {len(self.frequencies)} frequency points."
        )

        job, matset = self.assembler.assemble_eig_matrix(
            self.basis, materials, self.bc.boundary_conditions, search_frequency
        )

        er, ur, cond = matset
        logger.info("Solving complete")

        A, C, solve_ids = job.get_AC()

        target_k0 = 2 * np.pi * search_frequency / 299792458

        eigen_values, eigen_modes, report = self.solveroutine.eig(
            A, C, solve_ids, nmodes, direct, target_k0, which=mode
        )

        eigen_modes = job.unpermute_solutions(eigen_modes)

        logger.debug(f"Eigenvalues: {np.sqrt(eigen_values)} rad/m")

        nmodes_found = eigen_values.shape[0]

        for i in range(nmodes_found):
            eig_k0 = np.sqrt(eigen_values[i])
            if eig_k0 < k0_limit:
                logger.debug(f"Ignoring mode due to low k0: {eig_k0} < {k0_limit}")
                continue
            eig_freq = eig_k0 * 299792458 / (2 * np.pi)
            Q = 0.5 * eig_freq.real / eig_freq.imag
            logger.debug(
                f"Found k0={eig_k0:.2f}, f0={eig_freq / 1e9:.2f} GHz (Q = {Q:.1f})"
            )
            Emode = eigen_modes[:, i]

            scalardata = self.data.scalar.new(**self._params)
            scalardata.k0 = eig_k0
            scalardata.Q = Q
            scalardata.freq = eig_freq

            fielddata = self.data.field.new(**self._params)
            fielddata.freq = eig_freq
            fielddata.Q = Q
            fielddata._der = np.squeeze(er[0, 0, :])
            fielddata._dur = np.squeeze(ur[0, 0, :])
            fielddata._dsig = np.squeeze(cond[0, 0, :])
            fielddata._mode_field = Emode
            fielddata.basis = self.basis
        ### Compute S-parameters and return

        return self.data

    def _check_port_powers(self) -> dict[int | float, float]:
        """Performs a port mode power check.

        If all is correct, the port powers should add up to 1W.
        If it doesn't we need to correct for it during S-parameter computation.

        Returns:
            dict[int | float, float]: Dict mapping S-matrix indices to poewr normalization constants.
        """
        freq = self.frequencies[0]
        k0 = 2 * np.pi * freq / C0

        p_constants = dict()
        for active_port, smat_index, mode_nr in self.bc.iter_port_modes():
            if isinstance(active_port, (LumpedPort,UserDefinedPort) ):
                p_constants[smat_index] = 1.0
                continue
            

            inward_normal = self.mesh.inward_normal(active_port.tags)

            def portfE(x, y, z):
                E = active_port.port_mode_3d_global(
                    x, y, z, k0=k0, which="E", mode_nr=mode_nr
                )
                return E[0, :], E[1, :], E[2, :]

            def portfH(x, y, z):
                H = active_port.port_mode_3d_global(
                    x, y, z, k0=k0, which="H", mode_nr=mode_nr
                )
                return H[0, :], H[1, :], H[2, :]

            tris = self.mesh.get_triangles(active_port.tags)
            power = compute_port_power_flux(
                self.mesh.nodes, self.mesh.tris[:, tris], portfE, portfH, inward_normal
            )

            p_constants[smat_index] = 1 / (abs(power)) ** 0.5
            logger.debug(f"Port {active_port} power = {power:.4f} [W]")
        return p_constants

    def _post_process(
        self,
        results: list[SimJob],
        materials: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ):
        """Compute the S-parameters after Frequency sweep

        Args:
            results (list[SimJob]): The set of simulation results
            er (np.ndarray): The domain εᵣ
            ur (np.ndarray): The domain μᵣ
            cond (np.ndarray): The domain conductivity
        """

        if self.basis is None:
            raise SimulationError(
                "Cannot post-process. Simulation basis function is undefined."
            )

        port_mode_powers = self._check_port_powers()

        mesh = self.mesh
        matrix_indices = [smati for port, smati, modenr in self.bc.iter_port_modes()]

        logger.info("Computing S-parameters")

        not_conserved = False
        conserve_margin = 0.0
        single_corr = self._settings.mw_cap_sp_single
        col_corr = self._settings.mw_cap_sp_col
        recip_corr = self._settings.mw_recip_sp

        ertri = np.zeros((3, 3, self.mesh.n_tris), dtype=np.complex128)
        urtri = np.zeros((3, 3, self.mesh.n_tris), dtype=np.complex128)
        condtri = np.zeros((self.mesh.n_tris,), dtype=np.complex128)

        for job, mats in zip(results, materials):
            job.load_solutions()
            freq = job.freq
            er, ur, cond = mats

            er_scal = (er[0, 0, :] + er[1, 1, :] + er[2, 2, :]) / 3
            ur_scal = (ur[0, 0, :] + ur[1, 1, :] + ur[2, 2, :]) / 3
            cond_scal = (cond[0, 0, :] + cond[1, 1, :] + cond[2, 2, :]) / 3

            ertri[:, :, :] = er[:, :, self.mesh.tri_to_tet[0, :]]
            urtri[:, :, :] = ur[:, :, self.mesh.tri_to_tet[0, :]]
            condtri[:] = cond[0, 0, self.mesh.tri_to_tet[0, :]]

            k0 = 2 * np.pi * freq / 299792458

            scalardata = self.data.scalar.new(freq=freq, **self._params)
            scalardata.k0 = k0
            scalardata.freq = freq
            scalardata.init_sp(matrix_indices)  # type: ignore

            fielddata = self.data.field.new(freq=freq, **self._params)
            fielddata.freq = freq
            fielddata._der = np.squeeze(er_scal)
            fielddata._dur = np.squeeze(ur_scal)
            fielddata._dsig = np.squeeze(cond_scal)

            logger.info(f"Post Processing simulation frequency = {freq / 1e9:.3f} GHz")

            # Recording port information
            for active_port, smat_index_j, mode_nr_j in self.bc.iter_port_modes():
                if not active_port.driven:
                    continue

                port_tets = self.mesh.get_face_tets(active_port.tags)

                fielddata.add_port_properties(
                    active_port.port_number,
                    mode_number=mode_nr_j,
                    smat_index=smat_index_j,
                    k0=k0,
                    beta=active_port.get_beta(k0),
                    Z0=active_port.portZ0(k0),
                    Pout=active_port.power,
                )

                scalardata.add_port_properties(
                    active_port.port_number,
                    mode_number=mode_nr_j,
                    smat_index=smat_index_j,
                    k0=k0,
                    beta=active_port.get_beta(k0),
                    Z0=active_port.portZ0(k0),
                    Pout=active_port.power,
                )

                solution = job._solutions_dict[smat_index_j]

                fielddata._fields = job._solutions_dict
                fielddata.basis = self.basis

                # Passive ports
                for passive_port, smat_index_i, mode_nr_i in self.bc.iter_port_modes():
                    if smat_index_i == smat_index_j:
                        active = True
                        func = "(F-E)"
                    else:
                        active = False
                        func = "F"

                    port_tets = self.mesh.get_face_tets(passive_port.tags)
                    fieldf = self.basis.interpolate_Ef(solution, tetids=port_tets)
                    tris = mesh.get_triangles(passive_port.tags)
                    tri_vertices = mesh.tris[:, tris]
                    EdotF_pas, EdotE_pas = self._compute_s_data(
                        passive_port,
                        mode_nr_i,
                        active,
                        fieldf,
                        tri_vertices,
                        k0,
                        ertri[:, :, tris],
                        urtri[:, :, tris],
                    )
                    bi = EdotF_pas / (EdotE_pas)
                    logger.debug(
                        f"[{smat_index_i},{smat_index_j}] ∬ ({func} · E*) ds = {np.abs(EdotF_pas):.3f}, ∬ (E · E*) ds = {np.abs(EdotE_pas):.3f}, (ampl={np.abs(bi):.3f})"
                    )

                    Sij = bi * np.abs(port_mode_powers[smat_index_j])
                    scalardata.write_S(smat_index_i, smat_index_j, Sij)
                    if abs(Sij) > 1.0:
                        logger.debug(
                            f"S-parameter ({smat_index_i},{smat_index_j}) > 1.0 detected: {np.abs(Sij)}"
                        )
                        not_conserved = True
                        conserve_margin = abs(Sij) - 1.0
                active_port.active = False

            fielddata.set_field_vector()

            N = scalardata.Sp.shape[1]

            # Enforce reciprocity
            if recip_corr:
                scalardata.Sp = (scalardata.Sp + scalardata.Sp.T) / 2

            # Enforce energy conservation
            if col_corr:
                for j in range(N):
                    scalardata.Sp[:, j] = scalardata.Sp[:, j] / max(
                        1.0, np.sum(np.abs(scalardata.Sp[:, j]) ** 2)
                    )

            # Enforce S-parameter limit to 1.0
            if single_corr:
                for i, j in product(range(N), range(N)):
                    scalardata.Sp[i, j] = scalardata.Sp[i, j] / max(
                        1.0, np.abs(scalardata.Sp[i, j])
                    )

            job.clear_solutions()

        if not_conserved and conserve_margin > 0.01:
            _GlobalHandler.active().debugcollector.add_report(
                f"S-parameters with an amplitude greater than 1.0 detected. ({20 * np.log10(conserve_margin):.2f}dB error. This could be due to a ModalPort with the wrong mode type.\n"
                + "Specify the type of mode (TE/TM/TEM) in the constructor using ModalPort(..., modetype='TE') for example."
            )

        logger.info("Simulation Complete!")
        self._simend = time.time()
        logger.info(f"Elapsed time = {(self._simend - self._simstart):.2f} seconds.")
        self._state.set_modified()

    def _post_process_scatter(
        self,
        results: list[SimJob],
        materials: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ):
        """Compute the S-parameters after Frequency sweep

        Args:
            results (list[SimJob]): The set of simulation results
            er (np.ndarray): The domain εᵣ
            ur (np.ndarray): The domain μᵣ
            cond (np.ndarray): The domain conductivity
        """

        if self.basis is None:
            raise SimulationError(
                "Cannot post-process. Simulation basis function is undefined."
            )
        logger.info("Post processing Scattered Field solutions")

        ertri = np.zeros((3, 3, self.mesh.n_tris), dtype=np.complex128)
        urtri = np.zeros((3, 3, self.mesh.n_tris), dtype=np.complex128)
        condtri = np.zeros((self.mesh.n_tris,), dtype=np.complex128)

        scatter_bc: ScatteredField = self.bc.oftype(ScatteredField)[0]

        for job, mats in zip(results, materials):
            job.load_solutions()
            freq = job.freq
            er, ur, cond = mats

            er_scal = (er[0, 0, :] + er[1, 1, :] + er[2, 2, :]) / 3
            ur_scal = (ur[0, 0, :] + ur[1, 1, :] + ur[2, 2, :]) / 3
            cond_scal = (cond[0, 0, :] + cond[1, 1, :] + cond[2, 2, :]) / 3

            ertri[:, :, :] = er[:, :, self.mesh.tri_to_tet[0, :]]
            urtri[:, :, :] = ur[:, :, self.mesh.tri_to_tet[0, :]]
            condtri[:] = cond[0, 0, self.mesh.tri_to_tet[0, :]]

            k0 = 2 * np.pi * freq / 299792458

            scalardata = self.data.scalar.new(freq=freq, **self._params)
            scalardata.k0 = k0
            scalardata.freq = freq

            fielddata = self.data.field.new(freq=freq, **self._params)
            fielddata.freq = freq
            fielddata._der = np.squeeze(er_scal)
            fielddata._dur = np.squeeze(ur_scal)
            fielddata._dsig = np.squeeze(cond_scal)
            fielddata._fields = job._solutions_dict
            fielddata.basis = self.basis

            logger.info(f"Post Processing simulation frequency = {freq / 1e9:.3f} GHz")

            # Recording port information
            i = 1
            for bf in scatter_bc._iter_fields(k0):
                fielddata.add_field_properties(bf)
                i += 1

            fielddata.set_field_vector()
            job.clear_solutions()

        logger.info("Simulation Complete!")
        self._simend = time.time()
        logger.info(f"Elapsed time = {(self._simend - self._simstart):.2f} seconds.")
        self._state.set_modified()

    def _compute_s_data(
        self,
        bc: PortBC,
        mode_nr: int,
        active: bool,
        fieldfunction: Callable,
        tri_vertices: np.ndarray,
        k0: float,
        erp: np.ndarray,
        urp: np.ndarray,
    ) -> tuple[complex, complex]:
        """Computes the S-parameter data for a given boundary condition and field function.

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
            if bc.voltage_integration_line is None:
                raise SimulationError(
                    "Trying to compute characteristic impedance but no integration line is defined."
                )
            if bc.Z0 is None:
                raise SimulationError(
                    "Trying to compute the impedance of a boundary condition with no characteristic impedance."
                )

            # Compute all integration points
            xs = np.concat([line.xs for line in bc.voltage_integration_line])
            ys = np.concat([line.ys for line in bc.voltage_integration_line])
            zs = np.concat([line.zs for line in bc.voltage_integration_line])

            # Evaluate the E-field
            Ex, Ey, Ez = fieldfunction(xs, ys, zs)

            ctr = 0
            V = 0
            Nlines = len(bc.voltage_integration_line)
            for line in bc.voltage_integration_line:
                npts = line.xs.shape[0]
                slc = slice(ctr, ctr + npts)
                ctr += npts
                V += line.line_integral_precalc(Ex[slc], Ey[slc], Ez[slc]) / Nlines

            if active:
                if bc.voltage is None:
                    raise ValueError(
                        "Cannot compute port S-paramer with a None port voltage."
                    )
                a = bc.voltage
                b = V - bc.voltage
            else:
                a = bc.voltage
                b = V

            a_sig = a * csqrt(1 / (2 * bc.Z0))
            b_sig = b * csqrt(1 / (2 * bc.Z0))

            return b_sig, a_sig
        else:
            const = np.ones_like(erp[0, 0, :].squeeze(), dtype=np.complex128)

            field_p = sparam_field_power(
                self.mesh.nodes,
                tri_vertices,
                bc,
                mode_nr,
                active,
                k0,
                fieldfunction,
                const,
                5,
            )
            mode_p = sparam_mode_power(
                self.mesh.nodes, tri_vertices, bc, mode_nr, k0, const, 7
            )

            return field_p, mode_p

    ############################################################
    #                     DEPRICATED FUNCTIONS                #
    ############################################################

    def frequency_domain(self, *args, **kwargs):
        """DEPRICATED VERSION: Use run_sweep() instead."""
        logger.warning("This function is depricated. Please use run_sweep() instead")
        return self.run_sweep(*args, **kwargs)

