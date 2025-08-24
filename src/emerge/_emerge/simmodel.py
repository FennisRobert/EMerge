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

from __future__ import annotations
from .mesher import Mesher, unpack_lists
from .geometry import GeoObject, _GEOMANAGER
from .geo.modeler import Modeler
from .physics.microwave.microwave_3d import Microwave3D
from .mesh3d import Mesh3D
from .selection import Selector, FaceSelection, Selection
from .logsettings import LOG_CONTROLLER
from .plot.pyvista import PVDisplay
from .dataset import SimulationDataset
from .periodic import PeriodicCell
from .bc import BoundaryCondition
from typing import Literal, Generator, Any
from loguru import logger
import numpy as np
import gmsh # type: ignore
import joblib # type: ignore
import os
import inspect
from pathlib import Path
from atexit import register
import signal
from .. import __version__

############################################################
#                   EXCEPTION DEFINITIONS                  #
############################################################

_GMSH_ERROR_TEXT = """
--------------------------
Known problems/solutions:
(1) - PLC Error:  A segment and a facet intersect at point
    This can be caused when approximating thin curved volumes. Try to decrease the mesh size for that region.
--------------------------
"""

class SimulationError(Exception):
    pass

class VersionError(Exception):
    pass

############################################################
#                 BASE 3D SIMULATION MODEL                 #
############################################################

class Simulation:

    def __init__(self, 
                 modelname: str, 
                 loglevel: Literal['TRACE','DEBUG','INFO','WARNING','ERROR'] = 'INFO',
                 load_file: bool = False,
                 save_file: bool = False,
                 logfile: bool = False,
                 path_suffix: str = ".EMResults"):
        """Generate a Simulation class object.

        As a minimum a file name should be provided. Additionally you may provide it with any
        class that inherits from BaseDisplay. This will then be used for geometry displaying.

        Args:
            modelname (str): The name of the simulation model. This will be used for filenames and path names when saving data.
            loglevel ("DEBUG","INFO","WARNING","ERROR", optional): The loglevel to use for loguru. Defaults to 'INFO'.
            load_file (bool, optional): If the simulatio model should be loaded from a file. Defaults to False.
            save_file (bool, optional): if the simulation file should be stored to a file. Defaults to False.
            logfile (bool, optional): If a file should be created that contains the entire log of the simulation. Defaults to False.
            path_suffix (str, optional): The suffix that will be added to the results directory. Defaults to ".EMResults".
        """

        caller_file = Path(inspect.stack()[1].filename).resolve()
        base_path = caller_file.parent

        self.modelname = modelname
        self.modelpath = base_path / (modelname.lower()+path_suffix)
        self.mesher: Mesher = Mesher()
        self.modeler: Modeler = Modeler()
        
        self.mesh: Mesh3D = Mesh3D(self.mesher)
        self.select: Selector = Selector()

        ## STATES
        self.__active: bool = False
        self._defined_geometries: bool = False
        self._cell: PeriodicCell | None = None

        self.display: PVDisplay = PVDisplay(self.mesh)

        self.save_file: bool = save_file
        self.load_file: bool = load_file

        self.data: SimulationDataset = SimulationDataset()

        ## Physics
        self.mw: Microwave3D = Microwave3D(self.mesher, self.data.mw)

        self._initialize_simulation()

        self.set_loglevel(loglevel)
        if logfile:
            self.set_logfile()

        self._update_data()
    

    ############################################################
    #                       PRIVATE FUNCTIONS                  #
    ############################################################

    def __setitem__(self, name: str, value: Any) -> None:
        """Store data in the current data container"""
        self.data.sim[name] = value

    def __getitem__(self, name: str) -> Any:
        """Get the data from the current data container"""
        return self.data.sim[name]
    
    def __enter__(self) -> Simulation:
        """This method is depricated with the new atexit system. It still exists for backwards compatibility.

        Returns:
            Simulation: the Simulation object
        """
        return self

    def __exit__(self, type, value, tb):
        """This method no longer does something. It only serves as backwards compatibility."""
        self._exit_gmsh()
        return False
    
    def _install_signal_handlers(self):
        # on SIGINT (Ctrl-C) or SIGTERM, call our exit routine
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """
        Signal handler: do our cleanup, then re-raise
        the default handler so that exit code / traceback
        is as expected.
        """
        try:
            # run your atexit-style cleanup
            self._exit_gmsh()
        except Exception:
            # log but don’t block shutdown
            logger.exception("Error during signal cleanup")
        finally:
            # restore default handler and re‐send the signal
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    def _initialize_simulation(self):
        """Initializes the Simulation data and GMSH API with proper shutdown routines.
        """
        _GEOMANAGER.sign_in(self.modelname)
        
        # If GMSH is not yet initialized (Two simulation in a file)
        if gmsh.isInitialized() == 0:
            logger.debug('Initializing GMSH')
            gmsh.initialize()
            
            # Set an exit handler for Ctrl+C cases
            self._install_signal_handlers()

            # Restier the Exit GMSH function on proper program abortion
            register(self._exit_gmsh)

        # Create a new GMSH model or load it
        if not self.load_file:
            gmsh.model.add(self.modelname)
            self.data: SimulationDataset = SimulationDataset()
        else:
            self.load()

        # Set the Simulation state to active
        self.__active = True
        return self

    def _exit_gmsh(self):
        # If the simulation object state is still active (GMSH is running)
        if not self.__active:
            return
        logger.debug('Exiting program')
        # Save the file first
        if self.save_file:
            self.save()
        # Finalize GMSH
        if gmsh.isInitialized():
            gmsh.finalize()

        logger.debug('GMSH Shut down successful')
        # set the state to active
        self.__active = False

    def _update_data(self) -> None:
        """Writes the stored physics data to each phyics class insatnce"""
        self.mw.data = self.data.mw

    def all_geometries(self) -> list[GeoObject]:
        """Returns all geometries stored in the simulation file."""
        return [obj for obj in self.data.sim.default.values() if isinstance(obj, GeoObject)]
    
    def all_bcs(self) -> list[BoundaryCondition]:
        """Returns all boundary condition objects stored in the simulation file"""
        return [obj for obj in self.data.sim.default.values() if isinstance(obj, BoundaryCondition)]
    
    def _set_mesh(self, mesh: Mesh3D) -> None:
        """Set the current model mesh to a given mesh."""
        self.mesh = mesh
        self.mw.mesh = mesh
        self.display._mesh = mesh
    
    ############################################################
    #                       PUBLIC FUNCTIONS                  #
    ############################################################

    def check_version(self, version: str) -> None:
        """Compares the provided version number with the version number of EMerge that is running the script.
        
        You may remove any call to check_version to suppress VersionErrors and warnings.
        
        Args:
            version (str): The EMerge version you intend to write this code for.

        Raises:
            VersionError: A potential version error if incompatibility is possible
        """
        vM, vm, vp = [float(x) for x in version.split('.')]
        cM, cm, cp = [float(x) for x in __version__.split('.')]
        if vM != cM:
            raise VersionError(f"You are running a script designed for version {version} with a possibly incompatible version of EMerge {__version__}")
        if vm != cm:
            raise VersionError(f"You are running a script designed for version {version} with a possibly incompatible version of EMerge {__version__}")
        if vp != cp:
            logger.warning(f"You are running a script designed for version {version} with a possibly incompatible version of EMerge {__version__}")
            logger.warning("You may suppress this error by removing the call to .check_version().")
            input('Press enter to proceed...')

    def save(self) -> None:
        """Saves the current model in the provided project directory."""
        # Ensure directory exists
        if not self.modelpath.exists():
            self.modelpath.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {self.modelpath}")

        # Save mesh
        mesh_path = self.modelpath / 'mesh.msh'
        brep_path = self.modelpath / 'model.brep'

        gmsh.option.setNumber('Mesh.SaveParametric', 1)
        gmsh.option.setNumber('Mesh.SaveAll', 1)
        gmsh.model.geo.synchronize()
        gmsh.model.occ.synchronize()

        gmsh.write(str(mesh_path))
        gmsh.write(str(brep_path))
        logger.info(f"Saved mesh to: {mesh_path}")

        # Pack and save data
        dataset = dict(simdata=self.data, mesh=self.mesh)
        data_path = self.modelpath / 'simdata.emerge'
        joblib.dump(dataset, str(data_path))
        logger.info(f"Saved simulation data to: {data_path}")

    def load(self) -> None:
        """Loads the model from the project directory."""
        mesh_path = self.modelpath / 'mesh.msh'
        brep_path = self.modelpath / 'model.brep'
        data_path = self.modelpath / 'simdata.emerge'

        if not mesh_path.exists() or not data_path.exists():
            raise FileNotFoundError("Missing required mesh or data file.")

        # Load mesh
        gmsh.open(str(brep_path))
        gmsh.merge(str(mesh_path))
        gmsh.model.geo.synchronize()
        gmsh.model.occ.synchronize()
        logger.info(f"Loaded mesh from: {mesh_path}")
        #self.mesh.update([])

        # Load data
        datapack = joblib.load(str(data_path))
        self.data = datapack['simdata']
        self._set_mesh(datapack['mesh'])
        logger.info(f"Loaded simulation data from: {data_path}")
    
    def set_loglevel(self, loglevel: Literal['DEBUG','INFO','WARNING','ERROR']) -> None:
        """Set the loglevel for loguru.

        Args:
            loglevel ('DEBUG','INFO','WARNING','ERROR'): The loglevel
        """
        LOG_CONTROLLER.set_std_loglevel(loglevel)
        if loglevel not in ('TRACE','DEBUG'):
            gmsh.option.setNumber("General.Terminal", 0)

    def set_logfile(self) -> None:
        """Adds a file output for the logger."""
        LOG_CONTROLLER.set_write_file(self.modelpath)
        
    def view(self, 
             selections: list[Selection] | None = None, 
             use_gmsh: bool = False,
             plot_mesh: bool = False,
             volume_mesh: bool = True,
             opacity: float | None = None) -> None:
        """View the current geometry in either the BaseDisplay object (PVDisplay only) or
        the GMSH viewer.

        Args:
            selections (list[Selection] | None, optional): Optional selections to highlight. Defaults to None.
            use_gmsh (bool, optional): If GMSH's GUI should be used. Defaults to False.
            plot_mesh (bool, optional): If the mesh should be plot instead of the object. Defaults to False.
            volume_mesh (bool, optional): If the internal mesh should be plot instead of only the surface boundary mesh. Defaults to True
            opacity (float | None, optional): The object/mesh opacity. Defaults to None.

        """
        if not (self.display is not None and self.mesh.defined) or use_gmsh:
            gmsh.model.occ.synchronize()
            gmsh.fltk.run()
            return
        for geo in _GEOMANAGER.all_geometries():
            self.display.add_object(geo, mesh=plot_mesh, opacity=opacity, volume_mesh=volume_mesh)
        if selections:
            [self.display.add_object(sel, color='red', opacity=0.3) for sel in selections]
        self.display.show()

        return None
        
    def set_periodic_cell(self, cell: PeriodicCell, excluded_faces: list[FaceSelection] | None = None):
        """Set the given periodic cell object as the simulations peridicity.

        Args:
            cell (PeriodicCell): The PeriodicCell class
            excluded_faces (list[FaceSelection], optional): Faces to exclude from the periodic boundary condition. Defaults to None.
        """
        self.mw.bc._cell = cell
        self._cell = cell

    def commit_geometry(self, *geometries: GeoObject | list[GeoObject]) -> None:
        """Finalizes and locks the current geometry state of the simulation.

        The geometries may be provided (legacy behavior) but are automatically managed in the background.
        
        """
        geometries_parsed: Any = None
        if not geometries:
            geometries_parsed = _GEOMANAGER.all_geometries()
        else:
            geometries_parsed = unpack_lists(geometries + tuple([item for item in self.data.sim.default.values() if isinstance(item, GeoObject)]))
        
        self.data.sim['geometries'] = geometries_parsed
        self.mesher.submit_objects(geometries_parsed)
        self._defined_geometries = True
        self.display._facetags = [dt[1] for dt in gmsh.model.get_entities(2)]
          
    def generate_mesh(self) -> None:
        """Generate the mesh. 
        This can only be done after commit_geometry(...) is called and if frequencies are defined.

        Args:
            name (str, optional): The mesh file name. Defaults to "meshname.msh".

        Raises:
            ValueError: ValueError if no frequencies are defined.
        """
        if not self._defined_geometries:
            self.commit_geometry()
        
        # Set the cell periodicity in GMSH
        if self._cell is not None:
            self.mesher.set_periodic_cell(self._cell)
            
        self.mw._initialize_bcs()

        # Check if frequencies are defined: TODO: Replace with a more generic check
        if self.mw.frequencies is None:
            raise ValueError('No frequencies defined for the simulation. Please set frequencies before generating the mesh.')

        gmsh.model.occ.synchronize()

        # Set the mesh size
        self.mesher.set_mesh_size(self.mw.get_discretizer(), self.mw.resolution)
        
        try:
            gmsh.logger.start()
            gmsh.model.mesh.generate(3)
            logs = gmsh.logger.get()
            gmsh.logger.stop()
            for log in logs:
                logger.trace('[GMSH] '+log)
        except Exception:
            logger.error('GMSH Mesh error detected.')
            print(_GMSH_ERROR_TEXT)
            raise
        self.mesh.update(self.mesher._get_periodic_bcs())
        self.mesh.exterior_face_tags = self.mesher.domain_boundary_face_tags
        gmsh.model.occ.synchronize()
        self._set_mesh(self.mesh)

    def parameter_sweep(self, clear_mesh: bool = True, **parameters: np.ndarray) -> Generator[tuple[float,...], None, None]:
        """Executes a parameteric sweep iteration.

        The first argument clear_mesh determines if the mesh should be cleared and rebuild in between sweeps. This is usually needed
        except when you change only port-properties or material properties. The parameters of the sweep can be provided as a set of 
        keyword arguments. As an example if you defined the axes: width=np.linspace(...) and height=np.linspace(...). You can
        add them as arguments using .parameter_sweep(True, width=width, height=height).

        The rest of the simulation commands should be inside the iteration scope

        Args:
            clear_mesh (bool, optional): Wether to clear the mesh in between sweeps. Defaults to True.

        Yields:
            Generator[tuple[float,...], None, None]: The parameters provided

        Example:
         >>> for W, H in model.parameter_sweep(True, width=widths, height=heights):
         >>>    // build simulation
         >>>    data = model.run_sweep()
         >>> // Extract the data
         >>> widths, heights, frequencies, S21 = data.ax('width','height','freq').S(2,1)
        """
        paramlist = sorted(list(parameters.keys()))
        dims = np.meshgrid(*[parameters[key] for key in paramlist], indexing='ij')
        dims_flat = [dim.flatten() for dim in dims]
        self.mw.cache_matrices = False
        for i_iter in range(dims_flat[0].shape[0]):
            if clear_mesh:
                logger.info('Cleaning up mesh.')
                gmsh.clear()
                mesh = Mesh3D(self.mesher)
                _GEOMANAGER.reset(self.modelname)
                self._set_mesh(mesh)
                self.mw.reset()
            
            params = {key: dim[i_iter] for key,dim in zip(paramlist, dims_flat)}
            self.mw._params = params
            self.data.sim.new(**params)

            logger.info(f'Iterating: {params}')
            if len(dims_flat)==1:
                yield (dims_flat[0][i_iter],)
            else:
                yield (dim[i_iter] for dim in dims_flat) # type: ignore
        self.mw.cache_matrices = True
    
    def export(self, filename: str):
        """Exports the model or mesh depending on the extension. 
        
        Exporting is realized by GMSH.
        Supported file formats are:
        
        3D Model: .opt, .geo_unrolled, .brep, .xao ,.step and .iges
        Mesh: .msh, .inp, .key, ._0000.rad, .celum, .cgns, .diff, .unv, .ir3, .mes, .mesh 
              .mail, .m, .bdf, .off, .p3d, .stl, .wrl, .vtk, .dat, .ply2, .su2, .neu, .x3d

        Args:
            filename (str): The filename
        """
        gmsh.write(filename)
    ############################################################
    #                     DEPRICATED FUNCTIONS                #
    ############################################################

    def define_geometry(self, *args):
        """DEPRICATED VERSION: Use .commit_geometry()
        """
        logger.warning('define_geometry() will be derpicated. Use commit_geometry() instead.')
        self.commit_geometry(*args)
        