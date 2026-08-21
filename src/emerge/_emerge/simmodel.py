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
from __future__ import annotations

import sys
from ._global import _GlobalHandler
from .geometry import GeoObject
from .geo.modeler import Modeler
from .physics.microwave.microwave_3d import Microwave3D
from .physics.heatconduction.heatconduction_3d import HeatConduction3D
from .mesh3d import Mesh3D
from .mesher import Mesher
from .dataset import SimulationDataset
from .plot.pyvista import PVDisplay
from .periodic import PeriodicCell
from .cacherun import get_build_section, get_run_section
from .settings import DEFAULT_SETTINGS, Settings
from .solver import EMSolver, Solver
from .simstate import SimState
from .selection import Selector, Selection, FaceSelection
from .optim import Optimizer
from typing import Literal, Generator, Any
from loguru import logger
import numpy as np
import gmsh  # type: ignore
import os
import inspect
from pathlib import Path
import joblib
from pickle import PicklingError
from atexit import register, _clear
import signal
from .. import __version__
from .file import save_object, load_object

############################################################
#                   EXCEPTION DEFINITIONS                  #
############################################################

_GMSH_ERROR_TEXT = """
--------------------------
Known problems/solutions:
(1) - PLC Error:  A segment and a facet intersect at point
    This can be caused when approximating thin curved volumes. Possible solutions:
    1. Increase the curved boundary resolution:
        model.mesher.set_curved_boundary_meshing(20)
         (!) Pick a higher number than 10 like 20, 30, 40 etc.
         
    2. Decrease the mesh size of the object:
        object.max_meshsize = ...
--------------------------
"""


class SimulationError(Exception):
    pass


class VersionError(Exception):
    pass


def install_refined_mesh(volume_dim, volume_tag, node_ids_global, steiner, tetra_out,
                          old_element_tags):
    """
    node_ids_global : (n_orig,) global gmsh node tags corresponding to local_nodes
                       columns [0 .. n_orig-1] (i.e. the return value of tet_to_node's
                       node-index output, or your global_to_local inverse).
    steiner         : flat (3*n_new,) new point coordinates from refineTetrahedra.
    tetra_out       : flat, 1-based, 4-per-tet connectivity from refineTetrahedra
                       (confirmed convention -- subtract 1 before use).
    old_element_tags: global gmsh element tags of the 95 original flagged tets,
                       to be removed and replaced by the refined set.
    """
    steiner = np.asarray(steiner, dtype=np.float64)
    tetra_out = np.asarray(tetra_out, dtype=np.int64)
    n_new = steiner.size // 3
    n_orig = node_ids_global.shape[0]

    # 1. Register the new (Steiner) nodes on the volume with fresh global tags.
    if n_new:
        max_node_tag = gmsh.model.mesh.getMaxNodeTag()
        new_global_tags = np.arange(max_node_tag + 1, max_node_tag + 1 + n_new, dtype=np.int64)
        gmsh.model.mesh.addNodes(volume_dim, volume_tag, new_global_tags.tolist(), steiner.tolist())
    else:
        new_global_tags = np.empty(0, dtype=np.int64)

    # 2. Local index space is [node_ids_global ++ new_global_tags], confirmed
    #    1-based on both ends -- subtract 1, then look up global tags.
    combined_global = np.concatenate([node_ids_global, new_global_tags])
    tetra_out_0based = tetra_out - 1
    tet_node_tags = combined_global[tetra_out_0based]  # flat, global node tags

    # 3. Remove the original (now-superseded) tets before adding the refined set.
    gmsh.model.mesh.removeElements(volume_dim, volume_tag, old_element_tags.tolist())

    # 4. Add the refined tets with fresh element tags.
    n_tets = tet_node_tags.size // 4
    max_elem_tag = gmsh.model.mesh.getMaxElementTag()
    new_elem_tags = np.arange(max_elem_tag + 1, max_elem_tag + 1 + n_tets, dtype=np.int64)
    gmsh.model.mesh.addElementsByType(volume_tag, 4, new_elem_tags.tolist(), tet_node_tags.tolist())

    return new_global_tags, new_elem_tags
############################################################
#                 BASE 3D SIMULATION MODEL                 #
############################################################
def _get_caller_dir() -> Path:
    """
    Walk the call stack and return the directory of the first frame
    that is NOT inside the emerge package itself.
    """
    val = os.environ.get("EMERGE_BASE_PATH")
    if val is not None:
        return Path(val)
    emerge_src = Path(__file__).resolve().parent  # emerge's own src dir

    for frame_info in inspect.stack():
        caller = Path(frame_info.filename).resolve()
        # Skip frames inside emerge itself and stdlib/site-packages
        if (
            not str(caller).startswith(str(emerge_src))
            and caller.suffix == ".py"
            and caller.exists()
        ):
            return caller.parent

    # Fallback to cwd if nothing found
    return Path.cwd()


class Simulation:
    def __init__(
        self,
        modelname: str,
        loglevel: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
        load_file: bool = False,
        save_file: bool = False,
        write_log: bool = False,
        path_suffix: str = ".EMResults",
        store_system: Literal["msgpack", "joblib"] = "joblib",
    ):
        """Generate a Simulation class object.

        As a minimum a file name should be provided. Additionally you may provide it with any
        class that inherits from BaseDisplay. This will then be used for geometry displaying.

        Args:
            modelname (str): The name of the simulation model. This will be used for filenames and path names when saving data.
            loglevel ("DEBUG","INFO","WARNING","ERROR", optional): The loglevel to use for loguru. Defaults to 'INFO'.
            load_file (bool, optional): If the simulatio model should be loaded from a file. Defaults to False.
            save_file (bool, optional): if the simulation file should be stored to a file. Defaults to False.
            write_log (bool, optional): If a file should be created that contains the entire log of the simulation. Defaults to False.
            path_suffix (str, optional): The suffix that will be added to the results directory. Defaults to ".EMResults".
            store_system: ("msgpack","joblib", optional): The serialization module to use. Uses Joblib by default for legacy reasons. Msgpack is safe.
        """

        base_path = _get_caller_dir()

        self.base_path: Path = base_path
        self.modelname: str = modelname
        self.modelpath: Path = base_path / (modelname.lower() + path_suffix)
        self.mesher: Mesher = Mesher()
        self.modeler: Modeler = Modeler()

        self.state: SimState = SimState(self.modelname, self.modelpath)
        self.select: Selector = Selector()
        self.settings: Settings = DEFAULT_SETTINGS
        self.opt: Optimizer = Optimizer()
        self.opt.callback = self._optim_callback
        self.settings._save_method = store_system

        ## Display
        self.display: PVDisplay = PVDisplay(self.state)

        ## STATES
        self.__active: bool = False
        self._cell: PeriodicCell | None = None
        self.save_file: bool = save_file
        self.load_file: bool = load_file
        self._cache_run: bool = False
        self._file_lines: str = ""
        self._saved: bool = False

        ## Physics
        self.mw: Microwave3D = Microwave3D(self.state, self.mesher, self.settings)
        self.hc: HeatConduction3D = HeatConduction3D(
            self.state, self.mesher, self.settings
        )
        self._geo_scale_factor: float = 1.0

        self._mw_active: bool = True
        self._hc_active: bool = False

        self._initialize_simulation()

        self.set_loglevel(loglevel)

        if write_log:
            self.set_write_log()

        _GlobalHandler.active().logcontroller._flush_log_buffer()
        _GlobalHandler.active().logcontroller._sys_info()

        self.__post_init__()

    ############################################################
    #                       PRIVATE FUNCTIONS                  #
    ############################################################

    def _reset_mesh(self, reset_gmsh: bool = True) -> None:
        if reset_gmsh:
            gmsh.model.mesh.clear()
        self.mw.reset(_reset_bc=False)
        self.state.reset_mesh()

    def _reset_geometry(self) -> None:
        gmsh.clear()
        gmsh.model.mesh.clear()
        self.state.reset_geostate()
        self.mw.reset(_reset_bc=False)
        self.state.reset_mesh()

    def _reset_physics(self) -> None:
        self.mw.reset()

    def _reset_data(self) -> None:
        self.state.data.reset()

    @property
    def data(self) -> SimulationDataset:
        return self.state.data

    @property
    def mesh(self) -> Mesh3D:
        return self.state.mesh

    def __del__(self):
        self.clean()

    def __post_init__(self):
        pass

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
        self.clean()
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

    def _autosave(self):
        """Called by atexit as an emergency matter."""
        if not self.settings.auto_save:
            return

        if not self.settings.save_after_sim:
            if not self.mw._completed:
                return

        self._exit_gmsh()

    def _initialize_simulation(self):
        """Initializes the Simulation data and GMSH API with proper shutdown routines."""
        self.state.init()

        # If GMSH is not yet initialized (Two simulation in a file)
        if gmsh.isInitialized() == 0:
            logger.debug("Initializing GMSH")

            gmsh.initialize(readConfigFiles=False)
            # Set an exit handler for Ctrl+C cases
            self._install_signal_handlers()

            # Restier the Exit GMSH function on proper program abortion
            _clear()
            register(self._autosave)
            register(self._print_summary)
        else:
            gmsh.finalize()
            gmsh.initialize(readConfigFiles=False)

        # Create a new GMSH model or load it
        if not self.load_file:
            gmsh.model.add(self.modelname)
        else:
            self.load()

        # Set the Simulation state to active
        self.__active = True
        return self

    def _print_summary(self):
        if _GlobalHandler.active().debugcollector.any_warnings:
            logger.warning("EMerge simulation warnings:")
        for i, report in _GlobalHandler.active().debugcollector.all_reports():
            logger.warning(f"{i}: {report}")

    def _exit_gmsh(self):
        # If the simulation object state is still active (GMSH is running)
        if not self.__active:
            return

        logger.debug("Exiting program")

        self._print_summary()

        # Save the file first
        if self.save_file:
            self.save(_force_save=False)

        # Finalize GMSH
        if gmsh.isInitialized():
            gmsh.clear()
            gmsh.finalize()

        logger.debug("GMSH Shut down successful")
        # set the state to active
        self.__active = False

    def _optim_callback(self):
        self.mw.cache_matrices = False
        if self.opt.clear_mesh and self.opt.N > 0:
            logger.info("Cleaning up mesh.")
            self.reset(mesh=True, geometry=True, physics=True)
        self.state.set_parameters(self.opt.params)
        if not self.opt.clear_mesh:
            self.state.store_geometry_data()

    def _commit_quick_geometries(self) -> None:
        """Used to temporarily submit geometries"""
        logger.trace("Committing temporary geometry.")
        self.state.store_geometry_data()
        logger.trace(f"Parsed geometries = {self.state.current_geo_state}")
        self.mesher.submit_objects(self.state.current_geo_state)
        self.display._facetags = [dt[1] for dt in gmsh.model.get_entities(2)]

    def _apply_size_scaling(self) -> None:
        """Applies size scaling factor


        Returns:
            _type_: _description_

        Yields:
            _type_: _description_
        """
        if self._geo_scale_factor == 1.0:
            return
        logger.info(f'Applying global size scaling with factor: {self._geo_scale_factor}')
        # Apply scale factor
        
        s = 1/self._geo_scale_factor
        scale_down_matrix = [
            s,   0.0, 0.0, 0.0,
            0.0, s,   0.0, 0.0,
            0.0, 0.0, s,   0.0,
            0.0, 0.0, 0.0, 1.0
        ]
        # Apply to all nodes in the mesh
        gmsh.model.mesh.affineTransform(scale_down_matrix)

        for geo in self.all_geos():
            geo._apply_size_scaling(self._geo_scale_factor)
        
    ############################################################
    #                       PUBLIC FUNCTIONS                  #
    ############################################################

    def set_scale_factor(self, scale_factor: float) -> None:
        """Define the scaling factor at which modelling occurs relative to the actual size.
        The physical mesh will be descaled with 1/factor after meshing.

        For example, if you want to model some object 1000x larger than it really is, the scale factor becomes 1000
        
        This can be used when modelling tiny structures like bond wires which may trip up GMSH and OpenCASCADE

        Args:
            scale_factor (float): _description_
        """
        self._geo_scale_factor = scale_factor
        self.mesher._scale = scale_factor
        msg = """Applying a global scaling is an experimental feature that may break EMerge features.
        If you run into issues, please contact me through robertfennis@emerge-software.com so that I can resolve any issues."""
        logger.warning(msg)
        
    def ping(self, message: str, channel: str) -> None:
        """Send a message to a ntfy app channel. Messages are public. Pick a channel name that only you would guess."""
        import urllib.request
        import urllib.parse

        if channel.lower().strip() == "emerge":
            _GlobalHandler.active().debugcollector.add_report(
                "You are using the EMerge ping channel. All pings are public! Everybody can read them who is subscribed to the same channel. This is being broadcast. Please use private channels."
            )
        url = f"https://ntfy.sh/{channel}"
        data = f"{self.modelname}: {message}".encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req) as response:
            return response.read()

    def clean(self) -> None:
        """Cleans up the simulation object references by deleting all major components."""

        ## SIGN OFF SIMUALTION STATE
        if "state" in self.__dict__:
            if self.state is not None:
                self.state.sign_off()
            self.state = None

        ## Remove the State object
        if "display" in self.__dict__:
            if self.display:
                self.display.clean()
        self.mesher = None
        self.modeler = None
        self.mw = None
        self.display = None

    def reset(
        self,
        mesh: bool = False,
        geometry: bool = False,
        physics: bool = False,
        data: bool = False,
        all: bool = False,
    ):
        """Resets the simulation model states

        Args:
            mesh (bool, optional): Reset the Mesh. Defaults to False.
            geometry (bool, optional): Reset the CAD geometry. Defaults to False.
            physics (bool, optional): Reset the physics settings. Defaults to False.
            data (bool, optional): Reset the simulation data. Defaults to False.
            all (bool, optional): Reset everything. Defaults to False.

        """
        if all:
            self._reset_geometry()
            self._reset_physics()
            self._reset_data()
            return
        if mesh:
            self._reset_mesh()
        if physics:
            self._reset_physics()
        if geometry:
            self._reset_geometry()
        if data:
            self._reset_data()

    def cache_build(self) -> bool:
        """Checks if all the lines inside this if statement block are the same as those
        stored from a previous run. If so, then it returns false. Else it returns True.

        Can be used to capture an entire model simulation.

        Example:

        >>> if model.cache_build():
        >>>     box = em.geo.Box(...)
        >>>     # Other lines
        >>>     model.mw.run_sweep()
        >>> data = model.data.mw

        Returns:
            bool: If the code is not the same
        """

        self.save_file = True
        self._cache_run = True
        filestr = get_build_section()
        self._file_lines = filestr
        cachepath = self.modelpath / "pylines.txt"

        # If there is not pylines file, simulate (can't have been run).
        if not cachepath.exists():
            logger.info("No cached data detected, running file")
            return True

        with open(cachepath, "r") as file:
            lines = file.read()

        if lines == filestr:
            logger.info("Cached data detected! Loading data!")
            self.load()
            return False
        logger.info("Different cached data detected, rebuilding file.")
        return True

    def cache_run(self) -> bool:
        """Checks if all the lines before this call are the same as the lines
        stored from a previous run. If so, then it returns false. Else it returns True.

        Can be used to capture a run_sweep() call.

        Example:

        >>> if model.cache_run():
        >>>     model.mw.run_sweep()
        >>> data = model.data.mw

        Returns:
            bool: If the code is not the same
        """

        self._cache_run = True
        filestr = get_run_section()
        self._file_lines = filestr
        cachepath = self.modelpath / "pylines.txt"

        # If there is not pylines file, simulate (can't have been run).
        if not cachepath.exists():
            self.save_file = True
            logger.info("No cached data detected, running simulation!")
            return True

        with open(cachepath, "r") as file:
            lines = file.read()

        if lines == filestr:
            logger.info("Cached data detected! Loading data!")
            self.load()
            return False
        logger.info("Different cached data detected, rerunning simulation.")
        return True

    def check_version(self, target_version: str, *, log: bool = False) -> None:
        """
        Ensure the script targets an EMerge major.minor compatible with the current runtime.

        Only major.minor is enforced. Patch differences are ignored.

        Parameters
        ----------
        target_version : str
            The EMerge version this script was written for (e.g. "1.4.0" or "1.4").
        log : bool, optional
            If True and a `logger` is available, emit a single WARNING with the same
            message as the exception. Defaults to False.

        Raises
        ------
        VersionError
            If the script's target major.minor differs from the running EMerge major.minor.
        """
        try:
            from packaging.version import Version as _V

            def _mm(v: str):
                pv = _V(v)
                return (pv.major, pv.minor)

            script_mm = _mm(target_version)
            runtime_mm = _mm(__version__)
            newer = script_mm > runtime_mm
            older = script_mm < runtime_mm
        except Exception:

            def _parse_mm(v: str):
                parts = v.split(".")
                try:
                    major = int(parts[0])
                    minor = int(parts[1]) if len(parts) > 1 else 0
                    return (major, minor)
                except Exception:
                    # Fallback: compare as strings to avoid crashing the check itself
                    major = parts[0] if parts else "0"
                    minor = parts[1] if len(parts) > 1 else "0"
                    return (str(major), str(minor))

            script_mm = _parse_mm(target_version)
            runtime_mm = _parse_mm(__version__)
            newer = script_mm > runtime_mm
            older = script_mm < runtime_mm

        if not newer and not older:
            return  # major.minor match

        if newer:
            msg = (
                f"Script targets EMerge {target_version} (major.minor {script_mm[0]}.{script_mm[1]}), "
                f"but runtime is {__version__} (major.minor {runtime_mm[0]}.{runtime_mm[1]}). "
                "The script may rely on features added after your installed major.minor. "
                "Recommended: upgrade EMerge (`pip install --upgrade emerge`). "
                "If you know the script is compatible, you may remove this check."
            )
        else:  # older
            msg = (
                f"Script targets EMerge {target_version} (major.minor {script_mm[0]}.{script_mm[1]}), "
                f"but runtime is {__version__} (major.minor {runtime_mm[0]}.{runtime_mm[1]}). "
                "APIs may have changed since the targeted major.minor. "
                "Recommended: update the script for the current EMerge, or run a matching older release. "
                "If you know the script is compatible, you may remove this check."
            )

        if log:
            try:
                logger.warning(msg)
            except Exception:
                pass

        raise VersionError(msg)

    def set_physics(self, microwave: bool, heatconduction: bool):
        """Define which physics modules are active

        Args:
            microwave (bool): If the microwave physics should be active
            heatconduction (bool): If the heatconduction physics should be active
        """
        self._mw_active = microwave
        self._hc_active = heatconduction

    def activate(self, _indx: int | None = None, **variables) -> Simulation:
        """Searches for the permutaions of parameter sweep variables and sets the current geometry to the provided set."""
        self.state.activate(_indx, **variables)
        return self

    def save(self, _force_save: bool = True, screenshot: bool = False) -> None:
        """Saves the current model in the provided project directory."""
        # Ensure directory exists

        if not self.state.has_been_modified:
            logger.warning(
                "Simulation state has not been modified. Saving is aborted to prevent overwrite."
            )
            return
        else:
            logger.trace(
                "Modified simulation data detected, proceeding with storing data!"
            )

        if self._saved and not _force_save:
            logger.debug("File already saved. Terminating save procedure.")
            return

        if not self.modelpath.exists():
            self.modelpath.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {self.modelpath}")

        # Pack and save data
        dataset = self.state.get_dataset()
        data_path = self.modelpath / "simdata.emerge"
        screenshot_path = self.base_path / "model.png"

        if self.settings._save_method == "msgpack":
            save_object(str(data_path), dataset)
        else:
            try:
                joblib.dump(dataset, str(data_path))
            except PicklingError as e:
                _GlobalHandler.active().debugcollector.add_report(
                    'EMerge cannot save python functions. Set the save system to msgpack instead using em.Simulation(..., store_system="msgpack")'
                )
                raise e

        if screenshot:
            self.view(screenshot=screenshot_path, off_screen=True)

        if self._cache_run:
            cachepath = self.modelpath / "pylines.txt"
            with open(str(cachepath), "w") as f_out:
                f_out.write(self._file_lines)

        logger.info(f"Saved simulation data to: {data_path}")
        self._saved = True

    def load(self) -> None:
        """Loads the model from the project directory."""
        mesh_path = self.modelpath / "mesh.msh"
        data_path = self.modelpath / "simdata.emerge"

        if not data_path.exists():
            raise FileNotFoundError(f"Cannot find file {data_path}.")

        logger.info(f"Loaded mesh from: {mesh_path}")
        if self.settings._save_method == "msgpack":
            datapack = load_object(str(data_path))
        else:
            datapack = joblib.load(str(data_path))

        self.state.load_dataset(datapack)
        self.state.activate(0)

        logger.info(f"Loaded simulation data from: {data_path}")

    def set_loglevel(
        self, loglevel: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    ) -> None:
        """Set the loglevel for loguru.

        Args:
            loglevel ('DEBUG','INFO','WARNING','ERROR'): The loglevel
        """
        logger.trace(f"Setting loglevel to {loglevel}")
        _GlobalHandler.active().logcontroller.set_std_loglevel(loglevel)
        if loglevel not in ("TRACE"):
            gmsh.option.setNumber("General.Terminal", 0)

    def set_write_log(self) -> None:
        """Adds a file output for the logger."""
        logger.trace(f"Writing log to path = {self.modelpath}")
        _GlobalHandler.active().logcontroller.set_write_file(self.modelpath)

    def view(
        self,
        selections: list[Selection] | None = None,
        use_gmsh: bool = False,
        plot_mesh: bool = False,
        volume_mesh: bool = True,
        opacity: float | None = None,
        labels: bool = False,
        bc: bool = False,
        bw: bool = False,
        face_labels: bool = False,
        screenshot: str | None = None,
        assigned_materials: bool = False,
        off_screen: bool = False,
    ) -> None:
        """View the current geometry in either the BaseDisplay object (PVDisplay only) or
        the GMSH viewer.

        Args:
            selections (list[Selection] | None, optional): Optional selections to highlight. Defaults to None.
            use_gmsh (bool, optional): If GMSH's GUI should be used. Defaults to False.
            plot_mesh (bool, optional): If the mesh should be plot instead of the object. Defaults to False.
            volume_mesh (bool, optional): If the internal mesh should be plot instead of only the surface boundary mesh. Defaults to True
            opacity (float | None, optional): The object/mesh opacity. Defaults to None.
            labels: (bool, optional): If geometry name labels should be shown. Defaults to False.
            bc: (bool, optional): If you wish to show boundary condition selections in the view. Defaults to False
            bw: (bool, optional): If you want to view in black-white mode. Defaults to False
            face_labels: (bool, optional): Prints the labels of geometry faces. Defaults to False
            screenshot: (bool, optional): Saves a screenshot and stores the file. Defaults to False
            assigned_materials: (bool, optional): Shows the materials in geometries as assigned per unique domain. Defaults to False
            off_screen: (bool, optional): Only shows off screen (useful for combinations with Screenshot). Defaults to False
        """

        # Use GMSH if there are only 3D geometries
        if len(self.state.all3d) == 0:
            use_gmsh = True
        
        
        def default_if_none(value, default):
            if value is None:
                return default
            return value

        if use_gmsh or self.display is None:
            gmsh.model.occ.synchronize()
            gmsh.fltk.run()
            return

        if not self.mesh.defined:
            self.quick_mesh()

        if bw:
            self.display.drawing_bw()

        if assigned_materials:
            geolist = self.state.simulation_geostate()
        else:
            geolist = self.state.current_geo_state

        for geo in geolist:
            self.display.add_object(
                geo,
                mesh=plot_mesh,
                opacity=default_if_none(opacity, geo.opacity),
                volume_mesh=volume_mesh,
                label=labels,
                selectable_as=geo.name,
            )

            if face_labels and geo.dim == 3:
                for face_name in geo._face_pointers.keys():
                    if geo.face(face_name).invalid:
                        continue
                    self.display.add_object(
                        geo.face(face_name),
                        color="yellow",
                        opacity=0.1,
                        label=face_name,
                        selectable_as=face_name,
                    )

        if selections:
            [
                self.display.add_object(
                    sel, color="EMERGE-SELECT", opacity="EMERGE-SELECT", label=sel.name
                )
                for sel in selections
            ]
        if bc:
            if self._hc_active:
                for bc in self.hc.bc.boundary_conditions:
                    if bc.selection.invalid:
                        continue
                    self.display.add_object(
                        bc.selection,
                        color=bc._color,
                        opacity="EMERGE-BC",
                        label=True,
                        label_text=bc._name,
                        texture=bc._texture,
                        pbr=False,
                    )

            if self._mw_active:
                for bc in self.mw.bc.boundary_conditions:
                    if bc.selection.invalid:
                        continue
                    self.display.add_object(
                        bc.selection,
                        color=bc._color,
                        opacity=0.4,
                        label=True,
                        label_text=bc._name,
                        texture=bc._texture,
                        pbr=False,
                    )

        self.display.show(screenshot=screenshot, off_screen=off_screen)

        return None

    def set_periodic_cell(self, cell: PeriodicCell):
        """Set the given periodic cell object as the simulations peridicity.

        Args:
            cell (PeriodicCell): The PeriodicCell class
        """
        logger.trace(f"Setting {cell} as periodic cell object")
        self.mw.bc._cell = cell
        self._cell = cell
        self.mesher.periodic_cell = cell

    def set_resolution(self, resolution: float) -> Simulation:
        """Sets the discretization resolution in the various physics interfaces.


        Args:
            resolution (float): The resolution as a float. Lower resolution is a finer mesh

        Returns:
            Simulation: The same simulation object
        """
        self.mw.set_resolution(resolution)

    def commit_geometry(self) -> None:
        """Finalizes and locks the current geometry state of the simulation.

        The geometries may be provided (legacy behavior) but are automatically managed in the background.

        """
        logger.trace("Committing final geometry.")

        self.state.store_geometry_data()

        logger.trace(f"Parsed geometries = {self.state.current_geo_state}")
        self.state.manager.self_destruct()
        self.mesher.submit_objects(self.state.current_geo_state)

        # Cache GMSH properties
        for geo in self.all_geos():
            geo._cache_gmsh()

        self.state._geometry_committed = True
        self.display._facetags = [dt[1] for dt in gmsh.model.get_entities(2)]

        if self._mw_active:
            self.mw._autogenerate_bcs()

    def all_geos(self) -> list[GeoObject]:
        """Returns all geometries in a list

        Returns:
            list[GeoObject]: A list of all GeoObjects
        """
        return self.state.current_geo_state

    def all_boundaries(self) -> FaceSelection:
        """Returns a selection of all geometry boundaries

        Returns:
            FaceSelection: A selection of all boundaries
        """
        geos = self.all_geos()
        sel = geos[0].boundary()
        for geo in geos[1:]:
            sel = sel + geo.boundary()
        return sel

    def find_geo(self, name: str) -> GeoObject:
        """Searches first for an exact match and then the first partial match for the provided geometry name

        Args:
            name (str): _description_

        Returns:
            GeoObject: _description_
        """
        return self.state.manager.find_geo(name)

    def quick_mesh(self) -> None:
        logger.info("Starting quick mesh generation phase.")

        self._commit_quick_geometries()

        logger.trace(" (1) Installing periodic boundaries in mesher.")
        # Set the cell periodicity in GMSH
        if self._cell is not None:
            self.mesher.set_periodic_cell(self._cell)

        # Check if frequencies are defined: TODO: Replace with a more generic check
        if self.mw.frequencies is None:
            raise ValueError(
                "No frequencies defined for the simulation. Please set frequencies before generating the mesh."
            )

        gmsh.model.occ.synchronize()

        # Set the mesh size
        # self.mesher.set_algorithm(Algorithm3D.INITIAL_MESH_ONLY)
        self.mesher._fix_curved_boundary_meshing()
        self.mesher._configure_coarse_size()  # This makes no sense to do this here

        logger.trace(" (2) Calling GMSH mesher")
        try:
            gmsh.logger.start()
            gmsh.model.mesh.generate(3)
            logs = gmsh.logger.get()
            gmsh.logger.stop()
            for log in logs:
                logger.trace("[GMSH] " + log)
        except Exception:
            logger.error("GMSH Mesh error detected.")
            # ALLOWED PRINT
            print(_GMSH_ERROR_TEXT)
            raise

        gmsh.model.occ.synchronize()

        logger.info("GMSH Meshing complete!")
        self.mesh._pre_update(self.mesher._get_periodic_bcs(), 1/self._geo_scale_factor)
        self.mesh.exterior_face_tags = self.mesher.domain_boundary_face_tags
        self.mesh._quick_mesh = True
        logger.trace(" (3) Mesh routine complete")

    def _cleanup_entities(self):
        gmsh.model.occ.synchronize()

        # 1. Get all 3D volumes
        volumes = gmsh.model.getEntities(dim=3)

        # 2. Extract every surface, curve, and point that forms the boundary of these 3D volumes
        valid_boundary_entities = set()
        if volumes:
            valid_boundary_entities = set(
                gmsh.model.getBoundary(volumes, combined=False, oriented=False, recursive=True)
            )

        # 3. Gather all 0D, 1D, and 2D entities currently in the model
        entities_0to2 = (
            gmsh.model.getEntities(dim=0) + 
            gmsh.model.getEntities(dim=1) + 
            gmsh.model.getEntities(dim=2)
        )

        # 4. Filter for entities that aren't in the 3D boundary tree
        orphans = [ent for ent in entities_0to2 if ent not in valid_boundary_entities]

        # 5. Purge the orphans from OCC
        if orphans:
            logger.debug(f' Removing orphans: {orphans}')
            gmsh.model.occ.remove(orphans, recursive=True)
            gmsh.model.occ.synchronize()

        return orphans

    def generate_mesh(self, regenerate: bool = False) -> None:
        """Generate the mesh.
        This can only be done after commit_geometry(...) is called and if frequencies are defined.

        Args:
            name (str, optional): The mesh file name. Defaults to "meshname.msh".

        Raises:
            ValueError: ValueError if no frequencies are defined.
        """
        
        # Check if there is at least one 3D geometry
        if len(self.state.all3d) == 0:
            raise SimulationError("EMerge can only generate meshes of 3D geometries. You need at least one 3D geometry to start the mesh procedure.")
        logger.info("Starting mesh generation phase.")

        if self.mesh.defined and self.mesh._quick_mesh:
            logger.debug("Resetting quick mesh before full mesh generation.")
            self._reset_mesh()

        if not regenerate:
            if not self.state._geometry_committed:
                self.commit_geometry()

            # Set the cell periodicity in GMSH
            if self._cell is not None:
                logger.trace(" (1) Installing periodic boundaries in mesher.")
                self.mesher.set_periodic_cell(self._cell)

            # Check if frequencies are defined: TODO: Replace with a more generic check
            if self.mw.frequencies is None:
                raise ValueError(
                    "No frequencies defined for the simulation. Please set frequencies before generating the mesh."
                )

        gmsh.model.occ.synchronize()

        # Set the mesh size
        if self._mw_active:
            self.mesher._configure_bc_size(self.mw.bc.boundary_conditions)
            self.mesher._configure_mesh_size(
                self.mw.get_discretizer(), self.mw.resolution
            )  # This makes no sense to do this here

            # Validity check
            x1, y1, z1, x2, y2, z2 = gmsh.model.getBoundingBox(-1, -1)
            bb_volume = (x2 - x1) * (y2 - y1) * (z2 - z1)
            wl = self._geo_scale_factor * 299792458 / self.mw.frequencies[-1]
            Nelem = int(5 * bb_volume / (wl**3))

            if Nelem > 100_000 and DEFAULT_SETTINGS.size_check:
                _GlobalHandler.active().debugcollector.add_report(
                    f"An estimated {Nelem} tetrahedra are required for the bounding box of the geometry. This may imply a simulation domain that is very large."
                    + "To disable this message. Set the .size_check parameter in model.settings to False."
                )

                raise SimulationError("Simulation requires too many elements.")

        if self._hc_active:
            from .geo.operations import bounding_box

            xmin, xmax, ymin, ymax, zmin, zmax = bounding_box()
            diagmax = (
                (xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2
            ) ** 0.5
            self.mesher._configure_mesh_size(lambda x: diagmax / 10, 1.0)

        # Remove orphaned surfaces
        self._cleanup_entities()
        logger.info("Calling GMSH mesher")
        
        try:
            gmsh.logger.start()
            gmsh.model.mesh.generate(3)
            logs = gmsh.logger.get()
            gmsh.logger.stop()
            for log in logs:
                logger.trace("[GMSH] " + log)
        except Exception:
            logger.error("GMSH Mesh error detected.")
            # ALLOWED PRINT
            print(_GMSH_ERROR_TEXT)
            raise

        self._apply_size_scaling()
        
        logger.info("GMSH Meshing complete!")
        self.mesh._pre_update(self.mesher._get_periodic_bcs(), 1/self._geo_scale_factor)
        if self.settings.safe_mode:
            self.mesh.diagnose()

        self.mesh.exterior_face_tags = self.mesher.domain_boundary_face_tags
        gmsh.model.occ.synchronize()
        self.state.store_geometry_data()
        logger.trace(" (3) Mesh routine complete")

    def parameter_sweep(
        self, clear_mesh: bool = True, **parameters: np.ndarray
    ) -> Generator[tuple[float, ...], None, None]:
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
        dims = np.meshgrid(*[parameters[key] for key in paramlist], indexing="ij")
        dims_flat = [dim.flatten() for dim in dims]

        self.mw.cache_matrices = False
        logger.trace("Starting parameter sweep.")

        for i_iter in range(dims_flat[0].shape[0]):
            if clear_mesh and i_iter > 0:
                logger.info("Cleaning up mesh.")
                self.reset(geometry=True, physics=True, mesh=True)

            # Set the outer variable parameters
            params = {key: dim[i_iter] for key, dim in zip(paramlist, dims_flat)}
            self.state.set_parameters(params)

            logger.info(f"Iterating: {params}")

            # Yield the parameters as tuples
            if len(dims_flat) == 1:
                yield dims_flat[0][i_iter]
            else:
                yield (dim[i_iter] for dim in dims_flat)  # type: ignore

            # Store the geometry data (geometries and mesh) to the current parameter setting
            if not clear_mesh:
                self.state.store_geometry_data()

        if not clear_mesh:
            self.state.store_geometry_data()

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
        logger.trace(f"Writing geometry to {filename}")
        gmsh.write(filename)

    def set_solver(self, solver: EMSolver | Solver):
        """Set a given Solver class instance as the main solver.
        Solvers will be checked on validity for the given problem.

        Args:
            solver (EMSolver | Solver): The solver objects
        """
        logger.trace(f"Setting solver to {solver}")
        self.mw.solveroutine.set_solver(solver)

    @staticmethod
    def from_simulations(name: str, *sims: Simulation, **kwargs) -> Simulation:
        new = Simulation(name, **kwargs)
        for sim in sims:
            new.state.import_from(sim.state)
        new.state.data.remove_empty_datasets()
        new.state.set_modified()
        new.state.activate(0)
        return new

    ############################################################
    #                     DEPRICATED FUNCTIONS                #
    ############################################################

    ############################################################
    #                   ADAPTIVE MESH REFINEMENT              #
    ############################################################

    def adaptive_mesh_refinement(
        self,
        max_steps: int = 6,
        min_refined_passes: int = 1,
        convergence: float = 0.02,
        magnitude_convergence: float = 2.0,
        phase_convergence: float = 180,
        max_tets: int = 1_000_000,
        order: float = 3.0,
        theta: float = 0.7,
        minimum_steps: int = 1,
        frequency: float | list[float] | None = None,
        show_mesh: bool = False,
    ) -> SimulationDataset:
        """Adaptive mesh refinement via full remesh-from-scratch, driven by a
        single native gmsh background size field per pass.
    
        SIZE PREDICTION: equidistribution formula --
    
            h_new = h_current * (error_target / error_current) ** (1/order)
    
        SIZE DELIVERY: self.mesher.set_error_size_field() installs a NodeData
        view on a frozen auxiliary model, registered into self.mesher.mesh_fields
        alongside every other size source. IMPORTANT: h_new from the
        equidistribution formula is per-TET; set_error_size_field needs
        per-NODE data. The scatter step below (minimum over each node's
        incident tets) is required -- skipping it doesn't crash (gmsh silently
        accepts a too-long values array, since n_tets > n_nodes is the normal
        case for a tet mesh) but produces a field that fails to interpolate
        anywhere, which looks exactly like "no refinement happening" with no
        error to point at.
        """
        from .physics.microwave.adaptive_mesh import compute_convergence
    
        max_freq = np.max(self.mw.frequencies)
        sim_freqs = frequency if frequency is not None else [max_freq]
        if isinstance(sim_freqs, float):
            sim_freqs = [sim_freqs]
        NF = len(sim_freqs)
    
        S_matrices: list[list[np.ndarray]] = []
        logger.info(f"Initial mesh has {self.mesh.n_tets} tetrahedra")
        passed = 0
        self.state.stash()
    
        logger.info("Starting adaptive mesh refinement (full remesh, background field).")
    
        self._cleanup_entities()

        previous_tet_count = self.mesh.n_tets
        for step in range(1, max_steps + 1):
            self.data.sim.new(iter_step=step)
    
            fields, Smats = [], []
            logger.debug("Running frequency simulation.")
            for sf in sim_freqs:
                data, solve_ids = self.mw._run_adaptive_mesh(step, sf)
                fields.append(data.field[-1])
                Smats.append(data.scalar[-1].Sp)
            S_matrices.append(Smats)
    
            # -------------------------------------------------------------------
            # Convergence check
            # -------------------------------------------------------------------
            if step > minimum_steps:
                S0s, S1s = S_matrices[-2], S_matrices[-1]
                max_complx = max_mag = max_phase = 0.0
                for i in range(NF):
                    conv_complex, conv_mag, conv_phase = compute_convergence(S0s[i], S1s[i])
                    max_complx = max(max_complx, conv_complex)
                    max_mag = max(max_mag, conv_mag)
                    max_phase = max(max_phase, conv_phase)
    
                logger.info(
                    f"Pass {step}: Convergence = {max_complx:.3f}, Mag = {max_mag:.3f}, Phase = {max_phase:.1f} deg"
                )
    
                if max_complx <= convergence and max_phase < phase_convergence and max_mag < magnitude_convergence:
                    logger.info(f"Pass {step}: Mesh refinement passed!")
                    passed += 1
                else:
                    passed = 0
    
            if passed >= min_refined_passes and step > minimum_steps:
                logger.info(f"Adaptive mesh refinement successful with {self.mesh.n_tets} tetrahedra.")
                break
    
            # -------------------------------------------------------------------
            # Error + current size per tet
            # -------------------------------------------------------------------
            errors = np.empty((self.mesh.n_tets, NF), dtype=np.float64)
            h_current_per_freq = np.empty((self.mesh.n_tets, NF), dtype=np.float64)
            for i in range(NF):
                error, h_current = fields[i]._solution_quality(solve_ids)
                errors[:, i] = error
                h_current_per_freq[:, i] = h_current
            error_current = np.max(errors, axis=1)
            h_current = h_current_per_freq[:, 0]

            logger.debug(
                f"h_current: min={h_current.min():.3e}, max={h_current.max():.3e}, "
                f"NaN={np.isnan(h_current).sum()}, zero={(h_current == 0).sum()}"
            )
            logger.debug(
                f"error_current: min={error_current.min():.3e}, max={error_current.max():.3e}, "
                f"NaN={np.isnan(error_current).sum()}, zero={(error_current == 0).sum()}"
            )
    
            # -------------------------------------------------------------------
            # Equidistribution size prediction, per TET.
            # -------------------------------------------------------------------
            n_tets = self.mesh.n_tets
            total_sq_error = np.sum(error_current**2)
            error_target = np.sqrt(theta * total_sq_error / n_tets)
    
            ratio = error_target / np.maximum(error_current, 1e-300)
            h_new = h_current * ratio ** (1.0 / order)

            h_new = np.clip(h_new, 0.7 * h_current, h_current * 1.5)
    
            ratio_applied = h_new / h_current
            logger.info(
                f" - h_new range: min={float(h_new.min())*1000:.4f}mm, "
                f"max={float(h_new.max())*1000:.4f}mm "
                f"(theta={theta}, order={order})"
            )
            logger.info(
                f" - applied ratio range: min={float(ratio_applied.min()):.3f}, "
                f"max={float(ratio_applied.max()):.3f}"
            )
    
            # -------------------------------------------------------------------
            # Scatter per-TET h_new onto NODES (minimum over incident tets) --
            # set_error_size_field needs per-node data, not per-tet.
            # -------------------------------------------------------------------
            nodes = np.asarray(self.mesh.nodes, dtype=np.float64)  # (3, N)
            tets = np.asarray(self.mesh.tets, dtype=np.int64)      # (4, M)
    
            size_at_node = np.full(nodes.shape[1], np.inf, dtype=np.float64)
            for corner in range(4):
                np.minimum.at(size_at_node, tets[corner, :], h_new)
    
            if np.isinf(size_at_node).any():
                # Shouldn't happen (every node belongs to at least one tet), but
                # gmsh needs finite values regardless.
                size_at_node[np.isinf(size_at_node)] = h_current.max()
    
            logger.info(
                f" - size_at_node range: min={float(size_at_node.min())*1000:.4f}mm, "
                f"max={float(size_at_node.max())*1000:.4f}mm"
            )
    
            # -------------------------------------------------------------------
            # Install via the Mesher's own size-field convention, then let the
            # NORMAL remesh pipeline handle everything else.
            # -------------------------------------------------------------------
            self.mesher.clear_error_size_field()
            self.mesher.set_error_size_field(nodes, tets, size_at_node)
    
            self._reset_mesh()
            self.generate_mesh(regenerate=True)
    
            logger.info(f"    Pass {step}: New mesh has {self.mesh.n_tets} tetrahedra (Δ={(self.mesh.n_tets/previous_tet_count-1)*100:.1f}%)")
            previous_tet_count = self.mesh.n_tets
            if show_mesh:
                self.view(plot_mesh=True, volume_mesh=True)
    
            if self.mesh.n_tets > max_tets:
                logger.warning(
                    f"Aborting refinement because the number of tets exceeds the maximum: {self.mesh.n_tets}>{max_tets}"
                )
                break
    
        self.mesher.clear_error_size_field()
    
        if passed < min_refined_passes:
            logger.warning("Adaptive mesh refinement did not converge!")
    
        if show_mesh:
            self.view(plot_mesh=True, volume_mesh=True)
    
        old = self.state.reload()
        self.state.store_geometry_data()
        return old

class SimulationBeta(Simulation):
    def __post_init__(self):
        pass
        # self.mesher.set_algorithm(Algorithm3D.HXT)
        # logger.debug('Setting mesh algorithm to HXT')

    def __enter__(self) -> SimulationBeta:
        """This method is depricated with the new atexit system. It still exists for backwards compatibility.

        Returns:
            SimulationBeta: the SimulationBeta object
        """
        return self

    def _reset_mesh(self):
        gmsh.model.mesh.clear()
        self.mw.reset(_reset_bc=False)
        self.state.reset_mesh()

    @staticmethod
    def guess_R(P: float, last_ratio: float = 1.0) -> float:
        # Coefficients for the refinement ratio calculation.

        a0 = 0.5
        c0 = 0.85
        x0 = 12
        q0 = (1 - a0) * 2 / np.pi
        b0 = np.tan((c0 - a0) / q0) / x0
        q0 = (0.8 - a0) * 2 / np.pi

        ratio = a0 + np.arctan(b0 * P) * q0
        if last_ratio > 1.0:
            ratio = ratio / 0.9
        if last_ratio < 1.0:
            ratio = ratio * 0.9

        return ratio

    @staticmethod
    def compute_ratio(
        new_point_percentage: float,
        ratios: np.ndarray,
        percentages: np.ndarray,
        P_target: float,
    ) -> float:
        """
        Strategy:
        - n=0: use guess_R(P_target, throttle=1.0)
        - n=1: use guess_R with throttle 0.5 if last P < target, else 2.0 if last P > 2*target,
                else keep the same R (already acceptable)
        - n=2: fit P = a1*(1/R) + a0 and solve for R; fallback to through-origin model if needed

        Returns R_guess in (0, 1].
        """
        ratios = np.asarray(ratios, dtype=float)
        percentages = np.asarray(percentages, dtype=float)

        # Clean
        n = ratios.size

        if n == 1:
            last_R = float(ratios[-1])
            last_P = float(percentages[-1])

            if P_target <= last_P <= 2.0 * P_target:
                # Already acceptable
                return last_R * ((1.5 * P_target) / last_P) ** 0.2

            if last_P > P_target * 2.0:
                return last_R / 0.8
            else:
                return last_R * 0.8

        P_target = P_target * 1.5
        x = 1.0 / ratios
        y = percentages
        dy = y - P_target

        # Indices by side of target
        idx_lo = np.where(dy < 0)[0]  # below target
        idx_hi = np.where(dy > 0)[0]  # above target
        idx_eq = np.where(dy == 0)[0]

        # Exact hit
        if idx_eq.size > 0:
            # Return corresponding R (already perfect)
            R_exact = float(ratios[idx_eq[0]])
            return float(np.clip(R_exact, 1e-6, 1.0))

        def pick_two(indices, reverse=False):
            # pick two closest to target among given indices
            order = np.argsort(np.abs(dy[indices]))
            if reverse:
                # when all below, we want the two with largest y (closest from below)
                order = np.argsort(-y[indices])
            return indices[order[:2]]

        # Choose two points
        if idx_lo.size > 0 and idx_hi.size > 0:
            # Bracket: closest below and closest above
            i_lo = idx_lo[np.argmin(np.abs(dy[idx_lo]))]
            i_hi = idx_hi[np.argmin(np.abs(dy[idx_hi]))]
            i1, i2 = i_lo, i_hi
        elif idx_lo.size == 0:
            # All above: take two smallest-above (closest to target)
            i12 = pick_two(idx_hi)  # by |dy|
            if i12.size < 2 and idx_hi.size >= 2:
                i12 = idx_hi[:2]
            i1, i2 = i12[0], i12[1]
        elif idx_hi.size == 0:
            # All below: take two largest-below (closest from below)
            i12 = pick_two(idx_lo, reverse=True)
            if i12.size < 2 and idx_lo.size >= 2:
                i12 = idx_lo[:2]
            i1, i2 = i12[0], i12[1]
        else:
            # Fallback (should not trigger)
            i1, i2 = 0, 1

        x1, y1 = float(x[i1]), float(y[i1])
        x2, y2 = float(x[i2]), float(y[i2])

        # Linear solve in y(x): y = y1 + (y2-y1)/(x2-x1) * (x - x1)
        # Target xt: xt = x1 + (P_target - y1) * (x2 - x1) / (y2 - y1)
        if abs(y2 - y1) > 1e-16:
            xt = x1 + (P_target - y1) * (x2 - x1) / (y2 - y1)
        else:
            # Degenerate: same y; choose midpoint in x
            xt = 0.5 * (x1 + x2)

        # Ensure positive xt
        xt = float(max(xt, 1e-16))
        R_new = 1.0 / xt

        # Clamp to (0, 1]
        return float(np.clip(R_new, 1e-6, 1.0))

    def adaptive_mesh_refinement(
        self,
        max_steps: int = 6,
        min_refined_passes: int = 1,
        convergence: float = 0.02,
        magnitude_convergence: float = 2.0,
        phase_convergence: float = 180,
        max_tets: int = 1e6,
        refinement_ratio: float = 0.6,
        growth_rate: float = 1.6,
        minimum_refinement_percentage: float = 20.0,
        error_field_inclusion_percentage: float = 50.0,
        minimum_steps: int = 1,
        frequency: float | list[float] = None,
        show_mesh: bool = False,
    ) -> SimulationDataset:
        """A beta-version of adaptive mesh refinement.

        Convergence Criteria:
            (1): max(abs(S[n]-S[n-1]))
            (2): max(abs(abs(S[n]) - abs(S[n-1])))
            (3): max(angle(S[n]/S[n-1])) * 180/π

        Args:
            max_steps (int, optional): The maximum number of refinement steps. Defaults to 6.
            min_refined_passes (int, optional): The minimum number of refined passes. Defaults to 1.
            convergence (float, optional): The S-paramerter convergence (1). Defaults to 0.02.
            magnitude_convergence (float, optional): The S-parameter magnitude convergence (2). Defaults to 2.0.
            phase_convergence (float, optional): The S-parameter Phase convergence (3). Defaults to 180.
            refinement_ratio (float, optional): The size reduction of mesh elements by original length. Defaults to 0.75.
            growth_rate (float, optional): The mesh size growth rate. Defaults to 3.0.
            minimum_refinement_percentage (float, optional): The minimum mesh size increase . Defaults to 15.0.
            error_field_inclusion_percentage (float, optional): A percentage of tet elements to be included for refinement. Defaults to 5.0.
            minimum_steps (int, optional): The minimum number of adaptive steps to execute. Defaults to 1.
            frequency (float, optional): The refinement frequency. Defaults to None.
            show_mesh (bool, optional): If the intermediate meshes should be shown (freezes simulation). Defaults to False

        Returns:
            SimulationDataset: _description_
        """
        from .physics.microwave.adaptive_mesh import (
            select_refinement_indices,
            reduce_point_set,
            compute_convergence,
            tet_to_node,
        )

        max_freq = np.max(self.mw.frequencies)

        if frequency is not None:
            sim_freqs = frequency
            if isinstance(sim_freqs, float):
                sim_freqs = [sim_freqs]
        else:
            sim_freqs = [max_freq]

        S_matrices: list[list[np.ndarray]] = []

        last_n_tets: int = self.mesh.n_tets
        logger.info(f"Initial mesh has {last_n_tets} tetrahedra")

        passed = 0

        self.state.stash()

        NF = len(sim_freqs)

        original_ratio = refinement_ratio
        logger.info("Stating adaptive mesh refinement process...")

        for step in range(1, max_steps + 1):
            self.data.sim.new(iter_step=step)

            datas = []
            fields = []
            Smats = []

            logger.debug("Running frequency simulation.")
            for sf in sim_freqs:
                data, solve_ids = self.mw._run_adaptive_mesh(step, sf)
                datas.append(data)
                fields.append(data.field[-1])
                Smats.append(data.scalar[-1].Sp)

            S_matrices.append(Smats)

            if step > minimum_steps:
                S0s = S_matrices[-2]
                S1s = S_matrices[-1]

                max_complx = 0
                max_mag = 0
                max_phase = 0

                for i in range(NF):
                    conv_complex, conv_mag, conv_phase = compute_convergence(
                        S0s[i], S1s[i]
                    )
                    max_complx = max(max_complx, conv_complex)
                    max_mag = max(max_mag, conv_mag)
                    max_phase = max(max_phase, conv_phase)

                logger.info(
                    f"Pass {step}: Convergence = {max_complx:.3f}, Mag = {max_mag:.3f}, Phase = {max_phase:.1f} deg"
                )

                if (
                    max_complx <= convergence
                    and max_phase < phase_convergence
                    and max_mag < magnitude_convergence
                ):
                    logger.info(f"Pass {step}: Mesh refinement passed!")
                    passed += 1
                else:
                    passed = 0

            if passed >= min_refined_passes and step > minimum_steps:
                logger.info(
                    f"Adaptive mesh refinement successfull with {self.mesh.n_tets} tetrahedra."
                )
                break

            errors = np.empty((self.mesh.n_tets, NF), dtype=np.float64)
            logger.debug(
                f"Computing error estimates for {self.mesh.n_tets} tetrahedra."
            )
            for i in range(NF):
                error, lengths = fields[i]._solution_quality(solve_ids)
                errors[:, i] = error
            error = np.max(errors, axis=1)

            logger.debug("Selecting refinement indices.")
            refine_tet_ids = select_refinement_indices(
                error, error_field_inclusion_percentage / 100
            )
            refine_tet_ids = refine_tet_ids[::-1]

            n_refine_points = refine_tet_ids.shape[0]
            rine_points_percentage = n_refine_points / self.mesh.n_tets * 100
            original_ratio = 0.75 * original_ratio + 0.25 * refinement_ratio
            refinement_ratio = original_ratio

            Ratios = []
            Percentages = []

            included = np.zeros((self.mesh.n_tets,), dtype=np.bool_)
            included[refine_tet_ids] = True

            coords, sizes = tet_to_node(
                self.mesh.nodes, self.mesh.tets, lengths, included
            )
            self.mesher._amrobj.add_refinement_points(
                coords, sizes, refinement_ratio * np.ones_like(sizes)
            )

            new_ids = reduce_point_set(
                self.mesher._amrobj._amr_coords,
                growth_rate,
                self.mesher._amrobj._amr_sizes,
                refinement_ratio,
                0.20,
            )

            nremoved = self.mesher._amrobj.npts - len(new_ids)

            logger.info(
                f"    Pass {step}: Added {len(sizes) - nremoved} new refinement points with ratio {refinement_ratio}."
            )

            self.mesher._amrobj.reduce_set(new_ids)

            logger.debug(f"    Initial refinement ratio: {refinement_ratio}")

            # Mesh refinement loop. Only escapes if the mesh refined a certain set percentage.
            counter = 0
            while True:
                counter += 1
                if counter == 10:
                    logger.warning(
                        "    More than 10 attempts at reaching the target refinement. Continuing with current."
                    )
                    break
                counter += 1
                self._reset_mesh()
                self.mesher._amrobj.set_refinement_function(growth_rate, 2.0)
                self.generate_mesh(True)
                percentage = (self.mesh.n_tets / last_n_tets - 1) * 100
                logger.info(
                    f"    Pass {step}: New mesh has {self.mesh.n_tets} (+{percentage:.1f}%) tetrahedra."
                )

                Ratios.append(refinement_ratio)
                Percentages.append(percentage)

                if len(Percentages) >= 2:
                    if abs(Percentages[-2] - Percentages[-1]) == 0.0:
                        logger.warning(
                            f"No refinement realized, decreasing refinment ratio."
                        )
                        refinement_ratio = refinement_ratio * 0.5
                        self.mesher._amrobj.set_ratio(refinement_ratio)
                        continue

                if percentage < minimum_refinement_percentage or percentage > (
                    minimum_refinement_percentage * 2
                ):
                    refinement_ratio = self.compute_ratio(
                        rine_points_percentage,
                        Ratios,
                        Percentages,
                        minimum_refinement_percentage,
                    )
                    logger.info(
                        f"    Refinement target not reached! New ratio = {refinement_ratio:.3f}"
                    )
                    self.mesher._amrobj.set_ratio(refinement_ratio)
                    if refinement_ratio >= 1.0:
                        logger.warning(
                            f"Refinement ratio pushed above 1.0... continuing with current percentage."
                        )
                        break
                    continue

                break

            last_n_tets = self.mesh.n_tets
            if show_mesh:
                self.view(plot_mesh=True, volume_mesh=True)

            if last_n_tets > max_tets:
                logger.warning(
                    f"Aborting refinement because the number of tets exceeds the maximum: {last_n_tets}>{max_tets}"
                )
                break
        if passed < min_refined_passes:
            logger.warning("Adaptive mesh refinement did not converge!")

        if show_mesh:
            self.view(plot_mesh=True, volume_mesh=True)
        old = self.state.reload()
        self.state.store_geometry_data()

        return old
