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
from ._global import _GlobalHandler, _BaseManager
from .mesh3d import Mesh3D
from .geometry import GeoObject, GeoVolume, GeoSurface, _GeometryManager
from .dataset import SimulationDataset
from loguru import logger
from typing import Any
import numpy as np
import gmsh
from pathlib import Path


class _SimStateManager(_BaseManager):
    def __init__(self):
        self.states: list[SimState] = []
        self.active: SimState | None = None

    def sign_on(self, state: SimState) -> None:
        if state not in self.states:
            self.states.append(state)
        self.active = state

    def sign_off(self, state: SimState) -> None:
        if state in self.states:
            self.states.remove(state)
        if self.active is state:
            self.active = None if not self.states else self.states[-1]
    
    def reset(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.active = None
        for state in self.states:
            state.reset_data()
        self.states.clear()
        self.states = []


class SimState:
    def __init__(self, modelname: str, modelpath: Path):
        self.modelname: str = modelname
        self.modelpath: Path = modelpath
        self.mesh: Mesh3D = Mesh3D()
        self.data: SimulationDataset = SimulationDataset()
        self.params: dict[str, float] = dict()
        self._stashed: SimulationDataset | None = None
        self.has_been_modified: bool = False

        # --- STATES
        self._geometry_committed: bool = False
        self.sign_on()

    @property
    def manager(self) -> _GeometryManager:
        return _GlobalHandler.active().geomanager
    def import_from(self, other: SimState) -> None:
        """Imports the dataset from another simulation

        Args:
            other (SimState): _description_
        """
        self.data.merge_with(other.data)

    def set_modified(self) -> None:
        logger.trace(
            "Detected a change in simulation data. Setting modified flag to True"
        )
        self.has_been_modified = True

    def sign_on(self):
        _GlobalHandler.active().simstates.sign_on(self)

    def sign_off(self) -> None:
        _GlobalHandler.active().simstates.sign_off(self)

    @property
    def current_geo_state(self) -> list[GeoObject]:
        return self.manager.all_geometries()

    @property
    def all3d(self) -> list[GeoVolume]:
        return [geo for geo in self.manager.all_geometries() if geo.dim == 3]
    
    @property
    def all2d(self) -> list[GeoVolume]:
        return [geo for geo in self.manager.all_geometries() if geo.dim == 2]
    
    def simulation_geostate(self) -> list[GeoObject]:
        geodict: dict[int, dict[int, GeoObject]] = {
            0: dict(),
            1: dict(),
            2: dict(),
            3: dict(),
        }

        for geo in self.manager.all_geometries():
            dim = geo.dim
            prio = geo._priority
            for tag in geo.tags:
                current = geodict[dim].get(tag, None)
                if current is None:
                    geodict[dim][tag] = geo
                else:
                    if prio >= current._priority:
                        geodict[dim][tag] = geo

        geolist = []
        for dim, dct in geodict.items():
            for tag, geometry in dct.items():
                geolist.append(
                    GeoObject.from_dimtags(
                        [(dim, tag)], _submit_geometry=False
                    ).set_material(geometry.material)
                )
        return geolist

    def reset_geostate(self) -> None:
        self.manager.reset()
        self.manager.sign_in(self.modelname)
        self.clear_mesh()

    def reset_data(self) -> None:
        """Resets the simulation dataset to an empty one."""
        self.data.clean()
        del self.mesh
        del self.data
        del self._stashed
        self.modelname: str = ""
        self.mesh: Mesh3D = Mesh3D()
        self.data: SimulationDataset = SimulationDataset()
        self.params: dict[str, float] = dict()
        self._stashed: SimulationDataset | None = None
        self.init()

    def init(self) -> None:
        """Initializes the Simstate to a clean starting point."""
        self.mesh = Mesh3D()
        self.reset_geostate()
        self.data.initialize(**self.params)
        self.sign_on()

    def stash(self) -> None:
        """Stashes the simstate data to run simulations and restore it later."""
        self._stashed = self.data
        self.data = SimulationDataset()

    def set_parameters(self, parameters: dict[str, float]) -> None:
        """Define the simulation parameter sweep to a set of outer variables
        defined by a dictionary of string, float pairs.

        Args:
            parameters (dict[str, float]): The parameter sweep slice.
        """
        self.params = parameters
        self.data.initialize(**parameters)

    def reload(self) -> SimulationDataset:
        """Reload stashed data into the simstate memory

        Returns:
            SimulationDataset: _description_
        """
        old = self._stashed
        self.data = self._stashed
        self._stashed = None
        return old

    def reset_mesh(self) -> None:
        """Resets and clears the current mesh."""
        self.mesh = Mesh3D()

    def set_mesh(self, mesh: Mesh3D) -> None:
        """Overwrite the mesh with a new one."""
        self.mesh = mesh

    def set_geos(self, geos: list[GeoObject]) -> None:
        """Activate a set of geometry objects to the simstate geometry field.

        Args:
            geos (list[GeoObject]): _description_
        """

        self.manager.set_geometries(geos)

    def clear_mesh(self) -> None:
        """resets the current mesh object to an empty one."""
        self.mesh = Mesh3D()

    def store_geometry_data(self) -> None:
        """Saves the current geometry state to the simulatin dataset"""
        logger.trace("Storing geometries in data.sim")
        self.data.sim["geos"] = self.current_geo_state
        self.data.sim["mesh"] = self.mesh

    def get_dataset(self) -> dict[str, Any]:
        """Create a dict of the file data to store to the harddrive.

        Returns:
            dict[str, Any]: _description_
        """
        self.data.remove_empty_datasets()
        return dict(simdata=self.data, mesh=self.mesh)

    def load_dataset(self, dataset: dict[str, Any]):
        """Load the data from a dataset

        Args:
            dataset (dict[str, Any]): _description_
        """
        self.data = dataset["simdata"]
        self.mesh = dataset["mesh"]

    def activate(self, _indx: int | None = None, **variables):
        """Searches for the permutaions of parameter sweep variables and sets the current geometry to the provided set."""
        if _indx is not None:
            dataset = self.data.sim.index(_indx)
        else:
            dataset = self.data.sim.find(**variables)

        variables = ", ".join([f"{key}={value}" for key, value in dataset.vars.items()])
        logger.info(f"Activated entry with variables: {variables}")
        self.set_mesh(dataset["mesh"])
        self.set_geos(dataset["geos"])
        return self

    ############################################################
    #                       GMSH LIKE METHODS                  #
    ############################################################

    def getCenterOfMass(self, dim: int, tag: int) -> tuple[float, float, float]:
        if self.mesh.defined is False:
            return gmsh.model.occ.getCenterOfMass(dim, tag)
        return self.mesh.dimtag_to_center[(dim, tag)]

    def getPoints(self, dimtags: list[tuple[int, int]]) -> list[np.ndarray]:
        """Returns a list of np.array([x,y,z]) coordinates corresponding to the provided dimtags

        Args:
            dimtags (list[tuple[int, int]]): _description_

        Returns:
            list[np.ndarray]: _description_
        """
        if not self.mesh.defined:
            points = gmsh.model.get_boundary(dimtags, recursive=True, oriented=False)
            coordinates = [gmsh.model.getValue(*p, []) for p in points]
            return coordinates
        points = []
        id_set = []
        for dt in dimtags:
            id_set.append(self.mesh.dimtag_to_nodes[dt])
        ids = np.unique(np.concatenate(id_set))
        points = [self.mesh.nodes[:, i] for i in ids]
        return points

    def getBoundingBox(
        self, dim: int, tag: int
    ) -> tuple[float, float, float, float, float, float]:
        """Returns the bounding box corresponding to entity defined by (dim,tag)

        Args:
            dim (int): _description_
            tag (int): _description_

        Returns:
            tuple[float, float, float, float, float, float]: _description_
        """
        if self.mesh.defined is False:
            return gmsh.model.occ.getBoundingBox(dim, tag)
        return self.mesh.dimtag_to_bb[(dim, tag)]

    def getNormal(self, facetag: int) -> np.ndarray:
        """Returns the normal vector of a facetag

        Args:
            facetag (int): _description_

        Returns:
            np.ndarray: _description_
        """
        if self.mesh.defined:
            return self.mesh.ftag_to_normal[facetag]
        else:
            return np.array(gmsh.model.getNormal(facetag, (0, 0)))

    def getCharPoint(self, facetag: int) -> np.ndarray:
        """Returns a coordinate that is always on the surface and roughly in the center

        Args:
            facetag (int): _description_

        Returns:
            np.ndarray: _description_
        """
        return self.mesh.ftag_to_point[facetag]

    def getArea(self, tag: int) -> float:
        if self.mesh.defined is True:
            area = sum([self.mesh.areas[tri] for tri in self.mesh.ftag_to_tri[tag]])
            return area
        else:
            area = gmsh.model.occ.getMass(2, tag)
            return area
