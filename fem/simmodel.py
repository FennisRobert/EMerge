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
from .geo3d import GMSHObject
import gmsh
from .physics.edm.emfreq3d import Electrodynamics3D
from .mesh3d import Mesh3D
from typing import Literal, Type
from loguru import logger
import numpy as np
from .selection import Selector, FaceSelection
import sys
from .logsettings import logger_format
from .modeling.modeler import Modeler
from .plotting.display import BaseDisplay

class Simulation3D:

    def __init__(self, 
                 modelname: str, 
                 display: Type[BaseDisplay] = None,
                 loglevel: Literal['DEBUG','INFO','WARNING','ERROR'] = 'INFO'):
        """Generate a Simulation3D class object.

        As a minimum a file name should be provided. Additionally you may provide it with any
        class that inherits from BaseDisplay. This will then be used for geometry displaying.

        Args:
            modelname (str): The model name
            display (BaseDisplay, optional): The BaseDisplay class type to use. Defaults to None.
            loglevel ("DEBUG","INFO","WARNING","ERROR, optional): _description_. Defaults to 'INFO'.
        
        """
        self.modelname = modelname
        self.mesher: Mesher = Mesher()
        self.physics: Electrodynamics3D = Electrodynamics3D(self.mesher)
        self.mesh: Mesh3D = Mesh3D(self.mesher)
        self.select: Selector = Selector()
        self.modeler: Modeler = Modeler()
        self.display: BaseDisplay = None
        self._geometries: list[GMSHObject] = []
        self.set_loglevel(loglevel)

        if display is not None:
            self.display = display(self.mesh)
        

    def set_loglevel(self, loglevel: Literal['DEBUG','INFO','WARNING','ERROR']) -> None:
        handler = {"sink": sys.stdout, "level": loglevel, "format": logger_format}
        logger.configure(handlers=[handler])
        #logger.remove()
        #logger.add(sys.stderr, format=logger_format)
    
    def view(self) -> None:
        """Preview the geometry as currently defined using the GMSH viewer.
        
        This function simply calls: 
        >>> gmsh.model.occ.synchronize()
        >>> gmsh.fltk.run()
        """
        if self.display is not None and self.mesh.defined:
            try:
                for obj in self._geometries:
                    self.display.add_object(obj, color=obj.color, opacity=obj.opacity)
                self.display.show()
                return
            except NotImplementedError as e:
                logger.warning('The provided BaseDisplay class does not support object display. Please make' \
                'sure that this method is properly implemented.')
        gmsh.model.occ.synchronize()
        gmsh.fltk.run()
       
    
    def define_geometry(self, *geometries: list[GMSHObject]) -> None:
        """Provide the physics engine with the geometries that are contained and ought to be included
        in the simulation. Please make sure to include all geometries. Its currently unclear how the
        system behaves if only a part of all geometries are included.

        """
        self._geometries = unpack_lists(geometries)
        self.mesher.submit_objects(self._geometries)
        self.physics._initialize_bcs()

    
    def generate_mesh(self, name: str = "meshname.msh"):
        """Generate the mesh. 
        This can only be done after define_geometry(...) is called and if frequencies are defined.

        Args:
            name (str, optional): The mesh file name. Defaults to "meshname.msh".

        Raises:
            ValueError: ValueError if no frequencies are defined.
        """
        if self.physics.frequencies is None:
            raise ValueError('No frequencies defined for the simulation. Please set frequencies before generating the mesh.')
        if '.msh' not in name:
            name = name + '.msh'
            logger.warning(f'No .msh extension added, renamed to {name}')

        gmsh.model.occ.synchronize()
        self.mesher.set_mesh_size(self.physics.get_discretizer(), self.physics.resolution)
        gmsh.model.mesh.generate(3)
        gmsh.write(name)
        self.mesh.update()
        gmsh.model.occ.synchronize()
        self.physics.mesh = self.mesh

    def get_boundary(self, face: FaceSelection = None, tags: list[int] = None) -> tuple[np.ndarray, np.ndarray]:
        ''' Return boundary data. 
        
        Parameters
        ----------
        obj: GMSHObject
        tags: list[int]
        
        Returns:
        ----------
        nodes: np.ndarray
        triangles: np.ndarray
        '''
        if tags is None:
            tags = face.tags
        tri_ids = self.mesh.get_triangles(tags)
        return self.mesh.nodes, self.mesh.tris[:,tri_ids]

    def step1(self) -> None:
        '''
        Step 1: Initialize the model and create a new geometry.

        >>> with fem.Simulation3D("My Model") as model:
                object1 = fem.modeler.Box(...)
                object2 = fem.modeler.CoaxCyllinder(...)
                model.define_geometry(...)
        '''
        pass
    def step2(self) -> None:
        '''
        Step 2: Create the mesh

        example:
        >>> model.generate_mesh()
        '''
        pass
    def step3(self) -> None:
        '''
        Step 3: Setup the physics

        >>> model.physics.resolution = 0.2
        >>> model.physics.assign(*bcs)
        >>> model.physics.set_frequency(np.linspace(f1, f2, N))
        >>> data = model.run_frequency_domain()
        '''
        pass
    def step4(self) -> None:
        '''
        Step 4: Post processing

        The data is provided as a set of data defined for global variables. Each simulation run gets
        A value for these global variables. Therefor, you can select which variable you want as your inner axis.

        example:
        >>> freq, S21 = data.ax('freq').S(2,1)

        Returns the S21 parameter AND the frequency for which it is  known.
        '''

    def howto_ff(self) -> None:
        '''
        >>> topsurf = model.mesh.boundary_surface(model.select.face.near(0, 0, H).tags)
        >>> Ein, Hin = data.item(0).interpolate(*topsurf.exyz).EH
        >>> theta = np.linspace(-np.pi/2, 1.5*np.pi, 201)
        >>> phi = 0*theta
        >>> E, H = fem.physics.edm.stratton_chu(Ein, Hin, topsurf, theta, phi, data.item(0).glob('k0')[0])
        '''
        pass
    def __enter__(self):
        gmsh.initialize()
        gmsh.model.add(self.modelname)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        gmsh.finalize()


   
# class Simulation2D:
#     def __init__(self, modelname: str):
#         self.modelname = modelname
#         self.geo: Geometry = Geometry()
#         self.physics: EMFreqDomain2D = EMFreqDomain2D(self.geo)
#         self.mesh: Mesh2D = None
#         self.resolution: float = 0.2

#     
#     def define_geometry(self, polygons: list[shp.Polygon]) -> None:
#         self.geo._merge_polygons(polygons)
#         self.geo.compile()
#         self.physics._initialize_bcs()

#     
#     def generate_mesh(self, name: str = None, element_order: int = 2) -> Mesh2D:
#         logger.info('Generating mesh')
#         discretizer = self.physics.get_discretizer()
#         logger.info('Updating mesh point disscretization size.')
#         self.geo.update_point_ds(discretizer, self.resolution)
#         logger.info('Calling GMSH mesh routine.')
#         self.geo._commit_gmsh()
#         if name is None:
#             name = self.modelname + '_mesh'
#         logger.info('Generating Mesh2D object')
#         self.mesh = Mesh2D(name, element_order=element_order)
#         logger.info('Mesh complete')
#         return self.mesh
    
#     
#     def update_boundary_conditions(self):
#         logger.info('Updating boundary conditions')
#         for bc in self.physics.boundary_conditions:
#             logger.debug(f'Parsing BC: {bc}')
#             if bc.dimension is BCDimension.EDGE:
#                 vertices = []
#                 for tag in bc.tags:
#                     vertices.append(self.mesh.get_edge(tag))
#                 bc.node_indices = [np.array(_list, dtype=np.int32) for _list in _stitch_lists(vertices)]
#             elif bc.dimension is BCDimension.NODE:
#                 vertices = []
#                 for tag in bc.tags:
#                     vertices.append(self.mesh.get_point(tag))
#                 bc.node_indices = vertices
#             else:
#                 raise NotImplementedError(f'Boundary condition type {bc.dimension} is not implemented yet.')
    
#     def run_frequency_domain(self, solver: callable = None) -> FEMBasis:
#         logger.info('Starting frequency domain simulation routine.')
#         if self.mesh is None:
#             logger.info('No mesh detected.')
#             self.generate_mesh()
#         self.update_boundary_conditions()
#         self.physics.solve(self.mesh, solver=solver)
        
#         return self.physics.solution

#     def run_eigenmodes(self, nsolutions: int = 6) -> FEMBasis:
#         logger.info('Starting eigenmode simulation routine.')
#         if self.mesh is None:
#             logger.info('No mesh detected.')
#             self.generate_mesh()
#         self.update_boundary_conditions()
#         self.physics.eigenmode(self.mesh, num_sols=nsolutions)
#         return self.physics.solution
    
#     def __enter__(self):
#         gmsh.initialize()
#         gmsh.model.add(self.modelname)
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         gmsh.finalize()

