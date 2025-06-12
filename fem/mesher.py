
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

import gmsh
from .material import Material, AIR
from .geo3d import GMSHVolume, GMSHObject, GMSHSurface
from .selection import Selection, FaceSelection
import numpy as np
from typing import Iterable, Callable
from collections import defaultdict
from loguru import logger
from enum import Enum

class MeshError(Exception):
    pass

class Algorithm2D(Enum):
    MESHADAPT = 1
    AUTOMATIC = 2
    INITIAL_MESH_ONLY = 3
    DELAUNAY = 5
    FRONTAL_DELAUNAY = 6
    BAMG = 7
    FRONTAL_DELAUNAY_QUADS = 8
    PACKING_PARALLELOGRAMS = 9
    QUASI_STRUCTURED_QUAD = 11

#(1: Delaunay, 3: Initial mesh only, 4: Frontal, 7: MMG3D, 9: R-tree, 10: HXT)

class Algorithm3D(Enum):
    DELAUNAY = 1
    INITIAL_MESH_ONLY = 3
    FRONTAL = 4
    MMG3D = 7
    RTREE = 9
    HXT = 10

def unpack_lists(_list: list[list], collector: list = None) -> list:
    '''Unpack a recursive list of lists'''
    if collector is None:
        collector = []
    for item in _list:
        if isinstance(item, list):
            unpack_lists(item, collector)
        else:
            collector.append(item)
    
    return collector

class Mesher:

    def __init__(self):
        self.objects: list[GMSHObject] = []
        self.size_definitions: list[tuple[int, float]] = []
        self.mesh_fields: list[int] = []
        self.min_size: float = None
        self.max_size: float = None

    @property
    def edge_tags(self) -> list[int]:
        return [tag[1] for tag in gmsh.model.getEntities(1)]
    
    @property
    def face_tags(self) -> list[int]:
        return [tag[1] for tag in gmsh.model.getEntities(2)]
    
    @property
    def node_tags(self) -> list[int]:
        return [tag[1] for tag in gmsh.model.getEntities(0)]
    
    @property
    def volumes(self) -> list[GMSHVolume]:
        return [obj for obj in self.objects if isinstance(obj, GMSHVolume)]
    
    @property
    def domain_boundary_face_tags(self) -> list[int]:
        '''Get the face tags of the domain boundaries'''
        domain_tags = gmsh.model.getEntities(3)
        tags = gmsh.model.getBoundary(domain_tags, combined=True, oriented=False)
        return [int(tag[1]) for tag in tags]
    
    @property
    def domain_internal_face_tags(self) -> list[int]:
        alltags = self.face_tags
        boundary = self.domain_boundary_face_tags
        return [tag for tag in alltags if tag not in boundary]
    
    def _check_ready(self) -> None:
        if self.max_size is None or self.min_size is None:
            raise MeshError('Either maximum or minimum mesh size is undefined. Make sure \
                            to set the simulation frequency range before calling mesh instructions.')
    
    def submit_objects(self, objects: list[GMSHObject]) -> None:
        """Takes al ist of GMSHObjects and computes the fragment. 

        Args:
            objects (list[GMSHObject]): The set of GMSHObjects
        """
        if not isinstance(objects, list):
            objects = [objects,]

        objects = unpack_lists(objects)
        embeddings = []

        gmsh.model.occ.synchronize()

        final_dimtags = unpack_lists([domain.dimtags for domain in objects])

        dom_mapping = dict()
        for dom in objects:
            embeddings.extend(dom._embeddings)
            for dt in dom.dimtags:
                dom_mapping[dt] = dom
        

        embedding_dimtags = unpack_lists([emb.dimtags for emb in embeddings])
        for dom in objects:
            embedding_dimtags.extend(dom.embeddings)

        tag_mapping: dict[int, dict] = {0: dict(),
                                        1: dict(),
                                        2: dict(),
                                        3: dict()}
        if len(objects) > 1:
            dimtags, output_mapping = gmsh.model.occ.fragment(final_dimtags, embedding_dimtags)
            for domain, mapping in zip(final_dimtags + embedding_dimtags, output_mapping):
                print(domain, mapping)
                tag_mapping[domain[0]][domain[1]] = [o[1] for o in mapping]

            for dom in objects:
                dom.update_tags(tag_mapping)

        else:
            dimtags = final_dimtags
        
        self.objects = objects
        
        gmsh.model.occ.synchronize()

    def set_size_in_domain(self, tags: list[int], max_size: float) -> None:
        """Define the size of the mesh inside a domain

        Args:
            tags (list[int]): The tags of the geometry
            max_size (float): The maximum size (in meters)
        """
        ctag = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.set_numbers(ctag, "VolumesList", tags)
        gmsh.model.mesh.field.set_number(ctag, "VIn", max_size)
        self.mesh_fields.append(ctag)

    def set_mesh_size(self, discretizer: Callable, resolution: float):
        
        dimtags = gmsh.model.occ.get_entities(2)
        for dim, tag in dimtags:
            gmsh.model.mesh.setSizeFromBoundary(2, tag, 0)

        mintag = gmsh.model.mesh.field.add("Min")

        for obj in self.objects:
            if obj._unset_constraints:
                self.unset_constraints(obj.dimtags)

            size = discretizer(obj.material)*resolution*obj.mesh_multiplier
            size = min(size, obj.max_meshsize)
            logger.info(f'Setting mesh size for domain {obj.dim} {obj.tags} to {size}')
            self.set_size_in_domain(obj.tags, size)

        gmsh.model.mesh.field.setNumbers(mintag, "FieldsList", self.mesh_fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(mintag)

        for tag, size in self.size_definitions:
            print('overwriting:', tag, size)
            gmsh.model.mesh.setSize([tag,], size)

    def unset_constraints(self, dimtags: list[tuple[int,int]]):
        '''Unset the mesh constraints for the given dimension tags.'''
        for dimtag in dimtags:
            gmsh.model.mesh.setSizeFromBoundary(dimtag[0], dimtag[1], 0)
            
    def set_boundary_size(self, object: GMSHSurface | FaceSelection, 
                          size:float,
                          growth_distance: float = 5,
                          max_size: float = None):
        """D

        Args:
            dimtags (list[tuple[int,int]]): _description_
            size (float): _description_
            growth_distance (float, optional): _description_. Defaults to 5.
            max_size (float, optional): _description_. Defaults to None.
        """
        dimtags = object.dimtags

        if max_size is None:
            self._check_ready()
            max_size = self.max_size
        
        nodes = gmsh.model.getBoundary(dimtags, combined=False, oriented=False, recursive=False)

        disttag = gmsh.model.mesh.field.add("Distance")

        gmsh.model.mesh.field.setNumbers(disttag, "CurvesList", [n[1] for n in nodes])
        gmsh.model.mesh.field.setNumber(disttag, "Sampling", 100)

        thtag = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thtag, "InField", disttag)
        gmsh.model.mesh.field.setNumber(thtag, "SizeMin", size)
        gmsh.model.mesh.field.setNumber(thtag, "SizeMax", max_size)
        gmsh.model.mesh.field.setNumber(thtag, "DistMin", size)
        gmsh.model.mesh.field.setNumber(thtag, "DistMax", growth_distance*size)
    
        self.mesh_fields.append(thtag)

    def refine_conductor_edge(self, dimtags: list[tuple[int,int]], size):
        nodes = gmsh.model.getBoundary(dimtags, combined=False, recursive=False)

        # for node in nodes:
        #     pcoords = np.linspace(0, 0.5, 10)
        #     gmsh.model.mesh.setSizeAtParametricPoints(node[0], node[1], pcoords, size*np.ones_like(pcoords))
        #     #self.size_definitions.append((node, size))
        # gmsh.model.mesh.setSizeFromBoundary(dimtag[0], dimtag[1], 0)

        tag = gmsh.model.mesh.field.add("Distance")

        #gmsh.model.mesh.field.setNumbers(1, "PointsList", [5])
        gmsh.model.mesh.field.setNumbers(tag, "CurvesList", [n[1] for n in nodes])
        gmsh.model.mesh.field.setNumber(tag, "Sampling", 100)

        # We then define a `Threshold' field, which uses the return value of the
        # `Distance' field 1 in order to define a simple change in element size
        # depending on the computed distances
        #
        # SizeMax -                     /------------------
        #                              /
        #                             /
        #                            /
        # SizeMin -o----------------/
        #          |                |    |
        #        Point         DistMin  DistMax
        thtag = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thtag, "InField", tag)
        gmsh.model.mesh.field.setNumber(thtag, "SizeMin", size)
        gmsh.model.mesh.field.setNumber(thtag, "SizeMax", 100)
        gmsh.model.mesh.field.setNumber(thtag, "DistMin", 0.2*size)
        gmsh.model.mesh.field.setNumber(thtag, "DistMax", 5*size)

        self.mesh_fields.append(thtag)
        

        for dimtag in dimtags:
            gmsh.model.mesh.setSizeFromBoundary(dimtag[0], dimtag[1], 0)

