
import gmsh
from .material import Material, AIR
from .geo3d import GMSHVolume, GMSHObject, GMSHSurface
import numpy as np
from typing import Iterable, Callable
from collections import defaultdict
from loguru import logger
from enum import Enum

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
        
    def submit_objects(self, objects: list[GMSHObject]) -> None:

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
        if len(objects) > 1:
            
            dimtags, output_mapping = gmsh.model.occ.fragment(final_dimtags, embedding_dimtags)
            for domain, mapping in zip(final_dimtags, output_mapping):
                dom_mapping[domain].update_tags([dt[1] for dt in mapping])
        else:
            dimtags = final_dimtags
        
        self.objects = objects
        
        gmsh.model.occ.synchronize()

    def set_mesh_size(self, discretizer: Callable, resolution: float):
        
        mintag = gmsh.model.mesh.field.add("Min")

        gmsh.model.mesh.field.setNumbers(mintag, "FieldsList", self.mesh_fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(mintag)

        for obj in self.objects:
            if obj._unset_constraints:
                self.unset_constraints(obj.dimtags)

            size = discretizer(obj.material)*resolution*obj.mesh_multiplier
            size = min(size, obj.max_meshsize)
            logger.info(f'Setting mesh size for domain {obj.dim} {obj.tags} to {size}')
            
            gmsh.model.mesh.setSize(gmsh.model.getBoundary(obj.dimtags, recursive=True), size)
            
        for tag, size in self.size_definitions:
            print('overwriting:', tag, size)
            gmsh.model.mesh.setSize([tag,], size)

    def unset_constraints(self, dimtags: list[tuple[int,int]]):
        '''Unset the mesh constraints for the given dimension tags.'''
        for dimtag in dimtags:
            gmsh.model.mesh.setSizeFromBoundary(dimtag[0], dimtag[1], 0)
            
    def set_boundary_size(self, dimtags: list[tuple[int,int]], 
                          size:float, 
                          max_size: float, 
                          edge_only: bool = False,
                          growth_distance: float = 3):
        
        nodes = gmsh.model.getBoundary(dimtags, combined=False, oriented=False, recursive=False)

        print(f'Setting size for {dimtags} yielding nodes:', nodes)

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

        if not edge_only:
            return
        
        for dimtag in dimtags:
            gmsh.model.mesh.setSizeFromBoundary(dimtag[0], dimtag[1], 0)

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

