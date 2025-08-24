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
import gmsh # type: ignore
import numpy as np
from numba import njit, f8 # type: ignore
from .mesher import Mesher
from typing import Union, List, Tuple, Callable, Any
from collections import defaultdict
from .geometry import GeoVolume
from .mth.optimized import outward_normal
from loguru import logger
from functools import cache
from .bc import Periodic
from .mth.pairing import pair_coordinates

@njit(f8(f8[:], f8[:], f8[:]), cache=True, nogil=True)
def area(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray):
    e1 = x2 - x1
    e2 = x3 - x1
    av = np.array([e1[1]*e2[2] - e1[2]*e2[1], e1[2]*e2[0] - e1[0]*e2[2], e1[0]*e2[1] - e1[1]*e2[0]])
    return np.sqrt(av[0]**2 + av[1]**2 + av[2]**2)/2

def shortest_distance(point_cloud):
    """
    Compute the shortest distance between any two points in a 3D point cloud.

    Parameters:
    - point_cloud: np.ndarray of shape (3, N)

    Returns:
    - min_dist: float, the shortest distance
    """
    # Transpose to shape (N, 3)
    points = point_cloud.T  # Shape (N, 3)

    # Compute pairwise squared distances (broadcasting)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # Shape (N, N, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)  # Shape (N, N)

    # Avoid zero on diagonal (distance to self), set to np.inf
    np.fill_diagonal(dist_sq, np.inf)

    # Return minimum distance
    return np.sqrt(np.min(dist_sq))

def tri_ordering(i1: int, i2: int, i3: int) -> int:
    ''' Takes two integer indices of triangle verticces and determines if they are in increasing order or decreasing order.
    It ignores cyclic shifts of the indices. so (4,10,21) == (10,21,4) == (21,4,10)

    for triangle (4,10,20) (for example)
    (4,10,20): In co-ordering of phase = 0: i1 < i2, i2 < i3, i3 > i1: 4-10-20-4 diffs = +6 +10 -16
    (10,20,4): In co-order shift 1: i1 < i2, i2 > i3, i3 < i1: 10-20-4-10 diffs = 10 -16 - (6)
    (20,4,10): In co-order shift 2: i1 > i2, i2 < i3, i3 < i1: 

    For triangle (20,10,4) 
    (20,10,3): i1 > i2, i2 > i3
    (10,3,20): i1 > i2, i2 < i3
    (3,20,10): i1 < i2, i2 > i3
    '''
    return np.sign(np.sign(i2-1) + np.sign(i3-i2) + np.sign(i1-i3))

class Mesh:
    pass

class Mesh3D(Mesh):
    """A Mesh managing all 3D mesh related properties.

    Relevant mesh data such as mappings between nodes(vertices), edges, triangles and tetrahedra
    are managed by the Mesh3D class. Specific information regarding to how actual field values
    are mapped to mesh elements is managed by the FEMBasis class.
    
    """
    def __init__(self, mesher: Mesher):
        
        self.geometry: Mesher = mesher

        # All spatial objects
        self.nodes: np.ndarray = np.array([])
        self.n_i2t: dict = dict()
        self.n_t2i: dict = dict()

        # tets colletions
        self.tets: np.ndarray = np.array([])
        self.tet_i2t: dict = dict()
        self.tet_t2i: dict = dict()
        self.centers: np.ndarray = np.array([])

        # triangles
        self.tris: np.ndarray = np.array([])
        self.tri_i2t: dict = dict()
        self.tri_t2i: dict = dict()
        self.areas: np.ndarray = np.array([])
        self.tri_centers: np.ndarray = np.array([])

        # edges
        self.edges: np.ndarray = np.array([])
        self.edge_i2t: dict = dict()
        self.edge_t2i: dict = dict()
        self.edge_centers: np.ndarray = np.array([])
        self.edge_lengths: np.ndarray = np.array([])
        
        # Inverse mappings
        self.inv_edges: dict = dict()
        self.inv_tris: dict = dict()
        self.inv_tets: dict = dict()

        # Mappings

        self.tet_to_edge: np.ndarray = np.array([])
        self.tet_to_edge_sign: np.ndarray = np.array([])
        self.tet_to_tri: np.ndarray = np.array([])
        self.tri_to_tet: np.ndarray = np.array([])
        self.tri_to_edge: np.ndarray = np.array([])
        self.tri_to_edge_sign: np.ndarray = np.array([])
        self.edge_to_tri: defaultdict | dict = defaultdict()
        self.node_to_edge: defaultdict | dict = defaultdict()

        # Physics mappings

        self.tet_to_field: np.ndarray = np.array([])
        self.edge_to_field: np.ndarray = np.array([])
        self.tri_to_field: np.ndarray = np.array([])

        ## States
        self.defined: bool = False


        ## Memory
        self.ftag_to_tri: dict[int, list[int]] = dict()
        self.ftag_to_node: dict[int, list[int]] = dict()
        self.ftag_to_edge: dict[int, list[int]] = dict()
        self.vtag_to_tet:  dict[int, list[int]] = dict()
        self.etag_to_edge: dict[int, list[int]] = dict()
        
        self.exterior_face_tags: list[int] = []
    
    @property
    def n_edges(self) -> int:
        '''Return the number of edges'''
        return self.edges.shape[1]
    
    @property
    def n_tets(self) -> int:
        '''Return the number of tets'''
        return self.tets.shape[1]
    
    @property
    def n_tris(self) -> int:
        '''Return the number of triangles'''
        return self.tris.shape[1]
    
    @property
    def n_nodes(self) -> int:
        '''Return the number of nodes'''
        return self.nodes.shape[1]

    def get_edge(self, i1: int, i2: int, skip: bool = False) -> int:
        '''Return the edge index given the two node indices'''
        if i1==i2:
            raise ValueError("Edge cannot be formed by the same node.")
        search = (min(int(i1),int(i2)), max(int(i1),int(i2)))
        result =  self.inv_edges.get(search, -10)
        if result == -10 and not skip:
            raise ValueError(f'There is no edge with indices {i1}, {i2}')
        return result
    
    def get_edge_sign(self, i1: int, i2: int) -> int:
        '''Return the edge index given the two node indices'''
        if i1==i2:
            raise ValueError("Edge cannot be formed by the same node.")
        if i1 > i2:
            return -1
        return 1
        
    def get_tri(self, i1, i2, i3) -> int:
        '''Return the triangle index given the three node indices'''
        output = self.inv_tris.get(tuple(sorted((int(i1), int(i2), int(i3)))), None)
        if output is None:
            raise ValueError(f'There is no triangle with indices {i1}, {i2}, {i3}')
        return output
    
    def get_tet(self, i1, i2, i3, i4) -> int:
        '''Return the tetrahedron index given the four node indices'''
        output = self.inv_tets.get(tuple(sorted((int(i1), int(i2), int(i3), int(i4)))), None)
        if output is None:
            raise ValueError(f'There is no tetrahedron with indices {i1}, {i2}, {i3}, {i4}')
        return output
    
    def boundary_triangles(self, dimtags: list[tuple[int, int]] | None = None) -> np.ndarray:
        if dimtags is None:
            outputtags = []
            for tags in self.ftag_to_tri.values():
                outputtags.extend(tags)
            return np.array(outputtags)
        else:
            dts = []
            for dimtag in dimtags:
                if dimtag[0]==2:
                    dts.append(dimtag)
                elif dimtag[0]==3:
                    dts.extend(gmsh.model.get_boundary(dimtags))
            return self.get_triangles([tag[1] for tag in dts])
        

    def get_tetrahedra(self, vol_tags: Union[int, list[int]]) -> np.ndarray:
        if isinstance(vol_tags, int):
            vol_tags = [vol_tags,]
        
        indices = []
        for voltag in vol_tags:
            indices.extend(self.vtag_to_tet[voltag])
        return np.array(indices)
    
    def get_triangles(self, face_tags: Union[int, list[int]]) -> np.ndarray:
        '''Returns a numpyarray of all the triangles that belong to the given face tags'''
        if isinstance(face_tags, int):
            face_tags = [face_tags,]
        indices = []
        for facetag in face_tags:
            indices.extend(self.ftag_to_tri[facetag])
        if any([(i is None) for i in indices]):
            logger.error('Clearing None indices: ', [i for i, ind in enumerate(indices) if ind is None])
            logger.error('This is usually a sign of boundaries sticking out of domains. Please check your Geometry.')
            indices = [i for i in indices if i is not None]

        return np.array(indices)
    
    def domain_edges(self, dimtags: list[tuple[int,int]]) -> np.ndarray:
        """Returns a np.ndarray of all edge indices corresponding to a set of dimension tags.

        Args:
            dimtags (list[tuple[int,int]]): A list of dimtags.

        Returns:
            np.ndarray: The list of mesh edge element indices.
        """
        dimtags_edge = []
        for (d,t) in dimtags:
            if d==1:
                dimtags_edge.append(t)
            if d==2:
                dimtags_edge.extend(gmsh.model.getBoundary([(d,t),], False, False))
            if d==3:
                dts = gmsh.model.getBoundary([(d,t),], False, False)
                dimtags_edge.extend(gmsh.model.getBoundary(dts, False, False))
        
        edge_ids = []
        for tag in dimtags_edge:
            edge_ids.extend(self.etag_to_edge[tag[1]])
        edge_ids = np.array(edge_ids)
        return edge_ids
        
    def get_face_tets(self, *taglist: list[int]) -> np.ndarray:
        ''' Return a list of a tetrahedrons that share a node with any of the nodes in the provided face.'''
        nodes: set = set()
        for tags in taglist:
            nodes.update(self.get_nodes(tags))
        return np.array([i for i, tet in enumerate(self.tets.T) if not set(tet).isdisjoint(nodes)])

    def get_nodes(self, face_tags: Union[int, list[int]]) -> np.ndarray:
        '''Returns a numpyarray of all the nodes that belong to the given face tags'''
        if isinstance(face_tags, int):
            face_tags = [face_tags,]
        
        nodes = []
        for facetag in face_tags:
            nodes.extend(self.ftag_to_node[facetag])
        
        return np.array(sorted(list(set(nodes))))

    def get_edges(self, face_tags: Union[int, list[int]]) -> np.ndarray:
        '''Returns a numpyarray of all the edges that belong to the given face tags'''
        if isinstance(face_tags, int):
            face_tags = [face_tags,]
        
        edges = []
        for facetag in face_tags:
            edges.extend(self.ftag_to_edge[facetag])
        
        return np.array(sorted(list(set(edges))))
    
    
    def update(self, periodic_bcs: list[Periodic] | None = None):
        
        logger.trace('Generating mesh data.')
        if periodic_bcs is None:
            periodic_bcs = []
            
        nodes, lin_coords, _  = gmsh.model.mesh.get_nodes()
        
        coords = lin_coords.reshape(-1, 3).T
        
        ## Vertices
        self.nodes = coords
        self.n_i2t = {i: int(t) for i, t in enumerate(nodes)}
        self.n_t2i = {t: i for i, t in self.n_i2t.items()}
        logger.trace(f'Total of {self.nodes.shape[1]} nodes imported.')
        ## Tetrahedras

        _, tet_tags, tet_node_tags = gmsh.model.mesh.get_elements(3)
        
        # The algorithm assumes that only one domain tag is returned in this function. 
        # Hence the use of tri_node_tags[0] in the next line. If domains are missing.
        # Make sure to combine all the entries in the tri-node-tags list
        
        tet_node_tags = [self.n_t2i[int(t)] for t in tet_node_tags[0]]
        tet_tags = np.squeeze(np.array(tet_tags))

        self.tets = np.array(tet_node_tags).reshape(-1,4).T
        self.tet_i2t = {i: int(t) for i, t in enumerate(tet_tags)}
        self.tet_t2i = {t: i for i, t in self.tet_i2t.items()}
        self.centers = (self.nodes[:,self.tets[0,:]] + self.nodes[:,self.tets[1,:]] + self.nodes[:,self.tets[2,:]] + self.nodes[:,self.tets[3,:]]) / 4
        logger.trace(f'Total of {self.tets.shape[1]} tetrahedra imported.')
        
        # Resort node indices to be sorted on all periodic conditions
        # This sorting makes sure that each edge and triangle on a source face is 
        # sorted in the same order as the corresponding target face triangle or edge.
        # In other words, if a source face triangle or edge index i1, i2, i3 is mapped to j1, j2, j3 respectively
        # Then this ensures that if i1>i2>i3 then j1>j2>j3
    
        for bc in periodic_bcs:
            logger.trace(f'reassigning ordered node numbers for periodic boundary {bc}')
            nodemap, ids1, ids2 = self._derive_node_map(bc)
            nodemap = {int(a): int(b) for a,b in nodemap.items()}
            self.nodes[:,ids2] = self.nodes[:,ids1]
            for itet in range(self.tets.shape[1]):
                self.tets[:,itet] = [nodemap.get(i, i) for i in self.tets[:,itet]]
            self.n_t2i = {t: nodemap.get(i,i) for t,i in self.n_t2i.items()}
            self.n_i2t = {t: i for i, t in self.n_t2i.items()}

        # Extract unique edges and triangles
        edgeset = set()
        triset = set()
        for itet in range(self.tets.shape[1]):
            i1, i2, i3, i4 = sorted([int(ind) for ind in self.tets[:, itet]])
            edgeset.add((i1, i2))
            edgeset.add((i1, i3))
            edgeset.add((i1, i4))
            edgeset.add((i2, i3))
            edgeset.add((i2, i4))
            edgeset.add((i3, i4))
            triset.add((i1,i2,i3))
            triset.add((i1,i2,i4))
            triset.add((i1,i3,i4))
            triset.add((i2,i3,i4))
        logger.trace(f'Total of {len(edgeset)} unique edges and {len(triset)} unique triangles.')
        
        # Edges are effectively Randomly sorted
        # It contains index pairs of vertices edge 1 = (ev1, ev2) etc.
        # Same for triangles
        self.edges = np.array(sorted(list(edgeset))).T
        self.tris = np.array(sorted(list(triset))).T
        self.tri_centers = (self.nodes[:,self.tris[0,:]] + self.nodes[:,self.tris[1,:]] + self.nodes[:,self.tris[2,:]]) / 3
        
        def _hash(ints):
            return tuple(sorted([int(x) for x in ints]))
        
        # Map edge index tuples to edge indices
        # This mapping tells which characteristic index pair (4,3) maps to which edge
        logger.trace('Constructing tet/tri/edge and node mappings.')
        self.inv_edges = {(int(self.edges[0,i]), int(self.edges[1,i])): i for i in range(self.edges.shape[1])}
        self.inv_tris = {_hash((self.tris[0,i], self.tris[1,i], self.tris[2,i])): i for i in range(self.tris.shape[1])}
        self.inv_tets = {_hash((self.tets[0,i], self.tets[1,i], self.tets[2,i], self.tets[3,i])): i for i in range(self.tets.shape[1])}
        
        # Tet links
        self.tet_to_edge = np.zeros((6, self.tets.shape[1]), dtype=int)-99999
        self.tet_to_edge_sign = np.zeros((6, self.tets.shape[1]), dtype=int)-999999
        self.tet_to_tri = np.zeros((4, self.tets.shape[1]), dtype=int)-99999
        self.tet_to_tri_sign = np.zeros((4, self.tets.shape[1]), dtype=int)-999999

        tri_to_tet = defaultdict(list)
        for itet in range(self.tets.shape[1]):
            edge_ids = [self.get_edge(self.tets[i-1,itet],self.tets[j-1,itet]) for i,j in zip([1, 1, 1, 2, 4, 3], [2, 3, 4, 3, 2, 4])]
            id_signs = [self.get_edge_sign(self.tets[i-1,itet],self.tets[j-1,itet]) for i,j in zip([1, 1, 1, 2, 4, 3], [2, 3, 4, 3, 2, 4])]
            self.tet_to_edge[:,itet] = edge_ids
            self.tet_to_edge_sign[:,itet] = id_signs
            self.tet_to_tri[:,itet] = [self.get_tri(self.tets[i-1,itet],self.tets[j-1,itet],self.tets[k-1,itet]) for i,j,k in zip([1, 1, 1, 2], [2, 3, 4, 3], [3, 4, 2, 4])]
            
            
            self.tet_to_tri_sign[0,itet] = tri_ordering(self.tets[0,itet], self.tets[1,itet], self.tets[2,itet])
            self.tet_to_tri_sign[1,itet] = tri_ordering(self.tets[0,itet], self.tets[2,itet], self.tets[3,itet])
            self.tet_to_tri_sign[2,itet] = tri_ordering(self.tets[0,itet], self.tets[3,itet], self.tets[1,itet])
            self.tet_to_tri_sign[3,itet] = tri_ordering(self.tets[1,itet], self.tets[2,itet], self.tets[3,itet])
            
            tri_to_tet[self.tet_to_tri[0, itet]].append(itet)
            tri_to_tet[self.tet_to_tri[1, itet]].append(itet)
            tri_to_tet[self.tet_to_tri[2, itet]].append(itet)
            tri_to_tet[self.tet_to_tri[3, itet]].append(itet)
        
        # Tri links
        self.tri_to_tet = np.zeros((2, self.tris.shape[1]), dtype=int)-1
        for itri in range(self.tris.shape[1]):
            tets = tri_to_tet[itri]
            self.tri_to_tet[:len(tets), itri] = tets
        
        _, tet_tags, tet_node_tags = gmsh.model.mesh.get_elements(2)
        
        # The algorithm assumes that only one domain tag is returned in this function. 
        # Hence the use of tri_node_tags[0] in the next line. If domains are missing.
        # Make sure to combine all the entries in the tri-node-tags list
        tet_node_tags = [self.n_t2i[int(t)] for t in tet_node_tags[0]]
        tet_tags = np.squeeze(np.array(tet_tags))

        self.tri_i2t = {self.get_tri(*self.tris[:,i]): int(t) for i, t in enumerate(tet_tags)}
        self.tri_t2i = {t: i for i, t in self.tri_i2t.items()}

        self.tri_to_edge = np.ndarray((3, self.tris.shape[1]), dtype=int)
        self.tri_to_edge_sign = np.ndarray((3, self.tris.shape[1]), dtype=int)
        self.edge_to_tri = defaultdict(list)

        for itri in range(self.tris.shape[1]):
            i1, i2, i3 = self.tris[:, itri]
            ie1 = self.get_edge(i1,i2)
            ie2 = self.get_edge(i2,i3)
            ie3 = self.get_edge(i1,i3)
            self.tri_to_edge[:,itri] = [ie1, ie2, ie3]
            self.tri_to_edge_sign[:,itri] = [self.get_edge_sign(i1,i2), self.get_edge_sign(i2,i3), self.get_edge_sign(i3,i1)]
            self.edge_to_tri[ie1].append(itri)
            self.edge_to_tri[ie2].append(itri)
            self.edge_to_tri[ie3].append(itri)

        self.node_to_edge = defaultdict(list)
        for eid in range(self.n_edges):
            v1, v2 = self.edges[0, eid], self.edges[1, eid]
            self.node_to_edge[v1].append(eid)
            self.node_to_edge[v2].append(eid)

        self.node_to_edge = {key: sorted(list(set(val))) for key, val in self.node_to_edge.items()}

        ## Quantities
        logger.trace('Computing derived quantaties (centres, areas and lengths).')
        self.edge_centers = (self.nodes[:,self.edges[0,:]] + self.nodes[:,self.edges[1,:]]) / 2
        self.edge_lengths = np.sqrt(np.sum((self.nodes[:,self.edges[0,:]] - self.nodes[:,self.edges[1,:]])**2, axis=0))
        self.areas = np.array([area(self.nodes[:,self.tris[0,i]], self.nodes[:,self.tris[1,i]], self.nodes[:,self.tris[2,i]]) for i in range(self.tris.shape[1])])
        
        ## Edge tag pairings
        _, edge_tags, edge_node_tags = gmsh.model.mesh.get_elements(1)
        edge_tags = np.array(edge_tags).flatten()
        ent = np.array(edge_node_tags).reshape(-1,2).T
        nET = ent.shape[1]
        self.edge_t2i = {int(edge_tags[i]): self.get_edge(self.n_t2i[ent[0,i]], self.n_t2i[ent[1,i]], skip=True) for i in range(nET)}
        self.edge_t2i = {key: value for key,value in self.edge_t2i.items() if value!=-10}
        self.edge_i2t = {i: t for t, i in self.edge_t2i.items()}
        
        edge_dimtags = gmsh.model.get_entities(1)
        for _d, t in edge_dimtags:
            _, edge_tags, node_tags = gmsh.model.mesh.get_elements(1, t)
            if not edge_tags:
                self.etag_to_edge[t] = []
                continue
            self.etag_to_edge[t] = [int(self.edge_t2i.get(tag,None)) for tag in edge_tags[0] if tag in self.edge_t2i]
        
        
        ## Tag bindings
        logger.trace('Constructing geometry to mesh mappings.')
        face_dimtags = gmsh.model.get_entities(2)
        for _d,t in face_dimtags:
            domain_tag, f_tags, node_tags = gmsh.model.mesh.get_elements(2, t)
            node_tags = [self.n_t2i[int(t)] for t in node_tags[0]]
            self.ftag_to_node[t] = node_tags
            node_tags = np.squeeze(np.array(node_tags)).reshape(-1,3).T
            self.ftag_to_tri[t] = [self.get_tri(node_tags[0,i], node_tags[1,i], node_tags[2,i]) for i in range(node_tags.shape[1])]
            self.ftag_to_edge[t] = sorted(list(np.unique(self.tri_to_edge[:,self.ftag_to_tri[t]].flatten())))
        
        vol_dimtags = gmsh.model.get_entities(3)
        for _d,t in vol_dimtags:
            domain_tag, v_tags, node_tags = gmsh.model.mesh.get_elements(3, t)
            node_tags = [self.n_t2i[int(t)] for t in node_tags[0]]
            node_tags = np.squeeze(np.array(node_tags)).reshape(-1,4).T
            self.vtag_to_tet[t] = [self.get_tet(node_tags[0,i], node_tags[1,i], node_tags[2,i], node_tags[3,i]) for i in range(node_tags.shape[1])]

        self.defined = True
        logger.trace('Done analyzing mesh.')


    ## Higher order functions

    def _derive_node_map(self, bc: Periodic) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
        """Computes an old to new node index mapping that preserves global sorting

        Since basis function field direction is based on the order of indices in tetrahedron
        for periodic boundaries it is important that all triangles and edges in each source
        face are in the same order as the target face. This method computes the mapping for the
        secondary face nodes

        Args:
            bc (Periodic): The Periodic boundary condition

        Returns:
            tuple[dict[int, int], np.ndarray, np.ndarray]: The node index mapping and the node index arrays
        """

        node_ids_1 = []
        node_ids_2 = []

        face_dimtags = gmsh.model.get_entities(2)
        
        for d,t in face_dimtags:
            domain_tag, f_tags, node_tags = gmsh.model.mesh.get_elements(2, t)
            node_tags = [self.n_t2i[int(t)] for t in node_tags[0]]
            if t in bc.face1.tags:
                node_ids_1.extend(node_tags)
            if t in bc.face2.tags:
                node_ids_2.extend(node_tags)


        node_ids_1 = sorted(list(set(node_ids_1)))
        node_ids_2 = sorted(list(set(node_ids_2)))
        
        all_node_ids = np.unique(np.array(node_ids_1 + node_ids_2))
        dsmin = shortest_distance(self.nodes[:,all_node_ids])

        node_ids_1_arry = np.array(node_ids_1)
        node_ids_2_arry = np.array(node_ids_2)
        dv = np.array(bc.dv)
        
        nodemap = pair_coordinates(self.nodes, node_ids_1_arry, node_ids_2_arry, dv, dsmin/2)
        node_ids_2_unsorted = [nodemap[i] for i in sorted(node_ids_1)]
        node_ids_2_sorted = sorted(node_ids_2_unsorted)
        conv_map = {i1: i2 for i1, i2 in zip(node_ids_2_unsorted, node_ids_2_sorted)}

        return conv_map, np.array(node_ids_2_unsorted), np.array(node_ids_2_sorted)


    def retreive(self, material_selector: Callable, volumes: list[GeoVolume]) -> np.ndarray:
        '''Retrieve the material properties of the geometry'''
        arry = np.zeros((3,3,self.n_tets,), dtype=np.complex128)

        xs = self.centers[0,:]
        ys = self.centers[1,:]
        zs = self.centers[2,:]
        
        for volume in sorted(volumes, key=lambda x: x._priority):
        
            for dimtag in volume.dimtags:
                etype, etag_list, ntags = gmsh.model.mesh.get_elements(*dimtag)
                for etags in etag_list:
                    tet_ids = [self.tet_t2i[t] for t in etags]

                    value = material_selector(volume.material, xs[tet_ids], ys[tet_ids], zs[tet_ids])
                    arry[:,:,tet_ids] = value
        return arry
    
    def plot_gmsh(self) -> None:
        gmsh.fltk.run()

    def find_edge_groups(self, edge_ids: np.ndarray) -> list[tuple[Any,...]]:
        """
        Find the groups of edges in the mesh.

        Split an edge list into sets (islands) whose vertices are mutually connected.

        Parameters
        ----------
        edges : np.ndarray, shape (2, N)
            edges[0, i] and edges[1, i] are the two vertex indices of edge *i*.
            The array may contain any (hashable) integer vertex labels, in any order.

        Returns
        -------
        List[Tuple[int, ...]]
            A list whose *k*‑th element is a `tuple` with the (zero‑based) **edge IDs**
            that belong to the *k*‑th connected component.  Ordering is:
            • components appear in the order in which their first edge is met,  
            • edge IDs inside each tuple are sorted increasingly.

        Notes
        -----
        * Only the connectivity of the supplied edges is considered.  
        In particular, vertices that never occur in `edges` do **not** create extra
        components.
        * Runtime is *O*(N + V), with N = number of edges, V = number of
        distinct vertices.  No external libraries are needed.
        """
        edges = self.edges[:,edge_ids]
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("`edges` must have shape (2, N)")

        #n_edges: int = edges.shape[1]

        # --- build “vertex ⇒ incident edge IDs” map ------------------------------
        vert2edges = defaultdict(list)
        for eid in edge_ids:
            v1, v2 = self.edges[0, eid], self.edges[1, eid]
            vert2edges[v1].append(eid)
            vert2edges[v2].append(eid)
        
        groups = []

        ungrouped = set(edge_ids)

        group = [edge_ids[0],]
        ungrouped.remove(edge_ids[0])

        while True:
            new_edges = set()
            for edge in group:
                v1, v2 = self.edges[0, edge], self.edges[1, edge]
                new_edges.update(set(vert2edges[v1]))
                new_edges.update(set(vert2edges[v2]))

            new_edges = new_edges.intersection(ungrouped)
            if len(new_edges) == 0:
                groups.append(tuple(sorted(group)))
                if len(ungrouped) == 0:
                    break
                group = [ungrouped.pop(),]
            else:
                group += list(new_edges)
                ungrouped.difference_update(new_edges)

        return groups

    def boundary_surface(self, 
                         face_tags: Union[int, list[int]], 
                         origin: tuple[float, float, float] | None = None) -> SurfaceMesh:
        """Returns a SurfaceMesh class that is a 2D mesh isolated from the 3D mesh

        The mesh will be based on the given set of face tags.

        In order to properly allign the normal vectors, an alignment origin can be provided.
        If not provided, the center point of all boundaries will be used.

        Args:
            face_tags (Union[int, list[int]]): The list of face tags to use
            origin (tuple[float, float, float], optional): The normal vecor alignment origin.. Defaults to None.

        Returns:
            SurfaceMesh: The resultant surface mesh
        """
        tri_ids = self.get_triangles(face_tags)
        if origin is None:
            nodes = self.nodes[:,self.get_nodes(face_tags)]
            x0 = float(np.mean(nodes[0,:]))
            y0 = float(np.mean(nodes[1,:]))
            z0 = float(np.mean(nodes[2,:]))
            origin = (x0, y0, z0)

        return SurfaceMesh(self, tri_ids, origin)
    
class SurfaceMesh(Mesh):

    def __init__(self,
                 original: Mesh3D,
                 tri_ids: np.ndarray,
                 origin: tuple[float, float, float]):
        
        ## Compute derived mesh properties
        tris = original.tris[:, tri_ids]
        unique_nodes = np.sort(np.unique(tris.flatten()))
        new_ids = np.arange(unique_nodes.shape[0])
        old_to_new_node_id_map = {a: b for a,b in zip(unique_nodes, new_ids)}
        new_tris = np.array([[old_to_new_node_id_map[tris[i,j]] for i in range(3)] for j in range(tris.shape[1])]).T


        ### Store information
        self._tri_ids = tri_ids
        self._origin = origin

        self.original_tris: np.ndarray = original.tris

        self.old_new_node_map: dict = old_to_new_node_id_map
        self.original: Mesh3D = original
        self._alignment_origin: np.ndarray = np.array(origin).astype(np.float64)
        self.nodes: np.ndarray = original.nodes[:, unique_nodes]
        self.tris: np.ndarray = new_tris

        ## initialize derived
        self.edge_centers: np.ndarray = np.array([])
        self.edge_tris: np.ndarray = np.array([])
        self.n_nodes = self.nodes.shape[1]
        self.n_tris = self.tris.shape[1]
        self.n_edges: float = -1
        self.areas: np.ndarray = np.array([])
        self.normals: np.ndarray = np.array([])
        self.tri_to_edge: np.ndarray = np.array([])
        self.edge_to_tri: dict | defaultdict = dict()
        # Generate derived
        self.update()

    def copy(self) -> SurfaceMesh:
        return SurfaceMesh(self.original, self._tri_ids, self._origin)
    
    def flip(self, ax: str) -> SurfaceMesh:
        if ax.lower()=='x':
            self.flipX()
        if ax.lower()=='y':
            self.flipY()
        if ax.lower()=='z':
            self.flipZ()
        return self
        #self.tris[(0,1),:] = self.tris[(1,0),:]

    def flipX(self) -> SurfaceMesh:
        self.nodes[0,:] = -self.nodes[0,:]
        self.normals[0,:] = -self.normals[0,:]
        self.edge_centers[0,:] = -self.edge_centers[0,:]
        return self

    def flipY(self) -> SurfaceMesh:
        self.nodes[1,:] = -self.nodes[1,:]
        self.normals[1,:] = -self.normals[1,:]
        self.edge_centers[1,:] = -self.edge_centers[1,:]
        return self
    
    def flipZ(self) -> SurfaceMesh:
        self.nodes[2,:] = -self.nodes[2,:]
        self.normals[2,:] = -self.normals[2,:]
        self.edge_centers[2,:] = -self.edge_centers[2,:]
        return self

    def from_source_tri(self, triid: int) -> int | None:
        ''' Returns a triangle index from the old mesh to the new mesh.'''
        i1in = self.original.tris[0,triid]
        i2in = self.original.tris[1,triid]
        i3in = self.original.tris[2,triid]
        i1 = self.old_new_node_map.get(i1in,None)
        i2 = self.old_new_node_map.get(i2in,None)
        i3 = self.old_new_node_map.get(i3in,None)
        if i1 is None or i2 is None or i3 is None:
            return None
        return self.get_tri(i1, i2, i3)
    
    def from_source_edge(self, edgeid: int) -> int | None:
        ''' Returns an edge index form the old mesh to the new mesh.'''
        i1 = self.old_new_node_map.get(self.original.edges[0,edgeid],None)
        i2 = self.old_new_node_map.get(self.original.edges[1,edgeid],None)
        if i1 is None or i2 is None:
            return None
        return self.get_edge(i1, i2)
    
    def get_edge(self, i1: int, i2: int) -> int:
        '''Return the edge index given the two node indices'''
        if i1==i2:
            raise ValueError("Edge cannot be formed by the same node.")
        search = (min(int(i1),int(i2)), max(int(i1),int(i2)))
        result =  self.inv_edges.get(search, None)
        if result is None:
            raise ValueError(f'There is no edge with indices {i1}, {i2}')
        return result
    
    def get_edge_sign(self, i1: int, i2: int) -> int:
        '''Return the edge index given the two node indices'''
        if i1==i2:
            raise ValueError("Edge cannot be formed by the same node.")
        if i1 > i2:
            return -1
        return 1
        
    def get_tri(self, i1, i2, i3) -> int:
        '''Return the triangle index given the three node indices'''
        result = self.inv_tris.get(tuple(sorted((int(i1), int(i2), int(i3)))), None)
        if result is None:
            raise ValueError(f'There is no triangle with indices {i1}, {i2}, {i3}')
        return result

    
    def update(self) -> None:
        ## First Edges

        edges = set()
        for i in range(self.n_tris):
            i1, i2, i3 = self.tris[:,i]
            edges.add((i1, i2))
            edges.add((i2, i3))
            edges.add((i1, i3))

        edgelist = list(edges)

        self.edges = np.array(edgelist).T
        self.n_edges = self.edges.shape[1]
        self.edge_centers = (self.nodes[:,self.edges[0,:]] + self.nodes[:,self.edges[1,:]])/2

        ## Mapping from edge pairs to edge index
         
        def _hash(ints):
            return tuple(sorted([int(x) for x in ints]))
        
        self.inv_edges = {(int(self.edges[0,i]), int(self.edges[1,i])): i for i in range(self.edges.shape[1])}
        self.inv_tris = {_hash((self.tris[0,i], self.tris[1,i], self.tris[2,i])): i for i in range(self.tris.shape[1])}
        ##
        origin = self._alignment_origin

        self.areas = np.array([area(self.nodes[:,self.tris[0,i]], 
                                    self.nodes[:,self.tris[1,i]], 
                                    self.nodes[:,self.tris[2,i]]) for i in range(self.n_tris)]).T
        self.normals = np.array([outward_normal(
                                    self.nodes[:,self.tris[0,i]], 
                                    self.nodes[:,self.tris[1,i]], 
                                    self.nodes[:,self.tris[2,i]], 
                                    origin) for i in range(self.n_tris)]).T
        
        self.tri_to_edge = np.ndarray((3, self.tris.shape[1]), dtype=int)
        self.edge_to_tri = defaultdict(list)

        for itri in range(self.tris.shape[1]):
            i1, i2, i3 = self.tris[:, itri]
            ie1 = self.get_edge(i1,i2)
            ie2 = self.get_edge(i2,i3)
            ie3 = self.get_edge(i1,i3)
            self.tri_to_edge[:,itri] = [ie1, ie2, ie3]
        
    @property
    def exyz(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.edge_centers[0,:], self.edge_centers[1,:], self.edge_centers[2,:]
    