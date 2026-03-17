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
from __future__ import annotations
import gmsh # type: ignore
import numpy as np
from typing import Union, List, Tuple, Callable, Any
from collections import defaultdict
from loguru import logger
from .bc import Periodic
from .logsettings import DEBUG_COLLECTOR
from emsutil import Saveable

_MISSING_ID: int = -1234

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


def _map_array_by_dict(arr: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Convertes a numpy array of integers by a dictionary from int to int

    Args:
        arr (np.ndarray): _description_
        mapping (dict[int, int]): _description_

    Returns:
        np.ndarray: _description_
    """
    keys   = np.array(list(mapping.keys()))
    vals   = np.array(list(mapping.values()))
    sort_idx = np.argsort(keys)
    keys, vals = keys[sort_idx], vals[sort_idx]

    idx    = np.searchsorted(keys, arr)
    return vals[idx]

class Mesh:
    pass

class Mesh3D(Mesh, Saveable):
    """A Mesh managing all 3D mesh related properties.

    Relevant mesh data such as mappings between nodes(vertices), edges, triangles and tetrahedra
    are managed by the Mesh3D class. Specific information regarding to how actual field values
    are mapped to mesh elements is managed by the FEMBasis class.
    
    """
    def __init__(self):

        
        
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
        self.tet_to_tri: np.ndarray = np.array([])
        self.tri_to_tet: np.ndarray = np.array([])
        self.tri_to_edge: np.ndarray = np.array([])
        self.edge_to_tri: defaultdict | dict = defaultdict()
        self.node_to_edge: defaultdict | dict = defaultdict()

        # Physics mappings
        self.tet_to_field: np.ndarray = np.array([])
        self.edge_to_field: np.ndarray = np.array([])
        self.tri_to_field: np.ndarray = np.array([])

        ## States
        self.defined: bool = False
        self._quick_mesh: bool = False

        ## Memory
        self.ftag_to_tri: dict[int, list[int]] = dict()
        self.ftag_to_node: dict[int, list[int]] = dict()
        self.ftag_to_edge: dict[int, list[int]] = dict()
        self.vtag_to_tet:  dict[int, list[int]] = dict()
        self.etag_to_edge: dict[int, list[int]] = dict()
        
        ## Dervied
        self.dimtag_to_center: dict[tuple[int, int], tuple[float, float, float]] = dict()
        self.dimtag_to_edges: dict[tuple[int, int], np.ndarray] = dict()
        self.dimtag_to_nodes: dict[tuple[int, int], np.ndarray] = dict()
        self.dimtag_to_bb: dict[tuple[int, int], np.ndarray] = dict()
        self.ftag_to_normal: dict[int, np.ndarray] = dict()
        self.ftag_to_point: dict[int, np.ndarray] = dict()
        
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
        result =  self.inv_edges.get(search, _MISSING_ID)
        if result == _MISSING_ID and not skip:
            raise ValueError(f'There is no edge with indices {i1}, {i2}')
        return result
        
    def get_tri(self, i1, i2, i3) -> int:
        '''Return the triangle index given the three node indices'''
        i11, i21, i31 = tuple(sorted((int(i1), int(i2), int(i3))))
        output = self.inv_tris.get(tuple(sorted((int(i1), int(i2), int(i3)))), None)
        if output is None:
            DEBUG_COLLECTOR.add_report(f'Mesh3D: The program is crashed due to a non existing triangle {i11}, {i21}, {i31}. This occurs often if surfaces stick out of the 3D domain.\n' + 
                                       'Only 3D volumes can be meshed. Parts or entire simulations that are two dimensional will cause this problem.')
            raise ValueError(f'There is no triangle with indices {i11}, {i21}, {i31}')
        return output
    
    def get_tet(self, i1, i2, i3, i4) -> int:
        '''Return the tetrahedron index given the four node indices'''
        output = self.inv_tets.get(tuple(sorted((int(i1), int(i2), int(i3), int(i4)))), None)
        if output is None:
            raise ValueError(f'There is no tetrahedron with indices {i1}, {i2}, {i3}, {i4}')
        return output
        
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
    
    def _domain_edge(self, dimtag: tuple[int,int]) -> np.ndarray:
        """Returns a np.ndarray of all edge indices corresponding to a set of dimension tags.

        Args:
            dimtags (list[tuple[int,int]]): A list of dimtags.

        Returns:
            np.ndarray: The list of mesh edge element indices.
        """
        dimtags_edge = []
        d,t = dimtag
        if d==0:
            return np.ndarray([], dtype=np.int64)
        if d==1:
            dimtags_edge.append((1,t))
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
    
    def domain_edges(self, dimtags: list[tuple[int,int]]) -> np.ndarray:
        """Returns a np.ndarray of all edge indices corresponding to a set of dimension tags.

        Args:
            dimtags (list[tuple[int,int]]): A list of dimtags.

        Returns:
            np.ndarray: The list of mesh edge element indices.
        """
        
        edge_ids = []
        for dt in dimtags:
            edge_ids.extend(self.dimtag_to_edges[dt])
        edge_ids = np.array(edge_ids)
        return edge_ids
    
    def get_face_tets(self, *taglist: list[int]) -> np.ndarray:
        ''' Return a list of a tetrahedrons that share a node with any of the nodes in the provided face.'''
        tritags = []
        for tags in taglist:
            for tag in tags:
                tritags.extend(self.ftag_to_tri[tag])
        tettags = np.unique(self.tri_to_tet[:, tritags].flatten())
        tettags = tettags[tettags != _MISSING_ID]
        return np.sort(tettags)


    def _get_dimtags(self, nodes: list[int] | None = None, edges: list[int] | None = None) -> list[tuple[int, int]]:
        """Returns the geometry dimtags associated with a set of nodes and edges"""
        if nodes is None:
            nodes = []
        if edges is None:
            edges = []
        nodes = set(nodes)
        edges = set(edges)
        dimtags = []
        
        # Test faces
        for tag, f_nodes in self.ftag_to_node.items():
            if set(f_nodes).isdisjoint(nodes):
                continue
            dimtags.append((2,tag))
        
        for tag, f_edges in self.ftag_to_edge.items():
            if set(f_edges).isdisjoint(edges):
                continue
            dimtags.append((2,tag))
        
        # test volumes
        for tag, f_tets in self.vtag_to_tet.items():
            v_nodes = set(self.tets[:,f_tets].flatten())
            if not v_nodes.isdisjoint(nodes):
                dimtags.append((3,tag))
            v_edges = set(self.tet_to_edge[:,f_tets].flatten())
            if not v_edges.isdisjoint(edges):
                dimtags.append((3,tag))
        return sorted(dimtags)
    
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
    
    
    def _pre_update(self, periodic_bcs: list[Periodic] | None = None):
        """Builds the mesh data properties

        Args:
            periodic_bcs (list[Periodic] | None, optional): A list of periodic boundary conditions. Defaults to None.

        Returns:
            None: None
        """

        # NEW VERSION
        logger.info('Generating internal mesh data.')
        if periodic_bcs is None:
            periodic_bcs = []
        
        # -----------------------------------------------------------------------------
        # Pull all mesh information out of GMSH
        # -----------------------------------------------------------------------------

        nodes, lin_coords, _  = gmsh.model.mesh.get_nodes()
        point_dimtags = gmsh.model.get_entities(0)
        edge_dimtags = gmsh.model.get_entities(1)
        face_dimtags = gmsh.model.get_entities(2)
        vol_dimtags = gmsh.model.get_entities(3)
        entity_set = {
            0: point_dimtags,
            1: edge_dimtags,
            2: face_dimtags,
            3: vol_dimtags,
        }
        _, edge_tags, edge_node_tags = gmsh.model.mesh.get_elements(1)
        _, tri_tags, tri_node_tags = gmsh.model.mesh.get_elements(2)
        _, tet_tags, tet_node_tags = gmsh.model.mesh.get_elements(3)
        _edge_entity_map = {t: gmsh.model.mesh.get_elements(1, t)[1] for d,t in gmsh.model.get_entities(1)}
        _face_entity_map = {t: gmsh.model.mesh.get_elements(2, t)[2] for d,t in gmsh.model.get_entities(2)}
        _vol_entity_map = {t: gmsh.model.mesh.get_elements(3, t)[2] for d,t in gmsh.model.get_entities(3)}

        for dim, dts in entity_set.items():
            self.dimtag_to_center.update({dt: gmsh.model.occ.get_center_of_mass(*dt) for dt in dts})
            self.dimtag_to_bb.update({dt: np.array(gmsh.model.occ.get_bounding_box(*dt)) for dt in dts})

        # -----------------------------------------------------------------------------
        # Start of Processing
        # -----------------------------------------------------------------------------
        
        # -----------------------------------------------------------------------------
        # Vertices
        # -----------------------------------------------------------------------------
        self.nodes = lin_coords.reshape(-1, 3).T
        self.n_i2t = {i: int(t) for i, t in enumerate(nodes)}
        self.n_t2i = {t: i for i, t in self.n_i2t.items()}

        logger.trace(f'Total of {self.nodes.shape[1]} nodes imported.')
        # -----------------------------------------------------------------------------
        # Tetrahedra
        # -----------------------------------------------------------------------------
        
        # The algorithm assumes that only one domain tag is returned in this function. 
        # Hence the use of tri_node_tags[0] in the next line. If domains are missing.
        # Make sure to combine all the entries in the tri-node-tags list
        
        if not tet_node_tags:
            DEBUG_COLLECTOR.add_report('Only 3D volumes can be meshed. Parts or entire simulations that are two dimensional will cause this problem.')
            raise ValueError('Not a 3D volume')

        tet_node_tags = [self.n_t2i[int(t)] for t in tet_node_tags[0]]
        tet_tags = np.squeeze(np.array(tet_tags))

        self.tets = np.array(tet_node_tags).reshape(-1,4).T
        self.tet_i2t = {i: int(t) for i, t in enumerate(tet_tags)}
        self.tet_t2i = {t: i for i, t in self.tet_i2t.items()}
        self.centers = (self.nodes[:,self.tets[0,:]] + self.nodes[:,self.tets[1,:]] + self.nodes[:,self.tets[2,:]] + self.nodes[:,self.tets[3,:]]) / 4
        logger.trace(f'Total of {self.tets.shape[1]} tetrahedra imported.')
        
        # -----------------------------------------------------------------------------
        # Resorting nodes to keep Periodic boundaries consistent
        # -----------------------------------------------------------------------------

        # Resort node indices to be sorted on all periodic conditions
        # This sorting makes sure that each edge and triangle on a source face is 
        # sorted in the same order as the corresponding target face triangle or edge.
        # In other words, if a source face triangle or edge index i1, i2, i3 is mapped to j1, j2, j3 respectively
        # Then this ensures that if i1>i2>i3 then j1>j2>j3

        for bc in periodic_bcs:
            logger.trace(f'reassigning ordered node numbers for periodic boundary {bc}')
            nodemap, ids1, ids2 = self._pre_derive_node_map(bc)
            nodemap = {int(a): int(b) for a,b in nodemap.items()}
            self.nodes[:,ids2] = self.nodes[:,ids1]
            for itet in range(self.tets.shape[1]):
                self.tets[:,itet] = [nodemap.get(i, i) for i in self.tets[:,itet]]
            self.n_t2i = {t: nodemap.get(i,i) for t,i in self.n_t2i.items()}
            self.n_i2t = {t: i for i, t in self.n_t2i.items()}

        # -----------------------------------------------------------------------------
        # Extracting Edges and Triangles
        # -----------------------------------------------------------------------------

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
        
        self.edges = np.array(sorted(list(edgeset))).T
        self.tris = np.array(sorted(list(triset))).T

        # -----------------------------------------------------------------------------
        # Creating inverse mappings from edge/tri/tet id-tuples to integers
        # -----------------------------------------------------------------------------
        
        # Map edge index tuples to edge indices
        # This mapping tells which characteristic index pair (4,3) maps to which edge
        logger.trace('Constructing tet/tri/edge and node mappings.')
        nN = self.nodes.shape[1]
        nE = self.edges.shape[1]
        nR = self.tris.shape[1]
        nT = self.tets.shape[1]

        _sorted_edges = self.edges
        _sorted_tris = np.sort(self.tris, axis=0)
        _sorted_tets = np.sort(self.tets, axis=0)

        self.inv_edges = {tuple(_sorted_edges[:,i].tolist()): i for i in range(nE)}
        self.inv_tris = {tuple(_sorted_tris[:,i].tolist()): i for i in range(nR)}
        self.inv_tets = {tuple(_sorted_tets[:,i].tolist()): i for i in range(nT)}
        
        # -----------------------------------------------------------------------------
        # Constructucting Tet -> Edge/Tri mapping
        # -----------------------------------------------------------------------------
        self.tet_to_edge = np.zeros((6, self.tets.shape[1]), dtype=int) + _MISSING_ID
        self.tet_to_tri = np.zeros((4, self.tets.shape[1]), dtype=int) + _MISSING_ID

        _idset1 = ((1,2),(1,3),(1,4),(2,3),(4,2),(3,4))
        _idset2 = ((1,2,3),(1,3,4),(1,4,2),(2,3,4))

        for itet in range(self.tets.shape[1]):
            t = self.tets[:, itet]
            self.tet_to_edge[:,itet] = [self.inv_edges[tuple(sorted((int(t[i-1]), int(t[j-1]))))] for i,j in _idset1]
            self.tet_to_tri[:,itet]  = [self.inv_tris[tuple(sorted((int(t[i-1]), int(t[j-1]), int(t[k-1]))))] for i,j,k in _idset2]
        
        # -----------------------------------------------------------------------------
        # Constructing Tri -> Tet mapping
        # -----------------------------------------------------------------------------

        n_tets = self.tets.shape[1]
        n_tris = self.tris.shape[1]
        flat_tris = self.tet_to_tri.ravel(order='F')
        flat_tets = np.repeat(np.arange(n_tets, dtype=int), 4)
        sort_idx    = np.argsort(flat_tris, stable=True)
        sorted_tris = flat_tris[sort_idx]
        sorted_tets = flat_tets[sort_idx]
        self.tri_to_tet = np.full((2, n_tris), _MISSING_ID, dtype=int)
        _, first_idx, counts = np.unique(sorted_tris, return_index=True, return_counts=True)
        self.tri_to_tet[0, sorted_tris[first_idx]] = sorted_tets[first_idx]
        mask2 = counts == 2
        self.tri_to_tet[1, sorted_tris[first_idx[mask2]]] = sorted_tets[first_idx[mask2] + 1]
        
        # -----------------------------------------------------------------------------
        # Triangle GMSH Tag to Index Mapping
        # -----------------------------------------------------------------------------

        # The algorithm assumes that only one domain tag is returned in this function. 
        # Hence the use of tri_node_tags[0] in the next line. If domains are missing.
        # Make sure to combine all the entries in the tri-node-tags list
        # assuming only one element type here (tri3)
        
        # tri tags are all labels for all triangular elements in the Mesh

        tri_tags = np.array(tri_tags[0], dtype=int)
        tri_nodes = np.array([self.n_t2i[int(t)] for t in tri_node_tags[0]], dtype=int).reshape(-1, 3)

        self.tri_i2t = {}
        _sorted_tri_nodes = np.sort(tri_nodes, axis=1)

        for k, tag in enumerate(tri_tags):
            key = tuple(_sorted_tri_nodes[k,:].tolist())
            if key not in self.inv_tris:
                DEBUG_COLLECTOR.add_report(f'Mesh3D: The program is crashed due to a non existing triangle {key}. This occurs often if surfaces stick out of the 3D domain.\n')
                raise ValueError(f'There is no triangle with indices {key}')
            tri_id = self.inv_tris[key]
            self.tri_i2t[tri_id] = int(tag)

        self.tri_t2i = {t: i for i, t in self.tri_i2t.items()}

        # -----------------------------------------------------------------------------
        # Triangle -> Edge and Edge -> Tri
        # -----------------------------------------------------------------------------

        _edge_lookup_mat = np.full((nN, nN), _MISSING_ID, dtype=np.int32)
        _edge_lookup_mat[self.edges[0], self.edges[1]] = np.arange(nE)
        _edge_lookup_mat[self.edges[1], self.edges[0]] = np.arange(nE)

        self.tri_to_edge = np.zeros((3, self.tris.shape[1]), dtype=int)
        self.tri_to_edge[0] = _edge_lookup_mat[_sorted_tris[0], _sorted_tris[1]]
        self.tri_to_edge[1] = _edge_lookup_mat[_sorted_tris[1], _sorted_tris[2]]
        self.tri_to_edge[2] = _edge_lookup_mat[_sorted_tris[0], _sorted_tris[2]]
        
        
        flat_edges = self.tri_to_edge.ravel(order='F')
        flat_tris  = np.repeat(np.arange(nR, dtype=int), 3)
        sort_idx     = np.argsort(flat_edges, stable=True)
        sorted_edges = flat_edges[sort_idx]
        sorted_tris_e  = flat_tris[sort_idx]
        _, first_idx, counts = np.unique(sorted_edges, return_index=True, return_counts=True)
        max_count = int(counts.max())
        self.edge_to_tri = np.full((max_count, nE), _MISSING_ID, dtype=int)
        for k in range(max_count):
            mask = counts > k
            self.edge_to_tri[k, sorted_edges[first_idx[mask]]] = sorted_tris_e[first_idx[mask] + k]
        
        # -----------------------------------------------------------------------------
        # Node to Edge mapping
        # -----------------------------------------------------------------------------

        flat_nodes = self.edges.ravel(order='F')
        flat_edges = np.repeat(np.arange(self.edges.shape[1], dtype=int), 2)

        sort_idx     = np.argsort(flat_nodes, stable=True)
        sorted_nodes = flat_nodes[sort_idx]
        sorted_edges = flat_edges[sort_idx]

        _, first_idx, counts = np.unique(sorted_nodes, return_index=True, return_counts=True)
        max_count = int(counts.max())

        self.node_to_edge = np.full((max_count, self.nodes.shape[1]), _MISSING_ID, dtype=int)
        for k in range(max_count):
            mask = counts > k
            self.node_to_edge[k, sorted_nodes[first_idx[mask]]] = sorted_edges[first_idx[mask] + k]
        
        # -----------------------------------------------------------------------------
        # Mesh Quantities
        # -----------------------------------------------------------------------------

        logger.trace('Computing derived quantaties (centres, areas and lengths).')
        # TRIANGLE CENTERS
        self.tri_centers = (self.nodes[:,self.tris[0,:]] + self.nodes[:,self.tris[1,:]] + self.nodes[:,self.tris[2,:]]) / 3
        
        # EDGE CENTERS
        self.edge_centers = (self.nodes[:,self.edges[0,:]] + self.nodes[:,self.edges[1,:]]) / 2
        
        # EDGE LENGTHS
        self.edge_lengths = np.sqrt(np.sum((self.nodes[:,self.edges[0,:]] - self.nodes[:,self.edges[1,:]])**2, axis=0))

        # TRIANGLE AREAS
        Nxyz1 = self.nodes[:, self.tris[0,:]]
        Nxyz2 = self.nodes[:, self.tris[1,:]]
        Nxyz3 = self.nodes[:, self.tris[2,:]]
        
        e1 = Nxyz2 - Nxyz1
        e2 = Nxyz3 - Nxyz1
        av1 = e1[1,:]*e2[2,:] - e1[2,:]*e2[1,:]
        av2 = e1[2,:]*e2[0,:] - e1[0,:]*e2[2,:]
        av3 = e1[0,:]*e2[1,:] - e1[1,:]*e2[0,:]
        self.areas = np.sqrt(av1**2 + av2**2 + av3**2)/2

        # -----------------------------------------------------------------------------
        # Edge GMSH Tag to Index Mapping
        # -----------------------------------------------------------------------------

        edge_tags = np.array(edge_tags).flatten()
        ent = np.array(edge_node_tags).reshape(-1,2).T
        nET = ent.shape[1]
        self.edge_t2i = {int(edge_tags[i]): self.get_edge(self.n_t2i[ent[0,i]], self.n_t2i[ent[1,i]], skip=True) for i in range(nET)}
        self.edge_t2i = {key: value for key,value in self.edge_t2i.items() if value!=-_MISSING_ID}
        self.edge_i2t = {i: t for t, i in self.edge_t2i.items()}
        
        # -----------------------------------------------------------------------------
        # OpenCASCADE Domain to Mesh mappings
        # -----------------------------------------------------------------------------

        # -----------------------------------------------------------------------------
        # Geometry Edges to Mesh edges
        # -----------------------------------------------------------------------------
        for _d, t in edge_dimtags:
            _edge_tags = _edge_entity_map[t]
            if not _edge_tags:
                self.etag_to_edge[t] = []
                continue
            self.etag_to_edge[t] = [int(self.edge_t2i.get(tag,None)) for tag in _edge_tags[0] if tag in self.edge_t2i]
        

        # -----------------------------------------------------------------------------
        # Geometry Faces to Triangles/Nodes and Edges
        # -----------------------------------------------------------------------------
        logger.trace('Constructing geometry to mesh mappings.')
        
        for _d,t in face_dimtags:
            node_tags = _face_entity_map[t]
            node_tags = [self.n_t2i[int(t)] for t in node_tags[0]]
            self.ftag_to_node[t] = node_tags
            node_tags = np.squeeze(np.array(node_tags)).reshape(-1,3).T
            self.ftag_to_tri[t] = [self.get_tri(node_tags[0,i], node_tags[1,i], node_tags[2,i]) for i in range(node_tags.shape[1])]
            self.ftag_to_edge[t] = sorted(list(np.unique(self.tri_to_edge[:,self.ftag_to_tri[t]].flatten())))
            self.ftag_to_normal[t] = gmsh.model.get_normal(t, np.array([0,0]))

        # -----------------------------------------------------------------------------
        # Geometry Volume to Tetrahedra mapping
        # -----------------------------------------------------------------------------

        for _d, t in vol_dimtags:
            node_tags = _map_array_by_dict(_vol_entity_map[t][0], self.n_t2i) 
            node_tags = node_tags.reshape(-1, 4)
            sorted_nodes = np.sort(node_tags, axis=1) 
            self.vtag_to_tet[t] = [self.inv_tets[tuple(row)] for row in sorted_nodes.tolist()]
        self.defined = True

        # -----------------------------------------------------------------------------
        # Final GMSH CACHE
        # -----------------------------------------------------------------------------
        
        # Very short, takes a millisecond

        for dim in (0,1,2,3):
            dts = entity_set[dim]
            for dt in dts:
                self.dimtag_to_edges[dt] = self._domain_edge(dt)
                self.dimtag_to_nodes[dt] = np.array([self.n_t2i[gmsh.model.mesh.get_nodes(*dt)[0][0]] for dt in gmsh.model.get_boundary([dt,], True, False, True)])
                if dim==2:
                    center = self.dimtag_to_center[dt]
                    xyz, _ = gmsh.model.get_closest_point(*dt, center)
                    self.ftag_to_point[dt[1]] = np.array(xyz)
                
        logger.trace('Finalized mesh data generation!')

    ## Higher order functions

    def _pre_derive_node_map(self, bc: Periodic) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
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

        from .mth.pairing import pair_coordinates
        
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
        
        nodemap = pair_coordinates(self.nodes, node_ids_1_arry, node_ids_2_arry, dv, dsmin/4)
        node_ids_2_unsorted = [nodemap[i] for i in sorted(node_ids_1)]
        node_ids_2_sorted = sorted(node_ids_2_unsorted)
        conv_map = {i1: i2 for i1, i2 in zip(node_ids_2_unsorted, node_ids_2_sorted)}

        return conv_map, np.array(node_ids_2_unsorted), np.array(node_ids_2_sorted)

    
    def plot_gmsh(self) -> None:
        gmsh.fltk.run()

    def find_edge_groups(self, edge_ids: np.ndarray) -> list[tuple[int,...]]:
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

        groups = sorted(groups, key = lambda x: sum(self.edge_lengths[np.array(x)]))
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
            sf_origin = (x0, y0, z0)
        else:
            sf_origin = origin

        smesh = SurfaceMesh(self, tri_ids, sf_origin)
        
        if origin is None:
            tet_ids = np.max(self.tri_to_tet[:,tri_ids], axis=0)
            tet_centers = self.centers[:,tet_ids]
            tri_centers = self.tri_centers[:,tri_ids]
            align = tri_centers-tet_centers
            signflip = np.sign(np.sum(align*smesh.normals, axis=0))
            smesh.normals = signflip*smesh.normals
        return smesh


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

        from .mth.optimized import outward_normal, area
        
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
    
