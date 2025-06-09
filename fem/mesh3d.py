from __future__ import annotations
import gmsh
import numpy as np
from numba import njit
from .mesher import Mesher
from typing import Union, List, Tuple, Callable
from collections import defaultdict
from .geo3d import GMSHVolume
from .mth.optimized import outward_normal
from loguru import logger

@njit(cache=True)
def area(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray):
    e1 = x2 - x1
    e2 = x3 - x1
    av = np.array([e1[1]*e2[2] - e1[2]*e2[1], e1[2]*e2[0] - e1[0]*e2[2], e1[0]*e2[1] - e1[1]*e2[0]])
    return np.sqrt(av[0]**2 + av[1]**2 + av[2]**2)/2


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


class Mesh3D:
    """A Mesh managing all 3D mesh related properties.

    Relevant mesh data such as mappings between nodes(vertices), edges, triangles and tetrahedra
    are managed by the Mesh3D class. Specific information regarding to how actual field values
    are mapped to mesh elements is managed by the FEMBasis class.
    
    """
    def __init__(self, geometry: Mesher):
        
        self.geometry: Mesher = geometry

        # All spatial objects
        self.nodes: np.ndarray = None
        self.n_i2t: dict = None
        self.n_t2i: dict = None

        # tets colletions
        self.tets: np.ndarray = None
        self.tet_i2t: dict = None
        self.tet_t2i: dict = None
        self.centers: np.ndarray = None

        # triangles
        self.tris: np.ndarray = None
        self.tri_i2t: dict = None
        self.tri_t2i: dict = None
        self.areas: np.ndarray = None
        self.tri_centers: np.ndarray = None

        # edges
        self.edges: np.ndarray = None
        self.edge_i2t: dict = None
        self.edge_t2i: dict = None
        self.edge_centers: np.ndarray = None
        self.edge_lengths: np.ndarray = None
        
        # Inverse mappings
        self.inv_edges: dict = None
        self.inv_tris: dict = None
        self.inv_tets: dict = None

        # Mappings

        self.tet_to_edge: np.ndarray = None
        self.tet_to_edge_sign: np.ndarray = None
        self.tet_to_tri: np.ndarray = None
        self.tri_to_tet: np.ndarray = None
        self.tri_to_edge: np.ndarray = None
        self.tri_to_edge_sign: np.ndarray = None
        self.edge_to_tri: defaultdict = None
        self.node_to_edge: defaultdict = None

        # Physics mappings

        self.tet_to_field: np.ndarray = None
        self.edge_to_field: np.ndarray = None
        self.tri_to_field: np.ndarray = None

        ## States
        self.defined = False

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

    def get_edge(self, i1: int, i2: int) -> int:
        '''Return the edge index given the two node indices'''
        if i1==i2:
            raise ValueError("Edge cannot be formed by the same node.")
        search = (min(int(i1),int(i2)), max(int(i1),int(i2)))
        result =  self.inv_edges.get(search, None)
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
        return self.inv_tris.get(tuple(sorted((int(i1), int(i2), int(i3)))), None)
    
    def get_tet(self, i1, i2, i3, i4) -> int:
        '''Return the tetrahedron index given the four node indices'''
        return self.inv_tets.get(tuple(sorted((int(i1), int(i2), int(i3), int(i4)))), None)
    
    def boundary_triangles(self, dimtags: list[tuple[int, int]] = None) -> np.ndarray:
        if dimtags is None:
            domain_tag, face_tags, node_tags = gmsh.model.mesh.get_elements(2)
            node_tags = [self.n_t2i[int(t)] for t in node_tags[0]]
            node_tags = np.squeeze(np.array(node_tags)).reshape(-1,3).T
            indices = [self.get_tri(node_tags[0,i], node_tags[1,i], node_tags[2,i]) for i in range(node_tags.shape[1])]
            return np.array(indices)
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
            domain_tag, v_tags, node_tags = gmsh.model.mesh.get_elements(3, voltag)
            node_tags = [self.n_t2i[int(t)] for t in node_tags[0]]
            node_tags = np.squeeze(np.array(node_tags)).reshape(-1,4).T
            indices.extend([self.get_tet(node_tags[0,i], node_tags[1,i], node_tags[2,i], node_tags[3,i]) for i in range(node_tags.shape[1])])
        return np.array(indices)
    
    def get_triangles(self, face_tags: Union[int, list[int]]) -> np.ndarray:
        '''Returns a numpyarray of all the triangles that belong to the given face tags'''
        if isinstance(face_tags, int):
            face_tags = [face_tags,]
        
        indices = []
        for facetag in face_tags:
            domain_tag, f_tags, node_tags = gmsh.model.mesh.get_elements(2, facetag)
            node_tags = [self.n_t2i[int(t)] for t in node_tags[0]]
            node_tags = np.squeeze(np.array(node_tags)).reshape(-1,3).T
            indices.extend([self.get_tri(node_tags[0,i], node_tags[1,i], node_tags[2,i]) for i in range(node_tags.shape[1])])
        if any([(i is None) for i in indices]):
            logger.error('Clearing None indices: ', [i for i, ind in enumerate(indices) if ind is None])
            logger.error('This is usually a sign of boundaries sticking out of domains. Please check your Geometry.')
            indices = [i for i in indices if i is not None]

        return np.array(indices)
    

    def get_nodes(self, face_tags: Union[int, list[int]]) -> np.ndarray:
        '''Returns a numpyarray of all the nodes that belong to the given face tags'''
        if isinstance(face_tags, int):
            face_tags = [face_tags,]
        
        nodes = []
        for facetag in face_tags:
            domain_tag, f_tags, node_tags = gmsh.model.mesh.get_elements(2, facetag)
            node_tags = [self.n_t2i[int(t)] for t in node_tags[0]]
            nodes.extend(node_tags)
        
        return np.array(sorted(list(set(nodes))))
    
    
    def update(self):
        nodes, lin_coords, _  = gmsh.model.mesh.get_nodes()
        
        coords = lin_coords.reshape(-1, 3).T
        
        ## Vertices
        self.nodes = coords
        self.n_i2t = {i: int(t) for i, t in enumerate(nodes)}
        self.n_t2i = {t: i for i, t in self.n_i2t.items()}

        ## Tetrahedras

        _, tri_tags, tri_node_tags = gmsh.model.mesh.get_elements(3)
        
        # The algorithm assumes that only one domain tag is returned in this function. 
        # Hence the use of tri_node_tags[0] in the next line. If domains are missing.
        # Make sure to combine all the entries in the tri-node-tags list
        
        tri_node_tags = [self.n_t2i[int(t)] for t in tri_node_tags[0]]
        tri_tags = np.squeeze(np.array(tri_tags))

        self.tets = np.array(tri_node_tags).reshape(-1,4).T
        self.tet_i2t = {i: int(t) for i, t in enumerate(tri_tags)}
        self.tet_t2i = {t: i for i, t in self.tet_i2t.items()}

        self.centers = (self.nodes[:,self.tets[0,:]] + self.nodes[:,self.tets[1,:]] + self.nodes[:,self.tets[2,:]] + self.nodes[:,self.tets[3,:]]) / 4
        #Extract unique edges and triangles

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

        # Edges are effectively Randomly sorted
        # It contains index pairs of vertices edge 1 = (ev1, ev2) etc.
        # Same for traingles
        self.edges = np.array(list(edgeset)).T
        self.tris = np.array(list(triset)).T
        
        self.tri_centers = (self.nodes[:,self.tris[0,:]] + self.nodes[:,self.tris[1,:]] + self.nodes[:,self.tris[2,:]]) / 3
        def _hash(ints):
            return tuple(sorted([int(x) for x in ints]))
        
        # Map edge index tuples to edge indices
        # This mapping tells which characteristic index pair (4,3) maps to which edge
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
        
        _, tri_tags, tri_node_tags = gmsh.model.mesh.get_elements(2)
        
        # The algorithm assumes that only one domain tag is returned in this function. 
        # Hence the use of tri_node_tags[0] in the next line. If domains are missing.
        # Make sure to combine all the entries in the tri-node-tags list
        tri_node_tags = [self.n_t2i[int(t)] for t in tri_node_tags[0]]
        tri_tags = np.squeeze(np.array(tri_tags))

        self.tri_i2t = {self.get_tri(*self.tris[:,i]): int(t) for i, t in enumerate(tri_tags)}
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

        self.edge_centers = (self.nodes[:,self.edges[0,:]] + self.nodes[:,self.edges[1,:]]) / 2
        self.edge_lengths = np.sqrt(np.sum((self.nodes[:,self.edges[0,:]] - self.nodes[:,self.edges[1,:]])**2, axis=0))
        self.areas = np.array([area(self.nodes[:,self.tris[0,i]], self.nodes[:,self.tris[1,i]], self.nodes[:,self.tris[2,i]]) for i in range(self.tris.shape[1])])

        self.defined = True
    ## Higher order functions

    def retreive(self, material_selector: Callable, volumes: list[GMSHVolume]) -> np.ndarray:
        '''Retrieve the material properties of the geometry'''
        arry = np.zeros((3,3,self.n_tets,), dtype=np.complex128)

        xs = self.centers[0,:]
        ys = self.centers[1,:]
        zs = self.centers[2,:]

        dimtags = []
        
        for volume in volumes:
        
            for dimtag in volume.dimtags:
                
                etype, etag_list, ntags = gmsh.model.mesh.get_elements(*dimtag)
                for etags in etag_list:
                    tet_ids = [self.tet_t2i[t] for t in etags]

                    value = material_selector(volume.material, xs[tet_ids], ys[tet_ids], zs[tet_ids])
                    arry[:,:,tet_ids] = value
        return arry
    
    def plot_gmsh(self) -> None:
        gmsh.fltk.run()

    def find_edge_groups(self, edge_ids: np.ndarray) -> dict:
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
                         origin: tuple[float, float, float]) -> SurfaceMesh:
        
        tri_ids = self.get_triangles(face_tags)

        return SurfaceMesh(self, tri_ids, origin)#self.nodes[:,unique_nodes], new_tris, origin=origin, nodemap=mapper, original=self, original_tris=ids)

class SurfaceMesh:

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
        self.original_tris: np.ndarray = original.tris
        self.old_new_node_map: dict[int,int] = old_to_new_node_id_map
        self.original: Mesh3D = original
        self._alignment_origin: np.ndarray = np.array(origin).astype(np.float64)
        self.nodes: np.ndarray = original.nodes[:, unique_nodes]
        self.tris: np.ndarray = new_tris

        ## initialize derived
        self.edge_centers: np.ndarray = None
        self.edge_tris: np.ndarray = None
        self.n_nodes = self.nodes.shape[1]
        self.n_tris = self.tris.shape[1]
        self.n_edges = None
        self.areas: np.ndarray = None

        # Generate derived
        self.update()

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
        return self.inv_tris.get(tuple(sorted((int(i1), int(i2), int(i3)))), None)
    
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
    