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
from ._global import _GlobalHandler
import gmsh  # type: ignore
import numpy as np
from typing import Union, List, Tuple, Callable, Any
from collections import defaultdict
from loguru import logger
from .bc import Periodic
from emsutil import Saveable

_MISSING_ID: int = -1234


class MeshException(Exception):
    pass


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
    dist_sq = np.einsum("ijk,ijk->ij", diff, diff)  # Shape (N, N)

    # Avoid zero on diagonal (distance to self), set to np.inf
    np.fill_diagonal(dist_sq, np.inf)

    # Return minimum distance
    return np.sqrt(np.min(dist_sq))


def tri_ordering(i1: int, i2: int, i3: int) -> int:
    """Takes two integer indices of triangle verticces and determines if they are in increasing order or decreasing order.
    It ignores cyclic shifts of the indices. so (4,10,21) == (10,21,4) == (21,4,10)

    for triangle (4,10,20) (for example)
    (4,10,20): In co-ordering of phase = 0: i1 < i2, i2 < i3, i3 > i1: 4-10-20-4 diffs = +6 +10 -16
    (10,20,4): In co-order shift 1: i1 < i2, i2 > i3, i3 < i1: 10-20-4-10 diffs = 10 -16 - (6)
    (20,4,10): In co-order shift 2: i1 > i2, i2 < i3, i3 < i1:

    For triangle (20,10,4)
    (20,10,3): i1 > i2, i2 > i3
    (10,3,20): i1 > i2, i2 < i3
    (3,20,10): i1 < i2, i2 > i3
    """
    return np.sign(np.sign(i2 - 1) + np.sign(i3 - i2) + np.sign(i1 - i3))


def _map_array_by_dict(arr: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Convertes a numpy array of integers by a dictionary from int to int

    Args:
        arr (np.ndarray): _description_
        mapping (dict[int, int]): _description_

    Returns:
        np.ndarray: _description_
    """
    keys = np.array(list(mapping.keys()))
    vals = np.array(list(mapping.values()))
    sort_idx = np.argsort(keys)
    keys, vals = keys[sort_idx], vals[sort_idx]

    idx = np.searchsorted(keys, arr)
    return vals[idx]


class Mesh:
    _MISSING_ID: int = _MISSING_ID
    pass


class Mesh3D(Mesh, Saveable):
    """A Mesh managing all 3D mesh related properties.

    Relevant mesh data such as mappings between nodes(vertices), edges, triangles and tetrahedra
    are managed by the Mesh3D class. Specific information regarding to how actual field values
    are mapped to mesh elements is managed by the FEMBasis class.

    The goal of this class is to take all information from GMSH and put it into EMerge so that in principle,
    GMSH can be forgotton about after generating the mesh.

    The reason is that reloading GMSH .msh files seems to be unreliable causing assignments of tags to be corrupted.

    The Mesh3D class is thus called after GMSH is done meshing.

    Mesh post-processing in Python is slow and because GMSH is needed, this is hard to delegate to external libraries.
    As a consequence, EMerge opts to rely a lot on caching mappings between mesh entities which will increase RAM usage but save
    time when requireing specific Mesh properties.

    """

    _MISSING_ID: int = _MISSING_ID

    def __init__(self):

        # Because GMSH associates "tags" with each entity, in EMerge we link GMSH tag numbers with EMerge entiti indices in t2i and i2t notation.

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

        # These mapping where possible associate mesh entities such as triangles, tetrahedra and vertices
        # with those attached/or contained by them. tet_to_edge tells you which edges are contained in a tet
        # where edge_to_tet (not used) would tell you which tets contain any given edge.
        self.tet_to_edge: np.ndarray = np.array([])
        self.tet_to_tri: np.ndarray = np.array([])
        self.tri_to_tet: np.ndarray = np.array([])
        self.tri_to_edge: np.ndarray = np.array([])
        self.edge_to_tri: defaultdict | dict = defaultdict()
        self.node_to_edge: defaultdict | dict = defaultdict()

        # Physics mappings
        # TODO: These should be removable.
        self.tet_to_field: np.ndarray = np.array([])
        self.edge_to_field: np.ndarray = np.array([])
        self.tri_to_field: np.ndarray = np.array([])

        ## States
        # If the mesh is defined after GMSH meshing
        self.defined: bool = False
        # If the mesh is only that of a quick mesh (in which case actual meshing should be done)
        self._quick_mesh: bool = False

        ## Memory
        # These
        self.geonodes: list[
            int
        ] = []  # All nodes that are characteristic of the geometry (excluding mesh nodes)

        # These quantities link more abstract geometric entities such as boxes/faces etc to their
        # respective triangles, nodes etc. They link GMSH tags to mesh indices
        self.ftag_to_tri: dict[int, list[int]] = dict()
        self.ftag_to_node: dict[int, list[int]] = dict()
        self.ftag_to_edge: dict[int, list[int]] = dict()
        self.vtag_to_tet: dict[int, list[int]] = dict()
        self.etag_to_edge: dict[int, list[int]] = dict()

        ## Dervied

        # These quanties link dimension tag pairs (dim,tag) to geometric quantities such as coordiantes and indices
        self.dimtag_to_center: dict[tuple[int, int], tuple[float, float, float]] = (
            dict()
        )
        self.dimtag_to_edges: dict[tuple[int, int], np.ndarray] = dict()
        self.dimtag_to_nodes: dict[tuple[int, int], np.ndarray] = dict()
        self.dimtag_to_bb: dict[tuple[int, int], np.ndarray] = dict()
        self.ftag_to_normal: dict[int, np.ndarray] = dict()
        self.ftag_to_point: dict[int, np.ndarray] = dict()

        # This list specifically contains those faces that form the exterior of the simulation domain.
        self.exterior_face_tags: list[int] = []

    @property
    def n_edges(self) -> int:
        """Return the number of edges"""
        return self.edges.shape[1]

    @property
    def n_tets(self) -> int:
        """Return the number of tets"""
        return self.tets.shape[1]

    @property
    def n_tris(self) -> int:
        """Return the number of triangles"""
        return self.tris.shape[1]

    @property
    def n_nodes(self) -> int:
        """Return the number of nodes"""
        return self.nodes.shape[1]

    def get_edge(self, i1: int, i2: int, skip: bool = False) -> int:
        """Return the edge index given the two node indices"""
        if i1 == i2:
            raise MeshException("Edge cannot be formed by the same node.")
        search = (min(int(i1), int(i2)), max(int(i1), int(i2)))
        result = self.inv_edges.get(search, _MISSING_ID)
        if result == _MISSING_ID and not skip:
            raise MeshException(f"There is no edge with indices {i1}, {i2}")
        return result

    def get_tri(self, i1, i2, i3) -> int:
        """Return the triangle index given the three node indices"""
        i11, i21, i31 = tuple(sorted((int(i1), int(i2), int(i3))))
        output = self.inv_tris.get(tuple(sorted((int(i1), int(i2), int(i3)))), None)
        if output is None:
            _GlobalHandler.active().debugcollector.add_report(
                f"Mesh3D: The program is crashed due to a non existing triangle {i11}, {i21}, {i31}. This occurs often if surfaces stick out of the 3D domain.\n"
                + "Only 3D volumes can be meshed. Parts or entire simulations that are two dimensional will cause this problem."
            )
            raise MeshException(
                f"There is no triangle with indices {i11}, {i21}, {i31}"
            )
        return output

    def get_tet(self, i1, i2, i3, i4) -> int:
        """Return the tetrahedron index given the four node indices"""
        output = self.inv_tets.get(
            tuple(sorted((int(i1), int(i2), int(i3), int(i4)))), None
        )
        if output is None:
            raise MeshException(
                f"There is no tetrahedron with indices {i1}, {i2}, {i3}, {i4}"
            )
        return output

    def get_tetrahedra(self, vol_tags: Union[int, list[int]]) -> np.ndarray:
        if isinstance(vol_tags, int):
            vol_tags = [vol_tags]

        indices = []
        for voltag in vol_tags:
            indices.extend(self.vtag_to_tet[voltag])
        return np.array(indices)

    def get_triangles(self, face_tags: Union[int, list[int]]) -> np.ndarray:
        """Returns a numpyarray of all the triangles that belong to the given face tags"""
        if isinstance(face_tags, int):
            face_tags = [face_tags]
        indices = []
        for facetag in face_tags:
            indices.extend(self.ftag_to_tri[facetag])
        if any([(i is None) for i in indices]):
            logger.error(
                "Clearing None indices: ",
                [i for i, ind in enumerate(indices) if ind is None],
            )
            logger.error(
                "This is usually a sign of boundaries sticking out of domains. Please check your Geometry."
            )
            indices = [i for i in indices if i is not None]

        return np.array(indices)

    def _domain_edge(self, dimtag: tuple[int, int]) -> np.ndarray:
        """Returns a np.ndarray of all edge indices corresponding to a set of dimension tags.

        Args:
            dimtags (list[tuple[int,int]]): A list of dimtags.

        Returns:
            np.ndarray: The list of mesh edge element indices.
        """
        dimtags_edge = []
        d, t = dimtag
        if d == 0:
            return np.ndarray([], dtype=np.int64)
        if d == 1:
            dimtags_edge.append((1, t))
        if d == 2:
            dimtags_edge.extend(gmsh.model.getBoundary([(d, t)], False, False))
        if d == 3:
            dts = gmsh.model.getBoundary([(d, t)], False, False)
            dimtags_edge.extend(gmsh.model.getBoundary(dts, False, False))

        edge_ids = []
        for tag in dimtags_edge:
            edge_ids.extend(self.etag_to_edge[tag[1]])
        edge_ids = np.array(edge_ids)
        return edge_ids

    def domain_edges(self, dimtags: list[tuple[int, int]]) -> np.ndarray:
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
        """Return a list of a tetrahedrons that share a node with any of the nodes in the provided face."""
        tritags = []
        for tags in taglist:
            for tag in tags:
                tritags.extend(self.ftag_to_tri[tag])
        tettags = np.unique(self.tri_to_tet[:, tritags].flatten())
        tettags = tettags[tettags != _MISSING_ID]
        return np.sort(tettags)

    def _get_tet_to_tag(self) -> np.ndarray:
        """Returns an array that counts which domain tag is assigned to each tetrahedron

        Returns:
            np.ndarray: _description_
        """
        tags = np.zeros((self.n_tets,), dtype=np.int64)
        for tag, tets in self.vtag_to_tet.items():
            tags[tets] = tag
        return tags
    
    def _get_dimtags(
        self, nodes: list[int] | None = None, edges: list[int] | None = None
    ) -> list[tuple[int, int]]:
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
            dimtags.append((2, tag))

        for tag, f_edges in self.ftag_to_edge.items():
            if set(f_edges).isdisjoint(edges):
                continue
            dimtags.append((2, tag))

        # test volumes
        for tag, f_tets in self.vtag_to_tet.items():
            v_nodes = set(self.tets[:, f_tets].flatten())
            if not v_nodes.isdisjoint(nodes):
                dimtags.append((3, tag))
            v_edges = set(self.tet_to_edge[:, f_tets].flatten())
            if not v_edges.isdisjoint(edges):
                dimtags.append((3, tag))
        return sorted(dimtags)

    def get_nodes(self, face_tags: Union[int, list[int]]) -> np.ndarray:
        """Returns a numpyarray of all the nodes that belong to the given face tags"""
        if isinstance(face_tags, int):
            face_tags = [face_tags]

        nodes = []
        for facetag in face_tags:
            nodes.extend(self.ftag_to_node[facetag])

        return np.array(sorted(list(set(nodes))))

    def get_edges(self, face_tags: Union[int, list[int]]) -> np.ndarray:
        """Returns a numpyarray of all the edges that belong to the given face tags"""
        if isinstance(face_tags, int):
            face_tags = [face_tags]

        edges = []
        for facetag in face_tags:
            edges.extend(self.ftag_to_edge[facetag])

        return np.array(sorted(list(set(edges))))

    def diagnose(self) -> None:
        """Executes a diagnosis for the mesh to make sure that it is properly defined."""

        # Check 1: Make sure that all i2t bindings are complete
        for i in range(self.n_nodes):
            if i not in self.n_i2t:
                raise MeshException(
                    f"Node {i} is not referenced in the node-to-tet mapping. "
                    f"This indicates an orphan node in the mesh."
                )
        for i in range(self.n_tets):
            if i not in self.tet_i2t:
                raise MeshException(
                    f"Tetrahedron {i} is not referenced in the tet-to-tet mapping. "
                    f"This indicates a disconnected element in the mesh."
                )

        # Check if the mesh is densely connected
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        row = self.edges[0, :]
        col = self.edges[1, :]
        data = np.ones(len(row))
        adj_matrix = csr_matrix((data, (row, col)), shape=(self.n_nodes, self.n_nodes))
        n_components, labels = connected_components(
            csgraph=adj_matrix, directed=False, return_labels=True
        )
        if n_components != 1:
            component_sizes = np.bincount(labels)
            raise MeshException(
                f"Mesh is not densely connected: found {n_components} disconnected components "
                f"with sizes {component_sizes.tolist()}. "
                f"This may indicate gaps or disconnected regions in the geometry."
            )

        # Check tet-to-edge and tet-to-tri
        pointed_edges = np.zeros((self.n_edges,), dtype=np.bool_)
        pointed_tris = np.zeros((self.n_tris,), dtype=np.bool_)
        for itet in range(self.n_tets):
            pointed_edges[self.tet_to_edge[:, itet]] = True
            pointed_tris[self.tet_to_tri[:, itet]] = True
        if not np.all(pointed_edges):
            orphan_edges = np.where(~pointed_edges)[0]
            raise MeshException(
                f"{len(orphan_edges)} edges in the mesh are not referenced by any tetrahedron. "
                f"First few orphan edge indices: {orphan_edges[:10].tolist()}. "
                f"This may indicate surface-only elements from the geometry import."
            )
        if not np.all(pointed_tris):
            orphan_tris = np.where(~pointed_tris)[0]
            raise MeshException(
                f"{len(orphan_tris)} triangles in the mesh are not referenced by any tetrahedron. "
                f"First few orphan triangle indices: {orphan_tris[:10].tolist()}. "
                f"This may indicate surface-only elements from the geometry import."
            )

        # Compute volumes and areas
        from .mth.optimized import calc_volume, calc_area

        for i in range(self.n_tets):
            i1, i2, i3, i4 = self.tets[:, i]
            V = calc_volume(
                self.nodes[:, i1],
                self.nodes[:, i2],
                self.nodes[:, i3],
                self.nodes[:, i4],
            )
            if V == 0.0:
                logger.error(
                    f"Zero volume tetrahedron at index {i}. This points to degenerate geometries."
                )
                raise MeshException(
                    f"Zero volume tetrahedron at index {i}. This points to degenerate geometries."
                )
        for i in range(self.n_tris):
            i1, i2, i3 = self.tris[:, i]
            V = calc_area(self.nodes[:, i1], self.nodes[:, i2], self.nodes[:, i3])
            if V == 0.0:
                logger.error(
                    f"Zero area triangle at index {i}. This points to degenerate geometries."
                )
                raise MeshException(
                    f"Zero area triangle at index {i}. This points to degenerate geometries."
                )

    def fix_zero_volumes(
        self, tol: float = 1e-25, perturbation: float = 1e-12, respect_boundaries=True
    ):
        """Attempt to fix zero-volume tetrahedra by minimally perturbing a non-boundary node.

        For each degenerate tet, finds a node that is not on any boundary (i.e. not in
        ftag_to_node) and offsets it along the tet's normal direction by a small amount.

        Args:
            tol (float): Volume threshold below which a tet is considered degenerate.
            perturbation (float): The displacement magnitude applied to the chosen node.

        Raises:
            MeshException: If a degenerate tet has all four nodes on boundaries.
        """
        from .mth.optimized import calc_volume

        # Collect all boundary-constrained nodes into a set for fast lookup
        boundary_nodes = set()
        for node_list in self.ftag_to_node.values():
            boundary_nodes.update(node_list)

        n_fixed = 0

        for itet in range(self.tets.shape[1]):
            i1, i2, i3, i4 = self.tets[:, itet]
            v1, v2, v3, v4 = (
                self.nodes[:, i1],
                self.nodes[:, i2],
                self.nodes[:, i3],
                self.nodes[:, i4],
            )
            V = calc_volume(v1, v2, v3, v4)

            if V >= tol:
                continue

            # Find a node we're allowed to move (not on any boundary)
            tet_nodes = [i1, i2, i3, i4]
            if respect_boundaries:
                movable = [n for n in tet_nodes if n not in boundary_nodes]
            else:
                movable = tet_nodes
            if len(movable) == 0:
                raise MeshException(
                    f"Zero-volume tetrahedron at index {itet} with nodes {tet_nodes}, "
                    f"but all four nodes lie on boundary surfaces. "
                    f"Cannot fix by perturbation — the geometry itself is degenerate."
                )

            # Compute the normal of the plane formed by the other three nodes
            target = movable[0]
            others = [n for n in tet_nodes if n != target]
            p0, p1, p2 = (
                self.nodes[:, others[0]],
                self.nodes[:, others[1]],
                self.nodes[:, others[2]],
            )

            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)

            if norm < 1e-30:
                # The three other nodes are also degenerate (collinear) — perturb in an arbitrary direction
                # Pick the axis along which the tet has the least extent
                all_coords = np.column_stack([self.nodes[:, n] for n in tet_nodes])
                extents = np.ptp(all_coords, axis=1)
                axis = np.argmin(extents)
                direction = np.zeros(3)
                direction[axis] = 1.0
            else:
                direction = normal / norm

            self.nodes[:, target] += direction * perturbation
            n_fixed += 1

            # Verify it actually worked
            v1, v2, v3, v4 = (
                self.nodes[:, i1],
                self.nodes[:, i2],
                self.nodes[:, i3],
                self.nodes[:, i4],
            )
            V_new = calc_volume(v1, v2, v3, v4)
            if V_new < tol:
                logger.warning(
                    f"Tetrahedron {itet} still degenerate after perturbation (V={V_new}). "
                    f"Increasing perturbation."
                )
                self.nodes[:, target] += direction * perturbation * 100
                n_fixed  # already counted

        if n_fixed > 0:
            logger.warning(
                f"Fixed {n_fixed} zero-volume tetrahedra by perturbing non-boundary nodes "
                f"with magnitude {perturbation}."
            )

    def _pre_update(self, periodic_bcs: list[Periodic] | None = None, scale: float = 1.0):
        """Builds the mesh data properties

        Args:
            periodic_bcs (list[Periodic] | None, optional): A list of periodic boundary conditions. Defaults to None.

        Returns:
            None: None
        """

        # This function will take all GMSH information and build the mesh data properties.
        # Specific algorithmic choices are made for performance reasons. Before making any changes
        # tho this function, make sure to benchmark if the new code does not significantly slow down performance.

        # NEW VERSION
        logger.info("Generating internal mesh data.")
        if periodic_bcs is None:
            periodic_bcs = []

        # -----------------------------------------------------------------------------
        # Pull all mesh information out of GMSH
        # -----------------------------------------------------------------------------

        nodes, lin_coords, _ = gmsh.model.mesh.get_nodes()
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
        _edge_entity_map = {
            t: gmsh.model.mesh.get_elements(1, t)[1]
            for d, t in gmsh.model.get_entities(1)
        }
        _edge_entity_map_2 = {
            t: gmsh.model.mesh.get_elements(1, t)[2]
            for d, t in gmsh.model.get_entities(1)
        }
        _face_entity_map = {
            t: gmsh.model.mesh.get_elements(2, t)[2]
            for d, t in gmsh.model.get_entities(2)
        }
        _vol_entity_map = {
            t: gmsh.model.mesh.get_elements(3, t)[2]
            for d, t in gmsh.model.get_entities(3)
        }

        for dim, dts in entity_set.items():
            self.dimtag_to_center.update(
                {dt: gmsh.model.occ.get_center_of_mass(*dt) for dt in dts}
            )
            
            self.dimtag_to_bb.update(
                {dt: np.array(gmsh.model.occ.get_bounding_box(*dt))*scale for dt in dts}
            )
        if scale != 1.0:
            self.dimtag_to_center = {dt: (x*scale, y*scale, z*scale) for dt, (x,y,z) in self.dimtag_to_center.items()}


        # -----------------------------------------------------------------------------
        # Start of Processing
        # -----------------------------------------------------------------------------

        # -----------------------------------------------------------------------------
        # Vertices
        # -----------------------------------------------------------------------------
        self.nodes = lin_coords.reshape(-1, 3).T
        self.n_i2t = {i: int(t) for i, t in enumerate(nodes)}
        self.n_t2i = {t: i for i, t in self.n_i2t.items()}

        logger.trace(f"Total of {self.nodes.shape[1]} nodes imported.")

        # -----------------------------------------------------------------------------
        # Tetrahedra
        # -----------------------------------------------------------------------------

        # The algorithm assumes that only one domain tag is returned in this function.
        # Hence the use of tri_node_tags[0] in the next line. If domains are missing.
        # Make sure to combine all the entries in the tri-node-tags list

        tet_node_tags = [self.n_t2i[int(t)] for t in tet_node_tags[0]]
        tet_tags = np.squeeze(np.array(tet_tags))

        self.tets = np.array(tet_node_tags).reshape(-1, 4).T
        self.tet_i2t = {i: int(t) for i, t in enumerate(tet_tags)}
        self.tet_t2i = {t: i for i, t in self.tet_i2t.items()}
        self.centers = (
            self.nodes[:, self.tets[0, :]]
            + self.nodes[:, self.tets[1, :]]
            + self.nodes[:, self.tets[2, :]]
            + self.nodes[:, self.tets[3, :]]
        ) / 4
        logger.trace(f"Total of {self.tets.shape[1]} tetrahedra imported.")

        # -----------------------------------------------------------------------------
        # Resorting nodes to keep Periodic boundaries consistent
        # -----------------------------------------------------------------------------

        # Resort node indices to be sorted on all periodic conditions
        # This sorting makes sure that each edge and triangle on a source face is
        # sorted in the same order as the corresponding target face triangle or edge.
        # In other words, if a source face triangle or edge index i1, i2, i3 is mapped to j1, j2, j3 respectively
        # Then this ensures that if i1>i2>i3 then j1>j2>j3

        for bc in periodic_bcs:
            logger.trace(f"reassigning ordered node numbers for periodic boundary {bc}")
            nodemap, ids1, ids2 = self._pre_derive_node_map(bc)
            nodemap = {int(a): int(b) for a, b in nodemap.items()}
            self.nodes[:, ids2] = self.nodes[:, ids1]
            for itet in range(self.tets.shape[1]):
                self.tets[:, itet] = [nodemap.get(i, i) for i in self.tets[:, itet]]
            self.n_t2i = {t: nodemap.get(i, i) for t, i in self.n_t2i.items()}
            self.n_i2t = {t: i for i, t in self.n_t2i.items()}

        # -----------------------------------------------------------------------------
        # Extracting Edges and Triangles
        # -----------------------------------------------------------------------------

        edgeset = set()
        triset = set()

        # As much as possible, the same loop is reused for as many properties.
        # EMerge uses the ordered numbering logic. Edges and triangles
        # can be hashed assuming that indices of their constituent vertices
        # are always ordered, thus a triangle (10,5,1) is hashed as (1,5,10)
        for itet in range(self.tets.shape[1]):
            i1, i2, i3, i4 = sorted([int(ind) for ind in self.tets[:, itet]])
            edgeset.add((i1, i2))
            edgeset.add((i1, i3))
            edgeset.add((i1, i4))
            edgeset.add((i2, i3))
            edgeset.add((i2, i4))
            edgeset.add((i3, i4))
            triset.add((i1, i2, i3))
            triset.add((i1, i2, i4))
            triset.add((i1, i3, i4))
            triset.add((i2, i3, i4))

        logger.trace(
            f"Total of {len(edgeset)} unique edges and {len(triset)} unique triangles."
        )

        self.edges = np.array(sorted(list(edgeset))).T
        self.tris = np.array(sorted(list(triset))).T

        # -----------------------------------------------------------------------------
        # Creating inverse mappings from edge/tri/tet id-tuples to integers
        # -----------------------------------------------------------------------------

        # Map edge index tuples to edge indices
        # This mapping tells which characteristic index pair (4,3) maps to which edge
        logger.trace("Constructing tet/tri/edge and node mappings.")
        nN = self.nodes.shape[1]
        nE = self.edges.shape[1]
        nR = self.tris.shape[1]
        nT = self.tets.shape[1]

        _sorted_edges = self.edges
        _sorted_tris = np.sort(self.tris, axis=0)
        _sorted_tets = np.sort(self.tets, axis=0)

        self.inv_edges = {tuple(_sorted_edges[:, i].tolist()): i for i in range(nE)}
        self.inv_tris = {tuple(_sorted_tris[:, i].tolist()): i for i in range(nR)}
        self.inv_tets = {tuple(_sorted_tets[:, i].tolist()): i for i in range(nT)}

        # -----------------------------------------------------------------------------
        # Constructucting Tet -> Edge/Tri mapping
        # -----------------------------------------------------------------------------
        self.tet_to_edge = np.zeros((6, self.tets.shape[1]), dtype=int) + _MISSING_ID
        self.tet_to_tri = np.zeros((4, self.tets.shape[1]), dtype=int) + _MISSING_ID

        _idset1 = ((1, 2), (1, 3), (1, 4), (2, 3), (4, 2), (3, 4))
        _idset2 = ((1, 2, 3), (1, 3, 4), (1, 4, 2), (2, 3, 4))

        for itet in range(self.tets.shape[1]):
            t = self.tets[:, itet]
            self.tet_to_edge[:, itet] = [
                self.inv_edges[tuple(sorted((int(t[i - 1]), int(t[j - 1]))))]
                for i, j in _idset1
            ]
            self.tet_to_tri[:, itet] = [
                self.inv_tris[
                    tuple(sorted((int(t[i - 1]), int(t[j - 1]), int(t[k - 1]))))
                ]
                for i, j, k in _idset2
            ]

        # -----------------------------------------------------------------------------
        # Constructing Tri -> Tet mapping
        # -----------------------------------------------------------------------------

        n_tets = self.tets.shape[1]
        n_tris = self.tris.shape[1]
        flat_tris = self.tet_to_tri.ravel(order="F")
        flat_tets = np.repeat(np.arange(n_tets, dtype=int), 4)
        sort_idx = np.argsort(flat_tris, stable=True)
        sorted_tris = flat_tris[sort_idx]
        sorted_tets = flat_tets[sort_idx]
        self.tri_to_tet = np.full((2, n_tris), _MISSING_ID, dtype=int)
        _, first_idx, counts = np.unique(
            sorted_tris, return_index=True, return_counts=True
        )
        self.tri_to_tet[0, sorted_tris[first_idx]] = sorted_tets[first_idx]
        mask2 = counts == 2
        self.tri_to_tet[1, sorted_tris[first_idx[mask2]]] = sorted_tets[
            first_idx[mask2] + 1
        ]

        # -----------------------------------------------------------------------------
        # Triangle GMSH Tag to Index Mapping
        # -----------------------------------------------------------------------------

        # The algorithm assumes that only one domain tag is returned in this function.
        # Hence the use of tri_node_tags[0] in the next line. If domains are missing.
        # Make sure to combine all the entries in the tri-node-tags list
        # assuming only one element type here (tri3)

        # tri tags are all labels for all triangular elements in the Mesh

        tri_tags = np.array(tri_tags[0], dtype=int)
        tri_nodes = np.array(
            [self.n_t2i[int(t)] for t in tri_node_tags[0]], dtype=int
        ).reshape(-1, 3)

        self.tri_i2t = {}
        _sorted_tri_nodes = np.sort(tri_nodes, axis=1)

        try:
            for k, tag in enumerate(tri_tags):
                tri_id = self.inv_tris[tuple(_sorted_tri_nodes[k, :].tolist())]
                self.tri_i2t[tri_id] = int(tag)
        except KeyError as e:
            logger.error(e)
            _GlobalHandler.active().debugcollector.add_report(
                "A missing key error usually indicates that there are flat geometries sticking out of the simulation domain.\nEMerge can only deal with volumetric data. Please check the geometry using .view(use_gmsh=True) before meshing to make sure no surfaces are sticking out of the domain."
            )
            raise e
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

        # This algorithm is optimized with help of Claude code.
        # Because each edge may be part of multiple triangles, an efficient algorithm
        # using numpy optimized functions is needed

        flat_edges = self.tri_to_edge.ravel(order="F")
        flat_tris = np.repeat(np.arange(nR, dtype=int), 3)
        sort_idx = np.argsort(flat_edges, stable=True)
        sorted_edges = flat_edges[sort_idx]
        sorted_tris_e = flat_tris[sort_idx]
        _, first_idx, counts = np.unique(
            sorted_edges, return_index=True, return_counts=True
        )
        max_count = int(counts.max())
        self.edge_to_tri = np.full((max_count, nE), _MISSING_ID, dtype=int)
        for k in range(max_count):
            mask = counts > k
            self.edge_to_tri[k, sorted_edges[first_idx[mask]]] = sorted_tris_e[
                first_idx[mask] + k
            ]

        # -----------------------------------------------------------------------------
        # Node to Edge mapping
        # -----------------------------------------------------------------------------

        flat_nodes = self.edges.ravel(order="F")
        flat_edges = np.repeat(np.arange(self.edges.shape[1], dtype=int), 2)

        sort_idx = np.argsort(flat_nodes, stable=True)
        sorted_nodes = flat_nodes[sort_idx]
        sorted_edges = flat_edges[sort_idx]

        _, first_idx, counts = np.unique(
            sorted_nodes, return_index=True, return_counts=True
        )
        max_count = int(counts.max())

        self.node_to_edge = np.full(
            (max_count, self.nodes.shape[1]), _MISSING_ID, dtype=int
        )
        for k in range(max_count):
            mask = counts > k
            self.node_to_edge[k, sorted_nodes[first_idx[mask]]] = sorted_edges[
                first_idx[mask] + k
            ]

        # -----------------------------------------------------------------------------
        # Mesh Quantities
        # -----------------------------------------------------------------------------

        logger.trace("Computing derived quantaties (centres, areas and lengths).")
        # TRIANGLE CENTERS
        self.tri_centers = (
            self.nodes[:, self.tris[0, :]]
            + self.nodes[:, self.tris[1, :]]
            + self.nodes[:, self.tris[2, :]]
        ) / 3

        # EDGE CENTERS
        self.edge_centers = (
            self.nodes[:, self.edges[0, :]] + self.nodes[:, self.edges[1, :]]
        ) / 2

        # EDGE LENGTHS
        self.edge_lengths = np.sqrt(
            np.sum(
                (self.nodes[:, self.edges[0, :]] - self.nodes[:, self.edges[1, :]])
                ** 2,
                axis=0,
            )
        )

        # TRIANGLE AREAS
        Nxyz1 = self.nodes[:, self.tris[0, :]]
        Nxyz2 = self.nodes[:, self.tris[1, :]]
        Nxyz3 = self.nodes[:, self.tris[2, :]]

        e1 = Nxyz2 - Nxyz1
        e2 = Nxyz3 - Nxyz1
        av1 = e1[1, :] * e2[2, :] - e1[2, :] * e2[1, :]
        av2 = e1[2, :] * e2[0, :] - e1[0, :] * e2[2, :]
        av3 = e1[0, :] * e2[1, :] - e1[1, :] * e2[0, :]
        self.areas = np.sqrt(av1**2 + av2**2 + av3**2) / 2

        # -----------------------------------------------------------------------------
        # Edge GMSH Tag to Index Mapping
        # -----------------------------------------------------------------------------

        edge_tags = np.array(edge_tags).flatten()
        ent = np.array(edge_node_tags).reshape(-1, 2).T
        nET = ent.shape[1]
        self.edge_t2i = {
            int(edge_tags[i]): self.get_edge(
                self.n_t2i[ent[0, i]], self.n_t2i[ent[1, i]], skip=True
            )
            for i in range(nET)
        }
        self.edge_t2i = {
            key: value for key, value in self.edge_t2i.items() if value != -_MISSING_ID
        }
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
            self.etag_to_edge[t] = [
                int(self.edge_t2i.get(tag, None))
                for tag in _edge_tags[0]
                if tag in self.edge_t2i
            ]
            self.geonodes.extend([int(i) for i in _edge_entity_map_2[t][0]])

        # Geo nodes are all unique node ides that are characteirstic of the input geometry.
        self.geonodes = sorted(list(set(self.geonodes)))

        # -----------------------------------------------------------------------------
        # Geometry Faces to Triangles/Nodes and Edges
        # -----------------------------------------------------------------------------
        logger.trace("Constructing geometry to mesh mappings.")

        for _d, t in face_dimtags:
            node_tags = _face_entity_map[t]
            node_tags = [self.n_t2i[int(t)] for t in node_tags[0]]
            self.ftag_to_node[t] = node_tags
            node_tags = np.squeeze(np.array(node_tags)).reshape(-1, 3).T
            self.ftag_to_tri[t] = [
                self.get_tri(node_tags[0, i], node_tags[1, i], node_tags[2, i])
                for i in range(node_tags.shape[1])
            ]
            self.ftag_to_edge[t] = sorted(
                list(np.unique(self.tri_to_edge[:, self.ftag_to_tri[t]].flatten()))
            )
            self.ftag_to_normal[t] = gmsh.model.get_normal(t, np.array([0, 0]))

        # -----------------------------------------------------------------------------
        # Geometry Volume to Tetrahedra mapping
        # -----------------------------------------------------------------------------

        for _d, t in vol_dimtags:
            node_tags = _map_array_by_dict(_vol_entity_map[t][0], self.n_t2i)
            node_tags = node_tags.reshape(-1, 4)
            sorted_nodes = np.sort(node_tags, axis=1)
            self.vtag_to_tet[t] = [
                self.inv_tets[tuple(row)] for row in sorted_nodes.tolist()
            ]
        self.defined = True

        # -----------------------------------------------------------------------------
        # Final GMSH CACHE
        # -----------------------------------------------------------------------------

        # Very short, takes a millisecond

        for dim in (0, 1, 2, 3):
            dts = entity_set[dim]
            for dt in dts:
                self.dimtag_to_edges[dt] = self._domain_edge(dt)
                self.dimtag_to_nodes[dt] = np.array(
                    [
                        self.n_t2i[gmsh.model.mesh.get_nodes(*dt)[0][0]]
                        for dt in gmsh.model.get_boundary([dt], True, False, True)
                    ]
                )
                if dim == 2:
                    (x0,y0,z0) = self.dimtag_to_center[dt]
                    xyz, _ = gmsh.model.get_closest_point(*dt, (x0/scale, y0/scale, z0/scale))
                    self.ftag_to_point[dt[1]] = np.array(xyz)*scale

        logger.info("Finalized mesh data generation!")

    ## Higher order functions

    def _pre_derive_node_map(
        self, bc: Periodic
    ) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
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

        for d, t in face_dimtags:
            domain_tag, f_tags, node_tags = gmsh.model.mesh.get_elements(2, t)
            node_tags = [self.n_t2i[int(t)] for t in node_tags[0]]
            if t in bc.face1.tags:
                node_ids_1.extend(node_tags)
            if t in bc.face2.tags:
                node_ids_2.extend(node_tags)

        node_ids_1 = sorted(list(set(node_ids_1)))
        node_ids_2 = sorted(list(set(node_ids_2)))

        all_node_ids = np.unique(np.array(node_ids_1 + node_ids_2))
        dsmin = shortest_distance(self.nodes[:, all_node_ids])

        node_ids_1_arry = np.array(node_ids_1)
        node_ids_2_arry = np.array(node_ids_2)
        dv = np.array(bc.dv)

        nodemap = pair_coordinates(
            self.nodes, node_ids_1_arry, node_ids_2_arry, dv, dsmin / 4
        )
        node_ids_2_unsorted = [nodemap[i] for i in sorted(node_ids_1)]
        node_ids_2_sorted = sorted(node_ids_2_unsorted)
        conv_map = {i1: i2 for i1, i2 in zip(node_ids_2_unsorted, node_ids_2_sorted)}

        return conv_map, np.array(node_ids_2_unsorted), np.array(node_ids_2_sorted)

    def plot_gmsh(self) -> None:
        gmsh.fltk.run()

    def find_edge_groups(self, edge_ids: np.ndarray) -> list[tuple[int, ...]]:
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
        edges = self.edges[:, edge_ids]
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("`edges` must have shape (2, N)")

        # n_edges: int = edges.shape[1]

        # --- build “vertex ⇒ incident edge IDs” map ------------------------------
        vert2edges = defaultdict(list)
        for eid in edge_ids:
            v1, v2 = self.edges[0, eid], self.edges[1, eid]
            vert2edges[v1].append(eid)
            vert2edges[v2].append(eid)

        groups = []

        ungrouped = set(edge_ids)

        group = [edge_ids[0]]
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
                group = [ungrouped.pop()]
            else:
                group += list(new_edges)
                ungrouped.difference_update(new_edges)

        groups = sorted(groups, key=lambda x: sum(self.edge_lengths[np.array(x)]))
        return groups

    def outward_normals(
        self,
        tri_ids: list[int],
        alignment_origin: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """Computes strictly outward facing normals for given triangle indices.

        Outwards is defined as pointing away from the simulation domain.
        This assumes that the tri_ids are on the boundary of the simualtion domain.
        It checks the centroid of the connected tetrahedron and then points the normal
        away from that centroid.

        Args:
            tri_ids (list[int]): _description_
            alignment_origin (tuple[float, float, float], optional): A normal alignment origin in case inside vs. outside is not apparant.

        Returns:
            np.ndarray: _description_
        """
        v1s = self.nodes[:, self.tris[0, tri_ids]]
        v2s = self.nodes[:, self.tris[1, tri_ids]]
        v3s = self.nodes[:, self.tris[2, tri_ids]]
        e1 = v2s - v1s
        e2 = v3s - v1s
        nx = e1[1, :] * e2[2, :] - e1[2, :] * e2[1, :]
        ny = e1[2, :] * e2[0, :] - e1[0, :] * e2[2, :]
        nz = e1[0, :] * e2[1, :] - e1[1, :] * e2[0, :]
        nn = (nx**2 + ny**2 + nz**2) ** 0.5
        nx = nx / nn
        ny = ny / nn
        nz = nz / nn
        normals = np.array([nx, ny, nz])
        tet_ids = np.max(self.tri_to_tet[:, tri_ids], axis=0)
        tet_centers = self.centers[:, tet_ids]
        tri_centers = self.tri_centers[:, tri_ids]
        align = tri_centers - tet_centers
        signflip = np.sign(np.sum(align * normals, axis=0))
        normals = signflip * normals

        if alignment_origin is not None:
            rad_vec = tri_centers.copy()
            rad_vec[0, :] -= alignment_origin[0]
            rad_vec[1, :] -= alignment_origin[1]
            rad_vec[2, :] -= alignment_origin[2]
            signflip = np.sign(np.sum(normals * rad_vec, axis=0))
            normals = normals * signflip
        return normals

    def boundary_surface(
        self,
        face_tags: Union[int, list[int]],
        inward_normal: bool = True,
        origin: tuple[float, float, float] | None = None,
    ) -> SurfaceMesh:
        """Returns a SurfaceMesh class that is a 2D mesh isolated from the 3D mesh

        The mesh will be based on the given set of face tags.

        In order to properly allign the normal vectors, an alignment origin can be provided.
        If not provided, the center point of all boundaries will be used.

        Args:
            face_tags (Union[int, list[int]]): The list of face tags to use
            origin (tuple[float, float, float], optional): The normal vecor alignment origin. Defaults to None.

        Returns:
            SurfaceMesh: The resultant surface mesh
        """
        tri_ids = self.get_triangles(face_tags)

        normals = self.outward_normals(tri_ids, alignment_origin=origin)
        if inward_normal:
            normals = -normals

        smesh = SurfaceMesh(self, tri_ids, normals)

        return smesh

    def get_normal(self, itri: int) -> np.ndarray:
        """Returns a triangle normal which with an
        arbitrarily assigned normal direction.
        (Deterministic but not consistent with faces)

        Args:
            itri (int): The triangle index

        Returns:
            np.ndarray: _description_
        """
        i1, i2, i3 = self.tris[:, itri]
        p0, p1, p2 = (
            self.nodes[:, i1],
            self.nodes[:, i2],
            self.nodes[:, i3],
        )

        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        return normal / norm

    def inward_normal(self, face_tags: list[int]) -> np.ndarray:
        """Compute the inward pointing normal vector and return it."""
        tri_ids = self.get_triangles(face_tags)

        tri_ids = np.array(sorted(list(set(tri_ids))))

        tri1 = tri_ids[0]
        tri_center = self.tri_centers[:, tri1]
        tet_center = self.centers[:, self.tri_to_tet[0, tri1]]

        inside = tet_center - tri_center
        port_normal = self.get_normal(tri1)

        # Align normal
        if sum(inside * port_normal) < 0:
            port_normal = -port_normal

        return port_normal
    
    def apply_local_refinement(
        self,
        tet_indices: np.ndarray,
        local_node_ids: np.ndarray,
        steiner: np.ndarray,
        tetra_out: np.ndarray,
    ) -> None:
        """Installs the output of gmsh.algorithm.refine_tetrahedra back into the gmsh model.

        This only updates the gmsh model itself (new nodes, removed/added elements).
        Call self._pre_update(periodic_bcs) afterward to rebuild all derived EMerge
        mesh properties (edges, triangles, mappings, etc.) from the updated model --
        patching those incrementally by hand here would duplicate and risk diverging
        from the already-tested logic in _pre_update.

        Args:
            tet_indices: EMerge-local indices into self.tets that were passed to
                refine_tetrahedra (e.g. your `refine_tet_ids`). Must all belong to
                the same gmsh volume.
            local_node_ids: EMerge-local indices into self.nodes, in the same order
                as the `coord`/`sizeAtNode` arrays passed to refine_tetrahedra (e.g.
                the node-index output of tet_to_node).
            steiner: flat (3*n_new,) new point coordinates returned by
                refine_tetrahedra.
            tetra_out: flat, 1-based, 4-per-tet connectivity returned by
                refine_tetrahedra (confirmed convention: subtract 1 for local use).
        """
        steiner = np.asarray(steiner, dtype=np.float64)
        tetra_out = np.asarray(tetra_out, dtype=np.int64) - 1  # 1-based -> 0-based
        tet_indices = np.asarray(tet_indices, dtype=np.int64)
        local_node_ids = np.asarray(local_node_ids, dtype=np.int64)

        # Determine the owning gmsh volume from the flagged tets themselves --
        # reuses vtag_to_tet rather than re-deriving anything from gmsh.
        vol_tag = None
        for t, tets in self.vtag_to_tet.items():
            if int(tet_indices[0]) in tets:
                vol_tag = t
                break
        if vol_tag is None:
            raise MeshException(
                "apply_local_refinement: could not determine the owning gmsh "
                "volume for the given tet_indices."
            )
        tet_set = set(self.vtag_to_tet[vol_tag])
        if not all(int(i) in tet_set for i in tet_indices):
            raise MeshException(
                "apply_local_refinement: tet_indices span more than one gmsh "
                "volume. Group flagged tets per volume before calling this."
            )

        n_new = steiner.size // 3

        # 1. Register new (Steiner) nodes on the volume with fresh gmsh tags.
        if n_new:
            max_node_tag = gmsh.model.mesh.get_max_node_tag()
            new_gmsh_node_tags = np.arange(
                max_node_tag + 1, max_node_tag + 1 + n_new, dtype=np.int64
            )
            gmsh.model.mesh.add_nodes(
                3, vol_tag, new_gmsh_node_tags.tolist(), steiner.tolist()
            )
        else:
            new_gmsh_node_tags = np.empty(0, dtype=np.int64)

        # 2. Translate refine_tetrahedra's local index space
        #    [local_node_ids ++ steiner points] into gmsh node tags.
        orig_gmsh_node_tags = np.array(
            [self.n_i2t[int(i)] for i in local_node_ids], dtype=np.int64
        )
        combined_gmsh_tags = np.concatenate([orig_gmsh_node_tags, new_gmsh_node_tags])
        tet_node_gmsh_tags = combined_gmsh_tags[tetra_out]

        # 3. Remove the superseded tets (element tags via tet_i2t -- no need
        #    to re-query gmsh for them).
        old_element_tags = np.array(
            [self.tet_i2t[int(i)] for i in tet_indices], dtype=np.int64
        )
        gmsh.model.mesh.remove_elements(3, vol_tag, old_element_tags.tolist())

        # 4. Add the refined tets with fresh gmsh element tags.
        n_new_tets = tet_node_gmsh_tags.size // 4
        max_elem_tag = gmsh.model.mesh.get_max_element_tag()
        new_elem_tags = np.arange(
            max_elem_tag + 1, max_elem_tag + 1 + n_new_tets, dtype=np.int64
        )
        gmsh.model.mesh.add_elements_by_type(
            vol_tag, 4, new_elem_tags.tolist(), tet_node_gmsh_tags.tolist()
        )

    def expand_with_neighbors(self, tet_indices: np.ndarray, allowed_tets: set, layers: int = 1) -> np.ndarray:
        """Grows a set of tet indices by `layers` rings of face-adjacency,
        using tet_to_tri / tri_to_tet -- no new mesh bookkeeping needed, this
        is exactly the adjacency info Mesh3D already maintains.
    
        `allowed_tets` restricts expansion to a single gmsh volume (e.g.
        set(mesh.vtag_to_tet[vol_tag])) -- expansion must NOT cross a material
        boundary into a different volume; that's a different, harder problem
        than sliver avoidance and out of scope here.
        """
        current = set(int(t) for t in tet_indices)
        frontier = set(current)
    
        for _ in range(layers):
            new_frontier = set()
            for itet in frontier:
                for iface in range(4):
                    tri_id = self.tet_to_tri[iface, itet]
                    for j in range(2):
                        other = self.tri_to_tet[j, tri_id]
                        if other == self._MISSING_ID or other == itet:
                            continue
                        other = int(other)
                        if other in allowed_tets and other not in current:
                            new_frontier.add(other)
            if not new_frontier:
                break
            current |= new_frontier
            frontier = new_frontier
    
        return np.array(sorted(current), dtype=np.int64)

class SurfaceMesh(Mesh):
    """The surface mesh class is used to assemble the Modal port matrix.


    Args:
        Mesh (_type_): _description_
    """

    def __init__(
        self,
        original: Mesh3D,
        tri_ids: np.ndarray,
        normals: tuple[float, float, float],
    ):

        ## Compute derived mesh properties
        tris = original.tris[:, tri_ids]
        unique_nodes = np.sort(np.unique(tris.flatten()))
        new_ids = np.arange(unique_nodes.shape[0])
        old_to_new_node_id_map = {a: b for a, b in zip(unique_nodes, new_ids)}
        new_tris = np.array(
            [
                [old_to_new_node_id_map[tris[i, j]] for i in range(3)]
                for j in range(tris.shape[1])
            ]
        ).T

        ### Store information
        self._tri_ids = tri_ids
        self.normals = normals

        self.original_tris: np.ndarray = original.tris

        self.old_new_node_map: dict = old_to_new_node_id_map
        self.original: Mesh3D = original
        self.nodes: np.ndarray = original.nodes[:, unique_nodes]
        self.tris: np.ndarray = new_tris

        ## initialize derived
        self.edge_centers: np.ndarray = np.array([])
        self.edge_tris: np.ndarray = np.array([])
        self.n_nodes = self.nodes.shape[1]
        self.n_tris = self.tris.shape[1]
        self.n_edges: float = -1
        self.areas: np.ndarray = np.array([])
        self.centroids: np.ndarray = np.array([])
        self.tri_to_edge: np.ndarray = np.array([])
        self.edge_to_tri: dict | defaultdict = dict()
        # Generate derived
        self.update()

    def copy(self) -> SurfaceMesh:
        return SurfaceMesh(self.original, self._tri_ids.copy(), self.normals.copy())

    def flip(self, ax: str) -> SurfaceMesh:
        if ax.lower() == "x":
            self.flipX()
        if ax.lower() == "y":
            self.flipY()
        if ax.lower() == "z":
            self.flipZ()
        return self
        # self.tris[(0,1),:] = self.tris[(1,0),:]

    def flipX(self) -> SurfaceMesh:
        self.nodes[0, :] = -self.nodes[0, :]
        self.normals[0, :] = -self.normals[0, :]
        self.edge_centers[0, :] = -self.edge_centers[0, :]
        self.centroids[0, :] = -self.centroids[0, :]
        return self

    def flipY(self) -> SurfaceMesh:
        self.nodes[1, :] = -self.nodes[1, :]
        self.normals[1, :] = -self.normals[1, :]
        self.edge_centers[1, :] = -self.edge_centers[1, :]
        self.centroids[1, :] = -self.centroids[1, :]
        return self

    def flipZ(self) -> SurfaceMesh:
        self.nodes[2, :] = -self.nodes[2, :]
        self.normals[2, :] = -self.normals[2, :]
        self.edge_centers[2, :] = -self.edge_centers[2, :]
        self.centroids[2, :] = -self.centroids[2, :]
        return self

    def from_source_tri(self, triid: int) -> int | None:
        """Returns a triangle index from the old mesh to the new mesh."""
        i1in = self.original.tris[0, triid]
        i2in = self.original.tris[1, triid]
        i3in = self.original.tris[2, triid]
        i1 = self.old_new_node_map.get(i1in, None)
        i2 = self.old_new_node_map.get(i2in, None)
        i3 = self.old_new_node_map.get(i3in, None)
        if i1 is None or i2 is None or i3 is None:
            return None
        return self.get_tri(i1, i2, i3)

    def from_source_edge(self, edgeid: int) -> int | None:
        """Returns an edge index form the old mesh to the new mesh."""
        i1 = self.old_new_node_map.get(self.original.edges[0, edgeid], None)
        i2 = self.old_new_node_map.get(self.original.edges[1, edgeid], None)
        if i1 is None or i2 is None:
            return None
        return self.get_edge(i1, i2)

    def get_edge(self, i1: int, i2: int) -> int:
        """Return the edge index given the two node indices"""
        if i1 == i2:
            raise ValueError("Edge cannot be formed by the same node.")
        search = (min(int(i1), int(i2)), max(int(i1), int(i2)))
        result = self.inv_edges.get(search, None)
        if result is None:
            raise ValueError(f"There is no edge with indices {i1}, {i2}")
        return result

    def get_edge_sign(self, i1: int, i2: int) -> int:
        """Return the edge index given the two node indices"""
        if i1 == i2:
            raise ValueError("Edge cannot be formed by the same node.")
        if i1 > i2:
            return -1
        return 1

    def get_tri(self, i1, i2, i3) -> int:
        """Return the triangle index given the three node indices"""
        result = self.inv_tris.get(tuple(sorted((int(i1), int(i2), int(i3)))), None)
        if result is None:
            raise ValueError(f"There is no triangle with indices {i1}, {i2}, {i3}")
        return result

    def update(self) -> None:
        ## First Edges

        from .mth.optimized import area

        edges = set()
        for i in range(self.n_tris):
            i1, i2, i3 = self.tris[:, i]
            edges.add((i1, i2))
            edges.add((i2, i3))
            edges.add((i1, i3))

        edgelist = list(edges)

        self.edges = np.array(edgelist).T
        self.n_edges = self.edges.shape[1]
        self.edge_centers = (
            self.nodes[:, self.edges[0, :]] + self.nodes[:, self.edges[1, :]]
        ) / 2

        ## Mapping from edge pairs to edge index

        def _hash(ints):
            return tuple(sorted([int(x) for x in ints]))

        self.inv_edges = {
            (int(self.edges[0, i]), int(self.edges[1, i])): i
            for i in range(self.edges.shape[1])
        }
        self.inv_tris = {
            _hash((self.tris[0, i], self.tris[1, i], self.tris[2, i])): i
            for i in range(self.tris.shape[1])
        }
        ##
        self.areas = np.array(
            [
                area(
                    self.nodes[:, self.tris[0, i]],
                    self.nodes[:, self.tris[1, i]],
                    self.nodes[:, self.tris[2, i]],
                )
                for i in range(self.n_tris)
            ]
        ).T

        n1 = self.nodes[:, self.tris[0, :]]
        n2 = self.nodes[:, self.tris[1, :]]
        n3 = self.nodes[:, self.tris[2, :]]

        self.centroids = 1 / 3 * (n1 + n2 + n3)
        self.tri_to_edge = np.ndarray((3, self.tris.shape[1]), dtype=int)
        self.edge_to_tri = defaultdict(list)

        for itri in range(self.tris.shape[1]):
            i1, i2, i3 = self.tris[:, itri]
            ie1 = self.get_edge(i1, i2)
            ie2 = self.get_edge(i2, i3)
            ie3 = self.get_edge(i1, i3)
            self.tri_to_edge[:, itri] = [ie1, ie2, ie3]

    @property
    def exyz(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.edge_centers[0, :], self.edge_centers[1, :], self.edge_centers[2, :]
