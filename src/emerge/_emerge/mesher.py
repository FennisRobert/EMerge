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
import gmsh  # type: ignore
from .geometry import GeoVolume, GeoObject, GeoSurface
from .selection import Selection, FaceSelection
from .periodic import PeriodicCell
import numpy as np
from typing import Iterable, Callable, Any, TypeVar
from loguru import logger
from enum import Enum
from .bc import Periodic, BoundaryCondition


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


# (1: Delaunay, 3: Initial mesh only, 4: Frontal, 7: MMG3D, 9: R-tree, 10: HXT)


class Algorithm3D(Enum):
    DELAUNAY = 1
    INITIAL_MESH_ONLY = 3
    FRONTAL = 4
    MMG3D = 7
    RTREE = 9
    HXT = 10


_DOM_TO_STR = {
    0: "point",
    1: "edge",
    2: "face",
    3: "volume",
}

T = TypeVar("T")


def unpack_lists(_list: Any, collector: list | None = None) -> list[Any]:
    """Unpack a recursive list of lists"""
    if collector is None:
        collector = []
    for item in _list:
        if isinstance(item, list):
            unpack_lists(item, collector)
        else:
            collector.append(item)

    return collector


class AMRPoints:
    def __init__(self):
        self._amr_fields: list[int] = []
        self._amr_coords: np.ndarray = None
        self._amr_sizes: np.ndarray = None
        self._amr_ratios: np.ndrray = None
        self._amr_new: np.ndarray = None

    def reset(self):
        self._amr_fields: list[int] = []
        self._amr_coords: np.ndarray = None
        self._amr_sizes: np.ndarray = None
        self._amr_ratios: np.ndrray = None
        self._amr_new: np.ndarray = None
        self._reset_amr_points()

    @property
    def npts(self) -> int:
        return self._amr_coords.shape[1]

    def reduce_set(self, ids: np.ndarray) -> None:
        self._amr_coords = self._amr_coords[:, ids]
        self._amr_sizes = self._amr_sizes[ids]
        self._amr_ratios = self._amr_ratios[ids]
        self._amr_new = self._amr_new[ids]

    def _reset_amr_points(self) -> None:
        for tag in self._amr_fields:
            gmsh.model.mesh.field.remove(tag)
        self._amr_fields = []

    def add_refinement_points(
        self, coords: np.ndarray, sizes: np.ndarray, ratios: np.ndarray
    ):
        if self._amr_coords is None:
            self._amr_coords = coords
        else:
            self._amr_coords = np.hstack((self._amr_coords, coords))

        if self._amr_sizes is None:
            self._amr_sizes = sizes
        else:
            self._amr_sizes = np.hstack((self._amr_sizes, sizes))

        if self._amr_ratios is None:
            self._amr_ratios = ratios
        else:
            self._amr_ratios = np.hstack((self._amr_ratios, ratios))

        if self._amr_new is None:
            self._amr_new = np.ones_like(sizes)
        else:
            self._amr_new = np.hstack((0.0 * self._amr_new, np.ones_like(sizes)))

    def set_refinement_function(self, gr: float = 1.5, _qf: float = 1.0):
        xs = self._amr_coords[0, :]
        ys = self._amr_coords[1, :]
        zs = self._amr_coords[2, :]
        newsize = self._amr_ratios * self._amr_sizes
        A = newsize / gr
        B = (1 - gr) / gr

        from numba import njit, i8, f8

        @njit(f8(i8, i8, f8, f8, f8, f8), nogil=True, fastmath=True, parallel=False)
        def func(dim, tag, x, y, z, lc):
            sizes = np.maximum(
                newsize,
                A - B * _qf * np.sqrt((x - xs) ** 2 + (y - ys) ** 2 + (z - zs) ** 2),
            )
            return min(lc, float(np.min(sizes)))

        gmsh.model.mesh.setSizeCallback(func)

    def set_ratio(self, ratio: float) -> None:
        newids = self._amr_new == 1
        self._amr_ratios[newids] = ratio
        return self._amr_ratios[newids][0]


class Mesher:
    def __init__(self):
        self.objects: list[GeoObject] = []
        self.size_definitions: list[tuple[int, float]] = []
        self.mesh_fields: list[int] = []
        self.max_mesh_fields: list[int] = []
        self.curved_boundary_segments: float = 10.0

        self._coarse_tags: list[int] = []

        self._amr_fields: list[int] = []
        self._amr_coords: np.ndarray = None
        self._amr_sizes: np.ndarray = None
        self._amr_ratios: np.ndrray = None
        self._amr_new: np.ndarray = None
        self._amrobj = AMRPoints()
        self.min_size: float = None
        self.max_size: float = None
        self.periodic_cell: PeriodicCell = None

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
    def volumes(self) -> list[GeoVolume]:
        return [
            obj for obj in self.objects if isinstance(obj, GeoVolume) and obj._exists
        ]

    @property
    def domain_boundary_face_tags(self) -> list[int]:
        """Get the face tags of the domain boundaries"""
        domain_tags = gmsh.model.getEntities(3)
        tags = gmsh.model.getBoundary(domain_tags, combined=True, oriented=False)
        return [int(tag[1]) for tag in tags]

    @property
    def domain_internal_face_tags(self) -> list[int]:
        alltags = self.face_tags
        boundary = self.domain_boundary_face_tags
        return [tag for tag in alltags if tag not in boundary]

    def _get_periodic_bcs(self) -> list[Periodic]:
        if self.periodic_cell is None:
            return []
        return self.periodic_cell._bcs

    def _check_ready(self) -> None:
        if self.max_size is None or self.min_size is None:
            raise MeshError(
                "Either maximum or minimum mesh size is undefined. Make sure \
                            to set the simulation frequency range before calling mesh instructions."
            )

    def submit_objects(
        self, objects: GeoObject | list[GeoObject] | list[list[GeoObject]]
    ) -> None:
        """Takes al ist of GeoObjects and computes the fragment.

        Args:
            objects (list[GeoObject]): The set of GeoObjects
        """
        if not isinstance(objects, list):
            objects = [
                objects,
            ]

        objects = unpack_lists(objects)
        embeddings: list = []
        gmsh.model.occ.synchronize()

        final_dimtags = unpack_lists([domain.dimtags for domain in objects])  # type: ignore

        dom_mapping = dict()
        for dom in objects:  # type: ignore
            embeddings.extend(dom._embeddings)  # type: ignore
            for dt in dom.dimtags:  # type: ignore
                dom_mapping[dt] = dom

        embedding_dimtags = unpack_lists([emb.dimtags for emb in embeddings])

        tag_mapping: dict[int, dict] = {0: dict(), 1: dict(), 2: dict(), 3: dict()}
        if len(objects) > 0:  # type: ignore
            dimtags, output_mapping = gmsh.model.occ.fragment(
                final_dimtags, embedding_dimtags
            )
            for domain, mapping in zip(
                final_dimtags + embedding_dimtags, output_mapping
            ):
                tag_mapping[domain[0]][domain[1]] = [o[1] for o in mapping]
            for dom in objects:  # type: ignore
                dom.update_tags(tag_mapping)  # type: ignore
        else:
            dimtags = final_dimtags

        self.objects = objects  # type: ignore

        gmsh.model.occ.synchronize()

    def _set_mesh_periodicity(
        self, face1: Selection, face2: Selection, lattice: np.ndarray
    ):
        translation = [
            1,
            0,
            0,
            lattice[0],
            0,
            1,
            0,
            lattice[1],
            0,
            0,
            1,
            lattice[2],
            0,
            0,
            0,
            1,
        ]
        gmsh.model.mesh.set_periodic(2, face2.tags, face1.tags, translation)

    def set_algorithm(self, algorithm: Algorithm3D) -> None:

        gmsh.option.setNumber("General.NumThreads", 16)
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm.value)

    def set_periodic_cell(
        self, cell: PeriodicCell, excluded_faces: Selection | None = None
    ):
        """Sets the periodic cell information based on the PeriodicCell class object"""
        if excluded_faces is None:
            for f1, f2, lat in cell.cell_data():
                self._set_mesh_periodicity(f1, f2, lat)
        else:
            for f1, f2, lat in cell.cell_data():
                self._set_mesh_periodicity(
                    f1 - excluded_faces, f2 - excluded_faces, lat
                )
        self.periodic_cell = cell

    def _set_size_in_domain(self, tags: list[int], max_size: float) -> None:
        """Define the size of the mesh inside a domain

        Args:
            tags (list[int]): The tags of the geometry
            max_size (float): The maximum size (in meters)
        """
        ctag = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.set_numbers(ctag, "VolumesList", tags)
        gmsh.model.mesh.field.set_number(ctag, "VIn", max_size)
        self.mesh_fields.append(ctag)

    def _set_size_on_face(
        self, tags: list[int], max_size: float, set_max: bool = False
    ) -> None:
        """Define the size of the mesh on a face

        Args:
            tags (list[int]): The tags of the geometry
            max_size (float): The maximum size (in meters)
        """
        ctag = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.set_numbers(ctag, "SurfacesList", tags)
        gmsh.model.mesh.field.set_number(ctag, "VIn", max_size)
        if not set_max:
            self.mesh_fields.append(ctag)
        else:
            gmsh.model.mesh.field.set_number(ctag, "IncludeBoundary", 0)
            gmsh.model.mesh.field.set_number(ctag, "VOut", 1e-9)
            self.max_mesh_fields.append(ctag)

    def _set_size_on_edge(self, tags: list[int], max_size: float) -> None:
        """Define the size of the mesh on an edge

        Args:
            tags (list[int]): The tags of the geometry
            max_size (float): The maximum size (in meters)
        """
        ctag = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.set_numbers(ctag, "CurvesList", tags)
        gmsh.model.mesh.field.set_number(ctag, "VIn", max_size)
        self.mesh_fields.append(ctag)

    def _reset_amr_points(self) -> None:
        for tag in self._amr_fields:
            gmsh.model.mesh.field.remove(tag)
        self._amr_fields = []

    def _configure_bc_size(self, bcs: list[BoundaryCondition]) -> None:
        """Writes size constraints for the boundary conditions

        Args:
            bcs (list[BoundaryCondition]): List of all the boundary conditions
        """
        for bc in bcs:
            if bc._size_constraint is None:
                continue
            if bc.dim != 2:
                continue
            size = bc._size_constraint
            logger.debug(
                f"Setting size constraint for {bc} of size {size * 1000:.2f}mm"
            )
            self.set_face_size(bc.selection, size)

    def _configure_coarse_size(self):

        tag = gmsh.model.mesh.field.add("Max")
        maxval_tag = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(maxval_tag, "F", "100")
        gmsh.model.mesh.field.setNumbers(
            tag,
            "FieldsList",
            [
                maxval_tag,
            ],
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(tag)
        self._coarse_tags = (tag, maxval_tag)

    def _configure_mesh_size(self, discretizer: Callable, resolution: float):
        """Defines the mesh sizes based on a discretization callable.
        The discretizer must take a material and return a maximum
        size for that material.

        Args:
            discretizer (Callable): The discretization function
            resolution (float): The resolution
        """
        logger.debug("Starting initial mesh size computation.")

        gmsh.option.setNumber(
            "Mesh.MeshSizeFromCurvature", self.curved_boundary_segments
        )

        for tag in self._coarse_tags:
            gmsh.model.mesh.field.remove(tag)
            self._coarse_tags = []

        dimtags = gmsh.model.occ.get_entities(2)
        for dim, tag in dimtags:
            gmsh.model.mesh.setSizeFromBoundary(2, tag, 0)

        mintag = gmsh.model.mesh.field.add("Min")

        size_mapping = dict()

        for obj in self.objects:
            if obj.dim != 2 and obj.max_meshsize < 1e9:
                continue
            self._set_size_on_face(obj.tags, obj.max_meshsize)

        for obj in sorted(self.objects, key=lambda x: x._priority):
            if obj.dim != 3:
                continue
            if obj._unset_constraints:
                self.unset_constraints(obj.dimtags)

            size = discretizer(obj.material) * resolution * obj.mesh_multiplier
            size = min(size, obj.max_meshsize)

            for dimtag in obj.dimtags:
                size_mapping[dimtag] = size

        for (dim, tag), size in size_mapping.items():
            
            if dim == 2:
                logger.debug(
                    f"Somehow setting mesh size:{1000 * size:.3f}mm on boundary: {tag}"
                )
                self._set_size_on_face(
                    [
                        tag,
                    ],
                    size,
                )
            if dim == 3:
                logger.debug(f"Setting mesh size:{1000 * size:.3f}mm in domains: {tag}")
                self._set_size_in_domain(
                    [
                        tag,
                    ],
                    size,
                )

        gmsh.model.mesh.field.setNumbers(
            mintag, "FieldsList", self.mesh_fields + self._amr_fields
        )

        if len(self.max_mesh_fields) > 0:
            maxtag = gmsh.model.mesh.field.add("Max")
            gmsh.model.mesh.field.setNumbers(
                maxtag,
                "FieldsList",
                self.max_mesh_fields
                + [
                    mintag,
                ],
            )
            gmsh.model.mesh.field.setAsBackgroundMesh(maxtag)
        else:
            gmsh.model.mesh.field.setAsBackgroundMesh(mintag)

        for tag, size in self.size_definitions:
            logger.debug(
                f"Setting aux size definition: {1000 * size:.3f}mm in domain {tag}."
            )
            gmsh.model.mesh.setSize(
                [
                    tag,
                ],
                size,
            )

    def _fix_curved_boundary_meshing(self) -> None:
        """Fix the meshing on curved boundaries by setting a finer mesh on edges"""
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 10)

    def set_curved_boundary_meshing(self, factor: float = 10) -> None:
        """Set the meshing on curved boundaries by setting a finer mesh on edges

        Args:
            factor (float, optional): The factor to reduce the mesh size on curved edges. Defaults to 10.
        """
        self.curved_boundary_segments = factor

    def unset_constraints(self, dimtags: list[tuple[int, int]]):
        """Unset the mesh constraints for the given dimension tags."""
        logger.trace(f"Unsetting mesh size constraint for domains: {dimtags}")
        for dimtag in dimtags:
            gmsh.model.mesh.setSizeFromBoundary(dimtag[0], dimtag[1], 0)

    def add_refinement_points(
        self, coords: np.ndarray, sizes: np.ndarray, ratios: np.ndarray
    ):
        if self._amr_coords is None:
            self._amr_coords = coords
        else:
            self._amr_coords = np.hstack((self._amr_coords, coords))

        if self._amr_sizes is None:
            self._amr_sizes = sizes
        else:
            self._amr_sizes = np.hstack((self._amr_sizes, sizes))

        if self._amr_ratios is None:
            self._amr_ratios = ratios
        else:
            self._amr_ratios = np.hstack((self._amr_ratios, ratios))

        if self._amr_new is None:
            self._amr_new = np.ones_like(sizes)
        else:
            self._amr_new = np.hstack((0.0 * self._amr_new, np.ones_like(sizes)))

    def _set_refinement_function(self, gr: float = 1.5, _qf: float = 1.0):
        """Define the refinement function based on AMR points. Not to be used by you.

        Args:
            gr (float, optional): The growth ratio . Defaults to 1.5.
            _qf (float, optional): The growth ratio scale factor. Defaults to 1.0.

        """
        xs = self._amr_coords[0, :]
        ys = self._amr_coords[1, :]
        zs = self._amr_coords[2, :]
        newsize = self._amr_ratios * self._amr_sizes
        A = newsize / gr
        B = (1 - gr) / gr
        from numba import njit, i8, f8

        @njit(f8(i8, i8, f8, f8, f8, f8), nogil=True, fastmath=True, parallel=False)
        def func(dim, tag, x, y, z, lc):
            sizes = np.maximum(
                newsize,
                A
                - B
                * _qf
                * np.clip(
                    np.sqrt((x - xs) ** 2 + (y - ys) ** 2 + (z - zs) ** 2)
                    - newsize * 0,
                    a_min=0,
                    a_max=None,
                ),
            )
            return min(lc, float(np.min(sizes)))

        gmsh.model.mesh.setSizeCallback(func)

    def _set_refinement_ratio(self, ratio: float) -> None:
        """Update the adaptive mesh refinement ratio. Not to be used by users.

        Args:
            ratio (float): The new refinement ratio

        """
        newids = self._amr_new == 1
        self._amr_ratios[newids] = ratio
        return self._amr_ratios[newids][0]

    def set_boundary_size(
        self,
        boundary: GeoObject | Selection | Iterable,
        size: float,
        growth_rate: float = 3,
        max_size: float | None = None,
    ) -> None:
        """Refine the mesh size along the boundary of a conducting surface

        The growth rate determines how quickly the mesh size is allowed to increase away from the face boundary.

        Args:
            boundary (GeoSurface | FaceSelection | Iterable): The boundary/surface object to refine the mesh on
            size (float): The mesh size limit in meters
            growth_rate (float, optional): The mesh growth rate. Defaults to 3.
            max_size (float, optional): The maximum mesh size. Defaults to None.
        """
        if isinstance(boundary, Iterable):
            for bound in boundary:
                self.set_boundary_size(bound, size, growth_rate, max_size)
            return

        dimtags = boundary.dimtags

        if max_size is None:
            self._check_ready()
            max_size = self.max_size

        growth_distance = (growth_rate * max_size - size) / (growth_rate - 1)
        logger.debug(
            f"Setting boundary size for region {dimtags} to {size * 1000:.3f}mm, GR={growth_rate:.3f}, dist={growth_distance * 1000:.2f}mm, Max={max_size * 1000:.3f}mm"
        )

        nodes = gmsh.model.getBoundary(
            dimtags, combined=False, oriented=False, recursive=False
        )

        disttag = gmsh.model.mesh.field.add("Distance")
        if boundary.dim == 2:
            gmsh.model.mesh.field.setNumbers(
                disttag, "CurvesList", [n[1] for n in nodes]
            )
        if boundary.dim == 3:
            gmsh.model.mesh.field.setNumbers(
                disttag, "SurfacesList", [n[1] for n in nodes]
            )
        gmsh.model.mesh.field.setNumber(disttag, "Sampling", 100)

        thtag = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thtag, "InField", disttag)
        gmsh.model.mesh.field.setNumber(thtag, "SizeMin", size)
        gmsh.model.mesh.field.setNumber(thtag, "SizeMax", max_size)
        gmsh.model.mesh.field.setNumber(thtag, "DistMin", size)
        gmsh.model.mesh.field.setNumber(thtag, "DistMax", growth_distance)

        self.mesh_fields.append(thtag)

    def set_trace_refinement(
        self,
        obj: GeoSurface | FaceSelection | Iterable,
        qualtity_factor: float = 3.0,
        growth_rate: float = 3.0,
        min_size: float = 1e-5,
        max_size: float | None = None,
        edge_size: float | None = None,
        min_overlap: float = 0.75,
    ):
        """Performs an intelligent size constraint refinement for stripline edge gaps based on the distance between them.

        This algorithm looks for parallel stripline edges and refines the mesh size around them iff the
        they are at minimum overlapping by at least 75%(default) and not too close.
        The mesh edge size constraint will be the distance between the line segments divided by the quality factor.
        Args:
            obj (GeoSurface | FaceSelection | Iterable): The trace or set of traces to refine
            qualtity_factor (float, optional): How much to refine. Greater means finer mesh. Defaults to 3.0.
            growth_rate (float, optional): How quickly the mesh size is allowed to grow. Defaults to 3.0.
            min_size (float, optional): The minimum size of the mesh in meters. Defaults to 1e-5.
            max_size (float | None, optional): The maximum size to constrain to. Defaults to None.
            min_overlap (float | optional): The minimum overlap for two edges to be refined. Defaults to 0.75 = 75%

        Raises:
            MeshError: _description_
        """
        from collections import defaultdict

        if max_size is None:
            self._check_ready()
            max_size = self.max_size

        if isinstance(obj, Iterable):
            from .geometry import select

            obj = select(*obj)

        if obj.dim != 2:
            raise MeshError(
                'Can only do trace refinement for 2D structures. In case of thick metal traces, use the bottom layer with .face("-z")'
            )

        edge_dimtags = gmsh.model.get_boundary(obj.dimtags, True)
        edges = [t for d, t in edge_dimtags if d == 1]
        edge_nodes: dict[int, tuple[int, int]] = {
            tag: gmsh.model.get_boundary(
                [
                    (1, tag),
                ]
            )
            for tag in edges
        }

        edges = dict()
        for edgetag, (dt1, dt2) in edge_nodes.items():
            xyz1 = np.array([gmsh.model.get_value(0, dt1[1], [])]).squeeze()
            xyz2 = np.array([gmsh.model.get_value(0, dt2[1], [])]).squeeze()
            edgedir = (xyz2 - xyz1) / np.linalg.norm(xyz2 - xyz1)
            edge_nodes[edgetag] = (xyz1, xyz2, edgedir)

        sizes = defaultdict(lambda: 1e9)
        for t1, (p11, p12, e1) in edge_nodes.items():
            for t2, (p21, p22, e2) in edge_nodes.items():
                if t1 == t2:
                    continue
                if np.abs(np.dot(e1, e2)) < 0.7:
                    continue
                # Lines are parallel
                a = e1
                b = p21 - p11
                du = np.cross(a, np.cross(a, b))
                du = du / (np.linalg.norm(du) + 1e-12)
                line_dist = np.abs(np.dot(du, b))
                if line_dist < min_size:
                    continue
                c1 = (p11 + p12) / 2
                c2 = (p21 + p22) / 2
                dl = np.linalg.norm(c2 - c1)
                l1 = np.linalg.norm(p12 - p11)
                l2 = np.linalg.norm(p22 - p21)
                if dl < 2 * l1 * (1 - min_overlap):
                    sizes[t1] = min(
                        sizes[t1], max(min_size, line_dist / qualtity_factor)
                    )
                if dl < 2 * l2 * (1 - min_overlap):
                    sizes[t2] = min(
                        sizes[t2], max(min_size, line_dist / qualtity_factor)
                    )
        
        if isinstance(edge_size, (float,int)):
            self.set_boundary_size(obj, edge_size, growth_rate=growth_rate)

        for edge_tag, size in sizes.items():
            if size == 1e9:
                continue
            growth_distance = (growth_rate * max_size - size) / (growth_rate - 1)

            disttag = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(
                disttag,
                "CurvesList",
                [
                    edge_tag,
                ],
            )
            gmsh.model.mesh.field.setNumber(disttag, "Sampling", 100)

            thtag = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thtag, "InField", disttag)
            gmsh.model.mesh.field.setNumber(thtag, "SizeMin", size)
            gmsh.model.mesh.field.setNumber(thtag, "SizeMax", max_size)
            gmsh.model.mesh.field.setNumber(thtag, "DistMin", size / 2)
            gmsh.model.mesh.field.setNumber(thtag, "DistMax", growth_distance)
            self.mesh_fields.append(thtag)

    def set_domain_size(self, obj: GeoObject | Selection, size: float):
        """Manually set the maximum element size inside a domain

        Args:
            obj (GeoVolume | Selection): The volumetric domain
            size (float): The maximum mesh size
        """
        if obj.dim != 3:
            logger.warning("Provided object is not a volume.")
            if obj.dim == 2:
                logger.warning("Forwarding to set_face_size")
                self.set_face_size(obj, size)
                return
        logger.debug(f"Setting size {size * 1000:.3f}mm for object {obj}")
        self._set_size_in_domain(obj.tags, size)

    def set_face_size(self, obj: GeoSurface | Selection, size: float):
        """Manually set the maximum element size on a face

        Args:
            obj (GeoSurface | Selection): The surface domain
            size (float): The maximum size
        """
        if obj.dim != 2:
            logger.warning("Provided object is not a surface.")
            if obj.dim == 3:
                logger.warning("Forwarding to set_domain_size")
                self.set_domain_size(obj, size)
                return

        logger.debug(f"Setting size {size * 1000:.3f}mm for face {obj}")
        self._set_size_on_face(obj.tags, size)

    def set_pec_face(self, obj: GeoSurface | Selection, size: float = -1) -> None:
        """Sets the size on PEC faces to be very coarse to save computational time.

        Args:
            obj (GeoSurface | Selection): The surface domain
            size (float): The maximum size
        """
        if size < 0:
            size = self.max_size
        if obj.dim != 2:
            logger.warning("Provided object is not a surface.")
            if obj.dim == 3:
                logger.warning("Forwarding to set_domain_size")
                self.set_face_size(obj, size)

        logger.debug(f"Setting PEC size {size * 1000:.3f}mm for face {obj}")
        self._set_size_on_face(obj.tags, size, True)

    def set_size(self, obj: GeoObject, size: float) -> None:
        """Manually set the size in or on an object

        Args:
            obj (GeoObject): The object to change the size for
            size (float): The maximum mesh edge size
        """
        if obj.dim == 2:
            self._set_size_on_face(obj.tags, size)
        elif obj.dim == 3:
            self._set_size_in_domain(obj.tags, size)
        elif obj.dim == 1:
            self._set_size_on_edge(obj.tags, size)
        elif obj.dim == 0:
            self._set_size_on_point(obj.tags, size)

    def refine_conductor_edge(self, dimtags: list[tuple[int, int]], size):
        nodes = gmsh.model.getBoundary(dimtags, combined=False, recursive=False)

        # for node in nodes:
        #     pcoords = np.linspace(0, 0.5, 10)
        #     gmsh.model.mesh.setSizeAtParametricPoints(node[0], node[1], pcoords, size*np.ones_like(pcoords))
        #     #self.size_definitions.append((node, size))
        # gmsh.model.mesh.setSizeFromBoundary(dimtag[0], dimtag[1], 0)

        tag = gmsh.model.mesh.field.add("Distance")

        # gmsh.model.mesh.field.setNumbers(1, "PointsList", [5])
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
        gmsh.model.mesh.field.setNumber(thtag, "DistMin", 0.2 * size)
        gmsh.model.mesh.field.setNumber(thtag, "DistMax", 5 * size)

        self.mesh_fields.append(thtag)

        for dimtag in dimtags:
            gmsh.model.mesh.setSizeFromBoundary(dimtag[0], dimtag[1], 0)
