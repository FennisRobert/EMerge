"""AURA: Adaptive Unstructured Refinement Algorithm for EMerge (Rust core in emr/src/aura).

Local, conforming refinement of tetrahedral meshes by longest-edge bisection,
followed by flip/smoothing quality improvement. This package does not depend
on the rest of EMerge; :mod:`.emerge_tools` holds the helpers that work on
EMerge ``Mesh3D`` objects.

Conventions of this module: arrays are row-major, i.e. ``nodes`` is ``(N, 3)``
and ``tets`` is ``(M, 4)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from emr import aura as _aura
from .brep import BrepSnapshot

__all__ = ["BrepSnapshot", "RefineResult", "refine", "quality"]


@dataclass
class RefineResult:
    """Output of :func:`refine`.

    The first ``N_old`` nodes are the original nodes, unchanged. New nodes are appended.
    """

    nodes: np.ndarray  # (N', 3) float64
    tets: np.ndarray  # (M', 4) int64, same orientation convention as the input
    tet_root: np.ndarray  # (M',) an original tet this tet descends from (for flipped tets: one of them)
    tet_region: np.ndarray  # (M',) region (volume tag) of each tet, inherited from the input
    tet_new: np.ndarray  # (M',) bool, True for tets created or modified in this pass
    face_tris: np.ndarray  # (F, 3) int64, sorted node ids of every tagged triangle
    face_tri_tags: np.ndarray  # (F,) face tag per triangle (a triangle may appear once per tag)
    curve_edges: np.ndarray  # (C, 2) int64, sorted node ids of every tagged curve edge
    curve_edge_tags: np.ndarray  # (C,) curve tag per edge
    stats: dict = field(default_factory=dict)


def refine(
    nodes: np.ndarray,
    tets: np.ndarray,
    error: np.ndarray,
    tet_region: np.ndarray | None = None,
    face_tris: np.ndarray | None = None,
    face_tri_tags: np.ndarray | None = None,
    curve_edges: np.ndarray | None = None,
    curve_edge_tags: np.ndarray | None = None,
    periodic_pairs: np.ndarray | None = None,
    brep: BrepSnapshot | None = None,
    growth: float = 0.3,
    order: float = 2.0,
    max_depth: int = 6,
    min_projected_volume: float = 0.2,
    lepp_ratio: float = 0.7,
    improve_sweeps: int = 3,
    flip_quality: float = 0.6,
    smooth_iters: int = 2,
    improve_layers: int = 2,
    improve_global: bool = False,
    strip_ratio: float = 0.0,
    strip_max_growth: float = 2.0,
    strip_gated: bool = True,
) -> RefineResult:
    """Refine a tetrahedral mesh locally until it has ``(1 + growth)`` times as many tets.

    Tets are refined greedily in order of decreasing error. Each bisection gives
    both children the parent error times ``2**-(order/3 + 1/2)`` (the expected
    reduction of an h^order error indicator integrated over half the volume), so
    a high-error tet can be bisected several times in one pass while refinement
    still spreads to the next-worst regions.

    With ``strip_ratio > 0`` curve edges along narrow strips (e.g. thin PCB traces)
    are split until they are at most ``strip_ratio`` times the local strip width, so
    the triangles across the strip are not slivers. With ``strip_gated`` (default)
    this happens after the error-driven refinement and only for strips next to tets
    whose error reached the level of the tets refined in this pass; otherwise it is a
    purely geometric pass over all narrow strips, run first.

    Afterwards the new tets and ``improve_layers`` rings of neighbouring old tets
    are improved by 2-3 flips, edge removal (3 to 7 tets around an edge) and smart
    Laplacian smoothing of free interior nodes. Each operation must strictly
    improve the worst tet it touches. Tagged faces, tagged curve edges, their
    nodes and region boundaries are never changed.

    Args:
        nodes: (N, 3) node coordinates.
        tets: (M, 4) node indices per tetrahedron.
        error: (M,) error indicator per tetrahedron.
        tet_region: (M,) region id per tet (e.g. the volume tag). Flips never
            mix regions, and smoothing never moves nodes on region boundaries.
        face_tris / face_tri_tags: triangles lying on tagged geometric faces.
        curve_edges / curve_edge_tags: edges lying on tagged geometric curves.
        periodic_pairs: (P, 3) rows ``(k, source_node, target_node)`` for periodic
            boundary ``k``. Paired faces are refined identically, the maps stay order
            preserving, and paired nodes are never moved by smoothing.
        brep: Optional geometry snapshot used to project new boundary nodes.
        growth: Target relative increase of the number of tetrahedra (0.3 = +30 %).
        order: Assumed convergence order of the error indicator.
        max_depth: Maximum bisection generations per pass for a selected tet.
        min_projected_volume: Minimum child volume, as a fraction of an exact half,
            accepted when projecting a new node onto curved geometry.
        lepp_ratio: Neighbours are bisected through a shared edge that is not their
            own longest edge if it is at least this fraction of it. 1.0 gives strict
            longest-edge propagation (best quality); lower values keep refinement
            more local.
        improve_sweeps: Number of flip + smoothing sweeps (0 disables improvement).
        flip_quality: Only tets with a mean ratio below this are flip candidates.
        smooth_iters: Smoothing iterations per sweep.
        improve_layers: Rings of node-adjacent old tets around the refined region that
            are improved as well (where new and old elements meet).
        improve_global: Improve the whole mesh instead of only the refined region.
        strip_ratio: Step 0 narrow-strip regularisation: maximum curve-edge length along
            a narrow strip relative to its width (e.g. 4). 0 disables it.
        strip_max_growth: Strip regularisation stops once it has grown the tet count by
            this fraction.
        strip_gated: Only regularise strips in high-error regions (see above).
    """
    nodes = np.ascontiguousarray(nodes, dtype=np.float64)
    tets = np.ascontiguousarray(tets, dtype=np.int64)
    error = np.ascontiguousarray(error, dtype=np.float64)
    if tet_region is None:
        tet_region = np.zeros(len(tets), dtype=np.int32)
    if face_tris is None:
        face_tris = np.zeros((0, 3), dtype=np.int64)
        face_tri_tags = np.zeros((0,), dtype=np.int32)
    if curve_edges is None:
        curve_edges = np.zeros((0, 2), dtype=np.int64)
        curve_edge_tags = np.zeros((0,), dtype=np.int32)
    if brep is None:
        brep = BrepSnapshot.empty()
    if periodic_pairs is None:
        periodic_pairs = np.zeros((0, 3), dtype=np.int64)

    out = _aura.refine(
        nodes,
        tets,
        error,
        np.ascontiguousarray(tet_region, dtype=np.int32),
        np.ascontiguousarray(face_tris, dtype=np.int64).reshape(-1, 3),
        np.ascontiguousarray(face_tri_tags, dtype=np.int32),
        np.ascontiguousarray(curve_edges, dtype=np.int64).reshape(-1, 2),
        np.ascontiguousarray(curve_edge_tags, dtype=np.int32),
        np.ascontiguousarray(periodic_pairs, dtype=np.int64).reshape(-1, 3),
        *brep.as_args(),
        float(growth),
        float(2.0 ** -(order / 3.0 + 0.5)),
        int(max_depth),
        float(min_projected_volume),
        float(lepp_ratio),
        int(improve_sweeps),
        float(flip_quality),
        int(smooth_iters),
        int(improve_layers),
        bool(improve_global),
        float(strip_ratio),
        float(strip_max_growth),
        bool(strip_gated),
    )
    return RefineResult(**out)


def quality(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Mean-ratio quality per tetrahedron: 1 for a regular tet, 0 for a degenerate one."""
    return _aura.quality(
        np.ascontiguousarray(nodes, dtype=np.float64),
        np.ascontiguousarray(tets, dtype=np.int64),
    )
