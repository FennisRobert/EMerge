"""Helpers that operate on EMerge ``Mesh3D`` objects.

Every function receives the EMerge objects it works on. EMerge's
``Simulation.adaptive_mesh_refinement`` drives the AMR loop and calls into this module:

* :func:`refine_mesh_step`    One complete refinement: Mesh3D + error -> new Mesh3D.
* :func:`build_topology`      GMSH-free port of the topology part of ``Mesh3D._pre_update``.
* :func:`mesh_to_aura_inputs` Extract the arrays AURA needs from a ``Mesh3D``.
* :func:`refine_mesh3d`       Build a new ``Mesh3D`` from a :class:`RefineResult`.
* :func:`growth_for_pass`     Resolve the per-pass growth setting (fixed, list, "auto", callable).
* :func:`periodic_node_pairs` Pair the nodes of periodic faces.

Periodic boundaries are described as a list of ``(face1_tags, face2_tags, dv)``
with face2 = face1 + dv, as in EMerge's ``Periodic`` boundary conditions.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Union

import numpy as np

from . import RefineResult, quality, refine
from .brep import BrepSnapshot

# Order in which EMerge enumerates the 6 edges and 4 faces of a tet (0-based
# version of _idset1/_idset2 in Mesh3D._pre_update).
_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (3, 1), (2, 3))
_TET_TRIS = ((0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 2, 3))

# Geometry-level attributes that refinement does not change.
_GEOMETRY_ATTRS = (
    "dimtag_to_center",
    "dimtag_to_bb",
    "dimtag_to_nodes",
    "ftag_to_normal",
    "ftag_to_point",
    "exterior_face_tags",
)


# -----------------------------------------------------------------------------
# Mesh3D topology without GMSH
# -----------------------------------------------------------------------------


def build_topology(mesh: Any, nodes: np.ndarray, tets: np.ndarray) -> None:
    """Fill all connectivity and derived quantities of ``mesh`` from nodes/tets.

    Produces exactly the arrays and dicts ``Mesh3D._pre_update`` produces in its
    "Extracting Edges and Triangles" ... "Mesh Quantities" sections, but
    vectorised and from plain arrays instead of the GMSH model.

    Args:
        mesh: An (empty) EMerge ``Mesh3D``.
        nodes: (3, N) coordinates.
        tets: (4, M) node indices.
    """
    MISSING = mesh._MISSING_ID
    nodes = np.ascontiguousarray(nodes, dtype=np.float64)
    tets = np.ascontiguousarray(tets, dtype=np.int64)
    nN, nT = nodes.shape[1], tets.shape[1]

    mesh.nodes = nodes
    mesh.tets = tets
    mesh.centers = (nodes[:, tets[0]] + nodes[:, tets[1]] + nodes[:, tets[2]] + nodes[:, tets[3]]) / 4

    # Edges: unique sorted node pairs, lexicographically ordered (== sorted(set)).
    pe = np.stack([np.sort(np.stack([tets[i], tets[j]]), axis=0) for i, j in _TET_EDGES])  # (6, 2, M)
    pe = pe.transpose(0, 2, 1).reshape(-1, 2)  # row k*M + itet
    edges, e_inv = np.unique(pe, axis=0, return_inverse=True)
    mesh.edges = edges.T.copy()
    mesh.tet_to_edge = e_inv.reshape(6, nT)

    # Triangles: unique sorted node triples.
    pt = np.stack([np.sort(np.stack([tets[i], tets[j], tets[k]]), axis=0) for i, j, k in _TET_TRIS])
    pt = pt.transpose(0, 2, 1).reshape(-1, 3)
    tris, t_inv = np.unique(pt, axis=0, return_inverse=True)
    mesh.tris = tris.T.copy()
    mesh.tet_to_tri = t_inv.reshape(4, nT)
    nE, nR = mesh.edges.shape[1], mesh.tris.shape[1]

    mesh.inv_edges = {(a, b): i for i, (a, b) in enumerate(edges.tolist())}
    mesh.inv_tris = {(a, b, c): i for i, (a, b, c) in enumerate(tris.tolist())}
    mesh.inv_tets = {tuple(t): i for i, t in enumerate(np.sort(tets, axis=0).T.tolist())}

    # Tri -> Tet (same algorithm as _pre_update)
    flat_tris = mesh.tet_to_tri.ravel(order="F")
    flat_tets = np.repeat(np.arange(nT, dtype=int), 4)
    sort_idx = np.argsort(flat_tris, stable=True)
    sorted_tris, sorted_tets = flat_tris[sort_idx], flat_tets[sort_idx]
    mesh.tri_to_tet = np.full((2, nR), MISSING, dtype=int)
    _, first_idx, counts = np.unique(sorted_tris, return_index=True, return_counts=True)
    mesh.tri_to_tet[0, sorted_tris[first_idx]] = sorted_tets[first_idx]
    mask2 = counts == 2
    mesh.tri_to_tet[1, sorted_tris[first_idx[mask2]]] = sorted_tets[first_idx[mask2] + 1]

    # Tri -> Edge
    keys = edges[:, 0].astype(np.int64) * nN + edges[:, 1]  # already sorted ascending

    def _lookup_edge(a, b):
        k = np.minimum(a, b).astype(np.int64) * nN + np.maximum(a, b)
        pos = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
        return np.where(keys[pos] == k, pos, MISSING)

    mesh.tri_to_edge = np.zeros((3, nR), dtype=int)
    mesh.tri_to_edge[0] = _lookup_edge(mesh.tris[0], mesh.tris[1])
    mesh.tri_to_edge[1] = _lookup_edge(mesh.tris[1], mesh.tris[2])
    mesh.tri_to_edge[2] = _lookup_edge(mesh.tris[0], mesh.tris[2])

    # Edge -> Tri
    flat_edges = mesh.tri_to_edge.ravel(order="F")
    flat_tris = np.repeat(np.arange(nR, dtype=int), 3)
    sort_idx = np.argsort(flat_edges, stable=True)
    sorted_edges, sorted_tris_e = flat_edges[sort_idx], flat_tris[sort_idx]
    _, first_idx, counts = np.unique(sorted_edges, return_index=True, return_counts=True)
    max_count = int(counts.max())
    mesh.edge_to_tri = np.full((max_count, nE), MISSING, dtype=int)
    for k in range(max_count):
        m = counts > k
        mesh.edge_to_tri[k, sorted_edges[first_idx[m]]] = sorted_tris_e[first_idx[m] + k]

    # Node -> Edge
    flat_nodes = mesh.edges.ravel(order="F")
    flat_edges = np.repeat(np.arange(nE, dtype=int), 2)
    sort_idx = np.argsort(flat_nodes, stable=True)
    sorted_nodes, sorted_edges = flat_nodes[sort_idx], flat_edges[sort_idx]
    _, first_idx, counts = np.unique(sorted_nodes, return_index=True, return_counts=True)
    max_count = int(counts.max())
    mesh.node_to_edge = np.full((max_count, nN), MISSING, dtype=int)
    for k in range(max_count):
        m = counts > k
        mesh.node_to_edge[k, sorted_nodes[first_idx[m]]] = sorted_edges[first_idx[m] + k]

    # Derived quantities
    t = mesh.tris
    mesh.tri_centers = (nodes[:, t[0]] + nodes[:, t[1]] + nodes[:, t[2]]) / 3
    mesh.edge_centers = (nodes[:, mesh.edges[0]] + nodes[:, mesh.edges[1]]) / 2
    mesh.edge_lengths = np.sqrt(np.sum((nodes[:, mesh.edges[0]] - nodes[:, mesh.edges[1]]) ** 2, axis=0))
    e1 = nodes[:, t[1]] - nodes[:, t[0]]
    e2 = nodes[:, t[2]] - nodes[:, t[0]]
    mesh.areas = np.linalg.norm(np.cross(e1.T, e2.T), axis=1) / 2


# -----------------------------------------------------------------------------
# Mesh3D <-> AURA
# -----------------------------------------------------------------------------


def mesh_to_aura_inputs(mesh: Any, periodic: Periodic | None = None) -> dict[str, np.ndarray]:
    """Extract nodes, tets, regions, tagged faces/curves and periodic node pairs from a Mesh3D."""
    face_tris, face_tags = [], []
    for tag, tri_ids in mesh.ftag_to_tri.items():
        ids = np.asarray([i for i in tri_ids if i is not None], dtype=np.int64)
        if ids.size == 0:
            continue
        face_tris.append(mesh.tris[:, ids].T)
        face_tags.append(np.full(ids.size, tag, dtype=np.int32))
    curve_edges, curve_tags = [], []
    for tag, edge_ids in mesh.etag_to_edge.items():
        ids = np.asarray(edge_ids, dtype=np.int64)
        if ids.size == 0:
            continue
        curve_edges.append(mesh.edges[:, ids].T)
        curve_tags.append(np.full(ids.size, tag, dtype=np.int32))

    def cat(lst, shape, dtype):
        return np.concatenate(lst).astype(dtype) if lst else np.zeros(shape, dtype=dtype)

    return dict(
        nodes=mesh.nodes.T,
        tets=mesh.tets.T,
        tet_region=mesh._get_tet_to_tag().astype(np.int32),
        face_tris=cat(face_tris, (0, 3), np.int64),
        face_tri_tags=cat(face_tags, (0,), np.int32),
        curve_edges=cat(curve_edges, (0, 2), np.int64),
        curve_edge_tags=cat(curve_tags, (0,), np.int32),
        periodic_pairs=periodic_node_pairs(mesh, periodic) if periodic else np.zeros((0, 3), dtype=np.int64),
    )


def refine_mesh3d(mesh: Any, result: RefineResult) -> Any:
    """Return a new Mesh3D equal to ``mesh`` refined according to ``result``.

    Every mapping that ``_pre_update`` derives from GMSH is reconstructed from
    the old mesh plus AURA's tag inheritance, so no GMSH call is made.
    """
    new = type(mesh)()
    for attr in _GEOMETRY_ATTRS:
        val = getattr(mesh, attr)
        setattr(new, attr, val.copy() if hasattr(val, "copy") else val)

    build_topology(new, result.nodes.T, result.tets.T)
    nN, nT = new.nodes.shape[1], new.tets.shape[1]

    # Node ids are stable (new nodes are appended), so node tags carry over.
    new.n_i2t = dict(mesh.n_i2t)
    next_tag = max(mesh.n_i2t.values(), default=0) + 1
    for i in range(mesh.nodes.shape[1], nN):
        new.n_i2t[i] = next_tag
        next_tag += 1
    new.n_t2i = {t: i for i, t in new.n_i2t.items()}
    new.tet_i2t = {i: i + 1 for i in range(nT)}
    new.tet_t2i = {t: i for i, t in new.tet_i2t.items()}

    # Volumes: every tet carries the volume tag it was refined/flipped within.
    region = np.asarray(result.tet_region)
    new.vtag_to_tet = {tag: np.flatnonzero(region == tag).tolist() for tag in mesh.vtag_to_tet}

    # Faces
    ftags = np.asarray(result.face_tri_tags)
    tri_ids = np.array([new.inv_tris[tuple(f)] for f in result.face_tris.tolist()], dtype=np.int64)
    new.ftag_to_tri, new.ftag_to_node, new.ftag_to_edge = {}, {}, {}
    for tag in mesh.ftag_to_tri:
        ids = tri_ids[ftags == tag]
        new.ftag_to_tri[tag] = ids.tolist()
        new.ftag_to_node[tag] = new.tris[:, ids].T.ravel().tolist()
        new.ftag_to_edge[tag] = sorted(np.unique(new.tri_to_edge[:, ids]).tolist())
    new.tri_i2t = {int(i): k + 1 for k, i in enumerate(np.unique(tri_ids))}
    new.tri_t2i = {t: i for i, t in new.tri_i2t.items()}

    # Curves
    ctags = np.asarray(result.curve_edge_tags)
    edge_ids = np.array([new.inv_edges[tuple(e)] for e in result.curve_edges.tolist()], dtype=np.int64)
    new.etag_to_edge = {tag: edge_ids[ctags == tag].tolist() for tag in mesh.etag_to_edge}
    new.edge_i2t = {int(i): k + 1 for k, i in enumerate(np.unique(edge_ids))}
    new.edge_t2i = {t: i for i, t in new.edge_i2t.items()}
    curve_nodes = new.edges[:, edge_ids].ravel() if edge_ids.size else np.zeros(0, dtype=int)
    new.geonodes = sorted(set(mesh.geonodes) | set(curve_nodes.tolist()))

    # dimtag -> edges: the edges of the bounding curves of each entity. Recover
    # which curves bound each entity from the old mesh, then take their new edges.
    old_edge_curves: dict[int, list[int]] = {}
    for tag, ids in mesh.etag_to_edge.items():
        for i in ids:
            old_edge_curves.setdefault(int(i), []).append(tag)
    new.dimtag_to_edges = {}
    for dt, old_ids in mesh.dimtag_to_edges.items():
        if dt[0] == 0:
            new.dimtag_to_edges[dt] = old_ids
            continue
        curves = sorted({c for i in np.asarray(old_ids, dtype=int).tolist() for c in old_edge_curves.get(i, [])})
        new.dimtag_to_edges[dt] = np.array(
            [e for c in curves for e in new.etag_to_edge[c]], dtype=np.int64
        )

    new.defined = True
    new._quick_mesh = False
    return new


# -----------------------------------------------------------------------------
# Growth control
# -----------------------------------------------------------------------------

Periodic = Sequence[tuple[Sequence[int], Sequence[int], np.ndarray]]


def _match_points(src: np.ndarray, dst: np.ndarray, tol: float) -> np.ndarray:
    """For each row of ``src`` the index of the row of ``dst`` within ``tol`` (or -1)."""
    cell = 4 * tol
    keys = np.floor(dst / cell).astype(np.int64)
    grid: dict[tuple[int, int, int], list[int]] = {}
    for j, k in enumerate(map(tuple, keys.tolist())):
        grid.setdefault(k, []).append(j)
    out = np.full(len(src), -1, dtype=np.int64)
    offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for i, (p, k) in enumerate(zip(src, np.floor(src / cell).astype(np.int64).tolist())):
        best, bd = -1, tol * tol
        for o in offsets:
            for j in grid.get((k[0] + o[0], k[1] + o[1], k[2] + o[2]), ()):
                d = float(np.sum((dst[j] - p) ** 2))
                if d <= bd:
                    best, bd = j, d
        out[i] = best
    return out


def _face_nodes(mesh: Any, tags: Sequence[int]) -> np.ndarray:
    tris = [i for t in tags for i in mesh.ftag_to_tri[t]]
    return np.unique(mesh.tris[:, np.asarray(tris, dtype=np.int64)])


def periodic_node_pairs(mesh: Any, periodic: Periodic, rel_tol: float = 1e-9) -> np.ndarray:
    """Pair the nodes of every periodic face pair.

    Returns:
        (P, 3) int64 rows ``(k, source_node, target_node)`` for periodic boundary ``k``.
    """
    tol = rel_tol * float(np.linalg.norm(np.ptp(mesh.nodes, axis=1)))
    rows = []
    for k, (tags1, tags2, dv) in enumerate(periodic):
        n1, n2 = _face_nodes(mesh, tags1), _face_nodes(mesh, tags2)
        idx = _match_points(mesh.nodes[:, n1].T + np.asarray(dv, dtype=np.float64), mesh.nodes[:, n2].T, tol)
        if len(n1) != len(n2) or np.any(idx < 0) or len(np.unique(idx)) != len(idx):
            raise ValueError(
                f"Periodic boundary {k}: faces {list(tags1)} and {list(tags2)} do not have matching meshes "
                f"({len(n1)} vs {len(n2)} nodes, {int(np.sum(idx < 0))} unmatched)."
            )
        rows.append(np.stack([np.full(len(n1), k), n1, n2[idx]], axis=1))
    return np.concatenate(rows).astype(np.int64) if rows else np.zeros((0, 3), dtype=np.int64)


def periodic_report(mesh: Any, periodic: Periodic) -> list[dict]:
    """Check periodic faces: matching triangles and an order preserving node map.

    Returns one dict per boundary with the triangle counts, the largest distance
    between a source triangle centre + dv and its partner, and whether the node map
    is order preserving (needed for consistent edge orientations in EMerge).
    """
    out = []
    pairs = periodic_node_pairs(mesh, periodic)
    for k, (tags1, tags2, dv) in enumerate(periodic):
        t1 = np.asarray([i for t in tags1 for i in mesh.ftag_to_tri[t]], dtype=np.int64)
        t2 = np.asarray([i for t in tags2 for i in mesh.ftag_to_tri[t]], dtype=np.int64)
        c1 = mesh.tri_centers[:, t1].T + np.asarray(dv)
        c2 = mesh.tri_centers[:, t2].T
        tol = 1e-9 * float(np.linalg.norm(np.ptp(mesh.nodes, axis=1)))
        idx = _match_points(c1, c2, tol)
        dist = np.linalg.norm(c1 - c2[idx], axis=1) if np.all(idx >= 0) else np.array([np.inf])
        p = pairs[pairs[:, 0] == k]
        order = np.argsort(p[:, 1])
        out.append(
            dict(
                n_tris=(len(t1), len(t2)),
                unmatched=int(np.sum(idx < 0)),
                max_center_distance=float(dist.max()) if dist.size else 0.0,
                order_preserving=bool(np.all(np.diff(p[order, 2]) > 0)),
            )
        )
    return out


GrowthSpec = Union[float, str, Sequence[float], Callable[[int, dict], float]]


def estimate_growth(error: np.ndarray, theta: float = 0.5, order: float = 2.0) -> float:
    """Predict the relative tet growth needed to equidistribute the error.

    Uses the same equidistribution model as EMerge's remeshing AMR: a tet with
    error e needs its size scaled by (e_target / e)^(1/order), with
    e_target = sqrt(theta * sum(e^2) / N), so it turns into
    max(1, (e / e_target)^(3/order)) tets (refinement only).
    """
    e = np.asarray(error, dtype=np.float64)
    e_target = np.sqrt(theta * np.sum(e**2) / e.size)
    factors = np.maximum(1.0, (e / max(e_target, 1e-300)) ** (3.0 / order))
    return float(np.sum(factors) / e.size - 1.0)


def growth_for_pass(
    growth: GrowthSpec,
    step: int,
    info: dict,
    theta: float = 0.5,
    order: float = 2.0,
    limits: tuple[float, float] = (0.1, 1.0),
) -> float:
    """Resolve the growth setting for one refinement pass.

    Args:
        growth: One of
            * a float: fixed growth every pass (0.3 = +30 %),
            * a sequence: growth per pass, the last value is repeated,
            * ``"auto"``: :func:`estimate_growth` from the current error indicator,
            * a callable ``f(step, info) -> float``; ``info`` holds ``n_tets``,
              ``error`` (per-tet indicator), ``convergence`` (last max |dS| or
              None) and ``auto`` (the value ``"auto"`` would give).
        step: 1-based pass number.
        info: See above.
        limits: (min, max) clamp applied to every resolved value.
    """
    if isinstance(growth, str):
        if growth != "auto":
            raise ValueError(f"Unknown growth mode {growth!r}; use a float, a list, a callable or 'auto'.")
        g = info["auto"]
    elif callable(growth):
        g = float(growth(step, info))
    elif isinstance(growth, (int, float)):
        g = float(growth)
    else:
        seq = list(growth)
        g = float(seq[min(step - 1, len(seq) - 1)])
    return float(np.clip(g, limits[0], limits[1]))


# -----------------------------------------------------------------------------
# One refinement step
# -----------------------------------------------------------------------------


def refine_mesh_step(
    mesh: Any,
    error: np.ndarray,
    brep: BrepSnapshot | None = None,
    growth: float = 0.3,
    order: float = 2.0,
    max_depth: int = 6,
    lepp_ratio: float = 0.7,
    improve_sweeps: int = 3,
    improve_layers: int = 2,
    improve_global: bool = False,
    strip_ratio: float = 0.0,
    strip_max_growth: float = 2.0,
    strip_gated: bool = True,
    periodic: Periodic | None = None,
) -> tuple[Any, dict]:
    """Refine an EMerge ``Mesh3D`` once and return the new mesh plus a report.

    Args:
        mesh: The current ``Mesh3D`` (left unchanged).
        error: (n_tets,) error indicator per tetrahedron.
        brep: Geometry snapshot for projecting new boundary nodes.
        growth: Relative increase of the number of tets (0.3 = +30 %).
        order, max_depth, lepp_ratio, improve_sweeps, improve_layers, improve_global:
            See :func:`emerge._emerge.aura.refine`. With ``growth=0`` and ``improve_global=True``
            this is a pure quality-improvement pass over the whole mesh.
        strip_ratio, strip_max_growth, strip_gated: Narrow-strip regularisation, see
            :func:`emerge._emerge.aura.refine`.
        periodic: Periodic boundaries as ``[(face1_tags, face2_tags, dv), ...]``.

    Returns:
        (new_mesh, report): the refined ``Mesh3D`` and a dict with ``n_tets_before``,
        ``n_tets``, quality statistics and the refinement/improvement counters.
    """
    inputs = mesh_to_aura_inputs(mesh, periodic)
    q_before = quality(inputs["nodes"], inputs["tets"])
    result = refine(
        **inputs,
        error=error,
        brep=brep,
        growth=growth,
        order=order,
        max_depth=max_depth,
        lepp_ratio=lepp_ratio,
        improve_sweeps=improve_sweeps,
        improve_layers=improve_layers,
        improve_global=improve_global,
        strip_ratio=strip_ratio,
        strip_max_growth=strip_max_growth,
        strip_gated=strip_gated,
    )
    q_after = quality(result.nodes, result.tets)
    new = refine_mesh3d(mesh, result)
    report = dict(
        n_tets_before=int(mesh.tets.shape[1]),
        n_tets=int(new.tets.shape[1]),
        q_min_before=float(q_before.min()),
        q_min=float(q_after.min()),
        q_mean_before=float(q_before.mean()),
        q_mean=float(q_after.mean()),
        **result.stats,
    )
    return new, report
