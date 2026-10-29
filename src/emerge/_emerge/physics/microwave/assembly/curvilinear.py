# [Written by Claude] All code in this module was written by Claude (AI).
"""Curved (isoparametric) tetrahedra at curved CAD surfaces for the curl-curl assembly.

The mesh maps every tetrahedron affinely, so a curved CAD surface is replaced by flat facets. On a strongly
reflecting curved object (a post in a waveguide, a coax inner conductor, a sphere) this faceting error can
dominate the S-parameters long before the field is resolved, and the error estimator cannot see it.

With `Settings.mw_curved_elements` the tetrahedra next to curved PEC and material surfaces get quadratic
edges: the midpoint of every mesh edge on a curved CAD surface or curve is moved onto the true geometry
(gmsh.model.getClosestPoint). That gives the usual 10-node isoparametric geometry
x(y) = y + sum_e d_e phi_e(y), phi_e = 4 l_i l_j, on the straight tetrahedron y. The element matrices of
those tetrahedra are then recomputed with point-wise effective materials (emr/src/microwave/curlcurl_iso.rs);
all other tetrahedra keep the matrices of `curlcurl._matrix_builder`.

Faces with surface integrals (ports, absorbing boundaries, lumped elements, impedance sheets, periodic faces)
stay straight, so nothing changes in the boundary terms. Every curved face connected to a wave port stays
straight as well, so the port modes are solved on the same facets as the line behind the port.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

_FLAT = {"Plane", "Line", "Point"}
_PAIRS = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
_CACHE: dict = {}


def _curved_type(dim: int, tag: int) -> bool:
    """CAD entity that may be curved. gmsh calls swept ellipses, splines, pipes and lofts "Unknown", so only
    planes and lines count as flat (a flat face that is not recognised just gets zero displacements).
    Discrete entities (a mesh loaded from file) have no geometry to project onto."""
    import gmsh

    t = gmsh.model.getType(dim, tag)
    return t not in _FLAT and "iscrete" not in t


def curvable(bc) -> bool:
    """Boundaries whose faces may be curved: PEC and PMC (no surface integral) and metal walls given as a
    SurfaceImpedance of a material (the impedance term is small next to the field)."""
    from ..bcs.boundary_conditions import PEC, PMC, SurfaceImpedance

    if isinstance(bc, (PEC, PMC)):
        return True
    return type(bc) is SurfaceImpedance and getattr(bc, "_Zf", None) is None


def exclude_tags(bcs) -> frozenset:
    """Face tags with surface integrals: they stay straight."""
    return frozenset(int(t) for bc in bcs if not curvable(bc) for t in (getattr(bc, "tags", None) or []))


def spread_tags(bcs) -> frozenset:
    """Wave port faces: every curved face connected to them stays straight. The port mode is solved on the
    straight facets, and a curved line behind a straight port would be an impedance step (coax, circular
    waveguide). Absorbing boundaries, lumped ports and other surface terms only keep their own edges straight."""
    from ..bcs.port_bcs import FloquetPort, LumpedPort, PortBC

    return frozenset(int(t) for bc in bcs if isinstance(bc, PortBC) and not isinstance(bc, (LumpedPort, FloquetPort))
                     for t in (getattr(bc, "tags", None) or []))


def curved_data_bcs(mesh, bcs) -> tuple[np.ndarray, np.ndarray] | None:
    """curved_data for the boundary conditions bcs: the curved tets the assembly uses."""
    return curved_data(mesh, exclude_tags(bcs), spread_tags(bcs))


def quadrature_rule(n: int = 4) -> np.ndarray:
    """Conical product rule on the tetrahedron, (5, n^3): weights (sum 1) and the 4 barycentric rows.
    n = 4 (64 points) is exact to degree 7."""
    from scipy.special import roots_jacobi

    tu, wu = roots_jacobi(n, 2.0, 0.0)
    tv, wv = roots_jacobi(n, 1.0, 0.0)
    tw, ww = roots_jacobi(n, 0.0, 0.0)
    u, v, w = (1 + tu) / 2, (1 + tv) / 2, (1 + tw) / 2
    U, V, W = np.meshgrid(u, v, w, indexing="ij")
    WU, WV, WW = np.meshgrid(wu, wv, ww, indexing="ij")
    wt = (WU * WV * WW).ravel()
    out = np.empty((5, wt.size))
    out[0] = wt / wt.sum()
    out[2] = U.ravel()
    out[3] = (V * (1 - U)).ravel()
    out[4] = (W * (1 - U) * (1 - V)).ravel()
    out[1] = 1 - out[2] - out[3] - out[4]
    return out


_QP = quadrature_rule(4)


def _grads(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric gradients (nt, 4, 3) and signed volumes of tetrahedra X (nt, 4, 3)."""
    J = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=1)
    det = np.linalg.det(J)
    Ji = np.linalg.inv(J)
    g = np.empty((len(X), 4, 3))
    g[:, 1:] = np.transpose(Ji, (0, 2, 1))
    g[:, 0] = -g[:, 1:].sum(1)
    return g, det / 6


def _detD(g: np.ndarray, d: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """det(D) (nt, np) at barycentric points lam (np, 4) for gradients g (nt, 4, 3), displacements d (nt, 6, 3)."""
    out = np.empty((g.shape[0], lam.shape[0]))
    for p, l in enumerate(lam):
        D = np.broadcast_to(np.eye(3), (g.shape[0], 3, 3)).copy()
        for e, (i, j) in enumerate(_PAIRS):
            gp = 4 * (l[i] * g[:, j] + l[j] * g[:, i])
            D += d[:, e, :, None] * gp[:, None, :]
        out[:, p] = np.linalg.det(D)
    return out


def _check_points() -> np.ndarray:
    pts = [np.eye(4)[k] for k in range(4)]
    pts += [(np.eye(4)[i] + np.eye(4)[j]) / 2 for i, j in _PAIRS]
    pts += [np.full(4, 0.25)]
    pts += [(np.ones(4) - np.eye(4)[k]) / 3 for k in range(4)]
    return np.vstack([np.array(pts), _QP[1:].T])


_CHK = _check_points()


def _edge_keys(e0: np.ndarray, e1: np.ndarray, n: int) -> np.ndarray:
    a, b = np.minimum(e0, e1).astype(np.int64), np.maximum(e0, e1).astype(np.int64)
    return a * np.int64(n) + b


def _closest(dim: int, tag: int, pts: np.ndarray) -> np.ndarray:
    import gmsh

    if not len(pts):
        return pts
    q, _ = gmsh.model.getClosestPoint(dim, tag, np.ascontiguousarray(pts, float).ravel())
    q = np.asarray(q, float).reshape(-1, 3)
    return q if q.shape == pts.shape else pts


def curved_data(mesh, exclude: frozenset = frozenset(),
                spread: frozenset | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """The curved tets of a mesh: (tet ids, edge midpoint displacements (nc, 6, 3) in _PAIRS order), or None.

    exclude: faces that stay straight. spread: faces whose connected curved faces stay straight too
    (default: all of exclude; see spread_tags). Needs the gmsh model the mesh was generated from.
    """
    spread = exclude if spread is None else spread
    import gmsh

    nodes = np.asarray(mesh.nodes, float)
    tets = np.asarray(mesh.tets, np.int64)
    edges = np.asarray(mesh.edges, np.int64)
    nn, nt, ne = nodes.shape[1], tets.shape[1], edges.shape[1]
    fp = (id(mesh), nn, nt, ne, float(nodes[:, :: max(1, nn // 97)].sum()), int(tets[:, :: max(1, nt // 97)].sum()),
          tuple(sorted(exclude)), tuple(sorted(spread)))
    if fp in _CACHE:
        return _CACHE[fp]
    _CACHE.clear()
    disp = np.zeros((ne, 3))
    done = np.zeros(ne, bool)
    for t in exclude:
        ids = mesh.ftag_to_edge.get(t, [])
        done[np.asarray(ids, np.int64)] = True
    try:
        curves = [t for _, t in gmsh.model.getEntities(1) if _curved_type(1, t)]
        faces = [t for _, t in gmsh.model.getEntities(2) if _curved_type(2, t)]
        # A curved surface that meets a wave port (a face in spread) stays straight as a whole: the port mode
        # is solved on the straight facets, and a curved line behind a straight port would be an impedance
        # step (coax, circular waveguide). Curves on such faces stay straight too.
        rim = set()
        for t in spread:
            rim.update(abs(c) for _, c in gmsh.model.getBoundary([(2, int(t))], False, False, False))
        bnd = {f: {abs(c) for _, c in gmsh.model.getBoundary([(2, f)], False, False, False)} for f in faces}
        changed = True
        while changed:
            changed = False
            for f in list(bnd):
                if bnd[f] & rim:
                    rim.update(bnd.pop(f))
                    changed = True
        faces = sorted(bnd)
        curves = [c for c in curves if c not in rim]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Curved elements: no CAD geometry available ({e}), all elements stay straight")
        curves, faces = [], []
    mid = 0.5 * (nodes[:, edges[0]] + nodes[:, edges[1]]).T
    for dim, tags, emap in ((1, curves, mesh.etag_to_edge), (2, faces, mesh.ftag_to_edge)):
        for t in tags:
            ids = np.asarray(emap.get(t, []), np.int64)
            if not len(ids):
                continue
            ids = ids[~done[ids]]
            if not len(ids):
                continue
            q = _closest(dim, t, mid[ids])
            disp[ids] = q - mid[ids]
            done[ids] = True
    L = np.linalg.norm(nodes[:, edges[0]] - nodes[:, edges[1]], axis=0)
    dn = np.linalg.norm(disp, axis=1)
    disp[dn > 0.25 * L] = 0.0  # the projection went elsewhere (seam, wrong patch): keep straight
    disp[dn < 1e-12 * np.maximum(L, 1e-300)] = 0.0
    ek = _edge_keys(edges[0], edges[1], nn)
    order = np.argsort(ek)
    ek_s = ek[order]
    a, b = tets[_PAIRS[:, 0]], tets[_PAIRS[:, 1]]
    pos = np.searchsorted(ek_s, _edge_keys(a, b, nn))
    eid = order[np.minimum(pos, len(order) - 1)]  # (6, nt) edge ids of every tet in _PAIRS order
    curved_edge = np.linalg.norm(disp, axis=1) > 0
    tc = np.where(curved_edge[eid].any(0))[0]
    if not len(tc):
        _CACHE[fp] = None
        return None
    X = np.transpose(nodes[:, tets[:, tc]], (2, 1, 0))
    g, _ = _grads(X)
    # a valid map needs det D well above 0 everywhere; otherwise shrink the offending edges (globally, so
    # that neighbouring tets keep the same face shape)
    scale = np.ones(ne)
    for _ in range(6):
        d = disp[eid[:, tc].T] * scale[eid[:, tc].T][..., None]
        bad = _detD(g, d, _CHK).min(1) < 0.3
        if not bad.any():
            break
        e_bad = np.unique(eid[:, tc[bad]].ravel())
        e_bad = e_bad[curved_edge[e_bad]]
        scale[e_bad] *= 0.5
    else:
        # still invalid: straighten those tets completely, and repeat for neighbours that lose a
        # compensating edge (ends at the latest when all tets are straight, det D = 1)
        while True:
            d = disp[eid[:, tc].T] * scale[eid[:, tc].T][..., None]
            bad = _detD(g, d, _CHK).min(1) < 0.3
            if not bad.any():
                break
            scale[np.unique(eid[:, tc[bad]].ravel())] = 0.0
    keep = np.abs(d).sum((1, 2)) > 0
    res = (tc[keep], np.ascontiguousarray(d[keep]))
    logger.debug(f" - Curved elements: {int(keep.sum())} tets, {int((scale * curved_edge > 0).sum())} curved edges, "
                 f"{int(((scale < 1) & curved_edge).sum())} shrunk")
    _CACHE[fp] = res
    return res


def apply_curved(field, er: np.ndarray, ur: np.ndarray, conductor_tets: np.ndarray, dataE: np.ndarray,
                 dataB: np.ndarray, bcs) -> int:
    """Overwrite the element blocks of the curved tets in the volume data of `tet_mass_stiffness_matrices`.

    Returns the number of curved tets.
    """
    from emr.curlcurl_iso import _matrix_builder_iso

    mesh = field.mesh
    cd = curved_data_bcs(mesh, bcs)
    if cd is None:
        return 0
    ctets, disp = cd
    nt = mesh.tets.shape[1]
    pos = np.full(nt, -1, np.int64)
    assy = np.arange(nt)
    if len(conductor_tets):
        assy = np.delete(assy, np.asarray(conductor_tets, np.int64))
    pos[assy] = np.arange(len(assy))
    sel = pos[ctets] >= 0
    if not sel.any():
        return 0
    E, B = _matrix_builder_iso(
        np.asarray(mesh.nodes, np.float64), np.asarray(mesh.tets, np.int64), np.asarray(mesh.tris, np.int64),
        np.asarray(mesh.edges, np.int64), np.asarray(mesh.tet_to_edge, np.int64),
        np.asarray(mesh.tet_to_tri, np.int64), ur, er, ctets[sel], np.ascontiguousarray(disp[sel]),
        np.asarray(field.dofcodes3d, np.int64), _QP,
    )
    nsq = field.dofcodes3d.shape[0] ** 2
    idx = (pos[ctets[sel]][:, None] * nsq + np.arange(nsq)[None, :]).ravel()
    dataE[idx] = E
    dataB[idx] = B
    return int(sel.sum())


def curved_volume(mesh, exclude: frozenset = frozenset()) -> tuple[float, float]:
    """(straight, curved) volume of all tets: a check of the geometry map."""
    X = np.transpose(np.asarray(mesh.nodes, float)[:, np.asarray(mesh.tets)], (2, 1, 0))
    g, V = _grads(X)
    V = np.abs(V)
    tot = V.sum()
    cd = curved_data(mesh, exclude)
    if cd is None:
        return tot, tot
    tc, d = cd
    dets = _detD(g[tc], d, _QP[1:].T)
    return tot, tot + (V[tc] * (dets @ _QP[0] - 1.0)).sum()
