"""A one-time snapshot of the BREP geometry used to project new boundary nodes.

Every geometric face and curve is reduced to a small descriptor:

* faces:  flat (planes), an exact sphere or cylinder fit, or a dense sample
  cloud with normals for any other surface (cones, tori, B-splines, ...).
* curves: straight (lines), an exact circle fit, or an ordered polyline.

The snapshot is plain numpy data, so AURA itself never talks to GMSH.
:func:`snapshot_gmsh` builds it from a live GMSH/OCC model; ``gmsh`` is
imported lazily and is not a dependency of AURA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SURF_FLAT, SURF_SPHERE, SURF_CYLINDER, SURF_SAMPLED = 0, 1, 2, 3
CURVE_STRAIGHT, CURVE_CIRCLE, CURVE_POLYLINE = 0, 1, 2


@dataclass
class BrepSnapshot:
    surf_tags: np.ndarray  # (S,) int32
    surf_kind: np.ndarray  # (S,) int32
    surf_params: np.ndarray  # (S, 7) sphere [c, r] / cylinder [p, axis, r]
    samp_owner: np.ndarray  # (K,) int32 face tag of each sample
    samp_pts: np.ndarray  # (K, 3)
    samp_nrm: np.ndarray  # (K, 3)
    curve_tags: np.ndarray  # (C,) int32
    curve_kind: np.ndarray  # (C,) int32
    curve_params: np.ndarray  # (C, 7) circle [c, n, r]
    poly_owner: np.ndarray  # (P,) int32 curve tag of each polyline point
    poly_pts: np.ndarray  # (P, 3) ordered along each curve

    @classmethod
    def empty(cls) -> "BrepSnapshot":
        i = np.zeros((0,), dtype=np.int32)
        p7 = np.zeros((0, 7))
        p3 = np.zeros((0, 3))
        return cls(i, i, p7, i, p3, p3, i, i, p7, i, p3)

    def as_args(self) -> tuple:
        def f(a, cols=None):
            a = np.ascontiguousarray(a, dtype=np.float64)
            return a.reshape(-1, cols) if cols else a

        def i(a):
            return np.ascontiguousarray(a, dtype=np.int32)

        return (
            i(self.surf_tags), i(self.surf_kind), f(self.surf_params, 7),
            i(self.samp_owner), f(self.samp_pts, 3), f(self.samp_nrm, 3),
            i(self.curve_tags), i(self.curve_kind), f(self.curve_params, 7),
            i(self.poly_owner), f(self.poly_pts, 3),
        )

    def summary(self) -> dict[str, int]:
        names_s = {SURF_FLAT: "flat", SURF_SPHERE: "sphere", SURF_CYLINDER: "cylinder", SURF_SAMPLED: "sampled"}
        names_c = {CURVE_STRAIGHT: "straight", CURVE_CIRCLE: "circle", CURVE_POLYLINE: "polyline"}
        out: dict[str, int] = {}
        for k in self.surf_kind.tolist():
            key = "face_" + names_s[k]
            out[key] = out.get(key, 0) + 1
        for k in self.curve_kind.tolist():
            key = "curve_" + names_c[k]
            out[key] = out.get(key, 0) + 1
        return out


# ---------------------------------------------------------------------------
# Least-squares primitive fits
# ---------------------------------------------------------------------------


def fit_sphere(pts: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Returns (center, radius, max relative residual)."""
    A = np.hstack([2 * pts, np.ones((len(pts), 1))])
    b = np.sum(pts**2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = float(np.sqrt(sol[3] + c @ c))
    res = float(np.max(np.abs(np.linalg.norm(pts - c, axis=1) - r)) / r)
    return c, r, res


def _fit_circle_2d(uv: np.ndarray) -> tuple[np.ndarray, float]:
    A = np.hstack([2 * uv, np.ones((len(uv), 1))])
    b = np.sum(uv**2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    return c, float(np.sqrt(sol[2] + c @ c))


def _basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def fit_cylinder(pts: np.ndarray, nrm: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Returns (point on axis, unit axis, radius, max relative residual)."""
    w, v = np.linalg.eigh(nrm.T @ nrm)
    axis = v[:, 0]  # normals are perpendicular to the axis
    u1, u2 = _basis(axis)
    c2, r = _fit_circle_2d(np.stack([pts @ u1, pts @ u2], axis=1))
    p = c2[0] * u1 + c2[1] * u2
    d = pts - p
    radial = d - np.outer(d @ axis, axis)
    res = float(np.max(np.abs(np.linalg.norm(radial, axis=1) - r)) / r)
    return p, axis, r, res


def fit_circle_3d(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Returns (center, unit plane normal, radius, max relative residual)."""
    c0 = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c0)
    n = vt[2]
    u1, u2 = _basis(n)
    d = pts - c0
    c2, r = _fit_circle_2d(np.stack([d @ u1, d @ u2], axis=1))
    c = c0 + c2[0] * u1 + c2[1] * u2
    res = float(np.max(np.abs(np.linalg.norm(pts - c, axis=1) - r)) / r)
    res = max(res, float(np.max(np.abs((pts - c) @ n)) / r))
    return c, n, r, res


# ---------------------------------------------------------------------------
# GMSH snapshot
# ---------------------------------------------------------------------------


def snapshot_gmsh(
    scale: float = 1.0,
    n_surface_samples: int = 40,
    n_curve_samples: int = 129,
    fit_tol: float = 1e-6,
) -> BrepSnapshot:
    """Snapshot the geometry of the currently loaded GMSH/OCC model.

    Args:
        scale: Factor from GMSH geometry coordinates to mesh coordinates.
        n_surface_samples: Samples per parametric direction for non-analytic faces.
        n_curve_samples: Samples along each non-analytic curve.
        fit_tol: Relative residual below which a sphere/cylinder/circle fit is accepted.
    """
    import gmsh  # type: ignore  # lazily: gmsh is only required for this function

    surf_tags, surf_kind, surf_params = [], [], []
    samp_owner, samp_pts, samp_nrm = [], [], []
    for _, tag in gmsh.model.getEntities(2):
        typ = gmsh.model.getType(2, tag)
        params = np.zeros(7)
        kind = SURF_FLAT
        if typ != "Plane":
            (u0, v0), (u1, v1) = gmsh.model.getParametrizationBounds(2, tag)
            us = np.linspace(u0, u1, n_surface_samples)
            vs = np.linspace(v0, v1, n_surface_samples)
            U, V = np.meshgrid(us, vs, indexing="ij")
            uv = np.stack([U.ravel(), V.ravel()], axis=1).ravel()
            xyz = np.asarray(gmsh.model.getValue(2, tag, uv)).reshape(-1, 3) * scale
            nrm = np.asarray(gmsh.model.getNormal(tag, uv)).reshape(-1, 3)
            ok = np.all(np.isfinite(xyz), axis=1) & np.all(np.isfinite(nrm), axis=1)
            ok &= np.linalg.norm(nrm, axis=1) > 0.5
            xyz, nrm = xyz[ok], nrm[ok]
            kind = SURF_SAMPLED
            if typ == "Sphere":
                c, r, res = fit_sphere(xyz)
                if res < fit_tol:
                    kind, params[:4] = SURF_SPHERE, [*c, r]
            elif typ == "Cylinder":
                p, ax, r, res = fit_cylinder(xyz, nrm)
                if res < fit_tol:
                    kind, params[:] = SURF_CYLINDER, [*p, *ax, r]
            if kind == SURF_SAMPLED:
                samp_owner.append(np.full(len(xyz), tag, dtype=np.int32))
                samp_pts.append(xyz)
                samp_nrm.append(nrm)
        surf_tags.append(tag)
        surf_kind.append(kind)
        surf_params.append(params)

    curve_tags, curve_kind, curve_params = [], [], []
    poly_owner, poly_pts = [], []
    for _, tag in gmsh.model.getEntities(1):
        typ = gmsh.model.getType(1, tag)
        params = np.zeros(7)
        kind = CURVE_STRAIGHT
        if typ != "Line":
            (t0,), (t1,) = gmsh.model.getParametrizationBounds(1, tag)
            ts = np.linspace(t0, t1, n_curve_samples)
            xyz = np.asarray(gmsh.model.getValue(1, tag, ts)).reshape(-1, 3) * scale
            xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
            kind = CURVE_POLYLINE
            if len(xyz) >= 3:
                c, n, r, res = fit_circle_3d(xyz)
                if res < fit_tol:
                    kind, params[:] = CURVE_CIRCLE, [*c, *n, r]
                elif np.ptp(xyz, axis=0).max() > 0 and _is_straight(xyz, fit_tol):
                    kind = CURVE_STRAIGHT
            if kind == CURVE_POLYLINE:
                poly_owner.append(np.full(len(xyz), tag, dtype=np.int32))
                poly_pts.append(xyz)
        curve_tags.append(tag)
        curve_kind.append(kind)
        curve_params.append(params)

    def cat(lst, shape):
        return np.concatenate(lst) if lst else np.zeros(shape)

    return BrepSnapshot(
        np.array(surf_tags, dtype=np.int32),
        np.array(surf_kind, dtype=np.int32),
        np.array(surf_params).reshape(-1, 7),
        cat(samp_owner, (0,)).astype(np.int32),
        cat(samp_pts, (0, 3)),
        cat(samp_nrm, (0, 3)),
        np.array(curve_tags, dtype=np.int32),
        np.array(curve_kind, dtype=np.int32),
        np.array(curve_params).reshape(-1, 7),
        cat(poly_owner, (0,)).astype(np.int32),
        cat(poly_pts, (0, 3)),
    )


def _is_straight(xyz: np.ndarray, tol: float) -> bool:
    d = xyz[-1] - xyz[0]
    L = np.linalg.norm(d)
    if L == 0:
        return False
    d /= L
    rel = xyz - xyz[0]
    off = rel - np.outer(rel @ d, d)
    return bool(np.max(np.linalg.norm(off, axis=1)) < tol * L)
