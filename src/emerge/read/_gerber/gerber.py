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

"""Gerber (RS-274X) layer import.

Pipeline:
    1. pygerber parses the file into draw/flash/region commands.
    2. Every command is converted into emcad polygons (meters).
    3. Dark/clear polarity runs are merged with emcad booleans (fast, 2D).
    4. The merged outline is simplified.
    5. Each resulting polygon (with holes) becomes one gmsh plane surface.

All boolean work happens in 2D before anything is sent to gmsh, so gmsh never
has to fuse or cut OCC faces.
"""

from __future__ import annotations

import math
from typing import Iterable

import gmsh
import numpy as np
import emcad as cad
from emcad.poly import GeometryException
from emcad.kernel.api import is_simple_ring, dekeyhole_polygon, sanitize_polygon
from loguru import logger

from pygerber.gerberx3.api.v2 import GerberFile
from pygerber.gerberx3.parser2.commands2.region2 import Region2
from pygerber.gerberx3.parser2.commands2.line2 import Line2
from pygerber.gerberx3.parser2.commands2.flash2 import Flash2
from pygerber.gerberx3.parser2.commands2.arc2 import Arc2, CCArc2
from pygerber.gerberx3.parser2.apertures2.polygon2 import Polygon2
from pygerber.gerberx3.parser2.apertures2.circle2 import Circle2
from pygerber.gerberx3.parser2.apertures2.rectangle2 import Rectangle2
from pygerber.gerberx3.parser2.apertures2.obround2 import Obround2

from ..._emerge.cs import CoordinateSystem, GCS
from ..._emerge.geometry import GeoSurface

N_CIRC_MIN = 6
N_CIRC_MAX = 31

MM = 1e-3      # Gerber coordinates are handled in mm, polygons are stored in meters
_EPS = 5e-4    # mm, features below this size are treated as degenerate


def _calc_via_segs(diameter: float,
                   nsegments: int,
                   edge_size: float | None = None) -> int:
    """Number of segments used to discretize a full circle of the given diameter."""
    if edge_size is not None:
        N = int(np.ceil(np.pi*diameter/edge_size))
    else:
        N = nsegments
    return max(min(N, N_CIRC_MAX), N_CIRC_MIN)


def _mm(value) -> float:
    return float(value.as_millimeters())


def _xy(vec) -> tuple[float, float]:
    return _mm(vec.x), _mm(vec.y)


def _is_dark(cmd) -> bool:
    return 'dark' in str(cmd.transform.polarity).lower()


def _arc_sweep(cmd: Arc2) -> float:
    """Signed sweep (CCW positive) of an arc command. Coinciding start/end is a full circle."""
    x0, y0 = _xy(cmd.start_point)
    x1, y1 = _xy(cmd.end_point)
    xc, yc = _xy(cmd.center_point)
    a0 = math.atan2(y0 - yc, x0 - xc)
    a1 = math.atan2(y1 - yc, x1 - xc)
    if isinstance(cmd, CCArc2):
        sweep = (a1 - a0) % (2*math.pi)
        return sweep if sweep > 1e-9 else 2*math.pi
    sweep = -((a0 - a1) % (2*math.pi))
    return sweep if sweep < -1e-9 else -2*math.pi


def _ring(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ring in meters with duplicate/degenerate points removed (Gerber writers, KiCad
    included, often repeat the closing point)."""
    return sanitize_polygon(xy[:, 0]*MM, xy[:, 1]*MM, 1e-9, True)


def _poly(xy: np.ndarray, holes: list[np.ndarray] | None = None) -> cad.Polygon:
    """emcad polygon (meters) from (n, 2) point arrays in mm."""
    return cad.Polygon(*_ring(xy), [cad.Polygon(*_ring(h)) for h in (holes or [])])


def _translate(p: cad.Polygon, dx: float, dy: float) -> cad.Polygon:
    return cad.Polygon(np.asarray(p.xs) + dx, np.asarray(p.ys) + dy, [_translate(h, dx, dy) for h in p.holes])


def _iter_rings(polys: Iterable[cad.Polygon]):
    """Yields every ring (outer boundaries, holes, islands, ...) as (xs, ys)."""
    for p in polys:
        yield p.xs, p.ys
        yield from _iter_rings(p.holes)


############################################################
#                     RING SIMPLIFICATION                  #
############################################################

def _simplify_ring(xy: np.ndarray,
                   min_dist: float,
                   min_incl_angle: float,
                   min_remove_angle: float,
                   min_points: int = 4) -> np.ndarray:
    """Removes redundant vertices from an open (non-repeated) ring.

    A vertex is removed iff it duplicates its predecessor, OR
        turn angle < min_incl_angle, OR
        both adjacent edges < min_dist AND turn angle < min_remove_angle.

    Each pass is fully vectorized. Within a pass no two neighbouring vertices
    are removed, so angles are re-evaluated after every removal just like a
    sequential sweep would, but in O(n log n) instead of O(n^2) or worse.
    """
    while len(xy) > min_points:
        d_prev = xy - np.roll(xy, 1, axis=0)
        d_next = np.roll(xy, -1, axis=0) - xy
        l_prev = np.hypot(d_prev[:, 0], d_prev[:, 1])
        l_next = np.hypot(d_next[:, 0], d_next[:, 1])

        valid = (l_prev > 0) & (l_next > 0)
        cos_a = np.einsum('ij,ij->i', d_prev, d_next) / np.where(valid, l_prev*l_next, 1.0)
        angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

        remove = (l_prev == 0) | (valid & (
            (angle < min_incl_angle)
            | ((l_prev < min_dist) & (l_next < min_dist) & (angle < min_remove_angle))
        ))
        if not remove.any():
            break

        # Only remove every other vertex in a run of candidates
        idx = np.flatnonzero(remove)
        run_start = np.r_[True, np.diff(idx) != 1]
        run_id = np.cumsum(run_start) - 1
        pos_in_run = np.arange(len(idx)) - np.flatnonzero(run_start)[run_id]
        sel = idx[pos_in_run % 2 == 0]
        if len(sel) > 1 and sel[0] == 0 and sel[-1] == len(xy) - 1:
            sel = sel[:-1]  # first and last vertex are neighbours on a ring
        sel = sel[:max(len(xy) - min_points, 0)]
        if len(sel) == 0:
            break
        xy = np.delete(xy, sel, axis=0)
    return xy


def _simplify_polygon(poly: cad.Polygon, min_dist: float, min_incl_angle: float, min_remove_angle: float) -> cad.Polygon:
    """Simplifies every ring of a polygon (recursing into holes and islands). A candidate
    that is no longer a valid, properly nested polygon is rejected in favour of the original."""
    xy = _simplify_ring(np.column_stack((poly.xs, poly.ys)), min_dist, min_incl_angle, min_remove_angle)
    holes = [_simplify_polygon(h, min_dist, min_incl_angle, min_remove_angle) for h in poly.holes]
    if not is_simple_ring(xy[:, 0], xy[:, 1]):
        return poly
    try:
        return cad.Polygon(xy[:, 0], xy[:, 1], holes)
    except GeometryException:
        return poly


############################################################
#                     PRIMITIVE GEOMETRY                   #
############################################################

class _Builder:
    """Converts pygerber commands into emcad polygons (meters). Shapes are built from
    point arrays in mm; round shapes are discretized here."""

    def __init__(self, arc_tol_mm: float, n_circ_segments: int, seg_size_mm: float | None):
        self.arc_tol = arc_tol_mm
        self.nseg = n_circ_segments
        self.seg_size = seg_size_mm

    # ---------------- point generators (mm) ----------------

    def _n_arc(self, r: float, sweep: float) -> int:
        """Segments for an arc of radius r such that the chord error stays below arc_tol."""
        dth = 2*math.acos(1 - self.arc_tol/r) if self.arc_tol < r else math.pi/2
        return max(1, math.ceil(abs(sweep)/max(dth, 1e-6)), math.ceil(abs(sweep)/(2*math.pi)*N_CIRC_MIN))

    def _n_cap(self, diameter: float) -> int:
        """Segments for a half circle of an aperture."""
        return max(2, math.ceil(_calc_via_segs(diameter, self.nseg, self.seg_size)/2))

    @staticmethod
    def _arc(cx: float, cy: float, r: float, a0: float, sweep: float, n: int) -> np.ndarray:
        th = a0 + np.linspace(0, sweep, n + 1)
        return np.column_stack((cx + r*np.cos(th), cy + r*np.sin(th)))

    def _circle_xy(self, cx: float, cy: float, d: float) -> np.ndarray:
        return self._arc(cx, cy, d/2, 0, 2*math.pi, _calc_via_segs(d, self.nseg, self.seg_size))[:-1]

    def _stadium_xy(self, x1: float, y1: float, x2: float, y2: float, r: float) -> np.ndarray:
        """A segment stroked with a round aperture of radius r (two half circle caps)."""
        L = math.hypot(x2 - x1, y2 - y1)
        if L < _EPS:
            return self._circle_xy(x1, y1, 2*r)
        a = math.atan2(-(x2 - x1), y2 - y1)  # direction of the right hand normal
        n = self._n_cap(2*r)
        end_cap = self._arc(x2, y2, r, a, math.pi, n)            # right side -> left side around p2
        start_cap = self._arc(x1, y1, r, a + math.pi, math.pi, n)  # left side -> right side around p1
        return np.vstack((end_cap, start_cap))

    def _stroke_arc_xy(self, xc: float, yc: float, R: float, a0: float, sweep: float, w: float) -> np.ndarray:
        """An arc stroked with a round aperture: annular sector with round caps (requires R > w)."""
        s = math.copysign(1.0, sweep)
        a1 = a0 + sweep
        n_cap = self._n_cap(2*w)
        outer = self._arc(xc, yc, R + w, a0, sweep, self._n_arc(R + w, sweep))
        end_cap = self._arc(xc + R*math.cos(a1), yc + R*math.sin(a1), w, a1, s*math.pi, n_cap)
        inner = self._arc(xc, yc, R - w, a1, -sweep, self._n_arc(R - w, sweep))
        start_cap = self._arc(xc + R*math.cos(a0), yc + R*math.sin(a0), w, a0 + math.pi, s*math.pi, n_cap)
        return np.vstack((outer, end_cap[1:], inner[1:], start_cap[1:-1]))

    def _arc_points(self, cmd: Arc2) -> np.ndarray:
        """Samples the centre line of an arc command (mm), start and end inclusive."""
        x0, y0 = _xy(cmd.start_point)
        x1, y1 = _xy(cmd.end_point)
        xc, yc = _xy(cmd.center_point)
        R = math.hypot(x0 - xc, y0 - yc)
        if R == 0:
            return np.array([[x0, y0], [x1, y1]])
        sweep = _arc_sweep(cmd)
        pts = self._arc(xc, yc, R, math.atan2(y0 - yc, x0 - xc), sweep, self._n_arc(R, sweep))
        pts[-1] = (x1, y1)  # land exactly on the end point
        return pts

    def _with_hole(self, xy: np.ndarray, aperture, cx: float, cy: float) -> list[cad.Polygon]:
        hole = getattr(aperture, 'hole_diameter', None)
        if hole is None or _mm(hole) <= 0:
            return [_poly(xy)]
        hole_xy = self._circle_xy(cx, cy, _mm(hole))
        try:
            return [_poly(xy, [hole_xy])]
        except GeometryException:  # hole not strictly inside the aperture
            return cad.subtract_polygons((_poly(xy),), (_poly(hole_xy),))

    # ---------------- commands ----------------

    def _stroke(self, cmd: Line2 | Arc2) -> list[cad.Polygon]:
        """Strokes a line or arc with its aperture (D01 draw)."""
        ap = cmd.aperture
        if isinstance(ap, Circle2):
            w = _mm(ap.diameter)/2
        elif isinstance(ap, Rectangle2) and not isinstance(ap, Obround2) and isinstance(cmd, Line2):
            # Minkowski sum of a rectangle and a segment = hull of the two end-rectangles
            hx, hy = _mm(ap.x_size)/2, _mm(ap.y_size)/2
            pts = np.array([(x + sx*hx, y + sy*hy) for x, y in (_xy(cmd.start_point), _xy(cmd.end_point))
                            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            return [_poly(pts[cad.convex_hull(pts[:, 0], pts[:, 1])])]
        else:
            logger.warning(f'Drawing with aperture {type(ap).__name__} is approximated by a round stroke.')
            w = _mm(ap.get_stroke_width())/2
        if w <= 0:
            return []

        if isinstance(cmd, Line2):
            return [_poly(self._stadium_xy(*_xy(cmd.start_point), *_xy(cmd.end_point), w))]

        x0, y0 = _xy(cmd.start_point)
        xc, yc = _xy(cmd.center_point)
        R = math.hypot(x0 - xc, y0 - yc)
        sweep = _arc_sweep(cmd)
        if R*abs(sweep) < _EPS:
            return [_poly(self._circle_xy(x0, y0, 2*w))]
        if R - w < _EPS:
            # Aperture wider than the arc radius: union of the strokes of the sampled centre line
            pts = self._arc_points(cmd)
            return cad.add_polygons(*[_poly(self._stadium_xy(*p, *q, w)) for p, q in zip(pts[:-1], pts[1:])])
        if abs(abs(sweep) - 2*math.pi) < 1e-9:
            return [_poly(self._circle_xy(xc, yc, 2*(R + w)), [self._circle_xy(xc, yc, 2*(R - w))])]
        return [_poly(self._stroke_arc_xy(xc, yc, R, math.atan2(y0 - yc, x0 - xc), sweep, w))]

    def _flash(self, cmd: Flash2) -> list[cad.Polygon]:
        ap = cmd.aperture
        cx, cy = _xy(cmd.flash_point)

        if isinstance(ap, Circle2):
            d = _mm(ap.diameter)
            return self._with_hole(self._circle_xy(cx, cy, d), ap, cx, cy) if d > 0 else []

        if isinstance(ap, Obround2):
            sx, sy = _mm(ap.x_size), _mm(ap.y_size)
            r = min(sx, sy)/2
            h = max(sx, sy)/2 - r
            rot = math.radians(float(ap.rotation)) + (0 if sx >= sy else math.pi/2)
            dx, dy = h*math.cos(rot), h*math.sin(rot)
            return self._with_hole(self._stadium_xy(cx - dx, cy - dy, cx + dx, cy + dy, r), ap, cx, cy)

        if isinstance(ap, Rectangle2):
            hx, hy = _mm(ap.x_size)/2, _mm(ap.y_size)/2
            c, s = math.cos(math.radians(float(ap.rotation))), math.sin(math.radians(float(ap.rotation)))
            xy = np.array([(cx + x*c - y*s, cy + x*s + y*c) for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))])
            return self._with_hole(xy, ap, cx, cy)

        if isinstance(ap, Polygon2):
            n = int(ap.number_vertices)
            R = _mm(ap.outer_diameter)/2
            th = np.radians(float(ap.rotation)) + 2*np.pi*np.arange(n)/n
            return self._with_hole(np.column_stack((cx + R*np.cos(th), cy + R*np.sin(th))), ap, cx, cy)

        if hasattr(ap, 'command_buffer'):
            # Macro (AM) or block (AB) aperture: primitives are relative to the flash point
            # and their exposure only affects the aperture itself.
            return [_translate(p, cx*MM, cy*MM) for p in merge_commands(ap.command_buffer, self)]

        logger.error(f'Unsupported aperture {type(ap).__name__} in flash at ({cx}, {cy}) mm. '
                     'Please contact the EMerge developers with this Gerber file to get it supported.')
        return []

    def region_contours(self, cmd: Region2) -> list[list]:
        """Splits a region (G36/G37) into contours: a new contour starts wherever a
        segment does not continue from the previous end point (D02 move)."""
        contours: list[list] = []
        last_end = None
        for seg in cmd.command_buffer:
            if not isinstance(seg, (Line2, Arc2)):
                logger.warning(f'Unhandled region segment {type(seg).__name__}')
                continue
            start, end = _xy(seg.start_point), _xy(seg.end_point)
            if last_end is None or math.hypot(start[0] - last_end[0], start[1] - last_end[1]) > _EPS:
                contours.append([])
            contours[-1].append(seg)
            last_end = end
        return contours

    def contour_ring(self, segs: list) -> np.ndarray:
        """Discretized ring (mm) of a region contour."""
        parts = [np.array([_xy(segs[0].start_point)])]
        for seg in segs:
            parts.append(self._arc_points(seg)[1:] if isinstance(seg, Arc2) else np.array([_xy(seg.end_point)]))
        return np.vstack(parts)

    def contour_polygon(self, segs: list) -> cad.Polygon | None:
        """Region contour as a polygon, with KiCad style keyhole (cut-in) holes resolved."""
        ring = self.contour_ring(segs)
        if len(ring) < 4:
            return None
        return dekeyhole_polygon(_poly(ring))

    def _region(self, cmd: Region2) -> list[cad.Polygon]:
        polys = [p for p in map(self.contour_polygon, self.region_contours(cmd)) if p is not None]
        return cad.add_polygons(*polys) if polys else []

    def geometry(self, cmd) -> list[cad.Polygon] | None:
        if isinstance(cmd, Region2):
            return self._region(cmd)
        if isinstance(cmd, (Line2, Arc2)):
            return self._stroke(cmd)
        if isinstance(cmd, Flash2):
            return self._flash(cmd)
        logger.warning(f'Unprocessed Gerber command: {type(cmd).__name__}')
        return None


def _join(polys: list[cad.Polygon]) -> list[cad.Polygon]:
    """Fuses boolean output pieces that only touch edge-to-edge (add_polygons keeps
    those separate), e.g. a trace ending exactly on the edge of a copper region."""
    if len(polys) < 2:
        return polys
    try:
        return cad.join_polygons(polys)
    except GeometryException as e:
        logger.debug(f'join_polygons skipped: {e}')
        return polys


def merge_commands(commands: Iterable, builder: _Builder) -> list[cad.Polygon]:
    """Converts a stream of commands to polygons and applies the dark/clear polarity
    in order. Consecutive commands of equal polarity are merged in a single boolean."""
    result: list[cad.Polygon] = []
    batch: list[cad.Polygon] = []
    batch_dark = True

    def flush() -> None:
        nonlocal result
        if not batch:
            return
        if batch_dark:
            result = _join(cad.add_polygons(*result, *batch))
        elif result:
            result = _join(cad.subtract_polygons(tuple(result), tuple(batch)))
        batch.clear()

    for cmd in commands:
        polys = builder.geometry(cmd)
        if not polys:
            continue
        dark = _is_dark(cmd)
        if dark != batch_dark:
            flush()
            batch_dark = dark
        batch.extend(polys)
    flush()
    return result


############################################################
#                       GERBER CLASS                      #
############################################################

class GerberLayer:
    """A single Gerber copper layer converted to merged 2D geometry.

    Args:
        filename (str): Path to the Gerber file.
        res_mm (float): Geometric resolution in mm. Used as the arc chord tolerance and as
            the edge length below which near-collinear vertices are removed.
        n_circ_segments (int): Segments per full circle for round apertures. Defaults to 8.
        seg_size_mm (float | None): Target edge length for round apertures. Overrides n_circ_segments.
        min_incl_angle (float): Vertices with a turn angle below this (deg) are removed.
        min_remove_angle (float): Vertices between two edges shorter than res_mm with a
            turn angle below this (deg) are removed.
        cs (CoordinateSystem): The coordinate system to place the layer in.
        simplify (bool): Whether to simplify the merged outline.
    """

    def __init__(self,
                 filename: str,
                 res_mm: float,
                 n_circ_segments: int = 8,
                 seg_size_mm: float | None = None,
                 min_incl_angle: float = 2.0,
                 min_remove_angle: float = 10.0,
                 ignore_via_pads: bool = True,
                 cs: CoordinateSystem = GCS,
                 simplify: bool = True
                 ):
        self.fname: str = filename
        self.res_mm: float = res_mm
        self.cs: CoordinateSystem = cs

        commands = GerberFile.from_file(filename).parse()._command_buffer
        polys = merge_commands(commands, _Builder(res_mm, n_circ_segments, seg_size_mm))

        if simplify:
            polys = [_simplify_polygon(p, res_mm*MM, min_incl_angle, min_remove_angle) for p in polys]
            for p in polys:
                p.dezigzag(res_mm*MM)  # fail-safe: collapses round cap step artifacts

        #: Merged layer polygons in meters. Holes may carry islands (holes of holes).
        self.polygons: list[cad.Polygon] = polys

        if polys:
            self.xs: list[float] = [min(min(p.xs) for p in polys), max(max(p.xs) for p in polys)]
            self.ys: list[float] = [min(min(p.ys) for p in polys), max(max(p.ys) for p in polys)]
        else:
            logger.warning(f'Gerber file {filename} produced no copper geometry.')
            self.xs, self.ys = [], []

    @property
    def n_points(self) -> int:
        return sum(len(xs) for xs, _ in _iter_rings(self.polygons))

    def bounds(self, margin: float | tuple[float, float, float, float] = 0.0) -> tuple[float, float, float, float]:
        if isinstance(margin, (float, int)):
            margin = (margin, margin, margin, margin)
        return min(self.xs)-margin[0], min(self.ys)-margin[1], max(self.xs)+margin[2], max(self.ys)+margin[3]

    def _curve_loop(self, xs, ys, z: float) -> int:
        xg, yg, zg = self.cs.in_global_cs(np.asarray(xs), np.asarray(ys), np.full(len(xs), z))
        occ = gmsh.model.occ
        ptags = [occ.addPoint(x, y, zz) for x, y, zz in zip(xg, yg, zg)]
        n = len(ptags)
        lines = [occ.addLine(ptags[i], ptags[(i + 1) % n]) for i in range(n)]
        return occ.addCurveLoop(lines)

    def _surfaces(self, poly: cad.Polygon, z: float) -> list[int]:
        loops = [self._curve_loop(poly.xs, poly.ys, z)]
        loops += [self._curve_loop(h.xs, h.ys, z) for h in poly.holes]
        tags = [gmsh.model.occ.addPlaneSurface(loops)]
        for hole in poly.holes:  # islands inside holes are separate surfaces
            for island in hole.holes:
                tags += self._surfaces(island, z)
        return tags

    def flatten(self, z: float = 0.0) -> GeoSurface:
        """Creates the gmsh surfaces of this layer at height z (meters).

        Returns:
            GeoSurface: One surface object containing all disjoint copper areas.
        """
        if not self.polygons:
            raise ValueError(f'Gerber file {self.fname} contains no copper geometry.')
        tags = [t for poly in self.polygons for t in self._surfaces(poly, z)]
        return GeoSurface(tags, name='GerberLayer')
