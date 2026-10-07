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
    2. Every command is converted into a 2D shapely geometry (in mm).
    3. Dark/clear polarity runs are merged with GEOS booleans (fast, 2D).
    4. The merged outline is simplified and snapped to a grid.
    5. Each resulting polygon (with holes) becomes one gmsh plane surface.

All boolean work happens in 2D before anything is sent to gmsh, so gmsh never
has to fuse or cut OCC faces.
"""

from __future__ import annotations

import math
from typing import Iterable

import gmsh
import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
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

_EMPTY = Polygon()


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


def _simplify_polygon(poly: Polygon, min_dist: float, min_incl_angle: float, min_remove_angle: float) -> Polygon:
    def ring(r) -> np.ndarray:
        xy = np.asarray(r.coords)[:-1]
        return _simplify_ring(xy, min_dist, min_incl_angle, min_remove_angle)

    new = Polygon(ring(poly.exterior), [ring(h) for h in poly.interiors])
    if new.is_valid and not new.is_empty:
        return new
    return poly


############################################################
#                     PRIMITIVE GEOMETRY                   #
############################################################

class _Builder:
    """Converts pygerber commands into shapely geometries (in mm)."""

    def __init__(self, arc_tol_mm: float, n_circ_segments: int, seg_size_mm: float | None):
        self.arc_tol = arc_tol_mm
        self.nseg = n_circ_segments
        self.seg_size = seg_size_mm

    # ---------------- helpers ----------------

    def _quad_segs(self, diameter: float) -> int:
        return max(1, math.ceil(_calc_via_segs(diameter, self.nseg, self.seg_size)/4))

    def _circle(self, cx: float, cy: float, diameter: float) -> BaseGeometry:
        if diameter <= 0:
            return _EMPTY
        return Point(cx, cy).buffer(diameter/2, quad_segs=self._quad_segs(diameter))

    def _with_hole(self, geom: BaseGeometry, aperture, cx: float, cy: float) -> BaseGeometry:
        hole = getattr(aperture, 'hole_diameter', None)
        if hole is None:
            return geom
        return geom.difference(self._circle(cx, cy, _mm(hole)))

    def _arc_points(self, cmd: Arc2) -> np.ndarray:
        """Samples the centre line of an arc command, start and end inclusive."""
        x0, y0 = _xy(cmd.start_point)
        x1, y1 = _xy(cmd.end_point)
        xc, yc = _xy(cmd.center_point)
        R = math.hypot(x0 - xc, y0 - yc)
        if R == 0:
            return np.array([[x0, y0], [x1, y1]])

        a0 = math.atan2(y0 - yc, x0 - xc)
        a1 = math.atan2(y1 - yc, x1 - xc)
        if isinstance(cmd, CCArc2):
            sweep = (a1 - a0) % (2*math.pi)
            if sweep < 1e-9:
                sweep = 2*math.pi
        else:
            sweep = -((a0 - a1) % (2*math.pi))
            if sweep > -1e-9:
                sweep = -2*math.pi

        # Angular step such that the chord deviates at most arc_tol from the true arc
        dth = 2*math.acos(max(1 - self.arc_tol/R, -1.0)) if self.arc_tol < R else math.pi/2
        n = max(2, math.ceil(abs(sweep)/max(dth, 1e-6)) + 1, math.ceil(abs(sweep)/(2*math.pi)*N_CIRC_MIN) + 1)
        th = a0 + np.linspace(0, sweep, n)
        pts = np.column_stack((xc + R*np.cos(th), yc + R*np.sin(th)))
        pts[-1] = (x1, y1)  # land exactly on the end point
        return pts

    def _stroke(self, pts: np.ndarray, aperture) -> BaseGeometry:
        """Strokes a polyline with an aperture (D01 draw)."""
        if isinstance(aperture, Circle2):
            d = _mm(aperture.diameter)
            if d <= 0:
                return _EMPTY
            if np.allclose(pts[0], pts[1:]):
                return self._circle(pts[0, 0], pts[0, 1], d)
            return LineString(pts).buffer(d/2, quad_segs=self._quad_segs(d))

        if isinstance(aperture, Rectangle2) and not isinstance(aperture, Obround2) and len(pts) == 2:
            # Minkowski sum of a rectangle and a segment = hull of the two end-rectangles
            hx, hy = _mm(aperture.x_size)/2, _mm(aperture.y_size)/2
            rects = [box(x - hx, y - hy, x + hx, y + hy) for x, y in pts]
            return shapely.union_all(rects).convex_hull

        # Rare in practice: approximate with a round aperture of the stroke width
        logger.warning(f'Drawing with aperture {type(aperture).__name__} is approximated by a round stroke.')
        d = _mm(aperture.get_stroke_width())
        return LineString(pts).buffer(d/2, quad_segs=self._quad_segs(d)) if d > 0 else _EMPTY

    def _flash(self, cmd: Flash2) -> BaseGeometry:
        ap = cmd.aperture
        cx, cy = _xy(cmd.flash_point)

        if isinstance(ap, Circle2):
            return self._with_hole(self._circle(cx, cy, _mm(ap.diameter)), ap, cx, cy)

        if isinstance(ap, Obround2):
            sx, sy = _mm(ap.x_size), _mm(ap.y_size)
            r = min(sx, sy)/2
            if sx >= sy:
                core = LineString([(-sx/2 + r, 0), (sx/2 - r, 0)])
            else:
                core = LineString([(0, -sy/2 + r), (0, sy/2 - r)])
            geom = core.buffer(r, quad_segs=self._quad_segs(2*r)) if core.length > 0 else Point(0, 0).buffer(r, quad_segs=self._quad_segs(2*r))
            geom = affinity.rotate(geom, float(ap.rotation), origin=(0, 0))
            return self._with_hole(affinity.translate(geom, cx, cy), ap, cx, cy)

        if isinstance(ap, Rectangle2):
            hx, hy = _mm(ap.x_size)/2, _mm(ap.y_size)/2
            geom = affinity.rotate(box(-hx, -hy, hx, hy), float(ap.rotation), origin=(0, 0))
            return self._with_hole(affinity.translate(geom, cx, cy), ap, cx, cy)

        if isinstance(ap, Polygon2):
            n = int(ap.number_vertices)
            R = _mm(ap.outer_diameter)/2
            th = np.radians(float(ap.rotation)) + 2*np.pi*np.arange(n)/n
            geom = Polygon(np.column_stack((cx + R*np.cos(th), cy + R*np.sin(th))))
            return self._with_hole(geom, ap, cx, cy)

        if hasattr(ap, 'command_buffer'):
            # Macro (AM) or block (AB) aperture: primitives are relative to the flash point
            # and their exposure only affects the aperture itself.
            geom = merge_commands(ap.command_buffer, self)
            return affinity.translate(geom, cx, cy)

        logger.error(f'Unsupported aperture {type(ap).__name__} in flash at ({cx}, {cy}) mm. '
                     'Please contact the EMerge developers with this Gerber file to get it supported.')
        return _EMPTY

    def _region(self, cmd: Region2) -> BaseGeometry:
        """Builds a region (G36/G37). A new contour starts wherever a segment does not
        continue from the previous end point (D02 move)."""
        contours: list[list[np.ndarray]] = []
        last_end = None
        for seg in cmd.command_buffer:
            if isinstance(seg, Arc2):
                pts = self._arc_points(seg)
            elif isinstance(seg, Line2):
                pts = np.array([_xy(seg.start_point), _xy(seg.end_point)])
            else:
                logger.warning(f'Unhandled region segment {type(seg).__name__}')
                continue
            if last_end is None or not np.allclose(pts[0], last_end, atol=1e-9):
                contours.append([pts])
            else:
                contours[-1].append(pts[1:])
            last_end = pts[-1]

        polys = []
        for parts in contours:
            ring = np.vstack(parts)
            if len(ring) < 3:
                continue
            # 'structure' turns KiCad style cut-in contours into proper holes
            polys.append(shapely.make_valid(Polygon(ring), method='structure', keep_collapsed=False))
        return shapely.union_all(polys) if polys else _EMPTY

    # ---------------- dispatch ----------------

    def geometry(self, cmd) -> BaseGeometry | None:
        if isinstance(cmd, Region2):
            return self._region(cmd)
        if isinstance(cmd, Arc2):
            return self._stroke(self._arc_points(cmd), cmd.aperture)
        if isinstance(cmd, Line2):
            pts = np.array([_xy(cmd.start_point), _xy(cmd.end_point)])
            return self._stroke(pts, cmd.aperture)
        if isinstance(cmd, Flash2):
            return self._flash(cmd)
        logger.warning(f'Unprocessed Gerber command: {type(cmd).__name__}')
        return None


def merge_commands(commands: Iterable, builder: _Builder) -> BaseGeometry:
    """Converts a stream of commands to geometry and applies the dark/clear polarity
    in order. Consecutive commands of equal polarity are merged in a single union."""
    result: BaseGeometry = _EMPTY
    batch: list[BaseGeometry] = []
    batch_dark = True

    def flush() -> None:
        nonlocal result
        if not batch:
            return
        merged = shapely.union_all(batch)
        if batch_dark:
            result = shapely.union(result, merged)
        elif not result.is_empty:
            result = shapely.difference(result, merged)
        batch.clear()

    for cmd in commands:
        geom = builder.geometry(cmd)
        if geom is None or geom.is_empty:
            continue
        dark = _is_dark(cmd)
        if dark != batch_dark:
            flush()
            batch_dark = dark
        batch.append(geom)
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
        builder = _Builder(res_mm, n_circ_segments, seg_size_mm)
        geom = merge_commands(commands, builder)

        # Snap to a fine grid: removes slivers and near-coincident vertices left by the booleans
        geom = shapely.set_precision(geom, res_mm/10)
        polys = [p for p in shapely.get_parts(geom) if isinstance(p, Polygon) and not p.is_empty]
        if simplify:
            polys = [_simplify_polygon(p, res_mm, min_incl_angle, min_remove_angle) for p in polys]

        #: Merged layer polygons in mm
        self.polygons: list[Polygon] = polys

        if polys:
            x0, y0, x1, y1 = shapely.MultiPolygon(polys).bounds
            self.xs: list[float] = [x0*1e-3, x1*1e-3]
            self.ys: list[float] = [y0*1e-3, y1*1e-3]
        else:
            logger.warning(f'Gerber file {filename} produced no copper geometry.')
            self.xs, self.ys = [], []

    @property
    def n_points(self) -> int:
        return sum(len(p.exterior.coords) - 1 + sum(len(h.coords) - 1 for h in p.interiors) for p in self.polygons)

    def bounds(self, margin: float | tuple[float, float, float, float] = 0.0) -> tuple[float, float, float, float]:
        if isinstance(margin, (float, int)):
            margin = (margin, margin, margin, margin)
        return min(self.xs)-margin[0], min(self.ys)-margin[1], max(self.xs)+margin[2], max(self.ys)+margin[3]

    def _curve_loop(self, ring, z: float) -> int:
        xy = np.asarray(ring.coords)[:-1]*1e-3
        xg, yg, zg = self.cs.in_global_cs(xy[:, 0], xy[:, 1], np.full(len(xy), z))
        occ = gmsh.model.occ
        ptags = [occ.addPoint(x, y, zz) for x, y, zz in zip(xg, yg, zg)]
        n = len(ptags)
        lines = [occ.addLine(ptags[i], ptags[(i + 1) % n]) for i in range(n)]
        return occ.addCurveLoop(lines)

    def flatten(self, z: float = 0.0) -> GeoSurface:
        """Creates the gmsh surfaces of this layer at height z (meters).

        Returns:
            GeoSurface: One surface object containing all disjoint copper areas.
        """
        if not self.polygons:
            raise ValueError(f'Gerber file {self.fname} contains no copper geometry.')
        tags = []
        for poly in self.polygons:
            loops = [self._curve_loop(poly.exterior, z)]
            loops += [self._curve_loop(h, z) for h in poly.interiors]
            tags.append(gmsh.model.occ.addPlaneSurface(loops))
        return GeoSurface(tags, name='GerberLayer')
