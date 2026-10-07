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

"""Gerber import with exact curved boundaries.

Instead of discretizing every round shape into a polygon, each Gerber object is
built directly as an OCC face with true circular arcs:

    * round-aperture lines   -> stadium (2 lines + 2 half circles)
    * round-aperture arcs    -> annular sector with round caps
    * circle / obround flash -> disk / stadium
    * regions                -> wires of lines and exact arcs

Dark/clear polarity is applied with OCC booleans. No polygon simplification is
needed since round features contribute only a handful of exact edges. Shapes that
cannot be represented exactly (KiCad cut-in regions, arcs thinner than their
aperture) fall back to the polygonal representation of the shapely path.
"""

from __future__ import annotations

import math
from typing import Iterable

import gmsh
import numpy as np
import shapely
from shapely.geometry import Polygon
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
from .gerber import _Builder, _mm, _xy, _is_dark

MM = 1e-3            # Gerber geometry is handled in mm, gmsh works in meters
_MAX_ARC = math.radians(120)  # longest single OCC arc segment
_EPS = 5e-4          # mm, well above the OCC tolerance of 1e-7 m

DimTags = list[tuple[int, int]]


def _arc_sweep(cmd: Arc2) -> float:
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


class _Wire:
    """Builds a closed OCC wire from lines and exact circular arcs (coordinates in mm)."""

    def __init__(self, x: float, y: float, centers: list[int]):
        self._first = self._last = gmsh.model.occ.addPoint(x*MM, y*MM, 0)
        self._xy0 = self._xy = (x, y)
        self._curves: list[int] = []
        self._centers = centers
        self._area2 = 0.0  # twice the signed area, to orient every face towards +z
        self._prev: tuple[int, int | None] | None = None  # (start point, arc center) of the last curve

    def _track(self, x: float, y: float) -> None:
        self._area2 += self._xy[0]*y - x*self._xy[1]

    def _point(self, x: float, y: float, close: bool) -> int:
        if close:
            return self._first
        return gmsh.model.occ.addPoint(x*MM, y*MM, 0)

    def _reclose(self) -> None:
        """Re-ends the last curve on the first point when the closing gap is negligible."""
        occ = gmsh.model.occ
        start, center = self._prev
        occ.remove([(1, self._curves.pop())])
        occ.remove([(0, self._last)])
        self._curves.append(occ.addLine(start, self._first) if center is None
                            else occ.addCircleArc(start, center, self._first))
        self._last = self._first

    def line_to(self, x: float, y: float, close: bool = False) -> _Wire:
        if math.hypot(x - self._xy[0], y - self._xy[1]) < _EPS:
            if close and self._last != self._first and self._prev is not None:
                self._reclose()
            return self
        p = self._point(x, y, close)
        self._prev = (self._last, None)
        self._curves.append(gmsh.model.occ.addLine(self._last, p))
        self._track(x, y)
        self._last, self._xy = p, (x, y)
        return self

    def arc_to(self, cx: float, cy: float, sweep: float, close: bool = False) -> _Wire:
        """Arc around (cx, cy) from the current point over `sweep` radians (CCW positive)."""
        x0, y0 = self._xy
        r = math.hypot(x0 - cx, y0 - cy)
        a0 = math.atan2(y0 - cy, x0 - cx)
        n = max(1, math.ceil(abs(sweep)/_MAX_ARC - 1e-9))
        c = gmsh.model.occ.addPoint(cx*MM, cy*MM, 0)
        self._centers.append(c)
        for i in range(1, n + 1):
            a = a0 + sweep*i/n
            x, y = cx + r*math.cos(a), cy + r*math.sin(a)
            p = self._point(x, y, close and i == n)
            self._prev = (self._last, c)
            self._curves.append(gmsh.model.occ.addCircleArc(self._last, c, p))
            am = a0 + sweep*(i - 0.5)/n
            self._track(cx + r*math.cos(am), cy + r*math.sin(am))
            self._xy = (cx + r*math.cos(am), cy + r*math.sin(am))
            self._track(x, y)
            self._last, self._xy = p, (x, y)
        return self

    def face(self) -> int:
        if self._last != self._first:
            self.line_to(*self._xy0, close=True)
        curves = self._curves if self._area2 >= 0 else [-c for c in reversed(self._curves)]
        loop = gmsh.model.occ.addCurveLoop(curves)
        return gmsh.model.occ.addPlaneSurface([loop])


class _OCCBuilder:
    """Converts pygerber commands into OCC faces with exact arcs (built at z=0, meters)."""

    def __init__(self, res_mm: float, n_circ_segments: int, seg_size_mm: float | None):
        self._poly = _Builder(res_mm, n_circ_segments, seg_size_mm)
        self.centers: list[int] = []
        self.n_fallback = 0

    # ---------------- primitives ----------------

    def disk(self, cx: float, cy: float, d: float) -> DimTags:
        if d <= 0:
            return []
        return [(2, gmsh.model.occ.addDisk(cx*MM, cy*MM, 0, d/2*MM, d/2*MM))]

    def stadium(self, x1: float, y1: float, x2: float, y2: float, r: float) -> DimTags:
        """A line stroked with a round aperture of radius r."""
        if r <= 0:
            return []
        L = math.hypot(x2 - x1, y2 - y1)
        if L < _EPS:
            return self.disk(x1, y1, 2*r)
        ux, uy = (x2 - x1)/L, (y2 - y1)/L
        nx, ny = uy, -ux  # right hand normal
        w = _Wire(x1 + r*nx, y1 + r*ny, self.centers)
        w.line_to(x2 + r*nx, y2 + r*ny).arc_to(x2, y2, math.pi)
        w.line_to(x1 - r*nx, y1 - r*ny).arc_to(x1, y1, math.pi, close=True)
        return [(2, w.face())]

    def polygon(self, pts: Iterable[tuple[float, float]]) -> DimTags:
        pts = list(pts)
        w = _Wire(*pts[0], self.centers)
        for x, y in pts[1:]:
            w.line_to(x, y)
        return [(2, w.face())]

    def from_shapely(self, geom) -> DimTags:
        """Fallback: build linear faces from a shapely (multi)polygon in mm."""
        self.n_fallback += 1
        out = []
        for p in shapely.get_parts(geom):
            if not isinstance(p, Polygon) or p.is_empty:
                continue
            loops = []
            for ring in [p.exterior, *p.interiors]:
                xy = np.asarray(ring.coords)[:-1]
                pts = [gmsh.model.occ.addPoint(x*MM, y*MM, 0) for x, y in xy]
                lines = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
                loops.append(gmsh.model.occ.addCurveLoop(lines))
            out.append((2, gmsh.model.occ.addPlaneSurface(loops)))
        return out

    def _cut_hole(self, faces: DimTags, ap, cx: float, cy: float) -> DimTags:
        hole = getattr(ap, 'hole_diameter', None)
        if hole is None or not faces:
            return faces
        out, _ = gmsh.model.occ.cut(faces, self.disk(cx, cy, _mm(hole)))
        return out

    # ---------------- commands ----------------

    def _stroke_line(self, cmd: Line2) -> DimTags:
        ap = cmd.aperture
        (x1, y1), (x2, y2) = _xy(cmd.start_point), _xy(cmd.end_point)
        if isinstance(ap, Circle2):
            return self.stadium(x1, y1, x2, y2, _mm(ap.diameter)/2)
        return self.from_shapely(self._poly.geometry(cmd))

    def _stroke_arc(self, cmd: Arc2) -> DimTags:
        ap = cmd.aperture
        if not isinstance(ap, Circle2):
            return self.from_shapely(self._poly.geometry(cmd))
        w = _mm(ap.diameter)/2
        if w <= 0:
            return []
        x0, y0 = _xy(cmd.start_point)
        xc, yc = _xy(cmd.center_point)
        R = math.hypot(x0 - xc, y0 - yc)
        sweep = _arc_sweep(cmd)

        if R*abs(sweep) < _EPS:
            return self.disk(x0, y0, 2*w)
        if R - w < _EPS:
            # Aperture wider than the arc radius: no inner boundary, use the polygon path
            return self.from_shapely(self._poly.geometry(cmd))

        if abs(abs(sweep) - 2*math.pi) < 1e-9:
            out, _ = gmsh.model.occ.cut(self.disk(xc, yc, 2*(R + w)), self.disk(xc, yc, 2*(R - w)))
            return out

        s = math.copysign(1.0, sweep)
        a0 = math.atan2(y0 - yc, x0 - xc)
        a1 = a0 + sweep
        ex, ey = xc + R*math.cos(a1), yc + R*math.sin(a1)
        wire = _Wire(xc + (R + w)*math.cos(a0), yc + (R + w)*math.sin(a0), self.centers)
        wire.arc_to(xc, yc, sweep)                    # outer arc
        wire.arc_to(ex, ey, s*math.pi)               # end cap
        wire.arc_to(xc, yc, -sweep)                  # inner arc, back
        wire.arc_to(x0, y0, s*math.pi, close=True)   # start cap
        return [(2, wire.face())]

    def _flash(self, cmd: Flash2) -> DimTags:
        ap = cmd.aperture
        cx, cy = _xy(cmd.flash_point)

        if isinstance(ap, Circle2):
            return self._cut_hole(self.disk(cx, cy, _mm(ap.diameter)), ap, cx, cy)

        if isinstance(ap, Obround2):
            sx, sy = _mm(ap.x_size), _mm(ap.y_size)
            r = min(sx, sy)/2
            h = max(sx, sy)/2 - r
            rot = math.radians(float(ap.rotation)) + (0 if sx >= sy else math.pi/2)
            dx, dy = h*math.cos(rot), h*math.sin(rot)
            faces = self.stadium(cx - dx, cy - dy, cx + dx, cy + dy, r)
            return self._cut_hole(faces, ap, cx, cy)

        if isinstance(ap, Rectangle2):
            hx, hy = _mm(ap.x_size)/2, _mm(ap.y_size)/2
            c, s = math.cos(math.radians(float(ap.rotation))), math.sin(math.radians(float(ap.rotation)))
            pts = [(cx + x*c - y*s, cy + x*s + y*c) for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]
            return self._cut_hole(self.polygon(pts), ap, cx, cy)

        if isinstance(ap, Polygon2):
            n = int(ap.number_vertices)
            R = _mm(ap.outer_diameter)/2
            th0 = math.radians(float(ap.rotation))
            pts = [(cx + R*math.cos(th0 + 2*math.pi*i/n), cy + R*math.sin(th0 + 2*math.pi*i/n)) for i in range(n)]
            return self._cut_hole(self.polygon(pts), ap, cx, cy)

        if hasattr(ap, 'command_buffer'):
            faces = merge_commands_occ(ap.command_buffer, self)
            if faces:
                gmsh.model.occ.translate(faces, cx*MM, cy*MM, 0)
            return faces

        logger.error(f'Unsupported aperture {type(ap).__name__} in flash at ({cx}, {cy}) mm.')
        return []

    def _region(self, cmd: Region2) -> DimTags:
        # Split into contours at every discontinuity (D02 move)
        contours: list[list] = []
        last_end = None
        for seg in cmd.command_buffer:
            if not isinstance(seg, (Line2, Arc2)):
                continue
            start, end = _xy(seg.start_point), _xy(seg.end_point)
            if last_end is None or math.hypot(start[0] - last_end[0], start[1] - last_end[1]) > _EPS:
                contours.append([])
            contours[-1].append(seg)
            last_end = end

        faces: DimTags = []
        for segs in contours:
            # Self touching (cut-in) contours are not valid OCC faces: use the polygon path
            ring = [_xy(segs[0].start_point)]
            for seg in segs:
                ring.extend(self._poly._arc_points(seg)[1:].tolist() if isinstance(seg, Arc2) else [_xy(seg.end_point)])
            if len(ring) < 4:
                continue
            if not Polygon(ring).is_valid:
                faces += self.from_shapely(shapely.make_valid(Polygon(ring), method='structure', keep_collapsed=False))
                continue

            wire = _Wire(*ring[0], self.centers)
            for i, seg in enumerate(segs):
                last = i == len(segs) - 1
                if isinstance(seg, Arc2):
                    wire.arc_to(*_xy(seg.center_point), _arc_sweep(seg), close=last)
                else:
                    wire.line_to(*_xy(seg.end_point), close=last)
            faces.append((2, wire.face()))
        if len(faces) > 1:
            faces, _ = gmsh.model.occ.fuse(faces[:1], faces[1:])
        return faces

    def faces(self, cmd) -> DimTags:
        if isinstance(cmd, Region2):
            return self._region(cmd)
        if isinstance(cmd, Arc2):
            return self._stroke_arc(cmd)
        if isinstance(cmd, Line2):
            return self._stroke_line(cmd)
        if isinstance(cmd, Flash2):
            return self._flash(cmd)
        logger.warning(f'Unprocessed Gerber command: {type(cmd).__name__}')
        return []


def merge_commands_occ(commands: Iterable, builder: _OCCBuilder) -> DimTags:
    """Applies dark/clear polarity in order with OCC booleans. Consecutive objects of
    equal polarity are combined in a single boolean operation."""
    result: DimTags = []
    batch: DimTags = []
    batch_dark = True

    def flush() -> None:
        nonlocal result
        if not batch:
            return
        if batch_dark:
            objs = result or batch[:1]
            tools = batch if result else batch[1:]
            result = gmsh.model.occ.fuse(objs, tools)[0] if tools else objs
        elif result:
            result = gmsh.model.occ.cut(result, batch)[0]
        else:
            gmsh.model.occ.remove(batch, recursive=True)
        batch.clear()

    for cmd in commands:
        faces = builder.faces(cmd)
        if not faces:
            continue
        dark = _is_dark(cmd)
        if dark != batch_dark:
            flush()
            batch_dark = dark
        batch.extend(faces)
    flush()
    return result


class CurvedGerberLayer:
    """A Gerber copper layer built from exact lines and circular arcs.

    Args:
        filename (str): Path to the Gerber file.
        res_mm (float): Resolution used only for shapes that fall back to polygons.
        n_circ_segments (int): Circle segments for polygon fallbacks.
        seg_size_mm (float | None): Segment size for polygon fallbacks.
        cs (CoordinateSystem): The coordinate system to place the layer in.
    """

    def __init__(self,
                 filename: str,
                 res_mm: float = 0.01,
                 n_circ_segments: int = 16,
                 seg_size_mm: float | None = None,
                 cs: CoordinateSystem = GCS):
        self.fname = filename
        self.cs = cs
        self._commands = GerberFile.from_file(filename).parse()._command_buffer
        self._builder_args = (res_mm, n_circ_segments, seg_size_mm)
        self.xs: list[float] = []
        self.ys: list[float] = []

    def flatten(self, z: float = 0.0) -> GeoSurface:
        """Builds the layer in gmsh at local height z (meters) and returns its surfaces."""
        builder = _OCCBuilder(*self._builder_args)
        faces = merge_commands_occ(self._commands, builder)
        if builder.centers:
            gmsh.model.occ.remove([(0, t) for t in builder.centers])
        if builder.n_fallback:
            logger.info(f'{self.fname}: {builder.n_fallback} object(s) built from polygons (no exact representation).')
        if not faces:
            raise ValueError(f'Gerber file {self.fname} contains no copper geometry.')

        x0, y0, _, x1, y1, _ = np.array([gmsh.model.occ.getBoundingBox(d, t) for d, t in faces]).T
        self.xs = [float(x0.min()), float(x1.max())]
        self.ys = [float(y0.min()), float(y1.max())]

        gmsh.model.occ.translate(faces, 0, 0, z)
        if not getattr(self.cs, '_is_global', False):
            gmsh.model.occ.affineTransform(faces, self.cs.affine_to_global()[:3, :].flatten().tolist())
        return GeoSurface([t for _, t in faces], name='GerberLayer')
