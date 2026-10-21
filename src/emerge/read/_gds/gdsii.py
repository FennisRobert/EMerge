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

"""Dependency free GDSII stream reader.

Reads the binary GDSII records into cells and flattens the cell hierarchy
into plain polygons (in meters) per (layer, datatype) and text labels.

Supported elements: BOUNDARY, BOX, PATH (path types 0, 1, 2 and 4), SREF,
AREF (with reflection, magnification and rotation) and TEXT. NODE elements
and properties are skipped.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

############################################################
#                       RECORD TYPES                       #
############################################################

HEADER, BGNLIB, LIBNAME, UNITS, ENDLIB = 0x00, 0x01, 0x02, 0x03, 0x04
BGNSTR, STRNAME, ENDSTR = 0x05, 0x06, 0x07
BOUNDARY, PATH, SREF, AREF, TEXT = 0x08, 0x09, 0x0A, 0x0B, 0x0C
LAYER, DATATYPE, WIDTH, XY, ENDEL, SNAME, COLROW = 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13
NODE, TEXTTYPE, STRING, STRANS, MAG, ANGLE = 0x15, 0x16, 0x19, 0x1A, 0x1B, 0x1C
PATHTYPE, BOX, BOXTYPE, BGNEXTN, ENDEXTN = 0x21, 0x2D, 0x2E, 0x30, 0x31

_ELEMENTS = (BOUNDARY, PATH, SREF, AREF, TEXT, NODE, BOX)

# Data type codes of the record payload
_DT_INT2, _DT_INT4, _DT_REAL8, _DT_ASCII = 0x02, 0x03, 0x05, 0x06

N_ROUND_CAP = 16  # segments of a full circle for round (type 1) path ends


class GDSParseError(Exception):
    pass


def _real8(b: bytes) -> float:
    """GDSII 8-byte excess-64 base-16 real."""
    sign = -1.0 if b[0] & 0x80 else 1.0
    exponent = (b[0] & 0x7F) - 64
    mantissa = int.from_bytes(b[1:8], "big")
    return sign * mantissa * 16.0 ** (exponent - 14)


def _decode(datatype: int, body: bytes):
    if datatype == _DT_INT2:
        return struct.unpack(f">{len(body) // 2}h", body)
    if datatype == _DT_INT4:
        return struct.unpack(f">{len(body) // 4}i", body)
    if datatype == _DT_REAL8:
        return tuple(_real8(body[i:i + 8]) for i in range(0, len(body), 8))
    if datatype == _DT_ASCII:
        return body.rstrip(b"\0").decode("ascii", errors="replace")
    return body


def _records(data: bytes):
    """Yields (record_type, value) for every record in the stream."""
    i, n = 0, len(data)
    while i + 4 <= n:
        size, rtype, dtype = struct.unpack(">HBB", data[i:i + 4])
        if size < 4:
            if size == 0:  # zero padding after ENDLIB
                return
            raise GDSParseError(f"Invalid record size {size} at byte {i}")
        yield rtype, _decode(dtype, data[i + 4:i + size])
        i += size
        if rtype == ENDLIB:
            return


############################################################
#                        DATA CLASSES                      #
############################################################


@dataclass
class GDSPolygon:
    layer: int
    datatype: int
    xy: np.ndarray  # (N, 2) in meters, not closed (first point is not repeated)


@dataclass
class GDSLabel:
    """A TEXT element, often used to mark ports or pins."""
    text: str
    layer: int
    texttype: int
    x: float  # meters
    y: float  # meters


@dataclass
class _Ref:
    cell: str
    origin: tuple[float, float]
    reflect: bool = False
    mag: float = 1.0
    angle: float = 0.0  # degrees
    cols: int = 1
    rows: int = 1
    col_step: tuple[float, float] = (0.0, 0.0)
    row_step: tuple[float, float] = (0.0, 0.0)


@dataclass
class GDSCell:
    name: str
    polygons: list[GDSPolygon] = field(default_factory=list)
    labels: list[GDSLabel] = field(default_factory=list)
    refs: list[_Ref] = field(default_factory=list)


############################################################
#                       PATH OUTLINES                      #
############################################################


def path_outline(xy: np.ndarray, width: float, pathtype: int = 0,
                 bgn_ext: float = 0.0, end_ext: float = 0.0) -> list[np.ndarray]:
    """Converts a GDS path centerline into polygon outline(s) with mitered joints.

    Round paths (type 1) return the outline plus a circle polygon at each end;
    these overlap and are merged later together with the rest of the layer.
    """
    pts = _dedupe(np.asarray(xy, dtype=float))
    hw = abs(width) / 2.0
    if len(pts) < 2 or hw == 0.0:
        return []

    if pathtype == 2:
        bgn_ext = end_ext = hw
    elif pathtype not in (4,):
        bgn_ext = end_ext = 0.0

    pts = pts.copy()
    d0 = pts[1] - pts[0]
    d1 = pts[-1] - pts[-2]
    pts[0] -= d0 / np.linalg.norm(d0) * bgn_ext
    pts[-1] += d1 / np.linalg.norm(d1) * end_ext

    left, right = [], []
    for i in range(len(pts)):
        if i == 0:
            n = _normal(pts[1] - pts[0])
            offset = n * hw
        elif i == len(pts) - 1:
            n = _normal(pts[-1] - pts[-2])
            offset = n * hw
        else:
            n1 = _normal(pts[i] - pts[i - 1])
            n2 = _normal(pts[i + 1] - pts[i])
            denom = 1.0 + float(np.dot(n1, n2))
            if denom < 1e-6:  # path folds back on itself, fall back to a plain offset
                offset = n1 * hw
            else:
                offset = (n1 + n2) * hw / denom
        left.append(pts[i] + offset)
        right.append(pts[i] - offset)

    outlines = [np.array(left + right[::-1])]
    if pathtype == 1:
        t = np.linspace(0, 2 * math.pi, N_ROUND_CAP, endpoint=False)
        circle = np.column_stack([np.cos(t), np.sin(t)]) * hw
        outlines += [pts[0] + circle, pts[-1] + circle]
    return outlines


def _normal(d: np.ndarray) -> np.ndarray:
    return np.array([-d[1], d[0]]) / np.linalg.norm(d)


def _dedupe(xy: np.ndarray) -> np.ndarray:
    """Removes consecutive duplicate points."""
    if len(xy) == 0:
        return xy
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = np.any(np.diff(xy, axis=0) != 0, axis=1)
    return xy[keep]


def clean_ring(xy: np.ndarray) -> np.ndarray | None:
    """Drops the closing point and duplicates. Returns None for degenerate rings."""
    xy = _dedupe(np.asarray(xy, dtype=float))
    if len(xy) > 1 and np.array_equal(xy[0], xy[-1]):
        xy = xy[:-1]
    if len(xy) < 3:
        return None
    x, y = xy[:, 0], xy[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area == 0.0:
        return None
    return xy


def drop_short_edges(xy: np.ndarray, min_edge: float) -> np.ndarray | None:
    """Removes vertices so that no edge is shorter than min_edge. GDS grid rounding often leaves
    one-database-unit steps at corners, which gmsh merges into zero-length lines.
    Returns None if fewer than 3 points remain."""
    kept = [xy[0]]
    for pt in xy[1:]:
        if np.linalg.norm(pt - kept[-1]) >= min_edge:
            kept.append(pt)
    while len(kept) > 2 and np.linalg.norm(kept[-1] - kept[0]) < min_edge:
        kept.pop()
    return clean_ring(np.array(kept)) if len(kept) >= 3 else None


############################################################
#                         LIBRARY                          #
############################################################


class GDSLibrary:
    """A parsed GDSII file.

    Args:
        filename (str | Path): Path to the .gds file.
    """

    def __init__(self, filename: str | Path):
        self.filename = Path(filename)
        self.name: str = ""
        self.user_unit: float = 1e-6   # meters per user unit (informational)
        self.db_unit: float = 1e-9     # meters per database unit
        self.cells: dict[str, GDSCell] = {}
        self._parse(self.filename.read_bytes())

    # ---------------------------------------------------------------- parse

    def _parse(self, data: bytes) -> None:
        cell: GDSCell | None = None
        el: dict | None = None
        skipped: set[str] = set()

        for rtype, value in _records(data):
            if rtype == LIBNAME:
                self.name = value
            elif rtype == UNITS:
                db_in_user, db_in_m = value
                self.db_unit = db_in_m
                self.user_unit = db_in_m / db_in_user
            elif rtype == BGNSTR:
                cell = GDSCell(name="")
            elif rtype == STRNAME and cell is not None:
                cell.name = value
            elif rtype == ENDSTR and cell is not None:
                self.cells[cell.name] = cell
                cell = None
            elif rtype in _ELEMENTS:
                el = {"type": rtype}
            elif rtype == ENDEL:
                if el is not None and cell is not None:
                    self._add_element(cell, el, skipped)
                el = None
            elif el is not None:
                el[rtype] = value

        if skipped:
            logger.debug(f"GDS: skipped unsupported elements: {sorted(skipped)}")
        if not self.cells:
            raise GDSParseError(f"No cells found in {self.filename}")

    def _xy(self, el: dict) -> np.ndarray:
        return np.asarray(el[XY], dtype=float).reshape(-1, 2) * self.db_unit

    def _add_element(self, cell: GDSCell, el: dict, skipped: set[str]) -> None:
        kind = el["type"]
        if kind in (BOUNDARY, BOX):
            ring = clean_ring(self._xy(el))
            if ring is not None:
                cell.polygons.append(GDSPolygon(el[LAYER][0], el.get(DATATYPE, el.get(BOXTYPE, (0,)))[0], ring))
        elif kind == PATH:
            width = el.get(WIDTH, (0,))[0] * self.db_unit
            pathtype = el.get(PATHTYPE, (0,))[0]
            bgn = el.get(BGNEXTN, (0,))[0] * self.db_unit
            end = el.get(ENDEXTN, (0,))[0] * self.db_unit
            for outline in path_outline(self._xy(el), width, pathtype, bgn, end):
                ring = clean_ring(outline)
                if ring is not None:
                    cell.polygons.append(GDSPolygon(el[LAYER][0], el.get(DATATYPE, (0,))[0], ring))
        elif kind in (SREF, AREF):
            cell.refs.append(self._ref(el))
        elif kind == TEXT:
            x, y = self._xy(el)[0]
            cell.labels.append(GDSLabel(el.get(STRING, ""), el[LAYER][0], el.get(TEXTTYPE, (0,))[0], float(x), float(y)))
        else:
            skipped.add("NODE")

    def _ref(self, el: dict) -> _Ref:
        strans = el.get(STRANS, b"\x00\x00")
        flags = int.from_bytes(strans, "big") if isinstance(strans, bytes) else 0
        if flags & 0x0006:
            logger.warning("GDS: absolute magnification/angle flags are not supported, treated as relative.")
        xy = self._xy(el)
        ref = _Ref(
            cell=el[SNAME],
            origin=(xy[0, 0], xy[0, 1]),
            reflect=bool(flags & 0x8000),
            mag=el.get(MAG, (1.0,))[0],
            angle=el.get(ANGLE, (0.0,))[0],
        )
        if el["type"] == AREF:
            ref.cols, ref.rows = el[COLROW]
            ref.col_step = tuple((xy[1] - xy[0]) / ref.cols)
            ref.row_step = tuple((xy[2] - xy[0]) / ref.rows)
        return ref

    # -------------------------------------------------------------- queries

    @property
    def top_cells(self) -> list[str]:
        """Cells that are not referenced by any other cell."""
        referenced = {r.cell for c in self.cells.values() for r in c.refs}
        return [name for name in self.cells if name not in referenced]

    def flatten(self, cell: str | None = None) -> tuple[list[GDSPolygon], list[GDSLabel]]:
        """Returns all polygons and labels of a cell with its hierarchy resolved.

        Args:
            cell (str | None): Cell name. Defaults to the single top cell.
        """
        if cell is None:
            tops = self.top_cells
            if len(tops) != 1:
                raise ValueError(f"The GDS file has {len(tops)} top cells {tops}, choose one with cell=...")
            cell = tops[0]
        if cell not in self.cells:
            raise KeyError(f"Cell '{cell}' not found. Available cells: {list(self.cells)}")

        polygons: list[GDSPolygon] = []
        labels: list[GDSLabel] = []
        self._flatten(cell, np.eye(3), polygons, labels, depth=0)
        return polygons, labels

    def _flatten(self, name: str, T: np.ndarray, polygons: list, labels: list, depth: int) -> None:
        if depth > 64:
            raise GDSParseError(f"Cell hierarchy too deep (recursive reference to '{name}'?)")
        cell = self.cells.get(name)
        if cell is None:
            logger.warning(f"GDS: referenced cell '{name}' is missing, skipped.")
            return

        for p in cell.polygons:
            xy = _apply(T, p.xy)
            if np.linalg.det(T[:2, :2]) < 0:  # keep orientation consistent after mirroring
                xy = xy[::-1]
            polygons.append(GDSPolygon(p.layer, p.datatype, xy))
        for lb in cell.labels:
            (x, y), = _apply(T, np.array([[lb.x, lb.y]]))
            labels.append(GDSLabel(lb.text, lb.layer, lb.texttype, float(x), float(y)))
        for ref in cell.refs:
            for c in range(ref.cols):
                for r in range(ref.rows):
                    ox = ref.origin[0] + c * ref.col_step[0] + r * ref.row_step[0]
                    oy = ref.origin[1] + c * ref.col_step[1] + r * ref.row_step[1]
                    self._flatten(ref.cell, T @ _transform(ox, oy, ref), polygons, labels, depth + 1)


def _transform(ox: float, oy: float, ref: _Ref) -> np.ndarray:
    """Reference transform: reflect about x, magnify, rotate, then translate."""
    a = math.radians(ref.angle)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]]) * ref.mag
    if ref.reflect:
        R = R @ np.diag([1.0, -1.0])
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = (ox, oy)
    return T


def _apply(T: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return xy @ T[:2, :2].T + T[:2, 2]
