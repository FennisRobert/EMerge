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

"""Minimal Excellon (NC drill) reader that extracts drill hits and their radii.

Supported:
  * METRIC / INCH (also M71 / M72), with optional LZ/TZ and digit format (e.g. ``METRIC,TZ,000.000``)
  * ``;FILE_FORMAT=i:d`` comments
  * Explicit decimal coordinates (KiCad default) and implicit-decimal coordinates
  * Tool tables (``T1C0.300``, ``T01F00S00C0.0354``) and tool selection
  * Modal coordinates, absolute (G90) and incremental (G91) mode

Slots (G85) and routed paths (G00/M15/M16) are skipped with a warning since they are not vias.
"""

from __future__ import annotations

import re
from loguru import logger


class ExcellonParseError(Exception):
    pass


_RE_FILEFMT = re.compile(r"FILE_FORMAT\s*=\s*(\d+)\s*:\s*(\d+)", re.I)
_RE_UNITS = re.compile(r"^(INCH|METRIC)\b(?:\s*,\s*(LZ|TZ))?(?:\s*,\s*(0*)\.(0*))?", re.I)
_RE_TOOL_DEF = re.compile(r"^T(\d+)(?:[ABD-Z][+-]?[\d.]*)*C\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.I)
_RE_TOOL_SEL = re.compile(r"^T(\d+)(?=[XY]|$)", re.I)
_RE_X = re.compile(r"X([+-]?[\d.]+)", re.I)
_RE_Y = re.compile(r"Y([+-]?[\d.]+)", re.I)


def _parse_number(s: str, int_digits: int, dec_digits: int, zero_sup: str) -> float:
    """Parse a coordinate field. Explicit decimals are used as is; otherwise the
    digit format and zero suppression mode decide where the decimal point goes.

    LZ: leading zeros are kept (trailing omitted) -> digits are read from the left.
    TZ: trailing zeros are kept (leading omitted) -> digits are read from the right.
    """
    if "." in s:
        return float(s)
    neg = s.startswith("-")
    digits = s.lstrip("+-")
    if not digits.isdigit():
        raise ExcellonParseError(f"Bad coordinate field: {s}")
    if zero_sup == "LZ":
        digits = digits.ljust(int_digits + dec_digits, "0")
    value = int(digits) / 10**dec_digits
    return -value if neg else value


def parse_excellon(text: str, *, convert_to: str | None = "mm") -> list[dict]:
    """Parse an Excellon file and return the drill hits.

    Args:
        text (str): The file contents.
        convert_to (str | None): "mm", "inch" or None for the native file unit.

    Returns:
        list[dict]: Dicts with keys x, y, radius (None if the tool size is unknown), unit and tool.
    """
    unit: str | None = None
    zero_sup = "LZ"
    fmt: tuple[int, int] | None = None
    tools: dict[int, float] = {}
    tool: int | None = None
    x = y = None
    incremental = False
    routing = False
    n_slots = n_routed = 0
    raw_hits: list[tuple[float, float, int | None]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(";"):
            m = _RE_FILEFMT.search(line)
            if m:
                fmt = (int(m.group(1)), int(m.group(2)))
            continue

        up = line.upper()
        if up.startswith(("M30", "M00")):
            break
        if up in ("G90", "G91"):
            incremental = up == "G91"
            continue
        if up in ("M71", "M72"):
            unit = "mm" if up == "M71" else "inch"
            continue
        if up.startswith(("G05", "M16", "M17")):
            routing = False
            continue
        if up.startswith(("G00", "M15")):
            routing = True
        if up in ("M48", "%", "M95"):
            continue

        m = _RE_UNITS.match(up)
        if m:
            unit = "mm" if m.group(1) == "METRIC" else "inch"
            if m.group(2):
                zero_sup = m.group(2)
            if m.group(4) is not None:
                fmt = (len(m.group(3)), len(m.group(4)))
            continue

        m = _RE_TOOL_DEF.match(up)
        if m:
            tools[int(m.group(1))] = float(m.group(2))
            continue

        m = _RE_TOOL_SEL.match(up)
        if m:
            tool = int(m.group(1))
            up = up[m.end():]
            if not up:
                continue

        is_slot = "G85" in up
        if is_slot:
            n_slots += 1
            up = up.split("G85")[-1]  # keep the modal position at the slot end

        mx, my = _RE_X.search(up), _RE_Y.search(up)
        if mx is None and my is None:
            continue

        int_d, dec_d = fmt if fmt is not None else ((3, 3) if unit != "inch" else (2, 4))
        if mx:
            v = _parse_number(mx.group(1), int_d, dec_d, zero_sup)
            x = (x or 0.0) + v if incremental else v
        if my:
            v = _parse_number(my.group(1), int_d, dec_d, zero_sup)
            y = (y or 0.0) + v if incremental else v

        if is_slot:
            continue
        if routing or up.startswith(("G01", "G02", "G03")):
            n_routed += 1
            continue
        if x is None or y is None:
            continue
        raw_hits.append((x, y, tool))

    if n_slots:
        logger.warning(f"Excellon: skipped {n_slots} slot(s) (G85).")
    if n_routed:
        logger.warning(f"Excellon: skipped {n_routed} routed coordinate(s).")

    native = unit or "mm"
    scale = 1.0
    if convert_to is not None and convert_to != native:
        if (native, convert_to) == ("inch", "mm"):
            scale = 25.4
        elif (native, convert_to) == ("mm", "inch"):
            scale = 1/25.4
        else:
            raise ExcellonParseError(f"Unknown conversion {native}->{convert_to}")

    hits = []
    for hx, hy, t in raw_hits:
        diam = tools.get(t) if t is not None else None
        hits.append({
            "x": hx*scale,
            "y": hy*scale,
            "radius": None if diam is None else diam/2*scale,
            "unit": convert_to or native,
            "tool": f"T{t:02d}" if t is not None else "T??",
        })
    return hits


def parse_excellon_file(path: str, **kwargs) -> list[dict]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse_excellon(f.read(), **kwargs)
