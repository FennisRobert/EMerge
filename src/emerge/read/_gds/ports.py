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

"""Port edges derived from GDS text labels.

GDS has no notion of ports, only text labels (a string at an XY point on a layer).
When a label is placed on the edge of a metal shape, which is the common convention
for pins, that edge defines where the port sits, how wide it is and which way it faces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gdsii import GDSLabel, GDSPolygon


@dataclass
class GDSPortEdge:
    """The metal edge a port label sits on. All coordinates are in meters, in the layout plane.

    Attributes:
        label (GDSLabel): The label that defines the port.
        p0, p1 (np.ndarray): End points of the edge (2D).
        normal (np.ndarray): Unit vector pointing out of the metal across the edge (2D).
        z_bottom, z_top (float): Bottom and top of the metal layer, from the GDSLayer stack-up.
    """
    label: GDSLabel
    p0: np.ndarray
    p1: np.ndarray
    normal: np.ndarray
    z_bottom: float
    z_top: float

    @property
    def width(self) -> float:
        """Length of the edge (the port width along the metal end)."""
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def tangent(self) -> np.ndarray:
        return (self.p1 - self.p0) / self.width

    @property
    def center(self) -> np.ndarray:
        return (self.p0 + self.p1) / 2.0


def find_label_edge(label: GDSLabel, polygons: list[GDSPolygon], max_distance: float,
                    layer: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finds the polygon edge nearest to the label.

    Args:
        layer (int | None): Layer whose edges are searched. Defaults to the label's own layer. Use this
            when labels live on a separate text/pin layer next to the metal.

    Returns:
        (p0, p1, outward_normal) of the edge.

    Raises:
        ValueError: if no edge of the label's layer is within max_distance of the label.
    """
    point = np.array([label.x, label.y])
    layer = label.layer if layer is None else layer
    best: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None

    for poly in polygons:
        if poly.layer != layer:
            continue
        xy = poly.xy
        x, y = xy[:, 0], xy[:, 1]
        ccw = (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) > 0
        for a, b in zip(xy, np.roll(xy, -1, axis=0)):
            d = b - a
            t = np.clip(np.dot(point - a, d) / np.dot(d, d), 0.0, 1.0)
            dist = float(np.linalg.norm(a + t * d - point))
            if best is None or dist < best[0]:
                n = np.array([d[1], -d[0]]) / np.linalg.norm(d)  # outward for a counter-clockwise ring
                best = (dist, a.copy(), b.copy(), n if ccw else -n)

    if best is None:
        raise ValueError(f"Layer {layer} (searched for label '{label.text}') has no polygons.")
    dist, p0, p1, normal = best
    if dist > max_distance:
        raise ValueError(
            f"Label '{label.text}' is not on a metal edge of layer {layer} "
            f"(nearest edge is {dist:.3g} m away, max_distance={max_distance:.3g} m). "
            f"Pass a larger max_distance if the label sits next to the edge, or layer=... to search another layer."
        )
    return p0, p1, normal
